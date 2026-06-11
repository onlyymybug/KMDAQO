from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ensure_output_dirs, load_config
from .db import PostgresRunner
from .drift import DriftDetector
from .editor import NoOpEditor
from .embedding import Embedder
from .features import merge_features
from .hints import build_candidate_hints
from .kb import KnowledgeBase
from .llm import HintGenerator
from .metrics import metric_row, read_baseline_csv, summarize, write_csv
from .router import QueryRouter
from .utils import fingerprint_sql, is_valid_hint_for_aliases, list_sql_files, read_sql, stable_case_id


class KMDAQO:
    def __init__(self, config_path: str = "configs/default.yaml", mock_db: bool = False, mock_llm: Optional[bool] = None, mock_milvus: bool = False) -> None:
        self.cfg = load_config(config_path)
        ensure_output_dirs(self.cfg)
        import os
        device = os.environ.get("KMDAQO_DEVICE") or self.cfg["models"].get("device", "cuda")
        if device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            runtime_device = "cuda:0"
        else:
            runtime_device = device
        self.cfg["models"]["device"] = device
        self.embedder = Embedder(self.cfg["models"]["embedding_model"], self.cfg["models"].get("embedding_device", runtime_device), self.cfg["milvus"].get("dim", 384))
        self.kb = KnowledgeBase(self.cfg["milvus"], self.cfg["paths"]["cases_path"], self.embedder.dim, use_mock=mock_milvus)
        self.router = QueryRouter(self.cfg["optimizer"])
        self.db = PostgresRunner(self.cfg["postgres"], mock=mock_db)
        use_mock_llm = self.cfg["models"].get("mock_llm", False) if mock_llm is None else mock_llm
        self.llm = HintGenerator(
            self.cfg["models"]["llm_path"],
            runtime_device,
            mock=use_mock_llm,
            adapter_path=self.cfg["models"].get("llm_adapter_path"),
            domain_file=self.cfg["models"].get("domain_file", "prompts/IMDB/domain.nl"),
            generation_config={
                "max_new_tokens": self.cfg["models"].get("max_new_tokens", 160),
                "do_sample": self.cfg["models"].get("do_sample", False),
                "temperature": self.cfg["models"].get("temperature", 1.0),
                "top_p": self.cfg["models"].get("top_p", 1.0),
                "torch_dtype": self.cfg["models"].get("torch_dtype", "bfloat16"),
                "device_map": self.cfg["models"].get("device_map", "auto"),
                "local_files_only": self.cfg["models"].get("local_files_only", True),
            },
        )
        self.drift = DriftDetector(self.cfg["drift"], self.cfg["paths"]["drift_log_path"])
        self.editor = NoOpEditor(self.cfg["paths"]["edit_log_path"])
        self.baselines = read_baseline_csv(self.cfg["paths"]["baseline_csv"])

    def _evaluate_candidate_hints(self, sql: str, query_features: Dict[str, Any], baseline_latency: Optional[float]) -> tuple[Optional[Any], Optional[str], List[Dict[str, Any]]]:
        aliases = (query_features.get("alias_to_table") or {}).keys()
        trial_hints = build_candidate_hints(sql, query_features, int(self.cfg["optimizer"].get("max_candidate_hints", 12)))
        seen_hints = set()
        best_result = None
        best_hint = None
        trials = []
        original_timeout = self.cfg["postgres"].get("statement_timeout_ms")
        candidate_timeout = int(self.cfg["optimizer"].get("candidate_timeout_ms", 5000))
        try:
            self.db.set_statement_timeout(candidate_timeout)
            for trial_hint in trial_hints:
                if trial_hint in seen_hints or not is_valid_hint_for_aliases(trial_hint, aliases):
                    continue
                seen_hints.add(trial_hint)
                trial_result = self.db.explain_analyze(sql, hint=trial_hint)
                trials.append({
                    "hint": trial_hint,
                    "latency_ms": trial_result.execution_time_ms,
                    "error": trial_result.error,
                })
                if trial_result.error or trial_result.execution_time_ms is None:
                    continue
                if best_result is None or float(trial_result.execution_time_ms) < float(best_result.execution_time_ms or 1e18):
                    best_result = trial_result
                    best_hint = trial_hint
        finally:
            self.db.set_statement_timeout(int(original_timeout) if original_timeout else None)
        if best_result is not None and (baseline_latency is None or float(best_result.execution_time_ms or 1e18) < baseline_latency):
            return best_result, best_hint, trials
        return None, None, trials

    def build_case_from_sql(self, sql_file: Path, hint: Optional[str] = None) -> Dict[str, Any]:
        sql = read_sql(sql_file)
        baseline_latency = self.baselines.get(sql_file.name)
        result = self.db.explain_analyze(sql, hint=hint)
        latency = result.execution_time_ms or baseline_latency or 0.0
        speedup = (baseline_latency / latency) if baseline_latency and latency else 1.0
        fp = fingerprint_sql(sql)
        vector = self.embedder.embed([sql])[0]
        row = {
            "case_id": stable_case_id(sql_file.name, sql),
            "sql_file": sql_file.name,
            "fingerprint": fp,
            "sql": sql,
            "hint": hint or "",
            "latency_ms": latency,
            "baseline_latency_ms": baseline_latency or latency,
            "speedup": speedup,
            "plan_json": result.raw_plan,
            "plan_features": result.plan_features,
            "error": result.error,
        }
        self.kb.insert_case(row, vector)
        self.router.remember(fp, hint, latency, speedup)
        return row

    def build_kb(self, limit: Optional[int] = None, reset: bool = False) -> List[Dict[str, Any]]:
        if reset:
            self.kb.reset()
        rows = []
        files = list_sql_files(self.cfg["paths"]["job_sql_dir"])
        for sql_file in files[:limit]:
            rows.append(self.build_case_from_sql(sql_file))
        removed = self.kb.enforce_capacity(int(self.cfg["optimizer"]["memory_capacity"]))
        if removed:
            self.editor.validate_and_log(removed, {"reason": "capacity_eviction"})
        return rows

    def optimize_sql(self, sql: str, sql_file: str = "<inline>", execute: bool = True, use_rag: bool = True) -> Dict[str, Any]:
        fp = fingerprint_sql(sql)
        baseline_latency = self.baselines.get(sql_file)
        current_baseline_result = None
        if execute and self.cfg["optimizer"].get("measure_current_baseline", False):
            current_baseline_result = self.db.explain_analyze(sql, hint=None)
            if current_baseline_result.execution_time_ms is not None and not current_baseline_result.error:
                baseline_latency = current_baseline_result.execution_time_ms
        query_features = merge_features(sql, None)
        planner_result = None
        if not self.db.mock:
            planner_result = self.db.explain_plan(sql)
            if planner_result.raw_plan is not None:
                query_features = planner_result.plan_features
        vector = self.embedder.embed([sql])[0]
        predicted_gain = 1.1 if baseline_latency is None or baseline_latency >= self.cfg["optimizer"]["short_query_threshold_ms"] else 1.0
        route = self.router.decide(fp, baseline_latency, predicted_gain)
        retrieval_ms = 0.0
        opt_start = time.perf_counter()
        hint = None
        fallback = False
        retrieved = []
        raw_llm_output = None
        llm_confidence = None
        parsed_hint = None
        fallback_retrieved_hint = None
        prompt_messages = None
        final_prompt = None
        candidate_trials = []
        if route.route == "cache":
            hint = route.cached_hint
            parsed_hint = hint
        elif route.route == "llm":
            if use_rag:
                r_start = time.perf_counter()
                cases = self.kb.search(vector, fp, query_features, int(self.cfg["milvus"]["top_k"]), accepted_only=True)
                retrieval_ms = (time.perf_counter() - r_start) * 1000.0
                retrieved = [{"row": c.row, "vector_score": c.vector_score, "rerank_score": c.rerank_score} for c in cases]
            generation = self.llm.generate(sql, retrieved, query_features=query_features)
            raw_llm_output = generation.get("raw")
            llm_confidence = generation.get("confidence")
            parsed_hint = generation.get("raw_parsed_hint")
            fallback_retrieved_hint = generation.get("fallback_retrieved_hint")
            prompt_messages = generation.get("prompt_messages")
            final_prompt = generation.get("final_prompt")
            aliases = (query_features.get("alias_to_table") or {}).keys()
            if generation["confidence"] >= self.cfg["optimizer"]["min_confidence"] and is_valid_hint_for_aliases(generation.get("hint"), aliases):
                hint = generation["hint"]
            else:
                fallback = True
        else:
            fallback = True
        opt_ms = (time.perf_counter() - opt_start) * 1000.0
        result = None
        if execute and route.route == "llm" and self.cfg["optimizer"].get("verify_candidate_hints", True):
            aliases = (query_features.get("alias_to_table") or {}).keys()
            trial_hints = []
            if is_valid_hint_for_aliases(hint, aliases):
                trial_hints.append(hint)
            trial_hints.extend(build_candidate_hints(sql, query_features, int(self.cfg["optimizer"].get("max_candidate_hints", 12))))
            seen_hints = set()
            best_result = None
            best_hint = None
            original_timeout = self.cfg["postgres"].get("statement_timeout_ms")
            candidate_timeout = int(self.cfg["optimizer"].get("candidate_timeout_ms", 5000))
            try:
                self.db.set_statement_timeout(candidate_timeout)
                for trial_hint in trial_hints:
                    if trial_hint in seen_hints:
                        continue
                    seen_hints.add(trial_hint)
                    trial_result = self.db.explain_analyze(sql, hint=trial_hint)
                    candidate_trials.append({
                        "hint": trial_hint,
                        "latency_ms": trial_result.execution_time_ms,
                        "error": trial_result.error,
                    })
                    if trial_result.error or trial_result.execution_time_ms is None:
                        continue
                    if best_result is None or float(trial_result.execution_time_ms) < float(best_result.execution_time_ms or 1e18):
                        best_result = trial_result
                        best_hint = trial_hint
            finally:
                self.db.set_statement_timeout(int(original_timeout) if original_timeout else None)
            if best_result is not None and (baseline_latency is None or float(best_result.execution_time_ms or 1e18) < baseline_latency):
                result = best_result
                hint = best_hint
                fallback = False
            else:
                hint = None
                fallback = True
        if result is None:
            if execute and hint is None and current_baseline_result is not None:
                result = current_baseline_result
            else:
                result = self.db.explain_analyze(sql, hint=hint) if execute else None
        db_latency = result.execution_time_ms if result else None
        row = metric_row(sql_file, route.route, db_latency, baseline_latency, opt_ms, retrieval_ms, fallback)
        speedup = row["speedup"]
        drift_event = None
        if result and execute:
            case = {
                "case_id": stable_case_id(sql_file, sql),
                "sql_file": sql_file,
                "fingerprint": fp,
                "sql": sql,
                "hint": hint or "",
                "latency_ms": db_latency,
                "baseline_latency_ms": baseline_latency or db_latency,
                "speedup": speedup,
                "plan_json": result.raw_plan,
                "plan_features": result.plan_features,
                "error": result.error,
                "accepted_for_rag": bool(hint) and speedup > 1.0 and not result.error,
            }
            self.kb.insert_case(case, vector)
            self.router.remember(fp, hint, db_latency, speedup)
            drift_record = dict(row)
            drift_record["sql"] = sql
            drift_record["sql_file"] = sql_file
            drift_record["route"] = route.route
            drift_record["embedding"] = vector
            drift_record["estimate_error"] = result.plan_features.get("estimate_error")
            drift_event = self.drift.observe(drift_record)
        return {
            "metric": row,
            "hint": hint,
            "route": route,
            "retrieved": retrieved,
            "raw_llm_output": raw_llm_output,
            "llm_confidence": llm_confidence,
            "parsed_hint": parsed_hint,
            "fallback_retrieved_hint": fallback_retrieved_hint,
            "prompt_messages": prompt_messages,
            "final_prompt": final_prompt,
            "use_rag": use_rag,
            "candidate_trials": candidate_trials,
            "drift_event": drift_event,
        }

    def adapt_to_drift(self, representatives: Optional[List[Dict[str, Any]]] = None, limit: Optional[int] = None, min_speedup: float = 1.0) -> Dict[str, Any]:
        """Offline replay path for drifted representative queries.

        This mirrors the paper's drift-aware case acquisition stage: evaluate a
        bounded set of representative queries with candidate hints, keep only
        validated wins, and add them to the bounded knowledge base.
        """
        representatives = representatives or self.drift.representative_queries(limit=limit or 5)
        if limit is not None:
            representatives = representatives[:limit]
        records = []
        for item in representatives:
            sql = item.get("sql")
            sql_file = item.get("sql_file") or "<drift>"
            if not sql:
                continue
            baseline_result = self.db.explain_analyze(sql, hint=None)
            baseline_latency = baseline_result.execution_time_ms or self.baselines.get(sql_file)
            features = baseline_result.plan_features if baseline_result.raw_plan is not None or self.db.mock else merge_features(sql, None)
            vector = self.embedder.embed([sql])[0]
            best_result, best_hint, trials = self._evaluate_candidate_hints(sql, features, baseline_latency)
            record: Dict[str, Any] = {
                "sql_file": sql_file,
                "accepted": False,
                "baseline_latency_ms": baseline_latency,
                "candidate_trials": trials,
            }
            if best_result is None or best_hint is None or best_result.execution_time_ms is None:
                record["reason"] = "no_candidate_improved_baseline"
                records.append(record)
                continue
            speedup = (baseline_latency / best_result.execution_time_ms) if baseline_latency else 1.0
            if speedup < min_speedup:
                record.update({"reason": "speedup_below_threshold", "speedup": speedup})
                records.append(record)
                continue
            case = {
                "case_id": stable_case_id(f"drift:{sql_file}:{best_hint}", sql),
                "source": "drift_replay",
                "sql_file": sql_file,
                "fingerprint": fingerprint_sql(sql),
                "sql": sql,
                "hint": best_hint,
                "latency_ms": best_result.execution_time_ms,
                "baseline_latency_ms": baseline_latency or best_result.execution_time_ms,
                "speedup": speedup,
                "plan_json": best_result.raw_plan,
                "plan_features": best_result.plan_features,
                "error": best_result.error,
                "accepted_for_rag": True,
            }
            inserted = self.kb.insert_case(case, vector)
            self.router.remember(case["fingerprint"], best_hint, best_result.execution_time_ms, speedup)
            record.update({
                "accepted": True,
                "case_id": inserted.get("case_id"),
                "hint": best_hint,
                "latency_ms": best_result.execution_time_ms,
                "speedup": speedup,
            })
            records.append(record)
        removed = self.kb.enforce_capacity(int(self.cfg["optimizer"]["memory_capacity"]))
        return {
            "processed": len(records),
            "accepted": sum(1 for row in records if row.get("accepted")),
            "rejected": sum(1 for row in records if not row.get("accepted")),
            "capacity_evictions": len(removed),
            "memory_size": len(self.kb.rows),
            "records": records,
        }

    def maintain_knowledge(self, summary: Optional[Dict[str, Any]] = None, k: int = 5) -> Dict[str, Any]:
        candidates = self.kb.top_edit_candidates(k=k)
        if not candidates:
            return {"candidate_count": 0, "internalized": 0, "memory_size": len(self.kb.rows)}
        event = self.editor.validate_and_log(candidates, summary or {})
        accepted_ids = event.get("delete_from_kb_allowed") or []
        marked = self.kb.mark_internalized(accepted_ids)
        return {
            "candidate_count": len(candidates),
            "internalized": len(marked),
            "internalized_case_ids": [row.get("case_id") for row in marked],
            "memory_size": len(self.kb.rows),
            "editor_event": event,
        }

    def run_job(self, limit: Optional[int] = None) -> Dict[str, Any]:
        rows = []
        for path in list_sql_files(self.cfg["paths"]["job_sql_dir"])[:limit]:
            rows.append(self.optimize_sql(read_sql(path), sql_file=path.name)["metric"])
        write_csv(self.cfg["paths"]["metrics_path"], rows)
        summary = summarize(rows, memory_size=len(self.kb.rows))
        summary_path = Path(self.cfg["paths"]["outputs_dir"]) / "summary.json"
        import json
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["knowledge_maintenance"] = self.maintain_knowledge(dict(summary), k=5)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
