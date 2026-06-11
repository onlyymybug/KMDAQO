#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from kmdaqo.features import merge_features
from kmdaqo.llm import build_postgres_hint, build_query_statistics
from kmdaqo.pipeline import KMDAQO
from kmdaqo.utils import fingerprint_sql, is_valid_hint_for_aliases, list_sql_files, read_sql


def parse_sql_files(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_files(sql_dir: str, names: List[str], limit: int | None) -> List[Path]:
    files = list_sql_files(sql_dir)
    if names:
        by_name = {path.name: path for path in files}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise FileNotFoundError(f"SQL files not found under {sql_dir}: {', '.join(missing)}")
        files = [by_name[name] for name in names]
    if limit is not None:
        files = files[:limit]
    return files


def rag_case_id(sql_file: str, fingerprint: str, hint: str) -> str:
    payload = f"rag:{sql_file}:{fingerprint}:{hint}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def inspect_kb(system: KMDAQO) -> Dict[str, Any]:
    rows = system.kb.rows
    usable = [row for row in rows if system.kb.is_usable_rag_case(row)]
    speedups = [float(row.get("speedup") or 0.0) for row in usable]
    top_cases = sorted(
        usable,
        key=lambda row: (float(row.get("speedup") or 0.0), float(row.get("usefulness_score") or 0.0)),
        reverse=True,
    )[:10]
    return {
        "total_cases": len(rows),
        "usable_rag_cases": len(usable),
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        "top_cases": [
            {
                "sql_file": row.get("sql_file"),
                "speedup": row.get("speedup"),
                "latency_ms": row.get("latency_ms"),
                "baseline_latency_ms": row.get("baseline_latency_ms"),
                "hint": row.get("hint"),
            }
            for row in top_cases
        ],
    }


def build_one_case(system: KMDAQO, sql_file: Path, min_speedup: float) -> Dict[str, Any]:
    sql = read_sql(sql_file)
    fingerprint = fingerprint_sql(sql)
    baseline_latency = system.baselines.get(sql_file.name)

    planner_result = system.db.explain_plan(sql)
    query_features = planner_result.plan_features if planner_result.raw_plan is not None or system.db.mock else merge_features(sql, None)
    query_statistics = build_query_statistics(sql, query_features)
    postgres_hint = build_postgres_hint(query_features)

    generation = system.llm.generate(sql, [], query_features=query_features, postgres_hint=postgres_hint)
    hint = (generation.get("hint") or "").strip()
    accepted = is_valid_hint_for_aliases(hint, (query_features.get("alias_to_table") or {}).keys()) and hint != "/*+ */"
    record: Dict[str, Any] = {
        "sql_file": sql_file.name,
        "raw_llm_output": generation.get("raw"),
        "parsed_hint": hint or None,
        "accepted": False,
        "reason": None,
    }
    if not accepted:
        record["reason"] = "invalid_or_empty_hint"
        return record

    result = system.db.explain_analyze(sql, hint=hint)
    if result.error:
        record["reason"] = "db_error"
        record["error"] = result.error
        return record
    latency = result.execution_time_ms or 0.0
    baseline = baseline_latency or latency
    speedup = (baseline / latency) if baseline and latency else 0.0
    if speedup < min_speedup:
        record.update({
            "reason": "speedup_below_threshold",
            "latency_ms": latency,
            "baseline_latency_ms": baseline,
            "speedup": speedup,
        })
        return record

    vector = system.embedder.embed([sql])[0]
    case = {
        "case_id": rag_case_id(sql_file.name, fingerprint, hint),
        "source": "rag_build",
        "sql_file": sql_file.name,
        "fingerprint": fingerprint,
        "sql": sql,
        "hint": hint,
        "latency_ms": latency,
        "baseline_latency_ms": baseline,
        "speedup": speedup,
        "plan_json": result.raw_plan,
        "plan_features": result.plan_features,
        "error": result.error,
        "raw_llm_output": generation.get("raw"),
        "parsed_hint": hint,
        "prompt_messages": generation.get("prompt_messages"),
        "postgres_hint": postgres_hint,
        "query_statistics": query_statistics,
        "accepted_for_rag": True,
    }
    inserted = system.kb.insert_case(case, vector)
    record.update({
        "accepted": True,
        "case_id": inserted.get("case_id"),
        "latency_ms": latency,
        "baseline_latency_ms": baseline,
        "speedup": speedup,
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real KMDAQO RAG knowledge base from evaluated JOB hints.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sql-files", default=None, help="Comma-separated SQL file names, for example: 1a.sql,1b.sql")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--mock-db", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-milvus", action="store_true")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cuda:0, cuda:1, cpu.")
    args = parser.parse_args()
    if args.device:
        os.environ["KMDAQO_DEVICE"] = args.device

    if args.inspect:
        system = KMDAQO(args.config, mock_db=True, mock_llm=True, mock_milvus=args.mock_milvus)
        print(json.dumps(inspect_kb(system), ensure_ascii=False, indent=2))
        return

    system = KMDAQO(args.config, mock_db=args.mock_db, mock_llm=args.mock_llm, mock_milvus=args.mock_milvus)
    if args.reset:
        system.kb.reset()

    names = parse_sql_files(args.sql_files)
    files = select_files(system.cfg["paths"]["job_sql_dir"], names, args.limit)
    records = []
    for index, sql_file in enumerate(files, 1):
        record = build_one_case(system, sql_file, args.min_speedup)
        records.append(record)
        status = "accepted" if record.get("accepted") else f"rejected:{record.get('reason')}"
        speedup = record.get("speedup")
        speedup_text = f", speedup={speedup:.4f}" if isinstance(speedup, (int, float)) else ""
        print(f"[{index}/{len(files)}] {sql_file.name}: {status}{speedup_text}")

    removed = system.kb.enforce_capacity(int(system.cfg["optimizer"]["memory_capacity"]))
    if removed:
        system.editor.validate_and_log(removed, {"reason": "capacity_eviction_after_rag_build"})
    system.db.close()
    summary = {
        "processed": len(records),
        "accepted": sum(1 for row in records if row.get("accepted")),
        "rejected": sum(1 for row in records if not row.get("accepted")),
        "memory_size": len(system.kb.rows),
        "usable_rag_cases": len([row for row in system.kb.rows if system.kb.is_usable_rag_case(row)]),
        "records": records,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
