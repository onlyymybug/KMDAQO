from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from .utils import append_jsonl, is_valid_hint_for_aliases, jaccard, load_jsonl, rewrite_jsonl


@dataclass
class RetrievedCase:
    row: Dict[str, Any]
    vector_score: float
    rerank_score: float


class KnowledgeBase:
    def __init__(self, cfg: Dict[str, Any], cases_path: str, dim: int, use_mock: bool = False) -> None:
        self.cfg = cfg
        self.cases_path = cases_path
        self.dim = dim
        self.use_mock = use_mock
        self.collection = None
        self.rows: List[Dict[str, Any]] = load_jsonl(cases_path)
        if not use_mock:
            self._connect_or_mock()

    def _connect_or_mock(self) -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
            connections.connect(alias="default", host=self.cfg["host"], port=str(self.cfg["port"]))
            name = self.cfg["collection"]
            if not utility.has_collection(name):
                fields = [
                    FieldSchema(name="case_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                    FieldSchema(name="sql_file", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="fingerprint", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="hint", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="latency_ms", dtype=DataType.FLOAT),
                    FieldSchema(name="baseline_latency_ms", dtype=DataType.FLOAT),
                    FieldSchema(name="speedup", dtype=DataType.FLOAT),
                    FieldSchema(name="retrieval_count", dtype=DataType.INT64),
                    FieldSchema(name="usefulness_score", dtype=DataType.FLOAT),
                    FieldSchema(name="is_internalized", dtype=DataType.BOOL),
                ]
                schema = CollectionSchema(fields, description="KMDAQO optimization cases")
                Collection(name, schema=schema)
                collection = Collection(name)
                collection.create_index("vector", {"metric_type": "COSINE", "index_type": "AUTOINDEX", "params": {}})
            self.collection = Collection(name)
            self.collection.load()
        except Exception as exc:
            if self.cfg.get("use_mock_on_failure", True):
                print(f"[WARN] Milvus unavailable, using JSON mock KB: {exc}")
                self.use_mock = True
            else:
                raise

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def is_usable_rag_case(self, row: Dict[str, Any]) -> bool:
        hint = (row.get("hint") or "").strip()
        aliases = ((row.get("plan_features") or {}).get("alias_to_table") or {}).keys()
        return (
            bool(row.get("accepted_for_rag"))
            and not bool(row.get("is_internalized"))
            and float(row.get("speedup") or 0.0) > 1.0
            and is_valid_hint_for_aliases(hint, aliases)
            and hint != "/*+ */"
        )

    def _find_duplicate(self, row: Dict[str, Any]) -> Dict[str, Any] | None:
        case_id = row.get("case_id")
        fingerprint = row.get("fingerprint")
        hint = (row.get("hint") or "").strip()
        for existing in self.rows:
            if case_id and existing.get("case_id") == case_id:
                return existing
            if fingerprint and hint and existing.get("fingerprint") == fingerprint and (existing.get("hint") or "").strip() == hint:
                return existing
        return None

    def reset(self) -> None:
        self.rows = []
        rewrite_jsonl(self.cases_path, [])
        if self.collection is not None:
            try:
                self.collection.delete('case_id != ""')
                self.collection.flush()
            except Exception:
                pass

    def insert_case(self, row: Dict[str, Any], vector: List[float]) -> Dict[str, Any]:
        row = dict(row)
        row.setdefault("case_id", str(uuid.uuid4()))
        row.setdefault("retrieval_count", 0)
        row.setdefault("is_internalized", False)
        speedup = float(row.get("speedup") or 1.0)
        aliases = ((row.get("plan_features") or {}).get("alias_to_table") or {}).keys()
        row["accepted_for_rag"] = bool(row.get("accepted_for_rag")) and speedup > 1.0 and is_valid_hint_for_aliases(row.get("hint"), aliases)
        retrieval_count = int(row.get("retrieval_count") or 0)
        row["usefulness_score"] = float(row.get("usefulness_score") or (speedup * (1.0 + math.log1p(retrieval_count))))
        row["_vector"] = vector
        duplicate = self._find_duplicate(row)
        if duplicate is not None:
            existing_speedup = float(duplicate.get("speedup") or 0.0)
            should_update = bool(row.get("accepted_for_rag")) and (
                not duplicate.get("accepted_for_rag") or speedup >= existing_speedup
            )
            if should_update:
                preserved_retrieval_count = int(duplicate.get("retrieval_count") or 0)
                duplicate.update(row)
                duplicate["retrieval_count"] = preserved_retrieval_count
                duplicate["usefulness_score"] = float(speedup * (1.0 + math.log1p(preserved_retrieval_count)))
                rewrite_jsonl(self.cases_path, self.rows)
            return duplicate
        self.rows.append(row)
        append_jsonl(self.cases_path, row)
        if self.collection is not None and not self.use_mock:
            entity = [
                [row["case_id"]],
                [vector],
                [row.get("sql_file", "")],
                [row.get("fingerprint", "")],
                [row.get("hint", "") or ""],
                [float(row.get("latency_ms") or 0.0)],
                [float(row.get("baseline_latency_ms") or 0.0)],
                [float(row.get("speedup") or 1.0)],
                [int(row.get("retrieval_count") or 0)],
                [float(row.get("usefulness_score") or 0.0)],
                [bool(row.get("is_internalized", False))],
            ]
            self.collection.insert(entity)
            self.collection.flush()
        return row

    def _rerank_score(self, query_features: Dict[str, Any], row: Dict[str, Any], vector_score: float, fingerprint: str) -> float:
        features = row.get("plan_features") or {}
        fp_bonus = 1.0 if row.get("fingerprint") == fingerprint else 0.0
        join = jaccard(query_features.get("join_graph", []), features.get("join_graph", []))
        pred = jaccard(query_features.get("predicate_columns", []), features.get("predicate_columns", []))
        ops = jaccard(query_features.get("operator_sequence", []), features.get("operator_sequence", []))
        useful = min(float(row.get("usefulness_score") or 0.0), 5.0) / 5.0
        return 0.45 * vector_score + 0.20 * fp_bonus + 0.15 * join + 0.10 * pred + 0.05 * ops + 0.05 * useful

    def search(self, vector: List[float], fingerprint: str, features: Dict[str, Any], top_k: int = 8, accepted_only: bool = False) -> List[RetrievedCase]:
        candidates: List[RetrievedCase] = []
        if self.collection is not None and not self.use_mock:
            results = self.collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k * 3,
                output_fields=["case_id"],
            )
            ids = {hit.entity.get("case_id"): float(hit.score) for hit in results[0]}
            rows = [r for r in self.rows if r.get("case_id") in ids]
            for row in rows:
                if accepted_only and not self.is_usable_rag_case(row):
                    continue
                score = ids[row["case_id"]]
                candidates.append(RetrievedCase(row, score, self._rerank_score(features, row, score, fingerprint)))
        else:
            for row in self.rows:
                if accepted_only and not self.is_usable_rag_case(row):
                    continue
                score = self._cosine(vector, row.get("_vector", []))
                candidates.append(RetrievedCase(row, score, self._rerank_score(features, row, score, fingerprint)))
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        for item in candidates[:top_k]:
            item.row["retrieval_count"] = int(item.row.get("retrieval_count") or 0) + 1
        rewrite_jsonl(self.cases_path, self.rows)
        return candidates[:top_k]

    def enforce_capacity(self, capacity: int) -> List[Dict[str, Any]]:
        if len(self.rows) <= capacity:
            return []
        ranked = sorted(self.rows, key=lambda r: float(r.get("usefulness_score") or 0.0))
        remove_count = len(self.rows) - capacity
        removed = ranked[:remove_count]
        keep_ids = {r["case_id"] for r in ranked[remove_count:]}
        self.rows = [r for r in self.rows if r["case_id"] in keep_ids]
        rewrite_jsonl(self.cases_path, self.rows)
        return removed

    def mark_internalized(self, case_ids: List[str]) -> List[Dict[str, Any]]:
        case_id_set = set(case_ids)
        marked = []
        for row in self.rows:
            if row.get("case_id") in case_id_set:
                row["is_internalized"] = True
                row["accepted_for_rag"] = False
                marked.append(row)
        if marked:
            rewrite_jsonl(self.cases_path, self.rows)
        return marked

    def top_edit_candidates(self, k: int = 5) -> List[Dict[str, Any]]:
        rows = [r for r in self.rows if not r.get("is_internalized") and self.is_usable_rag_case(r)]
        rows.sort(key=lambda r: (float(r.get("usefulness_score") or 0.0), int(r.get("retrieval_count") or 0)), reverse=True)
        return rows[:k]
