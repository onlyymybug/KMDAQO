from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .utils import append_jsonl


class DriftDetector:
    def __init__(self, cfg: Dict[str, Any], log_path: str) -> None:
        self.cfg = cfg
        self.log_path = log_path
        self.window: Deque[Dict[str, Any]] = deque(maxlen=int(cfg.get("window_size", 20)))
        self.reference: Optional[List[float]] = None

    def _centroid(self, rows: List[Dict[str, Any]]) -> Optional[List[float]]:
        vectors = [r.get("embedding") for r in rows if r.get("embedding")]
        if not vectors:
            return None
        dim = len(vectors[0])
        return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

    def _distance(self, a: List[float], b: List[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def observe(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.window.append(record)
        if len(self.window) < self.window.maxlen:
            return None
        rows = list(self.window)
        centroid = self._centroid(rows)
        if centroid is not None and self.reference is None:
            self.reference = centroid
            return None
        embedding_shift = self._distance(self.reference, centroid) if self.reference and centroid else 0.0
        regressions = [r for r in rows if r.get("normalized_latency", 1.0) > self.cfg.get("latency_regression_threshold", 1.15)]
        failures = [r for r in rows if r.get("fallback") or r.get("regression")]
        estimate_errors = [r.get("estimate_error") for r in rows if r.get("estimate_error")]
        avg_est_error = sum(estimate_errors) / len(estimate_errors) if estimate_errors else 1.0
        event = {
            "embedding_shift": embedding_shift,
            "latency_regression_rate": len(regressions) / len(rows),
            "hint_failure_rate": len(failures) / len(rows),
            "optimizer_estimate_error": avg_est_error,
            "window_size": len(rows),
            "representative_queries": self.representative_queries(rows),
        }
        triggered = (
            embedding_shift > self.cfg.get("embedding_shift_threshold", 0.25)
            or event["latency_regression_rate"] > 0.25
            or event["hint_failure_rate"] > self.cfg.get("hint_failure_threshold", 0.25)
            or avg_est_error > self.cfg.get("estimate_error_threshold", 2.0)
        )
        if triggered:
            event["type"] = "workload_drift"
            append_jsonl(self.log_path, event)
            if centroid is not None:
                self.reference = centroid
            return event
        return None

    def representative_queries(self, rows: Optional[List[Dict[str, Any]]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        rows = rows or list(self.window)
        ranked = sorted(
            rows,
            key=lambda row: (
                bool(row.get("fallback") or row.get("regression")),
                float(row.get("regret") or 0.0),
                float(row.get("normalized_latency") or 1.0),
            ),
            reverse=True,
        )
        representatives = []
        seen = set()
        for row in ranked:
            sql = row.get("sql")
            sql_file = row.get("sql_file")
            if not sql or sql_file in seen:
                continue
            seen.add(sql_file)
            representatives.append({
                "sql_file": sql_file,
                "sql": sql,
                "route": row.get("route"),
                "normalized_latency": row.get("normalized_latency"),
                "regret": row.get("regret"),
                "fallback": row.get("fallback"),
            })
            if len(representatives) >= limit:
                break
        return representatives
