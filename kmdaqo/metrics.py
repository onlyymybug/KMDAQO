from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


def metric_row(
    sql_file: str,
    route: str,
    db_latency_ms: float | None,
    baseline_latency_ms: float | None,
    optimization_latency_ms: float,
    retrieval_latency_ms: float,
    fallback: bool,
    timeout: bool = False,
) -> Dict[str, Any]:
    db = float(db_latency_ms or 0.0)
    baseline = float(baseline_latency_ms or db or 1.0)
    e2e = db + float(optimization_latency_ms or 0.0)
    normalized = e2e / baseline if baseline > 0 else 1.0
    speedup = baseline / db if db > 0 else 1.0
    return {
        "sql_file": sql_file,
        "route": route,
        "db_latency_ms": db,
        "baseline_latency_ms": baseline,
        "optimization_latency_ms": optimization_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "end_to_end_latency_ms": e2e,
        "ret": db / baseline if baseline > 0 else 1.0,
        "normalized_latency": normalized,
        "regret": max(0.0, normalized - 1.0),
        "speedup": speedup,
        "win": db < baseline,
        "regression": e2e > baseline,
        "break_even": speedup >= (1.0 + optimization_latency_ms / max(db, 1.0)),
        "fallback": fallback,
        "timeout": timeout,
    }


def summarize(rows: List[Dict[str, Any]], memory_size: int = 0) -> Dict[str, Any]:
    if not rows:
        return {}
    e2e = [float(r["end_to_end_latency_ms"]) for r in rows]
    sorted_e2e = sorted(e2e)
    p95 = sorted_e2e[min(len(sorted_e2e) - 1, int(0.95 * (len(sorted_e2e) - 1)))]
    p99 = sorted_e2e[min(len(sorted_e2e) - 1, int(0.99 * (len(sorted_e2e) - 1)))]
    return {
        "queries": len(rows),
        "avg_end_to_end_latency_ms": statistics.mean(e2e),
        "p95_end_to_end_latency_ms": p95,
        "p99_end_to_end_latency_ms": p99,
        "avg_ret": statistics.mean(float(r["ret"]) for r in rows),
        "win_rate": sum(bool(r["win"]) for r in rows) / len(rows),
        "regression_rate": sum(bool(r["regression"]) for r in rows) / len(rows),
        "break_even_ratio": sum(bool(r["break_even"]) for r in rows) / len(rows),
        "timeout_rate": sum(bool(r["timeout"]) for r in rows) / len(rows),
        "fallback_rate": sum(bool(r["fallback"]) for r in rows) / len(rows),
        "avg_retrieval_latency_ms": statistics.mean(float(r["retrieval_latency_ms"]) for r in rows),
        "memory_size": memory_size,
    }


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_baseline_csv(path: str | Path) -> Dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("sql_file")
            delay = row.get("query_delay")
            if name and delay:
                value = float(delay)
                result[name] = value if value > 100 else value * 1000.0
    return result

