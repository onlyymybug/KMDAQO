#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from kmdaqo.config import ensure_output_dirs, load_config
from kmdaqo.db import PostgresRunner
from kmdaqo.utils import list_sql_files, read_sql


DEFAULT_OUTPUT = "outputs/res_job_pg_new.csv"


def write_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sql_file", "query_delay", "planning_time_ms", "execution_time_ms", "error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fresh PostgreSQL baseline for JOB SQL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-db", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    sql_files = list_sql_files(cfg["paths"]["job_sql_dir"])
    if args.limit is not None:
        sql_files = sql_files[: args.limit]

    runner = PostgresRunner(cfg["postgres"], mock=args.mock_db)
    rows: List[Dict[str, Any]] = []
    start_all = time.perf_counter()

    try:
        for idx, sql_file in enumerate(sql_files, start=1):
            sql = read_sql(sql_file)
            started = time.perf_counter()
            result = runner.explain_analyze(sql, hint=None)
            elapsed = time.perf_counter() - started
            execution_time_ms = result.execution_time_ms
            query_delay = execution_time_ms / 1000.0 if execution_time_ms is not None else ""
            row = {
                "sql_file": sql_file.name,
                "query_delay": query_delay,
                "planning_time_ms": result.planning_time_ms if result.planning_time_ms is not None else "",
                "execution_time_ms": execution_time_ms if execution_time_ms is not None else "",
                "error": result.error or "",
            }
            rows.append(row)
            write_rows(args.output, rows)
            status = "OK" if not result.error else "ERROR"
            print(
                f"[{idx}/{len(sql_files)}] {sql_file.name} {status} "
                f"execution_time_ms={row['execution_time_ms']} elapsed_s={elapsed:.3f}"
            )
            if result.error:
                print(f"  error: {result.error}")
    finally:
        runner.close()

    total = time.perf_counter() - start_all
    print(f"[DONE] wrote {len(rows)} rows to {args.output} in {total:.2f}s")


if __name__ == "__main__":
    main()
