#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kmdaqo.pipeline import KMDAQO
from kmdaqo.utils import list_sql_files, read_sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KMDAQO offline drift adaptation by replaying representative queries.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sql-files", default=None, help="Comma-separated SQL file names. Defaults to the first --limit JOB files.")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--mock-db", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-milvus", action="store_true")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cuda:0, cuda:1, cpu.")
    args = parser.parse_args()
    if args.device:
        os.environ["KMDAQO_DEVICE"] = args.device

    system = KMDAQO(args.config, mock_db=args.mock_db, mock_llm=args.mock_llm, mock_milvus=args.mock_milvus)
    files = list_sql_files(system.cfg["paths"]["job_sql_dir"])
    if args.sql_files:
        wanted = [name.strip() for name in args.sql_files.split(",") if name.strip()]
        by_name = {path.name: path for path in files}
        files = [by_name[name] for name in wanted]
    else:
        files = files[: args.limit]

    representatives = [
        {
            "sql_file": Path(path).name,
            "sql": read_sql(path),
            "route": "offline_replay_seed",
            "normalized_latency": None,
            "regret": None,
            "fallback": None,
        }
        for path in files
    ]
    summary = system.adapt_to_drift(representatives, limit=args.limit, min_speedup=args.min_speedup)
    summary["knowledge_maintenance"] = system.maintain_knowledge(dict(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
