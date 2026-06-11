#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os

from kmdaqo.pipeline import KMDAQO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JOB benchmark with KMDAQO.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-db", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-milvus", action="store_true")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cuda:0, cuda:1, cpu.")
    args = parser.parse_args()
    if args.device:
        os.environ["KMDAQO_DEVICE"] = args.device
    system = KMDAQO(args.config, mock_db=args.mock_db, mock_llm=args.mock_llm, mock_milvus=args.mock_milvus)
    summary = system.run_job(limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
