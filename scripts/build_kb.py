#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from kmdaqo.pipeline import KMDAQO


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KMDAQO knowledge base from JOB SQL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--mock-db", action="store_true")
    parser.add_argument("--mock-milvus", action="store_true")
    args = parser.parse_args()
    system = KMDAQO(args.config, mock_db=args.mock_db, mock_llm=True, mock_milvus=args.mock_milvus)
    rows = system.build_kb(limit=args.limit, reset=args.reset)
    print(json.dumps({"built_cases": len(rows), "memory_size": len(system.kb.rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
