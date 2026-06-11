#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from kmdaqo.pipeline import KMDAQO
from kmdaqo.utils import read_sql


def write_log(path: Path, payload: Dict[str, Any], include_final_prompt: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = payload.get("prompt_messages") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# KMDAQO run_one log\n\n")
        f.write(f"use_rag: {payload.get('use_rag')}\n\n")
        f.write("## Summary JSON\n")
        f.write(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        f.write("\n\n")
        f.write("## Prompt Messages\n")
        if messages:
            for idx, message in enumerate(messages, start=1):
                f.write(f"\n### Message {idx}: {message.get('role')}\n")
                f.write(message.get("content") or "")
                f.write("\n")
        else:
            f.write("None\n")
        if include_final_prompt:
            f.write("\n## Final Prompt\n")
            f.write(payload.get("final_prompt") or "None")
            f.write("\n\n")
        else:
            f.write("\n")
        f.write("## Raw LLM Output\n")
        f.write(payload.get("raw_llm_output") or "None")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize one SQL file with KMDAQO.")
    parser.add_argument("sql_file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mock-db", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-milvus", action="store_true")
    rag_group = parser.add_mutually_exclusive_group()
    rag_group.add_argument("--use-rag", dest="use_rag", action="store_true", default=True)
    rag_group.add_argument("--no-rag", dest="use_rag", action="store_false")
    parser.add_argument("--log-file", default="outputs/run_one.log")
    parser.add_argument("--log-final-prompt", action="store_true", help="Also log tokenizer-rendered final prompt.")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cuda:0, cuda:1, cpu.")
    args = parser.parse_args()
    if args.device:
        os.environ["KMDAQO_DEVICE"] = args.device
    path = Path(args.sql_file)
    system = KMDAQO(args.config, mock_db=args.mock_db, mock_llm=args.mock_llm, mock_milvus=args.mock_milvus)
    result = system.optimize_sql(read_sql(path), sql_file=path.name, use_rag=args.use_rag)
    summary = {
        "use_rag": result.get("use_rag"),
        "hint": result["hint"],
        "parsed_hint": result.get("parsed_hint"),
        "fallback_retrieved_hint": result.get("fallback_retrieved_hint"),
        "llm_confidence": result.get("llm_confidence"),
        "raw_llm_output": result.get("raw_llm_output"),
        "metric": result["metric"],
        "candidate_trials": result.get("candidate_trials") or [],
    }
    log_path = Path(args.log_file)
    write_log(
        log_path,
        {
            "summary": summary,
            "prompt_messages": result.get("prompt_messages"),
            "final_prompt": result.get("final_prompt"),
            "raw_llm_output": result.get("raw_llm_output"),
            "use_rag": result.get("use_rag"),
        },
        include_final_prompt=args.log_final_prompt,
    )
    summary["log_file"] = str(log_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.print_prompt:
        print("\n===== PROMPT MESSAGES =====")
        for message in result.get("prompt_messages") or []:
            print(f"\n--- {message.get('role')} ---")
            print(message.get("content") or "")
        print("\n===== FINAL PROMPT =====")
        print(result.get("final_prompt") or "None")


if __name__ == "__main__":
    main()
