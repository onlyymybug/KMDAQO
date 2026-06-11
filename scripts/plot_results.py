#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple KMDAQO latency figures.")
    parser.add_argument("--metrics", default="outputs/metrics.csv")
    parser.add_argument("--out", default="outputs/figures/end_to_end_latency.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}")
    rows = list(csv.DictReader(open(args.metrics, encoding="utf-8")))
    xs = list(range(len(rows)))
    e2e = [float(r["end_to_end_latency_ms"]) for r in rows]
    db = [float(r["db_latency_ms"]) for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(xs, db, label="DB execution")
    plt.plot(xs, e2e, label="End-to-end")
    plt.xlabel("Query index")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
