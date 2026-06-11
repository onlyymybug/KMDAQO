# KMDAQO

KMDAQO is a research prototype for the paper **"KMDAQO: A
Knowledge-Managed Drift-Adaptive Query Optimization System"**. It implements a
closed-loop, LLM-assisted query optimizer that generates PostgreSQL
`pg_hint_plan` hints under workload drift.

The system keeps a bounded optimization knowledge base containing SQL queries,
plan-structured features, generated hints, execution feedback, and observed
speedups. At inference time, KMDAQO retrieves relevant historical cases,
constructs a structured prompt, generates a hint with a fine-tuned LLM, validates
the hint, and falls back safely when needed.

## What Is Included

- JOB SQL files in `data/job/`.
- PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` runner.
- Milvus-backed vector memory with a JSONL fallback sidecar.
- Plan-aware reranking over SQL fingerprints, join graphs, predicates, operator
  sequences, and usefulness scores.
- Router/cache/budget gate before LLM inference.
- HuggingFace LLM hint generator with a mock fallback.
- Drift detector and log-only knowledge-maintenance editor.
- End-to-end metrics for optimization latency, DB latency, RET, normalized
  latency, win/regression rates, timeout/fallback rate, retrieval latency, and
  memory size.

## Quick Mock Run

Use this path first. It does not require PostgreSQL, Milvus, CUDA, or the LLM
checkpoint:

```bash
cd /path/to/KMDAQO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/build_kb.py --mock-db --mock-milvus --reset --limit 5
PYTHONPATH=. python scripts/run_job.py --mock-db --mock-llm --mock-milvus --limit 5
```

Generated files are written to `outputs/`.

## Real Experiment Setup

Install the package and base dependencies:

```bash
pip install -e .
```

Create a local configuration file:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Set PostgreSQL credentials in `configs/local.yaml` or through environment
variables:

```bash
export KMDAQO_PGHOST=127.0.0.1
export KMDAQO_PGPORT=5432
export KMDAQO_PGDATABASE=imdb
export KMDAQO_PGUSER=postgres
export KMDAQO_PGPASSWORD='your-password'
```

Then run:

```bash
PYTHONPATH=. python scripts/build_kb.py --config configs/local.yaml --reset
PYTHONPATH=. python scripts/run_job.py --config configs/local.yaml
PYTHONPATH=. python scripts/adapt_drift.py --config configs/local.yaml --limit 5
PYTHONPATH=. python scripts/plot_results.py
```

## LLM Checkpoints

Model checkpoints are not committed to the repository. For a full LLM-assisted
run, place the base model and LoRA adapter under `models/`, or set:

```bash
export KMDAQO_LLM_PATH=/path/to/base-model
export KMDAQO_LLM_ADAPTER_PATH=/path/to/lora-adapter
```

For CUDA 11.8 environments:

```bash
pip install -r requirements-torch-cu118.txt --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-embedding.txt
```

## Artifact Documentation

See `ARTIFACT.md` for validation commands, PostgreSQL setup notes, and the
recommended reproduction workflow.
