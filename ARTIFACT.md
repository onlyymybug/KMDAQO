# KMDAQO Artifact Guide

This repository contains the implementation accompanying:

> KMDAQO: A Knowledge-Managed Drift-Adaptive Query Optimization System

The artifact is organized to support three levels of use: a dependency-light
mock run, a PostgreSQL run without the LLM, and the full LLM-assisted run.

## Repository Contents

- `kmdaqo/`: KMDAQO Python package.
- `scripts/`: command-line entry points for building the knowledge base,
  running JOB queries, running one query, plotting metrics, and building
  baseline latencies.
- `configs/`: default and local example configuration files.
- `data/job/`: JOB SQL workload files used by the lightweight artifact run.
- `prompts/IMDB/domain.nl`: domain prompt for pg_hint_plan-style hint bodies.
- `KMDAQO_v2.pdf`: paper draft included for reference.

Large model checkpoints, PostgreSQL data directories, Milvus data, and generated
experiment outputs are intentionally excluded from the public repository.

## Quick Validation

The quickest check does not require PostgreSQL, Milvus, or a local LLM:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
PYTHONPATH=. pytest -q
PYTHONPATH=. python scripts/build_kb.py --mock-db --mock-milvus --reset --limit 5
PYTHONPATH=. python scripts/run_job.py --mock-db --mock-llm --mock-milvus --limit 5
PYTHONPATH=. python scripts/adapt_drift.py --mock-db --mock-llm --mock-milvus --limit 5
```

This exercises the pipeline with deterministic mock database timings, hash
embeddings when sentence-transformers is unavailable, a JSONL knowledge base,
and mock LLM hint generation.

## PostgreSQL Setup

For real execution, install PostgreSQL and `pg_hint_plan`, load the IMDB/JOB
database, and update a local config:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Set credentials either in `configs/local.yaml` or through environment variables:

```bash
export KMDAQO_PGHOST=127.0.0.1
export KMDAQO_PGPORT=5432
export KMDAQO_PGDATABASE=imdb
export KMDAQO_PGUSER=postgres
export KMDAQO_PGPASSWORD='your-password'
```

Then run:

```bash
PYTHONPATH=. python scripts/build_baseline.py --config configs/local.yaml
PYTHONPATH=. python scripts/build_kb.py --config configs/local.yaml --reset
PYTHONPATH=. python scripts/run_job.py --config configs/local.yaml
PYTHONPATH=. python scripts/adapt_drift.py --config configs/local.yaml --limit 5
```

## Full LLM-Assisted Run

Place model checkpoints under `models/` or point to external paths:

```bash
export KMDAQO_LLM_PATH=/path/to/8B-SFT
export KMDAQO_LLM_ADAPTER_PATH=/path/to/checkpoint-1000
```

Install the optional GPU stack as appropriate for your CUDA version. For CUDA
11.8:

```bash
pip install -r requirements-torch-cu118.txt --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-embedding.txt
```

Run a single query:

```bash
PYTHONPATH=. python scripts/run_one.py data/job/1a.sql \
  --config configs/local.yaml \
  --mock-milvus \
  --device cuda:0
```

Run the JOB workload:

```bash
PYTHONPATH=. python scripts/run_job.py --config configs/local.yaml --device cuda:0
```

## Notes on Reproducibility

The mock run verifies code paths and artifact usability, but it is not intended
to reproduce paper latencies. Paper-level results require the same database
contents, PostgreSQL version, `pg_hint_plan` setup, model checkpoints, hardware,
and workload-drift construction used in the experiments.
