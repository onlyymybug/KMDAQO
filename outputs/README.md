# Outputs

Generated experiment files are written here, including:

- `cases.jsonl`: JSONL knowledge-base sidecar.
- `metrics.csv`: per-query optimization metrics.
- `summary.json`: aggregate benchmark summary.
- `drift_events.jsonl`: drift detector events.
- `edit_log.jsonl`: knowledge-maintenance log.

The public repository ignores generated outputs by default. Recreate them with
the commands in `README.md` or `ARTIFACT.md`.
