from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "job_sql_dir": "data/job",
        "outputs_dir": "outputs",
        "cases_path": "outputs/cases.jsonl",
        "edit_log_path": "outputs/edit_log.jsonl",
        "drift_log_path": "outputs/drift_events.jsonl",
        "metrics_path": "outputs/metrics.csv",
        "baseline_csv": "outputs/res_job_pg_new.csv",
    },
    "postgres": {
        "host": "127.0.0.1",
        "port": 5432,
        "dbname": "imdb",
        "user": "postgres",
        "password": "",
        "connect_timeout": 10,
        "statement_timeout_ms": 120000,
    },
    "models": {
        "embedding_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "llm_path": "models/8B-SFT",
        "llm_adapter_path": "models/checkpoint-1000",
        "domain_file": "prompts/IMDB/domain.nl",
        "device": "cuda",
        "mock_llm": False,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "local_files_only": True,
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": 19530,
        "collection": "kmdaqo_cases",
        "dim": 384,
        "top_k": 8,
        "use_mock_on_failure": True,
    },
    "optimizer": {
        "memory_capacity": 1000,
        "short_query_threshold_ms": 200,
        "min_predicted_gain": 1.05,
        "min_confidence": 0.35,
        "llm_timeout_s": 60,
        "canary_baseline_rate": 0.0,
        "verify_candidate_hints": True,
        "max_candidate_hints": 24,
        "candidate_timeout_ms": 5000,
        "measure_current_baseline": True,
    },
    "drift": {
        "window_size": 20,
        "embedding_shift_threshold": 0.25,
        "latency_regression_threshold": 1.15,
        "hint_failure_threshold": 0.25,
        "estimate_error_threshold": 2.0,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_repo_paths(cfg: Dict[str, Any]) -> None:
    path_keys = (
        ("paths", "job_sql_dir"),
        ("paths", "outputs_dir"),
        ("paths", "cases_path"),
        ("paths", "edit_log_path"),
        ("paths", "drift_log_path"),
        ("paths", "metrics_path"),
        ("paths", "baseline_csv"),
        ("models", "llm_path"),
        ("models", "llm_adapter_path"),
        ("models", "domain_file"),
    )
    for section, key in path_keys:
        value = cfg.get(section, {}).get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            cfg[section][key] = str(REPO_ROOT / path)


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    import os

    env_map = {
        "KMDAQO_PGHOST": ("postgres", "host"),
        "KMDAQO_PGPORT": ("postgres", "port"),
        "KMDAQO_PGDATABASE": ("postgres", "dbname"),
        "KMDAQO_PGUSER": ("postgres", "user"),
        "KMDAQO_PGPASSWORD": ("postgres", "password"),
        "KMDAQO_LLM_PATH": ("models", "llm_path"),
        "KMDAQO_LLM_ADAPTER_PATH": ("models", "llm_adapter_path"),
    }
    for env_name, (section, key) in env_map.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        if key == "port":
            value = int(value)
        cfg[section][key] = value


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        _apply_env_overrides(cfg)
        _resolve_repo_paths(cfg)
        return cfg
    try:
        import yaml
    except Exception:
        _apply_env_overrides(cfg)
        _resolve_repo_paths(cfg)
        return cfg
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg = _merge(cfg, loaded)
    _apply_env_overrides(cfg)
    _resolve_repo_paths(cfg)
    return cfg


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["cases_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["outputs_dir"]) / "figures").mkdir(parents=True, exist_ok=True)
