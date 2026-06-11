from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .features import merge_features
from .utils import attach_hint


@dataclass
class ExecutionResult:
    sql: str
    hint: Optional[str]
    planning_time_ms: Optional[float]
    execution_time_ms: Optional[float]
    raw_plan: Optional[Dict[str, Any]]
    plan_features: Dict[str, Any]
    error: Optional[str] = None


class PostgresRunner:
    def __init__(self, cfg: Dict[str, Any], mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock
        self.conn = None

    def connect(self) -> None:
        if self.mock:
            return
        if self.conn is not None and getattr(self.conn, "closed", 1) == 0:
            return
        try:
            import psycopg2
        except Exception as exc:
            raise RuntimeError("psycopg2 is required for real PostgreSQL execution") from exc
        params = {k: v for k, v in self.cfg.items() if k != "statement_timeout_ms"}
        if not params.get("password"):
            params.pop("password", None)
        self.conn = psycopg2.connect(**params)
        self.conn.autocommit = True
        timeout = self.cfg.get("statement_timeout_ms")
        if timeout:
            with self.conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {int(timeout)}")

    def set_statement_timeout(self, timeout_ms: Optional[int]) -> None:
        if self.mock:
            return
        self.connect()
        with self.conn.cursor() as cur:
            if timeout_ms:
                cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            else:
                cur.execute("SET statement_timeout = 0")

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

    def explain_analyze(self, sql: str, hint: Optional[str] = None) -> ExecutionResult:
        return self._explain(sql, hint=hint, analyze=True)

    def explain_plan(self, sql: str, hint: Optional[str] = None) -> ExecutionResult:
        return self._explain(sql, hint=hint, analyze=False)

    def _explain(self, sql: str, hint: Optional[str] = None, analyze: bool = True) -> ExecutionResult:
        if self.mock:
            features = merge_features(sql, None)
            pseudo = max(10.0, min(10000.0, len(sql) * 0.25))
            if hint:
                pseudo *= 0.9
            return ExecutionResult(sql, hint, 0.1, pseudo if analyze else None, None, features)
        self.connect()
        hinted_sql = attach_hint(sql, hint)
        explain_sql = f"""
EXPLAIN (
  ANALYZE {str(analyze).upper()},
  BUFFERS {str(analyze).upper()},
  COSTS TRUE,
  TIMING {str(analyze).upper()},
  SUMMARY TRUE,
  FORMAT JSON
)
{hinted_sql}
"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(explain_sql)
                row = cur.fetchone()
            raw = row[0][0] if isinstance(row[0], list) else row[0]
            return ExecutionResult(
                sql=sql,
                hint=hint,
                planning_time_ms=raw.get("Planning Time"),
                execution_time_ms=raw.get("Execution Time"),
                raw_plan=raw,
                plan_features=merge_features(sql, raw),
            )
        except Exception as exc:
            return ExecutionResult(sql, hint, None, None, None, merge_features(sql, None), str(exc))
