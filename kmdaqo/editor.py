from __future__ import annotations

from typing import Any, Dict, List

from .utils import append_jsonl


class NoOpEditor:
    """Validation-gated, log-only stand-in for offline knowledge editing.

    The paper describes selective internalization with validation and rollback.
    This prototype records the same decision boundary without mutating model
    weights, so artifact runs remain lightweight and reproducible.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

    def validate_and_log(self, candidates: List[Dict[str, Any]], before_metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
        before_metrics = before_metrics or {}
        accepted = []
        rejected = []
        for row in candidates:
            if (
                row.get("accepted_for_rag")
                and row.get("hint")
                and float(row.get("speedup") or 1.0) >= 1.05
                and float(row.get("usefulness_score") or 0.0) >= 1.05
            ):
                accepted.append(row.get("case_id"))
            else:
                rejected.append(row.get("case_id"))
        event = {
            "editor": "ValidatedLogEditor",
            "candidate_count": len(candidates),
            "accepted_case_ids": accepted,
            "rejected_case_ids": rejected,
            "before_metrics": before_metrics,
            "after_metrics": before_metrics,
            "rollback": False,
            "delete_from_kb_allowed": accepted,
        }
        append_jsonl(self.log_path, event)
        return event
