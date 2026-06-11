from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RouteDecision:
    route: str
    reason: str
    cached_hint: Optional[str] = None
    confidence: float = 0.0


class QueryRouter:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.cache: Dict[str, Dict[str, Any]] = {}

    def remember(self, fingerprint: str, hint: Optional[str], latency_ms: Optional[float], speedup: float) -> None:
        if hint and speedup >= self.cfg.get("min_predicted_gain", 1.05):
            self.cache[fingerprint] = {"hint": hint, "latency_ms": latency_ms, "speedup": speedup}

    def decide(self, fingerprint: str, baseline_latency_ms: Optional[float], predicted_gain: float = 1.0) -> RouteDecision:
        cached = self.cache.get(fingerprint)
        if cached:
            return RouteDecision("cache", "fingerprint cache hit", cached_hint=cached["hint"], confidence=0.9)
        threshold = float(self.cfg.get("short_query_threshold_ms", 200))
        if baseline_latency_ms is not None and baseline_latency_ms < threshold:
            return RouteDecision("postgres", f"short query below {threshold} ms", confidence=0.7)
        if predicted_gain < float(self.cfg.get("min_predicted_gain", 1.05)):
            return RouteDecision("postgres", "predicted gain below threshold", confidence=0.5)
        return RouteDecision("llm", "eligible for retrieval plus LLM", confidence=0.6)

