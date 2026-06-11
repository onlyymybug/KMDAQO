from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from .utils import is_valid_hint_for_aliases


def _left_deep_leading(order: List[str]) -> str:
    expr = order[0]
    for alias in order[1:]:
        expr = f"({expr} {alias})"
    return f"Leading({expr})"


def _join_edges(sql: str, aliases: Iterable[str]) -> Dict[str, Set[str]]:
    alias_set = {alias.lower() for alias in aliases}
    graph = {alias: set() for alias in alias_set}
    for left, right in re.findall(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\.[A-Za-z_]\w*", sql):
        left = left.lower()
        right = right.lower()
        if left in alias_set and right in alias_set and left != right:
            graph[left].add(right)
            graph[right].add(left)
    return graph


def _connected_orders(aliases: List[str], cards: Dict[str, float], graph: Dict[str, Set[str]]) -> List[List[str]]:
    orders: List[List[str]] = []
    for start in sorted(aliases, key=lambda alias: (cards.get(alias, 1e18), alias)):
        chosen = [start]
        remaining = set(aliases) - {start}
        while remaining:
            neighbors = sorted(
                [alias for alias in remaining if any(alias in graph.get(done, set()) for done in chosen)],
                key=lambda alias: (cards.get(alias, 1e18), alias),
            )
            pool = neighbors or sorted(remaining, key=lambda alias: (cards.get(alias, 1e18), alias))
            nxt = pool[0]
            chosen.append(nxt)
            remaining.remove(nxt)
        orders.append(chosen)
    return orders


def _join_methods(order: List[str], cards: Dict[str, float], small_prefix: int) -> List[str]:
    methods = []
    prefix: List[str] = []
    for idx, alias in enumerate(order):
        prefix.append(alias)
        if idx == 0:
            continue
        method = "NestLoop" if idx <= small_prefix else "HashJoin"
        methods.append(f"{method}({' '.join(prefix)})")
    return methods


def build_candidate_hints(sql: str, features: Dict[str, Any], limit: int = 16) -> List[str]:
    alias_to_table = features.get("alias_to_table") or {}
    aliases = sorted(str(alias).lower() for alias in alias_to_table)
    if len(aliases) < 2:
        return []
    cards = {str(alias).lower(): float(value or 1e18) for alias, value in (features.get("table_cardinality") or {}).items()}
    for alias in aliases:
        cards.setdefault(alias, 1e18)
    graph = _join_edges(sql, aliases)
    orders = _connected_orders(aliases, cards, graph)
    orders.append(sorted(aliases, key=lambda alias: (cards.get(alias, 1e18), alias)))
    orders.append(sorted(aliases, key=lambda alias: (cards.get(alias, 0.0), alias), reverse=True))

    hints: List[str] = []
    seen: Set[Tuple[str, ...]] = set()
    unique_orders = []
    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(order)
    for small_prefix in (1, 2, 0):
        for order in unique_orders:
            pieces = [_left_deep_leading(order), *_join_methods(order, cards, small_prefix)]
            hint = "/*+ " + " ".join(pieces) + " */"
            if is_valid_hint_for_aliases(hint, aliases):
                hints.append(hint)
            if len(hints) >= limit:
                return hints
    return hints
