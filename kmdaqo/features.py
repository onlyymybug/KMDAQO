from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}
SCAN_NODES = {
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Tid Scan",
    "Subquery Scan",
    "CTE Scan",
    "Function Scan",
    "Foreign Scan",
    "Values Scan",
}


def _extract_from_clause(sql: str) -> str:
    match = re.search(r"\bfrom\b(.*?)(?:\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", sql, flags=re.I | re.S)
    return match.group(1) if match else ""


def extract_aliases(sql: str) -> Dict[str, str]:
    from_clause = _extract_from_clause(sql)
    alias_to_table: Dict[str, str] = {}
    for part in from_clause.split(","):
        part = part.strip()
        if not part:
            continue
        join_head = re.split(r"\bjoin\b", part, flags=re.I)[0].strip()
        match = re.match(r"([a-zA-Z_][\w.]*)\s+(?:as\s+)?([a-zA-Z_]\w*)\b", join_head, flags=re.I)
        if match:
            table = match.group(1).split(".")[-1].lower()
            alias = match.group(2).lower()
            alias_to_table[alias] = table
            continue
        match = re.match(r"([a-zA-Z_][\w.]*)\b", join_head)
        if match:
            table = match.group(1).split(".")[-1].lower()
            alias_to_table[table] = table
    for match in re.finditer(r"\bjoin\s+([a-zA-Z_][\w.]*)(?:\s+(?:as\s+)?([a-zA-Z_]\w*))?", sql, flags=re.I):
        table = match.group(1).split(".")[-1].lower()
        alias = (match.group(2) or table).lower()
        alias_to_table[alias] = table
    return alias_to_table


def sql_text_features(sql: str) -> Dict[str, Any]:
    alias_to_table = extract_aliases(sql)
    tables = set(alias_to_table.keys())
    columns = set()
    for match in re.finditer(r"([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)", sql):
        columns.add(f"{match.group(1).lower()}.{match.group(2).lower()}")
    predicates = set()
    for match in re.finditer(r"\bwhere\b(.*)", sql, flags=re.I | re.S):
        predicates.update(re.findall(r"([a-zA-Z_]\w*\.[a-zA-Z_]\w*)\s*(?:=|<|>|like|in)\b", match.group(1), flags=re.I))
    return {
        "tables": sorted(tables),
        "alias_to_table": dict(sorted(alias_to_table.items())),
        "predicate_columns": sorted({p.lower() for p in predicates} or columns),
        "schema_tokens": sorted(tables | {c.split(".")[-1] for c in columns}),
    }


def extract_plan_features(explain_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    joins: List[str] = []
    scans: List[str] = []
    tables: Set[str] = set()
    operator_sequence: List[str] = []
    join_graph_edges: Set[str] = set()
    predicate_columns: Set[str] = set()
    alias_to_table: Dict[str, str] = {}
    table_cardinality: Dict[str, float] = {}
    filter_cardinality: Dict[str, float] = {}
    estimated_rows = 0.0
    actual_rows = 0.0

    def add_predicate_columns(value: Any) -> None:
        if not isinstance(value, str):
            return
        for alias, col in re.findall(r"([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)", value):
            predicate_columns.add(f"{alias.lower()}.{col.lower()}")

    def walk(node: Dict[str, Any]) -> Set[str]:
        nonlocal estimated_rows, actual_rows
        node_type = node.get("Node Type", "Unknown")
        operator_sequence.append(node_type)
        if node_type in JOIN_NODES:
            joins.append(node_type)
        if node_type in SCAN_NODES:
            scans.append(node_type)
        alias = node.get("Alias")
        relation_name = node.get("Relation Name")
        relation = alias or relation_name
        subtree_tables: Set[str] = set()
        if relation:
            table = str(relation).lower()
            tables.add(table)
            subtree_tables.add(table)
            if relation_name:
                alias_key = str(alias or relation_name).lower()
                alias_to_table[alias_key] = str(relation_name).lower()
                if node.get("Plan Rows") is not None:
                    table_cardinality.setdefault(alias_key, float(node.get("Plan Rows") or 0.0))
                has_filter = any(node.get(k) is not None for k in ("Filter", "Index Cond", "Recheck Cond"))
                if has_filter and node.get("Plan Rows") is not None:
                    filter_cardinality[alias_key] = float(node.get("Plan Rows") or 0.0)
        for key in ("Filter", "Index Cond", "Hash Cond", "Merge Cond", "Join Filter", "Recheck Cond"):
            add_predicate_columns(node.get(key))
        estimated_rows += float(node.get("Plan Rows") or 0.0)
        actual_rows += float(node.get("Actual Rows") or 0.0)
        child_sets = []
        for child in node.get("Plans", []) or []:
            child_sets.append(walk(child))
        for child_set in child_sets:
            subtree_tables |= child_set
        if node_type in JOIN_NODES and len(child_sets) >= 2:
            left, right = child_sets[0], set().union(*child_sets[1:])
            for l in left:
                for r in right:
                    join_graph_edges.add("--".join(sorted([l, r])))
        return subtree_tables

    root = explain_json.get("Plan") if isinstance(explain_json, dict) else None
    if isinstance(root, dict):
        walk(root)
    planning_time_ms = explain_json.get("Planning Time") if isinstance(explain_json, dict) else None
    execution_time_ms = explain_json.get("Execution Time") if isinstance(explain_json, dict) else None
    estimate_error = None
    if estimated_rows > 0 and actual_rows > 0:
        ratio = max(actual_rows / estimated_rows, estimated_rows / actual_rows)
        estimate_error = float(ratio)
    return {
        "joins": joins,
        "scans": scans,
        "tables": sorted(tables),
        "alias_to_table": dict(sorted(alias_to_table.items())),
        "table_cardinality": dict(sorted(table_cardinality.items())),
        "filter_cardinality": dict(sorted(filter_cardinality.items())),
        "join_graph": sorted(join_graph_edges),
        "predicate_columns": sorted(predicate_columns),
        "operator_sequence": operator_sequence,
        "planning_time_ms": planning_time_ms,
        "execution_time_ms": execution_time_ms,
        "estimate_error": estimate_error,
    }


def merge_features(sql: str, explain_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = sql_text_features(sql)
    plan = extract_plan_features(explain_json)
    merged = dict(plan)
    for key, value in text.items():
        if isinstance(value, dict):
            combined = dict(value)
            combined.update(merged.get(key, {}) if isinstance(merged.get(key), dict) else {})
            merged[key] = dict(sorted(combined.items()))
        else:
            merged[key] = sorted(set(merged.get(key, [])) | set(value))
    return merged
