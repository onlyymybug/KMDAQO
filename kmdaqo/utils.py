from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def read_sql(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    while text.endswith(";"):
        text = text[:-1].strip()
    return text


def list_sql_files(sql_dir: str | Path) -> List[Path]:
    return sorted(Path(sql_dir).glob("*.sql"), key=lambda p: p.name)


def normalize_sql(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--.*?$", " ", sql, flags=re.M)
    sql = re.sub(r"\s+", " ", sql).strip().lower()
    return sql.rstrip(";")


def fingerprint_sql(sql: str) -> str:
    normalized = normalize_sql(sql)
    normalized = re.sub(r"'[^']*'", "?", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "?", normalized)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def stable_case_id(sql_file: str, sql: str) -> str:
    return hashlib.sha1(f"{sql_file}:{fingerprint_sql(sql)}".encode("utf-8")).hexdigest()[:20]


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rewrite_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now_ms() -> int:
    return int(time.time() * 1000)


HINT_FUNCTION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)")
LEADING_RE = re.compile(r"\bLeading\s*\((.*)\)\s*$", re.S)
HINT_KEYWORDS = {
    "SeqScan",
    "IndexScan",
    "IndexOnlyScan",
    "BitmapScan",
    "BitmapHeapScan",
    "BitmapIndexScan",
    "HashJoin",
    "NestLoop",
    "MergeJoin",
    "Leading",
}


def is_valid_hint(hint: str | None) -> bool:
    if not hint:
        return False
    hint = hint.strip()
    return (
        hint.startswith("/*+")
        and hint.endswith("*/")
        and len(hint) <= 4096
        and hint.count("(") == hint.count(")")
    )


def hint_body(hint: str | None) -> str:
    if not is_valid_hint(hint):
        return ""
    return hint.strip()[3:-2].strip()


def is_hint_body_function_sequence(body: str | None) -> bool:
    body = (body or "").strip()
    if not body:
        return True
    pos = 0
    length = len(body)
    while pos < length:
        while pos < length and body[pos].isspace():
            pos += 1
        if pos >= length:
            break
        name_match = re.match(r"[A-Za-z][A-Za-z0-9_]*", body[pos:])
        if not name_match:
            return False
        name = name_match.group(0)
        if name not in HINT_KEYWORDS:
            return False
        pos += len(name)
        while pos < length and body[pos].isspace():
            pos += 1
        if pos >= length or body[pos] != "(":
            return False
        depth = 0
        while pos < length:
            char = body[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    pos += 1
                    break
            pos += 1
        if depth != 0:
            return False
    return True


def extract_hint_relations(hint: str | None) -> Set[str]:
    body = hint_body(hint)
    relations: Set[str] = set()
    for name, args in HINT_FUNCTION_RE.findall(body):
        if name not in HINT_KEYWORDS:
            continue
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", args):
            if token not in HINT_KEYWORDS:
                relations.add(token.lower())
    return relations


def extract_leading_relations(hint: str | None) -> List[str]:
    body = hint_body(hint)
    leading = re.search(r"\bLeading\s*\(", body)
    if not leading:
        return []
    start = leading.end()
    depth = 1
    chars: List[str] = []
    for char in body[start:]:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "".join(chars))]


def is_valid_hint_for_aliases(hint: str | None, aliases: Iterable[str]) -> bool:
    if not is_valid_hint(hint):
        return False
    if not is_hint_body_function_sequence(hint_body(hint)):
        return False
    alias_set = {alias.lower() for alias in aliases}
    if not alias_set:
        return True
    relations = extract_hint_relations(hint)
    if relations and not relations <= alias_set:
        return False
    leading = extract_leading_relations(hint)
    if leading:
        if set(leading) - alias_set:
            return False
        if len(leading) != len(set(leading)):
            return False
    for name, args in HINT_FUNCTION_RE.findall(hint_body(hint)):
        tokens = [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", args)]
        relation_tokens = [token for token in tokens if token not in {keyword.lower() for keyword in HINT_KEYWORDS}]
        if name in HINT_KEYWORDS and set(relation_tokens) - alias_set:
            return False
        if name in {"SeqScan", "IndexScan", "IndexOnlyScan", "BitmapScan", "BitmapHeapScan", "BitmapIndexScan"} and len(relation_tokens) != 1:
            return False
    return True


def translate_hint_aliases(hint: str | None, source_alias_to_table: Dict[str, str], target_alias_to_table: Dict[str, str]) -> Optional[str]:
    if not is_valid_hint(hint):
        return None
    table_to_target: Dict[str, str] = {}
    for alias, table in target_alias_to_table.items():
        table = str(table).lower()
        if table in table_to_target:
            table_to_target[table] = ""
        else:
            table_to_target[table] = str(alias).lower()
    replacements: Dict[str, str] = {}
    for alias, table in source_alias_to_table.items():
        target_alias = table_to_target.get(str(table).lower())
        if target_alias:
            replacements[str(alias).lower()] = target_alias
    body = hint_body(hint)
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body))
    relation_tokens = {token for token in tokens if token not in HINT_KEYWORDS}
    if any(token.lower() not in replacements and token.lower() not in {a.lower() for a in target_alias_to_table} for token in relation_tokens):
        return None

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return replacements.get(token.lower(), token)

    translated = "/*+ " + re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace, body) + " */"
    aliases = target_alias_to_table.keys()
    return translated if is_valid_hint_for_aliases(translated, aliases) else None


def attach_hint(sql: str, hint: str | None) -> str:
    sql = sql.strip()
    if not is_valid_hint(hint):
        return sql
    return f"{hint.strip()}\n{sql}"


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
