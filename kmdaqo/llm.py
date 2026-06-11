from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import is_hint_body_function_sequence, is_valid_hint, is_valid_hint_for_aliases, translate_hint_aliases


OUTPUT_CONSTRAINT = """## Please provide a better hint in the same pg_hint_plan strategy style as Postgres's. Return ONLY pg_hint_plan hint body lines.
Do not include /*+, */, SQL, markdown, explanations, natural-language steps, or multiple alternatives.
The system will wrap your valid hint body lines into one final /*+ ... */ SQL hint comment.
Each alias may appear at most once in Leading; each scan hint has exactly one alias.
IMPORTANT: Leading must have an extra outer pair of parentheses and balanced brackets. For example:
Valid hint body line: Leading((((a b) c) d))
Invalid hint body line (missing extra outer parentheses): Leading(((a b) c) d)
Invalid hint body line (brackets do not match): Leading(((((a b) c) d))
Invalid hint body line (brackets do not match): Leading(((a b) c) d))"""

RAG_REFERENCE_CONSTRAINT = """## Reference guidance:
Retrieved cases are similar queries, not guaranteed to be the same query. Use them only for scan/join strategy.
Never copy a reference-only alias into the final hint."""


def load_domain_prompt(domain_file: str | Path) -> str:
    with open(domain_file, "r", encoding="utf-8") as f:
        domain_nl = f.read()
    return domain_nl

def _format_list(values: Any) -> str:
    if not values:
        return "unknown"
    if isinstance(values, (list, tuple, set)):
        return ", ".join(str(v) for v in values) or "unknown"
    return str(values)


def build_query_statistics(sql: str, features: Optional[Dict[str, Any]] = None) -> str:
    features = features or {}
    tables = features.get("tables") or []
    alias_to_table = features.get("alias_to_table") or {}
    table_cardinality = features.get("table_cardinality") or {}
    filter_cardinality = features.get("filter_cardinality") or {}
    predicates = features.get("predicate_columns") or []
    joins = features.get("joins") or []
    scans = features.get("scans") or []
    operators = features.get("operator_sequence") or []
    allowed_aliases = sorted(alias_to_table) or sorted(tables)

    table_lines = []
    for alias in sorted(alias_to_table) or tables:
        table = alias_to_table.get(alias, alias)
        cardinality = table_cardinality.get(alias, "unknown")
        table_lines.append(f"- {alias}/{table}: {cardinality}")
    if not table_lines:
        table_lines.append("- unknown: unknown")

    filter_lines = []
    for alias in sorted(filter_cardinality):
        table = alias_to_table.get(alias, alias)
        filter_lines.append(f"- {alias}/{table}: {filter_cardinality[alias]}")
    for pred in predicates:
        if not any(pred.startswith(f"{alias}.") for alias in filter_cardinality):
            filter_lines.append(f"- {pred}: unknown")
    if not filter_lines:
        filter_lines.append("- unknown: unknown")

    return "\n".join([
        "SQL:",
        sql.strip(),
        "",
        "Available Tables/Aliases:",
        "\n".join(table_lines),
        "",
        "Table Cardinality:",
        "\n".join(table_lines),
        "",
        "Filter Cardinality:",
        "\n".join(filter_lines),
        "",
        "Allowed Alias Tokens In Final Hint:",
        ", ".join(allowed_aliases) if allowed_aliases else "unknown",
        "",
        f"Join Operators: {_format_list(joins)}",
        f"Scan Operators: {_format_list(scans)}",
        f"Operator Sequence: {_format_list(operators)}",
    ])


def build_compact_query_statistics(
    sql: str,
    features: Optional[Dict[str, Any]] = None,
    include_sql: bool = True,
    final_hint_alias_rule: bool = False,
) -> str:
    features = features or {}
    tables = features.get("tables") or []
    alias_to_table = features.get("alias_to_table") or {}
    table_cardinality = features.get("table_cardinality") or {}
    filter_cardinality = features.get("filter_cardinality") or {}
    joins = features.get("joins") or []
    scans = features.get("scans") or []
    allowed_aliases = sorted(alias_to_table) or sorted(tables)

    alias_parts = []
    for alias in allowed_aliases:
        table = alias_to_table.get(alias, alias)
        card = table_cardinality.get(alias, "unknown")
        filt = filter_cardinality.get(alias)
        if filt is None:
            alias_parts.append(f"{alias}/{table}:{card}")
        else:
            alias_parts.append(f"{alias}/{table}:{card},filter={filt}")

    lines = []
    if include_sql:
        lines.extend(["SQL:", sql.strip()])
    lines.extend([
        "Aliases:",
        ", ".join(alias_parts) if alias_parts else "unknown",
        "Allowed Alias Tokens In Final Hint:",
        ", ".join(allowed_aliases) if allowed_aliases else "unknown",
    ])
    if final_hint_alias_rule:
        lines.append("Alias rule: Use only the aliases listed above in the final hint; do not invent aliases or reuse aliases from examples.")
    lines.extend([
        f"Join Operators: {_format_list(joins)}",
        f"Scan Operators: {_format_list(scans)}",
    ])
    return "\n".join(lines)


def build_postgres_hint(features: Optional[Dict[str, Any]] = None) -> str:
    features = features or {}
    hints: List[str] = []
    for op in features.get("operator_sequence") or []:
        normalized = str(op).replace(" ", "")
        if normalized in {"SeqScan", "IndexScan", "IndexOnlyScan", "BitmapHeapScan", "BitmapIndexScan", "HashJoin", "MergeJoin", "NestLoop"}:
            continue
    return "/*+ " + " ".join(hints).strip() + " */"


def normalize_model_hint_output(response_text: str, aliases: Optional[List[str]] = None) -> Optional[str]:
    action_sequence = response_text.strip()
    if not action_sequence:
        return None
    alias_set = aliases or []
    match = re.search(r"/\*\+.*?\*/", action_sequence, flags=re.S)
    if match:
        hint = re.sub(r"\s+", " ", match.group(0)).strip()
        valid = is_valid_hint_for_aliases(hint, alias_set) if alias_set else is_valid_hint(hint)
        return hint if valid else None
    action_sequence = re.sub(r"```(?:sql)?|```", "", action_sequence, flags=re.I).strip()
    if not action_sequence:
        return None
    if "/*+" in action_sequence or "*/" in action_sequence:
        return None
    if not is_hint_body_function_sequence(action_sequence):
        return None
    hint = "/*+ " + action_sequence + " */"
    hint = re.sub(r"\s+", " ", hint).strip()
    valid = is_valid_hint_for_aliases(hint, alias_set) if alias_set else is_valid_hint(hint)
    return hint if valid else None


def build_prompt_messages(
    domain_file: str | Path,
    question_nl: str,
    postgres_hint: str,
    use_rag: bool,
    reference_query_statistics: Optional[List[str]] = None,
    reference_optimal_hint: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    domain_nl = load_domain_prompt(domain_file)
    reference_query_statistics = reference_query_statistics or []
    reference_optimal_hint = reference_optimal_hint or []

    parts: List[str] = []
    if use_rag:
        n = min(len(reference_query_statistics), len(reference_optimal_hint))
        parts.append(f"## Here are {n} similar query-answer pairs you can reference:")
        parts.append(RAG_REFERENCE_CONSTRAINT)
        for idx, (stats, hint) in enumerate(zip(reference_query_statistics[:n], reference_optimal_hint[:n]), start=1):
            parts.extend([
                f"Reference {idx}:",
                stats,
                "Strategy hint:",
                hint,
                "",
            ])

    parts.extend([
        "## Here is the query and its corresponding statistics:",
        question_nl,
        "The hint provided by Postgres: ",
        postgres_hint,
        OUTPUT_CONSTRAINT,
    ])
    combined_query = "\n".join(parts)
    return [
        {"role": "system", "content": domain_nl},
        {"role": "user", "content": combined_query},
    ]


class HintGenerator:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        mock: bool = False,
        adapter_path: Optional[str] = None,
        domain_file: str = "prompts/IMDB/domain.nl",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.domain_file = domain_file
        self.device = device
        self.mock = mock
        self.generation_config = generation_config or {}
        self.tokenizer = None
        self.model = None
        if not mock:
            self._load()

    def _load(self) -> None:
        try:
            import os
            if self.device.startswith("cuda:"):
                os.environ["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            local_files_only = bool(self.generation_config.get("local_files_only", True))
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            dtype_name = str(self.generation_config.get("torch_dtype", "bfloat16")).lower()
            dtype = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "auto": "auto",
            }.get(dtype_name, torch.bfloat16)
            device_map: Any = self.generation_config.get("device_map", "auto")
            if isinstance(device_map, str) and device_map.lower() in {"none", "null"}:
                device_map = None
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=local_files_only,
            ).eval()
            if self.adapter_path and Path(self.adapter_path).exists():
                try:
                    from peft import PeftModel
                except Exception as exc:
                    raise RuntimeError("peft is required when models.llm_adapter_path is set") from exc
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.adapter_path,
                    local_files_only=local_files_only,
                ).eval()
            else:
                self.model = base_model
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.do_sample = bool(self.generation_config.get("do_sample", False))
                self.model.generation_config.temperature = float(self.generation_config.get("temperature", 1.0))
                self.model.generation_config.top_p = float(self.generation_config.get("top_p", 1.0))
        except Exception as exc:
            print(f"[WARN] LLM unavailable, using mock hint generator: {exc}")
            self.mock = True

    def build_prompt(
        self,
        sql: str,
        retrieved: List[Dict[str, Any]],
        query_features: Optional[Dict[str, Any]] = None,
        postgres_hint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        reference_query_statistics = []
        reference_optimal_hint = []
        target_aliases = (query_features or {}).get("alias_to_table") or {}
        for item in retrieved:
            row = item["row"]
            ref_hint = (row.get("hint") or "").strip()
            source_aliases = (row.get("plan_features") or {}).get("alias_to_table") or {}
            translated = translate_hint_aliases(ref_hint, source_aliases, target_aliases) if target_aliases else ref_hint
            if not row.get("accepted_for_rag") or not translated or translated == "/*+ */":
                continue
            reference_query_statistics.append(build_compact_query_statistics(row.get("sql") or "", row.get("plan_features") or {}, include_sql=False))
            reference_optimal_hint.append(translated)

        question_nl = build_compact_query_statistics(sql, query_features, include_sql=True, final_hint_alias_rule=True)
        return build_prompt_messages(
            domain_file=self.domain_file,
            question_nl=question_nl,
            postgres_hint=postgres_hint or build_postgres_hint(query_features),
            use_rag=bool(reference_optimal_hint),
            reference_query_statistics=reference_query_statistics,
            reference_optimal_hint=reference_optimal_hint,
        )

    def _mock_hint(self, retrieved: List[Dict[str, Any]]) -> str:
        for item in retrieved:
            hint = item["row"].get("hint")
            if is_valid_hint(hint):
                return hint
        return "/*+ */"

    def _best_retrieved_hint(self, retrieved: List[Dict[str, Any]], query_features: Optional[Dict[str, Any]]) -> Optional[str]:
        target_aliases = (query_features or {}).get("alias_to_table") or {}
        for item in retrieved:
            row = item["row"]
            if float(row.get("speedup") or 0.0) <= 1.0:
                continue
            source_aliases = (row.get("plan_features") or {}).get("alias_to_table") or {}
            translated = translate_hint_aliases(row.get("hint"), source_aliases, target_aliases)
            if translated and translated != "/*+ */":
                return translated
        return None

    def _extract_hint(self, text: str, query_features: Optional[Dict[str, Any]] = None) -> Optional[str]:
        aliases = list(((query_features or {}).get("alias_to_table") or {}).keys())
        return normalize_model_hint_output(text, aliases=aliases)

    def generate(
        self,
        sql: str,
        retrieved: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        query_features: Optional[Dict[str, Any]] = None,
        postgres_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mock:
            messages = self.build_prompt(sql, retrieved, query_features=query_features, postgres_hint=postgres_hint)
            fallback_retrieved_hint = self._best_retrieved_hint(retrieved, query_features)
            hint = fallback_retrieved_hint or self._mock_hint(retrieved)
            return {
                "hint": hint,
                "raw_parsed_hint": None,
                "fallback_retrieved_hint": fallback_retrieved_hint,
                "confidence": 0.4 if hint != "/*+ */" else 0.0,
                "raw": "mock",
                "prompt_messages": messages,
                "final_prompt": None,
            }
        messages = self.build_prompt(sql, retrieved, query_features=query_features, postgres_hint=postgres_hint)
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_device = getattr(self.model, "device", None)
        if input_device is None:
            input_device = next(self.model.parameters()).device
        inputs = self.tokenizer([text], return_tensors="pt").to(input_device)
        import torch
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.generation_config.get("max_new_tokens", 160)),
            "do_sample": bool(self.generation_config.get("do_sample", False)),
            "temperature": float(self.generation_config.get("temperature", 1.0)),
            "top_p": float(self.generation_config.get("top_p", 1.0)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            out = self.model.generate(**inputs, **generation_kwargs)
        generated = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        raw_parsed_hint = self._extract_hint(raw, query_features=query_features)
        fallback_retrieved_hint = None if raw_parsed_hint else self._best_retrieved_hint(retrieved, query_features)
        hint = raw_parsed_hint or fallback_retrieved_hint
        return {
            "hint": hint,
            "raw_parsed_hint": raw_parsed_hint,
            "fallback_retrieved_hint": fallback_retrieved_hint,
            "confidence": 0.75 if hint else 0.0,
            "raw": raw,
            "prompt_messages": messages,
            "final_prompt": text,
        }
