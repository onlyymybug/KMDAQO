import sys
from types import SimpleNamespace

from kmdaqo.config import load_config
from kmdaqo.features import merge_features
from kmdaqo.llm import HintGenerator, build_prompt_messages, normalize_model_hint_output
from kmdaqo.metrics import metric_row, summarize
from kmdaqo.pipeline import KMDAQO
from kmdaqo.router import QueryRouter
from kmdaqo.utils import fingerprint_sql, is_valid_hint, is_valid_hint_for_aliases


def test_fingerprint_abstracts_literals():
    a = "SELECT * FROM title WHERE id = 1"
    b = " select * from title where id = 2 "
    assert fingerprint_sql(a) == fingerprint_sql(b)


def test_hint_parser():
    assert is_valid_hint("/*+ HashJoin(a b) */")
    assert not is_valid_hint("HashJoin(a b)")


def test_hint_alias_validation_rejects_unknown_and_repeated_leading_aliases():
    aliases = ["cn", "ct", "mc", "t"]
    assert is_valid_hint_for_aliases("/*+ SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t)) */", aliases)
    assert not is_valid_hint_for_aliases("/*+ SeqScan(it1) HashJoin(cn it1) Leading(((cn it1) t)) */", aliases)
    assert not is_valid_hint_for_aliases("/*+ Leading((((cn mc) cn) t)) */", aliases)
    assert not is_valid_hint_for_aliases("/*+ SeqScan(cn mc) Leading(((cn mc) t)) */", aliases)


def test_normalize_model_hint_output_accepts_first_valid_comment():
    aliases = ["cn", "mc", "t"]
    raw = "/*+ SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t)) */\n/*+ SeqScan(it1) */"
    assert normalize_model_hint_output(raw, aliases) == "/*+ SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t)) */"


def test_normalize_model_hint_output_accepts_unwrapped_multiline_hint_body():
    aliases = ["cn", "mc", "t"]
    raw = "SeqScan(cn)\nHashJoin(cn mc)\nLeading(((cn mc) t))"
    assert normalize_model_hint_output(raw, aliases) == "/*+ SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t)) */"


def test_normalize_model_hint_output_rejects_unknown_aliases_and_bad_fragments():
    aliases = ["cn", "ct", "mc", "t"]
    assert normalize_model_hint_output("/*+ SeqScan(mi) HashJoin(cn mi) Leading(((cn mi) t)) */", aliases) is None
    assert normalize_model_hint_output("/*+ SeqScan(it1) HashJoin(cn it1) Leading(((cn it1) t)) */", aliases) is None
    assert normalize_model_hint_output("/*+ final hint is Leading(((cn mc) t)) */", aliases) is None
    assert normalize_model_hint_output("final hint is Leading(((cn mc) t))", aliases) is None
    assert normalize_model_hint_output("SeqScan(it2)\nHashJoin(cn it2)\nLeading(((cn it2) t))", aliases) is None
    assert normalize_model_hint_output("/*+ Leading(((cn mc) t))", aliases) is None
    assert normalize_model_hint_output("SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t))", aliases) == "/*+ SeqScan(cn) HashJoin(cn mc) Leading(((cn mc) t)) */"


def test_prompt_with_rag_includes_reference_alias_warning(tmp_path):
    domain_file = tmp_path / "domain.nl"
    domain_file.write_text("domain prompt", encoding="utf-8")

    messages = build_prompt_messages(
        domain_file=domain_file,
        question_nl="Allowed Alias Tokens In Final Hint:\ncn, mc, t",
        postgres_hint="/*+ */",
        use_rag=True,
        reference_query_statistics=["Allowed Alias Tokens In Final Hint:\nit1, mi"],
        reference_optimal_hint=["/*+ Leading((it1 mi)) */"],
    )

    user_prompt = messages[1]["content"]
    assert "similar query-answer pairs" in user_prompt
    assert "similar queries, not guaranteed to be the same query" in user_prompt
    assert "Strategy hint:" in user_prompt
    assert "Never copy a reference-only alias" in user_prompt
    assert "Allowed Alias Tokens In Final Hint:\nit1, mi\nAlias rule:" not in user_prompt


def test_prompt_without_rag_omits_zero_reference_section(tmp_path):
    domain_file = tmp_path / "domain.nl"
    domain_file.write_text("domain prompt", encoding="utf-8")

    messages = build_prompt_messages(
        domain_file=domain_file,
        question_nl="Allowed Alias Tokens In Final Hint:\ncn, mc, t",
        postgres_hint="/*+ */",
        use_rag=False,
    )

    user_prompt = messages[1]["content"]
    assert "0 similar query-answer pairs" not in user_prompt
    assert "similar query-answer pairs" not in user_prompt
    assert "Allowed Alias Tokens In Final Hint" in user_prompt
    assert "Allowed Alias Tokens In Final Hint:\ncn, mc, t\nAlias rule: Use only the aliases listed above in the final hint; do not invent aliases or reuse aliases from examples." in user_prompt
    assert 'Use only the current query aliases listed in "Allowed Alias Tokens In Final Hint"' not in user_prompt
    assert "Return ONLY pg_hint_plan hint body lines" in user_prompt
    assert "Return ONLY one final pg_hint_plan SQL hint comment" not in user_prompt
    assert "Valid hint body line: Leading" in user_prompt


def test_domain_prompt_matches_sefrqo_hint_body_style():
    domain_prompt = open("prompts/IMDB/domain.nl", encoding="utf-8").read()

    assert "Return only pg_hint_plan hint body lines" in domain_prompt
    assert "Do not include \"/*+\", \"*/\"" in domain_prompt
    assert "Do not output natural-language Check, Decide, Pick, or Join steps" in domain_prompt
    assert "Check whether there are any remaining subtrees" not in domain_prompt
    assert "Pick one table from the candidate set" not in domain_prompt


def test_default_transformers_inference_config():
    cfg = load_config("configs/default.yaml")
    models = cfg["models"]

    assert models["max_new_tokens"] == 512
    assert models["do_sample"] is True
    assert models["temperature"] == 1.0
    assert models["top_p"] == 1.0
    assert models["torch_dtype"] == "bfloat16"
    assert models["device_map"] == "auto"
    assert models["local_files_only"] is True


def test_hint_generator_loads_local_bf16_auto_device_map_with_peft(monkeypatch, tmp_path):
    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer"] = (args, kwargs)
            return cls()

    class FakeModel:
        def __init__(self):
            self.generation_config = SimpleNamespace()

        def eval(self):
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["base_model"] = (args, kwargs)
            return FakeModel()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["peft"] = (args, kwargs)
            return FakeModel()

    fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
    fake_transformers = SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeAutoModel)
    fake_peft = SimpleNamespace(PeftModel=FakePeftModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    HintGenerator(
        "local-model",
        adapter_path=str(adapter_path),
        generation_config={
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "local_files_only": True,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    )

    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["base_model"][1]["torch_dtype"] == "bf16"
    assert calls["base_model"][1]["device_map"] == "auto"
    assert calls["base_model"][1]["local_files_only"] is True
    assert calls["peft"][1]["local_files_only"] is True


def test_hint_generator_uses_sampling_generation_kwargs(monkeypatch):
    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeIds:
        shape = (1, 2)

    class FakeInputs(dict):
        def __init__(self):
            super().__init__({"input_ids": FakeIds()})

        def to(self, device):
            self["device"] = device
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "prompt"

        def __call__(self, texts, return_tensors="pt"):
            return FakeInputs()

        def decode(self, generated, skip_special_tokens=True):
            return "SeqScan(t)\nLeading((t))"

    class FakeModel:
        device = "cuda:0"

        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [[1, 2, 3, 4]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=lambda: NoGrad()))
    generator = HintGenerator.__new__(HintGenerator)
    generator.mock = False
    generator.tokenizer = FakeTokenizer()
    generator.model = FakeModel()
    generator.generation_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    generator.build_prompt = lambda *args, **kwargs: [{"role": "user", "content": "prompt"}]
    generator._extract_hint = lambda raw, query_features=None: None
    generator._best_retrieved_hint = lambda retrieved, query_features=None: None

    generator.generate("select 1", [], query_features={})

    assert generator.model.kwargs["max_new_tokens"] == 512
    assert generator.model.kwargs["do_sample"] is True
    assert generator.model.kwargs["temperature"] == 1.0
    assert generator.model.kwargs["top_p"] == 1.0
    assert "num_return_sequences" not in generator.model.kwargs


def test_hint_generator_uses_compact_reference_prompt(tmp_path):
    domain_file = tmp_path / "domain.nl"
    domain_file.write_text("domain prompt", encoding="utf-8")
    generator = HintGenerator("unused", mock=True, domain_file=str(domain_file))
    sql = "SELECT * FROM company_type AS ct, movie_companies AS mc WHERE ct.id = mc.company_type_id"
    features = merge_features(sql, None)
    retrieved = [{
        "row": {
            "accepted_for_rag": True,
            "speedup": 1.2,
            "sql": "SELECT * FROM old_table AS old_alias WHERE old_alias.id = 1",
            "hint": "/*+ Leading((ct mc)) HashJoin(ct mc) */",
            "plan_features": {
                "alias_to_table": {"ct": "company_type", "mc": "movie_companies"},
                "table_cardinality": {"ct": 4, "mc": 100},
                "filter_cardinality": {},
                "joins": ["Hash Join"],
                "scans": ["Seq Scan", "Seq Scan"],
            },
        }
    }]

    user_prompt = generator.build_prompt(sql, retrieved, query_features=features)[1]["content"]

    assert "old_table" not in user_prompt
    assert "old_alias" not in user_prompt
    assert "Reference 1:" in user_prompt
    assert "Strategy hint:" in user_prompt
    assert "Allowed Alias Tokens In Final Hint" in user_prompt


def test_sql_features():
    features = merge_features("SELECT * FROM title t JOIN movie_info mi ON t.id = mi.movie_id WHERE t.id = 1")
    assert features["alias_to_table"]["t"] == "title"
    assert features["alias_to_table"]["mi"] == "movie_info"
    assert features["predicate_columns"]


def test_router_short_query():
    router = QueryRouter({"short_query_threshold_ms": 200, "min_predicted_gain": 1.05})
    decision = router.decide("abc", 10, 2.0)
    assert decision.route == "postgres"


def test_metrics_summary():
    row = metric_row("1a.sql", "postgres", 100, 200, 10, 0, False)
    summary = summarize([row], memory_size=1)
    assert summary["win_rate"] == 1.0


def test_optimize_sql_discards_llm_hint_when_candidate_slower_than_baseline():
    class FakeDB:
        mock = True

        def __init__(self):
            self.hints = []
            self.timeout_values = []

        def explain_analyze(self, sql, hint=None):
            self.hints.append(hint)
            latency = 100.0 if hint is None else 150.0
            return SimpleNamespace(
                execution_time_ms=latency,
                error=None,
                raw_plan={"Plan": {}},
                plan_features=merge_features(sql, None),
            )

        def set_statement_timeout(self, value):
            self.timeout_values.append(value)

    class FakeEmbedder:
        def embed(self, values):
            return [[0.0, 0.0, 0.0] for _ in values]

    class FakeKB:
        rows = []

        def search(self, *args, **kwargs):
            return []

        def insert_case(self, *args, **kwargs):
            return True

    class FakeLLM:
        def generate(self, *args, **kwargs):
            return {
                "hint": "/*+ SeqScan(t) */",
                "confidence": 0.75,
                "raw": "/*+ SeqScan(t) */",
                "prompt_messages": [],
                "final_prompt": None,
            }

    system = KMDAQO.__new__(KMDAQO)
    system.cfg = {
        "optimizer": {
            "measure_current_baseline": True,
            "short_query_threshold_ms": 0,
            "min_confidence": 0.35,
            "verify_candidate_hints": True,
            "max_candidate_hints": 0,
            "candidate_timeout_ms": 5000,
        },
        "milvus": {"top_k": 0},
        "postgres": {"statement_timeout_ms": 120000},
    }
    system.baselines = {}
    system.db = FakeDB()
    system.embedder = FakeEmbedder()
    system.kb = FakeKB()
    system.llm = FakeLLM()
    system.router = QueryRouter(system.cfg["optimizer"])
    system.drift = SimpleNamespace(observe=lambda row: None)

    result = system.optimize_sql("SELECT * FROM title AS t WHERE t.id = 1", sql_file="q.sql", execute=True, use_rag=True)

    assert result["hint"] is None
    assert result["metric"]["fallback"] is True
    assert system.db.hints == [None, "/*+ SeqScan(t) */"]


def test_optimize_sql_keeps_raw_parsed_and_retrieved_fallback_hints_separate():
    class FakeDB:
        mock = True

        def explain_analyze(self, sql, hint=None):
            return SimpleNamespace(
                execution_time_ms=100.0,
                error=None,
                raw_plan={"Plan": {}},
                plan_features=merge_features(sql, None),
            )

        def set_statement_timeout(self, value):
            pass

    class FakeEmbedder:
        def embed(self, values):
            return [[0.0, 0.0, 0.0] for _ in values]

    class FakeKB:
        rows = []

        def search(self, *args, **kwargs):
            return []

        def insert_case(self, *args, **kwargs):
            return True

    class FakeLLM:
        def generate(self, *args, **kwargs):
            return {
                "hint": "/*+ SeqScan(t) */",
                "raw_parsed_hint": None,
                "fallback_retrieved_hint": "/*+ SeqScan(t) */",
                "confidence": 0.75,
                "raw": "SeqScan(it1)",
                "prompt_messages": [],
                "final_prompt": None,
            }

    system = KMDAQO.__new__(KMDAQO)
    system.cfg = {
        "optimizer": {
            "measure_current_baseline": False,
            "short_query_threshold_ms": 0,
            "min_confidence": 0.35,
            "verify_candidate_hints": False,
        },
        "milvus": {"top_k": 0},
        "postgres": {"statement_timeout_ms": 120000},
    }
    system.baselines = {"q.sql": 200.0}
    system.db = FakeDB()
    system.embedder = FakeEmbedder()
    system.kb = FakeKB()
    system.llm = FakeLLM()
    system.router = QueryRouter(system.cfg["optimizer"])
    system.drift = SimpleNamespace(observe=lambda row: None)

    result = system.optimize_sql("SELECT * FROM title AS t WHERE t.id = 1", sql_file="q.sql", execute=True, use_rag=True)

    assert result["parsed_hint"] is None
    assert result["fallback_retrieved_hint"] == "/*+ SeqScan(t) */"
    assert result["hint"] == "/*+ SeqScan(t) */"
