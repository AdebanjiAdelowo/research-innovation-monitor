import pytest

from rim.agents.decompose import (
    LLMDecomposer,
    SentenceDecomposer,
    decomposed_retrieve,
    parse_subqueries,
)
from rim.agents.orchestrator import Orchestrator, SearchTool, ToolResult
from rim.agents.router import (
    EmbeddingRouter,
    LLMRouter,
    RuleRouter,
    parse_tool_list,
    routing_metrics,
)
from rim.agents.tasks import TEMPLATES, TOOLS, build_tasks
from rim.rag.citations import ABSTAIN
from rim.rag.llm import ExtractiveProvider, ScriptedProvider
from rim.retrieval.encoders import HashingEncoder
from rim.retrieval.searcher import HybridSearcher

# ----------------------------------------------------------------------- tasks


def test_tasks_are_deterministic_balanced_and_disjoint_across_splits():
    tasks = build_tasks()
    assert tasks == build_tasks()
    assert len(tasks) == 2 * 15 * len(TEMPLATES)
    dev = {t.request for t in tasks if t.split == "dev"}
    test = {t.request for t in tasks if t.split == "test"}
    assert not dev & test and len(dev) == len(test) == 120
    assert all(t.tools <= set(TOOLS) and t.tools for t in tasks)
    for dev_templates, test_templates in TEMPLATES.values():
        assert not set(dev_templates) & set(test_templates)


# --------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "request_text,expected",
    [
        ("Which papers propose methods for federated learning?", {"literature"}),
        ("Find patents that claim solid-state batteries.", {"patents"}),
        ("Which NSF awards fund work on neural operators?", {"grants"}),
        ("Show research awards supporting carbon capture.", {"grants"}),
        ("Is quantum error correction an emerging topic?", {"trends"}),
        ("Compare the academic papers and the patents on CRISPR.", {"literature", "patents"}),
        ("Tell me about sparse matrix factorisation.", {"literature"}),
    ],
)
def test_rule_router(request_text, expected):
    assert RuleRouter().route(request_text) == frozenset(expected)


def test_embedding_router_learns_from_dev_tasks():
    tasks = build_tasks()
    dev = [t for t in tasks if t.split == "dev"]
    router = EmbeddingRouter(HashingEncoder(dim=512)).fit(dev)
    predicted = [router.route(t.request) for t in dev]
    assert routing_metrics(predicted, [t.tools for t in dev])["f1"] > 0.95
    assert router.route("anything at all")  # never returns an empty set


def test_tool_list_parsing():
    assert parse_tool_list('["literature", "trends"]') == {"literature", "trends"}
    assert parse_tool_list('Sure! Tools: ["Patents"] because...') == {"patents"}
    assert parse_tool_list("[literature, grants]") == {"literature", "grants"}
    assert parse_tool_list('["weather"]') is None
    assert parse_tool_list("I cannot decide") is None


def test_llm_router_counts_unparseable_replies_and_falls_back():
    router = LLMRouter(ScriptedProvider(['["patents","trends"]', "no idea"]))
    assert router.route("a") == {"patents", "trends"}
    assert router.route("b") == {"literature"}
    assert router.n_unparseable == 1 and router.completion_tokens > 0


def test_routing_metrics_by_hand():
    expected = [frozenset({"literature"}), frozenset({"patents", "trends"}), frozenset({"grants"})]
    predicted = [
        frozenset({"literature"}),
        frozenset({"patents", "literature"}),
        frozenset({"trends"}),
    ]
    m = routing_metrics(predicted, expected)
    # tp = 1 + 1 + 0, fp = 0 + 1 + 1, fn = 0 + 1 + 1
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["precision"] == m["recall"] == m["f1"] == pytest.approx(0.5)
    assert m["unnecessary_calls_per_request"] == pytest.approx(2 / 3)
    assert m["all_needed_tools_called"] == pytest.approx(1 / 3)


# --------------------------------------------------------------- decomposition


def test_subquery_parsing_and_sentence_baseline():
    text = '1. neural operators for PDEs\n- "error bounds for operator learning"\n\nx\n3) neural operators for PDEs'
    assert parse_subqueries(text) == [
        "neural operators for PDEs",
        "error bounds for operator learning",
    ]
    dec = SentenceDecomposer()
    assert dec.decompose("One short question about retrieval?") == []
    parts = dec.decompose(
        "We need models that solve PDEs quickly. They should also quantify uncertainty well."
    )
    assert len(parts) == 2
    llm = LLMDecomposer(ScriptedProvider(["a b c\nd e f\ng h i\nj k l\nm n o"]))
    assert len(llm.decompose("q")) == 4


def test_decomposed_retrieval_fuses_lists_and_reduces_to_baseline_without_subqueries():
    lists = {
        "main": [("a", 3.0), ("b", 2.0), ("c", 1.0)],
        "s1": [("d", 9.0), ("a", 1.0)],
        "s2": [("d", 5.0), ("b", 1.0)],
    }
    assert decomposed_retrieve("main", [], lists.__getitem__) == lists["main"]
    fused = decomposed_retrieve("main", ["s1", "s2"], lists.__getitem__, rrf_k=1)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 2 + 1 / 3)
    assert scores["d"] == pytest.approx(1 / 2 + 1 / 2)
    assert [d for d, _ in fused][:2] == ["d", "a"]


# ---------------------------------------------------------------- orchestrator


class TrendTool:
    name = "trends"

    def run(self, request):
        return ToolResult("trends", table=[{"topic": 3, "slope_per_year": 0.004, "q_value": 0.01}])


class BrokenTool:
    name = "patents"

    def run(self, request):
        raise RuntimeError("index unavailable")


@pytest.fixture
def tools(toy_dataset):
    searcher = HybridSearcher.build(toy_dataset.corpus, HashingEncoder(dim=512), depth=12)
    return {"literature": SearchTool("literature", searcher, top_k=3), "trends": TrendTool()}


def test_orchestrator_runs_only_routed_tools_and_keeps_numbers_out_of_the_model(tools):
    provider = ScriptedProvider(["Dual encoders embed questions and passages [1]."])
    orchestrator = Orchestrator(RuleRouter(), tools, provider)

    report = orchestrator.run(
        "Find papers on dual encoders for passages and say if the topic is growing."
    )
    assert report.tools_called == ["literature", "trends"]
    assert report.trend_rows == [{"topic": 3, "slope_per_year": 0.004, "q_value": 0.01}]
    assert "0.004" not in provider.prompts[0]  # statistics are not sent through the model
    assert report.sentences[0]["cited_doc_ids"] == [report.passages[0].doc_id]
    assert report.passages[0].title.startswith("(literature) ")
    assert report.completed and report.to_dict()["completed"]

    only_trend = orchestrator.run("Is this an emerging topic?")
    assert only_trend.tools_called == ["trends"] and only_trend.text == "" and only_trend.completed
    assert len(provider.prompts) == 1  # no generation without passages


def test_orchestrator_survives_tool_failure_and_reports_it(tools):
    tools["patents"] = BrokenTool()
    orchestrator = Orchestrator(RuleRouter(), tools, ExtractiveProvider())
    report = orchestrator.run("Find patents that claim rank fusion.")
    assert report.tools_called == ["patents"]
    assert report.to_dict()["errors"] == {"patents": "RuntimeError: index unavailable"}
    assert report.text == ABSTAIN and not report.completed
