import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rim.analytics.store import TrendStore
from rim.api.app import build_state, create_app
from rim.rag.llm import ExtractiveProvider
from rim.retrieval.encoders import HashingEncoder


@pytest.fixture
def trends_dir(tmp_path):
    pd.DataFrame(
        {
            "topic": [0, 1, 2],
            "terms": ["neural operators, pde", "finite element, mesh", "protein, genome"],
            "trend": ["rising", "falling", "none"],
            "share_first": [0.1, 0.6, 0.3],
            "share_last": [0.5, 0.2, 0.3],
            "slope_per_year": [0.2, -0.2, 0.0],
            "mk_z": [3.0, -3.0, 0.0],
            "p_value": [0.001, 0.001, 1.0],
            "q_value": [0.002, 0.002, 1.0],
        }
    ).to_csv(tmp_path / "topic_trends.csv", index=False)
    pd.DataFrame({"topic": [0, 1, 2], "2020": [10, 60, 30], "2021": [50, 20, 30]}).to_csv(
        tmp_path / "topic_year_counts.csv", index=False
    )
    return tmp_path


@pytest.fixture
def client(toy_dataset, trends_dir):
    state = build_state(
        {"literature": toy_dataset.corpus},
        HashingEncoder(dim=512),
        ExtractiveProvider(),
        trends_dir,
    )
    return TestClient(create_app(state))


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["corpora"] == {"literature": 12}
    assert body["encoder"]["kind"] == "hashing" and body["generator"]["kind"] == "extractive"


def test_search_modes_and_validation(client):
    for mode in ("bm25", "dense", "hybrid"):
        body = client.post(
            "/search", json={"query": "reciprocal rank fusion", "mode": mode, "top_k": 3}
        ).json()
        assert body["hits"][0]["doc_id"] == "toy:d05" and len(body["hits"]) == 3
        assert body["hits"][0]["metadata"] == {"year": 2019}
    assert client.post("/search", json={"query": "x", "corpus": "patents"}).status_code == 404
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 1000}).status_code == 422


def test_ask_returns_sentences_traceable_to_passages(client):
    body = client.post("/ask", json={"question": "What do dual encoders embed?"}).json()
    passage_docs = {p["doc_id"] for p in body["passages"]}
    assert body["sentences"] and not body["abstained"]
    for sentence in body["sentences"]:
        assert sentence["citations"] and set(sentence["cited_doc_ids"]) <= passage_docs


def test_trends_endpoints(client):
    body = client.get("/trends", params={"direction": "rising"}).json()
    assert [t["topic"] for t in body["topics"]] == [0]
    assert client.get("/trends", params={"direction": "falling"}).json()["topics"][0]["topic"] == 1
    series = client.get("/trends/0").json()
    assert series["years"] == [2020, 2021] and series["counts"] == [10, 50]
    assert series["shares"] == [0.1, 0.5]
    assert client.get("/trends/99").status_code == 404
    assert client.get("/trends", params={"direction": "sideways"}).status_code == 422


def test_report_routes_tools_and_returns_python_computed_statistics(client):
    body = client.post(
        "/report",
        json={
            "request": "Find papers on neural operators for PDE and say if the topic is growing."
        },
    ).json()
    assert body["tools_called"] == ["literature", "trends"]
    assert body["trends"][0]["topic"] == 0 and body["trends"][0]["slope_per_year"] == 0.2
    assert body["completed"] and body["evidence"]


def test_trends_unavailable_without_experiment_output(toy_dataset):
    state = build_state(
        {"literature": toy_dataset.corpus}, HashingEncoder(dim=64), ExtractiveProvider()
    )
    client = TestClient(create_app(state))
    assert client.get("/trends").status_code == 503
    assert "trends" not in state.orchestrator.tools


def test_trend_store_matching(trends_dir):
    store = TrendStore(trends_dir)
    assert [t["topic"] for t in store.match("mesh generation for finite elements")] == [1]
    assert store.match("unrelated words") == []
    with pytest.raises(ValueError):
        store.top("sideways")
