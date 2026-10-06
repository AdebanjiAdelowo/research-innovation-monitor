import json

import pandas as pd

from rim.config import RetrievalExperimentConfig
from rim.experiments.retrieval import format_table, run_retrieval_experiment

CONFIG = {
    "experiment": "toy_retrieval",
    "seed": 3,
    "device": "cpu",
    "dataset": {"name": "toy"},
    "bm25": {"k1_grid": [0.8, 1.2], "b_grid": [0.4, 0.75]},
    "encoders": [
        {"name": "hash64", "kind": "hashing", "dim": 64},
        {"name": "hash512", "kind": "hashing", "dim": 512},
    ],
    "fusion": {"alpha_grid": [0.0, 0.5, 1.0], "rrf_k_grid": [10, 60], "depth": 10},
    "rerankers": [{"name": "lex", "kind": "lexical", "depths": [2, 5]}],
    "evaluation": {
        "ks": [1, 5, 10],
        "n_bootstrap": 200,
        "n_permutations": 200,
        "n_latency_queries": 2,
    },
    "top_k": 10,
}


def run(tmp_path, toy_dataset, **overrides):
    cfg = RetrievalExperimentConfig.model_validate({**CONFIG, **overrides})
    return run_retrieval_experiment(cfg, toy_dataset, tmp_path / "out", tmp_path / "cache")


def test_experiment_runs_end_to_end_and_writes_traceable_outputs(tmp_path, toy_dataset):
    summary = run(tmp_path, toy_dataset)
    out = tmp_path / "out" / "toy_retrieval"

    assert summary["n_queries"] == {"dev": 3, "test": 3}
    assert {"bm25", "dense:hash64", "dense:hash512", "hybrid_rrf", "hybrid_convex"} <= set(
        summary["systems"]
    )
    assert any(name.endswith("+rerank:lex@5") for name in summary["systems"])
    for row in summary["systems"].values():
        assert 0.0 <= row["test"]["ndcg@10"] <= 1.0
        low, high = row["test_ci"]["ndcg@10"]
        assert low <= row["test"]["ndcg@10"] <= high

    selected = summary["selected_on_dev"]
    assert selected["dense"]["hybrid_encoder"] in ("hash64", "hash512")
    assert len(selected["bm25"]["dev_grid"]) == 4
    assert selected["fusion"]["reranker_input"] in ("hybrid_rrf", "hybrid_convex")
    assert all(0.0 < c["p_value"] <= c["p_holm"] <= 1.0 for c in summary["comparisons"])
    assert summary["latency_pipelines"]["bm25"]["median_ms"] > 0

    run_record = json.loads((out / "run.json").read_text())
    assert run_record["seed"] == 3 and run_record["config"]["experiment"] == "toy_retrieval"
    assert set(run_record["artifacts"]) == {"summary.json", "per_query_test.csv"}
    assert {m["role"] for m in run_record["models"]} == {"lexical", "encoder", "reranker"}

    per_query = pd.read_csv(out / "per_query_test.csv")
    assert set(per_query["query_id"]) == {"toy:q3", "toy:q4", "toy:q5"}
    means = per_query.groupby("system")["ndcg@10"].mean()
    assert abs(means["bm25"] - summary["systems"]["bm25"]["test"]["ndcg@10"]) < 1e-5
    assert "| bm25 |" in format_table(summary, ("ndcg@10", "recall@5"))


def test_selection_uses_dev_queries_only(tmp_path, toy_dataset):
    """Changing test judgments must not change any choice made during tuning."""
    baseline = run(tmp_path / "a", toy_dataset)
    altered = toy_dataset.qrels.copy()
    test_ids = set(toy_dataset.split_queries("test")["query_id"])
    mask = altered["query_id"].isin(test_ids)
    altered.loc[mask, "doc_id"] = "toy:d11"
    toy_dataset.qrels = altered.drop_duplicates(["query_id", "doc_id"])
    changed = run(tmp_path / "b", toy_dataset)
    assert changed["selected_on_dev"] == baseline["selected_on_dev"]
    assert changed["systems"]["bm25"]["test"] != baseline["systems"]["bm25"]["test"]


def test_results_are_reproducible(tmp_path, toy_dataset):
    first = run(tmp_path / "a", toy_dataset)
    second = run(tmp_path / "b", toy_dataset)
    assert first["systems"] == second["systems"]
    assert [c["p_value"] for c in first["comparisons"]] == [
        c["p_value"] for c in second["comparisons"]
    ]


def test_group_metrics(tmp_path, toy_dataset):
    toy_dataset.qrels = toy_dataset.qrels.assign(
        domain=["IN" if i % 2 == 0 else "OUT" for i in range(len(toy_dataset.qrels))]
    )
    summary = run(tmp_path, toy_dataset, dataset={"name": "toy", "qrels_group": "domain"})
    assert set(summary["groups"]) == {"IN", "OUT"}
    assert summary["groups"]["IN"]["n_queries"] >= 1
