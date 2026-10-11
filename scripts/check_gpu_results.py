"""Check imported session-1 records and reproduce scores without model inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rim.config import RetrievalExperimentConfig, config_hash
from rim.data.schema import RetrievalDataset
from rim.evaluation.metrics import evaluate
from rim.experiments.judge_validation import JudgeValidationConfig, classification_scores
from rim.results_import import check_run, commit_in_repo

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2fe9d3136c99b067e71ccfd40aa7719853b4c8a5"


def main() -> None:
    session = json.loads(
        next((ROOT / "results/gpu_sessions").glob("2fe9d3136c99-*.json")).read_text()
    )
    assert session["commit"] == COMMIT and session["tree_clean_at_end"] is True
    assert session["stages_requested"] == ["judge", "retrieval"]
    assert all(s["status"] == "completed" for s in session["stages"])
    expected = {n for s in session["stages"] for n in s["results"]}
    assert len(expected) == 5
    for name in sorted(expected):
        directory = ROOT / "results" / name
        record = check_run(directory, commit_in_repo(ROOT), COMMIT)
        judge = name == "judge_validation_gpu"
        config_path = (
            "configs/rag/judge_validation_gpu.yaml"
            if judge
            else f"configs/retrieval/{name.removeprefix('retrieval_')}.yaml"
        )
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{config_path}"], cwd=ROOT)
        cls = JudgeValidationConfig if judge else RetrievalExperimentConfig
        cfg = cls.model_validate(yaml.safe_load(raw))
        assert record["config"] == cfg.model_dump(mode="json")
        assert record["config_sha256"] == config_hash(cfg)
        assert record["finished_at"]
        assert record["environment"]["hardware"]["device"] == "cuda"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["run_id"] == record["run_id"]
        if judge:
            scores = pd.read_csv(directory / "dev_scores.csv")
            truth = (scores.label == "SUPPORT").to_numpy()
            assert len(scores) == summary["n_pairs"]["dev"]
            for model, row in summary["judges"].items():
                observed = classification_scores(
                    scores[model].to_numpy(), truth, row["selected_threshold"]
                )
                for metric, value in observed.items():
                    assert np.isclose(value, row["dev"][metric], atol=1e-12)
            print(
                f"{name}: checksums/configuration/classification pass; dataset fingerprint absent"
            )
            continue
        scores = pd.read_csv(directory / "per_query_test.csv")
        for system, row in summary["systems"].items():
            per_system = scores[scores.system == system]
            assert len(per_system) == summary["n_queries"]["test"]
            assert not per_system.query_id.duplicated().any()
            for metric, value in row["test"].items():
                assert np.isclose(per_system[metric].mean(), value, atol=5.1e-7, rtol=0)
        for comparison in summary["comparisons"]:
            metric = comparison["metric"]
            difference = (
                summary["systems"][comparison["system"]]["test"][metric]
                - summary["systems"][comparison["baseline"]]["test"][metric]
            )
            assert np.isclose(difference, comparison["difference"], atol=1e-12)
        local = ROOT / "results" / name.removesuffix("_gpu")
        old = json.loads((local / "run.json").read_text())["datasets"][0]
        dataset = record["datasets"][0]
        assert dataset["revision"] == old["revision"]
        baseline = pd.read_csv(local / "per_query_test.csv")
        for system in ("bm25", "dense:minilm-l6", "dense:bge-small"):
            a = scores[scores.system == system].set_index("query_id").sort_index()
            b = baseline[baseline.system == system].set_index("query_id").sort_index()
            assert a.index.equals(b.index)
            assert np.array_equal(
                a[["ndcg@10", "mrr@10", "recall@10"]], b[["ndcg@10", "mrr@10", "recall@10"]]
            )
        # The archive retains raw rankings locally; these are deliberately Git-ignored.
        raw_path = directory / "runs_raw/rankings_test.json"
        data_path = ROOT / "data/processed" / dataset["name"]
        if raw_path.exists() and data_path.exists():
            qrels = RetrievalDataset.load(data_path, verify=True).qrels_dict("test")
            for system, ranking in json.loads(raw_path.read_text()).items():
                reproduced = evaluate(ranking, qrels, tuple(cfg.evaluation.ks))
                stored = scores[scores.system == system].set_index("query_id")
                assert set(reproduced.index) == set(stored.index)
                assert np.allclose(
                    reproduced,
                    stored.loc[reproduced.index, reproduced.columns],
                    atol=5.1e-7,
                    rtol=0,
                )
            print(f"{name}: raw rankings reproduce all metrics against local qrels")
        else:
            print(f"{name}: raw-ranking check unavailable (local archive/data required)")
        print(
            f"{name}: checksums/configuration/means/shared baselines pass; "
            f"Parquet bytes match local: {dataset['files'] == old['files']}"
        )


if __name__ == "__main__":
    main()
