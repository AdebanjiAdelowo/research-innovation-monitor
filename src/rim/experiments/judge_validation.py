"""How well does an entailment judge agree with human evidence labels?

Citation metrics in the RAG experiment depend on a model deciding whether a passage
supports a sentence. This experiment measures that model against SciFact, where
experts labelled whether an abstract supports a claim.

The decision threshold is selected on the SciFact train claims (maximising balanced
accuracy) and all reported numbers are from the dev claims, which the threshold
selection never sees. The dev claims are the queries of the BEIR SciFact test split
used in the RAG experiment, so the judge is validated on the same domain it is used on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rim.config import StrictModel, data_dir
from rim.evaluation.stats import bootstrap_ci
from rim.experiments.common import finish_run, start_run
from rim.rag.attribution import EntailmentJudge, NLIJudge


class JudgeCandidate(StrictModel):
    name: str
    model_id: str
    revision: str | None = None
    max_length: int = 512


class JudgeValidationConfig(StrictModel):
    experiment: str = "judge_validation"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    judges: tuple[JudgeCandidate, ...]
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    n_bootstrap: int = 10_000


def classification_scores(probabilities: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": 1.0 - specificity,
    }


def validate_judge(
    judge: EntailmentJudge,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    thresholds: tuple[float, ...],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    train_probs = judge.entailment(train["premise"].tolist(), train["claim"].tolist())
    train_truth = train["entailed"].to_numpy()
    on_train = [classification_scores(train_probs, train_truth, t) for t in thresholds]
    threshold = max(on_train, key=lambda row: row["balanced_accuracy"])["threshold"]

    dev_probs = judge.entailment(dev["premise"].tolist(), dev["claim"].tolist())
    dev_truth = dev["entailed"].to_numpy()
    correct = ((dev_probs >= threshold) == dev_truth).astype(float)
    by_label = {
        label: float(np.mean(dev_probs[(dev["label"] == label).to_numpy()] >= threshold))
        for label in sorted(dev["label"].unique())
    }
    return {
        "selected_threshold": threshold,
        "train_threshold_grid": on_train,
        "dev": classification_scores(dev_probs, dev_truth, threshold),
        "dev_at_0.5": classification_scores(dev_probs, dev_truth, 0.5),
        "dev_accuracy_ci": list(bootstrap_ci(correct, n_bootstrap, seed=seed)),
        "dev_auroc": float(roc_auc_score(dev_truth, dev_probs)),
        "dev_share_judged_entailed_by_label": by_label,
        "n_dev": len(dev),
        "n_dev_by_label": dev["label"].value_counts().to_dict(),
    }, dev_probs


def run_judge_validation(cfg: JudgeValidationConfig, out_root: str | Path | None = None) -> dict:
    from rim.data.loaders.scifact_claims import load_pairs

    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    pairs = load_pairs(data_dir() / "raw")
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    scored = pairs["dev"][["claim_id", "doc_id", "label"]].copy()
    for candidate in cfg.judges:
        judge = NLIJudge(candidate.model_id, candidate.revision, device, candidate.max_length)
        results[candidate.name], probs = validate_judge(
            judge, pairs["train"], pairs["dev"], cfg.thresholds, cfg.n_bootstrap, cfg.seed
        )
        scored[candidate.name] = probs
        record.models.append({"role": "judge", "name": candidate.name, **judge.describe()})
        del judge
    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "source": "SciFact claims (train for threshold selection, dev for reporting)",
        "n_pairs": {k: len(v) for k, v in pairs.items()},
        "judges": results,
        "device": device,
    }
    scores_path = out_dir / "dev_scores.csv"
    scored.to_csv(scores_path, index=False, float_format="%.5f")
    finish_run(record, out_dir, summary, [scores_path])
    return summary
