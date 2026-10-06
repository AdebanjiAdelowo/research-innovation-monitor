"""How reproducible is a clustering?

A topic that appears only for one random seed, or only for one sample of the corpus,
is not a property of the data. Two checks are provided:

* seed stability: agreement between partitions fitted with different seeds;
* subsample stability: agreement, on shared documents, between partitions fitted on
  different random subsamples.

Agreement is measured with the adjusted Rand index (chance-corrected pair agreement)
and adjusted mutual information.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

from rim.analytics.topics import fit_topics


def pairwise_agreement(partitions: list[np.ndarray]) -> dict[str, float]:
    ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(partitions, 2)]
    ami = [adjusted_mutual_info_score(a, b) for a, b in itertools.combinations(partitions, 2)]
    return {
        "ari_mean": float(np.mean(ari)),
        "ari_min": float(np.min(ari)),
        "ami_mean": float(np.mean(ami)),
        "n_pairs": len(ari),
    }


def seed_stability(
    embeddings: np.ndarray, k: int, seeds: tuple[int, ...] = (0, 1, 2, 3, 4), **fit_kwargs
) -> dict[str, float]:
    partitions = [fit_topics(embeddings, k, seed, **fit_kwargs)[1] for seed in seeds]
    return pairwise_agreement(partitions)


def subsample_stability(
    embeddings: np.ndarray,
    k: int,
    n_repeats: int = 5,
    fraction: float = 0.8,
    seed: int = 0,
    **fit_kwargs,
) -> dict[str, float]:
    """Fit on random subsamples and compare the labels each model gives to all documents."""
    rng = np.random.default_rng(seed)
    partitions = []
    for repeat in range(n_repeats):
        sample = rng.choice(len(embeddings), int(fraction * len(embeddings)), replace=False)
        model, _ = fit_topics(embeddings[sample], k, seed + repeat, **fit_kwargs)
        partitions.append(model.assign(embeddings))
    return pairwise_agreement(partitions)


def scan_k(
    embeddings: np.ndarray,
    ks: tuple[int, ...],
    seeds: tuple[int, ...] = (0, 1, 2),
    silhouette_sample: int = 5000,
    seed: int = 0,
    **fit_kwargs,
) -> list[dict]:
    """Seed stability and silhouette for each candidate number of topics."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(embeddings), min(silhouette_sample, len(embeddings)), replace=False)
    rows = []
    for k in ks:
        fits = [fit_topics(embeddings, k, s, **fit_kwargs) for s in seeds]
        model, labels = fits[0]
        row = {"k": k, **pairwise_agreement([labels for _, labels in fits])}
        row["silhouette"] = float(
            silhouette_score(model.project(embeddings[sample]), labels[sample], metric="cosine")
        )
        row["smallest_topic_share"] = float(np.bincount(labels, minlength=k).min() / len(labels))
        rows.append(row)
    return rows
