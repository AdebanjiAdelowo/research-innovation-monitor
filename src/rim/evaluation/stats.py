"""Uncertainty and significance for per-query metric comparisons.

* Confidence intervals: percentile bootstrap over queries.
* Paired comparison of two systems on the same queries: Fisher randomisation
  (sign-flip) test on the per-query differences, two-sided (Smucker et al., 2007).
* Several comparisons: Holm step-down control of the family-wise error rate;
  Benjamini-Hochberg control of the false discovery rate for exploratory families.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    block = 1000
    for start in range(0, n_resamples, block):
        size = min(block, n_resamples - start)
        idx = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Interval for mean(a - b), resampling queries so the pairing is kept."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    return bootstrap_ci(a - b, n_resamples, confidence, seed)


def paired_randomization_test(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for H0: the systems are exchangeable on each query.

    Under H0 the sign of every per-query difference is arbitrary. The p-value is the
    share of random sign assignments whose absolute mean difference is at least the
    observed one, with the add-one correction that keeps a Monte Carlo p-value valid.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.ndim != 1:
        raise ValueError("expected one value per query")
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        return 1.0
    observed = abs(nonzero.sum())
    rng = np.random.default_rng(seed)
    hits = 0
    block = 2000
    for start in range(0, n_permutations, block):
        size = min(block, n_permutations - start)
        signs = rng.integers(0, 2, size=(size, nonzero.size), dtype=np.int8) * 2 - 1
        hits += int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - 1e-12))
    return (hits + 1) / (n_permutations + 1)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * m / (rank + 1))
        adjusted[index] = running
    return adjusted.tolist()
