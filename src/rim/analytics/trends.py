"""Topic time series and monotone-trend tests.

For topic c and year t, the share is s(c, t) = n(c, t) / n(t). Shares, not counts,
are analysed, because the corpus itself grows every year.

Trend in a share series is tested with the Mann-Kendall test (no distributional
assumption, robust to outliers) and its size estimated with the Theil-Sen slope, the
median of all pairwise slopes. P-values over topics are adjusted for the false
discovery rate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rim.evaluation.stats import benjamini_hochberg


def topic_year_counts(
    labels: np.ndarray, years: np.ndarray, k: int, year_range: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Counts as a (k, n_years) array, and the array of years."""
    first, last = year_range
    axis = np.arange(first, last + 1)
    years = np.asarray(years)
    keep = (years >= first) & (years <= last)
    counts = np.zeros((k, len(axis)), dtype=np.int64)
    np.add.at(counts, (np.asarray(labels)[keep], years[keep].astype(int) - first), 1)
    return counts, axis


def shares(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=0, keepdims=True)
    return counts / np.maximum(totals, 1)


def mann_kendall(series: np.ndarray) -> tuple[float, float, float]:
    """Mann-Kendall S statistic, normal score with tie and continuity corrections, two-sided p."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 1.0
    s = float(sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n)))
    _, tie_counts = np.unique(x, return_counts=True)
    variance = (
        n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
    ) / 18.0
    if variance <= 0:
        return s, 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return s, float(z), float(p)


def theil_sen(series: np.ndarray, times: np.ndarray | None = None) -> tuple[float, float]:
    """Median pairwise slope and the matching median intercept."""
    y = np.asarray(series, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64) if times is None else np.asarray(times, np.float64)
    i, j = np.triu_indices(len(y), k=1)
    valid = t[j] != t[i]
    if not valid.any():
        return 0.0, float(np.median(y)) if len(y) else 0.0
    slope = float(np.median((y[j] - y[i])[valid] / (t[j] - t[i])[valid]))
    return slope, float(np.median(y - slope * t))


def trend_table(
    counts: np.ndarray, years: np.ndarray, terms: list[list[str]] | None = None, fdr: float = 0.05
) -> pd.DataFrame:
    """One row per topic: share at both ends, Theil-Sen slope, Mann-Kendall test, q-value."""
    share = shares(counts)
    rows = []
    for topic in range(share.shape[0]):
        slope, _ = theil_sen(share[topic], years)
        _, z, p = mann_kendall(share[topic])
        rows.append(
            {
                "topic": topic,
                "n_documents": int(counts[topic].sum()),
                "share_first": float(share[topic, 0]),
                "share_last": float(share[topic, -1]),
                "slope_per_year": slope,
                "mk_z": z,
                "p_value": p,
            }
        )
    table = pd.DataFrame(rows)
    table["q_value"] = benjamini_hochberg(table["p_value"].tolist())
    significant = table["q_value"] <= fdr
    table["trend"] = np.where(
        significant & (table["mk_z"] > 0), "rising", np.where(significant, "falling", "none")
    )
    if terms is not None:
        table["terms"] = [", ".join(t) for t in terms]
    return table
