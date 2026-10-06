"""Lead-lag relation between two topic time series (RQ6, exploratory).

Given yearly topic shares in two corpora over the same years, with documents of both
assigned to the same topics, the statistic at lag L is the mean over topics of the
correlation between yearly changes in corpus A at year t and in corpus B at year
t + L. A positive lag with positive correlation means A moves first.

Null hypothesis: the coupling is not topic-specific. It is simulated by pairing each
topic's A series with the B series of a randomly chosen topic. This keeps every
series intact (its autocorrelation and any corpus-wide movement) and breaks only the
topic correspondence. With short yearly series the test has little power, and a null
result is uninformative rather than evidence of independence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rim.evaluation.stats import benjamini_hochberg


def lagged_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation of a[t] with b[t + lag]. NaN if a series is constant."""
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_lagged_correlation(da: np.ndarray, db: np.ndarray, lag: int) -> float:
    values = [lagged_correlation(x, y, lag) for x, y in zip(da, db, strict=True)]
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def lead_lag_test(
    shares_a: np.ndarray,
    shares_b: np.ndarray,
    max_lag: int = 3,
    n_permutations: int = 2000,
    seed: int = 0,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Mean lagged correlation of yearly log-share changes, with a permutation p-value per lag."""
    if shares_a.shape != shares_b.shape:
        raise ValueError("both corpora need the same topics and years")
    da = np.diff(np.log(shares_a + eps), axis=1)
    db = np.diff(np.log(shares_b + eps), axis=1)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(db)) for _ in range(n_permutations)]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        observed = mean_lagged_correlation(da, db, lag)
        null = np.array([mean_lagged_correlation(da, db[p], lag) for p in permutations])
        exceed = int(np.sum(np.abs(null - np.nanmean(null)) >= abs(observed - np.nanmean(null))))
        rows.append(
            {
                "lag_years": lag,
                "mean_correlation": observed,
                "null_mean": float(np.nanmean(null)),
                "null_sd": float(np.nanstd(null)),
                "p_value": (exceed + 1) / (n_permutations + 1),
            }
        )
    table = pd.DataFrame(rows)
    table["q_value"] = benjamini_hochberg(table["p_value"].tolist())
    return table
