"""Rolling-origin backtest of topic-growth forecasts (RQ5).

For each cut-off year T:

1. Topics are fitted on documents dated T or earlier.
2. Documents after T are assigned to the fitted topics, never used for fitting.
3. Each predictor forecasts, from the share series up to T, the log growth
   g_h(c) = log((s(c, T+h) + e) / (s(c, T) + e)).
4. Forecasts are compared with the realised growth.

Predictors
----------
* ``persistence``: shares stay where they are, g = 0. It has no ranking, so it is
  scored on absolute error only.
* ``last_growth``: the latest one-year log growth continues for h years. Since the
  yearly total is common to all topics, this ranks topics exactly as growth in raw
  document counts does.
* ``linear_trend``: Theil-Sen slope of the log share over a trailing window, times h.
* ``ridge``: ridge regression on trend features, trained only on earlier cut-offs
  whose targets were already observable at T (cut-offs T' with T' + h <= T).

Scores per cut-off: Spearman correlation between forecast and realised growth,
precision of the predicted top decile against the realised top decile, and mean
absolute error.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from rim.analytics.trends import shares, theil_sen, topic_year_counts

FEATURES = ("last_growth", "slope", "acceleration", "volatility", "log_share")
PREDICTORS = ("persistence", "last_growth", "linear_trend", "ridge")


def smoothing(counts: np.ndarray) -> float:
    """Half a document in an average year, so empty topic-years have a finite log share."""
    return 0.5 / max(float(counts.sum(axis=0).mean()), 1.0)


def growth_features(counts: np.ndarray, window: int = 5) -> np.ndarray:
    """Features of each topic from counts up to the cut-off (last column). Shape (k, 5)."""
    eps = smoothing(counts)
    log_share = np.log(shares(counts) + eps)
    window = min(window, log_share.shape[1])
    recent = log_share[:, -window:]
    half = max(window // 2, 2)
    rows = []
    for series in recent:
        steps = np.diff(series)
        slope = theil_sen(series)[0]
        early, late = theil_sen(series[:half])[0], theil_sen(series[-half:])[0]
        rows.append(
            [
                steps[-1] if len(steps) else 0.0,
                slope,
                late - early,
                float(np.std(steps)) if len(steps) else 0.0,
                series[-1],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def realised_growth(counts_past: np.ndarray, counts_future: np.ndarray, horizon: int) -> np.ndarray:
    eps = smoothing(counts_past)
    now = shares(counts_past)[:, -1]
    then = shares(counts_future)[:, horizon - 1]
    return np.log((then + eps) / (now + eps))


def top_fraction_precision(
    predicted: np.ndarray, actual: np.ndarray, fraction: float = 0.1
) -> float:
    n = max(1, round(fraction * len(actual)))
    top_predicted = set(np.argsort(-predicted, kind="stable")[:n])
    top_actual = set(np.argsort(-actual, kind="stable")[:n])
    return len(top_predicted & top_actual) / n


def score_forecast(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    constant = np.allclose(predicted, predicted[0])
    return {
        "spearman": float("nan") if constant else float(spearmanr(predicted, actual).statistic),
        "precision_top10pct": float("nan")
        if constant
        else top_fraction_precision(predicted, actual),
        "mae": float(np.mean(np.abs(predicted - actual))),
    }


def rolling_backtest(
    fit_assign: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    years: np.ndarray,
    cutoffs: list[int],
    horizons: tuple[int, ...] = (1, 2),
    first_year: int | None = None,
    window: int = 5,
    ridge_alpha: float = 1.0,
    min_train_cutoffs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest.

    ``fit_assign(train_mask)`` fits topics on the masked documents and returns
    ``(labels for all documents, k)``. It is the only place where documents are used
    for fitting, which makes the temporal separation explicit and testable.

    Returns per-cut-off scores and the per-topic forecasts they were computed from.
    """
    years = np.asarray(years)
    first_year = int(years.min()) if first_year is None else first_year
    last_year = int(years.max())
    max_h = max(horizons)
    history: dict[int, dict] = {}
    score_rows, forecast_rows = [], []

    for cutoff in sorted(cutoffs):
        if cutoff + max_h > last_year:
            raise ValueError(f"cut-off {cutoff} leaves fewer than {max_h} years to evaluate")
        labels, k = fit_assign(years <= cutoff)
        past, _ = topic_year_counts(labels, years, k, (first_year, cutoff))
        future, _ = topic_year_counts(labels, years, k, (cutoff + 1, cutoff + max_h))
        features = growth_features(past, window)
        history[cutoff] = {
            "features": features,
            "targets": {h: realised_growth(past, future, h) for h in horizons},
        }
        for h in horizons:
            actual = history[cutoff]["targets"][h]
            forecasts = {
                "persistence": np.zeros(k),
                "last_growth": features[:, 0] * h,
                "linear_trend": features[:, 1] * h,
            }
            # Earlier cut-offs whose h-year outcome was known at this cut-off.
            usable = [c for c in history if c + h <= cutoff]
            if len(usable) >= min_train_cutoffs:
                x_train = np.vstack([history[c]["features"] for c in usable])
                y_train = np.concatenate([history[c]["targets"][h] for c in usable])
                mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-12
                model = Ridge(alpha=ridge_alpha).fit((x_train - mean) / std, y_train)
                forecasts["ridge"] = model.predict((features - mean) / std)
            for name, predicted in forecasts.items():
                score_rows.append(
                    {
                        "cutoff": cutoff,
                        "horizon": h,
                        "predictor": name,
                        "n_topics": k,
                        **score_forecast(predicted, actual),
                    }
                )
                forecast_rows.extend(
                    {
                        "cutoff": cutoff,
                        "horizon": h,
                        "predictor": name,
                        "topic": topic,
                        "predicted": float(p),
                        "actual": float(a),
                    }
                    for topic, (p, a) in enumerate(zip(predicted, actual, strict=True))
                )
    return pd.DataFrame(score_rows), pd.DataFrame(forecast_rows)
