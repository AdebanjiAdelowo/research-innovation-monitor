import numpy as np
import pytest
from scipy import stats as scipy_stats

from rim.analytics.backtest import (
    growth_features,
    rolling_backtest,
    score_forecast,
    top_fraction_precision,
)
from rim.analytics.leadlag import lagged_correlation, lead_lag_test
from rim.analytics.stability import scan_k, seed_stability, subsample_stability
from rim.analytics.topics import fit_topics, topic_terms
from rim.analytics.trends import mann_kendall, shares, theil_sen, topic_year_counts, trend_table


def planted_corpus(
    k=6, dim=24, per_year=600, years=range(2010, 2022), seed=0, noise=0.08, max_rate=0.25
):
    """Documents drawn around k planted directions, with topic shares that drift linearly in
    log space: topic 0 rises fastest, topic k-1 falls fastest."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rates = np.linspace(max_rate, -max_rate, k)
    embeddings, labels, doc_years = [], [], []
    for year in years:
        weights = np.exp(rates * (year - years[0]))
        topics = rng.choice(k, size=per_year, p=weights / weights.sum())
        embeddings.append(centers[topics] + noise * rng.normal(size=(per_year, dim)))
        labels.append(topics)
        doc_years.append(np.full(per_year, year))
    x = np.vstack(embeddings).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.concatenate(labels), np.concatenate(doc_years), rates


# ---------------------------------------------------------------------- topics


def test_clustering_recovers_planted_topics_and_assigns_unseen_documents():
    x, truth, _, _ = planted_corpus()
    model, labels = fit_topics(x[:5000], k=6, seed=0, pca_dim=10)
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(truth[:5000], labels) > 0.95
    assert adjusted_rand_score(truth[5000:], model.assign(x[5000:])) > 0.95
    np.testing.assert_array_equal(labels, model.assign(x[:5000]))
    np.testing.assert_array_equal(labels, fit_topics(x[:5000], k=6, seed=0, pca_dim=10)[1])
    with pytest.raises(ValueError):
        fit_topics(x[:5], k=10)


def test_topic_terms_describe_each_cluster():
    texts = ["graph neural network molecules"] * 5 + ["bayesian inference sampling posterior"] * 5
    labels = np.array([0] * 5 + [1] * 5)
    terms = topic_terms(texts, labels, k=2, top_n=3)
    assert set(terms[0]) <= {
        "graph",
        "neural",
        "network",
        "molecules",
        "graph neural",
        "neural network",
        "network molecules",
    }
    assert any("bayesian" in t or "posterior" in t for t in terms[1])


def test_stability_separates_real_structure_from_noise():
    x, _, _, _ = planted_corpus(per_year=200)
    structured = seed_stability(x, k=6, seeds=(0, 1, 2), pca_dim=10)
    noise = np.random.default_rng(0).normal(size=x.shape).astype(np.float32)
    unstructured = seed_stability(noise, k=6, seeds=(0, 1, 2), pca_dim=10)
    assert structured["ari_mean"] > 0.95 and unstructured["ari_mean"] < 0.5
    assert subsample_stability(x, k=6, n_repeats=3, pca_dim=10)["ari_mean"] > 0.95
    scan = {
        row["k"]: row
        for row in scan_k(x, ks=(3, 6, 12), seeds=(0, 1), pca_dim=10, silhouette_sample=800)
    }
    assert scan[6]["silhouette"] == max(r["silhouette"] for r in scan.values())


# ---------------------------------------------------------------------- trends


def test_counts_and_shares():
    labels = np.array([0, 0, 1, 1, 1, 0])
    years = np.array([2020, 2020, 2020, 2021, 2021, 2030])
    counts, axis = topic_year_counts(labels, years, k=2, year_range=(2020, 2021))
    np.testing.assert_array_equal(counts, [[2, 0], [1, 2]])
    np.testing.assert_array_equal(axis, [2020, 2021])
    np.testing.assert_allclose(shares(counts), [[2 / 3, 0.0], [1 / 3, 1.0]])


def test_mann_kendall_matches_kendall_tau_and_handles_ties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=15) + 0.2 * np.arange(15)
        s, z, p = mann_kendall(x)
        tau = scipy_stats.kendalltau(np.arange(15), x)
        assert s == pytest.approx(tau.statistic * 15 * 14 / 2)
        assert np.sign(z) == np.sign(tau.statistic)
    s, z, p = mann_kendall(np.arange(10.0))
    assert s == 45 and p < 0.001
    assert mann_kendall(np.ones(8)) == (0.0, 0.0, 1.0)
    # Hand-computed with one tie: x = [1, 2, 2, 3] gives S = 5, Var = (4*3*13 - 2*1*9)/18 = 23/3.
    s, z, _ = mann_kendall(np.array([1.0, 2.0, 2.0, 3.0]))
    assert s == 5 and z == pytest.approx(4 / np.sqrt(23 / 3))


def test_theil_sen_matches_scipy_and_resists_outliers():
    t = np.arange(12.0)
    y = 2.0 * t + 1.0
    y[5] = 500.0
    slope, intercept = theil_sen(y, t)
    assert slope == pytest.approx(2.0) and intercept == pytest.approx(1.0)
    noisy = np.random.default_rng(1).normal(size=20).cumsum()
    assert theil_sen(noisy)[0] == pytest.approx(scipy_stats.theilslopes(noisy)[0])


def test_trend_table_finds_planted_directions():
    _, truth, years, _ = planted_corpus()
    counts, axis = topic_year_counts(truth, years, k=6, year_range=(2010, 2021))
    table = trend_table(counts, axis)
    assert list(table.loc[[0, 5], "trend"]) == ["rising", "falling"]
    assert table.loc[0, "slope_per_year"] > 0 > table.loc[5, "slope_per_year"]
    assert (table["q_value"] >= table["p_value"] - 1e-12).all()


# -------------------------------------------------------------------- backtest


def test_growth_features_of_exponential_series():
    totals = np.full(8, 10_000)
    rising = (100 * np.exp(0.2 * np.arange(8))).round()
    counts = np.vstack([rising, totals - rising]).astype(int)
    features = growth_features(counts, window=6)
    assert features[0, 0] == pytest.approx(0.2, abs=0.02)  # last growth
    assert features[0, 1] == pytest.approx(0.2, abs=0.02)  # slope
    assert abs(features[0, 2]) < 0.03  # no acceleration
    assert features[0, 4] == pytest.approx(np.log(rising[-1] / 10_000), abs=0.01)


def test_forecast_scores():
    actual = np.array([0.5, 0.1, -0.2, 0.0, -0.4, 0.3, 0.2, -0.1, 0.05, -0.3])
    perfect = score_forecast(actual, actual)
    assert perfect == pytest.approx({"spearman": 1.0, "precision_top10pct": 1.0, "mae": 0.0})
    flat = score_forecast(np.zeros(10), actual)
    assert np.isnan(flat["spearman"]) and flat["mae"] == pytest.approx(np.abs(actual).mean())
    assert top_fraction_precision(-actual, actual, 0.2) == 0.0


def test_backtest_trend_beats_persistence_on_planted_growth_without_leakage():
    x, _, years, _ = planted_corpus(k=8, per_year=3000, max_rate=0.12)
    seen_masks = []

    def fit_assign(train_mask):
        seen_masks.append(train_mask.copy())
        model, _ = fit_topics(x[train_mask], k=8, seed=0, pca_dim=10)
        return model.assign(x), 8

    scores, forecasts = rolling_backtest(fit_assign, years, cutoffs=[2015, 2016, 2017, 2018, 2019])
    # Topics were fitted only on documents up to each cut-off.
    for cutoff, mask in zip([2015, 2016, 2017, 2018, 2019], seen_masks, strict=True):
        assert years[mask].max() == cutoff and mask.sum() == (years <= cutoff).sum()

    mean = scores.groupby("predictor")[["spearman", "mae"]].mean()
    two_year = scores[scores["horizon"] == 2].groupby("predictor")["spearman"].mean()
    assert two_year["linear_trend"] > 0.8
    # A trend over several years is a better forecast than the last one-year change.
    assert mean.loc["linear_trend", "spearman"] > mean.loc["last_growth", "spearman"]
    assert mean.loc["linear_trend", "mae"] < mean.loc["persistence", "mae"]
    assert np.isnan(mean.loc["persistence", "spearman"])
    # Ridge needs two earlier cut-offs with observed targets, so it starts later.
    ridge = scores[scores["predictor"] == "ridge"]
    assert set(ridge[ridge["horizon"] == 1]["cutoff"]) == {2017, 2018, 2019}
    assert set(ridge[ridge["horizon"] == 2]["cutoff"]) == {2018, 2019}
    assert len(forecasts) == scores["n_topics"].sum()
    with pytest.raises(ValueError):
        rolling_backtest(fit_assign, years, cutoffs=[2020])


# -------------------------------------------------------------------- lead-lag


def test_lagged_correlation_sign_convention():
    a = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0])
    b = np.roll(a, 1)  # b repeats a one step later
    assert lagged_correlation(a, b, 1) == pytest.approx(1.0)
    assert abs(lagged_correlation(a, b, 0)) < 0.9
    assert np.isnan(lagged_correlation(np.ones(6), b[:6], 0))


def test_lead_lag_test_detects_planted_lead_and_is_quiet_on_noise():
    rng = np.random.default_rng(0)
    k, t = 40, 16
    steps = rng.normal(0, 0.3, size=(k, t + 2))
    log_a = np.cumsum(steps[:, 2:], axis=1)
    log_b = np.cumsum(
        steps[:, :-2] * 0 + steps[:, 1:-1] * 0 + np.roll(steps, 2, axis=1)[:, 2:], axis=1
    )
    planted = lead_lag_test(np.exp(log_a) / 1e3, np.exp(log_b) / 1e3, max_lag=3, n_permutations=300)
    best = planted.loc[planted["mean_correlation"].idxmax()]
    assert best["lag_years"] == 2 and best["q_value"] < 0.05

    independent = lead_lag_test(
        np.exp(np.cumsum(rng.normal(0, 0.3, (k, t)), axis=1)) / 1e3,
        np.exp(np.cumsum(rng.normal(0, 0.3, (k, t)), axis=1)) / 1e3,
        max_lag=3,
        n_permutations=300,
    )
    assert (independent["q_value"] > 0.05).all()
