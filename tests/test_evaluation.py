import math

import numpy as np
import pytest

from rim.evaluation.metrics import (
    evaluate,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rim.evaluation.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    holm,
    paired_bootstrap_ci,
    paired_randomization_test,
)

RANKED = ["a", "x", "b", "y", "c"]
REL = {"a": 1.0, "b": 1.0, "c": 1.0, "z": 1.0}


def test_set_metrics_by_hand():
    assert recall_at_k(RANKED, REL, 3) == 0.5
    assert precision_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], REL, 5) == 0.2  # short lists are penalised, not renormalised
    assert mrr_at_k(RANKED, REL, 5) == 1.0
    assert mrr_at_k(["x", "y", "a"], REL, 5) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "a"], REL, 2) == 0.0
    assert recall_at_k(RANKED, {}, 3) == 0.0


def test_ndcg_by_hand_binary_and_graded():
    dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, 5))
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / idcg)
    graded = {"a": 3.0, "b": 1.0}
    assert ndcg_at_k(["a", "b"], graded, 10) == pytest.approx(1.0)
    swapped = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], graded, 10) == pytest.approx(swapped)
    assert ndcg_at_k(["x"], graded, 10) == 0.0


def test_evaluate_keeps_queries_without_results_and_rejects_duplicates():
    qrels = {"q1": {"a": 1.0}, "q2": {"b": 1.0}}
    frame = evaluate({"q1": [("a", 1.0)]}, qrels, ks=(1, 10))
    assert list(frame.index) == ["q1", "q2"]
    assert frame.loc["q2"].sum() == 0.0
    assert mean_metrics(frame)["recall@1"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        evaluate({"q1": [("a", 1.0), ("a", 0.5)]}, qrels)


def test_bootstrap_ci_covers_mean_and_is_seeded():
    values = np.random.default_rng(1).normal(0.5, 0.1, size=400)
    low, high = bootstrap_ci(values, 4000, seed=3)
    assert low < values.mean() < high
    # Close to the normal-theory interval.
    half = 1.96 * values.std(ddof=1) / math.sqrt(values.size)
    assert high - low == pytest.approx(2 * half, rel=0.1)
    assert bootstrap_ci(values, 4000, seed=3) == (low, high)
    low_d, high_d = paired_bootstrap_ci(values + 0.05, values, 2000)
    assert low_d == pytest.approx(0.05) and high_d == pytest.approx(0.05)


def test_randomization_test_exact_small_case():
    # Five positive differences: only the two extreme sign assignments reach the observed
    # sum, so the exact two-sided p-value is 2 / 2**5 = 0.0625.
    a, b = np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.zeros(5)
    p = paired_randomization_test(a, b, n_permutations=200_000, seed=0)
    assert p == pytest.approx(0.0625, abs=0.004)


def test_randomization_test_behaviour():
    rng = np.random.default_rng(0)
    base = rng.uniform(size=300)
    assert paired_randomization_test(base, base) == 1.0
    assert paired_randomization_test(base + rng.normal(0.1, 0.05, 300), base) < 0.001
    assert paired_randomization_test(base + rng.normal(0.0, 0.05, 300), base) > 0.05
    noisy = base + rng.normal(0.02, 0.2, 300)
    assert paired_randomization_test(noisy, base, seed=1) == paired_randomization_test(
        base, noisy, seed=1
    )


def test_holm_and_benjamini_hochberg_by_hand():
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert benjamini_hochberg([]) == []
