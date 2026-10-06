"""Ranking metrics, computed per query so that systems can be compared with paired tests.

Definitions for a ranked list r_1..r_k and judged relevance rel(d) >= 0:

* recall@k    = |relevant in top k| / |relevant|
* precision@k = |relevant in top k| / k
* mrr@k       = 1 / rank of the first relevant document within the top k, else 0
* ndcg@k      = DCG@k / IDCG@k with DCG@k = sum_i rel(r_i) / log2(i + 1), linear gain,
                as in trec_eval and BEIR

Queries with judgments but no retrieved documents score zero; they are never dropped.
"""

from __future__ import annotations

import math

import pandas as pd

from rim.retrieval.types import Ranking

Qrels = dict[str, dict[str, float]]


def recall_at_k(ranked: list[str], relevant: dict[str, float], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if relevant.get(d, 0.0) > 0) / len(relevant)


def precision_at_k(ranked: list[str], relevant: dict[str, float], k: int) -> float:
    return sum(1 for d in ranked[:k] if relevant.get(d, 0.0) > 0) / k


def mrr_at_k(ranked: list[str], relevant: dict[str, float], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[str], relevant: dict[str, float], k: int) -> float:
    ideal = dcg(sorted((g for g in relevant.values() if g > 0), reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return dcg([relevant.get(d, 0.0) for d in ranked[:k]]) / ideal


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate(
    ranking: Ranking, qrels: Qrels, ks: tuple[int, ...] = (1, 5, 10, 20, 100)
) -> pd.DataFrame:
    """Per-query metrics: one row per judged query, one column per ``metric@k``."""
    rows = []
    for qid in sorted(qrels):
        relevant = qrels[qid]
        ranked = [doc_id for doc_id, _ in ranking.get(qid, [])]
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"ranking for {qid} contains duplicate documents")
        row = {"query_id": qid}
        for name, fn in METRICS.items():
            for k in ks:
                row[f"{name}@{k}"] = fn(ranked, relevant, k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("query_id")


def mean_metrics(per_query: pd.DataFrame) -> dict[str, float]:
    return {column: float(value) for column, value in per_query.mean().items()}
