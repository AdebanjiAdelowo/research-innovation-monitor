"""Fusion of ranked lists.

Two families are implemented:

* Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_i 1 / (k + rank_i(d)).
  It uses ranks only, so it needs no score calibration.
* Convex combination of normalised scores (Bruch et al., 2023):
  score(d) = alpha * dense(d) + (1 - alpha) * lexical(d), after per-query
  normalisation of each list. The weight alpha is selected on dev queries.

A document absent from a list contributes nothing to RRF and receives that list's
minimum normalised score in the convex combination.
"""

from __future__ import annotations

import numpy as np

from rim.retrieval.types import Ranking, sort_hits


def reciprocal_rank_fusion(rankings: list[Ranking], k: int = 60, depth: int = 100) -> Ranking:
    fused: Ranking = {}
    for qid in _all_queries(rankings):
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking.get(qid, []), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[qid] = sort_hits(scores, depth)
    return fused


def normalize_scores(hits: list[tuple[str, float]], method: str = "minmax") -> dict[str, float]:
    if not hits:
        return {}
    values = np.array([s for _, s in hits], dtype=np.float64)
    if method == "minmax":
        span = values.max() - values.min()
        normed = (values - values.min()) / span if span > 0 else np.ones_like(values)
    elif method == "zscore":
        std = values.std()
        normed = (values - values.mean()) / std if std > 0 else np.zeros_like(values)
    else:
        raise ValueError(f"unknown normalization {method!r}")
    return {doc_id: float(v) for (doc_id, _), v in zip(hits, normed, strict=True)}


def convex_fusion(
    lexical: Ranking,
    dense: Ranking,
    alpha: float,
    normalization: str = "minmax",
    depth: int = 100,
) -> Ranking:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    fused: Ranking = {}
    for qid in _all_queries([lexical, dense]):
        lex = normalize_scores(lexical.get(qid, []), normalization)
        den = normalize_scores(dense.get(qid, []), normalization)
        lex_floor = min(lex.values(), default=0.0)
        den_floor = min(den.values(), default=0.0)
        scores = {
            doc_id: alpha * den.get(doc_id, den_floor) + (1.0 - alpha) * lex.get(doc_id, lex_floor)
            for doc_id in lex.keys() | den.keys()
        }
        fused[qid] = sort_hits(scores, depth)
    return fused


def _all_queries(rankings: list[Ranking]) -> list[str]:
    seen: dict[str, None] = {}
    for ranking in rankings:
        for qid in ranking:
            seen.setdefault(qid)
    return list(seen)
