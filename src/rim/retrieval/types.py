"""Shared types for ranked results."""

from __future__ import annotations

# ``{query_id: [(doc_id, score), ...]}`` sorted by decreasing score.
Ranking = dict[str, list[tuple[str, float]]]


def truncate(ranking: Ranking, depth: int) -> Ranking:
    return {qid: hits[:depth] for qid, hits in ranking.items()}


def sort_hits(scores: dict[str, float], depth: int | None = None) -> list[tuple[str, float]]:
    """Order by decreasing score, breaking ties by document ID so results are reproducible."""
    hits = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return hits[:depth] if depth is not None else hits
