"""Second-stage reranking of a candidate list."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rim.config import RerankerConfig
from rim.retrieval.tokenize import Tokenizer
from rim.retrieval.types import Ranking


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> np.ndarray: ...
    def describe(self) -> dict: ...


class LexicalOverlapReranker:
    """Fraction of query terms present in the document. Deterministic stand-in for tests."""

    def __init__(self) -> None:
        self._tokenizer = Tokenizer()

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        terms = set(self._tokenizer(query))
        if not terms:
            return np.zeros(len(documents), dtype=np.float32)
        return np.array(
            [len(terms & set(self._tokenizer(d))) / len(terms) for d in documents],
            dtype=np.float32,
        )

    def describe(self) -> dict:
        return {"kind": "lexical"}


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.config = config
        self.device = device
        self.model = CrossEncoder(
            config.model_id, revision=config.revision, max_length=config.max_length, device=device
        )
        self.n_parameters = sum(p.numel() for p in self.model.model.parameters())

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            [(query, d) for d in documents],
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).reshape(len(documents), -1)[:, -1]

    def describe(self) -> dict:
        return {
            "kind": "cross-encoder",
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "n_parameters": self.n_parameters,
            "device": self.device,
        }


def build_reranker(config: RerankerConfig, device: str = "cpu") -> Reranker:
    if config.kind == "lexical":
        return LexicalOverlapReranker()
    return CrossEncoderReranker(config, device)


def score_candidates(
    reranker: Reranker,
    candidates: Ranking,
    query_text: dict[str, str],
    doc_text: dict[str, str],
    depth: int,
) -> dict[str, list[tuple[str, float]]]:
    """Reranker scores for the top ``depth`` candidates of every query, in candidate order."""
    scored = {}
    for qid, hits in candidates.items():
        top = [doc_id for doc_id, _ in hits[:depth]]
        scores = reranker.score(query_text[qid], [doc_text[d] for d in top])
        scored[qid] = list(zip(top, (float(s) for s in scores), strict=True))
    return scored


def apply_rerank(
    candidates: Ranking, scored: dict[str, list[tuple[str, float]]], depth: int
) -> Ranking:
    """Reorder the top ``depth`` candidates by reranker score; the tail keeps its order.

    ``scored`` may hold more candidates than ``depth`` (scores are computed once at the
    largest depth); only those within the first ``depth`` first-stage positions are used.
    The stable sort keeps the first-stage order among equal reranker scores.
    """
    out: Ranking = {}
    for qid, hits in candidates.items():
        head = scored.get(qid, [])[:depth]
        reordered = sorted(head, key=lambda item: -item[1])
        tail = hits[len(head) :]
        # Tail scores are shifted below the head so the list stays sorted by score.
        floor = min((s for _, s in reordered), default=0.0)
        out[qid] = reordered + [
            (doc_id, floor - rank) for rank, (doc_id, _) in enumerate(tail, start=1)
        ]
    return out
