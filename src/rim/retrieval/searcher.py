"""One object that answers queries over a corpus: lexical, dense, hybrid, optionally reranked.

This is the retrieval entry point shared by the RAG pipeline, the tools and the API.
The experiment runner in ``rim.experiments.retrieval`` uses the same building blocks
but keeps the stages separate so each can be tuned and timed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rim.data.schema import index_text
from rim.rerank.rerankers import Reranker
from rim.retrieval.bm25 import BM25Index
from rim.retrieval.dense import DenseIndex
from rim.retrieval.encoders import Encoder, encode_documents_cached
from rim.retrieval.fusion import convex_fusion, reciprocal_rank_fusion
from rim.retrieval.tokenize import Tokenizer

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    mode: str
    hits: list[Hit]
    timings_ms: dict[str, float]


class HybridSearcher:
    def __init__(
        self,
        corpus: pd.DataFrame,
        bm25: BM25Index,
        encoder: Encoder,
        dense: DenseIndex,
        reranker: Reranker | None = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        depth: int = 100,
        fields: tuple[str, ...] = ("title", "text"),
    ) -> None:
        if fusion not in ("rrf", "convex"):
            raise ValueError("fusion must be 'rrf' or 'convex'")
        self.corpus = corpus.set_index("doc_id", drop=False)
        self.bm25, self.encoder, self.dense, self.reranker = bm25, encoder, dense, reranker
        self.fusion, self.rrf_k, self.alpha, self.depth = fusion, rrf_k, alpha, depth
        self._index_text = dict(zip(corpus["doc_id"], index_text(corpus, fields), strict=True))
        self._meta_columns = [c for c in corpus.columns if c not in ("doc_id", "title", "text")]

    @classmethod
    def build(
        cls,
        corpus: pd.DataFrame,
        encoder: Encoder,
        cache_dir: str | Path | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        fields: tuple[str, ...] = ("title", "text"),
        **kwargs,
    ) -> HybridSearcher:
        doc_ids = list(corpus["doc_id"])
        texts = index_text(corpus, fields)
        bm25 = BM25Index(Tokenizer(), k1, b).fit(doc_ids, texts)
        embeddings, _ = encode_documents_cached(encoder, texts, cache_dir)
        return cls(corpus, bm25, encoder, DenseIndex(doc_ids, embeddings), fields=fields, **kwargs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        rerank_depth: int = 0,
        exclude: set[str] | None = None,
    ) -> SearchResponse:
        """Top ``top_k`` documents. ``exclude`` removes documents before ranking is cut,
        which is how evidence is withheld in the abstention experiment."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        exclude = exclude or set()
        depth = max(self.depth, top_k, rerank_depth) + len(exclude)
        timings: dict[str, float] = {}

        def timed(name, fn):
            start = time.perf_counter()
            value = fn()
            timings[name] = (time.perf_counter() - start) * 1000.0
            return value

        lexical = dense = None
        if mode in ("bm25", "hybrid"):
            lexical = timed("bm25_ms", lambda: self.bm25.search_one(query, depth))
        if mode in ("dense", "hybrid"):
            vector = timed("encode_ms", lambda: self.encoder.encode_queries([query]))
            dense = timed("dense_ms", lambda: self.dense.search(["q"], vector, depth)["q"])
        if mode == "bm25":
            ranked = lexical
        elif mode == "dense":
            ranked = dense
        elif self.fusion == "rrf":
            ranked = timed(
                "fusion_ms",
                lambda: reciprocal_rank_fusion([{"q": lexical}, {"q": dense}], self.rrf_k, depth)[
                    "q"
                ],
            )
        else:
            ranked = timed(
                "fusion_ms",
                lambda: convex_fusion({"q": lexical}, {"q": dense}, self.alpha, depth=depth)["q"],
            )
        ranked = [(d, s) for d, s in ranked if d not in exclude]

        if rerank_depth and self.reranker is not None:
            head = ranked[:rerank_depth]
            scores = timed(
                "rerank_ms",
                lambda: self.reranker.score(query, [self._index_text[d] for d, _ in head]),
            )
            order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
            ranked = [(head[i][0], float(scores[i])) for i in order] + ranked[rerank_depth:]

        hits = []
        for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
            row = self.corpus.loc[doc_id]
            metadata = {c: _plain(row[c]) for c in self._meta_columns}
            hits.append(Hit(doc_id, float(score), rank, row["title"], row["text"], metadata))
        timings["total_ms"] = sum(timings.values())
        return SearchResponse(query, mode, hits, timings)


def _plain(value):
    """Convert numpy and pandas scalars to JSON-friendly Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is pd.NA or (isinstance(value, float) and value != value):
        return None
    return value.item() if hasattr(value, "item") else value
