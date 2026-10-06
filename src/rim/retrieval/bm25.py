"""Okapi BM25 over a sparse term-document matrix.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

with the Lucene form of the inverse document frequency,
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)),
which is non-negative for every term. Repeated query terms count once per occurrence.

Raw term frequencies are kept separately from the weights, so k1 and b can be changed
without tokenising the corpus again.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from rim.retrieval.tokenize import Tokenizer
from rim.retrieval.types import Ranking


class BM25Index:
    def __init__(
        self, tokenizer: Tokenizer | None = None, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.tokenizer = tokenizer or Tokenizer()
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self._tf: sparse.csc_matrix | None = None  # documents x terms, raw counts
        self._weights: sparse.csc_matrix | None = None
        self._doc_len: np.ndarray | None = None

    # ------------------------------------------------------------------ build

    def fit(self, doc_ids: list[str], texts: list[str]) -> BM25Index:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts differ in length")
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(texts), dtype=np.float64)
        vocabulary: dict[str, int] = {}
        for row, text in enumerate(texts):
            counts = Counter(self.tokenizer(text))
            doc_len[row] = sum(counts.values())
            for term, count in counts.items():
                col = vocabulary.setdefault(term, len(vocabulary))
                rows.append(row)
                cols.append(col)
                vals.append(count)
        self.doc_ids = list(doc_ids)
        self.vocabulary = vocabulary
        self._doc_len = doc_len
        self._tf = sparse.csc_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(texts), len(vocabulary)),
        )
        self._reweight()
        return self

    def _reweight(self) -> None:
        tf = self._tf
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr).astype(np.float64)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avgdl = self._doc_len.mean() if n_docs else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * self._doc_len / max(avgdl, 1e-12))
        col_of_entry = np.repeat(np.arange(tf.shape[1]), np.diff(tf.indptr))
        data = tf.data.astype(np.float64)
        weights = idf[col_of_entry] * data * (self.k1 + 1.0) / (data + norm[tf.indices])
        self._weights = sparse.csc_matrix(
            (weights.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape
        )

    def with_params(self, k1: float, b: float) -> BM25Index:
        """A view of the same index with different parameters (shares the count matrix)."""
        other = BM25Index(self.tokenizer, k1, b)
        other.doc_ids, other.vocabulary = self.doc_ids, self.vocabulary
        other._tf, other._doc_len = self._tf, self._doc_len
        other._reweight()
        return other

    # ----------------------------------------------------------------- search

    def scores(self, query: str) -> np.ndarray:
        """Dense score vector over all documents."""
        counts = Counter(t for t in self.tokenizer(query) if t in self.vocabulary)
        out = np.zeros(len(self.doc_ids), dtype=np.float32)
        if not counts:
            return out
        cols = np.fromiter((self.vocabulary[t] for t in counts), dtype=np.int64)
        mult = np.fromiter(counts.values(), dtype=np.float32)
        return np.asarray(self._weights[:, cols] @ mult, dtype=np.float32).ravel()

    def search_one(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        scores = self.scores(query)
        k = min(top_k, int(np.count_nonzero(scores > 0)))
        if k == 0:
            return []
        candidates = np.argpartition(-scores, k - 1)[:k]
        # Sort by score, then by position in the corpus, for reproducible ties.
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 100) -> Ranking:
        return {qid: self.search_one(q, top_k) for qid, q in zip(query_ids, queries, strict=True)}

    def describe(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "idf": "lucene",
            "tokenizer": self.tokenizer.describe(),
            "n_documents": len(self.doc_ids),
            "n_terms": len(self.vocabulary),
        }
