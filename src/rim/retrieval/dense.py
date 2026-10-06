"""Exact nearest-neighbour search by inner product.

The corpora here have at most a few hundred thousand vectors, where a blocked matrix
product is fast and exact. An approximate index would add a second source of error
to every retrieval comparison without a measurable speed benefit at this scale.
"""

from __future__ import annotations

import numpy as np

from rim.retrieval.types import Ranking


class DenseIndex:
    def __init__(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids and embeddings differ in length")
        self.doc_ids = list(doc_ids)
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    def search_vectors(
        self, queries: np.ndarray, top_k: int = 100, block: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Indices and scores of the top documents for each query vector."""
        queries = np.asarray(queries, dtype=np.float32)
        k = min(top_k, len(self.doc_ids))
        indices = np.empty((len(queries), k), dtype=np.int64)
        scores = np.empty((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), block):
            sims = queries[start : start + block] @ self.embeddings.T
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(sims, part, axis=1)
            # Sort by score, then by corpus position, for reproducible ties.
            order = np.lexsort((part, -part_scores), axis=1)
            indices[start : start + block] = np.take_along_axis(part, order, axis=1)
            scores[start : start + block] = np.take_along_axis(part_scores, order, axis=1)
        return indices, scores

    def search(self, query_ids: list[str], queries: np.ndarray, top_k: int = 100) -> Ranking:
        indices, scores = self.search_vectors(queries, top_k)
        return {
            qid: [(self.doc_ids[i], float(s)) for i, s in zip(row, row_scores, strict=True)]
            for qid, row, row_scores in zip(query_ids, indices, scores, strict=True)
        }
