"""Topics as clusters of document embeddings.

Pipeline: L2-normalised embeddings -> PCA -> re-normalisation -> k-means. With unit
vectors, squared Euclidean distance is 2 - 2 cos, so k-means on the sphere groups
documents by cosine similarity. PCA removes low-variance directions that make
distances in the full space nearly uniform.

A fitted model can assign unseen documents to the nearest centroid. Temporal
backtests rely on this: topics are fitted on documents up to a cut-off year and later
documents are only assigned, never used for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from rim.retrieval.tokenize import STOPWORDS


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


@dataclass
class TopicModel:
    mean: np.ndarray
    components: np.ndarray  # (pca_dim, dim)
    centroids: np.ndarray  # (k, pca_dim), unit norm
    seed: int

    @property
    def k(self) -> int:
        return self.centroids.shape[0]

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        return _unit((np.asarray(embeddings, dtype=np.float32) - self.mean) @ self.components.T)

    def assign(self, embeddings: np.ndarray, block: int = 50_000) -> np.ndarray:
        """Nearest centroid by cosine similarity."""
        labels = np.empty(len(embeddings), dtype=np.int32)
        for start in range(0, len(embeddings), block):
            sims = self.project(embeddings[start : start + block]) @ self.centroids.T
            labels[start : start + block] = sims.argmax(axis=1)
        return labels


def fit_topics(
    embeddings: np.ndarray,
    k: int,
    seed: int = 0,
    pca_dim: int = 50,
    n_init: int = 3,
    minibatch_above: int = 200_000,
) -> tuple[TopicModel, np.ndarray]:
    """Fit a topic model. Returns the model and the label of each input document."""
    x = np.asarray(embeddings, dtype=np.float32)
    if k < 2 or k > len(x):
        raise ValueError("k must be between 2 and the number of documents")
    pca_dim = min(pca_dim, x.shape[1], len(x) - 1)
    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
    reduced = _unit(pca.fit_transform(x).astype(np.float32))
    if len(x) > minibatch_above:
        km = MiniBatchKMeans(k, random_state=seed, n_init=n_init, batch_size=4096)
    else:
        km = KMeans(k, random_state=seed, n_init=n_init)
    km.fit(reduced)
    model = TopicModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        centroids=_unit(km.cluster_centers_.astype(np.float32)),
        seed=seed,
    )
    # Labels come from the model's own assignment rule, so training documents and
    # later documents are treated identically.
    return model, model.assign(x)


def topic_terms(texts: list[str], labels: np.ndarray, k: int, top_n: int = 8) -> list[list[str]]:
    """Descriptive terms per topic by class-based TF-IDF (Grootendorst, 2022).

    Each topic is treated as one document. A term's weight in topic c is
    tf(t, c) * log(1 + A / f(t)), with tf the frequency of t in c normalised by the
    topic's length, f(t) its frequency over all topics and A the mean topic length.
    """
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=2, max_features=50_000
    )
    try:
        counts = vectorizer.fit_transform(texts)
    except ValueError:  # vocabulary empty after pruning
        return [[] for _ in range(k)]
    vocabulary = np.array(vectorizer.get_feature_names_out())
    per_topic = np.vstack(
        [np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in range(k)]
    ).astype(np.float64)
    lengths = per_topic.sum(axis=1, keepdims=True)
    tf = per_topic / np.maximum(lengths, 1.0)
    idf = np.log1p(lengths.mean() / np.maximum(per_topic.sum(axis=0), 1.0))
    weights = tf * idf
    return [list(vocabulary[np.argsort(-weights[c])[:top_n]]) for c in range(k)]
