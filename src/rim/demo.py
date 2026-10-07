"""A twelve-document synthetic corpus.

It lets the API, the dashboard and the tests run with no download and no model
weights. It is written for this repository and is not derived from any dataset.
"""

from __future__ import annotations

import pandas as pd

from rim.data.schema import RetrievalDataset

DOCS = [
    (
        "d01",
        "Graph neural networks for molecules",
        "Message passing networks predict molecular properties from atomic graphs.",
    ),
    (
        "d02",
        "Transformers for protein folding",
        "Attention models predict protein structure from amino acid sequences.",
    ),
    (
        "d03",
        "BM25 and probabilistic ranking",
        "The probabilistic relevance framework derives term weighting for document ranking.",
    ),
    (
        "d04",
        "Dense passage retrieval",
        "Dual encoders embed questions and passages for open domain question answering.",
    ),
    (
        "d05",
        "Reciprocal rank fusion",
        "Combining ranked lists by reciprocal rank outperforms individual ranking systems.",
    ),
    (
        "d06",
        "Cross-encoder reranking",
        "Joint encoding of query and passage improves ranking accuracy at higher cost.",
    ),
    (
        "d07",
        "Physics-informed neural networks",
        "Neural networks are trained to satisfy partial differential equations.",
    ),
    (
        "d08",
        "Fourier neural operators",
        "Operator learning maps between function spaces for parametric PDE solutions.",
    ),
    (
        "d09",
        "Topic models for science",
        "Latent Dirichlet allocation discovers research themes in document collections.",
    ),
    (
        "d10",
        "Citation analysis of patents",
        "Patent citations to scientific papers measure knowledge flow to industry.",
    ),
    (
        "d11",
        "Diffusion models for images",
        "Denoising diffusion generates images by reversing a noising process.",
    ),
    (
        "d12",
        "Retrieval-augmented generation",
        "Language models condition on retrieved passages to answer with evidence.",
    ),
]

QUERIES = [
    ("q1", "probabilistic term weighting for ranking documents", "dev", ["d03"]),
    ("q2", "embedding questions and passages with dual encoders", "dev", ["d04"]),
    ("q3", "combining ranked lists from several systems", "test", ["d05"]),
    ("q4", "neural networks that solve partial differential equations", "test", ["d07", "d08"]),
    ("q5", "language models answering with retrieved evidence", "test", ["d12", "d04"]),
    ("q6", "discovering research themes in collections", "dev", ["d09"]),
]


def build_demo_dataset() -> RetrievalDataset:
    corpus = pd.DataFrame(
        {
            "doc_id": [f"toy:{d}" for d, _, _ in DOCS],
            "title": [t for _, t, _ in DOCS],
            "text": [x for _, _, x in DOCS],
            "year": [2015 + i % 8 for i in range(len(DOCS))],
        }
    )
    queries = pd.DataFrame(
        {
            "query_id": [f"toy:{q}" for q, _, _, _ in QUERIES],
            "text": [t for _, t, _, _ in QUERIES],
            "split": [s for _, _, s, _ in QUERIES],
        }
    )
    qrels = pd.DataFrame(
        [(f"toy:{q}", f"toy:{d}", 1.0) for q, _, _, docs in QUERIES for d in docs],
        columns=["query_id", "doc_id", "relevance"],
    )
    return RetrievalDataset("toy", corpus, queries, qrels)
