import math

import numpy as np
import pytest

from rim.config import EncoderConfig, RerankerConfig
from rim.rerank.rerankers import apply_rerank, build_reranker, score_candidates
from rim.retrieval.bm25 import BM25Index
from rim.retrieval.dense import DenseIndex
from rim.retrieval.encoders import HashingEncoder, build_encoder, encode_documents_cached
from rim.retrieval.fusion import convex_fusion, normalize_scores, reciprocal_rank_fusion
from rim.retrieval.tokenize import Tokenizer

DOCS = [
    "the cat sat on the mat",
    "dogs chase cats",
    "a treatise on ranking functions",
    "cat cat cat",
]
IDS = ["a", "b", "c", "d"]

# ------------------------------------------------------------------- tokenizer


def test_tokenizer_lowercases_stems_and_removes_stopwords():
    assert Tokenizer()("The Cats are RANKING documents, 2024!") == [
        "cat",
        "rank",
        "document",
        "2024",
    ]
    assert Tokenizer(stem=False, remove_stopwords=False)("The Cats") == ["the", "cats"]


# ------------------------------------------------------------------------ BM25


def reference_bm25(query, docs, k1, b):
    """Direct transcription of the formula, independent of the sparse implementation."""
    tok = Tokenizer()
    tokens = [tok(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in tokens) / n
    scores = []
    for doc in tokens:
        total = 0.0
        for term in tok(query):
            df = sum(1 for t in tokens if term in t)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = doc.count(term)
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(total)
    return scores


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.6, 0.3), (2.0, 1.0), (1.0, 0.0)])
def test_bm25_matches_reference_formula(k1, b):
    index = BM25Index(k1=k1, b=b).fit(IDS, DOCS)
    for query in ["cat", "cats ranking", "cat cat dogs", "unknownterm"]:
        np.testing.assert_allclose(
            index.scores(query), reference_bm25(query, DOCS, k1, b), rtol=1e-5
        )


def test_bm25_reparameterisation_equals_refit():
    base = BM25Index(k1=1.2, b=0.75).fit(IDS, DOCS)
    np.testing.assert_allclose(
        base.with_params(0.5, 0.2).scores("cat ranking"),
        BM25Index(k1=0.5, b=0.2).fit(IDS, DOCS).scores("cat ranking"),
        rtol=1e-6,
    )
    # The original index is unchanged.
    assert (base.k1, base.b) == (1.2, 0.75)


def test_bm25_search_orders_by_score_and_omits_non_matching(toy_dataset):
    index = BM25Index().fit(list(toy_dataset.corpus["doc_id"]), toy_dataset.doc_texts())
    hits = index.search_one("probabilistic term weighting for ranking documents", top_k=5)
    assert hits[0][0] == "toy:d03"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert all(s > 0 for _, s in hits)
    assert index.search_one("zzzz qqqq") == []


def test_bm25_ties_are_broken_by_corpus_order():
    index = BM25Index().fit(["x", "y", "z"], ["alpha beta", "alpha beta", "gamma"])
    assert [d for d, _ in index.search_one("alpha")] == ["x", "y"]


# ----------------------------------------------------------------------- dense


def test_hashing_encoder_is_deterministic_and_normalised():
    enc = HashingEncoder(dim=64)
    a, b = enc.encode_documents(DOCS), enc.encode_documents(DOCS)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(enc.encode_queries(["cat"]), enc.encode_documents(["cats"]))


def test_dense_search_matches_brute_force():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    queries = rng.normal(size=(7, 16)).astype(np.float32)
    index = DenseIndex([f"d{i}" for i in range(200)], emb)
    indices, scores = index.search_vectors(queries, top_k=10, block=3)
    expected = np.argsort(-(queries @ emb.T), axis=1)[:, :10]
    np.testing.assert_array_equal(indices, expected)
    assert np.all(np.diff(scores, axis=1) <= 0)
    ranking = index.search([f"q{i}" for i in range(7)], queries, top_k=3)
    assert [d for d, _ in ranking["q0"]] == [f"d{i}" for i in expected[0, :3]]


def test_dense_retrieval_finds_lexically_similar_document(toy_dataset):
    enc = build_encoder(EncoderConfig(name="h", kind="hashing", dim=512))
    index = DenseIndex(
        list(toy_dataset.corpus["doc_id"]), enc.encode_documents(toy_dataset.doc_texts())
    )
    ranking = index.search(["q"], enc.encode_queries(["reciprocal rank fusion of ranked lists"]), 3)
    assert ranking["q"][0][0] == "toy:d05"


def test_embedding_cache_hits_only_for_identical_inputs(tmp_path):
    enc = HashingEncoder(dim=32)
    first, info1 = encode_documents_cached(enc, DOCS, tmp_path)
    second, info2 = encode_documents_cached(enc, DOCS, tmp_path)
    assert not info1["cache_hit"] and info2["cache_hit"]
    np.testing.assert_array_equal(first, second)
    assert info1["dim"] == 32 and info1["n_vectors"] == 4
    _, info3 = encode_documents_cached(enc, [*DOCS[:-1], "changed"], tmp_path)
    _, info4 = encode_documents_cached(HashingEncoder(dim=16), DOCS, tmp_path)
    assert not info3["cache_hit"] and not info4["cache_hit"]


# ---------------------------------------------------------------------- fusion


def test_rrf_scores_by_hand():
    lexical = {"q": [("a", 9.0), ("b", 5.0), ("c", 1.0)]}
    dense = {"q": [("b", 0.9), ("d", 0.8)]}
    fused = dict(reciprocal_rank_fusion([lexical, dense], k=60)["q"])
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert next(iter(fused)) == "b"


def test_rrf_is_invariant_to_monotone_score_changes():
    lexical = {"q": [("a", 9.0), ("b", 5.0)]}
    rescaled = {"q": [("a", 900.0), ("b", 0.001)]}
    dense = {"q": [("b", 0.9), ("a", 0.1)]}
    assert reciprocal_rank_fusion([lexical, dense]) == reciprocal_rank_fusion([rescaled, dense])


def test_normalisation():
    hits = [("a", 10.0), ("b", 6.0), ("c", 2.0)]
    assert normalize_scores(hits) == {"a": 1.0, "b": 0.5, "c": 0.0}
    z = normalize_scores(hits, "zscore")
    assert sum(z.values()) == pytest.approx(0.0) and z["a"] == pytest.approx(-z["c"])
    assert normalize_scores([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}
    assert normalize_scores([]) == {}


def test_convex_fusion_endpoints_and_missing_documents():
    lexical = {"q": [("a", 10.0), ("b", 6.0), ("c", 2.0)]}
    dense = {"q": [("c", 0.9), ("d", 0.5), ("a", 0.1)]}
    only_lexical = [d for d, _ in convex_fusion(lexical, dense, alpha=0.0)["q"]]
    only_dense = [d for d, _ in convex_fusion(lexical, dense, alpha=1.0)["q"]]
    assert only_lexical[:3] == ["a", "b", "c"] and only_dense[:2] == ["c", "d"]
    mixed = dict(convex_fusion(lexical, dense, alpha=0.5)["q"])
    # a: lexical 1.0, dense 0.0. c: lexical 0.0, dense 1.0. d: absent from lexical -> floor 0.0.
    assert mixed["a"] == pytest.approx(0.5) and mixed["c"] == pytest.approx(0.5)
    assert mixed["d"] == pytest.approx(0.25) and mixed["b"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        convex_fusion(lexical, dense, alpha=1.5)


# ------------------------------------------------------------------- reranking


def test_rerank_reorders_head_and_keeps_tail():
    reranker = build_reranker(RerankerConfig(name="lex", kind="lexical"))
    candidates = {"q": [("d1", 3.0), ("d2", 2.0), ("d3", 1.0), ("d4", 0.5)]}
    doc_text = {
        "d1": "unrelated words",
        "d2": "neural operators",
        "d3": "fourier neural operators",
        "d4": "fourier",
    }
    scored = score_candidates(
        reranker, candidates, {"q": "fourier neural operators"}, doc_text, depth=4
    )
    assert [d for d, _ in scored["q"]] == ["d1", "d2", "d3", "d4"]

    full = apply_rerank(candidates, scored, depth=4)["q"]
    assert [d for d, _ in full] == ["d3", "d2", "d4", "d1"]
    shallow = apply_rerank(candidates, scored, depth=2)["q"]
    assert [d for d, _ in shallow] == ["d2", "d1", "d3", "d4"]
    for hits in (full, shallow):
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True) and len({d for d, _ in hits}) == 4
    assert apply_rerank(candidates, scored, depth=0)["q"][0][0] == "d1"
