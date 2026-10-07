import itertools

import pandas as pd
import pytest

from rim.data.chunking import chunk_corpus, chunk_spans, split_sentences
from rim.data.ids import (
    content_doc_id,
    doc_id_of,
    make_chunk_id,
    make_doc_id,
    parse_chunk_id,
    source_of,
)
from rim.data.loaders.beir import build_beir
from rim.data.loaders.dapfam import build_dapfam
from rim.data.loaders.litsearch import build_litsearch, query_id_for
from rim.data.schema import DatasetError, RetrievalDataset, index_text
from rim.data.splits import hash_split, unit_hash

# ---------------------------------------------------------------- identifiers


def test_doc_id_round_trip():
    doc = make_doc_id("litsearch", 12345)
    assert doc == "litsearch:12345"
    assert source_of(doc) == "litsearch"
    chunk = make_chunk_id(doc, 3)
    assert chunk == "litsearch:12345#c0003"
    assert parse_chunk_id(chunk) == (doc, 3)
    assert doc_id_of(chunk) == doc
    assert doc_id_of(doc) == doc


@pytest.mark.parametrize(
    "source,native", [("Bad Source", "1"), ("ok", ""), ("ok", "a b"), ("ok", "a#b")]
)
def test_invalid_ids_are_rejected(source, native):
    with pytest.raises(ValueError):
        make_doc_id(source, native)


def test_content_id_is_stable_under_formatting():
    a = content_doc_id("nsf", "A Title", "Some  text\nhere")
    b = content_doc_id("nsf", "a title", "some text here")
    assert a == b
    assert a != content_doc_id("nsf", "A Title", "Other text")


# --------------------------------------------------------------------- splits


def test_hash_split_is_deterministic_and_balanced():
    ids = [f"q{i}" for i in range(4000)]
    first = [hash_split(i, 0.3) for i in ids]
    assert first == [hash_split(i, 0.3) for i in ids]
    share = first.count("dev") / len(first)
    assert 0.27 < share < 0.33
    assert all(0.0 <= unit_hash(i) < 1.0 for i in ids[:100])


def test_hash_split_does_not_depend_on_other_queries():
    assert hash_split("q7", 0.3) == hash_split("q7", 0.3, salt="rim-split-v1")
    assert {hash_split("q7", 0.3, salt=s) for s in "abcdefghij"} == {"dev", "test"}


# --------------------------------------------------------------------- schema


def test_dataset_validation_and_qrels(toy_dataset):
    assert len(toy_dataset.split_queries("dev")) == 3
    test_qrels = toy_dataset.qrels_dict("test")
    assert set(test_qrels) == {"toy:q3", "toy:q4", "toy:q5"}
    assert test_qrels["toy:q4"] == {"toy:d07": 1.0, "toy:d08": 1.0}
    assert toy_dataset.doc_texts()[0].startswith("Graph neural networks for molecules. Message")


def test_duplicate_doc_ids_are_rejected(toy_dataset):
    corpus = pd.concat([toy_dataset.corpus, toy_dataset.corpus.iloc[:1]])
    with pytest.raises(DatasetError, match="duplicate"):
        RetrievalDataset("bad", corpus, toy_dataset.queries, toy_dataset.qrels)


def test_qrels_must_reference_known_documents(toy_dataset):
    qrels = pd.concat(
        [
            toy_dataset.qrels,
            pd.DataFrame([("toy:q1", "toy:missing", 1.0)], columns=toy_dataset.qrels.columns),
        ]
    )
    with pytest.raises(DatasetError, match="missing from the corpus"):
        RetrievalDataset("bad", toy_dataset.corpus, toy_dataset.queries, qrels)


def test_save_load_preserves_metadata_and_verifies_checksums(toy_dataset, tmp_path):
    from rim.provenance import DatasetManifest, utc_now

    toy_dataset.manifest = DatasetManifest(
        name="toy",
        source="unit",
        source_url="https://example.org",
        license="CC0",
        redistributable=True,
        retrieved_at=utc_now(),
        n_documents=len(toy_dataset.corpus),
    )
    toy_dataset.save(tmp_path)
    loaded = RetrievalDataset.load(tmp_path, verify=True)
    pd.testing.assert_frame_equal(loaded.corpus, toy_dataset.corpus)
    assert list(loaded.corpus["year"]) == list(toy_dataset.corpus["year"])
    assert set(loaded.manifest.files) == {"corpus.parquet", "queries.parquet", "qrels.parquet"}

    (tmp_path / "qrels.parquet").write_bytes(b"corrupted")
    with pytest.raises(DatasetError, match="checksum"):
        RetrievalDataset.load(tmp_path, verify=True)


def test_index_text_skips_empty_fields():
    frame = pd.DataFrame({"title": ["T", ""], "text": ["", "body"]})
    assert index_text(frame) == ["T", "body"]


# ------------------------------------------------------------------- chunking

TEXT = (
    "Hybrid retrieval combines lexical and dense signals. It was studied by Smith et al. in 2021. "
    "The gain is 3.5 points, e.g. on scientific corpora. Is it robust? Reranking adds latency."
)


def test_sentence_splitting_respects_abbreviations_and_decimals():
    sentences = split_sentences(TEXT)
    assert sentences == [
        "Hybrid retrieval combines lexical and dense signals.",
        "It was studied by Smith et al. in 2021.",
        "The gain is 3.5 points, e.g. on scientific corpora.",
        "Is it robust?",
        "Reranking adds latency.",
    ]


def test_chunks_respect_budget_cover_text_and_overlap():
    spans = chunk_spans(TEXT, max_words=20, overlap_words=9)
    chunks = [s.of(TEXT) for s in spans]
    assert [len(c.split()) for c in chunks] == [16, 18, 15]
    # Offsets address the source text exactly.
    assert all(TEXT[s.start : s.end] == c for s, c in zip(spans, chunks, strict=True))
    # Every sentence appears in some chunk, and consecutive chunks never leave a gap.
    assert all(any(sent in c for c in chunks) for sent in split_sentences(TEXT))
    assert all(b.start <= a.end + 1 for a, b in itertools.pairwise(spans))
    assert any(b.start < a.end for a, b in itertools.pairwise(spans))
    assert spans[0].start == 0 and spans[-1].end == len(TEXT)
    # No chunk is contained in its predecessor, even when the overlap budget allows it.
    for max_words, overlap in [(16, 9), (16, 15), (12, 11)]:
        tight = chunk_spans(TEXT, max_words, overlap)
        assert all(b.end > a.end for a, b in itertools.pairwise(tight))


def test_long_sentence_is_hard_split_and_chunking_terminates():
    text = " ".join(f"w{i}" for i in range(95))
    spans = chunk_spans(text, max_words=20, overlap_words=5)
    assert [len(s.of(text).split()) for s in spans] == [20, 20, 20, 20, 15]
    assert chunk_spans("", 20, 5) == []


def test_chunk_parameters_are_validated():
    with pytest.raises(ValueError):
        chunk_spans("a b c", max_words=10, overlap_words=10)


def test_chunk_corpus_preserves_ids_and_metadata(toy_dataset):
    corpus = toy_dataset.corpus.copy()
    corpus.loc[0, "text"] = TEXT
    chunks = chunk_corpus(corpus, max_words=16, overlap_words=4)
    first = chunks[chunks["doc_id"] == "toy:d01"]
    assert len(first) > 1
    assert list(first["chunk_id"]) == [make_chunk_id("toy:d01", i) for i in range(len(first))]
    assert set(first["year"]) == {2015} and set(first["title"]) == {corpus.loc[0, "title"]}
    assert chunks["chunk_id"].is_unique
    assert set(chunks["doc_id"]) == set(corpus["doc_id"])
    row = first.iloc[1]
    assert TEXT[row["start_char"] : row["end_char"]] == row["text"]
    pd.testing.assert_frame_equal(chunks, chunk_corpus(corpus, max_words=16, overlap_words=4))


# -------------------------------------------------------------------- loaders


def test_build_beir_uses_native_splits_and_drops_unjudged_queries():
    corpus_raw = pd.DataFrame({"_id": ["1", "2"], "title": ["A", None], "text": ["a", "b"]})
    queries_raw = pd.DataFrame({"_id": ["10", "11", "12"], "text": ["x", "y", "unjudged"]})
    qrels = {
        "dev": pd.DataFrame({"query-id": [10], "corpus-id": [1], "score": [1]}),
        "test": pd.DataFrame({"query-id": [11], "corpus-id": [2], "score": [1]}),
    }
    corpus, queries, qrels_out = build_beir("scifact", corpus_raw, queries_raw, qrels)
    ds = RetrievalDataset("scifact", corpus, queries, qrels_out)
    assert list(ds.corpus["doc_id"]) == ["scifact:1", "scifact:2"]
    assert ds.corpus.loc[1, "title"] == ""
    assert dict(zip(ds.queries["query_id"], ds.queries["split"], strict=True)) == {
        "scifact:q10": "dev",
        "scifact:q11": "test",
    }


def test_build_litsearch_derives_stable_query_ids():
    corpus_raw = pd.DataFrame(
        {"corpusid": [5, 6, 6], "title": ["T5", "T6", "T6"], "abstract": ["a5", None, None]}
    )
    queries_raw = pd.DataFrame(
        {
            "query_set": ["inline_acl", "manual_iclr"],
            "query": ["Which paper studies X?", "Papers on Y"],
            "specificity": [1, 0],
            "quality": [2, 1],
            "corpusids": [[5], [6, 999]],
        }
    )
    corpus, queries, qrels = build_litsearch(corpus_raw, queries_raw)
    ds = RetrievalDataset("litsearch", corpus, queries, qrels)
    assert list(ds.corpus["doc_id"]) == ["litsearch:5", "litsearch:6"]
    assert ds.queries.loc[0, "query_id"] == query_id_for("which paper  studies x?")
    # The judgment pointing outside the corpus is dropped, the query is kept.
    assert len(ds.qrels) == 2
    assert list(ds.queries["specificity"]) == [1, 0]


def test_build_dapfam_keeps_domain_labels_and_metadata():
    ipc = "classifications_ipcr_list_first_three_chars_list"
    corpus_raw = pd.DataFrame(
        {
            "relevant_id": ["F1", "F2"],
            "title_en": ["Pump", "Valve"],
            "abstract_en": ["A pump.", "A valve."],
            "earliest_claim_year": ["2001", None],
            ipc: [["F04", "F04"], None],
        }
    )
    queries_raw = pd.DataFrame(
        {
            "query_id": ["Q1"],
            "title_en": ["Compressor"],
            "abstract_en": ["A compressor."],
            "earliest_claim_year": ["2010"],
        }
    )
    qrels_raw = pd.DataFrame(
        {
            "query_id": ["Q1", "Q1"],
            "relevant_id": ["F1", "F2"],
            "relevance_score": [1.0, 1.0],
            "domain_rel": ["IN", "OUT"],
        }
    )
    corpus, queries, qrels = build_dapfam(corpus_raw, queries_raw, qrels_raw)
    ds = RetrievalDataset("dapfam", corpus, queries, qrels)
    assert ds.queries.loc[0, "text"] == "Compressor. A compressor"
    assert list(ds.qrels["domain"]) == ["IN", "OUT"]
    assert ds.corpus.loc[0, "year"] == 2001 and pd.isna(ds.corpus.loc[1, "year"])
    assert ds.corpus.loc[0, "ipc3"] == ["F04"]


def test_build_dorismae_aggregates_aspect_scores_hierarchically():
    from rim.data.loaders.dorismae import build_dorismae, query_relevance

    # Aspect 1 has sub-aspects 10 and 11 (mean 1.5); aspect 2 is a leaf (0). Mean = 0.75.
    assert query_relevance({"1": ["10", "11"], "2": []}, {"10": 2, "11": 1, "2": 0, "1": 0}) == 0.75
    assert query_relevance({"1": []}, {}) == 0.0

    raw = {
        "aspect_id2aspect": {"1": "uses deep learning", "10": "a", "11": "b", "2": "has a dataset"},
        "Corpus": [
            {
                "abstract_id": 0,
                "title": "A_first_paper",
                "original_abstract": "line one\nline two",
                "primary_category": "cs.LG",
            },
            {
                "abstract_id": 1,
                "title": "Second",
                "original_abstract": "text",
                "primary_category": "cs.CL",
            },
            {
                "abstract_id": 2,
                "title": "Not in any pool",
                "original_abstract": "x",
                "primary_category": "cs.CL",
            },
        ],
        "Query": [
            {
                "query_text": "I need  a model.",
                "candidate_pool": [0, 1],
                "aspects": {"1": ["10", "11"], "2": []},
                "query_type": "ML",
            }
        ],
        "Annotation": [
            {"aspect_id": "10", "abstract_id": 0, "score": 2},
            {"aspect_id": "11", "abstract_id": 0, "score": 1},
            {"aspect_id": "2", "abstract_id": 0, "score": 0},
            {"aspect_id": "2", "abstract_id": 1, "score": 0},
        ],
    }
    corpus, queries, qrels = build_dorismae(raw)
    ds = RetrievalDataset("dorismae", corpus, queries, qrels)
    assert list(ds.corpus["doc_id"]) == ["dorismae:0", "dorismae:1"]
    assert (
        ds.corpus.loc[0, "title"] == "A first paper"
        and ds.corpus.loc[0, "text"] == "line one line two"
    )
    assert ds.queries.loc[0, "candidate_pool"] == ["dorismae:0", "dorismae:1"]
    assert ds.queries.loc[0, "aspects"] == ["uses deep learning", "has a dataset"]
    assert ds.qrels.to_dict("records") == [
        {"query_id": "dorismae:q000", "doc_id": "dorismae:0", "relevance": 0.75}
    ]
