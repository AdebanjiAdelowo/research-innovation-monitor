"""LitSearch (Ajith et al., EMNLP 2024): literature-search questions over ML/NLP papers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id, normalize_text
from rim.data.loaders import hub
from rim.data.schema import RetrievalDataset
from rim.data.splits import hash_split
from rim.provenance import DatasetManifest, utc_now

REPO = "princeton-nlp/LitSearch"
REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
N_CORPUS_SHARDS = 6


def query_id_for(text: str) -> str:
    """LitSearch has no query IDs, so one is derived from the question text."""
    return "litsearch:q" + hashlib.sha1(normalize_text(text).encode()).hexdigest()[:12]


def build_litsearch(
    corpus_raw: pd.DataFrame, queries_raw: pd.DataFrame, dev_fraction: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert the raw tables to the common schema. Pure function, tested offline."""
    corpus_raw = corpus_raw.drop_duplicates("corpusid")
    corpus = pd.DataFrame(
        {
            "doc_id": [make_doc_id("litsearch", i) for i in corpus_raw["corpusid"]],
            "title": corpus_raw["title"].fillna("").astype(str).to_numpy(),
            "text": corpus_raw["abstract"].fillna("").astype(str).to_numpy(),
        }
    )
    if "full_paper" in corpus_raw.columns:
        corpus["full_text"] = corpus_raw["full_paper"].fillna("").astype(str).to_numpy()

    query_ids = [query_id_for(t) for t in queries_raw["query"]]
    queries = pd.DataFrame(
        {
            "query_id": query_ids,
            "text": queries_raw["query"].astype(str).to_numpy(),
            "split": [hash_split(q, dev_fraction) for q in query_ids],
            "query_set": queries_raw["query_set"].to_numpy(),
            "specificity": queries_raw["specificity"].to_numpy(),
            "quality": queries_raw["quality"].to_numpy(),
        }
    )
    known = set(corpus["doc_id"])
    rows = [
        (qid, make_doc_id("litsearch", cid), 1.0)
        for qid, cids in zip(query_ids, queries_raw["corpusids"], strict=True)
        for cid in cids
    ]
    qrels = pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])
    qrels = qrels[qrels["doc_id"].isin(known)].drop_duplicates().reset_index(drop=True)
    queries = queries[queries["query_id"].isin(qrels["query_id"])].reset_index(drop=True)
    return corpus, queries, qrels


def prepare(
    raw_dir: str | Path, dev_fraction: float = 0.3, full_text: bool = False
) -> RetrievalDataset:
    """Titles and abstracts by default; ``full_text`` adds the paper bodies (about 1.3 GB)."""
    columns = ["corpusid", "title", "abstract"] + (["full_paper"] if full_text else [])
    shards = [
        hub.read_parquet_columns(
            REPO, f"corpus_clean/full-{i:05d}-of-{N_CORPUS_SHARDS:05d}.parquet", REVISION, columns
        )
        for i in range(N_CORPUS_SHARDS)
    ]
    queries_raw = pd.read_parquet(
        hub.download(REPO, "query/full-00000-of-00001.parquet", REVISION, raw_dir)
    )
    corpus, queries, qrels = build_litsearch(
        pd.concat(shards, ignore_index=True), queries_raw, dev_fraction
    )
    manifest = DatasetManifest(
        name="litsearch_fulltext" if full_text else "litsearch",
        source="LitSearch",
        source_url=f"https://huggingface.co/datasets/{REPO}",
        revision=REVISION,
        license="MIT (code repository); the dataset card states no licence",
        license_url="https://github.com/princeton-nlp/LitSearch/blob/main/LICENSE",
        redistributable=False,
        retrieved_at=utc_now(),
        n_documents=len(corpus),
        n_queries=len(queries),
        n_qrels=len(qrels),
        notes=(
            "Corpus text is title and abstract from the corpus_clean configuration. "
            "Query IDs are derived from the question text. Treated as not redistributable "
            "because the dataset card carries no explicit licence."
        ),
    )
    return RetrievalDataset(manifest.name, corpus, queries, qrels, manifest)
