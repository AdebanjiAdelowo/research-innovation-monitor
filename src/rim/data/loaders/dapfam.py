"""DAPFAM (Ayaou et al., 2025): family-level patent prior-art retrieval.

Licence: CC BY-NC-SA 4.0. The data is used for evaluation only and is never
committed or redistributed by this repository.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id
from rim.data.loaders import hub
from rim.data.schema import RetrievalDataset
from rim.data.splits import hash_split
from rim.provenance import DatasetManifest, utc_now

REPO = "datalyes/DAPFAM_patent"
REVISION = "a59a74ce31384165065af1823a83c6f94ccafd48"
_IPC = "classifications_ipcr_list_first_three_chars_list"


def _year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_dapfam(
    corpus_raw: pd.DataFrame,
    queries_raw: pd.DataFrame,
    qrels_raw: pd.DataFrame,
    dev_fraction: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Title and abstract view of queries and targets. Pure function, tested offline."""
    corpus = pd.DataFrame(
        {
            "doc_id": [make_doc_id("dapfam", i) for i in corpus_raw["relevant_id"]],
            "title": corpus_raw["title_en"].fillna("").astype(str).to_numpy(),
            "text": corpus_raw["abstract_en"].fillna("").astype(str).to_numpy(),
            "year": _year(corpus_raw["earliest_claim_year"]).to_numpy(),
            "ipc3": [sorted(set(v)) if v is not None else [] for v in corpus_raw[_IPC]],
        }
    ).drop_duplicates("doc_id", ignore_index=True)

    query_ids = [f"dapfam:q{i}" for i in queries_raw["query_id"]]
    title = queries_raw["title_en"].fillna("").astype(str)
    abstract = queries_raw["abstract_en"].fillna("").astype(str)
    queries = pd.DataFrame(
        {
            "query_id": query_ids,
            "text": (title + ". " + abstract).str.strip(". ").to_numpy(),
            "split": [hash_split(q, dev_fraction) for q in query_ids],
            "year": _year(queries_raw["earliest_claim_year"]).to_numpy(),
        }
    )
    qrels = pd.DataFrame(
        {
            "query_id": [f"dapfam:q{i}" for i in qrels_raw["query_id"]],
            "doc_id": [make_doc_id("dapfam", i) for i in qrels_raw["relevant_id"]],
            "relevance": qrels_raw["relevance_score"].astype(float).to_numpy(),
            # IN: query and target share an IPC3 class. OUT: they do not.
            "domain": qrels_raw["domain_rel"].astype(str).to_numpy(),
        }
    )
    qrels = qrels[qrels["doc_id"].isin(set(corpus["doc_id"]))].reset_index(drop=True)
    queries = queries[queries["query_id"].isin(qrels["query_id"])].reset_index(drop=True)
    return corpus, queries, qrels


def prepare(raw_dir: str | Path, dev_fraction: float = 0.3) -> RetrievalDataset:
    # Column projection: claims and descriptions (most of the 1.15 GB file) are not fetched.
    corpus_raw = hub.read_parquet_columns(
        REPO,
        "corpus.parquet",
        REVISION,
        ["relevant_id", "title_en", "abstract_en", "earliest_claim_year", _IPC],
    )
    queries_raw = hub.read_parquet_columns(
        REPO,
        "queries.parquet",
        REVISION,
        ["query_id", "title_en", "abstract_en", "earliest_claim_year"],
    )
    qrels_raw = pd.read_parquet(hub.download(REPO, "qrels_all.parquet", REVISION, raw_dir))
    corpus, queries, qrels = build_dapfam(corpus_raw, queries_raw, qrels_raw, dev_fraction)
    manifest = DatasetManifest(
        name="dapfam",
        source="DAPFAM",
        source_url=f"https://huggingface.co/datasets/{REPO}",
        revision=REVISION,
        license="CC BY-NC-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        redistributable=False,
        retrieved_at=utc_now(),
        n_documents=len(corpus),
        n_queries=len(queries),
        n_qrels=len(qrels),
        notes=(
            "Title and abstract view of query and target families. Relevance is "
            "citation-based. Non-commercial licence: evaluation only, never redistributed."
        ),
    )
    return RetrievalDataset("dapfam", corpus, queries, qrels, manifest)
