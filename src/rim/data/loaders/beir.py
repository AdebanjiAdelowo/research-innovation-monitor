"""BEIR scientific subsets: SciFact (claim verification) and SCIDOCS (citation prediction)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id
from rim.data.loaders import hub
from rim.data.schema import RetrievalDataset
from rim.data.splits import hash_split
from rim.provenance import DatasetManifest, utc_now

BEIR_SPECS = {
    "scifact": {
        "repo": "BeIR/scifact",
        "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels_repo": "BeIR/scifact-qrels",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        # The native train queries are used as the dev split.
        "qrels_files": {"train.tsv": "dev", "test.tsv": "test"},
    },
    "scidocs": {
        "repo": "BeIR/scidocs",
        "revision": "acebceec772628c6679d76d6cff30ad517d17f4e",
        "qrels_repo": "BeIR/scidocs-qrels",
        "qrels_revision": "735ea1048e37b1ebce14c6dc3d33a5edaf66d3dc",
        # Only a test file exists, so queries are split by hash.
        "qrels_files": {"test.tsv": None},
    },
}


def build_beir(
    name: str,
    corpus_raw: pd.DataFrame,
    queries_raw: pd.DataFrame,
    qrels_by_split: dict[str | None, pd.DataFrame],
    dev_fraction: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert BEIR tables to the common schema. Pure function, tested offline."""
    corpus = pd.DataFrame(
        {
            "doc_id": [make_doc_id(name, i) for i in corpus_raw["_id"]],
            "title": corpus_raw["title"].fillna("").astype(str),
            "text": corpus_raw["text"].fillna("").astype(str),
        }
    )
    query_text = dict(zip(queries_raw["_id"].astype(str), queries_raw["text"], strict=True))

    qrel_rows, split_of = [], {}
    for split, frame in qrels_by_split.items():
        for qid, did, score in zip(
            frame["query-id"].astype(str),
            frame["corpus-id"].astype(str),
            frame["score"],
            strict=True,
        ):
            query_id = f"{name}:q{qid}"
            split_of[query_id] = split or hash_split(query_id, dev_fraction)
            qrel_rows.append((query_id, make_doc_id(name, did), float(score)))
    qrels = pd.DataFrame(qrel_rows, columns=["query_id", "doc_id", "relevance"])

    judged = sorted(split_of)
    queries = pd.DataFrame(
        {
            "query_id": judged,
            "text": [query_text[q.split(":q", 1)[1]] for q in judged],
            "split": [split_of[q] for q in judged],
        }
    )
    return corpus, queries, qrels


def prepare(name: str, raw_dir: str | Path, dev_fraction: float = 0.3) -> RetrievalDataset:
    spec = BEIR_SPECS[name]
    corpus_raw = pd.read_parquet(
        hub.download(
            spec["repo"], "corpus/corpus-00000-of-00001.parquet", spec["revision"], raw_dir
        )
    )
    queries_raw = pd.read_parquet(
        hub.download(
            spec["repo"], "queries/queries-00000-of-00001.parquet", spec["revision"], raw_dir
        )
    )
    qrels_by_split = {
        split: pd.read_csv(
            hub.download(spec["qrels_repo"], filename, spec["qrels_revision"], raw_dir), sep="\t"
        )
        for filename, split in spec["qrels_files"].items()
    }
    corpus, queries, qrels = build_beir(name, corpus_raw, queries_raw, qrels_by_split, dev_fraction)
    manifest = DatasetManifest(
        name=name,
        source="BEIR",
        source_url=f"https://huggingface.co/datasets/{spec['repo']}",
        revision=f"{spec['revision']} (qrels {spec['qrels_revision']})",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        redistributable=True,
        retrieved_at=utc_now(),
        n_documents=len(corpus),
        n_queries=len(queries),
        n_qrels=len(qrels),
        notes="Queries without relevance judgments are dropped.",
    )
    return RetrievalDataset(name, corpus, queries, qrels, manifest)
