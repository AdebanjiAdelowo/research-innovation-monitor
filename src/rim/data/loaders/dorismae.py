"""DORIS-MAE (Wang et al., NeurIPS 2023): multi-aspect scientific queries.

Each of the 100 queries is a paragraph-long research need, broken down by the
dataset authors into aspects and sub-aspects, with a pool of about 100 candidate
abstracts. Relevance of an abstract to each aspect was scored 0, 1 or 2 by a language
model whose agreement with human annotators the authors report.

Only candidate-pool documents are judged, so ranking is evaluated within each
query's pool. The query-level relevance used here is the mean over top-level aspects
of the aspect score, where an aspect with sub-aspects takes the mean score of its
sub-aspects. It lies in [0, 2].

Licence: CC BY-NC 4.0. Evaluation only; never committed or redistributed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id
from rim.data.schema import RetrievalDataset
from rim.data.splits import hash_split
from rim.provenance import DatasetManifest, sha256_file, utc_now

ZENODO_URL = "https://zenodo.org/api/records/8299749/files/DORIS-MAE_dataset_v1.json/content"
FILENAME = "DORIS-MAE_dataset_v1.json"
MD5 = "eb176a948f576d6205d31af4f27be801"


def query_relevance(aspects: dict[str, list[str]], scores: dict[str, float]) -> float:
    """Mean over top-level aspects; an aspect with sub-aspects takes their mean score."""
    values = []
    for aspect, subs in aspects.items():
        if subs:
            sub_scores = [scores[s] for s in subs if s in scores]
            if sub_scores:
                values.append(sum(sub_scores) / len(sub_scores))
                continue
        if aspect in scores:
            values.append(scores[aspect])
    return sum(values) / len(values) if values else 0.0


def build_dorismae(
    raw: dict, dev_fraction: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert the raw JSON to the common schema. Pure function, tested offline."""
    aspect_text = raw["aspect_id2aspect"]
    annotation: dict[tuple[str, int], float] = {
        (str(a["aspect_id"]), int(a["abstract_id"])): float(a["score"]) for a in raw["Annotation"]
    }
    pool_ids = sorted({int(d) for q in raw["Query"] for d in q["candidate_pool"]})
    by_id = {int(d["abstract_id"]): d for d in raw["Corpus"]}
    corpus = pd.DataFrame(
        {
            "doc_id": [make_doc_id("dorismae", i) for i in pool_ids],
            "title": [str(by_id[i].get("title") or "").replace("_", " ") for i in pool_ids],
            "text": [
                " ".join(str(by_id[i].get("original_abstract") or "").split()) for i in pool_ids
            ],
            "category": [by_id[i].get("primary_category") for i in pool_ids],
        }
    )

    query_rows, qrel_rows = [], []
    for index, query in enumerate(raw["Query"]):
        query_id = f"dorismae:q{index:03d}"
        aspects = {str(a): [str(s) for s in subs] for a, subs in query["aspects"].items()}
        needed = set(aspects) | {s for subs in aspects.values() for s in subs}
        pool = [int(d) for d in query["candidate_pool"]]
        for doc in pool:
            scores = {a: annotation[(a, doc)] for a in needed if (a, doc) in annotation}
            relevance = query_relevance(aspects, scores)
            if relevance > 0:
                qrel_rows.append((query_id, make_doc_id("dorismae", doc), relevance))
        query_rows.append(
            {
                "query_id": query_id,
                "text": " ".join(query["query_text"].split()),
                "split": hash_split(query_id, dev_fraction),
                "query_type": query.get("query_type"),
                "candidate_pool": [make_doc_id("dorismae", d) for d in pool],
                # The authors' own decomposition, used as an oracle upper reference.
                "aspects": [aspect_text[a] for a in aspects],
            }
        )
    queries = pd.DataFrame(query_rows)
    qrels = pd.DataFrame(qrel_rows, columns=["query_id", "doc_id", "relevance"])
    return corpus, queries, qrels


def prepare(raw_dir: str | Path, dev_fraction: float = 0.3) -> RetrievalDataset:
    path = Path(raw_dir) / "dorismae" / FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it (1.08 GB) with:\n  curl -L -o {path} {ZENODO_URL}"
        )
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    corpus, queries, qrels = build_dorismae(raw, dev_fraction)
    manifest = DatasetManifest(
        name="dorismae",
        source="DORIS-MAE v1",
        source_url="https://doi.org/10.5281/zenodo.8299749",
        revision=f"Zenodo record 8299749, sha256 {sha256_file(path)}",
        license="CC BY-NC 4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        redistributable=False,
        retrieved_at=utc_now(),
        n_documents=len(corpus),
        n_queries=len(queries),
        n_qrels=len(qrels),
        notes=(
            "Corpus restricted to the union of the candidate pools; ranking is evaluated "
            "within each query's pool. Aspect relevance labels are model-generated "
            "(validated against humans by the dataset authors). Non-commercial licence: "
            "evaluation only, never redistributed."
        ),
    )
    return RetrievalDataset("dorismae", corpus, queries, qrels, manifest)
