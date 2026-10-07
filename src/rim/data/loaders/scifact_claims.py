"""SciFact claims with human evidence labels (Wadden et al., EMNLP 2020).

Used only to measure how well an entailment judge agrees with human annotators.
Claims and annotations: CC BY 4.0. Abstracts: ODC-By 1.0 (from S2ORC).
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def build_pairs(claims: list[dict], corpus: dict[int, dict]) -> pd.DataFrame:
    """One row per (claim, cited abstract) with the human label.

    ``SUPPORT`` pairs are positives. ``CONTRADICT`` pairs and cited abstracts for
    which annotators found no evidence (``NOINFO``) are negatives.
    """
    rows = []
    for claim in claims:
        evidence = {int(k): v for k, v in claim.get("evidence", {}).items()}
        for doc_id in claim.get("cited_doc_ids", []):
            doc = corpus.get(int(doc_id))
            if doc is None:
                continue
            labels = {e["label"] for e in evidence.get(int(doc_id), [])}
            if len(labels) > 1:
                continue  # annotators disagree within the document
            label = labels.pop() if labels else "NOINFO"
            rows.append(
                {
                    "claim_id": claim["id"],
                    "doc_id": int(doc_id),
                    "claim": claim["claim"],
                    "premise": f"{doc['title']}\n{' '.join(doc['abstract'])}",
                    "label": label,
                    "entailed": label == "SUPPORT",
                }
            )
    return pd.DataFrame(rows)


def load_pairs(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Labelled pairs for the ``train`` and ``dev`` claim files (the test file is unlabelled)."""
    raw_dir = Path(raw_dir) / "scifact_claims"
    archive = raw_dir / "data.tar.gz"
    if not archive.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:

        def read(name: str) -> list[dict]:
            member = next(m for m in tar.getmembers() if m.name.endswith(name))
            return [
                json.loads(line) for line in tar.extractfile(member).read().decode().splitlines()
            ]

        corpus = {int(d["doc_id"]): d for d in read("corpus.jsonl")}
        return {
            "train": build_pairs(read("claims_train.jsonl"), corpus),
            "dev": build_pairs(read("claims_dev.jsonl"), corpus),
        }
