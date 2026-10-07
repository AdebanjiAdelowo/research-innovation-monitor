"""arXiv metadata from the Kaggle snapshot (Cornell University, CC0 metadata).

The snapshot is one JSON object per line (about 5 GB). It is read as a stream and
filtered to the requested categories and years, so it never has to fit in memory.
On Kaggle the file is available at
``/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json`` without a download.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id
from rim.provenance import DatasetManifest, sha256_file, utc_now

DEFAULT_CATEGORIES = ("cs.LG", "cs.CL", "cs.CV", "cs.IR", "cs.AI", "stat.ML", "math.NA", "math.OC")


def parse_record(raw: dict, categories: set[str], first_year: int, last_year: int) -> dict | None:
    """Keep a paper whose primary category is requested and whose first version is in range.

    The year is that of the first submitted version, so later revisions do not move a
    paper forward in time.
    """
    listed = (raw.get("categories") or "").split()
    if not listed or listed[0] not in categories:
        return None
    versions = raw.get("versions") or []
    try:
        year = parsedate_to_datetime(versions[0]["created"]).year
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not first_year <= year <= last_year:
        return None
    abstract = " ".join((raw.get("abstract") or "").split())
    title = " ".join((raw.get("title") or "").split())
    if not abstract or not title:
        return None
    return {
        "doc_id": make_doc_id("arxiv", raw["id"]),
        "title": title,
        "text": abstract,
        "year": year,
        "primary_category": listed[0],
    }


def iter_records(
    lines: Iterable[str], categories: Iterable[str], first_year: int, last_year: int
) -> Iterator[dict]:
    wanted = set(categories)
    for line in lines:
        if line.strip():
            row = parse_record(json.loads(line), wanted, first_year, last_year)
            if row is not None:
                yield row


def prepare(
    snapshot: str | Path,
    root: str | Path,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    first_year: int = 2010,
    last_year: int = 2024,
) -> tuple[pd.DataFrame, DatasetManifest]:
    categories = tuple(categories)
    with open(snapshot, encoding="utf-8") as fh:
        papers = pd.DataFrame(iter_records(fh, categories, first_year, last_year))
    papers = papers.drop_duplicates("doc_id", ignore_index=True)
    out = Path(root) / "processed" / "arxiv"
    out.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(out / "corpus.parquet", index=False, compression="zstd")
    manifest = DatasetManifest(
        name="arxiv",
        source="arXiv metadata snapshot (Kaggle, Cornell University)",
        source_url="https://www.kaggle.com/datasets/Cornell-University/arxiv",
        revision=f"snapshot sha256 {sha256_file(snapshot)}",
        license="CC0 1.0 (metadata)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        redistributable=True,
        retrieved_at=utc_now(),
        n_documents=len(papers),
        files={"corpus.parquet": sha256_file(out / "corpus.parquet")},
        notes=(
            f"Primary category in {list(categories)}, first version submitted "
            f"{first_year} to {last_year}. The snapshot is updated weekly; the checksum "
            "identifies the version used."
        ),
    )
    manifest.write(out / "manifest.json")
    return papers, manifest
