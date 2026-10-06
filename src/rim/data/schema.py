"""In-memory and on-disk form of a retrieval dataset.

A prepared dataset is a directory with ``corpus.parquet``, ``queries.parquet``,
``qrels.parquet`` and ``manifest.json``. Columns beyond the required ones are
metadata and are preserved through chunking and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rim.provenance import DatasetManifest, sha256_file

CORPUS_COLUMNS = ("doc_id", "title", "text")
QUERY_COLUMNS = ("query_id", "text", "split")
QREL_COLUMNS = ("query_id", "doc_id", "relevance")
SPLITS = ("dev", "test")


class DatasetError(ValueError):
    pass


def _require(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DatasetError(f"{what}: missing columns {missing}")


def validate_corpus(corpus: pd.DataFrame) -> None:
    _require(corpus, CORPUS_COLUMNS, "corpus")
    if corpus["doc_id"].isna().any():
        raise DatasetError("corpus: null doc_id")
    if corpus["doc_id"].duplicated().any():
        dup = corpus.loc[corpus["doc_id"].duplicated(), "doc_id"].iloc[0]
        raise DatasetError(f"corpus: duplicate doc_id {dup!r}")
    if corpus[["title", "text"]].isna().any().any():
        raise DatasetError("corpus: null title or text (use empty strings)")


def validate_queries(queries: pd.DataFrame) -> None:
    _require(queries, QUERY_COLUMNS, "queries")
    if queries["query_id"].duplicated().any():
        raise DatasetError("queries: duplicate query_id")
    bad = set(queries["split"]) - set(SPLITS)
    if bad:
        raise DatasetError(f"queries: unknown splits {sorted(bad)}")


@dataclass
class RetrievalDataset:
    name: str
    corpus: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    manifest: DatasetManifest | None = None

    def __post_init__(self) -> None:
        validate_corpus(self.corpus)
        validate_queries(self.queries)
        _require(self.qrels, QREL_COLUMNS, "qrels")
        known_docs = set(self.corpus["doc_id"])
        unknown = set(self.qrels["doc_id"]) - known_docs
        if unknown:
            raise DatasetError(
                f"qrels reference {len(unknown)} documents missing from the corpus, "
                f"for example {next(iter(unknown))!r}"
            )
        unknown_q = set(self.qrels["query_id"]) - set(self.queries["query_id"])
        if unknown_q:
            raise DatasetError(f"qrels reference {len(unknown_q)} unknown queries")

    def split_queries(self, split: str) -> pd.DataFrame:
        if split not in SPLITS:
            raise DatasetError(f"unknown split {split!r}")
        return self.queries[self.queries["split"] == split].reset_index(drop=True)

    def qrels_dict(self, split: str | None = None) -> dict[str, dict[str, float]]:
        """``{query_id: {doc_id: relevance}}`` restricted to positive judgments."""
        qrels = self.qrels[self.qrels["relevance"] > 0]
        if split is not None:
            qrels = qrels[qrels["query_id"].isin(self.split_queries(split)["query_id"])]
        out: dict[str, dict[str, float]] = {}
        for qid, did, rel in zip(
            qrels["query_id"], qrels["doc_id"], qrels["relevance"], strict=True
        ):
            out.setdefault(qid, {})[did] = float(rel)
        return out

    def doc_texts(self, fields: tuple[str, ...] = ("title", "text")) -> list[str]:
        return index_text(self.corpus, fields)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for stem, frame in (
            ("corpus", self.corpus),
            ("queries", self.queries),
            ("qrels", self.qrels),
        ):
            path = directory / f"{stem}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            files[path.name] = sha256_file(path)
        if self.manifest is not None:
            self.manifest = self.manifest.model_copy(update={"files": files})
            self.manifest.write(directory / "manifest.json")

    @classmethod
    def load(cls, directory: str | Path, verify: bool = False) -> RetrievalDataset:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = DatasetManifest.read(manifest_path) if manifest_path.exists() else None
        if verify and manifest is not None:
            for filename, expected in manifest.files.items():
                actual = sha256_file(directory / filename)
                if actual != expected:
                    raise DatasetError(f"{filename}: checksum differs from the manifest")
        return cls(
            name=manifest.name if manifest else directory.name,
            corpus=pd.read_parquet(directory / "corpus.parquet"),
            queries=pd.read_parquet(directory / "queries.parquet"),
            qrels=pd.read_parquet(directory / "qrels.parquet"),
            manifest=manifest,
        )


def index_text(frame: pd.DataFrame, fields: tuple[str, ...] = ("title", "text")) -> list[str]:
    """Concatenate the chosen text fields of each row, skipping empty ones."""
    columns = [frame[f].fillna("").astype(str).tolist() for f in fields]
    return [
        ". ".join(part.strip() for part in parts if part.strip())
        for parts in zip(*columns, strict=True)
    ]
