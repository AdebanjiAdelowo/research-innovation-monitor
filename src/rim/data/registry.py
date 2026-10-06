"""Prepare (download and convert) and load datasets by name."""

from __future__ import annotations

from pathlib import Path

from rim.config import data_dir
from rim.data.schema import RetrievalDataset

RETRIEVAL_DATASETS = ("scifact", "scidocs", "litsearch", "litsearch_fulltext", "dapfam", "dorismae")


def processed_dir(name: str, root: str | Path | None = None) -> Path:
    return Path(root or data_dir()) / "processed" / name


def prepare_dataset(name: str, root: str | Path | None = None, force: bool = False) -> Path:
    """Download and convert a dataset. Returns the processed directory."""
    root = Path(root or data_dir())
    out = processed_dir(name, root)
    if (out / "manifest.json").exists() and not force:
        return out
    raw = root / "raw"
    if name in ("scifact", "scidocs"):
        from rim.data.loaders import beir

        dataset = beir.prepare(name, raw)
    elif name in ("litsearch", "litsearch_fulltext"):
        from rim.data.loaders import litsearch

        dataset = litsearch.prepare(raw, full_text=name.endswith("fulltext"))
    elif name == "dapfam":
        from rim.data.loaders import dapfam

        dataset = dapfam.prepare(raw)
    elif name == "dorismae":
        from rim.data.loaders import dorismae

        dataset = dorismae.prepare(raw)
    else:
        raise KeyError(f"unknown dataset {name!r}; known: {RETRIEVAL_DATASETS}")
    dataset.save(out)
    return out


def load_dataset(
    name: str, root: str | Path | None = None, verify: bool = False
) -> RetrievalDataset:
    path = processed_dir(name, root)
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"dataset {name!r} is not prepared; run: rim prepare {name}")
    return RetrievalDataset.load(path, verify=verify)
