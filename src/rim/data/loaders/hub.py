"""Access to dataset files on the Hugging Face Hub, pinned to a revision."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _require_hub():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'Dataset download needs the "data" extra: pip install -e ".[data]"'
        ) from exc
    return huggingface_hub


def download(repo_id: str, filename: str, revision: str, raw_dir: str | Path) -> Path:
    """Download one file into ``raw_dir`` (not into the shared Hub cache)."""
    hub = _require_hub()
    target = Path(raw_dir) / repo_id.replace("/", "__")
    path = hub.hf_hub_download(
        repo_id, filename, repo_type="dataset", revision=revision, local_dir=target
    )
    return Path(path)


def read_parquet_columns(
    repo_id: str, filename: str, revision: str, columns: list[str]
) -> pd.DataFrame:
    """Read selected columns of a remote parquet file with HTTP range requests.

    Only the requested column chunks are transferred, which keeps large full-text
    columns off the local disk.
    """
    hub = _require_hub()
    import pyarrow.parquet as pq

    fs = hub.HfFileSystem()
    with fs.open(f"datasets/{repo_id}@{revision}/{filename}", "rb") as fh:
        return pq.ParquetFile(fh).read(columns=columns).to_pandas()
