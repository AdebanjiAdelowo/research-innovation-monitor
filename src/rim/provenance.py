"""Provenance: what code, data, models and machine produced a result.

Every experiment writes a run record next to its outputs. A number that appears in
the README must be traceable to one of these records.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "torch",
    "sentence-transformers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "snowballstemmer",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Commit and cleanliness of the working tree.

    When the code does not run from a git checkout (a Kaggle upload, a Docker image),
    the commit is taken from ``RIM_GIT_COMMIT`` and marked as declared, not verified.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit:
        status = run("status", "--porcelain") or ""
        # Paths with uncommitted changes, so a reader can judge whether they matter.
        paths = sorted(line[3:] for line in status.splitlines() if line.strip())
        return {"commit": commit, "dirty": bool(paths), "dirty_paths": paths, "source": "git"}
    declared = os.environ.get("RIM_GIT_COMMIT", "").strip()
    if declared:
        return {"commit": declared, "dirty": None, "source": "declared"}
    return {"commit": None, "dirty": None, "source": "unknown"}


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_device(requested: str = "auto") -> str:
    """Pick cuda, then mps, then cpu. Never fails when torch is absent."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info(device: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": device or resolve_device(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return info


def environment_info(device: str | None = None) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "hardware": device_info(device),
    }


class DatasetManifest(BaseModel):
    """Where a prepared dataset came from and what it contains."""

    name: str
    source: str
    source_url: str
    revision: str | None = None
    license: str
    license_url: str | None = None
    redistributable: bool
    retrieved_at: str
    n_documents: int
    n_queries: int = 0
    n_qrels: int = 0
    files: dict[str, str] = {}
    notes: str = ""

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class RunRecord:
    experiment: str
    config: dict[str, Any]
    config_sha256: str
    seed: int | None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    git: dict[str, Any] = field(default_factory=git_info)
    environment: dict[str, Any] = field(default_factory=environment_info)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        commit = (self.git.get("commit") or "nogit")[:7]
        stamp = self.started_at.replace("-", "").replace(":", "")
        return f"{self.experiment}-{stamp}-{commit}-{self.config_sha256[:8]}"

    def add_artifact(self, path: str | Path, root: str | Path | None = None) -> None:
        p = Path(path)
        key = str(p.relative_to(root)) if root else p.name
        self.artifacts[key] = sha256_file(p)

    def finish(self) -> None:
        self.finished_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, **asdict(self)}

    def write(self, path: str | Path) -> None:
        if self.finished_at is None:
            self.finish()
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
