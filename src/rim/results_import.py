"""Import a results archive written by ``kaggle/run_gpu_experiments.py``.

The archive is checked before anything is copied: its checksum, the commit and tree
state each run record names, and the checksum of every artifact a record lists. A
result directory that already exists with a different run is moved to
``results/historical/<run_id>/`` first, so no record is lost by an import.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from rim.provenance import sha256_file

SESSIONS = "gpu_sessions"


class ImportRejected(ValueError):
    """The archive failed a check; nothing was copied."""


def commit_in_repo(repo: Path) -> Callable[[str], bool]:
    def exists(commit: str) -> bool:
        done = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        return done.returncode == 0

    return exists


def check_archive_checksum(archive: Path, expected: str | None = None) -> str:
    """SHA-256 of the archive, compared with ``expected`` or the ``.sha256`` file beside it."""
    actual = sha256_file(archive)
    sidecar = Path(f"{archive}.sha256")
    if expected is None and sidecar.exists():
        expected = sidecar.read_text().split()[0]
    if expected is not None and expected.lower() != actual:
        raise ImportRejected(f"archive checksum is {actual}, expected {expected}")
    return actual


def extract(archive: Path, target: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts or parts[:1] != ("results",):
                raise ImportRejected(f"unexpected path in archive: {name}")
        zf.extractall(target)
    return target / "results"


def check_run(run_dir: Path, commit_exists: Callable[[str], bool], commit: str | None) -> dict:
    """Validate one result directory against its own run record."""
    name = run_dir.name
    record_path = run_dir / "run.json"
    if not record_path.exists():
        raise ImportRejected(f"{name}: no run.json (the run did not finish)")
    record = json.loads(record_path.read_text())
    if record.get("experiment") != name:
        raise ImportRejected(f"{name}: record is for {record.get('experiment')!r}")
    git = record.get("git") or {}
    if git.get("source") != "git" or not git.get("commit"):
        raise ImportRejected(f"{name}: the commit was not read from a git checkout")
    if git.get("dirty"):
        raise ImportRejected(f"{name}: run from a modified tree: {git.get('dirty_paths')}")
    if commit and git["commit"] != commit:
        raise ImportRejected(f"{name}: run at {git['commit']}, expected {commit}")
    if not commit_exists(git["commit"]):
        raise ImportRejected(f"{name}: commit {git['commit']} is not in this repository")
    for artifact, digest in (record.get("artifacts") or {}).items():
        path = run_dir / artifact
        if not path.exists():
            raise ImportRejected(f"{name}: artifact {artifact} is missing")
        if sha256_file(path) != digest:
            raise ImportRejected(f"{name}: artifact {artifact} does not match its checksum")
    return record


def import_archive(
    archive: str | Path,
    repo: str | Path,
    commit: str | None = None,
    sha256: str | None = None,
    commit_exists: Callable[[str], bool] | None = None,
) -> dict:
    """Check ``archive`` and copy its runs into ``<repo>/results``. Returns a summary."""
    archive, repo = Path(archive), Path(repo)
    commit_exists = commit_exists or commit_in_repo(repo)
    summary = {"sha256": check_archive_checksum(archive, sha256)}
    summary |= {"imported": [], "unchanged": [], "archived": [], "sessions": []}
    with tempfile.TemporaryDirectory() as tmp:
        source = extract(archive, Path(tmp))
        runs = sorted(p for p in source.iterdir() if p.is_dir() and p.name != SESSIONS)
        # Every run is checked before the first one is copied.
        records = {run.name: check_run(run, commit_exists, commit) for run in runs}
        results = repo / "results"
        for run in runs:
            target = results / run.name
            if target.exists():
                old = json.loads((target / "run.json").read_text())["run_id"]
                if old == records[run.name]["run_id"]:
                    summary["unchanged"].append(run.name)
                    continue
                keep = results / "historical" / old
                if keep.exists():
                    raise ImportRejected(f"{keep} already exists")
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(keep))
                summary["archived"].append(old)
            shutil.copytree(run, target)
            summary["imported"].append(records[run.name]["run_id"])
        for session in sorted((source / SESSIONS).glob("*.json")):
            (results / SESSIONS).mkdir(parents=True, exist_ok=True)
            shutil.copy2(session, results / SESSIONS / session.name)
            summary["sessions"].append(session.name)
    return summary
