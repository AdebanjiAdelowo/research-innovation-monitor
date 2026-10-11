import json
import shutil

import pytest

from rim.provenance import sha256_bytes
from rim.results_import import ImportRejected, import_archive

COMMIT = "a" * 40
SUMMARY = b'{"ndcg@10": 0.5}\n'


def make_archive(tmp_path, run_id="demo_gpu-1", content=SUMMARY, commit=COMMIT, **git):
    build = tmp_path / "build"
    run = build / "results" / "demo_gpu"
    run.mkdir(parents=True)
    (run / "summary.json").write_bytes(content)
    record = {
        "experiment": "demo_gpu",
        "run_id": run_id,
        "git": {"commit": commit, "dirty": False, "source": "git"} | git,
        "artifacts": {"summary.json": sha256_bytes(SUMMARY)},
    }
    (run / "run.json").write_text(json.dumps(record))
    (build / "results" / "gpu_sessions").mkdir()
    (build / "results" / "gpu_sessions" / "session.json").write_text("{}")
    archive = shutil.make_archive(str(tmp_path / f"archive-{run_id}"), "zip", build, "results")
    shutil.rmtree(build)
    return archive


def known(commit):
    return commit == COMMIT


def test_import_copies_runs_and_session_manifest(tmp_path):
    repo = tmp_path / "repo"
    summary = import_archive(make_archive(tmp_path), repo, commit=COMMIT, commit_exists=known)
    assert summary["imported"] == ["demo_gpu-1"]
    assert (repo / "results/demo_gpu/summary.json").read_bytes() == SUMMARY
    assert (repo / "results/gpu_sessions/session.json").exists()


def test_existing_run_is_kept_as_historical(tmp_path):
    repo = tmp_path / "repo"
    import_archive(make_archive(tmp_path), repo, commit_exists=known)
    again = import_archive(make_archive(tmp_path), repo, commit_exists=known)
    assert again["unchanged"] == ["demo_gpu"]
    assert again["imported"] == []
    newer = import_archive(make_archive(tmp_path, run_id="demo_gpu-2"), repo, commit_exists=known)
    assert newer["archived"] == ["demo_gpu-1"]
    assert (repo / "results/historical/demo_gpu-1/run.json").exists()
    assert json.loads((repo / "results/demo_gpu/run.json").read_text())["run_id"] == "demo_gpu-2"


@pytest.mark.parametrize(
    ("git", "message"),
    [
        ({"dirty": True, "dirty_paths": ["src/x.py"]}, "modified tree"),
        ({"source": "declared"}, "not read from a git checkout"),
        ({"commit": "b" * 40}, "expected"),
    ],
)
def test_bad_provenance_is_rejected_before_copying(tmp_path, git, message):
    repo = tmp_path / "repo"
    archive = make_archive(tmp_path, **git)
    with pytest.raises(ImportRejected, match=message):
        import_archive(archive, repo, commit=COMMIT, commit_exists=lambda c: True)
    assert not (repo / "results").exists()


def test_unknown_commit_is_rejected(tmp_path):
    archive = make_archive(tmp_path, commit="c" * 40)
    with pytest.raises(ImportRejected, match="not in this repository"):
        import_archive(archive, tmp_path / "repo", commit_exists=known)


def test_archive_checksum_mismatch_is_rejected(tmp_path):
    archive = make_archive(tmp_path)
    with pytest.raises(ImportRejected, match="archive checksum"):
        import_archive(archive, tmp_path / "repo", sha256="0" * 64, commit_exists=known)


def test_altered_artifact_is_rejected(tmp_path):
    archive = make_archive(tmp_path, content=b"edited")
    with pytest.raises(ImportRejected, match="does not match its checksum"):
        import_archive(archive, tmp_path / "repo", commit_exists=known)
