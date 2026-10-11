import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"


def load(name):
    spec = importlib.util.spec_from_file_location(f"kaggle_{name}", KAGGLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare")
runner = load("run_gpu_experiments")


def git(repo, *args):
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A small repository with two commits and the runner at kaggle/."""
    root = tmp_path / "repo"
    (root / "kaggle").mkdir(parents=True)
    (root / "results" / "tracked_run").mkdir(parents=True)
    git(root, "init", "--quiet", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.org")
    (root / "results" / "tracked_run" / "summary.json").write_text("{}\n")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "first")
    shutil.copy(KAGGLE / "run_gpu_experiments.py", root / "kaggle" / "run_gpu_experiments.py")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "second")
    return root


def test_stage_writes_bundle_session_and_kernel(repo, tmp_path):
    staged = prepare.stage(tmp_path / "out", "someone", ["judge", "retrieval"], root=repo)
    commit = git(repo, "rev-parse", "HEAD")
    assert staged["commit"] == commit
    session = json.loads((staged["source"] / prepare.SESSION).read_text())
    assert session == {"commit": commit, "stages": ["judge", "retrieval"]}
    kernel = json.loads((staged["kernel"] / "kernel-metadata.json").read_text())
    assert kernel["is_private"] and kernel["enable_gpu"] and kernel["enable_internet"]
    assert kernel["dataset_sources"] == ["someone/research-innovation-monitor-src"]
    assert (staged["kernel"] / "run_gpu_experiments.py").read_bytes() == (
        repo / "kaggle" / "run_gpu_experiments.py"
    ).read_bytes()


def test_trends_session_attaches_the_arxiv_dataset(repo, tmp_path):
    staged = prepare.stage(tmp_path / "out", "someone", ["trends"], root=repo)
    kernel = json.loads((staged["kernel"] / "kernel-metadata.json").read_text())
    assert prepare.ARXIV_DATASET in kernel["dataset_sources"]


def test_stage_refuses_a_modified_tree_and_an_output_inside_the_repository(repo, tmp_path):
    with pytest.raises(SystemExit, match="outside"):
        prepare.stage(repo / "out", "someone", ["judge"], root=repo)
    with pytest.raises(SystemExit, match="stages"):
        prepare.stage(tmp_path / "out", "someone", ["nonsense"], root=repo)
    (repo / "untracked.txt").write_text("x")
    with pytest.raises(SystemExit, match="commit or stash"):
        prepare.stage(tmp_path / "out", "someone", ["judge"], root=repo)


def test_runner_checks_out_the_session_commit_from_the_bundle(repo, tmp_path, monkeypatch):
    staged = prepare.stage(tmp_path / "out", "someone", ["judge"], root=repo)
    # Kaggle mounts a dataset below /kaggle/input; the depth has varied.
    mount = tmp_path / "input" / "datasets" / "someone" / "research-innovation-monitor-src"
    shutil.copytree(staged["source"], mount)
    monkeypatch.setattr(runner, "INPUT", tmp_path / "input")
    monkeypatch.setattr(runner, "CHECKOUT", tmp_path / "checkout")

    commit, stages = runner.read_session(runner.find_input(runner.BUNDLE))
    assert (commit, stages) == (staged["commit"], ["judge"])
    checkout = runner.checkout("bundle", commit)
    assert git(checkout, "rev-parse", "HEAD") == commit
    assert git(checkout, "status", "--porcelain") == ""

    # Results written by a run are moved out, so the tree is clean for the next run.
    new_run = checkout / "results" / "new_run"
    new_run.mkdir()
    (new_run / "summary.json").write_text("{}\n")
    collected = tmp_path / "collected"
    (collected / "results").mkdir(parents=True)
    assert runner.collect_results(checkout, collected) == ["new_run"]
    assert git(checkout, "status", "--porcelain") == ""
    assert (collected / "results" / "new_run" / "summary.json").exists()


def test_runner_rejects_a_commit_the_bundle_does_not_hold(repo, tmp_path, monkeypatch):
    staged = prepare.stage(tmp_path / "out", "someone", ["judge"], root=repo)
    monkeypatch.setattr(runner, "INPUT", staged["source"])
    monkeypatch.setattr(runner, "CHECKOUT", tmp_path / "checkout")
    with pytest.raises(subprocess.CalledProcessError):
        runner.checkout("bundle", "0" * 40)


def test_find_input_needs_exactly_one_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "INPUT", tmp_path)
    with pytest.raises(SystemExit, match="expected one"):
        runner.find_input(runner.BUNDLE)
