"""Stage the files one Kaggle GPU session needs, locally. Uploads nothing.

    python kaggle/prepare.py --username <kaggle username> --stages judge retrieval \\
        --out /path/outside/the/repo

Writes two folders under ``--out``:

``source/``   a private Kaggle dataset: ``research-innovation-monitor.bundle`` (a Git
              bundle of the current commit and its history), ``SESSION.json`` (the
              commit and the stages to run) and ``dataset-metadata.json``
``kernel/``   the runner as it is at that commit, and ``kernel-metadata.json``

The bundle holds exactly what Git tracks: code, documentation and aggregate result
files. It contains no benchmark data and no model weights.

The script refuses to run on a modified working tree, because the point is that
Kaggle evaluates one identifiable commit. It prints the ``kaggle`` commands to run;
it does not run them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "research-innovation-monitor.bundle"
SESSION = "SESSION.json"
SOURCE_SLUG = "research-innovation-monitor-src"
KERNEL_SLUG = "research-innovation-monitor-gpu"
ARXIV_DATASET = "Cornell-University/arxiv"
STAGES = ("judge", "retrieval", "rag", "routing", "decomposition", "trends", "chunking")
# Kaggle's name for the two-card T4 machine.
MACHINE = "NvidiaTeslaT4"


def git(*args: str, root: Path = ROOT) -> str:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def stage(out: Path, username: str, stages: list[str], root: Path = ROOT) -> dict:
    unknown = [s for s in stages if s not in STAGES]
    if unknown or not stages:
        raise SystemExit(f"stages must be chosen from {list(STAGES)}; got {stages}")
    dirty = git("status", "--porcelain", root=root)
    if dirty:
        raise SystemExit(f"commit or stash these changes first:\n{dirty}")
    if out.resolve().is_relative_to(root.resolve()):
        raise SystemExit("--out must be outside the repository")
    commit = git("rev-parse", "HEAD", root=root)
    source, kernel = out / "source", out / "kernel"
    for folder in (source, kernel):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    git("bundle", "create", str(source / BUNDLE), "HEAD", root=root)
    write_json(source / SESSION, {"commit": commit, "stages": stages})
    write_json(
        source / "dataset-metadata.json",
        {
            "title": SOURCE_SLUG,
            "id": f"{username}/{SOURCE_SLUG}",
            "licenses": [{"name": "MIT"}],
        },
    )

    # The runner is taken from the commit, not from the working tree.
    runner = git("show", f"{commit}:kaggle/run_gpu_experiments.py", root=root)
    (kernel / "run_gpu_experiments.py").write_text(runner + "\n")
    datasets = [f"{username}/{SOURCE_SLUG}"]
    if "trends" in stages:
        datasets.append(ARXIV_DATASET)
    write_json(
        kernel / "kernel-metadata.json",
        {
            "id": f"{username}/{KERNEL_SLUG}",
            "title": KERNEL_SLUG,
            "code_file": "run_gpu_experiments.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": MACHINE,
            "dataset_sources": datasets,
            "competition_sources": [],
            "kernel_sources": [],
        },
    )
    return {"commit": commit, "stages": stages, "source": source, "kernel": kernel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--username", required=True, help="Kaggle username")
    parser.add_argument("--stages", required=True, nargs="+", choices=STAGES)
    parser.add_argument("--out", required=True, type=Path, help="staging folder outside the repo")
    args = parser.parse_args()
    staged = stage(args.out, args.username, args.stages)
    size = (staged["source"] / BUNDLE).stat().st_size / 1e6
    print(f"commit   {staged['commit']}")
    print(f"stages   {' '.join(staged['stages'])}")
    print(f"bundle   {staged['source'] / BUNDLE} ({size:.1f} MB)")
    print(f"kernel   {staged['kernel']}")
    print("\nNothing has been uploaded. To run on Kaggle:")
    print(f"  kaggle datasets create -p {staged['source']}        # first time (private)")
    print(f"  kaggle datasets version -p {staged['source']} -m {staged['commit'][:12]}   # later")
    print(f"  kaggle kernels push -p {staged['kernel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
