"""Run the GPU experiments on Kaggle (or any CUDA machine) at a pinned commit.

Paste into a Kaggle notebook cell with a GPU accelerator and internet enabled, set
REPO_URL and REF, and run. The script clones the repository at REF, installs it,
runs the selected stages and writes ``rim_results_<REF>.zip`` to /kaggle/working.

Every run record inside the archive names the commit, the device, the package
versions and the dataset revisions, so results copied back into ``results/`` keep
their provenance. REF must be a full commit hash, not a branch name: a branch can
move between the run and the write-up.

For the arXiv stage, attach the dataset "Cornell-University/arxiv" to the notebook.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/<user>/research-innovation-monitor.git"  # set before running
REF = ""  # full 40-character commit hash
STAGES = ["judge", "retrieval", "rag", "routing", "decomposition", "trends", "chunking"]

WORK = Path(os.environ.get("RIM_WORK_DIR", "/kaggle/working"))
ARXIV_SNAPSHOT = Path("/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json")
DORISMAE_URL = "https://zenodo.org/api/records/8299749/files/DORIS-MAE_dataset_v1.json/content"

COMMANDS = {
    "judge": [["rim", "judge-validation", "configs/rag/judge_validation_gpu.yaml"]],
    "retrieval": [
        ["rim", "prepare", "scifact", "scidocs", "litsearch", "dapfam"],
        *[
            ["rim", "retrieval", f"configs/retrieval/{d}_gpu.yaml"]
            for d in ("scifact", "scidocs", "litsearch", "dapfam")
        ],
    ],
    "rag": [["rim", "prepare", "scifact"], ["rim", "rag", "configs/rag/scifact_gpu.yaml"]],
    "routing": [["rim", "routing", "configs/agents/routing_gpu.yaml"]],
    "decomposition": [
        ["mkdir", "-p", "data/raw/dorismae"],
        ["curl", "-sL", "-o", "data/raw/dorismae/DORIS-MAE_dataset_v1.json", DORISMAE_URL],
        ["rim", "prepare", "dorismae", "litsearch"],
        ["rim", "decomposition", "configs/agents/decomposition_gpu.yaml"],
    ],
    "trends": [
        ["rim", "prepare-nsf", "2010", "2024"],
        ["rim", "prepare-arxiv", str(ARXIV_SNAPSHOT)],
        ["rim", "trends", "configs/trends/nsf.yaml"],
        ["rim", "trends", "configs/trends/arxiv.yaml"],
    ],
    "chunking": [
        ["rim", "prepare", "litsearch_fulltext"],
        ["rim", "chunking", "configs/retrieval/chunking_litsearch.yaml"],
    ],
}


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    if len(REF) != 40:
        sys.exit("Set REF to a full commit hash before running.")
    repo = WORK / "research-innovation-monitor"
    if repo.exists():
        shutil.rmtree(repo)
    sh(["git", "clone", REPO_URL, str(repo)])
    sh(["git", "checkout", "--detach", REF], cwd=repo)
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[ml,data]"], cwd=repo)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

    failed = []
    for stage in STAGES:
        for cmd in COMMANDS[stage]:
            try:
                sh(cmd, cwd=repo)
            except subprocess.CalledProcessError as exc:
                # A failed stage is reported and skipped; its results are simply absent.
                print(f"STAGE FAILED: {stage}: {exc}", flush=True)
                failed.append(stage)
                break

    archive = shutil.make_archive(str(WORK / f"rim_results_{REF[:12]}"), "zip", repo, "results")
    print(f"\nresults archive: {archive}")
    print(f"stages failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
