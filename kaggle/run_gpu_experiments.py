"""Run the GPU experiments on Kaggle (or any CUDA machine) at a pinned commit.

Run as a Kaggle script with the "GPU T4 x2" accelerator and internet enabled. The
script checks the repository out at one commit, installs it, runs the selected stages
and writes ``rim_results_<commit>-<time>.zip`` and its SHA-256 to /kaggle/working.

Every run record inside the archive names the commit, the device, the package
versions and the dataset revisions, so results copied back into ``results/`` keep
their provenance. The commit is always a full hash, not a branch name: a branch can
move between the run and the write-up.

The code comes from one of two places, chosen by SOURCE:

``bundle``  (default) a Git bundle in a private Kaggle dataset attached to the
            script, staged by ``kaggle/prepare.py``. ``SESSION.json`` beside the
            bundle names the commit and the stages. No credential is involved.
``github``  a clone of the private repository at REF. The clone needs a GitHub
            token, read from the Kaggle secret named by TOKEN_SECRET (or, off
            Kaggle, the RIM_GITHUB_TOKEN environment variable). It is passed to git
            through the environment of the clone command only: it is not written to
            the git configuration, to a URL, to the command line or to the
            environment of the experiments, and the finished archive is searched for
            it before the script reports success.

For the trends stage, attach the dataset "Cornell-University/arxiv" as well.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

SOURCE = os.environ.get("RIM_SOURCE", "bundle")  # "bundle" or "github"
REPO_URL = "https://github.com/AdebanjiAdelowo/research-innovation-monitor.git"
# Used with SOURCE = "github" only; a bundle session takes both from SESSION.json.
REF = ""  # full 40-character commit hash
# All stages together may not fit in one 12-hour session; see kaggle/README.md.
STAGES = ["judge", "retrieval", "rag", "routing", "decomposition", "trends", "chunking"]
TOKEN_SECRET = "GITHUB_TOKEN"

BUNDLE = "research-innovation-monitor.bundle"
SESSION = "SESSION.json"
INPUT = Path(os.environ.get("RIM_INPUT_DIR", "/kaggle/input"))
# Outputs. Everything left here is kept by Kaggle, so only the archive is.
WORK = Path(os.environ.get("RIM_WORK_DIR", "/kaggle/working"))
# The checkout, with the datasets it downloads, stays out of the kept outputs.
CHECKOUT = Path(os.environ.get("RIM_CHECKOUT_DIR", "/tmp/rim"))
ARXIV_FILE = "arxiv-metadata-oai-snapshot.json"
ARXIV_SNAPSHOT = "<arxiv snapshot>"  # replaced by the attached file's path at run time
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
    # The NSF trend run is not repeated here: it has a local record, and the arXiv
    # run loads the NSF awards itself for the lead-lag test.
    "trends": [
        ["rim", "prepare-nsf", "2010", "2024"],
        ["rim", "prepare-arxiv", ARXIV_SNAPSHOT],
        ["rim", "trends", "configs/trends/arxiv.yaml"],
    ],
    "chunking": [
        ["rim", "prepare", "litsearch_fulltext"],
        ["rim", "chunking", "configs/retrieval/chunking_litsearch.yaml"],
    ],
}


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def read_token() -> str:
    """The GitHub token, from the environment or the Kaggle secret store."""
    token = os.environ.pop("RIM_GITHUB_TOKEN", "")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret(TOKEN_SECRET)
        except Exception as exc:  # the secret store raises its own error types
            sys.exit(
                f"No GitHub token: attach the Kaggle secret {TOKEN_SECRET!r} to this "
                f"notebook (Add-ons > Secrets). ({type(exc).__name__})"
            )
    token = token.strip()
    if not token:
        sys.exit(f"The secret {TOKEN_SECRET!r} is empty.")
    return token


def basic_credential(token: str) -> str:
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


def git_with_token(args: list[str], token: str, cwd: Path | None = None) -> None:
    """Run a git command that talks to GitHub, without storing or printing the token."""
    basic = basic_credential(token)
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
    }
    print("+ git", " ".join(args), flush=True)
    done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    output = (done.stdout + done.stderr).replace(token, "***").replace(basic, "***")
    if output.strip():
        print(output.strip(), flush=True)
    if done.returncode != 0:
        sys.exit(
            "git could not reach the repository. Check that the token is valid and "
            "has read access to the repository contents."
        )


def find_input(name: str) -> Path:
    """The one attached file called ``name``.

    Kaggle's mount layout has changed over time (``/kaggle/input/<slug>`` and
    ``/kaggle/input/datasets/<owner>/<slug>``), so the file is searched for.
    """
    found = sorted(INPUT.rglob(name)) if INPUT.exists() else []
    if len(found) != 1:
        sys.exit(f"expected one {name} under {INPUT}, found {[str(f) for f in found]}")
    return found[0]


def read_session(bundle: Path) -> tuple[str, list[str]]:
    """Commit and stages of a bundle session, from the file staged beside the bundle."""
    session = json.loads((bundle.parent / SESSION).read_text())
    return session["commit"], list(session["stages"])


def checkout(source: str, ref: str, token: str = "") -> Path:
    """A checkout of exactly ``ref``, from the Git bundle or from GitHub."""
    repo = CHECKOUT / "research-innovation-monitor"
    if repo.exists():
        shutil.rmtree(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if source == "bundle":
        sh(["git", "clone", "--quiet", "--no-checkout", str(find_input(BUNDLE)), str(repo)])
    else:
        git_with_token(["clone", "--no-checkout", REPO_URL, str(repo)], token)
    sh(["git", "checkout", "--quiet", "--detach", ref], cwd=repo)
    head = capture(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != ref:
        sys.exit(f"Checked out {head}, expected {ref}.")
    return repo


def collect_results(repo: Path, collected: Path) -> list[str]:
    """Move result directories written by a run out of the checkout.

    The checkout then stays identical to REF, so the next run records a clean tree.
    """
    tracked = set(capture(["git", "ls-tree", "--name-only", "HEAD", "results/"], cwd=repo).split())
    moved = []
    for path in sorted((repo / "results").iterdir()):
        if path.is_dir() and f"results/{path.name}" not in tracked:
            target = collected / "results" / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(path), str(target))
            moved.append(path.name)
    return moved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def archive_contains(archive: Path, needles: list[bytes]) -> bool:
    with zipfile.ZipFile(archive) as zf:
        return any(n in zf.read(name) for name in zf.namelist() for n in needles)


def main() -> None:
    if SOURCE not in ("bundle", "github"):
        sys.exit(f"SOURCE must be 'bundle' or 'github', not {SOURCE!r}.")
    ref, stage_names, token = REF, STAGES, ""
    if SOURCE == "bundle":
        ref, stage_names = read_session(find_input(BUNDLE))
    if len(ref) != 40:
        sys.exit("The commit must be a full 40-character hash.")
    unknown = [s for s in stage_names if s not in COMMANDS]
    if unknown:
        sys.exit(f"Unknown stages: {unknown}")
    arxiv = str(find_input(ARXIV_FILE)) if "trends" in stage_names else ""
    started = utc_now()
    if SOURCE == "github":
        token = read_token()
    repo = checkout(SOURCE, ref, token)
    print(f"source: {SOURCE}  commit: {ref}  stages: {stage_names}", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[ml,data,gpu]"], cwd=repo)
    gpus = capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    print("GPUs:", gpus or "none found", flush=True)

    collected = CHECKOUT / "rim_collected"
    if collected.exists():
        shutil.rmtree(collected)
    (collected / "results").mkdir(parents=True)

    stages = []
    for stage in stage_names:
        entry = {"stage": stage, "status": "completed", "started_at": utc_now(), "results": []}
        clock = time.perf_counter()
        for cmd in COMMANDS[stage]:
            cmd = [arxiv if part == ARXIV_SNAPSHOT else part for part in cmd]
            try:
                sh(cmd, cwd=repo)
            except subprocess.CalledProcessError as exc:
                # A failed stage is reported and skipped; its results are simply absent.
                print(f"STAGE FAILED: {stage}: {exc}", flush=True)
                entry["status"] = "failed"
                entry["failed_command"] = " ".join(cmd)
                break
            finally:
                entry["results"] += collect_results(repo, collected)
        entry["seconds"] = round(time.perf_counter() - clock, 1)
        stages.append(entry)

    manifest = {
        "repository": REPO_URL,
        "source": SOURCE,
        "commit": ref,
        "tree_clean_at_end": capture(["git", "status", "--porcelain"], cwd=repo) == "",
        "started_at": started,
        "finished_at": utc_now(),
        "gpus": gpus.splitlines(),
        "python": sys.version.split()[0],
        "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "stages_requested": stage_names,
        "stages": stages,
    }
    stamp = started.replace("-", "").replace(":", "")
    manifest_path = collected / "results" / "gpu_sessions" / f"{ref[:12]}-{stamp}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    base = WORK / f"rim_results_{ref[:12]}-{stamp}"
    archive = Path(shutil.make_archive(str(base), "zip", collected, "results"))
    shutil.rmtree(collected)
    needles = [token.encode(), basic_credential(token).encode()] if token else []
    if needles and archive_contains(archive, needles):
        archive.unlink()
        sys.exit("The access token was found in the results; the archive was deleted.")
    digest = sha256_file(archive)
    Path(f"{archive}.sha256").write_text(f"{digest}  {archive.name}\n")

    print(f"\nresults archive: {archive}")
    print(f"sha256: {digest}")
    for entry in stages:
        print(f"  {entry['stage']:<14} {entry['status']:<10} {entry['seconds']:>9.1f} s")


if __name__ == "__main__":
    main()
