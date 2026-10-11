# Kaggle handoff

Where the GPU experiments stand and how to continue. Updated 2026-10-11.

## Current submission: session 1

| | |
|---|---|
| Kernel | `adebanjiadelowo/research-innovation-monitor-gpu` (private script), version 1 |
| URL | https://www.kaggle.com/code/adebanjiadelowo/research-innovation-monitor-gpu |
| Submitted | 2026-10-11, about 12:03 UTC; started 12:27 UTC |
| Commit run | `2fe9d3136c99b067e71ccfd40aa7719853b4c8a5` |
| Code source | Git bundle in the private dataset `adebanjiadelowo/research-innovation-monitor-src`, version 1 |
| Stages | `judge`, `retrieval` |
| Machine | GPU T4 x2 (`NvidiaTeslaT4`), internet on |
| Staging folder | `~/kaggle-staging/research-innovation-monitor/` (`source/`, `kernel/`) |

Configurations, all at the commit above:

| Stage | Config | Experiment name |
|---|---|---|
| judge | `configs/rag/judge_validation_gpu.yaml` | `judge_validation_gpu` |
| retrieval | `configs/retrieval/scifact_gpu.yaml` | `retrieval_scifact_gpu` |
| retrieval | `configs/retrieval/scidocs_gpu.yaml` | `retrieval_scidocs_gpu` |
| retrieval | `configs/retrieval/litsearch_gpu.yaml` | `retrieval_litsearch_gpu` |
| retrieval | `configs/retrieval/dapfam_gpu.yaml` | `retrieval_dapfam_gpu` |

Expected outputs in the kernel's output folder:

- `rim_results_2fe9d3136c99-<time>.zip`: one `results/<experiment>/` directory per
  experiment that finished (`run.json`, `summary.json`, per-query scores, raw runs)
  and `results/gpu_sessions/2fe9d3136c99-<time>.json`, the session manifest with
  the status and duration of each stage.
- `rim_results_2fe9d3136c99-<time>.zip.sha256`.

A stage that fails is marked `failed` in the manifest and its results are absent;
the other stages still run. Runtime was estimated at 2 to 5 hours before any GPU
run; the manifest records the real durations.

Commit `2fe9d31` is on GitHub: `main` was pushed at `84e49d7` on 2026-10-11 and CI
passed there. The kernel started at 12:27 UTC that day and was `RUNNING` when last checked, at 17:03 UTC.

## When resuming

Do not resubmit before checking. From the repository root:

```bash
# 1. Status: complete, running, error or cancelled.
kaggle kernels status adebanjiadelowo/research-innovation-monitor-gpu

# 2. Fetch the outputs and the log into a folder outside the repository.
kaggle kernels output adebanjiadelowo/research-innovation-monitor-gpu \
    -p ~/kaggle-staging/research-innovation-monitor/output/session1

# 3. Read the manifest inside the archive: every stage should be "completed" and
#    "tree_clean_at_end" should be true. Read the log for any "STAGE FAILED" line.

# 4. Import. This checks the archive checksum, the commit and clean tree in every
#    run record and the checksum of every result file before copying anything.
uv run rim import-results \
    ~/kaggle-staging/research-innovation-monitor/output/session1/rim_results_2fe9d3136c99-<time>.zip \
    --commit 2fe9d3136c99b067e71ccfd40aa7719853b4c8a5

# 5. Regenerate the README tables, then test and lint.
uv run python scripts/make_results_tables.py
uv run pytest && uv run ruff check .
```

Then review the numbers before committing: compare each GPU run's small-model rows
with the local run of the same benchmark (same models and seeds, different device),
and report the larger reranker and judge whichever way they come out.

On this Mac the Kaggle CLI hangs on uploads and some downloads because IPv6 is
advertised but does not work. If a `kaggle` command stalls, run it through a small
wrapper that limits name resolution to IPv4.

If the kernel ended in `error`, read the log first. A failure before the first stage
(bundle not found, install failed) means nothing ran and the fix is in
`kaggle/run_gpu_experiments.py`; commit the fix, restage and push a new version.

## Sessions not yet submitted

All remain PENDING. Submit one at a time, after the previous one is imported.

| Session | Stages | Notes |
|---|---|---|
| 2 | `rag routing decomposition` | Qwen2.5-7B in half precision across both T4s; estimate 3 to 6 hours |
| 3 | `trends chunking` | Attaches `Cornell-University/arxiv` (5.6 GB, read in place); estimate 2 to 5 hours |

To submit the next session from a clean tree:

```bash
uv run python kaggle/prepare.py --username adebanjiadelowo \
    --stages rag routing decomposition --out ~/kaggle-staging/research-innovation-monitor
kaggle datasets version -p ~/kaggle-staging/research-innovation-monitor/source -m "<commit>"
kaggle datasets status adebanjiadelowo/research-innovation-monitor-src    # wait for "ready"
kaggle kernels push -p ~/kaggle-staging/research-innovation-monitor/kernel
```

The kernel uses the newest version of the source dataset, so wait for `ready` before
pushing. Weekly GPU allowance on 2026-10-11 before this submission: 28.2 of 30 hours
left, shared with other projects; it resets on 2026-10-17.
