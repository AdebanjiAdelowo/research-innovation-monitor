# GPU runs

`run_gpu_experiments.py` runs the experiments that are too heavy for a laptop on a
CUDA machine, at a pinned commit:

| Stage | Configs | What it adds over the local runs |
|---|---|---|
| judge | `configs/rag/judge_validation_gpu.yaml` | A larger entailment judge |
| retrieval | `configs/retrieval/*_gpu.yaml` | Larger encoders and `bge-reranker-v2-m3` |
| rag | `configs/rag/scifact_gpu.yaml` | An open 7B generator on all 300 test claims |
| routing, decomposition | `configs/agents/*_gpu.yaml` | The 7B model as router and decomposer |
| trends | `configs/trends/arxiv.yaml` | arXiv backtest and the grants lead-lag test |
| chunking | `configs/retrieval/chunking_litsearch.yaml` | Chunk-size ablation on full text |

## Where the code comes from

The repository is private. The runner gets the code in one of two ways.

**Git bundle (default).** `prepare.py` writes a Git bundle of the current commit and a
`SESSION.json` naming that commit and the stages to run. Both go into a private
Kaggle dataset attached to the script. The runner clones the bundle, checks out the
commit and verifies the hash, so each run record names a real commit and a clean
tree. No credential is involved. The bundle holds only what Git tracks: no benchmark
data and no model weights.

**GitHub clone.** With `SOURCE = "github"` the runner clones the repository at `REF`
using a token from the Kaggle secret `GITHUB_TOKEN`: a fine-grained personal access
token limited to this repository with Contents read-only. The token is passed to
`git clone` through that command's environment only, and the archive is searched for
it before the script finishes. This route is for running the script by hand in a
notebook, where a secret can be attached.

## Running a session

```bash
# 1. Stage one session from a clean tree (uploads nothing).
uv run python kaggle/prepare.py --username <kaggle username> \
    --stages judge retrieval --out ~/kaggle-staging/research-innovation-monitor

# 2. Upload the bundle as a private dataset (create once, then add versions).
kaggle datasets create  -p ~/kaggle-staging/research-innovation-monitor/source
kaggle datasets version -p ~/kaggle-staging/research-innovation-monitor/source -m "<commit>"

# 3. Submit the script. Kaggle runs it on its own; nothing has to stay open.
kaggle kernels push -p ~/kaggle-staging/research-innovation-monitor/kernel

# 4. Later: check, then fetch the archive and its checksum.
kaggle kernels status <kaggle username>/research-innovation-monitor-gpu
kaggle kernels output <kaggle username>/research-innovation-monitor-gpu -p ~/kaggle-staging/output

# 5. Import and regenerate the tables.
uv run rim import-results ~/kaggle-staging/output/rim_results_<commit>-<time>.zip --commit <commit>
uv run python scripts/make_results_tables.py
```

The kernel metadata asks for a private script with a GPU ("T4 x2"), internet access
and the source dataset; a session with the trends stage also attaches
`Cornell-University/arxiv`. The script prints the GPUs it found, and the session
manifest records them.

The import checks the archive checksum, that every run record names the expected
commit and a clean tree, and the checksum of every result file, before it copies
anything. A result directory that already exists with a different run is moved to
`results/historical/` first.

## Sessions

A Kaggle session ends after 12 hours and the weekly GPU allowance is limited. The
durations below are estimates made before any GPU run, not measurements; the
session manifest in each archive (`results/gpu_sessions/`) records the real ones.

| Session | `STAGES` | Estimate |
|---|---|---|
| 1 | `["judge", "retrieval"]` | 2 to 4 hours |
| 2 | `["rag", "routing", "decomposition"]` | 3 to 6 hours |
| 3 | `["trends", "chunking"]` | 2 to 5 hours |

A failed stage does not stop the later ones: it is marked `failed` in the manifest
and its results are absent from the archive.

## Models and memory

| Model | Revision | Licence | Weights | Use |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | `a09a3545` | Apache 2.0 | 15.2 GB | Generator, router, decomposer |
| `BAAI/bge-base-en-v1.5` | `a5beb1e3` | MIT | 0.4 GB | Encoder |
| `sentence-transformers/allenai-specter` | `2c68eeca` | Apache 2.0 | 0.4 GB | Encoder |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6` | Apache 2.0 | 2.3 GB | Reranker |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | `b3546ea6` | MIT | 0.9 GB | Entailment judge |

Revisions, licences and sizes were read from the Hugging Face Hub on 2026-10-10;
none of the models is gated. The configs hold the full revision hashes.

The 7B generator in half precision needs about 15.2 GB for its weights alone, which
does not fit on one 16 GB T4 (about 15 GB usable). It is therefore loaded in half
precision across both cards of a "T4 x2" notebook (`device_map="auto"`), with greedy
decoding. It is not quantized. If only a single GPU is available, the stage fails with
an out-of-memory error rather than falling back to another configuration; a
quantized run would be a different experiment and would need its own config and name.

Approximate downloads on the GPU machine: models 20 GB, datasets 4 GB, plus the
attached arXiv snapshot (5.6 GB, read in place). The checkout and the datasets live
outside Kaggle's kept output folder, which receives only the results archive and its
checksum. The archive holds summaries, per-query scores, run records and the raw
per-run files that Git ignores.
