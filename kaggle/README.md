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

## One-time setup

The repository is private, so Kaggle needs a token to clone it.

1. On GitHub, create a fine-grained personal access token: Settings > Developer
   settings > Fine-grained tokens. Repository access: only
   `research-innovation-monitor`. Permissions: Contents, read-only. Nothing else.
2. On Kaggle, open the notebook, then Add-ons > Secrets > Add a new secret. Name it
   `GITHUB_TOKEN`, paste the token as the value, and tick the box that attaches it to
   the notebook.

The token never appears in the script, the notebook output or the results archive.
The script passes it to `git clone` through that command's environment and searches
the archive for it before finishing.

## Running

1. Create a Kaggle notebook. Settings: accelerator **GPU T4 x2**, internet on.
   For the trends stage, add the dataset `Cornell-University/arxiv` as input.
2. Paste `run_gpu_experiments.py` into a cell. `REPO_URL` and `REF` are already set;
   `REF` is the commit whose code is run. Trim `STAGES` as described below.
3. Run. The last lines print the archive name, its SHA-256 and the status and
   duration of each stage.
4. Download `rim_results_<ref>-<time>.zip` and the `.sha256` file beside it.
5. In the repository, import the archive:

   ```bash
   uv run rim import-results path/to/rim_results_<ref>-<time>.zip --commit <REF>
   uv run python scripts/make_results_tables.py
   ```

   The import checks the archive checksum, that every run record names `REF` and a
   clean tree, and the checksum of every result file, before it copies anything. A
   result directory that already exists with a different run is moved to
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
attached arXiv snapshot (5.6 GB, read in place). Nothing is written to the local
machine except the results archive, which holds summaries, per-query scores and
run records.
