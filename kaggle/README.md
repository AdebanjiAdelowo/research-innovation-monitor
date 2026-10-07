# GPU runs

`run_gpu_experiments.py` runs the experiments that are too heavy for a laptop on a
CUDA machine, at a pinned commit:

| Stage | Configs | What it adds over the local runs |
|---|---|---|
| judge | `configs/rag/judge_validation_gpu.yaml` | A larger entailment judge |
| retrieval | `configs/retrieval/*_gpu.yaml` | Larger encoders and `bge-reranker-v2-m3` |
| rag | `configs/rag/scifact_gpu.yaml` | An open 7B generator on all 300 test claims |
| routing, decomposition | `configs/agents/*_gpu.yaml` | The 7B model as router and decomposer |
| trends | `configs/trends/nsf.yaml`, `configs/trends/arxiv.yaml` | arXiv backtest and the grants lead-lag test |
| chunking | `configs/retrieval/chunking_litsearch.yaml` | Chunk-size ablation on full text |

Steps:

1. Create a Kaggle notebook with a GPU accelerator and internet access. For the
   trends stage, attach the dataset `Cornell-University/arxiv`.
2. Paste the script into a cell. Set `REPO_URL` and `REF` (a full commit hash), and
   trim `STAGES` if the session limit is a concern.
3. Run. Download `rim_results_<ref>.zip` from the output.
4. Unzip it into the repository root. Each `results/<experiment>/run.json` records the
   commit, device, package versions and dataset revisions of the run.

Approximate downloads on the GPU machine: models 20 GB (the 7B generator is 15 GB),
datasets 4 GB. Nothing is written to the local machine.
