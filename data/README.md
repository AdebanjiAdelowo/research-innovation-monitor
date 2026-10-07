# Data

Nothing in this directory is tracked except this file. Datasets are fetched with
`rim prepare <name>` into:

- `raw/`: files as downloaded, at the revision pinned in the loader.
- `processed/<name>/`: `corpus.parquet`, `queries.parquet`, `qrels.parquet` and
  `manifest.json` (source, revision, licence, row counts, SHA-256 of each file).
- `cache/`: embeddings and indices derived from processed data.

`RetrievalDataset.load(path, verify=True)` checks the files against the manifest.

## Licences

| Dataset | Licence | Consequence here |
|---|---|---|
| SciFact, SCIDOCS (BEIR) | CC BY-SA 4.0 | Downloaded on demand, not committed |
| LitSearch | MIT on the code repository; no licence on the dataset card | Treated as not redistributable |
| DAPFAM | CC BY-NC-SA 4.0 | Evaluation only, never committed or redistributed |

Tests use a small synthetic corpus defined in `tests/conftest.py` and need none of these.

## Sources fetched by other commands

| Data | Command | Notes |
|---|---|---|
| NSF awards | `rim prepare-nsf 2010 2024` | Public domain. About 2 GB transferred, 0.7 GB kept. Resumable by month. |
| DORIS-MAE | download the 1.08 GB JSON from Zenodo into `raw/dorismae/`, then `rim prepare dorismae` | CC BY-NC 4.0. The processed files are 4 MB; the raw file can be deleted afterwards. |
| SciFact claims | fetched on first use by `rim judge-validation` | 3 MB. CC BY 4.0 (claims), ODC-By 1.0 (abstracts). |
| arXiv metadata | `rim prepare-arxiv <snapshot.json>` | CC0. The snapshot is about 5 GB; intended to be read where it is hosted (Kaggle). |
