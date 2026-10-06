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
