# Research and Innovation Monitor

Hybrid retrieval, citation-grounded generation and quantitative trend analysis over
scientific literature, patents and public research grants.

The system answers three kinds of question about a body of research: which documents
are relevant to a technical query, what the retrieved evidence supports, and which
research themes are growing. Each component is evaluated against baselines on public
benchmarks, and every reported number is traceable to a configuration, dataset
revision, model, seed, machine and Git commit.

## Research questions

| | Question |
|---|---|
| RQ1 | Does hybrid lexical and dense retrieval outperform BM25-only and dense-only retrieval on scientific queries, and does the answer change for patents? |
| RQ2 | How much ranking quality does cross-encoder reranking add per unit of latency? |
| RQ3 | Does query decomposition improve retrieval for multi-aspect queries, and does it harm simple ones? |
| RQ4 | What fraction of generated sentences is supported by the passages they cite, and does the system abstain when the evidence is withheld? |
| RQ5 | Does topic growth measured up to year T predict growth in the following years better than persistence and count-based baselines? |
| RQ6 | Do topic shares in research grants lead or lag topic shares in papers? (exploratory) |

## Scope

Included: scholarly papers, patents (as a fixed retrieval benchmark) and public research
grants (NSF awards). Not included: startup and venture-investment data. No open,
redistributable source of startup funding rounds exists, so that part of the problem is
left as a possible extension rather than approximated.

## Data

No dataset is stored in this repository. Each one is downloaded at a pinned revision and
converted to a common schema, with a manifest recording source, revision, licence and
checksums.

| Dataset | Role | Size | Licence |
|---|---|---|---|
| [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) | Literature retrieval | 597 queries, 64,183 papers | MIT (code repository); dataset card states none |
| [SciFact](https://huggingface.co/datasets/BeIR/scifact), [SCIDOCS](https://huggingface.co/datasets/BeIR/scidocs) | Literature retrieval, second domain | 5,183 and 25,657 documents | CC BY-SA 4.0 |
| [DAPFAM](https://huggingface.co/datasets/datalyes/DAPFAM_patent) | Patent prior-art retrieval | 1,247 query families, 45,336 target families | CC BY-NC-SA 4.0 (evaluation only) |

## Installation

```bash
uv venv --python 3.12
uv pip install -e ".[ml,data,app,dev]"
```

The `ml` extra (PyTorch, sentence-transformers) is needed for neural encoders and
rerankers. The unit tests run without it.

## Usage

```bash
rim prepare scifact      # download and convert a dataset
rim env                  # print the provenance of the current environment
pytest                   # run the tests
```

## Results

No experiment has been run yet. Result tables are added only from executed runs.

## License

MIT. Datasets keep their own licences, listed above.
