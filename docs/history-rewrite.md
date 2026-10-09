# History rewrite of 2026-10-10

Before the repository was first pushed, one file was removed from every commit:
`results/decomposition/subqueries.csv`. It held the sub-queries a language model
wrote for benchmark queries, which paraphrase benchmark text that this repository
does not redistribute. The file is still written by the decomposition run and its
checksum is still listed in `results/decomposition/run.json`; it is ignored by Git.

Method: `git filter-repo --invert-paths --path results/decomposition/subqueries.csv`
on the only branch, `main`. No remote existed at the time.

## Commit mapping

The file entered history in the commit "Add query-decomposition results", so that
commit and the four after it have new hashes. The 22 earlier commits are unchanged.

| Original | Rewritten | Subject |
|---|---|---|
| `583975fbc2ddfe386d18b885d87f38eea430fdf6` | `47b8f545a80dce2f2cc5e004ab8034a6cff944df` | Add query-decomposition results |
| `914743da290e894dfe3c1d1648b13d2d10b10741` | `0310c42eaabc1eafa8935bb822cf3e0a21dd29fa` | Exempt the notebook from the line-length rule |
| `451faf08285f181ba3c5ded1832fc69ca661ff60` | `dfe9dd9f16297c33d0f21797c59801c4047c4d43` | Add NSF trend results; shorten panel titles in the trend figure |
| `bf542c89972b90995ba3a82d4a6f24b8e9c18e55` | `88a8af7a96c44ea8f68ca5fd9b12edca1a0e2b93` | Score claims in answers that also abstain |
| `270089f7eca71f6df63207ab9fc13beca9d3fb26` | `ae283ffe988390c8df4afc86d5d2c8e7b41bc541` | Add RAG and DAPFAM results, key findings; stop tracking files that quote benchmark text |

Apart from the removed file, each rewritten commit has the same tree, message,
author and dates as the original.

## Effect on run records

Two runs were made at original commit `bf542c8`, which no longer exists in this
repository: `rag_scifact_local` and `retrieval_dapfam`. Their code is the code of
rewritten commit `88a8af7`. The records were not edited. They are kept unchanged in
[results/historical/](../results/historical/README.md), and both experiments were
rerun at a commit that exists; the comparison is in the section below.

All other run records name commits older than the rewrite and are unaffected.

## Reruns
