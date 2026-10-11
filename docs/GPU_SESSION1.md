# GPU session 1: evaluation

Source commit: `2fe9d3136c99b067e71ccfd40aa7719853b4c8a5`. Two Tesla T4 GPUs; Python 3.13.15; seed 13. Judge thresholds use 919 train pairs and report 340 dev pairs. Retrieval model, fusion and headline rerank choices use dev queries only.

## Retrieval systems

95% intervals are query-bootstrap intervals. Comparisons use paired randomization tests with Holm correction within each benchmark. The intervals do not capture variation across training seeds or dataset versions.

### scifact

Run record: [run.json](../results/retrieval_scifact_gpu/run.json). Dev/test queries: 809/300.

Dev-selected headline: `hybrid_convex+rerank:bge-reranker-v2-m3@10`. Fusion encoder: `bge-base`.

| System | Test nDCG@10 | 95% CI | Recall@20 | MRR@10 |
|---|---:|---|---:|---:|
| bm25 | 0.6826 | [0.6385, 0.7261] | 0.8588 | 0.6497 |
| dense:minilm-l6 | 0.6484 | [0.6034, 0.6929] | 0.8440 | 0.6068 |
| dense:bge-small | 0.7057 | [0.6625, 0.7487] | 0.8763 | 0.6725 |
| dense:bge-base | 0.7421 | [0.7011, 0.7818] | 0.9182 | 0.7069 |
| dense:specter | 0.4798 | [0.4327, 0.5276] | 0.7286 | 0.4349 |
| hybrid_rrf | 0.7395 | [0.6999, 0.7778] | 0.9292 | 0.6992 |
| hybrid_convex | 0.7598 | [0.7201, 0.7986] | 0.9232 | 0.7272 |
| hybrid_convex+rerank:minilm-l6-ce@10 | 0.7292 | [0.6890, 0.7682] | 0.9232 | 0.6852 |
| hybrid_convex+rerank:minilm-l6-ce@20 | 0.7068 | [0.6642, 0.7480] | 0.9232 | 0.6692 |
| hybrid_convex+rerank:minilm-l6-ce@50 | 0.6941 | [0.6496, 0.7373] | 0.8853 | 0.6600 |
| hybrid_convex+rerank:minilm-l6-ce@100 | 0.6816 | [0.6355, 0.7262] | 0.8669 | 0.6522 |
| hybrid_convex+rerank:bge-reranker-v2-m3@10 | 0.7592 | [0.7194, 0.7986] | 0.9232 | 0.7240 |
| hybrid_convex+rerank:bge-reranker-v2-m3@20 | 0.7611 | [0.7208, 0.8001] | 0.9232 | 0.7269 |
| hybrid_convex+rerank:bge-reranker-v2-m3@50 | 0.7509 | [0.7089, 0.7916] | 0.9260 | 0.7214 |
| hybrid_convex+rerank:bge-reranker-v2-m3@100 | 0.7463 | [0.7041, 0.7880] | 0.9070 | 0.7187 |

| System | Baseline | nDCG@10 difference | Paired 95% CI | p (Holm) |
|---|---|---:|---|---:|
| dense:bge-base | bm25 | +0.0595 | [+0.0268, +0.0924] | 0.0020 |
| hybrid_rrf | bm25 | +0.0568 | [+0.0368, +0.0772] | 0.0007 |
| hybrid_rrf | dense:bge-base | -0.0026 | [-0.0230, +0.0180] | 1.0000 |
| hybrid_convex | bm25 | +0.0772 | [+0.0486, +0.1060] | 0.0007 |
| hybrid_convex | dense:bge-base | +0.0177 | [+0.0047, +0.0311] | 0.0412 |
| hybrid_convex | hybrid_rrf | +0.0204 | [+0.0048, +0.0364] | 0.0412 |
| hybrid_convex+rerank:bge-reranker-v2-m3@10 | hybrid_convex | -0.0006 | [-0.0197, +0.0186] | 1.0000 |

### scidocs

Run record: [run.json](../results/retrieval_scidocs_gpu/run.json). Dev/test queries: 298/702.

Dev-selected headline: `hybrid_convex+rerank:bge-reranker-v2-m3@10`. Fusion encoder: `minilm-l6`.

| System | Test nDCG@10 | 95% CI | Recall@20 | MRR@10 |
|---|---:|---|---:|---:|
| bm25 | 0.1508 | [0.1365, 0.1657] | 0.2107 | 0.2704 |
| dense:minilm-l6 | 0.2080 | [0.1917, 0.2246] | 0.3017 | 0.3563 |
| dense:bge-small | 0.1986 | [0.1827, 0.2151] | 0.2737 | 0.3393 |
| dense:bge-base | 0.2114 | [0.1947, 0.2285] | 0.2910 | 0.3535 |
| dense:specter | 0.1389 | [0.1253, 0.1531] | 0.2093 | 0.2550 |
| hybrid_rrf | 0.2009 | [0.1847, 0.2172] | 0.2831 | 0.3444 |
| hybrid_convex | 0.2104 | [0.1939, 0.2269] | 0.3062 | 0.3605 |
| hybrid_convex+rerank:minilm-l6-ce@10 | 0.2017 | [0.1858, 0.2176] | 0.3062 | 0.3335 |
| hybrid_convex+rerank:minilm-l6-ce@20 | 0.1939 | [0.1781, 0.2100] | 0.3062 | 0.3217 |
| hybrid_convex+rerank:minilm-l6-ce@50 | 0.1811 | [0.1653, 0.1968] | 0.2751 | 0.3099 |
| hybrid_convex+rerank:minilm-l6-ce@100 | 0.1656 | [0.1502, 0.1811] | 0.2441 | 0.2931 |
| hybrid_convex+rerank:bge-reranker-v2-m3@10 | 0.2007 | [0.1849, 0.2162] | 0.3062 | 0.3273 |
| hybrid_convex+rerank:bge-reranker-v2-m3@20 | 0.1936 | [0.1778, 0.2095] | 0.3062 | 0.3090 |
| hybrid_convex+rerank:bge-reranker-v2-m3@50 | 0.1819 | [0.1666, 0.1975] | 0.2851 | 0.2969 |
| hybrid_convex+rerank:bge-reranker-v2-m3@100 | 0.1680 | [0.1531, 0.1834] | 0.2604 | 0.2805 |

| System | Baseline | nDCG@10 difference | Paired 95% CI | p (Holm) |
|---|---|---:|---|---:|
| dense:minilm-l6 | bm25 | +0.0572 | [+0.0456, +0.0689] | 0.0007 |
| hybrid_rrf | bm25 | +0.0501 | [+0.0430, +0.0575] | 0.0007 |
| hybrid_rrf | dense:minilm-l6 | -0.0071 | [-0.0149, +0.0004] | 0.1294 |
| hybrid_convex | bm25 | +0.0595 | [+0.0485, +0.0707] | 0.0007 |
| hybrid_convex | dense:minilm-l6 | +0.0024 | [-0.0004, +0.0052] | 0.1294 |
| hybrid_convex | hybrid_rrf | +0.0095 | [+0.0029, +0.0164] | 0.0176 |
| hybrid_convex+rerank:bge-reranker-v2-m3@10 | hybrid_convex | -0.0097 | [-0.0165, -0.0032] | 0.0176 |

### litsearch

Run record: [run.json](../results/retrieval_litsearch_gpu/run.json). Dev/test queries: 173/424.

Dev-selected headline: `hybrid_rrf+rerank:bge-reranker-v2-m3@100`. Fusion encoder: `bge-base`.

| System | Test nDCG@10 | 95% CI | Recall@20 | MRR@10 |
|---|---:|---|---:|---:|
| bm25 | 0.4371 | [0.3963, 0.4787] | 0.6352 | 0.3990 |
| dense:minilm-l6 | 0.3661 | [0.3283, 0.4048] | 0.6081 | 0.3243 |
| dense:bge-small | 0.4499 | [0.4100, 0.4903] | 0.6659 | 0.4092 |
| dense:bge-base | 0.4651 | [0.4248, 0.5064] | 0.6863 | 0.4204 |
| dense:specter | 0.2037 | [0.1719, 0.2366] | 0.3605 | 0.1764 |
| hybrid_rrf | 0.5324 | [0.4915, 0.5731] | 0.7382 | 0.4956 |
| hybrid_convex | 0.5421 | [0.5008, 0.5836] | 0.7394 | 0.5079 |
| hybrid_rrf+rerank:minilm-l6-ce@10 | 0.5101 | [0.4706, 0.5504] | 0.7382 | 0.4650 |
| hybrid_rrf+rerank:minilm-l6-ce@20 | 0.5118 | [0.4730, 0.5517] | 0.7382 | 0.4630 |
| hybrid_rrf+rerank:minilm-l6-ce@50 | 0.4950 | [0.4552, 0.5349] | 0.7296 | 0.4510 |
| hybrid_rrf+rerank:minilm-l6-ce@100 | 0.4847 | [0.4444, 0.5247] | 0.7146 | 0.4409 |
| hybrid_rrf+rerank:bge-reranker-v2-m3@10 | 0.5595 | [0.5187, 0.6004] | 0.7382 | 0.5306 |
| hybrid_rrf+rerank:bge-reranker-v2-m3@20 | 0.5677 | [0.5278, 0.6074] | 0.7382 | 0.5321 |
| hybrid_rrf+rerank:bge-reranker-v2-m3@50 | 0.5612 | [0.5208, 0.6011] | 0.7669 | 0.5244 |
| hybrid_rrf+rerank:bge-reranker-v2-m3@100 | 0.5611 | [0.5209, 0.6010] | 0.7559 | 0.5243 |

| System | Baseline | nDCG@10 difference | Paired 95% CI | p (Holm) |
|---|---|---:|---|---:|
| dense:bge-base | bm25 | +0.0281 | [-0.0077, +0.0633] | 0.2526 |
| hybrid_rrf | bm25 | +0.0953 | [+0.0708, +0.1194] | 0.0007 |
| hybrid_rrf | dense:bge-base | +0.0672 | [+0.0423, +0.0928] | 0.0007 |
| hybrid_convex | bm25 | +0.1050 | [+0.0814, +0.1296] | 0.0007 |
| hybrid_convex | dense:bge-base | +0.0770 | [+0.0524, +0.1033] | 0.0007 |
| hybrid_convex | hybrid_rrf | +0.0097 | [-0.0026, +0.0225] | 0.2526 |
| hybrid_rrf+rerank:bge-reranker-v2-m3@100 | hybrid_rrf | +0.0288 | [+0.0009, +0.0573] | 0.1368 |

### dapfam

Run record: [run.json](../results/retrieval_dapfam_gpu/run.json). Dev/test queries: 355/892.

Dev-selected headline: `hybrid_convex+rerank:minilm-l6-ce@10`. Fusion encoder: `bge-base`.

| System | Test nDCG@10 | 95% CI | Recall@20 | MRR@10 |
|---|---:|---|---:|---:|
| bm25 | 0.1931 | [0.1792, 0.2075] | 0.1242 | 0.4046 |
| dense:minilm-l6 | 0.2104 | [0.1964, 0.2252] | 0.1400 | 0.4276 |
| dense:bge-small | 0.2175 | [0.2034, 0.2326] | 0.1427 | 0.4440 |
| dense:bge-base | 0.2354 | [0.2205, 0.2513] | 0.1590 | 0.4476 |
| dense:specter | 0.1835 | [0.1709, 0.1968] | 0.1213 | 0.3951 |
| hybrid_rrf | 0.2386 | [0.2236, 0.2549] | 0.1591 | 0.4546 |
| hybrid_convex | 0.2433 | [0.2282, 0.2596] | 0.1637 | 0.4651 |
| hybrid_convex+rerank:minilm-l6-ce@10 | 0.2318 | [0.2171, 0.2478] | 0.1637 | 0.4258 |
| hybrid_convex+rerank:minilm-l6-ce@20 | 0.2130 | [0.1990, 0.2277] | 0.1637 | 0.4035 |
| hybrid_convex+rerank:minilm-l6-ce@50 | 0.1932 | [0.1798, 0.2073] | 0.1468 | 0.3818 |
| hybrid_convex+rerank:minilm-l6-ce@100 | 0.1791 | [0.1661, 0.1929] | 0.1313 | 0.3673 |
| hybrid_convex+rerank:bge-reranker-v2-m3@10 | 0.2342 | [0.2194, 0.2502] | 0.1637 | 0.4330 |
| hybrid_convex+rerank:bge-reranker-v2-m3@20 | 0.2213 | [0.2071, 0.2368] | 0.1637 | 0.4184 |
| hybrid_convex+rerank:bge-reranker-v2-m3@50 | 0.1994 | [0.1860, 0.2138] | 0.1494 | 0.3940 |
| hybrid_convex+rerank:bge-reranker-v2-m3@100 | 0.1855 | [0.1729, 0.1995] | 0.1373 | 0.3788 |

| System | Baseline | nDCG@10 difference | Paired 95% CI | p (Holm) |
|---|---|---:|---|---:|
| dense:bge-base | bm25 | +0.0423 | [+0.0325, +0.0522] | 0.0007 |
| hybrid_rrf | bm25 | +0.0455 | [+0.0393, +0.0519] | 0.0007 |
| hybrid_rrf | dense:bge-base | +0.0032 | [-0.0028, +0.0092] | 0.3081 |
| hybrid_convex | bm25 | +0.0502 | [+0.0429, +0.0576] | 0.0007 |
| hybrid_convex | dense:bge-base | +0.0078 | [+0.0028, +0.0130] | 0.0090 |
| hybrid_convex | hybrid_rrf | +0.0046 | [+0.0016, +0.0077] | 0.0090 |
| hybrid_convex+rerank:minilm-l6-ce@10 | hybrid_convex | -0.0114 | [-0.0155, -0.0075] | 0.0007 |

## Judge evaluation

Run record: [run.json](../results/judge_validation_gpu/run.json).

Thresholds are selected on train; metrics below use dev pairs.

| Judge | Threshold | Accuracy | Accuracy 95% CI | Balanced accuracy | AUROC |
|---|---:|---:|---|---:|---:|
| nli-deberta-v3-xsmall | 0.1 | 0.6676 | [0.6176, 0.7147] | 0.6032 | 0.7693 |
| deberta-v3-base-mnli-fever-anli | 0.4 | 0.7676 | [0.7206, 0.8118] | 0.7551 | 0.8580 |
| deberta-v3-large-mnli-fever-anli-ling-wanli | 0.5 | 0.8324 | [0.7912, 0.8706] | 0.8199 | 0.9101 |

## Interpretation and limits

Dense retrieval beats BM25 in the pre-specified comparisons on all four datasets. Larger models do not uniformly improve retrieval: SPECTER scores below BM25 on every dataset, and bge-base is not the dev-selected encoder on SCIDOCS. These SPECTER comparisons are descriptive, without a recorded paired test.

Reranking harms SCIDOCS and DAPFAM at the dev-selected headline settings. SciFact is inconclusive. LitSearch has a positive estimate whose paired interval excludes zero, but its Holm-adjusted p is 0.1368, so the corrected test is inconclusive. Selecting a different depth after seeing test scores would change the protocol.

The larger judge improves observed dev accuracy, but this does not establish improved end-to-end RAG attribution. No GPU generation was requested in this session.

See [scientific audit](SCIENTIFIC_AUDIT.md) for provenance limits and checks.

Regenerate with `PYTHONPATH=src .venv/bin/python scripts/make_gpu_report.py`; check scores with `PYTHONPATH=src .venv/bin/python scripts/check_gpu_results.py`.
