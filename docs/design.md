# Design notes

This document explains the main design decisions: what each component is for, what
simpler baseline it is measured against, and what was considered and not used.

## Components

```
                     ┌────────────── data layer ──────────────┐
 Hub / Zenodo / API →│ loaders → common schema → manifest      │
                     │ stable IDs, dev/test split by hash      │
                     └───────────────┬─────────────────────────┘
                                     │
          ┌──────────────────────────┼───────────────────────────┐
          ▼                          ▼                           ▼
   BM25 (sparse matrix)      dense encoder + exact         topic model on
          │                  inner-product search          embeddings (PCA + k-means)
          └────────┬─────────────────┘                           │
                   ▼                                             ▼
        fusion (RRF, convex)                         yearly shares → trend tests
                   │                                 → rolling-origin backtest
                   ▼                                             │
        cross-encoder reranker                                   │
                   │                                             │
                   ▼                                             ▼
   numbered context → language model            TrendStore (precomputed tables)
   → citation parsing → attribution judge                        │
                   │                                             │
                   └──────────────┬──────────────────────────────┘
                                  ▼
                 router → tools → orchestrator → report
                                  │
                           FastAPI → Streamlit
```

## Decisions

### Lexical retrieval: BM25 implemented over a sparse matrix

- **Problem.** A strong lexical baseline whose parameters can be tuned cheaply.
- **Why this way.** Term counts are stored once; the weights for any (k1, b) are
  recomputed from them in milliseconds, so a 20-point parameter grid costs one
  tokenisation pass. The implementation is verified against a direct transcription
  of the formula.
- **Alternatives.** Pyserini/Lucene (needs a JVM, heavier to reproduce); `rank-bm25`
  (dense per-query loops, slow at 64k documents).

### Dense retrieval: exact search, no vector database

- **Problem.** Nearest neighbours of a query vector among 5k to 1M document vectors.
- **Why this way.** At this size a blocked matrix product is exact and takes
  milliseconds. Approximate indices trade recall for speed; using one here would mix
  index error into every comparison between retrievers.
- **Simpler baseline.** None needed: this is the simplest correct method.
- **When it stops being right.** Tens of millions of vectors, or tight memory. An
  IVF or HNSW index then becomes necessary, and its recall should be reported.

### Hybrid fusion: two methods, both evaluated

- **Problem.** Lexical and dense scores live on different scales.
- **Reciprocal rank fusion** uses ranks only. It needs no calibration and has one
  constant.
- **Convex combination** of min-max normalised scores has one weight, selected on dev
  queries. Bruch et al. (2023) argue it is more sample-efficient and often better
  than RRF; whether that holds on these corpora is an empirical question the
  experiments answer.
- **Baselines.** BM25 alone and dense alone, with BM25 parameters tuned on the same
  dev queries so that the hybrid is not compared against a weak lexical system.

### Reranking: evaluated as a cost, not assumed as a gain

- **Problem.** Cross-encoders read query and document jointly and can reorder a
  candidate list, at a cost linear in the list length.
- **Why evaluated this way.** The experiment reports quality at several depths and
  the measured latency of each, so the result is a quality-latency curve. Whether a
  reranker helps depends on how close its training distribution is to the corpus.

### Generation: citations are parsed and checked, not trusted

- **Problem.** A generated sentence can be unsupported while carrying a citation.
- **Approach.** The model sees numbered passages and must cite them. Markers are
  parsed deterministically; markers that match no passage are counted. Support is
  judged per sentence by an entailment model, following the ALCE definitions of
  citation recall and precision.
- **Why not a language model as judge.** It would be slower, costlier and not
  reproducible. The entailment judge is deterministic, and its own error rate is
  measured against human labels before its verdicts are used.
- **Baseline.** An extractive generator that copies sentences from the context. Its
  citations are correct by construction, which gives a reference for the metrics.
- **Abstention.** Measured by removing the gold documents from the index and
  checking whether the system declines to answer.

### Trend analysis: clustering with measured stability and a forecast that is scored

- **Problem.** "Which topics are emerging" has no ground truth when asked about the
  present.
- **Approach.** Ask the question about the past. Topics are fitted on documents up to
  a cut-off year; growth forecasts are scored against the following years.
- **Why PCA and k-means rather than UMAP and HDBSCAN.** The backtest needs a model
  that assigns unseen documents to existing topics and gives the same answer for the
  same seed. K-means centroids do both directly. HDBSCAN leaves a variable share of
  documents unassigned and UMAP adds a second stochastic stage.
- **Baselines.** Persistence (no change), continuation of the last one-year growth
  (equivalent to count growth for ranking), and a robust linear trend. The ridge
  model is trained only on cut-offs whose outcomes were already observable.
- **Stability.** Agreement across seeds, across subsamples, and between topic models
  fitted at consecutive cut-offs.

### Orchestration: one router, typed tools, no agent-to-agent conversation

- **Problem.** Decide which sources a request needs.
- **Approach.** Routing is multi-label classification over four tools. A rule-based
  router is the baseline; an embedding classifier and a language model are compared
  against it on the same tasks.
- **Why not several conversing agents.** Ingestion is a batch job and trend analysis
  is statistics. Neither benefits from a language model in the loop, and each
  model-to-model hop adds latency and a failure mode without adding information.
- **Separation.** The model writes prose over numbered evidence. Shares, slopes and
  p-values are computed in Python and returned verbatim; they never pass through the
  model.

### Provenance

Every experiment writes `run.json` next to its outputs: configuration and its hash,
seed, Git commit with any uncommitted paths, package versions, device, dataset
manifests (source, revision, licence, checksums), model identifiers and revisions,
and checksums of the result files.

## Evaluation protocol

| Rule | How it is enforced |
|---|---|
| Tuning never sees test queries | The retrieval runner selects every parameter with a function that only receives dev queries; a test changes the test judgments and asserts that no selection changes |
| Dev/test assignment is stable | Hash of the query ID with a fixed salt |
| Paired comparisons | Fisher randomisation test on per-query differences |
| Several comparisons | Holm correction within each pre-specified family |
| Uncertainty | Percentile bootstrap over queries |
| Latency | Measured per query, one query at a time, on the recorded device |
| No temporal leakage | The backtest receives a fitting function and a mask; a test asserts the mask never includes later years |
