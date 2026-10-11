# Scientific audit: Kaggle session 1

Audited 2026-10-12. This audit covers kernel version 1 identified in the submission
handoff, source dataset version 1 and commit
`2fe9d3136c99b067e71ccfd40aa7719853b4c8a5`. The fetched output and log name this exact
commit and the submitted `judge`, `retrieval` stage list. Kaggle reports COMPLETE.
The Kaggle kernel metadata API independently reports `current_version_number: 1`,
private visibility, `NvidiaTeslaT4` and the expected source dataset. Its
`last_run_time` field reports 2026-10-08, inconsistent with this execution's log and
manifest timestamps; execution timing therefore uses the log and session manifest.

## Archive and execution

Archive: `rim_results_2fe9d3136c99-20261011T122744Z.zip`.
SHA-256: `83345e374cc0a9728c1a5695ccd4bd287cefd5bbc48e00cd564da2edba3995f0`.
The supplied checksum, log checksum and computed archive checksum agree. ZIP CRC
verification passes. Members are confined to `results/`; the importer validates paths
before extraction. The archive and log remain outside Git in `/private/tmp/rim-session1/`.
Raw ranking files are retained locally and ignored by Git.

Session start/end: 2026-10-11 12:27:44 / 22:09:20 UTC. The manifest records two
Tesla T4 GPUs (15,360 MiB each), Python 3.13.15, bundle source and a clean final tree.
Each of the five run records has verified Git source, the expected full commit,
`dirty: false`, CUDA device and a finished timestamp. Every listed result artifact
passes SHA-256 validation. Raw rankings are not listed in the run records' artifact
maps, so their integrity is covered by the outer archive checksum and metric
recomputation, not individual run-record hashes.

| Stage | Manifest status | Duration | Finished results |
|---|---|---:|---|
| judge | completed | 162.2 s | judge_validation_gpu |
| retrieval | completed | 34,719.3 s | SciFact, SCIDOCS, LitSearch, DAPFAM GPU retrieval |
| rag, routing, decomposition, trends, chunking | not requested | unavailable | none from this session |

No requested stage failed or was skipped. Logs contain the five finished run IDs,
final archive checksum and both completion lines, with no `STAGE FAILED` marker.
Warnings include unauthenticated Hugging Face downloads, model loading and notebook
conversion syntax warnings; these do not negate the recorded completions.

## Configuration and reproducibility

Expanded Pydantic configurations and configuration hashes match YAML files read
from the source commit, including default parameters. Seed 13, pinned model revisions,
selection settings and recorded package versions are preserved in the run records.
GPU dependencies were installed with editable extras, rather than a frozen lockfile;
package versions therefore differ from the local environment. This is not a
controlled device-only experiment.

`scripts/check_gpu_results.py` checks session identity, all five records, source
configurations, artifact hashes, finished timestamps, classification scores, retrieval
summary means and comparison differences. CSV rounding is allowed up to 0.00000051
for retrieval metric reconstruction. All archived raw rankings reproduce every
stored per-query retrieval metric against checksum-verified local relevance data.
The raw-ranking check requires the locally retained archive and processed data;
a fresh checkout can still verify tracked records, summary means and classifications.

## Dataset evidence and limits

All four retrieval source revisions, dataset sizes and test-query identities match
local records. All twelve corpus/query/qrels Parquet hashes differ from the local
run records. A different serialization environment is a plausible explanation,
but the original GPU Parquet files were not archived, so byte-level or complete
semantic corpus equality cannot be established retrospectively. Recomputing raw
rankings against local qrels establishes agreement for the scored relevance
judgments. Shared BM25, MiniLM and bge-small nDCG@10, MRR@10 and Recall@10 match
exactly per query on all four datasets. DAPFAM has differences at deeper cutoffs
(BM25 maximum absolute metric difference 0.35; bge-small nDCG@100 difference
0.000126), so complete cross-device ranking identity is not claimed.

The judge run's `datasets` list is empty. Its loader uses a mutable SciFact `latest`
archive and records no archive checksum. Dev pair IDs and labels and shared judge
metrics can be compared, but source text identity cannot be verified from this run.
The GPU dev pair IDs and labels match both the local source pairs and local score
file exactly. Shared judge probability differences have maxima 0.00001 (xsmall)
and 0.00488 (base), while the reported classification metrics match. These checks
do not establish original GPU input text equality. The judge results are retained
with this explicit provenance qualification.

## Findings and statistical scope

The dev-selected dense systems improve over BM25 on all four retrieval benchmarks
with within-benchmark Holm-adjusted p < 0.003. Convex fusion improves over its selected
dense encoder on SciFact and DAPFAM, while the SCIDOCS difference is inconclusive.
SPECTER performs below BM25 in all four descriptive comparisons.

The dev-selected reranker lowers test nDCG@10 on SCIDOCS and DAPFAM (Holm p 0.0176
and 0.0007). SciFact is inconclusive (p 1.0). LitSearch improves numerically (0.5324
to 0.5611), but the adjusted test is inconclusive (p 0.1368), despite an unadjusted
paired bootstrap interval excluding zero. Do not select a different depth on test.

The large judge achieves balanced accuracy 0.8199, versus 0.7551 for the base judge.
Its accuracy is 0.8324 with a pair-bootstrap 95% interval [0.7912, 0.8706]. These
intervals are not balanced-accuracy intervals, are not claim-clustered, and do not
constitute a paired significance test between judges. Judge reporting uses dev,
not an untouched test set. No inference about end-to-end GPU generation follows.

Retrieval uncertainty is conditional on these query splits, seed and model weights.
Holm correction is within each benchmark's recorded comparison family, not across
all four datasets. No multiple-seed or multiple-session uncertainty was measured.
See [full measured tables](GPU_SESSION1.md), including all reranking depths, intervals
and corrected comparisons.

## Import and publication boundary

Five new GPU result directories and one session manifest were imported. No existing
verified result was replaced or moved. Only summaries, per-item numeric scores, run
records and the generated figure are intended for Git. Benchmark text, corpus files,
model weights, raw ranking dumps, signed download URLs and the 69 MB archive remain
excluded. GitHub privacy was checked as `private: true` before publication.

## Repository verification

The complete unit-test suite passed: 129 tests, with two environment/dependency
warnings. Ruff passed. README table freshness, relative links, no-em-dash policy,
and staged whitespace checks passed. Importing the archive again reported all five
runs unchanged. The confidence-interval figure was rendered and visually inspected.
The staged-file review found no credential patterns, signed download URLs or
benchmark text columns. No staged file exceeds 10 MiB; the largest is the 3,151,638-byte
DAPFAM numeric per-query CSV. Total staged content is approximately 8.7 MB.
