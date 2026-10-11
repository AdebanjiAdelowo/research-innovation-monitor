"""Render session-1 tables and a confidence-interval figure from measured summaries."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("scifact", "scidocs", "litsearch", "dapfam")


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "summary.json").read_text())


def gpu_section() -> list[str]:
    lines = [
        "### GPU retrieval and judge evaluation",
        "",
        "CUDA evaluation uses the same seed and pinned model revisions as the local "
        "baselines, with additional encoders and rerankers. Test nDCG@10 and 95% "
        "query-bootstrap intervals are shown below. The headline reranker and its depth "
        "are selected on dev, rather than by the highest test score.",
        "",
        "| Benchmark | Test queries | BM25 | Selected dense | Fusion input | "
        "Headline reranker | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for benchmark in BENCHMARKS:
        s = load(f"retrieval_{benchmark}_gpu")
        selected = s["selected_on_dev"]
        systems = s["systems"]
        dense = systems[f"dense:{selected['dense']['hybrid_encoder']}"]["test"]["ndcg@10"]
        fusion = systems[selected["fusion"]["reranker_input"]]["test"]["ndcg@10"]
        headline = systems[selected["headline_rerank"]]
        low, high = headline["test_ci"]["ndcg@10"]
        lines.append(
            f"| {benchmark} | {s['n_queries']['test']} | "
            f"{systems['bm25']['test']['ndcg@10']:.4f} | {dense:.4f} | "
            f"{fusion:.4f} | {headline['test']['ndcg@10']:.4f} | [{low:.4f}, {high:.4f}] |"
        )
    lines += [
        "",
        "The dev-selected dense system improves over BM25 on all four benchmarks "
        "(Holm-adjusted p < 0.003 within each benchmark). Reranking reduces nDCG@10 "
        "on SCIDOCS and DAPFAM (adjusted p = 0.0176 and 0.0007). Its change on SciFact "
        "is inconclusive (adjusted p = 1.0); LitSearch improves numerically but does "
        "not pass the corrected significance threshold (adjusted p = 0.1368).",
        "",
        "![GPU reranking changes with paired 95% confidence intervals]"
        "(results/gpu_sessions/reranking_effects.png)",
        "",
        "| GPU judge | Dev accuracy | 95% CI | Balanced accuracy | False-positive rate |",
        "|---|---:|---|---:|---:|",
    ]
    for name, row in load("judge_validation_gpu")["judges"].items():
        low, high = row["dev_accuracy_ci"]
        lines.append(
            f"| {name} | {row['dev']['accuracy']:.4f} | [{low:.4f}, {high:.4f}] | "
            f"{row['dev']['balanced_accuracy']:.4f} | {row['dev']['false_positive_rate']:.4f} |"
        )
    lines += [
        "",
        "The large judge has higher observed balanced accuracy than the base judge "
        "(0.8199 versus 0.7551). These are dev-set results, with pair-level accuracy "
        "intervals; no paired judge significance test or held-out test-set judge "
        "evaluation was recorded. The GPU judge run lacks source dataset fingerprints. "
        "Generation scores above still use the base judge.",
        "",
        "Retrieval source revisions match the local runs, and shared baseline nDCG@10 "
        "scores match per query. Serialized Parquet fingerprints differ, so byte-identical "
        "datasets and a controlled device-only comparison are not established. "
        "[Detailed GPU results and protocol](docs/GPU_SESSION1.md).",
        "",
    ]
    return lines


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(8, 3.5))
    lines = [
        "# GPU session 1: evaluation",
        "",
        "Source commit: `2fe9d3136c99b067e71ccfd40aa7719853b4c8a5`. "
        "Two Tesla T4 GPUs; Python 3.13.15; seed 13. "
        "Judge thresholds use 919 train pairs and report 340 dev pairs. "
        "Retrieval model, fusion and headline rerank choices use dev queries only.",
        "",
        "## Retrieval systems",
        "",
        "95% intervals are query-bootstrap intervals. Comparisons use paired "
        "randomization tests with Holm correction within each benchmark. "
        "The intervals do not capture variation across training seeds or dataset versions.",
        "",
    ]
    for position, benchmark in enumerate(BENCHMARKS):
        name = f"retrieval_{benchmark}_gpu"
        s = load(name)
        lines += [
            f"### {benchmark}",
            "",
            f"Run record: [run.json](../results/{name}/run.json). "
            f"Dev/test queries: {s['n_queries']['dev']}/{s['n_queries']['test']}.",
            "",
            f"Dev-selected headline: `{s['selected_on_dev']['headline_rerank']}`. "
            f"Fusion encoder: `{s['selected_on_dev']['dense']['hybrid_encoder']}`.",
            "",
            "| System | Test nDCG@10 | 95% CI | Recall@20 | MRR@10 |",
            "|---|---:|---|---:|---:|",
        ]
        for system, row in s["systems"].items():
            low, high = row["test_ci"]["ndcg@10"]
            lines.append(
                f"| {system} | {row['test']['ndcg@10']:.4f} | "
                f"[{low:.4f}, {high:.4f}] | {row['test']['recall@20']:.4f} | "
                f"{row['test']['mrr@10']:.4f} |"
            )
        lines += [
            "",
            "| System | Baseline | nDCG@10 difference | Paired 95% CI | p (Holm) |",
            "|---|---|---:|---|---:|",
        ]
        for c in s["comparisons"]:
            low, high = c["ci"]
            lines.append(
                f"| {c['system']} | {c['baseline']} | {c['difference']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {c['p_holm']:.4f} |"
            )
        effect = s["comparisons"][-1]
        delta = effect["difference"]
        low, high = effect["ci"]
        axes.errorbar(delta, position, xerr=[[delta - low], [high - delta]], fmt="o", capsize=4)
        lines += [""]
    axes.axvline(0, color="grey", linestyle="--", linewidth=1)
    axes.set_yticks(range(4), BENCHMARKS)
    axes.set_xlabel("Headline reranker minus its fusion input: test nDCG@10")
    axes.set_title("Dev-selected reranking: paired 95% bootstrap intervals")
    fig.tight_layout()
    fig.savefig(ROOT / "results/gpu_sessions/reranking_effects.png", dpi=160)
    plt.close(fig)
    lines += [
        "## Judge evaluation",
        "",
        "Run record: [run.json](../results/judge_validation_gpu/run.json).",
        "",
        "Thresholds are selected on train; metrics below use dev pairs.",
        "",
        "| Judge | Threshold | Accuracy | Accuracy 95% CI | Balanced accuracy | AUROC |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name, row in load("judge_validation_gpu")["judges"].items():
        low, high = row["dev_accuracy_ci"]
        d = row["dev"]
        lines.append(
            f"| {name} | {row['selected_threshold']} | {d['accuracy']:.4f} | "
            f"[{low:.4f}, {high:.4f}] | {d['balanced_accuracy']:.4f} | "
            f"{row['dev_auroc']:.4f} |"
        )
    lines += [""]
    lines += [
        "## Interpretation and limits",
        "",
        "Dense retrieval beats BM25 in the pre-specified comparisons on all four datasets. "
        "Larger models do not uniformly improve retrieval: SPECTER scores below BM25 "
        "on every dataset, and bge-base is not the dev-selected encoder on SCIDOCS. "
        "These SPECTER comparisons are descriptive, without a recorded paired test.",
        "",
        "Reranking harms SCIDOCS and DAPFAM at the dev-selected headline settings. "
        "SciFact is inconclusive. LitSearch has a positive estimate whose paired interval "
        "excludes zero, but its Holm-adjusted p is 0.1368, so the corrected test is inconclusive. "
        "Selecting a different depth after seeing test scores would change the protocol.",
        "",
        "The larger judge improves observed dev accuracy, but this does not establish "
        "improved end-to-end RAG attribution. No GPU generation was requested in this session.",
        "",
        "See [scientific audit](SCIENTIFIC_AUDIT.md) for provenance limits and checks.",
        "",
        "Regenerate with `PYTHONPATH=src .venv/bin/python scripts/make_gpu_report.py`; "
        "check scores with `PYTHONPATH=src .venv/bin/python scripts/check_gpu_results.py`.",
        "",
    ]
    (ROOT / "docs/GPU_SESSION1.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
