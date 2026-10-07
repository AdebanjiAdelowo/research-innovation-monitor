"""Render the results section of the README from the files experiment runs wrote.

Every number in the README comes from a ``results/<experiment>/summary.json``. This
script rewrites the text between the ``results:start`` and ``results:end`` markers,
so no number is typed by hand. An experiment without a summary is listed as not run.

Usage: python scripts/make_results_tables.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"

RETRIEVAL = [
    ("retrieval_litsearch", "LitSearch"),
    ("retrieval_scifact", "SciFact"),
    ("retrieval_scidocs", "SCIDOCS"),
    ("retrieval_dapfam", "DAPFAM (patents)"),
]
GPU_RUNS = [
    ("retrieval_*_gpu", "Larger encoders (bge-base, SPECTER) and `bge-reranker-v2-m3`"),
    ("rag_scifact_gpu", "RQ4 with an open 7B generator on all 300 test claims"),
    ("judge_validation_gpu", "A larger entailment judge"),
    ("routing_gpu, decomposition_gpu", "7B model as router and decomposer"),
    ("trends_arxiv", "RQ5 on arXiv and RQ6 (grants against papers)"),
    ("chunking_litsearch", "Chunk-size ablation on full text"),
]


def load(name: str) -> dict | None:
    path = RESULTS / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_ref(name: str) -> str:
    run = json.loads((RESULTS / name / "run.json").read_text())
    commit = (run["git"].get("commit") or "unknown")[:7]
    device = run["environment"]["hardware"]["device"]
    return f"[`{run['run_id']}`](results/{name}/run.json), commit `{commit}`, device `{device}`"


def f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}f}"


def p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def short(system: str) -> str:
    return system.replace("hybrid_convex", "hybrid (convex)").replace("hybrid_rrf", "hybrid (RRF)")


def retrieval_section() -> list[str]:
    out = ["### RQ1 and RQ2: retrieval", ""]
    for name, label in RETRIEVAL:
        s = load(name)
        if s is None:
            out += [f"**{label}**: not run.", ""]
            continue
        metric = s["primary_metric"]
        sel = s["selected_on_dev"]
        best_dense = f"dense:{sel['dense']['hybrid_encoder']}"
        headline = sel.get("headline_rerank")
        hybrid = sel["fusion"]["reranker_input"]
        lat = s["latency_pipelines"]
        out += [
            f"**{label}**: {s['n_documents']:,} documents, {s['n_queries']['test']} test and "
            f"{s['n_queries']['dev']} dev queries. Run {run_ref(name)}.",
            "",
            f"| System | {metric} | 95% CI | Recall@20 | Recall@100 | MRR@10 | Median latency (ms) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for system, row in s["systems"].items():
            if "+rerank:" in system and system != headline:
                continue
            low, high = row["test_ci"][metric]
            key = system if system in lat else (hybrid if system.startswith("hybrid") else None)
            if "+rerank:" in system:
                key = f"{hybrid}+rerank:{system.split('+rerank:')[1]}"
            latency = f(lat[key]["median_ms"], 1) if key in lat else "n/a"
            t = row["test"]
            out.append(
                f"| {short(system)} | {f(t[metric])} | [{f(low)}, {f(high)}] | "
                f"{f(t['recall@20'])} | {f(t['recall@100'])} | {f(t['mrr@10'])} | {latency} |"
            )
        b = sel["bm25"]["selected"]
        out += [
            "",
            f"Selected on dev: BM25 k1={b['k1']}, b={b['b']}; hybrid encoder "
            f"`{sel['dense']['hybrid_encoder']}`; RRF k={sel['fusion']['rrf_k']}; convex weight "
            f"of the dense score {sel['fusion']['alpha']}. Latency is for the hybrid encoder's "
            "pipeline; fused systems share one latency because fusion itself takes well under a millisecond.",
            "",
            f"| Comparison ({metric}, test) | Difference | 95% CI | p (Holm) |",
            "|---|---:|---|---:|",
        ]
        for c in s["comparisons"]:
            out.append(
                f"| {short(c['system'])} vs {short(c['baseline'])} | {c['difference']:+.3f} | "
                f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {p(c['p_holm'])} |"
            )
        out.append("")
        rr = next(iter(sel["rerank"].items()), None)
        if rr:
            rr_name, _ = rr
            depths = sorted(
                (int(k.rsplit("@", 1)[1]), k) for k in s["systems"] if f"+rerank:{rr_name}@" in k
            )
            out += [
                f"Reranking the {short(hybrid)} candidates with `{rr_name}`, by depth "
                f"(no reranking: {f(s['systems'][hybrid]['test'][metric])}):",
                "",
                "| Depth | " + " | ".join(str(d) for d, _ in depths) + " |",
                "|---|" + "---:|" * len(depths),
                f"| {metric} | "
                + " | ".join(f(s["systems"][k]["test"][metric]) for _, k in depths)
                + " |",
                "| Median latency (ms) | "
                + " | ".join(
                    f(lat[f"{hybrid}+rerank:{rr_name}@{d}"]["median_ms"], 0) for d, _ in depths
                )
                + " |",
                "",
            ]
        if s.get("groups"):
            names = ["bm25", best_dense, "hybrid_rrf", "hybrid_convex"]
            out += [
                "By relevance domain (IN: query and target share an IPC class; OUT: they do not):",
                "",
                "| Domain | Queries | " + " | ".join(short(n) for n in names) + " |",
                "|---|---:|" + "---:|" * len(names),
            ]
            for group, data in s["groups"].items():
                cells = " | ".join(f(data["systems"][n][metric]) for n in names)
                out.append(f"| {group} | {data['n_queries']} | {cells} |")
            out.append("")
    return out


def decomposition_section() -> list[str]:
    s = load("decomposition")
    out = ["### RQ3: query decomposition", ""]
    if s is None:
        return [*out, "Not run.", ""]
    out += [f"Run {run_ref('decomposition')}.", ""]
    for name, data in s["datasets"].items():
        metric = data["primary_metric"]
        out += [
            f"**{name}**: {data['n_queries']} test queries, ranking within the {data['ranking_scope']}.",
            "",
            f"| Condition | {metric} | 95% CI | Sub-queries per query | Difference vs none | p (Holm) | Improved / harmed |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
        comparisons = {c["condition"]: c for c in data["comparisons"]}
        for condition, row in data["conditions"].items():
            low, high = row["test_ci"][metric]
            c = comparisons.get(condition)
            tail = (
                f"{c['difference']:+.3f} | {p(c['p_holm'])} | {c['queries_improved']} / {c['queries_harmed']}"
                if c
                else " | | "
            )
            out.append(
                f"| {condition} | {f(row['test'][metric])} | [{f(low)}, {f(high)}] | "
                f"{f(row['mean_subqueries'], 1)} | {tail} |"
            )
        out.append("")
        if data.get("groups"):
            conditions = list(data["conditions"])
            out += [
                f"By `{data['groups']['by']}` as annotated in the dataset:",
                "",
                f"| {data['groups']['by']} | Queries | " + " | ".join(conditions) + " |",
                "|---|---:|" + "---:|" * len(conditions),
            ]
            for value, g in data["groups"]["values"].items():
                out.append(
                    f"| {value} | {g['n_queries']} | "
                    + " | ".join(f(g[c]) for c in conditions)
                    + " |"
                )
            out.append("")
    return out


def rag_section() -> list[str]:
    out = ["### RQ4: grounded generation", ""]
    j = load("judge_validation")
    if j:
        out += [
            f"Entailment judges against human SciFact evidence labels ({j['n_pairs']['dev']} dev "
            f"pairs; threshold chosen on {j['n_pairs']['train']} train pairs). Run {run_ref('judge_validation')}.",
            "",
            "| Judge | Threshold | Balanced accuracy | Precision | Recall | False-positive rate | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in j["judges"].items():
            d = row["dev"]
            out.append(
                f"| {name} | {d['threshold']} | {f(d['balanced_accuracy'])} | {f(d['precision'])} | "
                f"{f(d['recall'])} | {f(d['false_positive_rate'])} | {f(row['dev_auroc'])} |"
            )
        out.append("")
    s = load("rag_scifact_local")
    if s is None:
        return [*out, "Generation experiment: not run.", ""]
    out += [
        f"Generation on {s['n_questions']} SciFact test claims, judge `{s['judge'].get('model_id', s['judge']['kind'])}` "
        f"at threshold {s['judge']['threshold']}. Run {run_ref('rag_scifact_local')}.",
        "",
        "| Generator | Evidence | Pure abstention | Abstention with claims | Answers with claims | Citation recall | Citation precision | Unsupported sentences | Uncited sentences | Cited documents judged relevant |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for generator, conditions in s["generators"].items():
        for condition in ("available", "withheld"):
            row = conditions[condition]
            a = row["answered"]

            def cell(key, _a=a):
                item = _a[key]
                if item["mean"] is None:
                    return "n/a"
                return f"{f(item['mean'])} [{f(item['ci'][0])}, {f(item['ci'][1])}]"

            out.append(
                f"| {generator} | {condition} | {f(row['pure_abstention_rate']['mean'])} | "
                f"{f(row['abstention_with_claims_rate'])} | {a['n']} | "
                f"{cell('citation_recall')} | {cell('citation_precision')} | "
                f"{cell('unsupported_sentence_rate')} | {cell('uncited_sentence_share')} | "
                f"{cell('cited_docs_gold_share')} |"
            )
    out += [
        "",
        "A pure abstention makes no claim. An abstention with claims contains the abstention phrase and at least one other sentence; "
        "those sentences are scored like any others. Citation metrics are over answers with at least one claim and are conditional on the judge above. "
        "With evidence withheld no cited document can be one the benchmark judges relevant, so that column is zero by construction.",
        "",
    ]
    first = next(iter(s["generators"].values()))["available"]
    llm = [g for g in s["generators"] if g != "extractive"]
    if llm:
        row = s["generators"][llm[0]]["available"]
        out += [
            f"`{llm[0]}`: median generation time {f(row['generation_ms_median'] / 1000, 1)} s, mean "
            f"{f(row['prompt_tokens_mean'], 0)} prompt and {f(row['completion_tokens_mean'], 0)} completion tokens "
            f"per answer; median retrieval time {f(first['retrieval_ms_median'], 0)} ms.",
            "",
        ]
    return out


def routing_section() -> list[str]:
    s = load("routing")
    out = ["### Tool routing", ""]
    if s is None:
        return [*out, "Not run.", ""]
    out += [
        f"{s['n_tasks']['test']} templated test requests and {s['n_tasks']['paraphrase']} "
        f"paraphrases of them written by `{s['paraphrase_generator'].get('model', 'a model')}`. Run {run_ref('routing')}.",
        "",
        "| Router | Exact match (templated) | Exact match (paraphrased) | 95% CI | F1 | Unnecessary calls per request | Missed tools per request | Median latency (ms) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, row in s["routers"].items():
        para = row.get("paraphrase") or {}
        ci = row.get("paraphrase_exact_match_ci") or [None, None]
        out.append(
            f"| {name} | {f(row['test']['exact_match'])} | {f(para.get('exact_match'))} | "
            f"[{f(ci[0])}, {f(ci[1])}] | {f(para.get('f1'))} | "
            f"{f(para.get('unnecessary_calls_per_request'), 2)} | "
            f"{f(para.get('missed_tools_per_request'), 2)} | {f(row['latency_median_ms'], 1)} |"
        )
    out += ["", "F1, unnecessary calls and missed tools are on the paraphrased requests.", ""]
    return out


def trends_section() -> list[str]:
    out = ["### RQ5: trend forecasting", ""]
    s = load("trends_nsf")
    if s is None:
        return [*out, "Not run.", ""]
    k = str(s["k"])
    out += [
        f"NSF awards, {s['n_documents']:,} abstracts, {s['years'][0]} to {s['years'][1]}, {k} topics. "
        f"Run {run_ref('trends_nsf')}.",
        "",
        "Topic stability (adjusted Rand index):",
        "",
        "| Topics | Seed stability (mean ARI) | Silhouette | Smallest topic (share) |",
        "|---:|---:|---:|---:|",
    ]
    for row in s["k_scan"]:
        out.append(
            f"| {row['k']} | {f(row['ari_mean'])} | {f(row['silhouette'])} | {f(row['smallest_topic_share'], 4)} |"
        )
    ari = s["consecutive_cutoff_ari"]
    out += [
        "",
        f"At {k} topics: subsample stability {f(s['subsample_stability_at_k']['ari_mean'])}; agreement "
        f"between topic models fitted at consecutive cut-off years {f(min(ari))} to {f(max(ari))}. "
        f"Full-period trend tests at 5% false discovery rate: {s['trend_counts'].get('rising', 0)} rising, "
        f"{s['trend_counts'].get('falling', 0)} falling, {s['trend_counts'].get('none', 0)} without a monotone trend.",
        "",
        "Backtest, mean over cut-off years:",
        "",
        "| Predictor | Horizon (years) | Cut-offs | Spearman | Precision, top 10% | MAE (log growth) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(s["backtest"][k], key=lambda r: (r["horizon"], r["predictor"])):
        out.append(
            f"| {row['predictor'].replace('_', ' ')} | {row['horizon']} | {row['n_cutoffs']} | "
            f"{f(row['spearman'])} | {f(row['precision_top10pct'])} | {f(row['mae'], 4)} |"
        )
    out += ["", "Paired comparison of absolute forecast errors (topic by cut-off):", "",
            "| Horizon | Predictor | Baseline | Pairs | MAE difference | p (Holm) |", "|---:|---|---|---:|---:|---:|"]  # fmt: skip
    for c in s["backtest_error_comparisons"]:
        out.append(
            f"| {c['horizon']} | {c['predictor'].replace('_', ' ')} | {c['baseline'].replace('_', ' ')} | "
            f"{c['n']} | {c['mae_difference']:+.4f} | {p(c['p_holm'])} |"
        )
    out += ["", "Sensitivity to the number of topics (Spearman, mean over cut-offs):", "",
            "| Topics | Horizon | last growth | linear trend | ridge |", "|---:|---:|---:|---:|---:|"]  # fmt: skip
    for kk, rows in s["backtest"].items():
        for h in sorted({r["horizon"] for r in rows}):
            by = {r["predictor"]: r for r in rows if r["horizon"] == h}
            out.append(
                f"| {kk} | {h} | {f(by['last_growth']['spearman'])} | {f(by['linear_trend']['spearman'])} | "
                f"{f(by.get('ridge', {}).get('spearman'))} |"
            )
    out += ["", "Strongest full-period trends:", "", "| Direction | Topic terms | Share, first year | Share, last year | q |", "|---|---|---:|---:|---:|"]  # fmt: skip
    for direction, rows in (("rising", s["top_rising"]), ("falling", s["top_falling"])):
        for row in rows[:4]:
            out.append(
                f"| {direction} | {row['terms']} | {100 * row['share_first']:.2f}% | "
                f"{100 * row['share_last']:.2f}% | {p(row['q_value'])} |"
            )
    out += ["", "![Topic trends](results/trends_nsf/topic_trends.png)", "",
            "![Backtest](results/trends_nsf/backtest.png)", ""]  # fmt: skip
    return out


def pending_section() -> list[str]:
    out = [
        "### Not yet run",
        "",
        "These need a CUDA GPU or data that is only practical to process there. Configs and the "
        "runner are in place ([kaggle/](kaggle/README.md)); no numbers are reported for them.",
        "",
        "| Experiment | What it measures | Status |",
        "|---|---|---|",
    ]
    for name, what in GPU_RUNS:
        present = any(RESULTS.glob(f"{name.split(',')[0].strip()}/summary.json"))
        out.append(f"| `{name}` | {what} | {'run' if present else 'PENDING'} |")
    return [*out, ""]


def render() -> str:
    lines: list[str] = []
    for section in (retrieval_section, decomposition_section, rag_section, routing_section,
                    trends_section, pending_section):  # fmt: skip
        lines += section()
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the README is out of date")
    args = parser.parse_args()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{START}\n\n{render()}\n{END}{tail}"
    if args.check:
        sys.exit(0 if updated == text else "README results section is out of date")
    readme.write_text(updated, encoding="utf-8")
    print("README results section updated")


if __name__ == "__main__":
    main()
