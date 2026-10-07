"""Command line entry point: ``rim <command>``."""

from __future__ import annotations

import argparse
import json
import logging


def _prepare(args: argparse.Namespace) -> None:
    from rim.data.registry import prepare_dataset
    from rim.provenance import DatasetManifest

    for name in args.names:
        path = prepare_dataset(name, force=args.force)
        manifest = DatasetManifest.read(path / "manifest.json")
        print(f"{name}: {manifest.n_documents} documents, {manifest.n_queries} queries -> {path}")


def _prepare_nsf(args: argparse.Namespace) -> None:
    from rim.config import data_dir
    from rim.data.loaders import nsf

    awards, _ = nsf.prepare(data_dir(), args.first_year, args.last_year)
    print(f"nsf_awards: {len(awards)} awards")


def _retrieval(args: argparse.Namespace) -> None:
    from rim.config import RetrievalExperimentConfig, load_config
    from rim.experiments.retrieval import format_table, run_retrieval_experiment

    summary = run_retrieval_experiment(load_config(args.config, RetrievalExperimentConfig))
    print(format_table(summary))
    print(f"\nrun {summary['run_id']}")


def _routing(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.routing import RoutingExperimentConfig, run_routing_experiment

    summary = run_routing_experiment(load_config(args.config, RoutingExperimentConfig))
    for name, row in summary["routers"].items():
        m = row["test"]
        print(
            f"{name}: exact={m['exact_match']:.3f} f1={m['f1']:.3f} "
            f"unnecessary={m['unnecessary_calls_per_request']:.3f}"
        )
    print(f"\nrun {summary['run_id']}")


def _decomposition(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.decomposition import DecompositionConfig, run_decomposition_experiment

    summary = run_decomposition_experiment(load_config(args.config, DecompositionConfig))
    for name, data in summary["datasets"].items():
        metric = data["primary_metric"]
        for condition, row in data["conditions"].items():
            print(f"{name} {condition}: {metric}={row['test'][metric]:.4f}")
    print(f"\nrun {summary['run_id']}")


def _judge(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.judge_validation import JudgeValidationConfig, run_judge_validation

    summary = run_judge_validation(load_config(args.config, JudgeValidationConfig))
    for name, row in summary["judges"].items():
        d = row["dev"]
        print(
            f"{name}: threshold={d['threshold']} balanced_acc={d['balanced_accuracy']:.3f} "
            f"precision={d['precision']:.3f} recall={d['recall']:.3f} auroc={row['dev_auroc']:.3f}"
        )
    print(f"\nrun {summary['run_id']}")


def _rag(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.rag import RagExperimentConfig, run_rag_experiment

    summary = run_rag_experiment(load_config(args.config, RagExperimentConfig))
    for name, conditions in summary["generators"].items():
        for condition, row in conditions.items():
            a = row["answered"]
            print(
                f"{name} [{condition}]: pure_abstain={row['pure_abstention_rate']['mean']:.3f} "
                f"cit_recall={a['citation_recall']['mean']} "
                f"cit_precision={a['citation_precision']['mean']}"
            )
    print(f"\nrun {summary['run_id']}")


def _trends(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.trends import TrendExperimentConfig, run_trend_experiment

    summary = run_trend_experiment(load_config(args.config, TrendExperimentConfig))
    print(f"{summary['n_documents']} documents, k={summary['k']}, trends {summary['trend_counts']}")
    for row in summary["backtest"][str(summary["k"])]:
        print(
            f"  {row['predictor']} h={row['horizon']}: spearman={row['spearman']:.3f} "
            f"mae={row['mae']:.4f}"
        )
    print(f"\nrun {summary['run_id']}")


def _chunking(args: argparse.Namespace) -> None:
    from rim.config import load_config
    from rim.experiments.chunking import ChunkingConfig, run_chunking_experiment

    summary = run_chunking_experiment(load_config(args.config, ChunkingConfig))
    metric = summary["primary_metric"]
    for name, row in summary["conditions"].items():
        print(f"{name}: {metric}={row['test'][metric]:.4f} units={row['n_indexed_units']}")
    print(f"\nrun {summary['run_id']}")


def _prepare_arxiv(args: argparse.Namespace) -> None:
    from rim.config import data_dir
    from rim.data.loaders import arxiv

    papers, _ = arxiv.prepare(
        args.snapshot, data_dir(), first_year=args.first_year, last_year=args.last_year
    )
    print(f"arxiv: {len(papers)} papers")


def _env(_: argparse.Namespace) -> None:
    from rim.provenance import environment_info, git_info

    print(json.dumps({"git": git_info(), "environment": environment_info()}, indent=2))


def main(argv: list[str] | None = None) -> None:
    from rim.net import force_ipv4_if_requested

    force_ipv4_if_requested()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="rim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download and convert datasets")
    prepare.add_argument("names", nargs="+")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=_prepare)

    nsf = sub.add_parser("prepare-nsf", help="download NSF award abstracts (resumable)")
    nsf.add_argument("first_year", type=int)
    nsf.add_argument("last_year", type=int)
    nsf.set_defaults(func=_prepare_nsf)

    retrieval = sub.add_parser("retrieval", help="run a retrieval experiment (RQ1, RQ2)")
    retrieval.add_argument("config")
    retrieval.set_defaults(func=_retrieval)

    routing = sub.add_parser("routing", help="run the tool-routing experiment")
    routing.add_argument("config")
    routing.set_defaults(func=_routing)

    decomposition = sub.add_parser("decomposition", help="run the query-decomposition experiment")
    decomposition.add_argument("config")
    decomposition.set_defaults(func=_decomposition)

    judge = sub.add_parser("judge-validation", help="validate entailment judges on human labels")
    judge.add_argument("config")
    judge.set_defaults(func=_judge)

    rag = sub.add_parser("rag", help="run the grounded-generation experiment (RQ4)")
    rag.add_argument("config")
    rag.set_defaults(func=_rag)

    trends = sub.add_parser("trends", help="run the trend experiment (RQ5, RQ6)")
    trends.add_argument("config")
    trends.set_defaults(func=_trends)

    chunking = sub.add_parser("chunking", help="run the chunk-size ablation")
    chunking.add_argument("config")
    chunking.set_defaults(func=_chunking)

    arxiv = sub.add_parser("prepare-arxiv", help="filter the arXiv metadata snapshot")
    arxiv.add_argument("snapshot")
    arxiv.add_argument("--first-year", type=int, default=2010)
    arxiv.add_argument("--last-year", type=int, default=2024)
    arxiv.set_defaults(func=_prepare_arxiv)

    env = sub.add_parser("env", help="print the provenance of this environment")
    env.set_defaults(func=_env)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
