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

    env = sub.add_parser("env", help="print the provenance of this environment")
    env.set_defaults(func=_env)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
