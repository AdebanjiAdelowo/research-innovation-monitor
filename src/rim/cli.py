"""Command line entry point: ``rim <command>``."""

from __future__ import annotations

import argparse
import json


def _prepare(args: argparse.Namespace) -> None:
    from rim.data.registry import prepare_dataset
    from rim.provenance import DatasetManifest

    for name in args.names:
        path = prepare_dataset(name, force=args.force)
        manifest = DatasetManifest.read(path / "manifest.json")
        print(f"{name}: {manifest.n_documents} documents, {manifest.n_queries} queries -> {path}")


def _env(_: argparse.Namespace) -> None:
    from rim.provenance import environment_info, git_info

    print(json.dumps({"git": git_info(), "environment": environment_info()}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download and convert datasets")
    prepare.add_argument("names", nargs="+")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=_prepare)

    env = sub.add_parser("env", help="print the provenance of this environment")
    env.set_defaults(func=_env)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
