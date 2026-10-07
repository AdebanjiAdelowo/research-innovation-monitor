"""Helpers shared by the experiment runners."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from rim.config import StrictModel, config_hash
from rim.provenance import RunRecord, resolve_device, seed_everything


class ProviderConfig(StrictModel):
    """A language-model backend. ``options`` is passed to the provider constructor."""

    name: str
    kind: str  # extractive | ollama | transformers
    options: dict = Field(default_factory=dict)


def start_run(
    cfg: BaseModel, experiment: str, seed: int, device: str = "auto"
) -> tuple[RunRecord, str]:
    seed_everything(seed)
    resolved = resolve_device(device)
    record = RunRecord(
        experiment=experiment,
        config=cfg.model_dump(mode="json"),
        config_sha256=config_hash(cfg),
        seed=seed,
    )
    record.environment["hardware"]["device"] = resolved
    return record, resolved


def finish_run(record: RunRecord, out_dir: Path, summary: dict, extra_files: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", "utf-8")
    for path in [summary_path, *extra_files]:
        record.add_artifact(path)
    record.write(out_dir / "run.json")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, frozenset | set):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")
