"""Experiment configuration.

Configs are YAML files validated into pydantic models. Unknown keys are rejected so
that a typo in a config cannot silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    name: str
    # Text fields of the corpus that are concatenated into the indexed text.
    fields: tuple[str, ...] = ("title", "text")
    # Fraction of queries assigned to the dev split when the dataset has no native split.
    dev_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    # Cap on queries per split, for smoke runs. Queries are taken in ID order.
    max_queries: int | None = None
    # Optional qrels column (for example the DAPFAM domain label) by which test
    # metrics are also reported separately.
    qrels_group: str | None = None


class BM25Config(StrictModel):
    k1: float = 1.2
    b: float = 0.75
    stem: bool = True
    remove_stopwords: bool = True
    # Grids searched on the dev split only. Empty grids keep k1 and b as given.
    k1_grid: tuple[float, ...] = ()
    b_grid: tuple[float, ...] = ()


class EncoderConfig(StrictModel):
    """Everything that changes the vectors an encoder produces."""

    name: str
    kind: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    model_id: str = ""
    revision: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    normalize: bool = True
    max_seq_length: int = 256
    batch_size: int = 64
    dtype: Literal["float32", "float16"] = "float32"
    # Only used by the hashing encoder.
    dim: int = 256

    @model_validator(mode="after")
    def _check(self) -> EncoderConfig:
        if self.kind == "sentence-transformers" and not self.model_id:
            raise ValueError("sentence-transformers encoders need a model_id")
        return self


class FusionConfig(StrictModel):
    rrf_k: int = 60
    rrf_k_grid: tuple[int, ...] = ()
    # Weight of the dense score in the convex combination, searched on dev only.
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    normalization: Literal["minmax", "zscore"] = "minmax"
    # Depth of each candidate list before fusion.
    depth: int = 100


class RerankerConfig(StrictModel):
    name: str
    kind: Literal["cross-encoder", "lexical"] = "cross-encoder"
    model_id: str = ""
    revision: str | None = None
    max_length: int = 512
    batch_size: int = 32
    depths: tuple[int, ...] = (10, 20, 50, 100)
    # Dev queries used to select the rerank depth (first N in ID order). Bounds the
    # cost of tuning; test queries are always reranked in full.
    max_dev_queries: int | None = None


class EvalConfig(StrictModel):
    ks: tuple[int, ...] = (1, 5, 10, 20, 100)
    primary_metric: str = "ndcg@10"
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000
    confidence: float = 0.95
    # Number of test queries on which per-stage latency is measured, one query at a time.
    n_latency_queries: int = 50


class RetrievalExperimentConfig(StrictModel):
    experiment: str
    seed: int = 13
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    output_dir: str = "results"
    dataset: DatasetConfig
    bm25: BM25Config = BM25Config()
    encoders: tuple[EncoderConfig, ...] = ()
    # Encoder used for the hybrid and reranking stages. Chosen on dev when null.
    hybrid_encoder: str | None = None
    fusion: FusionConfig = FusionConfig()
    rerankers: tuple[RerankerConfig, ...] = ()
    evaluation: EvalConfig = EvalConfig()
    top_k: int = 100

    @model_validator(mode="after")
    def _check(self) -> RetrievalExperimentConfig:
        names = [e.name for e in self.encoders]
        if len(set(names)) != len(names):
            raise ValueError("encoder names must be unique")
        if self.hybrid_encoder is not None and self.hybrid_encoder not in names:
            raise ValueError(f"hybrid_encoder {self.hybrid_encoder!r} is not a configured encoder")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_config[T: BaseModel](path: str | Path, model: type[T]) -> T:
    return model.model_validate(load_yaml(path))


def config_hash(config: BaseModel | dict[str, Any]) -> str:
    """Stable hash of a config, independent of key order."""
    payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def data_dir() -> Path:
    return Path(os.environ.get("RIM_DATA_DIR", "data"))
