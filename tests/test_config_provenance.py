import json

import pytest
from pydantic import ValidationError

from rim.config import RetrievalExperimentConfig, config_hash, load_config
from rim.provenance import (
    DatasetManifest,
    RunRecord,
    environment_info,
    git_info,
    sha256_file,
    utc_now,
)

MINIMAL = """
experiment: demo
dataset: {name: toy}
encoders:
  - {name: hash, kind: hashing, dim: 64}
hybrid_encoder: hash
"""


def test_load_config_applies_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL)
    cfg = load_config(path, RetrievalExperimentConfig)
    assert cfg.seed == 13
    assert cfg.bm25.k1 == 1.2
    assert cfg.encoders[0].dim == 64


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL + "typo_key: 1\n")
    with pytest.raises(ValidationError):
        load_config(path, RetrievalExperimentConfig)


def test_hybrid_encoder_must_exist(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL.replace("hybrid_encoder: hash", "hybrid_encoder: other"))
    with pytest.raises(ValidationError):
        load_config(path, RetrievalExperimentConfig)


def test_sentence_transformer_encoder_needs_model_id(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("experiment: x\ndataset: {name: toy}\nencoders: [{name: a}]\n")
    with pytest.raises(ValidationError):
        load_config(path, RetrievalExperimentConfig)


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_git_info_falls_back_to_declared_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("RIM_GIT_COMMIT", "abc1234")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    info = git_info(tmp_path)
    assert info == {"commit": "abc1234", "dirty": None, "source": "declared"}


def test_run_record_round_trip(tmp_path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}")
    record = RunRecord(experiment="demo", config={"a": 1}, config_sha256="f" * 64, seed=7)
    record.add_artifact(artifact)
    record.write(tmp_path / "run.json")
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["seed"] == 7
    assert data["run_id"].startswith("demo-")
    assert data["artifacts"]["summary.json"] == sha256_file(artifact)
    assert data["finished_at"] is not None
    assert "python" in data["environment"]


def test_environment_info_reports_packages():
    info = environment_info("cpu")
    assert info["packages"]["numpy"] is not None
    assert info["hardware"]["device"] == "cpu"


def test_manifest_round_trip(tmp_path):
    manifest = DatasetManifest(
        name="toy",
        source="unit",
        source_url="https://example.org",
        license="CC0",
        redistributable=True,
        retrieved_at=utc_now(),
        n_documents=3,
    )
    manifest.write(tmp_path / "m.json")
    assert DatasetManifest.read(tmp_path / "m.json") == manifest
