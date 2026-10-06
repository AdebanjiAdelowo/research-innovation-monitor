"""Text encoders for dense retrieval, with an on-disk embedding cache.

``HashingEncoder`` is a deterministic, model-free encoder used in tests and CI.
``SentenceTransformerEncoder`` wraps a bi-encoder and records every setting that
changes the vectors it produces (revision, prefixes, normalisation, truncation).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from rim.config import EncoderConfig
from rim.retrieval.tokenize import Tokenizer


class Encoder(Protocol):
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def describe(self) -> dict: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class HashingEncoder:
    """Signed feature hashing of stemmed tokens, L2-normalised.

    It captures lexical overlap only. Its purpose is to exercise the dense pipeline
    without model weights, not to be a competitive retriever.
    """

    def __init__(self, dim: int = 256, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize
        self._tokenizer = Tokenizer()

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenizer(text):
                digest = hashlib.md5(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                out[row, index] += 1.0 if digest[4] & 1 else -1.0
        return _l2_normalize(out) if self.normalize else out

    encode_queries = _encode
    encode_documents = _encode

    def describe(self) -> dict:
        return {"kind": "hashing", "dim": self.dim, "normalize": self.normalize}


class SentenceTransformerEncoder:
    def __init__(self, config: EncoderConfig, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = device
        self.model = SentenceTransformer(config.model_id, revision=config.revision, device=device)
        self.model.max_seq_length = config.max_seq_length
        self.dim = int(self.model.encode(["dimension probe"]).shape[1])
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        # The revision actually loaded, so a run without a pinned revision is still traceable.
        self.resolved_revision = config.revision or _resolved_revision(config.model_id)

    def _encode(self, texts: list[str], prefix: str, batch_size: int | None = None) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, self.config.query_prefix, batch_size)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.config.doc_prefix)

    def describe(self) -> dict:
        return {
            "kind": "sentence-transformers",
            "model_id": self.config.model_id,
            "revision": self.resolved_revision,
            "dim": self.dim,
            "n_parameters": self.n_parameters,
            "normalize": self.config.normalize,
            "max_seq_length": self.config.max_seq_length,
            "query_prefix": self.config.query_prefix,
            "doc_prefix": self.config.doc_prefix,
            "batch_size": self.config.batch_size,
            "storage_dtype": self.config.dtype,
            "device": self.device,
        }


def _resolved_revision(model_id: str) -> str | None:
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                return revisions[0].commit_hash if revisions else None
    except Exception:  # the cache layout is not something a run should fail on
        return None
    return None


def build_encoder(config: EncoderConfig, device: str = "cpu") -> Encoder:
    if config.kind == "hashing":
        return HashingEncoder(config.dim, config.normalize)
    return SentenceTransformerEncoder(config, device)


def texts_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def encode_documents_cached(
    encoder: Encoder, texts: list[str], cache_dir: str | Path | None, dtype: str = "float32"
) -> tuple[np.ndarray, dict]:
    """Encode documents, reusing a cached matrix when encoder settings and texts match.

    The device and batch size are excluded from the cache key because they do not
    change which vectors are intended, only how they are computed.
    """
    spec = {k: v for k, v in encoder.describe().items() if k not in ("device", "batch_size")}
    key = hashlib.sha256(
        (json.dumps(spec, sort_keys=True) + texts_fingerprint(texts)).encode()
    ).hexdigest()[:24]
    path = Path(cache_dir) / f"emb-{key}.npy" if cache_dir else None
    if path is not None and path.exists():
        matrix = np.load(path)
        info = {"cache_hit": True, "encode_seconds": None}
    else:
        start = time.perf_counter()
        matrix = encoder.encode_documents(texts).astype(dtype)
        info = {"cache_hit": False, "encode_seconds": time.perf_counter() - start}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
    info.update(
        cache_key=key,
        n_vectors=int(matrix.shape[0]),
        dim=int(matrix.shape[1]),
        storage_mb=round(matrix.nbytes / 1e6, 2),
    )
    return matrix, info
