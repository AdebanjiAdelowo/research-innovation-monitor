"""Deterministic dev/test assignment of queries.

Hyperparameters (BM25 parameters, fusion weights, the hybrid encoder, rerank depth)
are selected on dev queries only. Test queries are used once, for reporting.
The assignment depends only on the query ID and a salt, so it is identical on every
machine and does not change when queries are added or reordered.
"""

from __future__ import annotations

import hashlib

DEFAULT_SALT = "rim-split-v1"


def unit_hash(key: str, salt: str = DEFAULT_SALT) -> float:
    """Map a string to a reproducible number in [0, 1)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hash_split(query_id: str, dev_fraction: float, salt: str = DEFAULT_SALT) -> str:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    return "dev" if unit_hash(query_id, salt) < dev_fraction else "test"
