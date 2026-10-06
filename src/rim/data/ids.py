"""Stable identifiers for documents and chunks.

A document ID is ``<source>:<native id>``. The native ID is the one the source
assigns (a corpus ID, a patent family ID, an award number), so IDs survive
re-ingestion. When a source has no native ID, the ID is derived from the content.
A chunk ID is ``<doc id>#c<index>``, so the parent document is always recoverable.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_CHUNK_RE = re.compile(r"^(?P<doc>.+)#c(?P<idx>\d{4,})$")


def normalize_text(text: str) -> str:
    """Canonical form used for hashing: NFKC, lower case, single spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.lower().split())


def make_doc_id(source: str, native_id: str | int) -> str:
    if not _SOURCE_RE.match(source):
        raise ValueError(f"invalid source name: {source!r}")
    native = str(native_id).strip()
    if not native or any(ch.isspace() for ch in native) or "#" in native:
        raise ValueError(f"invalid native id: {native_id!r}")
    return f"{source}:{native}"


def content_doc_id(source: str, title: str, text: str) -> str:
    """ID for sources without native identifiers. Insensitive to case and spacing."""
    blob = f"{normalize_text(title)}\n{normalize_text(text)}".encode()
    return make_doc_id(source, "h" + hashlib.sha1(blob).hexdigest()[:16])


def make_chunk_id(doc_id: str, index: int) -> str:
    if index < 0:
        raise ValueError("chunk index must be non-negative")
    return f"{doc_id}#c{index:04d}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    match = _CHUNK_RE.match(chunk_id)
    if not match:
        raise ValueError(f"not a chunk id: {chunk_id!r}")
    return match["doc"], int(match["idx"])


def doc_id_of(any_id: str) -> str:
    """Parent document of a chunk ID; a document ID maps to itself."""
    return parse_chunk_id(any_id)[0] if _CHUNK_RE.match(any_id) else any_id


def source_of(doc_id: str) -> str:
    return doc_id.split(":", 1)[0]
