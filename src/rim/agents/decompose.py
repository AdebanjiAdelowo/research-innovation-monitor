"""Query decomposition for multi-aspect questions (RQ3).

A complex request is split into sub-queries, each sub-query is retrieved with the
same pipeline as the original, and the lists are fused by reciprocal rank. The
original query is always kept as one of the lists, so decomposition can add evidence
but cannot discard what the undecomposed pipeline found at the top.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from rim.data.chunking import split_sentences
from rim.rag.llm import LLMProvider
from rim.retrieval.fusion import reciprocal_rank_fusion

Retrieve = Callable[[str], list[tuple[str, float]]]


class Decomposer(Protocol):
    def decompose(self, query: str) -> list[str]: ...
    def describe(self) -> dict: ...


class SentenceDecomposer:
    """One sub-query per sentence. A model-free baseline for long, multi-sentence queries."""

    def __init__(self, max_parts: int = 6, min_words: int = 4) -> None:
        self.max_parts, self.min_words = max_parts, min_words

    def decompose(self, query: str) -> list[str]:
        parts = [s for s in split_sentences(query) if len(s.split()) >= self.min_words]
        return parts[: self.max_parts] if len(parts) > 1 else []

    def describe(self) -> dict:
        return {"kind": "sentence", "max_parts": self.max_parts}


DECOMPOSE_SYSTEM = (
    "You rewrite a complex research question as short search queries. Write one query "
    "per line, at most four lines, each covering a different aspect of the question. "
    "Each line must be a self-contained search query. Output only the queries."
)


def parse_subqueries(text: str, max_parts: int = 4) -> list[str]:
    parts = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(line.split()) >= 2 and line not in parts:
            parts.append(line)
    return parts[:max_parts]


class LLMDecomposer:
    def __init__(self, provider: LLMProvider, max_parts: int = 4) -> None:
        self.provider, self.max_parts = provider, max_parts
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_s = 0.0

    def decompose(self, query: str) -> list[str]:
        response = self.provider.generate(f"Question: {query}\nQueries:", DECOMPOSE_SYSTEM, 160, 0)
        self.prompt_tokens += response.prompt_tokens or 0
        self.completion_tokens += response.completion_tokens or 0
        self.latency_s += response.latency_s
        return parse_subqueries(response.text, self.max_parts)

    def describe(self) -> dict:
        return {"kind": "llm", "provider": self.provider.describe(), "max_parts": self.max_parts}


def decomposed_retrieve(
    query: str, subqueries: list[str], retrieve: Retrieve, depth: int = 100, rrf_k: int = 60
) -> list[tuple[str, float]]:
    """RRF over the original query's list and one list per sub-query."""
    lists = [{"q": retrieve(q)} for q in [query, *subqueries]]
    if len(lists) == 1:
        return lists[0]["q"][:depth]
    return reciprocal_rank_fusion(lists, rrf_k, depth)["q"]
