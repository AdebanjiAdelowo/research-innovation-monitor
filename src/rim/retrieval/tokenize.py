"""Tokenisation for lexical retrieval: lower case, alphanumeric tokens, stopwords, stemming."""

from __future__ import annotations

import re

import snowballstemmer

_TOKEN = re.compile(r"[a-z0-9]+")

# The Lucene/Elasticsearch English stopword list.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)


class Tokenizer:
    def __init__(self, stem: bool = True, remove_stopwords: bool = True) -> None:
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        self._stemmer = snowballstemmer.stemmer("english") if stem else None
        self._cache: dict[str, str] = {}

    def _stem(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is None:
            cached = self._stemmer.stemWord(token)
            self._cache[token] = cached
        return cached

    def __call__(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self._stemmer is not None:
            tokens = [self._stem(t) for t in tokens]
        return tokens

    def describe(self) -> dict:
        return {
            "stem": self.stem,
            "remove_stopwords": self.remove_stopwords,
            "pattern": _TOKEN.pattern,
        }
