"""Sentence-aware chunking with overlap.

Chunks are built by packing whole sentences until a word budget is reached, then
starting the next chunk with the trailing sentences of the previous one. Character
offsets into the source text are kept, so a cited chunk can be located exactly.
Chunking is deterministic: the same text and parameters give the same chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from rim.data.ids import make_chunk_id

# A sentence ends at ., ! or ? followed by whitespace and an opening character.
# Common abbreviations and decimal numbers do not match because of the lookbehinds.
_SENT_END = re.compile(
    r"(?<!\b[A-Z])(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bEq)(?<!\bvs)(?<!\bcf)"
    r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])"
)
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def of(self, text: str) -> str:
        return text[self.start : self.end]


def sentence_spans(text: str) -> list[Span]:
    """Character spans of sentences; every non-space character lies in one span."""
    spans: list[Span] = []
    start = 0
    for match in _SENT_END.finditer(text):
        end = match.end()
        segment = text[start:end]
        if segment.strip():
            lead = len(segment) - len(segment.lstrip())
            spans.append(Span(start + lead, start + len(segment.rstrip())))
        start = end
    tail = text[start:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        spans.append(Span(start + lead, start + len(tail.rstrip())))
    return spans


def split_sentences(text: str) -> list[str]:
    return [span.of(text) for span in sentence_spans(text)]


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _hard_split(text: str, span: Span, max_words: int) -> list[Span]:
    """Split an over-long sentence at word boundaries."""
    words = list(_WORD.finditer(text, span.start, span.end))
    pieces = []
    for i in range(0, len(words), max_words):
        group = words[i : i + max_words]
        pieces.append(Span(group[0].start(), group[-1].end()))
    return pieces


def chunk_spans(text: str, max_words: int = 200, overlap_words: int = 40) -> list[Span]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words")

    units: list[Span] = []
    for span in sentence_spans(text):
        if _word_count(span.of(text)) > max_words:
            units.extend(_hard_split(text, span, max_words))
        else:
            units.append(span)
    if not units:
        return []
    sizes = [_word_count(u.of(text)) for u in units]

    chunks: list[Span] = []
    start = 0
    while start < len(units):
        end = start
        words = 0
        while end < len(units) and (end == start or words + sizes[end] <= max_words):
            words += sizes[end]
            end += 1
        chunks.append(Span(units[start].start, units[end - 1].end))
        if end >= len(units):
            break
        # Step back over trailing sentences that fit in the overlap budget, but
        # always advance by at least one sentence so the loop terminates.
        next_start = end
        carried = 0
        while next_start - 1 > start and carried + sizes[next_start - 1] <= overlap_words:
            next_start -= 1
            carried += sizes[next_start]
        # Drop overlap that would leave no room for new text: the next chunk must
        # reach past the end of this one, or it would only repeat it.
        while next_start < end and sum(sizes[next_start : end + 1]) > max_words:
            next_start += 1
        start = next_start
    return chunks


def chunk_corpus(
    corpus: pd.DataFrame,
    text_column: str = "text",
    max_words: int = 200,
    overlap_words: int = 40,
) -> pd.DataFrame:
    """One row per chunk. All other corpus columns are carried over unchanged."""
    carried = [c for c in corpus.columns if c != text_column]
    rows = []
    for record in corpus.to_dict("records"):
        text = record[text_column] or ""
        spans = chunk_spans(text, max_words, overlap_words) or [Span(0, 0)]
        for index, span in enumerate(spans):
            row = {c: record[c] for c in carried}
            row.update(
                chunk_id=make_chunk_id(record["doc_id"], index),
                chunk_index=index,
                text=span.of(text),
                start_char=span.start,
                end_char=span.end,
            )
            rows.append(row)
    columns = ["chunk_id", *carried, "chunk_index", "text", "start_char", "end_char"]
    return pd.DataFrame(rows, columns=columns)
