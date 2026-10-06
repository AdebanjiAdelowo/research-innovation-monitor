"""Citation markers in generated text and their mapping to retrieved passages.

Answers cite passages with bracketed numbers, ``[2]`` or ``[1, 3]``, that index the
numbered context shown to the model. Parsing and validation are deterministic: a
marker that does not correspond to a passage in the context is reported as invalid,
never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rim.data.chunking import split_sentences

_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
ABSTAIN = "INSUFFICIENT EVIDENCE"


@dataclass(frozen=True)
class Passage:
    index: int  # 1-based position in the context shown to the model
    doc_id: str
    title: str
    text: str
    score: float = 0.0

    def render(self) -> str:
        return f"[{self.index}] {self.title}\n{self.text}".strip()


@dataclass
class CitedSentence:
    text: str  # the sentence without citation markers
    raw: str
    citations: list[int]  # valid passage indices, in order of appearance, without repeats
    invalid: list[int]  # cited numbers that match no passage


@dataclass
class AnswerAnalysis:
    abstained: bool
    sentences: list[CitedSentence] = field(default_factory=list)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_uncited(self) -> int:
        return sum(1 for s in self.sentences if not s.citations)

    @property
    def n_invalid_markers(self) -> int:
        return sum(len(s.invalid) for s in self.sentences)

    def cited_indices(self) -> list[int]:
        seen: dict[int, None] = {}
        for sentence in self.sentences:
            for index in sentence.citations:
                seen.setdefault(index)
        return list(seen)


def parse_markers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in _MARKER.finditer(text):
        numbers.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return numbers


def strip_markers(text: str) -> str:
    cleaned = _MARKER.sub("", text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return " ".join(cleaned.split())


def is_abstention(text: str) -> bool:
    return ABSTAIN in text.upper()


def analyze_answer(text: str, n_passages: int) -> AnswerAnalysis:
    """Split an answer into sentences and resolve each sentence's citations."""
    if is_abstention(text):
        return AnswerAnalysis(abstained=True)
    sentences = []
    for raw in _sentences_keeping_markers(text):
        valid, invalid = [], []
        for number in parse_markers(raw):
            bucket = valid if 1 <= number <= n_passages else invalid
            if number not in bucket:
                bucket.append(number)
        stripped = strip_markers(raw)
        if stripped:
            sentences.append(CitedSentence(stripped, raw, valid, invalid))
    return AnswerAnalysis(abstained=False, sentences=sentences)


def _sentences_keeping_markers(text: str) -> list[str]:
    """Sentence split where a marker written after the full stop stays with its sentence."""
    text = re.sub(r"([.!?])\s*(" + _MARKER.pattern + r")", r" \2\1", text)
    pieces = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if line:
            pieces.extend(split_sentences(line))
    return pieces


def cited_documents(analysis: AnswerAnalysis, passages: list[Passage]) -> list[str]:
    by_index = {p.index: p.doc_id for p in passages}
    return [by_index[i] for i in analysis.cited_indices()]
