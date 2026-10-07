"""Retrieve, build a numbered context, generate, and resolve the citations."""

from __future__ import annotations

from dataclasses import dataclass

from rim.rag.citations import ABSTAIN, AnswerAnalysis, Passage, analyze_answer, cited_documents
from rim.rag.llm import LLMProvider, LLMResponse
from rim.retrieval.searcher import Hit, HybridSearcher

SYSTEM_PROMPT = (
    "You answer questions about research literature using only the numbered passages "
    "provided. Rules:\n"
    "1. Every sentence must end with the numbers of the passages that support it, in "
    "square brackets, for example [2] or [1][3].\n"
    "2. Use only information stated in the passages. Do not use outside knowledge.\n"
    f"3. If the passages do not contain the answer, reply with exactly: {ABSTAIN}\n"
    "4. Be concise: at most four sentences."
)


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def build_context(hits: list[Hit], max_passages: int = 5, max_words: int = 220) -> list[Passage]:
    """Number the top hits. Text is truncated per passage so the prompt size is bounded."""
    return [
        Passage(i, hit.doc_id, hit.title, truncate_words(hit.text, max_words), hit.score)
        for i, hit in enumerate(hits[:max_passages], start=1)
    ]


def build_prompt(question: str, passages: list[Passage]) -> str:
    context = "\n\n".join(p.render() for p in passages)
    return f"Passages:\n\n{context}\n\nQuestion: {question}\nAnswer:"


@dataclass
class RagAnswer:
    question: str
    passages: list[Passage]
    response: LLMResponse
    analysis: AnswerAnalysis
    retrieval_ms: float

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def cited_doc_ids(self) -> list[str]:
        return cited_documents(self.analysis, self.passages)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.analysis.abstained,
            "pure_abstention": self.analysis.pure_abstention,
            "sentences": [
                {
                    "text": s.text,
                    "citations": s.citations,
                    "cited_doc_ids": [self.passages[i - 1].doc_id for i in s.citations],
                    "invalid_markers": s.invalid,
                }
                for s in self.analysis.sentences
            ],
            "passages": [
                {"index": p.index, "doc_id": p.doc_id, "title": p.title, "text": p.text}
                for p in self.passages
            ],
            "usage": {
                "model": self.response.model,
                "prompt_tokens": self.response.prompt_tokens,
                "completion_tokens": self.response.completion_tokens,
                "generation_ms": self.response.latency_s * 1000.0,
                "retrieval_ms": self.retrieval_ms,
            },
        }


class RagPipeline:
    def __init__(
        self,
        searcher: HybridSearcher,
        provider: LLMProvider,
        mode: str = "hybrid",
        rerank_depth: int = 0,
        max_passages: int = 5,
        max_passage_words: int = 220,
        max_tokens: int = 300,
    ) -> None:
        self.searcher, self.provider = searcher, provider
        self.mode, self.rerank_depth = mode, rerank_depth
        self.max_passages, self.max_passage_words = max_passages, max_passage_words
        self.max_tokens = max_tokens

    def answer(self, question: str, exclude: set[str] | None = None, seed: int = 0) -> RagAnswer:
        found = self.searcher.search(
            question, self.max_passages, self.mode, self.rerank_depth, exclude
        )
        passages = build_context(found.hits, self.max_passages, self.max_passage_words)
        if not passages:
            response = LLMResponse(ABSTAIN, 0, 0, 0.0, "none")
        else:
            response = self.provider.generate(
                build_prompt(question, passages), SYSTEM_PROMPT, self.max_tokens, seed
            )
        analysis = analyze_answer(response.text, len(passages))
        return RagAnswer(question, passages, response, analysis, found.timings_ms["total_ms"])
