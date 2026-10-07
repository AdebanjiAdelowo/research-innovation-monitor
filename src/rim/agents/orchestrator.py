"""Orchestration: route a request, run the selected tools, write a cited report.

Responsibilities are separated on purpose:

* the router chooses tools;
* tools retrieve documents or compute statistics in Python;
* the language model only writes prose over numbered evidence.

Numbers in a report (shares, slopes, p-values) are computed by the trend tool and
passed to the output verbatim. The model is never asked to calculate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from rim.agents.router import Router
from rim.rag.citations import ABSTAIN, Passage, analyze_answer
from rim.rag.llm import LLMProvider
from rim.rag.pipeline import SYSTEM_PROMPT, build_prompt, truncate_words


@dataclass
class ToolResult:
    tool: str
    passages: list[dict] = field(default_factory=list)  # doc_id, title, text, score, metadata
    table: list[dict] = field(default_factory=list)  # quantitative rows, reported verbatim
    error: str | None = None
    latency_ms: float = 0.0


class Tool(Protocol):
    name: str

    def run(self, request: str) -> ToolResult: ...


class SearchTool:
    """Document search over one corpus."""

    def __init__(self, name: str, searcher, top_k: int = 4, rerank_depth: int = 0) -> None:
        self.name, self.searcher, self.top_k, self.rerank_depth = (
            name,
            searcher,
            top_k,
            rerank_depth,
        )

    def run(self, request: str) -> ToolResult:
        response = self.searcher.search(request, self.top_k, "hybrid", self.rerank_depth)
        passages = [
            {
                "doc_id": h.doc_id,
                "title": h.title,
                "text": h.text,
                "score": h.score,
                "metadata": h.metadata,
            }
            for h in response.hits
        ]
        return ToolResult(self.name, passages=passages)


@dataclass
class Report:
    request: str
    tools_called: list[str]
    results: list[ToolResult]
    passages: list[Passage]
    text: str
    sentences: list[dict]
    trend_rows: list[dict]
    abstained: bool
    timings_ms: dict[str, float]

    @property
    def completed(self) -> bool:
        """Every tool returned without error and the text is either cited or an abstention."""
        tools_ok = all(r.error is None for r in self.results)
        cited = any(s["citations"] for s in self.sentences)
        return tools_ok and (cited or self.abstained or bool(self.trend_rows))

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "tools_called": self.tools_called,
            "report": self.text,
            "abstained": self.abstained,
            "sentences": self.sentences,
            "evidence": [
                {"index": p.index, "doc_id": p.doc_id, "title": p.title, "text": p.text}
                for p in self.passages
            ],
            "trends": self.trend_rows,
            "errors": {r.tool: r.error for r in self.results if r.error},
            "timings_ms": self.timings_ms,
            "completed": self.completed,
        }


class Orchestrator:
    def __init__(
        self,
        router: Router,
        tools: dict[str, Tool],
        provider: LLMProvider,
        max_passage_words: int = 160,
        max_tokens: int = 350,
    ) -> None:
        self.router, self.tools, self.provider = router, tools, provider
        self.max_passage_words, self.max_tokens = max_passage_words, max_tokens

    def run(self, request: str, seed: int = 0) -> Report:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        selected = sorted(t for t in self.router.route(request) if t in self.tools)
        timings["routing_ms"] = (time.perf_counter() - start) * 1000.0

        results = []
        for name in selected:
            start = time.perf_counter()
            try:
                result = self.tools[name].run(request)
            except Exception as exc:  # a failing tool must not take the report down
                result = ToolResult(name, error=f"{type(exc).__name__}: {exc}")
            result.latency_ms = (time.perf_counter() - start) * 1000.0
            timings[f"tool_{name}_ms"] = result.latency_ms
            results.append(result)

        passages: list[Passage] = []
        for result in results:
            for item in result.passages:
                passages.append(
                    Passage(
                        len(passages) + 1,
                        item["doc_id"],
                        f"({result.tool}) {item['title']}",
                        truncate_words(item["text"], self.max_passage_words),
                        item["score"],
                    )
                )
        trend_rows = [row for r in results for row in r.table]

        if passages:
            start = time.perf_counter()
            response = self.provider.generate(
                build_prompt(request, passages), SYSTEM_PROMPT, self.max_tokens, seed
            )
            timings["generation_ms"] = (time.perf_counter() - start) * 1000.0
            text = response.text
        else:
            text = ABSTAIN if not trend_rows else ""
        analysis = analyze_answer(text, len(passages)) if text else None
        sentences = [
            {
                "text": s.text,
                "citations": s.citations,
                "cited_doc_ids": [passages[i - 1].doc_id for i in s.citations],
                "invalid_markers": s.invalid,
            }
            for s in (analysis.sentences if analysis else [])
        ]
        timings["total_ms"] = sum(timings.values())
        return Report(
            request,
            selected,
            results,
            passages,
            text,
            sentences,
            trend_rows,
            bool(analysis and analysis.abstained),
            timings,
        )
