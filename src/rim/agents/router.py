"""Routers: which tools does a request need?

Routing is multi-label classification over the tool set. Three routers share one
interface so they can be compared on the same tasks:

* ``RuleRouter``: keyword patterns. No model, no training, fully inspectable.
* ``EmbeddingRouter``: one logistic regression per tool on request embeddings,
  trained on dev tasks.
* ``LLMRouter``: a language model asked to return the tool names as JSON.

The rule router is the baseline a language model has to beat to justify its latency
and non-determinism.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from rim.agents.tasks import TOOLS, RoutingTask
from rim.rag.llm import LLMProvider
from rim.retrieval.encoders import Encoder


class Router(Protocol):
    def route(self, request: str) -> frozenset[str]: ...
    def describe(self) -> dict: ...


RULES: dict[str, str] = {
    "patents": r"\b(patent\w*|prior art|invention\w*|intellectual property|claims?|ip filings?)\b",
    "grants": r"\b(grants?|awards?|fund(?:ed|ing|s)?|nsf|proposals?)\b",
    "trends": (
        r"\b(trend\w*|grow\w*|declin\w*|emerg\w*|over time|over the years|trajectory|"
        r"increas\w*|decreas\w*|rising|falling|per year|yearly|decade)\b"
    ),
    "literature": (
        r"\b(papers?|publications?|literature|articles?|stud(?:y|ies)|research(?:ers)?|"
        r"academic|scientific|published)\b"
    ),
}


class RuleRouter:
    def __init__(self, rules: dict[str, str] | None = None, default: str = "literature") -> None:
        self.rules = {tool: re.compile(p, re.I) for tool, p in (rules or RULES).items()}
        self.default = default

    def route(self, request: str) -> frozenset[str]:
        tools = {tool for tool, pattern in self.rules.items() if pattern.search(request)}
        # "funded research" and "research awards" name grants, not literature.
        if "grants" in tools and "literature" in tools:
            stripped = re.sub(
                r"\bresearch (awards?|projects?|proposals?)\b", "", request, flags=re.I
            )
            stripped = re.sub(r"\bfunded research\b", "", stripped, flags=re.I)
            if not self.rules["literature"].search(stripped):
                tools.discard("literature")
        return frozenset(tools or {self.default})

    def describe(self) -> dict:
        return {"kind": "rules", "default": self.default}


class EmbeddingRouter:
    def __init__(self, encoder: Encoder, threshold: float = 0.5, c: float = 10.0) -> None:
        self.encoder, self.threshold, self.c = encoder, threshold, c
        self.models: dict[str, LogisticRegression] = {}

    def fit(self, tasks: list[RoutingTask]) -> EmbeddingRouter:
        x = self.encoder.encode_queries([t.request for t in tasks])
        for tool in TOOLS:
            y = np.array([tool in t.tools for t in tasks], dtype=int)
            self.models[tool] = LogisticRegression(C=self.c, max_iter=1000).fit(x, y)
        return self

    def route(self, request: str) -> frozenset[str]:
        x = self.encoder.encode_queries([request])
        probs = {tool: float(m.predict_proba(x)[0, 1]) for tool, m in self.models.items()}
        chosen = {tool for tool, p in probs.items() if p >= self.threshold}
        return frozenset(chosen or {max(probs, key=probs.get)})

    def describe(self) -> dict:
        return {
            "kind": "embedding",
            "threshold": self.threshold,
            "encoder": self.encoder.describe(),
        }


ROUTER_SYSTEM = (
    "You select tools for a research-monitoring assistant. Tools:\n"
    "- literature: search scientific papers\n"
    "- patents: search patents and prior art\n"
    "- grants: search funded research awards\n"
    "- trends: quantitative growth or decline of a topic over the years\n"
    "Reply with only a JSON list of the tool names the request needs, for example "
    '["literature", "trends"]. Select every tool the request needs and no others.'
)


def parse_tool_list(text: str) -> frozenset[str] | None:
    """Tool names from a model reply. None when the reply contains no valid tool list."""
    match = re.search(r"\[[^\[\]]*\]", text)
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = re.findall(r"[a-z]+", match.group(0).lower())
        tools = {str(i).strip().lower() for i in items if isinstance(i, str)}
        valid = tools & set(TOOLS)
        if valid:
            return frozenset(valid)
    return None


class LLMRouter:
    def __init__(self, provider: LLMProvider, default: str = "literature") -> None:
        self.provider, self.default = provider, default
        self.n_unparseable = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_s = 0.0

    def route(self, request: str) -> frozenset[str]:
        response = self.provider.generate(f"Request: {request}\nTools:", ROUTER_SYSTEM, 40, 0)
        self.prompt_tokens += response.prompt_tokens or 0
        self.completion_tokens += response.completion_tokens or 0
        self.latency_s += response.latency_s
        tools = parse_tool_list(response.text)
        if tools is None:
            self.n_unparseable += 1
            return frozenset({self.default})
        return tools

    def describe(self) -> dict:
        return {"kind": "llm", "provider": self.provider.describe(), "default": self.default}


def routing_metrics(predicted: list[frozenset[str]], expected: list[frozenset[str]]) -> dict:
    """Exact-set accuracy, micro precision/recall/F1 over tools, and per-request error counts."""
    tp = sum(len(p & e) for p, e in zip(predicted, expected, strict=True))
    fp = sum(len(p - e) for p, e in zip(predicted, expected, strict=True))
    fn = sum(len(e - p) for p, e in zip(predicted, expected, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n = len(expected)
    return {
        "n": n,
        "exact_match": sum(p == e for p, e in zip(predicted, expected, strict=True)) / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "unnecessary_calls_per_request": fp / n,
        "missed_tools_per_request": fn / n,
        "all_needed_tools_called": sum(e <= p for p, e in zip(predicted, expected, strict=True))
        / n,
    }
