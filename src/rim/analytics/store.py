"""Read-only access to the outputs of a trend experiment.

The API and the trend tool serve precomputed statistics from the files a run wrote.
Nothing is recomputed at request time and no language model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rim.retrieval.tokenize import Tokenizer

COLUMNS = [
    "topic",
    "terms",
    "trend",
    "share_first",
    "share_last",
    "slope_per_year",
    "p_value",
    "q_value",
]


class TrendStore:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.table = pd.read_csv(directory / "topic_trends.csv")
        counts = pd.read_csv(directory / "topic_year_counts.csv", index_col="topic")
        self.years = [int(c) for c in counts.columns]
        self.counts = counts.to_numpy()
        totals = self.counts.sum(axis=0)
        self.shares = self.counts / totals.clip(min=1)
        run = directory / "run.json"
        self.run_id = json.loads(run.read_text())["run_id"] if run.exists() else None
        self._tokenizer = Tokenizer()
        self._term_tokens = [
            set(self._tokenizer(str(terms)))
            for terms in self.table.get("terms", [""] * len(self.table))
        ]

    def _rows(self, frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame[[c for c in COLUMNS if c in frame]].to_json(orient="records"))

    def top(self, direction: str = "rising", limit: int = 10) -> list[dict]:
        if direction not in ("rising", "falling"):
            raise ValueError("direction must be 'rising' or 'falling'")
        part = self.table[self.table["trend"] == direction].sort_values(
            "mk_z", ascending=direction == "falling"
        )
        return self._rows(part.head(limit))

    def series(self, topic: int) -> dict:
        if not 0 <= topic < len(self.table):
            raise KeyError(topic)
        return {
            "topic": topic,
            "years": self.years,
            "counts": [int(c) for c in self.counts[topic]],
            "shares": [float(s) for s in self.shares[topic]],
            **self._rows(self.table[self.table["topic"] == topic])[0],
        }

    def match(self, query: str, limit: int = 3) -> list[dict]:
        """Topics whose descriptive terms share the most stems with the query."""
        terms = set(self._tokenizer(query))
        overlap = [len(terms & tokens) for tokens in self._term_tokens]
        order = sorted(range(len(overlap)), key=lambda i: (-overlap[i], i))
        chosen = [i for i in order[:limit] if overlap[i] > 0]
        return self._rows(self.table.iloc[chosen])


class TrendTool:
    """Orchestrator tool: trend statistics for the topics that match a request."""

    name = "trends"

    def __init__(self, store: TrendStore, limit: int = 3) -> None:
        self.store, self.limit = store, limit

    def run(self, request: str):
        from rim.agents.orchestrator import ToolResult

        return ToolResult(self.name, table=self.store.match(request, self.limit))
