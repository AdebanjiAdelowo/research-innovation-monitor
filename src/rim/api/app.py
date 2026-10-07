"""HTTP API over the retrieval, generation, trend and reporting components.

Configuration is by environment variable. With none set, the service starts on a
small synthetic corpus with a model-free encoder and an extractive generator, so it
runs in a container or in CI without downloads, weights or secrets.

    RIM_LITERATURE_DATASET, RIM_PATENT_DATASET   prepared dataset names
    RIM_GRANTS_CORPUS                            1 to serve data/processed/nsf_awards
    RIM_ENCODER                                  "hashing" or a sentence-transformers id
    RIM_ENCODER_QUERY_PREFIX                     query prefix, if the encoder needs one
    RIM_LLM                                      "extractive" or "ollama"
    RIM_ROUTER                                   "rules" or "llm"
    RIM_TRENDS_DIR                               directory written by a trend experiment
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rim import __version__
from rim.agents.orchestrator import Orchestrator, SearchTool
from rim.agents.router import LLMRouter, RuleRouter
from rim.analytics.store import TrendStore, TrendTool
from rim.config import EncoderConfig, data_dir
from rim.provenance import git_info, resolve_device
from rim.rag.llm import LLMProvider, build_provider
from rim.rag.pipeline import RagPipeline
from rim.retrieval.encoders import build_encoder
from rim.retrieval.searcher import HybridSearcher


@dataclass
class AppState:
    searchers: dict[str, HybridSearcher]
    provider: LLMProvider
    orchestrator: Orchestrator
    trends: TrendStore | None = None
    info: dict = field(default_factory=dict)


def build_state(
    corpora: dict[str, object],
    encoder,
    provider: LLMProvider,
    trends_dir: str | Path | None = None,
    router: str = "rules",
    cache_root: str | Path | None = None,
) -> AppState:
    """``corpora`` maps a tool name (literature, patents, grants) to a corpus DataFrame."""
    searchers = {
        name: HybridSearcher.build(frame, encoder, Path(cache_root) / name if cache_root else None)
        for name, frame in corpora.items()
    }
    tools: dict = {name: SearchTool(name, searcher) for name, searcher in searchers.items()}
    store = None
    if trends_dir and (Path(trends_dir) / "topic_trends.csv").exists():
        store = TrendStore(trends_dir)
        tools["trends"] = TrendTool(store)
    chosen_router = LLMRouter(provider) if router == "llm" else RuleRouter()
    info = {
        "corpora": {name: len(s.corpus) for name, s in searchers.items()},
        "encoder": encoder.describe(),
        "generator": provider.describe(),
        "router": chosen_router.describe(),
        "trends_run": store.run_id if store else None,
    }
    return AppState(searchers, provider, Orchestrator(chosen_router, tools, provider), store, info)


def state_from_env() -> AppState:
    import pandas as pd

    from rim.data.registry import load_dataset
    from rim.demo import build_demo_dataset

    env = os.environ.get
    corpora: dict[str, object] = {}
    if env("RIM_LITERATURE_DATASET"):
        corpora["literature"] = load_dataset(env("RIM_LITERATURE_DATASET")).corpus
    else:
        corpora["literature"] = build_demo_dataset().corpus
    if env("RIM_PATENT_DATASET"):
        corpora["patents"] = load_dataset(env("RIM_PATENT_DATASET")).corpus
    if env("RIM_GRANTS_CORPUS") == "1":
        corpora["grants"] = pd.read_parquet(data_dir() / "processed/nsf_awards/corpus.parquet")

    model = env("RIM_ENCODER", "hashing")
    if model == "hashing":
        encoder_cfg = EncoderConfig(name="hashing", kind="hashing", dim=512)
    else:
        encoder_cfg = EncoderConfig(
            name=model.split("/")[-1],
            model_id=model,
            query_prefix=env("RIM_ENCODER_QUERY_PREFIX", ""),
        )
    encoder = build_encoder(encoder_cfg, resolve_device())
    provider = build_provider(env("RIM_LLM", "extractive"))
    return build_state(
        corpora,
        encoder,
        provider,
        env("RIM_TRENDS_DIR"),
        env("RIM_ROUTER", "rules"),
        data_dir() / "cache" / "api",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    corpus: str = "literature"
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    corpus: str = "literature"
    max_passages: int = Field(default=5, ge=1, le=10)


class ReportRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)


def create_app(state: AppState | None = None) -> FastAPI:
    app = FastAPI(
        title="Research and Innovation Monitor",
        version=__version__,
        description="Hybrid retrieval, citation-grounded answers and topic trends.",
    )
    app.state.rim = state

    def get_state() -> AppState:
        if app.state.rim is None:
            app.state.rim = state_from_env()
        return app.state.rim

    def searcher_for(corpus: str) -> HybridSearcher:
        searchers = get_state().searchers
        if corpus not in searchers:
            raise HTTPException(404, f"unknown corpus {corpus!r}; available: {sorted(searchers)}")
        return searchers[corpus]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "git": git_info(), **get_state().info}

    @app.post("/search")
    def search(body: SearchRequest) -> dict:
        response = searcher_for(body.corpus).search(body.query, body.top_k, body.mode)
        return {
            "query": body.query,
            "corpus": body.corpus,
            "mode": body.mode,
            "hits": [asdict(hit) for hit in response.hits],
            "timings_ms": response.timings_ms,
        }

    @app.post("/ask")
    def ask(body: AskRequest) -> dict:
        pipeline = RagPipeline(
            searcher_for(body.corpus), get_state().provider, max_passages=body.max_passages
        )
        return pipeline.answer(body.question).to_dict()

    @app.get("/trends")
    def trends(direction: Literal["rising", "falling"] = "rising", limit: int = 10) -> dict:
        store = get_state().trends
        if store is None:
            raise HTTPException(503, "no trend experiment output is configured (RIM_TRENDS_DIR)")
        return {
            "run_id": store.run_id,
            "direction": direction,
            "topics": store.top(direction, limit),
        }

    @app.get("/trends/{topic}")
    def trend_series(topic: int) -> dict:
        store = get_state().trends
        if store is None:
            raise HTTPException(503, "no trend experiment output is configured (RIM_TRENDS_DIR)")
        try:
            return store.series(topic)
        except KeyError:
            raise HTTPException(404, f"unknown topic {topic}") from None

    @app.post("/report")
    def report(body: ReportRequest) -> dict:
        return get_state().orchestrator.run(body.request).to_dict()

    return app


app = create_app()
