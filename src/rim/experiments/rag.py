"""Grounded-generation experiment (RQ4).

Each test question is answered twice by every generator:

* ``available``: normal retrieval;
* ``withheld``: all documents judged relevant to the question are removed from the
  index before retrieval. The retrieved context then contains no gold evidence, and
  the instructed behaviour is to abstain.

Per answer the following are measured, all without a language-model judge:

* abstention;
* sentences without a citation and citation markers that match no passage;
* ALCE citation recall and precision under an entailment judge;
* whether the cited documents are ones the benchmark judges relevant.

The entailment judge is itself a model. Its agreement with human labels is measured
separately (``rim.experiments.judge_validation``) and should be read with these numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rim.config import EncoderConfig, RerankerConfig, StrictModel, data_dir
from rim.data.registry import load_dataset
from rim.data.schema import RetrievalDataset
from rim.evaluation.stats import bootstrap_ci
from rim.experiments.common import ProviderConfig, finish_run, start_run
from rim.rag.attribution import LexicalJudge, NLIJudge, score_attribution
from rim.rag.llm import build_provider
from rim.rag.pipeline import RagPipeline
from rim.rerank.rerankers import build_reranker
from rim.retrieval.encoders import build_encoder
from rim.retrieval.searcher import HybridSearcher

log = logging.getLogger(__name__)
CONDITIONS = ("available", "withheld")


class JudgeConfig(StrictModel):
    kind: str = "nli"  # nli | lexical
    model_id: str = ""
    revision: str | None = None
    threshold: float = 0.5
    max_length: int = 512


class RagExperimentConfig(StrictModel):
    experiment: str = "rag"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    dataset: str = "scifact"
    n_questions: int = 100
    question_template: str = "{query}"
    encoder: EncoderConfig
    k1: float = 1.2
    b: float = 0.75
    fusion: str = "rrf"
    rrf_k: int = 60
    alpha: float = 0.5
    reranker: RerankerConfig | None = None
    rerank_depth: int = 0
    max_passages: int = 5
    max_passage_words: int = 220
    max_tokens: int = 300
    providers: tuple[ProviderConfig, ...]
    judge: JudgeConfig = JudgeConfig()
    n_bootstrap: int = 10_000


def _mean_ci(values: list[float], cfg: RagExperimentConfig) -> dict:
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return {"mean": None, "ci": None, "n": 0}
    low, high = bootstrap_ci(arr, cfg.n_bootstrap, seed=cfg.seed)
    return {"mean": float(arr.mean()), "ci": [low, high], "n": int(arr.size)}


def run_rag_experiment(
    cfg: RagExperimentConfig,
    dataset: RetrievalDataset | None = None,
    out_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    dataset = dataset or load_dataset(cfg.dataset)
    if dataset.manifest is not None:
        record.datasets.append(dataset.manifest.model_dump(mode="json"))

    encoder = build_encoder(cfg.encoder, device)
    reranker = build_reranker(cfg.reranker, device) if cfg.reranker else None
    searcher = HybridSearcher.build(
        dataset.corpus,
        encoder,
        cache_dir or data_dir() / "cache" / dataset.name,
        cfg.k1,
        cfg.b,
        reranker=reranker,
        fusion=cfg.fusion,
        rrf_k=cfg.rrf_k,
        alpha=cfg.alpha,
    )
    record.models.append({"role": "encoder", **encoder.describe()})
    if cfg.judge.kind == "lexical":
        judge = LexicalJudge()
    else:
        judge = NLIJudge(cfg.judge.model_id, cfg.judge.revision, device, cfg.judge.max_length)
    record.models.append({"role": "judge", "threshold": cfg.judge.threshold, **judge.describe()})

    queries = dataset.split_queries("test").sort_values("query_id").head(cfg.n_questions)
    qrels = dataset.qrels_dict("test")
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, answers = [], []
    for provider_cfg in cfg.providers:
        provider = build_provider(provider_cfg.kind, **provider_cfg.options)
        record.models.append(
            {"role": "generator", "name": provider_cfg.name, **provider.describe()}
        )
        pipeline = RagPipeline(
            searcher,
            provider,
            "hybrid",
            cfg.rerank_depth,
            cfg.max_passages,
            cfg.max_passage_words,
            cfg.max_tokens,
        )
        log.info("generator %s: %d questions x %d conditions", provider_cfg.name, len(queries), 2)
        for qid, text in zip(queries["query_id"], queries["text"], strict=True):
            gold = set(qrels.get(qid, {}))
            for condition in CONDITIONS:
                question = cfg.question_template.format(query=text)
                answer = pipeline.answer(
                    question, gold if condition == "withheld" else None, cfg.seed
                )
                scores = score_attribution(
                    answer.analysis, answer.passages, judge, cfg.judge.threshold
                )
                cited = answer.cited_doc_ids
                context_docs = {p.doc_id for p in answer.passages}
                analysis = answer.analysis
                rows.append(
                    {
                        "generator": provider_cfg.name,
                        "condition": condition,
                        "query_id": qid,
                        "abstained": analysis.abstained,
                        "gold_in_context": bool(gold & context_docs),
                        "n_sentences": analysis.n_sentences,
                        "uncited_share": (
                            analysis.n_uncited / analysis.n_sentences
                            if analysis.n_sentences
                            else float("nan")
                        ),
                        "invalid_markers": analysis.n_invalid_markers,
                        "citation_recall": scores.citation_recall,
                        "citation_precision": scores.citation_precision,
                        "cited_docs_gold_share": (
                            len(set(cited) & gold) / len(set(cited)) if cited else float("nan")
                        ),
                        "prompt_tokens": answer.response.prompt_tokens,
                        "completion_tokens": answer.response.completion_tokens,
                        "generation_ms": answer.response.latency_s * 1000.0,
                        "retrieval_ms": answer.retrieval_ms,
                    }
                )
                answers.append(
                    {
                        "generator": provider_cfg.name,
                        "condition": condition,
                        "query_id": qid,
                        "supported": scores.supported,
                        **answer.to_dict(),
                    }
                )

    frame = pd.DataFrame(rows)
    results = {}
    for (generator, condition), part in frame.groupby(["generator", "condition"]):
        answered = part[~part["abstained"]]
        results.setdefault(generator, {})[condition] = {
            "n_questions": len(part),
            "abstention_rate": _mean_ci(part["abstained"].astype(float).tolist(), cfg),
            "gold_in_context_rate": float(part["gold_in_context"].mean()),
            "answered": {
                "n": len(answered),
                "citation_recall": _mean_ci(answered["citation_recall"].tolist(), cfg),
                "citation_precision": _mean_ci(answered["citation_precision"].tolist(), cfg),
                "unsupported_sentence_rate": _mean_ci(
                    (1.0 - answered["citation_recall"]).tolist(), cfg
                ),
                "uncited_sentence_share": _mean_ci(answered["uncited_share"].tolist(), cfg),
                "answers_with_invalid_marker": float((answered["invalid_markers"] > 0).mean())
                if len(answered)
                else None,
                "cited_docs_gold_share": _mean_ci(answered["cited_docs_gold_share"].tolist(), cfg),
                "mean_sentences": float(answered["n_sentences"].mean()) if len(answered) else None,
            },
            "prompt_tokens_mean": float(part["prompt_tokens"].mean()),
            "completion_tokens_mean": float(part["completion_tokens"].mean()),
            "generation_ms_median": float(part["generation_ms"].median()),
            "retrieval_ms_median": float(part["retrieval_ms"].median()),
        }

    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "dataset": dataset.name,
        "n_questions": len(queries),
        "judge": {"threshold": cfg.judge.threshold, **judge.describe()},
        "generators": results,
        "device": device,
    }
    per_answer = out_dir / "per_answer.csv"
    frame.to_csv(per_answer, index=False, float_format="%.6f")
    answers_path = out_dir / "answers.jsonl"
    answers_path.write_text("".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8")
    finish_run(record, out_dir, summary, [per_answer, answers_path])
    return summary
