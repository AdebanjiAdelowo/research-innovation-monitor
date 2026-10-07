"""Query decomposition experiment (RQ3).

Every condition uses the identical retrieval pipeline: BM25 and one dense encoder
fused by reciprocal rank with fixed k. The only difference between conditions is the
set of query strings sent through it:

* ``none``: the original query;
* ``sentence``: the original query plus one sub-query per sentence (no model);
* ``llm:<name>``: the original query plus sub-queries written by a language model;
* ``oracle_aspects``: the original query plus the aspects written by the dataset
  authors, where the dataset provides them. This is a reference for what a good
  decomposition can achieve, not a deployable system.

Nothing is tuned, so the dev split is left untouched and all numbers are from test.
Conditions are compared with the undecomposed pipeline by a paired randomisation
test, Holm-corrected within each dataset.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rim.agents.decompose import LLMDecomposer, SentenceDecomposer, decomposed_retrieve
from rim.config import EncoderConfig, StrictModel, data_dir
from rim.data.registry import load_dataset
from rim.data.schema import RetrievalDataset, index_text
from rim.evaluation.metrics import evaluate, mean_metrics
from rim.evaluation.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_randomization_test
from rim.experiments.common import ProviderConfig, finish_run, start_run
from rim.rag.llm import build_provider
from rim.retrieval.bm25 import BM25Index
from rim.retrieval.dense import DenseIndex
from rim.retrieval.encoders import build_encoder, encode_documents_cached
from rim.retrieval.fusion import reciprocal_rank_fusion

log = logging.getLogger(__name__)


class DecompositionDataset(StrictModel):
    name: str
    k1: float = 1.2
    b: float = 0.75
    # Query column used to report results by subgroup (for example LitSearch specificity).
    group_by: str | None = None
    max_queries: int | None = None


class DecompositionConfig(StrictModel):
    experiment: str = "decomposition"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    datasets: tuple[DecompositionDataset, ...]
    encoder: EncoderConfig
    llm_decomposers: tuple[ProviderConfig, ...] = ()
    rrf_k: int = 60
    depth: int = 100
    ks: tuple[int, ...] = (5, 10, 20, 100)
    primary_metric: str = "ndcg@10"
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000


def run_decomposition_experiment(
    cfg: DecompositionConfig,
    datasets: dict[str, RetrievalDataset] | None = None,
    out_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> dict:
    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    encoder = build_encoder(cfg.encoder, device)
    record.models.append({"role": "encoder", **encoder.describe()})
    decomposers = {"sentence": SentenceDecomposer()}
    for provider_cfg in cfg.llm_decomposers:
        provider = build_provider(provider_cfg.kind, **provider_cfg.options)
        decomposers[f"llm:{provider_cfg.name}"] = LLMDecomposer(provider)
    for name, decomposer in decomposers.items():
        record.models.append({"role": "decomposer", "name": name, **decomposer.describe()})

    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = cfg.primary_metric
    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "device": device,
        "datasets": {},
    }
    per_query_frames, subquery_rows = [], []

    for ds_cfg in cfg.datasets:
        dataset = (datasets or {}).get(ds_cfg.name) or load_dataset(ds_cfg.name)
        if dataset.manifest is not None:
            record.datasets.append(dataset.manifest.model_dump(mode="json"))
        doc_ids = list(dataset.corpus["doc_id"])
        texts = index_text(dataset.corpus)
        bm25 = BM25Index(k1=ds_cfg.k1, b=ds_cfg.b).fit(doc_ids, texts)
        cache = Path(cache_root or data_dir() / "cache") / dataset.name
        embeddings, _ = encode_documents_cached(encoder, texts, cache, cfg.encoder.dtype)
        dense = DenseIndex(doc_ids, embeddings)

        queries = dataset.split_queries("test").sort_values("query_id")
        if ds_cfg.max_queries:
            queries = queries.head(ds_cfg.max_queries)
        qids = list(queries["query_id"])
        qrels = {q: r for q, r in dataset.qrels_dict("test").items() if q in set(qids)}
        has_pool = "candidate_pool" in queries.columns
        has_aspects = "aspects" in queries.columns
        # With candidate pools, every list is retrieved in full and then filtered.
        depth = len(doc_ids) if has_pool else cfg.depth

        memo: dict[str, list[tuple[str, float]]] = {}

        def retrieve(text: str, _memo=memo, _bm25=bm25, _dense=dense, _depth=depth):
            if text not in _memo:
                lexical = {"q": _bm25.search_one(text, _depth)}
                vector = encoder.encode_queries([text])
                semantic = _dense.search(["q"], vector, _depth)
                _memo[text] = reciprocal_rank_fusion([lexical, semantic], cfg.rrf_k, _depth)["q"]
            return _memo[text]

        conditions = ["none", *decomposers] + (["oracle_aspects"] if has_aspects else [])
        rankings = {c: {} for c in conditions}
        n_subqueries = {c: [] for c in conditions}
        seconds = {c: 0.0 for c in decomposers}
        log.info("%s: %d test queries, conditions %s", dataset.name, len(qids), conditions)
        for row in queries.to_dict("records"):
            qid, text = row["query_id"], row["text"]
            pool = set(row["candidate_pool"]) if has_pool else None
            for condition in conditions:
                if condition == "none":
                    subs = []
                elif condition == "oracle_aspects":
                    subs = list(row["aspects"])
                else:
                    start = time.perf_counter()
                    subs = decomposers[condition].decompose(text)
                    seconds[condition] += time.perf_counter() - start
                    subquery_rows.append(
                        {
                            "dataset": dataset.name,
                            "query_id": qid,
                            "condition": condition,
                            "subqueries": " || ".join(subs),
                        }
                    )
                ranked = decomposed_retrieve(text, subs, retrieve, depth, cfg.rrf_k)
                if pool is not None:
                    ranked = [(d, s) for d, s in ranked if d in pool]
                rankings[condition][qid] = ranked[: cfg.depth]
                n_subqueries[condition].append(len(subs))

        frames = {c: evaluate(rankings[c], qrels, cfg.ks) for c in conditions}
        base = frames["none"][primary].to_numpy()
        results, comparisons = {}, []
        for condition in conditions:
            values = frames[condition][primary].to_numpy()
            low, high = bootstrap_ci(values, cfg.n_bootstrap, seed=cfg.seed)
            results[condition] = {
                "test": mean_metrics(frames[condition]),
                "test_ci": {primary: [low, high]},
                "mean_subqueries": float(np.mean(n_subqueries[condition])),
                "share_decomposed": float(np.mean(np.array(n_subqueries[condition]) > 0)),
            }
            if condition in decomposers:
                results[condition]["decompose_seconds_per_query"] = seconds[condition] / len(qids)
            if condition != "none":
                d_low, d_high = paired_bootstrap_ci(values, base, cfg.n_bootstrap, seed=cfg.seed)
                comparisons.append(
                    {
                        "condition": condition,
                        "baseline": "none",
                        "metric": primary,
                        "difference": float(values.mean() - base.mean()),
                        "ci": [d_low, d_high],
                        "p_value": paired_randomization_test(
                            values, base, cfg.n_permutations, cfg.seed
                        ),
                        "queries_improved": int(np.sum(values > base + 1e-12)),
                        "queries_harmed": int(np.sum(values < base - 1e-12)),
                    }
                )
        for item, adjusted in zip(
            comparisons, holm([c["p_value"] for c in comparisons]), strict=True
        ):
            item["p_holm"] = adjusted

        groups = {}
        if ds_cfg.group_by:
            group_of = dict(zip(queries["query_id"], queries[ds_cfg.group_by], strict=True))
            for value in sorted(set(group_of.values())):
                members = [q for q in frames["none"].index if group_of[q] == value]
                groups[str(value)] = {
                    "n_queries": len(members),
                    **{c: float(frames[c].loc[members, primary].mean()) for c in conditions},
                }

        for decomposer in decomposers.values():
            if isinstance(decomposer, LLMDecomposer):
                results[next(n for n, d in decomposers.items() if d is decomposer)].update(
                    prompt_tokens=decomposer.prompt_tokens,
                    completion_tokens=decomposer.completion_tokens,
                )
                decomposer.prompt_tokens = decomposer.completion_tokens = 0
        summary["datasets"][dataset.name] = {
            "n_queries": len(qids),
            "n_documents": len(doc_ids),
            "ranking_scope": "candidate pool" if has_pool else "full corpus",
            "primary_metric": primary,
            "conditions": results,
            "comparisons": comparisons,
            "groups": {"by": ds_cfg.group_by, "values": groups} if groups else None,
        }
        per_query_frames.extend(
            frame.assign(condition=c, dataset=dataset.name).reset_index()
            for c, frame in frames.items()
        )

    per_query = out_dir / "per_query_test.csv"
    pd.concat(per_query_frames).to_csv(per_query, index=False, float_format="%.6f")
    subqueries = out_dir / "subqueries.csv"
    pd.DataFrame(subquery_rows).to_csv(subqueries, index=False)
    finish_run(record, out_dir, summary, [per_query, subqueries])
    return summary
