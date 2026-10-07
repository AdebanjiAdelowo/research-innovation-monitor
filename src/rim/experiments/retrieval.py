"""Retrieval experiment (RQ1, RQ2): BM25, dense, hybrid fusion and reranking.

Protocol
--------
1. Every tunable choice is made on dev queries only: BM25 (k1, b), the encoder used
   for the hybrid, the RRF constant, the convex fusion weight, which fusion feeds the
   reranker, and the rerank depth reported as the headline.
2. Test queries are scored once, after all choices are frozen.
3. Systems are compared on test with a paired randomisation test on the primary
   metric, Holm-corrected over the pre-specified family of comparisons below.
4. Latency is measured per query, one query at a time, on the device recorded in the
   run record.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rim.config import RetrievalExperimentConfig, config_hash, data_dir
from rim.data.schema import RetrievalDataset, index_text
from rim.evaluation.metrics import evaluate, mean_metrics
from rim.evaluation.stats import (
    bootstrap_ci,
    holm,
    paired_bootstrap_ci,
    paired_randomization_test,
)
from rim.provenance import RunRecord, resolve_device, seed_everything
from rim.rerank.rerankers import apply_rerank, build_reranker, score_candidates
from rim.retrieval.bm25 import BM25Index
from rim.retrieval.dense import DenseIndex
from rim.retrieval.encoders import build_encoder, encode_documents_cached
from rim.retrieval.fusion import convex_fusion, reciprocal_rank_fusion
from rim.retrieval.tokenize import Tokenizer
from rim.retrieval.types import Ranking

log = logging.getLogger(__name__)


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _split(ranking: Ranking, query_ids: list[str]) -> Ranking:
    return {qid: ranking[qid] for qid in query_ids}


def _ms(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples) * 1000.0
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "mean_ms": float(arr.mean()),
        "n": len(samples),
    }


def run_retrieval_experiment(
    cfg: RetrievalExperimentConfig,
    dataset: RetrievalDataset | None = None,
    out_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    from rim.data.registry import load_dataset

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    record = RunRecord(
        experiment=cfg.experiment,
        config=cfg.model_dump(mode="json"),
        config_sha256=config_hash(cfg),
        seed=cfg.seed,
    )
    record.environment["hardware"]["device"] = device

    dataset = dataset or load_dataset(cfg.dataset.name)
    if dataset.manifest is not None:
        record.datasets.append(dataset.manifest.model_dump(mode="json"))
    cache_dir = Path(cache_dir) if cache_dir else data_dir() / "cache" / dataset.name
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = cfg.evaluation.primary_metric
    ks = cfg.evaluation.ks
    doc_ids = list(dataset.corpus["doc_id"])
    doc_texts = index_text(dataset.corpus, cfg.dataset.fields)
    doc_text = dict(zip(doc_ids, doc_texts, strict=True))

    queries = {}
    for split in ("dev", "test"):
        frame = dataset.split_queries(split).sort_values("query_id")
        if cfg.dataset.max_queries:
            frame = frame.head(cfg.dataset.max_queries)
        queries[split] = frame
    qids = {s: list(f["query_id"]) for s, f in queries.items()}
    all_qids = qids["dev"] + qids["test"]
    query_text = dict(
        zip(all_qids, list(queries["dev"]["text"]) + list(queries["test"]["text"]), strict=True)
    )
    all_texts = [query_text[q] for q in all_qids]
    qrels = {
        s: {q: r for q, r in dataset.qrels_dict(s).items() if q in set(qids[s])}
        for s in ("dev", "test")
    }

    def dev_score(ranking: Ranking) -> float:
        return mean_metrics(evaluate(_split(ranking, qids["dev"]), qrels["dev"], ks))[primary]

    selection: dict = {}
    systems: dict[str, Ranking] = {}

    # ------------------------------------------------------------------ BM25
    tokenizer = Tokenizer(cfg.bm25.stem, cfg.bm25.remove_stopwords)
    start = time.perf_counter()
    bm25 = BM25Index(tokenizer, cfg.bm25.k1, cfg.bm25.b).fit(doc_ids, doc_texts)
    bm25_build_seconds = time.perf_counter() - start
    grid = list(
        itertools.product(cfg.bm25.k1_grid or (cfg.bm25.k1,), cfg.bm25.b_grid or (cfg.bm25.b,))
    )
    bm25_dev = []
    for k1, b in grid:
        candidate = bm25.with_params(k1, b)
        ranking = candidate.search(qids["dev"], [query_text[q] for q in qids["dev"]], cfg.top_k)
        bm25_dev.append({"k1": k1, "b": b, primary: dev_score(ranking)})
    best = max(bm25_dev, key=lambda row: row[primary])
    bm25 = bm25.with_params(best["k1"], best["b"])
    selection["bm25"] = {"selected": best, "dev_grid": bm25_dev}
    log.info("bm25 selected k1=%s b=%s", best["k1"], best["b"])
    lexical = bm25.search(all_qids, all_texts, cfg.fusion.depth)
    systems["bm25"] = lexical
    record.models.append(
        {"role": "lexical", "build_seconds": bm25_build_seconds, **bm25.describe()}
    )

    # ----------------------------------------------------------------- dense
    dense_runs: dict[str, Ranking] = {}
    dense_dev: dict[str, float] = {}
    latency_qids = qids["test"][: cfg.evaluation.n_latency_queries]
    latency: dict[str, dict] = {}
    stage_samples: dict[str, list[float]] = {}

    samples = []
    for qid in latency_qids:
        start = time.perf_counter()
        bm25.search_one(query_text[qid], cfg.fusion.depth)
        samples.append(time.perf_counter() - start)
    stage_samples["bm25"] = samples

    for enc_cfg in cfg.encoders:
        encoder = build_encoder(enc_cfg, device)
        embeddings, info = encode_documents_cached(encoder, doc_texts, cache_dir, enc_cfg.dtype)
        start = time.perf_counter()
        query_vectors = encoder.encode_queries(all_texts)
        query_seconds = time.perf_counter() - start
        index = DenseIndex(doc_ids, embeddings)
        ranking = index.search(all_qids, query_vectors, cfg.fusion.depth)
        dense_runs[enc_cfg.name] = ranking
        dense_dev[enc_cfg.name] = dev_score(ranking)
        log.info("encoder %s: dev %s=%.4f", enc_cfg.name, primary, dense_dev[enc_cfg.name])
        systems[f"dense:{enc_cfg.name}"] = ranking

        encode_samples, search_samples = [], []
        for qid in latency_qids:
            start = time.perf_counter()
            vector = encoder.encode_queries([query_text[qid]])
            encode_samples.append(time.perf_counter() - start)
            start = time.perf_counter()
            index.search_vectors(vector, cfg.fusion.depth)
            search_samples.append(time.perf_counter() - start)
        stage_samples[f"encode:{enc_cfg.name}"] = encode_samples
        stage_samples[f"dense_search:{enc_cfg.name}"] = search_samples

        docs_per_second = (
            info["n_vectors"] / info["encode_seconds"] if info["encode_seconds"] else None
        )
        record.models.append(
            {
                "role": "encoder",
                "name": enc_cfg.name,
                **encoder.describe(),
                "corpus_embedding": info,
                "corpus_docs_per_second": docs_per_second,
                "batched_query_encode_seconds": query_seconds,
            }
        )
        del encoder, index, embeddings
        _release()

    if not dense_runs:
        raise ValueError("at least one encoder is required")
    hybrid_encoder = cfg.hybrid_encoder or max(dense_dev, key=dense_dev.get)
    selection["dense"] = {"dev": dense_dev, "hybrid_encoder": hybrid_encoder}
    dense = dense_runs[hybrid_encoder]

    # ---------------------------------------------------------------- fusion
    rrf_dev = [
        {"k": k, primary: dev_score(reciprocal_rank_fusion([lexical, dense], k, cfg.top_k))}
        for k in (cfg.fusion.rrf_k_grid or (cfg.fusion.rrf_k,))
    ]
    rrf_k = max(rrf_dev, key=lambda row: row[primary])["k"]
    systems["hybrid_rrf"] = reciprocal_rank_fusion([lexical, dense], rrf_k, cfg.top_k)

    alpha_dev = [
        {
            "alpha": alpha,
            primary: dev_score(
                convex_fusion(lexical, dense, alpha, cfg.fusion.normalization, cfg.top_k)
            ),
        }
        for alpha in cfg.fusion.alpha_grid
    ]
    alpha = max(alpha_dev, key=lambda row: row[primary])["alpha"]
    systems["hybrid_convex"] = convex_fusion(
        lexical, dense, alpha, cfg.fusion.normalization, cfg.top_k
    )
    hybrid_dev = {name: dev_score(systems[name]) for name in ("hybrid_rrf", "hybrid_convex")}
    best_hybrid = max(hybrid_dev, key=hybrid_dev.get)
    selection["fusion"] = {
        "rrf_k": rrf_k,
        "rrf_dev_grid": rrf_dev,
        "alpha": alpha,
        "alpha_dev_grid": alpha_dev,
        "dev": hybrid_dev,
        "reranker_input": best_hybrid,
    }

    samples = []
    for qid in latency_qids:
        one_lex, one_den = {qid: lexical[qid]}, {qid: dense[qid]}
        start = time.perf_counter()
        if best_hybrid == "hybrid_rrf":
            reciprocal_rank_fusion([one_lex, one_den], rrf_k, cfg.top_k)
        else:
            convex_fusion(one_lex, one_den, alpha, cfg.fusion.normalization, cfg.top_k)
        samples.append(time.perf_counter() - start)
    stage_samples["fusion"] = samples

    # ------------------------------------------------------------- reranking
    candidates = systems[best_hybrid]
    headline_rerank: str | None = None
    selection["rerank"] = {}
    for rr_cfg in cfg.rerankers:
        reranker = build_reranker(rr_cfg, device)
        max_depth = max(rr_cfg.depths)
        tune_qids = qids["dev"][: rr_cfg.max_dev_queries or None]
        scored_qids = tune_qids + qids["test"]
        log.info("reranking %d queries with %s", len(scored_qids), rr_cfg.name)
        start = time.perf_counter()
        scored = score_candidates(
            reranker, _split(candidates, scored_qids), query_text, doc_text, max_depth
        )
        rerank_seconds = time.perf_counter() - start
        tune_qrels = {q: qrels["dev"][q] for q in tune_qids if q in qrels["dev"]}
        depth_dev = {}
        for depth in rr_cfg.depths:
            name = f"{best_hybrid}+rerank:{rr_cfg.name}@{depth}"
            systems[name] = apply_rerank(_split(candidates, scored_qids), scored, depth)
            depth_dev[depth] = mean_metrics(
                evaluate(_split(systems[name], tune_qids), tune_qrels, ks)
            )[primary]
            samples = []
            for qid in latency_qids:
                top = [d for d, _ in candidates[qid][:depth]]
                start = time.perf_counter()
                reranker.score(query_text[qid], [doc_text[d] for d in top])
                samples.append(time.perf_counter() - start)
            stage_samples[f"rerank:{rr_cfg.name}@{depth}"] = samples
        best_depth = max(depth_dev, key=depth_dev.get)
        selection["rerank"][rr_cfg.name] = {
            "dev_by_depth": depth_dev,
            "selected_depth": best_depth,
            "n_dev_queries": len(tune_qids),
            "dev_without_rerank": mean_metrics(
                evaluate(_split(candidates, tune_qids), tune_qrels, ks)
            )[primary],
        }
        record.models.append(
            {
                "role": "reranker",
                "name": rr_cfg.name,
                **reranker.describe(),
                "batched_seconds_all_queries": rerank_seconds,
                "pairs_scored": sum(len(v) for v in scored.values()),
            }
        )
        del reranker
        _release()
    if selection["rerank"]:
        best_rr = max(
            selection["rerank"],
            key=lambda n: selection["rerank"][n]["dev_by_depth"][
                selection["rerank"][n]["selected_depth"]
            ],
        )
        depth = selection["rerank"][best_rr]["selected_depth"]
        headline_rerank = f"{best_hybrid}+rerank:{best_rr}@{depth}"
        selection["headline_rerank"] = headline_rerank

    # --------------------------------------------- test evaluation (one pass)
    per_query = {
        name: evaluate(_split(ranking, qids["test"]), qrels["test"], ks)
        for name, ranking in systems.items()
    }
    ecfg = cfg.evaluation
    results = {}
    for name, frame in per_query.items():
        low, high = bootstrap_ci(
            frame[primary].to_numpy(), ecfg.n_bootstrap, ecfg.confidence, cfg.seed
        )
        results[name] = {
            "test": mean_metrics(frame),
            "test_ci": {primary: [low, high]},
            "dev": {primary: dev_score(systems[name])} if "+rerank:" not in name else None,
        }

    best_dense = f"dense:{hybrid_encoder}"
    pairs = [
        (best_dense, "bm25"),
        ("hybrid_rrf", "bm25"),
        ("hybrid_rrf", best_dense),
        ("hybrid_convex", "bm25"),
        ("hybrid_convex", best_dense),
        ("hybrid_convex", "hybrid_rrf"),
    ]
    if headline_rerank:
        pairs.append((headline_rerank, best_hybrid))
    comparisons = []
    for a, b in pairs:
        xa, xb = per_query[a][primary].to_numpy(), per_query[b][primary].to_numpy()
        low, high = paired_bootstrap_ci(xa, xb, ecfg.n_bootstrap, ecfg.confidence, cfg.seed)
        comparisons.append(
            {
                "system": a,
                "baseline": b,
                "metric": primary,
                "difference": float(xa.mean() - xb.mean()),
                "ci": [low, high],
                "p_value": paired_randomization_test(xa, xb, ecfg.n_permutations, cfg.seed),
            }
        )
    for row, adjusted in zip(comparisons, holm([c["p_value"] for c in comparisons]), strict=True):
        row["p_holm"] = adjusted

    groups = {}
    if cfg.dataset.qrels_group:
        column = cfg.dataset.qrels_group
        judged = dataset.qrels[
            (dataset.qrels["relevance"] > 0) & dataset.qrels["query_id"].isin(qids["test"])
        ]
        for value, part in judged.groupby(column):
            sub: dict[str, dict[str, float]] = {}
            for qid, did, rel in zip(
                part["query_id"], part["doc_id"], part["relevance"], strict=True
            ):
                sub.setdefault(qid, {})[did] = float(rel)
            groups[str(value)] = {
                "n_queries": len(sub),
                "systems": {
                    name: mean_metrics(evaluate(_split(r, list(sub)), sub, ks))
                    for name, r in systems.items()
                },
            }

    # --------------------------------------------------------------- latency
    for stage, values in stage_samples.items():
        latency[stage] = _ms(values)

    def pipeline(*stages: str) -> dict[str, float]:
        return _ms(list(np.sum([stage_samples[s] for s in stages], axis=0)))

    enc_stage, search_stage = f"encode:{hybrid_encoder}", f"dense_search:{hybrid_encoder}"
    pipelines = {
        "bm25": pipeline("bm25"),
        best_dense: pipeline(enc_stage, search_stage),
        best_hybrid: pipeline("bm25", enc_stage, search_stage, "fusion"),
    }
    for enc_cfg in cfg.encoders:
        pipelines.setdefault(
            f"dense:{enc_cfg.name}",
            pipeline(f"encode:{enc_cfg.name}", f"dense_search:{enc_cfg.name}"),
        )
    for stage in stage_samples:
        if stage.startswith("rerank:"):
            pipelines[f"{best_hybrid}+{stage}"] = pipeline(
                "bm25", enc_stage, search_stage, "fusion", stage
            )

    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "dataset": dataset.name,
        "n_documents": len(doc_ids),
        "n_queries": {s: len(qids[s]) for s in qids},
        "primary_metric": primary,
        "selected_on_dev": selection,
        "systems": results,
        "comparisons": comparisons,
        "groups": groups,
        "latency_stages": latency,
        "latency_pipelines": pipelines,
        "device": device,
    }

    # --------------------------------------------------------------- outputs
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    long = pd.concat([frame.assign(system=name).reset_index() for name, frame in per_query.items()])
    per_query_path = out_dir / "per_query_test.csv"
    long.to_csv(per_query_path, index=False, float_format="%.6f")
    raw_dir = out_dir / "runs_raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "rankings_test.json").write_text(
        json.dumps({n: _split(r, qids["test"]) for n, r in systems.items()})
    )
    for path in (summary_path, per_query_path):
        record.add_artifact(path)
    record.write(out_dir / "run.json")
    return summary


def format_table(summary: dict, metrics: tuple[str, ...] | None = None) -> str:
    """Markdown table of test metrics, built only from a summary produced by a run."""
    primary = summary["primary_metric"]
    metrics = metrics or (primary, "recall@5", "recall@20", "recall@100", "mrr@10")
    lines = [
        "| System | " + " | ".join(metrics) + f" | 95% CI ({primary}) |",
        "|---|" + "---:|" * len(metrics) + "---|",
    ]
    for name, row in summary["systems"].items():
        low, high = row["test_ci"][primary]
        cells = " | ".join(f"{row['test'][m]:.4f}" for m in metrics)
        lines.append(f"| {name} | {cells} | [{low:.4f}, {high:.4f}] |")
    return "\n".join(lines)
