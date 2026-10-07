"""Chunk-size ablation on full-text documents.

Documents are split into chunks of a given size, chunks are retrieved with the same
hybrid pipeline (BM25 and a dense encoder fused by reciprocal rank), and a document
is scored by its best chunk. The reference condition indexes title and abstract only.

BM25 parameters are fixed in the config and nothing is tuned, so all numbers are from
the test split. Conditions are compared with the abstract-only reference by a paired
randomisation test with Holm correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rim.config import EncoderConfig, StrictModel, data_dir
from rim.data.chunking import chunk_corpus
from rim.data.ids import doc_id_of
from rim.data.registry import load_dataset
from rim.data.schema import RetrievalDataset, index_text
from rim.evaluation.metrics import evaluate, mean_metrics
from rim.evaluation.stats import bootstrap_ci, holm, paired_randomization_test
from rim.experiments.common import finish_run, start_run
from rim.retrieval.bm25 import BM25Index
from rim.retrieval.dense import DenseIndex
from rim.retrieval.encoders import build_encoder, encode_documents_cached
from rim.retrieval.fusion import reciprocal_rank_fusion

log = logging.getLogger(__name__)


class ChunkingConfig(StrictModel):
    experiment: str = "chunking"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    dataset: str = "litsearch_fulltext"
    full_text_column: str = "full_text"
    chunk_words: tuple[int, ...] = (128, 256, 512)
    overlap_fraction: float = 0.2
    encoder: EncoderConfig
    k1: float = 1.2
    b: float = 0.75
    rrf_k: int = 60
    chunk_depth: int = 300
    depth: int = 100
    ks: tuple[int, ...] = (5, 10, 20, 100)
    primary_metric: str = "ndcg@10"
    max_docs: int | None = None
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000


def best_chunk_per_document(hits: list[tuple[str, float]], depth: int) -> list[tuple[str, float]]:
    """Collapse a ranked chunk list to documents, keeping each document's best chunk."""
    seen: dict[str, float] = {}
    for unit_id, score in hits:
        seen.setdefault(doc_id_of(unit_id), score)
        if len(seen) >= depth:
            break
    return list(seen.items())


def run_chunking_experiment(
    cfg: ChunkingConfig,
    dataset: RetrievalDataset | None = None,
    out_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    dataset = dataset or load_dataset(cfg.dataset)
    if dataset.manifest is not None:
        record.datasets.append(dataset.manifest.model_dump(mode="json"))
    corpus = dataset.corpus.head(cfg.max_docs) if cfg.max_docs else dataset.corpus
    encoder = build_encoder(cfg.encoder, device)
    record.models.append({"role": "encoder", **encoder.describe()})
    cache = Path(cache_dir) if cache_dir else data_dir() / "cache" / dataset.name

    queries = dataset.split_queries("test").sort_values("query_id")
    qids, texts = list(queries["query_id"]), list(queries["text"])
    known = set(corpus["doc_id"])
    qrels = {
        q: {d: r for d, r in rel.items() if d in known}
        for q, rel in dataset.qrels_dict("test").items()
    }
    qrels = {q: rel for q, rel in qrels.items() if rel}
    query_vectors = encoder.encode_queries(texts)

    def run_condition(unit_ids: list[str], unit_texts: list[str]) -> dict:
        bm25 = BM25Index(k1=cfg.k1, b=cfg.b).fit(unit_ids, unit_texts)
        embeddings, info = encode_documents_cached(encoder, unit_texts, cache, cfg.encoder.dtype)
        lexical = bm25.search(qids, texts, cfg.chunk_depth)
        dense = DenseIndex(unit_ids, embeddings).search(qids, query_vectors, cfg.chunk_depth)
        fused = reciprocal_rank_fusion([lexical, dense], cfg.rrf_k, cfg.chunk_depth)
        ranking = {q: best_chunk_per_document(hits, cfg.depth) for q, hits in fused.items()}
        return {
            "frame": evaluate(ranking, qrels, cfg.ks),
            "n_units": len(unit_ids),
            "embedding": info,
        }

    conditions = {"abstract": run_condition(list(corpus["doc_id"]), index_text(corpus))}
    for words in cfg.chunk_words:
        log.info("chunking at %d words", words)
        body = corpus[["doc_id", "title", cfg.full_text_column]].rename(
            columns={cfg.full_text_column: "text"}
        )
        chunks = chunk_corpus(body, "text", words, int(cfg.overlap_fraction * words))
        chunk_texts = (chunks["title"] + ". " + chunks["text"]).tolist()
        conditions[f"chunks_{words}"] = run_condition(list(chunks["chunk_id"]), chunk_texts)

    primary = cfg.primary_metric
    base = conditions["abstract"]["frame"][primary].to_numpy()
    results, comparisons = {}, []
    for name, data in conditions.items():
        values = data["frame"][primary].to_numpy()
        low, high = bootstrap_ci(values, cfg.n_bootstrap, seed=cfg.seed)
        results[name] = {
            "test": mean_metrics(data["frame"]),
            "test_ci": {primary: [low, high]},
            "n_indexed_units": data["n_units"],
            "embedding_storage_mb": data["embedding"].get("storage_mb"),
            "embedding_seconds": data["embedding"].get("encode_seconds"),
        }
        if name != "abstract":
            comparisons.append(
                {
                    "condition": name,
                    "baseline": "abstract",
                    "difference": float(values.mean() - base.mean()),
                    "p_value": paired_randomization_test(
                        values, base, cfg.n_permutations, cfg.seed
                    ),
                }
            )
    for row, adjusted in zip(comparisons, holm([c["p_value"] for c in comparisons]), strict=True):
        row["p_holm"] = adjusted

    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "dataset": dataset.name,
        "n_documents": len(corpus),
        "n_queries": len(qrels),
        "primary_metric": primary,
        "conditions": results,
        "comparisons": comparisons,
        "device": device,
    }
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = out_dir / "per_query_test.csv"
    pd.concat([d["frame"].assign(condition=n).reset_index() for n, d in conditions.items()]).to_csv(
        per_query, index=False, float_format="%.6f"
    )
    finish_run(record, out_dir, summary, [per_query])
    return summary
