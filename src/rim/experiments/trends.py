"""Trend experiment (RQ5, and RQ6 when a second corpus is configured).

1. Embed the corpus and scan the number of topics for seed stability and silhouette.
   The number of topics used downstream is fixed in the config; the scan and the
   backtest at other values are reported as sensitivity, not used for selection.
2. Describe: fit topics on all years, label them, test each share series for trend.
3. Backtest: for each cut-off year fit topics on earlier documents only, forecast
   growth, and score against what happened (``rim.analytics.backtest``).
4. Optionally assign a second corpus to the same topics and test for lead or lag.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.metrics import adjusted_rand_score

from rim.analytics.backtest import rolling_backtest
from rim.analytics.leadlag import lead_lag_test
from rim.analytics.stability import scan_k, subsample_stability
from rim.analytics.topics import fit_topics, topic_terms
from rim.analytics.trends import shares, topic_year_counts, trend_table
from rim.config import EncoderConfig, StrictModel, data_dir
from rim.data.schema import index_text
from rim.evaluation.stats import holm, paired_randomization_test
from rim.experiments.common import finish_run, start_run
from rim.provenance import DatasetManifest
from rim.retrieval.encoders import build_encoder, encode_documents_cached
from rim.viz import plot_backtest, plot_topic_trends

log = logging.getLogger(__name__)

# Sentence appended to NSF abstracts from 2018 on. Left in place it would be a
# time-dependent feature unrelated to the research content.
_NSF_BOILERPLATE = re.compile(
    r"This award reflects NSF'?s statutory mission and has been deemed worthy of support "
    r"through evaluation using the Foundation'?s intellectual merit and broader impacts "
    r"review criteria\.?",
    re.I,
)
_TAGS = re.compile(r"<[^>]{1,20}>|&lt;br\s*/?&gt;")


def clean_text(text: str) -> str:
    return " ".join(_TAGS.sub(" ", _NSF_BOILERPLATE.sub(" ", text or "")).split())


class CorpusConfig(StrictModel):
    name: str
    first_year: int
    last_year: int
    min_words: int = 40
    # Optional column filter, for example {"directorate": ["CSE", "MPS", "ENG"]}.
    where: dict[str, list[str]] = Field(default_factory=dict)


class TrendExperimentConfig(StrictModel):
    experiment: str = "trends"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    corpus: CorpusConfig
    secondary: CorpusConfig | None = None
    encoder: EncoderConfig
    k: int = 60
    k_sensitivity: tuple[int, ...] = (30, 120)
    k_scan: tuple[int, ...] = (20, 40, 60, 80, 120)
    scan_sample: int = 30_000
    pca_dim: int = 50
    cutoffs: tuple[int, ...]
    horizons: tuple[int, ...] = (1, 2)
    window: int = 5
    max_lag: int = 3
    n_permutations: int = 10_000


def load_corpus(cfg: CorpusConfig, root: Path) -> tuple[pd.DataFrame, DatasetManifest | None]:
    directory = root / "processed" / cfg.name
    frame = pd.read_parquet(directory / "corpus.parquet")
    frame = frame[frame["year"].between(cfg.first_year, cfg.last_year)]
    for column, allowed in cfg.where.items():
        frame = frame[frame[column].isin(allowed)]
    frame = frame.assign(text=frame["text"].map(clean_text))
    frame = frame[frame["text"].str.split().str.len() >= cfg.min_words]
    frame = frame.reset_index(drop=True)
    frame["year"] = frame["year"].astype(int)
    manifest_path = directory / "manifest.json"
    return frame, DatasetManifest.read(manifest_path) if manifest_path.exists() else None


def run_trend_experiment(
    cfg: TrendExperimentConfig,
    corpus: pd.DataFrame | None = None,
    secondary: pd.DataFrame | None = None,
    out_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    root = data_dir()
    if corpus is None:
        corpus, manifest = load_corpus(cfg.corpus, root)
        if manifest:
            record.datasets.append(manifest.model_dump(mode="json"))
    if secondary is None and cfg.secondary is not None:
        secondary, manifest = load_corpus(cfg.secondary, root)
        if manifest:
            record.datasets.append(manifest.model_dump(mode="json"))
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else root / "cache" / cfg.corpus.name

    encoder = build_encoder(cfg.encoder, device)
    texts = index_text(corpus)
    embeddings, info = encode_documents_cached(encoder, texts, cache, cfg.encoder.dtype)
    embeddings = embeddings.astype(np.float32)
    record.models.append({"role": "encoder", **encoder.describe(), "corpus_embedding": info})
    years = corpus["year"].to_numpy()
    span = (cfg.corpus.first_year, cfg.corpus.last_year)
    log.info("%d documents, %d-%d", len(corpus), *span)
    fit = {"pca_dim": cfg.pca_dim}
    artifacts: list[Path] = []

    # ------------------------------------------------------- number of topics
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(len(corpus), min(cfg.scan_sample, len(corpus)), replace=False)
    scan = pd.DataFrame(scan_k(embeddings[sample], cfg.k_scan, seed=cfg.seed, **fit))
    scan.to_csv(out_dir / "k_scan.csv", index=False, float_format="%.4f")
    artifacts.append(out_dir / "k_scan.csv")
    stability = subsample_stability(embeddings[sample], cfg.k, seed=cfg.seed, **fit)

    # ------------------------------------------------- description, all years
    model, labels = fit_topics(embeddings, cfg.k, cfg.seed, **fit)
    terms = topic_terms(texts, labels, cfg.k)
    counts, axis = topic_year_counts(labels, years, cfg.k, span)
    table = trend_table(counts, axis, terms)
    table.to_csv(out_dir / "topic_trends.csv", index=False, float_format="%.6g")
    pd.DataFrame(counts, columns=axis).rename_axis("topic").to_csv(
        out_dir / "topic_year_counts.csv"
    )
    artifacts += [out_dir / "topic_trends.csv", out_dir / "topic_year_counts.csv"]
    artifacts.append(plot_topic_trends(shares(counts), axis, table, out_dir / "topic_trends.png"))

    # ---------------------------------------------------------------- backtest
    backtests, consistency = {}, []
    all_scores = []
    for k in (cfg.k, *cfg.k_sensitivity):
        previous: dict = {}

        def fit_assign(train_mask, _k=k, _previous=previous):
            fitted, _ = fit_topics(embeddings[train_mask], _k, cfg.seed, **fit)
            assigned = fitted.assign(embeddings)
            if "labels" in _previous and _k == cfg.k:
                shared = _previous["mask"]
                consistency.append(
                    float(adjusted_rand_score(_previous["labels"][shared], assigned[shared]))
                )
            _previous.update(labels=assigned, mask=train_mask)
            return assigned, _k

        scores, forecasts = rolling_backtest(
            fit_assign, years, list(cfg.cutoffs), cfg.horizons, span[0], cfg.window
        )
        all_scores.append(scores.assign(k=k))
        mean = scores.groupby(["predictor", "horizon"])[["spearman", "precision_top10pct", "mae"]]
        backtests[str(k)] = [
            {"predictor": p, "horizon": int(h), "n_cutoffs": len(g), **g.mean().to_dict()}
            for (p, h), g in mean
        ]
        if k == cfg.k:
            forecasts.to_csv(out_dir / "forecasts.csv", index=False, float_format="%.6f")
            artifacts.append(out_dir / "forecasts.csv")
            artifacts.append(plot_backtest(scores, out_dir / "backtest.png"))
            comparisons = _compare_errors(forecasts, cfg)
    scores_path = out_dir / "backtest_scores.csv"
    pd.concat(all_scores).to_csv(scores_path, index=False, float_format="%.6f")
    artifacts.append(scores_path)

    # ---------------------------------------------------------------- lead-lag
    lead_lag = None
    if secondary is not None:
        other_texts = index_text(secondary)
        other_cache = root / "cache" / cfg.secondary.name
        other_emb, _ = encode_documents_cached(encoder, other_texts, other_cache, cfg.encoder.dtype)
        other_labels = model.assign(other_emb.astype(np.float32))
        common = (max(span[0], cfg.secondary.first_year), min(span[1], cfg.secondary.last_year))
        a, _ = topic_year_counts(labels, years, cfg.k, common)
        b, _ = topic_year_counts(other_labels, secondary["year"].to_numpy(), cfg.k, common)
        result = lead_lag_test(shares(a), shares(b), cfg.max_lag, 2000, cfg.seed)
        result.to_csv(out_dir / "lead_lag.csv", index=False, float_format="%.5f")
        artifacts.append(out_dir / "lead_lag.csv")
        lead_lag = {
            "a": cfg.corpus.name,
            "b": cfg.secondary.name,
            "years": list(common),
            "convention": "positive lag: changes in a precede changes in b",
            "by_lag": result.to_dict("records"),
        }

    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "corpus": cfg.corpus.name,
        "n_documents": len(corpus),
        "years": list(span),
        "documents_per_year": {
            int(y): int(n) for y, n in zip(axis, counts.sum(axis=0), strict=True)
        },
        "k": cfg.k,
        "k_scan": scan.to_dict("records"),
        "subsample_stability_at_k": stability,
        "trend_counts": table["trend"].value_counts().to_dict(),
        "top_rising": table.sort_values("mk_z")
        .tail(5)[::-1][["topic", "terms", "share_first", "share_last", "q_value"]]
        .to_dict("records"),
        "top_falling": table.sort_values("mk_z")
        .head(5)[["topic", "terms", "share_first", "share_last", "q_value"]]
        .to_dict("records"),
        "backtest": backtests,
        "backtest_error_comparisons": comparisons,
        "consecutive_cutoff_ari": consistency,
        "lead_lag": lead_lag,
        "device": device,
    }
    finish_run(record, out_dir, summary, artifacts)
    return summary


def _compare_errors(forecasts: pd.DataFrame, cfg: TrendExperimentConfig) -> list[dict]:
    """Paired tests on absolute forecast error per (cut-off, topic), Holm-corrected."""
    forecasts = forecasts.assign(error=(forecasts["predicted"] - forecasts["actual"]).abs())
    wide = forecasts.pivot_table(
        index=["cutoff", "horizon", "topic"], columns="predictor", values="error"
    ).reset_index()
    pairs = [
        ("linear_trend", "persistence"),
        ("linear_trend", "last_growth"),
        ("ridge", "persistence"),
        ("ridge", "last_growth"),
        ("ridge", "linear_trend"),
    ]
    rows = []
    for horizon in sorted(wide["horizon"].unique()):
        part = wide[wide["horizon"] == horizon]
        for a, b in pairs:
            if a not in part or b not in part:
                continue
            both = part.dropna(subset=[a, b])
            if both.empty:
                continue
            rows.append(
                {
                    "horizon": int(horizon),
                    "predictor": a,
                    "baseline": b,
                    "n": len(both),
                    "mae_difference": float(both[a].mean() - both[b].mean()),
                    "p_value": paired_randomization_test(
                        both[a].to_numpy(), both[b].to_numpy(), cfg.n_permutations, cfg.seed
                    ),
                }
            )
    for row, adjusted in zip(rows, holm([r["p_value"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
    return rows
