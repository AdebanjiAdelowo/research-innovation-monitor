"""Tool-routing experiment: rule, embedding and language-model routers on the same tasks.

The embedding router is fitted on dev tasks. The rule router has no parameters
fitted here, but its patterns were written by the same author as the task templates,
so its dev and test scores are both reported. All routers are scored on test tasks,
compared pairwise on per-task exact match with a paired randomisation test.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rim.agents.router import EmbeddingRouter, LLMRouter, RuleRouter, routing_metrics
from rim.agents.tasks import PARAPHRASE_FILE, build_tasks, load_paraphrased_tasks
from rim.config import EncoderConfig, StrictModel
from rim.evaluation.stats import bootstrap_ci, holm, paired_randomization_test
from rim.experiments.common import ProviderConfig, finish_run, start_run
from rim.rag.llm import build_provider
from rim.retrieval.encoders import build_encoder


class RoutingExperimentConfig(StrictModel):
    experiment: str = "routing"
    seed: int = 13
    device: str = "auto"
    output_dir: str = "results"
    per_label_set: int = 15
    # Also score routers on model-written paraphrases of the test requests, if present.
    use_paraphrases: bool = True
    encoder: EncoderConfig
    llm_routers: tuple[ProviderConfig, ...] = ()
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000


def run_routing_experiment(
    cfg: RoutingExperimentConfig, out_root: str | Path | None = None
) -> dict:
    record, device = start_run(cfg, cfg.experiment, cfg.seed, cfg.device)
    tasks = build_tasks(cfg.per_label_set, cfg.seed)
    dev = [t for t in tasks if t.split == "dev"]
    test = [t for t in tasks if t.split == "test"]

    paraphrased, paraphrase_meta = [], {}
    if cfg.use_paraphrases and PARAPHRASE_FILE.exists():
        paraphrased, paraphrase_meta = load_paraphrased_tasks()

    encoder = build_encoder(cfg.encoder, device)
    routers = {"rules": RuleRouter(), "embedding": EmbeddingRouter(encoder).fit(dev)}
    for provider_cfg in cfg.llm_routers:
        provider = build_provider(provider_cfg.kind, **provider_cfg.options)
        routers[f"llm:{provider_cfg.name}"] = LLMRouter(provider)

    results, rows, exact = {}, [], {}
    for name, router in routers.items():
        predictions, latencies = [], []
        for task in test:
            start = time.perf_counter()
            predictions.append(router.route(task.request))
            latencies.append(time.perf_counter() - start)
        expected = [t.tools for t in test]
        metrics = routing_metrics(predictions, expected)
        exact[name] = np.array([p == e for p, e in zip(predictions, expected, strict=True)], float)
        low, high = bootstrap_ci(exact[name], cfg.n_bootstrap, seed=cfg.seed)
        by_size = {}
        for size in sorted({len(e) for e in expected}):
            idx = [i for i, e in enumerate(expected) if len(e) == size]
            by_size[str(size)] = float(exact[name][idx].mean())
        results[name] = {
            "test": metrics,
            "exact_match_ci": [low, high],
            "exact_match_by_n_tools": by_size,
            "latency_median_ms": float(np.median(latencies) * 1000),
            "latency_p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        }
        if isinstance(router, LLMRouter):
            results[name]["unparseable_replies"] = router.n_unparseable
            results[name]["prompt_tokens"] = router.prompt_tokens
            results[name]["completion_tokens"] = router.completion_tokens
        if name == "rules":
            results[name]["dev"] = routing_metrics(
                [router.route(t.request) for t in dev], [t.tools for t in dev]
            )
        if paraphrased:
            para_predictions = [router.route(t.request) for t in paraphrased]
            para_expected = [t.tools for t in paraphrased]
            para_exact = np.array(
                [p == e for p, e in zip(para_predictions, para_expected, strict=True)], float
            )
            low, high = bootstrap_ci(para_exact, cfg.n_bootstrap, seed=cfg.seed)
            results[name]["paraphrase"] = routing_metrics(para_predictions, para_expected)
            results[name]["paraphrase_exact_match_ci"] = [low, high]
            rows.extend(
                {
                    "router": name,
                    "split": "paraphrase",
                    "task_id": t.task_id,
                    "request": t.request,
                    "expected": "+".join(sorted(t.tools)),
                    "predicted": "+".join(sorted(p)),
                    "exact": p == t.tools,
                }
                for t, p in zip(paraphrased, para_predictions, strict=True)
            )
        record.models.append({"role": "router", "name": name, **router.describe()})
        rows.extend(
            {
                "router": name,
                "split": "test",
                "task_id": t.task_id,
                "request": t.request,
                "expected": "+".join(sorted(t.tools)),
                "predicted": "+".join(sorted(p)),
                "exact": p == t.tools,
            }
            for t, p in zip(test, predictions, strict=True)
        )

    comparisons = []
    for a, b in itertools.combinations(routers, 2):
        comparisons.append(
            {
                "a": a,
                "b": b,
                "difference_exact_match": float(exact[a].mean() - exact[b].mean()),
                "p_value": paired_randomization_test(
                    exact[a], exact[b], cfg.n_permutations, cfg.seed
                ),
            }
        )
    for row, adjusted in zip(comparisons, holm([c["p_value"] for c in comparisons]), strict=True):
        row["p_holm"] = adjusted

    summary = {
        "experiment": cfg.experiment,
        "run_id": record.run_id,
        "n_tasks": {"dev": len(dev), "test": len(test), "paraphrase": len(paraphrased)},
        "paraphrase_generator": paraphrase_meta,
        "routers": results,
        "comparisons": comparisons,
        "device": device,
    }
    out_dir = Path(out_root or cfg.output_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    per_task = out_dir / "per_task_test.csv"
    pd.DataFrame(rows).to_csv(per_task, index=False)
    finish_run(record, out_dir, summary, [per_task])
    return summary
