"""Paraphrase the routing test requests with a local language model.

The rule router's patterns and the task templates have the same author, so its score
on the templated test set says little about robustness to other phrasings. This
script rewrites every test request with a model the rule author does not control and
stores the result. Labels are inherited from the original request; a paraphrase that
drifts in meaning is label noise, which affects all routers alike.

Usage: python scripts/make_routing_paraphrases.py [--model llama3.2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rim.agents.tasks import build_tasks
from rim.rag.llm import OllamaProvider

SYSTEM = (
    "Rewrite the user's request in different words. Keep exactly the same meaning and "
    "ask for exactly the same kinds of information. Do not answer the request. Output "
    "only the rewritten request on one line."
)
OUT = Path(__file__).resolve().parents[1] / "src/rim/agents/data/paraphrased_test.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    provider = OllamaProvider(args.model)
    meta = {"model": args.model, "digest": provider.digest(), "seed": args.seed, "system": SYSTEM}
    lines = [json.dumps({"_meta": meta})]
    for task in build_tasks(seed=args.seed):
        if task.split != "test":
            continue
        text = provider.generate(task.request, SYSTEM, 80, args.seed).text
        text = " ".join(text.strip().strip('"').split())
        lines.append(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "original": task.request,
                    "request": text,
                    "tools": sorted(task.tools),
                }
            )
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines) - 1} paraphrases to {OUT}")


if __name__ == "__main__":
    main()
