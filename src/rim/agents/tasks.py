"""Labelled routing tasks.

Each task is a request and the set of tools needed to answer it. The set is
generated from hand-written templates and a list of technical topics. Dev and test
tasks use disjoint templates and disjoint topics, so a router cannot score well by
memorising a phrasing or a topic.

Limitation: the requests are written by the author of the tools. The benchmark
measures whether a router separates the intents the tools were designed for, not how
it behaves on requests from real users.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

TOOLS = ("literature", "patents", "grants", "trends")

# label set -> (dev templates, test templates)
TEMPLATES: dict[tuple[str, ...], tuple[list[str], list[str]]] = {
    ("literature",): (
        [
            "Which papers propose methods for {t}?",
            "Find recent publications on {t}.",
            "Summarise what the research literature says about {t}.",
        ],
        [
            "What have researchers published about {t}?",
            "I need the key academic studies on {t}.",
            "Point me to articles that evaluate {t}.",
        ],
    ),
    ("patents",): (
        [
            "Is there prior art for an invention using {t}?",
            "Find patents that claim {t}.",
            "Which patent families cover {t}?",
        ],
        [
            "Search granted patents and applications related to {t}.",
            "Has anyone filed intellectual property on {t}?",
            "List inventions disclosed in patent documents about {t}.",
        ],
    ),
    ("grants",): (
        [
            "Which NSF awards fund work on {t}?",
            "Find funded research projects on {t}.",
            "Who received grants for {t}?",
        ],
        [
            "What public funding has gone to {t}?",
            "Show research awards supporting {t}.",
            "Which funded proposals address {t}?",
        ],
    ),
    ("trends",): (
        [
            "Is research activity on {t} growing or declining?",
            "How has the share of work on {t} changed over the years?",
            "Is {t} an emerging topic?",
        ],
        [
            "Show the yearly trajectory of {t}.",
            "Has interest in {t} increased over time?",
            "Quantify the growth rate of {t} over the last decade.",
        ],
    ),
    ("literature", "patents"): (
        [
            "Compare the academic papers and the patents on {t}.",
            "Has the research on {t} been turned into patented inventions?",
            "Find both publications and prior art about {t}.",
        ],
        [
            "What do scientific articles and patent filings each say about {t}?",
            "Trace {t} from published studies to patent documents.",
            "Give me the papers and the intellectual property on {t}.",
        ],
    ),
    ("literature", "trends"): (
        [
            "Find key papers on {t} and tell me whether the topic is growing.",
            "What are the main publications on {t}, and how has activity changed over time?",
            "Summarise the literature on {t} and quantify its yearly trajectory.",
        ],
        [
            "Is {t} an emerging area, and which articles should I read first?",
            "Show the growth of {t} over the years together with representative studies.",
            "Which academic works define {t}, and is interest in it increasing?",
        ],
    ),
    ("grants", "trends"): (
        [
            "Which awards fund {t}, and is the topic growing?",
            "Find funded projects on {t} and show how activity has changed over the years.",
            "Is {t} an emerging topic, and who received grants for it?",
        ],
        [
            "Quantify the trajectory of {t} over time and list the public funding behind it.",
            "Has interest in {t} increased, and which research awards support it?",
            "Show the yearly growth of {t} next to the funded proposals in that area.",
        ],
    ),
    ("literature", "patents", "grants", "trends"): (
        [
            "Give me a full landscape of {t}: papers, patents, funding and how it is trending.",
            "Prepare a report on {t} covering publications, prior art, grants and growth over time.",
            "For {t}, find the research, the patents and the awards, and say whether it is emerging.",
        ],
        [
            "I need a complete overview of {t}: academic studies, patent filings, public funding and yearly trajectory.",
            "Monitor {t} across articles, intellectual property and funded projects, and quantify its growth.",
            "Brief me on {t} from every angle: scientific work, inventions, research awards and whether interest is increasing.",
        ],
    ),
}

DEV_TOPICS = [
    "physics-informed neural networks",
    "graph neural networks for molecules",
    "federated learning",
    "solid-state batteries",
    "retrieval-augmented generation",
    "CRISPR gene editing",
    "neural operators for PDEs",
    "quantum error correction",
    "perovskite solar cells",
    "reinforcement learning for robotics",
]
TEST_TOPICS = [
    "diffusion models for image synthesis",
    "photoacoustic imaging",
    "topology optimisation",
    "carbon capture materials",
    "large language model fine-tuning",
    "data assimilation for weather forecasting",
    "lithium-sulfur batteries",
    "protein structure prediction",
    "autonomous vehicle perception",
    "sparse matrix factorisation",
]


@dataclass(frozen=True)
class RoutingTask:
    task_id: str
    split: str
    request: str
    tools: frozenset[str]


def build_tasks(per_label_set: int = 15, seed: int = 0) -> list[RoutingTask]:
    """``per_label_set`` tasks per tool combination and split, sampled without repeats."""
    rng = random.Random(seed)
    tasks = []
    for labels, (dev_templates, test_templates) in TEMPLATES.items():
        for split, templates, topics in (
            ("dev", dev_templates, DEV_TOPICS),
            ("test", test_templates, TEST_TOPICS),
        ):
            pairs = [(tpl, topic) for tpl in templates for topic in topics]
            for i, (template, topic) in enumerate(rng.sample(pairs, per_label_set)):
                tasks.append(
                    RoutingTask(
                        f"{split}-{'+'.join(labels)}-{i:02d}",
                        split,
                        template.format(t=topic),
                        frozenset(labels),
                    )
                )
    return tasks


PARAPHRASE_FILE = Path(__file__).parent / "data" / "paraphrased_test.jsonl"


def load_paraphrased_tasks(path: str | Path = PARAPHRASE_FILE) -> tuple[list[RoutingTask], dict]:
    """Model-written paraphrases of the test requests, with the generator's metadata."""
    tasks, meta = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "_meta" in row:
            meta = row["_meta"]
        else:
            tasks.append(
                RoutingTask(row["task_id"], "paraphrase", row["request"], frozenset(row["tools"]))
            )
    return tasks, meta
