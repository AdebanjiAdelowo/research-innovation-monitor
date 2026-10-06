"""Does the cited evidence support what the answer says?

Metrics follow ALCE (Gao et al., 2023), computed per answer over its sentences:

* citation recall: share of sentences entailed by the concatenation of the passages
  they cite. A sentence without citations counts as unsupported.
* citation precision: share of citations that are needed. A citation is unneeded when
  its passage alone does not entail the sentence and the remaining cited passages
  still do. Citations of unsupported sentences count as incorrect.
* unsupported-sentence rate: 1 - citation recall.

Entailment is decided by an ``EntailmentJudge``. The judge is a model with its own
error rate, so these are estimates conditional on the judge, which is evaluated
separately against human labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rim.rag.citations import AnswerAnalysis, Passage
from rim.retrieval.tokenize import Tokenizer


class EntailmentJudge(Protocol):
    def entailment(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Probability in [0, 1] that each premise entails the paired hypothesis."""

    def describe(self) -> dict: ...


class LexicalJudge:
    """Share of the hypothesis's content terms found in the premise. For tests only:
    term containment is not entailment, and negation is invisible to it."""

    def __init__(self) -> None:
        self._tokenizer = Tokenizer()

    def entailment(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            terms = set(self._tokenizer(hypothesis))
            found = terms & set(self._tokenizer(premise))
            out.append(len(found) / len(terms) if terms else 0.0)
        return np.asarray(out, dtype=np.float32)

    def describe(self) -> dict:
        return {"kind": "lexical"}


class NLIJudge:
    """Sequence-pair classifier with an entailment class (NLI or fact-checking model)."""

    def __init__(self, model_id: str, revision: str | None = None, device: str = "cpu",
                 max_length: int = 512, batch_size: int = 16) -> None:  # fmt: skip
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id, self.revision, self.device = model_id, revision, device
        self.max_length, self.batch_size = max_length, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision
        ).to(device)
        self.model.eval()
        self._torch = torch
        labels = {int(i): str(name).lower() for i, name in self.model.config.id2label.items()}
        matches = [i for i, name in labels.items() if name.startswith("entail")]
        if len(matches) != 1:
            raise ValueError(f"cannot identify the entailment class among {labels}")
        self.entailment_index = matches[0]

    def entailment(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for start in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**batch).logits, dim=-1)
            out.append(probs[:, self.entailment_index].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "kind": "nli",
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length": self.max_length,
            "entailment_index": self.entailment_index,
            "device": self.device,
        }


@dataclass
class AttributionScores:
    n_sentences: int
    n_citations: int
    supported: list[bool]  # per sentence
    needed: list[list[bool]]  # per sentence, per citation
    citation_recall: float
    citation_precision: float

    @property
    def unsupported_rate(self) -> float:
        return 1.0 - self.citation_recall


def score_attribution(
    analysis: AnswerAnalysis,
    passages: list[Passage],
    judge: EntailmentJudge,
    threshold: float = 0.5,
) -> AttributionScores:
    """ALCE citation recall and precision for one answer. NaN when there is nothing to score."""
    text_of = {p.index: p.render() for p in passages}

    def entails(indices: list[int], hypothesis: str) -> bool:
        if not indices:
            return False
        premise = "\n\n".join(text_of[i] for i in indices)
        return bool(judge.entailment([premise], [hypothesis])[0] >= threshold)

    supported, needed = [], []
    for sentence in analysis.sentences:
        cites = sentence.citations
        ok = entails(cites, sentence.text)
        supported.append(ok)
        flags = []
        for cite in cites:
            if not ok:
                flags.append(False)
            elif len(cites) == 1 or entails([cite], sentence.text):
                flags.append(True)
            else:
                # Not sufficient alone: it is needed only if the others fail without it.
                flags.append(not entails([c for c in cites if c != cite], sentence.text))
        needed.append(flags)

    n_citations = sum(len(f) for f in needed)
    recall = float(np.mean(supported)) if supported else float("nan")
    precision = sum(sum(f) for f in needed) / n_citations if n_citations else float("nan")
    return AttributionScores(len(supported), n_citations, supported, needed, recall, precision)
