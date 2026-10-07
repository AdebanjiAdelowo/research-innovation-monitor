"""Language-model backends behind one small interface.

The reported experiments use open models only: Ollama for local runs and a
transformers model on a GPU. ``ExtractiveProvider`` is a model-free baseline that
copies sentences from the context, which gives a reference point for citation
quality and keeps the test suite free of model downloads. A hosted API can be added
by implementing ``generate``.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

from rim.data.chunking import split_sentences
from rim.rag.citations import ABSTAIN
from rim.retrieval.tokenize import Tokenizer


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    model: str = ""


class LLMProvider(Protocol):
    def generate(
        self, prompt: str, system: str = "", max_tokens: int = 400, seed: int = 0
    ) -> LLMResponse: ...

    def describe(self) -> dict: ...


class ScriptedProvider:
    """Returns prepared responses in order. For tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, system="", max_tokens=400, seed=0) -> LLMResponse:
        self.prompts.append(prompt)
        text = self.responses[(len(self.prompts) - 1) % len(self.responses)]
        return LLMResponse(text, len(prompt.split()), len(text.split()), 0.0, "scripted")

    def describe(self) -> dict:
        return {"kind": "scripted"}


_PASSAGE = re.compile(r"^\[(\d+)\] (.*?)\n(.*?)(?=^\[\d+\] |\Z)", re.S | re.M)


class ExtractiveProvider:
    """Answers with the context sentences that share the most terms with the question.

    Every output sentence is copied from one passage and cites it, so its citations
    are correct by construction. It abstains when no sentence reaches ``min_overlap``.
    """

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.34) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self._tokenizer = Tokenizer()

    def generate(self, prompt, system="", max_tokens=400, seed=0) -> LLMResponse:
        start = time.perf_counter()
        question = prompt.rsplit("Question:", 1)[-1].split("\n", 1)[0]
        terms = set(self._tokenizer(question))
        scored = []
        for match in _PASSAGE.finditer(prompt.split("Question:", 1)[0]):
            index, body = int(match.group(1)), match.group(3).strip()
            for sentence in split_sentences(body):
                overlap = len(terms & set(self._tokenizer(sentence))) / max(len(terms), 1)
                scored.append((overlap, -index, sentence))
        scored.sort(reverse=True)
        chosen = [s for s in scored[: self.max_sentences] if s[0] >= self.min_overlap]
        if chosen:
            text = " ".join(
                f"{sentence.rstrip('.!?')} [{-neg_index}]." for _, neg_index, sentence in chosen
            )
        else:
            text = ABSTAIN
        return LLMResponse(
            text, len(prompt.split()), len(text.split()), time.perf_counter() - start, "extractive"
        )

    def describe(self) -> dict:
        return {
            "kind": "extractive",
            "max_sentences": self.max_sentences,
            "min_overlap": self.min_overlap,
        }


class OllamaProvider:
    def __init__(self, model: str | None = None, url: str | None = None, timeout: float = 300.0):
        self.model = model or os.environ.get("RIM_OLLAMA_MODEL", "llama3.2")
        self.url = (url or os.environ.get("RIM_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt, system="", max_tokens=400, seed=0) -> LLMResponse:
        import httpx

        start = time.perf_counter()
        response = httpx.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.0, "seed": seed, "num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            data.get("response", "").strip(),
            data.get("prompt_eval_count"),
            data.get("eval_count"),
            time.perf_counter() - start,
            self.model,
        )

    def digest(self) -> str | None:
        """Content digest of the local model, so a run names the exact weights."""
        import httpx

        try:
            tags = httpx.get(f"{self.url}/api/tags", timeout=10).json()
        except (httpx.HTTPError, ValueError):
            return None
        for model in tags.get("models", []):
            if model.get("name", "").split(":")[0] == self.model.split(":")[0]:
                return model.get("digest")
        return None

    def describe(self) -> dict:
        return {"kind": "ollama", "model": self.model, "digest": self.digest(), "temperature": 0.0}


class TransformersProvider:
    """Greedy decoding with a chat model from the Hugging Face Hub (intended for GPU runs)."""

    def __init__(self, model_id: str, revision: str | None = None, device: str = "cuda",
                 dtype: str = "float16") -> None:  # fmt: skip
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.revision, self.device, self.dtype = model_id, revision, device, dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # On CUDA the weights are spread over all visible GPUs, so a 7B model in
        # half precision fits on two 16 GB cards.
        placement = {"device_map": "auto"} if device == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype), **placement
        )
        if not placement:
            self.model.to(device)
        self.model.eval()

    def generate(self, prompt, system="", max_tokens=400, seed=0) -> LLMResponse:
        import torch

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        latency = time.perf_counter() - start
        n_prompt = inputs["input_ids"].shape[1]
        completion = output[0, n_prompt:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return LLMResponse(text, int(n_prompt), int(completion.shape[0]), latency, self.model_id)

    def describe(self) -> dict:
        return {
            "kind": "transformers",
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "decoding": "greedy",
            "device": self.device,
        }


def build_provider(kind: str, **kwargs) -> LLMProvider:
    if kind == "extractive":
        return ExtractiveProvider(**kwargs)
    if kind == "ollama":
        return OllamaProvider(**kwargs)
    if kind == "transformers":
        return TransformersProvider(**kwargs)
    raise ValueError(f"unknown provider {kind!r}")
