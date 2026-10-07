import math

import numpy as np
import pytest

from rim.rag.attribution import LexicalJudge, score_attribution
from rim.rag.citations import (
    ABSTAIN,
    Passage,
    analyze_answer,
    cited_documents,
    parse_markers,
    strip_markers,
)
from rim.rag.llm import ExtractiveProvider, ScriptedProvider
from rim.rag.pipeline import RagPipeline, build_context, build_prompt
from rim.retrieval.encoders import HashingEncoder
from rim.retrieval.searcher import HybridSearcher

PASSAGES = [
    Passage(1, "toy:d03", "BM25", "The probabilistic relevance framework derives term weighting."),
    Passage(2, "toy:d04", "DPR", "Dual encoders embed questions and passages."),
    Passage(3, "toy:d05", "RRF", "Reciprocal rank fusion combines ranked lists."),
]


@pytest.fixture
def searcher(toy_dataset):
    return HybridSearcher.build(toy_dataset.corpus, HashingEncoder(dim=512), depth=12)


# ------------------------------------------------------------------- citations


def test_marker_parsing_and_stripping():
    assert parse_markers("Fusion helps [1][3], see also [2, 3; 4].") == [1, 3, 2, 3, 4]
    assert parse_markers("An interval [a, b] and a year (2020).") == []
    assert strip_markers("Fusion helps [1] [3] .") == "Fusion helps."


def test_analysis_maps_sentences_to_passages_and_flags_invalid_markers():
    text = "Term weighting is derived probabilistically [1]. Fusion helps. [3] Dense models embed text [2][7]. No source here."
    analysis = analyze_answer(text, n_passages=3)
    assert [s.citations for s in analysis.sentences] == [[1], [3], [2], []]
    assert analysis.sentences[2].invalid == [7]
    assert analysis.sentences[1].text == "Fusion helps."
    assert analysis.n_uncited == 1 and analysis.n_invalid_markers == 1
    assert analysis.cited_indices() == [1, 3, 2]
    assert cited_documents(analysis, PASSAGES) == ["toy:d03", "toy:d05", "toy:d04"]


def test_bulleted_answers_and_abstention():
    analysis = analyze_answer(
        "- Dual encoders embed text [2].\n2. Fusion combines lists [3, 1].", 3
    )
    assert [s.citations for s in analysis.sentences] == [[2], [3, 1]]
    abstained = analyze_answer(f"{ABSTAIN.lower()}.", 3)
    assert abstained.abstained and abstained.pure_abstention and abstained.n_sentences == 0
    # Declining and then asserting something anyway: the claim is still scored.
    hedged = analyze_answer(
        "There is insufficient evidence for this [1]. Still, fusion helps [3].", 3
    )
    assert hedged.abstained and not hedged.pure_abstention
    assert [(s.text, s.citations) for s in hedged.sentences] == [("Still, fusion helps.", [3])]


# ----------------------------------------------------------------- attribution


def test_attribution_recall_and_precision_by_hand():
    text = (
        "Dual encoders embed questions and passages [2]. "  # supported
        "Reciprocal rank fusion combines ranked lists [3][1]. "  # supported, [1] unneeded
        "Transformers predict protein structure [1]. "  # unsupported
        "Fusion combines ranked lists."  # uncited
    )
    scores = score_attribution(analyze_answer(text, 3), PASSAGES, LexicalJudge(), threshold=0.8)
    assert scores.supported == [True, True, False, False]
    assert scores.needed == [[True], [True, False], [False], []]
    assert scores.citation_recall == 0.5
    assert scores.citation_precision == 0.5
    assert scores.unsupported_rate == 0.5


def test_citation_needed_jointly_counts_as_precise():
    passages = [Passage(1, "a", "A", "alpha beta"), Passage(2, "b", "B", "gamma delta")]
    scores = score_attribution(
        analyze_answer("alpha beta gamma delta [1][2].", 2), passages, LexicalJudge(), 0.99
    )
    assert scores.supported == [True] and scores.needed == [[True, True]]


def test_attribution_of_empty_answer_is_nan():
    scores = score_attribution(analyze_answer(ABSTAIN, 3), PASSAGES, LexicalJudge())
    assert math.isnan(scores.citation_recall) and math.isnan(scores.citation_precision)


# -------------------------------------------------------------------- pipeline


def test_searcher_modes_exclusion_and_metadata(searcher):
    query = "probabilistic term weighting for ranking documents"
    for mode in ("bm25", "dense", "hybrid"):
        response = searcher.search(query, top_k=3, mode=mode)
        assert response.hits[0].doc_id == "toy:d03" and response.hits[0].rank == 1
        assert response.timings_ms["total_ms"] > 0
    hit = searcher.search(query, top_k=1).hits[0]
    assert hit.metadata == {"year": 2017} and hit.title == "BM25 and probabilistic ranking"
    withheld = searcher.search(query, top_k=3, exclude={"toy:d03"})
    assert "toy:d03" not in [h.doc_id for h in withheld.hits] and len(withheld.hits) == 3
    with pytest.raises(ValueError):
        searcher.search(query, mode="other")


def test_context_is_numbered_truncated_and_rendered(searcher):
    hits = searcher.search("dual encoders", top_k=4).hits
    passages = build_context(hits, max_passages=2, max_words=3)
    assert [p.index for p in passages] == [1, 2]
    assert passages[0].text.endswith(" ...") and len(passages[0].text.split()) == 4
    prompt = build_prompt("What embeds passages?", passages)
    assert prompt.index("[1] ") < prompt.index("[2] ") < prompt.index("Question: What embeds")


def test_pipeline_traces_every_sentence_to_a_retrieved_document(searcher):
    provider = ScriptedProvider(
        ["Dual encoders embed questions and passages [1]. Unsupported claim [9]."]
    )
    answer = RagPipeline(searcher, provider, max_passages=3).answer(
        "embedding questions and passages"
    )
    assert answer.passages[0].doc_id == "toy:d04"
    payload = answer.to_dict()
    assert payload["sentences"][0]["cited_doc_ids"] == ["toy:d04"]
    assert payload["sentences"][1]["citations"] == [] and payload["sentences"][1][
        "invalid_markers"
    ] == [9]
    assert answer.cited_doc_ids == ["toy:d04"]
    assert "[1] Dense passage retrieval" in provider.prompts[0]
    assert payload["usage"]["retrieval_ms"] > 0


def test_extractive_provider_cites_its_source_and_abstains(searcher):
    pipeline = RagPipeline(searcher, ExtractiveProvider(max_sentences=1), max_passages=3)
    answer = pipeline.answer("dual encoders embed questions and passages")
    assert not answer.analysis.abstained
    sentence = answer.analysis.sentences[0]
    source = answer.passages[sentence.citations[0] - 1]
    assert sentence.text.rstrip(".") in source.text
    scores = score_attribution(answer.analysis, answer.passages, LexicalJudge(), 0.99)
    assert scores.citation_recall == 1.0 and scores.citation_precision == 1.0
    assert pipeline.answer("quantum chromodynamics lattice simulations").analysis.abstained


def test_lexical_judge_is_bounded():
    probs = LexicalJudge().entailment(["a cat sat", "nothing"], ["cat sat", ""])
    np.testing.assert_allclose(probs, [1.0, 0.0])


def test_rag_experiment_withholds_gold_and_reports_both_conditions(tmp_path, toy_dataset):
    import json

    import pandas as pd

    from rim.experiments.rag import RagExperimentConfig, run_rag_experiment

    cfg = RagExperimentConfig.model_validate(
        {
            "experiment": "rag_toy",
            "device": "cpu",
            "n_questions": 3,
            "encoder": {"name": "hash", "kind": "hashing", "dim": 512},
            "providers": [
                {"name": "extractive", "kind": "extractive", "options": {"max_sentences": 1}}
            ],
            "judge": {"kind": "lexical", "threshold": 0.9},
            "max_passages": 3,
            "n_bootstrap": 200,
        }
    )
    summary = run_rag_experiment(cfg, toy_dataset, tmp_path, tmp_path / "cache")
    per_answer = pd.read_csv(tmp_path / "rag_toy" / "per_answer.csv")
    assert len(per_answer) == 6
    withheld = per_answer[per_answer["condition"] == "withheld"]
    assert not withheld["gold_in_context"].any()
    assert per_answer[per_answer["condition"] == "available"]["gold_in_context"].all()

    result = summary["generators"]["extractive"]
    assert result["withheld"]["gold_in_context_rate"] == 0.0
    answered = result["available"]["answered"]
    # The extractive generator copies sentences, so every answered sentence is supported.
    assert (
        answered["citation_recall"]["mean"] == 1.0 and answered["citation_precision"]["mean"] == 1.0
    )
    low, high = result["available"]["abstention_rate"]["ci"]
    assert 0.0 <= low <= high <= 1.0

    answers = [
        json.loads(line)
        for line in (tmp_path / "rag_toy" / "answers.jsonl").read_text().splitlines()
    ]
    gold_q4 = {"toy:d07", "toy:d08"}
    for answer in answers:
        if answer["query_id"] == "toy:q4" and answer["condition"] == "withheld":
            assert not gold_q4 & {p["doc_id"] for p in answer["passages"]}
        for sentence in answer["sentences"]:
            assert set(sentence["cited_doc_ids"]) <= {p["doc_id"] for p in answer["passages"]}


def test_scifact_pairs_and_judge_validation():
    import pandas as pd

    from rim.data.loaders.scifact_claims import build_pairs
    from rim.experiments.judge_validation import classification_scores, validate_judge

    corpus = {
        1: {"title": "T1", "abstract": ["cats sleep often.", "dogs bark."]},
        2: {"title": "T2", "abstract": ["fish swim."]},
        3: {"title": "T3", "abstract": ["birds fly south."]},
    }
    claims = [
        {
            "id": 10,
            "claim": "cats sleep often",
            "cited_doc_ids": [1, 2],
            "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]},
        },
        {
            "id": 11,
            "claim": "birds never fly",
            "cited_doc_ids": [3, 99],
            "evidence": {"3": [{"sentences": [0], "label": "CONTRADICT"}]},
        },
    ]
    pairs = build_pairs(claims, corpus)
    assert list(pairs["label"]) == ["SUPPORT", "NOINFO", "CONTRADICT"]
    assert list(pairs["entailed"]) == [True, False, False]
    assert pairs.loc[0, "premise"] == "T1\ncats sleep often. dogs bark."

    probs = np.array([0.9, 0.2, 0.6, 0.4])
    truth = np.array([True, False, False, True])
    scores = classification_scores(probs, truth, 0.5)
    assert scores["accuracy"] == 0.5 and scores["precision"] == 0.5 and scores["recall"] == 0.5
    assert scores["false_positive_rate"] == 0.5

    result, dev_probs = validate_judge(LexicalJudge(), pairs, pairs, (0.5, 0.99), n_bootstrap=100)
    # The lexical judge cannot see negation: the contradicted claim shares most terms.
    assert result["dev_share_judged_entailed_by_label"]["SUPPORT"] == 1.0
    assert result["selected_threshold"] in (0.5, 0.99) and len(dev_probs) == 3
    assert isinstance(pd.Series(result["n_dev_by_label"]).sum().item(), int)
