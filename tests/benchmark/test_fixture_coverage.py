"""Coverage guarantees for the committed deterministic benchmark corpus."""

from __future__ import annotations

from pathlib import Path

from experience_hub.benchmark.cases import (
    ColdCueCase,
    InspirationCase,
    IrrelevantDistractorCase,
    PropagationCase,
    RetrievalCase,
    load_cases,
    load_seed,
)
from experience_hub.retrieval import RetrievalMode

REPOSITORY_ROOT = Path(__file__).parents[2]
SEED_PATH = REPOSITORY_ROOT / "benchmarks" / "seed.json"
CASES_PATH = REPOSITORY_ROOT / "benchmarks" / "cases.jsonl"
REQUIRED_SEED_TAGS = frozenset(
    {
        "coverage:english",
        "coverage:chinese",
        "coverage:mixed",
        "coverage:operational_causality",
        "coverage:counterfactual_applicability",
        "coverage:distant_mechanism_analogy",
        "coverage:cold_relevant",
        "coverage:cold_distractor",
    }
)


def _has_latin(value: str) -> bool:
    return any(character.isascii() and character.isalpha() for character in value)


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)


def test_case_corpus_covers_every_required_category_and_language() -> None:
    seed = load_seed(SEED_PATH)
    cases = load_cases(CASES_PATH, seed=seed)

    retrieval = tuple(case for case in cases if isinstance(case, RetrievalCase))
    cold = tuple(case for case in cases if isinstance(case, ColdCueCase))
    distractors = tuple(
        case for case in cases if isinstance(case, IrrelevantDistractorCase)
    )
    propagation = tuple(case for case in cases if isinstance(case, PropagationCase))
    inspiration = tuple(case for case in cases if isinstance(case, InspirationCase))

    assert len(retrieval) >= 3
    assert all(case.mode is RetrievalMode.FOCUSED for case in retrieval)
    retrieval_languages = {
        (
            "mixed"
            if _has_latin(case.query) and _has_cjk(case.query)
            else "english"
            if _has_latin(case.query)
            else "chinese"
        )
        for case in retrieval
    }
    assert {"english", "chinese", "mixed"} <= retrieval_languages

    assert sum(case.mode is RetrievalMode.FOCUSED for case in cold) >= 2
    assert sum(case.mode is RetrievalMode.ASSOCIATIVE for case in cold) >= 2
    assert len(distractors) >= 3
    assert all(case.expected_false_reactivations == 0 for case in distractors)
    assert len(propagation) >= 1
    assert all(not case.pending_relevant_labels for case in propagation)
    assert len(inspiration) >= 4
    assert sum(case.expected_min_valid_ideas for case in inspiration) >= 12


def test_seed_covers_required_memory_semantics_with_runnable_evidence() -> None:
    seed = load_seed(SEED_PATH)
    cases = load_cases(CASES_PATH, seed=seed)
    by_label = {experience.label: experience for experience in seed.experiences}
    corpus_tags = frozenset(
        tag for experience in seed.experiences for tag in experience.tags
    )

    assert corpus_tags >= REQUIRED_SEED_TAGS

    cold_labels = {
        label
        for case in cases
        if isinstance(case, ColdCueCase)
        for label in case.relevant_labels
    }
    assert cold_labels
    assert all(
        by_label[label].target_temperature.value == "cold" for label in cold_labels
    )

    distractor_memories = tuple(
        experience
        for experience in seed.experiences
        if "coverage:cold_distractor" in experience.tags
    )
    assert len(distractor_memories) >= 3
    assert all(
        experience.target_temperature.value == "cold"
        for experience in distractor_memories
    )

    inspiration_cases = tuple(
        case for case in cases if isinstance(case, InspirationCase)
    )
    for case in inspiration_cases:
        evidence = tuple(by_label[label] for label in case.evidence_labels)
        assert len(evidence) >= 3
        assert all(item.applicability for item in evidence)
        assert all(item.evidence for item in evidence)
        assert all(item.falsifiers for item in evidence)
