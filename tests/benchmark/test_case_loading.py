from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from experience_hub.benchmark.cases import (
    BenchmarkFixtureError,
    ColdCueCase,
    InspirationCase,
    IrrelevantDistractorCase,
    PropagationCase,
    RetrievalCase,
    load_cases,
    load_seed,
)


def _valid_seed() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "clock": {
            "started_at": "2026-01-01T00:00:00Z",
            "cold_at": "2026-01-17T00:15:01Z",
        },
        "config": {
            "retrieval_limit": 5,
            "content_budget_bytes": 65536,
            "expand_cold": True,
            "generator": "deterministic",
            "operators": [
                "causal_gap",
                "counterfactual",
                "distant_analogy",
            ],
            "include_inbox": False,
            "branches_per_operator": 3,
            "output_tokens_per_operator": 1200,
            "total_output_tokens": 3600,
            "operator_timeout_seconds": 30,
            "global_timeout_seconds": 90,
            "lifecycle": {
                "recency_half_life_hours": 168.0,
                "frequency_half_life_hours": 336.0,
                "warm_to_hot_threshold": 0.75,
                "hot_to_warm_threshold": 0.62,
                "warm_to_cold_threshold": 0.30,
                "demotion_cycles": 2,
                "archive_after_days": 90.0,
                "archive_importance_threshold": 0.75,
                "archive_confidence_threshold": 0.25,
                "archive_strength_threshold": 0.10,
                "minimum_cycle_interval_seconds": 900.0,
                "worker_interval_seconds": 900.0,
                "lease_duration_seconds": 300.0,
            },
        },
        "agents": [{"label": "alice"}, {"label": "bob"}],
        "experiences": [
            {
                "label": "lease-causal",
                "owner_label": "alice",
                "kind": "procedural",
                "body": "Renew the lease before rotating the worker.",
                "summary": "Lease renewal prevents split ownership.",
                "mechanism": "A bounded lease preserves exclusive ownership.",
                "tags": ["lease", "worker"],
                "applicability": ["single active coordinator"],
                "evidence": [{"type": "incident", "id": "inc-lease-01"}],
                "falsifiers": ["Two coordinators own the same lease."],
                "importance": 0.80,
                "confidence": 0.90,
                "target_temperature": "hot",
            },
            {
                "label": "cold-backpressure",
                "owner_label": "alice",
                "kind": "semantic",
                "body": "A token bucket smooths burst pressure.",
                "summary": "Token buckets bound bursts.",
                "mechanism": "Stored capacity absorbs short bursts.",
                "tags": ["backpressure", "token-bucket"],
                "applicability": ["bursty ingress"],
                "evidence": [{"type": "experiment", "id": "exp-bucket-01"}],
                "falsifiers": ["Ingress is perfectly constant."],
                "importance": 0.20,
                "confidence": 0.30,
                "target_temperature": "cold",
            },
        ],
    }


def _valid_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "focused-lease",
            "type": "retrieval",
            "owner_label": "alice",
            "query": "lease worker ownership",
            "mode": "focused",
            "relevant_labels": ["lease-causal"],
        },
        {
            "id": "cold-bucket",
            "type": "cold_cue",
            "owner_label": "alice",
            "query": "token bucket burst capacity",
            "mode": "focused",
            "relevant_labels": ["cold-backpressure"],
            "expected_reactivated_labels": ["cold-backpressure"],
        },
        {
            "id": "distractor-nebula",
            "type": "irrelevant_distractor",
            "owner_label": "alice",
            "query": "nebula orchid melody",
            "mode": "associative",
            "relevant_labels": [],
            "expected_false_reactivations": 0,
        },
        {
            "id": "propagate-lease",
            "type": "propagation",
            "owner_label": "bob",
            "sender_label": "alice",
            "recipient_label": "bob",
            "source_labels": ["lease-causal"],
            "query": "lease worker ownership",
            "pending_relevant_labels": [],
            "adopted_relevant_labels": ["lease-causal"],
        },
        {
            "id": "inspire-pressure",
            "type": "inspiration",
            "owner_label": "alice",
            "goal": "Design a safer burst-control mechanism.",
            "context": "The input rate is uncertain.",
            "mode": "associative",
            "evidence_labels": ["lease-causal", "cold-backpressure"],
            "expected_min_valid_ideas": 3,
            "expected_min_distinct_mechanisms": 2,
        },
    ]


def _write_seed(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_cases(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n"
            for value in values
        ),
        encoding="utf-8",
    )


def test_loaders_return_strict_discriminated_models(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    _write_cases(cases_path, _valid_cases())

    seed = load_seed(seed_path)
    cases = load_cases(cases_path, seed=seed)

    assert seed.schema_version == 1
    assert seed.clock.started_at.isoformat() == "2026-01-01T00:00:00+00:00"
    assert tuple(experience.label for experience in seed.experiences) == (
        "lease-causal",
        "cold-backpressure",
    )
    assert tuple(type(case) for case in cases) == (
        RetrievalCase,
        ColdCueCase,
        IrrelevantDistractorCase,
        PropagationCase,
        InspirationCase,
    )
    assert cases[0].relevant_labels == frozenset({"lease-causal"})
    assert cases[1].expected_reactivated_labels == frozenset({"cold-backpressure"})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda seed: seed.pop("clock"),
            "clock",
        ),
        (
            lambda seed: seed["clock"].pop("cold_at"),
            "cold_at",
        ),
        (
            lambda seed: seed["config"].pop("lifecycle"),
            "lifecycle",
        ),
        (
            lambda seed: seed["config"].pop("generator"),
            "generator",
        ),
        (
            lambda seed: seed["config"].pop("operators"),
            "operators",
        ),
        (
            lambda seed: seed["config"].pop("include_inbox"),
            "include_inbox",
        ),
        (
            lambda seed: seed["config"].pop("expand_cold"),
            "expand_cold",
        ),
        (
            lambda seed: seed["config"]["lifecycle"].pop("recency_half_life_hours"),
            "recency_half_life_hours",
        ),
        (
            lambda seed: seed["config"].__setitem__("retrieval_limit", "5"),
            "retrieval_limit",
        ),
        (
            lambda seed: seed.__setitem__("unknown", True),
            "unknown",
        ),
    ],
)
def test_seed_rejects_unpinned_or_non_strict_configuration(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    seed_value = _valid_seed()
    mutation(seed_value)
    path = tmp_path / "seed.json"
    _write_seed(path, seed_value)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_seed(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda seed: seed["clock"].__setitem__("started_at", "2026-01-01T00:00:00"),
            "UTC",
        ),
        (
            lambda seed: seed["clock"].__setitem__("cold_at", "2025-12-31T00:00:00Z"),
            "after started_at",
        ),
        (
            lambda seed: seed["agents"].append({"label": "alice"}),
            "Duplicate agent label",
        ),
        (
            lambda seed: seed["experiences"].append(deepcopy(seed["experiences"][0])),
            "Duplicate experience label",
        ),
        (
            lambda seed: seed["experiences"][0].__setitem__(
                "owner_label", "unknown-agent"
            ),
            "Unknown owner label",
        ),
    ],
)
def test_seed_rejects_invalid_pins_and_label_registry(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    seed_value = _valid_seed()
    mutation(seed_value)
    path = tmp_path / "seed.json"
    _write_seed(path, seed_value)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_seed(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda seed: seed["experiences"][0].__setitem__(
                "body",
                "界" * 21_846,
            ),
            "64 KiB",
        ),
        (
            lambda seed: seed["experiences"][0].__setitem__(
                "tags",
                [f"tag-{index}" for index in range(33)],
            ),
            "32",
        ),
    ],
)
def test_seed_rejects_content_the_public_experience_domain_cannot_store(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    seed_value = _valid_seed()
    mutation(seed_value)
    path = tmp_path / "seed.json"
    _write_seed(path, seed_value)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_seed(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda seed: seed["config"].__setitem__(
                "generator",
                "openai_compatible",
            ),
            "deterministic",
        ),
        (
            lambda seed: seed["config"].__setitem__(
                "operators",
                ["counterfactual", "causal_gap", "distant_analogy"],
            ),
            "canonical",
        ),
        (
            lambda seed: seed["config"].__setitem__("include_inbox", True),
            "include_inbox",
        ),
        (
            lambda seed: seed["config"].__setitem__("expand_cold", False),
            "expand_cold",
        ),
    ],
)
def test_seed_rejects_nondeterministic_or_incomplete_benchmark_modes(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    seed_value = _valid_seed()
    mutation(seed_value)
    path = tmp_path / "seed.json"
    _write_seed(path, seed_value)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_seed(path)


@pytest.mark.parametrize(
    ("case_index", "field", "value", "match"),
    [
        (0, "relevant_labels", [], "nonempty"),
        (0, "relevant_labels", "lease-causal", "array"),
        (
            0,
            "relevant_labels",
            ["lease-causal", "lease-causal"],
            "duplicates",
        ),
        (1, "relevant_labels", [], "nonempty"),
        (1, "expected_reactivated_labels", [], "nonempty"),
        (
            1,
            "expected_reactivated_labels",
            ["lease-causal"],
            "subset",
        ),
        (
            2,
            "relevant_labels",
            ["lease-causal"],
            "empty",
        ),
        (2, "expected_false_reactivations", 1, "0"),
        (
            3,
            "pending_relevant_labels",
            ["lease-causal"],
            "empty",
        ),
        (3, "source_labels", [], "nonempty"),
        (3, "adopted_relevant_labels", [], "nonempty"),
        (4, "evidence_labels", [], "nonempty"),
    ],
)
def test_cases_reject_every_invalid_expected_set_shape(
    tmp_path: Path,
    case_index: int,
    field: str,
    value: object,
    match: str,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    cases[case_index][field] = value
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_cases(cases_path, seed=load_seed(seed_path))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cases: cases[0].__setitem__(
                "relevant_labels", ["unknown-seed-label"]
            ),
            "Unknown seed label",
        ),
        (
            lambda cases: cases[4].__setitem__(
                "evidence_labels", ["unknown-seed-label"]
            ),
            "Unknown seed label",
        ),
        (
            lambda cases: cases[0].__setitem__("owner_label", "nobody"),
            "Unknown agent label",
        ),
        (
            lambda cases: cases[0].__setitem__("query", 7),
            "query",
        ),
        (
            lambda cases: cases[0].__setitem__("extra", True),
            "extra",
        ),
        (
            lambda cases: cases[0].__setitem__("type", "unknown"),
            "type",
        ),
    ],
)
def test_cases_reject_unknown_references_fields_and_types(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    mutation(cases)
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_cases(cases_path, seed=load_seed(seed_path))


@pytest.mark.parametrize(
    ("case_index", "mutation", "match"),
    [
        (
            0,
            lambda case: case.__setitem__("owner_label", "bob"),
            "owner",
        ),
        (
            1,
            lambda case: case.__setitem__("owner_label", "bob"),
            "owner",
        ),
        (
            4,
            lambda case: case.__setitem__("owner_label", "bob"),
            "owner",
        ),
        (
            3,
            lambda case: case.__setitem__("sender_label", "bob"),
            "sender",
        ),
    ],
)
def test_cases_reject_labels_inaccessible_to_the_executing_owner(
    tmp_path: Path,
    case_index: int,
    mutation: Any,
    match: str,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    mutation(cases[case_index])
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match=match):
        load_cases(cases_path, seed=load_seed(seed_path))


def test_propagation_requires_adopted_labels_to_match_capsule_sources(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    cases[3]["adopted_relevant_labels"] = ["cold-backpressure"]
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match="source_labels"):
        load_cases(cases_path, seed=load_seed(seed_path))


def test_cold_cue_labels_must_reference_cold_seed_experiences(
    tmp_path: Path,
) -> None:
    seed_value = _valid_seed()
    seed_value["experiences"][1]["target_temperature"] = "warm"
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, seed_value)
    _write_cases(cases_path, _valid_cases())

    with pytest.raises(BenchmarkFixtureError, match="target cold"):
        load_cases(cases_path, seed=load_seed(seed_path))


def test_inspiration_expectations_must_fit_the_pinned_branch_budget(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    cases[4]["expected_min_valid_ideas"] = 10
    cases[4]["expected_min_distinct_mechanisms"] = 10
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match="branch budget"):
        load_cases(cases_path, seed=load_seed(seed_path))


def test_cases_reject_duplicate_case_ids(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases = _valid_cases()
    cases[1]["id"] = cases[0]["id"]
    _write_cases(cases_path, cases)

    with pytest.raises(BenchmarkFixtureError, match="Duplicate case ID"):
        load_cases(cases_path, seed=load_seed(seed_path))


@pytest.mark.parametrize(
    "body",
    [
        "",
        "\n",
        "[]\n",
        '{"id":"broken"\n',
        '{"id":"one"}\n\n{"id":"two"}\n',
    ],
)
def test_case_loader_rejects_empty_nonobject_malformed_or_blank_lines(
    tmp_path: Path,
    body: str,
) -> None:
    seed_path = tmp_path / "seed.json"
    cases_path = tmp_path / "cases.jsonl"
    _write_seed(seed_path, _valid_seed())
    cases_path.write_text(body, encoding="utf-8")

    with pytest.raises(BenchmarkFixtureError):
        load_cases(cases_path, seed=load_seed(seed_path))
