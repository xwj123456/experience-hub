from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.benchmark.experiencebench_factories import (
    valid_case_document,
    valid_manifest_document,
    valid_source_agent_document,
    valid_source_candidate_document,
    valid_source_experience_document,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    BenchmarkArmEvidenceV1,
    BenchmarkArmKind,
    BenchmarkCaseV1,
    BenchmarkCategoryScoresV1,
    BenchmarkEvidenceDataV1,
    BenchmarkPackManifestV1,
    BenchmarkSourceRecordV1,
    BenchmarkSummaryDataV1,
)


def _manifest_document() -> dict[str, object]:
    return valid_manifest_document(cases_sha256="a" * 64, source_sha256="b" * 64)


def _validate_case(document: dict[str, object]) -> BenchmarkCaseV1:
    return BenchmarkCaseV1.model_validate_json(canonical_json_bytes(document))


def test_manifest_pins_the_four_required_arms_in_fixed_order() -> None:
    manifest = BenchmarkPackManifestV1.model_validate_json(
        canonical_json_bytes(_manifest_document())
    )

    assert tuple(arm.kind for arm in manifest.arms) == BENCHMARK_ARM_ORDER
    assert BENCHMARK_ARM_ORDER == (
        BenchmarkArmKind.NO_MEMORY,
        BenchmarkArmKind.RECENT_NOTES,
        BenchmarkArmKind.SQLITE_BM25,
        BenchmarkArmKind.EXPERIENCE_HUB,
    )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"unexpected": "value"}, "Extra inputs are not permitted"),
        ({"schema_version": 2}, "Input should be 1"),
        (
            {
                "arms": list(reversed(_manifest_document()["arms"])),
            },
            "ordered required benchmark arms",
        ),
        (
            {
                "composition": {
                    **_manifest_document()["composition"],
                    "case_count": 29,
                }
            },
            "case_count",
        ),
    ],
)
def test_manifest_rejects_contract_breaks(
    update: dict[str, object], message: str
) -> None:
    document = _manifest_document()
    document.update(update)

    with pytest.raises(ValidationError, match=message):
        BenchmarkPackManifestV1.model_validate_json(canonical_json_bytes(document))


def test_case_keeps_penalty_order_for_the_oracle() -> None:
    case = _validate_case(valid_case_document())

    assert tuple(item.label for item in case.penalties) == (
        "queue-forbidden",
        "queue-stale",
        "queue-misleading",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["required"].append(
                {"label": "queue-required", "weight_micros": 0}
            ),
            "pairwise disjoint",
        ),
        (
            lambda document: document.update(
                {"optional": [{"label": "queue-required", "weight_micros": 0}]}
            ),
            "disjoint",
        ),
        (
            lambda document: document.update(
                {"required": [{"label": "queue-required", "weight_micros": 449999}]}
            ),
            "450000",
        ),
        (
            lambda document: document.update(
                {"forbidden": [{"label": "queue-forbidden", "weight_micros": 99999}]}
            ),
            "300000",
        ),
        (
            lambda document: document.update(
                {
                    "checkpoints": [
                        {
                            "predicate": "ordered_subsequence",
                            "labels": ["queue-required"],
                            "weight_micros": 149999,
                        }
                    ]
                }
            ),
            "150000",
        ),
        (
            lambda document: document.update({"required": []}),
            "at least one",
        ),
    ],
)
def test_case_rejects_invalid_rubric(
    mutate: object, message: str
) -> None:
    document = deepcopy(valid_case_document())
    assert callable(mutate)
    mutate(document)

    with pytest.raises(ValidationError, match=message):
        _validate_case(document)


def test_canonical_scores_reject_float_values() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCategoryScoresV1.model_validate(
            {
                "schema_version": 1,
                "required_coverage_micros": 450000.0,
                "avoidance_micros": 300000,
                "recovery_order_micros": 150000,
                "evidence_efficiency_micros": 100000,
                "utility_micros": 1000000,
            }
        )


def test_source_records_are_strict_discriminated_documents() -> None:
    adapter = TypeAdapter(BenchmarkSourceRecordV1)
    agent = adapter.validate_json(canonical_json_bytes(valid_source_agent_document()))
    experience = adapter.validate_json(
        canonical_json_bytes(valid_source_experience_document())
    )
    candidate = adapter.validate_json(
        canonical_json_bytes(valid_source_candidate_document())
    )

    assert (agent.record_type, experience.record_type, candidate.record_type) == (
        "agent",
        "experience",
        "candidate",
    )


def test_source_experience_rejects_raw_uuid_evidence_labels() -> None:
    document = valid_source_experience_document()
    document["evidence"] = [
        {"type": "fixture", "label": "550e8400-e29b-41d4-a716-446655440000"}
    ]

    with pytest.raises(ValidationError, match="benchmark label"):
        TypeAdapter(BenchmarkSourceRecordV1).validate_json(
            canonical_json_bytes(document)
        )


def test_manifest_requires_distinct_case_and_source_files() -> None:
    document = _manifest_document()
    document["source"] = {"file": "pilot-cases.jsonl", "sha256": "b" * 64}

    with pytest.raises(ValidationError, match="distinct"):
        BenchmarkPackManifestV1.model_validate_json(canonical_json_bytes(document))


def _complete_arm_document(arm_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "arm_id": arm_id,
        "status": "complete",
        "observation": {
            "schema_version": 1,
            "returned_labels": [],
            "selected_content_bytes": 0,
        },
        "oracle": {
            "schema_version": 1,
            "returned_labels": [],
            "required_labels": [],
            "optional_labels": [],
            "violation_labels": [],
            "satisfied_checkpoint_indexes": [],
            "scores": {
                "schema_version": 1,
                "required_coverage_micros": 0,
                "avoidance_micros": 0,
                "recovery_order_micros": 0,
                "evidence_efficiency_micros": 0,
                "utility_micros": 0,
            },
        },
        "error_code": None,
        "error_stage": None,
    }


@pytest.mark.parametrize(
    ("error_code", "error_stage"),
    (
        ("benchmark_policy_failed", None),
        (None, "policy"),
    ),
)
def test_complete_arm_rejects_partial_error_details(
    error_code: str | None, error_stage: str | None
) -> None:
    document = _complete_arm_document("no_memory")
    document["error_code"] = error_code
    document["error_stage"] = error_stage

    with pytest.raises(
        ValidationError, match="complete arms require observation and oracle only"
    ):
        BenchmarkArmEvidenceV1.model_validate_json(canonical_json_bytes(document))


@pytest.mark.parametrize("result_field", ("observation", "oracle"))
def test_failed_arm_rejects_partial_result_details(result_field: str) -> None:
    document = _complete_arm_document("no_memory")
    retained_result = document[result_field]
    document.update(
        {
            "status": "failed",
            "observation": None,
            "oracle": None,
            "error_code": "benchmark_policy_failed",
            "error_stage": "policy",
        }
    )
    document[result_field] = retained_result

    with pytest.raises(
        ValidationError, match="failed arms require stable error details only"
    ):
        BenchmarkArmEvidenceV1.model_validate_json(canonical_json_bytes(document))


def _valid_safety_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "owner_leak_count": 0,
        "quarantine_leak_count": 0,
        "cross_arm_contamination_count": 0,
        "source_mutation_count": 0,
        "source_unchanged": True,
        "clone_isolation_verified": True,
    }


def _valid_aggregate_document() -> dict[str, object]:
    scopes = (
        "recurring_workflow",
        "environment_gotcha",
        "state_change",
        "failure_recovery",
        "irrelevant_distractor",
    )
    return {
        "schema_version": 1,
        "overall": {
            "schema_version": 1,
            "scope": "overall",
            "sum_delta_micros": 0,
            "case_count": 30,
            "mean_delta_micros": 0,
        },
        "strata": [
            {
                "schema_version": 1,
                "scope": scope,
                "sum_delta_micros": 0,
                "case_count": 6,
                "mean_delta_micros": 0,
            }
            for scope in scopes
        ],
    }


def _valid_pass_payload_document() -> dict[str, object]:
    strata = (
        "recurring_workflow",
        "environment_gotcha",
        "state_change",
        "failure_recovery",
        "irrelevant_distractor",
    )
    return {
        "schema_version": 1,
        "resolved_manifest": {
            "schema_version": 1,
            "pack_id": "experiencebench-s-pilot",
            "maturity": "pilot-30",
            "manifest_sha256": "a" * 64,
            "cases_sha256": "b" * 64,
            "source_fixture_sha256": "c" * 64,
            "snapshot_sha256": "d" * 64,
            "source_schema_revision": 1,
            "frozen_at": "2026-08-20T00:00:00Z",
            "seed": 20260820,
            "arms": _manifest_document()["arms"],
            "oracle_version": 1,
            "metric_version": 1,
            "gate_version": 1,
            "evidence_schema_version": 1,
            "summary_schema_version": 1,
            "profile_schema_version": 1,
        },
        "cases": [
            {
                "schema_version": 1,
                "case": {
                    **valid_case_document(),
                    "case_id": f"case-{index}",
                    "stratum": strata[index // 6],
                },
                "case_id": f"case-{index}",
                "source_class": "public_authored",
                "stratum": strata[index // 6],
                "status": "complete",
                "arms": [
                    _complete_arm_document(arm_id)
                    for arm_id in (
                        "no_memory",
                        "recent_notes",
                        "sqlite_bm25",
                        "experience_hub",
                    )
                ],
                "comparator_arm_id": "no_memory",
                "comparator_utility_micros": 0,
                "experience_hub_utility_micros": 0,
                "delta_utility_micros": 0,
            }
            for index in range(30)
        ],
        "comparison_complete": True,
        "safety": _valid_safety_document(),
        "aggregate": _valid_aggregate_document(),
    }


def _valid_evidence_data_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pass_payload": _valid_pass_payload_document(),
        "deterministic_replay_match": True,
        "gates": [{"schema_version": 1, "gate_id": "pilot_gate", "passed": True}],
        "expansion_gate_passed": True,
        "valid": True,
    }


def test_evidence_requires_pilot_cardinality_and_consistent_state() -> None:
    document = _valid_evidence_data_document()
    document["pass_payload"]["cases"] = document["pass_payload"]["cases"][:-1]

    with pytest.raises(ValidationError, match="exactly 30"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))


def test_case_evidence_requires_one_identity_bound_canonical_case() -> None:
    document = _valid_evidence_data_document()
    cases = document["pass_payload"]["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    first.pop("case")

    with pytest.raises(ValidationError, match="case"):
        BenchmarkEvidenceDataV1.model_validate_json(
            canonical_json_bytes(document)
        )

    document = _valid_evidence_data_document()
    document["deterministic_replay_match"] = False

    with pytest.raises(ValidationError, match="valid must match"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))

    document = _valid_evidence_data_document()
    document["gates"] = []

    with pytest.raises(ValidationError, match="at least one"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))

    document = _valid_evidence_data_document()
    document["gates"] = [
        {"schema_version": 1, "gate_id": "pilot_gate", "passed": False}
    ]

    with pytest.raises(ValidationError, match="expansion_gate_passed"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))

    document = _valid_evidence_data_document()
    document["pass_payload"]["aggregate"]["overall"]["case_count"] = 29

    with pytest.raises(ValidationError, match="overall aggregate"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))

    document = _valid_evidence_data_document()
    document["pass_payload"]["aggregate"]["strata"][0]["case_count"] = 5

    with pytest.raises(ValidationError, match="stratum aggregate"):
        BenchmarkEvidenceDataV1.model_validate_json(canonical_json_bytes(document))


def test_summary_requires_fixed_pilot_cardinality_and_consistent_state() -> None:
    document = {
        "schema_version": 1,
        "pack_id": "experiencebench-s-pilot",
        "manifest_sha256": "a" * 64,
        "cases_sha256": "b" * 64,
        "source_fixture_sha256": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "evidence_sha256": "e" * 64,
        "case_count": 29,
        "arm_count": 4,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "safety": _valid_safety_document(),
        "aggregate": _valid_aggregate_document(),
        "gates": [{"schema_version": 1, "gate_id": "pilot_gate", "passed": True}],
        "expansion_gate_passed": True,
        "valid": True,
        "claim_boundary": "Pilot results are limited to this pack.",
    }

    with pytest.raises(ValidationError, match="Input should be 30"):
        BenchmarkSummaryDataV1.model_validate_json(canonical_json_bytes(document))

    document["case_count"] = 30
    document["arm_count"] = 1

    with pytest.raises(ValidationError, match="Input should be 4"):
        BenchmarkSummaryDataV1.model_validate_json(canonical_json_bytes(document))

    document["arm_count"] = 4
    document["deterministic_replay_match"] = False

    with pytest.raises(ValidationError, match="valid must match"):
        BenchmarkSummaryDataV1.model_validate_json(canonical_json_bytes(document))
