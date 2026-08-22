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
    BenchmarkArmKind,
    BenchmarkCaseV1,
    BenchmarkCategoryScoresV1,
    BenchmarkPackManifestV1,
    BenchmarkSourceRecordV1,
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
