from __future__ import annotations

from copy import deepcopy

import pytest
from tests.benchmark.experiencebench_factories import valid_case_document

from experience_hub.canonical import canonical_json_bytes
from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmObservationV1,
    BenchmarkCaseV1,
)
from experience_hub.experiments.benchmarks.oracles import score_benchmark_observation
from experience_hub.experiments.errors import ExperimentInputError


def _case() -> BenchmarkCaseV1:
    document = deepcopy(valid_case_document())
    document["source_labels"] = [
        "queue-required-one",
        "queue-required-two",
        "queue-required-three",
        "queue-optional",
        "queue-forbidden",
        "queue-stale",
        "queue-misleading",
    ]
    document["required"] = [
        {"label": "queue-required-one", "weight_micros": 200_000},
        {"label": "queue-required-two", "weight_micros": 150_000},
        {"label": "queue-required-three", "weight_micros": 100_000},
    ]
    document["optional"] = [{"label": "queue-optional", "weight_micros": 0}]
    document["checkpoints"] = [
        {
            "predicate": "required_set",
            "labels": ["queue-required-one", "queue-required-two"],
            "weight_micros": 50_000,
        },
        {
            "predicate": "ordered_subsequence",
            "labels": ["queue-required-two", "queue-required-three"],
            "weight_micros": 100_000,
        },
    ]
    return BenchmarkCaseV1.model_validate_json(canonical_json_bytes(document))


def _observation(*labels: str, selected_bytes: int = 0) -> BenchmarkArmObservationV1:
    return BenchmarkArmObservationV1(
        schema_version=1,
        returned_labels=labels,
        selected_content_bytes=selected_bytes,
    )


def test_empty_observation_receives_only_avoidance_credit() -> None:
    evidence = score_benchmark_observation(_case(), _observation())

    assert evidence.returned_labels == ()
    assert evidence.scores.utility_micros == 300_000
    assert evidence.scores.evidence_efficiency_micros == 0


def test_full_required_set_scores_one_million_micros() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation(
            "queue-required-one",
            "queue-required-two",
            "queue-required-three",
        ),
    )

    assert evidence.required_labels == (
        "queue-required-one",
        "queue-required-two",
        "queue-required-three",
    )
    assert evidence.satisfied_checkpoint_indexes == (0, 1)
    assert evidence.scores.utility_micros == 1_000_000
    assert evidence.scores.utility_micros == sum(
        (
            evidence.scores.required_coverage_micros,
            evidence.scores.avoidance_micros,
            evidence.scores.recovery_order_micros,
            evidence.scores.evidence_efficiency_micros,
        )
    )


def test_partial_weighted_coverage_keeps_declared_label_order() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation("queue-required-three", "queue-required-one"),
    )

    assert evidence.returned_labels == ("queue-required-three", "queue-required-one")
    assert evidence.required_labels == ("queue-required-one", "queue-required-three")
    assert evidence.scores.required_coverage_micros == 300_000
    assert evidence.scores.recovery_order_micros == 0


@pytest.mark.parametrize(
    ("label", "expected_avoidance"),
    (
        ("queue-forbidden", 200_000),
        ("queue-stale", 200_000),
        ("queue-misleading", 200_000),
    ),
)
def test_each_penalty_group_reduces_avoidance(
    label: str, expected_avoidance: int
) -> None:
    evidence = score_benchmark_observation(_case(), _observation(label))

    assert evidence.violation_labels == (label,)
    assert evidence.scores.avoidance_micros == expected_avoidance


def test_combined_penalties_floor_avoidance_at_zero() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation("queue-forbidden", "queue-stale", "queue-misleading"),
    )

    assert evidence.violation_labels == (
        "queue-forbidden",
        "queue-stale",
        "queue-misleading",
    )
    assert evidence.scores.avoidance_micros == 0


def test_ordered_checkpoint_allows_unrelated_labels_between_required_labels() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation(
            "queue-required-two",
            "queue-optional",
            "queue-required-three",
        ),
    )

    assert evidence.satisfied_checkpoint_indexes == (1,)
    assert evidence.optional_labels == ("queue-optional",)


def test_out_of_order_recovery_does_not_satisfy_ordered_checkpoint() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation("queue-required-three", "queue-required-two"),
    )

    assert evidence.satisfied_checkpoint_indexes == ()


def test_required_set_checkpoint_ignores_returned_order() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation("queue-required-two", "queue-required-one"),
    )

    assert evidence.satisfied_checkpoint_indexes == (0,)


def test_optional_label_counts_as_efficient_evidence_without_utility_weight() -> None:
    evidence = score_benchmark_observation(
        _case(),
        _observation("queue-required-one", "queue-optional", "queue-forbidden"),
    )

    assert evidence.optional_labels == ("queue-optional",)
    assert evidence.scores.evidence_efficiency_micros == 66_666


@pytest.mark.parametrize(
    "observation",
    (
        BenchmarkArmObservationV1.model_construct(
            schema_version=1,
            returned_labels=("queue-required-one", "queue-required-one"),
            selected_content_bytes=0,
        ),
        BenchmarkArmObservationV1.model_construct(
            schema_version=1,
            returned_labels=("not-declared",),
            selected_content_bytes=0,
        ),
        BenchmarkArmObservationV1.model_construct(
            schema_version=1,
            returned_labels=(
                "queue-required-one",
                "queue-required-two",
                "queue-required-three",
                "queue-optional",
                "queue-forbidden",
                "queue-stale",
            ),
            selected_content_bytes=0,
        ),
        BenchmarkArmObservationV1.model_construct(
            schema_version=1,
            returned_labels=(),
            selected_content_bytes=4_097,
        ),
    ),
)
def test_invalid_observations_fail_with_the_stable_oracle_code(
    observation: BenchmarkArmObservationV1,
) -> None:
    with pytest.raises(ExperimentInputError, match="benchmark_oracle_invalid"):
        score_benchmark_observation(_case(), observation)
