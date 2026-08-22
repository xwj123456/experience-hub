"""Deterministic integer scoring for ExperienceBench-S observations."""

from __future__ import annotations

from pydantic import ValidationError

from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmObservationV1,
    BenchmarkCaseV1,
    BenchmarkCategoryScoresV1,
    BenchmarkCheckpointPredicate,
    BenchmarkOracleEvidenceV1,
)
from experience_hub.experiments.errors import ExperimentInputError


def _invalid_oracle_input() -> ExperimentInputError:
    return ExperimentInputError(
        "benchmark_oracle_invalid",
        "Benchmark observation cannot be scored by the pilot oracle",
    )


def _is_ordered_subsequence(labels: tuple[str, ...], returned: tuple[str, ...]) -> bool:
    position = 0
    for returned_label in returned:
        if returned_label == labels[position]:
            position += 1
            if position == len(labels):
                return True
    return False


def _checkpoint_satisfied(
    predicate: BenchmarkCheckpointPredicate,
    labels: tuple[str, ...],
    returned: tuple[str, ...],
    returned_membership: set[str],
) -> bool:
    if predicate is BenchmarkCheckpointPredicate.REQUIRED_SET:
        return all(label in returned_membership for label in labels)
    if predicate is BenchmarkCheckpointPredicate.ORDERED_SUBSEQUENCE:
        return _is_ordered_subsequence(labels, returned)
    raise _invalid_oracle_input()


def score_benchmark_observation(
    case: BenchmarkCaseV1,
    observation: BenchmarkArmObservationV1,
) -> BenchmarkOracleEvidenceV1:
    """Score one bounded logical-label observation without reordering evidence."""
    if not isinstance(case, BenchmarkCaseV1) or not isinstance(
        observation, BenchmarkArmObservationV1
    ):
        raise _invalid_oracle_input()
    try:
        case = BenchmarkCaseV1.model_validate(case, strict=True)
        observation = BenchmarkArmObservationV1.model_validate(observation, strict=True)
    except ValidationError:
        raise _invalid_oracle_input() from None

    returned = observation.returned_labels
    returned_membership = set(returned)
    if (
        len(returned_membership) != len(returned)
        or any(label not in case.source_labels for label in returned)
        or len(returned) > case.limit
        or observation.selected_content_bytes > case.content_budget_bytes
    ):
        raise _invalid_oracle_input()

    required_labels = tuple(
        item.label for item in case.required if item.label in returned_membership
    )
    optional_labels = tuple(
        item.label for item in case.optional if item.label in returned_membership
    )
    violation_labels = tuple(
        item.label for item in case.penalties if item.label in returned_membership
    )
    required_score = sum(
        item.weight_micros
        for item in case.required
        if item.label in returned_membership
    )
    avoidance_score = max(
        0,
        300_000
        - sum(
            item.weight_micros
            for item in case.penalties
            if item.label in returned_membership
        ),
    )
    satisfied_indexes = tuple(
        index
        for index, checkpoint in enumerate(case.checkpoints)
        if _checkpoint_satisfied(
            checkpoint.predicate,
            checkpoint.labels,
            returned,
            returned_membership,
        )
    )
    recovery_score = sum(
        case.checkpoints[index].weight_micros for index in satisfied_indexes
    )
    relevant_labels = {item.label for item in (*case.required, *case.optional)}
    relevant_returned = tuple(label for label in returned if label in relevant_labels)
    efficiency_score = (
        0 if not returned else 100_000 * len(relevant_returned) // len(returned)
    )
    utility_score = required_score + avoidance_score + recovery_score + efficiency_score

    return BenchmarkOracleEvidenceV1(
        schema_version=1,
        returned_labels=returned,
        required_labels=required_labels,
        optional_labels=optional_labels,
        violation_labels=violation_labels,
        satisfied_checkpoint_indexes=satisfied_indexes,
        scores=BenchmarkCategoryScoresV1(
            schema_version=1,
            required_coverage_micros=required_score,
            avoidance_micros=avoidance_score,
            recovery_order_micros=recovery_score,
            evidence_efficiency_micros=efficiency_score,
            utility_micros=utility_score,
        ),
    )


__all__ = ["score_benchmark_observation"]
