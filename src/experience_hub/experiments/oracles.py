"""Deterministic truth-source scoring for replay retrieval observations."""

from __future__ import annotations

from pydantic import ValidationError

from experience_hub.experiments.contracts import (
    ArmObservationV1,
    OracleEvidenceV1,
    ReplayCaseV1,
)
from experience_hub.experiments.errors import ExperimentInputError


def _invalid_case() -> ExperimentInputError:
    return ExperimentInputError(
        "replay_oracle_invalid_case",
        "Replay case cannot be scored by the retrieval oracle",
    )


def _invalid_observation() -> ExperimentInputError:
    return ExperimentInputError(
        "replay_oracle_invalid_observation",
        "Policy observation cannot be scored by the retrieval oracle",
    )


def score_retrieval_observation(
    case: ReplayCaseV1,
    observation: ArmObservationV1,
) -> OracleEvidenceV1:
    """Score only logical labels declared by the case truth source."""
    if not isinstance(case, ReplayCaseV1):
        raise _invalid_case()
    if not isinstance(observation, ArmObservationV1):
        raise _invalid_observation()
    try:
        case = ReplayCaseV1.model_validate(case, strict=True)
    except ValidationError:
        raise _invalid_case() from None
    try:
        observation = ArmObservationV1.model_validate(
            observation,
            strict=True,
        )
    except ValidationError:
        raise _invalid_observation() from None

    expected_labels = tuple(item.label for item in case.expected)
    forbidden_labels = tuple(item.label for item in case.forbidden)
    if not expected_labels or set(expected_labels) & set(forbidden_labels):
        raise _invalid_case()

    known_labels = set(expected_labels) | set(forbidden_labels)
    if any(label not in known_labels for label in observation.returned_labels):
        raise _invalid_observation()

    expected = set(expected_labels)
    forbidden = set(forbidden_labels)
    returned = observation.returned_labels
    expected_found = tuple(label for label in returned if label in expected)
    expected_missing = tuple(
        label for label in expected_labels if label not in expected_found
    )
    forbidden_found = tuple(label for label in returned if label in forbidden)
    utility_micros = (
        1_000_000 * len(expected_found) // len(expected_labels)
    )
    if forbidden_found or observation.unmapped_count:
        utility_micros = 0

    return OracleEvidenceV1(
        schema_version=1,
        expected_found=expected_found,
        expected_missing=expected_missing,
        forbidden_found=forbidden_found,
        unmapped_count=observation.unmapped_count,
        utility_micros=utility_micros,
    )


__all__ = ["score_retrieval_observation"]
