from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.experiments.contracts import (
    ArmObservationV1,
    ExperienceLabelV1,
    ReplayCaseV1,
)
from experience_hub.experiments.errors import ExperimentInputError
from experience_hub.experiments.oracles import score_retrieval_observation
from experience_hub.retrieval.ranking import RetrievalMode


def _label(name: str, value: int) -> ExperienceLabelV1:
    return ExperienceLabelV1(
        label=name,
        experience_id=UUID(f"00000000-0000-4000-8000-{value:012d}"),
    )


def _case(
    *,
    expected: tuple[str, ...] = ("one", "two"),
    forbidden: tuple[str, ...] = ("foreign",),
) -> ReplayCaseV1:
    return ReplayCaseV1(
        schema_version=1,
        case_id="oracle-case",
        owner_agent_id=UUID("00000000-0000-4000-8000-000000000001"),
        query="bounded replay retrieval",
        mode=RetrievalMode.FOCUSED,
        tags=(),
        mechanism_cues=(),
        limit=10,
        content_budget_bytes=2_048,
        expand_cold=False,
        expected=tuple(
            _label(name, index)
            for index, name in enumerate(expected, start=100)
        ),
        forbidden=tuple(
            _label(name, index)
            for index, name in enumerate(forbidden, start=200)
        ),
    )


def test_oracle_scores_expected_labels_without_trusting_the_arm() -> None:
    evidence = score_retrieval_observation(
        _case(expected=("one", "two"), forbidden=("foreign",)),
        ArmObservationV1(
            schema_version=1,
            returned_labels=("one", "two"),
            unmapped_count=0,
        ),
    )

    assert evidence.utility_micros == 1_000_000
    assert evidence.expected_found == ("one", "two")
    assert evidence.expected_missing == ()
    assert evidence.forbidden_found == ()
    assert evidence.unmapped_count == 0


def test_oracle_uses_return_order_and_exact_floor_formula() -> None:
    evidence = score_retrieval_observation(
        _case(expected=("one", "two", "three")),
        ArmObservationV1(
            schema_version=1,
            returned_labels=("three", "one"),
            unmapped_count=0,
        ),
    )

    assert evidence.expected_found == ("three", "one")
    assert evidence.expected_missing == ("two",)
    assert evidence.utility_micros == 666_666


@pytest.mark.parametrize(
    ("returned_labels", "unmapped_count", "forbidden_found"),
    [
        (("one", "foreign"), 0, ("foreign",)),
        (("one",), 1, ()),
    ],
)
def test_oracle_forces_zero_for_forbidden_or_unmapped_hits(
    returned_labels: tuple[str, ...],
    unmapped_count: int,
    forbidden_found: tuple[str, ...],
) -> None:
    evidence = score_retrieval_observation(
        _case(),
        ArmObservationV1(
            schema_version=1,
            returned_labels=returned_labels,
            unmapped_count=unmapped_count,
        ),
    )

    assert evidence.expected_found == ("one",)
    assert evidence.forbidden_found == forbidden_found
    assert evidence.unmapped_count == unmapped_count
    assert evidence.utility_micros == 0


def test_oracle_rejects_a_case_without_expected_labels() -> None:
    invalid_case = _case().model_copy(update={"expected": ()})

    with pytest.raises(ExperimentInputError) as raised:
        score_retrieval_observation(
            invalid_case,
            ArmObservationV1(
                schema_version=1,
                returned_labels=(),
                unmapped_count=0,
            ),
        )

    assert raised.value.code == "replay_oracle_invalid_case"
    assert "00000000" not in str(raised.value)


def test_oracle_rejects_unknown_logical_labels_without_echoing_them() -> None:
    with pytest.raises(ExperimentInputError) as raised:
        score_retrieval_observation(
            _case(),
            ArmObservationV1(
                schema_version=1,
                returned_labels=("unknown-private-label",),
                unmapped_count=0,
            ),
        )

    assert raised.value.code == "replay_oracle_invalid_observation"
    assert "unknown-private-label" not in str(raised.value)


def test_observation_rejects_duplicate_returned_labels() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ArmObservationV1(
            schema_version=1,
            returned_labels=("one", "one"),
            unmapped_count=0,
        )


@pytest.mark.parametrize(
    ("expected", "forbidden"),
    [
        (
            (_label("one", 100), _label("one", 101)),
            (_label("foreign", 200),),
        ),
        (
            (_label("one", 100), _label("two", 100)),
            (_label("foreign", 200),),
        ),
        (
            (_label("one", 100),),
            (_label("foreign", 200), _label("foreign", 201)),
        ),
        (
            (_label("one", 100),),
            (_label("foreign", 200), _label("outsider", 200)),
        ),
        (
            (_label("one", 100),),
            (_label("foreign", 100),),
        ),
    ],
)
def test_oracle_revalidates_unsafe_case_identity_closure_without_echo(
    expected: tuple[ExperienceLabelV1, ...],
    forbidden: tuple[ExperienceLabelV1, ...],
) -> None:
    invalid_case = _case().model_copy(
        update={"expected": expected, "forbidden": forbidden}
    )

    with pytest.raises(ExperimentInputError) as raised:
        score_retrieval_observation(
            invalid_case,
            ArmObservationV1(
                schema_version=1,
                returned_labels=(),
                unmapped_count=0,
            ),
        )

    assert raised.value.code == "replay_oracle_invalid_case"
    assert raised.value.__cause__ is None
    assert "00000000" not in str(raised.value)
    assert "foreign" not in str(raised.value)


@pytest.mark.parametrize(
    "observation",
    [
        ArmObservationV1(
            schema_version=1,
            returned_labels=("one",),
            unmapped_count=0,
        ).model_copy(update={"returned_labels": ("one", "one")}),
        ArmObservationV1(
            schema_version=1,
            returned_labels=("one",),
            unmapped_count=0,
        ).model_copy(update={"unmapped_count": -1}),
        ArmObservationV1(
            schema_version=1,
            returned_labels=("one",),
            unmapped_count=0,
        ).model_copy(update={"schema_version": 2}),
    ],
)
def test_oracle_strictly_revalidates_unsafe_observation_without_echo(
    observation: ArmObservationV1,
) -> None:
    with pytest.raises(ExperimentInputError) as raised:
        score_retrieval_observation(_case(), observation)

    assert raised.value.code == "replay_oracle_invalid_observation"
    assert raised.value.__cause__ is None
    assert "one" not in str(raised.value)
    assert "validation" not in str(raised.value).lower()
