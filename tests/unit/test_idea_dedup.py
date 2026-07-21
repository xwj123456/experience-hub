from __future__ import annotations

from uuid import UUID

import pytest

from experience_hub.inspiration.dedup import (
    MAX_IDEAS_PER_OPERATOR,
    MAX_IDEAS_PER_RUN,
    RetainedIdea,
    RunDeduplicationResult,
    deduplicate_run_batches,
)
from experience_hub.inspiration.generators.base import OperatorFailureCode
from experience_hub.inspiration.hashing import (
    NEAR_DUPLICATE_THRESHOLD,
    mechanism_similarity,
)
from experience_hub.inspiration.models import (
    IdeaDraft,
    InspirationOperator,
    OperatorOutcome,
    SnapshotEvidenceReference,
)
from experience_hub.inspiration.validation import ValidatedOperatorBatch

BASE_MECHANISM = "P8knKXFMQJfIGud0rUOpSyExZtgR7CTmcv1zjw5H3LB"
AT_THRESHOLD_SIDE = "P8knKXFMQJfIGud0rUOpSy@E#ZtgR7CTmcv1zjw5H3LB"
BELOW_THRESHOLD_SIDE = "IjWca&m#is4R5xZo9CwQSf7qVJPduOlrvn3tN8gp0EX"
BELOW_THRESHOLD_BASE = "IjWcamHis4R5xZo9CwQSf7qVJPduOlrvn3tN8gp0EX"


def _draft(mechanism: str, *, marker: str = "") -> IdeaDraft:
    suffix = f" {marker}" if marker else ""
    return IdeaDraft(
        title=f"Idea{suffix}",
        hypothesis=f"Hypothesis{suffix}",
        mechanism=mechanism,
        predictions=(f"Prediction{suffix}",),
        falsifiers=(f"Falsifier{suffix}",),
        assumptions=(f"Assumption{suffix}",),
        proposed_test=f"Test{suffix}",
        evidence=(
            SnapshotEvidenceReference(
                id=UUID("00000000-0000-0000-0000-000000002001"),
                stable_evidence_key="a" * 64,
            ),
        ),
    )


def _batch(
    operator: InspirationOperator,
    *ideas: IdeaDraft,
    error_code: OperatorFailureCode | None = None,
) -> ValidatedOperatorBatch:
    return ValidatedOperatorBatch(
        operator=operator,
        ideas=tuple(ideas),
        error_code=error_code,
        output_tokens_consumed=11,
    )


def _outcome(
    result: RunDeduplicationResult,
    operator: InspirationOperator,
):
    return next(
        outcome for outcome in result.outcomes if outcome.operator is operator
    )


def test_exact_duplicate_retains_only_the_earliest_ordinal() -> None:
    first = _draft("same normalized mechanism", marker="first")
    second = _draft("same normalized mechanism", marker="second")

    result = deduplicate_run_batches(
        (_batch(InspirationOperator.CAUSAL_GAP, first, second),)
    )

    assert result.ideas == (
        RetainedIdea.from_draft(
            operator=InspirationOperator.CAUSAL_GAP,
            ordinal=1,
            draft=first,
        ),
    )
    assert result.outcomes[0].succeeded is True
    assert result.outcomes[0].persisted_ideas == 1
    assert result.outcomes[0].duplicate_count == 1


def test_near_duplicate_threshold_is_inclusive_and_below_is_retained() -> None:
    at_threshold = mechanism_similarity(BASE_MECHANISM, AT_THRESHOLD_SIDE)
    immediately_below = mechanism_similarity(
        BELOW_THRESHOLD_BASE,
        BELOW_THRESHOLD_SIDE,
    )
    assert at_threshold == NEAR_DUPLICATE_THRESHOLD == 0.82
    assert immediately_below < NEAR_DUPLICATE_THRESHOLD

    result = deduplicate_run_batches(
        (
            _batch(
                InspirationOperator.DISTANT_ANALOGY,
                _draft(BELOW_THRESHOLD_SIDE, marker="below"),
            ),
            _batch(
                InspirationOperator.COUNTERFACTUAL,
                _draft(AT_THRESHOLD_SIDE, marker="threshold"),
            ),
            _batch(
                InspirationOperator.CAUSAL_GAP,
                _draft(BASE_MECHANISM, marker="base"),
                _draft(BELOW_THRESHOLD_BASE, marker="below-base"),
            ),
        )
    )

    assert tuple(idea.operator for idea in result.ideas) == (
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.DISTANT_ANALOGY,
    )
    assert _outcome(
        result,
        InspirationOperator.COUNTERFACTUAL,
    ).error_code == OperatorFailureCode.NO_VALID_BRANCHES
    assert _outcome(
        result,
        InspirationOperator.COUNTERFACTUAL,
    ).duplicate_count == 1
    assert _outcome(
        result,
        InspirationOperator.DISTANT_ANALOGY,
    ).succeeded


def test_operator_with_every_branch_discarded_never_completes_empty() -> None:
    result = deduplicate_run_batches(
        (
            _batch(
                InspirationOperator.CAUSAL_GAP,
                _draft("shared mechanism"),
            ),
            _batch(
                InspirationOperator.COUNTERFACTUAL,
                _draft("shared mechanism"),
            ),
        )
    )

    failed = _outcome(result, InspirationOperator.COUNTERFACTUAL)
    assert failed.succeeded is False
    assert failed.persisted_ideas == 0
    assert failed.error_code == OperatorFailureCode.NO_VALID_BRANCHES
    assert all(
        outcome.persisted_ideas >= 1
        for outcome in result.outcomes
        if outcome.succeeded
    )


def test_validation_failure_is_preserved_without_raw_details() -> None:
    result = deduplicate_run_batches(
        (
            _batch(
                InspirationOperator.CAUSAL_GAP,
                error_code=OperatorFailureCode.INVALID_EVIDENCE_REFERENCE,
            ),
        )
    )

    assert result.ideas == ()
    assert result.outcomes[0].model_dump(mode="json") == {
        "operator": "causal_gap",
        "succeeded": False,
        "persisted_ideas": 0,
        "duplicate_count": 0,
        "error_code": "invalid_evidence_reference",
        "output_tokens_consumed": 11,
    }


def test_operator_and_run_caps_report_every_discarded_branch() -> None:
    operators = (
        InspirationOperator.DISTANT_ANALOGY,
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.COUNTERFACTUAL,
    )
    batches = tuple(
        _batch(
            operator,
            *tuple(
                _draft(
                    chr(0x4E00 + operator_index * 5 + ordinal) * 20,
                    marker=f"{operator.value}-{ordinal}",
                )
                for ordinal in range(5)
            ),
        )
        for operator_index, operator in enumerate(operators)
    )

    result = deduplicate_run_batches(batches)

    assert len(result.ideas) == MAX_IDEAS_PER_RUN == 9
    assert all(
        outcome.persisted_ideas == MAX_IDEAS_PER_OPERATOR == 3
        for outcome in result.outcomes
    )
    assert all(outcome.duplicate_count == 2 for outcome in result.outcomes)
    assert all(1 <= idea.ordinal <= 3 for idea in result.ideas)
    assert tuple(idea.operator for idea in result.ideas) == (
        (InspirationOperator.CAUSAL_GAP,) * 3
        + (InspirationOperator.COUNTERFACTUAL,) * 3
        + (InspirationOperator.DISTANT_ANALOGY,) * 3
    )


def test_repeated_operator_batch_is_rejected() -> None:
    first = _batch(InspirationOperator.CAUSAL_GAP, _draft("first mechanism"))
    second = _batch(InspirationOperator.CAUSAL_GAP, _draft("second mechanism"))

    try:
        deduplicate_run_batches((first, second))
    except ValueError as error:
        assert str(error) == "operator batches must not repeat an operator"
    else:
        raise AssertionError("duplicate operator batch was accepted")


def test_result_rejects_noncanonical_idea_order() -> None:
    result = deduplicate_run_batches(
        (
            _batch(
                InspirationOperator.CAUSAL_GAP,
                _draft("first distinct mechanism"),
            ),
            _batch(
                InspirationOperator.COUNTERFACTUAL,
                _draft("second distinct mechanism"),
            ),
        )
    )

    with pytest.raises(ValueError, match="canonical operator"):
        RunDeduplicationResult(
            ideas=tuple(reversed(result.ideas)),
            outcomes=result.outcomes,
        )


def test_operator_outcome_rejects_unsanitized_or_inconsistent_failure() -> None:
    with pytest.raises(ValueError):
        OperatorOutcome(
            operator=InspirationOperator.CAUSAL_GAP,
            succeeded=False,
            persisted_ideas=0,
            error_code="secret provider traceback",
        )
    with pytest.raises(ValueError, match="cannot persist ideas"):
        OperatorOutcome(
            operator=InspirationOperator.CAUSAL_GAP,
            succeeded=False,
            persisted_ideas=1,
            error_code=OperatorFailureCode.GENERATOR_ERROR,
        )


def test_operator_outcome_model_copy_is_revalidated_at_nested_boundaries() -> None:
    valid = OperatorOutcome(
        operator=InspirationOperator.CAUSAL_GAP,
        succeeded=False,
        persisted_ideas=0,
        error_code=OperatorFailureCode.GENERATOR_ERROR,
    )
    forged = valid.model_copy(update={"error_code": "secret traceback"})

    with pytest.raises(ValueError):
        OperatorOutcome.model_validate(forged, strict=True)
    with pytest.raises(ValueError, match="structurally valid"):
        RunDeduplicationResult(ideas=(), outcomes=(forged,))
