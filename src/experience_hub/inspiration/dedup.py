"""Deterministic run-local mechanism deduplication."""

from __future__ import annotations

from dataclasses import dataclass

from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.hashing import (
    NEAR_DUPLICATE_THRESHOLD,
    hash_idea_content,
    hash_mechanism,
    mechanism_similarity,
)
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    IdeaDraft,
    InspirationOperator,
    OperatorOutcome,
)
from experience_hub.inspiration.validation import ValidatedOperatorBatch

MAX_IDEAS_PER_OPERATOR = 3
MAX_IDEAS_PER_RUN = 9
_OPERATOR_ORDER = {
    operator: index for index, operator in enumerate(INSPIRATION_OPERATOR_ORDER)
}


@dataclass(frozen=True, slots=True)
class RetainedIdea:
    """One branch retained with its contiguous persisted ordinal."""

    operator: InspirationOperator
    ordinal: int
    draft: IdeaDraft
    idea_content_hash: str
    mechanism_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.operator, InspirationOperator):
            raise TypeError("operator must be an InspirationOperator")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or not 1 <= self.ordinal <= MAX_IDEAS_PER_OPERATOR
        ):
            raise ValueError("ordinal must be a strict integer from one to three")
        if not isinstance(self.draft, IdeaDraft):
            raise TypeError("draft must be an IdeaDraft")
        if self.idea_content_hash != hash_idea_content(self.draft):
            raise ValueError("idea_content_hash must match the retained draft")
        if self.mechanism_hash != hash_mechanism(self.draft.mechanism):
            raise ValueError("mechanism_hash must match the retained draft")

    @classmethod
    def from_draft(
        cls,
        *,
        operator: InspirationOperator,
        ordinal: int,
        draft: IdeaDraft,
    ) -> RetainedIdea:
        """Construct a hash-locked retained branch."""
        return cls(
            operator=operator,
            ordinal=ordinal,
            draft=draft,
            idea_content_hash=hash_idea_content(draft),
            mechanism_hash=hash_mechanism(draft.mechanism),
        )


@dataclass(frozen=True, slots=True)
class RunDeduplicationResult:
    """Canonical retained branches and sanitized per-operator outcomes."""

    ideas: tuple[RetainedIdea, ...]
    outcomes: tuple[OperatorOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ideas, tuple) or any(
            not isinstance(idea, RetainedIdea) for idea in self.ideas
        ):
            raise TypeError("ideas must be an immutable RetainedIdea tuple")
        if not isinstance(self.outcomes, tuple) or any(
            not isinstance(outcome, OperatorOutcome) for outcome in self.outcomes
        ):
            raise TypeError("outcomes must be an immutable OperatorOutcome tuple")
        try:
            for outcome in self.outcomes:
                OperatorOutcome.model_validate(
                    outcome.model_dump(mode="python", warnings=False),
                    strict=True,
                )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "outcomes must contain structurally valid OperatorOutcome values"
            ) from error
        if len(self.ideas) > MAX_IDEAS_PER_RUN:
            raise ValueError("a run may retain at most nine ideas")
        idea_order = tuple(
            (_OPERATOR_ORDER[idea.operator], idea.ordinal)
            for idea in self.ideas
        )
        if idea_order != tuple(sorted(idea_order)):
            raise ValueError("ideas must follow canonical operator and ordinal order")
        for operator in INSPIRATION_OPERATOR_ORDER:
            retained = tuple(
                idea for idea in self.ideas if idea.operator is operator
            )
            if tuple(idea.ordinal for idea in retained) != tuple(
                range(1, len(retained) + 1)
            ):
                raise ValueError("retained ordinals must be contiguous per operator")
        outcome_operators = tuple(
            outcome.operator for outcome in self.outcomes
        )
        if len(set(outcome_operators)) != len(outcome_operators):
            raise ValueError("outcomes must not repeat an operator")
        if outcome_operators != tuple(
            sorted(outcome_operators, key=_OPERATOR_ORDER.__getitem__)
        ):
            raise ValueError("outcomes must follow canonical operator order")
        outcome_by_operator = {
            outcome.operator: outcome for outcome in self.outcomes
        }
        if any(
            idea.operator not in outcome_by_operator for idea in self.ideas
        ):
            raise ValueError("every retained idea requires an operator outcome")
        for outcome in self.outcomes:
            persisted = sum(
                idea.operator is outcome.operator for idea in self.ideas
            )
            if outcome.persisted_ideas != persisted:
                raise ValueError(
                    "outcome persisted count must match retained ideas"
                )


def _is_duplicate(
    draft: IdeaDraft,
    retained: tuple[RetainedIdea, ...],
) -> bool:
    mechanism_hash = hash_mechanism(draft.mechanism)
    return any(
        mechanism_hash == candidate.mechanism_hash
        or mechanism_similarity(
            draft.mechanism,
            candidate.draft.mechanism,
        )
        >= NEAR_DUPLICATE_THRESHOLD
        for candidate in retained
    )


def _failed_outcome(batch: ValidatedOperatorBatch) -> OperatorOutcome:
    if batch.error_code is None:
        raise ValueError("failed outcome requires a failed validated batch")
    return OperatorOutcome(
        operator=batch.operator,
        succeeded=False,
        persisted_ideas=0,
        error_code=batch.error_code,
        output_tokens_consumed=batch.output_tokens_consumed,
    )


def deduplicate_run_batches(
    batches: tuple[ValidatedOperatorBatch, ...],
) -> RunDeduplicationResult:
    """Retain earliest mechanisms in canonical operator/ordinal order."""
    if not isinstance(batches, tuple) or any(
        not isinstance(batch, ValidatedOperatorBatch) for batch in batches
    ):
        raise TypeError(
            "batches must be an immutable tuple of ValidatedOperatorBatch values"
        )
    if not batches:
        raise ValueError("batches must not be empty")
    if len({batch.operator for batch in batches}) != len(batches):
        raise ValueError("operator batches must not repeat an operator")
    canonical_batches = tuple(
        sorted(batches, key=lambda batch: _OPERATOR_ORDER[batch.operator])
    )

    retained: tuple[RetainedIdea, ...] = ()
    outcomes: list[OperatorOutcome] = []
    for batch in canonical_batches:
        if not batch.succeeded:
            outcomes.append(_failed_outcome(batch))
            continue

        operator_retained: list[RetainedIdea] = []
        discarded = 0
        for draft in batch.ideas:
            if (
                len(operator_retained) >= MAX_IDEAS_PER_OPERATOR
                or len(retained) >= MAX_IDEAS_PER_RUN
                or _is_duplicate(draft, retained)
            ):
                discarded += 1
                continue
            candidate = RetainedIdea.from_draft(
                operator=batch.operator,
                ordinal=len(operator_retained) + 1,
                draft=draft,
            )
            operator_retained.append(candidate)
            retained = (*retained, candidate)

        if not operator_retained:
            outcomes.append(
                OperatorOutcome(
                    operator=batch.operator,
                    succeeded=False,
                    persisted_ideas=0,
                    duplicate_count=discarded,
                    error_code=OperatorFailureCode.NO_VALID_BRANCHES,
                    output_tokens_consumed=batch.output_tokens_consumed,
                )
            )
            continue
        outcomes.append(
            OperatorOutcome(
                operator=batch.operator,
                succeeded=True,
                persisted_ideas=len(operator_retained),
                duplicate_count=discarded,
                output_tokens_consumed=batch.output_tokens_consumed,
            )
        )

    return RunDeduplicationResult(
        ideas=retained,
        outcomes=tuple(outcomes),
    )


__all__ = [
    "MAX_IDEAS_PER_OPERATOR",
    "MAX_IDEAS_PER_RUN",
    "NEAR_DUPLICATE_THRESHOLD",
    "RetainedIdea",
    "RunDeduplicationResult",
    "deduplicate_run_batches",
]
