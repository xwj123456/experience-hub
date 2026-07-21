"""Reproducible inspiration generation from frozen evidence only."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from experience_hub.experiences.models import (
    MAX_MECHANISM_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
)
from experience_hub.inspiration.commands import (
    MAX_CONTEXT_CHARACTERS,
    MAX_GOAL_CHARACTERS,
)
from experience_hub.inspiration.generators.base import (
    GeneratorResult,
    OperatorFailureCode,
)
from experience_hub.inspiration.models import (
    MAX_IDEA_TEXT_CHARACTERS,
    MAX_SNAPSHOT_ITEMS,
    IdeaDraft,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.operators import (
    AnalogyPair,
    CausalPair,
    CounterfactualCandidate,
    causal_gap_pairs,
    counterfactual_candidates,
    distant_analogy_pairs,
)
from experience_hub.retrieval.tokenizer import normalize_text

_DISPLAY_GOAL_CHARACTERS = 512
_DISPLAY_CONTEXT_CHARACTERS = 1_000
_DISPLAY_TITLE_SUMMARY_CHARACTERS = 440
_DISPLAY_BODY_SUMMARY_CHARACTERS = 640
_DISPLAY_MECHANISM_CHARACTERS = 700
_DISPLAY_APPLICABILITY_CHARACTERS = 800
_DISPLAY_TITLE_APPLICABILITY_CHARACTERS = 900
_DISPLAY_SHARED_TERMS_CHARACTERS = 400


def _validate_text(
    name: str,
    value: Any,
    *,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if value != value.strip():
        raise ValueError(f"{name} must already be trimmed")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum:,} characters")
    return value


def _validate_frozen_items(value: Any) -> tuple[SnapshotItem, ...]:
    if not isinstance(value, tuple):
        raise ValueError("frozen_items must be an immutable tuple")
    if len(value) > MAX_SNAPSHOT_ITEMS:
        raise ValueError("frozen_items may contain at most 12 items")
    if any(not isinstance(item, SnapshotItem) for item in value):
        raise ValueError("frozen_items must contain only SnapshotItem values")
    items: tuple[SnapshotItem, ...] = value
    if tuple(item.rank for item in items) != tuple(range(1, len(items) + 1)):
        raise ValueError("frozen_items must retain contiguous canonical rank order")
    if len({item.snapshot_item_id for item in items}) != len(items):
        raise ValueError("frozen_items must not repeat snapshot identities")
    if items and any(item.run_id != items[0].run_id for item in items):
        raise ValueError("frozen_items must belong to one inspiration run")
    return items


def _validate_branch_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("branch_limit must be a strict integer from 1 to 3")
    if not 1 <= value <= 3:
        raise ValueError("branch_limit must be from 1 to 3")
    return int(value)


def _validate_output_token_limit(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 1_200
    ):
        raise ValueError(
            "output_token_limit must be a strict integer from 0 to 1200"
        )
    return int(value)


def _abbreviate(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    retained = maximum - 1
    prefix = (retained + 1) // 2
    suffix = retained - prefix
    return f"{value[:prefix]}…{value[-suffix:]}"


def _reference(item: SnapshotItem) -> SnapshotEvidenceReference:
    return SnapshotEvidenceReference(
        id=item.snapshot_item_id,
        stable_evidence_key=item.stable_evidence_key,
    )


def _conflict_conditions(
    left: SnapshotItem,
    right: SnapshotItem,
    *,
    mechanism_left: str,
    mechanism_right: str,
) -> tuple[str, str]:
    left_applicability = left.applicability[0] if left.applicability else ""
    right_applicability = right.applicability[0] if right.applicability else ""
    if (
        left_applicability
        and right_applicability
        and normalize_text(left_applicability)
        != normalize_text(right_applicability)
    ):
        retained_left = _abbreviate(
            left_applicability,
            _DISPLAY_APPLICABILITY_CHARACTERS,
        )
        retained_right = _abbreviate(
            right_applicability,
            _DISPLAY_APPLICABILITY_CHARACTERS,
        )
        if normalize_text(retained_left) != normalize_text(retained_right):
            return retained_left, retained_right
    if normalize_text(mechanism_left) != normalize_text(mechanism_right):
        return (
            f'activate "{mechanism_left}" while suppressing "{mechanism_right}"',
            f'activate "{mechanism_right}" while suppressing "{mechanism_left}"',
        )
    frozen_falsifier = next(
        (
            value
            for value in (*left.falsifiers, *right.falsifiers)
            if normalize_text(value)
        ),
        None,
    )
    if frozen_falsifier is not None:
        retained = _abbreviate(
            frozen_falsifier,
            _DISPLAY_APPLICABILITY_CHARACTERS,
        )
        return (
            f'apply the preregistered stress "{retained}"',
            f'withhold the preregistered stress "{retained}"',
        )
    return (
        f'activate the frozen mechanism "{mechanism_left}"',
        f'suppress the frozen mechanism "{mechanism_left}"',
    )


def _idea(
    *,
    title: str,
    hypothesis: str,
    mechanism: str,
    predictions: Iterable[str],
    falsifiers: Iterable[str],
    assumptions: Iterable[str],
    proposed_test: str,
    evidence: tuple[SnapshotEvidenceReference, ...],
) -> IdeaDraft:
    return IdeaDraft(
        title=_abbreviate(title, MAX_SUMMARY_CHARACTERS),
        hypothesis=_abbreviate(hypothesis, MAX_IDEA_TEXT_CHARACTERS),
        mechanism=_abbreviate(mechanism, MAX_MECHANISM_CHARACTERS),
        predictions=tuple(
            _abbreviate(value, MAX_IDEA_TEXT_CHARACTERS) for value in predictions
        ),
        falsifiers=tuple(
            _abbreviate(value, MAX_IDEA_TEXT_CHARACTERS) for value in falsifiers
        ),
        assumptions=tuple(
            _abbreviate(value, MAX_IDEA_TEXT_CHARACTERS) for value in assumptions
        ),
        proposed_test=_abbreviate(proposed_test, MAX_IDEA_TEXT_CHARACTERS),
        evidence=evidence,
    )


def _causal_idea(
    pair: CausalPair,
    *,
    goal: str,
    context: str,
) -> IdeaDraft:
    left = pair.left
    right = pair.right
    retained_goal = _abbreviate(goal, _DISPLAY_GOAL_CHARACTERS)
    title_left = _abbreviate(
        left.summary,
        _DISPLAY_TITLE_SUMMARY_CHARACTERS,
    )
    title_right = _abbreviate(
        right.summary,
        _DISPLAY_TITLE_SUMMARY_CHARACTERS,
    )
    body_left = _abbreviate(
        left.summary,
        _DISPLAY_BODY_SUMMARY_CHARACTERS,
    )
    body_right = _abbreviate(
        right.summary,
        _DISPLAY_BODY_SUMMARY_CHARACTERS,
    )
    mechanism_left = _abbreviate(
        left.mechanism,
        _DISPLAY_MECHANISM_CHARACTERS,
    )
    mechanism_right = _abbreviate(
        right.mechanism,
        _DISPLAY_MECHANISM_CHARACTERS,
    )
    retained_context = _abbreviate(
        context,
        _DISPLAY_CONTEXT_CHARACTERS,
    )

    if pair.conflict_basis is not None:
        left_condition, right_condition = _conflict_conditions(
            left,
            right,
            mechanism_left=mechanism_left,
            mechanism_right=mechanism_right,
        )
        assumptions = [
            (
                "The conflict is a testable lexical disagreement, not proof "
                "that either frozen claim is true."
            ),
            pair.conflict_basis.value,
        ]
        if retained_context:
            assumptions.append(f"Context remains fixed: {retained_context}")
        return _idea(
            title=f"Causal conflict: {title_left} <-> {title_right}",
            hypothesis=(
                f'For goal "{retained_goal}", the frozen claims "{body_left}" '
                f'and "{body_right}" disagree under the '
                f"{pair.conflict_basis.value} signal; one controlled boundary "
                "condition must determine which claim holds."
            ),
            mechanism=(
                f'Resolve the contradiction between "{mechanism_left}" and '
                f'"{mechanism_right}" by isolating one measurable boundary '
                "condition."
            ),
            predictions=(
                f'Under the preregistered condition "{left_condition}", the '
                f'signed contrast favors claim "{body_left}" with mechanism '
                f'"{mechanism_left}" over the competing claim.',
                f'Under the contrasting condition "{right_condition}", the '
                f'signed contrast favors claim "{body_right}" with mechanism '
                f'"{mechanism_right}" over the competing claim.',
            ),
            falsifiers=(
                "The preregistered signed contrast does not reverse direction "
                "between the two named boundary conditions.",
            ),
            assumptions=assumptions,
            proposed_test=(
                f'Hold common inputs constant, compare matched trials under '
                f'"{left_condition}" and "{right_condition}", operationalize '
                "both mechanism-specific claims before measurement, and reject "
                "the hypothesis unless their signed contrast reverses."
            ),
            evidence=(_reference(left), _reference(right)),
        )

    assumptions = [
        "The two frozen observations describe comparable stages of the same goal."
    ]
    if context:
        assumptions.append(f"Context remains fixed: {retained_context}")
    return _idea(
        title=f"Causal bridge: {title_left} -> {title_right}",
        hypothesis=(
            f'For goal "{retained_goal}", "{body_left}" and "{body_right}" are '
            "linked by an unobserved transition controlled by the change from "
            f'"{mechanism_left}" to "{mechanism_right}".'
        ),
        mechanism=(
            f'Bridge the transition between "{mechanism_left}" and '
            f'"{mechanism_right}" through one measurable intermediate state.'
        ),
        predictions=(
            f'The intermediate state changes after "{body_left}" and '
            f'before "{body_right}".',
            "Intervening on the intermediate state changes the probability of "
            f'"{body_right}".',
        ),
        falsifiers=(
            "No measurable intermediate improves prediction of "
            f'"{body_right}" beyond either frozen observation alone.',
        ),
        assumptions=assumptions,
        proposed_test=(
            "Measure candidate intermediate states between the two observations, "
            "then intervene on the best predictor and compare against both "
            "frozen-source baselines."
        ),
        evidence=(_reference(left), _reference(right)),
    )


def _counterfactual_idea(
    candidate: CounterfactualCandidate,
    *,
    goal: str,
) -> IdeaDraft:
    item = candidate.item
    retained_goal = _abbreviate(goal, _DISPLAY_GOAL_CHARACTERS)
    applicability = _abbreviate(
        candidate.applicability,
        _DISPLAY_APPLICABILITY_CHARACTERS,
    )
    title_applicability = _abbreviate(
        candidate.applicability,
        _DISPLAY_TITLE_APPLICABILITY_CHARACTERS,
    )
    summary = _abbreviate(
        item.summary,
        _DISPLAY_BODY_SUMMARY_CHARACTERS,
    )
    mechanism = _abbreviate(
        item.mechanism,
        _DISPLAY_MECHANISM_CHARACTERS,
    )
    inverted = f"it is not true that {applicability}"
    return _idea(
        title=f"Counterfactual: {title_applicability}",
        hypothesis=(
            f'For goal "{retained_goal}", if {inverted}, the outcome '
            f'"{summary}" '
            "should change in a way not explained by the original assumption."
        ),
        mechanism=(
            f'Invert the applicability assumption "{applicability}" while '
            f'holding the frozen mechanism "{mechanism}" fixed.'
        ),
        predictions=(
            "Under the inverted assumption, the observed outcome differs from "
            f'"{summary}".',
            "Restoring the original assumption restores the prior outcome.",
        ),
        falsifiers=(
            "The outcome remains unchanged across the original and inverted "
            "assumption.",
        ),
        assumptions=(
            inverted,
            "All non-target conditions in the frozen evidence remain fixed.",
        ),
        proposed_test=(
            "Run matched trials with the applicability assumption present and "
            "inverted; compare the outcome and restore it in a crossover trial."
        ),
        evidence=(_reference(item),),
    )


def _analogy_idea(
    pair: AnalogyPair,
    *,
    goal: str,
    context: str,
) -> IdeaDraft:
    left = pair.left
    right = pair.right
    retained_goal = _abbreviate(goal, _DISPLAY_GOAL_CHARACTERS)
    title_left = _abbreviate(
        left.summary,
        _DISPLAY_TITLE_SUMMARY_CHARACTERS,
    )
    title_right = _abbreviate(
        right.summary,
        _DISPLAY_TITLE_SUMMARY_CHARACTERS,
    )
    body_left = _abbreviate(
        left.summary,
        _DISPLAY_BODY_SUMMARY_CHARACTERS,
    )
    body_right = _abbreviate(
        right.summary,
        _DISPLAY_BODY_SUMMARY_CHARACTERS,
    )
    mechanism_left = _abbreviate(
        left.mechanism,
        _DISPLAY_MECHANISM_CHARACTERS,
    )
    mechanism_right = _abbreviate(
        right.mechanism,
        _DISPLAY_MECHANISM_CHARACTERS,
    )
    retained_context = _abbreviate(
        context,
        _DISPLAY_CONTEXT_CHARACTERS,
    )
    shared = _abbreviate(
        ", ".join(pair.shared_terms),
        _DISPLAY_SHARED_TERMS_CHARACTERS,
    )
    bracketed = f"[{shared}]"
    assumptions = [
        "Mapping limit: only the shared mechanism terms transfer; actors, scale, "
        "and boundary conditions do not."
    ]
    if retained_context:
        assumptions.append(f"Context remains fixed: {retained_context}")
    return _idea(
        title=f"Distant analogy: {title_left} <-> {title_right}",
        hypothesis=(
            f'For goal "{retained_goal}", the shared mechanism terms '
            f'{bracketed} transfer from "{body_left}" to "{body_right}" despite low '
            "lexical overlap."
        ),
        mechanism=(
            f'Map the shared mechanism {bracketed} from "{mechanism_left}" '
            f'onto "{mechanism_right}" without assuming the surrounding '
            "domains are equivalent."
        ),
        predictions=(
            f"A perturbation of {bracketed} produces directionally similar "
            "changes in both evidence domains.",
            "Domain-specific variables explain residual differences after the "
            "shared mechanism is controlled.",
        ),
        falsifiers=(
            f"Perturbing {bracketed} affects only one evidence domain or "
            "produces opposite effects.",
        ),
        assumptions=assumptions,
        proposed_test=(
            "Apply matched perturbations to the shared mechanism in both domains "
            "and compare normalized response shapes while recording "
            "domain-specific failures."
        ),
        evidence=(_reference(left), _reference(right)),
    )


class DeterministicIdeaGenerator:
    """Generate inspectable branches using only one frozen evidence boundary."""

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {}

    @property
    def reserves_output_tokens(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        retained_goal = _validate_text(
            "goal",
            goal,
            maximum=MAX_GOAL_CHARACTERS,
            allow_empty=False,
        )
        retained_context = _validate_text(
            "context",
            context,
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )
        items = _validate_frozen_items(frozen_items)
        if not isinstance(operator, InspirationOperator):
            raise ValueError("operator must be an InspirationOperator")
        limit = _validate_branch_limit(branch_limit)
        _validate_output_token_limit(output_token_limit)

        if operator is InspirationOperator.CAUSAL_GAP:
            ideas = tuple(
                _causal_idea(
                    pair,
                    goal=retained_goal,
                    context=retained_context,
                )
                for pair in causal_gap_pairs(items)[:limit]
            )
        elif operator is InspirationOperator.COUNTERFACTUAL:
            ideas = tuple(
                _counterfactual_idea(candidate, goal=retained_goal)
                for candidate in counterfactual_candidates(items)[:limit]
            )
        else:
            ideas = tuple(
                _analogy_idea(
                    pair,
                    goal=retained_goal,
                    context=retained_context,
                )
                for pair in distant_analogy_pairs(items)[:limit]
            )

        if not ideas:
            return GeneratorResult(
                ideas=(),
                error_code=OperatorFailureCode.INSUFFICIENT_EVIDENCE,
                output_tokens_consumed=0,
            )
        return GeneratorResult(
            ideas=ideas,
            error_code=None,
            output_tokens_consumed=0,
        )


__all__ = ["DeterministicIdeaGenerator"]
