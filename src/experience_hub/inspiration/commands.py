"""Strict commands for bounded inspiration generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from experience_hub.domain import StructuredReason
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    GeneratorKind,
    InspirationOperator,
)
from experience_hub.retrieval.ranking import RetrievalMode

MAX_GOAL_CHARACTERS = 2_000
MAX_CONTEXT_CHARACTERS = 4_000

type IdeaDecisionReason = StructuredReason | str


def _validate_idea_identity(
    owner_agent_id: UUID,
    idea_id: UUID,
) -> None:
    if not isinstance(owner_agent_id, UUID):
        raise ValueError("owner_agent_id must be a UUID")
    if not isinstance(idea_id, UUID):
        raise ValueError("idea_id must be a UUID")


def _validate_decision_reason(reason: object) -> None:
    if not isinstance(reason, (str, StructuredReason)):
        raise ValueError("reason must be a string or StructuredReason")


def _unit_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return converted


@dataclass(frozen=True, slots=True)
class StartInspirationRun:
    """One fully bounded, canonical request to start a synchronous run."""

    owner_agent_id: UUID
    goal: str
    context: str = ""
    mode: RetrievalMode = RetrievalMode.ASSOCIATIVE
    generator: GeneratorKind = GeneratorKind.DETERMINISTIC
    operators: tuple[InspirationOperator, ...] = INSPIRATION_OPERATOR_ORDER
    include_inbox: bool = False
    branches_per_operator: int = 3
    output_tokens_per_operator: int = 1_200
    total_output_tokens: int = 3_600
    operator_timeout_seconds: int = 30
    global_timeout_seconds: int = 90

    def __post_init__(self) -> None:
        if not isinstance(self.owner_agent_id, UUID):
            raise ValueError("owner_agent_id must be a UUID")
        self._validate_text(
            "goal",
            self.goal,
            maximum=MAX_GOAL_CHARACTERS,
            allow_empty=False,
        )
        self._validate_text(
            "context",
            self.context,
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )
        if not isinstance(self.mode, RetrievalMode):
            raise ValueError("mode must be a RetrievalMode")
        if not isinstance(self.generator, GeneratorKind):
            raise ValueError("generator must be a GeneratorKind")
        self._validate_operators(self.operators)
        if not isinstance(self.include_inbox, bool):
            raise ValueError("include_inbox must be a bool")
        self._validate_integer(
            "branches_per_operator",
            self.branches_per_operator,
            lower=1,
            upper=3,
        )
        self._validate_integer(
            "output_tokens_per_operator",
            self.output_tokens_per_operator,
            lower=1,
            upper=1_200,
        )
        self._validate_integer(
            "total_output_tokens",
            self.total_output_tokens,
            lower=1,
            upper=3_600,
        )
        self._validate_integer(
            "operator_timeout_seconds",
            self.operator_timeout_seconds,
            lower=1,
            upper=30,
        )
        self._validate_integer(
            "global_timeout_seconds",
            self.global_timeout_seconds,
            lower=1,
            upper=90,
        )
        if self.global_timeout_seconds < self.operator_timeout_seconds:
            raise ValueError(
                "global_timeout_seconds must not be less than "
                "operator_timeout_seconds"
            )

    @staticmethod
    def _validate_text(
        name: str,
        value: Any,
        *,
        maximum: int,
        allow_empty: bool,
    ) -> None:
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
            raise ValueError(
                f"{name} must contain at most {maximum:,} characters"
            )

    @staticmethod
    def _validate_operators(values: Any) -> None:
        if not isinstance(values, tuple):
            raise ValueError("operators must be an immutable tuple")
        if not values:
            raise ValueError("operators must not be empty")
        if any(not isinstance(value, InspirationOperator) for value in values):
            raise ValueError(
                "operators must contain only InspirationOperator values"
            )
        if len(values) != len(set(values)):
            raise ValueError("operators must not contain duplicates")
        expected = tuple(
            operator
            for operator in INSPIRATION_OPERATOR_ORDER
            if operator in values
        )
        if values != expected:
            raise ValueError("operators must follow the fixed canonical order")

    @staticmethod
    def _validate_integer(
        name: str,
        value: Any,
        *,
        lower: int,
        upper: int,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise ValueError(
                f"{name} must be an integer between {lower:,} and {upper:,}"
            )


@dataclass(frozen=True, slots=True)
class RejectIdea:
    """Record an owner's terminal rejection of an active or archived idea."""

    owner_agent_id: UUID
    idea_id: UUID
    reason: IdeaDecisionReason

    def __post_init__(self) -> None:
        _validate_idea_identity(self.owner_agent_id, self.idea_id)
        _validate_decision_reason(self.reason)


@dataclass(frozen=True, slots=True)
class ArchiveIdea:
    """Move an active idea out of the owner's working set."""

    owner_agent_id: UUID
    idea_id: UUID
    reason: IdeaDecisionReason

    def __post_init__(self) -> None:
        _validate_idea_identity(self.owner_agent_id, self.idea_id)
        _validate_decision_reason(self.reason)


@dataclass(frozen=True, slots=True)
class AdoptIdea:
    """Adopt one private idea as an owned hypothesis experience."""

    owner_agent_id: UUID
    idea_id: UUID
    importance: float = 0.40
    confidence: float = 0.35

    def __post_init__(self) -> None:
        _validate_idea_identity(self.owner_agent_id, self.idea_id)
        object.__setattr__(
            self,
            "importance",
            _unit_float("importance", self.importance),
        )
        object.__setattr__(
            self,
            "confidence",
            _unit_float("confidence", self.confidence),
        )


__all__ = [
    "AdoptIdea",
    "ArchiveIdea",
    "IdeaDecisionReason",
    "MAX_CONTEXT_CHARACTERS",
    "MAX_GOAL_CHARACTERS",
    "RejectIdea",
    "StartInspirationRun",
]
