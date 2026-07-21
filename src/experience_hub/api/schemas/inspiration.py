"""Strict HTTP contracts for bounded inspiration and idea decisions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, StructuredReason
from experience_hub.inspiration import (
    INSPIRATION_OPERATOR_ORDER,
    MAX_CONTEXT_CHARACTERS,
    MAX_GOAL_CHARACTERS,
    AdoptIdea,
    ArchiveIdea,
    EvaluationVerdict,
    ExperienceVersionEvidenceReference,
    GeneratorKind,
    IdeaEvaluation,
    InspirationOperator,
    RejectIdea,
    SnapshotEvidenceReference,
    StartInspirationRun,
)
from experience_hub.retrieval import RetrievalMode

UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})\Z"
)
_CANONICAL_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")


class _RequestModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _trimmed_text(
    value: Any,
    *,
    field_name: str,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    retained = value.strip()
    try:
        retained.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must contain valid Unicode") from error
    if not allow_empty and not retained:
        raise ValueError(f"{field_name} must not be blank")
    if len(retained) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum:,} characters")
    return retained


class StartInspirationRunRequest(_RequestModel):
    goal: str
    context: str = ""
    mode: RetrievalMode = RetrievalMode.ASSOCIATIVE
    generator: GeneratorKind = GeneratorKind.DETERMINISTIC
    operators: tuple[InspirationOperator, ...] = INSPIRATION_OPERATOR_ORDER
    include_inbox: StrictBool = False
    branches_per_operator: Annotated[StrictInt, Field(ge=1, le=3)] = 3
    output_tokens_per_operator: Annotated[
        StrictInt,
        Field(ge=1, le=1_200),
    ] = 1_200
    total_output_tokens: Annotated[
        StrictInt,
        Field(ge=1, le=3_600),
    ] = 3_600
    operator_timeout_seconds: Annotated[
        StrictInt,
        Field(ge=1, le=30),
    ] = 30
    global_timeout_seconds: Annotated[
        StrictInt,
        Field(ge=1, le=90),
    ] = 90

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal(cls, value: Any) -> str:
        return _trimmed_text(
            value,
            field_name="goal",
            maximum=MAX_GOAL_CHARACTERS,
            allow_empty=False,
        )

    @field_validator("context", mode="before")
    @classmethod
    def normalize_context(cls, value: Any) -> str:
        return _trimmed_text(
            value,
            field_name="context",
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )

    @field_validator("operators", mode="before")
    @classmethod
    def require_operator_list(cls, values: Any) -> Any:
        if not isinstance(values, (list, tuple)):
            raise ValueError("operators must be an array")
        if not 1 <= len(values) <= len(INSPIRATION_OPERATOR_ORDER):
            raise ValueError("operators must contain one to three values")
        return values

    @field_validator("operators")
    @classmethod
    def canonicalize_operators(
        cls,
        values: tuple[InspirationOperator, ...],
    ) -> tuple[InspirationOperator, ...]:
        if len(values) != len(set(values)):
            raise ValueError("operators must not contain duplicates")
        return tuple(
            operator for operator in INSPIRATION_OPERATOR_ORDER if operator in values
        )

    @model_validator(mode="after")
    def validate_timeout_budget(self) -> Self:
        if self.global_timeout_seconds < self.operator_timeout_seconds:
            raise ValueError(
                "global_timeout_seconds must not be less than operator_timeout_seconds"
            )
        return self

    def to_command(self, *, owner_agent_id: UUID) -> StartInspirationRun:
        return StartInspirationRun(
            owner_agent_id=owner_agent_id,
            goal=self.goal,
            context=self.context,
            mode=self.mode,
            generator=self.generator,
            operators=self.operators,
            include_inbox=self.include_inbox,
            branches_per_operator=self.branches_per_operator,
            output_tokens_per_operator=self.output_tokens_per_operator,
            total_output_tokens=self.total_output_tokens,
            operator_timeout_seconds=self.operator_timeout_seconds,
            global_timeout_seconds=self.global_timeout_seconds,
        )

    def command_body(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "context": self.context,
            "mode": self.mode.value,
            "generator": self.generator.value,
            "operators": tuple(value.value for value in self.operators),
            "include_inbox": self.include_inbox,
            "branches_per_operator": self.branches_per_operator,
            "output_tokens_per_operator": self.output_tokens_per_operator,
            "total_output_tokens": self.total_output_tokens,
            "operator_timeout_seconds": self.operator_timeout_seconds,
            "global_timeout_seconds": self.global_timeout_seconds,
        }


class IdeaListQuery(StrictModel):
    limit: Annotated[int, Field(ge=1, le=100)] = 100
    cursor: str | None = None

    @field_validator("limit", mode="before")
    @classmethod
    def require_canonical_limit(cls, value: Any) -> int:
        if type(value) is int:
            return value
        if isinstance(value, str) and _CANONICAL_POSITIVE_INTEGER.fullmatch(value):
            return int(value)
        raise ValueError("limit must be a canonical positive integer")


class AdoptIdeaRequest(_RequestModel):
    importance: UnitFloat = 0.40
    confidence: UnitFloat = 0.35

    def to_command(
        self,
        *,
        owner_agent_id: UUID,
        idea_id: UUID,
    ) -> AdoptIdea:
        return AdoptIdea(
            owner_agent_id=owner_agent_id,
            idea_id=idea_id,
            importance=self.importance,
            confidence=self.confidence,
        )


class IdeaReasonRequest(_RequestModel):
    reason: str

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> str:
        return _trimmed_text(
            value,
            field_name="reason",
            maximum=2_000,
            allow_empty=False,
        )

    def to_reason(self) -> StructuredReason:
        return StructuredReason.from_user_text(self.reason)

    def to_reject(
        self,
        *,
        owner_agent_id: UUID,
        idea_id: UUID,
    ) -> RejectIdea:
        return RejectIdea(
            owner_agent_id=owner_agent_id,
            idea_id=idea_id,
            reason=self.to_reason(),
        )

    def to_archive(
        self,
        *,
        owner_agent_id: UUID,
        idea_id: UUID,
    ) -> ArchiveIdea:
        return ArchiveIdea(
            owner_agent_id=owner_agent_id,
            idea_id=idea_id,
            reason=self.to_reason(),
        )


class SnapshotEvidenceRequest(_RequestModel):
    type: Literal["snapshot_item"]
    id: UUID
    stable_evidence_key: str

    @field_validator("stable_evidence_key")
    @classmethod
    def require_stable_key(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(
                "stable_evidence_key must be a lowercase SHA-256 hex digest"
            )
        return value

    def to_domain(self) -> SnapshotEvidenceReference:
        return SnapshotEvidenceReference(
            id=self.id,
            stable_evidence_key=self.stable_evidence_key,
        )


class ExperienceVersionEvidenceRequest(_RequestModel):
    type: Literal["experience_version"]
    id: UUID

    def to_domain(self) -> ExperienceVersionEvidenceReference:
        return ExperienceVersionEvidenceReference(id=self.id)


EvaluationEvidenceRequest = Annotated[
    SnapshotEvidenceRequest | ExperienceVersionEvidenceRequest,
    Field(discriminator="type"),
]


class EvaluateIdeaRequest(_RequestModel):
    verdict: EvaluationVerdict
    evidence: tuple[EvaluationEvidenceRequest, ...]
    evaluated_at: datetime
    reason: str | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def require_evidence_list(cls, values: Any) -> Any:
        if not isinstance(values, (list, tuple)):
            raise ValueError("evidence must be an array")
        if not 1 <= len(values) <= 32:
            raise ValueError("evidence must contain one to 32 references")
        return values

    @field_validator("evidence")
    @classmethod
    def require_unique_evidence(
        cls,
        values: tuple[EvaluationEvidenceRequest, ...],
    ) -> tuple[EvaluationEvidenceRequest, ...]:
        identities = tuple(
            (
                value.type,
                value.id,
                (
                    value.stable_evidence_key
                    if isinstance(value, SnapshotEvidenceRequest)
                    else None
                ),
            )
            for value in values
        )
        if len(identities) != len(set(identities)):
            raise ValueError("evidence references must not repeat")
        snapshot_keys = tuple(
            value.stable_evidence_key
            for value in values
            if isinstance(value, SnapshotEvidenceRequest)
        )
        if len(snapshot_keys) != len(set(snapshot_keys)):
            raise ValueError("snapshot evidence keys must not repeat")
        return values

    @field_validator("evaluated_at", mode="before")
    @classmethod
    def require_evaluated_at_shape(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and _RFC3339_TIMESTAMP.fullmatch(value):
            return value
        raise ValueError("evaluated_at must be an RFC 3339 timestamp")

    @field_validator("evaluated_at")
    @classmethod
    def normalize_evaluated_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_optional_reason(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _trimmed_text(
            value,
            field_name="reason",
            maximum=2_000,
            allow_empty=False,
        )

    def to_evaluation(
        self,
        *,
        evaluator_agent_id: UUID,
        idea_id: UUID,
    ) -> IdeaEvaluation:
        return IdeaEvaluation(
            evaluator_agent_id=evaluator_agent_id,
            idea_id=idea_id,
            verdict=self.verdict,
            reason=(
                None
                if self.reason is None
                else StructuredReason.from_user_text(self.reason)
            ),
            evidence=tuple(value.to_domain() for value in self.evidence),
            evaluated_at=self.evaluated_at,
        )


__all__ = [
    "AdoptIdeaRequest",
    "EvaluateIdeaRequest",
    "IdeaListQuery",
    "IdeaReasonRequest",
    "StartInspirationRunRequest",
]
