"""Immutable values for evidence-grounded inspiration and incubation."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, StructuredReason
from experience_hub.experiences.models import (
    MAX_MECHANISM_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    MAX_VERSION_LIST_ITEMS,
)
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import normalize_text

MAX_SNAPSHOT_ITEMS = 12
MAX_SNAPSHOT_EXCERPT_UTF8_BYTES = 2_048
MAX_SNAPSHOT_UTF8_BYTES = 24_576
MAX_IDEA_TEXT_CHARACTERS = 4_000

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    strict=True,
    revalidate_instances="always",
)


class InspirationOperator(StrEnum):
    """The fixed sequence of supported inspiration transformations."""

    CAUSAL_GAP = "causal_gap"
    COUNTERFACTUAL = "counterfactual"
    DISTANT_ANALOGY = "distant_analogy"


INSPIRATION_OPERATOR_ORDER: tuple[InspirationOperator, ...] = (
    InspirationOperator.CAUSAL_GAP,
    InspirationOperator.COUNTERFACTUAL,
    InspirationOperator.DISTANT_ANALOGY,
)


class GeneratorKind(StrEnum):
    """A selected, explicitly configured idea generator."""

    DETERMINISTIC = "deterministic"
    OPENAI_COMPATIBLE = "openai_compatible"


class InspirationRunStatus(StrEnum):
    """Durable run states exposed by the inspiration protocol."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class IdeaOwnerDecision(StrEnum):
    """An owner's independent decision about one immutable idea."""

    ACTIVE = "active"
    ADOPTED = "adopted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class MechanismMaturity(StrEnum):
    """Evidence maturity of a mechanism cluster."""

    SPECULATIVE = "speculative"
    INCUBATING = "incubating"
    CANDIDATE = "candidate"


# Concise domain names are retained as public aliases for service/query contracts.
RunStatus = InspirationRunStatus
OwnerDecision = IdeaOwnerDecision
Maturity = MechanismMaturity


class EvaluationVerdict(StrEnum):
    """A bounded human or agent evaluation of an idea."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class EvidenceSourceType(StrEnum):
    """The two evidence domains permitted in a frozen snapshot."""

    EXPERIENCE = "experience"
    CAPSULE = "capsule"


class EvidenceSourceState(StrEnum):
    """The visible lifecycle state captured at snapshot time."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    QUARANTINED = "quarantined"


def _require_hash(name: str, value: str) -> str:
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_timestamp(name: str, value: datetime) -> datetime:
    try:
        return require_utc(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timezone-aware datetime") from error


def _require_text(
    name: str,
    value: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return value


def _require_string_tuple(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) > MAX_VERSION_LIST_ITEMS:
        raise ValueError(
            f"{name} may contain at most {MAX_VERSION_LIST_ITEMS} values"
        )
    for value in values:
        _require_text(
            f"{name} value",
            value,
            maximum=MAX_IDEA_TEXT_CHARACTERS,
        )
    return values


def _require_unit_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite number between zero and one")
    return converted


def _validate_evidence_source(
    *,
    source_type: EvidenceSourceType,
    source_state: EvidenceSourceState,
    source_trust: float,
) -> None:
    if source_type is EvidenceSourceType.EXPERIENCE:
        if source_state not in {
            EvidenceSourceState.HOT,
            EvidenceSourceState.WARM,
            EvidenceSourceState.COLD,
        }:
            raise ValueError(
                "experience evidence must use a retrievable temperature"
            )
        return
    if (
        source_state is not EvidenceSourceState.QUARANTINED
        or source_trust != 0.25
    ):
        raise ValueError(
            "capsule evidence must be quarantined with fixed trust 0.25"
        )


class InspirationModel(StrictModel):
    """Base class that rejects coercion and mutation at domain boundaries."""

    model_config = _STRICT_CONFIG


class EvidenceCandidate(InspirationModel):
    """A ranked, read-only candidate before it receives a snapshot identity."""

    source_type: EvidenceSourceType
    source_id: UUID
    source_version_id: UUID
    source_state: EvidenceSourceState
    source_trust: float
    relevance: float
    summary: str
    mechanism: str
    applicability: tuple[str, ...]
    tags: tuple[str, ...]
    falsifiers: tuple[str, ...]
    excerpt: str
    content_hash: str

    @field_validator("source_trust", mode="before")
    @classmethod
    def validate_source_trust(cls, value: Any) -> float:
        return _require_unit_float("source_trust", value)

    @field_validator("relevance", mode="before")
    @classmethod
    def validate_relevance(cls, value: Any) -> float:
        return _require_unit_float("relevance", value)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _require_text(
            "summary",
            value,
            maximum=MAX_SUMMARY_CHARACTERS,
        )

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        retained = _require_text(
            "mechanism",
            value,
            maximum=MAX_MECHANISM_CHARACTERS,
        )
        if not normalize_text(retained):
            raise ValueError("mechanism must contain normalized semantic text")
        return retained

    @field_validator("applicability", "tags", "falsifiers")
    @classmethod
    def validate_metadata(
        cls,
        values: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _require_string_tuple(info.field_name, values, allow_empty=True)

    @field_validator("excerpt")
    @classmethod
    def validate_excerpt(cls, value: str) -> str:
        retained = _require_text(
            "excerpt",
            value,
            maximum=MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
            allow_empty=True,
        )
        if len(retained.encode("utf-8")) > MAX_SNAPSHOT_EXCERPT_UTF8_BYTES:
            raise ValueError("excerpt must be at most 2,048 UTF-8 bytes")
        return retained

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _require_hash("content_hash", value)

    @model_validator(mode="after")
    def validate_source_semantics(self) -> Self:
        _validate_evidence_source(
            source_type=self.source_type,
            source_state=self.source_state,
            source_trust=self.source_trust,
        )
        return self


class SnapshotItem(InspirationModel):
    """One immutable, bounded piece of evidence frozen for a run."""

    snapshot_item_id: UUID
    stable_evidence_key: str
    run_id: UUID
    source_type: EvidenceSourceType
    source_id: UUID
    source_version_id: UUID
    source_state: EvidenceSourceState
    source_trust: float
    rank: Annotated[int, Field(strict=True, ge=1, le=MAX_SNAPSHOT_ITEMS)]
    summary: str
    mechanism: str
    applicability: tuple[str, ...]
    tags: tuple[str, ...]
    falsifiers: tuple[str, ...]
    excerpt: str
    content_hash: str
    captured_at: datetime

    @field_validator("stable_evidence_key", "content_hash")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _require_hash(info.field_name, value)

    @field_validator("source_trust", mode="before")
    @classmethod
    def validate_source_trust(cls, value: Any) -> float:
        return _require_unit_float("source_trust", value)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _require_text(
            "summary",
            value,
            maximum=MAX_SUMMARY_CHARACTERS,
        )

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        retained = _require_text(
            "mechanism",
            value,
            maximum=MAX_MECHANISM_CHARACTERS,
        )
        if not normalize_text(retained):
            raise ValueError("mechanism must contain normalized semantic text")
        return retained

    @field_validator("applicability", "tags", "falsifiers")
    @classmethod
    def validate_metadata(
        cls,
        values: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _require_string_tuple(info.field_name, values, allow_empty=True)

    @field_validator("excerpt")
    @classmethod
    def validate_excerpt(cls, value: str) -> str:
        retained = _require_text(
            "excerpt",
            value,
            maximum=MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
            allow_empty=True,
        )
        if len(retained.encode("utf-8")) > MAX_SNAPSHOT_EXCERPT_UTF8_BYTES:
            raise ValueError("excerpt must be at most 2,048 UTF-8 bytes")
        return retained

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _require_timestamp("captured_at", value)

    @model_validator(mode="after")
    def validate_source_semantics(self) -> Self:
        _validate_evidence_source(
            source_type=self.source_type,
            source_state=self.source_state,
            source_trust=self.source_trust,
        )
        return self


class FrozenSnapshot(InspirationModel):
    """The complete evidence boundary consumed by all run operators."""

    run_id: UUID
    items: tuple[SnapshotItem, ...]
    snapshot_hash: str
    frozen_at: datetime

    @field_validator("items")
    @classmethod
    def validate_items(
        cls,
        values: tuple[SnapshotItem, ...],
    ) -> tuple[SnapshotItem, ...]:
        if len(values) > MAX_SNAPSHOT_ITEMS:
            raise ValueError("items may contain at most 12 snapshot items")
        ranks = tuple(item.rank for item in values)
        if ranks != tuple(range(1, len(values) + 1)):
            raise ValueError("items must use contiguous canonical rank order")
        if len({item.snapshot_item_id for item in values}) != len(values):
            raise ValueError("items must not repeat snapshot identities")
        return values

    @field_validator("snapshot_hash")
    @classmethod
    def validate_snapshot_hash(cls, value: str) -> str:
        return _require_hash("snapshot_hash", value)

    @field_validator("frozen_at")
    @classmethod
    def validate_frozen_at(cls, value: datetime) -> datetime:
        return _require_timestamp("frozen_at", value)

    @model_validator(mode="after")
    def validate_run_identity(self) -> Self:
        if any(item.run_id != self.run_id for item in self.items):
            raise ValueError("every snapshot item must belong to the snapshot run")
        return self


class SnapshotEvidenceReference(InspirationModel):
    """A generation reference resolved against the same run's frozen rows."""

    type: Literal["snapshot_item"] = "snapshot_item"
    id: UUID
    stable_evidence_key: str

    @field_validator("stable_evidence_key")
    @classmethod
    def validate_stable_key(cls, value: str) -> str:
        return _require_hash("stable_evidence_key", value)


class ExperienceVersionEvidenceReference(InspirationModel):
    """An owned experience version cited by a later evaluation."""

    type: Literal["experience_version"] = "experience_version"
    id: UUID


EvaluationEvidenceReference = Annotated[
    SnapshotEvidenceReference | ExperienceVersionEvidenceReference,
    Field(discriminator="type"),
]


class IdeaDraft(InspirationModel):
    """Strict provider-independent idea content before persistence."""

    title: str
    hypothesis: str
    mechanism: str
    predictions: tuple[str, ...]
    falsifiers: tuple[str, ...]
    assumptions: tuple[str, ...]
    proposed_test: str
    evidence: tuple[SnapshotEvidenceReference, ...]

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _require_text(
            "title",
            value,
            maximum=MAX_SUMMARY_CHARACTERS,
        )

    @field_validator("hypothesis", "proposed_test")
    @classmethod
    def validate_long_text(cls, value: str, info: Any) -> str:
        return _require_text(
            info.field_name,
            value,
            maximum=MAX_IDEA_TEXT_CHARACTERS,
        )

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        retained = _require_text(
            "mechanism",
            value,
            maximum=MAX_MECHANISM_CHARACTERS,
        )
        if not normalize_text(retained):
            raise ValueError("mechanism must contain normalized semantic text")
        return retained

    @field_validator("predictions", "falsifiers", "assumptions")
    @classmethod
    def validate_lists(
        cls,
        values: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _require_string_tuple(info.field_name, values, allow_empty=False)

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[SnapshotEvidenceReference, ...],
    ) -> tuple[SnapshotEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > MAX_SNAPSHOT_ITEMS:
            raise ValueError("evidence may contain at most 12 references")
        return values


class OperatorOutcome(InspirationModel):
    """Sanitized, replayable terminal result for one enabled operator."""

    operator: InspirationOperator
    succeeded: bool
    persisted_ideas: Annotated[int, Field(strict=True, ge=0, le=3)]
    duplicate_count: Annotated[int, Field(strict=True, ge=0)] = 0
    error_code: OperatorFailureCode | None = None
    output_tokens_consumed: Annotated[int, Field(strict=True, ge=0)] = 0

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.succeeded:
            if self.persisted_ideas < 1:
                raise ValueError("a successful operator must persist at least one idea")
            if self.error_code is not None:
                raise ValueError("a successful operator cannot carry an error code")
        else:
            if self.error_code is None:
                raise ValueError("a failed operator must carry an error code")
            if self.persisted_ideas != 0:
                raise ValueError("a failed operator cannot persist ideas")
        return self


class InspirationRun(InspirationModel):
    """Owner-visible persisted run identity, configuration, and state."""

    run_id: UUID
    owner_agent_id: UUID
    goal: str
    context: str
    mode: RetrievalMode
    generator: GeneratorKind
    operators: tuple[InspirationOperator, ...]
    include_inbox: bool
    branches_per_operator: int
    output_tokens_per_operator: int
    total_output_tokens: int
    operator_timeout_seconds: int
    global_timeout_seconds: int
    request_hash: str
    snapshot_hash: str | None
    status: InspirationRunStatus
    operator_outcomes: tuple[OperatorOutcome, ...]
    output_tokens_reserved: int
    output_tokens_consumed: int
    elapsed_milliseconds: int
    created_at: datetime
    completed_at: datetime | None

    @field_validator("request_hash")
    @classmethod
    def validate_request_hash(cls, value: str) -> str:
        return _require_hash("request_hash", value)

    @field_validator("snapshot_hash")
    @classmethod
    def validate_optional_snapshot_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_hash("snapshot_hash", value)

    @field_validator("created_at", "completed_at")
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
        info: Any,
    ) -> datetime | None:
        if value is None:
            return None
        return _require_timestamp(info.field_name, value)


class Idea(InspirationModel):
    """Owner-visible immutable idea plus its independent mutable projections."""

    idea_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    operator: InspirationOperator
    ordinal: Annotated[int, Field(strict=True, ge=1, le=3)]
    draft: IdeaDraft
    idea_content_hash: str
    mechanism_hash: str
    duplicate_relation: UUID | None
    owner_decision: IdeaOwnerDecision
    mechanism_cluster_id: str
    maturity: MechanismMaturity
    last_signal_at: datetime
    resulting_experience_id: UUID | None = None
    resulting_version_id: UUID | None = None

    @field_validator(
        "idea_content_hash",
        "mechanism_hash",
        "mechanism_cluster_id",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _require_hash(info.field_name, value)

    @field_validator("last_signal_at")
    @classmethod
    def validate_last_signal_at(cls, value: datetime) -> datetime:
        return _require_timestamp("last_signal_at", value)


class IdeaEvaluation(InspirationModel):
    """One strict evaluation revision; repository logic selects the latest."""

    evaluator_agent_id: UUID
    idea_id: UUID
    verdict: EvaluationVerdict
    reason: StructuredReason | None = None
    evidence: tuple[EvaluationEvidenceReference, ...]
    evaluated_at: datetime

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[EvaluationEvidenceReference, ...],
    ) -> tuple[EvaluationEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > MAX_VERSION_LIST_ITEMS:
            raise ValueError("evidence contains too many references")
        identities = tuple(
            (
                value.type,
                value.id,
                (
                    value.stable_evidence_key
                    if isinstance(value, SnapshotEvidenceReference)
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
            if isinstance(value, SnapshotEvidenceReference)
        )
        if len(snapshot_keys) != len(set(snapshot_keys)):
            raise ValueError("snapshot evidence keys must not repeat")
        return values

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_timestamp("evaluated_at", value)


class MechanismIncubation(InspirationModel):
    """Aggregate-only public state for one deterministic mechanism cluster."""

    cluster_id: str
    canonical_mechanism_hash: str
    member_hashes: tuple[str, ...]
    occurrence_count: Annotated[int, Field(strict=True, ge=1)]
    distinct_snapshot_count: Annotated[int, Field(strict=True, ge=1)]
    distinct_adopter_count: Annotated[int, Field(strict=True, ge=0)]
    supported_count: Annotated[int, Field(strict=True, ge=0)]
    refuted_count: Annotated[int, Field(strict=True, ge=0)]
    maturity: MechanismMaturity
    candidate_since: datetime | None
    last_signal_at: datetime

    @field_validator("cluster_id", "canonical_mechanism_hash")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _require_hash(info.field_name, value)

    @field_validator("member_hashes")
    @classmethod
    def validate_member_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("member_hashes must not be empty")
        for value in values:
            _require_hash("member_hash", value)
        if len(values) != len(set(values)):
            raise ValueError("member_hashes must not contain duplicates")
        return values

    @field_validator("candidate_since", "last_signal_at")
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
        info: Any,
    ) -> datetime | None:
        if value is None:
            return None
        return _require_timestamp(info.field_name, value)

    @model_validator(mode="after")
    def validate_candidate_since(self) -> Self:
        if (
            self.maturity is MechanismMaturity.CANDIDATE
            and self.candidate_since is None
        ):
            raise ValueError("candidate maturity requires candidate_since")
        if (
            self.maturity is not MechanismMaturity.CANDIDATE
            and self.candidate_since is not None
        ):
            raise ValueError("non-candidate maturity must clear candidate_since")
        return self


__all__ = [
    "INSPIRATION_OPERATOR_ORDER",
    "MAX_IDEA_TEXT_CHARACTERS",
    "MAX_SNAPSHOT_EXCERPT_UTF8_BYTES",
    "MAX_SNAPSHOT_ITEMS",
    "MAX_SNAPSHOT_UTF8_BYTES",
    "EvaluationEvidenceReference",
    "EvaluationVerdict",
    "EvidenceCandidate",
    "EvidenceSourceState",
    "EvidenceSourceType",
    "ExperienceVersionEvidenceReference",
    "FrozenSnapshot",
    "GeneratorKind",
    "Idea",
    "IdeaDraft",
    "IdeaEvaluation",
    "IdeaOwnerDecision",
    "InspirationModel",
    "InspirationOperator",
    "InspirationRun",
    "InspirationRunStatus",
    "MechanismIncubation",
    "MechanismMaturity",
    "Maturity",
    "OperatorOutcome",
    "OwnerDecision",
    "RunStatus",
    "SnapshotEvidenceReference",
    "SnapshotItem",
]
