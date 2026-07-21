"""Strict retrieval requests, repository values, and response projections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import ConfigDict, field_validator, model_validator

from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, TypedEvidence
from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.experiences.models import (
    MAX_MECHANISM_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    MAX_VERSION_LIST_ITEMS,
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.retrieval.ranking import (
    MAX_RETRIEVAL_LIMIT,
    RetrievalMode,
)
from experience_hub.retrieval.tokenizer import TermCue

MAX_QUERY_CHARACTERS = 2_000
MAX_PEEK_QUERY_CHARACTERS = 6_001
MAX_CONTENT_BUDGET_BYTES = 64 * 1_024
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_CONTENT_BUDGET_BYTES = MAX_CONTENT_BUDGET_BYTES
DEFAULT_PEEK_LIMIT = 12
DEFAULT_PEEK_CONTENT_BUDGET_BYTES = 24_576
DEFAULT_PEEK_EXCERPT_BYTES = 2_048

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _uuid(name: str, value: UUID) -> UUID:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} must be a UUID")
    return value


def _enum(name: str, value: Any, enum_type: type[Any]) -> Any:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _timestamp(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a timezone-aware datetime")
    try:
        return require_utc(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a timezone-aware datetime") from error


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite float between zero and one")
    return converted


def _immutable_strings(
    name: str,
    values: tuple[str, ...],
    *,
    max_items: int = MAX_VERSION_LIST_ITEMS,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple")
    if len(values) > max_items:
        raise ValueError(f"{name} contains too many values")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain only nonblank strings")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{name} must contain valid Unicode") from error
    return values


def _immutable_evidence(
    values: tuple[TypedEvidence, ...],
) -> tuple[TypedEvidence, ...]:
    if not isinstance(values, tuple):
        raise ValueError("evidence must be an immutable tuple")
    if len(values) > MAX_VERSION_LIST_ITEMS:
        raise ValueError("evidence contains too many values")
    if any(not isinstance(value, TypedEvidence) for value in values):
        raise ValueError("evidence must contain only TypedEvidence values")
    return values


def _immutable_terms(
    name: str,
    values: tuple[TermCue, ...],
    *,
    allow_empty: bool,
) -> tuple[TermCue, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, TermCue) for value in values):
        raise ValueError(f"{name} must contain only TermCue values")
    keys = [(value.term, value.term_kind) for value in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must not contain duplicate cue kinds")
    if tuple(sorted(keys)) != tuple(keys):
        raise ValueError(f"{name} must use canonical term order")
    return values


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """Owner-scoped immutable metadata plus the current lifecycle snapshot."""

    experience_id: UUID
    owner_agent_id: UUID
    kind: ExperienceKind
    origin: ExperienceOrigin
    created_at: datetime
    current_version_id: UUID
    current_version_number: int
    current_version_created_at: datetime
    current_content_hash: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[TypedEvidence, ...]
    falsifiers: tuple[str, ...]
    state: ExperienceStateSnapshotV1
    projection_event_id: int
    latest_causal_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experience_id",
            _uuid("experience_id", self.experience_id),
        )
        object.__setattr__(
            self,
            "owner_agent_id",
            _uuid("owner_agent_id", self.owner_agent_id),
        )
        object.__setattr__(
            self,
            "kind",
            _enum("kind", self.kind, ExperienceKind),
        )
        object.__setattr__(
            self,
            "origin",
            _enum("origin", self.origin, ExperienceOrigin),
        )
        object.__setattr__(
            self,
            "created_at",
            _timestamp("created_at", self.created_at),
        )
        object.__setattr__(
            self,
            "current_version_id",
            _uuid("current_version_id", self.current_version_id),
        )
        object.__setattr__(
            self,
            "current_version_number",
            _positive_integer(
                "current_version_number",
                self.current_version_number,
            ),
        )
        object.__setattr__(
            self,
            "current_version_created_at",
            _timestamp(
                "current_version_created_at",
                self.current_version_created_at,
            ),
        )
        if (
            not isinstance(self.current_content_hash, str)
            or not _SHA256_HEX.fullmatch(self.current_content_hash)
        ):
            raise ValueError(
                "current_content_hash must be lowercase SHA-256 hex"
            )
        if (
            not isinstance(self.summary, str)
            or not self.summary.strip()
            or len(self.summary) > MAX_SUMMARY_CHARACTERS
        ):
            raise ValueError("summary must be a nonblank bounded string")
        if (
            not isinstance(self.mechanism, str)
            or not self.mechanism.strip()
            or len(self.mechanism) > MAX_MECHANISM_CHARACTERS
        ):
            raise ValueError("mechanism must be a nonblank bounded string")
        _immutable_strings("tags", self.tags)
        _immutable_strings("applicability", self.applicability)
        _immutable_evidence(self.evidence)
        _immutable_strings("falsifiers", self.falsifiers)
        if not isinstance(self.state, ExperienceStateSnapshotV1):
            raise ValueError("state must be an ExperienceStateSnapshotV1")
        if (
            self.state.experience_id != self.experience_id
            or self.state.owner_agent_id != self.owner_agent_id
            or self.state.current_version_id != self.current_version_id
            or self.state.current_content_hash != self.current_content_hash
        ):
            raise ValueError("Retrieval record state anchors are inconsistent")
        object.__setattr__(
            self,
            "projection_event_id",
            _positive_integer(
                "projection_event_id",
                self.projection_event_id,
            ),
        )
        object.__setattr__(
            self,
            "latest_causal_at",
            _timestamp("latest_causal_at", self.latest_causal_at),
        )
        known_causal_times = (
            self.created_at,
            self.current_version_created_at,
            self.state.strength_updated_at,
            self.state.last_transition_at,
            *(
                ()
                if self.state.last_accessed_at is None
                else (self.state.last_accessed_at,)
            ),
            *(
                ()
                if self.state.last_lifecycle_evaluated_at is None
                else (self.state.last_lifecycle_evaluated_at,)
            ),
        )
        if self.latest_causal_at < max(known_causal_times):
            raise ValueError("latest_causal_at is behind an aggregate timestamp")


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Owner and canonical cues used to select bounded temperature pools."""

    owner_agent_id: UUID
    query_cues: tuple[TermCue, ...]
    mode: RetrievalMode
    requested_limit: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_agent_id",
            _uuid("owner_agent_id", self.owner_agent_id),
        )
        _immutable_terms("query_cues", self.query_cues, allow_empty=False)
        object.__setattr__(
            self,
            "mode",
            _enum("mode", self.mode, RetrievalMode),
        )
        if (
            isinstance(self.requested_limit, bool)
            or not isinstance(self.requested_limit, int)
            or not 1 <= self.requested_limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("requested_limit must be between 1 and 50")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A selected active experience and all current terms used for ranking."""

    record: RetrievalRecord
    terms: tuple[TermCue, ...]
    raw_overlap: float

    def __post_init__(self) -> None:
        if not isinstance(self.record, RetrievalRecord):
            raise ValueError("record must be a RetrievalRecord")
        if self.record.state.temperature is Temperature.ARCHIVED:
            raise ValueError("Retrieval candidates cannot be archived")
        _immutable_terms("terms", self.terms, allow_empty=False)
        if (
            isinstance(self.raw_overlap, bool)
            or not isinstance(self.raw_overlap, (int, float))
        ):
            raise ValueError("raw_overlap must be a finite positive float")
        converted = float(self.raw_overlap)
        if not math.isfinite(converted) or converted <= 0.0:
            raise ValueError("raw_overlap must be a finite positive float")
        object.__setattr__(self, "raw_overlap", converted)


def _query_text(value: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    retained = value.strip()
    try:
        retained.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("query must contain valid Unicode") from error
    if not retained:
        raise ValueError("query must not be blank")
    if len(retained) > maximum:
        raise ValueError(
            f"query must contain at most {maximum:,} characters"
        )
    return retained


def _query_values(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    immutable = _immutable_strings(name, values)
    retained = tuple(value.strip() for value in immutable)
    return tuple(sorted(set(retained), key=lambda value: value.encode("utf-8")))


def _search_limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_RETRIEVAL_LIMIT
    ):
        raise ValueError("limit must be between 1 and 50")
    return value


def _peek_limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= DEFAULT_PEEK_LIMIT
    ):
        raise ValueError("peek limit must be between 1 and 12")
    return value


def _content_budget(value: int, *, positive: bool) -> int:
    lower_bound = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower_bound <= value <= MAX_CONTENT_BUDGET_BYTES
    ):
        qualifier = "positive and " if positive else ""
        raise ValueError(
            f"content_budget_bytes must be {qualifier}at most 65,536"
        )
    return value


def _bounded_positive_bytes(name: str, value: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} must be positive and at most {maximum:,}")
    return value


def _strict_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class SearchExperiences:
    """Public mutating search semantics, excluding transport idempotency."""

    owner_agent_id: UUID
    query: str
    mode: RetrievalMode
    tags: tuple[str, ...] = ()
    mechanism_cues: tuple[str, ...] = ()
    limit: int = DEFAULT_SEARCH_LIMIT
    content_budget_bytes: int = DEFAULT_CONTENT_BUDGET_BYTES
    expand_cold: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_agent_id",
            _uuid("owner_agent_id", self.owner_agent_id),
        )
        object.__setattr__(
            self,
            "query",
            _query_text(self.query, maximum=MAX_QUERY_CHARACTERS),
        )
        object.__setattr__(
            self,
            "mode",
            _enum("mode", self.mode, RetrievalMode),
        )
        object.__setattr__(self, "tags", _query_values("tags", self.tags))
        object.__setattr__(
            self,
            "mechanism_cues",
            _query_values("mechanism_cues", self.mechanism_cues),
        )
        object.__setattr__(self, "limit", _search_limit(self.limit))
        object.__setattr__(
            self,
            "content_budget_bytes",
            _content_budget(self.content_budget_bytes, positive=False),
        )
        object.__setattr__(
            self,
            "expand_cold",
            _strict_bool("expand_cold", self.expand_cold),
        )


@dataclass(frozen=True, slots=True)
class PeekExperiences:
    """Internal bounded read-only evidence retrieval semantics."""

    owner_agent_id: UUID
    query: str
    mode: RetrievalMode
    tags: tuple[str, ...] = ()
    mechanism_cues: tuple[str, ...] = ()
    limit: int = DEFAULT_PEEK_LIMIT
    content_budget_bytes: int = DEFAULT_PEEK_CONTENT_BUDGET_BYTES
    expand_cold: bool = True
    per_hit_excerpt_bytes: int = DEFAULT_PEEK_EXCERPT_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_agent_id",
            _uuid("owner_agent_id", self.owner_agent_id),
        )
        object.__setattr__(
            self,
            "query",
            _query_text(self.query, maximum=MAX_PEEK_QUERY_CHARACTERS),
        )
        object.__setattr__(
            self,
            "mode",
            _enum("mode", self.mode, RetrievalMode),
        )
        object.__setattr__(self, "tags", _query_values("tags", self.tags))
        object.__setattr__(
            self,
            "mechanism_cues",
            _query_values("mechanism_cues", self.mechanism_cues),
        )
        object.__setattr__(self, "limit", _peek_limit(self.limit))
        object.__setattr__(
            self,
            "content_budget_bytes",
            _bounded_positive_bytes(
                "content_budget_bytes",
                self.content_budget_bytes,
                DEFAULT_PEEK_CONTENT_BUDGET_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "expand_cold",
            _strict_bool("expand_cold", self.expand_cold),
        )
        object.__setattr__(
            self,
            "per_hit_excerpt_bytes",
            _bounded_positive_bytes(
                "per_hit_excerpt_bytes",
                self.per_hit_excerpt_bytes,
                DEFAULT_PEEK_EXCERPT_BYTES,
            ),
        )


class ExperienceView(StrictModel):
    """A full or blurred owner-scoped experience representation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    experience_id: UUID
    owner_agent_id: UUID
    kind: ExperienceKind
    origin: ExperienceOrigin
    created_at: datetime
    version_id: UUID
    version_number: int
    version_created_at: datetime
    content_hash: str
    temperature: Temperature
    importance: float
    confidence: float
    activation_score: float
    source_trust: float
    access_count: int
    access_strength: float
    strength_updated_at: datetime
    last_accessed_at: datetime | None
    last_transition_at: datetime
    last_lifecycle_evaluated_at: datetime | None
    consecutive_below_threshold: int
    pinned: bool
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[TypedEvidence, ...]
    falsifiers: tuple[str, ...]
    blurred: bool
    body: str | None
    body_is_excerpt: bool = False

    @field_validator("version_number", mode="before")
    @classmethod
    def validate_version_number(cls, value: Any) -> Any:
        _positive_integer("version_number", value)
        return value

    @field_validator(
        "importance",
        "confidence",
        "activation_score",
        "source_trust",
        mode="before",
    )
    @classmethod
    def validate_score(cls, value: Any, info: Any) -> Any:
        return _unit_float(info.field_name, value)

    @field_validator(
        "version_created_at",
        "created_at",
        "strength_updated_at",
        "last_accessed_at",
        "last_transition_at",
        "last_lifecycle_evaluated_at",
        mode="after",
    )
    @classmethod
    def normalize_time(
        cls,
        value: datetime | None,
        info: Any,
    ) -> datetime | None:
        return None if value is None else _timestamp(info.field_name, value)

    @field_validator(
        "access_count",
        "consecutive_below_threshold",
        mode="before",
    )
    @classmethod
    def validate_counter(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @field_validator("access_strength", mode="before")
    @classmethod
    def validate_strength(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("access_strength must be a finite float")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 20.0:
            raise ValueError("access_strength must be between zero and twenty")
        return converted

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("content_hash must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_content_visibility(self) -> Self:
        if self.blurred and self.body is not None:
            raise ValueError("blurred experience must not contain a body")
        if not self.blurred and self.body is None:
            raise ValueError("full experience must contain a body")
        if self.blurred and self.body_is_excerpt:
            raise ValueError("blurred experience cannot contain an excerpt")
        return self


class SearchHit(StrictModel):
    """One ranked response item with complete scoring evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    experience: ExperienceView
    score: float
    ranking_relevance: float
    lexical_or_trigram_relevance: float
    mechanism_relevance: float
    activation: float
    expanded: bool
    reactivated: bool

    @field_validator(
        "score",
        "ranking_relevance",
        "lexical_or_trigram_relevance",
        "mechanism_relevance",
        "activation",
        mode="before",
    )
    @classmethod
    def validate_component(cls, value: Any, info: Any) -> Any:
        return _unit_float(info.field_name, value)

    @model_validator(mode="after")
    def validate_flags(self) -> Self:
        if self.expanded == self.experience.blurred:
            raise ValueError("expanded must be the inverse of blurred")
        if self.reactivated and not self.expanded:
            raise ValueError("reactivated hit must be expanded")
        return self


class SearchResult(StrictModel):
    """Deterministically ordered hits and the unconsumed body budget."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    hits: tuple[SearchHit, ...]
    remaining_content_budget_bytes: int

    @field_validator("remaining_content_budget_bytes", mode="before")
    @classmethod
    def validate_remaining_budget(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "remaining_content_budget_bytes must be non-negative"
            )
        return value


__all__ = [
    "DEFAULT_CONTENT_BUDGET_BYTES",
    "DEFAULT_PEEK_CONTENT_BUDGET_BYTES",
    "DEFAULT_PEEK_EXCERPT_BYTES",
    "DEFAULT_PEEK_LIMIT",
    "DEFAULT_SEARCH_LIMIT",
    "MAX_CONTENT_BUDGET_BYTES",
    "MAX_PEEK_QUERY_CHARACTERS",
    "MAX_QUERY_CHARACTERS",
    "CandidateSelection",
    "ExperienceView",
    "PeekExperiences",
    "RetrievalCandidate",
    "RetrievalRecord",
    "SearchExperiences",
    "SearchHit",
    "SearchResult",
]
