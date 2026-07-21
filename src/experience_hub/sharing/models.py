"""Strict values for quarantined social experience propagation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, computed_field, field_validator, model_validator

from experience_hub import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, StructuredReason, TypedEvidence
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.models import (
    MAX_BODY_UTF8_BYTES,
    MAX_MECHANISM_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    MAX_VERSION_LIST_ITEMS,
    ExperienceKind,
    Temperature,
)

MAX_TOPIC_NAME_CHARACTERS = 200
MAX_TOPIC_DESCRIPTION_CHARACTERS = 2_000
MAX_PROVENANCE_HOPS = 4

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    strict=True,
)


class CapsuleStatus(StrEnum):
    ACTIVE = "active"
    RETRACTED = "retracted"


class EffectiveAvailability(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    RETRACTED = "retracted"


class InboxState(StrEnum):
    PENDING = "pending"
    ADOPTED = "adopted"
    REJECTED = "rejected"


class FeedbackVerdict(StrEnum):
    USEFUL = "useful"
    REFUTED = "refuted"
    HARMFUL = "harmful"


@dataclass(frozen=True, slots=True)
class CreateTopic:
    """Request creation of an immutable publication topic."""

    owner_agent_id: UUID
    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner_agent_id, UUID):
            raise ValueError("owner_agent_id must be a UUID")
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("description must be a string or None")


@dataclass(frozen=True, slots=True)
class CreateSubscription:
    """Request a non-backfilling subscription to one topic."""

    subscriber_agent_id: UUID
    topic_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.subscriber_agent_id, UUID):
            raise ValueError("subscriber_agent_id must be a UUID")
        if not isinstance(self.topic_id, UUID):
            raise ValueError("topic_id must be a UUID")


@dataclass(frozen=True, slots=True)
class PublishCapsule:
    """Request immutable publication of one owned experience version."""

    owner_agent_id: UUID
    topic_id: UUID
    experience_id: UUID
    version_id: UUID | None
    expires_at: datetime
    parent_adoption_id: UUID | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("owner_agent_id", self.owner_agent_id),
            ("topic_id", self.topic_id),
            ("experience_id", self.experience_id),
        ):
            if not isinstance(value, UUID):
                raise ValueError(f"{name} must be a UUID")
        for name, optional_value in (
            ("version_id", self.version_id),
            ("parent_adoption_id", self.parent_adoption_id),
        ):
            if optional_value is not None and not isinstance(
                optional_value,
                UUID,
            ):
                raise ValueError(f"{name} must be a UUID or None")
        if not isinstance(self.expires_at, datetime):
            raise ValueError("expires_at must be a datetime")


@dataclass(frozen=True, slots=True)
class AdoptCapsule:
    """Request deliberate adoption of one caller-owned inbox item."""

    adopter_agent_id: UUID
    item_id: UUID
    importance: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.adopter_agent_id, UUID):
            raise ValueError("adopter_agent_id must be a UUID")
        if not isinstance(self.item_id, UUID):
            raise ValueError("item_id must be a UUID")
        object.__setattr__(
            self,
            "importance",
            _unit_float("importance", self.importance),
        )


type SharingMutationReason = StructuredReason | str


@dataclass(frozen=True, slots=True)
class RetractCapsule:
    """Request publisher-owned withdrawal of one active capsule."""

    publisher_agent_id: UUID
    capsule_id: UUID
    reason: SharingMutationReason

    def __post_init__(self) -> None:
        if not isinstance(self.publisher_agent_id, UUID):
            raise ValueError("publisher_agent_id must be a UUID")
        if not isinstance(self.capsule_id, UUID):
            raise ValueError("capsule_id must be a UUID")
        if not isinstance(self.reason, (str, StructuredReason)):
            raise ValueError("reason must be a string or StructuredReason")


@dataclass(frozen=True, slots=True)
class RejectInboxItem:
    """Request recipient-owned rejection of one pending inbox item."""

    recipient_agent_id: UUID
    item_id: UUID
    reason: SharingMutationReason

    def __post_init__(self) -> None:
        if not isinstance(self.recipient_agent_id, UUID):
            raise ValueError("recipient_agent_id must be a UUID")
        if not isinstance(self.item_id, UUID):
            raise ValueError("item_id must be a UUID")
        if not isinstance(self.reason, (str, StructuredReason)):
            raise ValueError("reason must be a string or StructuredReason")


@dataclass(frozen=True, slots=True)
class RecordCapsuleFeedback:
    """Request one immutable revision of post-quarantine feedback."""

    observer_agent_id: UUID
    capsule_id: UUID
    verdict: FeedbackVerdict
    reason: SharingMutationReason
    evidence: tuple[TypedEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observer_agent_id, UUID):
            raise ValueError("observer_agent_id must be a UUID")
        if not isinstance(self.capsule_id, UUID):
            raise ValueError("capsule_id must be a UUID")
        if not isinstance(self.verdict, FeedbackVerdict):
            raise ValueError("verdict must be a FeedbackVerdict")
        if not isinstance(self.reason, (str, StructuredReason)):
            raise ValueError("reason must be a string or StructuredReason")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, TypedEvidence) for item in self.evidence
        ):
            raise ValueError("evidence must be a tuple of TypedEvidence values")


def _timestamp(name: str, value: datetime) -> datetime:
    try:
        return require_utc(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timezone-aware datetime") from error


def _unit_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite number between zero and one")
    return converted


def _canonical_tuple(values: tuple[Any, ...]) -> tuple[Any, ...]:
    unique = {canonical_json_bytes(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _validate_utf8(name: str, value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error


class Topic(StrictModel):
    """One immutable publication namespace."""

    model_config = _STRICT_CONFIG

    topic_id: UUID
    owner_agent_id: UUID
    name: str
    description: str | None
    created_at: datetime

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        _validate_utf8("name", normalized)
        if not 1 <= len(normalized) <= MAX_TOPIC_NAME_CHARACTERS:
            raise ValueError("name must contain 1-200 characters after trimming")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        _validate_utf8("description", normalized)
        if not 1 <= len(normalized) <= MAX_TOPIC_DESCRIPTION_CHARACTERS:
            raise ValueError(
                "description must contain 1-2,000 characters after trimming"
            )
        return normalized

    @field_validator("created_at", mode="after")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _timestamp("created_at", value)


class Subscription(StrictModel):
    """One immutable non-backfilling topic subscription."""

    model_config = _STRICT_CONFIG

    subscription_id: UUID
    subscriber_agent_id: UUID
    topic_id: UUID
    creation_event_id: int
    created_at: datetime

    @field_validator("creation_event_id", mode="before")
    @classmethod
    def validate_creation_event_id(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("creation_event_id must be a positive integer")
        return value

    @field_validator("created_at", mode="after")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _timestamp("created_at", value)


class ProvenanceHop(StrictModel):
    """One prior capsule and publisher in a root-first propagation chain."""

    model_config = _STRICT_CONFIG

    capsule_id: UUID
    publisher_agent_id: UUID


class Capsule(StrictModel):
    """Immutable transported experience content plus its current status."""

    model_config = _STRICT_CONFIG

    capsule_id: UUID
    transport_schema_version: Literal[1]
    topic_id: UUID
    source_experience_id: UUID
    source_version_id: UUID
    publisher_agent_id: UUID
    kind: ExperienceKind
    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[TypedEvidence, ...]
    falsifiers: tuple[str, ...]
    publisher_confidence: float
    provenance_chain: tuple[ProvenanceHop, ...]
    root_fingerprint: str
    source_content_hash: str
    created_at: datetime
    expires_at: datetime
    hop_count: int
    capsule_hash: str
    status: CapsuleStatus
    last_transition_at: datetime

    @field_validator(
        "tags",
        "applicability",
        "evidence",
        "falsifiers",
        mode="before",
    )
    @classmethod
    def enforce_input_array_limit(cls, values: Any) -> Any:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            return values
        if len(values) > MAX_VERSION_LIST_ITEMS:
            raise ValueError(
                f"capsule arrays may contain at most "
                f"{MAX_VERSION_LIST_ITEMS} input items"
            )
        return values

    @field_validator("provenance_chain", mode="before")
    @classmethod
    def enforce_chain_limit(cls, values: Any) -> Any:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            return values
        if len(values) > MAX_PROVENANCE_HOPS:
            raise ValueError(
                f"provenance_chain may contain at most {MAX_PROVENANCE_HOPS} hops"
            )
        return values

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        encoded = _validate_utf8("body", value)
        if not value.strip():
            raise ValueError("body must not be blank")
        if len(encoded) > MAX_BODY_UTF8_BYTES:
            raise ValueError("body must be at most 64 KiB when UTF-8 encoded")
        return value

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        _validate_utf8("summary", value)
        if not value.strip():
            raise ValueError("summary must not be blank")
        if len(value) > MAX_SUMMARY_CHARACTERS:
            raise ValueError("summary must contain at most 1,000 characters")
        return value

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        _validate_utf8("mechanism", value)
        if not value.strip():
            raise ValueError("mechanism must not be blank")
        if len(value) > MAX_MECHANISM_CHARACTERS:
            raise ValueError("mechanism must contain at most 2,000 characters")
        return value

    @field_validator("tags", "applicability", "falsifiers", mode="after")
    @classmethod
    def canonicalize_string_array(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        for value in values:
            if not value.strip():
                raise ValueError("capsule string arrays must not contain blanks")
            _validate_utf8("capsule string array value", value)
        return _canonical_tuple(values)

    @field_validator("evidence", mode="after")
    @classmethod
    def canonicalize_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        return _canonical_tuple(values)

    @field_validator(
        "root_fingerprint",
        "source_content_hash",
        "capsule_hash",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex")
        return value

    @field_validator("publisher_confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: Any) -> float:
        return _unit_float("publisher_confidence", value)

    @field_validator("hop_count", mode="before")
    @classmethod
    def validate_hop_count(cls, value: Any) -> Any:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_PROVENANCE_HOPS
        ):
            raise ValueError("hop_count must be an integer between zero and four")
        return value

    @field_validator(
        "created_at",
        "expires_at",
        "last_transition_at",
        mode="after",
    )
    @classmethod
    def normalize_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _timestamp(info.field_name, value)

    @model_validator(mode="after")
    def validate_transport_invariants(self) -> Self:
        if self.hop_count != len(self.provenance_chain):
            raise ValueError("hop_count must equal the prior provenance-chain length")
        capsule_ids = tuple(hop.capsule_id for hop in self.provenance_chain)
        if len(capsule_ids) != len(set(capsule_ids)):
            raise ValueError("provenance_chain must not repeat a capsule")
        if self.capsule_id in capsule_ids:
            raise ValueError("a capsule cannot occur in its own provenance chain")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if self.last_transition_at < self.created_at:
            raise ValueError("last_transition_at must not precede created_at")
        return self


class InboxItem(StrictModel):
    """One owner-visible quarantined route and its effective availability."""

    model_config = _STRICT_CONFIG

    item_id: UUID
    recipient_agent_id: UUID
    capsule_id: UUID
    capsule: Capsule
    state: InboxState
    effective_availability: EffectiveAvailability

    @model_validator(mode="after")
    def validate_capsule_identity(self) -> Self:
        if self.capsule_id != self.capsule.capsule_id:
            raise ValueError("capsule_id must match the nested capsule")
        if self.effective_availability is EffectiveAvailability.ACTIVE:
            if self.capsule.status is not CapsuleStatus.ACTIVE:
                raise ValueError("an active availability requires an active capsule")
        elif (
            self.effective_availability is EffectiveAvailability.RETRACTED
            and self.capsule.status is not CapsuleStatus.RETRACTED
        ):
            raise ValueError("a retracted availability requires a retracted capsule")
        elif (
            self.effective_availability is EffectiveAvailability.EXPIRED
            and self.capsule.status is not CapsuleStatus.ACTIVE
        ):
            raise ValueError("an expired availability requires an active capsule")
        return self


class AdoptionResult(StrictModel):
    """Metadata-only outcome of explicit capsule adoption."""

    model_config = _STRICT_CONFIG

    experience: ExperienceRecord
    created: bool
    corroboration_applied: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        record = self.experience
        if any(
            not isinstance(value, UUID)
            for value in (
                record.experience_id,
                record.owner_agent_id,
                record.current_version_id,
            )
        ):
            raise ValueError("adoption experience identities must be UUIDs")
        if not isinstance(
            record.current_content_hash, str
        ) or not _SHA256_HEX.fullmatch(record.current_content_hash):
            raise ValueError(
                "adoption experience content hash must be lowercase SHA-256 hex"
            )
        if not isinstance(record.temperature, Temperature):
            raise ValueError("adoption experience temperature must be a Temperature")
        if self.created and self.corroboration_applied:
            raise ValueError("a newly created experience cannot receive corroboration")
        return self


class FeedbackRevision(StrictModel):
    """One immutable observer/capsule feedback revision."""

    model_config = _STRICT_CONFIG

    feedback_id: UUID
    observer_agent_id: UUID
    capsule_id: UUID
    revision: int
    verdict: FeedbackVerdict
    reason: StructuredReason
    evidence: tuple[TypedEvidence, ...]
    created_at: datetime

    @field_validator("revision", mode="before")
    @classmethod
    def validate_revision(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("revision must be a positive integer")
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def enforce_evidence_limit(cls, values: Any) -> Any:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            return values
        if len(values) > MAX_VERSION_LIST_ITEMS:
            raise ValueError(
                f"evidence may contain at most {MAX_VERSION_LIST_ITEMS} input items"
            )
        return values

    @field_validator("evidence", mode="after")
    @classmethod
    def canonicalize_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        return _canonical_tuple(values)

    @field_validator("created_at", mode="after")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _timestamp("created_at", value)


class Reputation(StrictModel):
    """Observer-relative effective feedback counts and Bayesian trust."""

    model_config = _STRICT_CONFIG

    subject_agent_id: UUID
    observer_agent_id: UUID
    useful_count: int
    refuted_count: int
    harmful_count: int
    alpha: int
    beta: int
    last_feedback_at: datetime

    @field_validator(
        "useful_count",
        "refuted_count",
        "harmful_count",
        mode="before",
    )
    @classmethod
    def validate_count(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator("alpha", "beta", mode="before")
    @classmethod
    def validate_prior_count(cls, value: Any, info: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{info.field_name} must be an integer at least two")
        return int(value)

    @field_validator("last_feedback_at", mode="after")
    @classmethod
    def normalize_feedback_at(cls, value: datetime) -> datetime:
        return _timestamp("last_feedback_at", value)

    @model_validator(mode="after")
    def validate_effective_counts(self) -> Self:
        if self.subject_agent_id == self.observer_agent_id:
            raise ValueError("subject and observer agents must be distinct")
        expected_alpha = 2 + self.useful_count
        expected_beta = 2 + self.refuted_count + self.harmful_count
        if self.alpha != expected_alpha:
            raise ValueError("alpha must equal the prior plus useful_count")
        if self.beta != expected_beta:
            raise ValueError(
                "beta must equal the prior plus refuted_count and harmful_count"
            )
        return self

    @computed_field(return_type=float)  # type: ignore[prop-decorator]
    @property
    def trust(self) -> float:
        """Return derived trust without accepting or persisting a trust field."""
        return self.alpha / (self.alpha + self.beta)


__all__ = [
    "MAX_PROVENANCE_HOPS",
    "MAX_TOPIC_DESCRIPTION_CHARACTERS",
    "MAX_TOPIC_NAME_CHARACTERS",
    "AdoptionResult",
    "AdoptCapsule",
    "Capsule",
    "CapsuleStatus",
    "CreateSubscription",
    "CreateTopic",
    "EffectiveAvailability",
    "FeedbackRevision",
    "FeedbackVerdict",
    "InboxItem",
    "InboxState",
    "ProvenanceHop",
    "PublishCapsule",
    "RecordCapsuleFeedback",
    "RejectInboxItem",
    "Reputation",
    "RetractCapsule",
    "SharingMutationReason",
    "Subscription",
    "Topic",
]
