"""Strict version-one events for immutable experience creation."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, ClassVar, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import StructuredReason, TypedEvidence
from experience_hub.domain.events import EventPayload, EventRegistry
from experience_hub.experiences.models import LinkRelation, Temperature

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ACCESS_STRENGTH = 20.0
_CORRECTION_FIELDS = frozenset(
    {
        "current_version_id",
        "current_content_hash",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_ACCESS_FIELDS = frozenset(
    {
        "access_count",
        "access_strength",
        "strength_updated_at",
        "last_accessed_at",
        "activation_score",
    }
)
_TEMPERATURE_CHANGE_FIELDS = frozenset(
    {
        "temperature",
        "last_transition_at",
        "consecutive_below_threshold",
    }
)
_LIFECYCLE_EVALUATION_FIELDS = frozenset(
    {
        "access_strength",
        "strength_updated_at",
        "activation_score",
        "last_lifecycle_evaluated_at",
        "consecutive_below_threshold",
    }
)
_CONFIDENCE_EVENT_FIELDS = frozenset(
    {
        "confidence",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_PIN_EVENT_FIELDS = frozenset(
    {
        "pinned",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_RESTORE_EVENT_FIELDS = frozenset(
    {
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)

type RetrievalEventMode = Literal["focused", "associative"]
type LifecycleThresholdTarget = Literal[
    "none",
    "promote_hot",
    "demote_warm",
    "demote_cold",
    "archive",
]
type TemperatureChangeCause = Literal[
    "cold_reactivation",
    "confirmation",
    "pin",
    "lifecycle_activation",
    "lifecycle_demotion",
    "policy_archive",
    "restore",
    "capsule_corroboration",
]

_TEMPERATURE_TRANSITIONS: dict[
    TemperatureChangeCause,
    frozenset[tuple[Temperature, Temperature]],
] = {
    "cold_reactivation": frozenset(
        {(Temperature.COLD, Temperature.WARM)}
    ),
    "confirmation": frozenset(
        {
            (Temperature.WARM, Temperature.HOT),
            (Temperature.COLD, Temperature.HOT),
        }
    ),
    "pin": frozenset(
        {
            (Temperature.WARM, Temperature.HOT),
            (Temperature.COLD, Temperature.HOT),
        }
    ),
    "lifecycle_activation": frozenset(
        {(Temperature.WARM, Temperature.HOT)}
    ),
    "lifecycle_demotion": frozenset(
        {
            (Temperature.HOT, Temperature.WARM),
            (Temperature.WARM, Temperature.COLD),
        }
    ),
    "policy_archive": frozenset(
        {(Temperature.COLD, Temperature.ARCHIVED)}
    ),
    "restore": frozenset(
        {(Temperature.ARCHIVED, Temperature.WARM)}
    ),
    "capsule_corroboration": frozenset(
        {(Temperature.COLD, Temperature.HOT)}
    ),
}
_CYCLE_CAUSES = frozenset(
    {
        "lifecycle_activation",
        "lifecycle_demotion",
        "policy_archive",
    }
)


class ExperienceStateSnapshotV1(BaseModel):
    """All semantic experience-state fields, excluding the event checkpoint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    experience_id: UUID
    owner_agent_id: UUID
    current_version_id: UUID
    current_content_hash: str
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

    @field_validator("current_content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Content hash must be lowercase SHA-256 hex")
        return value

    @field_validator(
        "importance",
        "confidence",
        "activation_score",
        "source_trust",
        mode="before",
    )
    @classmethod
    def validate_unit_float(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("State score must be a finite float")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError("State score must be between zero and one")
        return converted

    @field_validator("access_strength", mode="before")
    @classmethod
    def validate_access_strength(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Access strength must be a finite float")
        converted = float(value)
        if (
            not math.isfinite(converted)
            or not 0.0 <= converted <= _MAX_ACCESS_STRENGTH
        ):
            raise ValueError("Access strength must be between zero and twenty")
        return converted

    @field_validator(
        "access_count",
        "consecutive_below_threshold",
        mode="before",
    )
    @classmethod
    def validate_counter(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("State counters must be non-negative integers")
        return value

    @field_validator(
        "strength_updated_at",
        "last_accessed_at",
        "last_transition_at",
        "last_lifecycle_evaluated_at",
        mode="after",
    )
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return None if value is None else require_utc(value)


class VersionLinkRefV1(BaseModel):
    """The immutable event reference corresponding to one source link row."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    target_experience_id: UUID
    relation: LinkRelation


def _changed_snapshot_fields(
    before: ExperienceStateSnapshotV1,
    after: ExperienceStateSnapshotV1,
) -> frozenset[str]:
    return frozenset(
        name
        for name in ExperienceStateSnapshotV1.model_fields
        if getattr(before, name) != getattr(after, name)
    )


class ExperienceCreatedV1(EventPayload):
    event_type: ClassVar[str] = "experience.created"

    experience_id: UUID
    version_id: UUID
    after: ExperienceStateSnapshotV1

    @model_validator(mode="after")
    def validate_anchors(self) -> Self:
        if self.experience_id != self.after.experience_id:
            raise ValueError("Created experience ID does not match after state")
        if self.version_id != self.after.current_version_id:
            raise ValueError("Created version ID does not match after state")
        return self


class ExperienceVersionCreatedV1(EventPayload):
    event_type: ClassVar[str] = "experience.version_created"

    experience_id: UUID
    version_id: UUID
    version_number: int
    supersedes_version_id: UUID | None
    links: tuple[VersionLinkRefV1, ...]
    before: ExperienceStateSnapshotV1
    after: ExperienceStateSnapshotV1

    @field_validator("version_number", mode="before")
    @classmethod
    def validate_version_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Version number must be a positive integer")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if (
            self.experience_id != self.before.experience_id
            or self.experience_id != self.after.experience_id
        ):
            raise ValueError("Version experience ID does not match snapshots")
        if self.before.owner_agent_id != self.after.owner_agent_id:
            raise ValueError("A correction cannot change experience owner")
        if self.after.current_version_id != self.version_id:
            raise ValueError("Version ID does not match after state")

        pairs = [
            (link.target_experience_id, link.relation) for link in self.links
        ]
        if any(target == self.experience_id for target, _ in pairs):
            raise ValueError("An experience version cannot link to itself")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Version links must not contain duplicate pairs")
        if tuple(
            sorted(pairs, key=lambda pair: (pair[0].bytes, pair[1].value))
        ) != tuple(pairs):
            raise ValueError("Version links must use canonical order")

        if self.version_number == 1:
            if self.supersedes_version_id is not None:
                raise ValueError("Initial version cannot supersede another version")
            if self.before != self.after:
                raise ValueError("Initial version event must be a semantic no-op")
            return self

        if self.supersedes_version_id != self.before.current_version_id:
            raise ValueError("Correction must supersede the before-state version")
        if self.version_id == self.before.current_version_id:
            raise ValueError("Correction must allocate a new version ID")
        changed = _changed_snapshot_fields(self.before, self.after)
        if not changed <= _CORRECTION_FIELDS:
            raise ValueError("Correction changes unauthorized experience state")
        return self


class ExperienceAccessedV1(EventPayload):
    """One full current-version payload returned to a caller."""

    event_type: ClassVar[str] = "experience.accessed"

    experience_id: UUID
    version_id: UUID
    before: ExperienceStateSnapshotV1
    after: ExperienceStateSnapshotV1

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if (
            self.experience_id != self.before.experience_id
            or self.experience_id != self.after.experience_id
        ):
            raise ValueError("Accessed experience ID does not match snapshots")
        if (
            self.version_id != self.before.current_version_id
            or self.version_id != self.after.current_version_id
        ):
            raise ValueError("Accessed version ID does not match snapshots")
        if self.before.temperature is Temperature.ARCHIVED:
            raise ValueError("Archived experience cannot be accessed")
        changed = _changed_snapshot_fields(self.before, self.after)
        if not changed <= _ACCESS_FIELDS:
            raise ValueError("Accessed event changes unauthorized state")
        if self.after.access_count != self.before.access_count + 1:
            raise ValueError("Accessed event must increment access count once")
        if (
            self.after.last_accessed_at is None
            or self.after.strength_updated_at
            != self.after.last_accessed_at
        ):
            raise ValueError("Accessed event must share one access timestamp")
        return self


class ExperienceReactivatedV1(EventPayload):
    """A qualifying cold-recall signal without raw query disclosure."""

    event_type: ClassVar[str] = "experience.reactivated"

    experience_id: UUID
    query_hash: str
    mode: RetrievalEventMode
    signal: float
    before: ExperienceStateSnapshotV1
    after: ExperienceStateSnapshotV1

    @field_validator("query_hash")
    @classmethod
    def validate_query_hash(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Query hash must be lowercase SHA-256 hex")
        return value

    @field_validator("signal", mode="before")
    @classmethod
    def validate_signal(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Reactivation signal must be a finite float")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError("Reactivation signal must be between zero and one")
        return converted

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if (
            self.experience_id != self.before.experience_id
            or self.experience_id != self.after.experience_id
        ):
            raise ValueError(
                "Reactivated experience ID does not match snapshots"
            )
        if self.before != self.after:
            raise ValueError("Reactivated event must be a semantic no-op")
        if self.before.temperature is not Temperature.COLD:
            raise ValueError("Reactivated event requires cold state")
        return self


class ExperienceTemperatureChangedV1(EventPayload):
    """The only reducer input authorized to change experience temperature."""

    event_type: ClassVar[str] = "experience.temperature_changed"

    experience_id: UUID
    cause: TemperatureChangeCause
    cycle_id: UUID | None
    before: ExperienceStateSnapshotV1
    after: ExperienceStateSnapshotV1

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if (
            self.experience_id != self.before.experience_id
            or self.experience_id != self.after.experience_id
        ):
            raise ValueError(
                "Temperature-changed experience ID does not match snapshots"
            )
        transition = (self.before.temperature, self.after.temperature)
        if transition not in _TEMPERATURE_TRANSITIONS[self.cause]:
            raise ValueError(
                "Temperature-changed cause does not permit this transition"
            )
        if self.cause in _CYCLE_CAUSES:
            if self.cycle_id is None:
                raise ValueError(
                    "Temperature-changed lifecycle cause requires a cycle ID"
                )
        elif self.cycle_id is not None:
            raise ValueError(
                "Temperature-changed command cause forbids a cycle ID"
            )
        changed = _changed_snapshot_fields(self.before, self.after)
        if not changed <= _TEMPERATURE_CHANGE_FIELDS:
            raise ValueError(
                "Temperature-changed event changes unauthorized state"
            )
        if self.after.consecutive_below_threshold != 0:
            raise ValueError(
                "Temperature change must reset the lifecycle counter"
            )
        return self


class _LifecycleStateEventV1(EventPayload):
    """Shared strict anchors for Task 6 state-event payloads."""

    allowed_state_fields: ClassVar[frozenset[str]]

    experience_id: UUID
    before: ExperienceStateSnapshotV1
    after: ExperienceStateSnapshotV1

    @model_validator(mode="after")
    def validate_state_anchors_and_fields(self) -> Self:
        if (
            self.experience_id != self.before.experience_id
            or self.experience_id != self.after.experience_id
        ):
            raise ValueError(
                "Lifecycle experience ID does not match snapshots"
            )
        changed = _changed_snapshot_fields(self.before, self.after)
        if not changed <= self.allowed_state_fields:
            raise ValueError("Lifecycle event changes unauthorized state")
        if self.before.temperature is not self.after.temperature:
            raise ValueError("Lifecycle event cannot change temperature")
        return self


class _EvidenceLifecycleStateEventV1(_LifecycleStateEventV1):
    reason: StructuredReason | None
    evidence: tuple[TypedEvidence, ...]

    @field_validator("evidence", mode="after")
    @classmethod
    def canonicalize_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        unique = {canonical_json_bytes(value): value for value in values}
        return tuple(unique[key] for key in sorted(unique))


class _ReasonLifecycleStateEventV1(_LifecycleStateEventV1):
    reason: StructuredReason | None


class ExperienceLifecycleEvaluatedV1(_LifecycleStateEventV1):
    """One eligible deterministic lifecycle evaluation."""

    event_type: ClassVar[str] = "experience.lifecycle_evaluated"
    allowed_state_fields: ClassVar[frozenset[str]] = (
        _LIFECYCLE_EVALUATION_FIELDS
    )

    cycle_id: UUID
    evaluated_at: datetime
    threshold_target: LifecycleThresholdTarget

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def normalize_evaluated_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def validate_evaluation_times(self) -> Self:
        if (
            self.after.last_lifecycle_evaluated_at != self.evaluated_at
            or self.after.strength_updated_at != self.evaluated_at
        ):
            raise ValueError(
                "Lifecycle evaluation must materialize at evaluated_at"
            )
        return self


class ExperienceConfirmedV1(_EvidenceLifecycleStateEventV1):
    """Explicit positive evidence with the locked confidence formula."""

    event_type: ClassVar[str] = "experience.confirmed"
    allowed_state_fields: ClassVar[frozenset[str]] = _CONFIDENCE_EVENT_FIELDS

    @model_validator(mode="after")
    def validate_confidence_formula(self) -> Self:
        expected = self.before.confidence + (
            1.0 - self.before.confidence
        ) * 0.20
        if not math.isclose(
            self.after.confidence,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Confirmed confidence does not use the locked formula"
            )
        return self


class ExperienceRefutedV1(_EvidenceLifecycleStateEventV1):
    """Explicit negative evidence with the locked confidence formula."""

    event_type: ClassVar[str] = "experience.refuted"
    allowed_state_fields: ClassVar[frozenset[str]] = _CONFIDENCE_EVENT_FIELDS

    @model_validator(mode="after")
    def validate_confidence_formula(self) -> Self:
        expected = self.before.confidence * 0.65
        if not math.isclose(
            self.after.confidence,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Refuted confidence does not use the locked formula"
            )
        return self


class ExperienceCorroboratedV1(_LifecycleStateEventV1):
    """One independent capsule root's trust-weighted confidence contribution."""

    event_type: ClassVar[str] = "experience.corroborated"
    allowed_state_fields: ClassVar[frozenset[str]] = _CONFIDENCE_EVENT_FIELDS

    adoption_id: UUID
    capsule_id: UUID
    root_fingerprint: str
    captured_trust: float

    @field_validator("root_fingerprint")
    @classmethod
    def validate_root_fingerprint(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Root fingerprint must be lowercase SHA-256 hex")
        return value

    @field_validator("captured_trust", mode="before")
    @classmethod
    def validate_captured_trust(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Captured trust must be a finite unit float")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError("Captured trust must be a finite unit float")
        return converted

    @model_validator(mode="after")
    def validate_confidence_formula(self) -> Self:
        expected = self.before.confidence + (
            1.0 - self.before.confidence
        ) * 0.20 * self.captured_trust
        if not math.isclose(
            self.after.confidence,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Corroborated confidence does not use the locked formula"
            )
        return self


class ExperiencePinnedV1(_ReasonLifecycleStateEventV1):
    """An explicit false-to-true pin state change."""

    event_type: ClassVar[str] = "experience.pinned"
    allowed_state_fields: ClassVar[frozenset[str]] = _PIN_EVENT_FIELDS

    @model_validator(mode="after")
    def validate_pin_flip(self) -> Self:
        if self.before.pinned or not self.after.pinned:
            raise ValueError("Pinned event must change false to true")
        return self


class ExperienceUnpinnedV1(_ReasonLifecycleStateEventV1):
    """An explicit true-to-false pin state change."""

    event_type: ClassVar[str] = "experience.unpinned"
    allowed_state_fields: ClassVar[frozenset[str]] = _PIN_EVENT_FIELDS

    @model_validator(mode="after")
    def validate_pin_flip(self) -> Self:
        if not self.before.pinned or self.after.pinned:
            raise ValueError("Unpinned event must change true to false")
        return self


class ExperienceArchivedV1(_LifecycleStateEventV1):
    """The fixed policy explanation immediately before archival."""

    event_type: ClassVar[str] = "experience.archived"
    allowed_state_fields: ClassVar[frozenset[str]] = frozenset()

    cycle_id: UUID
    reason: StructuredReason

    @model_validator(mode="after")
    def validate_archive_explanation(self) -> Self:
        if self.before != self.after:
            raise ValueError("Archived event must be a semantic no-op")
        if self.before.temperature is not Temperature.COLD:
            raise ValueError("Archived event requires cold state")
        if self.reason != StructuredReason.policy_due():
            raise ValueError("Archived event requires the policy-due reason")
        return self


class ExperienceRestoredV1(_ReasonLifecycleStateEventV1):
    """The materialized explanation immediately before explicit restore."""

    event_type: ClassVar[str] = "experience.restored"
    allowed_state_fields: ClassVar[frozenset[str]] = _RESTORE_EVENT_FIELDS

    @model_validator(mode="after")
    def validate_archived_state(self) -> Self:
        if (
            self.before.temperature is not Temperature.ARCHIVED
            or self.after.temperature is not Temperature.ARCHIVED
        ):
            raise ValueError(
                "Restored event snapshots must remain archived"
            )
        return self


_TASK2_EXPERIENCE_EVENT_PAYLOAD_TYPES = (
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
)
TASK2_EXPERIENCE_EVENT_TYPES = frozenset(
    payload_type.event_type
    for payload_type in _TASK2_EXPERIENCE_EVENT_PAYLOAD_TYPES
)
_RETRIEVAL_EXPERIENCE_EVENT_PAYLOAD_TYPES = (
    ExperienceAccessedV1,
    ExperienceReactivatedV1,
    ExperienceTemperatureChangedV1,
)
RETRIEVAL_EXPERIENCE_EVENT_TYPES = frozenset(
    payload_type.event_type
    for payload_type in _RETRIEVAL_EXPERIENCE_EVENT_PAYLOAD_TYPES
)
_LIFECYCLE_EXPERIENCE_EVENT_PAYLOAD_TYPES = (
    ExperienceLifecycleEvaluatedV1,
    ExperienceConfirmedV1,
    ExperienceRefutedV1,
    ExperiencePinnedV1,
    ExperienceUnpinnedV1,
    ExperienceArchivedV1,
    ExperienceRestoredV1,
)
LIFECYCLE_EXPERIENCE_EVENT_TYPES = frozenset(
    payload_type.event_type
    for payload_type in _LIFECYCLE_EXPERIENCE_EVENT_PAYLOAD_TYPES
)
_CORROBORATION_EXPERIENCE_EVENT_PAYLOAD_TYPES = (
    ExperienceCorroboratedV1,
)
CORROBORATION_EXPERIENCE_EVENT_TYPES = frozenset(
    payload_type.event_type
    for payload_type in _CORROBORATION_EXPERIENCE_EVENT_PAYLOAD_TYPES
)
STATE_EXPERIENCE_EVENT_TYPES = (
    TASK2_EXPERIENCE_EVENT_TYPES
    | RETRIEVAL_EXPERIENCE_EVENT_TYPES
    | LIFECYCLE_EXPERIENCE_EVENT_TYPES
    | CORROBORATION_EXPERIENCE_EVENT_TYPES
)
_STATE_EXPERIENCE_EVENT_PAYLOAD_TYPES = (
    *_TASK2_EXPERIENCE_EVENT_PAYLOAD_TYPES,
    *_RETRIEVAL_EXPERIENCE_EVENT_PAYLOAD_TYPES,
    *_LIFECYCLE_EXPERIENCE_EVENT_PAYLOAD_TYPES,
    *_CORROBORATION_EXPERIENCE_EVENT_PAYLOAD_TYPES,
)


def is_valid_version_event_sequence(
    *,
    version_number: int,
    aggregate_sequence: int,
) -> bool:
    """Keep version numbering independent from later aggregate event kinds."""
    if version_number == 1:
        return aggregate_sequence == 2
    return version_number > 1 and aggregate_sequence > 2


def register_experience_events(registry: EventRegistry) -> None:
    """Register the cumulative strict state-event protocol through Task 6."""
    for payload_type in _STATE_EXPERIENCE_EVENT_PAYLOAD_TYPES:
        registry.register(payload_type)


__all__ = [
    "CORROBORATION_EXPERIENCE_EVENT_TYPES",
    "LIFECYCLE_EXPERIENCE_EVENT_TYPES",
    "LifecycleThresholdTarget",
    "RETRIEVAL_EXPERIENCE_EVENT_TYPES",
    "RetrievalEventMode",
    "STATE_EXPERIENCE_EVENT_TYPES",
    "ExperienceAccessedV1",
    "ExperienceArchivedV1",
    "ExperienceConfirmedV1",
    "ExperienceCorroboratedV1",
    "ExperienceCreatedV1",
    "ExperienceLifecycleEvaluatedV1",
    "ExperiencePinnedV1",
    "ExperienceReactivatedV1",
    "ExperienceRefutedV1",
    "ExperienceRestoredV1",
    "ExperienceStateSnapshotV1",
    "ExperienceTemperatureChangedV1",
    "ExperienceUnpinnedV1",
    "ExperienceVersionCreatedV1",
    "TASK2_EXPERIENCE_EVENT_TYPES",
    "TemperatureChangeCause",
    "VersionLinkRefV1",
    "is_valid_version_event_sequence",
    "register_experience_events",
]
