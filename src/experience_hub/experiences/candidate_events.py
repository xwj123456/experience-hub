"""Strict events for trajectory capture and candidate quarantine."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import field_validator

from experience_hub.domain import EventPayload, EventRegistry, StructuredReason
from experience_hub.experiences.candidate_models import CandidateDecision

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _validate_hash(value: str, *, field_name: str) -> str:
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex")
    return value


def _reject_repeated_ids(
    values: tuple[UUID, ...],
    *,
    field_name: str,
) -> tuple[UUID, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not repeat")
    return values


class TrajectoryCapturedV1(EventPayload):
    """Anchor one immutable capture bundle and its writer-declared order."""

    event_type: ClassVar[str] = "trajectory.captured"

    bundle_id: UUID
    owner_agent_id: UUID
    manifest_hash: str
    evidence_ids: tuple[UUID, ...]
    candidate_ids: tuple[UUID, ...]

    @field_validator("manifest_hash")
    @classmethod
    def validate_manifest_hash(cls, value: str) -> str:
        return _validate_hash(value, field_name="manifest_hash")

    @field_validator("evidence_ids", "candidate_ids")
    @classmethod
    def validate_identifier_order(
        cls,
        values: tuple[UUID, ...],
        info: Any,
    ) -> tuple[UUID, ...]:
        return _reject_repeated_ids(values, field_name=str(info.field_name))


class CandidateCreatedV1(EventPayload):
    """Enter one immutable extracted candidate into pending quarantine."""

    event_type: ClassVar[str] = "candidate.created"

    candidate_id: UUID
    bundle_id: UUID
    owner_agent_id: UUID
    content_hash: str
    evidence_ids: tuple[UUID, ...]
    decision_after: Literal[CandidateDecision.PENDING]

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _validate_hash(value, field_name="content_hash")

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(
        cls,
        values: tuple[UUID, ...],
    ) -> tuple[UUID, ...]:
        return _reject_repeated_ids(values, field_name="evidence_ids")


class CandidateAdoptedV1(EventPayload):
    """Record a pending-to-adopted transition and its exact target lineage."""

    event_type: ClassVar[str] = "candidate.adopted"

    candidate_id: UUID
    owner_agent_id: UUID
    decision_before: Literal[CandidateDecision.PENDING]
    decision_after: Literal[CandidateDecision.ADOPTED]
    adoption_id: UUID
    resulting_experience_id: UUID
    resulting_version_id: UUID
    resulting_content_hash: str
    created: bool

    @field_validator("resulting_content_hash")
    @classmethod
    def validate_resulting_content_hash(cls, value: str) -> str:
        return _validate_hash(value, field_name="resulting_content_hash")


class CandidateRejectedV1(EventPayload):
    """Record a pending-to-rejected transition without retaining raw errors."""

    event_type: ClassVar[str] = "candidate.rejected"

    candidate_id: UUID
    owner_agent_id: UUID
    decision_before: Literal[CandidateDecision.PENDING]
    decision_after: Literal[CandidateDecision.REJECTED]
    reason: StructuredReason


_CANDIDATE_EVENT_PAYLOAD_TYPES = (
    TrajectoryCapturedV1,
    CandidateCreatedV1,
    CandidateAdoptedV1,
    CandidateRejectedV1,
)

CANDIDATE_EVENT_TYPES = frozenset(
    payload_type.event_type for payload_type in _CANDIDATE_EVENT_PAYLOAD_TYPES
)


def register_candidate_events(registry: EventRegistry) -> None:
    """Register the immutable capture and candidate event vocabulary."""
    for payload_type in _CANDIDATE_EVENT_PAYLOAD_TYPES:
        registry.register(payload_type)


__all__ = [
    "CANDIDATE_EVENT_TYPES",
    "CandidateAdoptedV1",
    "CandidateCreatedV1",
    "CandidateRejectedV1",
    "TrajectoryCapturedV1",
    "register_candidate_events",
]
