"""Strict version-one events for Passport quarantine and explicit decisions."""

from __future__ import annotations

import math
import re
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import field_validator

from experience_hub.domain import EventPayload, EventRegistry, StructuredReason
from experience_hub.passports.contracts import PassportState


class _PassportEventV1(EventPayload):
    import_id: UUID
    owner_agent_id: UUID

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Passport event schema version must be integer 1")
        return value


class PassportImportedV1(_PassportEventV1):
    """Anchor the exact owned immutable file and enter pending quarantine."""

    event_type: ClassVar[str] = "passport.imported"
    passport_hash: str
    state_after: Literal[PassportState.PENDING]

    @field_validator("passport_hash")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Passport hash must be lowercase SHA-256 hex")
        return value


class PassportAdoptedV1(_PassportEventV1):
    """Bind one pending decision to its immutable local target and inputs."""

    event_type: ClassVar[str] = "passport.adopted"
    state_before: Literal[PassportState.PENDING]
    state_after: Literal[PassportState.ADOPTED]
    adoption_id: UUID
    resulting_experience_id: UUID
    resulting_version_id: UUID
    resulting_content_hash: str
    created: bool
    importance: float
    confidence: float

    @field_validator("resulting_content_hash")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Resulting hash must be lowercase SHA-256 hex")
        return value

    @field_validator("importance", "confidence")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Decision scores must be finite values between 0 and 1")
        return value


class PassportRejectedV1(_PassportEventV1):
    """Retain a structured rejection without creating an ordinary experience."""

    event_type: ClassVar[str] = "passport.rejected"
    state_before: Literal[PassportState.PENDING]
    state_after: Literal[PassportState.REJECTED]
    reason: StructuredReason


_PAYLOADS = (PassportImportedV1, PassportAdoptedV1, PassportRejectedV1)
PASSPORT_EVENT_TYPES = frozenset(payload.event_type for payload in _PAYLOADS)


def register_passport_events(registry: EventRegistry) -> None:
    """Register the fixed Passport ledger vocabulary."""
    for payload in _PAYLOADS:
        registry.register(payload)


__all__ = [
    "PASSPORT_EVENT_TYPES",
    "PassportImportedV1",
    "PassportAdoptedV1",
    "PassportRejectedV1",
    "register_passport_events",
]
