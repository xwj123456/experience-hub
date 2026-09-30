"""Frozen values for one unsigned, offline evidence Passport."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from experience_hub.capture.models import TrajectoryField
from experience_hub.domain import StrictModel, TypedEvidence
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    VersionContent,
)

MAX_PASSPORT_BYTES = 512 * 1024
MAX_PASSPORT_DEPTH = 16
MAX_PASSPORT_HOPS = 4
type Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PassportValue(StrictModel):
    model_config = ConfigDict(strict=True, revalidate_instances="always")


class PassportState(StrEnum):
    PENDING = "pending"
    ADOPTED = "adopted"
    REJECTED = "rejected"


def _bounded_text(value: str, limit: int, *, nonblank: bool) -> str:
    # Count first: malicious callers must not cause an unbounded UTF-8 allocation.
    size = 0
    for character in value:
        point = ord(character)
        if 0xD800 <= point <= 0xDFFF:
            raise ValueError("Invalid Unicode")
        size += 1 if point < 128 else 2 if point < 2048 else 3 if point < 65536 else 4
        if size > limit:
            raise ValueError("Text exceeds its UTF-8 limit")
    if nonblank and not value.strip():
        raise ValueError("Text must not be blank")
    return value


class PassportSubjectV1(PassportValue):
    source_agent_id: UUID
    source_experience_id: UUID
    source_version_id: UUID
    source_origin: ExperienceOrigin
    kind: ExperienceKind
    content: VersionContent
    content_hash: Sha256

    @field_validator("content", mode="before")
    @classmethod
    def retain_json_arrays(cls, value: object) -> object:
        # VersionContent's legacy before-validator switches JSON arrays into
        # Python mode. Preserve exact scalar types while adapting only arrays.
        if isinstance(value, Mapping):
            return {
                key: tuple(item)
                if key in {"tags", "applicability", "evidence", "falsifiers"}
                and isinstance(item, list)
                else item
                for key, item in value.items()
            }
        return value


class ReferenceOnlySnapshotV1(PassportValue):
    mode: Literal["reference_only"]
    reference: TypedEvidence


class EmbeddedExcerptSnapshotV1(PassportValue):
    mode: Literal["embedded_excerpt"]
    reference: TypedEvidence
    excerpt: str
    excerpt_hash: Sha256
    source_hash: Sha256
    source_manifest_hash: Sha256
    step_id: str
    field: TrajectoryField

    @field_validator("excerpt")
    @classmethod
    def bounded_excerpt(cls, value: str) -> str:
        return _bounded_text(value, 512, nonblank=False)

    @field_validator("step_id")
    @classmethod
    def bounded_step(cls, value: str) -> str:
        return _bounded_text(value, MAX_PASSPORT_BYTES, nonblank=True)


type EvidenceSnapshotV1 = Annotated[
    ReferenceOnlySnapshotV1 | EmbeddedExcerptSnapshotV1,
    Field(discriminator="mode"),
]


class PassportHopV1(PassportValue):
    source_agent_id: UUID
    source_experience_id: UUID
    source_version_id: UUID
    source_origin: ExperienceOrigin
    content_hash: Sha256
    parent_passport_hash: Sha256 | None


class PassportProvenanceV1(PassportValue):
    scope: Literal["passport_transfers_only"]
    hops: Annotated[tuple[PassportHopV1, ...], Field(min_length=1, max_length=4)]
    origin_fingerprint: Sha256


class PassportDeclarationV1(PassportValue):
    input_sanitized: Literal[True]
    profile_id: str
    sharing_authorized: Literal[True]

    @field_validator("input_sanitized", "sharing_authorized", mode="before")
    @classmethod
    def require_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Explicit sharing and sanitization declarations required")
        return value

    @field_validator("profile_id")
    @classmethod
    def bounded_profile(cls, value: str) -> str:
        return _bounded_text(value, 100, nonblank=True)


class EvidencePassportV1(PassportValue):
    schema_version: Literal[1]
    format: Literal["experience_passport"]
    subject: PassportSubjectV1
    evidence_snapshots: Annotated[tuple[EvidenceSnapshotV1, ...], Field(max_length=32)]
    provenance: PassportProvenanceV1
    declaration: PassportDeclarationV1
    passport_hash: Sha256

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_one(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Schema version must be integer one")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        # Delayed import keeps transport values independent of module import order.
        from experience_hub.passports.codec import validate_document

        validate_document(self)
        return self


class PassportInspectionV1(PassportValue):
    passport_hash: Sha256
    source_agent_id: UUID
    kind: ExperienceKind
    embedded_excerpt_count: Annotated[int, Field(ge=0)]
    reference_only_count: Annotated[int, Field(ge=0)]
    unavailable_preimage_count: Annotated[int, Field(ge=0)]
    publisher_identity: Literal["unverified"] = "unverified"
    semantic_assessment: Literal["not_assessed"] = "not_assessed"
    persisted: Literal[False] = False

    @field_validator("persisted", mode="before")
    @classmethod
    def require_false(cls, value: object) -> object:
        if value is not False:
            raise ValueError("Inspection cannot persist source data")
        return value
