"""Public command and sharing contracts for immutable experiences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ConfigDict

from experience_hub.domain import StrictModel, StructuredReason, TypedEvidence
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    Temperature,
    VersionContent,
)

if TYPE_CHECKING:
    from experience_hub.experiences.events import VersionLinkRefV1


class VersionLinkInput(StrictModel):
    """One complete version-scoped dependency link supplied by a caller."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        strict=True,
    )

    target_experience_id: UUID
    relation: LinkRelation


@dataclass(frozen=True, slots=True)
class CreateExperience:
    owner_agent_id: UUID
    kind: ExperienceKind
    content: VersionContent
    importance: float = 0.35
    confidence: float = 0.50
    links: tuple[VersionLinkInput, ...] = ()


@dataclass(frozen=True, slots=True)
class CreateExperienceVersion:
    owner_agent_id: UUID
    experience_id: UUID
    content: VersionContent
    links: tuple[VersionLinkInput, ...] = ()


type ExperienceMutationReason = StructuredReason | str | None


def _validate_mutation_identity(
    owner_agent_id: UUID,
    experience_id: UUID,
) -> None:
    if not isinstance(owner_agent_id, UUID):
        raise ValueError("owner_agent_id must be a UUID")
    if not isinstance(experience_id, UUID):
        raise ValueError("experience_id must be a UUID")


def _validate_reason(reason: ExperienceMutationReason) -> None:
    if reason is not None and not isinstance(reason, (str, StructuredReason)):
        raise ValueError("reason must be a string, StructuredReason, or None")


def _validate_evidence(evidence: tuple[TypedEvidence, ...]) -> None:
    if not isinstance(evidence, tuple) or any(
        not isinstance(item, TypedEvidence) for item in evidence
    ):
        raise ValueError("evidence must be a tuple of TypedEvidence values")


@dataclass(frozen=True, slots=True)
class ConfirmExperience:
    owner_agent_id: UUID
    experience_id: UUID
    reason: ExperienceMutationReason = None
    evidence: tuple[TypedEvidence, ...] = ()

    def __post_init__(self) -> None:
        _validate_mutation_identity(self.owner_agent_id, self.experience_id)
        _validate_reason(self.reason)
        _validate_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class RefuteExperience:
    owner_agent_id: UUID
    experience_id: UUID
    reason: ExperienceMutationReason = None
    evidence: tuple[TypedEvidence, ...] = ()

    def __post_init__(self) -> None:
        _validate_mutation_identity(self.owner_agent_id, self.experience_id)
        _validate_reason(self.reason)
        _validate_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class PinExperience:
    owner_agent_id: UUID
    experience_id: UUID
    reason: ExperienceMutationReason = None

    def __post_init__(self) -> None:
        _validate_mutation_identity(self.owner_agent_id, self.experience_id)
        _validate_reason(self.reason)


@dataclass(frozen=True, slots=True)
class UnpinExperience:
    owner_agent_id: UUID
    experience_id: UUID
    reason: ExperienceMutationReason = None

    def __post_init__(self) -> None:
        _validate_mutation_identity(self.owner_agent_id, self.experience_id)
        _validate_reason(self.reason)


@dataclass(frozen=True, slots=True)
class RestoreExperience:
    owner_agent_id: UUID
    experience_id: UUID
    reason: ExperienceMutationReason = None

    def __post_init__(self) -> None:
        _validate_mutation_identity(self.owner_agent_id, self.experience_id)
        _validate_reason(self.reason)


@dataclass(frozen=True, slots=True)
class ExperienceDraft:
    owner_agent_id: UUID
    actor_agent_id: UUID
    kind: ExperienceKind
    origin: ExperienceOrigin
    content: VersionContent
    importance: float
    confidence: float
    source_trust: float
    initial_temperature: Temperature
    links: tuple[VersionLinkInput, ...]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExperienceCreation:
    experience_id: UUID
    version_id: UUID
    content_hash: str


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    experience_id: UUID
    owner_agent_id: UUID
    current_version_id: UUID
    current_content_hash: str
    temperature: Temperature


@dataclass(frozen=True, slots=True)
class ShareableExperienceVersion:
    experience_id: UUID
    owner_agent_id: UUID
    origin: ExperienceOrigin
    kind: ExperienceKind
    version_id: UUID
    content: VersionContent
    content_hash: str
    confidence: float
    temperature: Temperature
    latest_causal_at: datetime


def canonicalize_version_links(
    *,
    source_experience_id: UUID,
    links: tuple[VersionLinkInput, ...],
) -> tuple[VersionLinkRefV1, ...]:
    """Validate and retain one deterministic complete link set."""
    from experience_hub.experiences.events import VersionLinkRefV1

    pairs: set[tuple[UUID, LinkRelation]] = set()
    canonical: list[VersionLinkRefV1] = []
    for link in links:
        if link.target_experience_id == source_experience_id:
            raise ValueError("An experience version cannot link to itself")
        pair = (link.target_experience_id, link.relation)
        if pair in pairs:
            raise ValueError("Duplicate experience version link")
        pairs.add(pair)
        canonical.append(
            VersionLinkRefV1(
                target_experience_id=link.target_experience_id,
                relation=link.relation,
            )
        )
    return tuple(
        sorted(
            canonical,
            key=lambda item: (
                item.target_experience_id.bytes,
                item.relation.value,
            ),
        )
    )


__all__ = [
    "ConfirmExperience",
    "CreateExperience",
    "CreateExperienceVersion",
    "ExperienceCreation",
    "ExperienceDraft",
    "ExperienceMutationReason",
    "ExperienceRecord",
    "PinExperience",
    "RefuteExperience",
    "RestoreExperience",
    "ShareableExperienceVersion",
    "UnpinExperience",
    "VersionLinkInput",
    "canonicalize_version_links",
]
