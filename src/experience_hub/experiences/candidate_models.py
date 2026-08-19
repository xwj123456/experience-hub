"""Stable values for quarantined candidate decisions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Final, Literal
from uuid import UUID

from pydantic import field_validator, model_validator

from experience_hub.capture.models import CapturedEvidenceV1
from experience_hub.domain import StrictModel, StructuredReason
from experience_hub.experiences.models import ExperienceKind, VersionContent

type CandidateDecisionScope = Literal[
    "experience.candidate.adopt",
    "experience.candidate.reject",
]

CANDIDATE_ADOPT_SCOPE: Final[Literal["experience.candidate.adopt"]] = (
    "experience.candidate.adopt"
)
CANDIDATE_REJECT_SCOPE: Final[Literal["experience.candidate.reject"]] = (
    "experience.candidate.reject"
)


class CandidateDecision(StrEnum):
    """The replayable owner decision for one extracted candidate."""

    PENDING = "pending"
    ADOPTED = "adopted"
    REJECTED = "rejected"


class CandidateViewV1(StrictModel):
    candidate_id: UUID
    bundle_id: UUID
    owner_agent_id: UUID
    decision: CandidateDecision
    kind: ExperienceKind
    content: VersionContent
    content_hash: str
    evidence: tuple[CapturedEvidenceV1, ...]
    extractor_kind: str
    extractor_configuration_hash: str
    resulting_experience_id: UUID | None
    resulting_version_id: UUID | None
    reason: StructuredReason | None
    created_at: datetime
    decided_at: datetime | None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> CandidateViewV1:
        has_result = (
            self.resulting_experience_id is not None
            and self.resulting_version_id is not None
        )
        has_no_result = (
            self.resulting_experience_id is None
            and self.resulting_version_id is None
        )
        if self.decision is CandidateDecision.PENDING:
            valid = (
                has_no_result
                and self.reason is None
                and self.decided_at is None
            )
        elif self.decision is CandidateDecision.ADOPTED:
            valid = has_result and self.reason is None and self.decided_at is not None
        else:
            valid = (
                has_no_result
                and self.reason is not None
                and self.decided_at is not None
            )
        if not valid:
            raise ValueError("Candidate decision fields are inconsistent")
        return self


class CandidatePageV1(StrictModel):
    items: tuple[CandidateViewV1, ...]
    next_cursor: str | None


class AdoptCandidate(StrictModel):
    owner_agent_id: UUID
    candidate_id: UUID
    importance: float
    confidence: float

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def validate_score(cls, value: Any) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Candidate scores must be finite values from 0 to 1")
        return float(value)


class RejectCandidate(StrictModel):
    owner_agent_id: UUID
    candidate_id: UUID
    reason: StructuredReason


__all__ = [
    "CANDIDATE_ADOPT_SCOPE",
    "CANDIDATE_REJECT_SCOPE",
    "AdoptCandidate",
    "CandidateDecision",
    "CandidateDecisionScope",
    "CandidatePageV1",
    "CandidateViewV1",
    "RejectCandidate",
]
