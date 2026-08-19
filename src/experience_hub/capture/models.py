"""Strict immutable values for normalized trajectory capture."""

from __future__ import annotations

import re
from collections.abc import Sized
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import ConfigDict, field_validator, model_validator

from experience_hub import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, TypedEvidence
from experience_hub.experiences.models import ExperienceKind, VersionContent

MAX_TRAJECTORY_STEPS = 2_000
MAX_STEP_FIELD_UTF8_BYTES = 4_096
MAX_SIGNAL_LIST_ITEMS = 32
MAX_CAPTURE_EVIDENCE_ITEMS = 8
MAX_CAPTURE_EXCERPT_UTF8_BYTES = 512

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _nonblank(name: str, value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _canonical_string_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        _nonblank("Signal list item", value)
    unique = {canonical_json_bytes(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_utc_timestamp(name: str, value: datetime) -> datetime:
    try:
        return require_utc(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timezone-aware datetime") from error


class AdapterDescriptorV1(StrictModel):
    kind: Literal["generic_jsonl"]
    version: Literal[1]

    @field_validator("version", mode="before")
    @classmethod
    def require_exact_integer_version(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise ValueError("Adapter version must be integer one")
        return value


class SanitizationDeclarationV1(StrictModel):
    profile_id: str
    input_sanitized: Literal[True]

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _nonblank("Sanitization profile ID", value)

    @field_validator("input_sanitized", mode="before")
    @classmethod
    def require_explicit_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Input must be explicitly declared sanitized")
        return value


class TrajectoryField(StrEnum):
    OBSERVATION = "observation"
    ACTION = "action"
    OUTCOME = "outcome"


class SensitiveField(StrEnum):
    TRAJECTORY_ID = "trajectory_id"
    SANITIZATION_PROFILE_ID = "sanitization_profile_id"
    STEP_ID = "step_id"
    OBSERVATION = "observation"
    ACTION = "action"
    OUTCOME = "outcome"
    CANDIDATE_SIGNAL = "candidate_signal"


class SensitiveMatchV1(StrictModel):
    rule_id: str
    step_id: str
    field: SensitiveField

    @field_validator("rule_id", "step_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _nonblank("Sensitive match identifier", value)


class OutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EvidencePointerV1(StrictModel):
    step_id: str
    field: TrajectoryField

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, value: str) -> str:
        return _nonblank("Evidence step ID", value)


class CandidateSignalV1(StrictModel):
    kind: ExperienceKind
    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[EvidencePointerV1, ...]
    falsifiers: tuple[str, ...]

    @field_validator(
        "tags",
        "applicability",
        "evidence",
        "falsifiers",
        mode="before",
    )
    @classmethod
    def enforce_input_list_limit(cls, value: object) -> object:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sized):
            return value
        if len(value) > MAX_SIGNAL_LIST_ITEMS:
            raise ValueError(
                f"Signal arrays may contain at most {MAX_SIGNAL_LIST_ITEMS} items"
            )
        return value

    @field_validator("body", "summary", "mechanism")
    @classmethod
    def validate_retained_text(cls, value: str) -> str:
        return _nonblank("Candidate signal text", value)

    @field_validator("tags", "applicability", "falsifiers", mode="after")
    @classmethod
    def canonicalize_set_like_arrays(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _canonical_string_tuple(values)


class CapturedEvidenceV1(StrictModel):
    step_id: str
    field: TrajectoryField
    excerpt: str
    source_hash: str
    excerpt_hash: str

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, value: str) -> str:
        return _nonblank("Captured evidence step ID", value)

    @field_validator("excerpt")
    @classmethod
    def validate_excerpt_bound(cls, value: str) -> str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Evidence excerpt must contain valid Unicode") from error
        if len(encoded) > MAX_CAPTURE_EXCERPT_UTF8_BYTES:
            raise ValueError(
                "Evidence excerpt must be at most "
                f"{MAX_CAPTURE_EXCERPT_UTF8_BYTES} UTF-8 bytes"
            )
        return value

    @field_validator("source_hash", "excerpt_hash")
    @classmethod
    def validate_evidence_hash_shape(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Evidence hash must be a SHA-256 hex digest")
        return value


class CandidateDraftV1(StrictModel):
    model_config = ConfigDict(revalidate_instances="always")

    source_manifest_hash: str
    kind: ExperienceKind
    content: VersionContent
    content_hash: str
    evidence: tuple[CapturedEvidenceV1, ...]
    extractor_kind: Literal["deterministic_signal_v1"]
    extractor_configuration_hash: str

    @field_validator(
        "source_manifest_hash",
        "content_hash",
        "extractor_configuration_hash",
    )
    @classmethod
    def validate_hash_shape(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Candidate hash must be a SHA-256 hex digest")
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def enforce_evidence_limit(cls, value: object) -> object:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sized):
            return value
        if len(value) > MAX_CAPTURE_EVIDENCE_ITEMS:
            raise ValueError(
                "Candidate evidence may contain at most "
                f"{MAX_CAPTURE_EVIDENCE_ITEMS} items"
            )
        return value

    @model_validator(mode="after")
    def validate_reconstructed_hashes(self) -> Self:
        from experience_hub.capture.hashing import extractor_configuration_hash
        from experience_hub.experiences.content import encode_version_content

        expected_content_hash = encode_version_content(
            kind=self.kind,
            content=self.content,
        ).content_hash
        if self.content_hash != expected_content_hash:
            raise ValueError("Candidate content hash does not match content")
        if self.extractor_configuration_hash != extractor_configuration_hash():
            raise ValueError(
                "Candidate extractor configuration hash does not match configuration"
            )

        expected_evidence = VersionContent(
            body=self.content.body,
            summary=self.content.summary,
            mechanism=self.content.mechanism,
            tags=self.content.tags,
            applicability=self.content.applicability,
            evidence=tuple(
                TypedEvidence(
                    type="trajectory_field",
                    id=(
                        f"{self.source_manifest_hash}:{item.step_id}:"
                        f"{item.field.value}"
                    ),
                )
                for item in self.evidence
            ),
            falsifiers=self.content.falsifiers,
        ).evidence
        if self.content.evidence != expected_evidence:
            raise ValueError("Candidate content evidence does not match excerpts")
        return self


class TrajectoryStepV1(StrictModel):
    step_id: str
    ordinal: int
    occurred_at: datetime
    observation: str
    action: str
    outcome: str
    status: OutcomeStatus
    candidate_signal: CandidateSignalV1 | None = None

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, value: str) -> str:
        return _nonblank("Trajectory step ID", value)

    @field_validator("ordinal", mode="before")
    @classmethod
    def validate_ordinal_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Trajectory step ordinal must be an integer")
        return value

    @field_validator("ordinal")
    @classmethod
    def validate_positive_ordinal(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Trajectory step ordinal must be positive")
        return value

    @field_validator("occurred_at", mode="after")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return _require_utc_timestamp("Step timestamp", value)

    @field_validator("observation", "action", "outcome")
    @classmethod
    def validate_step_text_bound(cls, value: str) -> str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Step text must contain valid Unicode") from error
        if len(encoded) > MAX_STEP_FIELD_UTF8_BYTES:
            raise ValueError(
                f"Step text must be at most {MAX_STEP_FIELD_UTF8_BYTES} UTF-8 bytes"
            )
        return value


class TrajectoryBundleV1(StrictModel):
    model_config = ConfigDict(revalidate_instances="always")

    schema_version: Literal[1]
    adapter: AdapterDescriptorV1
    owner_agent_id: UUID
    trajectory_id: str
    source_started_at: datetime
    source_completed_at: datetime
    sanitization: SanitizationDeclarationV1
    steps: tuple[TrajectoryStepV1, ...]
    manifest_hash: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_integer_schema_version(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise ValueError("Trajectory schema version must be integer one")
        return value

    @field_validator("trajectory_id")
    @classmethod
    def validate_trajectory_id(cls, value: str) -> str:
        return _nonblank("Trajectory ID", value)

    @field_validator("source_started_at", "source_completed_at", mode="after")
    @classmethod
    def normalize_source_timestamp(cls, value: datetime) -> datetime:
        return _require_utc_timestamp("Source timestamp", value)

    @field_validator("steps", mode="before")
    @classmethod
    def enforce_input_step_limit(cls, value: object) -> object:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sized):
            return value
        if not 1 <= len(value) <= MAX_TRAJECTORY_STEPS:
            raise ValueError(
                f"Trajectory must contain 1-{MAX_TRAJECTORY_STEPS} steps"
            )
        return value

    @field_validator("manifest_hash")
    @classmethod
    def validate_manifest_hash_shape(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Trajectory manifest hash must be a SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_bundle_invariants(self) -> Self:
        if self.source_started_at > self.source_completed_at:
            raise ValueError("Source start must not follow source completion")

        step_ids: set[str] = set()
        previous_timestamp = self.source_started_at
        for expected_ordinal, step in enumerate(self.steps, start=1):
            if step.ordinal != expected_ordinal:
                raise ValueError("Step ordinals must be contiguous from one")
            if step.step_id in step_ids:
                raise ValueError("Trajectory step IDs must be unique")
            step_ids.add(step.step_id)
            within_source_bounds = (
                self.source_started_at
                <= step.occurred_at
                <= self.source_completed_at
            )
            if not within_source_bounds:
                raise ValueError("Step timestamp must be within the source time bounds")
            if step.occurred_at < previous_timestamp:
                raise ValueError("Step timestamps must follow declared step order")
            previous_timestamp = step.occurred_at

        for step in self.steps:
            if step.candidate_signal is None:
                continue
            for pointer in step.candidate_signal.evidence:
                if pointer.step_id not in step_ids:
                    raise ValueError("Evidence pointer must name a retained step")

        from experience_hub.capture.hashing import hash_trajectory_manifest

        if self.manifest_hash != hash_trajectory_manifest(self):
            raise ValueError("Trajectory manifest hash does not match bundle fields")
        return self


class PreparedCaptureV1(StrictModel):
    model_config = ConfigDict(revalidate_instances="always")

    bundle: TrajectoryBundleV1
    manifest_json: bytes
    candidates: tuple[CandidateDraftV1, ...]

    @model_validator(mode="after")
    def validate_bundle_anchors(self) -> Self:
        from experience_hub.capture.extraction import DeterministicSignalExtractor
        from experience_hub.capture.hashing import trajectory_manifest_document

        expected_manifest = canonical_json_bytes(
            trajectory_manifest_document(self.bundle)
        )
        if self.manifest_json != expected_manifest:
            raise ValueError("Prepared manifest JSON does not match bundle")
        if any(
            candidate.source_manifest_hash != self.bundle.manifest_hash
            for candidate in self.candidates
        ):
            raise ValueError("Prepared candidate manifest does not match bundle")
        expected_candidates = DeterministicSignalExtractor().extract(self.bundle)
        if self.candidates != expected_candidates:
            raise ValueError(
                "Prepared candidates do not match deterministic extraction"
            )
        return self
