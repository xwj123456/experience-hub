"""Strict, versioned contracts for deterministic replay experiments."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from experience_hub.domain import StrictModel
from experience_hub.retrieval.contracts import MAX_CONTENT_BUDGET_BYTES
from experience_hub.retrieval.ranking import MAX_RETRIEVAL_LIMIT, RetrievalMode

_STRICT = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
    revalidate_instances="always",
)
_LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)

MAX_REPLAY_CASES = 256
MAX_REPLAY_INPUT_BYTES = 2 * 1024 * 1024
MAX_REPLAY_OUTPUT_BYTES = 2 * 1024 * 1024

NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveLimit = Annotated[StrictInt, Field(ge=1, le=MAX_RETRIEVAL_LIMIT)]
ContentBudget = Annotated[
    StrictInt,
    Field(ge=1, le=MAX_CONTENT_BUDGET_BYTES),
]
UtilityMicros = Annotated[StrictInt, Field(ge=0, le=1_000_000)]


class _ReplayModel(StrictModel):
    model_config = _STRICT


class PolicyArmKind(StrEnum):
    NO_MEMORY = "no_memory"
    EXPERIENCE_HUB = "experience_hub"


def _label(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase replay label")
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _utc_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, str):
        if not _UTC_TIMESTAMP.fullmatch(value):
            raise ValueError(f"{field_name} must be a canonical UTC timestamp")
        return datetime.fromisoformat(value.removesuffix("Z")).replace(tzinfo=UTC)
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a UTC datetime")
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be a UTC datetime")
    return value


def _label_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an ordered array")
    labels = tuple(_label(item, field_name=field_name) for item in value)
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field_name} must contain unique labels")
    return labels


def _policy_arm_id(value: Any) -> str:
    allowed = {item.value for item in PolicyArmKind}
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("arm_id must name a supported policy arm")
    return value


class ExperienceLabelV1(_ReplayModel):
    """A portable fixture label bound to one concrete experience identity."""

    label: str
    experience_id: UUID

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: Any) -> str:
        return _label(value, field_name="label")


class ReplayDatasetDescriptorV1(_ReplayModel):
    schema_version: Literal[1]
    dataset_id: str
    cases_file: str
    cases_sha256: str

    @field_validator("dataset_id", mode="before")
    @classmethod
    def validate_dataset_id(cls, value: Any) -> str:
        return _label(value, field_name="dataset_id")

    @field_validator("cases_file", mode="before")
    @classmethod
    def validate_cases_file(cls, value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("cases_file must be one plain filename")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("cases_file must be one plain filename")
        return value

    @field_validator("cases_sha256", mode="before")
    @classmethod
    def validate_cases_sha256(cls, value: Any) -> str:
        return _sha256(value, field_name="cases_sha256")


class PolicyArmDescriptorV1(_ReplayModel):
    schema_version: Literal[1]
    arm_id: str
    kind: PolicyArmKind
    required: StrictBool

    @field_validator("arm_id", mode="before")
    @classmethod
    def validate_arm_id(cls, value: Any) -> str:
        return _policy_arm_id(value)


class OracleDescriptorV1(_ReplayModel):
    schema_version: Literal[1]
    kind: Literal["retrieval_labels"]
    version: Literal[1]


def _require_ordered_policy_arms(
    arms: tuple[PolicyArmDescriptorV1, ...],
) -> None:
    required = tuple(arm for arm in arms if arm.required)
    expected = (PolicyArmKind.NO_MEMORY, PolicyArmKind.EXPERIENCE_HUB)
    if len(arms) != 2 or len(required) != 2:
        raise ValueError("arms must contain exactly two ordered required arms")
    if tuple(arm.kind for arm in arms) != expected:
        raise ValueError("arms must use the ordered required policy kinds")
    arm_ids = tuple(arm.arm_id for arm in arms)
    expected_ids = tuple(item.value for item in expected)
    if arm_ids != expected_ids:
        raise ValueError("arms must use the ordered required arm IDs")


class ReplayCaseV1(_ReplayModel):
    schema_version: Literal[1]
    case_id: str
    owner_agent_id: UUID
    query: str
    mode: RetrievalMode
    tags: tuple[str, ...]
    mechanism_cues: tuple[str, ...]
    limit: PositiveLimit
    content_budget_bytes: ContentBudget
    expand_cold: StrictBool
    expected: tuple[ExperienceLabelV1, ...]
    forbidden: tuple[ExperienceLabelV1, ...]

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_case_id(cls, value: Any) -> str:
        return _label(value, field_name="case_id")

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("query must be a nonempty trimmed string")
        return value

    @field_validator("tags", "mechanism_cues", mode="before")
    @classmethod
    def validate_label_sequences(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _label_tuple(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_expected_and_forbidden(self) -> Self:
        expected_labels = tuple(item.label for item in self.expected)
        forbidden_labels = tuple(item.label for item in self.forbidden)
        expected_ids = tuple(item.experience_id for item in self.expected)
        forbidden_ids = tuple(item.experience_id for item in self.forbidden)
        if len(expected_labels) != len(set(expected_labels)):
            raise ValueError("expected labels must be unique")
        if len(forbidden_labels) != len(set(forbidden_labels)):
            raise ValueError("forbidden labels must be unique")
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected experience IDs must be unique")
        if len(forbidden_ids) != len(set(forbidden_ids)):
            raise ValueError("forbidden experience IDs must be unique")
        if set(expected_ids) & set(forbidden_ids):
            raise ValueError("expected and forbidden experience IDs must not overlap")
        if set(expected_labels) & set(forbidden_labels):
            raise ValueError("expected and forbidden labels must not overlap")
        return self


class ReplayManifestV1(_ReplayModel):
    schema_version: Literal[1]
    experiment_id: str
    dataset: ReplayDatasetDescriptorV1
    snapshot_binding: Literal["validated_source"]
    frozen_at: datetime
    seed: NonnegativeInt
    arms: tuple[PolicyArmDescriptorV1, ...]
    oracle: OracleDescriptorV1
    evidence_schema_version: Literal[1]
    profile_schema_version: Literal[1]
    deterministic_replay_runs: Literal[2]

    @field_validator("experiment_id", mode="before")
    @classmethod
    def validate_experiment_id(cls, value: Any) -> str:
        return _label(value, field_name="experiment_id")

    @field_validator("frozen_at", mode="before")
    @classmethod
    def validate_frozen_at(cls, value: Any) -> datetime:
        return _utc_datetime(value, field_name="frozen_at")

    @model_validator(mode="after")
    def validate_required_arm_order(self) -> Self:
        _require_ordered_policy_arms(self.arms)
        return self


class ArmObservationV1(_ReplayModel):
    schema_version: Literal[1]
    returned_labels: tuple[str, ...]
    unmapped_count: NonnegativeInt

    @field_validator("returned_labels", mode="before")
    @classmethod
    def validate_returned_labels(cls, value: Any) -> tuple[str, ...]:
        return _label_tuple(value, field_name="returned_labels")


class OracleEvidenceV1(_ReplayModel):
    """Structured retrieval-oracle evidence computed outside a policy arm."""

    schema_version: Literal[1]
    expected_found: tuple[str, ...]
    expected_missing: tuple[str, ...]
    forbidden_found: tuple[str, ...]
    unmapped_count: NonnegativeInt
    utility_micros: UtilityMicros

    @field_validator(
        "expected_found",
        "expected_missing",
        "forbidden_found",
        mode="before",
    )
    @classmethod
    def validate_label_sequences(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _label_tuple(value, field_name=info.field_name)


class ArmEvidenceV1(_ReplayModel):
    schema_version: Literal[1]
    arm_id: str
    status: Literal["complete", "failed"]
    observation: ArmObservationV1 | None
    utility_micros: UtilityMicros | None
    error_code: str | None
    error_stage: str | None

    @field_validator("arm_id", mode="before")
    @classmethod
    def validate_arm_id(cls, value: Any) -> str:
        return _policy_arm_id(value)


class CaseEvidenceV1(_ReplayModel):
    schema_version: Literal[1]
    case_id: str
    status: Literal["complete", "incomplete"]
    arms: tuple[ArmEvidenceV1, ...]
    delta_utility_micros: StrictInt | None

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_case_id(cls, value: Any) -> str:
        return _label(value, field_name="case_id")

    @model_validator(mode="after")
    def validate_unique_arm_ids(self) -> Self:
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("arms must contain unique arm IDs")
        return self


class ResolvedReplayManifestV1(_ReplayModel):
    schema_version: Literal[1]
    experiment_id: str
    manifest_sha256: str
    dataset_id: str
    cases_sha256: str
    snapshot_sha256: str
    source_schema_revision: NonnegativeInt
    frozen_at: datetime
    seed: NonnegativeInt
    policy_arms: tuple[PolicyArmDescriptorV1, ...]
    oracle: OracleDescriptorV1
    evidence_schema_version: Literal[1]
    profile_schema_version: Literal[1]

    @field_validator("experiment_id", "dataset_id", mode="before")
    @classmethod
    def validate_labels(cls, value: Any, info: Any) -> str:
        return _label(value, field_name=info.field_name)

    @field_validator(
        "manifest_sha256",
        "cases_sha256",
        "snapshot_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: Any, info: Any) -> str:
        return _sha256(value, field_name=info.field_name)

    @field_validator("frozen_at", mode="before")
    @classmethod
    def validate_frozen_at(cls, value: Any) -> datetime:
        return _utc_datetime(value, field_name="frozen_at")

    @model_validator(mode="after")
    def validate_policy_arm_order(self) -> Self:
        _require_ordered_policy_arms(self.policy_arms)
        return self


class ReplayEvidenceDataV1(_ReplayModel):
    schema_version: Literal[1]
    resolved_manifest: ResolvedReplayManifestV1
    cases: tuple[CaseEvidenceV1, ...]
    comparison_complete: StrictBool
    source_unchanged: StrictBool
    clone_isolation_verified: StrictBool
    deterministic_replay_match: StrictBool
    valid: StrictBool

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("cases must contain unique case IDs")
        return self


class ReplayEvidenceReportV1(_ReplayModel):
    data: ReplayEvidenceDataV1


class ReplayProfileDataV1(_ReplayModel):
    schema_version: Literal[1]
    experiment_id: str
    profile_complete: StrictBool
    wall_duration_ns: NonnegativeInt | None
    database_bytes: NonnegativeInt

    @field_validator("experiment_id", mode="before")
    @classmethod
    def validate_experiment_id(cls, value: Any) -> str:
        return _label(value, field_name="experiment_id")


class ReplayProfileReportV1(_ReplayModel):
    data: ReplayProfileDataV1
