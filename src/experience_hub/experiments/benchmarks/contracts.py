"""Strict, versioned contracts for the ExperienceBench-S pilot."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from experience_hub.domain import StrictModel
from experience_hub.experiences.models import ExperienceKind, Temperature
from experience_hub.experiments.contracts import (
    MAX_REPLAY_CASES,
    MAX_REPLAY_INPUT_BYTES,
    MAX_REPLAY_OUTPUT_BYTES,
    ContentBudget,
    NonnegativeInt,
    PositiveLimit,
    UtilityMicros,
)
from experience_hub.retrieval.ranking import RetrievalMode

_STRICT = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
    revalidate_instances="always",
)
_LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR_CODE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)

MAX_BENCHMARK_CASES = MAX_REPLAY_CASES
MAX_BENCHMARK_INPUT_BYTES = MAX_REPLAY_INPUT_BYTES
MAX_BENCHMARK_TOTAL_INPUT_BYTES = 6 * 1024 * 1024
MAX_BENCHMARK_SOURCE_RECORDS = 4_096
MAX_BENCHMARK_OUTPUT_BYTES = MAX_REPLAY_OUTPUT_BYTES


class BenchmarkArmKind(StrEnum):
    NO_MEMORY = "no_memory"
    RECENT_NOTES = "recent_notes"
    SQLITE_BM25 = "sqlite_bm25"
    EXPERIENCE_HUB = "experience_hub"


BENCHMARK_ARM_ORDER = tuple(BenchmarkArmKind)


class BenchmarkSourceClass(StrEnum):
    PUBLIC_AUTHORED = "public_authored"
    REVIEWED_ABSTRACTION = "reviewed_abstraction"


class BenchmarkStratum(StrEnum):
    RECURRING_WORKFLOW = "recurring_workflow"
    ENVIRONMENT_GOTCHA = "environment_gotcha"
    STATE_CHANGE = "state_change"
    FAILURE_RECOVERY = "failure_recovery"
    IRRELEVANT_DISTRACTOR = "irrelevant_distractor"


BENCHMARK_STRATUM_ORDER = tuple(BenchmarkStratum)


class BenchmarkLanguage(StrEnum):
    CHINESE = "zh"
    ENGLISH = "en"
    MIXED = "mixed"


class BenchmarkCheckpointPredicate(StrEnum):
    REQUIRED_SET = "required_set"
    ORDERED_SUBSEQUENCE = "ordered_subsequence"


class _BenchmarkModel(StrictModel):
    model_config = _STRICT


def _label(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase benchmark label")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _utc_datetime(value: object, *, field_name: str) -> datetime:
    if isinstance(value, str):
        if not _UTC_TIMESTAMP.fullmatch(value):
            raise ValueError(f"{field_name} must be a canonical UTC timestamp")
        return datetime.fromisoformat(value.removesuffix("Z")).replace(tzinfo=UTC)
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a UTC datetime")
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be a UTC datetime")
    return value


def _label_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an ordered array")
    labels = tuple(_label(item, field_name=field_name) for item in value)
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field_name} must contain unique labels")
    return labels


class BenchmarkFileDescriptorV1(_BenchmarkModel):
    file: str
    sha256: str

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("file must be one plain filename")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("file must be one plain filename")
        return value

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> str:
        return _sha256(value, field_name="sha256")


class BenchmarkCompositionV1(_BenchmarkModel):
    case_count: Annotated[StrictInt, Field(ge=1, le=MAX_REPLAY_CASES)]
    public_authored: NonnegativeInt
    reviewed_abstractions: NonnegativeInt
    cases_per_stratum: NonnegativeInt
    chinese: NonnegativeInt
    english: NonnegativeInt
    mixed: NonnegativeInt

    @model_validator(mode="after")
    def validate_pilot_composition(self) -> Self:
        if self.case_count != 30:
            raise ValueError("case_count must be exactly 30 for the pilot")
        if self.public_authored + self.reviewed_abstractions != self.case_count:
            raise ValueError("source-class counts must total case_count")
        if self.cases_per_stratum * len(BENCHMARK_STRATUM_ORDER) != self.case_count:
            raise ValueError("stratum counts must total case_count")
        if self.chinese + self.english + self.mixed != self.case_count:
            raise ValueError("language counts must total case_count")
        return self


class BenchmarkArmDescriptorV1(_BenchmarkModel):
    schema_version: Literal[1]
    arm_id: str
    kind: BenchmarkArmKind
    required: StrictBool

    @field_validator("arm_id", mode="before")
    @classmethod
    def validate_arm_id(cls, value: object) -> str:
        allowed = {item.value for item in BenchmarkArmKind}
        if not isinstance(value, str) or value not in allowed:
            raise ValueError("arm_id must name a supported benchmark arm")
        return value

    @model_validator(mode="after")
    def validate_matching_identifier(self) -> Self:
        if self.arm_id != self.kind.value:
            raise ValueError("arm_id must match its benchmark arm kind")
        return self


class BenchmarkWeightedLabelV1(_BenchmarkModel):
    label: str
    weight_micros: NonnegativeInt

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: object) -> str:
        return _label(value, field_name="label")


class BenchmarkCheckpointV1(_BenchmarkModel):
    predicate: BenchmarkCheckpointPredicate
    labels: tuple[str, ...]
    weight_micros: NonnegativeInt

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, value: object) -> tuple[str, ...]:
        return _label_tuple(value, field_name="labels")

    @model_validator(mode="after")
    def validate_positive_checkpoint(self) -> Self:
        if not self.labels or self.weight_micros <= 0:
            raise ValueError("checkpoints require labels and a positive weight")
        return self


class BenchmarkCaseV1(_BenchmarkModel):
    schema_version: Literal[1]
    case_id: str
    source_class: BenchmarkSourceClass
    review_status: Literal["authored", "maintainer_reviewed"]
    stratum: BenchmarkStratum
    language: BenchmarkLanguage
    difficulty: Literal["A", "B", "I"]
    owner_label: str
    query: str
    mode: RetrievalMode
    tags: tuple[str, ...]
    mechanism_cues: tuple[str, ...]
    limit: PositiveLimit
    content_budget_bytes: ContentBudget
    source_labels: tuple[str, ...]
    required: tuple[BenchmarkWeightedLabelV1, ...]
    optional: tuple[BenchmarkWeightedLabelV1, ...]
    forbidden: tuple[BenchmarkWeightedLabelV1, ...]
    stale: tuple[BenchmarkWeightedLabelV1, ...]
    misleading: tuple[BenchmarkWeightedLabelV1, ...]
    checkpoints: tuple[BenchmarkCheckpointV1, ...]
    oracle_version: Literal[1]

    @field_validator("case_id", "owner_label", mode="before")
    @classmethod
    def validate_labels(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "label")
        return _label(value, field_name=field_name)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("query must be a nonempty trimmed string")
        return value

    @field_validator("tags", "mechanism_cues", "source_labels", mode="before")
    @classmethod
    def validate_label_sequences(
        cls, value: object, info: object
    ) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "labels")
        return _label_tuple(value, field_name=field_name)

    @property
    def penalties(self) -> tuple[BenchmarkWeightedLabelV1, ...]:
        return (*self.forbidden, *self.stale, *self.misleading)

    @model_validator(mode="after")
    def validate_rubric(self) -> Self:
        semantic_groups = (
            self.required,
            self.optional,
            self.forbidden,
            self.stale,
            self.misleading,
        )
        semantic_labels = tuple(
            item.label for group in semantic_groups for item in group
        )
        if len(semantic_labels) != len(set(semantic_labels)):
            raise ValueError("semantic label groups must be pairwise disjoint")
        if not self.required:
            raise ValueError("required must contain at least one label")
        if any(item.weight_micros <= 0 for item in self.required):
            raise ValueError("required labels must have positive weights")
        if any(item.weight_micros != 0 for item in self.optional):
            raise ValueError("optional labels must have zero weights")
        if any(item.weight_micros <= 0 for item in self.penalties):
            raise ValueError("penalty labels must have positive weights")
        if sum(item.weight_micros for item in self.required) != 450_000:
            raise ValueError("required weights must total 450000")
        if sum(item.weight_micros for item in self.penalties) != 300_000:
            raise ValueError("penalty weights must total 300000")
        if not self.checkpoints or (
            sum(item.weight_micros for item in self.checkpoints) != 150_000
        ):
            raise ValueError("checkpoint weights must total 150000")
        required_labels = {item.label for item in self.required}
        if any(
            not set(checkpoint.labels).issubset(required_labels)
            for checkpoint in self.checkpoints
        ):
            raise ValueError("checkpoint labels must be declared required labels")
        if not set(semantic_labels).issubset(set(self.source_labels)):
            raise ValueError("source_labels must include every semantic label")
        if self.source_class is BenchmarkSourceClass.PUBLIC_AUTHORED:
            if self.review_status != "authored":
                raise ValueError("public cases must be declared authored")
        elif self.review_status != "maintainer_reviewed":
            raise ValueError("reviewed abstractions require maintainer review")
        return self


class BenchmarkSourceAgentV1(_BenchmarkModel):
    schema_version: Literal[1]
    record_type: Literal["agent"]
    label: str

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: object) -> str:
        return _label(value, field_name="label")


class BenchmarkSourceEvidenceV1(_BenchmarkModel):
    """One logical source-evidence reference without a storage identity."""

    type: str
    label: str

    @field_validator("type", "label", mode="before")
    @classmethod
    def validate_labels(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "label")
        return _label(value, field_name=field_name)


class _BenchmarkSourceContentV1(_BenchmarkModel):
    label: str
    owner_label: str
    created_at: datetime
    kind: ExperienceKind
    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    falsifiers: tuple[str, ...]

    @field_validator("label", "owner_label", mode="before")
    @classmethod
    def validate_labels(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "label")
        return _label(value, field_name=field_name)

    @field_validator("created_at", mode="before")
    @classmethod
    def validate_created_at(cls, value: object) -> datetime:
        return _utc_datetime(value, field_name="created_at")

    @field_validator("body", "summary", "mechanism", mode="before")
    @classmethod
    def validate_nonempty_text(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "content")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be nonempty")
        return value

    @field_validator("tags", "applicability", "falsifiers", mode="before")
    @classmethod
    def validate_text_sequences(
        cls, value: object, info: object
    ) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "content")
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be an ordered array")
        result = tuple(value)
        if any(not isinstance(item, str) or not item.strip() for item in result):
            raise ValueError(f"{field_name} must contain nonempty strings")
        if len(result) != len(set(result)):
            raise ValueError(f"{field_name} must contain unique values")
        return result


class BenchmarkSourceExperienceV1(_BenchmarkSourceContentV1):
    schema_version: Literal[1]
    record_type: Literal["experience"]
    temperature: Temperature
    evidence: tuple[BenchmarkSourceEvidenceV1, ...]
    importance_micros: UtilityMicros
    confidence_micros: UtilityMicros


class BenchmarkSourceCandidateV1(_BenchmarkSourceContentV1):
    schema_version: Literal[1]
    record_type: Literal["candidate"]


BenchmarkSourceRecordV1 = Annotated[
    BenchmarkSourceAgentV1 | BenchmarkSourceExperienceV1 | BenchmarkSourceCandidateV1,
    Field(discriminator="record_type"),
]


class BenchmarkArmObservationV1(_BenchmarkModel):
    schema_version: Literal[1]
    returned_labels: tuple[str, ...]
    selected_content_bytes: NonnegativeInt

    @field_validator("returned_labels", mode="before")
    @classmethod
    def validate_returned_labels(cls, value: object) -> tuple[str, ...]:
        return _label_tuple(value, field_name="returned_labels")


class BenchmarkCategoryScoresV1(_BenchmarkModel):
    schema_version: Literal[1]
    required_coverage_micros: UtilityMicros
    avoidance_micros: UtilityMicros
    recovery_order_micros: UtilityMicros
    evidence_efficiency_micros: UtilityMicros
    utility_micros: UtilityMicros

    @model_validator(mode="after")
    def validate_utility_sum(self) -> Self:
        if self.utility_micros != sum(
            (
                self.required_coverage_micros,
                self.avoidance_micros,
                self.recovery_order_micros,
                self.evidence_efficiency_micros,
            )
        ):
            raise ValueError("utility_micros must equal category score sum")
        return self


class BenchmarkOracleEvidenceV1(_BenchmarkModel):
    schema_version: Literal[1]
    returned_labels: tuple[str, ...]
    required_labels: tuple[str, ...]
    optional_labels: tuple[str, ...]
    violation_labels: tuple[str, ...]
    satisfied_checkpoint_indexes: tuple[NonnegativeInt, ...]
    scores: BenchmarkCategoryScoresV1

    @field_validator(
        "returned_labels",
        "required_labels",
        "optional_labels",
        "violation_labels",
        mode="before",
    )
    @classmethod
    def validate_label_sequences(
        cls, value: object, info: object
    ) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "labels")
        return _label_tuple(value, field_name=field_name)

    @field_validator("satisfied_checkpoint_indexes", mode="before")
    @classmethod
    def validate_checkpoint_indexes(cls, value: object) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("satisfied_checkpoint_indexes must be an ordered array")
        indexes = tuple(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in indexes):
            raise ValueError("satisfied_checkpoint_indexes must contain integers")
        if any(item < 0 for item in indexes) or indexes != tuple(sorted(set(indexes))):
            raise ValueError("satisfied_checkpoint_indexes must be unique and ordered")
        return indexes


class BenchmarkArmEvidenceV1(_BenchmarkModel):
    schema_version: Literal[1]
    arm_id: str
    status: Literal["complete", "failed"]
    observation: BenchmarkArmObservationV1 | None
    oracle: BenchmarkOracleEvidenceV1 | None
    error_code: str | None
    error_stage: str | None

    @field_validator("arm_id", mode="before")
    @classmethod
    def validate_arm_id(cls, value: object) -> str:
        allowed = {item.value for item in BenchmarkArmKind}
        if not isinstance(value, str) or value not in allowed:
            raise ValueError("arm_id must name a supported benchmark arm")
        return value

    @field_validator("error_code", "error_stage", mode="before")
    @classmethod
    def validate_error_labels(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _ERROR_CODE.fullmatch(value):
            raise ValueError("error fields must be stable lowercase error codes")
        return value

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        complete = self.status == "complete"
        has_any_result = self.observation is not None or self.oracle is not None
        has_complete_result = self.observation is not None and self.oracle is not None
        has_any_error = self.error_code is not None or self.error_stage is not None
        has_complete_error = (
            self.error_code is not None and self.error_stage is not None
        )
        if complete and (not has_complete_result or has_any_error):
            raise ValueError("complete arms require observation and oracle only")
        if not complete and (has_any_result or not has_complete_error):
            raise ValueError("failed arms require stable error details only")
        return self


class BenchmarkCaseEvidenceV1(_BenchmarkModel):
    schema_version: Literal[1]
    case: BenchmarkCaseV1
    case_id: str
    source_class: BenchmarkSourceClass
    stratum: BenchmarkStratum
    status: Literal["complete", "incomplete"]
    arms: tuple[BenchmarkArmEvidenceV1, ...]
    comparator_arm_id: str | None
    comparator_utility_micros: UtilityMicros | None
    experience_hub_utility_micros: UtilityMicros | None
    delta_utility_micros: StrictInt | None

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_case_id(cls, value: object) -> str:
        return _label(value, field_name="case_id")

    @field_validator("comparator_arm_id", mode="before")
    @classmethod
    def validate_comparator_arm_id(cls, value: object) -> str | None:
        if value is None:
            return None
        allowed = {
            BenchmarkArmKind.NO_MEMORY.value,
            BenchmarkArmKind.RECENT_NOTES.value,
            BenchmarkArmKind.SQLITE_BM25.value,
        }
        if not isinstance(value, str) or value not in allowed:
            raise ValueError("comparator_arm_id must name a baseline arm")
        return value

    @model_validator(mode="after")
    def validate_complete_case(self) -> Self:
        if (
            self.case.case_id != self.case_id
            or self.case.source_class is not self.source_class
            or self.case.stratum is not self.stratum
        ):
            raise ValueError("case evidence identity must match its canonical case")
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        expected = tuple(item.value for item in BENCHMARK_ARM_ORDER)
        if arm_ids != expected:
            raise ValueError("arms must use the ordered required benchmark arms")
        complete = all(arm.status == "complete" for arm in self.arms)
        comparison = (
            self.comparator_arm_id,
            self.comparator_utility_micros,
            self.experience_hub_utility_micros,
            self.delta_utility_micros,
        )
        if self.status == "complete":
            if not complete or any(value is None for value in comparison):
                raise ValueError("complete cases require complete arms and comparison")
            comparator_utility = self.comparator_utility_micros
            experience_hub_utility = self.experience_hub_utility_micros
            delta_utility = self.delta_utility_micros
            if (
                comparator_utility is None
                or experience_hub_utility is None
                or delta_utility is None
            ):
                raise ValueError("complete cases require complete utility values")
            if (
                delta_utility != experience_hub_utility - comparator_utility
            ):
                raise ValueError("delta_utility_micros must match case utilities")
        elif complete or any(value is not None for value in comparison):
            raise ValueError(
                "incomplete cases require an incomplete arm and null comparison"
            )
        return self


class BenchmarkSafetyEvidenceV1(_BenchmarkModel):
    schema_version: Literal[1]
    owner_leak_count: NonnegativeInt
    quarantine_leak_count: NonnegativeInt
    cross_arm_contamination_count: NonnegativeInt
    source_mutation_count: NonnegativeInt
    source_unchanged: StrictBool
    clone_isolation_verified: StrictBool

    @property
    def is_safe(self) -> bool:
        return (
            self.owner_leak_count == 0
            and self.quarantine_leak_count == 0
            and self.cross_arm_contamination_count == 0
            and self.source_mutation_count == 0
            and self.source_unchanged
            and self.clone_isolation_verified
        )


class BenchmarkDeltaAggregateV1(_BenchmarkModel):
    schema_version: Literal[1]
    scope: str
    sum_delta_micros: StrictInt
    case_count: Annotated[StrictInt, Field(ge=1, le=MAX_REPLAY_CASES)]
    mean_delta_micros: StrictInt

    @field_validator("scope", mode="before")
    @classmethod
    def validate_scope(cls, value: object) -> str:
        if value == "overall":
            return "overall"
        if isinstance(value, str) and value in {
            item.value for item in BENCHMARK_STRATUM_ORDER
        }:
            return value
        return _label(value, field_name="scope")

    @model_validator(mode="after")
    def validate_derived_mean(self) -> Self:
        if self.mean_delta_micros != self.sum_delta_micros // self.case_count:
            raise ValueError(
                "mean_delta_micros must be derived from sum and case count"
            )
        return self


class BenchmarkAggregateV1(_BenchmarkModel):
    schema_version: Literal[1]
    overall: BenchmarkDeltaAggregateV1
    strata: tuple[BenchmarkDeltaAggregateV1, ...]

    @model_validator(mode="after")
    def validate_stratum_order(self) -> Self:
        if self.overall.scope != "overall":
            raise ValueError("overall aggregate must use overall scope")
        expected = tuple(item.value for item in BENCHMARK_STRATUM_ORDER)
        if len(self.strata) != len(expected) or tuple(
            item.scope for item in self.strata
        ) != expected:
            raise ValueError("strata must use the fixed benchmark stratum order")
        return self


def _validate_pilot_aggregate(aggregate: BenchmarkAggregateV1) -> None:
    if aggregate.overall.case_count != 30:
        raise ValueError("overall aggregate must contain exactly 30 cases")
    if any(item.case_count != 6 for item in aggregate.strata):
        raise ValueError("each stratum aggregate must contain exactly 6 cases")


class BenchmarkGateResultV1(_BenchmarkModel):
    schema_version: Literal[1]
    gate_id: str
    passed: StrictBool

    @field_validator("gate_id", mode="before")
    @classmethod
    def validate_gate_id(cls, value: object) -> str:
        if not isinstance(value, str) or not _ERROR_CODE.fullmatch(value):
            raise ValueError("gate_id must be a stable lowercase identifier")
        return value


class ResolvedBenchmarkManifestV1(_BenchmarkModel):
    schema_version: Literal[1]
    pack_id: str
    maturity: str
    manifest_sha256: str
    cases_sha256: str
    source_fixture_sha256: str
    snapshot_sha256: str
    source_schema_revision: NonnegativeInt
    frozen_at: datetime
    seed: NonnegativeInt
    arms: tuple[BenchmarkArmDescriptorV1, ...]
    oracle_version: Literal[1]
    metric_version: Literal[1]
    gate_version: Literal[1]
    evidence_schema_version: Literal[1]
    summary_schema_version: Literal[1]
    profile_schema_version: Literal[1]

    @field_validator("pack_id", mode="before")
    @classmethod
    def validate_pack_id(cls, value: object) -> str:
        return _label(value, field_name="pack_id")

    @field_validator(
        "manifest_sha256",
        "cases_sha256",
        "source_fixture_sha256",
        "snapshot_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "sha256")
        return _sha256(value, field_name=field_name)

    @field_validator("frozen_at", mode="before")
    @classmethod
    def validate_frozen_at(cls, value: object) -> datetime:
        return _utc_datetime(value, field_name="frozen_at")

    @model_validator(mode="after")
    def validate_arm_order(self) -> Self:
        _require_ordered_benchmark_arms(self.arms)
        return self


def _require_ordered_benchmark_arms(
    arms: tuple[BenchmarkArmDescriptorV1, ...],
) -> None:
    expected = BENCHMARK_ARM_ORDER
    if len(arms) != len(expected) or not all(arm.required for arm in arms):
        raise ValueError("arms must contain all ordered required benchmark arms")
    if tuple(arm.kind for arm in arms) != expected:
        raise ValueError("arms must contain ordered required benchmark arms")
    if tuple(arm.arm_id for arm in arms) != tuple(item.value for item in expected):
        raise ValueError("arms must contain ordered required benchmark arm IDs")


class BenchmarkPassPayloadV1(_BenchmarkModel):
    schema_version: Literal[1]
    resolved_manifest: ResolvedBenchmarkManifestV1
    cases: tuple[BenchmarkCaseEvidenceV1, ...]
    comparison_complete: StrictBool
    safety: BenchmarkSafetyEvidenceV1
    aggregate: BenchmarkAggregateV1 | None

    @model_validator(mode="after")
    def validate_payload_cases(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != 30 or len(case_ids) != len(set(case_ids)):
            raise ValueError("cases must contain exactly 30 unique case IDs")
        if any(
            sum(case.stratum is stratum for case in self.cases) != 6
            for stratum in BENCHMARK_STRATUM_ORDER
        ):
            raise ValueError("cases must contain exactly 6 cases per stratum")
        complete = all(case.status == "complete" for case in self.cases)
        if self.comparison_complete != complete:
            raise ValueError("comparison_complete must match case completeness")
        if (self.aggregate is not None) != complete:
            raise ValueError("aggregate is present only for complete comparisons")
        if self.aggregate is not None:
            _validate_pilot_aggregate(self.aggregate)
        return self


class BenchmarkEvidenceDataV1(_BenchmarkModel):
    schema_version: Literal[1]
    pass_payload: BenchmarkPassPayloadV1
    deterministic_replay_match: StrictBool
    gates: tuple[BenchmarkGateResultV1, ...]
    expansion_gate_passed: StrictBool
    valid: StrictBool

    @model_validator(mode="after")
    def validate_gates(self) -> Self:
        gate_ids = tuple(gate.gate_id for gate in self.gates)
        if not gate_ids or len(gate_ids) != len(set(gate_ids)):
            raise ValueError("gates must contain at least one unique gate ID")
        expected_valid = (
            self.pass_payload.comparison_complete
            and self.deterministic_replay_match
            and self.pass_payload.safety.is_safe
        )
        if self.valid != expected_valid:
            raise ValueError("valid must match completeness, replay, and safety")
        expected_expansion = expected_valid and all(gate.passed for gate in self.gates)
        if self.expansion_gate_passed != expected_expansion:
            raise ValueError("expansion_gate_passed must match valid gate results")
        return self


class BenchmarkEvidenceReportV1(_BenchmarkModel):
    data: BenchmarkEvidenceDataV1


class BenchmarkSummaryDataV1(_BenchmarkModel):
    schema_version: Literal[1]
    pack_id: str
    manifest_sha256: str
    cases_sha256: str
    source_fixture_sha256: str
    snapshot_sha256: str
    evidence_sha256: str
    case_count: Literal[30]
    arm_count: Literal[4]
    comparison_complete: StrictBool
    deterministic_replay_match: StrictBool
    safety: BenchmarkSafetyEvidenceV1
    aggregate: BenchmarkAggregateV1 | None
    gates: tuple[BenchmarkGateResultV1, ...]
    expansion_gate_passed: StrictBool
    valid: StrictBool
    claim_boundary: str

    @field_validator("pack_id", mode="before")
    @classmethod
    def validate_pack_id(cls, value: object) -> str:
        return _label(value, field_name="pack_id")

    @field_validator(
        "manifest_sha256",
        "cases_sha256",
        "source_fixture_sha256",
        "snapshot_sha256",
        "evidence_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: object, info: object) -> str:
        field_name = getattr(info, "field_name", "sha256")
        return _sha256(value, field_name=field_name)

    @field_validator("claim_boundary", mode="before")
    @classmethod
    def validate_claim_boundary(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("claim_boundary must be nonempty")
        return value

    @model_validator(mode="after")
    def validate_summary_state(self) -> Self:
        if (self.aggregate is not None) != self.comparison_complete:
            raise ValueError("aggregate is present only for complete comparisons")
        if self.aggregate is not None:
            _validate_pilot_aggregate(self.aggregate)
        gate_ids = tuple(gate.gate_id for gate in self.gates)
        if not gate_ids or len(gate_ids) != len(set(gate_ids)):
            raise ValueError("gates must contain at least one unique gate ID")
        expected_valid = (
            self.comparison_complete
            and self.deterministic_replay_match
            and self.safety.is_safe
        )
        if self.valid != expected_valid:
            raise ValueError("valid must match completeness, replay, and safety")
        expected_expansion = expected_valid and all(gate.passed for gate in self.gates)
        if self.expansion_gate_passed != expected_expansion:
            raise ValueError("expansion_gate_passed must match valid gate results")
        return self


class BenchmarkSummaryReportV1(_BenchmarkModel):
    data: BenchmarkSummaryDataV1


class BenchmarkProfileDataV1(_BenchmarkModel):
    schema_version: Literal[1]
    pack_id: str
    profile_complete: StrictBool
    wall_duration_ns: NonnegativeInt | None
    database_bytes: NonnegativeInt
    clone_count: NonnegativeInt
    fts5_available: StrictBool

    @field_validator("pack_id", mode="before")
    @classmethod
    def validate_pack_id(cls, value: object) -> str:
        return _label(value, field_name="pack_id")

    @model_validator(mode="after")
    def validate_profile_state(self) -> Self:
        if self.profile_complete != (self.wall_duration_ns is not None):
            raise ValueError("profile_complete must match wall_duration_ns presence")
        return self


class BenchmarkProfileReportV1(_BenchmarkModel):
    data: BenchmarkProfileDataV1


__all__ = [
    "BENCHMARK_ARM_ORDER",
    "BENCHMARK_STRATUM_ORDER",
    "MAX_BENCHMARK_CASES",
    "MAX_BENCHMARK_INPUT_BYTES",
    "MAX_BENCHMARK_OUTPUT_BYTES",
    "MAX_BENCHMARK_SOURCE_RECORDS",
    "MAX_BENCHMARK_TOTAL_INPUT_BYTES",
    "BenchmarkAggregateV1",
    "BenchmarkArmDescriptorV1",
    "BenchmarkArmEvidenceV1",
    "BenchmarkArmKind",
    "BenchmarkArmObservationV1",
    "BenchmarkCaseEvidenceV1",
    "BenchmarkCaseV1",
    "BenchmarkCategoryScoresV1",
    "BenchmarkCheckpointPredicate",
    "BenchmarkCheckpointV1",
    "BenchmarkCompositionV1",
    "BenchmarkDeltaAggregateV1",
    "BenchmarkEvidenceDataV1",
    "BenchmarkEvidenceReportV1",
    "BenchmarkFileDescriptorV1",
    "BenchmarkGateResultV1",
    "BenchmarkLanguage",
    "BenchmarkOracleEvidenceV1",
    "BenchmarkPackManifestV1",
    "BenchmarkPassPayloadV1",
    "BenchmarkProfileDataV1",
    "BenchmarkProfileReportV1",
    "BenchmarkSourceAgentV1",
    "BenchmarkSourceCandidateV1",
    "BenchmarkSourceClass",
    "BenchmarkSourceEvidenceV1",
    "BenchmarkSourceExperienceV1",
    "BenchmarkSourceRecordV1",
    "BenchmarkStratum",
    "BenchmarkSummaryDataV1",
    "BenchmarkSummaryReportV1",
    "BenchmarkWeightedLabelV1",
    "ResolvedBenchmarkManifestV1",
]


class BenchmarkPackManifestV1(_BenchmarkModel):
    schema_version: Literal[1]
    pack_id: str
    maturity: Literal["pilot-30"]
    cases: BenchmarkFileDescriptorV1
    source: BenchmarkFileDescriptorV1
    frozen_at: datetime
    seed: NonnegativeInt
    arms: tuple[BenchmarkArmDescriptorV1, ...]
    oracle_version: Literal[1]
    metric_version: Literal[1]
    gate_version: Literal[1]
    evidence_schema_version: Literal[1]
    summary_schema_version: Literal[1]
    profile_schema_version: Literal[1]
    deterministic_replay_runs: Literal[2]
    composition: BenchmarkCompositionV1

    @field_validator("pack_id", mode="before")
    @classmethod
    def validate_pack_id(cls, value: object) -> str:
        return _label(value, field_name="pack_id")

    @field_validator("frozen_at", mode="before")
    @classmethod
    def validate_frozen_at(cls, value: object) -> datetime:
        return _utc_datetime(value, field_name="frozen_at")

    @model_validator(mode="after")
    def validate_required_arm_order(self) -> Self:
        _require_ordered_benchmark_arms(self.arms)
        if self.cases.file == self.source.file:
            raise ValueError("cases and source files must be distinct")
        return self
