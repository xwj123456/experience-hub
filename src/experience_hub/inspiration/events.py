"""Strict versioned events for inspiration runs and idea incubation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from experience_hub.clock import require_utc
from experience_hub.domain import EventPayload, EventRegistry
from experience_hub.domain.values import StructuredReason
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.incubation import ClusterTransition
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    EvaluationEvidenceReference,
    EvaluationVerdict,
    IdeaOwnerDecision,
    InspirationOperator,
    InspirationRunStatus,
    MechanismMaturity,
    OperatorOutcome,
    SnapshotEvidenceReference,
)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_Counter = Annotated[int, Field(strict=True, ge=0)]
_UnitFloat = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


def _require_running(name: str, value: InspirationRunStatus) -> None:
    if value is not InspirationRunStatus.RUNNING:
        raise ValueError(f"{name} must remain running")


def _validated_outcomes(
    values: tuple[OperatorOutcome, ...],
) -> tuple[OperatorOutcome, ...]:
    validated = tuple(
        OperatorOutcome.model_validate(
            value.model_dump(mode="python", warnings=False),
            strict=True,
        )
        for value in values
    )
    operators = tuple(value.operator for value in validated)
    canonical = tuple(
        operator for operator in INSPIRATION_OPERATOR_ORDER if operator in operators
    )
    if operators != canonical or len(operators) != len(set(operators)):
        raise ValueError("operator outcomes must use canonical operator order")
    return validated


def _validate_maturity_transition(
    *,
    maturity_before: MechanismMaturity,
    maturity_after: MechanismMaturity,
    candidate_since_before: datetime | None,
    candidate_since_after: datetime | None,
    last_signal_at_before: datetime,
    last_signal_at_after: datetime,
) -> None:
    if (maturity_before is MechanismMaturity.CANDIDATE) is (
        candidate_since_before is None
    ):
        raise ValueError("candidate_since_before must match candidate maturity")
    if (maturity_after is MechanismMaturity.CANDIDATE) is (
        candidate_since_after is None
    ):
        raise ValueError("candidate_since_after must match candidate maturity")
    if last_signal_at_after < last_signal_at_before:
        raise ValueError("last_signal_at must not move backward")
    if (
        candidate_since_before is not None
        and candidate_since_before > last_signal_at_before
    ):
        raise ValueError("candidate_since_before cannot follow last signal")
    if candidate_since_after is not None:
        if candidate_since_after > last_signal_at_after:
            raise ValueError("candidate_since_after cannot follow last signal")
        if maturity_before is MechanismMaturity.CANDIDATE:
            if candidate_since_after != candidate_since_before:
                raise ValueError("candidate maturity must retain candidate_since")
        elif candidate_since_after != last_signal_at_after:
            raise ValueError(
                "candidate_since_after must equal the candidate entry signal"
            )


class InspirationRunFailureCode(StrEnum):
    """Sanitized terminal causes that never retain exception text."""

    PREPARATION_FAILED = "preparation_failed"
    ALL_OPERATORS_FAILED = "all_operators_failed"
    PROCESS_INTERRUPTED = "process_interrupted"


class InspirationStartedV1(EventPayload):
    event_type: ClassVar[str] = "inspiration.started"

    run_id: UUID
    owner_agent_id: UUID
    status_after: InspirationRunStatus

    @model_validator(mode="after")
    def validate_initial_state(self) -> Self:
        _require_running("status_after", self.status_after)
        return self


class InspirationSnapshotFrozenV1(EventPayload):
    event_type: ClassVar[str] = "inspiration.snapshot_frozen"

    run_id: UUID
    snapshot_hash: str
    snapshot_item_ids: tuple[UUID, ...]
    status_before: InspirationRunStatus
    status_after: InspirationRunStatus

    @field_validator("snapshot_hash")
    @classmethod
    def validate_snapshot_hash(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("snapshot_hash must be lowercase SHA-256 hex")
        return value

    @field_validator("snapshot_item_ids")
    @classmethod
    def validate_snapshot_item_ids(
        cls,
        values: tuple[UUID, ...],
    ) -> tuple[UUID, ...]:
        if len(values) > 12:
            raise ValueError("snapshot_item_ids may contain at most 12 values")
        if len(values) != len(set(values)):
            raise ValueError("snapshot_item_ids must not repeat")
        return values

    @model_validator(mode="after")
    def validate_state_transition(self) -> Self:
        _require_running("status_before", self.status_before)
        _require_running("status_after", self.status_after)
        return self


class _InspirationOperatorEventV1(EventPayload):
    run_id: UUID
    operator: InspirationOperator
    outcome: OperatorOutcome
    status_before: InspirationRunStatus
    status_after: InspirationRunStatus
    output_tokens_reserved_before: _Counter
    output_tokens_reserved_after: _Counter
    output_tokens_consumed_before: _Counter
    output_tokens_consumed_after: _Counter
    elapsed_milliseconds_before: _Counter
    elapsed_milliseconds_after: _Counter

    @model_validator(mode="after")
    def validate_operator_accounting(self) -> Self:
        _require_running("status_before", self.status_before)
        _require_running("status_after", self.status_after)
        if self.operator is not self.outcome.operator:
            raise ValueError("operator must match the retained outcome operator")
        reservation = (
            self.output_tokens_reserved_after - self.output_tokens_reserved_before
        )
        consumption = (
            self.output_tokens_consumed_after - self.output_tokens_consumed_before
        )
        if not 0 <= reservation <= 1_200:
            raise ValueError(
                "operator reservation must increase by zero to 1200 tokens"
            )
        if consumption != self.outcome.output_tokens_consumed:
            raise ValueError("operator consumption must match the retained outcome")
        if not 0 <= consumption <= reservation:
            raise ValueError("operator consumption cannot exceed its reservation")
        if (
            self.output_tokens_reserved_after > 3_600
            or self.output_tokens_consumed_after > self.output_tokens_reserved_after
        ):
            raise ValueError("run consumption cannot exceed its reservation")
        if self.elapsed_milliseconds_after < self.elapsed_milliseconds_before:
            raise ValueError("elapsed milliseconds must not move backward")
        return self


class InspirationOperatorCompletedV1(_InspirationOperatorEventV1):
    event_type: ClassVar[str] = "inspiration.operator_completed"

    @model_validator(mode="after")
    def validate_success(self) -> Self:
        if not self.outcome.succeeded:
            raise ValueError("operator_completed requires a successful outcome")
        return self


class InspirationOperatorFailedV1(_InspirationOperatorEventV1):
    event_type: ClassVar[str] = "inspiration.operator_failed"

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if self.outcome.succeeded:
            raise ValueError("operator_failed requires a failed outcome")
        return self


class InspirationIdeaGeneratedV1(EventPayload):
    event_type: ClassVar[str] = "inspiration.idea_generated"

    idea_id: UUID
    occurrence_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    operator: InspirationOperator
    ordinal: Annotated[int, Field(strict=True, ge=1, le=3)]
    snapshot_hash: str
    evidence: tuple[SnapshotEvidenceReference, ...]
    idea_content_hash: str
    mechanism_hash: str
    duplicate_relation: UUID | None
    owner_decision_after: IdeaOwnerDecision
    cluster_id: str
    canonical_mechanism_hash: str
    member_hashes_before: tuple[str, ...]
    member_hashes_after: tuple[str, ...]
    occurrence_count_before: _Counter
    occurrence_count_after: _Counter
    distinct_snapshot_count_before: _Counter
    distinct_snapshot_count_after: _Counter
    distinct_adopter_count_before: _Counter
    distinct_adopter_count_after: _Counter
    supported_count_before: _Counter
    supported_count_after: _Counter
    refuted_count_before: _Counter
    refuted_count_after: _Counter
    maturity_before: MechanismMaturity | None
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime | None
    last_signal_at_after: datetime

    @field_validator(
        "snapshot_hash",
        "idea_content_hash",
        "mechanism_hash",
        "cluster_id",
        "canonical_mechanism_hash",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex")
        return value

    @field_validator("member_hashes_before", "member_hashes_after")
    @classmethod
    def validate_member_hashes(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _SHA256_HEX.fullmatch(value) for value in values):
            raise ValueError("member hashes must be lowercase SHA-256 hex")
        return values

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[SnapshotEvidenceReference, ...],
    ) -> tuple[SnapshotEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > 12:
            raise ValueError("evidence may contain at most 12 references")
        validated = tuple(
            SnapshotEvidenceReference.model_validate(
                value.model_dump(mode="python", warnings=False),
                strict=True,
            )
            for value in values
        )
        ids = tuple(value.id for value in validated)
        stable_keys = tuple(value.stable_evidence_key for value in validated)
        if len(ids) != len(set(ids)) or len(stable_keys) != len(set(stable_keys)):
            raise ValueError("evidence references must not repeat")
        return validated

    @field_validator(
        "candidate_since_before",
        "candidate_since_after",
        "last_signal_at_before",
        "last_signal_at_after",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else require_utc(value)

    @model_validator(mode="after")
    def validate_generated_transition(self) -> Self:
        if self.owner_decision_after is not IdeaOwnerDecision.ACTIVE:
            raise ValueError("a generated idea must start active")
        if self.duplicate_relation == self.idea_id:
            raise ValueError("an idea cannot duplicate itself")
        ClusterTransition(
            cluster_id=self.cluster_id,
            canonical_mechanism_hash=self.canonical_mechanism_hash,
            member_hashes_before=self.member_hashes_before,
            member_hashes_after=self.member_hashes_after,
            occurrence_count_before=self.occurrence_count_before,
            occurrence_count_after=self.occurrence_count_after,
            distinct_snapshot_count_before=self.distinct_snapshot_count_before,
            distinct_snapshot_count_after=self.distinct_snapshot_count_after,
            distinct_adopter_count_before=self.distinct_adopter_count_before,
            distinct_adopter_count_after=self.distinct_adopter_count_after,
            supported_count_before=self.supported_count_before,
            supported_count_after=self.supported_count_after,
            refuted_count_before=self.refuted_count_before,
            refuted_count_after=self.refuted_count_after,
            maturity_before=self.maturity_before,
            maturity_after=self.maturity_after,
            candidate_since_before=self.candidate_since_before,
            candidate_since_after=self.candidate_since_after,
            last_signal_at_before=self.last_signal_at_before,
            last_signal_at_after=self.last_signal_at_after,
        )
        if self.mechanism_hash not in self.member_hashes_after:
            raise ValueError(
                "mechanism_hash must belong to the declared cluster members"
            )
        return self


class _InspirationTerminalEventV1(EventPayload):
    run_id: UUID
    status_before: InspirationRunStatus
    status_after: InspirationRunStatus
    operator_outcomes: tuple[OperatorOutcome, ...]
    output_tokens_reserved_before: _Counter
    output_tokens_reserved_after: _Counter
    output_tokens_consumed_before: _Counter
    output_tokens_consumed_after: _Counter
    elapsed_milliseconds_before: _Counter
    elapsed_milliseconds_after: _Counter

    @field_validator("operator_outcomes")
    @classmethod
    def validate_outcomes(
        cls,
        values: tuple[OperatorOutcome, ...],
    ) -> tuple[OperatorOutcome, ...]:
        return _validated_outcomes(values)

    @model_validator(mode="after")
    def validate_terminal_accounting(self) -> Self:
        _require_running("status_before", self.status_before)
        before = (
            self.output_tokens_reserved_before,
            self.output_tokens_consumed_before,
            self.elapsed_milliseconds_before,
        )
        after = (
            self.output_tokens_reserved_after,
            self.output_tokens_consumed_after,
            self.elapsed_milliseconds_after,
        )
        if before != after:
            raise ValueError("terminal accounting must not change accumulated counters")
        if (
            not 0
            <= self.output_tokens_consumed_after
            <= (self.output_tokens_reserved_after)
            <= 3_600
        ):
            raise ValueError("terminal accounting exceeds the run token budget")
        expected_consumed = sum(
            outcome.output_tokens_consumed for outcome in self.operator_outcomes
        )
        if self.output_tokens_consumed_after != expected_consumed:
            raise ValueError(
                "terminal accounting must equal operator outcome consumption"
            )
        return self


class InspirationCompletedV1(_InspirationTerminalEventV1):
    event_type: ClassVar[str] = "inspiration.completed"

    @model_validator(mode="after")
    def validate_completed_status(self) -> Self:
        outcomes = self.operator_outcomes
        succeeded = sum(outcome.succeeded for outcome in outcomes)
        failed = len(outcomes) - succeeded
        if self.status_after is InspirationRunStatus.COMPLETED:
            if not outcomes or failed:
                raise ValueError("completed status requires every operator to succeed")
        elif self.status_after is InspirationRunStatus.COMPLETED_WITH_ERRORS:
            if not succeeded or not failed:
                raise ValueError(
                    "completed_with_errors requires mixed operator outcomes"
                )
        else:
            raise ValueError("inspiration.completed requires a completed status")
        return self


class InspirationFailedV1(_InspirationTerminalEventV1):
    event_type: ClassVar[str] = "inspiration.failed"

    failure_code: InspirationRunFailureCode

    @model_validator(mode="after")
    def validate_failed_status(self) -> Self:
        if self.status_after is not InspirationRunStatus.FAILED:
            raise ValueError("inspiration.failed requires failed status")
        if any(outcome.succeeded for outcome in self.operator_outcomes):
            raise ValueError("a failed run cannot retain a successful operator outcome")
        if self.failure_code is InspirationRunFailureCode.ALL_OPERATORS_FAILED:
            if not self.operator_outcomes:
                raise ValueError(
                    "all_operators_failed requires failed operator outcomes"
                )
        elif self.operator_outcomes:
            raise ValueError(
                "preparation or recovery failure must not retain operator outcomes"
            )
        return self


class InspirationTimedOutV1(_InspirationTerminalEventV1):
    event_type: ClassVar[str] = "inspiration.timed_out"

    failure_code: OperatorFailureCode

    @model_validator(mode="after")
    def validate_timed_out_status(self) -> Self:
        if self.status_after is not InspirationRunStatus.TIMED_OUT:
            raise ValueError("inspiration.timed_out requires timed_out status")
        if (
            self.failure_code is not OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
            or not any(
                outcome.error_code is OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
                for outcome in self.operator_outcomes
            )
        ):
            raise ValueError("timed_out requires global deadline exhaustion")
        return self


class InspirationIdeaEvaluatedV1(EventPayload):
    event_type: ClassVar[str] = "inspiration.idea_evaluated"

    idea_id: UUID
    evaluator_agent_id: UUID
    mechanism_cluster_id: str
    revision: Annotated[int, Field(strict=True, ge=1)]
    previous_verdict: EvaluationVerdict | None
    current_verdict: EvaluationVerdict
    evidence: tuple[EvaluationEvidenceReference, ...]
    reason: StructuredReason | None
    owner_decision_before: IdeaOwnerDecision
    owner_decision_after: IdeaOwnerDecision
    supported_count_before: _Counter
    supported_count_after: _Counter
    refuted_count_before: _Counter
    refuted_count_after: _Counter
    maturity_before: MechanismMaturity
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime
    last_signal_at_after: datetime

    @field_validator("mechanism_cluster_id")
    @classmethod
    def validate_cluster_id(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("mechanism_cluster_id must be lowercase SHA-256 hex")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[EvaluationEvidenceReference, ...],
    ) -> tuple[EvaluationEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > 32:
            raise ValueError("evidence may contain at most 32 references")
        identities: list[tuple[str, UUID, str | None]] = []
        for value in values:
            if isinstance(value, SnapshotEvidenceReference):
                identities.append((value.type, value.id, value.stable_evidence_key))
            else:
                identities.append((value.type, value.id, None))
        if len(identities) != len(set(identities)):
            raise ValueError("evaluation evidence must not repeat")
        snapshot_keys = tuple(
            value.stable_evidence_key
            for value in values
            if isinstance(value, SnapshotEvidenceReference)
        )
        if len(snapshot_keys) != len(set(snapshot_keys)):
            raise ValueError("evaluation snapshot evidence must not repeat")
        return values

    @field_validator(
        "candidate_since_before",
        "candidate_since_after",
        "last_signal_at_before",
        "last_signal_at_after",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else require_utc(value)

    @model_validator(mode="after")
    def validate_evaluation_transition(self) -> Self:
        if (self.revision == 1) is not (self.previous_verdict is None):
            raise ValueError("only the first revision may omit previous_verdict")
        if self.owner_decision_before not in {
            IdeaOwnerDecision.ACTIVE,
            IdeaOwnerDecision.ARCHIVED,
        }:
            raise ValueError("evaluation requires an active or archived owner decision")
        if self.owner_decision_after is not self.owner_decision_before:
            raise ValueError("evaluation cannot change the owner decision")
        expected_supported = self.supported_count_before
        expected_refuted = self.refuted_count_before
        if self.previous_verdict is EvaluationVerdict.SUPPORTED:
            expected_supported -= 1
        elif self.previous_verdict is EvaluationVerdict.REFUTED:
            expected_refuted -= 1
        if self.current_verdict is EvaluationVerdict.SUPPORTED:
            expected_supported += 1
        elif self.current_verdict is EvaluationVerdict.REFUTED:
            expected_refuted += 1
        if (
            expected_supported < 0
            or expected_refuted < 0
            or self.supported_count_after != expected_supported
            or self.refuted_count_after != expected_refuted
        ):
            raise ValueError(
                "evaluation effective counts do not match the verdict revision"
            )
        _validate_maturity_transition(
            maturity_before=self.maturity_before,
            maturity_after=self.maturity_after,
            candidate_since_before=self.candidate_since_before,
            candidate_since_after=self.candidate_since_after,
            last_signal_at_before=self.last_signal_at_before,
            last_signal_at_after=self.last_signal_at_after,
        )
        return self


class InspirationIdeaAdoptedV1(EventPayload):
    event_type: ClassVar[str] = "inspiration.idea_adopted"

    adoption_id: UUID
    idea_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    snapshot_hash: str
    evidence: tuple[SnapshotEvidenceReference, ...]
    resulting_experience_id: UUID
    resulting_version_id: UUID
    created: bool
    mechanism_cluster_id: str
    owner_decision_before: IdeaOwnerDecision
    owner_decision_after: IdeaOwnerDecision
    distinct_adopter_count_before: _Counter
    distinct_adopter_count_after: _Counter
    maturity_before: MechanismMaturity
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime
    last_signal_at_after: datetime

    @field_validator("snapshot_hash", "mechanism_cluster_id")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[SnapshotEvidenceReference, ...],
    ) -> tuple[SnapshotEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > 12:
            raise ValueError("evidence may contain at most 12 references")
        ids = tuple(value.id for value in values)
        stable_keys = tuple(value.stable_evidence_key for value in values)
        if len(ids) != len(set(ids)) or len(stable_keys) != len(set(stable_keys)):
            raise ValueError("adoption evidence must not repeat")
        return values

    @field_validator(
        "candidate_since_before",
        "candidate_since_after",
        "last_signal_at_before",
        "last_signal_at_after",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else require_utc(value)

    @model_validator(mode="after")
    def validate_adoption_transition(self) -> Self:
        if self.owner_decision_before not in {
            IdeaOwnerDecision.ACTIVE,
            IdeaOwnerDecision.ARCHIVED,
        }:
            raise ValueError("adoption requires an active or archived owner decision")
        if self.owner_decision_after is not IdeaOwnerDecision.ADOPTED:
            raise ValueError("adoption must transition the owner to adopted")
        if self.distinct_adopter_count_after not in {
            self.distinct_adopter_count_before,
            self.distinct_adopter_count_before + 1,
        }:
            raise ValueError(
                "adoption may increase distinct adopter count by at most one"
            )
        _validate_maturity_transition(
            maturity_before=self.maturity_before,
            maturity_after=self.maturity_after,
            candidate_since_before=self.candidate_since_before,
            candidate_since_after=self.candidate_since_after,
            last_signal_at_before=self.last_signal_at_before,
            last_signal_at_after=self.last_signal_at_after,
        )
        return self


class InspirationIdeaAdoptedV2(EventPayload):
    """Adoption event with exact caller-selected lifecycle parameters."""

    event_type: ClassVar[str] = "inspiration.idea_adopted_v2"
    # Pydantic intentionally narrows the inherited wire discriminator here.
    schema_version: Literal[2]  # type: ignore[assignment]

    adoption_id: UUID
    idea_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    snapshot_hash: str
    evidence: tuple[SnapshotEvidenceReference, ...]
    resulting_experience_id: UUID
    resulting_version_id: UUID
    created: bool
    requested_importance: _UnitFloat
    requested_confidence: _UnitFloat
    mechanism_cluster_id: str
    owner_decision_before: IdeaOwnerDecision
    owner_decision_after: IdeaOwnerDecision
    distinct_adopter_count_before: _Counter
    distinct_adopter_count_after: _Counter
    maturity_before: MechanismMaturity
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime
    last_signal_at_after: datetime

    @field_validator("snapshot_hash", "mechanism_cluster_id")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls,
        values: tuple[SnapshotEvidenceReference, ...],
    ) -> tuple[SnapshotEvidenceReference, ...]:
        if not values:
            raise ValueError("evidence must not be empty")
        if len(values) > 12:
            raise ValueError("evidence may contain at most 12 references")
        ids = tuple(value.id for value in values)
        stable_keys = tuple(value.stable_evidence_key for value in values)
        if len(ids) != len(set(ids)) or len(stable_keys) != len(set(stable_keys)):
            raise ValueError("adoption evidence must not repeat")
        return values

    @field_validator(
        "candidate_since_before",
        "candidate_since_after",
        "last_signal_at_before",
        "last_signal_at_after",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else require_utc(value)

    @model_validator(mode="after")
    def validate_adoption_transition(self) -> Self:
        if self.owner_decision_before not in {
            IdeaOwnerDecision.ACTIVE,
            IdeaOwnerDecision.ARCHIVED,
        }:
            raise ValueError("adoption requires an active or archived owner decision")
        if self.owner_decision_after is not IdeaOwnerDecision.ADOPTED:
            raise ValueError("adoption must transition the owner to adopted")
        if self.distinct_adopter_count_after not in {
            self.distinct_adopter_count_before,
            self.distinct_adopter_count_before + 1,
        }:
            raise ValueError(
                "adoption may increase distinct adopter count by at most one"
            )
        _validate_maturity_transition(
            maturity_before=self.maturity_before,
            maturity_after=self.maturity_after,
            candidate_since_before=self.candidate_since_before,
            candidate_since_after=self.candidate_since_after,
            last_signal_at_before=self.last_signal_at_before,
            last_signal_at_after=self.last_signal_at_after,
        )
        return self


class _InspirationIdeaDecisionEventV1(EventPayload):
    idea_id: UUID
    owner_agent_id: UUID
    reason: StructuredReason
    owner_decision_before: IdeaOwnerDecision
    owner_decision_after: IdeaOwnerDecision


class InspirationIdeaRejectedV1(_InspirationIdeaDecisionEventV1):
    event_type: ClassVar[str] = "inspiration.idea_rejected"

    @model_validator(mode="after")
    def validate_rejection_transition(self) -> Self:
        if self.owner_decision_before not in {
            IdeaOwnerDecision.ACTIVE,
            IdeaOwnerDecision.ARCHIVED,
        }:
            raise ValueError("rejection requires an active or archived owner decision")
        if self.owner_decision_after is not IdeaOwnerDecision.REJECTED:
            raise ValueError("rejection must transition the owner to rejected")
        return self


class InspirationIdeaArchivedV1(_InspirationIdeaDecisionEventV1):
    event_type: ClassVar[str] = "inspiration.idea_archived"

    cycle_id: UUID | None

    @model_validator(mode="after")
    def validate_archive_transition(self) -> Self:
        if self.owner_decision_before is not IdeaOwnerDecision.ACTIVE:
            raise ValueError("archive requires an active owner decision")
        if self.owner_decision_after is not IdeaOwnerDecision.ARCHIVED:
            raise ValueError("archive must transition the owner to archived")
        if self.cycle_id is not None and self.reason != StructuredReason.policy_due():
            raise ValueError("automatic archive requires the fixed policy-due reason")
        return self


_INSPIRATION_EVENT_PAYLOAD_TYPES = (
    InspirationStartedV1,
    InspirationSnapshotFrozenV1,
    InspirationOperatorCompletedV1,
    InspirationOperatorFailedV1,
    InspirationIdeaGeneratedV1,
    InspirationCompletedV1,
    InspirationFailedV1,
    InspirationTimedOutV1,
    InspirationIdeaEvaluatedV1,
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
    InspirationIdeaRejectedV1,
    InspirationIdeaArchivedV1,
)

INSPIRATION_EVENT_TYPES = frozenset(
    payload_type.event_type for payload_type in _INSPIRATION_EVENT_PAYLOAD_TYPES
)

INSPIRATION_EVENT_AGGREGATE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        InspirationStartedV1.event_type: "inspiration_run",
        InspirationSnapshotFrozenV1.event_type: "inspiration_run",
        InspirationOperatorCompletedV1.event_type: "inspiration_run",
        InspirationOperatorFailedV1.event_type: "inspiration_run",
        InspirationIdeaGeneratedV1.event_type: "idea",
        InspirationCompletedV1.event_type: "inspiration_run",
        InspirationFailedV1.event_type: "inspiration_run",
        InspirationTimedOutV1.event_type: "inspiration_run",
        InspirationIdeaEvaluatedV1.event_type: "idea",
        InspirationIdeaAdoptedV1.event_type: "idea",
        InspirationIdeaAdoptedV2.event_type: "idea",
        InspirationIdeaRejectedV1.event_type: "idea",
        InspirationIdeaArchivedV1.event_type: "idea",
    }
)


def register_inspiration_events(registry: EventRegistry) -> None:
    """Register the complete immutable inspiration event vocabulary."""
    for payload_type in _INSPIRATION_EVENT_PAYLOAD_TYPES:
        registry.register(payload_type)


__all__ = [
    "INSPIRATION_EVENT_AGGREGATE_TYPES",
    "INSPIRATION_EVENT_TYPES",
    "InspirationRunFailureCode",
    "InspirationCompletedV1",
    "InspirationFailedV1",
    "InspirationIdeaAdoptedV1",
    "InspirationIdeaAdoptedV2",
    "InspirationIdeaArchivedV1",
    "InspirationIdeaEvaluatedV1",
    "InspirationIdeaGeneratedV1",
    "InspirationIdeaRejectedV1",
    "InspirationOperatorCompletedV1",
    "InspirationOperatorFailedV1",
    "InspirationSnapshotFrozenV1",
    "InspirationStartedV1",
    "InspirationTimedOutV1",
    "register_inspiration_events",
]
