from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.domain import (
    EventPayload,
    EventRegistry,
    PendingEvent,
    StructuredReason,
)
from experience_hub.inspiration.events import (
    INSPIRATION_EVENT_AGGREGATE_TYPES,
    INSPIRATION_EVENT_TYPES,
    InspirationCompletedV1,
    InspirationFailedV1,
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
    InspirationIdeaArchivedV1,
    InspirationIdeaEvaluatedV1,
    InspirationIdeaGeneratedV1,
    InspirationIdeaRejectedV1,
    InspirationOperatorCompletedV1,
    InspirationOperatorFailedV1,
    InspirationRunFailureCode,
    InspirationSnapshotFrozenV1,
    InspirationStartedV1,
    InspirationTimedOutV1,
    register_inspiration_events,
)
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.models import (
    EvaluationVerdict,
    ExperienceVersionEvidenceReference,
    IdeaOwnerDecision,
    InspirationOperator,
    InspirationRunStatus,
    MechanismMaturity,
    OperatorOutcome,
    SnapshotEvidenceReference,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
IDEA_ID = UUID("00000000-0000-0000-0000-000000000102")
OCCURRENCE_ID = UUID("00000000-0000-0000-0000-000000000103")
SNAPSHOT_ITEM_ID = UUID("00000000-0000-0000-0000-000000000104")
EXPERIENCE_VERSION_ID = UUID("00000000-0000-0000-0000-000000000105")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000201")
EVALUATOR_ID = UUID("00000000-0000-0000-0000-000000000202")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000301")
RESULTING_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000302")
RESULTING_VERSION_ID = UUID("00000000-0000-0000-0000-000000000303")
CYCLE_ID = UUID("00000000-0000-0000-0000-000000000304")

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
EARLIER = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
OLDER = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
SNAPSHOT_HASH = "a" * 64
STABLE_EVIDENCE_KEY = "b" * 64
IDEA_CONTENT_HASH = "c" * 64
MECHANISM_HASH = "d" * 64
CLUSTER_ID = MECHANISM_HASH


def _reason(*, code: str = "user_provided") -> StructuredReason:
    text = "Retained structured decision."
    return StructuredReason(
        code=code,
        text=text,
        text_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def _snapshot_evidence() -> SnapshotEvidenceReference:
    return SnapshotEvidenceReference(
        id=SNAPSHOT_ITEM_ID,
        stable_evidence_key=STABLE_EVIDENCE_KEY,
    )


def _successful_outcome() -> OperatorOutcome:
    return OperatorOutcome(
        operator=InspirationOperator.CAUSAL_GAP,
        succeeded=True,
        persisted_ideas=1,
        duplicate_count=0,
        error_code=None,
        output_tokens_consumed=300,
    )


def _failed_outcome(
    *,
    operator: InspirationOperator = InspirationOperator.COUNTERFACTUAL,
    code: OperatorFailureCode = OperatorFailureCode.PROVIDER_TIMEOUT,
    consumed: int = 1_200,
) -> OperatorOutcome:
    return OperatorOutcome(
        operator=operator,
        succeeded=False,
        persisted_ideas=0,
        duplicate_count=0,
        error_code=code,
        output_tokens_consumed=consumed,
    )


def _started() -> InspirationStartedV1:
    return InspirationStartedV1(
        schema_version=1,
        run_id=RUN_ID,
        owner_agent_id=OWNER_ID,
        status_after=InspirationRunStatus.RUNNING,
    )


def _snapshot_frozen() -> InspirationSnapshotFrozenV1:
    return InspirationSnapshotFrozenV1(
        schema_version=1,
        run_id=RUN_ID,
        snapshot_hash=SNAPSHOT_HASH,
        snapshot_item_ids=(SNAPSHOT_ITEM_ID,),
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.RUNNING,
    )


def _operator_completed() -> InspirationOperatorCompletedV1:
    return InspirationOperatorCompletedV1(
        schema_version=1,
        run_id=RUN_ID,
        operator=InspirationOperator.CAUSAL_GAP,
        outcome=_successful_outcome(),
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.RUNNING,
        output_tokens_reserved_before=0,
        output_tokens_reserved_after=1_200,
        output_tokens_consumed_before=0,
        output_tokens_consumed_after=300,
        elapsed_milliseconds_before=0,
        elapsed_milliseconds_after=100,
    )


def _operator_failed() -> InspirationOperatorFailedV1:
    return InspirationOperatorFailedV1(
        schema_version=1,
        run_id=RUN_ID,
        operator=InspirationOperator.COUNTERFACTUAL,
        outcome=_failed_outcome(),
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.RUNNING,
        output_tokens_reserved_before=1_200,
        output_tokens_reserved_after=2_400,
        output_tokens_consumed_before=300,
        output_tokens_consumed_after=1_500,
        elapsed_milliseconds_before=100,
        elapsed_milliseconds_after=30_100,
    )


def _idea_generated() -> InspirationIdeaGeneratedV1:
    return InspirationIdeaGeneratedV1(
        schema_version=1,
        idea_id=IDEA_ID,
        occurrence_id=OCCURRENCE_ID,
        run_id=RUN_ID,
        owner_agent_id=OWNER_ID,
        operator=InspirationOperator.CAUSAL_GAP,
        ordinal=1,
        snapshot_hash=SNAPSHOT_HASH,
        evidence=(_snapshot_evidence(),),
        idea_content_hash=IDEA_CONTENT_HASH,
        mechanism_hash=MECHANISM_HASH,
        duplicate_relation=None,
        owner_decision_after=IdeaOwnerDecision.ACTIVE,
        cluster_id=CLUSTER_ID,
        canonical_mechanism_hash=MECHANISM_HASH,
        member_hashes_before=(),
        member_hashes_after=(MECHANISM_HASH,),
        occurrence_count_before=0,
        occurrence_count_after=1,
        distinct_snapshot_count_before=0,
        distinct_snapshot_count_after=1,
        distinct_adopter_count_before=0,
        distinct_adopter_count_after=0,
        supported_count_before=0,
        supported_count_after=0,
        refuted_count_before=0,
        refuted_count_after=0,
        maturity_before=None,
        maturity_after=MechanismMaturity.SPECULATIVE,
        candidate_since_before=None,
        candidate_since_after=None,
        last_signal_at_before=None,
        last_signal_at_after=NOW,
    )


def _completed() -> InspirationCompletedV1:
    return InspirationCompletedV1(
        schema_version=1,
        run_id=RUN_ID,
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.COMPLETED,
        operator_outcomes=(_successful_outcome(),),
        output_tokens_reserved_before=1_200,
        output_tokens_reserved_after=1_200,
        output_tokens_consumed_before=300,
        output_tokens_consumed_after=300,
        elapsed_milliseconds_before=100,
        elapsed_milliseconds_after=100,
    )


def _failed() -> InspirationFailedV1:
    outcome = _failed_outcome()
    return InspirationFailedV1(
        schema_version=1,
        run_id=RUN_ID,
        failure_code=InspirationRunFailureCode.ALL_OPERATORS_FAILED,
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.FAILED,
        operator_outcomes=(outcome,),
        output_tokens_reserved_before=1_200,
        output_tokens_reserved_after=1_200,
        output_tokens_consumed_before=1_200,
        output_tokens_consumed_after=1_200,
        elapsed_milliseconds_before=30_000,
        elapsed_milliseconds_after=30_000,
    )


def _timed_out() -> InspirationTimedOutV1:
    outcome = _failed_outcome(
        operator=InspirationOperator.CAUSAL_GAP,
        code=OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED,
    )
    return InspirationTimedOutV1(
        schema_version=1,
        run_id=RUN_ID,
        failure_code=OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED,
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.TIMED_OUT,
        operator_outcomes=(outcome,),
        output_tokens_reserved_before=1_200,
        output_tokens_reserved_after=1_200,
        output_tokens_consumed_before=1_200,
        output_tokens_consumed_after=1_200,
        elapsed_milliseconds_before=90_000,
        elapsed_milliseconds_after=90_000,
    )


def _idea_evaluated() -> InspirationIdeaEvaluatedV1:
    return InspirationIdeaEvaluatedV1(
        schema_version=1,
        idea_id=IDEA_ID,
        evaluator_agent_id=EVALUATOR_ID,
        mechanism_cluster_id=CLUSTER_ID,
        revision=1,
        previous_verdict=None,
        current_verdict=EvaluationVerdict.SUPPORTED,
        evidence=(
            _snapshot_evidence(),
            ExperienceVersionEvidenceReference(id=EXPERIENCE_VERSION_ID),
        ),
        reason=None,
        owner_decision_before=IdeaOwnerDecision.ACTIVE,
        owner_decision_after=IdeaOwnerDecision.ACTIVE,
        supported_count_before=0,
        supported_count_after=1,
        refuted_count_before=0,
        refuted_count_after=0,
        maturity_before=MechanismMaturity.INCUBATING,
        maturity_after=MechanismMaturity.CANDIDATE,
        candidate_since_before=None,
        candidate_since_after=NOW,
        last_signal_at_before=EARLIER,
        last_signal_at_after=NOW,
    )


def _idea_adopted() -> InspirationIdeaAdoptedV2:
    return InspirationIdeaAdoptedV2(
        schema_version=2,
        adoption_id=ADOPTION_ID,
        idea_id=IDEA_ID,
        run_id=RUN_ID,
        owner_agent_id=OWNER_ID,
        snapshot_hash=SNAPSHOT_HASH,
        evidence=(_snapshot_evidence(),),
        resulting_experience_id=RESULTING_EXPERIENCE_ID,
        resulting_version_id=RESULTING_VERSION_ID,
        created=True,
        requested_importance=0.4,
        requested_confidence=0.35,
        mechanism_cluster_id=CLUSTER_ID,
        owner_decision_before=IdeaOwnerDecision.ACTIVE,
        owner_decision_after=IdeaOwnerDecision.ADOPTED,
        distinct_adopter_count_before=1,
        distinct_adopter_count_after=2,
        maturity_before=MechanismMaturity.INCUBATING,
        maturity_after=MechanismMaturity.CANDIDATE,
        candidate_since_before=None,
        candidate_since_after=NOW,
        last_signal_at_before=EARLIER,
        last_signal_at_after=NOW,
    )


def _legacy_idea_adopted() -> InspirationIdeaAdoptedV1:
    data = _idea_adopted().model_dump(
        mode="python",
        exclude={"requested_importance", "requested_confidence"},
    )
    data["schema_version"] = 1
    return InspirationIdeaAdoptedV1.model_validate(data)


def _idea_rejected() -> InspirationIdeaRejectedV1:
    return InspirationIdeaRejectedV1(
        schema_version=1,
        idea_id=IDEA_ID,
        owner_agent_id=OWNER_ID,
        reason=_reason(),
        owner_decision_before=IdeaOwnerDecision.ACTIVE,
        owner_decision_after=IdeaOwnerDecision.REJECTED,
    )


def _idea_archived() -> InspirationIdeaArchivedV1:
    return InspirationIdeaArchivedV1(
        schema_version=1,
        idea_id=IDEA_ID,
        owner_agent_id=OWNER_ID,
        cycle_id=CYCLE_ID,
        reason=StructuredReason.policy_due(),
        owner_decision_before=IdeaOwnerDecision.ACTIVE,
        owner_decision_after=IdeaOwnerDecision.ARCHIVED,
    )


def _golden_payloads() -> tuple[EventPayload, ...]:
    return (
        _started(),
        _snapshot_frozen(),
        _operator_completed(),
        _operator_failed(),
        _idea_generated(),
        _completed(),
        _failed(),
        _timed_out(),
        _idea_evaluated(),
        _legacy_idea_adopted(),
        _idea_adopted(),
        _idea_rejected(),
        _idea_archived(),
    )


def test_registry_has_exact_inspiration_vocabulary_and_aggregate_ownership() -> None:
    registry = EventRegistry()

    register_inspiration_events(registry)

    expected_payload_types = {
        "inspiration.started": InspirationStartedV1,
        "inspiration.snapshot_frozen": InspirationSnapshotFrozenV1,
        "inspiration.operator_completed": InspirationOperatorCompletedV1,
        "inspiration.operator_failed": InspirationOperatorFailedV1,
        "inspiration.idea_generated": InspirationIdeaGeneratedV1,
        "inspiration.completed": InspirationCompletedV1,
        "inspiration.failed": InspirationFailedV1,
        "inspiration.timed_out": InspirationTimedOutV1,
        "inspiration.idea_evaluated": InspirationIdeaEvaluatedV1,
        "inspiration.idea_adopted": InspirationIdeaAdoptedV1,
        "inspiration.idea_adopted_v2": InspirationIdeaAdoptedV2,
        "inspiration.idea_rejected": InspirationIdeaRejectedV1,
        "inspiration.idea_archived": InspirationIdeaArchivedV1,
    }
    expected_ownership = {
        event_type: (
            "idea"
            if event_type
            in {
                "inspiration.idea_generated",
                "inspiration.idea_evaluated",
                "inspiration.idea_adopted",
                "inspiration.idea_adopted_v2",
                "inspiration.idea_rejected",
                "inspiration.idea_archived",
            }
            else "inspiration_run"
        )
        for event_type in expected_payload_types
    }

    assert (
        registry.event_types
        == INSPIRATION_EVENT_TYPES
        == frozenset(expected_payload_types)
    )
    assert expected_ownership == INSPIRATION_EVENT_AGGREGATE_TYPES
    for event_type, payload_type in expected_payload_types.items():
        assert registry.payload_type(event_type) is payload_type


def test_payloads_freeze_exact_required_fields() -> None:
    expected_fields = {
        InspirationStartedV1: {
            "schema_version",
            "run_id",
            "owner_agent_id",
            "status_after",
        },
        InspirationSnapshotFrozenV1: {
            "schema_version",
            "run_id",
            "snapshot_hash",
            "snapshot_item_ids",
            "status_before",
            "status_after",
        },
        InspirationOperatorCompletedV1: {
            "schema_version",
            "run_id",
            "operator",
            "outcome",
            "status_before",
            "status_after",
            "output_tokens_reserved_before",
            "output_tokens_reserved_after",
            "output_tokens_consumed_before",
            "output_tokens_consumed_after",
            "elapsed_milliseconds_before",
            "elapsed_milliseconds_after",
        },
        InspirationOperatorFailedV1: {
            "schema_version",
            "run_id",
            "operator",
            "outcome",
            "status_before",
            "status_after",
            "output_tokens_reserved_before",
            "output_tokens_reserved_after",
            "output_tokens_consumed_before",
            "output_tokens_consumed_after",
            "elapsed_milliseconds_before",
            "elapsed_milliseconds_after",
        },
        InspirationIdeaGeneratedV1: {
            "schema_version",
            "idea_id",
            "occurrence_id",
            "run_id",
            "owner_agent_id",
            "operator",
            "ordinal",
            "snapshot_hash",
            "evidence",
            "idea_content_hash",
            "mechanism_hash",
            "duplicate_relation",
            "owner_decision_after",
            "cluster_id",
            "canonical_mechanism_hash",
            "member_hashes_before",
            "member_hashes_after",
            "occurrence_count_before",
            "occurrence_count_after",
            "distinct_snapshot_count_before",
            "distinct_snapshot_count_after",
            "distinct_adopter_count_before",
            "distinct_adopter_count_after",
            "supported_count_before",
            "supported_count_after",
            "refuted_count_before",
            "refuted_count_after",
            "maturity_before",
            "maturity_after",
            "candidate_since_before",
            "candidate_since_after",
            "last_signal_at_before",
            "last_signal_at_after",
        },
        InspirationCompletedV1: {
            "schema_version",
            "run_id",
            "status_before",
            "status_after",
            "operator_outcomes",
            "output_tokens_reserved_before",
            "output_tokens_reserved_after",
            "output_tokens_consumed_before",
            "output_tokens_consumed_after",
            "elapsed_milliseconds_before",
            "elapsed_milliseconds_after",
        },
        InspirationFailedV1: {
            "schema_version",
            "run_id",
            "failure_code",
            "status_before",
            "status_after",
            "operator_outcomes",
            "output_tokens_reserved_before",
            "output_tokens_reserved_after",
            "output_tokens_consumed_before",
            "output_tokens_consumed_after",
            "elapsed_milliseconds_before",
            "elapsed_milliseconds_after",
        },
        InspirationTimedOutV1: {
            "schema_version",
            "run_id",
            "failure_code",
            "status_before",
            "status_after",
            "operator_outcomes",
            "output_tokens_reserved_before",
            "output_tokens_reserved_after",
            "output_tokens_consumed_before",
            "output_tokens_consumed_after",
            "elapsed_milliseconds_before",
            "elapsed_milliseconds_after",
        },
        InspirationIdeaEvaluatedV1: {
            "schema_version",
            "idea_id",
            "evaluator_agent_id",
            "mechanism_cluster_id",
            "revision",
            "previous_verdict",
            "current_verdict",
            "evidence",
            "reason",
            "owner_decision_before",
            "owner_decision_after",
            "supported_count_before",
            "supported_count_after",
            "refuted_count_before",
            "refuted_count_after",
            "maturity_before",
            "maturity_after",
            "candidate_since_before",
            "candidate_since_after",
            "last_signal_at_before",
            "last_signal_at_after",
        },
        InspirationIdeaAdoptedV2: {
            "schema_version",
            "adoption_id",
            "idea_id",
            "run_id",
            "owner_agent_id",
            "snapshot_hash",
            "evidence",
            "resulting_experience_id",
            "resulting_version_id",
            "created",
            "requested_importance",
            "requested_confidence",
            "mechanism_cluster_id",
            "owner_decision_before",
            "owner_decision_after",
            "distinct_adopter_count_before",
            "distinct_adopter_count_after",
            "maturity_before",
            "maturity_after",
            "candidate_since_before",
            "candidate_since_after",
            "last_signal_at_before",
            "last_signal_at_after",
        },
        InspirationIdeaAdoptedV1: {
            "schema_version",
            "adoption_id",
            "idea_id",
            "run_id",
            "owner_agent_id",
            "snapshot_hash",
            "evidence",
            "resulting_experience_id",
            "resulting_version_id",
            "created",
            "mechanism_cluster_id",
            "owner_decision_before",
            "owner_decision_after",
            "distinct_adopter_count_before",
            "distinct_adopter_count_after",
            "maturity_before",
            "maturity_after",
            "candidate_since_before",
            "candidate_since_after",
            "last_signal_at_before",
            "last_signal_at_after",
        },
        InspirationIdeaRejectedV1: {
            "schema_version",
            "idea_id",
            "owner_agent_id",
            "reason",
            "owner_decision_before",
            "owner_decision_after",
        },
        InspirationIdeaArchivedV1: {
            "schema_version",
            "idea_id",
            "owner_agent_id",
            "cycle_id",
            "reason",
            "owner_decision_before",
            "owner_decision_after",
        },
    }

    assert {type(payload) for payload in _golden_payloads()} == set(expected_fields)
    for payload in _golden_payloads():
        assert set(type(payload).model_fields) == expected_fields[type(payload)]


@pytest.mark.parametrize(
    "payload",
    _golden_payloads(),
    ids=lambda payload: payload.event_type,
)
def test_every_payload_is_strict_required_and_registry_round_trips(
    payload: EventPayload,
) -> None:
    registry = EventRegistry()
    register_inspiration_events(registry)
    values = payload.model_dump(mode="python")
    payload_type = type(payload)

    assert (
        registry.decode(
            event_type=payload.event_type,
            payload=payload.model_dump_json().encode(),
        )
        == payload
    )
    with pytest.raises(ValidationError):
        payload_type.model_validate({**values, "unexpected": True})
    with pytest.raises(ValidationError):
        payload_type.model_validate(
            {
                **values,
                "schema_version": 1 if values["schema_version"] == 2 else 2,
            }
        )
    for field_name in payload_type.model_fields:
        incomplete = dict(values)
        incomplete.pop(field_name)
        with pytest.raises(ValidationError):
            payload_type.model_validate(incomplete)


def test_registry_decodes_the_original_v1_adoption_json_shape() -> None:
    registry = EventRegistry()
    register_inspiration_events(registry)
    legacy_json = _legacy_idea_adopted().model_dump_json().encode()

    assert b"requested_importance" not in legacy_json
    assert b"requested_confidence" not in legacy_json
    decoded = registry.decode(
        event_type="inspiration.idea_adopted",
        payload=legacy_json,
    )

    assert decoded == _legacy_idea_adopted()
    assert type(decoded) is InspirationIdeaAdoptedV1


def test_golden_success_order_uses_run_and_idea_aggregate_identities() -> None:
    payloads = (
        _started(),
        _snapshot_frozen(),
        _idea_generated(),
        _operator_completed(),
        _completed(),
    )
    events = tuple(
        PendingEvent(
            aggregate_type=INSPIRATION_EVENT_AGGREGATE_TYPES[payload.event_type],
            aggregate_id=(
                payload.idea_id
                if isinstance(payload, InspirationIdeaGeneratedV1)
                else payload.run_id
            ),
            event_type=payload.event_type,
            payload=payload,
            actor_agent_id=OWNER_ID,
            occurred_at=NOW,
        )
        for payload in payloads
    )

    assert [event.event_type for event in events] == [
        "inspiration.started",
        "inspiration.snapshot_frozen",
        "inspiration.idea_generated",
        "inspiration.operator_completed",
        "inspiration.completed",
    ]
    assert [event.aggregate_type for event in events] == [
        "inspiration_run",
        "inspiration_run",
        "idea",
        "inspiration_run",
        "inspiration_run",
    ]
    assert [event.aggregate_id for event in events] == [
        RUN_ID,
        RUN_ID,
        IDEA_ID,
        RUN_ID,
        RUN_ID,
    ]
    generated = events[2].payload
    assert isinstance(generated, InspirationIdeaGeneratedV1)
    assert generated.occurrence_id == OCCURRENCE_ID
    assert generated.run_id == RUN_ID
    assert generated.evidence == (_snapshot_evidence(),)
    assert (
        generated.occurrence_count_before,
        generated.occurrence_count_after,
        generated.distinct_snapshot_count_before,
        generated.distinct_snapshot_count_after,
    ) == (0, 1, 0, 1)


def test_empty_frozen_snapshot_is_valid_but_snapshot_ids_cannot_repeat() -> None:
    empty = _snapshot_frozen().model_copy(update={"snapshot_item_ids": ()})

    assert (
        InspirationSnapshotFrozenV1.model_validate(empty.model_dump(mode="python"))
        == empty
    )
    with pytest.raises(ValidationError, match="repeat"):
        InspirationSnapshotFrozenV1.model_validate(
            {
                **_snapshot_frozen().model_dump(mode="python"),
                "snapshot_item_ids": (SNAPSHOT_ITEM_ID, SNAPSHOT_ITEM_ID),
            }
        )


@pytest.mark.parametrize(
    "payload",
    _golden_payloads(),
    ids=lambda payload: payload.event_type,
)
@pytest.mark.parametrize(
    "forbidden_field",
    (
        "prompt",
        "provider_response",
        "raw_exception",
        "exception_text",
        "experience_body",
        "capsule_body",
        "body",
        "query",
    ),
)
def test_event_payloads_forbid_sensitive_or_raw_material(
    payload: EventPayload,
    forbidden_field: str,
) -> None:
    fields = set(type(payload).model_fields)
    assert forbidden_field not in fields
    with pytest.raises(ValidationError):
        type(payload).model_validate(
            {
                **payload.model_dump(mode="python"),
                forbidden_field: "private material",
            }
        )


def test_started_and_snapshot_events_require_running_state() -> None:
    with pytest.raises(ValidationError, match="running"):
        InspirationStartedV1.model_validate(
            {
                **_started().model_dump(mode="python"),
                "status_after": InspirationRunStatus.COMPLETED,
            }
        )
    for field_name in ("status_before", "status_after"):
        with pytest.raises(ValidationError, match="running"):
            InspirationSnapshotFrozenV1.model_validate(
                {
                    **_snapshot_frozen().model_dump(mode="python"),
                    field_name: InspirationRunStatus.COMPLETED,
                }
            )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"operator": InspirationOperator.COUNTERFACTUAL},
            "operator",
        ),
        (
            {
                "outcome": _failed_outcome(
                    operator=InspirationOperator.CAUSAL_GAP,
                ),
                "output_tokens_consumed_after": 1_200,
            },
            "successful outcome",
        ),
        (
            {"status_after": InspirationRunStatus.COMPLETED},
            "running",
        ),
        (
            {"output_tokens_reserved_after": 1_201},
            "reservation",
        ),
        (
            {"output_tokens_consumed_after": 299},
            "consumption",
        ),
        (
            {"output_tokens_reserved_after": 200},
            "reservation",
        ),
        (
            {"elapsed_milliseconds_after": -1},
            "greater than or equal to 0",
        ),
    ),
)
def test_operator_completed_rejects_forged_outcome_or_accounting(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        InspirationOperatorCompletedV1.model_validate(
            {
                **_operator_completed().model_dump(mode="python"),
                **changes,
            }
        )


def test_operator_failed_accepts_skipped_zero_reservation_and_rejects_success() -> None:
    skipped_outcome = _failed_outcome(
        code=OperatorFailureCode.INSUFFICIENT_TOKEN_RESERVATION,
        consumed=0,
    )
    skipped = InspirationOperatorFailedV1.model_validate(
        {
            **_operator_failed().model_dump(mode="python"),
            "outcome": skipped_outcome,
            "output_tokens_reserved_after": 1_200,
            "output_tokens_consumed_after": 300,
            "elapsed_milliseconds_after": 100,
        }
    )

    assert skipped.outcome == skipped_outcome
    with pytest.raises(ValidationError, match="failed outcome"):
        InspirationOperatorFailedV1.model_validate(
            {
                **_operator_failed().model_dump(mode="python"),
                "outcome": _successful_outcome(),
                "operator": InspirationOperator.CAUSAL_GAP,
                "output_tokens_consumed_after": 600,
            }
        )


@pytest.mark.parametrize(
    ("payload_type", "payload", "changes", "message"),
    (
        (
            InspirationCompletedV1,
            _completed(),
            {"status_after": InspirationRunStatus.FAILED},
            "completed status",
        ),
        (
            InspirationCompletedV1,
            _completed(),
            {
                "operator_outcomes": (
                    _failed_outcome(
                        operator=InspirationOperator.CAUSAL_GAP,
                    ),
                ),
                "output_tokens_reserved_before": 1_200,
                "output_tokens_reserved_after": 1_200,
                "output_tokens_consumed_before": 1_200,
                "output_tokens_consumed_after": 1_200,
            },
            "completed status",
        ),
        (
            InspirationCompletedV1,
            _completed(),
            {"status_after": InspirationRunStatus.COMPLETED_WITH_ERRORS},
            "mixed",
        ),
        (
            InspirationCompletedV1,
            _completed(),
            {"output_tokens_reserved_after": 1_201},
            "terminal accounting",
        ),
        (
            InspirationFailedV1,
            _failed(),
            {
                "operator_outcomes": (_successful_outcome(),),
                "output_tokens_consumed_before": 300,
                "output_tokens_consumed_after": 300,
            },
            "cannot retain a successful",
        ),
        (
            InspirationFailedV1,
            _failed(),
            {
                "failure_code": InspirationRunFailureCode.PREPARATION_FAILED,
            },
            "must not retain operator outcomes",
        ),
        (
            InspirationTimedOutV1,
            _timed_out(),
            {
                "operator_outcomes": (
                    _failed_outcome(
                        code=OperatorFailureCode.PROVIDER_TIMEOUT,
                    ),
                )
            },
            "global deadline",
        ),
    ),
)
def test_terminal_events_reject_status_outcome_and_accounting_forgery(
    payload_type: type[EventPayload],
    payload: EventPayload,
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        payload_type.model_validate(
            {
                **payload.model_dump(mode="python"),
                **changes,
            }
        )


def test_completed_with_errors_requires_canonical_mixed_outcomes() -> None:
    mixed = InspirationCompletedV1.model_validate(
        {
            **_completed().model_dump(mode="python"),
            "status_after": InspirationRunStatus.COMPLETED_WITH_ERRORS,
            "operator_outcomes": (
                _successful_outcome(),
                _failed_outcome(),
            ),
            "output_tokens_reserved_before": 2_400,
            "output_tokens_reserved_after": 2_400,
            "output_tokens_consumed_before": 1_500,
            "output_tokens_consumed_after": 1_500,
            "elapsed_milliseconds_before": 30_100,
            "elapsed_milliseconds_after": 30_100,
        }
    )

    assert mixed.status_after is InspirationRunStatus.COMPLETED_WITH_ERRORS
    with pytest.raises(ValidationError, match="canonical operator order"):
        InspirationCompletedV1.model_validate(
            {
                **mixed.model_dump(mode="python"),
                "operator_outcomes": tuple(reversed(mixed.operator_outcomes)),
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"occurrence_count_after": 2}, "increment occurrence"),
        ({"distinct_snapshot_count_after": 0}, "after snapshot"),
        ({"mechanism_hash": "e" * 64}, "mechanism_hash"),
        ({"cluster_id": "e" * 64}, "cluster identity"),
        (
            {"member_hashes_after": (MECHANISM_HASH, MECHANISM_HASH)},
            "unique",
        ),
        (
            {"maturity_after": MechanismMaturity.CANDIDATE},
            "candidate_since_after",
        ),
        (
            {"last_signal_at_after": EARLIER.replace(tzinfo=None)},
            "timezone-aware",
        ),
        ({"owner_decision_after": IdeaOwnerDecision.ARCHIVED}, "active"),
        ({"duplicate_relation": IDEA_ID}, "itself"),
        ({"evidence": ()}, "evidence must not be empty"),
        (
            {"evidence": (_snapshot_evidence(), _snapshot_evidence())},
            "repeat",
        ),
    ),
)
def test_generated_event_rejects_invalid_source_or_cluster_transition(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        InspirationIdeaGeneratedV1.model_validate(
            {
                **_idea_generated().model_dump(mode="python"),
                **changes,
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"revision": 2}, "previous_verdict"),
        (
            {
                "revision": 2,
                "previous_verdict": EvaluationVerdict.SUPPORTED,
                "supported_count_before": 1,
                "supported_count_after": 2,
            },
            "effective counts",
        ),
        ({"evidence": ()}, "evidence must not be empty"),
        (
            {"evidence": (_snapshot_evidence(), _snapshot_evidence())},
            "repeat",
        ),
        (
            {"owner_decision_after": IdeaOwnerDecision.ARCHIVED},
            "owner decision",
        ),
        (
            {"candidate_since_after": None},
            "candidate_since_after",
        ),
        (
            {"last_signal_at_after": OLDER},
            "last_signal_at",
        ),
    ),
)
def test_evaluated_event_rejects_invalid_revision_evidence_or_transition(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        InspirationIdeaEvaluatedV1.model_validate(
            {
                **_idea_evaluated().model_dump(mode="python"),
                **changes,
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"evidence": ()}, "evidence must not be empty"),
        (
            {"evidence": (_snapshot_evidence(), _snapshot_evidence())},
            "repeat",
        ),
        (
            {"owner_decision_before": IdeaOwnerDecision.REJECTED},
            "active or archived",
        ),
        (
            {"owner_decision_after": IdeaOwnerDecision.ACTIVE},
            "adopted",
        ),
        ({"distinct_adopter_count_after": 3}, "at most one"),
        ({"candidate_since_after": None}, "candidate_since_after"),
        ({"last_signal_at_after": OLDER}, "last_signal_at"),
    ),
)
def test_adopted_event_rejects_invalid_provenance_or_transition(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        InspirationIdeaAdoptedV2.model_validate(
            {
                **_idea_adopted().model_dump(mode="python"),
                **changes,
            }
        )


def test_reject_and_archive_require_exact_owner_decision_transitions() -> None:
    for before in (IdeaOwnerDecision.ACTIVE, IdeaOwnerDecision.ARCHIVED):
        rejected = InspirationIdeaRejectedV1.model_validate(
            {
                **_idea_rejected().model_dump(mode="python"),
                "owner_decision_before": before,
            }
        )
        assert rejected.owner_decision_after is IdeaOwnerDecision.REJECTED

    with pytest.raises(ValidationError, match="active or archived"):
        InspirationIdeaRejectedV1.model_validate(
            {
                **_idea_rejected().model_dump(mode="python"),
                "owner_decision_before": IdeaOwnerDecision.ADOPTED,
            }
        )
    with pytest.raises(ValidationError, match="active"):
        InspirationIdeaArchivedV1.model_validate(
            {
                **_idea_archived().model_dump(mode="python"),
                "owner_decision_before": IdeaOwnerDecision.ARCHIVED,
            }
        )
    with pytest.raises(ValidationError, match="policy-due"):
        InspirationIdeaArchivedV1.model_validate(
            {
                **_idea_archived().model_dump(mode="python"),
                "reason": _reason(),
            }
        )


def test_explicit_archive_has_no_cycle_and_retains_required_user_reason() -> None:
    explicit = InspirationIdeaArchivedV1.model_validate(
        {
            **_idea_archived().model_dump(mode="python"),
            "cycle_id": None,
            "reason": _reason(),
        }
    )

    assert explicit.cycle_id is None
    assert explicit.reason == _reason()


def test_snapshot_rejects_more_than_twelve_item_identities() -> None:
    item_ids = tuple(UUID(int=index) for index in range(1, 14))

    with pytest.raises(ValidationError, match="at most 12"):
        InspirationSnapshotFrozenV1.model_validate(
            {
                **_snapshot_frozen().model_dump(mode="python"),
                "snapshot_item_ids": item_ids,
            }
        )


def test_nested_outcome_is_revalidated_after_unsafe_construction() -> None:
    invalid = OperatorOutcome.model_construct(
        operator=InspirationOperator.CAUSAL_GAP,
        succeeded=True,
        persisted_ideas=0,
        duplicate_count=0,
        error_code=None,
        output_tokens_consumed=0,
    )

    with pytest.raises(ValidationError, match="persist"):
        InspirationOperatorCompletedV1.model_validate(
            {
                **_operator_completed().model_dump(mode="python"),
                "outcome": invalid,
                "output_tokens_consumed_after": 0,
            }
        )


@pytest.mark.parametrize(
    "payload_type",
    (
        InspirationIdeaGeneratedV1,
        InspirationIdeaAdoptedV2,
        InspirationIdeaEvaluatedV1,
    ),
)
def test_nested_evidence_is_revalidated_after_unsafe_construction(
    payload_type: type[EventPayload],
) -> None:
    invalid = SnapshotEvidenceReference.model_construct(
        type="snapshot_item",
        id=SNAPSHOT_ITEM_ID,
        stable_evidence_key="private",
    )
    payload_by_type: dict[type[EventPayload], EventPayload] = {
        InspirationIdeaGeneratedV1: _idea_generated(),
        InspirationIdeaAdoptedV2: _idea_adopted(),
        InspirationIdeaEvaluatedV1: _idea_evaluated(),
    }

    with pytest.raises(ValidationError, match="SHA-256"):
        payload_type.model_validate(
            {
                **payload_by_type[payload_type].model_dump(mode="python"),
                "evidence": (invalid,),
            }
        )


def test_nested_reason_is_revalidated_after_unsafe_construction() -> None:
    invalid = StructuredReason.model_construct(
        code="PRIVATE ERROR",
        text="raw traceback",
        text_hash="not-a-hash",
    )

    with pytest.raises(ValidationError, match="Reason"):
        InspirationIdeaRejectedV1.model_validate(
            {
                **_idea_rejected().model_dump(mode="python"),
                "reason": invalid,
            }
        )


def test_evaluation_rejects_same_snapshot_stable_key_under_another_id() -> None:
    repeated_key = SnapshotEvidenceReference(
        id=UUID("00000000-0000-0000-0000-000000000999"),
        stable_evidence_key=STABLE_EVIDENCE_KEY,
    )

    with pytest.raises(ValidationError, match="repeat"):
        InspirationIdeaEvaluatedV1.model_validate(
            {
                **_idea_evaluated().model_dump(mode="python"),
                "evidence": (_snapshot_evidence(), repeated_key),
            }
        )


@pytest.mark.parametrize(
    "failure_code",
    (
        InspirationRunFailureCode.PREPARATION_FAILED,
        InspirationRunFailureCode.PROCESS_INTERRUPTED,
    ),
)
def test_preparation_and_recovery_failures_are_sanitized_empty_terminals(
    failure_code: InspirationRunFailureCode,
) -> None:
    payload = InspirationFailedV1(
        schema_version=1,
        run_id=RUN_ID,
        failure_code=failure_code,
        status_before=InspirationRunStatus.RUNNING,
        status_after=InspirationRunStatus.FAILED,
        operator_outcomes=(),
        output_tokens_reserved_before=0,
        output_tokens_reserved_after=0,
        output_tokens_consumed_before=0,
        output_tokens_consumed_after=0,
        elapsed_milliseconds_before=0,
        elapsed_milliseconds_after=0,
    )

    assert payload.failure_code is failure_code
