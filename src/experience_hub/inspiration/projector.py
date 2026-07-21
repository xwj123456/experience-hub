"""Strict rebuildable projections for durable inspiration events."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry, StoredEvent
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.inspiration.adoption import adopted_hypothesis_content
from experience_hub.inspiration.events import (
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
)
from experience_hub.inspiration.incubation import (
    AdoptionTransition,
    ClusterTransition,
    EvaluationTransition,
    IncubationCluster,
    IncubationMember,
    plan_adoption_transition,
    plan_evaluation_transition,
    plan_occurrence,
)
from experience_hub.inspiration.models import (
    ExperienceVersionEvidenceReference,
    IdeaOwnerDecision,
    InspirationRunStatus,
    MechanismIncubation,
    MechanismMaturity,
    OperatorOutcome,
    SnapshotEvidenceReference,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationSnapshotItemRow,
)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NONCANDIDATE_RETENTION = timedelta(days=180)
_CANDIDATE_RETENTION = timedelta(days=365)
_ADOPTION_EVENT_TYPES = frozenset(
    {
        InspirationIdeaAdoptedV1.event_type,
        InspirationIdeaAdoptedV2.event_type,
    }
)
_ADOPTION_PAYLOAD_TYPES = (
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
)
type _AdoptionPayload = InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2


class InspirationProjectionIntegrityError(RuntimeError):
    """An event cannot be reconciled with inspiration source/projection state."""

    code = "inspiration_projection_integrity_error"


def _fail(message: str) -> InspirationProjectionIntegrityError:
    return InspirationProjectionIntegrityError(message)


def _target(prefix: str | None, name: str) -> str:
    table = name if prefix is None else f"{prefix}{name}"
    if not _SAFE_IDENTIFIER.fullmatch(table):
        raise ValueError("unsafe projection table identifier")
    return f'"{table}"'


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return require_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parsed_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _fail("projection timestamp is invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as error:
        raise _fail("projection timestamp is invalid") from error


def _stored_event(registry: EventRegistry, row: DomainEventRow) -> StoredEvent:
    try:
        payload = registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
    except (TypeError, ValueError) as error:
        raise _fail(f"event {row.event_id} payload is invalid") from error
    return StoredEvent(
        event_id=row.event_id,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        sequence=row.sequence,
        event_type=row.event_type,
        payload=payload,
        actor_agent_id=row.actor_agent_id,
        causation_id=row.causation_id,
        occurred_at=row.occurred_at,
    )


async def _experience_state_before_event(
    session: AsyncSession,
    *,
    experience_id: UUID,
    event_id: int,
    registry: EventRegistry,
) -> tuple[ExperienceStateSnapshotV1, StoredEvent]:
    row = await session.scalar(
        select(DomainEventRow)
        .where(
            DomainEventRow.aggregate_type == "experience",
            DomainEventRow.aggregate_id == experience_id,
            DomainEventRow.event_id < event_id,
        )
        .order_by(DomainEventRow.event_id.desc())
        .limit(1)
    )
    if row is None:
        raise _fail("adopted experience has no event state before adoption")
    prior = _stored_event(registry, row)
    after = getattr(prior.payload, "after", None)
    if not isinstance(after, ExperienceStateSnapshotV1):
        raise _fail("adopted experience predecessor has no state snapshot")
    return after, prior


def _require_run_anchor(event: StoredEvent, run_id: UUID) -> None:
    if (
        event.event_id < 1
        or event.aggregate_type != "inspiration_run"
        or event.aggregate_id != run_id
    ):
        raise _fail("run event has an invalid aggregate anchor")
    try:
        require_utc(event.occurred_at)
    except (TypeError, ValueError) as error:
        raise _fail("run event has an invalid causal time") from error


def _require_idea_anchor(event: StoredEvent, idea_id: UUID) -> None:
    if (
        event.event_id < 1
        or event.aggregate_type != "idea"
        or event.aggregate_id != idea_id
        or event.sequence != 1
    ):
        raise _fail("generated idea event has an invalid aggregate anchor")


def _decode_outcomes(raw: Any) -> tuple[OperatorOutcome, ...]:
    if not isinstance(raw, (bytes, bytearray)):
        raise _fail("run projection outcomes are invalid")
    encoded = bytes(raw)
    try:
        decoded = json.loads(encoded)
        if canonical_json_bytes(decoded) != encoded or not isinstance(decoded, list):
            raise ValueError
        retained: list[OperatorOutcome] = []
        for value in decoded:
            parsed = OperatorOutcome.model_validate_json(canonical_json_bytes(value))
            retained.append(
                OperatorOutcome.model_validate(
                    parsed.model_dump(mode="python", warnings=False),
                    strict=True,
                )
            )
        return tuple(retained)
    except (TypeError, ValueError, ValidationError) as error:
        raise _fail("run projection outcomes are invalid") from error


def _run_counters_match(
    row: Mapping[str, Any],
    *,
    status: InspirationRunStatus,
    reserved: int,
    consumed: int,
    elapsed: int,
) -> bool:
    return bool(
        row["status"] == status.value
        and row["output_tokens_reserved"] == reserved
        and row["output_tokens_consumed"] == consumed
        and row["elapsed_milliseconds"] == elapsed
    )


async def _one_mapping(
    session: AsyncSession,
    statement: str,
    parameters: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    row = (
        (await session.execute(text(statement), dict(parameters)))
        .mappings()
        .one_or_none()
    )
    return None if row is None else cast(Mapping[str, Any], row)


async def _require_run_predecessor(
    session: AsyncSession,
    *,
    current: Mapping[str, Any],
    event: StoredEvent,
    payload: object,
) -> None:
    prior_id = current.get("projection_event_id")
    if isinstance(prior_id, bool) or not isinstance(prior_id, int):
        raise _fail("run projection checkpoint is invalid")
    prior = await session.get(DomainEventRow, prior_id)
    source = await session.get(InspirationRunRow, event.aggregate_id)
    if (
        prior is None
        or source is None
        or prior.aggregate_type != "inspiration_run"
        or prior.aggregate_id != event.aggregate_id
        or prior.event_id >= event.event_id
        or event.sequence != prior.sequence + 1
        or prior.sequence < 1
    ):
        raise _fail("run event predecessor is inconsistent")
    is_recovery = (
        isinstance(payload, InspirationFailedV1)
        and payload.failure_code is InspirationRunFailureCode.PROCESS_INTERRUPTED
    )
    if is_recovery:
        if (
            event.actor_agent_id is not None
            or event.causation_id == prior.causation_id
            or event.occurred_at < prior.occurred_at
        ):
            raise _fail("recovery terminal has invalid causal semantics")
    elif (
        event.actor_agent_id != source.owner_agent_id
        or event.causation_id != prior.causation_id
        or event.occurred_at != source.created_at
        or prior.occurred_at != source.created_at
    ):
        raise _fail("normal run event violates its logical command time")


class InspirationRunProjector:
    """Reduce run phase and budget events into one terminal state row."""

    name = "inspiration_run_state"
    version = 1
    event_types = frozenset(
        {
            InspirationStartedV1.event_type,
            InspirationSnapshotFrozenV1.event_type,
            InspirationOperatorCompletedV1.event_type,
            InspirationOperatorFailedV1.event_type,
            InspirationCompletedV1.event_type,
            InspirationFailedV1.event_type,
            InspirationTimedOutV1.event_type,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "run_id VARCHAR(36) NOT NULL PRIMARY KEY,"
                "status VARCHAR(21) NOT NULL,"
                "snapshot_hash VARCHAR(64),"
                "operator_outcomes BLOB NOT NULL,"
                "output_tokens_reserved INTEGER NOT NULL,"
                "output_tokens_consumed INTEGER NOT NULL,"
                "elapsed_milliseconds INTEGER NOT NULL,"
                "started_at VARCHAR(27) NOT NULL,"
                "completed_at VARCHAR(27),"
                "projection_event_id INTEGER NOT NULL)"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._registry, row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        payload = event.payload
        run_id = getattr(payload, "run_id", None)
        if not isinstance(run_id, UUID):
            raise _fail("run event payload has no run identity")
        _require_run_anchor(event, run_id)
        target = _target(target_prefix, self.name)
        if isinstance(payload, InspirationStartedV1):
            await self._apply_started(session, target, event, payload)
        elif isinstance(payload, InspirationSnapshotFrozenV1):
            await self._apply_snapshot(session, target, event, payload)
        elif isinstance(
            payload,
            (InspirationOperatorCompletedV1, InspirationOperatorFailedV1),
        ):
            await self._apply_operator(session, target, event, payload)
        elif isinstance(
            payload,
            (InspirationCompletedV1, InspirationFailedV1, InspirationTimedOutV1),
        ):
            await self._apply_terminal(session, target, event, payload)
        else:
            raise _fail(f"unsupported run event {event.event_type!r}")

    async def _apply_started(
        self,
        session: AsyncSession,
        target: str,
        event: StoredEvent,
        payload: InspirationStartedV1,
    ) -> None:
        source = await session.get(InspirationRunRow, payload.run_id)
        existing = await _one_mapping(
            session,
            f"SELECT run_id FROM {target} WHERE run_id=:run_id",
            {"run_id": str(payload.run_id)},
        )
        if (
            event.event_type != InspirationStartedV1.event_type
            or event.sequence != 1
            or source is None
            or source.owner_agent_id != payload.owner_agent_id
            or event.actor_agent_id != payload.owner_agent_id
            or source.created_at != event.occurred_at
            or payload.status_after is not InspirationRunStatus.RUNNING
            or existing is not None
        ):
            raise _fail("started event does not match its run source")
        await session.execute(
            text(
                f"INSERT INTO {target} (run_id,status,snapshot_hash,"
                "operator_outcomes,output_tokens_reserved,"
                "output_tokens_consumed,elapsed_milliseconds,started_at,"
                "completed_at,projection_event_id) VALUES "
                "(:run_id,:status,NULL,:outcomes,0,0,0,:started_at,NULL,:event_id)"
            ),
            {
                "run_id": str(payload.run_id),
                "status": InspirationRunStatus.RUNNING.value,
                "outcomes": canonical_json_bytes(()),
                "started_at": _timestamp(event.occurred_at),
                "event_id": event.event_id,
            },
        )

    async def _current(
        self,
        session: AsyncSession,
        target: str,
        run_id: UUID,
    ) -> Mapping[str, Any]:
        row = await _one_mapping(
            session,
            f"SELECT * FROM {target} WHERE run_id=:run_id",
            {"run_id": str(run_id)},
        )
        if row is None:
            raise _fail("run projection is missing")
        return row

    async def _apply_snapshot(
        self,
        session: AsyncSession,
        target: str,
        event: StoredEvent,
        payload: InspirationSnapshotFrozenV1,
    ) -> None:
        current = await self._current(session, target, payload.run_id)
        await _require_run_predecessor(
            session,
            current=current,
            event=event,
            payload=payload,
        )
        source_ids = tuple(
            await session.scalars(
                select(InspirationSnapshotItemRow.snapshot_item_id)
                .where(InspirationSnapshotItemRow.run_id == payload.run_id)
                .order_by(InspirationSnapshotItemRow.rank)
            )
        )
        if (
            event.event_type != InspirationSnapshotFrozenV1.event_type
            or event.sequence != 2
            or payload.status_before is not InspirationRunStatus.RUNNING
            or payload.status_after is not InspirationRunStatus.RUNNING
            or current["status"] != InspirationRunStatus.RUNNING.value
            or current["snapshot_hash"] is not None
            or source_ids != payload.snapshot_item_ids
        ):
            raise _fail("snapshot event does not match its locked run state")
        await session.execute(
            text(
                f"UPDATE {target} SET snapshot_hash=:snapshot_hash,"
                "projection_event_id=:event_id WHERE run_id=:run_id"
            ),
            {
                "snapshot_hash": payload.snapshot_hash,
                "event_id": event.event_id,
                "run_id": str(payload.run_id),
            },
        )

    async def _apply_operator(
        self,
        session: AsyncSession,
        target: str,
        event: StoredEvent,
        payload: InspirationOperatorCompletedV1 | InspirationOperatorFailedV1,
    ) -> None:
        current = await self._current(session, target, payload.run_id)
        await _require_run_predecessor(
            session,
            current=current,
            event=event,
            payload=payload,
        )
        outcomes = _decode_outcomes(current["operator_outcomes"])
        expected_class = (
            InspirationOperatorCompletedV1
            if payload.outcome.succeeded
            else InspirationOperatorFailedV1
        )
        reserved_delta = (
            payload.output_tokens_reserved_after - payload.output_tokens_reserved_before
        )
        consumed_delta = (
            payload.output_tokens_consumed_after - payload.output_tokens_consumed_before
        )
        if (
            not isinstance(payload, expected_class)
            or payload.status_before is not InspirationRunStatus.RUNNING
            or payload.status_after is not InspirationRunStatus.RUNNING
            or not _run_counters_match(
                current,
                status=payload.status_before,
                reserved=payload.output_tokens_reserved_before,
                consumed=payload.output_tokens_consumed_before,
                elapsed=payload.elapsed_milliseconds_before,
            )
            or payload.operator is not payload.outcome.operator
            or payload.operator in {outcome.operator for outcome in outcomes}
            or not 0 <= reserved_delta <= 1_200
            or consumed_delta != payload.outcome.output_tokens_consumed
            or not 0 <= consumed_delta <= reserved_delta
            or payload.elapsed_milliseconds_after < payload.elapsed_milliseconds_before
        ):
            raise _fail("operator event does not match locked budget state")
        revised = (*outcomes, payload.outcome)
        await session.execute(
            text(
                f"UPDATE {target} SET operator_outcomes=:outcomes,"
                "output_tokens_reserved=:reserved,"
                "output_tokens_consumed=:consumed,"
                "elapsed_milliseconds=:elapsed,"
                "projection_event_id=:event_id WHERE run_id=:run_id"
            ),
            {
                "outcomes": canonical_json_bytes(revised),
                "reserved": payload.output_tokens_reserved_after,
                "consumed": payload.output_tokens_consumed_after,
                "elapsed": payload.elapsed_milliseconds_after,
                "event_id": event.event_id,
                "run_id": str(payload.run_id),
            },
        )

    async def _apply_terminal(
        self,
        session: AsyncSession,
        target: str,
        event: StoredEvent,
        payload: InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1,
    ) -> None:
        current = await self._current(session, target, payload.run_id)
        await _require_run_predecessor(
            session,
            current=current,
            event=event,
            payload=payload,
        )
        outcomes = _decode_outcomes(current["operator_outcomes"])
        unchanged = (
            payload.output_tokens_reserved_before
            == payload.output_tokens_reserved_after
            and payload.output_tokens_consumed_before
            == payload.output_tokens_consumed_after
            and payload.elapsed_milliseconds_before
            == payload.elapsed_milliseconds_after
        )
        valid_terminal = False
        if isinstance(payload, InspirationCompletedV1):
            succeeded = sum(outcome.succeeded for outcome in outcomes)
            expected = (
                InspirationRunStatus.COMPLETED
                if succeeded == len(outcomes) and succeeded > 0
                else InspirationRunStatus.COMPLETED_WITH_ERRORS
            )
            valid_terminal = payload.status_after is expected and succeeded > 0
        elif isinstance(payload, InspirationFailedV1):
            valid_terminal = payload.status_after is InspirationRunStatus.FAILED
            if payload.failure_code is InspirationRunFailureCode.PREPARATION_FAILED:
                valid_terminal = valid_terminal and not outcomes
            elif payload.failure_code is InspirationRunFailureCode.ALL_OPERATORS_FAILED:
                valid_terminal = (
                    valid_terminal
                    and bool(outcomes)
                    and not any(outcome.succeeded for outcome in outcomes)
                )
        else:
            valid_terminal = (
                payload.status_after is InspirationRunStatus.TIMED_OUT
                and any(
                    outcome.error_code is not None
                    and outcome.error_code.value == "global_deadline_exhausted"
                    for outcome in outcomes
                )
            )
        if (
            payload.status_before is not InspirationRunStatus.RUNNING
            or not _run_counters_match(
                current,
                status=payload.status_before,
                reserved=payload.output_tokens_reserved_before,
                consumed=payload.output_tokens_consumed_before,
                elapsed=payload.elapsed_milliseconds_before,
            )
            or payload.operator_outcomes != outcomes
            or not unchanged
            or not valid_terminal
        ):
            raise _fail("terminal event does not match locked run state")
        await session.execute(
            text(
                f"UPDATE {target} SET status=:status,completed_at=:completed_at,"
                "projection_event_id=:event_id WHERE run_id=:run_id"
            ),
            {
                "status": payload.status_after.value,
                "completed_at": _timestamp(event.occurred_at),
                "event_id": event.event_id,
                "run_id": str(payload.run_id),
            },
        )


def _transition(payload: InspirationIdeaGeneratedV1) -> ClusterTransition:
    try:
        return ClusterTransition(
            cluster_id=payload.cluster_id,
            canonical_mechanism_hash=payload.canonical_mechanism_hash,
            member_hashes_before=payload.member_hashes_before,
            member_hashes_after=payload.member_hashes_after,
            occurrence_count_before=payload.occurrence_count_before,
            occurrence_count_after=payload.occurrence_count_after,
            distinct_snapshot_count_before=payload.distinct_snapshot_count_before,
            distinct_snapshot_count_after=payload.distinct_snapshot_count_after,
            distinct_adopter_count_before=payload.distinct_adopter_count_before,
            distinct_adopter_count_after=payload.distinct_adopter_count_after,
            supported_count_before=payload.supported_count_before,
            supported_count_after=payload.supported_count_after,
            refuted_count_before=payload.refuted_count_before,
            refuted_count_after=payload.refuted_count_after,
            maturity_before=payload.maturity_before,
            maturity_after=payload.maturity_after,
            candidate_since_before=payload.candidate_since_before,
            candidate_since_after=payload.candidate_since_after,
            last_signal_at_before=payload.last_signal_at_before,
            last_signal_at_after=payload.last_signal_at_after,
        )
    except (TypeError, ValueError) as error:
        raise _fail("generated idea cluster transition is invalid") from error


def _decoded_member_hashes(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (bytes, bytearray)):
        raise _fail("mechanism projection member hashes are invalid")
    encoded = bytes(raw)
    try:
        decoded = json.loads(encoded)
        if (
            canonical_json_bytes(decoded) != encoded
            or not isinstance(decoded, list)
            or any(not isinstance(value, str) for value in decoded)
        ):
            raise ValueError
        return tuple(decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise _fail("mechanism projection member hashes are invalid") from error


def _mechanism_state(row: Mapping[str, Any]) -> MechanismIncubation:
    candidate_since = _parsed_timestamp(row["candidate_since"])
    last_signal_at = _parsed_timestamp(row["last_signal_at"])
    if last_signal_at is None:
        raise _fail("mechanism projection last signal is missing")
    try:
        return MechanismIncubation(
            cluster_id=row["cluster_id"],
            canonical_mechanism_hash=row["canonical_mechanism_hash"],
            member_hashes=_decoded_member_hashes(row["member_hashes"]),
            occurrence_count=row["occurrence_count"],
            distinct_snapshot_count=row["distinct_snapshot_count"],
            distinct_adopter_count=row["distinct_adopter_count"],
            supported_count=row["supported_count"],
            refuted_count=row["refuted_count"],
            maturity=MechanismMaturity(row["maturity"]),
            candidate_since=candidate_since,
            last_signal_at=last_signal_at,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _fail("mechanism projection state is invalid") from error


async def _validate_policy_archive_source(
    session: AsyncSession,
    *,
    event: StoredEvent,
    payload: InspirationIdeaArchivedV1,
    current: Mapping[str, Any],
    target_prefix: str | None,
) -> None:
    if payload.cycle_id is None:
        raise _fail("automatic archive cycle is missing")
    receipt = await session.get(IdempotencyRecordRow, event.causation_id)
    if (
        receipt is None
        or receipt.caller_scope != "system:local"
        or receipt.scope != "lifecycle.run"
        or receipt.result_resource_type != "lifecycle_cycle"
        or receipt.result_resource_id != payload.cycle_id
        or receipt.state not in {"in_progress", "completed"}
    ):
        raise _fail("automatic archive lacks a valid lifecycle receipt")
    if target_prefix is not None:
        # Repair validates the complete source ledger before rebuilding.
        # The mechanism temp table already contains its final state, which
        # may legitimately differ from the state at this historical archive.
        return
    mechanism_target = _target(target_prefix, "mechanism_incubation")
    cluster = await _one_mapping(
        session,
        f"SELECT * FROM {mechanism_target} WHERE cluster_id=:cluster_id",
        {"cluster_id": current["mechanism_cluster_id"]},
    )
    if cluster is None:
        raise _fail("automatic archive mechanism projection is missing")
    mechanism = _mechanism_state(cluster)
    last_signal_at = _parsed_timestamp(current["last_signal_at"])
    if last_signal_at is None:
        raise _fail("automatic archive idea signal is missing")
    if mechanism.maturity is MechanismMaturity.CANDIDATE:
        if mechanism.candidate_since is None:
            raise _fail("candidate archive source has no candidate timestamp")
        due_at = max(last_signal_at, mechanism.candidate_since) + _CANDIDATE_RETENTION
    else:
        due_at = last_signal_at + _NONCANDIDATE_RETENTION
    if event.occurred_at < due_at:
        raise _fail("automatic archive is not due")


def _evaluation_transition_matches(
    payload: InspirationIdeaEvaluatedV1,
    transition: EvaluationTransition,
) -> bool:
    return bool(
        payload.previous_verdict is transition.previous_verdict
        and payload.current_verdict is transition.current_verdict
        and payload.supported_count_before == transition.supported_count_before
        and payload.supported_count_after == transition.supported_count_after
        and payload.refuted_count_before == transition.refuted_count_before
        and payload.refuted_count_after == transition.refuted_count_after
        and payload.maturity_before is transition.maturity_before
        and payload.maturity_after is transition.maturity_after
        and payload.candidate_since_before == transition.candidate_since_before
        and payload.candidate_since_after == transition.candidate_since_after
        and payload.last_signal_at_before == transition.last_signal_at_before
        and payload.last_signal_at_after == transition.last_signal_at_after
    )


def _adoption_transition_matches(
    payload: _AdoptionPayload,
    transition: AdoptionTransition,
) -> bool:
    return bool(
        payload.distinct_adopter_count_before
        == transition.distinct_adopter_count_before
        and payload.distinct_adopter_count_after
        == transition.distinct_adopter_count_after
        and payload.maturity_before is transition.maturity_before
        and payload.maturity_after is transition.maturity_after
        and payload.candidate_since_before == transition.candidate_since_before
        and payload.candidate_since_after == transition.candidate_since_after
        and payload.last_signal_at_before == transition.last_signal_at_before
        and payload.last_signal_at_after == transition.last_signal_at_after
    )


async def _validate_adoption_source(
    session: AsyncSession,
    *,
    event: StoredEvent,
    payload: _AdoptionPayload,
    registry: EventRegistry,
    target_prefix: str | None,
) -> tuple[
    InspirationIdeaRow,
    InspirationRunRow,
    tuple[StoredEvent, ...],
    IdeaOwnerDecision,
    dict[UUID, InspirationIdeaEvaluatedV1],
]:
    idea, run, history, decision, latest = await _idea_history_before(
        session,
        event=event,
        registry=registry,
        target_prefix=target_prefix,
    )
    generated = cast(InspirationIdeaGeneratedV1, history[0].payload)
    occurrence = await session.get(
        IdeaOccurrenceRow,
        generated.occurrence_id,
    )
    record = await session.get(
        IdeaAdoptionRecordRow,
        payload.adoption_id,
    )
    identity = await session.get(
        ExperienceRow,
        payload.resulting_experience_id,
    )
    version = await session.get(
        ExperienceVersionRow,
        payload.resulting_version_id,
    )
    body_payload = await session.get(
        ExperiencePayloadRow,
        payload.resulting_version_id,
    )
    if (
        occurrence is None
        or record is None
        or identity is None
        or version is None
        or body_payload is None
    ):
        raise _fail("idea adoption source anchor is missing")
    expected_item_ids = canonical_json_bytes(
        tuple(reference.id for reference in payload.evidence)
    )
    expected_stable_keys = canonical_json_bytes(
        tuple(reference.stable_evidence_key for reference in payload.evidence)
    )
    if (
        event.event_type != payload.event_type
        or event.aggregate_type != "idea"
        or event.aggregate_id != payload.idea_id
        or event.actor_agent_id != payload.owner_agent_id
        or event.occurred_at != payload.last_signal_at_after
        or payload.idea_id != idea.idea_id
        or payload.run_id != idea.run_id
        or payload.run_id != run.run_id
        or payload.owner_agent_id != run.owner_agent_id
        or payload.owner_decision_before is not decision
        or payload.owner_decision_after is not IdeaOwnerDecision.ADOPTED
        or payload.mechanism_cluster_id != generated.cluster_id
        or payload.snapshot_hash != generated.snapshot_hash
        or payload.evidence != generated.evidence
        or occurrence.idea_id != payload.idea_id
        or occurrence.run_id != payload.run_id
        or occurrence.snapshot_hash != payload.snapshot_hash
        or record.owner_agent_id != payload.owner_agent_id
        or record.idea_id != payload.idea_id
        or record.run_id != payload.run_id
        or record.snapshot_hash != payload.snapshot_hash
        or record.evidence_snapshot_item_ids != expected_item_ids
        or record.evidence_stable_keys != expected_stable_keys
        or record.resulting_experience_id != payload.resulting_experience_id
        or record.resulting_version_id != payload.resulting_version_id
        or record.adopted_at != event.occurred_at
        or identity.owner_agent_id != payload.owner_agent_id
        or version.experience_id != identity.experience_id
    ):
        raise _fail("idea adoption event does not match its source anchors")
    try:
        expected_content = adopted_hypothesis_content(
            idea=idea,
            evidence=payload.evidence,
        )
        resulting_content = decode_and_verify_version(
            identity=identity,
            version=version,
            payload=body_payload,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise _fail(
            "idea adoption result cannot be verified as its mapped hypothesis"
        ) from error
    historical_state, historical_event = await _experience_state_before_event(
        session,
        experience_id=identity.experience_id,
        event_id=event.event_id,
        registry=registry,
    )
    if historical_state.temperature is Temperature.ARCHIVED:
        raise _fail("idea adoption cannot reuse an archived hypothesis")
    if (
        identity.kind is not ExperienceKind.HYPOTHESIS
        or (payload.created and identity.origin is not ExperienceOrigin.ADOPTED_IDEA)
        or resulting_content != expected_content
        or historical_state.experience_id != identity.experience_id
        or historical_state.owner_agent_id != identity.owner_agent_id
        or historical_state.current_version_id != version.version_id
        or historical_state.current_content_hash != version.content_hash
        or historical_event.occurred_at > event.occurred_at
        or version.created_at > event.occurred_at
    ):
        raise _fail("idea adoption result is not the exact mapped hypothesis")
    for reference in payload.evidence:
        item = await session.get(
            InspirationSnapshotItemRow,
            reference.id,
        )
        if (
            item is None
            or item.run_id != payload.run_id
            or item.stable_evidence_key != reference.stable_evidence_key
        ):
            raise _fail("idea adoption evidence does not match its frozen snapshot")

    experience_event_types = {
        ExperienceCreatedV1.event_type,
        ExperienceVersionCreatedV1.event_type,
    }
    rows = tuple(
        (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.causation_id == event.causation_id,
                    DomainEventRow.event_id < event.event_id,
                    DomainEventRow.event_type.in_(experience_event_types),
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    )
    experience_events = tuple(_stored_event(registry, row) for row in rows)
    if payload.created:
        if (
            len(experience_events) != 2
            or not isinstance(
                experience_events[0].payload,
                ExperienceCreatedV1,
            )
            or not isinstance(
                experience_events[1].payload,
                ExperienceVersionCreatedV1,
            )
        ):
            raise _fail("created adoption lacks its experience creation events")
        created = experience_events[0].payload
        version_created = experience_events[1].payload
        if (
            created.experience_id != payload.resulting_experience_id
            or created.version_id != payload.resulting_version_id
            or version_created.experience_id != payload.resulting_experience_id
            or version_created.version_id != payload.resulting_version_id
            or any(
                prior.actor_agent_id != payload.owner_agent_id
                or prior.occurred_at != event.occurred_at
                for prior in experience_events
            )
        ):
            raise _fail("created adoption experience events are inconsistent")
    elif experience_events:
        raise _fail("reused adoption unexpectedly creates experience events")
    return idea, run, history, decision, latest


async def _adopter_owners_before_event(
    session: AsyncSession,
    *,
    event: StoredEvent,
    cluster_id: str,
    registry: EventRegistry,
) -> frozenset[UUID]:
    rows = tuple(
        (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.event_type.in_(_ADOPTION_EVENT_TYPES),
                    DomainEventRow.event_id < event.event_id,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    )
    owners: set[UUID] = set()
    for row in rows:
        prior = _stored_event(registry, row)
        payload = prior.payload
        if not isinstance(payload, _ADOPTION_PAYLOAD_TYPES):
            raise _fail("prior idea adoption payload is invalid")
        if payload.mechanism_cluster_id != cluster_id:
            continue
        if (
            prior.aggregate_type != "idea"
            or prior.aggregate_id != payload.idea_id
            or prior.event_type != payload.event_type
            or prior.actor_agent_id != payload.owner_agent_id
            or prior.occurred_at != payload.last_signal_at_after
        ):
            raise _fail("prior idea adoption source is inconsistent")
        owners.add(payload.owner_agent_id)
    return frozenset(owners)


async def _validate_generated_source(
    session: AsyncSession,
    *,
    event: StoredEvent,
    payload: InspirationIdeaGeneratedV1,
    target_prefix: str | None,
) -> tuple[InspirationIdeaRow, IdeaOccurrenceRow, InspirationRunRow]:
    _require_idea_anchor(event, payload.idea_id)
    idea = await session.get(InspirationIdeaRow, payload.idea_id)
    occurrence = await session.get(IdeaOccurrenceRow, payload.occurrence_id)
    run = await session.get(InspirationRunRow, payload.run_id)
    run_state_target = _target(target_prefix, "inspiration_run_state")
    run_state = await _one_mapping(
        session,
        f"SELECT snapshot_hash FROM {run_state_target} WHERE run_id=:run_id",
        {"run_id": str(payload.run_id)},
    )
    started = await session.scalar(
        select(DomainEventRow).where(
            DomainEventRow.aggregate_type == "inspiration_run",
            DomainEventRow.aggregate_id == payload.run_id,
            DomainEventRow.sequence == 1,
            DomainEventRow.event_type == InspirationStartedV1.event_type,
        )
    )
    if (
        idea is None
        or occurrence is None
        or run is None
        or run_state is None
        or started is None
    ):
        raise _fail("generated idea source anchor is missing")
    try:
        raw_evidence = json.loads(idea.evidence_references)
        if canonical_json_bytes(raw_evidence) != idea.evidence_references:
            raise ValueError
        evidence = tuple(
            SnapshotEvidenceReference.model_validate_json(canonical_json_bytes(value))
            for value in raw_evidence
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _fail("generated idea evidence source is invalid") from error
    if (
        event.event_type != InspirationIdeaGeneratedV1.event_type
        or event.actor_agent_id != payload.owner_agent_id
        or run.owner_agent_id != payload.owner_agent_id
        or idea.run_id != payload.run_id
        or idea.operator != payload.operator.value
        or idea.ordinal != payload.ordinal
        or idea.idea_content_hash != payload.idea_content_hash
        or idea.mechanism_hash != payload.mechanism_hash
        or idea.duplicate_relation != payload.duplicate_relation
        or evidence != payload.evidence
        or occurrence.idea_id != payload.idea_id
        or occurrence.run_id != payload.run_id
        or occurrence.owner_agent_id != payload.owner_agent_id
        or occurrence.mechanism_hash != payload.mechanism_hash
        or occurrence.snapshot_hash != payload.snapshot_hash
        or occurrence.occurred_at != event.occurred_at
        or run_state["snapshot_hash"] != payload.snapshot_hash
        or event.occurred_at != run.created_at
        or event.causation_id != started.causation_id
        or payload.owner_decision_after is not IdeaOwnerDecision.ACTIVE
    ):
        raise _fail("generated idea event does not match its source anchors")
    for reference in evidence:
        item = await session.get(
            InspirationSnapshotItemRow,
            reference.id,
        )
        if (
            item is None
            or item.run_id != payload.run_id
            or item.stable_evidence_key != reference.stable_evidence_key
        ):
            raise _fail("generated idea evidence does not resolve in its run snapshot")
    return idea, occurrence, run


def _evaluation_document(
    payload: InspirationIdeaEvaluatedV1,
) -> dict[str, Any]:
    return {
        "evaluated_at": payload.last_signal_at_after,
        "evaluator_agent_id": payload.evaluator_agent_id,
        "evidence": tuple(
            reference.model_dump(mode="json", warnings=False)
            for reference in payload.evidence
        ),
        "reason": (
            None
            if payload.reason is None
            else payload.reason.model_dump(mode="json", warnings=False)
        ),
        "revision": payload.revision,
        "verdict": payload.current_verdict,
    }


def _latest_evaluation_bytes(
    latest: Mapping[UUID, InspirationIdeaEvaluatedV1],
) -> bytes:
    return canonical_json_bytes(
        tuple(
            _evaluation_document(payload)
            for evaluator_id, payload in sorted(
                latest.items(),
                key=lambda item: item[0].bytes,
            )
            if evaluator_id == payload.evaluator_agent_id
        )
    )


def _historical_decision_reason(
    history: tuple[StoredEvent, ...],
) -> bytes | None:
    return next(
        (
            canonical_json_bytes(prior.payload.reason)
            for prior in reversed(history)
            if isinstance(
                prior.payload,
                (InspirationIdeaArchivedV1, InspirationIdeaRejectedV1),
            )
        ),
        None,
    )


def _projected_decision_reason(value: Any) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise _fail("projected idea decision reason is invalid")
    return bytes(value)


async def _idea_history_before(
    session: AsyncSession,
    *,
    event: StoredEvent,
    registry: EventRegistry,
    target_prefix: str | None,
) -> tuple[
    InspirationIdeaRow,
    InspirationRunRow,
    tuple[StoredEvent, ...],
    IdeaOwnerDecision,
    dict[UUID, InspirationIdeaEvaluatedV1],
]:
    if event.event_id < 1 or event.aggregate_type != "idea" or event.sequence < 2:
        raise _fail("idea event has an invalid aggregate anchor")
    rows = tuple(
        (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "idea",
                    DomainEventRow.aggregate_id == event.aggregate_id,
                    DomainEventRow.event_id < event.event_id,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    )
    history = tuple(_stored_event(registry, row) for row in rows)
    if (
        len(history) != event.sequence - 1
        or tuple(item.sequence for item in history) != tuple(range(1, event.sequence))
        or not history
        or not isinstance(history[0].payload, InspirationIdeaGeneratedV1)
    ):
        raise _fail("idea event predecessor sequence is inconsistent")
    generated = history[0]
    generated_payload = cast(InspirationIdeaGeneratedV1, generated.payload)
    if (
        generated.aggregate_id != event.aggregate_id
        or generated_payload.idea_id != event.aggregate_id
    ):
        raise _fail("idea history does not start from its generated event")
    idea, _, run = await _validate_generated_source(
        session,
        event=generated,
        payload=generated_payload,
        target_prefix=target_prefix,
    )

    decision = IdeaOwnerDecision.ACTIVE
    latest: dict[UUID, InspirationIdeaEvaluatedV1] = {}
    prior_time = generated.occurred_at
    for prior in history[1:]:
        payload = prior.payload
        if (
            getattr(payload, "idea_id", None) != idea.idea_id
            or prior.aggregate_type != "idea"
            or prior.aggregate_id != idea.idea_id
            or prior.occurred_at < prior_time
        ):
            raise _fail("idea history has invalid identity or causal order")
        before = getattr(payload, "owner_decision_before", None)
        after = getattr(payload, "owner_decision_after", None)
        if before is not decision or not isinstance(after, IdeaOwnerDecision):
            raise _fail("idea history has a discontinuous owner decision")
        if isinstance(payload, InspirationIdeaEvaluatedV1):
            previous = latest.get(payload.evaluator_agent_id)
            expected_revision = 1 if previous is None else previous.revision + 1
            expected_verdict = None if previous is None else previous.current_verdict
            if (
                payload.revision != expected_revision
                or payload.previous_verdict is not expected_verdict
                or payload.evaluator_agent_id != run.owner_agent_id
                or prior.actor_agent_id != payload.evaluator_agent_id
                or prior.occurred_at != payload.last_signal_at_after
            ):
                raise _fail("idea evaluation history is inconsistent")
            latest[payload.evaluator_agent_id] = payload
        elif isinstance(payload, InspirationIdeaArchivedV1):
            automatic = payload.cycle_id is not None
            if (
                payload.owner_agent_id != run.owner_agent_id
                or (automatic and prior.actor_agent_id is not None)
                or (not automatic and prior.actor_agent_id != run.owner_agent_id)
            ):
                raise _fail("idea archive history has invalid ownership")
        elif isinstance(payload, InspirationIdeaRejectedV1):
            if (
                payload.owner_agent_id != run.owner_agent_id
                or prior.actor_agent_id != run.owner_agent_id
            ):
                raise _fail("idea rejection history has invalid ownership")
        elif isinstance(payload, _ADOPTION_PAYLOAD_TYPES):
            if (
                payload.owner_agent_id != run.owner_agent_id
                or prior.actor_agent_id != run.owner_agent_id
                or payload.run_id != run.run_id
            ):
                raise _fail("idea adoption history has invalid ownership")
        else:
            raise _fail("idea history has an unsupported event")
        decision = after
        prior_time = prior.occurred_at

    if event.occurred_at < prior_time:
        raise _fail("idea event causal time moves backward")
    return idea, run, history, decision, latest


async def _validate_evaluation_evidence(
    session: AsyncSession,
    *,
    payload: InspirationIdeaEvaluatedV1,
    idea: InspirationIdeaRow,
    run: InspirationRunRow,
    evaluated_at: datetime,
) -> None:
    for reference in payload.evidence:
        if isinstance(reference, SnapshotEvidenceReference):
            item = await session.get(
                InspirationSnapshotItemRow,
                reference.id,
            )
            if (
                item is None
                or item.run_id != idea.run_id
                or item.stable_evidence_key != reference.stable_evidence_key
            ):
                raise _fail("evaluation snapshot evidence does not match its source")
            continue
        if not isinstance(reference, ExperienceVersionEvidenceReference):
            raise _fail("evaluation evidence has an unsupported type")
        version = await session.get(ExperienceVersionRow, reference.id)
        experience = (
            None
            if version is None
            else await session.get(ExperienceRow, version.experience_id)
        )
        if (
            version is None
            or experience is None
            or experience.owner_agent_id != run.owner_agent_id
            or experience.created_at > evaluated_at
            or version.created_at > evaluated_at
        ):
            raise _fail(
                "evaluation experience evidence was not an owned version "
                "available at evaluation time"
            )


async def _validate_evaluation_source(
    session: AsyncSession,
    *,
    event: StoredEvent,
    payload: InspirationIdeaEvaluatedV1,
    registry: EventRegistry,
    target_prefix: str | None,
) -> tuple[
    InspirationIdeaRow,
    InspirationRunRow,
    dict[UUID, InspirationIdeaEvaluatedV1],
    IdeaOwnerDecision,
    int,
    bytes | None,
]:
    idea, run, history, historical_decision, latest = await _idea_history_before(
        session,
        event=event,
        registry=registry,
        target_prefix=target_prefix,
    )
    previous = latest.get(payload.evaluator_agent_id)
    expected_revision = 1 if previous is None else previous.revision + 1
    expected_verdict = None if previous is None else previous.current_verdict
    decision = historical_decision
    expected_reason = _historical_decision_reason(history)
    if target_prefix is None:
        state = await session.get(IdeaStateRow, payload.idea_id)
        if (
            state is None
            or state.owner_agent_id != run.owner_agent_id
            or state.mechanism_cluster_id != payload.mechanism_cluster_id
            or bytes(state.evaluations) != _latest_evaluation_bytes(latest)
            or _projected_decision_reason(state.decision_reason) != expected_reason
            or state.projection_event_id != history[-1].event_id
        ):
            raise _fail("online idea evaluation predecessor is inconsistent")
        try:
            decision = IdeaOwnerDecision(state.owner_decision)
        except ValueError as error:
            raise _fail("online idea owner decision is invalid") from error
    if (
        event.event_type != InspirationIdeaEvaluatedV1.event_type
        or event.aggregate_id != payload.idea_id
        or payload.evaluator_agent_id != run.owner_agent_id
        or event.actor_agent_id != payload.evaluator_agent_id
        or event.occurred_at != payload.last_signal_at_after
        or payload.revision != expected_revision
        or payload.previous_verdict is not expected_verdict
        or payload.owner_decision_before is not decision
        or payload.owner_decision_after is not decision
    ):
        raise _fail("evaluation event does not match its source history")
    await _validate_evaluation_evidence(
        session,
        payload=payload,
        idea=idea,
        run=run,
        evaluated_at=event.occurred_at,
    )
    return (
        idea,
        run,
        latest,
        decision,
        history[-1].event_id,
        expected_reason,
    )


async def _clusters_before_event(
    session: AsyncSession,
    *,
    target: str,
    target_prefix: str | None,
    event_id: int,
    registry: EventRegistry,
) -> tuple[IncubationCluster, ...]:
    projection_rows = tuple(
        (
            await session.execute(text(f"SELECT * FROM {target} ORDER BY cluster_id"))
        ).mappings()
    )
    states = tuple(
        _mechanism_state(cast(Mapping[str, Any], row)) for row in projection_rows
    )
    hash_to_cluster: dict[str, int] = {}
    for index, state in enumerate(states):
        for mechanism_hash in state.member_hashes:
            if mechanism_hash in hash_to_cluster:
                raise _fail(
                    "a mechanism member belongs to more than one rebuilt cluster"
                )
            hash_to_cluster[mechanism_hash] = index

    members: list[list[IncubationMember]] = [[] for _ in states]
    snapshots: list[set[str]] = [set() for _ in states]
    rows = tuple(
        (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.event_type == InspirationIdeaGeneratedV1.event_type,
                    DomainEventRow.event_id < event_id,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    )
    for row in rows:
        prior = _stored_event(registry, row)
        if not isinstance(prior.payload, InspirationIdeaGeneratedV1):
            raise _fail("generated idea history has an invalid payload")
        idea, occurrence, run = await _validate_generated_source(
            session,
            event=prior,
            payload=prior.payload,
            target_prefix=target_prefix,
        )
        cluster_index = hash_to_cluster.get(idea.mechanism_hash)
        if cluster_index is None:
            raise _fail("a prior generated mechanism is absent from rebuilt clusters")
        members[cluster_index].append(
            IncubationMember(
                idea_id=idea.idea_id,
                owner_agent_id=run.owner_agent_id,
                mechanism=idea.mechanism,
                mechanism_hash=idea.mechanism_hash,
            )
        )
        snapshots[cluster_index].add(occurrence.snapshot_hash)

    try:
        return tuple(
            IncubationCluster(
                state=state,
                members=tuple(members[index]),
                snapshot_hashes=frozenset(snapshots[index]),
            )
            for index, state in enumerate(states)
        )
    except (TypeError, ValueError) as error:
        raise _fail("rebuilt cluster sources disagree with projection state") from error


class MechanismIncubationProjector:
    """Replay generated assignments and effective evaluation signals."""

    name = "mechanism_incubation"
    version = 1
    event_types = frozenset(
        {
            InspirationIdeaGeneratedV1.event_type,
            InspirationIdeaEvaluatedV1.event_type,
            *_ADOPTION_EVENT_TYPES,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "cluster_id VARCHAR(64) NOT NULL PRIMARY KEY,"
                "canonical_mechanism_hash VARCHAR(64) NOT NULL,"
                "member_hashes BLOB NOT NULL,"
                "occurrence_count INTEGER NOT NULL,"
                "distinct_snapshot_count INTEGER NOT NULL,"
                "distinct_adopter_count INTEGER NOT NULL,"
                "supported_count INTEGER NOT NULL,"
                "refuted_count INTEGER NOT NULL,"
                "maturity VARCHAR(11) NOT NULL,"
                "candidate_since VARCHAR(27),"
                "last_signal_at VARCHAR(27) NOT NULL,"
                "projection_event_id INTEGER NOT NULL)"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._registry, row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        if isinstance(event.payload, InspirationIdeaGeneratedV1):
            await self._apply_generated(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, InspirationIdeaEvaluatedV1):
            await self._apply_evaluated(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, _ADOPTION_PAYLOAD_TYPES):
            await self._apply_adopted(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        raise _fail("unsupported mechanism event payload")

    async def _apply_generated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaGeneratedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        idea, occurrence, run = await _validate_generated_source(
            session,
            event=event,
            payload=payload,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        transition = _transition(payload)
        if target_prefix is not None:
            clusters = await _clusters_before_event(
                session,
                target=target,
                target_prefix=target_prefix,
                event_id=event.event_id,
                registry=self._registry,
            )
            try:
                plan = plan_occurrence(
                    owner_agent_id=run.owner_agent_id,
                    mechanism=idea.mechanism,
                    snapshot_hash=occurrence.snapshot_hash,
                    run_occurred_at=event.occurred_at,
                    clusters=clusters,
                )
            except (TypeError, ValueError) as error:
                raise _fail("generated idea assignment cannot be rebuilt") from error
            if (
                payload.mechanism_hash != plan.mechanism_hash
                or payload.duplicate_relation != plan.duplicate_relation
                or transition != plan.transition
            ):
                raise _fail(
                    "generated idea declares an invalid rebuilt cluster assignment"
                )
        current = await _one_mapping(
            session,
            f"SELECT * FROM {target} WHERE cluster_id=:cluster_id",
            {"cluster_id": transition.cluster_id},
        )
        if current is None:
            if transition.occurrence_count_before != 0:
                raise _fail("generated cluster before-state is missing")
            await session.execute(
                text(
                    f"INSERT INTO {target} (cluster_id,"
                    "canonical_mechanism_hash,member_hashes,occurrence_count,"
                    "distinct_snapshot_count,distinct_adopter_count,"
                    "supported_count,refuted_count,maturity,candidate_since,"
                    "last_signal_at,projection_event_id) VALUES "
                    "(:cluster_id,:canonical,:members,:occurrences,:snapshots,"
                    ":adopters,:supported,:refuted,:maturity,:candidate_since,"
                    ":last_signal,:event_id)"
                ),
                _cluster_after_parameters(transition, event.event_id),
            )
            return
        if not _cluster_before_matches(current, transition):
            raise _fail("generated cluster before-state does not match projection")
        result = await session.execute(
            text(
                f"UPDATE {target} SET member_hashes=:members,"
                "occurrence_count=:occurrences,"
                "distinct_snapshot_count=:snapshots,"
                "distinct_adopter_count=:adopters,supported_count=:supported,"
                "refuted_count=:refuted,maturity=:maturity,"
                "candidate_since=:candidate_since,last_signal_at=:last_signal,"
                "projection_event_id=:event_id WHERE cluster_id=:cluster_id"
            ),
            _cluster_after_parameters(transition, event.event_id),
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("generated cluster update did not affect one row")

    async def _apply_evaluated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaEvaluatedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        idea, _, _, _, _, _ = await _validate_evaluation_source(
            session,
            event=event,
            payload=payload,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await _one_mapping(
            session,
            f"SELECT * FROM {target} WHERE cluster_id=:cluster_id",
            {"cluster_id": payload.mechanism_cluster_id},
        )
        if current is None:
            raise _fail("evaluation mechanism cluster is missing")
        state = _mechanism_state(current)
        if (
            idea.mechanism_hash not in state.member_hashes
            or current["projection_event_id"] >= event.event_id
        ):
            raise _fail("evaluation mechanism membership is inconsistent")
        try:
            transition = plan_evaluation_transition(
                cluster=state,
                previous_verdict=payload.previous_verdict,
                current_verdict=payload.current_verdict,
                evaluated_at=event.occurred_at,
            )
        except (TypeError, ValueError) as error:
            raise _fail("evaluation transition cannot be rebuilt") from error
        if not _evaluation_transition_matches(payload, transition):
            raise _fail("evaluation transition does not match locked mechanism state")
        result = await session.execute(
            text(
                f"UPDATE {target} SET supported_count=:supported,"
                "refuted_count=:refuted,maturity=:maturity,"
                "candidate_since=:candidate_since,last_signal_at=:last_signal,"
                "projection_event_id=:event_id WHERE cluster_id=:cluster_id"
            ),
            {
                "supported": transition.supported_count_after,
                "refuted": transition.refuted_count_after,
                "maturity": transition.maturity_after.value,
                "candidate_since": _timestamp(transition.candidate_since_after),
                "last_signal": _timestamp(transition.last_signal_at_after),
                "event_id": event.event_id,
                "cluster_id": state.cluster_id,
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("evaluation mechanism update did not affect one row")

    async def _apply_adopted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: _AdoptionPayload,
        *,
        target_prefix: str | None,
    ) -> None:
        idea, _, _, _, _ = await _validate_adoption_source(
            session,
            event=event,
            payload=payload,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await _one_mapping(
            session,
            f"SELECT * FROM {target} WHERE cluster_id=:cluster_id",
            {"cluster_id": payload.mechanism_cluster_id},
        )
        if current is None:
            raise _fail("adoption mechanism cluster is missing")
        state = _mechanism_state(current)
        owners = await _adopter_owners_before_event(
            session,
            event=event,
            cluster_id=state.cluster_id,
            registry=self._registry,
        )
        if (
            idea.mechanism_hash not in state.member_hashes
            or current["projection_event_id"] >= event.event_id
            or state.distinct_adopter_count != len(owners)
        ):
            raise _fail("adoption mechanism membership is inconsistent")
        try:
            transition = plan_adoption_transition(
                cluster=state,
                owner_already_adopted=(payload.owner_agent_id in owners),
                adopted_at=event.occurred_at,
            )
        except (TypeError, ValueError) as error:
            raise _fail("adoption transition cannot be rebuilt") from error
        if not _adoption_transition_matches(payload, transition):
            raise _fail("adoption transition does not match locked mechanism state")
        result = await session.execute(
            text(
                f"UPDATE {target} SET distinct_adopter_count=:adopters,"
                "maturity=:maturity,candidate_since=:candidate_since,"
                "last_signal_at=:last_signal,projection_event_id=:event_id "
                "WHERE cluster_id=:cluster_id"
            ),
            {
                "adopters": transition.distinct_adopter_count_after,
                "maturity": transition.maturity_after.value,
                "candidate_since": _timestamp(transition.candidate_since_after),
                "last_signal": _timestamp(transition.last_signal_at_after),
                "event_id": event.event_id,
                "cluster_id": state.cluster_id,
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("adoption mechanism update did not affect one row")


def _cluster_before_matches(
    current: Mapping[str, Any],
    transition: ClusterTransition,
) -> bool:
    try:
        encoded_members = bytes(current["member_hashes"])
        members = json.loads(encoded_members)
        if canonical_json_bytes(members) != encoded_members:
            return False
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        current["canonical_mechanism_hash"] == transition.canonical_mechanism_hash
        and tuple(members) == transition.member_hashes_before
        and current["occurrence_count"] == transition.occurrence_count_before
        and current["distinct_snapshot_count"]
        == transition.distinct_snapshot_count_before
        and current["distinct_adopter_count"]
        == transition.distinct_adopter_count_before
        and current["supported_count"] == transition.supported_count_before
        and current["refuted_count"] == transition.refuted_count_before
        and current["maturity"]
        == cast(MechanismMaturity, transition.maturity_before).value
        and _parsed_timestamp(current["candidate_since"])
        == transition.candidate_since_before
        and _parsed_timestamp(current["last_signal_at"])
        == transition.last_signal_at_before
    )


def _cluster_after_parameters(
    transition: ClusterTransition,
    event_id: int,
) -> dict[str, Any]:
    return {
        "cluster_id": transition.cluster_id,
        "canonical": transition.canonical_mechanism_hash,
        "members": canonical_json_bytes(transition.member_hashes_after),
        "occurrences": transition.occurrence_count_after,
        "snapshots": transition.distinct_snapshot_count_after,
        "adopters": transition.distinct_adopter_count_after,
        "supported": transition.supported_count_after,
        "refuted": transition.refuted_count_after,
        "maturity": transition.maturity_after.value,
        "candidate_since": _timestamp(transition.candidate_since_after),
        "last_signal": _timestamp(transition.last_signal_at_after),
        "event_id": event_id,
    }


class IdeaStateProjector:
    """Replay generation, latest-effective evaluations, and archival."""

    name = "idea_state"
    version = 1
    event_types = frozenset(
        {
            InspirationIdeaGeneratedV1.event_type,
            InspirationIdeaEvaluatedV1.event_type,
            InspirationIdeaArchivedV1.event_type,
            InspirationIdeaRejectedV1.event_type,
            *_ADOPTION_EVENT_TYPES,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "idea_id VARCHAR(36) NOT NULL PRIMARY KEY,"
                "owner_agent_id VARCHAR(36) NOT NULL,"
                "mechanism_cluster_id VARCHAR(64) NOT NULL,"
                "owner_decision VARCHAR(8) NOT NULL,"
                "evaluations BLOB NOT NULL,"
                "decision_reason BLOB,"
                "resulting_experience_id VARCHAR(36),"
                "resulting_version_id VARCHAR(36),"
                "last_signal_at VARCHAR(27) NOT NULL,"
                "projection_event_id INTEGER NOT NULL)"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._registry, row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        if isinstance(event.payload, InspirationIdeaGeneratedV1):
            await self._apply_generated(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, InspirationIdeaEvaluatedV1):
            await self._apply_evaluated(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, InspirationIdeaArchivedV1):
            await self._apply_archived(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, InspirationIdeaRejectedV1):
            await self._apply_rejected(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, _ADOPTION_PAYLOAD_TYPES):
            await self._apply_adopted(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        raise _fail("unsupported idea-state event payload")

    async def _apply_generated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaGeneratedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        await _validate_generated_source(
            session,
            event=event,
            payload=payload,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        existing = await _one_mapping(
            session,
            f"SELECT idea_id FROM {target} WHERE idea_id=:idea_id",
            {"idea_id": str(payload.idea_id)},
        )
        mechanism_target = _target(target_prefix, "mechanism_incubation")
        cluster = await _one_mapping(
            session,
            f"SELECT cluster_id FROM {mechanism_target} WHERE cluster_id=:cluster_id",
            {"cluster_id": payload.cluster_id},
        )
        if existing is not None or cluster is None:
            raise _fail("generated idea state has invalid cluster or identity")
        await session.execute(
            text(
                f"INSERT INTO {target} (idea_id,owner_agent_id,"
                "mechanism_cluster_id,owner_decision,evaluations,"
                "decision_reason,resulting_experience_id,resulting_version_id,"
                "last_signal_at,projection_event_id) VALUES "
                "(:idea_id,:owner_id,:cluster_id,:decision,:evaluations,"
                "NULL,NULL,NULL,:last_signal,:event_id)"
            ),
            {
                "idea_id": str(payload.idea_id),
                "owner_id": str(payload.owner_agent_id),
                "cluster_id": payload.cluster_id,
                "decision": IdeaOwnerDecision.ACTIVE.value,
                "evaluations": canonical_json_bytes(()),
                "last_signal": _timestamp(payload.last_signal_at_after),
                "event_id": event.event_id,
            },
        )

    async def _current(
        self,
        session: AsyncSession,
        *,
        target: str,
        idea_id: UUID,
    ) -> Mapping[str, Any]:
        current = await _one_mapping(
            session,
            f"SELECT * FROM {target} WHERE idea_id=:idea_id",
            {"idea_id": str(idea_id)},
        )
        if current is None:
            raise _fail("idea state projection is missing")
        return current

    async def _apply_evaluated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaEvaluatedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        (
            idea,
            run,
            latest,
            decision,
            predecessor_event_id,
            expected_reason,
        ) = await _validate_evaluation_source(
            session,
            event=event,
            payload=payload,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await self._current(
            session,
            target=target,
            idea_id=payload.idea_id,
        )
        last_signal_at = _parsed_timestamp(current["last_signal_at"])
        expected_before = _latest_evaluation_bytes(latest)
        mechanism_target = _target(target_prefix, "mechanism_incubation")
        cluster = await _one_mapping(
            session,
            f"SELECT * FROM {mechanism_target} WHERE cluster_id=:cluster_id",
            {"cluster_id": payload.mechanism_cluster_id},
        )
        if cluster is None:
            raise _fail("evaluated idea mechanism cluster is missing")
        mechanism = _mechanism_state(cluster)
        if (
            current["owner_agent_id"] != str(run.owner_agent_id)
            or current["mechanism_cluster_id"] != payload.mechanism_cluster_id
            or current["owner_decision"] != decision.value
            or bytes(current["evaluations"]) != expected_before
            or _projected_decision_reason(current["decision_reason"]) != expected_reason
            or last_signal_at is None
            or event.occurred_at < last_signal_at
            or current["projection_event_id"] != predecessor_event_id
            or idea.mechanism_hash not in mechanism.member_hashes
        ):
            raise _fail("evaluation does not match locked owner-local idea state")
        revised = dict(latest)
        revised[payload.evaluator_agent_id] = payload
        result = await session.execute(
            text(
                f"UPDATE {target} SET evaluations=:evaluations,"
                "last_signal_at=:last_signal,projection_event_id=:event_id "
                "WHERE idea_id=:idea_id"
            ),
            {
                "evaluations": _latest_evaluation_bytes(revised),
                "last_signal": _timestamp(event.occurred_at),
                "event_id": event.event_id,
                "idea_id": str(payload.idea_id),
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("evaluation idea update did not affect one row")

    async def _apply_archived(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaArchivedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        idea, run, history, historical_decision, latest = await _idea_history_before(
            session,
            event=event,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await self._current(
            session,
            target=target,
            idea_id=payload.idea_id,
        )
        last_signal_at = _parsed_timestamp(current["last_signal_at"])
        automatic = payload.cycle_id is not None
        try:
            decision = IdeaOwnerDecision(current["owner_decision"])
        except ValueError as error:
            raise _fail("archived idea owner decision is invalid") from error
        if automatic:
            await _validate_policy_archive_source(
                session,
                event=event,
                payload=payload,
                current=current,
                target_prefix=target_prefix,
            )
        if (
            event.event_type != InspirationIdeaArchivedV1.event_type
            or event.aggregate_id != payload.idea_id
            or payload.owner_agent_id != run.owner_agent_id
            or (automatic and event.actor_agent_id is not None)
            or (not automatic and event.actor_agent_id != run.owner_agent_id)
            or decision is not historical_decision
            or payload.owner_decision_before is not decision
            or payload.owner_decision_after is not IdeaOwnerDecision.ARCHIVED
            or current["owner_agent_id"] != str(run.owner_agent_id)
            or bytes(current["evaluations"]) != _latest_evaluation_bytes(latest)
            or current["decision_reason"] is not None
            or current["resulting_experience_id"] is not None
            or current["resulting_version_id"] is not None
            or last_signal_at is None
            or event.occurred_at < last_signal_at
            or current["projection_event_id"] != history[-1].event_id
            or idea.idea_id != payload.idea_id
        ):
            raise _fail("archive event does not match locked idea state")
        result = await session.execute(
            text(
                f"UPDATE {target} SET owner_decision=:decision,"
                "decision_reason=:reason,projection_event_id=:event_id "
                "WHERE idea_id=:idea_id"
            ),
            {
                "decision": IdeaOwnerDecision.ARCHIVED.value,
                "reason": canonical_json_bytes(payload.reason),
                "event_id": event.event_id,
                "idea_id": str(payload.idea_id),
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("archive idea update did not affect one row")

    async def _apply_rejected(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: InspirationIdeaRejectedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        idea, run, history, historical_decision, latest = await _idea_history_before(
            session,
            event=event,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await self._current(
            session,
            target=target,
            idea_id=payload.idea_id,
        )
        last_signal_at = _parsed_timestamp(current["last_signal_at"])
        try:
            decision = IdeaOwnerDecision(current["owner_decision"])
        except ValueError as error:
            raise _fail("rejected idea owner decision is invalid") from error
        expected_reason = next(
            (
                canonical_json_bytes(prior.payload.reason)
                for prior in reversed(history)
                if isinstance(
                    prior.payload,
                    (InspirationIdeaArchivedV1, InspirationIdeaRejectedV1),
                )
            ),
            None,
        )
        retained_reason = current["decision_reason"]
        if retained_reason is not None:
            retained_reason = bytes(retained_reason)
        if (
            event.event_type != InspirationIdeaRejectedV1.event_type
            or event.aggregate_id != payload.idea_id
            or payload.owner_agent_id != run.owner_agent_id
            or event.actor_agent_id != run.owner_agent_id
            or decision is not historical_decision
            or payload.owner_decision_before is not decision
            or payload.owner_decision_after is not IdeaOwnerDecision.REJECTED
            or current["owner_agent_id"] != str(run.owner_agent_id)
            or bytes(current["evaluations"]) != _latest_evaluation_bytes(latest)
            or retained_reason != expected_reason
            or current["resulting_experience_id"] is not None
            or current["resulting_version_id"] is not None
            or last_signal_at is None
            or event.occurred_at < last_signal_at
            or current["projection_event_id"] != history[-1].event_id
            or idea.idea_id != payload.idea_id
        ):
            raise _fail("rejection event does not match locked idea state")
        result = await session.execute(
            text(
                f"UPDATE {target} SET owner_decision=:decision,"
                "decision_reason=:reason,projection_event_id=:event_id "
                "WHERE idea_id=:idea_id"
            ),
            {
                "decision": IdeaOwnerDecision.REJECTED.value,
                "reason": canonical_json_bytes(payload.reason),
                "event_id": event.event_id,
                "idea_id": str(payload.idea_id),
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("rejection idea update did not affect one row")

    async def _apply_adopted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: _AdoptionPayload,
        *,
        target_prefix: str | None,
    ) -> None:
        (
            idea,
            run,
            history,
            historical_decision,
            latest,
        ) = await _validate_adoption_source(
            session,
            event=event,
            payload=payload,
            registry=self._registry,
            target_prefix=target_prefix,
        )
        target = _target(target_prefix, self.name)
        current = await self._current(
            session,
            target=target,
            idea_id=payload.idea_id,
        )
        last_signal_at = _parsed_timestamp(current["last_signal_at"])
        try:
            decision = IdeaOwnerDecision(current["owner_decision"])
        except ValueError as error:
            raise _fail("adopted idea owner decision is invalid") from error
        expected_reason = next(
            (
                canonical_json_bytes(prior.payload.reason)
                for prior in reversed(history)
                if isinstance(
                    prior.payload,
                    (InspirationIdeaArchivedV1, InspirationIdeaRejectedV1),
                )
            ),
            None,
        )
        retained_reason = current["decision_reason"]
        if retained_reason is not None:
            retained_reason = bytes(retained_reason)
        if (
            event.event_type != payload.event_type
            or event.aggregate_id != payload.idea_id
            or payload.owner_agent_id != run.owner_agent_id
            or event.actor_agent_id != run.owner_agent_id
            or decision is not historical_decision
            or payload.owner_decision_before is not decision
            or payload.owner_decision_after is not IdeaOwnerDecision.ADOPTED
            or current["owner_agent_id"] != str(run.owner_agent_id)
            or current["mechanism_cluster_id"] != payload.mechanism_cluster_id
            or bytes(current["evaluations"]) != _latest_evaluation_bytes(latest)
            or retained_reason != expected_reason
            or current["resulting_experience_id"] is not None
            or current["resulting_version_id"] is not None
            or last_signal_at is None
            or event.occurred_at < last_signal_at
            or current["projection_event_id"] != history[-1].event_id
            or idea.idea_id != payload.idea_id
        ):
            raise _fail("adoption event does not match locked idea state")
        result = await session.execute(
            text(
                f"UPDATE {target} SET owner_decision=:decision,"
                "decision_reason=NULL,resulting_experience_id=:experience_id,"
                "resulting_version_id=:version_id,last_signal_at=:last_signal,"
                "projection_event_id=:event_id WHERE idea_id=:idea_id"
            ),
            {
                "decision": IdeaOwnerDecision.ADOPTED.value,
                "experience_id": str(payload.resulting_experience_id),
                "version_id": str(payload.resulting_version_id),
                "last_signal": _timestamp(event.occurred_at),
                "event_id": event.event_id,
                "idea_id": str(payload.idea_id),
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("adoption idea update did not affect one row")


__all__ = [
    "IdeaStateProjector",
    "InspirationProjectionIntegrityError",
    "InspirationRunProjector",
    "MechanismIncubationProjector",
]
