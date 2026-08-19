from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, text

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, StructuredReason
from experience_hub.experiences.candidate_events import (
    CandidateAdoptedV1,
    CandidateRejectedV1,
)
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
    AdoptCandidate,
    CandidateDecision,
    RejectCandidate,
)
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.models import (
    ExperienceOrigin,
    Temperature,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval import RetrievalMode, SearchExperiences
from experience_hub.storage import CommandResult, StoredResponse, UnitOfWork
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import (
    CandidateAdoptionRow,
    CandidateStateRow,
    DomainEventRow,
    ExperienceCandidateRow,
    ExperienceRow,
    ExperienceStateRow,
    IdempotencyRecordRow,
    TrajectoryBundleRow,
)
from experience_hub.storage.validation import SourceIntegrityError

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000601")
OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000602")
IDS = (
    UUID("00000000-0000-0000-0000-000000000603"),
    OWNER_ID,
    UUID("00000000-0000-0000-0000-000000000604"),
    OTHER_OWNER_ID,
    *(
        UUID(f"00000000-0000-0000-0000-{value:012d}")
        for value in range(605, 700)
    ),
)


class FailAt:
    def __init__(self) -> None:
        self.checkpoint: FaultCheckpoint | None = None
        self.calls: list[FaultCheckpoint] = []

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        self.calls.append(checkpoint)
        if checkpoint is self.checkpoint:
            raise RuntimeError(f"injected:{checkpoint.value}")

    def reset_calls(self) -> None:
        self.calls.clear()


@dataclass(slots=True)
class DecisionStack:
    container: ApplicationContainer
    fault: FailAt

    @property
    def clock(self) -> FrozenClock:
        assert isinstance(self.container.clock, FrozenClock)
        return self.container.clock


async def _create_agent(
    container: ApplicationContainer,
    *,
    key: str,
    name: str,
) -> None:
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        body={"name": name},
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=name),
            command_context=command,
        )

    result = await container.command_executor.execute(request, handler)
    assert result.status_code == 201


def _jsonl(
    *,
    owner_agent_id: UUID = OWNER_ID,
    label: str = "cache",
) -> bytes:
    timestamp = NOW.isoformat().replace("+00:00", "Z")
    return b"\n".join(
        (
            canonical_json_bytes(
                {
                    "adapter": {"kind": "generic_jsonl", "version": 1},
                    "owner_agent_id": owner_agent_id,
                    "record_type": "header",
                    "sanitization": {
                        "input_sanitized": True,
                        "profile_id": "trusted-v1",
                    },
                    "schema_version": 1,
                    "source_completed_at": timestamp,
                    "source_started_at": timestamp,
                    "trajectory_id": f"decision-trajectory-{label}",
                }
            ),
            canonical_json_bytes(
                {
                    "action": "Invalidated the cache and retried.",
                    "candidate_signal": {
                        "applicability": ["local caches"],
                        "body": f"Invalidate stale {label} before retrying.",
                        "evidence": [
                            {"field": "observation", "step_id": "step-1"}
                        ],
                        "falsifiers": ["A fresh cache still fails"],
                        "kind": "procedural",
                        "mechanism": f"Invalidation removes stale {label}.",
                        "summary": f"Invalidate stale {label}.",
                        "tags": [label, "retry"],
                    },
                    "observation": "A stale cache caused the command to fail.",
                    "occurred_at": timestamp,
                    "ordinal": 1,
                    "outcome": "The retry succeeded.",
                    "record_type": "step",
                    "status": "succeeded",
                    "step_id": "step-1",
                }
            ),
        )
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[DecisionStack]:
    database_path = tmp_path / "candidate-decisions.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")
    container = ApplicationContainer.build(
        Settings(database_url=f"sqlite+aiosqlite:///{database_path}"),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(IDS),
    )
    fault = FailAt()
    container.database._fault_injector = fault
    await _create_agent(container, key="create-owner", name="Owner")
    await _create_agent(container, key="create-other", name="Other owner")
    value = DecisionStack(container=container, fault=fault)
    try:
        yield value
    finally:
        await container.close()


async def _capture_pending(
    stack: DecisionStack,
    *,
    owner_agent_id: UUID = OWNER_ID,
    key: str = "capture-pending",
    label: str = "cache",
) -> UUID:
    container = stack.container
    prepared = container.capture_preparer.prepare_jsonl(
        _jsonl(owner_agent_id=owner_agent_id, label=label)
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=TRAJECTORY_IMPORT_SCOPE,
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/trajectory-bundles",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "candidate_count": 1,
            "manifest_hash": prepared.bundle.manifest_hash,
        },
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await container.capture_service.capture(
            uow=uow,
            prepared=prepared,
            command=command,
        )

    result = await container.command_executor.execute(request, handler)
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["candidate_ids"][0])


def _decision_request(
    *,
    candidate_id: UUID,
    action: str,
    scope: str,
    key: str,
    body: object,
    caller_agent_id: UUID = OWNER_ID,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{caller_agent_id}",
        operation_scope=scope,
        idempotency_key=key,
        method="POST",
        route_template=(
            "/v1/agents/{agent_id}/experience-candidates/"
            "{candidate_id}/" + action
        ),
        path_parameters={
            "agent_id": caller_agent_id,
            "candidate_id": candidate_id,
        },
        body=body,
    )


async def _adopt(
    stack: DecisionStack,
    *,
    candidate_id: UUID,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
    caller_agent_id: UUID | None = None,
) -> CommandResult:
    request = AdoptCandidate(
        owner_agent_id=owner_agent_id,
        candidate_id=candidate_id,
        importance=0.7,
        confidence=0.8,
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.container.candidate_service.adopt(
            uow=uow,
            request=request,
            command=command,
        )

    return await stack.container.command_executor.execute(
        _decision_request(
            candidate_id=candidate_id,
            action="adopt",
            scope=CANDIDATE_ADOPT_SCOPE,
            key=key,
            body=request,
            caller_agent_id=(
                owner_agent_id
                if caller_agent_id is None
                else caller_agent_id
            ),
        ),
        handler,
    )


async def _reject(
    stack: DecisionStack,
    *,
    candidate_id: UUID,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
    caller_agent_id: UUID | None = None,
    reason_text: str = "Not reusable.",
) -> CommandResult:
    request = RejectCandidate(
        owner_agent_id=owner_agent_id,
        candidate_id=candidate_id,
        reason=StructuredReason.from_user_text(reason_text),
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.container.candidate_service.reject(
            uow=uow,
            request=request,
            command=command,
        )

    return await stack.container.command_executor.execute(
        _decision_request(
            candidate_id=candidate_id,
            action="reject",
            scope=CANDIDATE_REJECT_SCOPE,
            key=key,
            body=request,
            caller_agent_id=(
                owner_agent_id
                if caller_agent_id is None
                else caller_agent_id
            ),
        ),
        handler,
    )


async def _create_equivalent(
    stack: DecisionStack,
    *,
    candidate_id: UUID,
    candidate_owner_id: UUID,
    experience_owner_id: UUID,
    key: str,
) -> tuple[UUID, UUID]:
    container = stack.container
    async with container.database.read_session() as session:
        candidate = await container.candidate_service.get_owned(
            session=session,
            owner_agent_id=candidate_owner_id,
            candidate_id=candidate_id,
        )
    request = CreateExperience(
        owner_agent_id=experience_owner_id,
        kind=candidate.kind,
        content=candidate.content,
        importance=0.4,
        confidence=0.6,
        links=(),
    )
    command_request = CommandRequest(
        caller_scope=f"agent:{experience_owner_id}",
        operation_scope="experience.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": experience_owner_id},
        body={"key": key},
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await container.experience_service.create(
            uow=uow,
            command=request,
            command_context=command,
        )

    result = await container.command_executor.execute(command_request, handler)
    assert result.status_code == 201
    body = json.loads(result.body)["data"]
    return UUID(body["experience_id"]), UUID(body["version_id"])


@pytest.mark.asyncio
async def test_adopt_pending_candidate_creates_experience(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    request = AdoptCandidate(
        owner_agent_id=OWNER_ID,
        candidate_id=candidate_id,
        importance=0.7,
        confidence=0.8,
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.container.candidate_service.adopt(
            uow=uow,
            request=request,
            command=command,
        )

    result = await stack.container.command_executor.execute(
        _decision_request(
            candidate_id=candidate_id,
            action="adopt",
            scope=CANDIDATE_ADOPT_SCOPE,
            key="adopt-create",
            body=request,
        ),
        handler,
    )

    assert result.status_code == 200
    async with stack.container.database.read_session() as session:
        candidate = await stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
        identity = await session.get(
            ExperienceRow,
            candidate.resulting_experience_id,
        )
        state = await session.get(
            ExperienceStateRow,
            candidate.resulting_experience_id,
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == CANDIDATE_ADOPT_SCOPE,
                IdempotencyRecordRow.idempotency_key == "adopt-create",
            )
        )
        assert receipt is not None
        adoption = await session.scalar(
            select(CandidateAdoptionRow).where(
                CandidateAdoptionRow.candidate_id == candidate_id
            )
        )
        event_types = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )

    assert candidate.decision is CandidateDecision.ADOPTED
    assert candidate.content_hash
    assert identity is not None
    assert identity.origin is ExperienceOrigin.ADOPTED_CANDIDATE
    assert state is not None
    assert state.temperature is Temperature.WARM
    assert state.importance == 0.7
    assert state.confidence == 0.8
    assert state.source_trust == 0.5
    assert adoption is not None
    assert adoption.adopted_at == receipt.created_at
    assert adoption.adopted_at == candidate.decided_at
    assert identity.created_at == receipt.created_at
    assert receipt.result_resource_type == "candidate_adoption"
    assert receipt.result_resource_id == adoption.adoption_id
    assert event_types == (
        "experience.created",
        "experience.version_created",
        CandidateAdoptedV1.event_type,
    )


@pytest.mark.asyncio
async def test_reject_pending_candidate_records_one_event(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    async with stack.container.database.read_session() as session:
        source_before = await session.get(ExperienceCandidateRow, candidate_id)
        assert source_before is not None
        source_snapshot = tuple(
            getattr(source_before, column.name)
            for column in ExperienceCandidateRow.__table__.columns
        )
    reason = StructuredReason.from_user_text("Not reusable.")
    request = RejectCandidate(
        owner_agent_id=OWNER_ID,
        candidate_id=candidate_id,
        reason=reason,
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.container.candidate_service.reject(
            uow=uow,
            request=request,
            command=command,
        )

    result = await stack.container.command_executor.execute(
        _decision_request(
            candidate_id=candidate_id,
            action="reject",
            scope=CANDIDATE_REJECT_SCOPE,
            key="reject-once",
            body=request,
        ),
        handler,
    )

    assert result.status_code == 200
    async with stack.container.database.read_session() as session:
        candidate = await stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == CANDIDATE_REJECT_SCOPE,
                IdempotencyRecordRow.idempotency_key == "reject-once",
            )
        )
        assert receipt is not None
        decision_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.causation_id == receipt.receipt_id
            )
        )
        event_types = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )
        experience_count = await session.scalar(select(ExperienceRow))
        source_after = await session.get(ExperienceCandidateRow, candidate_id)
        assert source_after is not None
        retained_snapshot = tuple(
            getattr(source_after, column.name)
            for column in ExperienceCandidateRow.__table__.columns
        )

    assert candidate.decision is CandidateDecision.REJECTED
    assert candidate.reason == reason
    assert candidate.decided_at == receipt.created_at
    assert decision_event is not None
    assert decision_event.occurred_at == receipt.created_at
    assert receipt.result_resource_type == "experience_candidate"
    assert receipt.result_resource_id == candidate_id
    assert event_types == (CandidateRejectedV1.event_type,)
    assert experience_count is None
    assert retained_snapshot == source_snapshot


@pytest.mark.asyncio
async def test_adopt_reuses_owned_current_equivalent_without_creation_events(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    existing_experience_id, existing_version_id = await _create_equivalent(
        stack,
        candidate_id=candidate_id,
        candidate_owner_id=OWNER_ID,
        experience_owner_id=OWNER_ID,
        key="create-owned-equivalent",
    )

    result = await _adopt(stack, candidate_id=candidate_id, key="adopt-reuse")

    assert result.status_code == 200
    async with stack.container.database.read_session() as session:
        candidate = await stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
        adoption = await session.scalar(
            select(CandidateAdoptionRow).where(
                CandidateAdoptionRow.candidate_id == candidate_id
            )
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == CANDIDATE_ADOPT_SCOPE,
                IdempotencyRecordRow.idempotency_key == "adopt-reuse",
            )
        )
        assert receipt is not None
        event_types = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )
        experience_count = await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        )

    assert candidate.decision is CandidateDecision.ADOPTED
    assert candidate.resulting_experience_id == existing_experience_id
    assert candidate.resulting_version_id == existing_version_id
    assert adoption is not None
    assert adoption.created is False
    assert event_types == (CandidateAdoptedV1.event_type,)
    assert experience_count == 1
    await stack.container.projection_manager.validate_startup(
        stack.container.database
    )


@pytest.mark.asyncio
async def test_adopt_does_not_reuse_foreign_equivalent(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    foreign_experience_id, _ = await _create_equivalent(
        stack,
        candidate_id=candidate_id,
        candidate_owner_id=OWNER_ID,
        experience_owner_id=OTHER_OWNER_ID,
        key="create-foreign-equivalent",
    )

    result = await _adopt(stack, candidate_id=candidate_id, key="adopt-owned")

    assert result.status_code == 200
    async with stack.container.database.read_session() as session:
        candidate = await stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
        identities = tuple(
            (
                await session.scalars(
                    select(ExperienceRow).order_by(ExperienceRow.experience_id)
                )
            ).all()
        )

    assert candidate.resulting_experience_id != foreign_experience_id
    assert {row.owner_agent_id for row in identities} == {
        OWNER_ID,
        OTHER_OWNER_ID,
    }
    assert len(identities) == 2


@pytest.mark.parametrize("action", ("adopt", "reject"))
@pytest.mark.asyncio
async def test_same_decision_key_replays_exact_response(
    stack: DecisionStack,
    action: str,
) -> None:
    candidate_id = await _capture_pending(stack)
    decide = _adopt if action == "adopt" else _reject

    first = await decide(stack, candidate_id=candidate_id, key="same-key")
    replay = await decide(stack, candidate_id=candidate_id, key="same-key")

    assert first.status_code == 200
    assert first.replayed is False
    assert replay.status_code == first.status_code
    assert replay.body == first.body
    assert replay.headers == first.headers
    assert replay.replayed is True
    async with stack.container.database.read_session() as session:
        decision_events = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(
                DomainEventRow.event_type.in_(
                    (
                        CandidateAdoptedV1.event_type,
                        CandidateRejectedV1.event_type,
                    )
                )
            )
        )
    assert decision_events == 1


@pytest.mark.parametrize(
    ("first_action", "second_action"),
    (
        ("reject", "adopt"),
        ("adopt", "reject"),
        ("adopt", "adopt"),
        ("reject", "reject"),
    ),
)
@pytest.mark.asyncio
async def test_terminal_candidate_rejects_every_different_decision_key(
    stack: DecisionStack,
    first_action: str,
    second_action: str,
) -> None:
    candidate_id = await _capture_pending(stack)
    first = _adopt if first_action == "adopt" else _reject
    second = _adopt if second_action == "adopt" else _reject
    initial = await first(stack, candidate_id=candidate_id, key="first-decision")

    conflict = await second(
        stack,
        candidate_id=candidate_id,
        key="different-decision-key",
    )

    assert initial.status_code == 200
    assert conflict.status_code == 409
    assert json.loads(conflict.body) == {
        "error": {
            "code": "candidate_already_decided",
            "details": {},
            "message": "Candidate already has a terminal decision",
        }
    }


@pytest.mark.asyncio
async def test_missing_foreign_and_forged_candidate_decisions_are_identical(
    stack: DecisionStack,
) -> None:
    owned_id = await _capture_pending(stack)
    foreign_id = await _capture_pending(
        stack,
        owner_agent_id=OTHER_OWNER_ID,
        key="capture-foreign",
        label="foreign-cache",
    )

    results = (
        await _adopt(stack, candidate_id=UUID(int=999_999), key="missing"),
        await _adopt(stack, candidate_id=foreign_id, key="foreign"),
        await _adopt(
            stack,
            candidate_id=owned_id,
            key="forged",
            owner_agent_id=OTHER_OWNER_ID,
            caller_agent_id=OWNER_ID,
        ),
    )

    observed = tuple(
        (result.status_code, json.loads(result.body)) for result in results
    )
    assert observed == (
        (
            404,
            {
                "error": {
                    "code": "candidate_not_found",
                    "details": {},
                    "message": "Candidate was not found",
                }
            },
        ),
    ) * 3
    async with stack.container.database.read_session() as session:
        states = tuple(
            await session.scalars(
                select(ExperienceStateRow).order_by(
                    ExperienceStateRow.experience_id
                )
            )
        )
    assert states == ()


@pytest.mark.parametrize(
    ("action", "checkpoint"),
    (
        ("adopt", FaultCheckpoint.AFTER_SOURCE_INSERT),
        ("reject", FaultCheckpoint.AFTER_EVENT_APPEND),
        ("reject", FaultCheckpoint.AFTER_PROJECTION_APPLY),
        ("adopt", FaultCheckpoint.AFTER_RECEIPT_COMPLETION),
    ),
)
@pytest.mark.asyncio
async def test_decision_fault_rolls_back_receipt_lineage_event_and_projection(
    stack: DecisionStack,
    action: str,
    checkpoint: FaultCheckpoint,
) -> None:
    candidate_id = await _capture_pending(stack)
    stack.fault.reset_calls()
    stack.fault.checkpoint = checkpoint
    decide = _adopt if action == "adopt" else _reject

    with pytest.raises(RuntimeError, match=f"injected:{checkpoint.value}"):
        await decide(stack, candidate_id=candidate_id, key="faulted-decision")
    stack.fault.checkpoint = None
    if checkpoint is FaultCheckpoint.AFTER_EVENT_APPEND:
        assert stack.fault.calls.count(checkpoint) == 1

    async with stack.container.database.read_session() as session:
        state = await session.get(CandidateStateRow, candidate_id)
        adoption_count = await session.scalar(
            select(func.count()).select_from(CandidateAdoptionRow)
        )
        experience_count = await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        )
        decision_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(
                DomainEventRow.event_type.in_(
                    (
                        CandidateAdoptedV1.event_type,
                        CandidateRejectedV1.event_type,
                    )
                )
            )
        )
        decision_receipt_count = await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.idempotency_key == "faulted-decision")
        )

    assert state is not None
    assert state.decision == CandidateDecision.PENDING.value
    assert adoption_count == 0
    assert experience_count == 0
    assert decision_event_count == 0
    assert decision_receipt_count == 0


@pytest.mark.asyncio
async def test_projection_rebuild_restores_adopted_and_rejected_decisions(
    stack: DecisionStack,
) -> None:
    adopted_id = await _capture_pending(
        stack,
        key="capture-adopted",
        label="adopted-cache",
    )
    rejected_id = await _capture_pending(
        stack,
        key="capture-rejected",
        label="rejected-cache",
    )
    assert (
        await _adopt(stack, candidate_id=adopted_id, key="adopt-for-rebuild")
    ).status_code == 200
    assert (
        await _reject(stack, candidate_id=rejected_id, key="reject-for-rebuild")
    ).status_code == 200
    async with stack.container.database.read_session() as session:
        before = (
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=adopted_id,
            ),
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=rejected_id,
            ),
        )
    async with stack.container.database.transaction() as uow:
        await uow.session.execute(
            CandidateStateRow.__table__.delete().where(
                CandidateStateRow.candidate_id.in_((adopted_id, rejected_id))
            )
        )

    report = await stack.container.projection_manager.repair(
        stack.container.database
    )

    assert report.matches is True
    async with stack.container.database.read_session() as session:
        after = (
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=adopted_id,
            ),
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=rejected_id,
            ),
        )
    assert after == before
    assert after[0].decision is CandidateDecision.ADOPTED
    assert after[1].decision is CandidateDecision.REJECTED


@pytest.mark.asyncio
async def test_candidate_enters_ordinary_retrieval_only_after_adoption(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="cache retry",
        mode=RetrievalMode.FOCUSED,
    )
    before = await stack.container.retrieval_adapter.search(
        query=query,
        idempotency_key="search-before-adoption",
    )

    adopted = await _adopt(
        stack,
        candidate_id=candidate_id,
        key="adopt-for-retrieval",
    )
    assert adopted.status_code == 200
    async with stack.container.database.read_session() as session:
        candidate = await stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
    after = await stack.container.retrieval_adapter.search(
        query=query,
        idempotency_key="search-after-adoption",
    )
    before_hits = json.loads(before.body)["data"]["hits"]
    after_hits = json.loads(after.body)["data"]["hits"]

    assert before_hits == []
    assert len(after_hits) == 1
    assert (
        after_hits[0]["experience"]["experience_id"]
        == str(candidate.resulting_experience_id)
    )
    assert after_hits[0]["experience"]["body"] == candidate.content.body


@pytest.mark.asyncio
async def test_adoption_maps_corrupt_candidate_lineage_to_stable_conflict(
    stack: DecisionStack,
) -> None:
    candidate_id = await _capture_pending(stack)
    async with stack.container.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_candidates_reject_update")
        )
        await uow.session.execute(
            text(
                "UPDATE experience_candidates SET body = 'CORRUPT_SOURCE' "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(candidate_id)},
        )

    result = await _adopt(
        stack,
        candidate_id=candidate_id,
        key="adopt-corrupt-lineage",
    )

    assert result.status_code == 409
    assert json.loads(result.body) == {
        "error": {
            "code": "candidate_lineage_invalid",
            "details": {},
            "message": "Candidate lineage could not be preserved",
        }
    }
    async with stack.container.database.read_session() as session:
        adoption_count = await session.scalar(
            select(func.count()).select_from(CandidateAdoptionRow)
        )
        experience_count = await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        )
    assert adoption_count == 0
    assert experience_count == 0


@pytest.mark.parametrize(
    "tamper_kind",
    ("excerpt", "source_hash", "ordinal", "manifest"),
)
@pytest.mark.asyncio
async def test_candidate_reads_and_decisions_reject_post_startup_source_tampering(
    stack: DecisionStack,
    tamper_kind: str,
) -> None:
    candidate_id = await _capture_pending(stack)
    async with stack.container.database.transaction() as uow:
        if tamper_kind == "manifest":
            bundle = await uow.session.scalar(select(TrajectoryBundleRow))
            assert bundle is not None
            document = json.loads(bundle.manifest)
            document["steps"][0]["observation_hash"] = "f" * 64
            manifest = canonical_json_bytes(document)
            await uow.session.execute(
                text("DROP TRIGGER trajectory_bundles_reject_update")
            )
            await uow.session.execute(
                text(
                    "UPDATE trajectory_bundles SET manifest = :manifest, "
                    "manifest_hash = :manifest_hash WHERE bundle_id = :bundle_id"
                ),
                {
                    "bundle_id": str(bundle.bundle_id),
                    "manifest": manifest,
                    "manifest_hash": sha256_hex(manifest),
                },
            )
        else:
            assignments = {
                "excerpt": "excerpt = 'valid utf8 tampering'",
                "source_hash": f"source_hash = '{'f' * 64}'",
                "ordinal": "ordinal = ordinal + 1",
            }
            await uow.session.execute(
                text("DROP TRIGGER trajectory_evidence_reject_update")
            )
            await uow.session.execute(
                text(f"UPDATE trajectory_evidence SET {assignments[tamper_kind]}")
            )

    with pytest.raises(SourceIntegrityError) as get_error:
        async with stack.container.database.read_session() as session:
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=candidate_id,
            )
    with pytest.raises(SourceIntegrityError) as list_error:
        async with stack.container.database.read_session() as session:
            await stack.container.candidate_service.list_owned(
                session=session,
                owner_agent_id=OWNER_ID,
            )
    assert get_error.value.mismatch_key == f"experience_candidate:{candidate_id}"
    assert list_error.value.mismatch_key == f"experience_candidate:{candidate_id}"
    assert "valid utf8 tampering" not in str(get_error.value)
    assert "valid utf8 tampering" not in str(list_error.value)

    for action in ("adopt", "reject"):
        result = (
            await _adopt(
                stack,
                candidate_id=candidate_id,
                key=f"{action}-{tamper_kind}-source",
            )
            if action == "adopt"
            else await _reject(
                stack,
                candidate_id=candidate_id,
                key=f"{action}-{tamper_kind}-source",
            )
        )
        assert result.status_code == 409
        assert json.loads(result.body) == {
            "error": {
                "code": "candidate_lineage_invalid",
                "details": {},
                "message": "Candidate lineage could not be preserved",
            }
        }

    async with stack.container.database.read_session() as session:
        state = await session.get(CandidateStateRow, candidate_id)
        adoption_count = await session.scalar(
            select(func.count()).select_from(CandidateAdoptionRow)
        )
        experience_count = await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        )
    assert state is not None
    assert state.decision == CandidateDecision.PENDING.value
    assert adoption_count == 0
    assert experience_count == 0


@pytest.mark.parametrize("action", ("get", "adopt", "reject"))
@pytest.mark.asyncio
async def test_candidate_operation_authenticates_manifest_once(
    stack: DecisionStack,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    candidate_id = await _capture_pending(stack)
    calls: list[UUID] = []
    original = capture_source_integrity.authenticate_trajectory_manifest

    def counting_authenticate(
        bundle: TrajectoryBundleRow,
    ) -> capture_source_integrity.AuthenticatedTrajectoryManifest:
        calls.append(bundle.bundle_id)
        return original(bundle)

    monkeypatch.setattr(
        capture_source_integrity,
        "authenticate_trajectory_manifest",
        counting_authenticate,
    )

    if action == "get":
        async with stack.container.database.read_session() as session:
            await stack.container.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=candidate_id,
            )
    elif action == "adopt":
        result = await _adopt(
            stack,
            candidate_id=candidate_id,
            key="count-adopt-manifest-auth",
        )
        assert result.status_code == 200
    else:
        result = await _reject(
            stack,
            candidate_id=candidate_id,
            key="count-reject-manifest-auth",
        )
        assert result.status_code == 200

    assert len(calls) == 1


@pytest.mark.parametrize("action", ("adopt", "reject"))
@pytest.mark.asyncio
async def test_candidate_decisions_refuse_non_immediate_unit_of_work(
    stack: DecisionStack,
    action: str,
) -> None:
    candidate_id = await _capture_pending(stack)
    command = CommandContext(
        receipt_id=UUID(int=999_001),
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope=(
            CANDIDATE_ADOPT_SCOPE
            if action == "adopt"
            else CANDIDATE_REJECT_SCOPE
        ),
        idempotency_key="non-immediate",
        request_hash="f" * 64,
    )
    request: AdoptCandidate | RejectCandidate
    if action == "adopt":
        request = AdoptCandidate(
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
            importance=0.5,
            confidence=0.5,
        )
    else:
        request = RejectCandidate(
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
            reason=StructuredReason.from_user_text("Not reusable."),
        )

    with pytest.raises(
        RuntimeError,
        match="Candidate decisions require an immediate transaction",
    ):
        async with stack.container.database.transaction() as uow:
            if isinstance(request, AdoptCandidate):
                await stack.container.candidate_service.adopt(
                    uow=uow,
                    request=request,
                    command=command,
                )
            else:
                await stack.container.candidate_service.reject(
                    uow=uow,
                    request=request,
                    command=command,
                )
