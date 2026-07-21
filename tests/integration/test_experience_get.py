from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_create_experience import (
    EXPERIENCE_IDS,
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    content,
    create,
    request,
)

from experience_hub import canonical_json_bytes
from experience_hub.domain import CommandContext, CommandRequest, PendingEvent
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
)
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    ExperienceAccessedV1,
    ExperienceReactivatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
)
from experience_hub.experiences.projector import (
    ExperienceProjectionIntegrityError,
)
from experience_hub.experiences.repository import snapshot_from_state_row
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.lifecycle import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
    record_access,
)
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

QUERY_HASH = "d" * 64


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-get.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def _create_cold(stack: Stack, *, key: str) -> UUID:
    created: list[UUID] = []

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        creation = await stack.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=OWNER_ID,
                actor_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=content(key),
                importance=0.35,
                confidence=0.5,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=stack.clock.now(),
            ),
            command=command,
        )
        created.append(creation.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    await stack.executor.execute(
        request(
            key=key,
            operation="experience.test_create_cold",
        ),
        handler,
    )
    assert created == [EXPERIENCE_IDS[0]]
    return created[0]


def _activation_inputs(
    state: ExperienceStateSnapshotV1,
    *,
    created_at: datetime,
) -> ActivationInputs:
    return ActivationInputs(
        importance=state.importance,
        confidence=state.confidence,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        created_at=created_at,
    )


async def _append(
    stack: Stack,
    *,
    key: str,
    events: Sequence[PendingEvent],
) -> None:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        await uow.append_events(command, events)
        return StoredResponse(status_code=200, body=b"{}")

    await stack.executor.execute(
        request(key=key, operation="experience.test_state_events"),
        handler,
    )


def _retrieval_service(stack: Stack) -> RetrievalService:
    return RetrievalService(
        clock=stack.clock,
        query=stack.query,
        mutation_writer=ExperienceMutationWriter(
            repository=stack.repository,
        ),
    )


async def _get(
    stack: Stack,
    *,
    service: RetrievalService,
    key: str,
    owner_agent_id: UUID,
    experience_id: UUID,
    caller_agent_id: UUID,
) -> CommandResult:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        view = await service.get(
            uow=uow,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
            command=command,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes(
                {"data": view.model_dump(mode="json")}
            ),
        )

    return await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{caller_agent_id}",
            operation_scope="experience.get",
            idempotency_key=key,
            method="GET",
            route_template="/v1/experiences/{experience_id}",
            path_parameters={"experience_id": str(experience_id)},
        ),
        handler,
    )


async def _event_count(
    stack: Stack,
    *,
    experience_id: UUID,
    event_type: str | None = None,
) -> int:
    statement = (
        select(func.count())
        .select_from(DomainEventRow)
        .where(DomainEventRow.aggregate_id == experience_id)
    )
    if event_type is not None:
        statement = statement.where(DomainEventRow.event_type == event_type)
    async with stack.database.read_session() as session:
        value = await session.scalar(statement)
    assert value is not None
    return value


@pytest.mark.asyncio
async def test_state_projector_reduces_access_reactivation_and_temperature(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="cold-state-events")
    stack.clock.advance(timedelta(hours=1))
    occurred_at = stack.clock.now()

    async with stack.database.read_session() as session:
        row = await session.get(ExperienceStateRow, experience_id)
    assert row is not None
    before = snapshot_from_state_row(row)
    access = record_access(
        _activation_inputs(before, created_at=NOW),
        occurred_at,
        LifecycleConfig(),
    )
    after_access = ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "access_count": access.access_count,
            "access_strength": access.access_strength,
            "strength_updated_at": access.strength_updated_at,
            "last_accessed_at": access.last_accessed_at,
            "activation_score": activation_at(
                ActivationInputs(
                    importance=before.importance,
                    confidence=before.confidence,
                    access_count=access.access_count,
                    access_strength=access.access_strength,
                    strength_updated_at=access.strength_updated_at,
                    last_accessed_at=access.last_accessed_at,
                    created_at=NOW,
                ),
                occurred_at,
                LifecycleConfig(),
            ).score,
        }
    )
    after_temperature = ExperienceStateSnapshotV1.model_validate(
        {
            **after_access.model_dump(mode="python"),
            "temperature": Temperature.WARM,
            "last_transition_at": occurred_at,
            "consecutive_below_threshold": 0,
        }
    )
    events = (
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=experience_id,
            event_type=ExperienceAccessedV1.event_type,
            payload=ExperienceAccessedV1(
                schema_version=1,
                experience_id=experience_id,
                version_id=before.current_version_id,
                before=before,
                after=after_access,
            ),
            actor_agent_id=OWNER_ID,
            occurred_at=occurred_at,
        ),
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=experience_id,
            event_type=ExperienceReactivatedV1.event_type,
            payload=ExperienceReactivatedV1(
                schema_version=1,
                experience_id=experience_id,
                query_hash=QUERY_HASH,
                mode="focused",
                signal=0.72,
                before=after_access,
                after=after_access,
            ),
            actor_agent_id=OWNER_ID,
            occurred_at=occurred_at,
        ),
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=experience_id,
            event_type=ExperienceTemperatureChangedV1.event_type,
            payload=ExperienceTemperatureChangedV1(
                schema_version=1,
                experience_id=experience_id,
                cause="cold_reactivation",
                cycle_id=None,
                before=after_access,
                after=after_temperature,
            ),
            actor_agent_id=OWNER_ID,
            occurred_at=occurred_at,
        ),
    )

    await _append(stack, key="state-event-chain", events=events)

    async with stack.database.read_session() as session:
        projected = await session.get(ExperienceStateRow, experience_id)
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )

    assert stack.projector.event_types == STATE_EXPERIENCE_EVENT_TYPES
    assert projected is not None
    assert snapshot_from_state_row(projected) == after_temperature
    assert projected.projection_event_id == event_rows[-1].event_id
    assert [row.event_type for row in event_rows] == [
        "experience.created",
        "experience.version_created",
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    ]
    assert [row.sequence for row in event_rows] == [1, 2, 3, 4, 5]
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_access_reducer_rejects_incorrect_materialized_strength(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="bad-access")
    stack.clock.advance(timedelta(hours=1))
    occurred_at = stack.clock.now()
    async with stack.database.read_session() as session:
        row = await session.get(ExperienceStateRow, experience_id)
    assert row is not None
    before = snapshot_from_state_row(row)
    invalid_after = ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "access_count": 1,
            "access_strength": 19.0,
            "strength_updated_at": occurred_at,
            "last_accessed_at": occurred_at,
            "activation_score": 0.9,
        }
    )
    event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceAccessedV1.event_type,
        payload=ExperienceAccessedV1(
            schema_version=1,
            experience_id=experience_id,
            version_id=before.current_version_id,
            before=before,
            after=invalid_after,
        ),
        actor_agent_id=OWNER_ID,
        occurred_at=occurred_at,
    )

    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="materialization",
    ):
        await _append(stack, key="bad-access-event", events=(event,))


@pytest.mark.asyncio
async def test_temperature_reducer_requires_transition_time_to_match_event(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="bad-transition-time")
    stack.clock.advance(timedelta(hours=1))
    occurred_at = stack.clock.now()
    async with stack.database.read_session() as session:
        row = await session.get(ExperienceStateRow, experience_id)
    assert row is not None
    before = snapshot_from_state_row(row)
    after = ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "temperature": Temperature.WARM,
            "last_transition_at": occurred_at + timedelta(seconds=1),
        }
    )
    event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceTemperatureChangedV1.event_type,
        payload=ExperienceTemperatureChangedV1(
            schema_version=1,
            experience_id=experience_id,
            cause="cold_reactivation",
            cycle_id=None,
            before=before,
            after=after,
        ),
        actor_agent_id=OWNER_ID,
        occurred_at=occurred_at,
    )

    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="transition time",
    ):
        await _append(stack, key="bad-transition-event", events=(event,))


@pytest.mark.parametrize(
    ("importance", "expected_temperature"),
    [
        (0.35, Temperature.WARM),
        (0.85, Temperature.HOT),
    ],
)
@pytest.mark.asyncio
async def test_real_get_returns_full_warm_or_hot_body_once_and_replays(
    stack: Stack,
    importance: float,
    expected_temperature: Temperature,
) -> None:
    key = f"real-get-{expected_temperature.value}"
    status, created = await create(
        stack,
        key=key,
        value=content(key),
        importance=importance,
    )
    assert status == 201
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    service = _retrieval_service(stack)

    first = await _get(
        stack,
        service=service,
        key=f"{key}-access",
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        caller_agent_id=OWNER_ID,
    )
    replay = await _get(
        stack,
        service=service,
        key=f"{key}-access",
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        caller_agent_id=OWNER_ID,
    )

    assert first.status_code == replay.status_code == 200
    assert not first.replayed
    assert replay.replayed
    assert first.body == replay.body
    response = json.loads(first.body)
    assert response["data"]["temperature"] == expected_temperature.value
    assert response["data"]["body"] == content(key).body
    assert response["data"]["blurred"] is False
    assert response["data"]["body_is_excerpt"] is False
    assert response["data"]["access_count"] == 1
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
    assert state is not None
    assert state.temperature is expected_temperature
    assert state.access_count == 1
    assert state.last_accessed_at == stack.clock.now()
    assert (
        await _event_count(
            stack,
            experience_id=experience_id,
            event_type=ExperienceAccessedV1.event_type,
        )
        == 1
    )
    assert await _event_count(stack, experience_id=experience_id) == 3


@pytest.mark.asyncio
async def test_real_get_keeps_cold_body_blurred_and_zlib_without_access(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="real-get-cold")
    service = _retrieval_service(stack)

    result = await _get(
        stack,
        service=service,
        key="real-get-cold-access",
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        caller_agent_id=OWNER_ID,
    )

    assert result.status_code == 200
    response = json.loads(result.body)
    assert response["data"]["temperature"] == Temperature.COLD.value
    assert response["data"]["body"] is None
    assert response["data"]["blurred"] is True
    assert response["data"]["body_is_excerpt"] is False
    assert response["data"]["access_count"] == 0
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
        assert state is not None
        payload = await session.get(
            ExperiencePayloadRow,
            state.current_version_id,
        )
    assert state.temperature is Temperature.COLD
    assert state.access_count == 0
    assert payload is not None and payload.codec is PayloadCodec.ZLIB
    assert (
        await _event_count(
            stack,
            experience_id=experience_id,
            event_type=ExperienceAccessedV1.event_type,
        )
        == 0
    )
    assert await _event_count(stack, experience_id=experience_id) == 2


@pytest.mark.asyncio
async def test_real_get_returns_identical_404_for_foreign_and_missing(
    stack: Stack,
) -> None:
    status, created = await create(stack, key="real-get-hidden")
    assert status == 201
    experience_id = UUID(created["data"]["experience_id"])
    missing_id = UUID("00000000-0000-0000-0000-000000009999")
    service = _retrieval_service(stack)

    foreign = await _get(
        stack,
        service=service,
        key="real-get-foreign",
        owner_agent_id=OTHER_OWNER_ID,
        experience_id=experience_id,
        caller_agent_id=OTHER_OWNER_ID,
    )
    missing = await _get(
        stack,
        service=service,
        key="real-get-missing",
        owner_agent_id=OWNER_ID,
        experience_id=missing_id,
        caller_agent_id=OWNER_ID,
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.body == missing.body
    assert json.loads(foreign.body)["error"]["code"] == "experience_not_found"
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
    assert state is not None and state.access_count == 0
    assert await _event_count(stack, experience_id=experience_id) == 2


@pytest.mark.asyncio
async def test_real_get_rejects_caller_owner_mismatch_without_side_effects(
    stack: Stack,
) -> None:
    status, created = await create(stack, key="real-get-scope-mismatch")
    assert status == 201
    experience_id = UUID(created["data"]["experience_id"])
    service = _retrieval_service(stack)

    result = await _get(
        stack,
        service=service,
        key="real-get-scope-mismatch-access",
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        caller_agent_id=OTHER_OWNER_ID,
    )

    assert result.status_code == 404
    assert json.loads(result.body)["error"]["code"] == "experience_not_found"
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
        assert state is not None
        payload = await session.get(
            ExperiencePayloadRow,
            state.current_version_id,
        )
    assert state.access_count == 0
    assert payload is not None and payload.codec is PayloadCodec.PLAIN
    assert await _event_count(stack, experience_id=experience_id) == 2
