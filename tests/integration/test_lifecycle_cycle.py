from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select, update
from tests.integration.test_create_experience import (
    OWNER_ID,
    Stack,
    build_stack,
    create,
)

from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.contracts import ConfirmExperience
from experience_hub.experiences.models import Temperature
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.lifecycle.contracts import (
    LifecycleResult,
    decode_lifecycle_result,
    encode_lifecycle_result,
)
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import LifecycleRunMode, LifecycleService
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "lifecycle-cycle.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def lifecycle_service(
    stack: Stack,
    *,
    config: LifecycleConfig | None = None,
) -> LifecycleService:
    return LifecycleService(
        clock=stack.clock,
        receipt_store=stack.receipts,
        repository=LifecycleRepository(),
        mutation_writer=ExperienceMutationWriter(repository=stack.repository),
        config=config,
    )


async def run_cycle(
    stack: Stack,
    service: LifecycleService,
    *,
    key: str,
    evaluated_at: datetime,
    mode: LifecycleRunMode = "manual",
) -> tuple[int, bytes, dict[str, Any]]:
    command_request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=key,
        method="POST",
        route_template="/v1/lifecycle:run",
        body={
            "evaluated_at": evaluated_at,
            "mode": mode,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode=mode,
        )

    result = await stack.executor.execute(command_request, handler)
    return result.status_code, result.body, json.loads(result.body)


@pytest.mark.parametrize(
    ("caller_scope", "operation_scope"),
    (
        (f"agent:{OWNER_ID}", "lifecycle.run"),
        ("system:local", "forged.lifecycle.run"),
    ),
)
@pytest.mark.asyncio
async def test_lifecycle_run_rejects_non_system_command_scope(
    stack: Stack,
    caller_scope: str,
    operation_scope: str,
) -> None:
    service = lifecycle_service(stack)
    evaluated_at = stack.clock.now()
    request = CommandRequest(
        caller_scope=caller_scope,
        operation_scope=operation_scope,
        idempotency_key=f"invalid-scope:{caller_scope}:{operation_scope}",
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": evaluated_at, "mode": "manual"},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode="manual",
        )

    result = await stack.executor.execute(request, handler)

    assert result.status_code == 404
    assert json.loads(result.body)["error"]["code"] == "resource_not_found"
    async with stack.database.read_session() as session:
        event_count = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow)) or 0
        )
    assert event_count == 0


@pytest.mark.asyncio
async def test_lifecycle_run_rejects_context_from_different_request_semantics(
    stack: Stack,
) -> None:
    service = lifecycle_service(stack)
    evaluated_at = stack.clock.now()
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key="mismatched-lifecycle-request",
        method="POST",
        route_template="/forged/lifecycle",
        body={
            "evaluated_at": evaluated_at - timedelta(days=1),
            "mode": "background",
            "extra": True,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode="manual",
        )

    result = await stack.executor.execute(request, handler)

    assert result.status_code == 404
    assert json.loads(result.body)["error"]["code"] == "resource_not_found"
    async with stack.database.read_session() as session:
        assert (
            int(
                await session.scalar(select(func.count()).select_from(DomainEventRow))
                or 0
            )
            == 0
        )


@pytest.mark.asyncio
async def test_omitted_evaluation_time_is_derived_from_its_receipt(
    stack: Stack,
) -> None:
    service = lifecycle_service(stack)
    receipt_time = stack.clock.now()
    mismatched_time = receipt_time - timedelta(minutes=15)
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key="omitted-time-owned-by-receipt",
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": None, "mode": "manual"},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.run(
            uow=uow,
            evaluated_at=mismatched_time,
            command=context,
            mode="manual",
            evaluated_at_was_omitted=True,
        )

    result = await stack.executor.execute(request, handler)

    assert result.status_code == 200
    assert decode_lifecycle_result(result.body).evaluated_at == receipt_time


@pytest.mark.asyncio
async def test_cycle_orders_experiences_and_cross_key_replays_after_changes(
    stack: Stack,
) -> None:
    _, first = await create(stack, key="cycle-first")
    _, second = await create(stack, key="cycle-second")
    first_id = UUID(first["data"]["experience_id"])
    second_id = UUID(second["data"]["experience_id"])
    evaluated_at = stack.clock.advance(timedelta(minutes=15))
    service = lifecycle_service(stack)

    status, first_body, first_decoded = await run_cycle(
        stack,
        service,
        key="cycle-key-a",
        evaluated_at=evaluated_at,
    )

    assert status == 200
    first_result = decode_lifecycle_result(first_body)
    assert first_result.evaluated_count == 2
    assert first_result.transition_count == 0
    assert first_decoded["data"]["cycle_id"] == str(first_result.cycle_id)
    assert first_decoded["data"]["evaluated_count"] == 2
    async with stack.database.read_session() as session:
        cycle_events = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == "experience.lifecycle_evaluated")
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    assert [event.aggregate_id for event in cycle_events] == sorted(
        (first_id, second_id),
        key=lambda value: value.bytes,
    )
    for row in cycle_events:
        payload = stack.registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
        assert getattr(payload, "cycle_id", None) == first_result.cycle_id

    stack.clock.advance(timedelta(minutes=1))

    async def confirm_handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.confirm(
            uow=uow,
            command=ConfirmExperience(
                owner_agent_id=OWNER_ID,
                experience_id=first_id,
            ),
            command_context=context,
        )

    confirm_request = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope="experience.confirm",
        idempotency_key="state-change-after-cycle",
        method="POST",
        route_template=("/v1/agents/{agent_id}/experiences/{experience_id}:confirm"),
        path_parameters={
            "agent_id": OWNER_ID,
            "experience_id": first_id,
        },
        body={},
    )
    confirm_result = await stack.executor.execute(
        confirm_request,
        confirm_handler,
    )
    assert confirm_result.status_code == 200
    status, replay_body, _ = await run_cycle(
        stack,
        service,
        key="cycle-key-b",
        evaluated_at=evaluated_at,
    )
    assert status == 200
    assert replay_body == first_body
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "experience.lifecycle_evaluated")
            )
            == 2
        )


@pytest.mark.asyncio
async def test_cross_key_replay_rejects_mismatched_result_cycle_id(
    stack: Stack,
) -> None:
    await create(stack, key="corrupt-cycle-source")
    evaluated_at = stack.clock.advance(timedelta(minutes=15))
    service = lifecycle_service(stack)
    _, body, _ = await run_cycle(
        stack,
        service,
        key="corrupt-cycle-original",
        evaluated_at=evaluated_at,
    )
    result = decode_lifecycle_result(body)
    wrong_cycle_id = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
    corrupt_body = encode_lifecycle_result(
        LifecycleResult(
            cycle_id=wrong_cycle_id,
            evaluated_at=result.evaluated_at,
            evaluated_count=result.evaluated_count,
            transition_count=result.transition_count,
            archive_count=result.archive_count,
            idea_archive_count=result.idea_archive_count,
        )
    )
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.result_resource_type == "lifecycle_cycle",
                IdempotencyRecordRow.result_resource_id == result.cycle_id,
            )
            .values(response_body=corrupt_body)
        )

    with pytest.raises(RuntimeError, match="mismatched cycle ID"):
        await run_cycle(
            stack,
            service,
            key="corrupt-cycle-replay",
            evaluated_at=evaluated_at,
        )


@pytest.mark.asyncio
async def test_custom_single_cycle_demotion_keeps_writer_protocol_aligned(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    config = LifecycleConfig(demotion_cycles=1)
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "single-cycle-demotion.sqlite3",
        lifecycle_config=config,
    )
    try:
        _, created = await create(
            stack,
            key="single-cycle-demotion-source",
            confidence=0.20,
            importance=0.20,
        )
        experience_id = UUID(created["data"]["experience_id"])
        evaluated_at = stack.clock.advance(timedelta(days=30))
        service = lifecycle_service(stack, config=config)

        status, body, _ = await run_cycle(
            stack,
            service,
            key="single-cycle-demotion",
            evaluated_at=evaluated_at,
        )

        result = decode_lifecycle_result(body)
        assert status == 200
        assert result.transition_count == 1
        async with stack.database.read_session() as session:
            state = await session.get(ExperienceStateRow, experience_id)
            event_types = tuple(
                (
                    await session.scalars(
                        select(DomainEventRow.event_type)
                        .where(DomainEventRow.aggregate_id == experience_id)
                        .order_by(DomainEventRow.sequence)
                    )
                ).all()
            )
        assert state is not None
        assert state.temperature is Temperature.COLD
        assert event_types[-2:] == (
            "experience.lifecycle_evaluated",
            "experience.temperature_changed",
        )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_archive_cycle_has_exact_three_events_and_compresses_payload(
    stack: Stack,
) -> None:
    _, created = await create(
        stack,
        key="archive-source",
        confidence=0.20,
        importance=0.20,
    )
    experience_id = UUID(created["data"]["experience_id"])
    evaluated_at = stack.clock.advance(timedelta(days=100))
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(
                temperature=Temperature.COLD,
                confidence=0.20,
                importance=0.20,
                access_strength=0.0,
                strength_updated_at=evaluated_at - timedelta(days=100),
                last_transition_at=evaluated_at - timedelta(days=91),
                last_lifecycle_evaluated_at=None,
            )
        )
    service = lifecycle_service(stack)

    status, body, _ = await run_cycle(
        stack,
        service,
        key="archive-cycle",
        evaluated_at=evaluated_at,
    )

    result = decode_lifecycle_result(body)
    assert status == 200
    assert (result.evaluated_count, result.transition_count, result.archive_count) == (
        1,
        1,
        1,
    )
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
        events = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.aggregate_id == experience_id)
                .order_by(DomainEventRow.sequence)
            )
        ).all()
        codecs = tuple(
            (await session.scalars(select(ExperiencePayloadRow.codec))).all()
        )
    assert state is not None
    assert state.temperature is Temperature.ARCHIVED
    assert [event.event_type for event in events[-3:]] == [
        "experience.lifecycle_evaluated",
        "experience.archived",
        "experience.temperature_changed",
    ]
    assert {
        getattr(
            stack.registry.decode(
                event_type=event.event_type,
                payload=event.payload,
            ),
            "cycle_id",
            None,
        )
        for event in events[-3:]
    } == {result.cycle_id}
    assert {codec.value for codec in codecs} == {"zlib"}


@pytest.mark.asyncio
async def test_zero_event_cycle_is_durable_only_as_completed_resource(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="already-archived")
    experience_id = UUID(created["data"]["experience_id"])
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(temperature=Temperature.ARCHIVED)
        )
    service = lifecycle_service(stack)

    status, body, _ = await run_cycle(
        stack,
        service,
        key="zero-event-cycle",
        evaluated_at=stack.clock.now(),
    )

    result = decode_lifecycle_result(body)
    assert status == 200
    assert result.evaluated_count == result.transition_count == 0
    async with stack.database.read_session() as session:
        lifecycle_events = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(
                DomainEventRow.event_type.in_(
                    (
                        "experience.lifecycle_evaluated",
                        "experience.archived",
                    )
                )
            )
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.result_resource_type == "lifecycle_cycle",
                IdempotencyRecordRow.result_resource_id == result.cycle_id,
                IdempotencyRecordRow.state == "completed",
            )
        )
    assert lifecycle_events == 0
    assert receipt is not None


@pytest.mark.asyncio
async def test_future_cycle_is_replayable_error_without_resource_or_events(
    stack: Stack,
) -> None:
    await create(stack, key="future-cycle-source")
    service = lifecycle_service(stack)

    status, _, body = await run_cycle(
        stack,
        service,
        key="future-cycle",
        evaluated_at=stack.clock.now() + timedelta(microseconds=1),
    )

    assert status == 422
    assert body["error"]["code"] == "invalid_evaluated_at"
    async with stack.database.read_session() as session:
        resources = await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.result_resource_type == "lifecycle_cycle")
        )
        events = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "experience.lifecycle_evaluated")
        )
    assert resources == events == 0


@pytest.mark.asyncio
async def test_live_manual_or_background_lease_excludes_another_cycle(
    stack: Stack,
) -> None:
    await create(stack, key="lease-exclusion-source")
    service = lifecycle_service(stack)
    lease_owner = UUID("90000000-0000-0000-0000-000000000001")
    repository = LifecycleRepository()
    async with stack.database.transaction(immediate=True) as uow:
        assert await repository.claim_lease(
            uow.session,
            owner_id=lease_owner,
            at=stack.clock.now(),
            ttl=timedelta(minutes=5),
        )

    blocked_status, _, blocked = await run_cycle(
        stack,
        service,
        key="blocked-background-cycle",
        evaluated_at=stack.clock.now(),
        mode="background",
    )

    assert blocked_status == 409
    assert blocked["error"]["code"] == "lifecycle_in_progress"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "experience.lifecycle_evaluated")
            )
            == 0
        )
    async with stack.database.transaction(immediate=True) as uow:
        assert await repository.release_lease(
            uow.session,
            owner_id=lease_owner,
        )

    resumed_status, _, _ = await run_cycle(
        stack,
        service,
        key="resumed-manual-cycle",
        evaluated_at=stack.clock.now(),
        mode="manual",
    )
    assert resumed_status == 200
