from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
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
    request,
)
from tests.integration.test_lifecycle_cycle import (
    lifecycle_service,
    run_cycle,
)

from experience_hub.domain import CommandContext, StructuredReason, TypedEvidence
from experience_hub.experiences.contracts import (
    ConfirmExperience,
    PinExperience,
    RefuteExperience,
    RestoreExperience,
    UnpinExperience,
)
from experience_hub.experiences.events import (
    ExperienceConfirmedV1,
    ExperiencePinnedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
)
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.service import ExperienceService
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import DomainEventRow, ExperienceStateRow
from experience_hub.storage.unit_of_work import UnitOfWork


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-mutations.sqlite3",
    )
    value.service = ExperienceService(
        clock=value.clock,
        receipt_store=value.receipts,
        writer=value.writer,
        mutation_writer=ExperienceMutationWriter(repository=value.repository),
        query=ExperienceQuery(event_registry=value.registry),
    )
    try:
        yield value
    finally:
        await value.database.dispose()


Mutation = (
    ConfirmExperience
    | RefuteExperience
    | PinExperience
    | UnpinExperience
    | RestoreExperience
)
MutationHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]


async def mutate(
    stack: Stack,
    *,
    key: str,
    command: Mutation,
    handler: MutationHandler,
) -> tuple[int, dict[str, Any]]:
    async def execute_handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await handler(uow, context)

    result = await stack.executor.execute(
        request(
            key=key,
            owner_agent_id=command.owner_agent_id,
            operation=f"experience.{key}",
        ),
        execute_handler,
    )
    return result.status_code, json.loads(result.body)


@pytest.mark.asyncio
async def test_confirm_materializes_formula_and_promotes_without_body(
    stack: Stack,
) -> None:
    _, created = await create(
        stack,
        key="confirm-source",
        confidence=0.50,
    )
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    evidence = (
        TypedEvidence(type="test", id="b"),
        TypedEvidence(type="test", id="a"),
        TypedEvidence(type="test", id="a"),
    )

    status, body = await mutate(
        stack,
        key="confirm",
        command=ConfirmExperience(
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            reason="  independently reproduced  ",
            evidence=evidence,
        ),
        handler=lambda uow, context: stack.service.confirm(
            uow=uow,
            command=ConfirmExperience(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                reason="  independently reproduced  ",
                evidence=evidence,
            ),
            command_context=context,
        ),
    )

    assert status == 200
    assert body["data"]["confidence"] == pytest.approx(0.60)
    assert body["data"]["temperature"] == "hot"
    assert body["data"]["blurred"] is True
    assert body["data"]["body"] is None
    async with stack.database.read_session() as session:
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.aggregate_id == experience_id)
                .order_by(DomainEventRow.sequence)
            )
        ).all()
    assert [row.event_type for row in rows[-2:]] == [
        ExperienceConfirmedV1.event_type,
        ExperienceTemperatureChangedV1.event_type,
    ]
    payload = stack.registry.decode(
        event_type=rows[-2].event_type,
        payload=rows[-2].payload,
    )
    assert isinstance(payload, ExperienceConfirmedV1)
    assert payload.reason == StructuredReason.from_user_text(
        "independently reproduced"
    )
    assert payload.evidence == tuple(
        sorted(set(evidence), key=lambda item: (item.type, item.id))
    )


@pytest.mark.asyncio
async def test_refute_applies_locked_formula_without_immediate_demotion(
    stack: Stack,
) -> None:
    _, created = await create(
        stack,
        key="refute-source",
        importance=0.90,
        confidence=0.80,
    )
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(minutes=1))

    status, body = await mutate(
        stack,
        key="refute",
        command=RefuteExperience(
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
        ),
        handler=lambda uow, context: stack.service.refute(
            uow=uow,
            command=RefuteExperience(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
            ),
            command_context=context,
        ),
    )

    assert status == 200
    assert body["data"]["confidence"] == pytest.approx(0.52)
    assert body["data"]["temperature"] == "hot"
    async with stack.database.read_session() as session:
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.aggregate_id == experience_id)
                .order_by(DomainEventRow.sequence)
            )
        ).all()
    assert rows[-1].event_type == ExperienceRefutedV1.event_type


@pytest.mark.asyncio
async def test_pin_unpin_and_matching_noops_have_exact_event_order(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="pin-source")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(minutes=1))

    pin = PinExperience(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
    )
    await mutate(
        stack,
        key="pin",
        command=pin,
        handler=lambda uow, context: stack.service.pin(
            uow=uow,
            command=pin,
            command_context=context,
        ),
    )
    await mutate(
        stack,
        key="pin-again",
        command=pin,
        handler=lambda uow, context: stack.service.pin(
            uow=uow,
            command=pin,
            command_context=context,
        ),
    )
    unpin = UnpinExperience(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
    )
    await mutate(
        stack,
        key="unpin",
        command=unpin,
        handler=lambda uow, context: stack.service.unpin(
            uow=uow,
            command=unpin,
            command_context=context,
        ),
    )
    stack.clock.advance(timedelta(hours=-1))
    noop_status, _ = await mutate(
        stack,
        key="unpin-again",
        command=unpin,
        handler=lambda uow, context: stack.service.unpin(
            uow=uow,
            command=unpin,
            command_context=context,
        ),
    )
    assert noop_status == 200

    async with stack.database.read_session() as session:
        types = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.event_type)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
    assert types[-3:] == (
        ExperiencePinnedV1.event_type,
        ExperienceTemperatureChangedV1.event_type,
        ExperienceUnpinnedV1.event_type,
    )


@pytest.mark.asyncio
async def test_restore_is_the_only_archived_mutation_and_rejects_old_clock(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="restore-source")
    experience_id = UUID(created["data"]["experience_id"])
    archived_at = stack.clock.advance(timedelta(days=100))
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(
                temperature="cold",
                confidence=0.20,
                importance=0.20,
                access_strength=0.0,
                last_transition_at=archived_at - timedelta(days=91),
                strength_updated_at=archived_at - timedelta(days=100),
            )
        )
    archive_status, _, _ = await run_cycle(
        stack,
        lifecycle_service(stack),
        key="archive-before-restore",
        evaluated_at=archived_at,
    )
    assert archive_status == 200

    archived_confirm_status, archived_confirm = await mutate(
        stack,
        key="archived-confirm",
        command=ConfirmExperience(
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
        ),
        handler=lambda uow, context: stack.service.confirm(
            uow=uow,
            command=ConfirmExperience(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
            ),
            command_context=context,
        ),
    )
    assert archived_confirm_status == 409
    assert archived_confirm["error"]["code"] == "restore_required"

    stack.clock.advance(timedelta(minutes=1))
    status, body = await mutate(
        stack,
        key="restore",
        command=RestoreExperience(
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            reason="needed for a new incident",
        ),
        handler=lambda uow, context: stack.service.restore(
            uow=uow,
            command=RestoreExperience(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                reason="needed for a new incident",
            ),
            command_context=context,
        ),
    )
    assert status == 200
    assert body["data"]["temperature"] == "warm"
    async with stack.database.read_session() as session:
        types = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.event_type)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
    assert types[-2:] == (
        ExperienceRestoredV1.event_type,
        ExperienceTemperatureChangedV1.event_type,
    )

    async with stack.database.transaction() as uow:
        before = await uow.session.scalar(
            select(func.count()).select_from(DomainEventRow)
        )
    stack.clock.advance(timedelta(days=-1))
    old_status, old_body = await mutate(
        stack,
        key="old-refute",
        command=RefuteExperience(
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
        ),
        handler=lambda uow, context: stack.service.refute(
            uow=uow,
            command=RefuteExperience(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
            ),
            command_context=context,
        ),
    )
    assert old_status == 409
    assert old_body["error"]["code"] == "clock_regression"
    async with stack.database.read_session() as session:
        assert await session.scalar(
            select(func.count()).select_from(DomainEventRow)
        ) == before
