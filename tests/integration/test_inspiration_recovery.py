from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from tests.integration.test_inspiration_run import (
    FakeGenerator,
    FakeSnapshotBuilder,
    Stack,
    build_stack,
    command,
    request,
)

from experience_hub.inspiration.recovery import InspirationRunRecovery
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
)
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.storage.tables import (
    DomainEventRow,
    IdempotencyRecordRow,
    InspirationRunStateRow,
)
from experience_hub.storage.validation import (
    SourceValidator,
    register_inspiration_source_validator,
)


class AdvancingClock:
    """Clock whose every read advances, like a production wall clock."""

    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        retained = self.current
        self.current += timedelta(microseconds=1)
        return retained


@pytest.fixture
async def interrupted_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-recovery.sqlite3",
        generator=FakeGenerator(cancellation=True),
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def recovery(stack: Stack) -> InspirationRunRecovery:
    receipts = stack.executor._receipt_store  # noqa: SLF001
    return InspirationRunRecovery(
        database=stack.database,
        receipt_store=receipts,
        repository=InspirationRepository(
            stack.executor._repository._event_registry  # noqa: SLF001
        ),
        response_codec=InspirationResponseCodec(),
        clock=stack.clock,
    )


@pytest.mark.asyncio
async def test_startup_recovery_fails_snapshot_trace_without_regeneration(
    interrupted_stack: Stack,
) -> None:
    with pytest.raises(asyncio.CancelledError):
        await interrupted_stack.executor.execute(
            request=request(key="interrupted-provider"),
            run=command(),
        )
    assert interrupted_stack.generator.calls
    calls_before = tuple(interrupted_stack.generator.calls)
    interrupted_stack.clock.advance(timedelta(minutes=5))

    recovered = await recovery(interrupted_stack).recover()

    assert len(recovered) == 1
    run_id = recovered[0]
    assert tuple(interrupted_stack.generator.calls) == calls_before
    async with interrupted_stack.database.read_session() as session:
        events = (
            await session.scalars(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).all()
        receipts = (
            await session.scalars(
                select(IdempotencyRecordRow).order_by(
                    IdempotencyRecordRow.created_at,
                    IdempotencyRecordRow.receipt_id,
                )
            )
        ).all()
        state = await session.get(InspirationRunStateRow, run_id)
    assert tuple(event.event_type for event in events) == (
        "inspiration.started",
        "inspiration.snapshot_frozen",
        "inspiration.failed",
    )
    assert events[-1].actor_agent_id is None
    assert events[-1].causation_id != events[0].causation_id
    assert events[-1].occurred_at == interrupted_stack.clock.now()
    assert state is not None and state.status == "failed"
    assert len(receipts) == 2
    assert {receipt.state for receipt in receipts} == {"completed"}
    assert receipts[0].response_body == receipts[1].response_body
    assert json.loads(receipts[0].response_body)["data"]["status"] == "failed"

    event_count = len(events)
    assert await recovery(interrupted_stack).recover() == ()
    async with interrupted_stack.database.read_session() as session:
        assert len(tuple(await session.scalars(select(DomainEventRow)))) == event_count


@pytest.mark.asyncio
async def test_recovery_uses_last_event_time_when_startup_clock_regresses(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    builder = FakeSnapshotBuilder(failure=asyncio.CancelledError())
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "recovery-clock.sqlite3",
        snapshot_builder=builder,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="interrupted-snapshot"),
                run=command(),
            )
        last_time = value.clock.now()
        value.clock.advance(timedelta(hours=-1))

        recovered = await recovery(value).recover()

        assert len(recovered) == 1
        async with value.database.read_session() as session:
            terminal = await session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.failed"
                )
            )
        assert terminal is not None
        assert terminal.occurred_at == last_time
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_recovery_uses_one_receipt_clock_sample_when_time_advances(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "recovery-advancing-clock.sqlite3",
        generator=FakeGenerator(cancellation=True),
        seed_experience_source_trace=True,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="interrupted-advancing-clock"),
                run=command(),
            )
        clock = AdvancingClock(value.clock.now() + timedelta(minutes=5))
        receipts = value.executor._receipt_store  # noqa: SLF001
        receipts._clock = clock  # noqa: SLF001
        registry = value.executor._repository._event_registry  # noqa: SLF001
        recovered = await InspirationRunRecovery(
            database=value.database,
            receipt_store=receipts,
            repository=InspirationRepository(registry),
            response_codec=InspirationResponseCodec(),
            clock=clock,
        ).recover()
        assert len(recovered) == 1

        async with value.database.read_session() as session:
            terminal = await session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.failed"
                )
            )
            recovery_receipt = await session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.scope == "inspiration.run.recover"
                )
            )
            assert terminal is not None
            assert recovery_receipt is not None
            assert (
                terminal.occurred_at
                == recovery_receipt.created_at
                == recovery_receipt.completed_at
            )
            validator = SourceValidator(registry)
            register_inspiration_source_validator(validator)
            await validator.validate(session)
    finally:
        await value.database.dispose()


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("caller_scope", "system:local"),
        ("scope", "forged.operation"),
        ("request_hash", "f" * 64),
    ],
)
@pytest.mark.asyncio
async def test_recovery_rejects_forged_original_receipt_identity_atomically(
    interrupted_stack: Stack,
    field_name: str,
    forged_value: str,
) -> None:
    with pytest.raises(asyncio.CancelledError):
        await interrupted_stack.executor.execute(
            request=request(key=f"forged-receipt-{field_name}"),
            run=command(),
        )
    async with interrupted_stack.database.transaction(immediate=True) as uow:
        receipt = await uow.session.scalar(select(IdempotencyRecordRow))
        assert receipt is not None
        setattr(receipt, field_name, forged_value)

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="original attached in-progress receipt",
    ):
        await recovery(interrupted_stack).recover()

    async with interrupted_stack.database.read_session() as session:
        events = tuple(
            await session.scalars(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        )
        receipts = tuple(await session.scalars(select(IdempotencyRecordRow)))
        state = await session.scalar(select(InspirationRunStateRow))
    assert tuple(event.event_type for event in events) == (
        "inspiration.started",
        "inspiration.snapshot_frozen",
    )
    assert len(receipts) == 1
    assert receipts[0].state == "in_progress"
    assert getattr(receipts[0], field_name) == forged_value
    assert state is not None and state.status == "running"
