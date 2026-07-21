from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.integration.test_create_experience import (
    Stack,
    build_stack,
    create,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandRequest
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.lifecycle.contracts import decode_lifecycle_result
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.lifecycle.service import LifecycleService
from experience_hub.lifecycle.worker import (
    LifecycleWorker,
    ManualLifecycleTicker,
    ProductionLifecycleTicker,
)
from experience_hub.storage.database import DatabaseBusy
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandHandler,
    CommandResult,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    IdempotencyRecordRow,
)


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "lifecycle-worker.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def lifecycle_service(stack: Stack) -> LifecycleService:
    return LifecycleService(
        clock=stack.clock,
        receipt_store=stack.receipts,
        repository=LifecycleRepository(),
        mutation_writer=ExperienceMutationWriter(repository=stack.repository),
    )


class RecordingExecutor:
    def __init__(self, delegate: CommandExecutor) -> None:
        self._delegate = delegate
        self.requests: list[CommandRequest] = []
        self.results: list[CommandResult] = []

    async def execute(
        self,
        request: CommandRequest,
        handler: CommandHandler,
    ) -> CommandResult:
        self.requests.append(request)
        result = await self._delegate.execute(request, handler)
        self.results.append(result)
        return result


class DatabaseBusyOnceExecutor(RecordingExecutor):
    def __init__(self, delegate: CommandExecutor) -> None:
        super().__init__(delegate)
        self._busy = True

    async def execute(
        self,
        request: CommandRequest,
        handler: CommandHandler,
    ) -> CommandResult:
        self.requests.append(request)
        if self._busy:
            self._busy = False
            raise DatabaseBusy
        result = await self._delegate.execute(request, handler)
        self.results.append(result)
        return result


class LifecycleBusyOnceExecutor(RecordingExecutor):
    def __init__(self, delegate: CommandExecutor) -> None:
        super().__init__(delegate)
        self._busy = True

    async def execute(
        self,
        request: CommandRequest,
        handler: CommandHandler,
    ) -> CommandResult:
        self.requests.append(request)
        if self._busy:
            self._busy = False
            result = CommandResult(
                status_code=409,
                body=canonical_json_bytes(
                    {
                        "error": {
                            "code": "lifecycle_in_progress",
                            "details": {"private": "not retained"},
                            "message": "not retained",
                        }
                    }
                ),
                content_type="application/json",
                headers={},
                replayed=False,
            )
        else:
            result = await self._delegate.execute(request, handler)
        self.results.append(result)
        return result


class CancelledExecutor:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def execute(
        self,
        request: CommandRequest,
        handler: CommandHandler,
    ) -> CommandResult:
        _ = request, handler
        self.entered.set()
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_one_tick_uses_system_command_and_runs_background_cycle(
    stack: Stack,
) -> None:
    await create(stack, key="worker-source")
    service = lifecycle_service(stack)
    executor = RecordingExecutor(stack.executor)
    ticker = ManualLifecycleTicker()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=executor,
        service=service,
    )

    worker.start()
    try:
        await ticker.tick()
    finally:
        await worker.stop()

    expected_cycle_id = service.cycle_id(stack.clock.now())
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.caller_scope == "system:local"
    assert request.operation_scope == "lifecycle.run"
    assert request.idempotency_key == f"lifecycle:{expected_cycle_id}"
    assert request.method == "POST"
    assert request.route_template == "/v1/lifecycle:run"
    assert dict(request.body) == {
        "evaluated_at": stack.clock.now(),
        "mode": "background",
    }
    assert len(executor.results) == 1
    assert executor.results[0].status_code == 200
    assert decode_lifecycle_result(
        executor.results[0].body
    ).cycle_id == expected_cycle_id
    assert worker.failures == ()

    async with stack.database.read_session() as session:
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.idempotency_key
                == f"lifecycle:{expected_cycle_id}"
            )
        )
        lifecycle_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(
                DomainEventRow.event_type
                == "experience.lifecycle_evaluated"
            )
        )
    assert receipt is not None
    assert receipt.caller_scope == "system:local"
    assert receipt.scope == "lifecycle.run"
    assert lifecycle_event_count == 1


@pytest.mark.asyncio
async def test_repeated_tick_for_same_cycle_replays_same_receipt(
    stack: Stack,
) -> None:
    await create(stack, key="worker-replay-source")
    service = lifecycle_service(stack)
    executor = RecordingExecutor(stack.executor)
    ticker = ManualLifecycleTicker()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=executor,
        service=service,
    )

    worker.start()
    try:
        await ticker.tick()
        await ticker.tick()
    finally:
        await worker.stop()

    assert [result.replayed for result in executor.results] == [False, True]
    assert executor.results[1].body == executor.results[0].body
    assert worker.failures == ()
    async with stack.database.read_session() as session:
        assert await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.scope == "lifecycle.run"
            )
        ) == 1
        assert await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(
                DomainEventRow.event_type
                == "experience.lifecycle_evaluated"
            )
        ) == 1


@pytest.mark.asyncio
async def test_database_busy_is_sanitized_and_next_tick_continues(
    stack: Stack,
) -> None:
    service = lifecycle_service(stack)
    executor = DatabaseBusyOnceExecutor(stack.executor)
    ticker = ManualLifecycleTicker()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=executor,
        service=service,
    )

    worker.start()
    try:
        await ticker.tick()
        assert len(worker.failures) == 1
        failure = worker.failures[0]
        assert failure.code == "database_busy"
        assert failure.status_code == 503
        assert failure.cycle_id == service.cycle_id(stack.clock.now())
        assert failure.evaluated_at == stack.clock.now()
        assert not hasattr(failure, "message")
        with pytest.raises(FrozenInstanceError):
            setattr(failure, "code", "leaked_detail")  # noqa: B010

        await ticker.tick()
    finally:
        await worker.stop()

    assert len(executor.requests) == 2
    assert len(executor.results) == 1
    assert executor.results[0].status_code == 200
    assert len(worker.failures) == 1


@pytest.mark.asyncio
async def test_lifecycle_busy_response_is_sanitized_and_loop_continues(
    stack: Stack,
) -> None:
    service = lifecycle_service(stack)
    executor = LifecycleBusyOnceExecutor(stack.executor)
    ticker = ManualLifecycleTicker()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=executor,
        service=service,
    )

    worker.start()
    try:
        await ticker.tick()
        assert len(worker.failures) == 1
        failure = worker.failures[0]
        assert failure.code == "lifecycle_in_progress"
        assert failure.status_code == 409
        assert not hasattr(failure, "details")
        assert not hasattr(failure, "message")

        await ticker.tick()
    finally:
        await worker.stop()

    assert [result.status_code for result in executor.results] == [409, 200]
    assert len(worker.failures) == 1


@pytest.mark.asyncio
async def test_stop_wakes_idle_worker_and_is_idempotent(
    stack: Stack,
) -> None:
    ticker = ManualLifecycleTicker()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=stack.executor,
        service=lifecycle_service(stack),
    )

    worker.start()
    assert worker.running

    await worker.stop()
    await worker.stop()

    assert not worker.running
    assert ticker.closed
    with pytest.raises(RuntimeError, match="closed"):
        await ticker.tick()


@pytest.mark.asyncio
async def test_production_ticker_close_interrupts_wait() -> None:
    ticker = ProductionLifecycleTicker(timedelta(hours=1))
    waiting = asyncio.create_task(ticker.wait_for_tick())

    ticker.close()

    assert not await waiting
    assert ticker.closed
    assert not await ticker.wait_for_tick()
    with pytest.raises(ValueError, match="positive"):
        ProductionLifecycleTicker(timedelta(0))


@pytest.mark.asyncio
async def test_repeated_start_is_rejected(
    stack: Stack,
) -> None:
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ManualLifecycleTicker(),
        executor=stack.executor,
        service=lifecycle_service(stack),
    )

    worker.start()
    try:
        with pytest.raises(RuntimeError, match="already been started"):
            worker.start()
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_cancelled_execution_propagates_without_failure_record(
    stack: Stack,
) -> None:
    ticker = ManualLifecycleTicker()
    executor = CancelledExecutor()
    worker = LifecycleWorker(
        clock=stack.clock,
        ticker=ticker,
        executor=executor,
        service=lifecycle_service(stack),
    )
    worker.start()
    tick = asyncio.create_task(ticker.tick())
    await executor.entered.wait()

    try:
        with pytest.raises(asyncio.CancelledError):
            await worker.stop()
    finally:
        tick.cancel()
        with suppress(asyncio.CancelledError):
            await tick

    assert worker.failures == ()
