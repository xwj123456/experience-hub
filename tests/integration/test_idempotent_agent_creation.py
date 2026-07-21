import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select

from experience_hub.agents import AgentCreated, AgentService, CreateAgent
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain.commands import (
    CommandContext,
    CommandRequest,
    ReplayableCommandError,
)
from experience_hub.domain.events import EventRegistry
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandHandler,
    NewReceipt,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
RECEIPT_IDS = [
    UUID("00000000-0000-0000-0000-000000000601"),
    UUID("00000000-0000-0000-0000-000000000602"),
    UUID("00000000-0000-0000-0000-000000000603"),
    UUID("00000000-0000-0000-0000-000000000604"),
]
AGENT_IDS = [
    UUID("00000000-0000-0000-0000-000000000701"),
    UUID("00000000-0000-0000-0000-000000000702"),
]
type Stack = tuple[Database, ReceiptStore, CommandExecutor, AgentService]


@pytest.fixture
async def stack(
    repository_root: Path, tmp_path: Path
) -> AsyncIterator[Stack]:
    database_path = tmp_path / "idempotent-agent.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")
    registry = EventRegistry()
    registry.register(AgentCreated)
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}", event_registry=registry
    )
    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock, id_generator=SequenceIdGenerator(RECEIPT_IDS)
    )
    executor = CommandExecutor(database=database, receipt_store=receipts, clock=clock)
    service = AgentService(
        clock=clock,
        id_generator=SequenceIdGenerator(AGENT_IDS),
        receipt_store=receipts,
    )
    try:
        yield database, receipts, executor, service
    finally:
        await database.dispose()


def request(key: str, name: str = "Alice") -> CommandRequest:
    return CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        path_parameters={},
        query_parameters=(),
        body={"name": name},
        semantic_headers={},
    )


def handler(service: AgentService, name: str = "Alice") -> CommandHandler:
    async def create(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.create(
            uow=uow, command=CreateAgent(name=name), command_context=context
        )

    return create


@pytest.mark.asyncio
async def test_completed_command_replays_exact_response_and_event_once(
    stack: Stack,
) -> None:
    database, _, executor, service = stack
    first = await executor.execute(request("agent-create-1"), handler(service))
    second = await executor.execute(request("agent-create-1"), handler(service))

    assert second.replayed is True
    assert (second.status_code, second.content_type, second.body, second.headers) == (
        first.status_code,
        first.content_type,
        first.body,
        first.headers,
    )
    async with database.read_session() as session:
        assert await session.scalar(select(func.count()).select_from(AgentRow)) == 1
        assert (
            await session.scalar(
                select(func.count()).select_from(DomainEventRow).where(
                    DomainEventRow.event_type == "agent.created"
                )
            )
            == 1
        )


@pytest.mark.asyncio
async def test_replayable_domain_error_rolls_back_savepoint_and_is_replayed(
    stack: Stack,
) -> None:
    database, _, executor, _ = stack

    async def rejected(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        _ = context
        uow.session.add(
            AgentRow(agent_id=AGENT_IDS[0], name="Transient", created_at=NOW)
        )
        await uow.session.flush()
        raise ReplayableCommandError(
            code="agent_rejected", message="Agent rejected", status_code=422
        )

    first = await executor.execute(request("rejected"), rejected)
    second = await executor.execute(request("rejected"), rejected)

    assert first.status_code == 422 and second.body == first.body
    assert second.replayed is True
    assert json.loads(first.body)["error"]["code"] == "agent_rejected"
    async with database.read_session() as session:
        assert await session.get(AgentRow, AGENT_IDS[0]) is None
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[0])
        assert receipt is not None and receipt.state == "completed"


@pytest.mark.asyncio
async def test_unexpected_exception_rolls_back_reservation(stack: Stack) -> None:
    database, _, executor, _ = stack

    async def crashes(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        _ = (uow, context)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await executor.execute(request("crash"), crashes)

    async with database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(IdempotencyRecordRow))
            == 0
        )


@pytest.mark.asyncio
async def test_precommitted_in_progress_returns_resource_without_handler(
    stack: Stack,
) -> None:
    _, receipts, executor, _ = stack
    command_request = request("visible")
    async with executor.database.transaction(immediate=True) as uow:
        decision = await receipts.reserve(uow=uow, request=command_request)
        assert isinstance(decision, NewReceipt)
        await receipts.attach_resource(
            uow=uow,
            receipt_id=decision.reservation.receipt_id,
            resource_type="agent",
            resource_id=AGENT_IDS[0],
        )
    called = False

    async def must_not_run(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        _ = (uow, context)
        nonlocal called
        called = True
        raise AssertionError("handler re-entered")

    result = await executor.execute(command_request, must_not_run)

    assert result.status_code == 409 and called is False
    assert result.content_type == "application/json"
    assert result.headers == {"retry-after": "1"}
    assert result.body == canonical_json_bytes(
        {
            "error": {
                "code": "operation_in_progress",
                "details": {
                    "receipt_id": str(decision.reservation.receipt_id),
                    "resource": {"id": str(AGENT_IDS[0]), "type": "agent"},
                },
                "message": "The operation is still in progress",
            }
        }
    )


@pytest.mark.asyncio
async def test_ordinary_same_key_calls_serialize_and_run_handler_once(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, executor, service = stack
    entered = asyncio.Event()
    second_transaction_attempted = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    transaction_calls = 0
    transaction = database.transaction

    def observed_transaction(
        *,
        immediate: bool = False,
        exclusive: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        nonlocal transaction_calls
        transaction_calls += 1
        if transaction_calls == 2:
            second_transaction_attempted.set()
        return transaction(immediate=immediate, exclusive=exclusive)

    monkeypatch.setattr(database, "transaction", observed_transaction)

    async def slow_handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await service.create(
            uow=uow, command=CreateAgent(name="Alice"), command_context=context
        )

    first_task = asyncio.create_task(
        executor.execute(request("concurrent"), slow_handler)
    )
    await entered.wait()
    second_task = asyncio.create_task(
        executor.execute(request("concurrent"), slow_handler)
    )
    await second_transaction_attempted.wait()
    assert calls == 1 and not second_task.done()
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert calls == 1 and first.replayed is False and second.replayed is True
    async with database.read_session() as session:
        causations = (
            await session.scalars(select(DomainEventRow.causation_id))
        ).all()
    assert causations == [RECEIPT_IDS[0]]


@pytest.mark.asyncio
async def test_handler_cannot_observe_receipt_from_an_independent_read_connection(
    stack: Stack,
) -> None:
    database, _, executor, service = stack
    observed_receipt: IdempotencyRecordRow | None = None

    async def inspect_then_create(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        nonlocal observed_receipt
        async with database.read_session() as independent_session:
            observed_receipt = await independent_session.get(
                IdempotencyRecordRow, context.receipt_id
            )
        return await service.create(
            uow=uow,
            command=CreateAgent(name="Alice"),
            command_context=context,
        )

    result = await executor.execute(request("not-visible"), inspect_then_create)

    assert result.status_code == 201
    assert observed_receipt is None


@pytest.mark.asyncio
async def test_same_name_domain_error_is_completed_and_replayed(
    stack: Stack,
) -> None:
    _, _, executor, service = stack
    await executor.execute(request("first-name"), handler(service))
    failed = await executor.execute(request("duplicate-name"), handler(service))
    replay = await executor.execute(request("duplicate-name"), handler(service))

    assert failed.status_code == 409 and replay.replayed is True
    assert replay.body == failed.body == canonical_json_bytes(
        {
            "error": {
                "code": "agent_name_conflict",
                "details": {"name": "Alice"},
                "message": "An agent with this name already exists",
            }
        }
    )


@pytest.mark.asyncio
async def test_blank_name_error_is_completed_and_replayed_without_side_effects(
    stack: Stack,
) -> None:
    database, _, executor, service = stack
    command_request = request("blank-name", " \t ")
    create_blank = handler(service, " \t ")

    first = await executor.execute(command_request, create_blank)
    replay = await executor.execute(command_request, create_blank)

    expected = canonical_json_bytes(
        {
            "error": {
                "code": "agent_name_required",
                "details": {},
                "message": "Agent name must not be empty",
            }
        }
    )
    assert first.status_code == 422
    assert first.content_type == "application/json"
    assert replay.replayed is True
    assert replay.body == first.body == expected
    assert replay.headers == first.headers == {}
    async with database.read_session() as session:
        assert await session.scalar(select(func.count()).select_from(AgentRow)) == 0
        assert (
            await session.scalar(select(func.count()).select_from(DomainEventRow))
            == 0
        )
