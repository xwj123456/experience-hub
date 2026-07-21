from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config

from experience_hub.clock import FrozenClock
from experience_hub.domain.commands import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CompletedReceipt,
    IdempotencyIntegrityError,
    InProgressReceipt,
    NewReceipt,
    ReceiptRecord,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import IdempotencyRecordRow
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
RECEIPT_IDS = (
    UUID("00000000-0000-0000-0000-000000000101"),
    UUID("00000000-0000-0000-0000-000000000102"),
    UUID("00000000-0000-0000-0000-000000000103"),
)
RESOURCE_ID = UUID("00000000-0000-0000-0000-000000000201")


class RegressingClock:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def now(self) -> datetime:
        return self._values.pop(0)


@pytest.fixture
async def database(repository_root: Path, tmp_path: Path) -> AsyncIterator[Database]:
    database_path = tmp_path / "idempotency.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    database = Database.create(f"sqlite+aiosqlite:///{database_path}")
    try:
        yield database
    finally:
        await database.dispose()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(NOW)


@pytest.fixture
def receipt_store(clock: FrozenClock) -> ReceiptStore:
    return ReceiptStore(clock=clock, id_generator=SequenceIdGenerator(RECEIPT_IDS))


def make_request(
    key: str = "agent-create-1",
    *,
    caller_scope: str = "system:local",
    body: object = {"name": "Alice"},
) -> CommandRequest:
    return CommandRequest(
        caller_scope=caller_scope,
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        path_parameters={},
        query_parameters=(),
        body=body,
        semantic_headers={},
    )


def response(name: str = "Alice") -> StoredResponse:
    return StoredResponse(
        status_code=201,
        body=f'{{"data":{{"name":"{name}"}}}}'.encode(),
        content_type="application/json",
        headers={"location": f"/v1/agents/{name.lower()}"},
    )


@pytest.mark.asyncio
async def test_command_completion_never_precedes_its_reserved_receipt(
    database: Database,
) -> None:
    clock = RegressingClock(NOW, NOW - timedelta(seconds=1))
    store = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator((RECEIPT_IDS[0],)),
    )
    executor = CommandExecutor(
        database=database,
        receipt_store=store,
        clock=clock,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        _ = (uow, context)
        return response()

    result = await executor.execute(make_request(), handler)

    assert result.status_code == 201
    async with database.read_session() as session:
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[0])
    assert receipt is not None
    assert receipt.completed_at == receipt.created_at == NOW


@pytest.mark.asyncio
async def test_reserve_replays_completed_receipt_without_mutating_bytes(
    database: Database,
    receipt_store: ReceiptStore,
    clock: FrozenClock,
) -> None:
    request = make_request()
    stored = response()
    async with database.transaction(immediate=True) as uow:
        decision = await receipt_store.reserve(uow=uow, request=request)
        assert isinstance(decision, NewReceipt)
        await receipt_store.complete(
            uow=uow,
            reservation=decision.reservation,
            response=stored,
            completed_at=clock.now(),
        )

    async with database.transaction(immediate=True) as uow:
        replay = await receipt_store.reserve(uow=uow, request=request)

    assert isinstance(replay, CompletedReceipt)
    assert replay.record.response == stored
    assert replay.record.receipt_id == decision.reservation.receipt_id
    assert replay.record.request_hash == request.request_hash
    assert replay.record.response is not None
    with pytest.raises(TypeError):
        replay.record.response.headers["location"] = "/changed"  # type: ignore[index]


@pytest.mark.asyncio
async def test_same_key_with_different_request_hash_is_a_stable_conflict(
    database: Database,
    receipt_store: ReceiptStore,
) -> None:
    async with database.transaction(immediate=True) as uow:
        await receipt_store.reserve(uow=uow, request=make_request())

    async with database.transaction(immediate=True) as uow:
        with pytest.raises(DomainError) as raised:
            await receipt_store.reserve(
                uow=uow,
                request=make_request(body={"name": "Bob"}),
            )

    error = raised.value
    assert error.code == "idempotency_key_conflict"
    assert error.status_code == 409


@pytest.mark.asyncio
async def test_receipts_are_isolated_by_caller_scope(
    database: Database,
    receipt_store: ReceiptStore,
) -> None:
    async with database.transaction(immediate=True) as uow:
        first = await receipt_store.reserve(uow=uow, request=make_request())
    async with database.transaction(immediate=True) as uow:
        second = await receipt_store.reserve(
            uow=uow,
            request=make_request(
                caller_scope="agent:00000000-0000-0000-0000-000000000001"
            ),
        )

    assert isinstance(first, NewReceipt)
    assert isinstance(second, NewReceipt)
    assert first.reservation.receipt_id != second.reservation.receipt_id


@pytest.mark.asyncio
async def test_visible_in_progress_receipt_includes_attached_resource(
    database: Database,
    receipt_store: ReceiptStore,
) -> None:
    request = make_request()
    async with database.transaction(immediate=True) as uow:
        new = await receipt_store.reserve(uow=uow, request=request)
        assert isinstance(new, NewReceipt)
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=new.reservation.receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )

    async with database.transaction(immediate=True) as uow:
        visible = await receipt_store.reserve(uow=uow, request=request)

    assert isinstance(visible, InProgressReceipt)
    assert visible.record.result_resource_type == "agent"
    assert visible.record.result_resource_id == RESOURCE_ID


@pytest.mark.asyncio
async def test_resource_attachment_is_one_time_and_identical_only(
    database: Database,
    receipt_store: ReceiptStore,
) -> None:
    async with database.transaction(immediate=True) as uow:
        new = await receipt_store.reserve(uow=uow, request=make_request())
        assert isinstance(new, NewReceipt)
        receipt_id = new.reservation.receipt_id
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )
        with pytest.raises(IdempotencyIntegrityError, match="resource"):
            await receipt_store.attach_resource(
                uow=uow,
                receipt_id=receipt_id,
                resource_type="agent",
                resource_id=RECEIPT_IDS[2],
            )


def test_receipt_record_rejects_an_unknown_state() -> None:
    with pytest.raises(ValueError, match="Unknown receipt state"):
        ReceiptRecord(
            receipt_id=RECEIPT_IDS[0],
            caller_scope="system:local",
            operation_scope="agent.create",
            idempotency_key="agent-create-1",
            request_hash="a" * 64,
            state="unknown",  # type: ignore[arg-type]
            result_resource_type=None,
            result_resource_id=None,
            response=None,
            created_at=NOW,
            completed_at=None,
        )


@pytest.mark.asyncio
async def test_completed_receipt_cannot_gain_its_first_resource_but_identical_is_noop(
    database: Database,
    receipt_store: ReceiptStore,
    clock: FrozenClock,
) -> None:
    async with database.transaction(immediate=True) as uow:
        without_resource = await receipt_store.reserve(
            uow=uow, request=make_request("without-resource")
        )
        assert isinstance(without_resource, NewReceipt)
        await receipt_store.complete(
            uow=uow,
            reservation=without_resource.reservation,
            response=response(),
            completed_at=clock.now(),
        )

    async with database.transaction(immediate=True) as uow:
        with pytest.raises(IdempotencyIntegrityError, match="resource"):
            await receipt_store.attach_resource(
                uow=uow,
                receipt_id=without_resource.reservation.receipt_id,
                resource_type="agent",
                resource_id=RESOURCE_ID,
            )

    async with database.transaction(immediate=True) as uow:
        with_resource = await receipt_store.reserve(
            uow=uow, request=make_request("with-resource")
        )
        assert isinstance(with_resource, NewReceipt)
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=with_resource.reservation.receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )
        await receipt_store.complete(
            uow=uow,
            reservation=with_resource.reservation,
            response=response(),
            completed_at=clock.now(),
        )

    async with database.transaction(immediate=True) as uow:
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=with_resource.reservation.receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )


@pytest.mark.asyncio
async def test_receipt_can_be_completed_by_id_in_a_later_unit_of_work(
    database: Database,
    receipt_store: ReceiptStore,
    clock: FrozenClock,
) -> None:
    async with database.transaction(immediate=True) as uow:
        new = await receipt_store.reserve(uow=uow, request=make_request())
        assert isinstance(new, NewReceipt)
        receipt_id = new.reservation.receipt_id
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=receipt_id,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )

    clock.advance(timedelta(seconds=1))
    stored = response()
    async with database.transaction(immediate=True) as uow:
        await receipt_store.complete_existing(
            uow=uow,
            receipt_id=receipt_id,
            response=stored,
            completed_at=clock.now(),
        )
    async with database.read_session() as session:
        record = await receipt_store.get_by_id(session=session, receipt_id=receipt_id)
        found = await receipt_store.find_completed_resource(
            session=session,
            resource_type="agent",
            resource_id=RESOURCE_ID,
        )

    assert record is not None
    assert record.state == "completed"
    assert record.response == stored
    assert found == record


@pytest.mark.asyncio
async def test_completion_is_compare_and_set_and_identical_only(
    database: Database,
    receipt_store: ReceiptStore,
    clock: FrozenClock,
) -> None:
    async with database.transaction(immediate=True) as uow:
        new = await receipt_store.reserve(uow=uow, request=make_request())
        assert isinstance(new, NewReceipt)
        receipt_id = new.reservation.receipt_id
        await receipt_store.complete(
            uow=uow,
            reservation=new.reservation,
            response=response(),
            completed_at=clock.now(),
        )
        await receipt_store.complete_existing(
            uow=uow,
            receipt_id=receipt_id,
            response=response(),
            completed_at=clock.now(),
        )
        with pytest.raises(IdempotencyIntegrityError, match="response"):
            await receipt_store.complete_existing(
                uow=uow,
                receipt_id=receipt_id,
                response=response("Bob"),
                completed_at=clock.now(),
            )
