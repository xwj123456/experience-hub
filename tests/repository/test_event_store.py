from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, cast
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain.commands import CommandContext
from experience_hub.domain.events import (
    EventPayload,
    EventRegistry,
    PendingEvent,
    StoredEvent,
)
from experience_hub.storage.database import Database
from experience_hub.storage.tables import DomainEventRow, IdempotencyRecordRow

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)


class ExampleRecorded(EventPayload):
    event_type: ClassVar[str] = "example.recorded"
    value: str


class OtherRecorded(EventPayload):
    event_type: ClassVar[str] = "other.recorded"
    value: str


class UnregisteredRecorded(EventPayload):
    event_type: ClassVar[str] = "unregistered.recorded"
    value: str


class RecordingProjectionApplier:
    def __init__(self) -> None:
        self.calls: list[tuple[AsyncSession, list[StoredEvent]]] = []

    async def apply(
        self,
        *,
        session: AsyncSession,
        events: Sequence[StoredEvent],
    ) -> None:
        self.calls.append((session, list(events)))


class FailingProjectionApplier:
    async def apply(
        self,
        *,
        session: AsyncSession,
        events: Sequence[StoredEvent],
    ) -> None:
        _ = (session, events)
        raise RuntimeError("projection failed")


@pytest.fixture
def event_registry() -> EventRegistry:
    registry = EventRegistry()
    registry.register(ExampleRecorded)
    registry.register(OtherRecorded)
    return registry


@pytest.fixture
async def database(
    repository_root: Path,
    tmp_path: Path,
    event_registry: EventRegistry,
) -> AsyncIterator[tuple[Database, RecordingProjectionApplier]]:
    database_path = tmp_path / "events.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")

    projection_applier = RecordingProjectionApplier()
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=event_registry,
        projection_applier=projection_applier,
    )
    try:
        yield database, projection_applier
    finally:
        await database.dispose()


@pytest.fixture
async def failing_projection_database(
    repository_root: Path,
    tmp_path: Path,
    event_registry: EventRegistry,
) -> AsyncIterator[Database]:
    database_path = tmp_path / "failing-projection.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")

    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=event_registry,
        projection_applier=FailingProjectionApplier(),
    )
    try:
        yield database
    finally:
        await database.dispose()


def command_context(receipt_id: UUID) -> CommandContext:
    return CommandContext(
        receipt_id=receipt_id,
        caller_scope="system:local",
        operation_scope="example.record",
        idempotency_key=f"record-{receipt_id}",
        request_hash="a" * 64,
    )


def pending_event(
    aggregate_id: UUID,
    value: str,
    *,
    event_type: str = ExampleRecorded.event_type,
    payload: EventPayload | None = None,
) -> PendingEvent:
    return PendingEvent(
        aggregate_type="example",
        aggregate_id=aggregate_id,
        event_type=event_type,
        payload=payload or ExampleRecorded(schema_version=1, value=value),
        actor_agent_id=None,
        occurred_at=NOW,
    )


async def seed_receipt(database: Database, receipt_id: UUID) -> None:
    async with database.transaction(immediate=True) as uow:
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=receipt_id,
                caller_scope="system:local",
                scope="example.record",
                idempotency_key=f"record-{receipt_id}",
                request_hash="a" * 64,
                state="in_progress",
                result_resource_type=None,
                result_resource_id=None,
                response_status_code=None,
                response_body=None,
                response_content_type=None,
                response_headers=None,
                created_at=NOW,
                completed_at=None,
            )
        )


@pytest.mark.asyncio
async def test_append_allocates_ordered_sequences_and_canonical_payloads(
    database: tuple[Database, RecordingProjectionApplier],
) -> None:
    db, projection_applier = database
    receipt_id = uuid4()
    aggregate_id = uuid4()
    await seed_receipt(db, receipt_id)

    async with db.transaction(immediate=True) as uow:
        first = await uow.append_events(
            command_context(receipt_id),
            [pending_event(aggregate_id, "first"), pending_event(aggregate_id, "two")],
        )
        second = await uow.append_events(
            command_context(receipt_id),
            [pending_event(aggregate_id, "third")],
        )

    stored = [*first, *second]
    assert [event.sequence for event in stored] == [1, 2, 3]
    assert [cast(ExampleRecorded, event.payload).value for event in stored] == [
        "first",
        "two",
        "third",
    ]
    assert all(type(event.payload) is ExampleRecorded for event in stored)
    assert [event.causation_id for event in stored] == [receipt_id] * 3
    assert [event.event_id for event in stored] == sorted(
        event.event_id for event in stored
    )

    async with db.read_session() as session:
        rows = (
            await session.execute(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).scalars()
        assert [row.payload for row in rows] == [
            b'{"schema_version":1,"value":"first"}',
            b'{"schema_version":1,"value":"two"}',
            b'{"schema_version":1,"value":"third"}',
        ]

    applied_sequences = [
        event.sequence for _, call in projection_applier.calls for event in call
    ]
    assert applied_sequences == [
        1,
        2,
        3,
    ]


@pytest.mark.asyncio
async def test_append_preserves_declared_order_across_aggregates(
    database: tuple[Database, RecordingProjectionApplier],
) -> None:
    db, _ = database
    receipt_id = uuid4()
    first_aggregate = uuid4()
    second_aggregate = uuid4()
    await seed_receipt(db, receipt_id)

    async with db.transaction(immediate=True) as uow:
        stored = await uow.append_events(
            command_context(receipt_id),
            [
                pending_event(first_aggregate, "a-1"),
                pending_event(second_aggregate, "b-1"),
                pending_event(first_aggregate, "a-2"),
            ],
        )

    assert [(event.aggregate_id, event.sequence) for event in stored] == [
        (first_aggregate, 1),
        (second_aggregate, 1),
        (first_aggregate, 2),
    ]
    assert [cast(ExampleRecorded, event.payload).value for event in stored] == [
        "a-1",
        "b-1",
        "a-2",
    ]


@pytest.mark.parametrize("mode", [{}, {"exclusive": True}])
@pytest.mark.asyncio
async def test_append_requires_an_immediate_unit_of_work(
    database: tuple[Database, RecordingProjectionApplier],
    mode: dict[str, bool],
) -> None:
    db, _ = database
    receipt_id = uuid4()
    await seed_receipt(db, receipt_id)

    with pytest.raises(RuntimeError, match="immediate transaction"):
        async with db.transaction(**mode) as uow:
            await uow.append_events(
                command_context(receipt_id),
                [pending_event(uuid4(), "rejected")],
            )


@pytest.mark.asyncio
async def test_append_rejects_missing_causation_receipt(
    database: tuple[Database, RecordingProjectionApplier],
) -> None:
    db, _ = database

    with pytest.raises(ValueError, match="causation receipt does not exist"):
        async with db.transaction(immediate=True) as uow:
            await uow.append_events(
                command_context(uuid4()),
                [pending_event(uuid4(), "orphaned")],
            )


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (
            OtherRecorded.event_type,
            ExampleRecorded(schema_version=1, value="mismatch"),
        ),
        (
            UnregisteredRecorded.event_type,
            UnregisteredRecorded(schema_version=1, value="unknown"),
        ),
    ],
    ids=["pending-payload-type-mismatch", "unregistered-pending-event"],
)
@pytest.mark.asyncio
async def test_append_rejects_invalid_pending_event_type(
    database: tuple[Database, RecordingProjectionApplier],
    event_type: str,
    payload: EventPayload,
) -> None:
    db, _ = database
    receipt_id = uuid4()
    await seed_receipt(db, receipt_id)

    with pytest.raises(ValueError, match="event type"):
        async with db.transaction(immediate=True) as uow:
            await uow.append_events(
                command_context(receipt_id),
                [
                    pending_event(
                        uuid4(),
                        "unused",
                        event_type=event_type,
                        payload=payload,
                    )
                ],
            )


@pytest.mark.asyncio
async def test_unique_sequence_rejects_an_injected_append_conflict(
    database: tuple[Database, RecordingProjectionApplier],
) -> None:
    db, _ = database
    receipt_id = uuid4()
    aggregate_id = uuid4()
    await seed_receipt(db, receipt_id)

    async with db.transaction(immediate=True) as uow:
        await uow.session.execute(
            text(
                "CREATE TRIGGER inject_sequence_conflict "
                "BEFORE INSERT ON domain_events "
                "WHEN NEW.event_type = 'example.recorded' "
                "BEGIN "
                "INSERT INTO domain_events "
                "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                "actor_agent_id, causation_id, occurred_at) VALUES "
                "(NEW.aggregate_type, NEW.aggregate_id, NEW.sequence, "
                "'other.recorded', NEW.payload, NEW.actor_agent_id, "
                "NEW.causation_id, NEW.occurred_at); "
                "END"
            )
        )

    with pytest.raises(IntegrityError):
        async with db.transaction(immediate=True) as uow:
            await uow.append_events(
                command_context(receipt_id),
                [pending_event(aggregate_id, "conflict")],
            )

    async with db.read_session() as session:
        count = await session.scalar(
            select(text("count(*)")).select_from(DomainEventRow)
        )
        assert count == 0


@pytest.mark.asyncio
async def test_domain_events_are_immutable_after_insert(
    database: tuple[Database, RecordingProjectionApplier],
) -> None:
    db, _ = database
    receipt_id = uuid4()
    await seed_receipt(db, receipt_id)
    async with db.transaction(immediate=True) as uow:
        [stored] = await uow.append_events(
            command_context(receipt_id),
            [pending_event(uuid4(), "immutable")],
        )

    for statement in (
        "UPDATE domain_events SET event_type = 'other.recorded' "
        "WHERE event_id = :event_id",
        "DELETE FROM domain_events WHERE event_id = :event_id",
    ):
        with pytest.raises(IntegrityError, match="domain_events rows are immutable"):
            async with db.transaction(immediate=True) as uow:
                await uow.session.execute(
                    text(statement),
                    {"event_id": stored.event_id},
                )


@pytest.mark.parametrize(
    "conflict",
    ["event-id", "aggregate-sequence"],
)
@pytest.mark.asyncio
async def test_insert_or_replace_cannot_replace_an_existing_event(
    database: tuple[Database, RecordingProjectionApplier],
    conflict: str,
) -> None:
    db, _ = database
    receipt_id = uuid4()
    aggregate_id = uuid4()
    await seed_receipt(db, receipt_id)
    async with db.transaction(immediate=True) as uow:
        [stored] = await uow.append_events(
            command_context(receipt_id),
            [pending_event(aggregate_id, "original")],
        )

    replacement_event_id = (
        stored.event_id if conflict == "event-id" else stored.event_id + 100
    )
    replacement_aggregate_id = (
        uuid4() if conflict == "event-id" else aggregate_id
    )
    with pytest.raises(
        IntegrityError,
        match="domain_events identity or sequence already exists",
    ):
        async with db.transaction(immediate=True) as uow:
            await uow.session.execute(
                text(
                    "INSERT OR REPLACE INTO domain_events "
                    "(event_id, aggregate_type, aggregate_id, sequence, "
                    "event_type, payload, actor_agent_id, causation_id, "
                    "occurred_at) VALUES "
                    "(:event_id, 'example', :aggregate_id, 1, "
                    "'other.recorded', :payload, NULL, :causation_id, "
                    ":occurred_at)"
                ),
                {
                    "event_id": replacement_event_id,
                    "aggregate_id": str(replacement_aggregate_id),
                    "payload": b'{"schema_version":1,"value":"replacement"}',
                    "causation_id": str(receipt_id),
                    "occurred_at": NOW.isoformat(timespec="microseconds").replace(
                        "+00:00",
                        "Z",
                    ),
                },
            )

    async with db.read_session() as session:
        rows = (
            await session.execute(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).scalars()
        retained = list(rows)

    assert len(retained) == 1
    assert retained[0].event_id == stored.event_id
    assert retained[0].aggregate_id == aggregate_id
    assert retained[0].event_type == ExampleRecorded.event_type
    assert retained[0].payload == b'{"schema_version":1,"value":"original"}'


@pytest.mark.asyncio
async def test_projection_failure_rolls_back_the_appended_events(
    failing_projection_database: Database,
) -> None:
    receipt_id = uuid4()
    await seed_receipt(failing_projection_database, receipt_id)

    with pytest.raises(RuntimeError, match="projection failed"):
        async with failing_projection_database.transaction(immediate=True) as uow:
            await uow.append_events(
                command_context(receipt_id),
                [pending_event(uuid4(), "rolled-back")],
            )

    async with failing_projection_database.read_session() as session:
        count = await session.scalar(
            select(text("count(*)")).select_from(DomainEventRow)
        )
        assert count == 0


def test_event_payload_requires_explicit_supported_schema_version() -> None:
    with pytest.raises(ValidationError):
        ExampleRecorded(value="missing")  # type: ignore[call-arg]
