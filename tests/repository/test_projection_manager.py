from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain.commands import CommandContext
from experience_hub.domain.events import (
    EventPayload,
    EventRegistry,
    PendingEvent,
    StoredEvent,
)
from experience_hub.storage.database import Database
from experience_hub.storage.projections import (
    EventHeadChanged,
    MaintenanceBlockedByInflight,
    ProjectionManager,
    ProjectionMismatch,
    ProjectionRegistry,
    ReducerVersionMismatch,
    SourceValidatorRequired,
    VerificationReport,
)
from experience_hub.storage.tables import IdempotencyRecordRow, ProjectionVersionRow
from experience_hub.storage.validation import SourceValidator

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)


class Relevant(EventPayload):
    event_type: ClassVar[str] = "example.relevant"
    value: str


class Irrelevant(EventPayload):
    event_type: ClassVar[str] = "example.irrelevant"
    value: str


class PrivateReducer:
    name = "example_projection"
    version = 1
    event_types = frozenset({Relevant.event_type})
    applications: list[int]

    def __init__(self) -> None:
        self.applications = []

    async def apply(
        self, session: AsyncSession, event: StoredEvent
    ) -> None:
        payload = event.payload
        assert isinstance(payload, Relevant)
        self.applications.append(event.event_id)
        await session.execute(
            text(
                'INSERT INTO main."example_projection" '
                "(projection_id, value, source_event_id) "
                "VALUES (:id, :value, :event_id)"
            ),
            {
                "id": str(event.aggregate_id),
                "value": payload.value,
                "event_id": event.event_id,
            },
        )

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = f"{target_prefix}{self.name}"
        await session.execute(
            text(
                f'CREATE TEMP TABLE "{target}" ('
                "projection_id TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "source_event_id INTEGER NOT NULL UNIQUE CHECK(source_event_id > 0))"
            )
        )
        await session.execute(
            text(
                f'INSERT INTO temp."{target}" '
                "(projection_id, value, source_event_id) "
                "SELECT aggregate_id, "
                "json_extract(CAST(payload AS TEXT), '$.value'), event_id "
                "FROM main.domain_events WHERE event_type = :event_type "
                "ORDER BY event_id"
            ),
            {"event_type": Relevant.event_type},
        )


@pytest.fixture
async def database(
    repository_root: Path, tmp_path: Path
) -> AsyncIterator[tuple[Database, ProjectionManager]]:
    path = tmp_path / "projections.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")
    event_registry = EventRegistry()
    event_registry.register(Relevant)
    event_registry.register(Irrelevant)
    registry = ProjectionRegistry()
    registry.register(PrivateReducer())
    manager = ProjectionManager(
        registry,
        source_validator=SourceValidator(event_registry),
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{path}",
        event_registry=event_registry,
        projection_applier=manager,
    )
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "CREATE TABLE example_projection ("
                "projection_id TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "source_event_id INTEGER NOT NULL)"
            )
        )
    try:
        yield database, manager
    finally:
        await database.dispose()


def stored(event_id: int, payload: EventPayload) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        aggregate_type="example",
        aggregate_id=uuid4(),
        sequence=1,
        event_type=type(payload).event_type,
        payload=payload,
        actor_agent_id=None,
        causation_id=uuid4(),
        occurred_at=NOW,
    )


def test_registry_rejects_duplicate_names_and_nonpositive_versions() -> None:
    registry = ProjectionRegistry()
    registry.register(PrivateReducer())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(PrivateReducer())
    invalid = PrivateReducer()
    invalid.version = 0
    with pytest.raises(ValueError, match="positive"):
        ProjectionRegistry([invalid])


def test_nonempty_projection_registry_requires_source_validator() -> None:
    registry = ProjectionRegistry([PrivateReducer()])

    with pytest.raises(SourceValidatorRequired):
        ProjectionManager(registry)


@pytest.mark.asyncio
async def test_apply_filters_events_and_advances_only_relevant_checkpoint(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    relevant = stored(1, Relevant(schema_version=1, value="first"))
    irrelevant = stored(2, Irrelevant(schema_version=1, value="ignored"))

    async with db.transaction() as uow:
        await manager.apply(session=uow.session, events=[irrelevant, relevant])

    async with db.read_session() as session:
        values = list(
            (
                await session.execute(
                    text(
                        "SELECT value FROM example_projection "
                        "ORDER BY source_event_id"
                    )
                )
            ).scalars()
        )
        version = await session.get(ProjectionVersionRow, "example_projection")

    assert values == ["first"]
    assert version is not None
    assert version.last_applied_event_id == 1
    assert version.reducer_version == 1


@pytest.mark.asyncio
async def test_database_default_factory_uses_noop_projection_manager(
    repository_root: Path, tmp_path: Path
) -> None:
    path = tmp_path / "default-manager.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")
    database = Database.create(f"sqlite+aiosqlite:///{path}")
    try:
        async with database.transaction() as uow:
            assert isinstance(uow.projection_applier, ProjectionManager)
            await uow.projection_applier.apply(session=uow.session, events=[])
    finally:
        await database.dispose()


async def append_relevant(*, database: Database, value: str = "golden") -> int:
    receipt_id = uuid4()
    aggregate_id = uuid4()
    async with database.transaction(immediate=True) as uow:
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=receipt_id,
                caller_scope="system:local",
                scope="example.record",
                idempotency_key=f"record-{receipt_id}",
                request_hash="a" * 64,
                state="completed",
                response_status_code=201,
                response_body=b"{}",
                response_content_type="application/json",
                response_headers=b"{}",
                created_at=NOW,
                completed_at=NOW,
            )
        )
        await uow.session.flush()
        result = await uow.append_events(
            CommandContext(
                receipt_id=receipt_id,
                caller_scope="system:local",
                operation_scope="example.record",
                idempotency_key=f"record-{receipt_id}",
                request_hash="a" * 64,
            ),
            [
                PendingEvent(
                    aggregate_type="example",
                    aggregate_id=aggregate_id,
                    event_type=Relevant.event_type,
                    payload=Relevant(schema_version=1, value=value),
                    actor_agent_id=None,
                    occurred_at=NOW,
                )
            ],
        )
    return result[0].event_id


@pytest.mark.asyncio
async def test_apply_orders_batch_and_advances_to_highest_checkpoint(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    reducer = manager.registry.reducers[0]
    assert isinstance(reducer, PrivateReducer)
    reducer.applications.clear()
    first = stored(7, Relevant(schema_version=1, value="first"))
    second = stored(8, Relevant(schema_version=1, value="second"))

    async with db.transaction() as uow:
        await manager.apply(session=uow.session, events=[second, first])

    async with db.read_session() as session:
        version = await session.get(ProjectionVersionRow, reducer.name)

    assert reducer.applications == [7, 8]
    assert version is not None and version.last_applied_event_id == 8


@pytest.mark.asyncio
async def test_apply_skips_batch_duplicates_and_already_checkpointed_events(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    reducer = manager.registry.reducers[0]
    assert isinstance(reducer, PrivateReducer)
    first = stored(7, Relevant(schema_version=1, value="first"))
    second = stored(8, Relevant(schema_version=1, value="second"))

    async with db.transaction() as uow:
        await manager.apply(session=uow.session, events=[second, first, first])
    reducer.applications.clear()
    async with db.transaction() as uow:
        await manager.apply(session=uow.session, events=[second, first])

    assert reducer.applications == []


@pytest.mark.asyncio
async def test_verify_and_repair_reject_runtime_source_validator_bypass(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    manager._source_validator = None

    with pytest.raises(SourceValidatorRequired):
        await manager.verify(db)
    with pytest.raises(SourceValidatorRequired):
        await manager.repair(db)


def test_projection_maintenance_errors_have_stable_codes() -> None:
    report = VerificationReport(event_head=0, differences=())

    assert ProjectionMismatch(report).code == "projection_mismatch"
    assert ReducerVersionMismatch().code == "reducer_version_mismatch"
    assert EventHeadChanged().code == "event_head_changed"
    assert (
        MaintenanceBlockedByInflight().code
        == "maintenance_blocked_by_inflight"
    )
    assert SourceValidatorRequired().code == "source_validator_required"


@pytest.mark.asyncio
async def test_verify_reports_mismatch_without_changing_online_rows(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'corrupt'")
        )

    with pytest.raises(ProjectionMismatch) as caught:
        await manager.verify(db)

    assert caught.value.report.matches is False
    assert caught.value.report.differences[0].projection == "example_projection"
    assert len(caught.value.report.differences[0].differing_keys) == 1
    async with db.read_session() as session:
        assert (
            await session.scalar(text("SELECT value FROM example_projection"))
            == "corrupt"
        )
        temp_count = await session.scalar(
            text(
                "SELECT count(*) FROM sqlite_temp_master "
                "WHERE name LIKE '_rebuild_%'"
            )
        )
    assert temp_count == 0


@pytest.mark.asyncio
async def test_verify_matches_and_cleans_temp_tables(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    head = await append_relevant(database=db)

    report = await manager.verify(db)

    assert report.matches and report.event_head == head
    async with db.read_session() as session:
        assert await session.scalar(
            text(
                "SELECT count(*) FROM sqlite_temp_master "
                "WHERE name LIKE '_rebuild_%'"
            )
        ) == 0


@pytest.mark.asyncio
async def test_verify_rechecks_event_head_and_cleans_temp_tables(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    manager._event_head = AsyncMock(side_effect=[1, 2])  # type: ignore[method-assign]

    with pytest.raises(EventHeadChanged):
        await manager.verify(db)

    async with db.read_session() as session:
        assert await session.scalar(
            text(
                "SELECT count(*) FROM sqlite_temp_master "
                "WHERE name LIKE '_rebuild_%'"
            )
        ) == 0


@pytest.mark.asyncio
async def test_verify_rejects_online_reducer_version_mismatch(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE projection_versions SET reducer_version = 2 "
                "WHERE name = 'example_projection'"
            )
        )

    with pytest.raises(ReducerVersionMismatch):
        await manager.verify(db)


@pytest.mark.asyncio
async def test_repair_swaps_rebuilt_rows_and_updates_version_and_head(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    head = await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'corrupt'")
        )

    report = await manager.repair(db)

    assert report.matches
    assert report.event_head == head
    async with db.read_session() as session:
        assert (
            await session.scalar(text("SELECT value FROM example_projection"))
            == "golden"
        )
        version = await session.get(ProjectionVersionRow, "example_projection")
    assert version is not None
    assert (version.reducer_version, version.last_applied_event_id) == (1, head)
    assert version.last_verified_hash is not None


@pytest.mark.asyncio
async def test_repair_rolls_back_swap_when_post_swap_hash_differs(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'before-repair'")
        )
        await uow.session.execute(
            text(
                "CREATE TRIGGER corrupt_repair AFTER INSERT ON example_projection "
                "BEGIN UPDATE example_projection SET value = value || '!'; END"
            )
        )
        await uow.session.execute(
            text(
                "UPDATE projection_versions SET reducer_version = 7, "
                "last_applied_event_id = 99, last_verified_hash = :hash, "
                "last_verified_at = :verified_at "
                "WHERE name = 'example_projection'"
            ),
            {"hash": "c" * 64, "verified_at": NOW.isoformat()},
        )

    with pytest.raises(ProjectionMismatch):
        await manager.repair(db)

    async with db.read_session() as session:
        assert (
            await session.scalar(text("SELECT value FROM example_projection"))
            == "before-repair"
        )
        version = await session.get(ProjectionVersionRow, "example_projection")
        temp_count = await session.scalar(
            text(
                "SELECT count(*) FROM sqlite_temp_master "
                "WHERE name LIKE '_rebuild_%'"
            )
        )
    assert version is not None
    assert (
        version.reducer_version,
        version.last_applied_event_id,
        version.last_verified_hash,
    ) == (7, 99, "c" * 64)
    assert temp_count == 0


@pytest.mark.asyncio
async def test_repair_refuses_in_progress_receipt_before_rebuild(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    async with db.transaction() as uow:
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=uuid4(),
                caller_scope="system:local",
                scope="example.record",
                idempotency_key="still-running",
                request_hash="b" * 64,
                state="in_progress",
                created_at=NOW,
            )
        )

    with pytest.raises(MaintenanceBlockedByInflight):
        await manager.repair(db)


@pytest.mark.asyncio
async def test_repair_validates_sources_before_rebuild_and_swap(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'before-repair'")
        )

    class InvalidSources:
        async def validate(self, session: AsyncSession) -> None:
            _ = session
            raise RuntimeError("source invalid")

    manager._source_validator = InvalidSources()
    with pytest.raises(RuntimeError, match="source invalid"):
        await manager.repair(db)

    async with db.read_session() as session:
        assert (
            await session.scalar(text("SELECT value FROM example_projection"))
            == "before-repair"
        )


@pytest.mark.asyncio
async def test_repair_rechecks_event_head_before_swap(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    await append_relevant(database=db)
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'before-repair'")
        )
    manager._event_head = AsyncMock(side_effect=[1, 2])  # type: ignore[method-assign]

    with pytest.raises(EventHeadChanged):
        await manager.repair(db)

    async with db.read_session() as session:
        assert (
            await session.scalar(text("SELECT value FROM example_projection"))
            == "before-repair"
        )


@pytest.mark.asyncio
async def test_verify_limits_differing_keys_to_fifty(
    database: tuple[Database, ProjectionManager],
) -> None:
    db, manager = database
    for index in range(51):
        await append_relevant(database=db, value=f"value-{index:02d}")
    async with db.transaction() as uow:
        await uow.session.execute(
            text("UPDATE example_projection SET value = 'corrupt'")
        )

    with pytest.raises(ProjectionMismatch) as caught:
        await manager.verify(db)

    assert len(caught.value.report.differences[0].differing_keys) == 50
