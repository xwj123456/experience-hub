import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.agents import AgentCreated
from experience_hub.domain.commands import CommandContext
from experience_hub.domain.events import EventRegistry, PendingEvent, StoredEvent
from experience_hub.storage.database import Database
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AgentRow,
    IdempotencyRecordRow,
    ProjectionVersionRow,
)
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_agent_source_validator,
)

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000a01")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000b01")
PROJECTION_NAME = "guarded_agent_projection"


class GuardedAgentProjection:
    def __init__(self) -> None:
        self.name = PROJECTION_NAME
        self.version = 1
        self.event_types = frozenset({AgentCreated.event_type})
        self.rebuild_calls = 0

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        payload = event.payload
        assert isinstance(payload, AgentCreated)
        await session.execute(
            text(
                "INSERT INTO guarded_agent_projection(agent_id, name) "
                "VALUES (:agent_id, :name)"
            ),
            {"agent_id": str(payload.agent_id), "name": payload.name},
        )

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        self.rebuild_calls += 1
        target = f"{target_prefix}{self.name}"
        await session.execute(
            text(
                f'CREATE TEMP TABLE "{target}" ('
                "agent_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
            )
        )


async def seeded_database(
    repository_root: Path,
    path: Path,
) -> AsyncGenerator[
    tuple[Database, ProjectionManager, GuardedAgentProjection]
]:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    source_validator = SourceValidator(registry)
    register_agent_source_validator(source_validator)
    reducer = GuardedAgentProjection()
    manager = ProjectionManager(
        ProjectionRegistry([reducer]),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{path}",
        event_registry=registry,
        projection_applier=manager,
    )
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "CREATE TABLE guarded_agent_projection("
                "agent_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
            )
        )
    async with database.transaction(immediate=True) as uow:
        uow.session.add(AgentRow(agent_id=AGENT_ID, name="Alice", created_at=NOW))
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=RECEIPT_ID,
                caller_scope="system:local",
                scope="agent.create",
                idempotency_key="repair-safety",
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
        await uow.append_events(
            CommandContext(
                receipt_id=RECEIPT_ID,
                caller_scope="system:local",
                operation_scope="agent.create",
                idempotency_key="repair-safety",
                request_hash="a" * 64,
            ),
            [
                PendingEvent(
                    aggregate_type="agent",
                    aggregate_id=AGENT_ID,
                    event_type=AgentCreated.event_type,
                    payload=AgentCreated(
                        schema_version=1,
                        agent_id=AGENT_ID,
                        name="Alice",
                    ),
                    actor_agent_id=None,
                    occurred_at=NOW,
                )
            ],
        )
    async with database.transaction() as uow:
        await uow.session.execute(
            text("UPDATE guarded_agent_projection SET name = 'sentinel'")
        )

    try:
        yield database, manager, reducer
    finally:
        await database.dispose()


def corrupt(path: Path, corruption: str) -> None:
    connection = sqlite3.connect(path)
    try:
        if corruption == "missing_causation_receipt":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DELETE FROM idempotency_records")
        else:
            connection.execute("DROP TRIGGER domain_events_reject_update")
            if corruption == "sequence_gap":
                connection.execute("UPDATE domain_events SET sequence = 2")
            elif corruption == "unknown_event":
                connection.execute(
                    "UPDATE domain_events SET event_type = 'unknown.event'"
                )
            elif corruption == "foreign_key":
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute(
                    "UPDATE domain_events SET actor_agent_id = ?",
                    ("00000000-0000-0000-0000-000000000fff",),
                )
            else:  # pragma: no cover - guarded by test parameters
                raise AssertionError(corruption)
        connection.commit()
    finally:
        connection.close()


async def snapshot(
    database: Database,
) -> tuple[
    list[tuple[str, str]],
    tuple[int, int, str | None, datetime | None] | None,
]:
    async with database.read_session() as session:
        rows = list(
            (
                await session.execute(
                    text(
                        "SELECT agent_id, name FROM guarded_agent_projection "
                        "ORDER BY agent_id"
                    )
                )
            ).tuples()
        )
        version = await session.get(
            ProjectionVersionRow, PROJECTION_NAME
        )
    version_state = (
        None
        if version is None
        else (
            version.reducer_version,
            version.last_applied_event_id,
            version.last_verified_hash,
            version.last_verified_at,
        )
    )
    return rows, version_state


@pytest.mark.parametrize("operation", ["verify", "repair"])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_causation_receipt",
        "sequence_gap",
        "unknown_event",
        "foreign_key",
    ],
)
@pytest.mark.asyncio
async def test_corrupt_sources_abort_before_projection_rebuild_or_swap(
    repository_root: Path,
    tmp_path: Path,
    corruption: str,
    operation: str,
) -> None:
    path = tmp_path / f"{corruption}-{operation}.sqlite3"
    stack = seeded_database(repository_root, path)
    database, manager, reducer = await anext(stack)
    try:
        before = await snapshot(database)
        await database.dispose()
        corrupt(path, corruption)

        with pytest.raises(SourceIntegrityError):
            await getattr(manager, operation)(database)

        assert reducer.rebuild_calls == 0
        assert await snapshot(database) == before
        async with database.read_session() as session:
            assert (
                await session.scalar(
                    text(
                        "SELECT count(*) FROM sqlite_temp_master "
                        "WHERE name LIKE '_rebuild_%'"
                    )
                )
                == 0
            )
            version_count = await session.scalar(
                select(ProjectionVersionRow).where(
                    ProjectionVersionRow.name == PROJECTION_NAME
                )
            )
        assert version_count is not None
    finally:
        await stack.aclose()
