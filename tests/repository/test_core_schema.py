from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from experience_hub.storage.database import Database
from experience_hub.storage.tables import Base


@pytest.fixture
def migrated_database_path(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Path]:
    database_path = tmp_path / "core.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    yield database_path


@pytest.fixture
def migrated_engine(migrated_database_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        yield engine
    finally:
        engine.dispose()


def _column_signature(engine: Engine, table_name: str) -> set[tuple[object, ...]]:
    inspector = inspect(engine)
    return {
        (
            column["name"],
            str(column["type"]),
            bool(column["nullable"]),
            bool(column["primary_key"]),  # type: ignore[typeddict-item]
        )
        for column in inspector.get_columns(table_name)
    }


def _metadata_column_signature(
    engine: Engine,
    table_name: str,
) -> set[tuple[object, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        (
            column.name,
            str(column.type.compile(dialect=engine.dialect)),
            column.nullable,
            bool(column.primary_key),
        )
        for column in table.columns
    }


def _index_signature(engine: Engine, table_name: str) -> set[tuple[object, ...]]:
    return {
        (
            index["name"],
            tuple(index["column_names"]),
            bool(index["unique"]),
        )
        for index in inspect(engine).get_indexes(table_name)
    }


def _metadata_index_signature(table_name: str) -> set[tuple[object, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            index.unique,
        )
        for index in table.indexes
    }


def test_migration_head_matches_declared_metadata(migrated_engine: Engine) -> None:
    reflected_tables = set(inspect(migrated_engine).get_table_names())
    assert reflected_tables == {*Base.metadata.tables, "alembic_version"}

    for table_name in Base.metadata.tables:
        assert _column_signature(
            migrated_engine, table_name
        ) == _metadata_column_signature(migrated_engine, table_name)
        assert _index_signature(
            migrated_engine, table_name
        ) == _metadata_index_signature(table_name)


@pytest.mark.asyncio
async def test_core_unique_and_check_constraints(
    migrated_database_path: Path,
) -> None:
    database = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    agent_id = str(uuid4())
    receipt_id = str(uuid4())
    now = "2026-07-17T00:00:00.000000Z"
    request_hash = "a" * 64

    try:
        async with database.transaction() as uow:
            await uow.session.execute(
                text(
                    "INSERT INTO agents (agent_id, name, created_at) "
                    "VALUES (:agent_id, 'Alice', :now)"
                ),
                {"agent_id": agent_id, "now": now},
            )
            await uow.session.execute(
                text(
                    "INSERT INTO idempotency_records "
                    "(receipt_id, caller_scope, scope, idempotency_key, "
                    "request_hash, state, created_at) "
                    "VALUES (:receipt_id, 'system:local', 'agent.create', "
                    "'create-alice', :request_hash, 'in_progress', :now)"
                ),
                {
                    "receipt_id": receipt_id,
                    "request_hash": request_hash,
                    "now": now,
                },
            )
            await uow.session.execute(
                text(
                    "INSERT INTO domain_events "
                    "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                    "actor_agent_id, causation_id, occurred_at) "
                    "VALUES ('agent', :agent_id, 1, 'agent.created', :payload, "
                    ":agent_id, :receipt_id, :now)"
                ),
                {
                    "agent_id": agent_id,
                    "payload": b'{"schema_version":1}',
                    "receipt_id": receipt_id,
                    "now": now,
                },
            )

        duplicate_statements = (
            (
                "INSERT INTO agents (agent_id, name, created_at) "
                "VALUES (:new_id, 'Alice', :now)",
                {"new_id": str(uuid4()), "now": now},
            ),
            (
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, created_at) VALUES (:new_id, 'system:local', "
                "'agent.create', 'create-alice', :request_hash, 'in_progress', :now)",
                {"new_id": str(uuid4()), "request_hash": request_hash, "now": now},
            ),
            (
                "INSERT INTO domain_events "
                "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                "actor_agent_id, causation_id, occurred_at) "
                "VALUES ('agent', :agent_id, 1, 'agent.renamed', :payload, "
                ":agent_id, :receipt_id, :now)",
                {
                    "agent_id": agent_id,
                    "payload": b'{"schema_version":1}',
                    "receipt_id": receipt_id,
                    "now": now,
                },
            ),
        )
        for statement, parameters in duplicate_statements:
            with pytest.raises(IntegrityError):
                async with database.transaction() as uow:
                    await uow.session.execute(text(statement), parameters)

        invalid_statements: tuple[tuple[str, dict[str, str]], ...] = (
            (
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, created_at) VALUES (:new_id, 'system:local', "
                "'agent.create', ' padded ', :request_hash, 'in_progress', :now)",
                {"new_id": str(uuid4()), "request_hash": request_hash, "now": now},
            ),
            (
                "INSERT INTO projection_versions "
                "(name, reducer_version, last_applied_event_id) "
                "VALUES ('experience_state', 0, 0)",
                {},
            ),
            (
                "INSERT INTO lifecycle_lease (lease_name) VALUES ('not-lifecycle')",
                {},
            ),
        )
        for statement, parameters in invalid_statements:
            with pytest.raises(IntegrityError):
                async with database.transaction() as uow:
                    await uow.session.execute(text(statement), parameters)
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_agents_are_immutable_after_insert(
    migrated_database_path: Path,
) -> None:
    database = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    agent_id = str(uuid4())
    try:
        async with database.transaction() as uow:
            await uow.session.execute(
                text(
                    "INSERT INTO agents (agent_id, name, created_at) "
                    "VALUES (:agent_id, 'Alice', :created_at)"
                ),
                {
                    "agent_id": agent_id,
                    "created_at": datetime(2026, 7, 17, tzinfo=UTC)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                },
            )

        for statement in (
            "UPDATE agents SET name = 'Changed' WHERE agent_id = :agent_id",
            "DELETE FROM agents WHERE agent_id = :agent_id",
        ):
            with pytest.raises(IntegrityError, match="agents rows are immutable"):
                async with database.transaction() as uow:
                    await uow.session.execute(text(statement), {"agent_id": agent_id})

        replacement_statements = (
            (
                "INSERT OR REPLACE INTO agents (agent_id, name, created_at) "
                "VALUES (:agent_id, 'Replacement', :created_at)",
                {"agent_id": agent_id},
            ),
            (
                "INSERT OR REPLACE INTO agents (agent_id, name, created_at) "
                "VALUES (:agent_id, 'Alice', :created_at)",
                {"agent_id": str(uuid4())},
            ),
        )
        for statement, parameters in replacement_statements:
            with pytest.raises(
                IntegrityError,
                match="agents identity or name already exists",
            ):
                async with database.transaction() as uow:
                    await uow.session.execute(
                        text(statement),
                        {
                            **parameters,
                            "created_at": datetime(2026, 7, 18, tzinfo=UTC)
                            .isoformat(timespec="microseconds")
                            .replace("+00:00", "Z"),
                        },
                    )

        async with database.read_session() as session:
            original = (
                await session.execute(
                    text(
                        "SELECT agent_id, name, created_at FROM agents "
                        "ORDER BY agent_id"
                    )
                )
            ).one()
        assert original == (
            agent_id,
            "Alice",
            "2026-07-17T00:00:00.000000Z",
        )
    finally:
        await database.dispose()


def test_migration_downgrade_removes_core_schema(
    repository_root: Path,
    migrated_database_path: Path,
) -> None:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{migrated_database_path}")

    command.downgrade(config, "base")

    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()
