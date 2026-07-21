import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from experience_hub.agents.events import AgentCreated
from experience_hub.domain.events import EventRegistry
from experience_hub.storage.database import Database
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_agent_source_validator,
)

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
AGENT_ID = UUID("00000000-0000-0000-0000-000000000501")
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000401")


@pytest.fixture
async def source_database(
    repository_root: Path, tmp_path: Path
) -> AsyncIterator[tuple[Database, Path, SourceValidator]]:
    path = tmp_path / "source-validation.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")
    registry = EventRegistry()
    registry.register(AgentCreated)
    validator = SourceValidator(registry)
    register_agent_source_validator(validator)
    database = Database.create(
        f"sqlite+aiosqlite:///{path}", event_registry=registry
    )
    try:
        yield database, path, validator
    finally:
        await database.dispose()


async def seed_valid_agent(database: Database) -> None:
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "INSERT INTO agents(agent_id, name, created_at) "
                "VALUES (:agent_id, 'Alice', :created_at)"
            ),
            {"agent_id": str(AGENT_ID), "created_at": NOW.isoformat()},
        )
        await uow.session.execute(
            text(
                "INSERT INTO idempotency_records("
                "receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, response_status_code, response_body, response_content_type, "
                "response_headers, created_at, completed_at"
                ") VALUES ("
                ":receipt_id, 'system:local', 'agent.create', 'agent-1', :hash, "
                "'completed', 201, '{}', 'application/json', '{}', "
                ":created_at, :created_at)"
            ),
            {
                "receipt_id": str(RECEIPT_ID),
                "hash": "a" * 64,
                "created_at": NOW.isoformat(),
            },
        )
        await uow.session.execute(
            text(
                "INSERT INTO domain_events("
                "aggregate_type, aggregate_id, sequence, event_type, payload, "
                "causation_id, occurred_at"
                ") VALUES ('agent', :agent_id, 1, 'agent.created', :payload, "
                ":receipt_id, :occurred_at)"
            ),
            {
                "agent_id": str(AGENT_ID),
                "payload": (
                    b'{"agent_id":"00000000-0000-0000-0000-000000000501",'
                    b'"name":"Alice","schema_version":1}'
                ),
                "receipt_id": str(RECEIPT_ID),
                "occurred_at": NOW.isoformat(),
            },
        )


@pytest.mark.asyncio
async def test_source_validator_accepts_valid_ledger_and_agent_correspondence(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, _, validator = source_database
    await seed_valid_agent(database)
    async with database.read_session() as session:
        await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_foreign_key_violation(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DROP TRIGGER domain_events_reject_update")
    connection.execute(
        "UPDATE domain_events SET actor_agent_id = ?",
        ("00000000-0000-0000-0000-000000000999",),
    )
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="foreign key"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_sequence_not_contiguous_from_one(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER domain_events_reject_update")
    connection.execute("UPDATE domain_events SET sequence = 2")
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="sequence"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_sequence_reversed_from_event_order(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, _ = source_database
    await seed_valid_agent(database)
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "INSERT INTO domain_events("
                "aggregate_type, aggregate_id, sequence, event_type, payload, "
                "causation_id, occurred_at"
                ") VALUES ('agent', :agent_id, 2, 'agent.created', :payload, "
                ":receipt_id, :occurred_at)"
            ),
            {
                "agent_id": str(AGENT_ID),
                "payload": (
                    b'{"agent_id":"00000000-0000-0000-0000-000000000501",'
                    b'"name":"Alice","schema_version":1}'
                ),
                "receipt_id": str(RECEIPT_ID),
                "occurred_at": NOW.isoformat(),
            },
        )
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER domain_events_reject_update")
    connection.execute("UPDATE domain_events SET sequence = 3 WHERE sequence = 1")
    connection.execute("UPDATE domain_events SET sequence = 1 WHERE sequence = 2")
    connection.execute("UPDATE domain_events SET sequence = 2 WHERE sequence = 3")
    connection.commit()
    connection.close()
    registry = EventRegistry()
    registry.register(AgentCreated)
    validator = SourceValidator(registry)

    with pytest.raises(SourceIntegrityError, match="sequence"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_unregistered_or_invalid_event_payload(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER domain_events_reject_update")
    connection.execute("UPDATE domain_events SET event_type = 'unknown.event'")
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="decode"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_registered_event_with_invalid_schema(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER domain_events_reject_update")
    connection.execute(
        "UPDATE domain_events SET payload = ?",
        (
            b'{"agent_id":"00000000-0000-0000-0000-000000000501",'
            b'"name":"Alice","schema_version":2,"extra":true}',
        ),
    )
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="decode"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_rejects_missing_causation_receipt(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DELETE FROM idempotency_records")
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="causation"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_source_validator_runs_registered_extension_hooks(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, _, validator = source_database

    class FailingHook:
        name = "semantic_hashes"

        async def validate(self, session: object) -> None:
            _ = session
            raise SourceIntegrityError("semantic hash mismatch")

    validator.register(FailingHook())
    with pytest.raises(SourceIntegrityError, match="semantic hash"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_agent_validator_rejects_orphan_row(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, _, validator = source_database
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "INSERT INTO agents(agent_id, name, created_at) "
                "VALUES (:agent_id, 'Alice', :created_at)"
            ),
            {"agent_id": str(AGENT_ID), "created_at": NOW.isoformat()},
        )
    with pytest.raises(SourceIntegrityError, match="agent.created"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_agent_validator_rejects_semantic_name_mismatch(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, path, validator = source_database
    await seed_valid_agent(database)
    await database.dispose()
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER agents_reject_update")
    connection.execute("UPDATE agents SET name = 'Mallory'")
    connection.commit()
    connection.close()

    with pytest.raises(SourceIntegrityError, match="agent.created"):
        async with database.read_session() as session:
            await validator.validate(session)


@pytest.mark.asyncio
async def test_agent_validator_rejects_orphan_created_event(
    source_database: tuple[Database, Path, SourceValidator],
) -> None:
    database, _, validator = source_database
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "INSERT INTO idempotency_records("
                "receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, response_status_code, response_body, response_content_type, "
                "response_headers, created_at, completed_at"
                ") VALUES ("
                ":receipt_id, 'system:local', 'agent.create', 'agent-1', :hash, "
                "'completed', 201, '{}', 'application/json', '{}', "
                ":created_at, :created_at)"
            ),
            {
                "receipt_id": str(RECEIPT_ID),
                "hash": "a" * 64,
                "created_at": NOW.isoformat(),
            },
        )
        await uow.session.execute(
            text(
                "INSERT INTO domain_events("
                "aggregate_type, aggregate_id, sequence, event_type, payload, "
                "causation_id, occurred_at"
                ") VALUES ('agent', :agent_id, 1, 'agent.created', :payload, "
                ":receipt_id, :occurred_at)"
            ),
            {
                "agent_id": str(AGENT_ID),
                "payload": (
                    b'{"agent_id":"00000000-0000-0000-0000-000000000501",'
                    b'"name":"Alice","schema_version":1}'
                ),
                "receipt_id": str(RECEIPT_ID),
                "occurred_at": NOW.isoformat(),
            },
        )

    with pytest.raises(SourceIntegrityError, match="agent.created"):
        async with database.read_session() as session:
            await validator.validate(session)


def test_source_integrity_error_has_stable_code() -> None:
    assert SourceIntegrityError("invalid").code == "source_integrity_error"
