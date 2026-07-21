import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import select

from experience_hub.agents import AgentCreated, AgentService, CreateAgent
from experience_hub.clock import FrozenClock
from experience_hub.domain.commands import CommandContext, CommandRequest
from experience_hub.domain.events import EventRegistry
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import (
    CommandExecutor,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import AgentRow, DomainEventRow
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000401")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000501")


@pytest.fixture
async def database(repository_root: Path, tmp_path: Path) -> AsyncIterator[Database]:
    database_path = tmp_path / "agent-create.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")
    registry = EventRegistry()
    registry.register(AgentCreated)
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}", event_registry=registry
    )
    try:
        yield database
    finally:
        await database.dispose()


def request(key: str = "agent-create-1", name: str = "  Alice  ") -> CommandRequest:
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


@pytest.mark.asyncio
async def test_agent_creation_persists_source_row_and_golden_event(
    database: Database,
) -> None:
    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock, id_generator=SequenceIdGenerator([RECEIPT_ID])
    )
    executor = CommandExecutor(database=database, receipt_store=receipts, clock=clock)
    service = AgentService(
        clock=clock,
        id_generator=SequenceIdGenerator([AGENT_ID]),
        receipt_store=receipts,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.create(
            uow=uow, command=CreateAgent(name="  Alice  "), command_context=context
        )

    result = await executor.execute(request(), handler)

    assert result.status_code == 201
    assert result.replayed is False
    assert json.loads(result.body) == {
        "data": {"agent_id": str(AGENT_ID), "name": "Alice"}
    }
    async with database.read_session() as session:
        agent = await session.get(AgentRow, AGENT_ID)
        event = await session.scalar(select(DomainEventRow))

    assert agent is not None and (agent.name, agent.created_at) == ("Alice", NOW)
    assert event is not None
    assert (
        event.aggregate_type,
        event.aggregate_id,
        event.sequence,
        event.event_type,
        event.causation_id,
    ) == ("agent", AGENT_ID, 1, "agent.created", RECEIPT_ID)
    assert json.loads(event.payload) == {
        "agent_id": str(AGENT_ID),
        "name": "Alice",
        "schema_version": 1,
    }
