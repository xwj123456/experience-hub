from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.agents import AgentCreated, AgentService, CreateAgent
from experience_hub.clock import FrozenClock
from experience_hub.domain.commands import CommandContext, CommandRequest
from experience_hub.domain.events import EventRegistry, StoredEvent
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import (
    CommandExecutor,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_agent_source_validator,
)

NOW = datetime(2026, 7, 17, 8, 30, tzinfo=UTC)
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000801")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000901")


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    def __init__(self, target: str) -> None:
        self.target = target
        self.calls: list[str] = []

    def __call__(self, checkpoint: object) -> None:
        name = str(checkpoint)
        self.calls.append(name)
        if name == self.target:
            raise InjectedFailure(name)


class AgentNameProjection:
    def __init__(self) -> None:
        self.name = "agent_name_projection"
        self.version = 1
        self.event_types = frozenset({AgentCreated.event_type})

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        payload = event.payload
        assert isinstance(payload, AgentCreated)
        await session.execute(
            text(
                "INSERT INTO agent_name_projection(agent_id, name) "
                "VALUES (:agent_id, :name)"
            ),
            {"agent_id": str(payload.agent_id), "name": payload.name},
        )

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = f"{target_prefix}{self.name}"
        await session.execute(
            text(
                f'CREATE TEMP TABLE "{target}" ('
                "agent_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
            )
        )


def request() -> CommandRequest:
    return CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key="atomic-agent-create",
        method="POST",
        route_template="/v1/agents",
        path_parameters={},
        query_parameters=(),
        body={"name": "Alice"},
        semantic_headers={},
    )


async def build_stack(
    repository_root: Path,
    database_path: Path,
    fault_injector: FailAt,
) -> AsyncGenerator[tuple[Database, CommandExecutor, AgentService]]:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    source_validator = SourceValidator(registry)
    register_agent_source_validator(source_validator)
    projection_manager = ProjectionManager(
        ProjectionRegistry([AgentNameProjection()]),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=projection_manager,
        fault_injector=fault_injector,
    )
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "CREATE TABLE agent_name_projection("
                "agent_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
            )
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator([RECEIPT_ID]),
    )
    executor = CommandExecutor(database=database, receipt_store=receipts, clock=clock)
    service = AgentService(
        clock=clock,
        id_generator=SequenceIdGenerator([AGENT_ID]),
        receipt_store=receipts,
    )
    try:
        yield database, executor, service
    finally:
        await database.dispose()


@pytest.mark.parametrize(
    "checkpoint",
    [
        "after_source_insert",
        "after_event_append",
        "after_projection_apply",
        "after_receipt_completion",
    ],
)
@pytest.mark.asyncio
async def test_command_fault_rolls_back_every_atomic_component(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: str,
) -> None:
    fault = FailAt(checkpoint)
    stack = build_stack(repository_root, tmp_path / f"{checkpoint}.sqlite3", fault)
    database, executor, service = await anext(stack)

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service.create(
            uow=uow,
            command=CreateAgent(name="Alice"),
            command_context=context,
        )

    try:
        with pytest.raises(InjectedFailure, match=checkpoint):
            await executor.execute(request(), handler)

        async with database.read_session() as session:
            assert await session.scalar(select(func.count()).select_from(AgentRow)) == 0
            assert (
                await session.scalar(
                    select(func.count()).select_from(DomainEventRow)
                )
                == 0
            )
            assert (
                await session.scalar(text("SELECT count(*) FROM agent_name_projection"))
                == 0
            )
            assert (
                await session.scalar(
                    select(func.count()).select_from(IdempotencyRecordRow)
                )
                == 0
            )
        assert fault.calls.count(checkpoint) == 1
    finally:
        await stack.aclose()
