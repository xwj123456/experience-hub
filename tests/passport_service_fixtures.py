"""Real transactional fixtures limited to Passport application-service tests."""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config

from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, StructuredReason
from experience_hub.ids import SequenceIdGenerator
from experience_hub.passports.codec import (
    VerifiedPassportV1,
    encode_passport_document,
    verify_passport_bytes,
)
from experience_hub.passports.requests import (
    passport_adopt_request,
    passport_import_request,
    passport_reject_request,
)
from experience_hub.storage import CommandResult, StoredResponse, UnitOfWork
from experience_hub.storage.faults import FaultCheckpoint
from tests.passport_fixtures import passport_document

if TYPE_CHECKING:
    from experience_hub.passports.queries import PassportQuery
    from experience_hub.passports.service import PassportService

NOW = datetime(2026, 9, 30, 8, tzinfo=UTC)
OWNER = UUID(int=5001)
OTHER = UUID(int=5002)
MISSING = UUID(int=5999)


class FailAt:
    def __init__(self) -> None:
        self.checkpoint: FaultCheckpoint | None = None
        self.occurrence = 1
        self.seen = 0

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is self.checkpoint:
            self.seen += 1
            if self.seen == self.occurrence:
                raise RuntimeError("injected passport checkpoint")


@dataclass(slots=True)
class PassportStack:
    container: ApplicationContainer
    service: PassportService
    query: PassportQuery
    fault: FailAt

    @property
    def clock(self) -> FrozenClock:
        assert isinstance(self.container.clock, FrozenClock)
        return self.container.clock


async def _agent(container: ApplicationContainer, name: str) -> None:
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=name,
        method="POST",
        route_template="/v1/agents",
        body={"name": name},
    )

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=name),
            command_context=context,
        )

    assert (
        await container.command_executor.execute(request, handler)
    ).status_code == 201


@pytest.fixture
async def passport_stack(
    repository_root: Path, tmp_path: Path
) -> AsyncIterator[PassportStack]:
    try:
        repository_module = importlib.import_module(
            "experience_hub.passports.repository"
        )
        query_module = importlib.import_module("experience_hub.passports.queries")
        service_module = importlib.import_module("experience_hub.passports.service")
    except ModuleNotFoundError:
        pytest.fail("Passport repository/query/service are not implemented")
    database_path = tmp_path / "passport-services.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")
    ids = (UUID(int=5003), OWNER, UUID(int=5004), OTHER)
    container = ApplicationContainer.build(
        Settings(database_url=f"sqlite+aiosqlite:///{database_path}"),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator((*ids, *(UUID(int=i) for i in range(5010, 5900)))),
    )
    fault = FailAt()
    container.database._fault_injector = fault
    await _agent(container, "Passport owner")
    await _agent(container, "Passport other owner")
    repository = repository_module.PassportRepository()
    stack = PassportStack(
        container=container,
        service=service_module.PassportService(
            repository=repository,
            experience_writer=container.experience_writer,
            receipt_store=container.receipt_store,
            clock=container.clock,
            id_generator=container.ids,
        ),
        query=query_module.PassportQuery(repository=repository),
        fault=fault,
    )
    try:
        yield stack
    finally:
        await container.close()


def prepared_passport(tag: str = "recovery") -> VerifiedPassportV1:
    return verify_passport_bytes(encode_passport_document(passport_document(tag=tag)))


async def import_passport(
    stack: PassportStack,
    *,
    owner: UUID = OWNER,
    key: str = "import",
    prepared: VerifiedPassportV1 | None = None,
    command_request: CommandRequest | None = None,
) -> CommandResult:
    from experience_hub.passports.contracts import ImportPassport

    prepared = prepared or prepared_passport()
    request = command_request or passport_import_request(
        owner_agent_id=owner,
        passport_hash=prepared.document.passport_hash,
        idempotency_key=key,
    )
    value = ImportPassport(owner_agent_id=owner, prepared=prepared)

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await stack.service.import_passport(
            uow=uow, request=value, command=context
        )

    return await stack.container.command_executor.execute(request, handler)


def result_id(result: CommandResult, field: str = "import_id") -> UUID:
    return UUID(json.loads(result.body)["data"][field])


async def decide_passport(
    stack: PassportStack,
    import_id: UUID,
    *,
    action: str = "adopt",
    owner: UUID = OWNER,
    key: str = "decision",
    importance: float = 0.7,
    confidence: float = 0.6,
    reason_text: str = "Not reusable here.",
) -> CommandResult:
    contracts = importlib.import_module("experience_hub.passports.contracts")
    command_type = getattr(
        contracts, "AdoptPassport" if action == "adopt" else "RejectPassport", None
    )
    if command_type is None:
        pytest.fail("Passport decision commands are not implemented")
    if action == "adopt":
        value = command_type(
            owner_agent_id=owner,
            import_id=import_id,
            importance=importance,
            confidence=confidence,
        )
        request = passport_adopt_request(
            owner_agent_id=owner,
            import_id=import_id,
            importance=importance,
            confidence=confidence,
            idempotency_key=key,
        )
    else:
        reason = StructuredReason.from_user_text(reason_text)
        value = command_type(owner_agent_id=owner, import_id=import_id, reason=reason)
        request = passport_reject_request(
            owner_agent_id=owner,
            import_id=import_id,
            reason=reason,
            idempotency_key=key,
        )

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        if action == "adopt":
            return await stack.service.adopt(uow=uow, request=value, command=context)
        return await stack.service.reject(uow=uow, request=value, command=context)

    return await stack.container.command_executor.execute(request, handler)
