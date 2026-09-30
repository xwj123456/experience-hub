"""Real service seeds for source export and cross-database CLI acceptance."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.engine import URL

from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.contracts import (
    CreateExperience,
    CreateExperienceVersion,
)
from experience_hub.experiences.models import ExperienceKind, VersionContent
from experience_hub.ids import SequenceIdGenerator
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork
from tests.passport_fixtures import passport_document

NOW = datetime(2026, 9, 30, 8, tzinfo=UTC)


@dataclass(frozen=True)
class ExportSeed:
    owner_agent_id: UUID
    experience_id: UUID
    version_id: UUID
    content_hash: str


def export_runtime(path: Path, *, offset: int = 1000) -> ApplicationRuntime:
    return ApplicationRuntime(
        Settings(database_url=URL.create("sqlite+aiosqlite", database=str(path))),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(
            tuple(UUID(int=i) for i in range(offset, offset + 300))
        ),
    )


async def create_export_owner(container: ApplicationContainer, key: str) -> UUID:
    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name="Synthetic " + key),
            command_context=command,
        )

    result = await container.command_executor.execute(
        CommandRequest(
            caller_scope="system:local",
            operation_scope="agent.create",
            idempotency_key=key,
            method="POST",
            route_template="/v1/agents",
            body={"name": "Synthetic " + key},
        ),
        handler,
    )
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["agent_id"])


async def create_export_experience(
    container: ApplicationContainer,
    owner: UUID,
    *,
    content: VersionContent | None = None,
    experience_id: UUID | None = None,
    key: str = "source-experience",
) -> ExportSeed:
    retained = content or passport_document().subject.content

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        if experience_id is None:
            return await container.experience_service.create(
                uow=uow,
                command_context=command,
                command=CreateExperience(
                    owner_agent_id=owner,
                    kind=ExperienceKind.PROCEDURAL,
                    content=retained,
                    importance=0.7,
                    confidence=0.6,
                ),
            )
        return await container.experience_service.create_version(
            uow=uow,
            command_context=command,
            command=CreateExperienceVersion(
                owner_agent_id=owner,
                experience_id=experience_id,
                content=retained,
            ),
        )

    result = await container.command_executor.execute(
        CommandRequest(
            caller_scope=f"agent:{owner}",
            operation_scope="experience.create"
            if experience_id is None
            else "experience.version",
            idempotency_key=key,
            method="POST",
            route_template="/synthetic-experience",
            body=retained.model_dump(mode="json"),
        ),
        handler,
    )
    assert result.status_code == 201
    data = json.loads(result.body)["data"]
    return ExportSeed(
        owner,
        UUID(data["experience_id"]),
        UUID(data["version_id"]),
        data["content_hash"],
    )


async def seed_export_database(path: Path, *, offset: int = 1000) -> ExportSeed:
    async with export_runtime(path, offset=offset).initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        owner = await create_export_owner(container, "publisher")
        return await create_export_experience(container, owner)
