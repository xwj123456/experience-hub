from __future__ import annotations

import json
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from tests.repository.test_passport_source_validation import (
    NOW,
    _seed_decision,
)
from tests.repository.test_passport_source_validation import (
    container as container,
)

from experience_hub.bootstrap import ApplicationContainer
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.sharing.models import CreateTopic, PublishCapsule
from experience_hub.sharing.validation import SharingSourceValidator
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import (
    ExperienceCapsuleRow,
    PassportAdoptionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


async def _publish(container: ApplicationContainer) -> CommandResult:
    async with container.database.read_session() as session:
        adoption = await session.scalar(select(PassportAdoptionRow))
        assert adoption is not None
        owner = adoption.owner_agent_id
        experience = adoption.resulting_experience_id

    async def create_topic(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await container.sharing_service.create_topic(
            uow=uow,
            command=CreateTopic(owner_agent_id=owner, name="Synthetic transfer"),
            command_context=command,
        )

    topic = await container.command_executor.execute(
        CommandRequest(
            caller_scope=f"agent:{owner}",
            operation_scope="topic.create",
            idempotency_key="topic",
            method="POST",
            route_template="/v1/topics",
            body={"name": "Synthetic transfer"},
        ),
        create_topic,
    )
    assert topic.status_code == 201
    topic_id = UUID(json.loads(topic.body)["data"]["topic_id"])

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await container.sharing_service.publish_capsule(
            uow=uow,
            command_context=command,
            command=PublishCapsule(
                owner_agent_id=owner,
                experience_id=experience,
                topic_id=topic_id,
                version_id=None,
                expires_at=NOW + timedelta(days=1),
            ),
        )

    return await container.command_executor.execute(
        CommandRequest(
            caller_scope=f"agent:{owner}",
            operation_scope="capsule.publish",
            idempotency_key="publish",
            method="POST",
            route_template="/v1/capsules",
            body={
                "experience_id": experience,
                "topic_id": topic_id,
                "version_id": None,
                "expires_at": NOW + timedelta(days=1),
            },
        ),
        handler,
    )


async def test_adopted_passport_cannot_be_published_as_capsule(
    container: ApplicationContainer,
) -> None:
    await _seed_decision(container, "created")
    result = await _publish(container)
    assert result.status_code == 400
    assert (
        json.loads(result.body)["error"]["code"] == "passport_publication_unsupported"
    )
    async with container.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceCapsuleRow))
            == 0
        )
        await container.source_validator.validate(session)


async def test_genuine_local_equivalent_remains_publishable(
    container: ApplicationContainer,
) -> None:
    await _seed_decision(container, "reused")
    result = await _publish(container)
    assert result.status_code == 201
    async with container.database.read_session() as session:
        await container.source_validator.validate(session)


async def test_forged_publication_origin_rejected_by_sharing_source_validation(
    container: ApplicationContainer,
) -> None:
    await _seed_decision(container, "reused")
    result = await _publish(container)
    assert result.status_code == 201
    async with container.database._engine.begin() as connection:
        await connection.execute(text("DROP TRIGGER experiences_reject_update"))
        await connection.execute(
            text("UPDATE experiences SET origin='adopted_passport'")
        )
    async with container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await SharingSourceValidator(container.event_registry).validate(session)
