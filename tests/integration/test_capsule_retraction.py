from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import delete, func, select
from tests.integration.test_capsule_adoption import (
    ADOPTED_EXPERIENCE_ID,
    ADOPTED_VERSION_ID,
    ADOPTER_ID,
    CAPSULE_ID,
    ITEM_ID,
    OTHER_AGENT_ID,
    PUBLISHER_ID,
    SOURCE_EXPERIENCE_ID,
    SOURCE_VERSION_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
    request,
)

from experience_hub.domain import CommandContext, StructuredReason
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.sharing.events import CapsuleRetractedV1
from experience_hub.sharing.models import (
    CapsuleStatus,
    InboxState,
    RetractCapsule,
)
from experience_hub.sharing.projector import (
    CapsuleStateProjector,
    InboxItemProjector,
)
from experience_hub.sharing.validation import (
    register_sharing_source_validator,
)
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_experience_source_validator,
)

RETRACTION_REASON = "传播范围已经不再适用"


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-retraction.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def retract(
    stack: AdoptionStack,
    *,
    key: str,
    publisher_agent_id: UUID = PUBLISHER_ID,
    caller_agent_id: UUID | None = None,
    capsule_id: UUID = CAPSULE_ID,
    reason: StructuredReason | str = RETRACTION_REASON,
) -> CommandResult:
    caller = caller_agent_id or publisher_agent_id
    command = RetractCapsule(
        publisher_agent_id=publisher_agent_id,
        capsule_id=capsule_id,
        reason=reason,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.retract_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.retract",
            route_template=("/v1/agents/{agent_id}/capsules/{capsule_id}:retract"),
            agent_id=caller,
            path_parameters={
                "agent_id": caller,
                "capsule_id": capsule_id,
            },
            body={"reason": reason},
        ),
        handler,
    )


def error_code(result: CommandResult) -> str:
    return str(json.loads(result.body)["error"]["code"])


async def retraction_rows(
    stack: AdoptionStack,
) -> tuple[DomainEventRow, ...]:
    async with stack.database.read_session() as session:
        return tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == CapsuleRetractedV1.event_type)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )


def test_retract_command_requires_a_non_null_reason() -> None:
    with pytest.raises(TypeError):
        RetractCapsule(  # type: ignore[call-arg]
            publisher_agent_id=PUBLISHER_ID,
            capsule_id=CAPSULE_ID,
        )

    with pytest.raises(ValueError, match="reason"):
        RetractCapsule(
            publisher_agent_id=PUBLISHER_ID,
            capsule_id=CAPSULE_ID,
            reason=cast(Any, None),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ("   ", "x" * 2_001))
async def test_retraction_rejects_invalid_reason_without_side_effects(
    stack: AdoptionStack,
    reason: str,
) -> None:
    await arrange_pending_capsule(stack)

    result = await retract(
        stack,
        key=f"invalid-reason-{len(reason)}",
        reason=reason,
    )

    assert result.status_code == 422
    assert error_code(result) == "invalid_reason"
    assert await retraction_rows(stack) == ()
    async with stack.database.read_session() as session:
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
    assert state is not None and state.status is CapsuleStatus.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("publisher_agent_id", "caller_agent_id"),
    (
        (OTHER_AGENT_ID, OTHER_AGENT_ID),
        (PUBLISHER_ID, OTHER_AGENT_ID),
    ),
)
async def test_only_publisher_can_retract_without_resource_disclosure(
    stack: AdoptionStack,
    publisher_agent_id: UUID,
    caller_agent_id: UUID,
) -> None:
    await arrange_pending_capsule(stack)

    result = await retract(
        stack,
        key=f"foreign-retract-{publisher_agent_id}-{caller_agent_id}",
        publisher_agent_id=publisher_agent_id,
        caller_agent_id=caller_agent_id,
    )

    assert result.status_code == 404
    assert error_code(result) == "resource_not_found"
    assert await retraction_rows(stack) == ()
    async with stack.database.read_session() as session:
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
        source = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
        item = await session.get(InboxItemRow, ITEM_ID)
    assert state is not None and state.status is CapsuleStatus.ACTIVE
    assert source is not None
    assert item is not None and item.state is InboxState.PENDING


@pytest.mark.asyncio
async def test_retraction_emits_one_strict_v1_event_retains_sources_and_replays(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    supplied = "  传播范围已经不再适用  "
    expected_reason = StructuredReason.from_user_text(RETRACTION_REASON)

    first = await retract(stack, key="retract", reason=supplied)
    replay = await retract(stack, key="retract", reason=supplied)

    assert first.status_code == 200
    assert not first.replayed
    assert replay.replayed
    assert (replay.status_code, replay.body, replay.headers) == (
        first.status_code,
        first.body,
        first.headers,
    )

    rows = await retraction_rows(stack)
    assert len(rows) == 1
    row = rows[0]
    assert (
        row.aggregate_type,
        row.aggregate_id,
        row.sequence,
        row.actor_agent_id,
    ) == ("capsule", CAPSULE_ID, 2, PUBLISHER_ID)
    payload = stack.registry.decode(
        event_type=row.event_type,
        payload=row.payload,
    )
    assert isinstance(payload, CapsuleRetractedV1)
    assert payload.model_dump(mode="json") == {
        "schema_version": 1,
        "capsule_id": str(CAPSULE_ID),
        "publisher_agent_id": str(PUBLISHER_ID),
        "reason": expected_reason.model_dump(mode="json"),
        "status_before": CapsuleStatus.ACTIVE.value,
        "status_after": CapsuleStatus.RETRACTED.value,
    }
    with pytest.raises(ValidationError):
        CapsuleRetractedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "unexpected": True,
            }
        )

    async with stack.database.read_session() as session:
        capsule_state = await session.get(CapsuleStateRow, CAPSULE_ID)
        capsule_source = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
        inbox_item = await session.get(InboxItemRow, ITEM_ID)
        source_experience = await session.get(
            ExperienceRow,
            SOURCE_EXPERIENCE_ID,
        )
        source_version = await session.get(
            ExperienceVersionRow,
            SOURCE_VERSION_ID,
        )
    assert capsule_state is not None
    assert (
        capsule_state.status,
        capsule_state.projection_event_id,
    ) == (CapsuleStatus.RETRACTED, row.event_id)
    assert capsule_source is not None
    assert source_experience is not None
    assert source_version is not None
    assert inbox_item is not None and inbox_item.state is InboxState.PENDING


@pytest.mark.asyncio
async def test_retraction_rejects_clock_regression_without_state_change(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(microseconds=-1))

    result = await retract(stack, key="old-retraction")

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    assert await retraction_rows(stack) == ()
    async with stack.database.read_session() as session:
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
        item = await session.get(InboxItemRow, ITEM_ID)
        source = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
    assert state is not None and state.status is CapsuleStatus.ACTIVE
    assert item is not None and item.state is InboxState.PENDING
    assert source is not None


@pytest.mark.asyncio
async def test_retracted_capsule_cannot_later_be_adopted(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await retract(stack, key="retract-before-adoption")).status_code == 200

    result = await adopt(stack, key="adopt-after-retraction")

    assert result.status_code == 409
    assert error_code(result) == "capsule_retracted"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        adoption_count = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
    assert item is not None and item.state is InboxState.PENDING
    assert adoption_count == 0


@pytest.mark.asyncio
async def test_retraction_does_not_delete_an_already_adopted_local_copy(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    adopted = await adopt(stack, key="adopt-before-retraction")
    assert adopted.status_code == 200
    async with stack.database.read_session() as session:
        experience_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_type == "experience")
        )

    result = await retract(stack, key="retract-after-adoption")

    assert result.status_code == 200
    async with stack.database.read_session() as session:
        capsule_source = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
        capsule_state = await session.get(CapsuleStateRow, CAPSULE_ID)
        inbox_item = await session.get(InboxItemRow, ITEM_ID)
        adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.capsule_id == CAPSULE_ID,
                AdoptionRecordRow.adopter_agent_id == ADOPTER_ID,
            )
        )
        experience = await session.get(
            ExperienceRow,
            ADOPTED_EXPERIENCE_ID,
        )
        version = await session.get(
            ExperienceVersionRow,
            ADOPTED_VERSION_ID,
        )
        payload = await session.scalar(
            select(ExperiencePayloadRow).where(
                ExperiencePayloadRow.version_id == ADOPTED_VERSION_ID
            )
        )
        experience_state = await session.get(
            ExperienceStateRow,
            ADOPTED_EXPERIENCE_ID,
        )
        after_experience_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_type == "experience")
        )
    assert capsule_source is not None
    assert capsule_state is not None
    assert capsule_state.status is CapsuleStatus.RETRACTED
    assert inbox_item is not None and inbox_item.state is InboxState.ADOPTED
    assert adoption is not None
    assert experience is not None
    assert version is not None
    assert payload is not None
    assert experience_state is not None
    assert after_experience_event_count == experience_event_count


@pytest.mark.asyncio
async def test_retracted_capsule_projection_verifies_rebuilds_and_repairs(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    result = await retract(stack, key="retract-before-rebuild")
    assert result.status_code == 200

    source_validator = SourceValidator(stack.registry)
    register_experience_source_validator(source_validator)
    register_sharing_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry(
            (
                ExperienceProjector(stack.registry, LifecycleConfig()),
                CapsuleStateProjector(stack.registry),
                InboxItemProjector(stack.registry),
            )
        ),
        source_validator=source_validator,
    )
    assert (await manager.verify(stack.database)).matches

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            delete(CapsuleStateRow).where(CapsuleStateRow.capsule_id == CAPSULE_ID)
        )
    with pytest.raises(ProjectionMismatch):
        await manager.verify(stack.database)

    report = await manager.repair(stack.database)

    assert report.matches
    async with stack.database.read_session() as session:
        capsule_state = await session.get(CapsuleStateRow, CAPSULE_ID)
        capsule_source = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
        inbox_item = await session.get(InboxItemRow, ITEM_ID)
    assert capsule_state is not None
    assert capsule_state.status is CapsuleStatus.RETRACTED
    assert capsule_source is not None
    assert inbox_item is not None and inbox_item.state is InboxState.PENDING
