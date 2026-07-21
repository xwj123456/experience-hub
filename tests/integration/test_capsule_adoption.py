from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, text, update

from experience_hub.agents import AgentCreated
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    StoredEvent,
)
from experience_hub.domain.values import TypedEvidence
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.contracts import (
    CreateExperience,
    ExperienceDraft,
)
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
    register_experience_events,
)
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
    decode_and_verify_version,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.sharing.events import (
    CapsuleAdoptedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import (
    AdoptCapsule,
    Capsule,
    CapsuleStatus,
    CreateSubscription,
    CreateTopic,
    InboxState,
    ProvenanceHop,
    PublishCapsule,
)
from experience_hub.sharing.projector import (
    AgentReputationProjector,
    CapsuleStateProjector,
    InboxItemProjector,
    SharingProjectionIntegrityError,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.service import SharingService
from experience_hub.sharing.validation import (
    register_sharing_source_validator,
)
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandResult,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    AgentRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceLinkRow,
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

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
PUBLISHER_ID = UUID("00000000-0000-0000-0000-000000000101")
ADOPTER_ID = UUID("00000000-0000-0000-0000-000000000102")
OTHER_AGENT_ID = UUID("00000000-0000-0000-0000-000000000103")

SOURCE_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000201")
SOURCE_VERSION_ID = UUID("00000000-0000-0000-0000-000000000202")
ADOPTED_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000203")
ADOPTED_VERSION_ID = UUID("00000000-0000-0000-0000-000000000204")
TOPIC_ID = UUID("00000000-0000-0000-0000-000000000301")
SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000302")
CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000303")
ITEM_ID = UUID("00000000-0000-0000-0000-000000000304")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000305")

UNKNOWN_ITEM_ID = UUID("00000000-0000-0000-0000-000000000999")
RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(1001, 1201)
)
EXTRA_WRITER_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(2001, 2061)
)
EXTRA_SHARING_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(3001, 3061)
)

SOURCE_CONTENT = VersionContent(
    body="传播时必须逐字保留正文，并将它隔离在事件载荷之外。",
    summary="显式采纳的不可变语义副本",
    mechanism="验证胶囊语义哈希后，在采纳者名下创建独立版本。",
    tags=("memory", "传播", "adoption"),
    applicability=("收到有效且未过期的胶囊", "明确选择采纳"),
    evidence=(
        TypedEvidence(type="experiment", id="exp:semantic-copy"),
        TypedEvidence(type="document", id="doc:provenance"),
    ),
    falsifiers=("复制后的语义哈希不同", "隔离区内容被自动检索"),
)


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    def __init__(self) -> None:
        self.checkpoint: FaultCheckpoint | None = None

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is self.checkpoint:
            raise InjectedFailure(checkpoint.value)


@dataclass(slots=True)
class AdoptionStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    receipts: ReceiptStore
    writer: ExperienceWriter
    experience_repository: ExperienceRepository
    experience_service: ExperienceService
    sharing_service: SharingService
    registry: EventRegistry
    fault: FailAt


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> AdoptionStack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    register_experience_events(registry)
    register_sharing_events(registry)
    lifecycle = LifecycleConfig()
    experience_projector = ExperienceProjector(registry, lifecycle)
    capsule_projector = CapsuleStateProjector(registry)
    inbox_projector = InboxItemProjector(registry)
    source_validator = SourceValidator(registry)
    register_experience_source_validator(source_validator)
    register_sharing_source_validator(source_validator)
    projection_manager = ProjectionManager(
        ProjectionRegistry((experience_projector, capsule_projector, inbox_projector)),
        source_validator=source_validator,
    )
    fault = FailAt()
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=projection_manager,
        fault_injector=fault,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            (
                AgentRow(
                    agent_id=PUBLISHER_ID,
                    name="Publisher",
                    created_at=NOW,
                ),
                AgentRow(
                    agent_id=ADOPTER_ID,
                    name="Adopter",
                    created_at=NOW,
                ),
                AgentRow(
                    agent_id=OTHER_AGENT_ID,
                    name="Other",
                    created_at=NOW,
                ),
            )
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    experience_repository = ExperienceRepository(event_registry=registry)
    writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(
            (
                SOURCE_EXPERIENCE_ID,
                SOURCE_VERSION_ID,
                ADOPTED_EXPERIENCE_ID,
                ADOPTED_VERSION_ID,
                *EXTRA_WRITER_IDS,
            )
        ),
        repository=experience_repository,
        lifecycle_config=lifecycle,
    )
    experience_service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=writer,
        query=ExperienceQuery(event_registry=registry),
        lifecycle_config=lifecycle,
    )
    sharing_service = SharingService(
        clock=clock,
        id_generator=SequenceIdGenerator(
            (
                TOPIC_ID,
                SUBSCRIPTION_ID,
                CAPSULE_ID,
                ITEM_ID,
                ADOPTION_ID,
                *EXTRA_SHARING_IDS,
            )
        ),
        receipt_store=receipts,
        repository=SharingRepository(),
        experience_query=ExperienceQuery(event_registry=registry),
        experience_writer=writer,
        experience_repository=experience_repository,
        experience_service=experience_service,
    )
    return AdoptionStack(
        database=database,
        clock=clock,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        receipts=receipts,
        writer=writer,
        experience_repository=experience_repository,
        experience_service=experience_service,
        sharing_service=sharing_service,
        registry=registry,
        fault=fault,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-adoption.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def request(
    *,
    key: str,
    operation_scope: str,
    route_template: str,
    agent_id: UUID,
    body: dict[str, Any],
    path_parameters: dict[str, Any] | None = None,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope=operation_scope,
        idempotency_key=key,
        method="POST",
        route_template=route_template,
        path_parameters=path_parameters or {"agent_id": agent_id},
        body=body,
    )


async def create_source_experience(stack: AdoptionStack) -> None:
    command = CreateExperience(
        owner_agent_id=PUBLISHER_ID,
        kind=ExperienceKind.PROCEDURAL,
        content=SOURCE_CONTENT,
        importance=0.72,
        confidence=0.80,
        links=(),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.experience_service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="create-source",
            operation_scope="experience.create",
            route_template="/v1/experiences",
            agent_id=PUBLISHER_ID,
            body={"summary": SOURCE_CONTENT.summary},
        ),
        handler,
    )
    assert result.status_code == 201


async def create_topic(stack: AdoptionStack) -> None:
    command = CreateTopic(
        owner_agent_id=PUBLISHER_ID,
        name="capsule-adoption",
        description="Explicit quarantine exit.",
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.create_topic(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="create-topic",
            operation_scope="topic.create",
            route_template="/v1/topics",
            agent_id=PUBLISHER_ID,
            body={"name": command.name, "description": command.description},
        ),
        handler,
    )
    assert result.status_code == 201


async def subscribe_adopter(stack: AdoptionStack) -> None:
    command = CreateSubscription(
        subscriber_agent_id=ADOPTER_ID,
        topic_id=TOPIC_ID,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.create_subscription(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="subscribe-adopter",
            operation_scope="subscription.create",
            route_template="/v1/agents/{agent_id}/subscriptions",
            agent_id=ADOPTER_ID,
            body={"topic_id": TOPIC_ID},
        ),
        handler,
    )
    assert result.status_code == 201


async def publish_capsule(stack: AdoptionStack) -> Capsule:
    command = PublishCapsule(
        owner_agent_id=PUBLISHER_ID,
        topic_id=TOPIC_ID,
        experience_id=SOURCE_EXPERIENCE_ID,
        version_id=SOURCE_VERSION_ID,
        expires_at=NOW + timedelta(days=7),
        parent_adoption_id=None,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="publish",
            operation_scope="capsule.publish",
            route_template="/v1/capsules",
            agent_id=PUBLISHER_ID,
            body={
                "topic_id": TOPIC_ID,
                "experience_id": SOURCE_EXPERIENCE_ID,
                "version_id": SOURCE_VERSION_ID,
                "expires_at": command.expires_at,
            },
        ),
        handler,
    )
    assert result.status_code == 201
    return Capsule.model_validate(json.loads(result.body)["data"], strict=False)


async def arrange_pending_capsule(stack: AdoptionStack) -> Capsule:
    await create_source_experience(stack)
    await create_topic(stack)
    await subscribe_adopter(stack)
    capsule = await publish_capsule(stack)
    assert capsule.capsule_id == CAPSULE_ID
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
    assert item is not None
    assert (
        item.recipient_agent_id,
        item.capsule_id,
        item.state,
    ) == (ADOPTER_ID, CAPSULE_ID, InboxState.PENDING)
    return capsule


async def adopt(
    stack: AdoptionStack,
    *,
    key: str,
    adopter_agent_id: UUID = ADOPTER_ID,
    item_id: UUID = ITEM_ID,
    importance: float = 0.50,
) -> CommandResult:
    command = AdoptCapsule(
        adopter_agent_id=adopter_agent_id,
        item_id=item_id,
        importance=importance,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.adopt_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.adopt",
            route_template="/v1/agents/{agent_id}/inbox/{item_id}:adopt",
            agent_id=adopter_agent_id,
            path_parameters={
                "agent_id": adopter_agent_id,
                "item_id": item_id,
            },
            body={"importance": importance},
        ),
        handler,
    )


async def create_archived_equivalent(stack: AdoptionStack) -> None:
    draft = ExperienceDraft(
        owner_agent_id=ADOPTER_ID,
        actor_agent_id=ADOPTER_ID,
        kind=ExperienceKind.PROCEDURAL,
        origin=ExperienceOrigin.LOCAL,
        content=SOURCE_CONTENT,
        importance=0.40,
        confidence=0.60,
        source_trust=1.0,
        initial_temperature=Temperature.ARCHIVED,
        links=(),
        occurred_at=stack.clock.now(),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        creation = await stack.writer.create_from_draft(
            uow=uow,
            draft=draft,
            command=context,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "experience_id": creation.experience_id,
                        "version_id": creation.version_id,
                        "content_hash": creation.content_hash,
                    }
                }
            ),
        )

    result = await stack.executor.execute(
        request(
            key="create-archived-equivalent",
            operation_scope="experience.create",
            route_template="/v1/experiences",
            agent_id=ADOPTER_ID,
            body={"summary": SOURCE_CONTENT.summary, "archived": True},
        ),
        handler,
    )
    assert result.status_code == 201
    assert UUID(json.loads(result.body)["data"]["experience_id"]) == (
        ADOPTED_EXPERIENCE_ID
    )


def error_code(result: CommandResult) -> str:
    return str(json.loads(result.body)["error"]["code"])


def stored_event(stack: AdoptionStack, row: DomainEventRow) -> StoredEvent:
    return StoredEvent(
        event_id=row.event_id,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        sequence=row.sequence,
        event_type=row.event_type,
        payload=stack.registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        ),
        actor_agent_id=row.actor_agent_id,
        causation_id=row.causation_id,
        occurred_at=row.occurred_at,
    )


def assert_no_body_key(value: Any) -> None:
    if isinstance(value, dict):
        assert "body" not in value
        for nested in value.values():
            assert_no_body_key(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_body_key(nested)


@pytest.mark.asyncio
async def test_pending_active_unexpired_adoption_copies_exact_semantics_and_provenance(
    stack: AdoptionStack,
) -> None:
    capsule = await arrange_pending_capsule(stack)

    result = await adopt(stack, key="adopt")

    assert result.status_code == 200
    assert not result.replayed
    response = json.loads(result.body)
    assert set(response) == {"data"}
    assert set(response["data"]) == {
        "experience",
        "created",
        "corroboration_applied",
    }
    assert response["data"]["created"] is True
    assert response["data"]["corroboration_applied"] is False
    assert response["data"]["experience"] == {
        "experience_id": str(ADOPTED_EXPERIENCE_ID),
        "owner_agent_id": str(ADOPTER_ID),
        "current_version_id": str(ADOPTED_VERSION_ID),
        "current_content_hash": capsule.source_content_hash,
        "temperature": "hot",
    }
    assert_no_body_key(response)

    async with stack.database.read_session() as session:
        identity = await session.get(ExperienceRow, ADOPTED_EXPERIENCE_ID)
        version = await session.get(ExperienceVersionRow, ADOPTED_VERSION_ID)
        payload = await session.get(ExperiencePayloadRow, ADOPTED_VERSION_ID)
        state = await session.get(ExperienceStateRow, ADOPTED_EXPERIENCE_ID)
        adoption = await session.get(AdoptionRecordRow, ADOPTION_ID)
        item = await session.get(InboxItemRow, ITEM_ID)
        links = await session.scalar(
            select(func.count())
            .select_from(ExperienceLinkRow)
            .where(ExperienceLinkRow.source_experience_id == ADOPTED_EXPERIENCE_ID)
        )
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            (
                                "experience.created",
                                "experience.version_created",
                                "capsule.adopted",
                            )
                        ),
                        DomainEventRow.actor_agent_id == ADOPTER_ID,
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )

    assert identity is not None
    assert version is not None
    assert payload is not None
    assert state is not None
    assert adoption is not None
    assert item is not None
    copied_content = decode_and_verify_version(
        identity=identity,
        version=version,
        payload=payload,
    )
    assert copied_content == SOURCE_CONTENT
    assert (
        identity.owner_agent_id,
        identity.kind,
        identity.origin,
        identity.created_at,
    ) == (
        ADOPTER_ID,
        capsule.kind,
        ExperienceOrigin.ADOPTED_CAPSULE,
        NOW,
    )
    assert (
        version.content_hash,
        state.current_content_hash,
        capsule.source_content_hash,
    ) == (
        capsule.source_content_hash,
        capsule.source_content_hash,
        capsule.source_content_hash,
    )
    assert (
        state.temperature,
        state.importance,
        state.access_count,
        state.access_strength,
    ) == (
        Temperature.HOT,
        0.50,
        0,
        0.0,
    )
    assert state.confidence == pytest.approx(0.80 * 0.50)
    assert state.source_trust == pytest.approx(0.50)
    assert links == 0
    assert (
        adoption.adopter_agent_id,
        adoption.capsule_id,
        adoption.resulting_experience_id,
        adoption.root_fingerprint,
        adoption.corroboration_applied,
        adoption.adopted_at,
    ) == (
        ADOPTER_ID,
        CAPSULE_ID,
        ADOPTED_EXPERIENCE_ID,
        capsule.root_fingerprint,
        False,
        NOW,
    )
    assert adoption.captured_trust == pytest.approx(0.50)
    expected_chain = (
        *capsule.provenance_chain,
        ProvenanceHop(
            capsule_id=CAPSULE_ID,
            publisher_agent_id=PUBLISHER_ID,
        ),
    )
    assert adoption.provenance_chain == canonical_json_bytes(expected_chain)
    assert (
        item.recipient_agent_id,
        item.capsule_id,
        item.state,
    ) == (ADOPTER_ID, CAPSULE_ID, InboxState.ADOPTED)

    assert [row.event_type for row in event_rows] == [
        "experience.created",
        "experience.version_created",
        "capsule.adopted",
    ]
    assert len({row.causation_id for row in event_rows}) == 1
    adopted_row = event_rows[-1]
    adopted_payload = stack.registry.decode(
        event_type=adopted_row.event_type,
        payload=adopted_row.payload,
    )
    assert isinstance(adopted_payload, CapsuleAdoptedV1)
    assert adopted_payload.model_dump(mode="json") == {
        "schema_version": 1,
        "item_id": str(ITEM_ID),
        "capsule_id": str(CAPSULE_ID),
        "adopter_agent_id": str(ADOPTER_ID),
        "adoption_id": str(ADOPTION_ID),
        "resulting_experience_id": str(ADOPTED_EXPERIENCE_ID),
        "root_fingerprint": capsule.root_fingerprint,
        "created": True,
        "corroboration_applied": False,
        "state_before": "pending",
        "state_after": "adopted",
    }
    assert (
        adopted_row.aggregate_type,
        adopted_row.aggregate_id,
        adopted_row.actor_agent_id,
        adopted_row.occurred_at,
    ) == ("inbox_item", ITEM_ID, ADOPTER_ID, NOW)
    assert (
        b'"body"' not in adopted_row.payload
        and b'"query"' not in adopted_row.payload
        and b'"error"' not in adopted_row.payload
    )


@pytest.mark.asyncio
async def test_adoption_captures_nondefault_observer_relative_trust(
    stack: AdoptionStack,
) -> None:
    from tests.integration.test_capsule_feedback import (
        publish_again,
        record_feedback,
    )
    from tests.integration.test_capsule_rejection import reject

    manager = cast(ProjectionManager, stack.database._projection_applier)  # noqa: SLF001
    manager.registry.register(AgentReputationProjector(stack.registry))
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-trust")).status_code == 200
    assert (
        await record_feedback(stack, key="useful-before-adoption")
    ).status_code == 201
    trusted_capsule, trusted_item_id = await publish_again(
        stack,
        key="publish-for-trusted-adoption",
    )

    result = await adopt(
        stack,
        key="adopt-trusted-publisher",
        item_id=trusted_item_id,
    )

    assert result.status_code == 200
    async with stack.database.read_session() as session:
        state = await session.get(
            ExperienceStateRow,
            ADOPTED_EXPERIENCE_ID,
        )
        adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.capsule_id
                == trusted_capsule.capsule_id,
                AdoptionRecordRow.adopter_agent_id == ADOPTER_ID,
            )
        )
    assert state is not None
    assert adoption is not None
    assert adoption.captured_trust == pytest.approx(0.60)
    assert state.source_trust == pytest.approx(0.60)
    assert state.confidence == pytest.approx(0.80 * 0.60)


@pytest.mark.asyncio
async def test_adoption_hides_foreign_and_missing_items_with_identical_404(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)

    foreign = await adopt(
        stack,
        key="foreign-item",
        adopter_agent_id=OTHER_AGENT_ID,
    )
    missing = await adopt(
        stack,
        key="missing-item",
        item_id=UNKNOWN_ITEM_ID,
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.body == missing.body
    assert error_code(foreign) == "resource_not_found"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(AdoptionRecordRow))
            == 0
        )
        item = await session.get(InboxItemRow, ITEM_ID)
        adopter_experiences = await session.scalar(
            select(func.count())
            .select_from(ExperienceRow)
            .where(ExperienceRow.owner_agent_id == ADOPTER_ID)
        )
    assert item is not None and item.state is InboxState.PENDING
    assert adopter_experiences == 0


@pytest.mark.asyncio
async def test_adoption_requires_pending_item_when_no_prior_result_exists(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(InboxItemRow)
            .where(InboxItemRow.item_id == ITEM_ID)
            .values(state=InboxState.REJECTED)
        )

    result = await adopt(stack, key="not-pending")

    assert result.status_code == 409
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(AdoptionRecordRow))
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "capsule.adopted")
            )
            == 0
        )


@pytest.mark.parametrize(
    ("availability", "expected_code"),
    (
        ("expired", "capsule_expired"),
        ("retracted", "capsule_retracted"),
    ),
)
@pytest.mark.asyncio
async def test_adoption_refuses_expired_or_retracted_capsules_without_side_effects(
    stack: AdoptionStack,
    availability: str,
    expected_code: str,
) -> None:
    await arrange_pending_capsule(stack)
    if availability == "expired":
        stack.clock.advance(timedelta(days=7))
    else:
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                update(CapsuleStateRow)
                .where(CapsuleStateRow.capsule_id == CAPSULE_ID)
                .values(status=CapsuleStatus.RETRACTED)
            )

    result = await adopt(stack, key=f"unavailable-{availability}")

    assert result.status_code == 409
    assert error_code(result) == expected_code
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        assert (
            await session.scalar(select(func.count()).select_from(AdoptionRecordRow))
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.owner_agent_id == ADOPTER_ID)
            )
            == 0
        )
    assert item is not None and item.state is InboxState.PENDING


@pytest.mark.asyncio
async def test_adoption_rejects_clock_regression_against_capsule_and_inbox_heads(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(microseconds=-1))

    result = await adopt(stack, key="clock-regression")

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        assert (
            await session.scalar(select(func.count()).select_from(AdoptionRecordRow))
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "capsule.adopted")
            )
            == 0
        )
    assert item is not None and item.state is InboxState.PENDING


@pytest.mark.asyncio
async def test_prior_adoption_result_precedes_pending_validation_across_receipts(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    first = await adopt(stack, key="first-adoption")
    assert first.status_code == 200
    assert not first.replayed
    async with stack.database.read_session() as session:
        baseline = {
            "adoptions": await session.scalar(
                select(func.count()).select_from(AdoptionRecordRow)
            ),
            "experiences": await session.scalar(
                select(func.count()).select_from(ExperienceRow)
            ),
            "events": await session.scalar(
                select(func.count()).select_from(DomainEventRow)
            ),
        }
        item = await session.get(InboxItemRow, ITEM_ID)
    assert item is not None and item.state is InboxState.ADOPTED

    different_receipt = await adopt(stack, key="second-adoption")
    same_receipt = await adopt(stack, key="first-adoption")

    assert different_receipt.status_code == 200
    assert not different_receipt.replayed
    assert (
        different_receipt.status_code,
        different_receipt.body,
        different_receipt.headers,
    ) == (first.status_code, first.body, first.headers)
    assert same_receipt.replayed
    assert (
        same_receipt.status_code,
        same_receipt.body,
        same_receipt.headers,
    ) == (first.status_code, first.body, first.headers)
    async with stack.database.read_session() as session:
        after = {
            "adoptions": await session.scalar(
                select(func.count()).select_from(AdoptionRecordRow)
            ),
            "experiences": await session.scalar(
                select(func.count()).select_from(ExperienceRow)
            ),
            "events": await session.scalar(
                select(func.count()).select_from(DomainEventRow)
            ),
        }
    assert after == baseline


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("temperature", Temperature.WARM),
        ("source_trust", 0.93),
        ("confidence", 0.91),
    ),
)
@pytest.mark.asyncio
async def test_adoption_replay_rejects_invalid_fixed_initial_policy(
    stack: AdoptionStack,
    field: str,
    invalid_value: object,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key=f"fixed-policy-{field}")).status_code == 200

    async with stack.database.transaction() as uow:
        rows = {
            row.event_type: row
            for row in (
                await uow.session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type.in_(
                            (
                                "experience.created",
                                "experience.version_created",
                                "capsule.received",
                                "capsule.adopted",
                            )
                        )
                    )
                )
            ).all()
        }
        created = stack.registry.decode(
            event_type=rows["experience.created"].event_type,
            payload=rows["experience.created"].payload,
        )
        version = stack.registry.decode(
            event_type=rows["experience.version_created"].event_type,
            payload=rows["experience.version_created"].payload,
        )
        assert isinstance(created, ExperienceCreatedV1)
        assert isinstance(version, ExperienceVersionCreatedV1)
        invalid_after = created.after.model_copy(
            update={field: invalid_value},
        )
        invalid_created = created.model_copy(update={"after": invalid_after})
        invalid_version = version.model_copy(
            update={
                "before": invalid_after,
                "after": invalid_after,
            }
        )
        adopted = stored_event(stack, rows["capsule.adopted"])
        item = await uow.session.get(InboxItemRow, ITEM_ID)
        assert item is not None

        await uow.session.execute(
            text("DROP TRIGGER domain_events_reject_update")
        )
        rows["experience.created"].payload = canonical_json_bytes(
            invalid_created
        )
        rows["experience.version_created"].payload = canonical_json_bytes(
            invalid_version
        )
        item.state = InboxState.PENDING
        item.projection_event_id = rows["capsule.received"].event_id
        await uow.session.flush()

        with pytest.raises(
            SharingProjectionIntegrityError,
            match="New adopted experience source",
        ):
            await InboxItemProjector(stack.registry).apply(
                uow.session,
                adopted,
            )


@pytest.mark.asyncio
async def test_archived_equivalent_requires_separate_restore(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    await create_archived_equivalent(stack)
    async with stack.database.read_session() as session:
        baseline_events = await session.scalar(
            select(func.count()).select_from(DomainEventRow)
        )

    result = await adopt(stack, key="archived-equivalent")

    assert result.status_code == 409
    assert error_code(result) == "restore_required"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        equivalent = await session.get(
            ExperienceStateRow,
            ADOPTED_EXPERIENCE_ID,
        )
        assert (
            await session.scalar(select(func.count()).select_from(AdoptionRecordRow))
            == 0
        )
        assert (
            await session.scalar(select(func.count()).select_from(DomainEventRow))
            == baseline_events
        )
    assert item is not None and item.state is InboxState.PENDING
    assert equivalent is not None
    assert equivalent.temperature is Temperature.ARCHIVED


@pytest.mark.parametrize(
    "checkpoint",
    (
        FaultCheckpoint.AFTER_SOURCE_INSERT,
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
        FaultCheckpoint.AFTER_RECEIPT_COMPLETION,
    ),
)
@pytest.mark.asyncio
async def test_adoption_is_atomic_at_every_command_fault_boundary(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"capsule-adoption-{checkpoint.value}.sqlite3",
    )
    try:
        await arrange_pending_capsule(stack)
        async with stack.database.read_session() as session:
            baseline = {
                "experiences": await session.scalar(
                    select(func.count()).select_from(ExperienceRow)
                ),
                "versions": await session.scalar(
                    select(func.count()).select_from(ExperienceVersionRow)
                ),
                "events": await session.scalar(
                    select(func.count()).select_from(DomainEventRow)
                ),
            }
        stack.fault.checkpoint = checkpoint

        with pytest.raises(InjectedFailure, match=checkpoint.value):
            await adopt(stack, key=f"fault-{checkpoint.value}")

        async with stack.database.read_session() as session:
            after_failure = {
                "experiences": await session.scalar(
                    select(func.count()).select_from(ExperienceRow)
                ),
                "versions": await session.scalar(
                    select(func.count()).select_from(ExperienceVersionRow)
                ),
                "events": await session.scalar(
                    select(func.count()).select_from(DomainEventRow)
                ),
            }
            item = await session.get(InboxItemRow, ITEM_ID)
            assert (
                await session.scalar(
                    select(func.count()).select_from(AdoptionRecordRow)
                )
                == 0
            )
        assert after_failure == baseline
        assert item is not None and item.state is InboxState.PENDING

        stack.fault.checkpoint = None
        retry = await adopt(stack, key=f"fault-{checkpoint.value}")
        assert retry.status_code == 200
        assert json.loads(retry.body)["data"]["created"] is True
    finally:
        await stack.database.dispose()
