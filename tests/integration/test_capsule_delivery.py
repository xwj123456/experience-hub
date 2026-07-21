from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import delete, func, select, text, update

from experience_hub.agents import AgentCreated
from experience_hub.clock import FrozenClock
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    StoredEvent,
)
from experience_hub.experiences import ExperienceKind, VersionContent
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.events import register_experience_events
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.sharing.events import (
    CapsulePublishedV1,
    CapsuleReceivedV1,
    register_sharing_events,
)
from experience_hub.sharing.hashing import compute_capsule_hash
from experience_hub.sharing.models import (
    Capsule,
    CapsuleStatus,
    CreateSubscription,
    CreateTopic,
    InboxState,
    PublishCapsule,
    Subscription,
)
from experience_hub.sharing.projector import (
    CapsuleStateProjector,
    InboxItemProjector,
    SharingProjectionIntegrityError,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.service import SharingService
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
    ProjectionMismatch,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AgentRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    InboxItemRow,
    SubscriptionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_experience_source_validator,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
PUBLISHER_ID = UUID("00000000-0000-0000-0000-000000000101")
RECIPIENT_LOW_ID = UUID("00000000-0000-0000-0000-000000000102")
RECIPIENT_HIGH_ID = UUID("00000000-0000-0000-0000-000000000104")
EQUAL_CUTOFF_ID = UUID("00000000-0000-0000-0000-000000000103")

EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000201")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000301")
TOPIC_ID = UUID("00000000-0000-0000-0000-000000000401")
HIGH_SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000402")
PUBLISHER_SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000403")
LOW_SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000404")
CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000405")
LOW_ITEM_ID = UUID("00000000-0000-0000-0000-000000000406")
HIGH_ITEM_ID = UUID("00000000-0000-0000-0000-000000000407")
EQUAL_CUTOFF_SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000499")

RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(1001, 1051)
)
SHARING_IDS = (
    TOPIC_ID,
    HIGH_SUBSCRIPTION_ID,
    PUBLISHER_SUBSCRIPTION_ID,
    LOW_SUBSCRIPTION_ID,
    CAPSULE_ID,
    LOW_ITEM_ID,
    HIGH_ITEM_ID,
)


@dataclass(slots=True)
class DeliveryStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    experience_service: ExperienceService
    sharing_service: SharingService
    sharing_repository: SharingRepository
    registry: EventRegistry
    capsule_projector: CapsuleStateProjector
    inbox_projector: InboxItemProjector
    projection_manager: ProjectionManager
    fault: FailAt


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    def __init__(self) -> None:
        self.checkpoint: FaultCheckpoint | None = None

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is self.checkpoint:
            raise InjectedFailure(checkpoint.value)


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> DeliveryStack:
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
    manager = ProjectionManager(
        ProjectionRegistry((experience_projector, capsule_projector, inbox_projector)),
        source_validator=source_validator,
    )
    fault = FailAt()
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
        fault_injector=fault,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            [
                AgentRow(
                    agent_id=agent_id,
                    name=f"Agent {agent_id.int}",
                    created_at=NOW,
                )
                for agent_id in (
                    PUBLISHER_ID,
                    RECIPIENT_LOW_ID,
                    EQUAL_CUTOFF_ID,
                    RECIPIENT_HIGH_ID,
                )
            ]
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    executor = CommandExecutor(
        database=database,
        receipt_store=receipts,
        clock=clock,
    )
    experience_repository = ExperienceRepository(event_registry=registry)
    experience_writer = ExperienceWriter(
        id_generator=SequenceIdGenerator((EXPERIENCE_ID, VERSION_ID)),
        repository=experience_repository,
        lifecycle_config=lifecycle,
    )
    experience_service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=experience_writer,
        lifecycle_config=lifecycle,
    )
    sharing_repository = SharingRepository()
    sharing_service = SharingService(
        clock=clock,
        id_generator=SequenceIdGenerator(SHARING_IDS),
        receipt_store=receipts,
        repository=sharing_repository,
        experience_query=ExperienceQuery(event_registry=registry),
    )
    return DeliveryStack(
        database=database,
        clock=clock,
        executor=executor,
        experience_service=experience_service,
        sharing_service=sharing_service,
        sharing_repository=sharing_repository,
        registry=registry,
        capsule_projector=capsule_projector,
        inbox_projector=inbox_projector,
        projection_manager=manager,
        fault=fault,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[DeliveryStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-delivery.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def _request(
    *,
    key: str,
    operation_scope: str,
    route_template: str,
    body: dict[str, Any],
    agent_id: UUID = PUBLISHER_ID,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope=operation_scope,
        idempotency_key=key,
        method="POST",
        route_template=route_template,
        path_parameters={"agent_id": agent_id},
        body=body,
    )


async def create_experience(stack: DeliveryStack) -> None:
    content = VersionContent(
        body="Keep exact delivery identities across projection rebuilds.",
        summary="Stable capsule delivery",
        mechanism="Allocate the route identity in the received event.",
        tags=("delivery", "memory"),
        applicability=("subscribed recipient",),
        evidence=(),
        falsifiers=("replay changes the inbox item ID",),
    )
    command = CreateExperience(
        owner_agent_id=PUBLISHER_ID,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        importance=0.7,
        confidence=0.8,
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
        _request(
            key="experience",
            operation_scope="experience.create",
            route_template="/v1/experiences",
            body={"summary": content.summary},
        ),
        handler,
    )
    assert result.status_code == 201


async def create_topic(stack: DeliveryStack) -> UUID:
    command = CreateTopic(
        owner_agent_id=PUBLISHER_ID,
        name="Delivery semantics",
        description=None,
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
        _request(
            key="topic",
            operation_scope="topic.create",
            route_template="/v1/topics",
            body={"name": command.name},
        ),
        handler,
    )
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["topic_id"])


async def subscribe(
    stack: DeliveryStack,
    *,
    subscriber_agent_id: UUID,
    key: str,
) -> None:
    command = CreateSubscription(
        subscriber_agent_id=subscriber_agent_id,
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
        _request(
            key=key,
            operation_scope="subscription.create",
            route_template="/v1/agents/{agent_id}/subscriptions",
            body={"topic_id": TOPIC_ID},
            agent_id=subscriber_agent_id,
        ),
        handler,
    )
    assert result.status_code == 201


async def arrange_publishable_graph(stack: DeliveryStack) -> None:
    await create_experience(stack)
    assert await create_topic(stack) == TOPIC_ID
    # Deliberately insert subscribers in the opposite of delivery order and
    # include the publisher as a subscriber.
    await subscribe(
        stack,
        subscriber_agent_id=RECIPIENT_HIGH_ID,
        key="subscribe-high",
    )
    await subscribe(
        stack,
        subscriber_agent_id=PUBLISHER_ID,
        key="subscribe-publisher",
    )
    await subscribe(
        stack,
        subscriber_agent_id=RECIPIENT_LOW_ID,
        key="subscribe-low",
    )


async def publish(
    stack: DeliveryStack,
    *,
    key: str = "publish",
) -> CommandResult:
    command = PublishCapsule(
        owner_agent_id=PUBLISHER_ID,
        topic_id=TOPIC_ID,
        experience_id=EXPERIENCE_ID,
        version_id=VERSION_ID,
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

    return await stack.executor.execute(
        _request(
            key=key,
            operation_scope="capsule.publish",
            route_template="/v1/capsules",
            body={
                "topic_id": TOPIC_ID,
                "experience_id": EXPERIENCE_ID,
                "version_id": VERSION_ID,
                "expires_at": command.expires_at,
            },
        ),
        handler,
    )


def _stored_event(
    *,
    stack: DeliveryStack,
    row: DomainEventRow,
) -> StoredEvent:
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


def test_publication_and_receipt_payloads_are_strict_body_free_v1() -> None:
    published = CapsulePublishedV1(
        schema_version=1,
        capsule_id=CAPSULE_ID,
        topic_id=TOPIC_ID,
        source_experience_id=EXPERIENCE_ID,
        source_version_id=VERSION_ID,
        publisher_agent_id=PUBLISHER_ID,
        capsule_hash="a" * 64,
        root_fingerprint="b" * 64,
        status_after=CapsuleStatus.ACTIVE,
    )
    received = CapsuleReceivedV1(
        schema_version=1,
        item_id=LOW_ITEM_ID,
        capsule_id=CAPSULE_ID,
        recipient_agent_id=RECIPIENT_LOW_ID,
        state_after=InboxState.PENDING,
    )

    assert published.model_dump(mode="json") == {
        "schema_version": 1,
        "capsule_id": str(CAPSULE_ID),
        "topic_id": str(TOPIC_ID),
        "source_experience_id": str(EXPERIENCE_ID),
        "source_version_id": str(VERSION_ID),
        "publisher_agent_id": str(PUBLISHER_ID),
        "capsule_hash": "a" * 64,
        "root_fingerprint": "b" * 64,
        "status_after": "active",
    }
    assert received.model_dump(mode="json") == {
        "schema_version": 1,
        "item_id": str(LOW_ITEM_ID),
        "capsule_id": str(CAPSULE_ID),
        "recipient_agent_id": str(RECIPIENT_LOW_ID),
        "state_after": "pending",
    }
    assert CapsulePublishedV1.event_type == "capsule.published"
    assert CapsuleReceivedV1.event_type == "capsule.received"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CapsulePublishedV1.model_validate(
            {**published.model_dump(), "body": "must stay in the source row"}
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CapsuleReceivedV1.model_validate(
            {**received.model_dump(), "query": "must never enter the event"}
        )


@pytest.mark.asyncio
async def test_publish_delivers_in_recipient_order_with_strict_cutoff(
    stack: DeliveryStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await arrange_publishable_graph(stack)
    original = stack.sharing_repository.list_eligible_subscriptions

    async def insert_equal_cutoff_then_list(
        *,
        session: Any,
        topic_id: UUID,
        publication_event_id: int,
        after: tuple[UUID, UUID] | None = None,
        limit: int = 100,
        exclude_subscriber_agent_id: UUID | None = None,
    ) -> tuple[Subscription, ...]:
        session.add(
            SubscriptionRow(
                subscription_id=EQUAL_CUTOFF_SUBSCRIPTION_ID,
                subscriber_agent_id=EQUAL_CUTOFF_ID,
                topic_id=topic_id,
                creation_event_id=publication_event_id,
                created_at=NOW,
            )
        )
        await session.flush()
        return await original(
            session=session,
            topic_id=topic_id,
            publication_event_id=publication_event_id,
            after=after,
            limit=limit,
            exclude_subscriber_agent_id=exclude_subscriber_agent_id,
        )

    monkeypatch.setattr(
        stack.sharing_repository,
        "list_eligible_subscriptions",
        insert_equal_cutoff_then_list,
    )

    result = await publish(stack)
    assert result.status_code == 201
    replay = await publish(stack)
    assert replay.replayed
    assert (replay.status_code, replay.body, replay.headers) == (
        result.status_code,
        result.body,
        result.headers,
    )

    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            ("capsule.published", "capsule.received")
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
        items = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).order_by(InboxItemRow.recipient_agent_id)
                )
            ).all()
        )

    assert [row.event_type for row in rows] == [
        "capsule.published",
        "capsule.received",
        "capsule.received",
    ]
    payloads = [
        stack.registry.decode(event_type=row.event_type, payload=row.payload)
        for row in rows
    ]
    assert isinstance(payloads[0], CapsulePublishedV1)
    receipts = payloads[1:]
    assert all(isinstance(payload, CapsuleReceivedV1) for payload in receipts)
    assert [
        (payload.recipient_agent_id, payload.item_id)
        for payload in receipts
        if isinstance(payload, CapsuleReceivedV1)
    ] == [
        (RECIPIENT_LOW_ID, LOW_ITEM_ID),
        (RECIPIENT_HIGH_ID, HIGH_ITEM_ID),
    ]
    assert [
        row.aggregate_id for row in rows if row.event_type == "capsule.received"
    ] == [LOW_ITEM_ID, HIGH_ITEM_ID]
    assert all(
        b'"body"' not in row.payload
        and b'"query"' not in row.payload
        and b'"error"' not in row.payload
        for row in rows
    )

    assert state is not None
    assert (state.status, state.projection_event_id) == (
        CapsuleStatus.ACTIVE,
        rows[0].event_id,
    )
    assert [
        (
            item.item_id,
            item.recipient_agent_id,
            item.capsule_id,
            item.state,
            item.projection_event_id,
        )
        for item in items
    ] == [
        (
            LOW_ITEM_ID,
            RECIPIENT_LOW_ID,
            CAPSULE_ID,
            InboxState.PENDING,
            rows[1].event_id,
        ),
        (
            HIGH_ITEM_ID,
            RECIPIENT_HIGH_ID,
            CAPSULE_ID,
            InboxState.PENDING,
            rows[2].event_id,
        ),
    ]
    assert PUBLISHER_ID not in {item.recipient_agent_id for item in items}
    assert EQUAL_CUTOFF_ID not in {item.recipient_agent_id for item in items}


@pytest.mark.asyncio
async def test_fanout_reuses_one_validated_capsule_per_inbox_session(
    stack: DeliveryStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await arrange_publishable_graph(stack)
    calls = 0
    original = compute_capsule_hash

    def counting_hash(**values: Any) -> str:
        nonlocal calls
        calls += 1
        return original(**values)

    monkeypatch.setattr(
        "experience_hub.sharing.projector.compute_capsule_hash",
        counting_hash,
    )

    assert (await publish(stack)).status_code == 201
    # Once for capsule-state creation and once for the first inbox receipt.
    # The second recipient reuses the immutable publication validation.
    assert calls == 2


@pytest.mark.asyncio
async def test_projection_replay_recreates_event_allocated_item_ids(
    stack: DeliveryStack,
) -> None:
    await arrange_publishable_graph(stack)
    assert (await publish(stack)).status_code == 201

    async with stack.database.transaction() as uow:
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            ("capsule.published", "capsule.received")
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        events = tuple(_stored_event(stack=stack, row=row) for row in rows)
        await uow.session.execute(delete(InboxItemRow))
        await uow.session.execute(delete(CapsuleStateRow))
        await uow.session.flush()
        for event in events:
            await stack.capsule_projector.apply(uow.session, event)
            await stack.inbox_projector.apply(uow.session, event)

    async with stack.database.read_session() as session:
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
        items = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).order_by(InboxItemRow.recipient_agent_id)
                )
            ).all()
        )

    assert state is not None and state.projection_event_id == events[0].event_id
    assert [item.item_id for item in items] == [LOW_ITEM_ID, HIGH_ITEM_ID]
    assert [item.projection_event_id for item in items] == [
        events[1].event_id,
        events[2].event_id,
    ]


@pytest.mark.asyncio
async def test_temp_rebuilds_are_independent_and_repair_exact_identities(
    stack: DeliveryStack,
) -> None:
    await arrange_publishable_graph(stack)
    assert (await publish(stack)).status_code == 201

    for reducer in (stack.capsule_projector, stack.inbox_projector):
        validator = SourceValidator(stack.registry)
        register_experience_source_validator(validator)
        manager = ProjectionManager(
            ProjectionRegistry((reducer,)),
            source_validator=validator,
        )
        assert (await manager.verify(stack.database)).matches

    async with stack.database.transaction() as uow:
        await uow.session.execute(delete(InboxItemRow))
        await uow.session.execute(delete(CapsuleStateRow))

    with pytest.raises(ProjectionMismatch):
        await stack.projection_manager.verify(stack.database)

    report = await stack.projection_manager.repair(stack.database)
    assert report.matches
    async with stack.database.read_session() as session:
        state = await session.get(CapsuleStateRow, CAPSULE_ID)
        items = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).order_by(InboxItemRow.recipient_agent_id)
                )
            ).all()
        )
    assert state is not None and state.capsule_id == CAPSULE_ID
    assert [item.item_id for item in items] == [LOW_ITEM_ID, HIGH_ITEM_ID]


@pytest.mark.asyncio
async def test_creation_reducers_reject_existing_state_and_wrong_aggregate(
    stack: DeliveryStack,
) -> None:
    await arrange_publishable_graph(stack)
    assert (await publish(stack)).status_code == 201
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            ("capsule.published", "capsule.received")
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    published = _stored_event(stack=stack, row=rows[0])
    received = _stored_event(stack=stack, row=rows[1])

    async with stack.database.transaction() as uow:
        with pytest.raises(
            SharingProjectionIntegrityError,
            match="already exists|before|absence",
        ):
            await stack.capsule_projector.apply(uow.session, published)
        with pytest.raises(
            SharingProjectionIntegrityError,
            match="already exists|before|absence",
        ):
            await stack.inbox_projector.apply(uow.session, received)
        with pytest.raises(
            SharingProjectionIntegrityError,
            match="aggregate anchor",
        ):
            await stack.capsule_projector.apply(
                uow.session,
                replace(
                    published,
                    aggregate_id=RECIPIENT_LOW_ID,
                ),
            )
        with pytest.raises(
            SharingProjectionIntegrityError,
            match="aggregate anchor",
        ):
            await stack.inbox_projector.apply(
                uow.session,
                replace(
                    received,
                    aggregate_id=RECIPIENT_LOW_ID,
                ),
            )


@pytest.mark.asyncio
async def test_capsule_reducer_rejects_semantic_source_hash_divergence(
    stack: DeliveryStack,
) -> None:
    await arrange_publishable_graph(stack)
    result = await publish(stack)
    capsule = Capsule.model_validate(
        json.loads(result.body)["data"],
        strict=False,
    )
    tampered_body = f"{capsule.body} Tampered after publication."
    tampered_hash = compute_capsule_hash(
        transport_schema_version=capsule.transport_schema_version,
        capsule_id=capsule.capsule_id,
        topic_id=capsule.topic_id,
        source_experience_id=capsule.source_experience_id,
        source_version_id=capsule.source_version_id,
        publisher_agent_id=capsule.publisher_agent_id,
        kind=capsule.kind,
        body=tampered_body,
        summary=capsule.summary,
        mechanism=capsule.mechanism,
        tags=capsule.tags,
        applicability=capsule.applicability,
        evidence=capsule.evidence,
        falsifiers=capsule.falsifiers,
        publisher_confidence=capsule.publisher_confidence,
        provenance_chain=capsule.provenance_chain,
        root_fingerprint=capsule.root_fingerprint,
        source_content_hash=capsule.source_content_hash,
        created_at=capsule.created_at,
        expires_at=capsule.expires_at,
        hop_count=capsule.hop_count,
    )

    async with stack.database.transaction() as uow:
        row = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsulePublishedV1.event_type
            )
        )
        assert row is not None
        event = _stored_event(stack=stack, row=row)
        assert isinstance(event.payload, CapsulePublishedV1)
        await uow.session.execute(
            text("DROP TRIGGER experience_capsules_reject_update")
        )
        await uow.session.execute(
            update(ExperienceCapsuleRow)
            .where(ExperienceCapsuleRow.capsule_id == capsule.capsule_id)
            .values(body=tampered_body, capsule_hash=tampered_hash)
        )
        await uow.session.execute(delete(CapsuleStateRow))
        await uow.session.flush()

        with pytest.raises(
            SharingProjectionIntegrityError,
            match="semantic hash",
        ):
            await stack.capsule_projector.apply(
                uow.session,
                replace(
                    event,
                    payload=event.payload.model_copy(
                        update={"capsule_hash": tampered_hash}
                    ),
                ),
            )


@pytest.mark.asyncio
async def test_inbox_reducer_rejects_unanchored_subscription_cutoff(
    stack: DeliveryStack,
) -> None:
    await arrange_publishable_graph(stack)
    async with stack.database.transaction() as uow:
        unrelated_event_id = await uow.session.scalar(
            select(func.min(DomainEventRow.event_id)).where(
                DomainEventRow.event_type != "subscription.created"
            )
        )
        assert unrelated_event_id is not None
        await uow.session.execute(text("DROP TRIGGER subscriptions_reject_update"))
        await uow.session.execute(
            update(SubscriptionRow)
            .where(SubscriptionRow.subscription_id == HIGH_SUBSCRIPTION_ID)
            .values(creation_event_id=unrelated_event_id)
        )

    with pytest.raises(
        SharingProjectionIntegrityError,
        match="subscription event",
    ):
        await publish(stack)

    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceCapsuleRow))
            == 0
        )
        assert await session.scalar(select(func.count()).select_from(InboxItemRow)) == 0


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
async def test_publication_and_all_deliveries_roll_back_as_one_command(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"delivery-{checkpoint.value}.sqlite3",
    )
    try:
        await arrange_publishable_graph(stack)
        async with stack.database.read_session() as session:
            baseline_events = await session.scalar(
                select(func.count()).select_from(DomainEventRow)
            )
        stack.fault.checkpoint = checkpoint

        with pytest.raises(InjectedFailure, match=checkpoint.value):
            await publish(stack, key=f"fault-{checkpoint.value}")

        async with stack.database.read_session() as session:
            assert (
                await session.scalar(
                    select(func.count()).select_from(ExperienceCapsuleRow)
                )
                == 0
            )
            assert (
                await session.scalar(select(func.count()).select_from(CapsuleStateRow))
                == 0
            )
            assert (
                await session.scalar(select(func.count()).select_from(InboxItemRow))
                == 0
            )
            assert (
                await session.scalar(select(func.count()).select_from(DomainEventRow))
                == baseline_events
            )
    finally:
        await stack.database.dispose()
