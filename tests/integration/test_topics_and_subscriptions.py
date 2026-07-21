from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select

from experience_hub.agents import AgentCreated
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import CommandContext, CommandRequest, EventRegistry
from experience_hub.experiences import ExperienceKind, ExperienceOrigin
from experience_hub.ids import SequenceIdGenerator
from experience_hub.sharing.events import register_sharing_events
from experience_hub.sharing.models import (
    CapsuleStatus,
    CreateSubscription,
    CreateTopic,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.service import SharingService
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CommandExecutor,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import (
    AgentRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
    InboxItemRow,
    SubscriptionRow,
    TopicRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 18, 9, 45, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000102")
SUBSCRIBER_IDS = (
    UUID("00000000-0000-0000-0000-000000000103"),
    UUID("00000000-0000-0000-0000-000000000104"),
    UUID("00000000-0000-0000-0000-000000000105"),
)
MISSING_AGENT_ID = UUID("00000000-0000-0000-0000-000000000199")
HISTORICAL_EXPERIENCE_ID = UUID(
    "00000000-0000-0000-0000-000000000801"
)
HISTORICAL_VERSION_ID = UUID("00000000-0000-0000-0000-000000000802")
HISTORICAL_CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000803")

RESOURCE_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}")
    for value in (
        903,
        901,
        902,
        913,
        911,
        912,
        923,
        921,
        922,
        933,
        931,
        932,
    )
)
RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(1001, 1051)
)


@dataclass(slots=True)
class Stack:
    database: Database
    executor: CommandExecutor
    repository: SharingRepository
    service: SharingService
    registry: EventRegistry


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    def __init__(self, checkpoint: FaultCheckpoint) -> None:
        self.checkpoint = checkpoint

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is self.checkpoint:
            raise InjectedFailure(checkpoint.value)


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
    fault: FailAt | None = None,
) -> Stack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    register_sharing_events(registry)
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        fault_injector=fault,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            [
                AgentRow(agent_id=OWNER_ID, name="Owner", created_at=NOW),
                AgentRow(
                    agent_id=OTHER_OWNER_ID,
                    name="Other Owner",
                    created_at=NOW,
                ),
                *(
                    AgentRow(
                        agent_id=agent_id,
                        name=f"Subscriber {index}",
                        created_at=NOW,
                    )
                    for index, agent_id in enumerate(SUBSCRIBER_IDS, start=1)
                ),
            ]
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    repository = SharingRepository()
    return Stack(
        database=database,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        repository=repository,
        service=SharingService(
            clock=clock,
            id_generator=SequenceIdGenerator(RESOURCE_IDS),
            receipt_store=receipts,
            repository=repository,
        ),
        registry=registry,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "sharing-topics.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def topic_request(
    *,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
    name: str,
    description: str | None,
    operation_scope: str = "topic.create",
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=operation_scope,
        idempotency_key=key,
        method="POST",
        route_template="/v1/topics",
        body={"name": name, "description": description},
    )


def subscription_request(
    *,
    key: str,
    subscriber_agent_id: UUID,
    topic_id: UUID,
    operation_scope: str = "subscription.create",
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{subscriber_agent_id}",
        operation_scope=operation_scope,
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/subscriptions",
        path_parameters={"agent_id": subscriber_agent_id},
        body={"topic_id": topic_id},
    )


async def create_topic(
    stack: Stack,
    *,
    key: str,
    name: str,
    description: str | None = None,
    owner_agent_id: UUID = OWNER_ID,
    caller_agent_id: UUID | None = None,
    operation_scope: str = "topic.create",
) -> tuple[int, dict[str, Any], bool]:
    command = CreateTopic(
        owner_agent_id=owner_agent_id,
        name=name,
        description=description,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create_topic(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        topic_request(
            key=key,
            owner_agent_id=(
                owner_agent_id
                if caller_agent_id is None
                else caller_agent_id
            ),
            name=name,
            description=description,
            operation_scope=operation_scope,
        ),
        handler,
    )
    return result.status_code, json.loads(result.body), result.replayed


async def create_subscription(
    stack: Stack,
    *,
    key: str,
    subscriber_agent_id: UUID,
    topic_id: UUID,
    caller_agent_id: UUID | None = None,
    operation_scope: str = "subscription.create",
) -> tuple[int, dict[str, Any], bool]:
    command = CreateSubscription(
        subscriber_agent_id=subscriber_agent_id,
        topic_id=topic_id,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create_subscription(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        subscription_request(
            key=key,
            subscriber_agent_id=(
                subscriber_agent_id
                if caller_agent_id is None
                else caller_agent_id
            ),
            topic_id=topic_id,
            operation_scope=operation_scope,
        ),
        handler,
    )
    return result.status_code, json.loads(result.body), result.replayed


async def seed_historical_capsule(stack: Stack, *, topic_id: UUID) -> None:
    """Insert a pre-subscription source that Task 2 must never route."""
    content_hash = "a" * 64
    async with stack.database.transaction() as uow:
        uow.session.add(
            ExperienceRow(
                experience_id=HISTORICAL_EXPERIENCE_ID,
                owner_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                created_at=NOW - timedelta(minutes=10),
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceVersionRow(
                version_id=HISTORICAL_VERSION_ID,
                experience_id=HISTORICAL_EXPERIENCE_ID,
                version_number=1,
                summary="Historical capsule",
                mechanism="Published before the subscription existed.",
                tags=canonical_json_bytes(("historical",)),
                applicability=canonical_json_bytes(()),
                evidence=canonical_json_bytes(()),
                falsifiers=canonical_json_bytes(()),
                content_hash=content_hash,
                supersedes_version_id=None,
                created_at=NOW - timedelta(minutes=10),
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceCapsuleRow(
                capsule_id=HISTORICAL_CAPSULE_ID,
                transport_schema_version=1,
                topic_id=topic_id,
                source_experience_id=HISTORICAL_EXPERIENCE_ID,
                source_version_id=HISTORICAL_VERSION_ID,
                publisher_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                body="Historical body",
                summary="Historical capsule",
                mechanism="Published before the subscription existed.",
                tags=canonical_json_bytes(("historical",)),
                applicability=canonical_json_bytes(()),
                evidence=canonical_json_bytes(()),
                falsifiers=canonical_json_bytes(()),
                publisher_confidence=0.5,
                provenance_chain=canonical_json_bytes(()),
                root_fingerprint="b" * 64,
                source_content_hash=content_hash,
                created_at=NOW - timedelta(minutes=5),
                expires_at=NOW + timedelta(days=1),
                hop_count=0,
                capsule_hash="c" * 64,
            )
        )
        await uow.session.flush()
        topic_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_type == "topic",
                DomainEventRow.aggregate_id == topic_id,
                DomainEventRow.event_type == "topic.created",
            )
        )
        assert topic_event is not None
        published_at = NOW - timedelta(minutes=5)
        publication = DomainEventRow(
            aggregate_type="capsule",
            aggregate_id=HISTORICAL_CAPSULE_ID,
            sequence=1,
            event_type="capsule.published",
            payload=canonical_json_bytes(
                {
                    "schema_version": 1,
                    "capsule_id": HISTORICAL_CAPSULE_ID,
                    "topic_id": topic_id,
                    "source_experience_id": HISTORICAL_EXPERIENCE_ID,
                    "source_version_id": HISTORICAL_VERSION_ID,
                    "publisher_agent_id": OWNER_ID,
                    "capsule_hash": "c" * 64,
                    "root_fingerprint": "b" * 64,
                    "status_after": CapsuleStatus.ACTIVE,
                }
            ),
            actor_agent_id=OWNER_ID,
            causation_id=topic_event.causation_id,
            occurred_at=published_at,
        )
        uow.session.add(publication)
        await uow.session.flush()
        uow.session.add(
            CapsuleStateRow(
                capsule_id=HISTORICAL_CAPSULE_ID,
                status=CapsuleStatus.ACTIVE,
                projection_event_id=publication.event_id,
            )
        )


@pytest.mark.asyncio
async def test_topic_is_trimmed_persisted_before_event_and_replayed_once(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_append = UnitOfWork.append_events
    observed_source_before_event = False

    async def append_after_source(
        uow: UnitOfWork,
        command: CommandContext,
        events: Any,
    ) -> Any:
        nonlocal observed_source_before_event
        if events and events[0].event_type == "topic.created":
            source = await uow.session.get(TopicRow, events[0].aggregate_id)
            observed_source_before_event = source is not None
        return await original_append(uow, command, events)

    monkeypatch.setattr(UnitOfWork, "append_events", append_after_source)

    first = await create_topic(
        stack,
        key="topic-trim",
        name="  Distributed Skills  ",
        description="  Experiences that survive transport.  ",
    )
    replay = await create_topic(
        stack,
        key="topic-trim",
        name="  Distributed Skills  ",
        description="  Experiences that survive transport.  ",
    )

    assert first == (
        201,
        {
            "data": {
                "created_at": "2026-07-18T09:45:00.000000Z",
                "description": "Experiences that survive transport.",
                "name": "Distributed Skills",
                "owner_agent_id": str(OWNER_ID),
                "topic_id": str(RESOURCE_IDS[0]),
            }
        },
        False,
    )
    assert replay == (first[0], first[1], True)
    assert observed_source_before_event

    async with stack.database.read_session() as session:
        topic = await session.get(TopicRow, RESOURCE_IDS[0])
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type == "topic.created"
                    )
                )
            ).all()
        )
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[0])

    assert topic is not None
    assert (
        topic.owner_agent_id,
        topic.name,
        topic.description,
        topic.created_at,
    ) == (
        OWNER_ID,
        "Distributed Skills",
        "Experiences that survive transport.",
        NOW,
    )
    assert len(events) == 1
    assert (
        events[0].aggregate_type,
        events[0].aggregate_id,
        events[0].sequence,
        events[0].actor_agent_id,
        events[0].occurred_at,
    ) == ("topic", RESOURCE_IDS[0], 1, OWNER_ID, NOW)
    decoded = stack.registry.decode(
        event_type=events[0].event_type,
        payload=events[0].payload,
    )
    assert decoded.model_dump(mode="json") == {
        "schema_version": 1,
        "topic_id": str(RESOURCE_IDS[0]),
        "owner_agent_id": str(OWNER_ID),
        "name": "Distributed Skills",
        "description": "Experiences that survive transport.",
    }
    assert receipt is not None
    assert (receipt.result_resource_type, receipt.result_resource_id) == (
        "topic",
        RESOURCE_IDS[0],
    )


@pytest.mark.asyncio
async def test_topic_optional_description_owner_and_trimmed_name_uniqueness(
    stack: Stack,
) -> None:
    created = await create_topic(
        stack,
        key="unique-topic",
        name="  Safe Replay  ",
    )
    duplicate = await create_topic(
        stack,
        key="duplicate-topic",
        name="Safe Replay",
        owner_agent_id=OWNER_ID,
    )
    other_owner_duplicate = await create_topic(
        stack,
        key="other-owner-duplicate-topic",
        name=" Safe Replay ",
        owner_agent_id=OTHER_OWNER_ID,
    )
    missing_owner = await create_topic(
        stack,
        key="missing-owner-topic",
        name="Orphaned Topic",
        owner_agent_id=MISSING_AGENT_ID,
    )

    assert created[0] == 201
    assert created[1]["data"]["description"] is None
    assert duplicate[0] == 409
    assert duplicate[1]["error"]["code"] == "topic_name_conflict"
    assert duplicate[1]["error"]["details"] == {"name": "Safe Replay"}
    # The schema declares a global topic namespace, not one namespace per owner.
    assert other_owner_duplicate[0] == 409
    assert other_owner_duplicate[1] == duplicate[1]
    assert missing_owner[0] == 404
    assert missing_owner[1]["error"]["code"] == "agent_not_found"

    async with stack.database.read_session() as session:
        assert await session.scalar(select(func.count()).select_from(TopicRow)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "topic.created")
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.state == "completed")
            )
            == 4
        )


@pytest.mark.asyncio
async def test_subscription_appends_event_before_source_captures_event_id_and_replays(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic_status, topic_body, _ = await create_topic(
        stack,
        key="subscription-topic",
        name="Non-backfilling",
    )
    assert topic_status == 201
    topic_id = UUID(topic_body["data"]["topic_id"])
    subscription_id = RESOURCE_IDS[1]

    original_append = UnitOfWork.append_events
    observed_absent_before_and_after_append = False

    async def append_before_source(
        uow: UnitOfWork,
        command: CommandContext,
        events: Any,
    ) -> Any:
        nonlocal observed_absent_before_and_after_append
        if events and events[0].event_type == "subscription.created":
            assert (
                await uow.session.get(
                    SubscriptionRow,
                    events[0].aggregate_id,
                )
                is None
            )
            stored = await original_append(uow, command, events)
            assert (
                await uow.session.get(
                    SubscriptionRow,
                    events[0].aggregate_id,
                )
                is None
            )
            observed_absent_before_and_after_append = True
            return stored
        return await original_append(uow, command, events)

    monkeypatch.setattr(UnitOfWork, "append_events", append_before_source)

    first = await create_subscription(
        stack,
        key="subscription-create",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=topic_id,
    )
    replay = await create_subscription(
        stack,
        key="subscription-create",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=topic_id,
    )

    assert first[0] == 201 and replay == (first[0], first[1], True)
    assert observed_absent_before_and_after_append
    assert first[1]["data"] == {
        "created_at": "2026-07-18T09:45:00.000000Z",
        "creation_event_id": first[1]["data"]["creation_event_id"],
        "subscriber_agent_id": str(SUBSCRIBER_IDS[0]),
        "subscription_id": str(subscription_id),
        "topic_id": str(topic_id),
    }

    async with stack.database.read_session() as session:
        source = await session.get(SubscriptionRow, subscription_id)
        event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "subscription.created"
            )
        )
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[1])

    assert source is not None and event is not None
    assert (
        source.subscriber_agent_id,
        source.topic_id,
        source.creation_event_id,
        source.created_at,
    ) == (
        SUBSCRIBER_IDS[0],
        topic_id,
        event.event_id,
        NOW,
    )
    assert first[1]["data"]["creation_event_id"] == event.event_id
    assert (
        event.aggregate_type,
        event.aggregate_id,
        event.sequence,
        event.actor_agent_id,
        event.occurred_at,
    ) == ("subscription", subscription_id, 1, SUBSCRIBER_IDS[0], NOW)
    decoded = stack.registry.decode(
        event_type=event.event_type,
        payload=event.payload,
    )
    assert decoded.model_dump(mode="json") == {
        "schema_version": 1,
        "subscription_id": str(subscription_id),
        "subscriber_agent_id": str(SUBSCRIBER_IDS[0]),
        "topic_id": str(topic_id),
    }
    assert receipt is not None
    assert (receipt.result_resource_type, receipt.result_resource_id) == (
        "subscription",
        subscription_id,
    )


@pytest.mark.asyncio
async def test_duplicate_subscription_is_stable_conflict_and_does_not_backfill(
    stack: Stack,
) -> None:
    _, topic_body, _ = await create_topic(
        stack,
        key="duplicate-subscription-topic",
        name="Immutable Delivery Cutoff",
    )
    topic_id = UUID(topic_body["data"]["topic_id"])
    await seed_historical_capsule(stack, topic_id=topic_id)
    first = await create_subscription(
        stack,
        key="subscribe-once",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=topic_id,
    )
    duplicate = await create_subscription(
        stack,
        key="subscribe-twice",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=topic_id,
    )
    duplicate_replay = await create_subscription(
        stack,
        key="subscribe-twice",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=topic_id,
    )

    assert first[0] == 201
    assert duplicate == (
        409,
        {
            "error": {
                "code": "already_subscribed",
                "details": {
                    "subscriber_agent_id": str(SUBSCRIBER_IDS[0]),
                    "topic_id": str(topic_id),
                },
                "message": "The agent is already subscribed to this topic",
            }
        },
        False,
    )
    assert duplicate_replay == (duplicate[0], duplicate[1], True)

    creation_event_id = first[1]["data"]["creation_event_id"]
    async with stack.database.read_session() as session:
        at_cutoff = await stack.repository.list_eligible_subscriptions(
            session=session,
            topic_id=topic_id,
            publication_event_id=creation_event_id,
            after=None,
            limit=100,
        )
        after_cutoff = await stack.repository.list_eligible_subscriptions(
            session=session,
            topic_id=topic_id,
            publication_event_id=creation_event_id + 1,
            after=None,
            limit=100,
        )
        inbox_count = await session.scalar(
            select(func.count()).select_from(InboxItemRow)
        )
        capsule_count = await session.scalar(
            select(func.count()).select_from(ExperienceCapsuleRow)
        )
        capsule_state_count = await session.scalar(
            select(func.count()).select_from(CapsuleStateRow)
        )
        subscription_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "subscription.created")
        )
        received_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "capsule.received")
        )

    assert at_cutoff == ()
    assert tuple(item.subscription_id for item in after_cutoff) == (
        UUID(first[1]["data"]["subscription_id"]),
    )
    assert inbox_count == 0
    assert capsule_count == 1
    assert capsule_state_count == 1
    assert subscription_event_count == 1
    assert received_event_count == 0


@pytest.mark.asyncio
async def test_subscription_requires_existing_agent_and_topic_without_events(
    stack: Stack,
) -> None:
    missing_topic_id = UUID("00000000-0000-0000-0000-000000009999")
    missing_topic = await create_subscription(
        stack,
        key="missing-topic-subscription",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        topic_id=missing_topic_id,
    )
    missing_agent = await create_subscription(
        stack,
        key="missing-agent-subscription",
        subscriber_agent_id=MISSING_AGENT_ID,
        topic_id=missing_topic_id,
    )

    assert missing_topic[0] == 404
    assert missing_topic[1]["error"]["code"] == "topic_not_found"
    # Existence checks do not leak topic state to a nonexistent caller.
    assert missing_agent[0] == 404
    assert missing_agent[1]["error"]["code"] == "agent_not_found"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(SubscriptionRow)) == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.event_type == "subscription.created")
            )
            == 0
        )


@pytest.mark.asyncio
async def test_command_scope_cannot_impersonate_another_agent(
    stack: Stack,
) -> None:
    topic_mismatch = await create_topic(
        stack,
        key="topic-impersonation",
        name="Forged Ownership",
        owner_agent_id=OWNER_ID,
        caller_agent_id=OTHER_OWNER_ID,
    )
    topic_mismatch_replay = await create_topic(
        stack,
        key="topic-impersonation",
        name="Forged Ownership",
        owner_agent_id=OWNER_ID,
        caller_agent_id=OTHER_OWNER_ID,
    )
    operation_mismatch = await create_topic(
        stack,
        key="topic-operation-mismatch",
        name="Wrong Operation",
        owner_agent_id=OWNER_ID,
        operation_scope="subscription.create",
    )
    assert topic_mismatch == (
        404,
        {
            "error": {
                "code": "resource_not_found",
                "details": {},
                "message": "The command resource was not found",
            }
        },
        False,
    )
    assert topic_mismatch_replay == (
        topic_mismatch[0],
        topic_mismatch[1],
        True,
    )
    assert operation_mismatch[0] == 404
    assert operation_mismatch[1] == topic_mismatch[1]

    _, topic_body, _ = await create_topic(
        stack,
        key="authorized-topic",
        name="Authorized Topic",
    )
    topic_id = UUID(topic_body["data"]["topic_id"])
    subscription_mismatch = await create_subscription(
        stack,
        key="subscription-impersonation",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        caller_agent_id=SUBSCRIBER_IDS[1],
        topic_id=topic_id,
    )
    subscription_mismatch_replay = await create_subscription(
        stack,
        key="subscription-impersonation",
        subscriber_agent_id=SUBSCRIBER_IDS[0],
        caller_agent_id=SUBSCRIBER_IDS[1],
        topic_id=topic_id,
    )
    assert subscription_mismatch[0] == 404
    assert subscription_mismatch[1] == topic_mismatch[1]
    assert subscription_mismatch_replay == (
        subscription_mismatch[0],
        subscription_mismatch[1],
        True,
    )

    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(TopicRow))
            == 1
        )
        assert (
            await session.scalar(
                select(func.count()).select_from(SubscriptionRow)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(
                    DomainEventRow.event_type.in_(
                        {"topic.created", "subscription.created"}
                    )
                )
            )
            == 1
        )


@pytest.mark.asyncio
async def test_repository_queries_use_primary_key_and_tuple_safe_cursors(
    stack: Stack,
) -> None:
    topic_ids: list[UUID] = []
    for index in range(3):
        status, body, _ = await create_topic(
            stack,
            key=f"page-topic-{index}",
            name=f"Page Topic {index}",
        )
        assert status == 201
        topic_ids.append(UUID(body["data"]["topic_id"]))

    expected_topics = tuple(sorted(topic_ids))
    async with stack.database.read_session() as session:
        topic_page_one = await stack.repository.list_topics(
            session=session,
            after_topic_id=None,
            limit=2,
        )
        topic_page_two = await stack.repository.list_topics(
            session=session,
            after_topic_id=topic_page_one[-1].topic_id,
            limit=2,
        )
        fetched_topic = await stack.repository.get_topic(
            session=session,
            topic_id=expected_topics[1],
        )

    assert tuple(item.topic_id for item in topic_page_one) == expected_topics[:2]
    assert tuple(item.topic_id for item in topic_page_two) == expected_topics[2:]
    assert fetched_topic is not None
    assert fetched_topic.topic_id == expected_topics[1]

    subscription_ids: list[UUID] = []
    for index, topic_id in enumerate(topic_ids):
        status, body, _ = await create_subscription(
            stack,
            key=f"page-subscription-{index}",
            subscriber_agent_id=SUBSCRIBER_IDS[0],
            topic_id=topic_id,
        )
        assert status == 201
        subscription_ids.append(UUID(body["data"]["subscription_id"]))

    for index, subscriber_agent_id in enumerate(SUBSCRIBER_IDS[1:], start=3):
        status, _, _ = await create_subscription(
            stack,
            key=f"tuple-subscription-{index}",
            subscriber_agent_id=subscriber_agent_id,
            topic_id=topic_ids[0],
        )
        assert status == 201

    expected_subscriptions = tuple(sorted(subscription_ids))
    async with stack.database.read_session() as session:
        subscription_page_one = await stack.repository.list_subscriptions(
            session=session,
            subscriber_agent_id=SUBSCRIBER_IDS[0],
            after_subscription_id=None,
            limit=2,
        )
        subscription_page_two = await stack.repository.list_subscriptions(
            session=session,
            subscriber_agent_id=SUBSCRIBER_IDS[0],
            after_subscription_id=subscription_page_one[-1].subscription_id,
            limit=2,
        )
        fetched_subscription = await stack.repository.get_subscription(
            session=session,
            subscription_id=expected_subscriptions[1],
        )
        eligible_page_one = await stack.repository.list_eligible_subscriptions(
            session=session,
            topic_id=topic_ids[0],
            publication_event_id=10_000,
            after=None,
            limit=2,
        )
        eligible_page_two = await stack.repository.list_eligible_subscriptions(
            session=session,
            topic_id=topic_ids[0],
            publication_event_id=10_000,
            after=(
                eligible_page_one[-1].subscriber_agent_id,
                eligible_page_one[-1].subscription_id,
            ),
            limit=2,
        )

    assert (
        tuple(item.subscription_id for item in subscription_page_one)
        == expected_subscriptions[:2]
    )
    assert (
        tuple(item.subscription_id for item in subscription_page_two)
        == expected_subscriptions[2:]
    )
    assert fetched_subscription is not None
    assert fetched_subscription.subscription_id == expected_subscriptions[1]
    eligible = (*eligible_page_one, *eligible_page_two)
    assert tuple(
        (item.subscriber_agent_id, item.subscription_id) for item in eligible
    ) == tuple(
        sorted(
            (
                item.subscriber_agent_id,
                item.subscription_id,
            )
            for item in eligible
        )
    )
    assert {item.subscriber_agent_id for item in eligible} == set(SUBSCRIBER_IDS)


@pytest.mark.parametrize("limit", [0, 101, True])
@pytest.mark.asyncio
async def test_repository_rejects_unsafe_page_sizes(
    stack: Stack,
    limit: int,
) -> None:
    async with stack.database.read_session() as session:
        with pytest.raises(ValueError, match="limit must be an integer"):
            await stack.repository.list_topics(
                session=session,
                limit=limit,
            )


@pytest.mark.parametrize(
    ("operation", "checkpoint"),
    [
        ("topic", FaultCheckpoint.AFTER_SOURCE_INSERT),
        ("topic", FaultCheckpoint.AFTER_EVENT_APPEND),
        ("topic", FaultCheckpoint.AFTER_PROJECTION_APPLY),
        ("topic", FaultCheckpoint.AFTER_RECEIPT_COMPLETION),
        ("subscription", FaultCheckpoint.AFTER_SOURCE_INSERT),
        ("subscription", FaultCheckpoint.AFTER_EVENT_APPEND),
        ("subscription", FaultCheckpoint.AFTER_PROJECTION_APPLY),
        ("subscription", FaultCheckpoint.AFTER_RECEIPT_COMPLETION),
    ],
)
@pytest.mark.asyncio
async def test_source_event_orders_are_atomic_at_fault_boundaries(
    repository_root: Path,
    tmp_path: Path,
    operation: str,
    checkpoint: FaultCheckpoint,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"{operation}-{checkpoint.value}.sqlite3",
        fault=FailAt(checkpoint),
    )
    try:
        if operation == "topic":
            with pytest.raises(InjectedFailure, match=checkpoint.value):
                await create_topic(
                    stack,
                    key=f"fault-{operation}-{checkpoint.value}",
                    name="Atomic Topic",
                )
        else:
            # The topic setup cannot use the fault-injected service because it
            # intentionally crosses both checkpoints under test.
            async with stack.database.transaction() as uow:
                uow.session.add(
                    TopicRow(
                        topic_id=RESOURCE_IDS[-1],
                        owner_agent_id=OWNER_ID,
                        name="Seed Topic",
                        description=None,
                        created_at=NOW,
                    )
                )
            with pytest.raises(InjectedFailure, match=checkpoint.value):
                await create_subscription(
                    stack,
                    key=f"fault-{operation}-{checkpoint.value}",
                    subscriber_agent_id=SUBSCRIBER_IDS[0],
                    topic_id=RESOURCE_IDS[-1],
                )

        async with stack.database.read_session() as session:
            expected_topics = 0 if operation == "topic" else 1
            assert (
                await session.scalar(select(func.count()).select_from(TopicRow))
                == expected_topics
            )
            assert (
                await session.scalar(select(func.count()).select_from(SubscriptionRow))
                == 0
            )
            assert (
                await session.scalar(select(func.count()).select_from(DomainEventRow))
                == 0
            )
            assert (
                await session.scalar(
                    select(func.count()).select_from(IdempotencyRecordRow)
                )
                == 0
            )
    finally:
        await stack.database.dispose()
