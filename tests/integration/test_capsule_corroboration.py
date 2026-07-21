from __future__ import annotations

import asyncio
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
from sqlalchemy import func, select, text

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
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.contracts import (
    ConfirmExperience,
    ExperienceCreation,
    ExperienceDraft,
)
from experience_hub.experiences.events import (
    ExperienceCorroboratedV1,
    ExperienceTemperatureChangedV1,
    register_experience_events,
)
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.sharing.events import (
    CapsuleAdoptedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import (
    AdoptCapsule,
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
    DomainEventRow,
    ExperienceCapsuleRow,
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
PUBLISHER_A = UUID("00000000-0000-0000-0000-000000000101")
PUBLISHER_B = UUID("00000000-0000-0000-0000-000000000102")
RELAY = UUID("00000000-0000-0000-0000-000000000103")
ADOPTER = UUID("00000000-0000-0000-0000-000000000104")


def _uuid_sequence(start: int, count: int) -> tuple[UUID, ...]:
    return tuple(UUID(int=value) for value in range(start, start + count))


@dataclass(frozen=True, slots=True)
class OwnedExperience:
    experience_id: UUID
    version_id: UUID
    content_hash: str


@dataclass(frozen=True, slots=True)
class Published:
    capsule_id: UUID
    item_ids: dict[UUID, UUID]


@dataclass(slots=True)
class CorroborationStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    receipts: ReceiptStore
    registry: EventRegistry
    experience_writer: ExperienceWriter
    experience_service: ExperienceService
    sharing_service: SharingService


def content() -> VersionContent:
    return VersionContent(
        body="An independent observation should strengthen one local memory once.",
        summary="Independent observations corroborate an experience",
        mechanism=(
            "Use root-scoped provenance claims so propagation echoes do not "
            "masquerade as independent evidence."
        ),
        tags=("corroboration", "distributed-memory"),
        applicability=("explicit capsule adoption",),
        evidence=(TypedEvidence(type="experiment", id="corroboration-case"),),
        falsifiers=("one root increases confidence more than once",),
    )


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> CorroborationStack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    register_experience_events(registry)
    register_sharing_events(registry)
    lifecycle = LifecycleConfig()
    source_validator = SourceValidator(registry)
    register_experience_source_validator(source_validator)
    register_sharing_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry(
            (
                ExperienceProjector(registry, lifecycle),
                CapsuleStateProjector(registry),
                InboxItemProjector(registry),
            )
        ),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
        busy_timeout_ms=10_000,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            [
                AgentRow(
                    agent_id=agent_id,
                    name=f"Agent {agent_id.int}",
                    created_at=NOW,
                )
                for agent_id in (PUBLISHER_A, PUBLISHER_B, RELAY, ADOPTER)
            ]
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(_uuid_sequence(30_000, 500)),
    )
    executor = CommandExecutor(
        database=database,
        receipt_store=receipts,
        clock=clock,
    )
    experience_repository = ExperienceRepository(event_registry=registry)
    experience_writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(_uuid_sequence(10_000, 500)),
        repository=experience_repository,
        lifecycle_config=lifecycle,
    )
    mutation_writer = ExperienceMutationWriter(
        repository=experience_repository,
        lifecycle_config=lifecycle,
    )
    experience_service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=experience_writer,
        mutation_writer=mutation_writer,
        query=ExperienceQuery(event_registry=registry),
        lifecycle_config=lifecycle,
    )
    sharing_service = SharingService(
        clock=clock,
        id_generator=SequenceIdGenerator(_uuid_sequence(20_000, 500)),
        receipt_store=receipts,
        repository=SharingRepository(),
        experience_query=ExperienceQuery(event_registry=registry),
        experience_repository=experience_repository,
        experience_writer=experience_writer,
        experience_service=experience_service,
    )
    return CorroborationStack(
        database=database,
        clock=clock,
        executor=executor,
        receipts=receipts,
        registry=registry,
        experience_writer=experience_writer,
        experience_service=experience_service,
        sharing_service=sharing_service,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[CorroborationStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-corroboration.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def _request(
    *,
    agent_id: UUID,
    scope: str,
    key: str,
    route: str,
    body: dict[str, Any],
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope=scope,
        idempotency_key=key,
        method="POST",
        route_template=route,
        path_parameters={"agent_id": agent_id},
        body=body,
    )


async def create_owned_experience(
    stack: CorroborationStack,
    *,
    owner_agent_id: UUID,
    key: str,
    confidence: float,
    temperature: Temperature = Temperature.WARM,
) -> OwnedExperience:
    created: ExperienceCreation | None = None

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        nonlocal created
        created = await stack.experience_writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=owner_agent_id,
                actor_agent_id=owner_agent_id,
                kind=ExperienceKind.SEMANTIC,
                origin=ExperienceOrigin.LOCAL,
                content=content(),
                importance=0.40,
                confidence=confidence,
                source_trust=1.0,
                initial_temperature=temperature,
                links=(),
                occurred_at=stack.clock.now(),
            ),
            command=context,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "experience_id": created.experience_id,
                        "version_id": created.version_id,
                        "content_hash": created.content_hash,
                    }
                }
            ),
        )

    result = await stack.executor.execute(
        _request(
            agent_id=owner_agent_id,
            scope="experience.create",
            key=key,
            route="/v1/experiences",
            body={"summary": content().summary},
        ),
        handler,
    )
    assert result.status_code == 201
    assert created is not None
    return OwnedExperience(
        experience_id=created.experience_id,
        version_id=created.version_id,
        content_hash=created.content_hash,
    )


async def create_topic(
    stack: CorroborationStack,
    *,
    owner_agent_id: UUID = PUBLISHER_A,
) -> UUID:
    command = CreateTopic(
        owner_agent_id=owner_agent_id,
        name=f"Corroboration {owner_agent_id}",
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
            agent_id=owner_agent_id,
            scope="topic.create",
            key=f"topic-{owner_agent_id}",
            route="/v1/topics",
            body={"name": command.name},
        ),
        handler,
    )
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["topic_id"])


async def subscribe(
    stack: CorroborationStack,
    *,
    subscriber_agent_id: UUID,
    topic_id: UUID,
) -> None:
    command = CreateSubscription(
        subscriber_agent_id=subscriber_agent_id,
        topic_id=topic_id,
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
            agent_id=subscriber_agent_id,
            scope="subscription.create",
            key=f"subscribe-{subscriber_agent_id}-{topic_id}",
            route="/v1/agents/{agent_id}/subscriptions",
            body={"topic_id": topic_id},
        ),
        handler,
    )
    assert result.status_code == 201


async def publish(
    stack: CorroborationStack,
    *,
    publisher_agent_id: UUID,
    topic_id: UUID,
    experience: OwnedExperience,
    key: str,
    parent_adoption_id: UUID | None = None,
) -> Published:
    command = PublishCapsule(
        owner_agent_id=publisher_agent_id,
        topic_id=topic_id,
        experience_id=experience.experience_id,
        version_id=experience.version_id,
        expires_at=stack.clock.now() + timedelta(days=7),
        parent_adoption_id=parent_adoption_id,
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
        _request(
            agent_id=publisher_agent_id,
            scope="capsule.publish",
            key=key,
            route="/v1/capsules",
            body={
                "topic_id": topic_id,
                "experience_id": experience.experience_id,
                "version_id": experience.version_id,
                "expires_at": command.expires_at,
                "parent_adoption_id": parent_adoption_id,
            },
        ),
        handler,
    )
    assert result.status_code == 201, json.loads(result.body)
    capsule_id = UUID(json.loads(result.body)["data"]["capsule_id"])
    async with stack.database.read_session() as session:
        items = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).where(InboxItemRow.capsule_id == capsule_id)
                )
            ).all()
        )
    return Published(
        capsule_id=capsule_id,
        item_ids={row.recipient_agent_id: row.item_id for row in items},
    )


async def adopt(
    stack: CorroborationStack,
    *,
    adopter_agent_id: UUID,
    item_id: UUID,
    key: str,
) -> CommandResult:
    command = AdoptCapsule(
        adopter_agent_id=adopter_agent_id,
        item_id=item_id,
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
        _request(
            agent_id=adopter_agent_id,
            scope="capsule.adopt",
            key=key,
            route="/v1/agents/{agent_id}/inbox/{item_id}:adopt",
            body={"item_id": item_id, "importance": command.importance},
        ),
        handler,
    )


async def confirm(
    stack: CorroborationStack,
    *,
    owner_agent_id: UUID,
    experience_id: UUID,
    key: str,
) -> CommandResult:
    command = ConfirmExperience(
        owner_agent_id=owner_agent_id,
        experience_id=experience_id,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.experience_service.confirm(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await stack.executor.execute(
        _request(
            agent_id=owner_agent_id,
            scope="experience.confirm",
            key=key,
            route="/v1/experiences/{experience_id}:confirm",
            body={"experience_id": experience_id},
        ),
        handler,
    )


async def adoption_row(
    stack: CorroborationStack,
    *,
    adopter_agent_id: UUID,
    capsule_id: UUID,
) -> AdoptionRecordRow:
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.adopter_agent_id == adopter_agent_id,
                AdoptionRecordRow.capsule_id == capsule_id,
            )
        )
    assert row is not None
    return row


def _result(result: CommandResult) -> dict[str, Any]:
    assert result.status_code == 200, json.loads(result.body)
    return dict(json.loads(result.body)["data"])


def stored_event(
    stack: CorroborationStack,
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


async def assert_task4_projections_rebuild(
    stack: CorroborationStack,
) -> None:
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


@pytest.mark.asyncio
async def test_two_independent_roots_corroborate_one_existing_experience(
    stack: CorroborationStack,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="local-equivalent",
        confidence=0.40,
    )
    source_a = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="source-a",
        confidence=0.80,
    )
    source_b = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_B,
        key="source-b",
        confidence=0.70,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published_a = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source_a,
        key="publish-a",
    )
    published_b = await publish(
        stack,
        publisher_agent_id=PUBLISHER_B,
        topic_id=topic_id,
        experience=source_b,
        key="publish-b",
    )

    first = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published_a.item_ids[ADOPTER],
            key="adopt-a",
        )
    )
    second = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published_b.item_ids[ADOPTER],
            key="adopt-b",
        )
    )

    assert first["created"] is False
    assert first["corroboration_applied"] is True
    assert second["created"] is False
    assert second["corroboration_applied"] is True
    assert UUID(first["experience"]["experience_id"]) == local.experience_id
    assert UUID(second["experience"]["experience_id"]) == local.experience_id

    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, local.experience_id)
        owner_experiences = await session.scalar(
            select(func.count())
            .select_from(ExperienceRow)
            .where(ExperienceRow.owner_agent_id == ADOPTER)
        )
        owner_versions = await session.scalar(
            select(func.count())
            .select_from(ExperienceVersionRow)
            .join(
                ExperienceRow,
                ExperienceRow.experience_id == ExperienceVersionRow.experience_id,
            )
            .where(ExperienceRow.owner_agent_id == ADOPTER)
        )
        links = await session.scalar(
            select(func.count()).select_from(ExperienceLinkRow)
        )
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            ("experience.corroborated", "capsule.adopted")
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        capsules = {
            row.capsule_id: row
            for row in (
                await session.scalars(
                    select(ExperienceCapsuleRow).where(
                        ExperienceCapsuleRow.capsule_id.in_(
                            (
                                published_a.capsule_id,
                                published_b.capsule_id,
                            )
                        )
                    )
                )
            ).all()
        }
    assert state is not None
    expected_after_first = 0.40 + (1.0 - 0.40) * 0.20 * 0.50
    expected_final = expected_after_first + (1.0 - expected_after_first) * 0.20 * 0.50
    assert state.confidence == pytest.approx(expected_final)
    assert state.source_trust == 1.0
    assert owner_experiences == owner_versions == 1
    assert links == 0
    assert [event.event_type for event in events] == [
        "experience.corroborated",
        "capsule.adopted",
        "experience.corroborated",
        "capsule.adopted",
    ]

    for published, publisher_id in (
        (published_a, PUBLISHER_A),
        (published_b, PUBLISHER_B),
    ):
        row = await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=published.capsule_id,
        )
        capsule = capsules[published.capsule_id]
        assert row.resulting_experience_id == local.experience_id
        assert row.captured_trust == 0.50
        assert row.root_fingerprint == capsule.root_fingerprint
        assert row.corroboration_applied is True
        assert row.provenance_chain == canonical_json_bytes(
            (
                ProvenanceHop(
                    capsule_id=published.capsule_id,
                    publisher_agent_id=publisher_id,
                ),
            )
        )


@pytest.mark.asyncio
async def test_equivalent_adoption_uses_captured_nondefault_trust_only_once(
    stack: CorroborationStack,
) -> None:
    from tests.integration.test_capsule_feedback import record_feedback
    from tests.integration.test_capsule_rejection import reject

    manager = cast(ProjectionManager, stack.database._projection_applier)  # noqa: SLF001
    manager.registry.register(AgentReputationProjector(stack.registry))
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="trusted-local",
        confidence=0.40,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="trusted-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    feedback_capsule = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="trusted-feedback-publish",
    )
    assert (
        await reject(
            cast(Any, stack),
            key="reject-before-trusted-equivalent",
            recipient_agent_id=ADOPTER,
            item_id=feedback_capsule.item_ids[ADOPTER],
        )
    ).status_code == 200
    assert (
        await record_feedback(
            cast(Any, stack),
            key="useful-before-trusted-equivalent",
            observer_agent_id=ADOPTER,
            capsule_id=feedback_capsule.capsule_id,
        )
    ).status_code == 201
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="trusted-publish",
    )

    result = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published.item_ids[ADOPTER],
            key="trusted-adopt",
        )
    )

    assert result["created"] is False
    assert result["corroboration_applied"] is True
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, local.experience_id)
        adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.capsule_id == published.capsule_id,
                AdoptionRecordRow.adopter_agent_id == ADOPTER,
            )
        )
    assert state is not None and adoption is not None
    assert adoption.captured_trust == pytest.approx(0.60)
    assert state.confidence == pytest.approx(
        0.40 + (1.0 - 0.40) * 0.20 * 0.60
    )
    assert state.source_trust == 1.0


@pytest.mark.asyncio
async def test_equivalent_adoption_rejects_clock_before_affected_experience(
    stack: CorroborationStack,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="clock-local",
        confidence=0.40,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="clock-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="clock-publish",
    )
    stack.clock.advance(timedelta(hours=1))
    assert (
        await confirm(
            stack,
            owner_agent_id=ADOPTER,
            experience_id=local.experience_id,
            key="clock-confirm",
        )
    ).status_code == 200
    stack.clock.advance(timedelta(minutes=-30))

    result = await adopt(
        stack,
        adopter_agent_id=ADOPTER,
        item_id=published.item_ids[ADOPTER],
        key="clock-adopt",
    )

    assert result.status_code == 409
    assert json.loads(result.body)["error"]["code"] == "clock_regression"
    async with stack.database.read_session() as session:
        item = await session.get(
            InboxItemRow,
            published.item_ids[ADOPTER],
        )
        adoption_count = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
    assert item is not None and item.state is InboxState.PENDING
    assert adoption_count == 0


@pytest.mark.asyncio
async def test_repeated_root_records_provenance_without_repeated_confidence(
    stack: CorroborationStack,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="repeat-local",
        confidence=0.40,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="repeat-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    first_capsule = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="repeat-publish-one",
    )
    second_capsule = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="repeat-publish-two",
    )

    first = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=first_capsule.item_ids[ADOPTER],
            key="repeat-adopt-one",
        )
    )
    second = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=second_capsule.item_ids[ADOPTER],
            key="repeat-adopt-two",
        )
    )

    assert first["corroboration_applied"] is True
    assert second["corroboration_applied"] is False
    rows = [
        await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=capsule.capsule_id,
        )
        for capsule in (first_capsule, second_capsule)
    ]
    assert rows[0].root_fingerprint == rows[1].root_fingerprint
    assert [row.corroboration_applied for row in rows] == [True, False]

    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, local.experience_id)
        event_types = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.event_type)
                    .where(
                        DomainEventRow.event_type.in_(
                            ("experience.corroborated", "capsule.adopted")
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        corroborated_claims = await session.scalar(
            select(func.count())
            .select_from(AdoptionRecordRow)
            .where(
                AdoptionRecordRow.resulting_experience_id == local.experience_id,
                AdoptionRecordRow.root_fingerprint == rows[0].root_fingerprint,
                AdoptionRecordRow.corroboration_applied.is_(True),
            )
        )
    assert state is not None
    assert state.confidence == pytest.approx(0.46)
    assert event_types == (
        "experience.corroborated",
        "capsule.adopted",
        "capsule.adopted",
    )
    assert corroborated_claims == 1
    await assert_task4_projections_rebuild(stack)


@pytest.mark.asyncio
async def test_initial_adoption_root_and_republished_echo_never_score_twice(
    stack: CorroborationStack,
) -> None:
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="echo-source",
        confidence=0.80,
    )
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="echo-local",
        confidence=0.40,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=RELAY, topic_id=topic_id)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    original = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="echo-original",
    )

    relay_result = _result(
        await adopt(
            stack,
            adopter_agent_id=RELAY,
            item_id=original.item_ids[RELAY],
            key="relay-adopt-original",
        )
    )
    adopter_first = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=original.item_ids[ADOPTER],
            key="adopter-adopt-original",
        )
    )
    assert relay_result["created"] is True
    assert relay_result["corroboration_applied"] is False
    assert adopter_first["corroboration_applied"] is True
    relay_experience = OwnedExperience(
        experience_id=UUID(relay_result["experience"]["experience_id"]),
        version_id=UUID(relay_result["experience"]["current_version_id"]),
        content_hash=relay_result["experience"]["current_content_hash"],
    )
    relay_adoption = await adoption_row(
        stack,
        adopter_agent_id=RELAY,
        capsule_id=original.capsule_id,
    )

    repeated_original = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="echo-original-repeat",
    )
    relay_repeat = _result(
        await adopt(
            stack,
            adopter_agent_id=RELAY,
            item_id=repeated_original.item_ids[RELAY],
            key="relay-adopt-repeat",
        )
    )
    assert relay_repeat["created"] is False
    assert relay_repeat["corroboration_applied"] is False

    echo = await publish(
        stack,
        publisher_agent_id=RELAY,
        topic_id=topic_id,
        experience=relay_experience,
        key="relay-republish",
        parent_adoption_id=relay_adoption.adoption_id,
    )
    adopter_echo = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=echo.item_ids[ADOPTER],
            key="adopter-adopt-echo",
        )
    )
    assert adopter_echo["created"] is False
    assert adopter_echo["corroboration_applied"] is False
    assert UUID(adopter_echo["experience"]["experience_id"]) == local.experience_id

    async with stack.database.read_session() as session:
        relay_state = await session.get(
            ExperienceStateRow,
            relay_experience.experience_id,
        )
        adopter_state = await session.get(
            ExperienceStateRow,
            local.experience_id,
        )
        echo_source = await session.get(ExperienceCapsuleRow, echo.capsule_id)
        links = await session.scalar(
            select(func.count())
            .select_from(ExperienceLinkRow)
            .where(
                ExperienceLinkRow.source_experience_id == relay_experience.experience_id
            )
        )
    assert relay_state is not None
    assert relay_state.confidence == pytest.approx(0.80 * 0.50)
    assert relay_state.source_trust == 0.50
    assert adopter_state is not None
    assert adopter_state.confidence == pytest.approx(0.46)
    assert links == 0
    assert echo_source is not None
    assert echo_source.root_fingerprint == relay_adoption.root_fingerprint
    assert echo_source.provenance_chain == canonical_json_bytes(
        (
            ProvenanceHop(
                capsule_id=original.capsule_id,
                publisher_agent_id=PUBLISHER_A,
            ),
        )
    )
    echo_adoption = await adoption_row(
        stack,
        adopter_agent_id=ADOPTER,
        capsule_id=echo.capsule_id,
    )
    assert echo_adoption.root_fingerprint == relay_adoption.root_fingerprint
    assert echo_adoption.provenance_chain == canonical_json_bytes(
        (
            ProvenanceHop(
                capsule_id=original.capsule_id,
                publisher_agent_id=PUBLISHER_A,
            ),
            ProvenanceHop(
                capsule_id=echo.capsule_id,
                publisher_agent_id=RELAY,
            ),
        )
    )


@pytest.mark.asyncio
async def test_independent_corroboration_promotes_cold_to_hot_in_event_order(
    stack: CorroborationStack,
) -> None:
    cold = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="cold-local",
        confidence=0.40,
        temperature=Temperature.COLD,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="cold-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="cold-publish",
    )

    result = _result(
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published.item_ids[ADOPTER],
            key="cold-adopt",
        )
    )

    assert result["created"] is False
    assert result["corroboration_applied"] is True
    assert result["experience"]["temperature"] == "hot"
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, cold.experience_id)
        payload = await session.get(ExperiencePayloadRow, cold.version_id)
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            (
                                "experience.corroborated",
                                "experience.temperature_changed",
                                "capsule.adopted",
                            )
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    assert state is not None and state.temperature is Temperature.HOT
    assert state.confidence == pytest.approx(0.46)
    assert payload is not None and payload.codec is PayloadCodec.PLAIN
    assert [row.event_type for row in rows] == [
        "experience.corroborated",
        "experience.temperature_changed",
        "capsule.adopted",
    ]
    decoded = [
        stack.registry.decode(event_type=row.event_type, payload=row.payload)
        for row in rows
    ]
    assert isinstance(decoded[0], ExperienceCorroboratedV1)
    assert decoded[0].before.temperature is Temperature.COLD
    assert decoded[0].after.temperature is Temperature.COLD
    assert decoded[0].captured_trust == 0.50
    assert isinstance(decoded[1], ExperienceTemperatureChangedV1)
    assert decoded[1].cause == "capsule_corroboration"
    assert decoded[1].before == decoded[0].after
    assert decoded[1].after.temperature is Temperature.HOT
    assert isinstance(decoded[2], CapsuleAdoptedV1)
    assert decoded[2].state_before is InboxState.PENDING
    assert decoded[2].state_after is InboxState.ADOPTED
    assert decoded[2].root_fingerprint == decoded[0].root_fingerprint
    assert decoded[2].adoption_id == decoded[0].adoption_id
    await assert_task4_projections_rebuild(stack)


@pytest.mark.parametrize(
    "corruption",
    ("cold_missing_promotion", "non_cold_extra_promotion"),
)
@pytest.mark.asyncio
async def test_adoption_replay_locks_promotion_to_corroborated_temperature(
    stack: CorroborationStack,
    corruption: str,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key=f"{corruption}-local",
        confidence=0.40,
        temperature=Temperature.COLD,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key=f"{corruption}-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key=f"{corruption}-publish",
    )
    item_id = published.item_ids[ADOPTER]
    assert (
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=item_id,
            key=f"{corruption}-adopt",
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        rows = {
            row.event_type: row
            for row in (
                await uow.session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.aggregate_id.in_(
                            (item_id, local.experience_id)
                        ),
                        DomainEventRow.event_type.in_(
                            (
                                "capsule.received",
                                "capsule.adopted",
                                "experience.corroborated",
                                "experience.temperature_changed",
                            )
                        ),
                    )
                )
            ).all()
        }
        adopted = stored_event(stack, rows["capsule.adopted"])
        item = await uow.session.get(InboxItemRow, item_id)
        assert item is not None
        await uow.session.execute(
            text("DROP TRIGGER domain_events_reject_update")
        )
        if corruption == "cold_missing_promotion":
            other_causation_id = await uow.session.scalar(
                select(DomainEventRow.causation_id).where(
                    DomainEventRow.causation_id != adopted.causation_id
                )
            )
            assert other_causation_id is not None
            rows[
                "experience.temperature_changed"
            ].causation_id = other_causation_id
        else:
            corroborated = stack.registry.decode(
                event_type=rows["experience.corroborated"].event_type,
                payload=rows["experience.corroborated"].payload,
            )
            assert isinstance(corroborated, ExperienceCorroboratedV1)
            before = corroborated.before.model_copy(
                update={"temperature": Temperature.WARM}
            )
            after = corroborated.after.model_copy(
                update={"temperature": Temperature.WARM}
            )
            rows["experience.corroborated"].payload = canonical_json_bytes(
                corroborated.model_copy(
                    update={"before": before, "after": after}
                )
            )
        received = rows["capsule.received"]
        item.state = InboxState.PENDING
        item.projection_event_id = received.event_id
        await uow.session.flush()

        with pytest.raises(
            SharingProjectionIntegrityError,
            match="command event sequence",
        ):
            await InboxItemProjector(stack.registry).apply(
                uow.session,
                adopted,
            )


@pytest.mark.asyncio
async def test_competing_independent_roots_are_serialized_without_lost_update(
    stack: CorroborationStack,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="concurrent-local",
        confidence=0.40,
    )
    source_a = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="concurrent-source-a",
        confidence=0.80,
    )
    source_b = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_B,
        key="concurrent-source-b",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published_a = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source_a,
        key="concurrent-publish-a",
    )
    published_b = await publish(
        stack,
        publisher_agent_id=PUBLISHER_B,
        topic_id=topic_id,
        experience=source_b,
        key="concurrent-publish-b",
    )

    results = await asyncio.gather(
        adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published_a.item_ids[ADOPTER],
            key="concurrent-adopt-a",
        ),
        adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=published_b.item_ids[ADOPTER],
            key="concurrent-adopt-b",
        ),
    )
    decoded = [_result(result) for result in results]
    assert all(result["corroboration_applied"] is True for result in decoded)

    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, local.experience_id)
        rows = tuple(
            (
                await session.scalars(
                    select(AdoptionRecordRow).where(
                        AdoptionRecordRow.resulting_experience_id
                        == local.experience_id,
                        AdoptionRecordRow.corroboration_applied.is_(True),
                    )
                )
            ).all()
        )
    assert state is not None
    after_first = 0.40 + (1.0 - 0.40) * 0.20 * 0.50
    expected = after_first + (1.0 - after_first) * 0.20 * 0.50
    assert state.confidence == pytest.approx(expected)
    assert len(rows) == 2
    assert len({row.root_fingerprint for row in rows}) == 2
