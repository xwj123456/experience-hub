from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select

from experience_hub.agents import AgentCreated, AgentService, CreateAgent
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import CommandContext, CommandRequest, EventRegistry
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.contracts import (
    CreateExperience,
    ExperienceDraft,
    VersionLinkInput,
)
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
    register_experience_events,
)
from experience_hub.experiences.projector import (
    ExperienceProjectionIntegrityError,
    ExperienceProjector,
)
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
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
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_experience_source_validator,
)

NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000102")
EXPERIENCE_IDS = (
    UUID("00000000-0000-0000-0000-000000000201"),
    UUID("00000000-0000-0000-0000-000000000202"),
    UUID("00000000-0000-0000-0000-000000000203"),
    UUID("00000000-0000-0000-0000-000000000204"),
)
VERSION_IDS = (
    UUID("00000000-0000-0000-0000-000000000301"),
    UUID("00000000-0000-0000-0000-000000000302"),
    UUID("00000000-0000-0000-0000-000000000303"),
    UUID("00000000-0000-0000-0000-000000000304"),
)
RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(401, 421)
)
WRITER_IDS = tuple(
    item
    for pair in zip(EXPERIENCE_IDS, VERSION_IDS, strict=True)
    for item in pair
)


def content(label: str = "lease-handoff") -> VersionContent:
    return VersionContent(
        body=f"Preserve body spacing for {label}.",
        summary=f"Summary {label}",
        mechanism=f"Mechanism {label}",
        tags=("memory", label),
        applicability=("single writer",),
        evidence=(),
        falsifiers=("overlap observed",),
    )


@dataclass(slots=True)
class Stack:
    database: Database
    clock: FrozenClock
    receipts: ReceiptStore
    executor: CommandExecutor
    repository: ExperienceRepository
    writer: ExperienceWriter
    service: ExperienceService
    query: ExperienceQuery
    projector: ExperienceProjector
    manager: ProjectionManager
    registry: EventRegistry


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    def __init__(self, checkpoint: FaultCheckpoint | None = None) -> None:
        self.checkpoint = checkpoint

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint == self.checkpoint:
            raise InjectedFailure(checkpoint.value)


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
    fault: FailAt | None = None,
    lifecycle_config: LifecycleConfig | None = None,
) -> Stack:
    lifecycle = lifecycle_config or LifecycleConfig()
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    register_experience_events(registry)
    projector = ExperienceProjector(registry, lifecycle)
    source_validator = SourceValidator(registry)
    register_experience_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry([projector]),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
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
            ]
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    repository = ExperienceRepository(event_registry=registry)
    writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(WRITER_IDS),
        repository=repository,
        lifecycle_config=lifecycle,
    )
    service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=writer,
        lifecycle_config=lifecycle,
    )
    return Stack(
        database=database,
        clock=clock,
        receipts=receipts,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        repository=repository,
        writer=writer,
        service=service,
        query=ExperienceQuery(event_registry=registry),
        projector=projector,
        manager=manager,
        registry=registry,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "create-experience.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def request(
    *,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
    operation: str = "experience.create",
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=operation,
        idempotency_key=key,
        method="POST",
        route_template="/v1/experiences",
        body={"key": key},
    )


async def create(
    stack: Stack,
    *,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
    value: VersionContent | None = None,
    importance: float = 0.35,
    confidence: float = 0.50,
    links: tuple[VersionLinkInput, ...] = (),
) -> tuple[int, dict[str, Any]]:
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=ExperienceKind.PROCEDURAL,
        content=value or content(key),
        importance=importance,
        confidence=confidence,
        links=links,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(key=key, owner_agent_id=owner_agent_id),
        handler,
    )
    return result.status_code, json.loads(result.body)


@pytest.mark.asyncio
async def test_local_creation_persists_defaults_two_events_and_noop_projection(
    stack: Stack,
) -> None:
    status, body = await create(stack, key="default")

    assert status == 201
    assert body["data"] == {
        "content_hash": body["data"]["content_hash"],
        "experience_id": str(EXPERIENCE_IDS[0]),
        "version_id": str(VERSION_IDS[0]),
    }
    async with stack.database.read_session() as session:
        identity = await session.get(ExperienceRow, EXPERIENCE_IDS[0])
        version = await session.get(ExperienceVersionRow, VERSION_IDS[0])
        payload = await session.get(ExperiencePayloadRow, VERSION_IDS[0])
        state = await session.get(ExperienceStateRow, EXPERIENCE_IDS[0])
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[0])

    assert identity is not None
    assert (
        identity.owner_agent_id,
        identity.kind,
        identity.origin,
        identity.created_at,
    ) == (
        OWNER_ID,
        ExperienceKind.PROCEDURAL,
        ExperienceOrigin.LOCAL,
        NOW,
    )
    assert version is not None
    assert (
        version.experience_id,
        version.version_number,
        version.supersedes_version_id,
        version.created_at,
    ) == (EXPERIENCE_IDS[0], 1, None, NOW)
    assert payload is not None and payload.codec is PayloadCodec.PLAIN
    assert state is not None
    assert (
        state.temperature,
        state.importance,
        state.confidence,
        state.source_trust,
        state.access_count,
        state.access_strength,
        state.strength_updated_at,
        state.last_accessed_at,
        state.last_transition_at,
        state.last_lifecycle_evaluated_at,
        state.consecutive_below_threshold,
        state.pinned,
    ) == (
        Temperature.WARM,
        0.35,
        0.50,
        1.0,
        0,
        0.0,
        NOW,
        None,
        NOW,
        None,
        0,
        False,
    )
    assert state.activation_score == pytest.approx(0.48, abs=1e-12)
    assert [event.event_type for event in events] == [
        "experience.created",
        "experience.version_created",
    ]
    assert [event.sequence for event in events] == [1, 2]
    created_payload = ExperienceCreatedV1.model_validate_json(events[0].payload)
    version_payload = ExperienceVersionCreatedV1.model_validate_json(
        events[1].payload
    )
    assert version_payload.before == version_payload.after == created_payload.after
    assert state.projection_event_id == events[1].event_id
    assert b'"body"' not in events[0].payload + events[1].payload
    assert receipt is not None
    assert (receipt.result_resource_type, receipt.result_resource_id) == (
        "experience",
        EXPERIENCE_IDS[0],
    )

    report = await stack.manager.verify(stack.database)
    assert report.matches


@pytest.mark.asyncio
async def test_high_importance_local_creation_starts_hot(stack: Stack) -> None:
    status, _ = await create(stack, key="important", importance=0.85)

    assert status == 201
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, EXPERIENCE_IDS[0])
    assert state is not None
    assert state.temperature is Temperature.HOT
    assert state.activation_score == pytest.approx(0.63, abs=1e-12)


@pytest.mark.parametrize(
    ("importance", "confidence"),
    [
        (-0.01, 0.5),
        (1.01, 0.5),
        (0.35, -0.01),
        (0.35, 1.01),
        (float("nan"), 0.5),
    ],
)
@pytest.mark.asyncio
async def test_invalid_local_scores_return_422_without_domain_state(
    stack: Stack,
    importance: float,
    confidence: float,
) -> None:
    status, body = await create(
        stack,
        key=f"invalid-{importance}-{confidence}",
        importance=importance,
        confidence=confidence,
    )

    assert status == 422
    assert body["error"]["code"] == "invalid_experience"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceRow))
            == 0
        )
        assert (
            await session.scalar(
                select(func.count()).select_from(DomainEventRow)
            )
            == 0
        )


@pytest.mark.asyncio
async def test_current_content_uniqueness_is_owner_scoped(stack: Stack) -> None:
    shared = content("shared")
    first_status, first = await create(stack, key="first", value=shared)
    duplicate_status, duplicate = await create(
        stack,
        key="duplicate",
        value=shared,
    )
    isolated_status, isolated = await create(
        stack,
        key="isolated",
        owner_agent_id=OTHER_OWNER_ID,
        value=shared,
    )

    assert first_status == isolated_status == 201
    assert duplicate_status == 409
    assert duplicate["error"]["code"] == "duplicate_experience"
    assert (
        first["data"]["content_hash"]
        == isolated["data"]["content_hash"]
    )


@pytest.mark.asyncio
async def test_complete_links_are_canonical_and_use_version_event_id(
    stack: Stack,
) -> None:
    await create(stack, key="target-a", value=content("target-a"))
    await create(stack, key="target-b", value=content("target-b"))
    status, _ = await create(
        stack,
        key="linked",
        value=content("linked"),
        links=(
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[1],
                relation=LinkRelation.TESTS,
            ),
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[0],
                relation=LinkRelation.SUPPORTS,
            ),
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[0],
                relation=LinkRelation.DERIVED_FROM,
            ),
        ),
    )

    assert status == 201
    async with stack.database.read_session() as session:
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == EXPERIENCE_IDS[2])
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        links = tuple(
            (
                await session.scalars(
                    select(ExperienceLinkRow)
                    .where(
                        ExperienceLinkRow.source_version_id
                        == VERSION_IDS[2]
                    )
                    .order_by(
                        ExperienceLinkRow.target_experience_id,
                        ExperienceLinkRow.relation,
                    )
                )
            ).all()
        )

    assert len(events) == 2 and len(links) == 3
    version_event = ExperienceVersionCreatedV1.model_validate_json(
        events[1].payload
    )
    assert tuple(
        (link.target_experience_id, link.relation)
        for link in version_event.links
    ) == (
        (EXPERIENCE_IDS[0], LinkRelation.DERIVED_FROM),
        (EXPERIENCE_IDS[0], LinkRelation.SUPPORTS),
        (EXPERIENCE_IDS[1], LinkRelation.TESTS),
    )
    assert {link.source_event_id for link in links} == {events[1].event_id}
    assert {link.source_experience_id for link in links} == {
        EXPERIENCE_IDS[2]
    }


@pytest.mark.asyncio
async def test_foreign_link_target_is_rejected_without_revealing_a_link(
    stack: Stack,
) -> None:
    await create(
        stack,
        key="foreign-target",
        owner_agent_id=OTHER_OWNER_ID,
        value=content("foreign-target"),
    )

    status, body = await create(
        stack,
        key="bad-link",
        value=content("bad-link"),
        links=(
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[0],
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )

    assert status == 422
    assert body["error"]["code"] == "invalid_experience_link"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count()).select_from(ExperienceLinkRow)
            )
            == 0
        )


@pytest.mark.asyncio
async def test_generic_writer_does_not_choose_the_outer_receipt_resource(
    stack: Stack,
) -> None:
    draft = ExperienceDraft(
        owner_agent_id=OWNER_ID,
        actor_agent_id=OWNER_ID,
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.ADOPTED_IDEA,
        content=content("generic"),
        importance=0.4,
        confidence=0.35,
        source_trust=0.8,
        initial_temperature=Temperature.COLD,
        links=(),
        occurred_at=NOW,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        result = await stack.writer.create_from_draft(
            uow=uow,
            draft=draft,
            command=context,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {"experience_id": str(result.experience_id)}
            ),
        )

    result = await stack.executor.execute(
        request(key="generic", operation="inspiration.idea_adopt"),
        handler,
    )

    assert result.status_code == 201
    async with stack.database.read_session() as session:
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_IDS[0])
        payload = await session.get(ExperiencePayloadRow, VERSION_IDS[0])
        equivalent = await stack.writer.find_current_equivalent(
            session=session,
            owner_agent_id=OWNER_ID,
            content_hash=(
                await session.get(ExperienceStateRow, EXPERIENCE_IDS[0])
            ).current_content_hash,  # type: ignore[union-attr]
        )
        current = await stack.query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=EXPERIENCE_IDS[0],
            version_id=None,
        )
        selected = await stack.query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=EXPERIENCE_IDS[0],
            version_id=VERSION_IDS[0],
        )
    assert receipt is not None
    assert receipt.result_resource_type is None
    assert receipt.result_resource_id is None
    assert payload is not None and payload.codec is PayloadCodec.ZLIB
    assert equivalent is not None
    assert equivalent.experience_id == EXPERIENCE_IDS[0]
    assert current == selected
    assert current.content == content("generic")


@pytest.mark.asyncio
async def test_agent_and_experience_events_coexist_append_and_decode(
    stack: Stack,
) -> None:
    agent_id = UUID("00000000-0000-0000-0000-000000000999")
    agent_service = AgentService(
        clock=stack.clock,
        id_generator=SequenceIdGenerator([agent_id]),
        receipt_store=stack.receipts,
    )

    async def create_agent(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await agent_service.create(
            uow=uow,
            command=CreateAgent(name="Combined Registry"),
            command_context=context,
        )

    agent_result = await stack.executor.execute(
        CommandRequest(
            caller_scope="system:local",
            operation_scope="agent.create",
            idempotency_key="combined-agent",
            method="POST",
            route_template="/v1/agents",
            body={"name": "Combined Registry"},
        ),
        create_agent,
    )
    experience_status, _ = await create(stack, key="combined-experience")

    assert agent_result.status_code == experience_status == 201
    assert stack.registry.event_types == (
        STATE_EXPERIENCE_EVENT_TYPES | {"agent.created"}
    )
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    decoded = tuple(
        stack.registry.decode(event_type=row.event_type, payload=row.payload)
        for row in rows
    )
    assert [type(value) for value in decoded] == [
        AgentCreated,
        ExperienceCreatedV1,
        ExperienceVersionCreatedV1,
    ]


@pytest.mark.parametrize(
    "checkpoint",
    [
        FaultCheckpoint.AFTER_SOURCE_INSERT,
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
        FaultCheckpoint.AFTER_RECEIPT_COMPLETION,
    ],
)
@pytest.mark.asyncio
async def test_creation_fault_rolls_back_every_atomic_component(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"fault-{checkpoint.value}.sqlite3",
        fault=FailAt(checkpoint),
    )
    try:
        command = CreateExperience(
            owner_agent_id=OWNER_ID,
            kind=ExperienceKind.PROCEDURAL,
            content=content("rollback"),
            links=(),
        )

        async def handler(
            uow: UnitOfWork,
            context: CommandContext,
        ) -> StoredResponse:
            return await stack.service.create(
                uow=uow,
                command=command,
                command_context=context,
            )

        with pytest.raises(InjectedFailure, match=checkpoint.value):
            await stack.executor.execute(
                request(key=f"fault-{checkpoint.value}"),
                handler,
            )

        async with stack.database.read_session() as session:
            for table in (
                ExperienceRow,
                ExperienceVersionRow,
                ExperiencePayloadRow,
                ExperienceStateRow,
                ExperienceLinkRow,
                DomainEventRow,
                IdempotencyRecordRow,
            ):
                assert (
                    await session.scalar(select(func.count()).select_from(table))
                    == 0
                )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_link_insertion_failure_rolls_back_new_source_events_and_projection(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await create(stack, key="link-target", value=content("link-target"))
    original = stack.repository.add_links

    def fail_after_staging_link(**kwargs: object) -> None:
        original(**kwargs)  # type: ignore[arg-type]
        raise InjectedFailure("link-insert")

    monkeypatch.setattr(stack.repository, "add_links", fail_after_staging_link)
    command = CreateExperience(
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.PROCEDURAL,
        content=content("link-source"),
        links=(
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[0],
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    with pytest.raises(InjectedFailure, match="link-insert"):
        await stack.executor.execute(
            request(key="link-insertion-failure"),
            handler,
        )

    async with stack.database.read_session() as session:
        assert await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(ExperienceVersionRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(ExperienceStateRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(DomainEventRow)
        ) == 2
        assert await session.scalar(
            select(func.count()).select_from(ExperienceLinkRow)
        ) == 0
        assert await session.scalar(
            select(func.count()).select_from(IdempotencyRecordRow)
        ) == 1


@pytest.mark.asyncio
async def test_receipt_completion_fault_rolls_back_a_staged_real_link(
    stack: Stack,
) -> None:
    await create(stack, key="fault-link-target", value=content("fault-link-target"))
    fault = FailAt(FaultCheckpoint.AFTER_RECEIPT_COMPLETION)
    stack.database._fault_injector = fault

    command = CreateExperience(
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.PROCEDURAL,
        content=content("fault-link-source"),
        links=(
            VersionLinkInput(
                target_experience_id=EXPERIENCE_IDS[0],
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    with pytest.raises(InjectedFailure, match="after_receipt_completion"):
        await stack.executor.execute(request(key="fault-linked-source"), handler)

    async with stack.database.read_session() as session:
        assert await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(ExperienceVersionRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(ExperienceStateRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(DomainEventRow)
        ) == 2
        assert await session.scalar(
            select(func.count()).select_from(ExperienceLinkRow)
        ) == 0
        assert await session.scalar(
            select(func.count()).select_from(IdempotencyRecordRow)
        ) == 1


@pytest.mark.asyncio
async def test_projector_rejects_wrong_created_aggregate(stack: Stack) -> None:
    await create(stack, key="anchor")
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == ExperienceCreatedV1.event_type
            )
        )
    assert row is not None
    stored = stack.projector.stored_event_from_row(row)
    wrong = stored.__class__(
        event_id=stored.event_id + 100,
        aggregate_type=stored.aggregate_type,
        aggregate_id=OTHER_OWNER_ID,
        sequence=stored.sequence,
        event_type=stored.event_type,
        payload=stored.payload,
        actor_agent_id=stored.actor_agent_id,
        causation_id=stored.causation_id,
        occurred_at=stored.occurred_at,
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="aggregate",
        ):
            await stack.projector.apply(uow.session, wrong)
