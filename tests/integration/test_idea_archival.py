from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, update
from tests.integration.test_create_experience import (
    OWNER_ID,
    create,
)
from tests.integration.test_inspiration_run import (
    FakeSnapshotBuilder,
    ImmediateDeadlineRunner,
)
from tests.integration.test_inspiration_run import (
    command as inspiration_command,
)
from tests.integration.test_inspiration_run import (
    request as inspiration_request,
)

from experience_hub.clock import FrozenClock
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    PendingEvent,
    StructuredReason,
)
from experience_hub.experiences.events import register_experience_events
from experience_hub.experiences.models import Temperature, VersionContent
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import Uuid4Generator
from experience_hub.inspiration.deadlines import BoundedGenerationRunner
from experience_hub.inspiration.events import (
    InspirationIdeaArchivedV1,
    register_inspiration_events,
)
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.lifecycle import (
    IdeaLifecycleService,
    InspirationIdeaArchivePlanner,
)
from experience_hub.inspiration.models import (
    EvaluationVerdict,
    IdeaDraft,
    IdeaEvaluation,
    IdeaOwnerDecision,
    InspirationOperator,
    MechanismMaturity,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationProjectionIntegrityError,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.repository import InspirationRepository
from experience_hub.inspiration.request_hashing import (
    evaluation_command_request,
)
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.service import InspirationRunExecutor
from experience_hub.lifecycle.contracts import decode_lifecycle_result
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import LifecycleService
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
    AgentRow,
    DomainEventRow,
    ExperienceStateRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    InspirationIdeaRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_inspiration_source_validator,
)

ORIGIN = datetime(2026, 1, 1, 12, tzinfo=UTC)
MICROSECOND = timedelta(microseconds=1)


@dataclass(slots=True)
class QueuedGenerator:
    """Deterministic in-memory adapter whose idea bodies vary by test."""

    mechanisms: list[str] = field(default_factory=list)

    @property
    def reserves_output_tokens(self) -> bool:
        return False

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        return None

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        _ = (goal, context, branch_limit, output_token_limit)
        if not self.mechanisms:
            raise AssertionError("A test mechanism must be queued before generation")
        item = frozen_items[0]
        mechanism = self.mechanisms.pop(0)
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title=f"Archive boundary for {mechanism}",
                    hypothesis=f"{mechanism} remains testable over time.",
                    mechanism=mechanism,
                    predictions=("The predicted signal remains measurable.",),
                    falsifiers=("The predicted signal cannot be measured.",),
                    assumptions=("The observation window remains stable.",),
                    proposed_test="Measure the signal at the policy boundary.",
                    evidence=(
                        SnapshotEvidenceReference(
                            id=item.snapshot_item_id,
                            stable_evidence_key=item.stable_evidence_key,
                        ),
                    ),
                ),
            ),
            output_tokens_consumed=0,
        )


@dataclass(slots=True)
class Stack:
    database: Database
    clock: FrozenClock
    receipts: ReceiptStore
    executor: CommandExecutor
    service: ExperienceService
    experience_repository: ExperienceRepository
    inspiration_executor: InspirationRunExecutor
    generator: QueuedGenerator
    snapshot_builder: FakeSnapshotBuilder
    lifecycle_config: LifecycleConfig
    snapshot_number: int = 0


@dataclass(frozen=True, slots=True)
class SeededIdea:
    idea_id: UUID
    run_id: UUID
    cluster_id: str


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> Stack:
    alembic = Config(repository_root / "alembic.ini")
    alembic.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(alembic, "head")

    lifecycle_config = LifecycleConfig()
    registry = EventRegistry()
    register_experience_events(registry)
    register_inspiration_events(registry)
    experience_projector = ExperienceProjector(registry, lifecycle_config)
    manager = ProjectionManager(
        ProjectionRegistry(
            (
                experience_projector,
                InspirationRunProjector(registry),
                MechanismIncubationProjector(registry),
                IdeaStateProjector(registry),
            )
        ),
        source_validator=SourceValidator(registry),
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
    )
    async with database.transaction() as uow:
        uow.session.add(
            AgentRow(
                agent_id=OWNER_ID,
                name="Idea archive owner",
                created_at=ORIGIN,
            )
        )

    clock = FrozenClock(ORIGIN)
    ids = Uuid4Generator()
    receipts = ReceiptStore(clock=clock, id_generator=ids)
    executor = CommandExecutor(
        database=database,
        receipt_store=receipts,
        clock=clock,
    )
    experience_repository = ExperienceRepository(event_registry=registry)
    writer = ExperienceWriter(
        id_generator=ids,
        repository=experience_repository,
        lifecycle_config=lifecycle_config,
    )
    service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=writer,
        lifecycle_config=lifecycle_config,
    )
    generator = QueuedGenerator()
    snapshot_builder = FakeSnapshotBuilder()
    inspiration_executor = InspirationRunExecutor(
        database=database,
        receipt_store=receipts,
        repository=InspirationRepository(registry),
        snapshot_builder=snapshot_builder,
        generator_factory=lambda _: generator,
        generation_runner=BoundedGenerationRunner(
            deadline_runner=ImmediateDeadlineRunner()
        ),
        response_codec=InspirationResponseCodec(),
        clock=clock,
        id_generator=ids,
    )
    return Stack(
        database=database,
        clock=clock,
        receipts=receipts,
        executor=executor,
        service=service,
        experience_repository=experience_repository,
        inspiration_executor=inspiration_executor,
        generator=generator,
        snapshot_builder=snapshot_builder,
        lifecycle_config=lifecycle_config,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "idea-archival.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def seed_idea(
    stack: Stack,
    *,
    key: str,
    mechanism: str,
    source_id: UUID | None = None,
    source_version_id: UUID | None = None,
    source_content_hash: str | None = None,
) -> SeededIdea:
    source_values = (source_id, source_version_id, source_content_hash)
    if any(value is None for value in source_values) and any(
        value is not None for value in source_values
    ):
        raise ValueError("snapshot source overrides must be provided together")
    stack.snapshot_number += 1
    stack.generator.mechanisms.append(mechanism)
    stack.snapshot_builder.item_ids.append(uuid4())
    if (
        source_id is not None
        and source_version_id is not None
        and source_content_hash is not None
    ):
        stack.snapshot_builder.source_ids.append(source_id)
        stack.snapshot_builder.source_version_ids.append(source_version_id)
        stack.snapshot_builder.content_hashes.append(source_content_hash)
    else:
        stack.snapshot_builder.content_hashes.append(f"{stack.snapshot_number:064x}")
    run = inspiration_command(goal=f"Observe {mechanism}")
    response = await stack.inspiration_executor.execute(
        request=inspiration_request(key=key, run=run),
        run=run,
    )
    assert response.status_code == 201
    run_id = UUID(json.loads(response.body)["data"]["run_id"])
    async with stack.database.read_session() as session:
        idea = await session.scalar(
            select(InspirationIdeaRow).where(InspirationIdeaRow.run_id == run_id)
        )
        assert idea is not None
        state = await session.get(IdeaStateRow, idea.idea_id)
        assert state is not None
        return SeededIdea(
            idea_id=idea.idea_id,
            run_id=run_id,
            cluster_id=state.mechanism_cluster_id,
        )


async def due_events(
    stack: Stack,
    *,
    evaluated_at: datetime,
    cycle_id: UUID,
) -> tuple[PendingEvent, ...]:
    async with stack.database.read_session() as session:
        return await InspirationIdeaArchivePlanner().due_archive_events(
            session=session,
            evaluated_at=evaluated_at,
            cycle_id=cycle_id,
        )


def assert_policy_archive_event(
    event: PendingEvent,
    *,
    idea_id: UUID,
    evaluated_at: datetime,
    cycle_id: UUID,
) -> None:
    assert event.aggregate_type == "idea"
    assert event.aggregate_id == idea_id
    assert event.event_type == "inspiration.idea_archived"
    assert event.actor_agent_id is None
    assert event.occurred_at == evaluated_at
    assert isinstance(event.payload, InspirationIdeaArchivedV1)
    assert event.payload.idea_id == idea_id
    assert event.payload.owner_agent_id == OWNER_ID
    assert event.payload.owner_decision_before is IdeaOwnerDecision.ACTIVE
    assert event.payload.owner_decision_after is IdeaOwnerDecision.ARCHIVED
    assert event.payload.cycle_id == cycle_id
    assert event.payload.cycle_id is not None
    assert event.payload.reason == StructuredReason.policy_due()


def lifecycle_service(stack: Stack) -> LifecycleService:
    return LifecycleService(
        clock=stack.clock,
        receipt_store=stack.receipts,
        repository=LifecycleRepository(),
        mutation_writer=ExperienceMutationWriter(
            repository=stack.experience_repository,
            lifecycle_config=stack.lifecycle_config,
        ),
        config=stack.lifecycle_config,
        idea_archive_planner=InspirationIdeaArchivePlanner(),
    )


async def run_lifecycle(
    stack: Stack,
    service: LifecycleService,
    *,
    key: str,
    evaluated_at: datetime,
) -> CommandResult:
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=key,
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": evaluated_at, "mode": "manual"},
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=command,
            mode="manual",
        )

    return await stack.executor.execute(request, handler)


@pytest.mark.asyncio
async def test_noncandidate_archives_at_180_day_boundary_in_uuid_order(
    stack: Stack,
) -> None:
    ideas = (
        await seed_idea(
            stack,
            key="noncandidate-zinc",
            mechanism="zinc lattice pressure memory",
        ),
        await seed_idea(
            stack,
            key="noncandidate-cedar",
            mechanism="cedar root moisture exchange",
        ),
        await seed_idea(
            stack,
            key="noncandidate-orbit",
            mechanism="orbital ice spectral reflection",
        ),
    )
    cycle_id = uuid4()
    boundary = ORIGIN + timedelta(days=180)

    assert (
        await due_events(
            stack,
            evaluated_at=boundary - MICROSECOND,
            cycle_id=cycle_id,
        )
        == ()
    )
    events = await due_events(
        stack,
        evaluated_at=boundary,
        cycle_id=cycle_id,
    )

    expected_ids = tuple(
        sorted((idea.idea_id for idea in ideas), key=lambda value: value.bytes)
    )
    assert tuple(event.aggregate_id for event in events) == expected_ids
    for event, idea_id in zip(events, expected_ids, strict=True):
        assert_policy_archive_event(
            event,
            idea_id=idea_id,
            evaluated_at=boundary,
            cycle_id=cycle_id,
        )


@pytest.mark.asyncio
async def test_rebuild_replays_policy_archive_before_later_candidate_promotion(
    stack: Stack,
) -> None:
    idea = await seed_idea(
        stack,
        key="historical-policy-archive-idea",
        mechanism="historical archive then supported promotion",
    )
    archived_at = stack.clock.advance(timedelta(days=180))
    archived = await run_lifecycle(
        stack,
        lifecycle_service(stack),
        key="historical-policy-archive-cycle",
        evaluated_at=archived_at,
    )
    assert archived.status_code == 200

    evaluated_at = stack.clock.advance(timedelta(minutes=1))
    async with stack.database.read_session() as session:
        item = await session.scalar(
            select(InspirationSnapshotItemRow).where(
                InspirationSnapshotItemRow.run_id == idea.run_id
            )
        )
    assert item is not None
    evaluation = IdeaEvaluation(
        evaluator_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        verdict=EvaluationVerdict.SUPPORTED,
        evidence=(
            SnapshotEvidenceReference(
                id=item.snapshot_item_id,
                stable_evidence_key=item.stable_evidence_key,
            ),
        ),
        evaluated_at=evaluated_at,
    )
    registry = EventRegistry()
    register_experience_events(registry)
    register_inspiration_events(registry)
    idea_service = IdeaLifecycleService(
        clock=stack.clock,
        receipt_store=stack.receipts,
        repository=InspirationRepository(registry),
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await idea_service.evaluate(
            uow=uow,
            evaluation=evaluation,
            command_context=command_context,
        )

    evaluated = await stack.executor.execute(
        evaluation_command_request(
            evaluation,
            idempotency_key="historical-post-archive-evaluation",
        ),
        handler,
    )
    assert evaluated.status_code == 200
    async with stack.database.read_session() as session:
        cluster = await session.get(MechanismIncubationRow, idea.cluster_id)
    assert cluster is not None
    assert cluster.maturity == MechanismMaturity.CANDIDATE.value

    manager = cast(Any, stack.database)._projection_applier
    assert isinstance(manager, ProjectionManager)
    report = await manager.verify(stack.database)
    assert report.matches


@pytest.mark.asyncio
async def test_omitted_time_archive_validates_after_restart_and_rebuild(
    stack: Stack,
    tmp_path: Path,
) -> None:
    source_status, source = await create(
        stack,
        key="omitted-time-policy-archive-source",
        value=VersionContent(
            body="A bounded observation.",
            summary="The queue drains when work is acknowledged.",
            mechanism="Acknowledgement releases capacity.",
            tags=("queue",),
            applicability=("bounded queue",),
            evidence=(),
            falsifiers=("Capacity remains blocked after acknowledgement.",),
        ),
    )
    assert source_status == 201
    idea = await seed_idea(
        stack,
        key="omitted-time-policy-archive-idea",
        mechanism="omitted receipt time archive proof",
        source_id=UUID(source["data"]["experience_id"]),
        source_version_id=UUID(source["data"]["version_id"]),
        source_content_hash=source["data"]["content_hash"],
    )
    receipt_time = stack.clock.advance(timedelta(days=180))
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key="omitted-time-policy-archive-cycle",
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": None, "mode": "manual"},
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await lifecycle_service(stack).run(
            uow=uow,
            evaluated_at=receipt_time - timedelta(minutes=15),
            command=command_context,
            mode="manual",
            evaluated_at_was_omitted=True,
        )

    archived = await stack.executor.execute(request, handler)

    assert archived.status_code == 200
    result = decode_lifecycle_result(archived.body)
    assert result.evaluated_at == receipt_time
    assert result.idea_archive_count == 1
    async with stack.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ARCHIVED.value

    await stack.database.dispose()
    registry = EventRegistry()
    register_experience_events(registry)
    register_inspiration_events(registry)
    source_validator = SourceValidator(registry)
    register_inspiration_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry(
            (
                ExperienceProjector(registry, stack.lifecycle_config),
                InspirationRunProjector(registry),
                MechanismIncubationProjector(registry),
                IdeaStateProjector(registry),
            )
        ),
        source_validator=source_validator,
    )
    restarted = Database.create(
        f"sqlite+aiosqlite:///{tmp_path / 'idea-archival.sqlite3'}",
        event_registry=registry,
        projection_applier=manager,
    )
    try:
        await manager.validate_startup(restarted)
        report = await manager.verify(restarted)
        assert report.matches
    finally:
        await restarted.dispose()


@pytest.mark.parametrize(
    ("operation_scope", "age", "expected_message"),
    (
        ("lifecycle.run", timedelta(0), "not due"),
        (
            "forged.lifecycle.run",
            timedelta(days=180),
            "lifecycle receipt",
        ),
    ),
)
@pytest.mark.asyncio
async def test_policy_archive_requires_due_state_and_a_lifecycle_receipt(
    stack: Stack,
    operation_scope: str,
    age: timedelta,
    expected_message: str,
) -> None:
    idea = await seed_idea(
        stack,
        key=f"forged-policy-{operation_scope}",
        mechanism=f"forged policy proof {operation_scope}",
    )
    archived_at = ORIGIN + age
    cycle_id = lifecycle_service(stack).cycle_id(archived_at)
    event = (
        await due_events(
            stack,
            evaluated_at=ORIGIN + timedelta(days=180),
            cycle_id=cycle_id,
        )
    )[0]
    forged = PendingEvent(
        aggregate_type=event.aggregate_type,
        aggregate_id=event.aggregate_id,
        event_type=event.event_type,
        payload=event.payload,
        actor_agent_id=event.actor_agent_id,
        occurred_at=archived_at,
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        await stack.receipts.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="lifecycle_cycle",
            resource_id=cycle_id,
        )
        await uow.append_events(command_context, (forged,))
        return StoredResponse(
            status_code=200,
            body=b'{"data":{"forged":true}}',
        )

    with pytest.raises(
        InspirationProjectionIntegrityError,
        match=expected_message,
    ):
        await stack.executor.execute(
            CommandRequest(
                caller_scope="system:local",
                operation_scope=operation_scope,
                idempotency_key=(f"forged-policy-archive:{operation_scope}"),
                method="POST",
                route_template="/v1/lifecycle:run",
                body={
                    "evaluated_at": archived_at,
                    "mode": "manual",
                },
            ),
            handler,
        )

    async with stack.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_later_recurrence_does_not_postpone_an_older_idea_archive(
    stack: Stack,
) -> None:
    first = await seed_idea(
        stack,
        key="recurrence-first",
        mechanism="recurring noncandidate signal",
    )
    stack.clock.advance(timedelta(days=100))
    second = await seed_idea(
        stack,
        key="recurrence-second",
        mechanism="recurring noncandidate signal",
    )
    assert second.cluster_id == first.cluster_id

    boundary = ORIGIN + timedelta(days=180)
    events = await due_events(
        stack,
        evaluated_at=boundary,
        cycle_id=uuid4(),
    )

    assert tuple(event.aggregate_id for event in events) == (first.idea_id,)


@pytest.mark.asyncio
async def test_candidate_uses_later_candidate_since_and_365_day_boundary(
    stack: Stack,
) -> None:
    first = await seed_idea(
        stack,
        key="candidate-first",
        mechanism="shared candidate transfer mechanism",
    )
    candidate_since = stack.clock.advance(timedelta(days=20))
    later_signal = await seed_idea(
        stack,
        key="candidate-signal",
        mechanism="shared candidate transfer mechanism",
    )
    assert later_signal.cluster_id == first.cluster_id
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(MechanismIncubationRow)
            .where(MechanismIncubationRow.cluster_id == first.cluster_id)
            .values(
                maturity=MechanismMaturity.CANDIDATE.value,
                candidate_since=candidate_since,
                supported_count=1,
            )
        )
        await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == later_signal.idea_id)
            .values(owner_decision=IdeaOwnerDecision.REJECTED.value)
        )

    cycle_id = uuid4()
    boundary = candidate_since + timedelta(days=365)
    assert (
        await due_events(
            stack,
            evaluated_at=boundary - MICROSECOND,
            cycle_id=cycle_id,
        )
        == ()
    )

    events = await due_events(
        stack,
        evaluated_at=boundary,
        cycle_id=cycle_id,
    )

    assert len(events) == 1
    assert_policy_archive_event(
        events[0],
        idea_id=first.idea_id,
        evaluated_at=boundary,
        cycle_id=cycle_id,
    )


@pytest.mark.asyncio
async def test_only_active_ideas_are_eligible_for_policy_archive(
    stack: Stack,
) -> None:
    active = await seed_idea(
        stack,
        key="active-idea",
        mechanism="active capillary transport",
    )
    adopted = await seed_idea(
        stack,
        key="adopted-idea",
        mechanism="adopted magnetic suspension",
    )
    rejected = await seed_idea(
        stack,
        key="rejected-idea",
        mechanism="rejected thermal braid",
    )
    archived = await seed_idea(
        stack,
        key="archived-idea",
        mechanism="archived acoustic coupling",
    )
    status, experience = await create(
        stack,
        key="adopted-result",
        importance=0.20,
        confidence=0.20,
    )
    assert status == 201
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == adopted.idea_id)
            .values(
                owner_decision=IdeaOwnerDecision.ADOPTED.value,
                resulting_experience_id=UUID(experience["data"]["experience_id"]),
                resulting_version_id=UUID(experience["data"]["version_id"]),
            )
        )
        await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == rejected.idea_id)
            .values(owner_decision=IdeaOwnerDecision.REJECTED.value)
        )
        await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == archived.idea_id)
            .values(owner_decision=IdeaOwnerDecision.ARCHIVED.value)
        )

    cycle_id = uuid4()
    evaluated_at = ORIGIN + timedelta(days=1_000)
    events = await due_events(
        stack,
        evaluated_at=evaluated_at,
        cycle_id=cycle_id,
    )

    assert len(events) == 1
    assert_policy_archive_event(
        events[0],
        idea_id=active.idea_id,
        evaluated_at=evaluated_at,
        cycle_id=cycle_id,
    )


@pytest.mark.asyncio
async def test_lifecycle_appends_idea_archives_after_experience_events_and_replays(
    stack: Stack,
) -> None:
    idea = await seed_idea(
        stack,
        key="lifecycle-due-idea",
        mechanism="lifecycle archive ordering mechanism",
    )
    status, experience = await create(
        stack,
        key="lifecycle-due-experience",
        importance=0.20,
        confidence=0.20,
    )
    assert status == 201
    experience_id = UUID(experience["data"]["experience_id"])
    evaluated_at = stack.clock.advance(timedelta(days=365))
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(
                temperature=Temperature.COLD,
                confidence=0.20,
                importance=0.20,
                activation_score=0.0,
                access_strength=0.0,
                strength_updated_at=ORIGIN,
                last_transition_at=evaluated_at - timedelta(days=91),
                last_lifecycle_evaluated_at=None,
                consecutive_below_threshold=0,
                pinned=False,
            )
        )
    service = lifecycle_service(stack)

    first = await run_lifecycle(
        stack,
        service,
        key="archive-cycle-first",
        evaluated_at=evaluated_at,
    )

    assert first.status_code == 200
    first_result = decode_lifecycle_result(first.body)
    assert (
        first_result.evaluated_count,
        first_result.transition_count,
        first_result.archive_count,
        first_result.idea_archive_count,
    ) == (1, 1, 1, 1)
    async with stack.database.read_session() as session:
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "lifecycle.run",
                IdempotencyRecordRow.idempotency_key == "archive-cycle-first",
            )
        )
        assert receipt is not None
        cycle_events = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
    assert [event.event_type for event in cycle_events] == [
        "experience.lifecycle_evaluated",
        "experience.archived",
        "experience.temperature_changed",
        "inspiration.idea_archived",
    ]
    assert {event.causation_id for event in cycle_events} == {receipt.receipt_id}
    assert idea_state is not None
    assert idea_state.owner_decision == IdeaOwnerDecision.ARCHIVED.value

    same_key = await run_lifecycle(
        stack,
        service,
        key="archive-cycle-first",
        evaluated_at=evaluated_at,
    )
    cross_key = await run_lifecycle(
        stack,
        service,
        key="archive-cycle-replay",
        evaluated_at=evaluated_at,
    )

    assert same_key.replayed
    assert same_key.body == cross_key.body == first.body
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(
                    DomainEventRow.event_type == InspirationIdeaArchivedV1.event_type
                )
            )
            == 1
        )


@pytest.mark.asyncio
async def test_archive_clock_regression_is_atomic_and_replayable(
    stack: Stack,
) -> None:
    idea = await seed_idea(
        stack,
        key="regression-idea",
        mechanism="regression causal anchor mechanism",
    )
    false_due_signal = ORIGIN - timedelta(days=182)
    evaluated_at = ORIGIN - timedelta(days=1)
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == idea.idea_id)
            .values(last_signal_at=false_due_signal)
        )
        await uow.session.execute(
            update(MechanismIncubationRow)
            .where(MechanismIncubationRow.cluster_id == idea.cluster_id)
            .values(last_signal_at=false_due_signal)
        )
    async with stack.database.read_session() as session:
        event_count_before = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow)) or 0
        )

    result = await run_lifecycle(
        stack,
        lifecycle_service(stack),
        key="archive-clock-regression",
        evaluated_at=evaluated_at,
    )

    assert result.status_code == 409
    assert json.loads(result.body)["error"]["code"] == "clock_regression"
    async with stack.database.read_session() as session:
        event_count_after = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow)) or 0
        )
        state = await session.get(IdeaStateRow, idea.idea_id)
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "lifecycle.run",
                IdempotencyRecordRow.idempotency_key == "archive-clock-regression",
            )
        )
    assert event_count_after == event_count_before
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value
    assert receipt is not None
    assert receipt.state == "completed"
    assert receipt.response_status_code == 409
    assert receipt.result_resource_type is None
    assert receipt.result_resource_id is None
