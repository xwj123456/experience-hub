from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    arrange_pending_capsule,
)
from tests.integration.test_capsule_adoption import (
    SOURCE_CONTENT as CAPSULE_SOURCE_CONTENT,
)
from tests.integration.test_capsule_adoption import (
    AdoptionStack as CapsuleAdoptionStack,
)
from tests.integration.test_capsule_adoption import (
    build_stack as build_capsule_adoption_stack,
)
from tests.integration.test_idea_adoption import (
    OWNER_A,
    OWNER_B,
    AdoptionStack,
    AllEvidenceGenerator,
    adopt,
    adoption_data,
    build_adoption_stack,
    create_experience,
    experience_spec,
    generate_idea,
    run_request,
    uid,
)
from tests.integration.test_inspiration_run import ImmediateDeadlineRunner
from tests.integration.test_snapshot_freeze import _create_cold

from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    VersionContent,
)
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.deadlines import BoundedGenerationRunner
from experience_hub.inspiration.events import register_inspiration_events
from experience_hub.inspiration.models import (
    EvidenceSourceType,
    InspirationOperator,
)
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.repository import InspirationRepository
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.service import InspirationRunExecutor
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.retrieval.service import ExperienceEvidenceReader
from experience_hub.sharing.queries import InboxEvidenceReader
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceTermRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    InboxItemRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationSnapshotItemRow,
)
from experience_hub.storage.validation import (
    SourceValidator,
    register_inspiration_source_validator,
)

SOURCE_CONTENT = VersionContent(
    body="Acknowledgement is observed before bounded capacity is released.",
    summary="An owned source for isolated inspiration",
    mechanism="Acknowledgement releases bounded capacity.",
    tags=("queue", "isolation"),
    applicability=("bounded queue",),
    evidence=(TypedEvidence(type="experiment", id="owner-scoped-source"),),
    falsifiers=("Capacity remains blocked after acknowledgement.",),
)

_EXPERIENCE_TABLES = (
    ExperienceRow,
    ExperienceVersionRow,
    ExperiencePayloadRow,
    ExperienceLinkRow,
    ExperienceStateRow,
    ExperienceTermRow,
)


def _enable_inspiration_source_validation(stack: AdoptionStack) -> None:
    validator = stack.manager._source_validator
    assert isinstance(validator, SourceValidator)
    register_inspiration_source_validator(validator)


@pytest.fixture
async def isolation_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-e2e-isolation.sqlite3",
    )
    _enable_inspiration_source_validation(stack)
    try:
        yield stack
    finally:
        await stack.database.dispose()


@pytest.fixture
async def production_isolation_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[
    tuple[CapsuleAdoptionStack, InspirationRunExecutor, ProjectionManager]
]:
    stack = await build_capsule_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "production-inspiration-isolation.sqlite3",
    )
    manager = cast(
        ProjectionManager,
        stack.database._projection_applier,  # noqa: SLF001
    )
    register_inspiration_events(stack.registry)
    validator = manager._source_validator  # noqa: SLF001
    assert isinstance(validator, SourceValidator)
    register_inspiration_source_validator(validator)
    manager.registry.register(ExperienceTermsProjector(stack.registry))
    manager.registry.register(InspirationRunProjector(stack.registry))
    manager.registry.register(MechanismIncubationProjector(stack.registry))
    manager.registry.register(IdeaStateProjector(stack.registry))

    snapshot_builder = SnapshotBuilder(
        experience_reader=ExperienceEvidenceReader(
            clock=stack.clock,
            query=ExperienceQuery(event_registry=stack.registry),
        ),
        inbox_reader=InboxEvidenceReader(
            repository=SharingRepository(event_registry=stack.registry)
        ),
        id_generator=SequenceIdGenerator(
            tuple(uid(value) for value in range(90_001, 90_051))
        ),
    )
    generator = AllEvidenceGenerator()
    run_executor = InspirationRunExecutor(
        database=stack.database,
        receipt_store=stack.receipts,
        repository=InspirationRepository(stack.registry),
        snapshot_builder=snapshot_builder,
        generator_factory=lambda _kind: generator,
        generation_runner=BoundedGenerationRunner(
            deadline_runner=ImmediateDeadlineRunner(),
        ),
        response_codec=InspirationResponseCodec(),
        clock=stack.clock,
        id_generator=SequenceIdGenerator(
            tuple(uid(value) for value in range(91_001, 91_051))
        ),
    )
    try:
        yield stack, run_executor, manager
    finally:
        await stack.database.dispose()


type Stack = AdoptionStack | CapsuleAdoptionStack


async def _experience_row_counts(stack: Stack) -> tuple[int, ...]:
    async with stack.database.read_session() as session:
        counts: list[int] = []
        for table in _EXPERIENCE_TABLES:
            counts.append(
                int(await session.scalar(select(func.count()).select_from(table)) or 0)
            )
        counts.append(
            int(
                await session.scalar(
                    select(func.count())
                    .select_from(DomainEventRow)
                    .where(DomainEventRow.aggregate_type == "experience")
                )
                or 0
            )
        )
    return tuple(counts)


async def _experience_source_rows(
    stack: Stack,
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    retained: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    async with stack.database.read_session() as session:
        for table in _EXPERIENCE_TABLES:
            rows = tuple(
                tuple(row)
                for row in (
                    await session.execute(
                        text(f'SELECT * FROM "{table.__tablename__}" ORDER BY rowid')
                    )
                )
            )
            retained.append((table.__tablename__, rows))
    return tuple(retained)


async def _experience_states(stack: Stack) -> tuple[tuple[object, ...], ...]:
    async with stack.database.read_session() as session:
        return tuple(
            tuple(row)
            for row in (
                await session.execute(
                    select(
                        ExperienceStateRow.experience_id,
                        ExperienceStateRow.current_version_id,
                        ExperienceStateRow.current_content_hash,
                        ExperienceStateRow.temperature,
                        ExperienceStateRow.importance,
                        ExperienceStateRow.confidence,
                        ExperienceStateRow.activation_score,
                        ExperienceStateRow.source_trust,
                        ExperienceStateRow.access_count,
                        ExperienceStateRow.access_strength,
                        ExperienceStateRow.strength_updated_at,
                        ExperienceStateRow.last_accessed_at,
                        ExperienceStateRow.last_transition_at,
                        ExperienceStateRow.last_lifecycle_evaluated_at,
                        ExperienceStateRow.consecutive_below_threshold,
                        ExperienceStateRow.pinned,
                        ExperienceStateRow.projection_event_id,
                    ).order_by(ExperienceStateRow.experience_id)
                )
            ).all()
        )


async def _inbox_states(stack: Stack) -> tuple[tuple[object, ...], ...]:
    async with stack.database.read_session() as session:
        return tuple(
            tuple(row)
            for row in (
                await session.execute(
                    select(
                        InboxItemRow.item_id,
                        InboxItemRow.recipient_agent_id,
                        InboxItemRow.capsule_id,
                        InboxItemRow.state,
                        InboxItemRow.projection_event_id,
                    ).order_by(InboxItemRow.item_id)
                )
            ).all()
        )


async def _external_event_counts(stack: Stack) -> tuple[tuple[str, int], ...]:
    external_aggregates = (
        "experience",
        "topic",
        "subscription",
        "capsule",
        "inbox_item",
    )
    async with stack.database.read_session() as session:
        rows = (
            await session.execute(
                select(
                    DomainEventRow.aggregate_type,
                    func.count(),
                )
                .where(DomainEventRow.aggregate_type.in_(external_aggregates))
                .group_by(DomainEventRow.aggregate_type)
                .order_by(DomainEventRow.aggregate_type)
            )
        ).all()
    return tuple((str(aggregate_type), int(count)) for aggregate_type, count in rows)


async def _hypothesis_counts(stack: AdoptionStack) -> tuple[int, int]:
    async with stack.database.read_session() as session:
        identities = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.kind == ExperienceKind.HYPOTHESIS)
            )
            or 0
        )
        versions = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceVersionRow)
                .join(
                    ExperienceRow,
                    ExperienceRow.experience_id == ExperienceVersionRow.experience_id,
                )
                .where(ExperienceRow.kind == ExperienceKind.HYPOTHESIS)
            )
            or 0
        )
    return identities, versions


async def _owned_source(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
    key: str,
) -> tuple[UUID, UUID, str]:
    created = await create_experience(
        stack,
        owner_agent_id=owner_agent_id,
        content=SOURCE_CONTENT,
        key=key,
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.LOCAL,
    )
    return created.experience_id, created.version_id, created.content_hash


@pytest.mark.asyncio
async def test_generation_is_read_only_for_every_experience_source_and_projection(
    production_isolation_stack: tuple[
        CapsuleAdoptionStack,
        InspirationRunExecutor,
        ProjectionManager,
    ],
) -> None:
    stack, run_executor, manager = production_isolation_stack
    assert ADOPTER_ID == OWNER_B
    capsule = await arrange_pending_capsule(stack)
    experience_id = await _create_cold(
        stack,
        key="generation-read-only-source",
        content=SOURCE_CONTENT,
    )
    before_counts = await _experience_row_counts(stack)
    before_sources = await _experience_source_rows(stack)
    before_states = await _experience_states(stack)
    before_inbox = await _inbox_states(stack)
    before_external_events = await _external_event_counts(stack)

    run = StartInspirationRun(
        owner_agent_id=ADOPTER_ID,
        goal=SOURCE_CONTENT.summary,
        context=(
            f"{SOURCE_CONTENT.mechanism}\n"
            f"{CAPSULE_SOURCE_CONTENT.summary}\n"
            f"{CAPSULE_SOURCE_CONTENT.mechanism}"
        ),
        operators=(InspirationOperator.CAUSAL_GAP,),
        include_inbox=True,
    )
    result = await run_executor.execute(
        request=run_request(run, key="production-generation-read-only-run"),
        run=run,
    )
    assert result.status_code == 201

    assert await _experience_row_counts(stack) == before_counts
    assert await _experience_source_rows(stack) == before_sources
    assert await _experience_states(stack) == before_states
    assert await _inbox_states(stack) == before_inbox
    assert await _external_event_counts(stack) == before_external_events
    async with stack.database.read_session() as session:
        inspiration_count_values: list[int] = []
        for table in (
            InspirationRunRow,
            InspirationSnapshotItemRow,
            InspirationIdeaRow,
            IdeaOccurrenceRow,
            IdeaStateRow,
        ):
            inspiration_count_values.append(
                int(await session.scalar(select(func.count()).select_from(table)) or 0)
            )
        snapshot_sources = tuple(
            (
                await session.scalars(
                    select(InspirationSnapshotItemRow.source_type).order_by(
                        InspirationSnapshotItemRow.rank
                    )
                )
            ).all()
        )
        snapshot_source_ids = set(
            (await session.scalars(select(InspirationSnapshotItemRow.source_id))).all()
        )
    inspiration_counts = tuple(inspiration_count_values)
    assert inspiration_counts == (1, 2, 1, 1, 1)
    assert set(snapshot_sources) == {
        EvidenceSourceType.EXPERIENCE,
        EvidenceSourceType.CAPSULE,
    }
    assert snapshot_source_ids == {experience_id, capsule.capsule_id}
    assert (await manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_explicit_adoption_creates_then_reuses_exactly_one_hypothesis(
    isolation_stack: AdoptionStack,
) -> None:
    stack = isolation_stack
    experience_id, version_id, content_hash = await _owned_source(
        stack,
        owner_agent_id=OWNER_A,
        key="adoption-reuse-source",
    )
    specs = tuple(
        (
            experience_spec(
                marker=marker,
                experience_id=experience_id,
                version_id=version_id,
                content_hash=content_hash,
            ),
        )
        for marker in (111, 112)
    )
    first = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="adoption-create-run",
        specs=specs[0],
    )
    second = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="adoption-reuse-run",
        specs=specs[1],
    )
    assert await _hypothesis_counts(stack) == (0, 0)

    created = adoption_data(
        await adopt(
            stack,
            owner_agent_id=OWNER_A,
            idea_id=first.idea_id,
            key="adoption-create",
        )
    )
    assert created["created"] is True
    assert await _hypothesis_counts(stack) == (1, 1)

    reused = adoption_data(
        await adopt(
            stack,
            owner_agent_id=OWNER_A,
            idea_id=second.idea_id,
            key="adoption-reuse",
        )
    )
    assert reused["created"] is False
    assert reused["experience"] == created["experience"]
    assert await _hypothesis_counts(stack) == (1, 1)

    async with stack.database.read_session() as session:
        records = tuple(
            (
                await session.scalars(
                    select(IdeaAdoptionRecordRow).order_by(
                        IdeaAdoptionRecordRow.idea_id
                    )
                )
            ).all()
        )
    assert len(records) == 2
    assert {record.idea_id for record in records} == {
        first.idea_id,
        second.idea_id,
    }
    assert len({record.resulting_experience_id for record in records}) == 1
    assert len({record.resulting_version_id for record in records}) == 1
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_private_ideas_never_become_cross_owner_duplicate_or_adoption_handles(
    isolation_stack: AdoptionStack,
) -> None:
    stack = isolation_stack
    a_source = await _owned_source(
        stack,
        owner_agent_id=OWNER_A,
        key="owner-a-private-source",
    )
    b_source = await _owned_source(
        stack,
        owner_agent_id=OWNER_B,
        key="owner-b-private-source",
    )

    first_a = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="owner-a-first-private-run",
        specs=(
            experience_spec(
                marker=121,
                experience_id=a_source[0],
                version_id=a_source[1],
                content_hash=a_source[2],
            ),
        ),
    )
    first_b = await generate_idea(
        stack,
        owner_agent_id=OWNER_B,
        key="owner-b-private-run",
        specs=(
            experience_spec(
                marker=122,
                experience_id=b_source[0],
                version_id=b_source[1],
                content_hash=b_source[2],
            ),
        ),
    )
    second_a = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="owner-a-second-private-run",
        specs=(
            experience_spec(
                marker=123,
                experience_id=a_source[0],
                version_id=a_source[1],
                content_hash=a_source[2],
            ),
        ),
    )

    async with stack.database.read_session() as session:
        rows = {
            row.idea_id: row
            for row in (
                await session.scalars(
                    select(InspirationIdeaRow).where(
                        InspirationIdeaRow.idea_id.in_(
                            (first_a.idea_id, first_b.idea_id, second_a.idea_id)
                        )
                    )
                )
            ).all()
        }
        states = {
            row.idea_id: row
            for row in (
                await session.scalars(
                    select(IdeaStateRow).where(
                        IdeaStateRow.idea_id.in_(
                            (first_a.idea_id, first_b.idea_id, second_a.idea_id)
                        )
                    )
                )
            ).all()
        }
    assert rows[first_a.idea_id].duplicate_relation is None
    assert rows[first_b.idea_id].duplicate_relation is None
    assert rows[second_a.idea_id].duplicate_relation == first_a.idea_id
    assert states[first_a.idea_id].owner_agent_id == OWNER_A
    assert states[first_b.idea_id].owner_agent_id == OWNER_B
    assert states[second_a.idea_id].owner_agent_id == OWNER_A

    foreign = await adopt(
        stack,
        owner_agent_id=OWNER_A,
        idea_id=first_b.idea_id,
        key="cross-owner-adoption",
    )
    unknown = await adopt(
        stack,
        owner_agent_id=OWNER_A,
        idea_id=uid(999_999),
        key="unknown-owner-adoption",
    )

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.body == unknown.body
    async with stack.database.read_session() as session:
        adoption_count = int(
            await session.scalar(
                select(func.count()).select_from(IdeaAdoptionRecordRow)
            )
            or 0
        )
    assert adoption_count == 0
    assert (await stack.manager.verify(stack.database)).matches
