from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    PUBLISHER_ID,
    SOURCE_CONTENT,
    SOURCE_EXPERIENCE_ID,
    SOURCE_VERSION_ID,
    TOPIC_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
    request,
)

from experience_hub.domain import CommandContext
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.hashing import snapshot_canonical_bytes
from experience_hub.inspiration.models import (
    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
    MAX_SNAPSHOT_UTF8_BYTES,
    EvidenceSourceState,
    EvidenceSourceType,
)
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.retrieval.contracts import PeekExperiences, SearchResult
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import ExperienceEvidenceReader
from experience_hub.retrieval.tokenizer import TermCue
from experience_hub.sharing.models import (
    Capsule,
    PublishCapsule,
    RetractCapsule,
)
from experience_hub.sharing.queries import (
    InboxEvidenceReader,
    QuarantinedCapsuleEvidence,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceStateRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


@pytest.fixture
async def snapshot_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "snapshot-freeze.sqlite3",
    )
    manager = cast(
        ProjectionManager,
        stack.database._projection_applier,  # noqa: SLF001
    )
    manager.registry.register(ExperienceTermsProjector(stack.registry))
    try:
        yield stack
    finally:
        await stack.database.dispose()


async def _publish(
    stack: AdoptionStack,
    *,
    key: str,
    expires_in: timedelta,
) -> Capsule:
    command = PublishCapsule(
        owner_agent_id=PUBLISHER_ID,
        topic_id=TOPIC_ID,
        experience_id=SOURCE_EXPERIENCE_ID,
        version_id=SOURCE_VERSION_ID,
        expires_at=stack.clock.now() + expires_in,
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
            key=key,
            operation_scope="capsule.publish",
            route_template="/v1/agents/{agent_id}/capsules",
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


async def _retract(
    stack: AdoptionStack,
    *,
    key: str,
    capsule_id: UUID,
) -> None:
    command = RetractCapsule(
        publisher_agent_id=PUBLISHER_ID,
        capsule_id=capsule_id,
        reason="Superseded before snapshot generation.",
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

    result = await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.retract",
            route_template="/v1/agents/{agent_id}/capsules/{capsule_id}:retract",
            agent_id=PUBLISHER_ID,
            path_parameters={
                "agent_id": PUBLISHER_ID,
                "capsule_id": capsule_id,
            },
            body={"reason": command.reason},
        ),
        handler,
    )
    assert result.status_code == 200


async def _item_ids(stack: AdoptionStack) -> dict[UUID, UUID]:
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).where(
                        InboxItemRow.recipient_agent_id == ADOPTER_ID
                    )
                )
            ).all()
        )
    return {row.capsule_id: row.item_id for row in rows}


async def _create_cold(
    stack: AdoptionStack,
    *,
    key: str,
    content: VersionContent,
) -> UUID:
    created: list[UUID] = []

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        record = await stack.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=ADOPTER_ID,
                actor_agent_id=ADOPTER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=content,
                importance=0.5,
                confidence=0.7,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=stack.clock.now(),
            ),
            command=context,
        )
        created.append(record.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    result = await stack.executor.execute(
        request(
            key=key,
            operation_scope="experience.snapshot_seed",
            route_template="/v1/agents/{agent_id}/experiences",
            agent_id=ADOPTER_ID,
            body={"summary": content.summary},
        ),
        handler,
    )
    assert result.status_code == 201
    assert len(created) == 1
    return created[0]


@dataclass(slots=True)
class RecordingExperienceReader:
    delegate: ExperienceEvidenceReader
    sessions: list[AsyncSession] = field(default_factory=list)

    async def peek(
        self,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        self.sessions.append(session)
        return await self.delegate.peek(session=session, query=query)


@dataclass(slots=True)
class RecordingInboxReader:
    delegate: InboxEvidenceReader
    sessions: list[AsyncSession] = field(default_factory=list)

    async def list_available_pending(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        as_of: datetime,
        query_cues: Iterable[TermCue],
        mode: RetrievalMode,
        limit: int,
    ) -> tuple[QuarantinedCapsuleEvidence, ...]:
        self.sessions.append(session)
        return await self.delegate.list_available_pending(
            session=session,
            recipient_agent_id=recipient_agent_id,
            as_of=as_of,
            query_cues=query_cues,
            mode=mode,
            limit=limit,
        )


async def _state_rows(
    session: AsyncSession,
    experience_ids: tuple[UUID, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in (
            await session.execute(
                select(
                    ExperienceStateRow.experience_id,
                    ExperienceStateRow.temperature,
                    ExperienceStateRow.access_count,
                    ExperienceStateRow.access_strength,
                    ExperienceStateRow.last_accessed_at,
                    ExperienceStateRow.last_transition_at,
                    ExperienceStateRow.projection_event_id,
                )
                .where(ExperienceStateRow.experience_id.in_(experience_ids))
                .order_by(ExperienceStateRow.experience_id)
            )
        ).all()
    )


@pytest.mark.asyncio
async def test_freeze_uses_one_session_without_access_or_quarantine_leakage(
    snapshot_stack: AdoptionStack,
) -> None:
    active = await arrange_pending_capsule(snapshot_stack)
    expired = await _publish(
        snapshot_stack,
        key="snapshot-expired-publish",
        expires_in=timedelta(days=1),
    )
    retracted = await _publish(
        snapshot_stack,
        key="snapshot-retracted-publish",
        expires_in=timedelta(days=7),
    )
    adopted = await _publish(
        snapshot_stack,
        key="snapshot-adopted-publish",
        expires_in=timedelta(days=7),
    )
    item_ids = await _item_ids(snapshot_stack)
    await _retract(
        snapshot_stack,
        key="snapshot-retract",
        capsule_id=retracted.capsule_id,
    )
    adoption = await adopt(
        snapshot_stack,
        key="snapshot-adopt",
        item_id=item_ids[adopted.capsule_id],
    )
    assert adoption.status_code == 200
    snapshot_stack.clock.advance(timedelta(days=2))

    goal = "alpha"
    context = SOURCE_CONTENT.summary
    combined_query = f"{goal}\n{context}"
    first = await _create_cold(
        snapshot_stack,
        key="snapshot-cold-first",
        content=VersionContent(
            body="记忆" * 2_000,
            summary=f"{combined_query} first",
            mechanism="bounded alpha evidence",
            tags=("snapshot",),
            applicability=("inspiration",),
            evidence=(),
            falsifiers=(),
        ),
    )
    second = await _create_cold(
        snapshot_stack,
        key="snapshot-cold-second",
        content=VersionContent(
            body="经验" * 2_000,
            summary=f"{combined_query} second",
            mechanism="bounded alpha evidence",
            tags=("snapshot",),
            applicability=("inspiration",),
            evidence=(),
            falsifiers=(),
        ),
    )
    blurred = await _create_cold(
        snapshot_stack,
        key="snapshot-cold-blurred",
        content=VersionContent(
            body="private cold body that must remain hidden",
            summary="alp unrelated",
            mechanism="unrelated mechanism",
            tags=("distractor",),
            applicability=("other context",),
            evidence=(),
            falsifiers=(),
        ),
    )
    cold_ids = (first, second, blurred)

    experience_reader = RecordingExperienceReader(
        ExperienceEvidenceReader(
            clock=snapshot_stack.clock,
            query=ExperienceQuery(event_registry=snapshot_stack.registry),
        )
    )
    inbox_reader = RecordingInboxReader(
        InboxEvidenceReader(
            repository=SharingRepository(event_registry=snapshot_stack.registry)
        )
    )
    builder = SnapshotBuilder(
        experience_reader=experience_reader,
        inbox_reader=inbox_reader,
        id_generator=SequenceIdGenerator(
            tuple(_uuid(90_000 + index) for index in range(30))
        ),
    )

    async with snapshot_stack.database.transaction(immediate=True) as uow:
        session = uow.session
        before_event_count = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow))
            or 0
        )
        before_states = await _state_rows(session, cold_ids)
        default_snapshot = await builder.freeze(
            uow=uow,
            request=StartInspirationRun(
                owner_agent_id=ADOPTER_ID,
                goal=goal,
                context=context,
                mode=RetrievalMode.FOCUSED,
            ),
            run_id=_uuid(80_001),
            at=snapshot_stack.clock.now(),
        )
        opted_in_snapshot = await builder.freeze(
            uow=uow,
            request=StartInspirationRun(
                owner_agent_id=ADOPTER_ID,
                goal=goal,
                context=context,
                mode=RetrievalMode.FOCUSED,
                include_inbox=True,
            ),
            run_id=_uuid(80_002),
            at=snapshot_stack.clock.now(),
        )
        after_states = await _state_rows(session, cold_ids)
        after_event_count = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow))
            or 0
        )
        assert all(value is session for value in experience_reader.sessions)
        assert inbox_reader.sessions == [session]

    assert all(
        item.source_type is EvidenceSourceType.EXPERIENCE
        for item in default_snapshot.items
    )
    capsule_items = tuple(
        item
        for item in opted_in_snapshot.items
        if item.source_type is EvidenceSourceType.CAPSULE
    )
    assert tuple(item.source_id for item in capsule_items) == (
        active.capsule_id,
    )
    assert expired.capsule_id not in {
        item.source_id for item in opted_in_snapshot.items
    }
    assert retracted.capsule_id not in {
        item.source_id for item in opted_in_snapshot.items
    }
    assert adopted.capsule_id not in {
        item.source_id for item in opted_in_snapshot.items
    }
    assert capsule_items[0].source_state is EvidenceSourceState.QUARANTINED
    assert capsule_items[0].source_trust == pytest.approx(0.25)
    assert capsule_items[0].falsifiers == SOURCE_CONTENT.falsifiers
    assert 0 < len(capsule_items[0].excerpt.encode("utf-8")) <= (
        MAX_SNAPSHOT_EXCERPT_UTF8_BYTES
    )

    by_source = {item.source_id: item for item in opted_in_snapshot.items}
    assert first in by_source and second in by_source and blurred in by_source
    assert 0 < len(by_source[first].excerpt.encode("utf-8")) <= (
        MAX_SNAPSHOT_EXCERPT_UTF8_BYTES
    )
    assert 0 < len(by_source[second].excerpt.encode("utf-8")) <= (
        MAX_SNAPSHOT_EXCERPT_UTF8_BYTES
    )
    assert by_source[blurred].source_state is EvidenceSourceState.COLD
    assert by_source[blurred].excerpt == ""
    assert len(snapshot_canonical_bytes(opted_in_snapshot.items)) <= (
        MAX_SNAPSHOT_UTF8_BYTES
    )
    assert before_states == after_states
    assert before_event_count == after_event_count
