from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_create_experience import (
    OWNER_ID,
    Stack,
    build_stack,
    request,
)

from experience_hub.domain import CommandContext
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.repository import snapshot_from_state_row
from experience_hub.retrieval.contracts import PeekExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import ExperienceEvidenceReader
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


@dataclass(slots=True)
class PeekStack:
    base: Stack
    reader: ExperienceEvidenceReader


@dataclass(frozen=True, slots=True)
class DomainSnapshot:
    state: ExperienceStateSnapshotV1
    event_count: int
    codec: PayloadCodec
    payload: bytes
    payload_hash: str


@pytest.fixture
async def peek_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[PeekStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-peek.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    try:
        yield PeekStack(
            base=stack,
            reader=ExperienceEvidenceReader(
                clock=stack.clock,
                query=stack.query,
            ),
        )
    finally:
        await stack.database.dispose()


def _content(
    *,
    body: str,
    summary: str,
    mechanism: str = "unrelated mechanism",
) -> VersionContent:
    return VersionContent(
        body=body,
        summary=summary,
        mechanism=mechanism,
        tags=("private",),
        applicability=("local runtime",),
        evidence=(),
        falsifiers=(),
    )


async def _create_cold(
    stack: PeekStack,
    *,
    key: str,
    value: VersionContent,
) -> UUID:
    created: list[UUID] = []

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        record = await stack.base.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=OWNER_ID,
                actor_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=value,
                importance=0.35,
                confidence=0.5,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=stack.base.clock.now(),
            ),
            command=command,
        )
        created.append(record.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    response = await stack.base.executor.execute(
        request(key=key, operation="experience.test_create_cold"),
        handler,
    )
    assert response.status_code == 201
    assert len(created) == 1
    return created[0]


async def _snapshot(
    stack: PeekStack,
    experience_id: UUID,
) -> DomainSnapshot:
    async with stack.base.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
        assert state is not None
        payload = await session.get(
            ExperiencePayloadRow,
            state.current_version_id,
        )
        assert payload is not None
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_id == experience_id)
        )
    return DomainSnapshot(
        state=snapshot_from_state_row(state),
        event_count=int(event_count or 0),
        codec=payload.codec,
        payload=payload.payload,
        payload_hash=payload.payload_hash,
    )


@pytest.mark.asyncio
async def test_real_peek_returns_utf8_bounded_cold_excerpt_without_writes(
    peek_stack: PeekStack,
) -> None:
    experience_id = await _create_cold(
        peek_stack,
        key="peek-qualifying-create",
        value=_content(
            body="记忆" * 2_000,
            summary="alpha evidence",
        ),
    )
    before = await _snapshot(peek_stack, experience_id)

    async with peek_stack.base.database.read_session() as session:
        result = await peek_stack.reader.peek(
            session=session,
            query=PeekExperiences(
                owner_agent_id=OWNER_ID,
                query="alpha",
                mode=RetrievalMode.FOCUSED,
            ),
        )

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.experience.experience_id == experience_id
    assert hit.experience.body is not None
    assert 0 < len(hit.experience.body.encode("utf-8")) <= 2_048
    assert hit.experience.body_is_excerpt is True
    assert hit.expanded is True
    assert hit.reactivated is False
    assert await _snapshot(peek_stack, experience_id) == before


@pytest.mark.asyncio
async def test_real_peek_never_leaks_nonqualifying_cold_body(
    peek_stack: PeekStack,
) -> None:
    experience_id = await _create_cold(
        peek_stack,
        key="peek-nonqualifying-create",
        value=_content(
            body="abc secret cold body",
            summary="unrelated summary",
        ),
    )
    before = await _snapshot(peek_stack, experience_id)

    async with peek_stack.base.database.read_session() as session:
        result = await peek_stack.reader.peek(
            session=session,
            query=PeekExperiences(
                owner_agent_id=OWNER_ID,
                query="abcdefghijklmnopqrstuvw",
                mode=RetrievalMode.FOCUSED,
            ),
        )

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert 0.05 <= hit.lexical_or_trigram_relevance < 0.72
    assert hit.experience.blurred is True
    assert hit.experience.body is None
    assert hit.expanded is False
    assert hit.reactivated is False
    assert await _snapshot(peek_stack, experience_id) == before


@pytest.mark.asyncio
async def test_real_peek_per_hit_caps_preserve_later_cold_evidence(
    peek_stack: PeekStack,
) -> None:
    first_id = await _create_cold(
        peek_stack,
        key="peek-cap-first",
        value=_content(
            body="记忆" * 2_000,
            summary="alpha first",
        ),
    )
    second_id = await _create_cold(
        peek_stack,
        key="peek-cap-second",
        value=_content(
            body="经验" * 2_000,
            summary="alpha second",
        ),
    )
    before = {
        first_id: await _snapshot(peek_stack, first_id),
        second_id: await _snapshot(peek_stack, second_id),
    }

    async with peek_stack.base.database.read_session() as session:
        result = await peek_stack.reader.peek(
            session=session,
            query=PeekExperiences(
                owner_agent_id=OWNER_ID,
                query="alpha",
                mode=RetrievalMode.FOCUSED,
                content_budget_bytes=24,
                per_hit_excerpt_bytes=12,
            ),
        )

    assert len(result.hits) == 2
    assert all(hit.experience.body is not None for hit in result.hits)
    assert [
        len(hit.experience.body.encode("utf-8"))
        for hit in result.hits
        if hit.experience.body is not None
    ] == [12, 12]
    assert result.remaining_content_budget_bytes == 0
    assert {
        first_id: await _snapshot(peek_stack, first_id),
        second_id: await _snapshot(peek_stack, second_id),
    } == before
