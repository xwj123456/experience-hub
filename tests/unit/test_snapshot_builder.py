from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.hashing import (
    hash_snapshot,
    snapshot_canonical_bytes,
)
from experience_hub.inspiration.models import (
    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
    MAX_SNAPSHOT_ITEMS,
    MAX_SNAPSHOT_UTF8_BYTES,
    EvidenceSourceState,
    EvidenceSourceType,
)
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.retrieval.contracts import (
    ExperienceView,
    PeekExperiences,
    SearchHit,
    SearchResult,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue, query_cues
from experience_hub.sharing.queries import QuarantinedCapsuleEvidence
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
RUN_ID = UUID("00000000-0000-0000-0000-000000000201")


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def _hash(value: int) -> str:
    return f"{value:064x}"


def _owned_hit(
    value: int,
    *,
    relevance: float,
    excerpt: str | None = "owned evidence",
    temperature: Temperature = Temperature.WARM,
    summary: str | None = None,
    mechanism: str = "commit before cache invalidation",
    applicability: tuple[str, ...] = ("transactional systems",),
    tags: tuple[str, ...] = ("owned",),
    falsifiers: tuple[str, ...] = ("No stale read is observable.",),
) -> SearchHit:
    blurred = excerpt is None
    experience = ExperienceView(
        experience_id=_uuid(1_000 + value),
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.PROCEDURAL,
        origin=ExperienceOrigin.LOCAL,
        created_at=NOW,
        version_id=_uuid(2_000 + value),
        version_number=1,
        version_created_at=NOW,
        content_hash=_hash(value),
        temperature=temperature,
        importance=0.7,
        confidence=0.8,
        activation_score=0.6,
        source_trust=1.0,
        access_count=0,
        access_strength=0.0,
        strength_updated_at=NOW,
        last_accessed_at=None,
        last_transition_at=NOW,
        last_lifecycle_evaluated_at=None,
        consecutive_below_threshold=0,
        pinned=False,
        summary=summary or f"owned summary {value}",
        mechanism=mechanism,
        tags=tags,
        applicability=applicability,
        evidence=(),
        falsifiers=falsifiers,
        blurred=blurred,
        body=excerpt,
        body_is_excerpt=not blurred,
    )
    return SearchHit(
        experience=experience,
        score=relevance,
        ranking_relevance=relevance,
        lexical_or_trigram_relevance=relevance,
        mechanism_relevance=relevance,
        activation=0.6,
        expanded=not blurred,
        reactivated=False,
    )


def _capsule(
    value: int,
    *,
    relevance: float,
    excerpt: str = "quarantined evidence",
) -> QuarantinedCapsuleEvidence:
    capsule_id = _uuid(3_000 + value)
    return QuarantinedCapsuleEvidence(
        item_id=_uuid(4_000 + value),
        capsule_id=capsule_id,
        publisher_agent_id=_uuid(5_000 + value),
        source_type="capsule",
        source_id=capsule_id,
        source_version_id=_uuid(6_000 + value),
        source_state="quarantined",
        content_hash=_hash(10_000 + value),
        summary=f"capsule summary {value}",
        mechanism="quarantine before explicit adoption",
        applicability=("explicit opt-in",),
        tags=("shared",),
        falsifiers=("Adoption occurs without an explicit decision.",),
        excerpt=excerpt,
        ranking_relevance=relevance,
        source_trust=0.25,
    )


@dataclass(slots=True)
class FakeExperienceReader:
    result: SearchResult
    calls: list[tuple[AsyncSession, PeekExperiences]] = field(
        default_factory=list
    )

    async def peek(
        self,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        self.calls.append((session, query))
        return self.result


@dataclass(slots=True)
class InboxCall:
    session: AsyncSession
    recipient_agent_id: UUID
    as_of: datetime
    query_cues: tuple[TermCue, ...]
    mode: RetrievalMode
    limit: int


@dataclass(slots=True)
class FakeInboxReader:
    result: tuple[QuarantinedCapsuleEvidence, ...] = ()
    calls: list[InboxCall] = field(default_factory=list)

    async def list_available_pending(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        as_of: datetime,
        query_cues: tuple[TermCue, ...],
        mode: RetrievalMode,
        limit: int,
    ) -> tuple[QuarantinedCapsuleEvidence, ...]:
        self.calls.append(
            InboxCall(
                session=session,
                recipient_agent_id=recipient_agent_id,
                as_of=as_of,
                query_cues=query_cues,
                mode=mode,
                limit=limit,
            )
        )
        return self.result


def _uow(session: AsyncSession) -> UnitOfWork:
    return cast(UnitOfWork, SimpleNamespace(session=session))


def _builder(
    owned: tuple[SearchHit, ...],
    *,
    capsules: tuple[QuarantinedCapsuleEvidence, ...] = (),
    ids: tuple[UUID, ...] | None = None,
) -> tuple[SnapshotBuilder, FakeExperienceReader, FakeInboxReader]:
    experience_reader = FakeExperienceReader(
        SearchResult(
            hits=owned,
            remaining_content_budget_bytes=0,
        )
    )
    inbox_reader = FakeInboxReader(capsules)
    builder = SnapshotBuilder(
        experience_reader=experience_reader,
        inbox_reader=inbox_reader,
        id_generator=SequenceIdGenerator(
            ids
            or tuple(_uuid(7_000 + index) for index in range(MAX_SNAPSHOT_ITEMS))
        ),
    )
    return builder, experience_reader, inbox_reader


@pytest.mark.asyncio
async def test_freeze_uses_same_session_and_locked_internal_peek_query() -> None:
    builder, experience_reader, inbox_reader = _builder(
        (_owned_hit(1, relevance=0.9),)
    )
    session = cast(AsyncSession, object())
    request = StartInspirationRun(
        owner_agent_id=OWNER_ID,
        goal="cache failure",
        context="commit boundary",
        mode=RetrievalMode.ASSOCIATIVE,
    )

    snapshot = await builder.freeze(
        uow=_uow(session),
        request=request,
        run_id=RUN_ID,
        at=NOW,
    )

    assert len(experience_reader.calls) == 1
    observed_session, query = experience_reader.calls[0]
    assert observed_session is session
    assert (
        query.owner_agent_id,
        query.query,
        query.mode,
        query.limit,
        query.content_budget_bytes,
        query.per_hit_excerpt_bytes,
        query.expand_cold,
    ) == (
        OWNER_ID,
        "cache failure\ncommit boundary",
        RetrievalMode.ASSOCIATIVE,
        12,
        24_576,
        2_048,
        True,
    )
    assert inbox_reader.calls == []
    assert len(snapshot.items) == 1
    assert snapshot.items[0].source_type is EvidenceSourceType.EXPERIENCE
    assert snapshot.items[0].source_state is EvidenceSourceState.WARM
    assert snapshot.items[0].falsifiers == ("No stale read is observable.",)
    assert snapshot.items[0].rank == 1
    assert snapshot.snapshot_hash == hash_snapshot(snapshot.items)
    assert snapshot.frozen_at == NOW
    assert len(snapshot_canonical_bytes(snapshot.items)) <= 24_576


@pytest.mark.asyncio
async def test_freeze_opt_in_merges_by_relevance_owned_first_and_source_ids() -> None:
    owned = (
        _owned_hit(3, relevance=0.8),
        _owned_hit(1, relevance=0.8),
    )
    capsules = (
        _capsule(2, relevance=0.9),
        _capsule(1, relevance=0.8),
    )
    builder, _, inbox_reader = _builder(owned, capsules=capsules)
    session = cast(AsyncSession, object())
    request = StartInspirationRun(
        owner_agent_id=OWNER_ID,
        goal="cache failure",
        context="commit boundary",
        mode=RetrievalMode.FOCUSED,
        include_inbox=True,
    )

    snapshot = await builder.freeze(
        uow=_uow(session),
        request=request,
        run_id=RUN_ID,
        at=NOW,
    )

    assert len(inbox_reader.calls) == 1
    call = inbox_reader.calls[0]
    assert call.session is session
    assert (
        call.recipient_agent_id,
        call.as_of,
        call.query_cues,
        call.mode,
        call.limit,
    ) == (
        OWNER_ID,
        NOW,
        query_cues("cache failure\ncommit boundary"),
        RetrievalMode.FOCUSED,
        12,
    )
    assert tuple(
        (item.source_type, item.source_id, item.rank, item.source_trust)
        for item in snapshot.items
    ) == (
        (
            EvidenceSourceType.CAPSULE,
            capsules[0].source_id,
            1,
            0.25,
        ),
        (
            EvidenceSourceType.EXPERIENCE,
            owned[1].experience.experience_id,
            2,
            1.0,
        ),
        (
            EvidenceSourceType.EXPERIENCE,
            owned[0].experience.experience_id,
            3,
            1.0,
        ),
        (
            EvidenceSourceType.CAPSULE,
            capsules[1].source_id,
            4,
            0.25,
        ),
    )
    assert snapshot.items[-1].source_state is EvidenceSourceState.QUARANTINED
    assert snapshot.items[-1].falsifiers == (
        "Adoption occurs without an explicit decision.",
    )


@pytest.mark.asyncio
async def test_freeze_limits_items_and_caps_each_excerpt_before_aggregate() -> None:
    large = "记忆" * 2_000
    owned = tuple(
        _owned_hit(index, relevance=1.0 - index / 100, excerpt=None)
        for index in range(13)
    )
    builder, _, _ = _builder(owned)
    snapshot = await builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=StartInspirationRun(owner_agent_id=OWNER_ID, goal="memory"),
        run_id=RUN_ID,
        at=NOW,
    )
    assert len(snapshot.items) == MAX_SNAPSHOT_ITEMS
    assert tuple(item.rank for item in snapshot.items) == tuple(range(1, 13))

    large_owned = tuple(
        _owned_hit(index, relevance=1.0 - index / 100, excerpt=large)
        for index in range(3)
    )
    builder, _, _ = _builder(large_owned)
    bounded = await builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=StartInspirationRun(owner_agent_id=OWNER_ID, goal="memory"),
        run_id=RUN_ID,
        at=NOW,
    )
    byte_lengths = [
        len(item.excerpt.encode("utf-8")) for item in bounded.items
    ]
    assert len(bounded.items) == 3
    assert byte_lengths == [2_046, 2_046, 2_046]
    assert all(
        length <= MAX_SNAPSHOT_EXCERPT_UTF8_BYTES for length in byte_lengths
    )
    assert len(snapshot_canonical_bytes(bounded.items)) <= (
        MAX_SNAPSHOT_UTF8_BYTES
    )


@pytest.mark.asyncio
async def test_freeze_shrinks_final_excerpt_and_stops_at_oversized_metadata() -> None:
    nearly_full = _owned_hit(
        1,
        relevance=1.0,
        excerpt="x" * 2_048,
        summary="s" * 1_000,
        mechanism="m" * 2_000,
        applicability=tuple(f"{index}:" + "a" * 2_900 for index in range(7)),
        tags=tuple("t" * 350 for _ in range(2)),
    )
    builder, _, _ = _builder((nearly_full,))
    snapshot = await builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=StartInspirationRun(owner_agent_id=OWNER_ID, goal="bounded"),
        run_id=RUN_ID,
        at=NOW,
    )

    assert len(snapshot.items) == 1
    assert 0 <= len(snapshot.items[0].excerpt) < 2_048
    assert len(snapshot_canonical_bytes(snapshot.items)) <= (
        MAX_SNAPSHOT_UTF8_BYTES
    )
    one_more = snapshot.items[0].model_copy(
        update={"excerpt": snapshot.items[0].excerpt + "x"}
    )
    assert len(snapshot_canonical_bytes((one_more,))) > (
        MAX_SNAPSHOT_UTF8_BYTES
    )

    first = _owned_hit(2, relevance=1.0, excerpt=None)
    oversized = _owned_hit(
        3,
        relevance=0.9,
        excerpt=None,
        applicability=tuple("a" * 4_000 for _ in range(7)),
    )
    later = _owned_hit(4, relevance=0.8, excerpt=None)
    builder, _, _ = _builder((first, oversized, later))
    stopped = await builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=StartInspirationRun(owner_agent_id=OWNER_ID, goal="bounded"),
        run_id=RUN_ID,
        at=NOW,
    )
    assert tuple(item.source_id for item in stopped.items) == (
        first.experience.experience_id,
    )


@pytest.mark.asyncio
async def test_equivalent_evidence_has_same_hash_across_run_ids_and_times() -> None:
    hit = _owned_hit(1, relevance=0.9, excerpt="stable evidence")
    first_builder, _, _ = _builder(
        (hit,),
        ids=(_uuid(8_001),),
    )
    second_builder, _, _ = _builder(
        (hit,),
        ids=(_uuid(8_002),),
    )
    request = StartInspirationRun(owner_agent_id=OWNER_ID, goal="stable")

    first = await first_builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=request,
        run_id=_uuid(9_001),
        at=NOW,
    )
    second = await second_builder.freeze(
        uow=_uow(cast(AsyncSession, object())),
        request=request,
        run_id=_uuid(9_002),
        at=NOW + timedelta(days=1),
    )

    assert first.items[0].snapshot_item_id != second.items[0].snapshot_item_id
    assert first.items[0].run_id != second.items[0].run_id
    assert first.items[0].captured_at != second.items[0].captured_at
    assert first.snapshot_hash == second.snapshot_hash
