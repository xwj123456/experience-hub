from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import CommandContext, PendingEvent
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.events import (
    ExperienceReactivatedV1,
    ExperienceStateSnapshotV1,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    PeekExperiences,
    RetrievalCandidate,
    RetrievalRecord,
    SearchExperiences,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import (
    ExperienceEvidenceReader,
    RetrievalService,
    retrieval_query_hash,
)
from experience_hub.retrieval.tokenizer import query_cues
from experience_hub.storage.unit_of_work import UnitOfWork

NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000401")


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def record(
    value: int,
    *,
    temperature: Temperature = Temperature.WARM,
    importance: float = 0.5,
    latest_causal_at: datetime = NOW,
) -> RetrievalRecord:
    experience_id = _id(200 + value)
    version_id = _id(300 + value)
    state = ExperienceStateSnapshotV1(
        experience_id=experience_id,
        owner_agent_id=OWNER_ID,
        current_version_id=version_id,
        current_content_hash=f"{value:x}".rjust(64, "0"),
        temperature=temperature,
        importance=importance,
        confidence=0.6,
        activation_score=0.5,
        source_trust=1.0,
        access_count=0,
        access_strength=0.0,
        strength_updated_at=NOW,
        last_accessed_at=None,
        last_transition_at=NOW,
        last_lifecycle_evaluated_at=None,
        consecutive_below_threshold=0,
        pinned=False,
    )
    return RetrievalRecord(
        experience_id=experience_id,
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.PROCEDURAL,
        origin=ExperienceOrigin.LOCAL,
        created_at=NOW,
        current_version_id=version_id,
        current_version_number=1,
        current_version_created_at=NOW,
        current_content_hash=state.current_content_hash,
        summary=f"summary {value}",
        mechanism="lease handoff",
        tags=("memory",),
        applicability=("runtime",),
        evidence=(),
        falsifiers=(),
        state=state,
        projection_event_id=value,
        latest_causal_at=latest_causal_at,
    )


def candidate(value: RetrievalRecord, query: str = "alpha") -> RetrievalCandidate:
    terms = query_cues(query)
    return RetrievalCandidate(
        record=value,
        terms=terms,
        raw_overlap=1.0,
    )


@dataclass
class FakeQuery:
    candidates: tuple[RetrievalCandidate, ...]
    payloads: dict[UUID, bytes]
    records: dict[UUID, RetrievalRecord]
    selections: list[CandidateSelection]

    async def select_retrieval_candidates(
        self,
        *,
        session: AsyncSession,
        selection: CandidateSelection,
    ) -> tuple[RetrievalCandidate, ...]:
        _ = session
        self.selections.append(selection)
        return self.candidates

    async def load_decoded_payloads(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        version_ids: Sequence[UUID],
    ) -> dict[UUID, bytes]:
        _ = session
        assert owner_agent_id == OWNER_ID
        return {version_id: self.payloads[version_id] for version_id in version_ids}

    async def get_owned_retrieval_record(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> RetrievalRecord | None:
        _ = session
        if owner_agent_id != OWNER_ID:
            return None
        return self.records.get(experience_id)


@dataclass
class WriterCall:
    experience_id: UUID
    resulting_state: ExperienceStateSnapshotV1
    events: tuple[PendingEvent, ...]


class FakeMutationWriter:
    def __init__(self) -> None:
        self.calls: list[WriterCall] = []

    async def apply_ordered_events(
        self,
        *,
        uow: UnitOfWork,
        experience_id: UUID,
        resulting_state: ExperienceStateSnapshotV1,
        events: Sequence[PendingEvent],
        command: CommandContext,
    ) -> ExperienceRecord:
        _ = (uow, command)
        self.calls.append(
            WriterCall(
                experience_id=experience_id,
                resulting_state=resulting_state,
                events=tuple(events),
            )
        )
        return ExperienceRecord(
            experience_id=experience_id,
            owner_agent_id=resulting_state.owner_agent_id,
            current_version_id=resulting_state.current_version_id,
            current_content_hash=resulting_state.current_content_hash,
            temperature=resulting_state.temperature,
        )


def context() -> CommandContext:
    return CommandContext(
        receipt_id=RECEIPT_ID,
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope="experience.search",
        idempotency_key="search-1",
        request_hash="f" * 64,
    )


def fake_uow() -> UnitOfWork:
    return cast(
        UnitOfWork,
        SimpleNamespace(session=object(), immediate=True),
    )


def service(
    query: FakeQuery,
) -> tuple[RetrievalService, FakeMutationWriter]:
    writer = FakeMutationWriter()
    return (
        RetrievalService(
            clock=FrozenClock(NOW),
            query=query,
            mutation_writer=writer,
        ),
        writer,
    )


def test_retrieval_query_hash_has_locked_cue_only_formula() -> None:
    value = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query=" lease handoff ",
        mode=RetrievalMode.FOCUSED,
        tags=("memory", "memory"),
        mechanism_cues=("single writer",),
        limit=3,
        content_budget_bytes=10,
        expand_cold=False,
    )

    assert retrieval_query_hash(value) == (
        "631b31fc33f60f4dd82cf072920340d4"
        "8c2ab04171144e84d8f6713bc48e8a73"
    )
    changed_transport = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="lease handoff",
        mode=RetrievalMode.ASSOCIATIVE,
        tags=("memory",),
        mechanism_cues=("single writer",),
        limit=50,
        content_budget_bytes=65_536,
        expand_cold=True,
    )
    assert retrieval_query_hash(changed_transport) == retrieval_query_hash(value)


@pytest.mark.asyncio
async def test_search_skips_oversized_early_body_and_accesses_later_body() -> None:
    first = record(1, importance=0.9)
    second = record(2, importance=0.1)
    query = FakeQuery(
        candidates=(candidate(first), candidate(second)),
        payloads={
            first.current_version_id: canonical_json_bytes({"body": "too large"}),
            second.current_version_id: canonical_json_bytes({"body": "ok"}),
        },
        records={
            first.experience_id: first,
            second.experience_id: second,
        },
        selections=[],
    )
    retrieval, writer = service(query)

    result = await retrieval.search(
        uow=fake_uow(),
        query=SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
            content_budget_bytes=2,
        ),
        command=context(),
    )

    assert [hit.experience.experience_id for hit in result.hits] == [
        first.experience_id,
        second.experience_id,
    ]
    assert result.hits[0].experience.blurred is True
    assert result.hits[1].experience.body == "ok"
    assert result.remaining_content_budget_bytes == 0
    assert [call.experience_id for call in writer.calls] == [
        second.experience_id
    ]
    assert [event.event_type for event in writer.calls[0].events] == [
        "experience.accessed"
    ]


@pytest.mark.asyncio
async def test_peek_caps_each_utf8_excerpt_and_never_calls_writer() -> None:
    first = record(1)
    second = record(2)
    query = FakeQuery(
        candidates=(candidate(first), candidate(second)),
        payloads={
            first.current_version_id: canonical_json_bytes(
                {"body": "记忆" * 20}
            ),
            second.current_version_id: canonical_json_bytes(
                {"body": "经验" * 20}
            ),
        },
        records={},
        selections=[],
    )
    writer = FakeMutationWriter()
    reader = ExperienceEvidenceReader(
        clock=FrozenClock(NOW),
        query=query,
    )

    result = await reader.peek(
        session=object(),  # type: ignore[arg-type]
        query=PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
            content_budget_bytes=24,
            per_hit_excerpt_bytes=12,
        ),
    )

    assert len(result.hits) == 2
    assert all(
        len(hit.experience.body.encode("utf-8")) == 12
        for hit in result.hits
        if hit.experience.body is not None
    )
    assert all(hit.experience.body_is_excerpt for hit in result.hits)
    assert result.remaining_content_budget_bytes == 0
    assert writer.calls == []


@pytest.mark.asyncio
async def test_get_cold_is_blurred_but_warm_records_one_access() -> None:
    cold = record(1, temperature=Temperature.COLD)
    warm = record(2, temperature=Temperature.WARM)
    query = FakeQuery(
        candidates=(),
        payloads={
            warm.current_version_id: canonical_json_bytes({"body": "visible"}),
        },
        records={
            cold.experience_id: cold,
            warm.experience_id: warm,
        },
        selections=[],
    )
    retrieval, writer = service(query)

    cold_view = await retrieval.get(
        uow=fake_uow(),
        owner_agent_id=OWNER_ID,
        experience_id=cold.experience_id,
        command=context(),
    )
    warm_view = await retrieval.get(
        uow=fake_uow(),
        owner_agent_id=OWNER_ID,
        experience_id=warm.experience_id,
        command=context(),
    )

    assert cold_view.blurred is True
    assert cold_view.body is None
    assert warm_view.body == "visible"
    assert [call.experience_id for call in writer.calls] == [
        warm.experience_id
    ]


@pytest.mark.asyncio
async def test_focused_cold_expansion_freezes_exact_three_event_intent() -> None:
    cold = record(1, temperature=Temperature.COLD)
    query = FakeQuery(
        candidates=(candidate(cold),),
        payloads={
            cold.current_version_id: canonical_json_bytes({"body": "visible"}),
        },
        records={cold.experience_id: cold},
        selections=[],
    )
    retrieval, writer = service(query)
    request = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="alpha",
        mode=RetrievalMode.FOCUSED,
    )

    result = await retrieval.search(
        uow=fake_uow(),
        query=request,
        command=context(),
    )

    assert result.hits[0].experience.body == "visible"
    assert result.hits[0].experience.temperature is Temperature.WARM
    assert result.hits[0].reactivated is True
    assert len(writer.calls) == 1
    assert [event.event_type for event in writer.calls[0].events] == [
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    ]
    reactivated = writer.calls[0].events[1].payload
    assert isinstance(reactivated, ExperienceReactivatedV1)
    assert reactivated.schema_version == 1
    assert reactivated.experience_id == cold.experience_id
    assert reactivated.query_hash == retrieval_query_hash(request)
    assert reactivated.mode == "focused"
    assert reactivated.signal == 1.0
    assert reactivated.before == reactivated.after
    assert "alpha" not in reactivated.model_dump_json()


@pytest.mark.asyncio
async def test_associative_cold_requires_mechanism_not_lexical_signal() -> None:
    cold = record(1, temperature=Temperature.COLD)
    query = FakeQuery(
        candidates=(candidate(cold),),
        payloads={},
        records={},
        selections=[],
    )
    retrieval, writer = service(query)

    result = await retrieval.search(
        uow=fake_uow(),
        query=SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.ASSOCIATIVE,
        ),
        command=context(),
    )

    assert result.hits[0].lexical_or_trigram_relevance == 1.0
    assert result.hits[0].mechanism_relevance == 0.0
    assert result.hits[0].experience.blurred is True
    assert result.hits[0].reactivated is False
    assert writer.calls == []


@pytest.mark.asyncio
async def test_late_clock_regression_prevents_all_planned_accesses() -> None:
    normal = record(1, importance=0.9)
    future = record(
        2,
        importance=0.1,
        latest_causal_at=NOW + timedelta(microseconds=1),
    )
    query = FakeQuery(
        candidates=(candidate(normal), candidate(future)),
        payloads={
            normal.current_version_id: canonical_json_bytes({"body": "first"}),
            future.current_version_id: canonical_json_bytes({"body": "second"}),
        },
        records={},
        selections=[],
    )
    retrieval, writer = service(query)

    with pytest.raises(ReplayableCommandError) as captured:
        await retrieval.search(
            uow=fake_uow(),
            query=SearchExperiences(
                owner_agent_id=OWNER_ID,
                query="alpha",
                mode=RetrievalMode.FOCUSED,
            ),
            command=context(),
        )

    assert captured.value.code == "clock_regression"
    assert writer.calls == []


@pytest.mark.asyncio
async def test_punctuation_only_query_returns_replayable_empty_query() -> None:
    query = FakeQuery(candidates=(), payloads={}, records={}, selections=[])
    retrieval, writer = service(query)

    with pytest.raises(ReplayableCommandError) as captured:
        await retrieval.search(
            uow=fake_uow(),
            query=SearchExperiences(
                owner_agent_id=OWNER_ID,
                query="!!!",
                mode=RetrievalMode.FOCUSED,
            ),
            command=context(),
        )

    assert captured.value.code == "empty_query"
    assert query.selections == []
    assert writer.calls == []


@pytest.mark.asyncio
async def test_service_rejects_caller_owner_mismatch_before_querying() -> None:
    query = FakeQuery(candidates=(), payloads={}, records={}, selections=[])
    retrieval, writer = service(query)
    foreign_command = CommandContext(
        receipt_id=RECEIPT_ID,
        caller_scope=f"agent:{_id(999)}",
        operation_scope="experience.search",
        idempotency_key="foreign",
        request_hash="e" * 64,
    )

    with pytest.raises(ReplayableCommandError) as captured:
        await retrieval.search(
            uow=fake_uow(),
            query=SearchExperiences(
                owner_agent_id=OWNER_ID,
                query="alpha",
                mode=RetrievalMode.FOCUSED,
            ),
            command=foreign_command,
        )

    assert captured.value.code == "experience_not_found"
    assert query.selections == []
    assert writer.calls == []
