from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import update
from tests.integration.test_create_experience import (
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    create,
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
from experience_hub.experiences.queries import (
    ExperienceNotFoundError,
    ExperienceQuery,
)
from experience_hub.experiences.repository import ExperienceWriter
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval.contracts import CandidateSelection
from experience_hub.retrieval.ranking import (
    CandidateMatch,
    RetrievalMode,
    raw_overlap,
    select_temperature_pools,
    temperature_pool_quota,
)
from experience_hub.retrieval.tokenizer import (
    TermCue,
    index_version_terms,
    query_cues,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import ExperienceStateRow
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


def searchable_content(label: str) -> VersionContent:
    return VersionContent(
        body=f"Preserve the complete {label} body.",
        summary=f"{label} summary",
        mechanism="single writer lease handoff",
        tags=("memory", label),
        applicability=("local runtime",),
        evidence=(),
        falsifiers=("overlapping writer",),
    )


@pytest.fixture
async def retrieval_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "retrieval-queries.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    try:
        yield stack
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_get_retrieval_record_is_owner_scoped_and_row_free(
    retrieval_stack: Stack,
) -> None:
    _, created = await create(
        retrieval_stack,
        key="owner-record",
        value=searchable_content("lease"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    version_id = UUID(created["data"]["version_id"])
    query = ExperienceQuery(event_registry=retrieval_stack.registry)

    async with retrieval_stack.database.read_session() as session:
        record = await query.get_owned_retrieval_record(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
        )
        foreign = await query.get_owned_retrieval_record(
            session=session,
            owner_agent_id=OTHER_OWNER_ID,
            experience_id=experience_id,
        )

    assert record is not None
    assert record.experience_id == experience_id
    assert record.current_version_id == version_id
    assert record.owner_agent_id == OWNER_ID
    assert record.state.temperature is Temperature.WARM
    assert record.summary == "lease summary"
    assert foreign is None
    assert not hasattr(record, "_sa_instance_state")


@pytest.mark.asyncio
async def test_candidate_selection_and_payload_batch_recheck_owner(
    retrieval_stack: Stack,
) -> None:
    _, owner_created = await create(
        retrieval_stack,
        key="owner-search",
        value=searchable_content("owner lease"),
    )
    _, foreign_created = await create(
        retrieval_stack,
        key="foreign-search",
        owner_agent_id=OTHER_OWNER_ID,
        value=searchable_content("foreign lease"),
    )
    owner_experience_id = UUID(owner_created["data"]["experience_id"])
    owner_version_id = UUID(owner_created["data"]["version_id"])
    foreign_version_id = UUID(foreign_created["data"]["version_id"])
    cues = query_cues("lease handoff")
    query = ExperienceQuery(event_registry=retrieval_stack.registry)

    async with retrieval_stack.database.read_session() as session:
        candidates = await query.select_retrieval_candidates(
            session=session,
            selection=CandidateSelection(
                owner_agent_id=OWNER_ID,
                query_cues=cues,
                mode=RetrievalMode.FOCUSED,
                requested_limit=10,
            ),
        )
        payloads = await query.load_decoded_payloads(
            session=session,
            owner_agent_id=OWNER_ID,
            version_ids=(owner_version_id,),
        )
        with pytest.raises(ExperienceNotFoundError):
            await query.load_decoded_payloads(
                session=session,
                owner_agent_id=OWNER_ID,
                version_ids=(owner_version_id, foreign_version_id),
            )

    assert [candidate.record.experience_id for candidate in candidates] == [
        owner_experience_id
    ]
    assert candidates[0].raw_overlap > 0.0
    assert payloads[owner_version_id].startswith(b'{"body":')


@pytest.mark.asyncio
async def test_payload_batch_empty_input_is_a_stable_noop(
    retrieval_stack: Stack,
) -> None:
    query = ExperienceQuery(event_registry=retrieval_stack.registry)

    async with retrieval_stack.database.read_session() as session:
        payloads = await query.load_decoded_payloads(
            session=session,
            owner_agent_id=OWNER_ID,
            version_ids=(),
        )

    assert payloads == {}


@pytest.mark.asyncio
async def test_candidate_selection_rejects_polluted_projection_owner(
    retrieval_stack: Stack,
) -> None:
    await create(
        retrieval_stack,
        key="polluted-owner",
        value=searchable_content("polluted lease"),
    )
    query = ExperienceQuery(event_registry=retrieval_stack.registry)
    async with retrieval_stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.owner_agent_id == OWNER_ID)
            .values(owner_agent_id=OTHER_OWNER_ID)
        )

    async with retrieval_stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="owner"):
            await query.select_retrieval_candidates(
                session=session,
                selection=CandidateSelection(
                    owner_agent_id=OWNER_ID,
                    query_cues=query_cues("polluted lease"),
                    mode=RetrievalMode.FOCUSED,
                    requested_limit=10,
                ),
            )


@pytest.mark.asyncio
async def test_valid_punctuation_only_experience_is_not_source_corruption(
    retrieval_stack: Stack,
) -> None:
    await create(
        retrieval_stack,
        key="punctuation-only",
        value=VersionContent(
            body="!!!",
            summary="？？？",
            mechanism="——",
            tags=("...",),
            applicability=("，。",),
            evidence=(),
            falsifiers=(),
        ),
    )
    query = ExperienceQuery(event_registry=retrieval_stack.registry)

    async with retrieval_stack.database.read_session() as session:
        candidates = await query.select_retrieval_candidates(
            session=session,
            selection=CandidateSelection(
                owner_agent_id=OWNER_ID,
                query_cues=query_cues("memory"),
                mode=RetrievalMode.FOCUSED,
                requested_limit=10,
            ),
        )

    assert candidates == ()


@pytest.mark.asyncio
async def test_candidate_query_exceeds_sqlite_variable_count_with_one_json_input(
    retrieval_stack: Stack,
) -> None:
    cues = tuple(
        TermCue(
            term=f"cue-{index:04d}",
            term_kind="word",
            weight=1.0,
        )
        for index in range(1_100)
    )
    query = ExperienceQuery(event_registry=retrieval_stack.registry)

    async with retrieval_stack.database.read_session() as session:
        candidates = await query.select_retrieval_candidates(
            session=session,
            selection=CandidateSelection(
                owner_agent_id=OWNER_ID,
                query_cues=cues,
                mode=RetrievalMode.FOCUSED,
                requested_limit=1,
            ),
        )

    assert candidates == ()


class _CandidateStageProbe:
    def __init__(self, result: Any, row_counts: list[int]) -> None:
        self._result = result
        self._row_counts = row_counts

    def all(self) -> Any:
        rows = self._result.all()
        self._row_counts.append(len(rows))
        return rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


@pytest.mark.asyncio
async def test_common_cue_candidate_stage_matches_oracle_and_is_pool_bounded(
    retrieval_stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_temperature = 36
    temperatures = (
        Temperature.HOT,
        Temperature.WARM,
        Temperature.COLD,
    )
    total = per_temperature * len(temperatures)
    contents = tuple(
        VersionContent(
            body=f"common shared body for candidate {index}",
            summary=f"common shared summary {index}",
            mechanism="common shared writer handoff",
            tags=(
                "common" if index % 3 == 0 else f"candidate-{index}",
                "memory",
            ),
            applicability=("common local runtime",),
            evidence=(),
            falsifiers=(f"candidate {index} mismatch",),
        )
        for index in range(total)
    )
    assigned_temperatures = tuple(
        temperatures[index % len(temperatures)] for index in range(total)
    )
    generated_ids = tuple(
        UUID(int=10_000 + index) for index in range(total * 2)
    )
    writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(generated_ids),
        repository=retrieval_stack.repository,
    )
    experience_ids: list[UUID] = []

    async def seed(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        for content, temperature in zip(
            contents,
            assigned_temperatures,
            strict=True,
        ):
            created = await writer.create_from_draft(
                uow=uow,
                draft=ExperienceDraft(
                    owner_agent_id=OWNER_ID,
                    actor_agent_id=OWNER_ID,
                    kind=ExperienceKind.PROCEDURAL,
                    origin=ExperienceOrigin.LOCAL,
                    content=content,
                    importance=0.35,
                    confidence=0.5,
                    source_trust=1.0,
                    initial_temperature=temperature,
                    links=(),
                    occurred_at=retrieval_stack.clock.now(),
                ),
                command=command,
            )
            experience_ids.append(created.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    seeded = await retrieval_stack.executor.execute(
        request(
            key="large-common-candidate-seed",
            operation="experience.test_candidate_seed",
        ),
        seed,
    )
    assert seeded.status_code == 201
    assert len(experience_ids) == total

    cues = query_cues(
        "common shared",
        tags=("common",),
        mechanisms=("writer",),
    )
    matches = tuple(
        CandidateMatch(
            experience_id=experience_id,
            temperature=temperature,
            raw_overlap=raw_overlap(cues, index_version_terms(content)),
        )
        for experience_id, temperature, content in zip(
            experience_ids,
            assigned_temperatures,
            contents,
            strict=True,
        )
    )
    expected = select_temperature_pools(
        matches,
        mode=RetrievalMode.FOCUSED,
        requested_limit=1,
    )
    query = ExperienceQuery(event_registry=retrieval_stack.registry)
    stage_row_counts: list[int] = []

    async with retrieval_stack.database.read_session() as session:
        original_execute = session.execute

        async def execute_with_probe(
            statement: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = await original_execute(statement, *args, **kwargs)
            if "bounded_candidate_stage" in str(statement):
                return _CandidateStageProbe(result, stage_row_counts)
            return result

        monkeypatch.setattr(session, "execute", execute_with_probe)
        candidates = await query.select_retrieval_candidates(
            session=session,
            selection=CandidateSelection(
                owner_agent_id=OWNER_ID,
                query_cues=cues,
                mode=RetrievalMode.FOCUSED,
                requested_limit=1,
            ),
        )

    assert [candidate.record.experience_id for candidate in candidates] == [
        match.experience_id for match in expected
    ]
    assert [candidate.raw_overlap for candidate in candidates] == pytest.approx(
        [match.raw_overlap for match in expected],
        abs=1e-12,
    )
    maximum_union = sum(
        temperature_pool_quota(
            RetrievalMode.FOCUSED,
            temperature,
            1,
        )
        for temperature in temperatures
    )
    assert len(expected) == maximum_union
    assert total > maximum_union
    assert stage_row_counts == [maximum_union]
