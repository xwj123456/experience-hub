from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_create_experience import (
    EXPERIENCE_IDS,
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    content,
    request,
)

from experience_hub import canonical_json_bytes
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.content import decode_payload
from experience_hub.experiences.contracts import (
    CreateExperienceVersion,
    ExperienceDraft,
)
from experience_hub.experiences.events import (
    ExperienceReactivatedV1,
    ExperienceStateSnapshotV1,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.repository import snapshot_from_state_row
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import (
    FOCUSED_COLD_EXPANSION_THRESHOLD,
    RetrievalService,
    retrieval_query_hash,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


@dataclass(slots=True)
class RetrievalStack:
    base: Stack
    service: RetrievalService


@dataclass(frozen=True, slots=True)
class AggregateSnapshot:
    state: ExperienceStateSnapshotV1
    event_count: int
    codec: PayloadCodec
    payload: bytes
    payload_hash: str


@pytest.fixture
async def retrieval_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[RetrievalStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "cold-reactivation.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    service = RetrievalService(
        clock=stack.clock,
        query=stack.query,
        mutation_writer=ExperienceMutationWriter(repository=stack.repository),
    )
    try:
        yield RetrievalStack(base=stack, service=service)
    finally:
        await stack.database.dispose()


async def create_cold(
    stack: RetrievalStack,
    *,
    key: str,
    label: str = "alpha",
    value: VersionContent | None = None,
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
                content=content(label) if value is None else value,
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

    result = await stack.base.executor.execute(
        request(key=key, operation="experience.test_create_cold"),
        handler,
    )
    assert result.status_code == 201
    assert created == [EXPERIENCE_IDS[0]]
    return created[0]


def threshold_content(
    *,
    body: str,
    mechanism: str = "zzz",
) -> VersionContent:
    return VersionContent(
        body=body,
        summary="zzz",
        mechanism=mechanism,
        tags=("zzz",),
        applicability=("zzz",),
        evidence=(),
        falsifiers=(),
    )


async def create_version(
    stack: RetrievalStack,
    *,
    experience_id: UUID,
    key: str,
    label: str,
) -> None:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.base.service.create_version(
            uow=uow,
            command=CreateExperienceVersion(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                content=content(label),
            ),
            command_context=command,
        )

    result = await stack.base.executor.execute(
        request(key=key, operation="experience.create_version"),
        handler,
    )
    assert result.status_code == 201


def search_request(
    query: SearchExperiences,
    *,
    key: str,
    caller_agent_id: UUID | None = None,
) -> CommandRequest:
    caller = query.owner_agent_id if caller_agent_id is None else caller_agent_id
    return CommandRequest(
        caller_scope=f"agent:{caller}",
        operation_scope="experience.search",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences:search",
        path_parameters={"agent_id": query.owner_agent_id},
        body={
            "query": query.query,
            "mode": query.mode,
            "tags": query.tags,
            "mechanism_cues": query.mechanism_cues,
            "limit": query.limit,
            "content_budget_bytes": query.content_budget_bytes,
            "expand_cold": query.expand_cold,
        },
    )


async def execute_search(
    stack: RetrievalStack,
    query: SearchExperiences,
    *,
    key: str,
    caller_agent_id: UUID | None = None,
) -> tuple[int, dict[str, Any], bool]:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        result = await stack.service.search(
            uow=uow,
            query=query,
            command=command,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": result}),
        )

    result = await stack.base.executor.execute(
        search_request(
            query,
            key=key,
            caller_agent_id=caller_agent_id,
        ),
        handler,
    )
    body = cast(dict[str, Any], json.loads(result.body))
    return result.status_code, body, result.replayed


async def execute_get(
    stack: RetrievalStack,
    *,
    experience_id: UUID,
    key: str,
) -> tuple[int, dict[str, Any]]:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        result = await stack.service.get(
            uow=uow,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            command=command,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": result}),
        )

    result = await stack.base.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_ID}",
            operation_scope="experience.get",
            idempotency_key=key,
            method="GET",
            route_template="/v1/agents/{agent_id}/experiences/{experience_id}",
            path_parameters={
                "agent_id": OWNER_ID,
                "experience_id": experience_id,
            },
        ),
        handler,
    )
    return result.status_code, cast(dict[str, Any], json.loads(result.body))


async def aggregate_snapshot(
    stack: RetrievalStack,
    experience_id: UUID,
) -> AggregateSnapshot:
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
    return AggregateSnapshot(
        state=snapshot_from_state_row(state),
        event_count=int(event_count or 0),
        codec=payload.codec,
        payload=payload.payload,
        payload_hash=payload.payload_hash,
    )


async def aggregate_events(
    stack: RetrievalStack,
    experience_id: UUID,
) -> tuple[DomainEventRow, ...]:
    async with stack.base.database.read_session() as session:
        return tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )


@pytest.mark.asyncio
async def test_qualifying_focused_cold_search_reactivates_atomically(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="qualifying-cold-create",
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    assert before.state.temperature is Temperature.COLD
    assert before.codec is PayloadCodec.ZLIB
    retrieval_stack.base.clock.advance(timedelta(hours=1))
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="alpha",
        mode=RetrievalMode.FOCUSED,
    )

    status, response, replayed = await execute_search(
        retrieval_stack,
        query,
        key="qualifying-cold-search",
    )

    assert status == 200
    assert replayed is False
    hits = response["data"]["hits"]
    assert len(hits) == 1
    hit = hits[0]
    assert hit["experience"]["experience_id"] == str(experience_id)
    assert hit["experience"]["body"] == content("alpha").body
    assert hit["experience"]["blurred"] is False
    assert hit["experience"]["temperature"] == "warm"
    assert hit["expanded"] is True
    assert hit["reactivated"] is True

    after = await aggregate_snapshot(retrieval_stack, experience_id)
    assert after.state.temperature is Temperature.WARM
    assert after.state.access_count == before.state.access_count + 1
    assert after.event_count == before.event_count + 3
    assert after.codec is PayloadCodec.PLAIN
    assert after.payload_hash == before.payload_hash
    assert after.state.current_content_hash == before.state.current_content_hash
    assert decode_payload(after.codec, after.payload) == decode_payload(
        before.codec,
        before.payload,
    )

    rows = await aggregate_events(retrieval_stack, experience_id)
    added = rows[before.event_count :]
    assert tuple(row.event_type for row in added) == (
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    )
    assert len({row.causation_id for row in added}) == 1
    reactivated = ExperienceReactivatedV1.model_validate_json(added[1].payload)
    assert reactivated.query_hash == retrieval_query_hash(query)
    assert reactivated.mode == "focused"
    assert reactivated.signal >= FOCUSED_COLD_EXPANSION_THRESHOLD
    assert reactivated.signal == pytest.approx(
        hit["lexical_or_trigram_relevance"],
        abs=1e-12,
    )
    assert (
        await retrieval_stack.base.manager.verify(retrieval_stack.base.database)
    ).matches


@pytest.mark.asyncio
async def test_focused_cold_threshold_equality_expands(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="focused-equality-create",
        value=threshold_content(
            body="abcdefghijklmnopqr secret",
        ),
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response, _ = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="abcdefghijklmnopqrstuvw",
            mode=RetrievalMode.FOCUSED,
        ),
        key="focused-equality-search",
    )

    assert status == 200
    hit = response["data"]["hits"][0]
    assert hit["lexical_or_trigram_relevance"] == pytest.approx(
        FOCUSED_COLD_EXPANSION_THRESHOLD,
        abs=1e-12,
    )
    assert hit["expanded"] is True
    assert hit["reactivated"] is True
    after = await aggregate_snapshot(retrieval_stack, experience_id)
    assert after.state.temperature is Temperature.WARM
    assert after.event_count == before.event_count + 3


@pytest.mark.asyncio
async def test_associative_cold_mechanism_threshold_equality_expands(
    retrieval_stack: RetrievalStack,
) -> None:
    mechanisms = tuple(f"m{index:02d}" for index in range(20))
    experience_id = await create_cold(
        retrieval_stack,
        key="associative-equality-create",
        value=threshold_content(
            body="hidden associative body",
            mechanism=" ".join(mechanisms[:13]),
        ),
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response, _ = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="!!!",
            mode=RetrievalMode.ASSOCIATIVE,
            mechanism_cues=mechanisms,
        ),
        key="associative-equality-search",
    )

    assert status == 200
    hit = response["data"]["hits"][0]
    assert hit["mechanism_relevance"] == pytest.approx(0.65, abs=1e-12)
    assert hit["expanded"] is True
    assert hit["reactivated"] is True
    after = await aggregate_snapshot(retrieval_stack, experience_id)
    assert after.state.temperature is Temperature.WARM
    assert after.event_count == before.event_count + 3


@pytest.mark.asyncio
async def test_qualifying_cold_body_one_byte_over_budget_stays_blurred(
    retrieval_stack: RetrievalStack,
) -> None:
    value = threshold_content(
        body="abcdefghijklmnopqr secret",
    )
    experience_id = await create_cold(
        retrieval_stack,
        key="budget-short-create",
        value=value,
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response, _ = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="abcdefghijklmnopqrstuvw",
            mode=RetrievalMode.FOCUSED,
            content_budget_bytes=len(value.body.encode("utf-8")) - 1,
        ),
        key="budget-short-search",
    )

    assert status == 200
    hit = response["data"]["hits"][0]
    assert hit["lexical_or_trigram_relevance"] == pytest.approx(
        FOCUSED_COLD_EXPANSION_THRESHOLD,
        abs=1e-12,
    )
    assert hit["experience"]["body"] is None
    assert hit["experience"]["blurred"] is True
    assert hit["expanded"] is False
    assert hit["reactivated"] is False
    assert await aggregate_snapshot(retrieval_stack, experience_id) == before


@pytest.mark.asyncio
async def test_expand_cold_false_stays_blurred_without_domain_mutation(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="no-expand-create",
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response, replayed = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
            expand_cold=False,
        ),
        key="no-expand-search",
    )

    assert status == 200
    assert replayed is False
    hits = response["data"]["hits"]
    assert len(hits) == 1
    assert hits[0]["experience"]["body"] is None
    assert hits[0]["experience"]["blurred"] is True
    assert hits[0]["experience"]["temperature"] == "cold"
    assert hits[0]["expanded"] is False
    assert hits[0]["reactivated"] is False
    after = await aggregate_snapshot(retrieval_stack, experience_id)
    assert after == before
    assert after.codec is PayloadCodec.ZLIB


@pytest.mark.asyncio
async def test_backward_service_clock_rolls_back_qualifying_cold_search(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="clock-create",
        label="initial",
    )
    retrieval_stack.base.clock.advance(timedelta(hours=1))
    await create_version(
        retrieval_stack,
        experience_id=experience_id,
        key="clock-version",
        label="alpha",
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    assert before.state.temperature is Temperature.COLD
    assert before.codec is PayloadCodec.ZLIB
    retrieval_stack.base.clock.advance(timedelta(minutes=-30))

    status, response, replayed = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
        ),
        key="clock-regression-search",
    )

    assert status == 409
    assert replayed is False
    assert response["error"]["code"] == "clock_regression"
    assert await aggregate_snapshot(retrieval_stack, experience_id) == before


@pytest.mark.asyncio
async def test_caller_owner_mismatch_has_no_domain_side_effects(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="owner-mismatch-create",
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response, replayed = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
        ),
        key="owner-mismatch-search",
        caller_agent_id=OTHER_OWNER_ID,
    )

    assert status == 404
    assert replayed is False
    assert response["error"]["code"] == "experience_not_found"
    assert await aggregate_snapshot(retrieval_stack, experience_id) == before


@pytest.mark.asyncio
async def test_direct_get_cold_is_blurred_and_read_only(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await create_cold(
        retrieval_stack,
        key="cold-get-create",
    )
    before = await aggregate_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))

    status, response = await execute_get(
        retrieval_stack,
        experience_id=experience_id,
        key="cold-get",
    )

    assert status == 200
    assert response["data"]["body"] is None
    assert response["data"]["blurred"] is True
    assert response["data"]["temperature"] == "cold"
    assert await aggregate_snapshot(retrieval_stack, experience_id) == before
