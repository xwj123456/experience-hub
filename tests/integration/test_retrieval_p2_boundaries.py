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
    OWNER_ID,
    Stack,
    build_stack,
    content,
    create,
    request,
)

from experience_hub import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    StructuredReason,
)
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.events import (
    ExperienceArchivedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
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
from experience_hub.lifecycle import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

ARCHIVE_CYCLE_ID = UUID("20000000-0000-0000-0000-000000000001")


@dataclass(slots=True)
class RetrievalStack:
    base: Stack
    service: RetrievalService
    mutation_writer: ExperienceMutationWriter


@dataclass(frozen=True, slots=True)
class DomainSnapshot:
    state: ExperienceStateSnapshotV1
    events: tuple[tuple[int, str, UUID, int], ...]
    payload_codec: PayloadCodec
    payload: bytes
    payload_hash: str


@pytest.fixture
async def retrieval_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[RetrievalStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "retrieval-p2-boundaries.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    mutation_writer = ExperienceMutationWriter(repository=stack.repository)
    service = RetrievalService(
        clock=stack.clock,
        query=stack.query,
        mutation_writer=mutation_writer,
    )
    try:
        yield RetrievalStack(
            base=stack,
            service=service,
            mutation_writer=mutation_writer,
        )
    finally:
        await stack.database.dispose()


async def _create_cold(
    stack: RetrievalStack,
    *,
    key: str,
    label: str = "alpha",
    importance: float = 0.35,
    confidence: float = 0.50,
) -> UUID:
    created: list[UUID] = []

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        value = await stack.base.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=OWNER_ID,
                actor_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=content(label),
                importance=importance,
                confidence=confidence,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=stack.base.clock.now(),
            ),
            command=command,
        )
        created.append(value.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    result = await stack.base.executor.execute(
        request(key=key, operation="experience.test_create_cold"),
        handler,
    )
    assert result.status_code == 201
    assert created == [EXPERIENCE_IDS[0]]
    return created[0]


async def _domain_snapshot(
    stack: RetrievalStack,
    experience_id: UUID,
) -> DomainSnapshot:
    async with stack.base.database.read_session() as session:
        state_row = await session.get(ExperienceStateRow, experience_id)
        assert state_row is not None
        payload_row = await session.get(
            ExperiencePayloadRow,
            state_row.current_version_id,
        )
        assert payload_row is not None
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        return DomainSnapshot(
            state=snapshot_from_state_row(state_row),
            events=tuple(
                (
                    row.event_id,
                    row.event_type,
                    row.aggregate_id,
                    row.sequence,
                )
                for row in event_rows
            ),
            payload_codec=payload_row.codec,
            payload=payload_row.payload,
            payload_hash=payload_row.payload_hash,
        )


async def _archive(
    stack: RetrievalStack,
    *,
    experience_id: UUID,
    key: str,
) -> None:
    stack.base.clock.advance(timedelta(days=91))
    before = (await _domain_snapshot(stack, experience_id)).state
    occurred_at = stack.base.clock.now()
    materialized = activation_at(
        ActivationInputs(
            importance=before.importance,
            confidence=before.confidence,
            access_count=before.access_count,
            access_strength=before.access_strength,
            strength_updated_at=before.strength_updated_at,
            last_accessed_at=before.last_accessed_at,
            created_at=before.last_transition_at,
        ),
        occurred_at,
        LifecycleConfig(),
    )
    evaluated = before.model_copy(
        update={
            "access_strength": materialized.decayed_strength,
            "strength_updated_at": occurred_at,
            "activation_score": materialized.score,
            "last_lifecycle_evaluated_at": occurred_at,
            "consecutive_below_threshold": 0,
        }
    )
    after = ExperienceStateSnapshotV1.model_validate(
        {
            **evaluated.model_dump(mode="python"),
            "temperature": Temperature.ARCHIVED,
            "last_transition_at": occurred_at,
            "consecutive_below_threshold": 0,
        }
    )
    evaluated_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceLifecycleEvaluatedV1.event_type,
        payload=ExperienceLifecycleEvaluatedV1(
            schema_version=1,
            experience_id=experience_id,
            cycle_id=ARCHIVE_CYCLE_ID,
            evaluated_at=occurred_at,
            threshold_target="archive",
            before=before,
            after=evaluated,
        ),
        actor_agent_id=None,
        occurred_at=occurred_at,
    )
    archived_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceArchivedV1.event_type,
        payload=ExperienceArchivedV1(
            schema_version=1,
            experience_id=experience_id,
            cycle_id=ARCHIVE_CYCLE_ID,
            reason=StructuredReason.policy_due(),
            before=evaluated,
            after=evaluated,
        ),
        actor_agent_id=None,
        occurred_at=occurred_at,
    )
    transition_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceTemperatureChangedV1.event_type,
        payload=ExperienceTemperatureChangedV1(
            schema_version=1,
            experience_id=experience_id,
            cause="policy_archive",
            cycle_id=ARCHIVE_CYCLE_ID,
            before=evaluated,
            after=after,
        ),
        actor_agent_id=None,
        occurred_at=occurred_at,
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        await stack.mutation_writer.apply_ordered_events(
            uow=uow,
            experience_id=experience_id,
            resulting_state=after,
            events=(evaluated_event, archived_event, transition_event),
            command=command,
        )
        return StoredResponse(status_code=200, body=b"{}")

    result = await stack.base.executor.execute(
        request(key=key, operation="experience.test_archive"),
        handler,
    )
    assert result.status_code == 200


def _search_request(
    query: SearchExperiences,
    *,
    key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{query.owner_agent_id}",
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


async def _search(
    stack: RetrievalStack,
    *,
    query: SearchExperiences,
    key: str,
) -> CommandResult:
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

    return await stack.base.executor.execute(
        _search_request(query, key=key),
        handler,
    )


async def _get(
    stack: RetrievalStack,
    *,
    experience_id: UUID,
    key: str,
) -> CommandResult:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        view = await stack.service.get(
            uow=uow,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            command=command,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": view}),
        )

    return await stack.base.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_ID}",
            operation_scope="experience.get",
            idempotency_key=key,
            method="GET",
            route_template=(
                "/v1/agents/{agent_id}/experiences/{experience_id}"
            ),
            path_parameters={
                "agent_id": OWNER_ID,
                "experience_id": experience_id,
            },
            body=None,
        ),
        handler,
    )


@pytest.mark.asyncio
async def test_archived_get_is_blurred_without_access_and_search_excludes_it(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await _create_cold(
        retrieval_stack,
        key="archived-boundary-create",
        importance=0.20,
        confidence=0.20,
    )
    await _archive(
        retrieval_stack,
        experience_id=experience_id,
        key="archived-boundary-transition",
    )
    before = await _domain_snapshot(retrieval_stack, experience_id)
    assert before.state.temperature is Temperature.ARCHIVED
    assert before.payload_codec is PayloadCodec.ZLIB

    get_result = await _get(
        retrieval_stack,
        experience_id=experience_id,
        key="archived-boundary-get",
    )
    search_result = await _search(
        retrieval_stack,
        query=SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
        ),
        key="archived-boundary-search",
    )

    assert get_result.status_code == 200
    get_body = cast(dict[str, Any], json.loads(get_result.body))
    assert get_body["data"]["temperature"] == Temperature.ARCHIVED.value
    assert get_body["data"]["blurred"] is True
    assert get_body["data"]["body"] is None
    assert get_body["data"]["body_is_excerpt"] is False
    assert get_body["data"]["access_count"] == 0
    assert search_result.status_code == 200
    search_body = cast(dict[str, Any], json.loads(search_result.body))
    assert search_body["data"]["hits"] == []
    assert await _domain_snapshot(retrieval_stack, experience_id) == before


@pytest.mark.parametrize(
    ("importance", "temperature"),
    [
        (0.35, Temperature.WARM),
        (0.90, Temperature.HOT),
    ],
)
@pytest.mark.asyncio
async def test_hot_or_warm_get_clock_regression_has_no_domain_side_effects(
    retrieval_stack: RetrievalStack,
    importance: float,
    temperature: Temperature,
) -> None:
    key = f"{temperature.value}-clock-boundary"
    status, created = await create(
        retrieval_stack.base,
        key=f"{key}-create",
        value=content(key),
        importance=importance,
    )
    assert status == 201
    experience_id = UUID(created["data"]["experience_id"])
    before = await _domain_snapshot(retrieval_stack, experience_id)
    assert before.state.temperature is temperature
    retrieval_stack.base.clock.advance(timedelta(microseconds=-1))

    first = await _get(
        retrieval_stack,
        experience_id=experience_id,
        key=f"{key}-get",
    )
    replay = await _get(
        retrieval_stack,
        experience_id=experience_id,
        key=f"{key}-get",
    )

    assert first.status_code == replay.status_code == 409
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.body == first.body
    error = cast(dict[str, Any], json.loads(first.body))
    assert error["error"]["code"] == "clock_regression"
    assert await _domain_snapshot(retrieval_stack, experience_id) == before


@pytest.mark.asyncio
async def test_qualifying_cold_search_same_key_replays_one_reactivation(
    retrieval_stack: RetrievalStack,
) -> None:
    experience_id = await _create_cold(
        retrieval_stack,
        key="cold-replay-boundary-create",
    )
    before = await _domain_snapshot(retrieval_stack, experience_id)
    retrieval_stack.base.clock.advance(timedelta(hours=1))
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="alpha",
        mode=RetrievalMode.FOCUSED,
    )

    first = await _search(
        retrieval_stack,
        query=query,
        key="cold-replay-boundary-search",
    )
    replay = await _search(
        retrieval_stack,
        query=query,
        key="cold-replay-boundary-search",
    )

    assert first.status_code == replay.status_code == 200
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.body == first.body
    after = await _domain_snapshot(retrieval_stack, experience_id)
    assert after.state.temperature is Temperature.WARM
    assert after.state.access_count == before.state.access_count + 1
    added_types = tuple(
        event_type for _, event_type, _, _ in after.events[len(before.events) :]
    )
    assert added_types == (
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    )


def _ranked_content(label: str) -> VersionContent:
    return VersionContent(
        body=f"{label} full body",
        summary=f"ranktoken {label}",
        mechanism="ranktoken shared mechanism",
        tags=("ranktoken", label),
        applicability=("ranking test",),
        evidence=(),
        falsifiers=("rank order differs",),
    )


@pytest.mark.asyncio
async def test_multiple_access_events_follow_final_response_rank_order(
    retrieval_stack: RetrievalStack,
) -> None:
    created_ids: dict[str, UUID] = {}
    for label, importance in (
        ("low", 0.10),
        ("high", 0.90),
        ("medium", 0.50),
    ):
        status, created = await create(
            retrieval_stack.base,
            key=f"rank-order-{label}",
            value=_ranked_content(label),
            importance=importance,
        )
        assert status == 201
        created_ids[label] = UUID(created["data"]["experience_id"])

    result = await _search(
        retrieval_stack,
        query=SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="ranktoken",
            mode=RetrievalMode.FOCUSED,
        ),
        key="rank-order-search",
    )

    assert result.status_code == 200
    response = cast(dict[str, Any], json.loads(result.body))
    hit_ids = tuple(
        UUID(hit["experience"]["experience_id"])
        for hit in response["data"]["hits"]
    )
    assert hit_ids == (
        created_ids["high"],
        created_ids["medium"],
        created_ids["low"],
    )
    assert all(
        hit["expanded"] is True and hit["experience"]["body"] is not None
        for hit in response["data"]["hits"]
    )
    async with retrieval_stack.base.database.read_session() as session:
        accessed_ids = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.aggregate_id)
                    .where(
                        DomainEventRow.event_type == "experience.accessed"
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        accessed_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "experience.accessed")
        )
    assert accessed_count == 3
    assert accessed_ids == hit_ids
