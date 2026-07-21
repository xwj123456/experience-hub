from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_create_experience import (
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    create,
)

from experience_hub import canonical_json_bytes
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.models import VersionContent
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import DomainEventRow, ExperienceStateRow
from experience_hub.storage.unit_of_work import UnitOfWork


def searchable_content(label: str, *, body: str | None = None) -> VersionContent:
    return VersionContent(
        body=body or f"Complete operational body for {label}.",
        summary=f"{label} summary",
        mechanism="single writer lease handoff",
        tags=("memory", label),
        applicability=("local runtime",),
        evidence=(),
        falsifiers=("overlapping writer",),
    )


@dataclass(slots=True)
class RetrievalStack:
    base: Stack
    service: RetrievalService


@pytest.fixture
async def retrieval_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[RetrievalStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-search.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    service = RetrievalService(
        clock=stack.clock,
        query=ExperienceQuery(event_registry=stack.registry),
        mutation_writer=ExperienceMutationWriter(repository=stack.repository),
    )
    try:
        yield RetrievalStack(base=stack, service=service)
    finally:
        await stack.database.dispose()


def search_request(
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


async def execute_search(
    stack: RetrievalStack,
    query: SearchExperiences,
    *,
    key: str,
) -> tuple[int, dict[str, object], bool]:
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
        search_request(query, key=key),
        handler,
    )
    return result.status_code, json.loads(result.body), result.replayed


async def access_count(stack: Stack, experience_id: UUID) -> int:
    async with stack.database.read_session() as session:
        value = await session.scalar(
            select(ExperienceStateRow.access_count).where(
                ExperienceStateRow.experience_id == experience_id
            )
        )
    assert value is not None
    return value


@pytest.mark.asyncio
async def test_search_is_owner_scoped_and_idempotent_for_full_content(
    retrieval_stack: RetrievalStack,
) -> None:
    _, owner_created = await create(
        retrieval_stack.base,
        key="owner-search-create",
        value=searchable_content("lease"),
    )
    await create(
        retrieval_stack.base,
        key="foreign-search-create",
        owner_agent_id=OTHER_OWNER_ID,
        value=searchable_content("foreign lease"),
    )
    experience_id = UUID(owner_created["data"]["experience_id"])
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="lease handoff",
        mode=RetrievalMode.FOCUSED,
    )

    first = await execute_search(retrieval_stack, query, key="search-once")
    replay = await execute_search(retrieval_stack, query, key="search-once")

    assert first[0] == 200
    assert first[2] is False
    assert replay[2] is True
    assert replay[1] == first[1]
    hits = first[1]["data"]["hits"]  # type: ignore[index]
    assert len(hits) == 1
    assert hits[0]["experience"]["experience_id"] == str(experience_id)
    assert hits[0]["experience"]["body"].startswith("Complete operational")
    assert await access_count(retrieval_stack.base, experience_id) == 1


@pytest.mark.asyncio
async def test_oversized_ranked_body_does_not_starve_later_small_body(
    retrieval_stack: RetrievalStack,
) -> None:
    _, large_created = await create(
        retrieval_stack.base,
        key="large-create",
        value=searchable_content("alpha", body="x" * 64),
        importance=0.9,
    )
    _, small_created = await create(
        retrieval_stack.base,
        key="small-create",
        value=searchable_content("alpha small", body="ok"),
        importance=0.1,
    )
    large_id = UUID(large_created["data"]["experience_id"])
    small_id = UUID(small_created["data"]["experience_id"])

    status, body, _ = await execute_search(
        retrieval_stack,
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="alpha",
            mode=RetrievalMode.FOCUSED,
            content_budget_bytes=2,
        ),
        key="budget-search",
    )

    assert status == 200
    hits = body["data"]["hits"]  # type: ignore[index]
    assert [hit["experience"]["experience_id"] for hit in hits] == [
        str(large_id),
        str(small_id),
    ]
    assert hits[0]["experience"]["blurred"] is True
    assert hits[1]["experience"]["body"] == "ok"
    assert body["data"]["remaining_content_budget_bytes"] == 0  # type: ignore[index]
    assert await access_count(retrieval_stack.base, large_id) == 0
    assert await access_count(retrieval_stack.base, small_id) == 1

    async with retrieval_stack.base.database.read_session() as session:
        accessed = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "experience.accessed")
        )
    assert accessed == 1
