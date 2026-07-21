from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_create_experience import (
    OWNER_ID,
    Stack,
    build_stack,
    create,
)

from experience_hub import canonical_json_bytes
from experience_hub.domain import CommandRequest
from experience_hub.experiences.models import VersionContent
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.service import ExperienceRetrievalAdapter
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.tables import (
    ExperienceStateRow,
    IdempotencyRecordRow,
)

ONE_SHOT_KEYS = (
    UUID("10000000-0000-0000-0000-000000000001"),
    UUID("10000000-0000-0000-0000-000000000002"),
)


@dataclass(slots=True)
class AdapterStack:
    base: Stack
    adapter: ExperienceRetrievalAdapter


@pytest.fixture
async def adapter_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdapterStack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "retrieval-adapter.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    retrieval = RetrievalService(
        clock=stack.clock,
        query=ExperienceQuery(event_registry=stack.registry),
        mutation_writer=ExperienceMutationWriter(repository=stack.repository),
    )
    adapter = ExperienceRetrievalAdapter(
        executor=stack.executor,
        retrieval_service=retrieval,
        id_generator=SequenceIdGenerator(ONE_SHOT_KEYS),
    )
    try:
        yield AdapterStack(base=stack, adapter=adapter)
    finally:
        await stack.database.dispose()


async def _access_count(stack: Stack, experience_id: UUID) -> int:
    async with stack.database.read_session() as session:
        value = await session.scalar(
            select(ExperienceStateRow.access_count).where(
                ExperienceStateRow.experience_id == experience_id
            )
        )
    assert value is not None
    return value


async def _receipt(
    stack: Stack,
    *,
    operation_scope: str,
    idempotency_key: str,
) -> IdempotencyRecordRow:
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.caller_scope == f"agent:{OWNER_ID}",
                IdempotencyRecordRow.scope == operation_scope,
                IdempotencyRecordRow.idempotency_key == idempotency_key,
            )
        )
    assert row is not None
    return row


async def _create_searchable(stack: Stack, *, key: str) -> UUID:
    _, created = await create(
        stack,
        key=key,
        value=VersionContent(
            body="Complete operational body for lease.",
            summary="Lease summary",
            mechanism="single writer lease handoff",
            tags=("memory", "lease"),
            applicability=("local runtime",),
            evidence=(),
            falsifiers=("overlapping writer",),
        ),
    )
    return UUID(created["data"]["experience_id"])


def _query() -> SearchExperiences:
    return SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="  lease handoff  ",
        mode=RetrievalMode.FOCUSED,
        tags=("memory", "memory"),
        mechanism_cues=(" single writer ",),
        limit=7,
        content_budget_bytes=4_096,
        expand_cold=True,
    )


@pytest.mark.asyncio
async def test_search_adapter_replays_exact_canonical_response_and_access_once(
    adapter_stack: AdapterStack,
) -> None:
    experience_id = await _create_searchable(
        adapter_stack.base,
        key="adapter-search-create",
    )
    query = _query()

    first = await adapter_stack.adapter.search(
        query=query,
        idempotency_key="adapter-search",
    )
    replay = await adapter_stack.adapter.search(
        query=query,
        idempotency_key="adapter-search",
    )

    assert first.replayed is False
    assert replay.replayed is True
    assert (
        replay.status_code,
        replay.content_type,
        replay.headers,
        replay.body,
    ) == (
        first.status_code,
        first.content_type,
        first.headers,
        first.body,
    )
    assert first.status_code == 200
    assert first.body == canonical_json_bytes(json.loads(first.body))
    assert await _access_count(adapter_stack.base, experience_id) == 1

    expected = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope="experience.search",
        idempotency_key="adapter-search",
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences:search",
        path_parameters={"agent_id": OWNER_ID},
        body={
            "query": "lease handoff",
            "mode": RetrievalMode.FOCUSED,
            "tags": ("memory",),
            "mechanism_cues": ("single writer",),
            "limit": 7,
            "content_budget_bytes": 4_096,
            "expand_cold": True,
        },
    )
    receipt = await _receipt(
        adapter_stack.base,
        operation_scope="experience.search",
        idempotency_key="adapter-search",
    )
    assert receipt.request_hash == expected.request_hash
    assert receipt.response_body == first.body


@pytest.mark.asyncio
async def test_search_without_key_retains_unique_one_shot_receipts(
    adapter_stack: AdapterStack,
) -> None:
    experience_id = await _create_searchable(
        adapter_stack.base,
        key="adapter-one-shot-search-create",
    )

    first = await adapter_stack.adapter.search(query=_query())
    second = await adapter_stack.adapter.search(query=_query())

    assert first.replayed is False
    assert second.replayed is False
    assert await _access_count(adapter_stack.base, experience_id) == 2
    for key in ONE_SHOT_KEYS:
        receipt = await _receipt(
            adapter_stack.base,
            operation_scope="experience.search",
            idempotency_key=str(key),
        )
        assert receipt.state == "completed"
        assert receipt.caller_scope == f"agent:{OWNER_ID}"


@pytest.mark.asyncio
async def test_get_adapter_uses_exact_route_and_replays_access_once(
    adapter_stack: AdapterStack,
) -> None:
    experience_id = await _create_searchable(
        adapter_stack.base,
        key="adapter-get-create",
    )

    first = await adapter_stack.adapter.get(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        idempotency_key="adapter-get",
    )
    replay = await adapter_stack.adapter.get(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        idempotency_key="adapter-get",
    )

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.body == first.body
    assert first.status_code == 200
    assert first.body == canonical_json_bytes(json.loads(first.body))
    assert await _access_count(adapter_stack.base, experience_id) == 1

    expected = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope="experience.get",
        idempotency_key="adapter-get",
        method="GET",
        route_template=(
            "/v1/agents/{agent_id}/experiences/{experience_id}"
        ),
        path_parameters={
            "agent_id": OWNER_ID,
            "experience_id": experience_id,
        },
        body=None,
    )
    receipt = await _receipt(
        adapter_stack.base,
        operation_scope="experience.get",
        idempotency_key="adapter-get",
    )
    assert receipt.request_hash == expected.request_hash
    assert receipt.response_body == first.body


@pytest.mark.asyncio
async def test_get_without_key_retains_unique_one_shot_receipts(
    adapter_stack: AdapterStack,
) -> None:
    experience_id = await _create_searchable(
        adapter_stack.base,
        key="adapter-one-shot-get-create",
    )

    first = await adapter_stack.adapter.get(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
    )
    second = await adapter_stack.adapter.get(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
    )

    assert first.replayed is False
    assert second.replayed is False
    assert await _access_count(adapter_stack.base, experience_id) == 2
    for key in ONE_SHOT_KEYS:
        receipt = await _receipt(
            adapter_stack.base,
            operation_scope="experience.get",
            idempotency_key=str(key),
        )
        assert receipt.state == "completed"
        assert receipt.caller_scope == f"agent:{OWNER_ID}"


@pytest.mark.asyncio
async def test_adapter_rejects_wrong_types_before_allocating_one_shot_key(
    adapter_stack: AdapterStack,
) -> None:
    with pytest.raises(ValueError, match="query must be SearchExperiences"):
        await adapter_stack.adapter.search(query=cast(Any, object()))
    with pytest.raises(ValueError, match="owner_agent_id must be a UUID"):
        await adapter_stack.adapter.get(
            owner_agent_id=cast(Any, str(OWNER_ID)),
            experience_id=UUID(int=1),
        )
    with pytest.raises(ValueError, match="experience_id must be a UUID"):
        await adapter_stack.adapter.get(
            owner_agent_id=OWNER_ID,
            experience_id=cast(Any, "not-a-uuid"),
        )
    with pytest.raises(ValueError, match="idempotency_key"):
        await adapter_stack.adapter.search(
            query=_query(),
            idempotency_key=cast(Any, 42),
        )

    result = await adapter_stack.adapter.search(query=_query())

    assert result.status_code == 200
    receipt = await _receipt(
        adapter_stack.base,
        operation_scope="experience.search",
        idempotency_key=str(ONE_SHOT_KEYS[0]),
    )
    assert receipt.state == "completed"
