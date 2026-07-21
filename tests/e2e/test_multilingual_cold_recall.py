from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from tests.integration.test_cold_reactivation import (
    AggregateSnapshot,
    RetrievalStack,
    aggregate_events,
    aggregate_snapshot,
    create_cold,
    execute_search,
    threshold_content,
)
from tests.integration.test_create_experience import OWNER_ID, build_stack

from experience_hub.experiences.content import decode_payload
from experience_hub.experiences.events import (
    ExperienceReactivatedV1,
    ExperienceTemperatureChangedV1,
)
from experience_hub.experiences.models import PayloadCodec, Temperature
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import (
    ASSOCIATIVE_COLD_EXPANSION_THRESHOLD,
    FOCUSED_COLD_EXPANSION_THRESHOLD,
    RetrievalService,
    retrieval_query_hash,
)
from experience_hub.retrieval.tokenizer import normalize_text


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            item
            for nested in value.values()
            for item in _string_values(nested)
        )
    if isinstance(value, list):
        return tuple(item for nested in value for item in _string_values(nested))
    return ()


def _assert_private_evidence_absent(
    payloads: tuple[bytes, ...],
    *,
    private_values: tuple[str, ...],
) -> None:
    event_values = tuple(
        item
        for payload in payloads
        for item in _string_values(json.loads(payload))
    )
    normalized_event_values = tuple(normalize_text(value) for value in event_values)
    for private_value in private_values:
        assert all(private_value not in event_value for event_value in event_values)
        normalized_private = normalize_text(private_value)
        if not normalized_private:
            continue
        assert all(
            normalized_private not in event_value
            for event_value in normalized_event_values
        )


@pytest.fixture
async def cold_recall_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[RetrievalStack]:
    base = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "multilingual-cold-recall.sqlite3",
    )
    base.manager.registry.register(ExperienceTermsProjector(base.registry))
    stack = RetrievalStack(
        base=base,
        service=RetrievalService(
            clock=base.clock,
            query=base.query,
            mutation_writer=ExperienceMutationWriter(
                repository=base.repository,
            ),
        ),
    )
    try:
        yield stack
    finally:
        await base.database.dispose()


async def assert_exact_cold_reactivation(
    stack: RetrievalStack,
    *,
    experience_id: UUID,
    before: AggregateSnapshot,
    query: SearchExperiences,
    body: str,
    expected_signal: float,
) -> None:
    after = await aggregate_snapshot(stack, experience_id)
    assert before.state.temperature is Temperature.COLD
    assert before.codec is PayloadCodec.ZLIB
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

    events = await aggregate_events(stack, experience_id)
    assert tuple(row.event_type for row in events) == (
        "experience.created",
        "experience.version_created",
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    )
    assert tuple(row.sequence for row in events) == (1, 2, 3, 4, 5)
    assert all(row.actor_agent_id == OWNER_ID for row in events)
    added = events[before.event_count :]
    assert len({row.causation_id for row in added}) == 1
    reactivated = ExperienceReactivatedV1.model_validate_json(added[1].payload)
    assert reactivated.query_hash == retrieval_query_hash(query)
    assert reactivated.mode == query.mode.value
    assert reactivated.signal == pytest.approx(expected_signal, abs=1e-12)
    temperature_changed = ExperienceTemperatureChangedV1.model_validate_json(
        added[2].payload
    )
    assert temperature_changed.cause == "cold_reactivation"
    assert temperature_changed.cycle_id is None

    _assert_private_evidence_absent(
        tuple(row.payload for row in added),
        private_values=(
            body,
            query.query,
            *query.tags,
            *query.mechanism_cues,
        ),
    )


@pytest.mark.parametrize(
    ("body", "query_text"),
    [
        ("租约交接保障单写者", "租约交接保障单写者"),
        ("lease handoff fencing token", "lease handoff fencing token"),
        ("缓存 cache 租约 handoff", "缓存 cache 租约 handoff"),
    ],
    ids=("chinese", "english", "mixed"),
)
@pytest.mark.asyncio
async def test_exact_multilingual_cues_expand_and_reactivate_cold_memory(
    cold_recall_stack: RetrievalStack,
    body: str,
    query_text: str,
) -> None:
    experience_id = await create_cold(
        cold_recall_stack,
        key="exact-multilingual-create",
        value=threshold_content(body=body),
    )
    before = await aggregate_snapshot(cold_recall_stack, experience_id)
    cold_recall_stack.base.clock.advance(timedelta(hours=1))
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query=query_text,
        mode=RetrievalMode.FOCUSED,
    )

    status, response, replayed = await execute_search(
        cold_recall_stack,
        query,
        key="exact-multilingual-search",
    )

    assert status == 200
    assert replayed is False
    hit = response["data"]["hits"][0]
    assert hit["lexical_or_trigram_relevance"] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert hit["experience"]["body"] == body
    assert hit["experience"]["blurred"] is False
    assert hit["experience"]["temperature"] == "warm"
    assert hit["expanded"] is True
    assert hit["reactivated"] is True
    await assert_exact_cold_reactivation(
        cold_recall_stack,
        experience_id=experience_id,
        before=before,
        query=query,
        body=body,
        expected_signal=1.0,
    )


@pytest.mark.asyncio
async def test_irrelevant_and_weak_mixed_cues_never_expand_cold_memory(
    cold_recall_stack: RetrievalStack,
) -> None:
    body = "租约交接 lease handoff"
    experience_id = await create_cold(
        cold_recall_stack,
        key="irrelevant-create",
        value=threshold_content(body=body),
    )
    before = await aggregate_snapshot(cold_recall_stack, experience_id)
    cold_recall_stack.base.clock.advance(timedelta(hours=1))

    irrelevant = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="量子果园 quantum orchard",
        mode=RetrievalMode.FOCUSED,
    )
    status, response, _ = await execute_search(
        cold_recall_stack,
        irrelevant,
        key="irrelevant-search",
    )
    assert status == 200
    assert response["data"]["hits"] == []
    assert await aggregate_snapshot(cold_recall_stack, experience_id) == before

    weak = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="租约断开 lease unknown",
        mode=RetrievalMode.FOCUSED,
    )
    status, response, _ = await execute_search(
        cold_recall_stack,
        weak,
        key="weak-search",
    )

    assert status == 200
    hits = response["data"]["hits"]
    assert len(hits) == 1
    hit = hits[0]
    assert hit["lexical_or_trigram_relevance"] == pytest.approx(
        0.5,
        abs=1e-12,
    )
    assert hit["experience"]["body"] is None
    assert hit["experience"]["blurred"] is True
    assert hit["experience"]["temperature"] == "cold"
    assert hit["expanded"] is False
    assert hit["reactivated"] is False
    assert await aggregate_snapshot(cold_recall_stack, experience_id) == before

    events = await aggregate_events(cold_recall_stack, experience_id)
    assert tuple(row.event_type for row in events) == (
        "experience.created",
        "experience.version_created",
    )
    event_payloads = b"\n".join(row.payload for row in events)
    assert body.encode("utf-8") not in event_payloads
    assert irrelevant.query.encode("utf-8") not in event_payloads
    assert weak.query.encode("utf-8") not in event_payloads


@pytest.mark.asyncio
async def test_focused_chinese_cue_expands_at_exact_point_seven_two_boundary(
    cold_recall_stack: RetrievalStack,
) -> None:
    body = "冷热记忆召回应使用上下文线索精准匹配"
    query_text = f"{body}并保持安全"
    experience_id = await create_cold(
        cold_recall_stack,
        key="focused-boundary-create",
        value=threshold_content(body=body),
    )
    before = await aggregate_snapshot(cold_recall_stack, experience_id)
    cold_recall_stack.base.clock.advance(timedelta(hours=1))
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query=query_text,
        mode=RetrievalMode.FOCUSED,
    )

    status, response, _ = await execute_search(
        cold_recall_stack,
        query,
        key="focused-boundary-search",
    )

    assert status == 200
    hit = response["data"]["hits"][0]
    assert hit["lexical_or_trigram_relevance"] == pytest.approx(
        FOCUSED_COLD_EXPANSION_THRESHOLD,
        abs=1e-12,
    )
    assert hit["experience"]["body"] == body
    assert hit["expanded"] is True
    assert hit["reactivated"] is True
    await assert_exact_cold_reactivation(
        cold_recall_stack,
        experience_id=experience_id,
        before=before,
        query=query,
        body=body,
        expected_signal=FOCUSED_COLD_EXPANSION_THRESHOLD,
    )


@pytest.mark.asyncio
async def test_associative_mechanism_cues_expand_at_exact_point_six_five_boundary(
    cold_recall_stack: RetrievalStack,
) -> None:
    mechanisms = (
        "租约",
        "handoff",
        "围栏",
        "token",
        "仲裁",
        "quorum",
        "快照",
        "snapshot",
        "回放",
        "replay",
        "压缩",
        "codec",
        "校验",
        "hash",
        "隔离",
        "isolation",
        "幂等",
        "idempotency",
        "因果",
        "causality",
    )
    body = "隐藏的混合关联经验"
    experience_id = await create_cold(
        cold_recall_stack,
        key="associative-boundary-create",
        value=threshold_content(
            body=body,
            mechanism=" ".join(mechanisms[:13]),
        ),
    )
    before = await aggregate_snapshot(cold_recall_stack, experience_id)
    cold_recall_stack.base.clock.advance(timedelta(hours=1))
    query = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="!!!",
        mode=RetrievalMode.ASSOCIATIVE,
        tags=("私密召回标签", "private recall tag"),
        mechanism_cues=mechanisms,
    )

    status, response, _ = await execute_search(
        cold_recall_stack,
        query,
        key="associative-boundary-search",
    )

    assert status == 200
    hit = response["data"]["hits"][0]
    assert hit["lexical_or_trigram_relevance"] < (FOCUSED_COLD_EXPANSION_THRESHOLD)
    assert hit["mechanism_relevance"] == pytest.approx(
        ASSOCIATIVE_COLD_EXPANSION_THRESHOLD,
        abs=1e-12,
    )
    assert hit["experience"]["body"] == body
    assert hit["expanded"] is True
    assert hit["reactivated"] is True
    await assert_exact_cold_reactivation(
        cold_recall_stack,
        experience_id=experience_id,
        before=before,
        query=query,
        body=body,
        expected_signal=ASSOCIATIVE_COLD_EXPANSION_THRESHOLD,
    )
