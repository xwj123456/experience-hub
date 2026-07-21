from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_capsule_corroboration import (
    ADOPTER,
    PUBLISHER_A,
    PUBLISHER_B,
    RELAY,
    CorroborationStack,
    OwnedExperience,
    adopt,
    adoption_row,
    build_stack,
    create_owned_experience,
    create_topic,
    publish,
    subscribe,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.sharing.models import ProvenanceHop
from experience_hub.storage.idempotency import CommandResult
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceStateRow,
)


def _adoption_data(result: CommandResult) -> dict[str, Any]:
    assert result.status_code == 200, json.loads(result.body)
    response = cast(dict[str, Any], json.loads(result.body))
    return cast(dict[str, Any], response["data"])


async def _confidence(
    stack: CorroborationStack,
    experience_id: UUID,
) -> float:
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
    assert state is not None
    return state.confidence


@pytest.mark.asyncio
async def test_three_agent_echo_preserves_root_and_only_independent_root_scores(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "echo-resistance.sqlite3",
    )

    try:
        source_a = await create_owned_experience(
            stack,
            owner_agent_id=PUBLISHER_A,
            key="echo-e2e-source-a",
            confidence=0.80,
        )
        source_b = await create_owned_experience(
            stack,
            owner_agent_id=PUBLISHER_B,
            key="echo-e2e-source-b",
            confidence=0.80,
        )
        topic_id = await create_topic(stack)
        await subscribe(stack, subscriber_agent_id=RELAY, topic_id=topic_id)
        await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)

        original = await publish(
            stack,
            publisher_agent_id=PUBLISHER_A,
            topic_id=topic_id,
            experience=source_a,
            key="echo-e2e-original",
        )
        relay_result = _adoption_data(
            await adopt(
                stack,
                adopter_agent_id=RELAY,
                item_id=original.item_ids[RELAY],
                key="echo-e2e-relay-adopts",
            )
        )
        assert relay_result["created"] is True
        assert relay_result["corroboration_applied"] is False
        relay_view = cast(dict[str, Any], relay_result["experience"])
        relay_experience = OwnedExperience(
            experience_id=UUID(relay_view["experience_id"]),
            version_id=UUID(relay_view["current_version_id"]),
            content_hash=cast(str, relay_view["current_content_hash"]),
        )
        relay_adoption = await adoption_row(
            stack,
            adopter_agent_id=RELAY,
            capsule_id=original.capsule_id,
        )

        echo = await publish(
            stack,
            publisher_agent_id=RELAY,
            topic_id=topic_id,
            experience=relay_experience,
            key="echo-e2e-relay-republishes",
            parent_adoption_id=relay_adoption.adoption_id,
        )
        echo_result = _adoption_data(
            await adopt(
                stack,
                adopter_agent_id=ADOPTER,
                item_id=echo.item_ids[ADOPTER],
                key="echo-e2e-adopter-takes-multihop-first",
            )
        )
        assert echo_result["created"] is True
        assert echo_result["corroboration_applied"] is False
        adopter_view = cast(dict[str, Any], echo_result["experience"])
        adopter_experience_id = UUID(adopter_view["experience_id"])
        confidence_after_echo = await _confidence(stack, adopter_experience_id)
        assert confidence_after_echo == pytest.approx(0.80 * 0.50 * 0.50)

        # A direct copy of the original observation overlaps the root already
        # represented by the multi-hop adoption and must not score again.
        overlapping = _adoption_data(
            await adopt(
                stack,
                adopter_agent_id=ADOPTER,
                item_id=original.item_ids[ADOPTER],
                key="echo-e2e-adopter-takes-overlapping-root",
            )
        )
        assert overlapping["created"] is False
        assert overlapping["corroboration_applied"] is False
        assert await _confidence(stack, adopter_experience_id) == pytest.approx(
            confidence_after_echo
        )

        independent = await publish(
            stack,
            publisher_agent_id=PUBLISHER_B,
            topic_id=topic_id,
            experience=source_b,
            key="echo-e2e-independent-root",
        )
        independent_result = _adoption_data(
            await adopt(
                stack,
                adopter_agent_id=ADOPTER,
                item_id=independent.item_ids[ADOPTER],
                key="echo-e2e-adopter-takes-independent-root",
            )
        )
        assert independent_result["created"] is False
        assert independent_result["corroboration_applied"] is True
        expected_after_independent = (
            confidence_after_echo + (1.0 - confidence_after_echo) * 0.20 * 0.50
        )
        assert await _confidence(stack, adopter_experience_id) == pytest.approx(
            expected_after_independent
        )

        # A second capsule with B's same root is another propagation copy, not
        # another independent observation.
        independent_repeat = await publish(
            stack,
            publisher_agent_id=PUBLISHER_B,
            topic_id=topic_id,
            experience=source_b,
            key="echo-e2e-independent-root-repeat",
        )
        repeated_result = _adoption_data(
            await adopt(
                stack,
                adopter_agent_id=ADOPTER,
                item_id=independent_repeat.item_ids[ADOPTER],
                key="echo-e2e-adopter-takes-independent-repeat",
            )
        )
        assert repeated_result["created"] is False
        assert repeated_result["corroboration_applied"] is False
        assert await _confidence(stack, adopter_experience_id) == pytest.approx(
            expected_after_independent
        )

        direct_adoption = await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=original.capsule_id,
        )
        echo_adoption = await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=echo.capsule_id,
        )
        independent_adoption = await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=independent.capsule_id,
        )
        repeated_adoption = await adoption_row(
            stack,
            adopter_agent_id=ADOPTER,
            capsule_id=independent_repeat.capsule_id,
        )
        async with stack.database.read_session() as session:
            original_capsule = await session.get(
                ExperienceCapsuleRow,
                original.capsule_id,
            )
            echo_capsule = await session.get(
                ExperienceCapsuleRow,
                echo.capsule_id,
            )
            independent_capsule = await session.get(
                ExperienceCapsuleRow,
                independent.capsule_id,
            )
            corroborated_events = await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_id == adopter_experience_id,
                    DomainEventRow.event_type == "experience.corroborated",
                )
            )
            scored_roots = await session.scalar(
                select(func.count())
                .select_from(AdoptionRecordRow)
                .where(
                    AdoptionRecordRow.resulting_experience_id == adopter_experience_id,
                    AdoptionRecordRow.corroboration_applied.is_(True),
                )
            )

        assert original_capsule is not None
        assert echo_capsule is not None
        assert independent_capsule is not None
        assert (
            relay_adoption.root_fingerprint
            == echo_capsule.root_fingerprint
            == echo_adoption.root_fingerprint
            == direct_adoption.root_fingerprint
            == original_capsule.root_fingerprint
        )
        assert independent_adoption.root_fingerprint != (
            original_capsule.root_fingerprint
        )
        assert (
            repeated_adoption.root_fingerprint
            == independent_adoption.root_fingerprint
            == independent_capsule.root_fingerprint
        )
        assert relay_adoption.provenance_chain == canonical_json_bytes(
            (
                ProvenanceHop(
                    capsule_id=original.capsule_id,
                    publisher_agent_id=PUBLISHER_A,
                ),
            )
        )
        assert echo_capsule.provenance_chain == relay_adoption.provenance_chain
        assert echo_adoption.provenance_chain == canonical_json_bytes(
            (
                ProvenanceHop(
                    capsule_id=original.capsule_id,
                    publisher_agent_id=PUBLISHER_A,
                ),
                ProvenanceHop(
                    capsule_id=echo.capsule_id,
                    publisher_agent_id=RELAY,
                ),
            )
        )
        assert direct_adoption.provenance_chain == relay_adoption.provenance_chain
        assert corroborated_events == 1
        assert scored_roots == 1
    finally:
        await stack.database.dispose()
