from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_create_experience import (
    OWNER_ID,
    Stack,
    build_stack,
    create,
)
from tests.integration.test_lifecycle_cycle import run_cycle

from experience_hub.canonical import sha256_hex
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventPayload,
    StructuredReason,
)
from experience_hub.experiences.content import decode_payload
from experience_hub.experiences.contracts import RestoreExperience
from experience_hub.experiences.events import (
    ExperienceAccessedV1,
    ExperienceArchivedV1,
    ExperienceCreatedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperienceRestoredV1,
    ExperienceTemperatureChangedV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import PayloadCodec, Temperature
from experience_hub.experiences.service import ExperienceRetrievalAdapter
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.contracts import decode_lifecycle_result
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import LifecycleService
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
    ExperienceVersionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


@dataclass(frozen=True, slots=True)
class _StoredMemory:
    temperature: Temperature
    codec: PayloadCodec
    current_content_hash: str
    version_content_hash: str
    payload_hash: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _ObservedEvent:
    sequence: int
    event_type: str
    actor_agent_id: UUID | None
    occurred_at: datetime
    payload: EventPayload


async def _stored_memories(
    stack: Stack,
    experience_to_version: dict[UUID, UUID],
) -> dict[UUID, _StoredMemory]:
    memories: dict[UUID, _StoredMemory] = {}
    async with stack.database.read_session() as session:
        for experience_id, version_id in experience_to_version.items():
            state = await session.get(ExperienceStateRow, experience_id)
            version = await session.get(ExperienceVersionRow, version_id)
            payload = await session.get(ExperiencePayloadRow, version_id)
            assert state is not None
            assert version is not None
            assert payload is not None
            memories[experience_id] = _StoredMemory(
                temperature=state.temperature,
                codec=payload.codec,
                current_content_hash=state.current_content_hash,
                version_content_hash=version.content_hash,
                payload_hash=payload.payload_hash,
                payload=payload.payload,
            )
    return memories


async def _events(
    stack: Stack,
    *,
    experience_id: UUID,
) -> tuple[_ObservedEvent, ...]:
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
    return tuple(
        _ObservedEvent(
            sequence=row.sequence,
            event_type=row.event_type,
            actor_agent_id=row.actor_agent_id,
            occurred_at=row.occurred_at,
            payload=stack.registry.decode(
                event_type=row.event_type,
                payload=row.payload,
            ),
        )
        for row in rows
    )


def _assert_storage_hashes(
    memories: dict[UUID, _StoredMemory],
    expected_content_hashes: dict[UUID, str],
) -> None:
    assert set(memories) == set(expected_content_hashes)
    for experience_id, memory in memories.items():
        assert memory.current_content_hash == expected_content_hashes[experience_id]
        assert memory.version_content_hash == expected_content_hashes[experience_id]
        assert memory.payload_hash == sha256_hex(
            decode_payload(memory.codec, memory.payload)
        )


async def _restore(
    stack: Stack,
    *,
    experience_id: UUID,
    reason: str,
) -> tuple[int, dict[str, Any]]:
    command = RestoreExperience(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        reason=reason,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.restore(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_ID}",
            operation_scope="experience.restore",
            idempotency_key="e2e-restore-fading",
            method="POST",
            route_template=(
                "/v1/agents/{agent_id}/experiences/{experience_id}:restore"
            ),
            path_parameters={
                "agent_id": OWNER_ID,
                "experience_id": experience_id,
            },
            body={"reason": reason},
        ),
        handler,
    )
    return result.status_code, json.loads(result.body)


@pytest.mark.asyncio
async def test_memory_lifecycle_diverges_archives_and_restores_exactly(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    config = LifecycleConfig()
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "memory-lifecycle.sqlite3",
        lifecycle_config=config,
    )
    try:
        creation_results = {}
        for key, importance, confidence in (
            ("e2e-hot", 1.0, 1.0),
            ("e2e-rehearsed", 0.3, 0.1),
            ("e2e-fading", 0.3, 0.1),
        ):
            status, response = await create(
                stack,
                key=key,
                importance=importance,
                confidence=confidence,
            )
            assert status == 201
            creation_results[key] = response["data"]

        experience_ids = {
            key: UUID(value["experience_id"]) for key, value in creation_results.items()
        }
        version_ids = {
            key: UUID(value["version_id"]) for key, value in creation_results.items()
        }
        expected_content_hashes = {
            experience_ids[key]: value["content_hash"]
            for key, value in creation_results.items()
        }
        experience_to_version = {
            experience_ids[key]: version_ids[key] for key in experience_ids
        }

        mutation_writer = ExperienceMutationWriter(
            repository=stack.repository,
            lifecycle_config=config,
        )
        retrieval = ExperienceRetrievalAdapter(
            executor=stack.executor,
            retrieval_service=RetrievalService(
                clock=stack.clock,
                query=stack.query,
                mutation_writer=mutation_writer,
                lifecycle_config=config,
            ),
            id_generator=SequenceIdGenerator(
                (UUID("00000000-0000-0000-0000-000000000999"),)
            ),
        )
        get_result = await retrieval.get(
            owner_agent_id=OWNER_ID,
            experience_id=experience_ids["e2e-rehearsed"],
            idempotency_key="e2e-rehearse-once",
        )
        assert get_result.status_code == 200
        assert json.loads(get_result.body)["data"]["experience_id"] == str(
            experience_ids["e2e-rehearsed"]
        )

        initial = await _stored_memories(stack, experience_to_version)
        _assert_storage_hashes(initial, expected_content_hashes)
        assert {
            key: initial[experience_ids[key]].temperature for key in experience_ids
        } == {
            "e2e-hot": Temperature.HOT,
            "e2e-rehearsed": Temperature.WARM,
            "e2e-fading": Temperature.WARM,
        }
        assert {memory.codec for memory in initial.values()} == {PayloadCodec.PLAIN}

        lifecycle = LifecycleService(
            clock=stack.clock,
            receipt_store=stack.receipts,
            repository=LifecycleRepository(),
            mutation_writer=mutation_writer,
            config=config,
        )
        first_cycle_at = stack.clock.advance(timedelta(days=5))
        first_status, first_body, _ = await run_cycle(
            stack,
            lifecycle,
            key="e2e-lifecycle-five-days",
            evaluated_at=first_cycle_at,
        )
        first_cycle = decode_lifecycle_result(first_body)
        assert first_status == 200
        assert (
            first_cycle.evaluated_count,
            first_cycle.transition_count,
            first_cycle.archive_count,
        ) == (3, 0, 0)

        second_cycle_at = stack.clock.advance(config.minimum_cycle_interval)
        second_status, second_body, _ = await run_cycle(
            stack,
            lifecycle,
            key="e2e-lifecycle-second-cycle",
            evaluated_at=second_cycle_at,
        )
        second_cycle = decode_lifecycle_result(second_body)
        assert second_status == 200
        assert (
            second_cycle.evaluated_count,
            second_cycle.transition_count,
            second_cycle.archive_count,
        ) == (3, 1, 0)

        diverged = await _stored_memories(stack, experience_to_version)
        _assert_storage_hashes(diverged, expected_content_hashes)
        assert {
            key: (
                diverged[experience_ids[key]].temperature,
                diverged[experience_ids[key]].codec,
            )
            for key in experience_ids
        } == {
            "e2e-hot": (Temperature.HOT, PayloadCodec.PLAIN),
            "e2e-rehearsed": (Temperature.WARM, PayloadCodec.PLAIN),
            "e2e-fading": (Temperature.COLD, PayloadCodec.ZLIB),
        }
        assert (
            diverged[experience_ids["e2e-hot"]].payload_hash
            == initial[experience_ids["e2e-hot"]].payload_hash
        )
        assert (
            diverged[experience_ids["e2e-rehearsed"]].payload_hash
            == initial[experience_ids["e2e-rehearsed"]].payload_hash
        )
        assert (
            diverged[experience_ids["e2e-fading"]].payload_hash
            == initial[experience_ids["e2e-fading"]].payload_hash
        )
        assert (
            diverged[experience_ids["e2e-fading"]].payload
            != initial[experience_ids["e2e-fading"]].payload
        )

        archive_cycle_at = stack.clock.advance(
            timedelta(days=config.archive_after_days)
        )
        archive_status, archive_body, _ = await run_cycle(
            stack,
            lifecycle,
            key="e2e-lifecycle-archive",
            evaluated_at=archive_cycle_at,
        )
        archive_cycle = decode_lifecycle_result(archive_body)
        assert archive_status == 200
        assert (
            archive_cycle.evaluated_count,
            archive_cycle.transition_count,
            archive_cycle.archive_count,
        ) == (3, 1, 1)

        archived = await _stored_memories(stack, experience_to_version)
        _assert_storage_hashes(archived, expected_content_hashes)
        assert (
            archived[experience_ids["e2e-fading"]].temperature is Temperature.ARCHIVED
        )
        assert archived[experience_ids["e2e-fading"]].codec is PayloadCodec.ZLIB
        assert (
            archived[experience_ids["e2e-fading"]].payload_hash
            == diverged[experience_ids["e2e-fading"]].payload_hash
        )

        restore_at = stack.clock.advance(timedelta(seconds=1))
        restore_reason = "  needed for a new incident  "
        restore_status, restore_body = await _restore(
            stack,
            experience_id=experience_ids["e2e-fading"],
            reason=restore_reason,
        )
        assert restore_status == 200
        assert restore_body["data"]["temperature"] == Temperature.WARM.value

        restored = await _stored_memories(stack, experience_to_version)
        _assert_storage_hashes(restored, expected_content_hashes)
        assert restored[experience_ids["e2e-fading"]].temperature is Temperature.WARM
        assert restored[experience_ids["e2e-fading"]].codec is PayloadCodec.PLAIN
        assert (
            restored[experience_ids["e2e-fading"]].payload_hash
            == initial[experience_ids["e2e-fading"]].payload_hash
        )
        assert (
            restored[experience_ids["e2e-fading"]].payload
            == initial[experience_ids["e2e-fading"]].payload
        )

        observed = {
            key: await _events(stack, experience_id=experience_ids[key])
            for key in experience_ids
        }
        expected_types = {
            "e2e-hot": (
                ExperienceCreatedV1.event_type,
                ExperienceVersionCreatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
            ),
            "e2e-rehearsed": (
                ExperienceCreatedV1.event_type,
                ExperienceVersionCreatedV1.event_type,
                ExperienceAccessedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
            ),
            "e2e-fading": (
                ExperienceCreatedV1.event_type,
                ExperienceVersionCreatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceTemperatureChangedV1.event_type,
                ExperienceLifecycleEvaluatedV1.event_type,
                ExperienceArchivedV1.event_type,
                ExperienceTemperatureChangedV1.event_type,
                ExperienceRestoredV1.event_type,
                ExperienceTemperatureChangedV1.event_type,
            ),
        }
        expected_actors = {
            "e2e-hot": (OWNER_ID, OWNER_ID, None, None, None),
            "e2e-rehearsed": (
                OWNER_ID,
                OWNER_ID,
                OWNER_ID,
                None,
                None,
                None,
            ),
            "e2e-fading": (
                OWNER_ID,
                OWNER_ID,
                None,
                None,
                None,
                None,
                None,
                None,
                OWNER_ID,
                OWNER_ID,
            ),
        }
        for key, events in observed.items():
            assert tuple(event.event_type for event in events) == expected_types[key]
            assert tuple(event.sequence for event in events) == tuple(
                range(1, len(events) + 1)
            )
            assert (
                tuple(event.actor_agent_id for event in events) == expected_actors[key]
            )

        lifecycle_expectations = {
            first_cycle.cycle_id: first_cycle_at,
            second_cycle.cycle_id: second_cycle_at,
            archive_cycle.cycle_id: archive_cycle_at,
        }
        for events in observed.values():
            lifecycle_events = (
                event
                for event in events
                if isinstance(event.payload, ExperienceLifecycleEvaluatedV1)
            )
            for event, (cycle_id, evaluated_at) in zip(
                lifecycle_events,
                lifecycle_expectations.items(),
                strict=True,
            ):
                assert event.occurred_at == evaluated_at
                assert event.payload.cycle_id == cycle_id
                assert event.payload.evaluated_at == evaluated_at

        fading_events = observed["e2e-fading"]
        demotion = fading_events[4].payload
        archive_explanation = fading_events[6].payload
        archive_transition = fading_events[7].payload
        restore_explanation = fading_events[8].payload
        restore_transition = fading_events[9].payload
        assert isinstance(demotion, ExperienceTemperatureChangedV1)
        assert demotion.cause == "lifecycle_demotion"
        assert demotion.cycle_id == second_cycle.cycle_id
        assert isinstance(archive_explanation, ExperienceArchivedV1)
        assert archive_explanation.cycle_id == archive_cycle.cycle_id
        assert archive_explanation.reason == StructuredReason.policy_due()
        assert isinstance(archive_transition, ExperienceTemperatureChangedV1)
        assert archive_transition.cause == "policy_archive"
        assert archive_transition.cycle_id == archive_cycle.cycle_id
        assert isinstance(restore_explanation, ExperienceRestoredV1)
        assert restore_explanation.reason == StructuredReason.from_user_text(
            restore_reason
        )
        assert fading_events[8].occurred_at == restore_at
        assert isinstance(restore_transition, ExperienceTemperatureChangedV1)
        assert restore_transition.cause == "restore"
        assert restore_transition.cycle_id is None
        assert fading_events[9].occurred_at == restore_at

        verification = await stack.manager.verify(stack.database)
        assert verification.matches
        assert verification.differences == ()
    finally:
        await stack.database.dispose()
