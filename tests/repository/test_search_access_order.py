from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from tests.integration.test_create_experience import (
    EXPERIENCE_IDS,
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    content,
    create,
    request,
)

from experience_hub.domain import (
    CommandContext,
    PendingEvent,
    StructuredReason,
)
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
)
from experience_hub.experiences.contracts import (
    CreateExperienceVersion,
    ExperienceDraft,
    ExperienceRecord,
)
from experience_hub.experiences.events import (
    ExperienceAccessedV1,
    ExperienceArchivedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperiencePinnedV1,
    ExperienceReactivatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
)
from experience_hub.experiences.repository import snapshot_from_state_row
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.lifecycle import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
    record_access,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.payload_rewrite import rewrite_payload_codec
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

QUERY_HASH = "e" * 64
CYCLE_ID = UUID("00000000-0000-0000-0000-000000000801")


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "search-access-order.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def _inputs(
    state: ExperienceStateSnapshotV1,
) -> ActivationInputs:
    return ActivationInputs(
        importance=state.importance,
        confidence=state.confidence,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        created_at=NOW,
    )


def _access_after(
    before: ExperienceStateSnapshotV1,
    *,
    occurred_at: datetime,
) -> ExperienceStateSnapshotV1:
    access = record_access(
        _inputs(before),
        occurred_at,
        LifecycleConfig(),
    )
    activation = activation_at(
        ActivationInputs(
            importance=before.importance,
            confidence=before.confidence,
            access_count=access.access_count,
            access_strength=access.access_strength,
            strength_updated_at=access.strength_updated_at,
            last_accessed_at=access.last_accessed_at,
            created_at=NOW,
        ),
        occurred_at,
        LifecycleConfig(),
    )
    return ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "access_count": access.access_count,
            "access_strength": access.access_strength,
            "strength_updated_at": access.strength_updated_at,
            "last_accessed_at": access.last_accessed_at,
            "activation_score": activation.score,
        }
    )


async def _snapshot(
    stack: Stack,
    experience_id: UUID,
) -> ExperienceStateSnapshotV1:
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
    assert state is not None
    return snapshot_from_state_row(state)


async def _create_cold(
    stack: Stack,
    *,
    key: str,
    importance: float = 0.35,
    confidence: float = 0.50,
) -> UUID:
    created: list[UUID] = []

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        value = await stack.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=OWNER_ID,
                actor_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=content(key),
                importance=importance,
                confidence=confidence,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=stack.clock.now(),
            ),
            command=command,
        )
        created.append(value.experience_id)
        return StoredResponse(status_code=201, body=b"{}")

    await stack.executor.execute(
        request(key=key, operation="experience.test_create_cold"),
        handler,
    )
    assert created == [EXPERIENCE_IDS[0]]
    return created[0]


async def _correct(
    stack: Stack,
    *,
    key: str,
    experience_id: UUID,
    label: str,
) -> tuple[int, dict[str, Any]]:
    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create_version(
            uow=uow,
            command=CreateExperienceVersion(
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                content=content(label),
            ),
            command_context=command_context,
        )

    result = await stack.executor.execute(
        request(
            key=key,
            operation="experience.create_version",
        ),
        handler,
    )
    return result.status_code, json.loads(result.body)


async def _apply(
    stack: Stack,
    *,
    key: str,
    experience_id: UUID,
    resulting_state: ExperienceStateSnapshotV1,
    events: Sequence[PendingEvent],
) -> tuple[int, ExperienceRecord | None, dict[str, Any]]:
    returned: list[ExperienceRecord] = []
    writer = ExperienceMutationWriter(repository=stack.repository)

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        returned.append(
            await writer.apply_ordered_events(
                uow=uow,
                experience_id=experience_id,
                resulting_state=resulting_state,
                events=events,
                command=command,
            )
        )
        return StoredResponse(status_code=200, body=b'{"data":{}}')

    result = await stack.executor.execute(
        request(key=key, operation="experience.test_mutation"),
        handler,
    )
    return (
        result.status_code,
        None if not returned else returned[0],
        json.loads(result.body),
    )


def _access_event(
    *,
    experience_id: UUID,
    before: ExperienceStateSnapshotV1,
    after: ExperienceStateSnapshotV1,
    occurred_at: datetime,
    actor_agent_id: UUID | None = OWNER_ID,
) -> PendingEvent:
    return PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceAccessedV1.event_type,
        payload=ExperienceAccessedV1(
            schema_version=1,
            experience_id=experience_id,
            version_id=before.current_version_id,
            before=before,
            after=after,
        ),
        actor_agent_id=actor_agent_id,
        occurred_at=occurred_at,
    )


def _cold_reactivation_events(
    *,
    experience_id: UUID,
    before: ExperienceStateSnapshotV1,
    after_access: ExperienceStateSnapshotV1,
    after_temperature: ExperienceStateSnapshotV1,
    occurred_at: datetime,
) -> tuple[PendingEvent, ...]:
    return (
        _access_event(
            experience_id=experience_id,
            before=before,
            after=after_access,
            occurred_at=occurred_at,
        ),
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=experience_id,
            event_type=ExperienceReactivatedV1.event_type,
            payload=ExperienceReactivatedV1(
                schema_version=1,
                experience_id=experience_id,
                query_hash=QUERY_HASH,
                mode="focused",
                signal=0.72,
                before=after_access,
                after=after_access,
            ),
            actor_agent_id=OWNER_ID,
            occurred_at=occurred_at,
        ),
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=experience_id,
            event_type=ExperienceTemperatureChangedV1.event_type,
            payload=ExperienceTemperatureChangedV1(
                schema_version=1,
                experience_id=experience_id,
                cause="cold_reactivation",
                cycle_id=None,
                before=after_access,
                after=after_temperature,
            ),
            actor_agent_id=OWNER_ID,
            occurred_at=occurred_at,
        ),
    )


@pytest.mark.asyncio
async def test_writer_applies_one_access_inside_the_caller_immediate_uow(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="ordinary-access")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    after = _access_after(before, occurred_at=stack.clock.now())

    status, returned, _ = await _apply(
        stack,
        key="ordinary-access-command",
        experience_id=experience_id,
        resulting_state=after,
        events=(
            _access_event(
                experience_id=experience_id,
                before=before,
                after=after,
                occurred_at=stack.clock.now(),
            ),
        ),
    )

    assert status == 200
    assert returned == ExperienceRecord(
        experience_id=experience_id,
        owner_agent_id=OWNER_ID,
        current_version_id=before.current_version_id,
        current_content_hash=before.current_content_hash,
        temperature=Temperature.WARM,
    )
    assert await _snapshot(stack, experience_id) == after


@pytest.mark.asyncio
async def test_cold_chain_rewrites_every_historical_version_to_plain(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="cold-history")
    stack.clock.advance(timedelta(hours=1))
    corrected_status, _ = await _correct(
        stack,
        key="cold-history-correction",
        experience_id=experience_id,
        label="cold-history-v2",
    )
    assert corrected_status == 201
    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    after_access = _access_after(before, occurred_at=stack.clock.now())
    after_temperature = ExperienceStateSnapshotV1.model_validate(
        {
            **after_access.model_dump(mode="python"),
            "temperature": Temperature.WARM,
            "last_transition_at": stack.clock.now(),
            "consecutive_below_threshold": 0,
        }
    )

    status, returned, _ = await _apply(
        stack,
        key="cold-history-reactivation",
        experience_id=experience_id,
        resulting_state=after_temperature,
        events=_cold_reactivation_events(
            experience_id=experience_id,
            before=before,
            after_access=after_access,
            after_temperature=after_temperature,
            occurred_at=stack.clock.now(),
        ),
    )

    assert status == 200
    assert returned is not None
    assert returned.temperature is Temperature.WARM
    async with stack.database.read_session() as session:
        codecs = tuple(
            (
                await session.scalars(
                    select(ExperiencePayloadRow.codec)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperiencePayloadRow.version_id,
                    )
                    .where(
                        ExperienceVersionRow.experience_id == experience_id
                    )
                    .order_by(ExperienceVersionRow.version_number)
                )
            ).all()
        )
        event_types = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.event_type)
                    .where(DomainEventRow.aggregate_id == experience_id)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
    assert codecs == (PayloadCodec.PLAIN, PayloadCodec.PLAIN)
    assert event_types[-3:] == (
        "experience.accessed",
        "experience.reactivated",
        "experience.temperature_changed",
    )
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_temperature_change_rejects_corrupt_historical_content_hash(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="corrupt-history")
    stack.clock.advance(timedelta(hours=1))
    corrected_status, _ = await _correct(
        stack,
        key="corrupt-history-correction",
        experience_id=experience_id,
        label="corrupt-history-v2",
    )
    assert corrected_status == 201
    async with stack.database.transaction(immediate=True) as uow:
        historical_version_id = await uow.session.scalar(
            select(ExperienceVersionRow.version_id).where(
                ExperienceVersionRow.experience_id == experience_id,
                ExperienceVersionRow.version_number == 1,
            )
        )
        assert historical_version_id is not None
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        await uow.session.execute(
            text(
                "UPDATE experience_versions SET content_hash = :content_hash "
                "WHERE version_id = :version_id"
            ),
            {
                "content_hash": "0" * 64,
                "version_id": str(historical_version_id),
            },
        )

    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    after_access = _access_after(before, occurred_at=stack.clock.now())
    after_temperature = ExperienceStateSnapshotV1.model_validate(
        {
            **after_access.model_dump(mode="python"),
            "temperature": Temperature.WARM,
            "last_transition_at": stack.clock.now(),
            "consecutive_below_threshold": 0,
        }
    )

    with pytest.raises(SourceIntegrityError, match="semantic validation"):
        await _apply(
            stack,
            key="corrupt-history-reactivation",
            experience_id=experience_id,
            resulting_state=after_temperature,
            events=_cold_reactivation_events(
                experience_id=experience_id,
                before=before,
                after_access=after_access,
                after_temperature=after_temperature,
                occurred_at=stack.clock.now(),
            ),
        )

    assert await _snapshot(stack, experience_id) == before
    async with stack.database.read_session() as session:
        codecs = tuple(
            (
                await session.scalars(
                    select(ExperiencePayloadRow.codec)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperiencePayloadRow.version_id,
                    )
                    .where(
                        ExperienceVersionRow.experience_id == experience_id
                    )
                    .order_by(ExperienceVersionRow.version_number)
                )
            ).all()
        )
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_id == experience_id)
        )
    assert codecs == (PayloadCodec.ZLIB, PayloadCodec.ZLIB)
    assert event_count == 3


@pytest.mark.asyncio
async def test_writer_rejects_standalone_cold_access(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="standalone-cold-access")
    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    after = _access_after(before, occurred_at=stack.clock.now())

    with pytest.raises(ValueError, match="three-event reactivation"):
        await _apply(
            stack,
            key="standalone-cold-access-command",
            experience_id=experience_id,
            resulting_state=after,
            events=(
                _access_event(
                    experience_id=experience_id,
                    before=before,
                    after=after,
                    occurred_at=stack.clock.now(),
                ),
            ),
        )

    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_writer_rejects_archived_access_even_with_bypassed_payload_validation(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(
        stack,
        key="archive-before-access",
        importance=0.20,
        confidence=0.20,
    )
    stack.clock.advance(timedelta(days=91))
    cold = await _snapshot(stack, experience_id)
    materialized = activation_at(
        _inputs(cold),
        stack.clock.now(),
        LifecycleConfig(),
    )
    evaluated = cold.model_copy(
        update={
            "access_strength": materialized.decayed_strength,
            "strength_updated_at": stack.clock.now(),
            "activation_score": materialized.score,
            "last_lifecycle_evaluated_at": stack.clock.now(),
            "consecutive_below_threshold": 0,
        }
    )
    archived = ExperienceStateSnapshotV1.model_validate(
        {
            **evaluated.model_dump(mode="python"),
            "temperature": Temperature.ARCHIVED,
            "last_transition_at": stack.clock.now(),
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
            cycle_id=CYCLE_ID,
            evaluated_at=stack.clock.now(),
            threshold_target="archive",
            before=cold,
            after=evaluated,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    archive_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceArchivedV1.event_type,
        payload=ExperienceArchivedV1(
            schema_version=1,
            experience_id=experience_id,
            cycle_id=CYCLE_ID,
            reason=StructuredReason.policy_due(),
            before=evaluated,
            after=evaluated,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    transition_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceTemperatureChangedV1.event_type,
        payload=ExperienceTemperatureChangedV1(
            schema_version=1,
            experience_id=experience_id,
            cause="policy_archive",
            cycle_id=CYCLE_ID,
            before=evaluated,
            after=archived,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    status, _, _ = await _apply(
        stack,
        key="archive-before-access-command",
        experience_id=experience_id,
        resulting_state=archived,
        events=(evaluated_event, archive_event, transition_event),
    )
    assert status == 200

    stack.clock.advance(timedelta(hours=1))
    after_access = _access_after(
        archived,
        occurred_at=stack.clock.now(),
    )
    bypassed_payload = ExperienceAccessedV1.model_construct(
        schema_version=1,
        experience_id=experience_id,
        version_id=archived.current_version_id,
        before=archived,
        after=after_access,
    )
    bypassed_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceAccessedV1.event_type,
        payload=bypassed_payload,
        actor_agent_id=OWNER_ID,
        occurred_at=stack.clock.now(),
    )

    with pytest.raises(ValueError, match="Archived"):
        await _apply(
            stack,
            key="archived-access-command",
            experience_id=experience_id,
            resulting_state=after_access,
            events=(bypassed_event,),
        )

    assert await _snapshot(stack, experience_id) == archived
    async with stack.database.read_session() as session:
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_id == experience_id)
        )
    assert event_count == 5


@pytest.mark.asyncio
async def test_writer_rejects_reactivation_without_exact_cold_chain(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="reactivation-only")
    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceReactivatedV1.event_type,
        payload=ExperienceReactivatedV1(
            schema_version=1,
            experience_id=experience_id,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=before,
            after=before,
        ),
        actor_agent_id=OWNER_ID,
        occurred_at=stack.clock.now(),
    )

    with pytest.raises(ValueError, match="exact three-event"):
        await _apply(
            stack,
            key="reactivation-only-command",
            experience_id=experience_id,
            resulting_state=before,
            events=(event,),
        )

    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_writer_rejects_foreign_owner_without_mutation(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="owner-isolation")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    foreign_before = ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "owner_agent_id": OTHER_OWNER_ID,
        }
    )
    stack.clock.advance(timedelta(hours=1))
    foreign_after = _access_after(
        foreign_before,
        occurred_at=stack.clock.now(),
    )

    status, returned, body = await _apply(
        stack,
        key="foreign-owner-mutation",
        experience_id=experience_id,
        resulting_state=foreign_after,
        events=(
            _access_event(
                experience_id=experience_id,
                before=foreign_before,
                after=foreign_after,
                occurred_at=stack.clock.now(),
                actor_agent_id=OTHER_OWNER_ID,
            ),
        ),
    )

    assert status == 404
    assert returned is None
    assert body["error"]["code"] == "experience_not_found"
    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_writer_rejects_retrieval_event_from_non_owner_actor(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="foreign-actor")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    before = await _snapshot(stack, experience_id)
    after = _access_after(before, occurred_at=stack.clock.now())

    with pytest.raises(ValueError, match="owner or system actor"):
        await _apply(
            stack,
            key="foreign-actor-command",
            experience_id=experience_id,
            resulting_state=after,
            events=(
                _access_event(
                    experience_id=experience_id,
                    before=before,
                    after=after,
                    occurred_at=stack.clock.now(),
                    actor_agent_id=OTHER_OWNER_ID,
                ),
            ),
        )

    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_writer_rejects_first_before_that_differs_from_locked_state(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="wrong-before")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    locked = await _snapshot(stack, experience_id)
    wrong_before = ExperienceStateSnapshotV1.model_validate(
        {
            **locked.model_dump(mode="python"),
            "confidence": 0.6,
        }
    )
    after_access = _access_after(
        wrong_before,
        occurred_at=stack.clock.now(),
    )

    with pytest.raises(ValueError, match="locked state"):
        await _apply(
            stack,
            key="wrong-before-command",
            experience_id=experience_id,
            resulting_state=after_access,
            events=(
                _access_event(
                    experience_id=experience_id,
                    before=wrong_before,
                    after=after_access,
                    occurred_at=stack.clock.now(),
                ),
            ),
        )

    assert await _snapshot(stack, experience_id) == locked


@pytest.mark.asyncio
async def test_writer_rejects_nonadjacent_event_snapshots(
    stack: Stack,
) -> None:
    experience_id = await _create_cold(stack, key="nonadjacent-events")
    stack.clock.advance(timedelta(hours=1))
    locked = await _snapshot(stack, experience_id)
    after_access = _access_after(
        locked,
        occurred_at=stack.clock.now(),
    )
    after_temperature = ExperienceStateSnapshotV1.model_validate(
        {
            **after_access.model_dump(mode="python"),
            "temperature": Temperature.WARM,
            "last_transition_at": stack.clock.now(),
        }
    )
    events = list(
        _cold_reactivation_events(
            experience_id=experience_id,
            before=locked,
            after_access=after_access,
            after_temperature=after_temperature,
            occurred_at=stack.clock.now(),
        )
    )
    reactivated = events[1]
    assert isinstance(reactivated.payload, ExperienceReactivatedV1)
    events[1] = PendingEvent(
        aggregate_type=reactivated.aggregate_type,
        aggregate_id=reactivated.aggregate_id,
        event_type=reactivated.event_type,
        payload=ExperienceReactivatedV1(
            schema_version=1,
            experience_id=experience_id,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=locked,
            after=locked,
        ),
        actor_agent_id=reactivated.actor_agent_id,
        occurred_at=reactivated.occurred_at,
    )

    with pytest.raises(ValueError, match="prior after state"):
        await _apply(
            stack,
            key="nonadjacent-events-command",
            experience_id=experience_id,
            resulting_state=after_temperature,
            events=events,
        )

    assert await _snapshot(stack, experience_id) == locked


@pytest.mark.asyncio
async def test_writer_rejects_incomplete_warm_pin_sequence(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="incomplete-warm-pin")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    assert before.temperature is Temperature.WARM
    after = before.model_copy(update={"pinned": True})
    event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperiencePinnedV1.event_type,
        payload=ExperiencePinnedV1(
            schema_version=1,
            experience_id=experience_id,
            reason=None,
            before=before,
            after=after,
        ),
        actor_agent_id=OWNER_ID,
        occurred_at=NOW,
    )

    with pytest.raises(ValueError, match="exact pinned event sequence"):
        await _apply(
            stack,
            key="incomplete-warm-pin-command",
            experience_id=experience_id,
            resulting_state=after,
            events=(event,),
        )

    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_writer_requires_caller_owned_immediate_uow(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="non-immediate")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    after = _access_after(before, occurred_at=NOW + timedelta(hours=1))
    writer = ExperienceMutationWriter(repository=stack.repository)

    async with stack.database.transaction() as uow:
        with pytest.raises(RuntimeError, match="immediate"):
            await writer.apply_ordered_events(
                uow=uow,
                experience_id=experience_id,
                resulting_state=after,
                events=(
                    _access_event(
                        experience_id=experience_id,
                        before=before,
                        after=after,
                        occurred_at=NOW + timedelta(hours=1),
                    ),
                ),
                command=CommandContext(
                    receipt_id=UUID(
                        "00000000-0000-0000-0000-000000000999"
                    ),
                    caller_scope=f"agent:{OWNER_ID}",
                    operation_scope="experience.test_mutation",
                    idempotency_key="non-immediate",
                    request_hash="f" * 64,
                ),
            )


@pytest.mark.asyncio
async def test_writer_returns_clock_regression_without_side_effects(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="clock-regression")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    regressed_at = NOW - timedelta(microseconds=1)
    after = ExperienceStateSnapshotV1.model_validate(
        {
            **before.model_dump(mode="python"),
            "access_count": 1,
            "access_strength": 1.0,
            "strength_updated_at": regressed_at,
            "last_accessed_at": regressed_at,
            "activation_score": 0.9,
        }
    )

    status, returned, body = await _apply(
        stack,
        key="clock-regression-command",
        experience_id=experience_id,
        resulting_state=after,
        events=(
            _access_event(
                experience_id=experience_id,
                before=before,
                after=after,
                occurred_at=regressed_at,
            ),
        ),
    )

    assert status == 409
    assert returned is None
    assert body["error"]["code"] == "clock_regression"
    assert await _snapshot(stack, experience_id) == before
    async with stack.database.read_session() as session:
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_id == experience_id)
        )
    assert event_count == 2


@pytest.mark.asyncio
async def test_codec_failure_rolls_back_events_projection_and_prior_rewrites(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, created = await create(stack, key="rewrite-rollback-v1")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=1))
    corrected_status, _ = await _correct(
        stack,
        key="rewrite-rollback-v2",
        experience_id=experience_id,
        label="rewrite-rollback-v2",
    )
    assert corrected_status == 201
    stack.clock.advance(timedelta(days=365))
    first_before = await _snapshot(stack, experience_id)
    first_materialized = activation_at(
        _inputs(first_before),
        stack.clock.now(),
        LifecycleConfig(),
    )
    first_evaluated = first_before.model_copy(
        update={
            "access_strength": first_materialized.decayed_strength,
            "strength_updated_at": stack.clock.now(),
            "activation_score": first_materialized.score,
            "last_lifecycle_evaluated_at": stack.clock.now(),
            "consecutive_below_threshold": 1,
        }
    )
    first_evaluation_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceLifecycleEvaluatedV1.event_type,
        payload=ExperienceLifecycleEvaluatedV1(
            schema_version=1,
            experience_id=experience_id,
            cycle_id=UUID("00000000-0000-0000-0000-000000000802"),
            evaluated_at=stack.clock.now(),
            threshold_target="demote_cold",
            before=first_before,
            after=first_evaluated,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    first_status, _, _ = await _apply(
        stack,
        key="rewrite-rollback-first-cycle",
        experience_id=experience_id,
        resulting_state=first_evaluated,
        events=(first_evaluation_event,),
    )
    assert first_status == 200

    stack.clock.advance(LifecycleConfig().minimum_cycle_interval)
    before = await _snapshot(stack, experience_id)
    materialized = activation_at(
        _inputs(before),
        stack.clock.now(),
        LifecycleConfig(),
    )
    evaluated = before.model_copy(
        update={
            "access_strength": materialized.decayed_strength,
            "strength_updated_at": stack.clock.now(),
            "activation_score": materialized.score,
            "last_lifecycle_evaluated_at": stack.clock.now(),
            "consecutive_below_threshold": 2,
        }
    )
    after = ExperienceStateSnapshotV1.model_validate(
        {
            **evaluated.model_dump(mode="python"),
            "temperature": Temperature.COLD,
            "last_transition_at": stack.clock.now(),
            "consecutive_below_threshold": 0,
        }
    )
    evaluation_event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceLifecycleEvaluatedV1.event_type,
        payload=ExperienceLifecycleEvaluatedV1(
            schema_version=1,
            experience_id=experience_id,
            cycle_id=CYCLE_ID,
            evaluated_at=stack.clock.now(),
            threshold_target="demote_cold",
            before=before,
            after=evaluated,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    event = PendingEvent(
        aggregate_type="experience",
        aggregate_id=experience_id,
        event_type=ExperienceTemperatureChangedV1.event_type,
        payload=ExperienceTemperatureChangedV1(
            schema_version=1,
            experience_id=experience_id,
            cause="lifecycle_demotion",
            cycle_id=CYCLE_ID,
            before=evaluated,
            after=after,
        ),
        actor_agent_id=None,
        occurred_at=stack.clock.now(),
    )
    calls = 0

    async def fail_second_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second rewrite failure")
        await rewrite_payload_codec(
            session=session,
            version_id=version_id,
            codec=codec,
        )

    monkeypatch.setattr(
        "experience_hub.experiences.transitions.rewrite_payload_codec",
        fail_second_rewrite,
    )

    with pytest.raises(RuntimeError, match="injected second rewrite failure"):
        await _apply(
            stack,
            key="rewrite-rollback-command",
            experience_id=experience_id,
            resulting_state=after,
            events=(evaluation_event, event),
        )

    assert calls == 2
    assert await _snapshot(stack, experience_id) == before
    async with stack.database.read_session() as session:
        codecs = tuple(
            (
                await session.scalars(
                    select(ExperiencePayloadRow.codec)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperiencePayloadRow.version_id,
                    )
                    .where(
                        ExperienceVersionRow.experience_id == experience_id
                    )
                    .order_by(ExperienceVersionRow.version_number)
                )
            ).all()
        )
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.aggregate_id == experience_id)
        )
        failed_receipt_count = await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.idempotency_key
                == "rewrite-rollback-command"
            )
        )
    assert codecs == (PayloadCodec.PLAIN, PayloadCodec.PLAIN)
    assert event_count == 4
    assert failed_receipt_count == 0
