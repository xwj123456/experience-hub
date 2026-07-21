from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, update
from tests.integration.test_create_experience import (
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    content,
    create,
    request,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandContext
from experience_hub.experiences import (
    ExperienceKind,
    LinkRelation,
    Temperature,
    VersionContent,
    encode_version_content,
)
from experience_hub.experiences.contracts import (
    CreateExperienceVersion,
    VersionLinkInput,
)
from experience_hub.experiences.events import (
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.projector import (
    ExperienceProjectionIntegrityError,
)
from experience_hub.experiences.queries import ExperienceNotFoundError
from experience_hub.storage.database import payload_rewrite_guard
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    ProjectionVersionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "create-experience-version.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def correct(
    stack: Stack,
    *,
    key: str,
    experience_id: UUID,
    value: VersionContent | None = None,
    owner_agent_id: UUID = OWNER_ID,
    links: tuple[VersionLinkInput, ...] = (),
) -> tuple[int, dict[str, Any]]:
    command = CreateExperienceVersion(
        owner_agent_id=owner_agent_id,
        experience_id=experience_id,
        content=value or content(key),
        links=links,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.create_version(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key=key,
            owner_agent_id=owner_agent_id,
            operation="experience.create_version",
        ),
        handler,
    )
    return result.status_code, json.loads(result.body)


async def restore_projection_snapshot(
    stack: Stack,
    *,
    snapshot: ExperienceStateSnapshotV1,
    projection_event_id: int,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(
                ExperienceStateRow.experience_id == snapshot.experience_id
            )
            .values(
                owner_agent_id=snapshot.owner_agent_id,
                current_version_id=snapshot.current_version_id,
                current_content_hash=snapshot.current_content_hash,
                temperature=snapshot.temperature,
                importance=snapshot.importance,
                confidence=snapshot.confidence,
                activation_score=snapshot.activation_score,
                source_trust=snapshot.source_trust,
                access_count=snapshot.access_count,
                access_strength=snapshot.access_strength,
                strength_updated_at=snapshot.strength_updated_at,
                last_accessed_at=snapshot.last_accessed_at,
                last_transition_at=snapshot.last_transition_at,
                last_lifecycle_evaluated_at=(
                    snapshot.last_lifecycle_evaluated_at
                ),
                consecutive_below_threshold=(
                    snapshot.consecutive_below_threshold
                ),
                pinned=snapshot.pinned,
                projection_event_id=projection_event_id,
            )
        )


async def append_unprojected_same_aggregate_event(
    stack: Stack,
    *,
    experience_id: UUID,
    event_type: str = "experience.future_nonversion",
) -> int:
    """Advance the authoritative aggregate ledger without its projection."""
    async with stack.database.transaction() as uow:
        head = await uow.session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "experience",
                DomainEventRow.aggregate_id == experience_id,
            )
            .order_by(DomainEventRow.sequence.desc())
            .limit(1)
        )
        assert head is not None
        event = DomainEventRow(
            aggregate_type=head.aggregate_type,
            aggregate_id=head.aggregate_id,
            sequence=head.sequence + 1,
            event_type=event_type,
            payload=canonical_json_bytes({"schema_version": 1}),
            actor_agent_id=head.actor_agent_id,
            causation_id=head.causation_id,
            occurred_at=head.occurred_at,
        )
        uow.session.add(event)
        await uow.session.flush()
        return event.event_id


@pytest.mark.asyncio
async def test_correction_appends_version_and_preserves_lifecycle_state(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="original", value=content("original"))
    experience_id = UUID(created["data"]["experience_id"])
    original_version_id = UUID(created["data"]["version_id"])
    stack.clock.advance(timedelta(hours=168))

    status, corrected = await correct(
        stack,
        key="correction",
        experience_id=experience_id,
        value=content("corrected"),
    )

    assert status == 201
    corrected_version_id = UUID(corrected["data"]["version_id"])
    async with stack.database.read_session() as session:
        versions = tuple(
            (
                await session.scalars(
                    select(ExperienceVersionRow)
                    .where(ExperienceVersionRow.experience_id == experience_id)
                    .order_by(ExperienceVersionRow.version_number)
                )
            ).all()
        )
        state = await session.get(ExperienceStateRow, experience_id)
        event_row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )

    assert [(row.version_number, row.supersedes_version_id) for row in versions] == [
        (1, None),
        (2, original_version_id),
    ]
    assert versions[1].version_id == corrected_version_id
    assert state is not None and event_row is not None
    assert (
        state.current_version_id,
        state.temperature,
        state.importance,
        state.confidence,
        state.source_trust,
        state.access_count,
        state.access_strength,
        state.strength_updated_at,
        state.last_accessed_at,
        state.last_transition_at,
        state.last_lifecycle_evaluated_at,
        state.consecutive_below_threshold,
        state.pinned,
    ) == (
        corrected_version_id,
        Temperature.WARM,
        0.35,
        0.5,
        1.0,
        0,
        0.0,
        stack.clock.now(),
        None,
        NOW,
        None,
        0,
        False,
    )
    assert state.activation_score == pytest.approx(0.33, abs=1e-12)
    event = ExperienceVersionCreatedV1.model_validate_json(event_row.payload)
    changed = {
        field_name
        for field_name in ExperienceStateSnapshotV1.model_fields
        if getattr(event.before, field_name) != getattr(event.after, field_name)
    }
    assert changed == {
        "current_version_id",
        "current_content_hash",
        "activation_score",
        "strength_updated_at",
    }
    assert state.projection_event_id == event_row.event_id
    assert event_row.sequence == 3
    assert (
        await _event_count(stack, experience_id, "experience.version_created")
    ) == 2
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_same_content_correction_on_same_identity_is_allowed(
    stack: Stack,
) -> None:
    original = content("same")
    _, created = await create(stack, key="same-create", value=original)
    experience_id = UUID(created["data"]["experience_id"])

    status, corrected = await correct(
        stack,
        key="same-correction",
        experience_id=experience_id,
        value=original,
    )

    assert status == 201
    assert corrected["data"]["content_hash"] == created["data"]["content_hash"]
    assert corrected["data"]["version_id"] != created["data"]["version_id"]
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ExperienceVersionRow)
                .where(ExperienceVersionRow.experience_id == experience_id)
            )
            == 2
        )


@pytest.mark.asyncio
async def test_correction_to_other_current_identity_hash_conflicts(
    stack: Stack,
) -> None:
    _, first = await create(stack, key="first", value=content("first"))
    await create(stack, key="second", value=content("second"))

    status, body = await correct(
        stack,
        key="duplicate-correction",
        experience_id=UUID(first["data"]["experience_id"]),
        value=content("second"),
    )

    assert status == 409
    assert body["error"]["code"] == "duplicate_experience"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count()).select_from(ExperienceVersionRow)
            )
            == 2
        )


async def _event_count(
    stack: Stack,
    experience_id: UUID,
    event_type: str | None = None,
) -> int:
    statement = (
        select(func.count())
        .select_from(DomainEventRow)
        .where(DomainEventRow.aggregate_id == experience_id)
    )
    if event_type is not None:
        statement = statement.where(DomainEventRow.event_type == event_type)
    async with stack.database.read_session() as session:
        return int(await session.scalar(statement) or 0)


@pytest.mark.asyncio
async def test_foreign_owner_correction_matches_missing_404_without_side_effects(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="foreign-correction")
    experience_id = UUID(created["data"]["experience_id"])
    missing_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")

    foreign_status, foreign = await correct(
        stack,
        key="foreign-owner-correction",
        experience_id=experience_id,
        owner_agent_id=OTHER_OWNER_ID,
    )
    missing_status, missing = await correct(
        stack,
        key="missing-correction",
        experience_id=missing_id,
    )

    assert foreign_status == missing_status == 404
    assert foreign == missing
    async with stack.database.read_session() as session:
        assert await session.scalar(
            select(func.count()).select_from(ExperienceVersionRow)
        ) == 1
        assert await session.scalar(
            select(func.count()).select_from(DomainEventRow)
        ) == 2


@pytest.mark.asyncio
async def test_archived_correction_returns_restore_required(stack: Stack) -> None:
    _, created = await create(stack, key="archived")
    experience_id = UUID(created["data"]["experience_id"])
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(temperature=Temperature.ARCHIVED)
        )

    status, body = await correct(
        stack,
        key="archived-correction",
        experience_id=experience_id,
    )

    assert status == 409
    assert body["error"]["code"] == "restore_required"


@pytest.mark.asyncio
async def test_correction_rejects_a_clock_behind_any_causal_timestamp(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="future-state")
    experience_id = UUID(created["data"]["experience_id"])
    future = NOW + timedelta(minutes=1)
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(
                strength_updated_at=future,
                last_accessed_at=future,
            )
        )

    status, body = await correct(
        stack,
        key="clock-regression",
        experience_id=experience_id,
    )

    assert status == 409
    assert body["error"]["code"] == "clock_regression"
    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ExperienceVersionRow)
                .where(ExperienceVersionRow.experience_id == experience_id)
            )
            == 1
        )


@pytest.mark.asyncio
async def test_correction_links_are_complete_and_reject_self_target(
    stack: Stack,
) -> None:
    _, target = await create(stack, key="target", value=content("target"))
    _, source = await create(stack, key="source", value=content("source"))
    source_id = UUID(source["data"]["experience_id"])
    target_id = UUID(target["data"]["experience_id"])

    status, corrected = await correct(
        stack,
        key="linked-correction",
        experience_id=source_id,
        value=content("source-v2"),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )
    invalid_status, invalid = await correct(
        stack,
        key="self-link",
        experience_id=source_id,
        value=content("source-v3"),
        links=(
            VersionLinkInput(
                target_experience_id=source_id,
                relation=LinkRelation.TESTS,
            ),
        ),
    )

    assert status == 201
    assert invalid_status == 422
    assert invalid["error"]["code"] == "invalid_experience_link"
    async with stack.database.read_session() as session:
        version_id = UUID(corrected["data"]["version_id"])
        link = await session.scalar(
            select(ExperienceLinkRow).where(
                ExperienceLinkRow.source_version_id == version_id
            )
        )
        event = await session.get(
            DomainEventRow,
            link.source_event_id if link is not None else -1,
        )
    assert link is not None and event is not None
    assert event.event_type == ExperienceVersionCreatedV1.event_type


@pytest.mark.asyncio
async def test_shareable_query_selects_current_or_owned_historical_version(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="history", value=content("history-v1"))
    experience_id = UUID(created["data"]["experience_id"])
    version_one_id = UUID(created["data"]["version_id"])
    stack.clock.advance(timedelta(hours=2))
    _, corrected = await correct(
        stack,
        key="history-v2",
        experience_id=experience_id,
        value=content("history-v2"),
    )
    version_two_id = UUID(corrected["data"]["version_id"])

    async with stack.database.read_session() as session:
        current = await stack.query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            version_id=None,
        )
        selected_current = await stack.query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            version_id=version_two_id,
        )
        historical = await stack.query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            version_id=version_one_id,
        )

    assert current == selected_current
    assert current.version_id == version_two_id
    assert historical.version_id == version_one_id
    assert current.content == content("history-v2")
    assert historical.content == content("history-v1")
    for value in (current, historical):
        encoded = encode_version_content(kind=value.kind, content=value.content)
        assert encoded.content_hash == value.content_hash
        assert value.confidence == 0.5
        assert value.temperature is Temperature.WARM
        assert value.latest_causal_at == stack.clock.now()


@pytest.mark.asyncio
async def test_shareable_query_hides_missing_foreign_and_wrong_history_equally(
    stack: Stack,
) -> None:
    _, first = await create(stack, key="owned-one", value=content("owned-one"))
    _, second = await create(stack, key="owned-two", value=content("owned-two"))
    first_id = UUID(first["data"]["experience_id"])
    second_version_id = UUID(second["data"]["version_id"])
    missing_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")

    errors: list[ExperienceNotFoundError] = []
    async with stack.database.read_session() as session:
        for owner_id, experience_id, version_id in (
            (OTHER_OWNER_ID, first_id, None),
            (OWNER_ID, missing_id, None),
            (OWNER_ID, first_id, second_version_id),
        ):
            with pytest.raises(ExperienceNotFoundError) as caught:
                await stack.query.get_owned_shareable_version(
                    session=session,
                    owner_agent_id=owner_id,
                    experience_id=experience_id,
                    version_id=version_id,
                )
            errors.append(caught.value)

    assert [
        (error.code, error.message, error.status_code, error.details)
        for error in errors
    ] == [
        ("experience_not_found", "Experience was not found", 404, {}),
    ] * 3


@pytest.mark.asyncio
async def test_shareable_query_treats_missing_or_mismatched_current_state_as_corrupt(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="projection-corrupt")
    experience_id = UUID(created["data"]["experience_id"])
    version_id = UUID(created["data"]["version_id"])

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(current_content_hash="f" * 64)
        )
    async with stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=version_id,
            )

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            delete(ExperienceStateRow).where(
                ExperienceStateRow.experience_id == experience_id
            )
        )
    async with stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=None,
            )
        with pytest.raises(ExperienceNotFoundError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OTHER_OWNER_ID,
                experience_id=experience_id,
                version_id=None,
            )


@pytest.mark.asyncio
async def test_shareable_query_rejects_stale_projection_checkpoint(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="stale-query")
    experience_id = UUID(created["data"]["experience_id"])
    await append_unprojected_same_aggregate_event(
        stack,
        experience_id=experience_id,
    )

    async with stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=None,
            )


@pytest.mark.asyncio
async def test_correction_rejects_stale_projection_checkpoint(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="stale-correction")
    experience_id = UUID(created["data"]["experience_id"])
    await append_unprojected_same_aggregate_event(
        stack,
        experience_id=experience_id,
    )

    with pytest.raises(SourceIntegrityError):
        await correct(
            stack,
            key="stale-correction-v2",
            experience_id=experience_id,
        )


@pytest.mark.asyncio
async def test_shareable_query_detects_payload_source_corruption(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="corrupt", value=content("corrupt"))
    experience_id = UUID(created["data"]["experience_id"])
    version_id = UUID(created["data"]["version_id"])
    replacement = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content("tampered"),
    )
    async with stack.database.transaction() as uow:
        connection = await uow.session.connection()
        with payload_rewrite_guard(connection):
            await uow.session.execute(
                update(ExperiencePayloadRow)
                .where(ExperiencePayloadRow.version_id == version_id)
                .values(payload=replacement.payload)
            )

    async with stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=None,
            )


@pytest.mark.asyncio
async def test_projector_rejects_wrong_before_on_reapplication(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="reapply-v1")
    experience_id = UUID(created["data"]["experience_id"])
    await correct(
        stack,
        key="reapply-v2",
        experience_id=experience_id,
        value=content("reapply-v2"),
    )
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
    assert row is not None
    stored = stack.projector.stored_event_from_row(row)
    async with stack.database.read_session() as session:
        before_state = await session.get(ExperienceStateRow, experience_id)
        before_checkpoint = await session.get(
            ProjectionVersionRow,
            "experience_state",
        )
        assert before_state is not None and before_checkpoint is not None
        state_event_id = before_state.projection_event_id
        checkpoint_event_id = before_checkpoint.last_applied_event_id

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="before",
        ):
            await stack.projector.apply(uow.session, stored)
    async with stack.database.read_session() as session:
        after_state = await session.get(ExperienceStateRow, experience_id)
        after_checkpoint = await session.get(
            ProjectionVersionRow,
            "experience_state",
        )
    assert after_state is not None and after_checkpoint is not None
    assert after_state.projection_event_id == state_event_id
    assert after_checkpoint.last_applied_event_id == checkpoint_event_id


@pytest.mark.asyncio
async def test_projector_rejects_nonadjacent_prior_checkpoint(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="nonadjacent-v1")
    experience_id = UUID(created["data"]["experience_id"])
    await correct(
        stack,
        key="nonadjacent-v2",
        experience_id=experience_id,
        value=content("nonadjacent-v2"),
    )
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        created_event_id = await session.scalar(
            select(DomainEventRow.event_id).where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.sequence == 1,
            )
        )
    assert row is not None and created_event_id is not None
    stored = stack.projector.stored_event_from_row(row)
    payload = stored.payload
    assert isinstance(payload, ExperienceVersionCreatedV1)
    await restore_projection_snapshot(
        stack,
        snapshot=payload.before,
        projection_event_id=created_event_id,
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="checkpoint",
        ):
            await stack.projector.apply(uow.session, stored)


@pytest.mark.asyncio
async def test_projector_recomputes_correction_materialization(
    stack: Stack,
) -> None:
    _, created = await create(stack, key="materialize-v1")
    experience_id = UUID(created["data"]["experience_id"])
    stack.clock.advance(timedelta(hours=24))
    await correct(
        stack,
        key="materialize-v2",
        experience_id=experience_id,
        value=content("materialize-v2"),
    )
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        initial_version_event_id = await session.scalar(
            select(DomainEventRow.event_id).where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.sequence == 2,
            )
        )
    assert row is not None and initial_version_event_id is not None
    stored = stack.projector.stored_event_from_row(row)
    payload = stored.payload
    assert isinstance(payload, ExperienceVersionCreatedV1)
    before = payload.before
    await restore_projection_snapshot(
        stack,
        snapshot=before,
        projection_event_id=initial_version_event_id,
    )
    forged_after = ExperienceStateSnapshotV1.model_validate(
        {
            **payload.after.model_dump(mode="python"),
            "access_strength": 1.0,
        }
    )
    forged_payload = ExperienceVersionCreatedV1(
        schema_version=1,
        experience_id=payload.experience_id,
        version_id=payload.version_id,
        version_number=payload.version_number,
        supersedes_version_id=payload.supersedes_version_id,
        links=payload.links,
        before=payload.before,
        after=forged_after,
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="materialization",
        ):
            await stack.projector.apply(
                uow.session,
                replace(stored, payload=forged_payload),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [
        "experience.future_nonversion",
        ExperienceVersionCreatedV1.event_type,
    ],
    ids=["unknown", "registered-with-malformed-payload"],
)
async def test_current_operations_reject_invalid_head_even_if_checkpoint_advanced(
    stack: Stack,
    event_type: str,
) -> None:
    _, created = await create(stack, key="unknown-head-v1")
    experience_id = UUID(created["data"]["experience_id"])
    invalid_event_id = await append_unprojected_same_aggregate_event(
        stack,
        experience_id=experience_id,
        event_type=event_type,
    )
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(
                ExperienceStateRow.experience_id == experience_id,
            )
            .values(projection_event_id=invalid_event_id)
        )

    async with stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await stack.query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=None,
            )

    with pytest.raises(SourceIntegrityError):
        await correct(
            stack,
            key="unknown-head-v2",
            experience_id=experience_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [
        "experience.future_nonversion",
        ExperienceVersionCreatedV1.event_type,
    ],
    ids=["unknown", "registered-with-malformed-payload"],
)
async def test_projector_rejects_invalid_adjacent_prior_checkpoint(
    stack: Stack,
    event_type: str,
) -> None:
    _, created = await create(stack, key="unknown-prior-v1")
    experience_id = UUID(created["data"]["experience_id"])
    await correct(
        stack,
        key="unknown-prior-v2",
        experience_id=experience_id,
        value=content("unknown-prior-v2"),
    )
    invalid_event_id = await append_unprojected_same_aggregate_event(
        stack,
        experience_id=experience_id,
        event_type=event_type,
    )
    async with stack.database.transaction() as uow:
        correction = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.sequence == 3,
            )
        )
        assert correction is not None
        later_correction = DomainEventRow(
            aggregate_type=correction.aggregate_type,
            aggregate_id=correction.aggregate_id,
            sequence=5,
            event_type=correction.event_type,
            payload=correction.payload,
            actor_agent_id=correction.actor_agent_id,
            causation_id=correction.causation_id,
            occurred_at=correction.occurred_at,
        )
        uow.session.add(later_correction)
        await uow.session.flush()
        later_correction_id = later_correction.event_id

    async with stack.database.read_session() as session:
        row = await session.get(DomainEventRow, later_correction_id)
    assert row is not None
    stored = stack.projector.stored_event_from_row(row)
    payload = stored.payload
    assert isinstance(payload, ExperienceVersionCreatedV1)
    await restore_projection_snapshot(
        stack,
        snapshot=payload.before,
        projection_event_id=invalid_event_id,
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="checkpoint",
        ):
            await stack.projector.apply(uow.session, stored)
