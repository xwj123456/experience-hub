from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from tests.integration.test_create_experience import (
    NOW,
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
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.events import (
    ExperienceArchivedV1,
    ExperienceConfirmedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperiencePinnedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.experiences.projector import (
    ExperienceProjectionIntegrityError,
)
from experience_hub.experiences.repository import snapshot_from_state_row
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import ExperienceStateRow
from experience_hub.storage.unit_of_work import UnitOfWork

type LifecycleProjectorPayload = (
    ExperienceLifecycleEvaluatedV1
    | ExperienceConfirmedV1
    | ExperienceRefutedV1
    | ExperiencePinnedV1
    | ExperienceUnpinnedV1
    | ExperienceArchivedV1
    | ExperienceRestoredV1
    | ExperienceTemperatureChangedV1
)


@pytest.fixture
async def lifecycle_projector_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "lifecycle-event-projector.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def _snapshot(
    stack: Stack,
    experience_id: UUID,
) -> ExperienceStateSnapshotV1:
    async with stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, experience_id)
        assert state is not None
        return snapshot_from_state_row(state)


def _materialize(
    before: ExperienceStateSnapshotV1,
    *,
    at: datetime,
    confidence: float | None = None,
    changes: dict[str, object] | None = None,
) -> ExperienceStateSnapshotV1:
    materialized_confidence = (
        before.confidence if confidence is None else confidence
    )
    result = activation_at(
        ActivationInputs(
            importance=before.importance,
            confidence=materialized_confidence,
            access_count=before.access_count,
            access_strength=before.access_strength,
            strength_updated_at=before.strength_updated_at,
            last_accessed_at=before.last_accessed_at,
            created_at=NOW,
        ),
        at,
        LifecycleConfig(),
    )
    updates: dict[str, object] = {
        "access_strength": result.decayed_strength,
        "strength_updated_at": at,
        "activation_score": result.score,
    }
    if confidence is not None:
        updates["confidence"] = confidence
    if changes is not None:
        updates.update(changes)
    return before.model_copy(update=updates)


async def _append(
    stack: Stack,
    *,
    key: str,
    events: Sequence[tuple[LifecycleProjectorPayload, datetime]],
) -> CommandResult:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        await uow.append_events(
            command,
            [
                PendingEvent(
                    aggregate_type="experience",
                    aggregate_id=payload.experience_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=OWNER_ID,
                    occurred_at=occurred_at,
                )
                for payload, occurred_at in events
            ],
        )
        return StoredResponse(status_code=200, body=b"{}")

    return await stack.executor.execute(
        request(
            key=key,
            operation="experience.test_lifecycle_projector",
        ),
        handler,
    )


async def _create_cold(stack: Stack) -> UUID:
    created_id: UUID | None = None

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        nonlocal created_id
        created = await stack.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=OWNER_ID,
                actor_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                content=content("cold-lifecycle"),
                importance=0.2,
                confidence=0.2,
                source_trust=1.0,
                initial_temperature=Temperature.COLD,
                links=(),
                occurred_at=NOW,
            ),
            command=command,
        )
        created_id = created.experience_id
        return StoredResponse(status_code=201, body=b"{}")

    result = await stack.executor.execute(
        request(
            key="cold-lifecycle-seed",
            operation="experience.test_lifecycle_seed",
        ),
        handler,
    )
    assert result.status_code == 201
    assert created_id is not None
    return created_id


@pytest.mark.asyncio
async def test_projector_reduces_all_nonarchive_lifecycle_events(
    lifecycle_projector_stack: Stack,
) -> None:
    stack = lifecycle_projector_stack
    _, created = await create(stack, key="lifecycle-reducer")
    experience_id = UUID(created["data"]["experience_id"])
    initial = await _snapshot(stack, experience_id)

    evaluated_at = NOW + timedelta(hours=1)
    evaluated = _materialize(
        initial,
        at=evaluated_at,
        changes={
            "last_lifecycle_evaluated_at": evaluated_at,
            "consecutive_below_threshold": 0,
        },
    )
    evaluation = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=UUID("00000000-0000-0000-0000-000000000501"),
        evaluated_at=evaluated_at,
        threshold_target="none",
        before=initial,
        after=evaluated,
    )
    await _append(
        stack,
        key="apply-evaluation",
        events=((evaluation, evaluated_at),),
    )

    confirmed_at = NOW + timedelta(hours=2)
    confirmed_confidence = (
        evaluated.confidence + (1.0 - evaluated.confidence) * 0.20
    )
    confirmed = _materialize(
        evaluated,
        at=confirmed_at,
        confidence=confirmed_confidence,
    )
    confirmation = ExperienceConfirmedV1(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        evidence=(),
        before=evaluated,
        after=confirmed,
    )
    promoted = confirmed.model_copy(
        update={
            "temperature": Temperature.HOT,
            "last_transition_at": confirmed_at,
            "consecutive_below_threshold": 0,
        }
    )
    promotion = ExperienceTemperatureChangedV1(
        schema_version=1,
        experience_id=experience_id,
        cause="confirmation",
        cycle_id=None,
        before=confirmed,
        after=promoted,
    )
    await _append(
        stack,
        key="apply-confirmation",
        events=((confirmation, confirmed_at), (promotion, confirmed_at)),
    )

    pinned_at = NOW + timedelta(hours=3)
    pinned = _materialize(
        promoted,
        at=pinned_at,
        changes={"pinned": True},
    )
    pin = ExperiencePinnedV1(
        schema_version=1,
        experience_id=experience_id,
        reason=StructuredReason.from_user_text("keep this"),
        before=promoted,
        after=pinned,
    )
    await _append(
        stack,
        key="apply-pin",
        events=((pin, pinned_at),),
    )

    unpinned_at = NOW + timedelta(hours=4)
    unpinned = _materialize(
        pinned,
        at=unpinned_at,
        changes={"pinned": False},
    )
    unpin = ExperienceUnpinnedV1(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        before=pinned,
        after=unpinned,
    )
    await _append(
        stack,
        key="apply-unpin",
        events=((unpin, unpinned_at),),
    )

    refuted_at = NOW + timedelta(hours=5)
    refuted = _materialize(
        unpinned,
        at=refuted_at,
        confidence=unpinned.confidence * 0.65,
    )
    refutation = ExperienceRefutedV1(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        evidence=(),
        before=unpinned,
        after=refuted,
    )
    await _append(
        stack,
        key="apply-refutation",
        events=((refutation, refuted_at),),
    )

    projected = await _snapshot(stack, experience_id)
    assert projected == refuted
    assert projected.temperature is Temperature.HOT
    assert projected.pinned is False
    assert projected.confidence == pytest.approx(0.39, abs=1e-12)


@pytest.mark.asyncio
async def test_projector_rejects_corrupt_lifecycle_math_time_and_pin_flip(
    lifecycle_projector_stack: Stack,
) -> None:
    stack = lifecycle_projector_stack
    _, created = await create(stack, key="corrupt-lifecycle-reducer")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    occurred_at = NOW + timedelta(hours=1)

    wrong_score = _materialize(
        before,
        at=occurred_at,
        changes={
            "activation_score": 0.01,
            "last_lifecycle_evaluated_at": occurred_at,
        },
    )
    evaluation = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=UUID("00000000-0000-0000-0000-000000000502"),
        evaluated_at=occurred_at,
        threshold_target="none",
        before=before,
        after=wrong_score,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="materialization",
    ):
        await _append(
            stack,
            key="reject-evaluation-score",
            events=((evaluation, occurred_at),),
        )

    mismatched_time = _materialize(
        before,
        at=occurred_at + timedelta(minutes=1),
        changes={
            "last_lifecycle_evaluated_at": (
                occurred_at + timedelta(minutes=1)
            ),
        },
    )
    wrong_time = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=UUID("00000000-0000-0000-0000-000000000503"),
        evaluated_at=occurred_at + timedelta(minutes=1),
        threshold_target="none",
        before=before,
        after=mismatched_time,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="evaluation time",
    ):
        await _append(
            stack,
            key="reject-evaluation-time",
            events=((wrong_time, occurred_at),),
        )

    wrong_confidence = _materialize(
        before,
        at=occurred_at,
        confidence=0.55,
    )
    corrupt_confirmation = ExperienceConfirmedV1.model_construct(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        evidence=(),
        before=before,
        after=wrong_confidence,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="confidence",
    ):
        await _append(
            stack,
            key="reject-confirmation-formula",
            events=((corrupt_confirmation, occurred_at),),
        )

    no_flip = _materialize(before, at=occurred_at)
    corrupt_pin = ExperiencePinnedV1.model_construct(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        before=before,
        after=no_flip,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="pin flip",
    ):
        await _append(
            stack,
            key="reject-pin-flip",
            events=((corrupt_pin, occurred_at),),
        )

    assert await _snapshot(stack, experience_id) == before


@pytest.mark.asyncio
async def test_projector_rejects_invalid_lifecycle_target_counter_and_interval(
    lifecycle_projector_stack: Stack,
) -> None:
    stack = lifecycle_projector_stack
    _, created = await create(stack, key="lifecycle-policy-validation")
    experience_id = UUID(created["data"]["experience_id"])
    before = await _snapshot(stack, experience_id)
    cycle_id = UUID("00000000-0000-0000-0000-000000000509")
    first_at = NOW + timedelta(minutes=15)

    wrong_target_state = _materialize(
        before,
        at=first_at,
        changes={
            "last_lifecycle_evaluated_at": first_at,
            "consecutive_below_threshold": 1,
        },
    )
    wrong_target = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        evaluated_at=first_at,
        threshold_target="demote_cold",
        before=before,
        after=wrong_target_state,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="threshold target",
    ):
        await _append(
            stack,
            key="reject-lifecycle-target",
            events=((wrong_target, first_at),),
        )

    wrong_counter_state = _materialize(
        before,
        at=first_at,
        changes={
            "last_lifecycle_evaluated_at": first_at,
            "consecutive_below_threshold": 1,
        },
    )
    wrong_counter = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        evaluated_at=first_at,
        threshold_target="none",
        before=before,
        after=wrong_counter_state,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="counter",
    ):
        await _append(
            stack,
            key="reject-lifecycle-counter",
            events=((wrong_counter, first_at),),
        )

    first_state = _materialize(
        before,
        at=first_at,
        changes={
            "last_lifecycle_evaluated_at": first_at,
            "consecutive_below_threshold": 0,
        },
    )
    first = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        evaluated_at=first_at,
        threshold_target="none",
        before=before,
        after=first_state,
    )
    await _append(
        stack,
        key="apply-first-lifecycle-interval",
        events=((first, first_at),),
    )
    too_soon_at = first_at + timedelta(minutes=1)
    too_soon_state = _materialize(
        first_state,
        at=too_soon_at,
        changes={
            "last_lifecycle_evaluated_at": too_soon_at,
            "consecutive_below_threshold": 0,
        },
    )
    too_soon = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=UUID("00000000-0000-0000-0000-000000000510"),
        evaluated_at=too_soon_at,
        threshold_target="none",
        before=first_state,
        after=too_soon_state,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="minimum interval",
    ):
        await _append(
            stack,
            key="reject-lifecycle-interval",
            events=((too_soon, too_soon_at),),
        )


@pytest.mark.asyncio
async def test_projector_enforces_archive_and_restore_predecessor_order(
    lifecycle_projector_stack: Stack,
) -> None:
    stack = lifecycle_projector_stack
    experience_id = await _create_cold(stack)
    cold = await _snapshot(stack, experience_id)
    cycle_id = UUID("00000000-0000-0000-0000-000000000504")
    archived_at = NOW + timedelta(days=91)
    direct_archive = ExperienceArchivedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        reason=StructuredReason.policy_due(),
        before=cold,
        after=cold,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="lifecycle evaluation",
    ):
        await _append(
            stack,
            key="reject-direct-archive",
            events=((direct_archive, archived_at),),
        )

    evaluated = _materialize(
        cold,
        at=archived_at,
        changes={"last_lifecycle_evaluated_at": archived_at},
    )
    evaluation = ExperienceLifecycleEvaluatedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        evaluated_at=archived_at,
        threshold_target="archive",
        before=cold,
        after=evaluated,
    )
    archived = ExperienceArchivedV1(
        schema_version=1,
        experience_id=experience_id,
        cycle_id=cycle_id,
        reason=StructuredReason.policy_due(),
        before=evaluated,
        after=evaluated,
    )
    archived_state = evaluated.model_copy(
        update={
            "temperature": Temperature.ARCHIVED,
            "last_transition_at": archived_at,
            "consecutive_below_threshold": 0,
        }
    )
    archive_transition = ExperienceTemperatureChangedV1(
        schema_version=1,
        experience_id=experience_id,
        cause="policy_archive",
        cycle_id=cycle_id,
        before=evaluated,
        after=archived_state,
    )
    await _append(
        stack,
        key="apply-archive",
        events=(
            (evaluation, archived_at),
            (archived, archived_at),
            (archive_transition, archived_at),
        ),
    )

    restored_at = archived_at + timedelta(days=1)
    direct_warm = archived_state.model_copy(
        update={
            "temperature": Temperature.WARM,
            "last_transition_at": restored_at,
        }
    )
    direct_restore_transition = ExperienceTemperatureChangedV1(
        schema_version=1,
        experience_id=experience_id,
        cause="restore",
        cycle_id=None,
        before=archived_state,
        after=direct_warm,
    )
    with pytest.raises(
        ExperienceProjectionIntegrityError,
        match="restored event",
    ):
        await _append(
            stack,
            key="reject-direct-restore-transition",
            events=((direct_restore_transition, restored_at),),
        )

    restored = _materialize(archived_state, at=restored_at)
    restore = ExperienceRestoredV1(
        schema_version=1,
        experience_id=experience_id,
        reason=None,
        before=archived_state,
        after=restored,
    )
    warm = restored.model_copy(
        update={
            "temperature": Temperature.WARM,
            "last_transition_at": restored_at,
            "consecutive_below_threshold": 0,
        }
    )
    restore_transition = ExperienceTemperatureChangedV1(
        schema_version=1,
        experience_id=experience_id,
        cause="restore",
        cycle_id=None,
        before=restored,
        after=warm,
    )
    await _append(
        stack,
        key="apply-restore",
        events=((restore, restored_at), (restore_transition, restored_at)),
    )

    projected = await _snapshot(stack, experience_id)
    assert projected == warm
    assert projected.temperature is Temperature.WARM
