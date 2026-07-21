from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select, text
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    CAPSULE_ID,
    PUBLISHER_ID,
    AdoptionStack,
    arrange_pending_capsule,
    build_stack,
    request,
)
from tests.integration.test_capsule_rejection import reject

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    PendingEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.sharing.events import CapsuleFeedbackRecordedV1
from experience_hub.sharing.models import FeedbackVerdict
from experience_hub.sharing.projector import (
    AgentReputationProjector,
    SharingProjectionIntegrityError,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
)
from experience_hub.storage.tables import (
    AgentReputationRow,
    CapsuleFeedbackRow,
    DomainEventRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

# Deliberately reverse lexical UUID order versus revision/event order. A rebuild
# that scans source rows by identity instead of following ordered events fails.
FIRST_FEEDBACK_ID = UUID("ffffffff-ffff-ffff-ffff-fffffffffff1")
SECOND_FEEDBACK_ID = UUID("00000000-0000-0000-0000-0000000000f2")
ORPHAN_FEEDBACK_ID = UUID("00000000-0000-0000-0000-0000000000f3")
FEEDBACK_REASON = StructuredReason.from_user_text(
    "Revised after checking the transported procedure against local evidence."
)
FEEDBACK_EVIDENCE = (TypedEvidence(type="experiment", id="exp:reputation-rebuild"),)


@dataclass(frozen=True, slots=True)
class FeedbackHistory:
    first_event_id: int
    second_event_id: int


def _manager(stack: AdoptionStack) -> ProjectionManager:
    manager = cast(Any, stack.database)._projection_applier
    assert isinstance(manager, ProjectionManager)
    return manager


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "reputation-rebuild.sqlite3",
    )
    _manager(value).registry.register(AgentReputationProjector(value.registry))
    try:
        yield value
    finally:
        await value.database.dispose()


async def _record_feedback_source_event(
    stack: AdoptionStack,
    *,
    key: str,
    feedback_id: UUID,
    revision: int,
    previous_verdict: FeedbackVerdict | None,
    current_verdict: FeedbackVerdict,
    alpha_before: int,
    beta_before: int,
    alpha_after: int,
    beta_after: int,
) -> int:
    occurred_at = stack.clock.now()
    payload = CapsuleFeedbackRecordedV1(
        schema_version=1,
        feedback_id=feedback_id,
        observer_agent_id=ADOPTER_ID,
        capsule_id=CAPSULE_ID,
        publisher_agent_id=PUBLISHER_ID,
        revision=revision,
        previous_verdict=previous_verdict,
        current_verdict=current_verdict,
        alpha_before=alpha_before,
        beta_before=beta_before,
        alpha_after=alpha_after,
        beta_after=beta_after,
    )
    appended_ids: list[int] = []

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        uow.session.add(
            CapsuleFeedbackRow(
                feedback_id=feedback_id,
                observer_agent_id=ADOPTER_ID,
                capsule_id=CAPSULE_ID,
                revision=revision,
                verdict=current_verdict,
                reason=canonical_json_bytes(FEEDBACK_REASON),
                evidence=canonical_json_bytes(FEEDBACK_EVIDENCE),
                created_at=occurred_at,
            )
        )
        await uow.session.flush()
        await stack.receipts.attach_resource(
            uow=uow,
            receipt_id=context.receipt_id,
            resource_type="feedback",
            resource_id=feedback_id,
        )
        stored = await uow.append_events(
            context,
            (
                PendingEvent(
                    aggregate_type="capsule",
                    aggregate_id=CAPSULE_ID,
                    event_type=CapsuleFeedbackRecordedV1.event_type,
                    payload=payload,
                    actor_agent_id=ADOPTER_ID,
                    occurred_at=occurred_at,
                ),
            ),
        )
        assert len(stored) == 1
        appended_ids.append(stored[0].event_id)
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "feedback_id": feedback_id,
                        "revision": revision,
                    }
                }
            ),
        )

    result = await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.feedback",
            route_template=("/v1/agents/{agent_id}/capsules/{capsule_id}:feedback"),
            agent_id=ADOPTER_ID,
            path_parameters={
                "agent_id": ADOPTER_ID,
                "capsule_id": CAPSULE_ID,
            },
            body={
                "verdict": current_verdict.value,
                "reason": FEEDBACK_REASON.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in FEEDBACK_EVIDENCE
                ],
            },
        ),
        handler,
    )
    assert result.status_code == 201
    assert len(appended_ids) == 1
    return appended_ids[0]


async def _seed_revised_feedback(stack: AdoptionStack) -> FeedbackHistory:
    await arrange_pending_capsule(stack)
    rejected = await reject(stack, key="reject-before-reputation-feedback")
    assert rejected.status_code == 200

    first_event_id = await _record_feedback_source_event(
        stack,
        key="reputation-feedback-1",
        feedback_id=FIRST_FEEDBACK_ID,
        revision=1,
        previous_verdict=None,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )
    stack.clock.advance(timedelta(minutes=1))
    second_event_id = await _record_feedback_source_event(
        stack,
        key="reputation-feedback-2",
        feedback_id=SECOND_FEEDBACK_ID,
        revision=2,
        previous_verdict=FeedbackVerdict.USEFUL,
        current_verdict=FeedbackVerdict.REFUTED,
        alpha_before=3,
        beta_before=2,
        alpha_after=2,
        beta_after=3,
    )
    assert first_event_id < second_event_id
    return FeedbackHistory(
        first_event_id=first_event_id,
        second_event_id=second_event_id,
    )


async def _feedback_sources(
    stack: AdoptionStack,
) -> tuple[tuple[Any, ...], ...]:
    async with stack.database.read_session() as session:
        rows = await session.execute(
            text(
                "SELECT feedback_id, observer_agent_id, capsule_id, revision, "
                "verdict, reason, evidence, created_at "
                "FROM capsule_feedback ORDER BY revision"
            )
        )
        return tuple(tuple(row) for row in rows)


async def _feedback_events(
    stack: AdoptionStack,
) -> tuple[DomainEventRow, ...]:
    async with stack.database.read_session() as session:
        return tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type
                        == CapsuleFeedbackRecordedV1.event_type
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )


async def _projection_state(
    stack: AdoptionStack,
) -> tuple[
    tuple[tuple[Any, ...], ...],
    tuple[tuple[Any, ...], ...],
]:
    async with stack.database.read_session() as session:
        reputation = tuple(
            tuple(row)
            for row in await session.execute(
                text(
                    "SELECT subject_agent_id, observer_agent_id, useful_count, "
                    "refuted_count, harmful_count, alpha, beta, "
                    "projection_event_id FROM agent_reputation "
                    "ORDER BY subject_agent_id, observer_agent_id"
                )
            )
        )
        versions = tuple(
            tuple(row)
            for row in await session.execute(
                text(
                    "SELECT name, reducer_version, last_applied_event_id, "
                    "last_verified_hash, last_verified_at "
                    "FROM projection_versions ORDER BY name"
                )
            )
        )
    return reputation, versions


@pytest.mark.asyncio
async def test_reputation_reducer_rejects_feedback_without_terminal_inbox_event(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)

    with pytest.raises(
        SharingProjectionIntegrityError,
        match="adoption|rejection|authorization",
    ):
        await _record_feedback_source_event(
            stack,
            key="unauthorized-pending-feedback",
            feedback_id=FIRST_FEEDBACK_ID,
            revision=1,
            previous_verdict=None,
            current_verdict=FeedbackVerdict.USEFUL,
            alpha_before=2,
            beta_before=2,
            alpha_after=3,
            beta_after=2,
        )

    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(text("count(*)")).select_from(CapsuleFeedbackRow)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(text("count(*)"))
                .select_from(DomainEventRow)
                .where(
                    DomainEventRow.event_type == CapsuleFeedbackRecordedV1.event_type
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(text("count(*)")).select_from(AgentReputationRow)
            )
            == 0
        )
    assert (await _manager(stack).verify(stack.database)).matches


@pytest.mark.asyncio
async def test_rebuild_follows_event_named_revisions_and_matches_incremental_trust(
    stack: AdoptionStack,
) -> None:
    history = await _seed_revised_feedback(stack)
    events = await _feedback_events(stack)
    assert tuple(event.event_id for event in events) == (
        history.first_event_id,
        history.second_event_id,
    )

    payloads = tuple(
        cast(
            CapsuleFeedbackRecordedV1,
            stack.registry.decode(
                event_type=event.event_type,
                payload=event.payload,
            ),
        )
        for event in events
    )
    assert tuple(payload.feedback_id for payload in payloads) == (
        FIRST_FEEDBACK_ID,
        SECOND_FEEDBACK_ID,
    )
    assert FIRST_FEEDBACK_ID.int > SECOND_FEEDBACK_ID.int
    assert tuple(payload.revision for payload in payloads) == (1, 2)
    assert tuple(
        (payload.previous_verdict, payload.current_verdict) for payload in payloads
    ) == (
        (None, FeedbackVerdict.USEFUL),
        (FeedbackVerdict.USEFUL, FeedbackVerdict.REFUTED),
    )
    assert tuple(
        (
            payload.alpha_before,
            payload.beta_before,
            payload.alpha_after,
            payload.beta_after,
        )
        for payload in payloads
    ) == ((2, 2, 3, 2), (3, 2, 2, 3))

    sources = await _feedback_sources(stack)
    assert tuple(UUID(str(row[0])) for row in sources) == (
        FIRST_FEEDBACK_ID,
        SECOND_FEEDBACK_ID,
    )
    incremental, _ = await _projection_state(stack)
    assert incremental == (
        (
            str(PUBLISHER_ID),
            str(ADOPTER_ID),
            0,
            1,
            0,
            2,
            3,
            history.second_event_id,
        ),
    )

    report = await _manager(stack).verify(stack.database)

    assert report.matches
    async with stack.database.read_session() as session:
        trust = await SharingRepository.observer_trust(
            session=session,
            subject_agent_id=PUBLISHER_ID,
            observer_agent_id=ADOPTER_ID,
        )
        row = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )
    assert row is not None
    assert round(trust, 12) == round(row.alpha / (row.alpha + row.beta), 12)
    assert round(trust, 12) == round(2 / 5, 12)


@pytest.mark.asyncio
async def test_corrupt_projection_is_reported_and_repaired_byte_exactly(
    stack: AdoptionStack,
) -> None:
    history = await _seed_revised_feedback(stack)
    golden_sources = await _feedback_sources(stack)
    golden_projection, _ = await _projection_state(stack)

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE agent_reputation SET useful_count = 1, "
                "refuted_count = 0, harmful_count = 0, alpha = 3, beta = 2 "
                "WHERE subject_agent_id = :subject "
                "AND observer_agent_id = :observer"
            ),
            {
                "subject": str(PUBLISHER_ID),
                "observer": str(ADOPTER_ID),
            },
        )

    with pytest.raises(ProjectionMismatch) as caught:
        await _manager(stack).verify(stack.database)
    assert tuple(
        difference.projection for difference in caught.value.report.differences
    ) == ("agent_reputation",)
    assert caught.value.report.differences[0].differing_keys == (
        json.dumps(
            [str(PUBLISHER_ID), str(ADOPTER_ID)],
            separators=(",", ":"),
        ),
    )

    report = await _manager(stack).repair(stack.database)

    assert report.matches
    repaired_projection, versions = await _projection_state(stack)
    assert repaired_projection == golden_projection
    assert await _feedback_sources(stack) == golden_sources
    reputation_version = next(row for row in versions if row[0] == "agent_reputation")
    assert reputation_version[1:3] == (1, history.second_event_id)
    assert reputation_version[3] is not None
    async with stack.database.read_session() as session:
        repaired_trust = await SharingRepository.observer_trust(
            session=session,
            subject_agent_id=PUBLISHER_ID,
            observer_agent_id=ADOPTER_ID,
        )
    assert round(repaired_trust, 12) == round(2 / 5, 12)


async def _rewrite_second_event(
    stack: AdoptionStack,
    *,
    changes: dict[str, Any],
) -> None:
    events = await _feedback_events(stack)
    assert len(events) == 2
    payload = json.loads(events[1].payload)
    payload.update(changes)
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text(
                "UPDATE domain_events SET payload = :payload WHERE event_id = :event_id"
            ),
            {
                "payload": canonical_json_bytes(payload),
                "event_id": events[1].event_id,
            },
        )


async def _corrupt_feedback_history(
    stack: AdoptionStack,
    corruption: str,
) -> None:
    if corruption == "revision_gap":
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text("DROP TRIGGER capsule_feedback_reject_update")
            )
            await uow.session.execute(
                text(
                    "UPDATE capsule_feedback SET revision = 3 "
                    "WHERE feedback_id = :feedback_id"
                ),
                {"feedback_id": str(SECOND_FEEDBACK_ID)},
            )
        await _rewrite_second_event(stack, changes={"revision": 3})
        return
    if corruption == "source_verdict_mismatch":
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text("DROP TRIGGER capsule_feedback_reject_update")
            )
            await uow.session.execute(
                text(
                    "UPDATE capsule_feedback SET verdict = 'harmful' "
                    "WHERE feedback_id = :feedback_id"
                ),
                {"feedback_id": str(SECOND_FEEDBACK_ID)},
            )
        return
    if corruption == "previous_verdict_mismatch":
        await _rewrite_second_event(
            stack,
            changes={
                "previous_verdict": FeedbackVerdict.HARMFUL.value,
                "alpha_after": 3,
                "beta_after": 2,
            },
        )
        return
    if corruption == "before_counts_mismatch":
        await _rewrite_second_event(
            stack,
            changes={
                "alpha_before": 4,
                "alpha_after": 3,
            },
        )
        return
    if corruption == "invalid_after_counts":
        await _rewrite_second_event(
            stack,
            changes={"alpha_after": 4},
        )
        return
    raise AssertionError(f"Unknown corruption: {corruption}")


@pytest.mark.parametrize("operation", ("verify", "repair"))
@pytest.mark.parametrize(
    "corruption",
    (
        "revision_gap",
        "source_verdict_mismatch",
        "previous_verdict_mismatch",
        "before_counts_mismatch",
        "invalid_after_counts",
    ),
)
@pytest.mark.asyncio
async def test_damaged_feedback_source_or_event_aborts_without_partial_write(
    stack: AdoptionStack,
    operation: str,
    corruption: str,
) -> None:
    await _seed_revised_feedback(stack)
    await _corrupt_feedback_history(stack, corruption)
    before_projection = await _projection_state(stack)
    before_sources = await _feedback_sources(stack)
    before_event_payloads = tuple(
        event.payload for event in await _feedback_events(stack)
    )

    with pytest.raises((SourceIntegrityError, SharingProjectionIntegrityError)):
        await getattr(_manager(stack), operation)(stack.database)

    assert await _projection_state(stack) == before_projection
    assert await _feedback_sources(stack) == before_sources
    assert (
        tuple(event.payload for event in await _feedback_events(stack))
        == before_event_payloads
    )
    async with stack.database.read_session() as session:
        temp_count = await session.scalar(
            text("SELECT count(*) FROM sqlite_temp_master WHERE name LIKE '_rebuild_%'")
        )
    assert temp_count == 0


@pytest.mark.parametrize("operation", ("verify", "repair"))
@pytest.mark.asyncio
async def test_orphan_feedback_source_aborts_without_partial_write(
    stack: AdoptionStack,
    operation: str,
) -> None:
    await arrange_pending_capsule(stack)
    rejected = await reject(
        stack,
        key=f"reject-before-orphan-{operation}",
    )
    assert rejected.status_code == 200
    async with stack.database.transaction() as uow:
        uow.session.add(
            CapsuleFeedbackRow(
                feedback_id=ORPHAN_FEEDBACK_ID,
                observer_agent_id=ADOPTER_ID,
                capsule_id=CAPSULE_ID,
                revision=1,
                verdict=FeedbackVerdict.USEFUL,
                reason=canonical_json_bytes(FEEDBACK_REASON),
                evidence=canonical_json_bytes(FEEDBACK_EVIDENCE),
                created_at=stack.clock.now(),
            )
        )
    before_projection = await _projection_state(stack)
    before_sources = await _feedback_sources(stack)

    async with stack.database.read_session() as session, session.begin():
        with pytest.raises(
            SharingProjectionIntegrityError,
            match="Feedback source.*event|orphan",
        ):
            await AgentReputationProjector(stack.registry).rebuild(
                session,
                f"_isolated_{operation}_",
            )

    with pytest.raises(
        SourceIntegrityError,
        match=rf"feedback:{ORPHAN_FEEDBACK_ID}:",
    ) as caught:
        await getattr(_manager(stack), operation)(stack.database)
    assert caught.value.mismatch_key == f"feedback:{ORPHAN_FEEDBACK_ID}"

    assert await _projection_state(stack) == before_projection
    assert await _feedback_sources(stack) == before_sources
    async with stack.database.read_session() as session:
        temp_count = await session.scalar(
            text("SELECT count(*) FROM sqlite_temp_master WHERE name LIKE '_rebuild_%'")
        )
    assert temp_count == 0
