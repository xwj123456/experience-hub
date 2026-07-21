from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import IntegrityError
from tests.integration.test_capsule_adoption import (
    ADOPTED_EXPERIENCE_ID,
    ADOPTER_ID,
    ADOPTION_ID,
    CAPSULE_ID,
    NOW,
    OTHER_AGENT_ID,
    PUBLISHER_ID,
    SOURCE_EXPERIENCE_ID,
    SOURCE_VERSION_ID,
    TOPIC_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
    error_code,
    request,
    stored_event,
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
from experience_hub.sharing.models import (
    Capsule,
    FeedbackRevision,
    FeedbackVerdict,
    PublishCapsule,
    RecordCapsuleFeedback,
)
from experience_hub.sharing.projector import AgentReputationProjector
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    AgentReputationRow,
    CapsuleFeedbackRow,
    DomainEventRow,
    ExperienceStateRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

UNKNOWN_CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000998")
FEEDBACK_REASON = StructuredReason.from_user_text(
    "  这条经验在独立复现中有效，并保留可核验的证据。  "
)
SECOND_REASON = StructuredReason.from_user_text(
    "新的实验推翻了上一版反馈，应只保留本版的有效贡献。"
)
THIRD_REASON = StructuredReason.from_user_text("风险复盘确认该经验会导致有害结果。")
EVIDENCE_INPUT = (
    TypedEvidence(type="experiment", id="exp:feedback-reproduction"),
    TypedEvidence(type="document", id="doc:feedback-runbook"),
    TypedEvidence(type="experiment", id="exp:feedback-reproduction"),
)
CANONICAL_EVIDENCE = (
    TypedEvidence(type="document", id="doc:feedback-runbook"),
    TypedEvidence(type="experiment", id="exp:feedback-reproduction"),
)
UNAUTHORIZED_FEEDBACK_ID = UUID("00000000-0000-0000-0000-0000000009f1")
ORPHAN_FEEDBACK_ID = UUID("00000000-0000-0000-0000-0000000009f2")
DOUBLE_ANCHOR_FEEDBACK_ID = UUID("00000000-0000-0000-0000-0000000009f3")
DOUBLE_ANCHOR_CAPSULE_ID = UUID("00000000-0000-0000-0000-0000000009f4")


@pytest.fixture(name="stack")
async def _stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-feedback.sqlite3",
    )
    projection_manager = cast(
        ProjectionManager,
        value.database._projection_applier,  # noqa: SLF001
    )
    projection_manager.registry.register(AgentReputationProjector(value.registry))
    try:
        yield value
    finally:
        await value.database.dispose()


async def record_feedback(
    stack: AdoptionStack,
    *,
    key: str,
    observer_agent_id: UUID = ADOPTER_ID,
    capsule_id: UUID = CAPSULE_ID,
    verdict: FeedbackVerdict = FeedbackVerdict.USEFUL,
    reason: StructuredReason | str = FEEDBACK_REASON,
    evidence: tuple[TypedEvidence, ...] = EVIDENCE_INPUT,
) -> CommandResult:
    command = RecordCapsuleFeedback(
        observer_agent_id=observer_agent_id,
        capsule_id=capsule_id,
        verdict=verdict,
        reason=reason,
        evidence=evidence,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.record_capsule_feedback(
            uow=uow,
            command=command,
            command_context=context,
        )

    reason_body = (
        reason.model_dump(mode="json")
        if isinstance(reason, StructuredReason)
        else reason
    )
    return await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.feedback",
            route_template="/v1/agents/{agent_id}/capsules/{capsule_id}:feedback",
            agent_id=observer_agent_id,
            path_parameters={
                "agent_id": observer_agent_id,
                "capsule_id": capsule_id,
            },
            body={
                "verdict": verdict,
                "reason": reason_body,
                "evidence": tuple(item.model_dump(mode="json") for item in evidence),
            },
        ),
        handler,
    )


async def publish_again(
    stack: AdoptionStack,
    *,
    key: str,
) -> tuple[Capsule, UUID]:
    command = PublishCapsule(
        owner_agent_id=PUBLISHER_ID,
        topic_id=TOPIC_ID,
        experience_id=SOURCE_EXPERIENCE_ID,
        version_id=SOURCE_VERSION_ID,
        expires_at=stack.clock.now() + timedelta(days=7),
        parent_adoption_id=None,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.publish",
            route_template="/v1/capsules",
            agent_id=PUBLISHER_ID,
            body={
                "topic_id": TOPIC_ID,
                "experience_id": SOURCE_EXPERIENCE_ID,
                "version_id": SOURCE_VERSION_ID,
                "expires_at": command.expires_at,
            },
        ),
        handler,
    )
    assert result.status_code == 201
    capsule = Capsule.model_validate(json.loads(result.body)["data"], strict=False)
    async with stack.database.read_session() as session:
        item_id = await session.scalar(
            select(InboxItemRow.item_id).where(
                InboxItemRow.recipient_agent_id == ADOPTER_ID,
                InboxItemRow.capsule_id == capsule.capsule_id,
            )
        )
    assert item_id is not None
    return capsule, item_id


async def feedback_counts(
    stack: AdoptionStack,
) -> tuple[int, int, int]:
    async with stack.database.read_session() as session:
        source_count = await session.scalar(
            select(func.count()).select_from(CapsuleFeedbackRow)
        )
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == CapsuleFeedbackRecordedV1.event_type)
        )
        reputation_count = await session.scalar(
            select(func.count()).select_from(AgentReputationRow)
        )
    assert source_count is not None
    assert event_count is not None
    assert reputation_count is not None
    return source_count, event_count, reputation_count


async def seed_coherent_but_unauthorized_reputation(
    stack: AdoptionStack,
) -> None:
    """Bypass the reputation reducer while retaining a coherent event/source row."""
    occurred_at = stack.clock.now()
    payload = CapsuleFeedbackRecordedV1(
        schema_version=1,
        feedback_id=UNAUTHORIZED_FEEDBACK_ID,
        observer_agent_id=ADOPTER_ID,
        capsule_id=CAPSULE_ID,
        publisher_agent_id=PUBLISHER_ID,
        revision=1,
        previous_verdict=None,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        uow.session.add(
            CapsuleFeedbackRow(
                feedback_id=UNAUTHORIZED_FEEDBACK_ID,
                observer_agent_id=ADOPTER_ID,
                capsule_id=CAPSULE_ID,
                revision=1,
                verdict=FeedbackVerdict.USEFUL,
                reason=canonical_json_bytes(FEEDBACK_REASON),
                evidence=canonical_json_bytes(CANONICAL_EVIDENCE),
                created_at=occurred_at,
            )
        )
        await uow.session.flush()
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
        uow.session.add(
            AgentReputationRow(
                subject_agent_id=PUBLISHER_ID,
                observer_agent_id=ADOPTER_ID,
                useful_count=1,
                refuted_count=0,
                harmful_count=0,
                alpha=3,
                beta=2,
                projection_event_id=stored[0].event_id,
            )
        )
        await uow.session.flush()
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes({"data": {"seeded": True}}),
        )

    result = await stack.executor.execute(
        request(
            key="seed-unauthorized-reputation",
            operation_scope="capsule.feedback",
            route_template="/v1/agents/{agent_id}/capsules/{capsule_id}:feedback",
            agent_id=ADOPTER_ID,
            path_parameters={
                "agent_id": ADOPTER_ID,
                "capsule_id": CAPSULE_ID,
            },
            body={"verdict": FeedbackVerdict.USEFUL.value},
        ),
        handler,
    )
    assert result.status_code == 201


async def append_double_anchor_corrupted_feedback_event(
    stack: AdoptionStack,
) -> None:
    """Append a payload-targeted event whose actor and capsule anchors are foreign."""
    payload = CapsuleFeedbackRecordedV1(
        schema_version=1,
        feedback_id=DOUBLE_ANCHOR_FEEDBACK_ID,
        observer_agent_id=ADOPTER_ID,
        capsule_id=DOUBLE_ANCHOR_CAPSULE_ID,
        publisher_agent_id=PUBLISHER_ID,
        revision=1,
        previous_verdict=None,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=3,
        beta_before=2,
        alpha_after=4,
        beta_after=2,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        uow.session.add(
            DomainEventRow(
                aggregate_type="capsule",
                aggregate_id=DOUBLE_ANCHOR_CAPSULE_ID,
                sequence=2,
                event_type=CapsuleFeedbackRecordedV1.event_type,
                payload=canonical_json_bytes(payload),
                actor_agent_id=OTHER_AGENT_ID,
                causation_id=context.receipt_id,
                occurred_at=stack.clock.now(),
            )
        )
        await uow.session.flush()
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes({"data": {"seeded": True}}),
        )

    result = await stack.executor.execute(
        request(
            key="append-double-anchor-corrupted-feedback",
            operation_scope="capsule.feedback",
            route_template="/v1/feedback",
            agent_id=OTHER_AGENT_ID,
            body={"feedback_id": DOUBLE_ANCHOR_FEEDBACK_ID},
        ),
        handler,
    )
    assert result.status_code == 201


@pytest.mark.asyncio
async def test_feedback_hides_unauthorized_capsules_but_allows_rejected_owner(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)

    pending = await record_feedback(stack, key="feedback-pending")
    foreign = await record_feedback(
        stack,
        key="feedback-foreign",
        observer_agent_id=OTHER_AGENT_ID,
    )
    missing = await record_feedback(
        stack,
        key="feedback-missing",
        capsule_id=UNKNOWN_CAPSULE_ID,
    )

    assert pending.status_code == foreign.status_code == missing.status_code == 404
    assert pending.body == foreign.body == missing.body
    assert error_code(pending) == "resource_not_found"
    assert await feedback_counts(stack) == (0, 0, 0)

    assert (await reject(stack, key="reject-before-feedback")).status_code == 200
    authorized = await record_feedback(
        stack,
        key="feedback-after-rejection",
    )

    assert authorized.status_code == 201
    assert await feedback_counts(stack) == (1, 1, 1)


@pytest.mark.asyncio
async def test_first_feedback_retains_canonical_source_emits_strict_event_and_replays(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-before-feedback")).status_code == 200

    first = await record_feedback(
        stack,
        key="feedback-once",
        reason="  这条经验在独立复现中有效，并保留可核验的证据。  ",
    )

    assert first.status_code == 201
    assert not first.replayed
    response = json.loads(first.body)
    assert set(response) == {"data"}
    revision = FeedbackRevision.model_validate(response["data"], strict=False)
    assert (
        revision.observer_agent_id,
        revision.capsule_id,
        revision.revision,
        revision.verdict,
        revision.reason,
        revision.evidence,
        revision.created_at,
    ) == (
        ADOPTER_ID,
        CAPSULE_ID,
        1,
        FeedbackVerdict.USEFUL,
        FEEDBACK_REASON,
        CANONICAL_EVIDENCE,
        NOW,
    )
    assert first.headers == {"location": f"/v1/feedback/{revision.feedback_id}"}

    async with stack.database.read_session() as session:
        source = await session.get(CapsuleFeedbackRow, revision.feedback_id)
        event_row = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleFeedbackRecordedV1.event_type
            )
        )
        reputation = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )
    assert source is not None
    assert event_row is not None
    assert reputation is not None
    assert (
        source.observer_agent_id,
        source.capsule_id,
        source.revision,
        source.verdict,
        source.reason,
        source.evidence,
        source.created_at,
    ) == (
        ADOPTER_ID,
        CAPSULE_ID,
        1,
        FeedbackVerdict.USEFUL,
        canonical_json_bytes(FEEDBACK_REASON),
        canonical_json_bytes(CANONICAL_EVIDENCE),
        NOW,
    )

    event = stored_event(stack, event_row)
    payload = event.payload
    assert isinstance(payload, CapsuleFeedbackRecordedV1)
    assert payload.model_dump(mode="json") == {
        "schema_version": 1,
        "feedback_id": str(revision.feedback_id),
        "observer_agent_id": str(ADOPTER_ID),
        "capsule_id": str(CAPSULE_ID),
        "publisher_agent_id": str(PUBLISHER_ID),
        "revision": 1,
        "previous_verdict": None,
        "current_verdict": "useful",
        "alpha_before": 2,
        "beta_before": 2,
        "alpha_after": 3,
        "beta_after": 2,
    }
    assert (
        event.aggregate_type,
        event.aggregate_id,
        event.sequence,
        event.actor_agent_id,
        event.occurred_at,
    ) == ("capsule", CAPSULE_ID, 2, ADOPTER_ID, NOW)
    assert set(json.loads(event_row.payload)) == {
        "schema_version",
        "feedback_id",
        "observer_agent_id",
        "capsule_id",
        "publisher_agent_id",
        "revision",
        "previous_verdict",
        "current_verdict",
        "alpha_before",
        "beta_before",
        "alpha_after",
        "beta_after",
    }
    assert b'"reason"' not in event_row.payload
    assert b'"evidence"' not in event_row.payload
    assert (
        reputation.subject_agent_id,
        reputation.observer_agent_id,
        reputation.useful_count,
        reputation.refuted_count,
        reputation.harmful_count,
        reputation.alpha,
        reputation.beta,
        reputation.projection_event_id,
    ) == (
        PUBLISHER_ID,
        ADOPTER_ID,
        1,
        0,
        0,
        3,
        2,
        event_row.event_id,
    )

    replay = await record_feedback(
        stack,
        key="feedback-once",
        reason="  这条经验在独立复现中有效，并保留可核验的证据。  ",
    )

    assert replay.replayed
    assert (
        replay.status_code,
        replay.body,
        replay.headers,
    ) == (first.status_code, first.body, first.headers)
    assert await feedback_counts(stack) == (1, 1, 1)


@pytest.mark.parametrize(
    ("verdict", "counts", "alpha", "beta"),
    (
        (FeedbackVerdict.USEFUL, (1, 0, 0), 3, 2),
        (FeedbackVerdict.REFUTED, (0, 1, 0), 2, 3),
        (FeedbackVerdict.HARMFUL, (0, 0, 1), 2, 3),
    ),
)
@pytest.mark.asyncio
async def test_first_feedback_supports_all_three_verdicts(
    stack: AdoptionStack,
    verdict: FeedbackVerdict,
    counts: tuple[int, int, int],
    alpha: int,
    beta: int,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key=f"reject-before-{verdict}")).status_code == 200

    result = await record_feedback(
        stack,
        key=f"first-{verdict}",
        verdict=verdict,
    )

    assert result.status_code == 201
    async with stack.database.read_session() as session:
        source = await session.scalar(select(CapsuleFeedbackRow))
        reputation = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )
    assert source is not None
    assert reputation is not None
    assert source.revision == 1
    assert source.verdict is verdict
    assert (
        reputation.useful_count,
        reputation.refuted_count,
        reputation.harmful_count,
    ) == counts
    assert (reputation.alpha, reputation.beta) == (alpha, beta)


@pytest.mark.asyncio
async def test_latest_revision_replaces_prior_contribution_and_is_immutable(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-revisions")).status_code == 200

    first = await record_feedback(
        stack,
        key="revision-one",
        verdict=FeedbackVerdict.USEFUL,
        reason=FEEDBACK_REASON,
        evidence=EVIDENCE_INPUT,
    )
    stack.clock.advance(timedelta(seconds=1))
    second = await record_feedback(
        stack,
        key="revision-two",
        verdict=FeedbackVerdict.REFUTED,
        reason=SECOND_REASON,
        evidence=(TypedEvidence(type="experiment", id="exp:counterexample"),),
    )
    stack.clock.advance(timedelta(seconds=1))
    third = await record_feedback(
        stack,
        key="revision-three",
        verdict=FeedbackVerdict.HARMFUL,
        reason=THIRD_REASON,
        evidence=(TypedEvidence(type="incident", id="incident:feedback"),),
    )

    assert first.status_code == second.status_code == third.status_code == 201
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(CapsuleFeedbackRow).order_by(CapsuleFeedbackRow.revision)
                )
            ).all()
        )
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type
                        == CapsuleFeedbackRecordedV1.event_type
                    )
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
        reputation = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )

    assert [row.revision for row in rows] == [1, 2, 3]
    assert [row.verdict for row in rows] == [
        FeedbackVerdict.USEFUL,
        FeedbackVerdict.REFUTED,
        FeedbackVerdict.HARMFUL,
    ]
    assert rows[0].reason == canonical_json_bytes(FEEDBACK_REASON)
    assert rows[0].evidence == canonical_json_bytes(CANONICAL_EVIDENCE)
    assert rows[1].reason == canonical_json_bytes(SECOND_REASON)
    assert rows[2].reason == canonical_json_bytes(THIRD_REASON)
    assert reputation is not None
    assert (
        reputation.useful_count,
        reputation.refuted_count,
        reputation.harmful_count,
        reputation.alpha,
        reputation.beta,
    ) == (0, 0, 1, 2, 3)

    payloads = [stored_event(stack, row).payload for row in event_rows]
    assert all(isinstance(payload, CapsuleFeedbackRecordedV1) for payload in payloads)
    typed_payloads = cast(list[CapsuleFeedbackRecordedV1], payloads)
    assert [row.sequence for row in event_rows] == [2, 3, 4]
    assert [
        (
            payload.revision,
            payload.previous_verdict,
            payload.current_verdict,
            payload.alpha_before,
            payload.beta_before,
            payload.alpha_after,
            payload.beta_after,
        )
        for payload in typed_payloads
    ] == [
        (1, None, FeedbackVerdict.USEFUL, 2, 2, 3, 2),
        (
            2,
            FeedbackVerdict.USEFUL,
            FeedbackVerdict.REFUTED,
            3,
            2,
            2,
            3,
        ),
        (
            3,
            FeedbackVerdict.REFUTED,
            FeedbackVerdict.HARMFUL,
            2,
            3,
            2,
            3,
        ),
    ]

    with pytest.raises(IntegrityError, match="immutable"):
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                update(CapsuleFeedbackRow)
                .where(CapsuleFeedbackRow.feedback_id == rows[0].feedback_id)
                .values(reason=canonical_json_bytes(SECOND_REASON))
            )
    with pytest.raises(IntegrityError, match="immutable"):
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                delete(CapsuleFeedbackRow).where(
                    CapsuleFeedbackRow.feedback_id == rows[0].feedback_id
                )
            )

    async with stack.database.read_session() as session:
        retained = await session.get(CapsuleFeedbackRow, rows[0].feedback_id)
    assert retained is not None
    assert retained.reason == canonical_json_bytes(FEEDBACK_REASON)
    assert retained.evidence == canonical_json_bytes(CANONICAL_EVIDENCE)


@pytest.mark.parametrize(
    ("reason", "evidence", "expected_code"),
    (
        ("   ", (), "invalid_reason"),
        (
            FEEDBACK_REASON,
            (TypedEvidence.model_construct(type="", id=""),),
            "invalid_feedback",
        ),
    ),
)
@pytest.mark.asyncio
async def test_feedback_revalidates_reason_and_typed_evidence_without_mutation(
    stack: AdoptionStack,
    reason: StructuredReason | str,
    evidence: tuple[TypedEvidence, ...],
    expected_code: str,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-invalid")).status_code == 200

    result = await record_feedback(
        stack,
        key=f"invalid-{expected_code}",
        reason=reason,
        evidence=evidence,
    )

    assert result.status_code == 422
    assert error_code(result) == expected_code
    assert await feedback_counts(stack) == (0, 0, 0)


@pytest.mark.asyncio
async def test_feedback_rejects_capsule_clock_regression_without_mutation(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-old-feedback")).status_code == 200
    stack.clock.advance(timedelta(microseconds=-1))

    result = await record_feedback(stack, key="feedback-before-capsule")

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    assert await feedback_counts(stack) == (0, 0, 0)


@pytest.mark.parametrize("terminal", ("adopted", "rejected"))
@pytest.mark.asyncio
async def test_feedback_rejects_time_between_publication_and_terminal_inbox_state(
    stack: AdoptionStack,
    terminal: str,
) -> None:
    await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(minutes=10))
    if terminal == "adopted":
        assert (await adopt(stack, key="late-terminal-adopt")).status_code == 200
    else:
        assert (await reject(stack, key="late-terminal-reject")).status_code == 200
    stack.clock.advance(timedelta(minutes=-5))

    result = await record_feedback(
        stack,
        key=f"feedback-before-{terminal}-terminal",
    )

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    assert await feedback_counts(stack) == (0, 0, 0)


@pytest.mark.asyncio
async def test_feedback_rejects_regression_against_revision_stream_and_reputation_head(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-first-capsule")).status_code == 200
    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-second-for-clock",
    )
    assert (
        await reject(
            stack,
            key="reject-second-capsule",
            item_id=second_item_id,
        )
    ).status_code == 200

    stack.clock.advance(timedelta(seconds=2))
    first = await record_feedback(
        stack,
        key="future-feedback",
        verdict=FeedbackVerdict.USEFUL,
    )
    assert first.status_code == 201
    stack.clock.advance(timedelta(seconds=-1))

    old_revision = await record_feedback(
        stack,
        key="old-revision",
        verdict=FeedbackVerdict.REFUTED,
    )
    old_reputation = await record_feedback(
        stack,
        key="old-reputation",
        capsule_id=second_capsule.capsule_id,
        verdict=FeedbackVerdict.HARMFUL,
    )

    assert old_revision.status_code == old_reputation.status_code == 409
    assert (
        error_code(old_revision) == error_code(old_reputation) == ("clock_regression")
    )
    assert await feedback_counts(stack) == (1, 1, 1)
    async with stack.database.read_session() as session:
        reputation = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )
    assert reputation is not None
    assert (
        reputation.useful_count,
        reputation.refuted_count,
        reputation.harmful_count,
        reputation.alpha,
        reputation.beta,
    ) == (1, 0, 0, 3, 2)


@pytest.mark.parametrize("corruption", ("deleted", "stale"))
@pytest.mark.asyncio
async def test_feedback_refuses_missing_or_stale_cross_capsule_reputation(
    stack: AdoptionStack,
    corruption: str,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-corruption")).status_code == 200
    second_capsule, second_item_id = await publish_again(
        stack,
        key=f"publish-second-{corruption}",
    )
    assert (
        await reject(
            stack,
            key=f"reject-second-{corruption}",
            item_id=second_item_id,
        )
    ).status_code == 200
    assert (
        await record_feedback(
            stack,
            key=f"first-feedback-{corruption}",
            verdict=FeedbackVerdict.USEFUL,
        )
    ).status_code == 201
    async with stack.database.transaction() as uow:
        if corruption == "deleted":
            await uow.session.execute(delete(AgentReputationRow))
        else:
            publication_event_id = await uow.session.scalar(
                select(DomainEventRow.event_id)
                .where(
                    DomainEventRow.aggregate_type == "capsule",
                    DomainEventRow.aggregate_id == CAPSULE_ID,
                    DomainEventRow.event_type == "capsule.published",
                )
            )
            assert publication_event_id is not None
            await uow.session.execute(
                update(AgentReputationRow).values(
                    useful_count=0,
                    refuted_count=0,
                    harmful_count=0,
                    alpha=2,
                    beta=2,
                    projection_event_id=publication_event_id,
                )
            )
    before = await feedback_counts(stack)

    with pytest.raises(SourceIntegrityError, match="reputation"):
        await record_feedback(
            stack,
            key=f"second-feedback-{corruption}",
            capsule_id=second_capsule.capsule_id,
            verdict=FeedbackVerdict.HARMFUL,
        )

    assert await feedback_counts(stack) == before


@pytest.mark.asyncio
async def test_feedback_refuses_reputation_with_swapped_negative_verdict_counts(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-count-swap")).status_code == 200
    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-second-count-swap",
    )
    assert (
        await reject(
            stack,
            key="reject-second-count-swap",
            item_id=second_item_id,
        )
    ).status_code == 200
    assert (
        await record_feedback(
            stack,
            key="harmful-before-count-swap",
            verdict=FeedbackVerdict.HARMFUL,
        )
    ).status_code == 201
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(AgentReputationRow).values(
                refuted_count=1,
                harmful_count=0,
            )
        )
    before = await feedback_counts(stack)

    with pytest.raises(SourceIntegrityError, match="reputation"):
        await record_feedback(
            stack,
            key="feedback-after-count-swap",
            capsule_id=second_capsule.capsule_id,
            verdict=FeedbackVerdict.USEFUL,
        )

    assert await feedback_counts(stack) == before
    async with stack.database.read_session() as session:
        corrupted = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )
    assert corrupted is not None
    assert (
        corrupted.useful_count,
        corrupted.refuted_count,
        corrupted.harmful_count,
        corrupted.alpha,
        corrupted.beta,
    ) == (0, 1, 0, 2, 3)


@pytest.mark.asyncio
async def test_future_adoption_refuses_reputation_with_missing_feedback_source(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-source-loss")).status_code == 200
    assert (
        await record_feedback(
            stack,
            key="useful-before-source-loss",
            verdict=FeedbackVerdict.USEFUL,
        )
    ).status_code == 201
    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-after-source-loss",
    )
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER capsule_feedback_reject_delete")
        )
        await uow.session.execute(delete(CapsuleFeedbackRow))
    async with stack.database.read_session() as session:
        before_adoptions = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
    assert before_adoptions == 0

    with pytest.raises(SourceIntegrityError, match="Feedback source"):
        await adopt(
            stack,
            key="adopt-after-source-loss",
            item_id=second_item_id,
        )

    async with stack.database.read_session() as session:
        after_adoptions = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
        second_adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.capsule_id == second_capsule.capsule_id,
            )
        )
    assert after_adoptions == before_adoptions
    assert second_adoption is None


@pytest.mark.asyncio
async def test_future_adoption_refuses_unterminated_feedback_authorization(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "unauthorized-reputation.sqlite3",
    )
    try:
        await arrange_pending_capsule(stack)
        await seed_coherent_but_unauthorized_reputation(stack)
        second_capsule, second_item_id = await publish_again(
            stack,
            key="publish-after-unauthorized-reputation",
        )

        with pytest.raises(SourceIntegrityError, match="Feedback|reputation"):
            await adopt(
                stack,
                key="adopt-after-unauthorized-reputation",
                item_id=second_item_id,
            )

        async with stack.database.read_session() as session:
            second_adoption = await session.scalar(
                select(AdoptionRecordRow).where(
                    AdoptionRecordRow.capsule_id == second_capsule.capsule_id,
                )
            )
        assert second_adoption is None
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_future_adoption_refuses_orphan_feedback_source(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-orphan-source")).status_code == 200
    async with stack.database.transaction() as uow:
        uow.session.add(
            CapsuleFeedbackRow(
                feedback_id=ORPHAN_FEEDBACK_ID,
                observer_agent_id=ADOPTER_ID,
                capsule_id=CAPSULE_ID,
                revision=1,
                verdict=FeedbackVerdict.USEFUL,
                reason=canonical_json_bytes(FEEDBACK_REASON),
                evidence=canonical_json_bytes(CANONICAL_EVIDENCE),
                created_at=stack.clock.now(),
            )
        )
    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-after-orphan-feedback-source",
    )

    with pytest.raises(SourceIntegrityError, match="Feedback|reputation"):
        await adopt(
            stack,
            key="adopt-after-orphan-feedback-source",
            item_id=second_item_id,
        )

    async with stack.database.read_session() as session:
        second_adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.capsule_id == second_capsule.capsule_id,
            )
        )
    assert second_adoption is None


@pytest.mark.asyncio
async def test_strict_trust_refuses_payload_targeted_double_anchor_damage(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-anchor-damage")).status_code == 200
    assert (
        await record_feedback(
            stack,
            key="useful-before-anchor-damage",
            verdict=FeedbackVerdict.USEFUL,
        )
    ).status_code == 201
    await append_double_anchor_corrupted_feedback_event(stack)
    repository = SharingRepository(event_registry=stack.registry)

    async with stack.database.read_session() as session:
        with pytest.raises(
            SourceIntegrityError,
            match="Feedback event history",
        ):
            await repository.strict_observer_trust(
                session=session,
                subject_agent_id=PUBLISHER_ID,
                observer_agent_id=ADOPTER_ID,
            )


@pytest.mark.asyncio
async def test_later_feedback_changes_future_trust_without_rewriting_earlier_adoption(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-at-prior-trust")).status_code == 200

    async with stack.database.read_session() as session:
        original_adoption = await session.get(AdoptionRecordRow, ADOPTION_ID)
        original_state = await session.get(
            ExperienceStateRow,
            ADOPTED_EXPERIENCE_ID,
        )
    assert original_adoption is not None
    assert original_state is not None
    assert original_adoption.captured_trust == pytest.approx(0.5)
    assert original_state.source_trust == pytest.approx(0.5)
    assert original_state.confidence == pytest.approx(0.4)

    feedback = await record_feedback(
        stack,
        key="useful-after-first-adoption",
        verdict=FeedbackVerdict.USEFUL,
    )
    assert feedback.status_code == 201

    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-after-feedback",
    )
    second_adoption = await adopt(
        stack,
        key="adopt-after-feedback",
        item_id=second_item_id,
    )
    assert second_adoption.status_code == 200

    async with stack.database.read_session() as session:
        retained_original = await session.get(AdoptionRecordRow, ADOPTION_ID)
        retained_state = await session.get(
            ExperienceStateRow,
            ADOPTED_EXPERIENCE_ID,
        )
        later_adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.adopter_agent_id == ADOPTER_ID,
                AdoptionRecordRow.capsule_id == second_capsule.capsule_id,
            )
        )
        reputation = await session.get(
            AgentReputationRow,
            (PUBLISHER_ID, ADOPTER_ID),
        )

    assert retained_original is not None
    assert retained_state is not None
    assert later_adoption is not None
    assert reputation is not None
    assert retained_original.captured_trust == pytest.approx(0.5)
    assert retained_state.source_trust == pytest.approx(0.5)
    assert retained_state.confidence == pytest.approx(0.4)
    assert later_adoption.captured_trust == pytest.approx(0.6)
    assert (reputation.alpha, reputation.beta) == (3, 2)
