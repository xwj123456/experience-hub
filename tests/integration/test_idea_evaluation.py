from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, update
from tests.integration.test_inspiration_run import (
    NOW,
    OWNER_ID,
    Stack,
    build_stack,
    command,
    request,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    PendingEvent,
    StructuredReason,
)
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    VersionContent,
)
from experience_hub.inspiration.events import (
    InspirationIdeaArchivedV1,
    InspirationIdeaEvaluatedV1,
    register_inspiration_events,
)
from experience_hub.inspiration.lifecycle import IdeaLifecycleService
from experience_hub.inspiration.models import (
    EvaluationEvidenceReference,
    EvaluationVerdict,
    ExperienceVersionEvidenceReference,
    IdeaEvaluation,
    IdeaOwnerDecision,
    MechanismMaturity,
    SnapshotEvidenceReference,
)
from experience_hub.inspiration.projector import (
    InspirationProjectionIntegrityError,
)
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
)
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandResult,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    InspirationIdeaRow,
    InspirationRunStateRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

OTHER_AGENT_ID = UUID("00000000-0000-0000-0000-000000000102")


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


@dataclass(frozen=True, slots=True)
class SeededIdea:
    idea_id: UUID
    run_id: UUID
    snapshot_item_id: UUID
    stable_evidence_key: str
    snapshot_hash: str
    mechanism_cluster_id: str

    @property
    def snapshot_evidence(self) -> SnapshotEvidenceReference:
        return SnapshotEvidenceReference(
            id=self.snapshot_item_id,
            stable_evidence_key=self.stable_evidence_key,
        )


@dataclass(slots=True)
class EvaluationStack:
    run: Stack
    registry: EventRegistry
    service: IdeaLifecycleService
    executor: CommandExecutor


@pytest.fixture
async def evaluation_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[EvaluationStack]:
    run = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "idea-evaluation.sqlite3",
    )
    async with run.database.transaction() as uow:
        uow.session.add(
            AgentRow(
                agent_id=OTHER_AGENT_ID,
                name="Other evaluator",
                created_at=NOW,
            )
        )

    registry = EventRegistry()
    register_inspiration_events(registry)
    receipts = cast(
        ReceiptStore,
        run.executor._receipt_store,  # noqa: SLF001 - shared test stack wiring
    )
    value = EvaluationStack(
        run=run,
        registry=registry,
        service=IdeaLifecycleService(
            clock=run.clock,
            receipt_store=receipts,
            repository=InspirationRepository(registry),
        ),
        executor=CommandExecutor(
            database=run.database,
            receipt_store=receipts,
            clock=run.clock,
        ),
    )
    try:
        yield value
    finally:
        await run.database.dispose()


async def _seed_idea(
    stack: EvaluationStack,
    *,
    key: str,
    snapshot_item: int,
    content_marker: int,
) -> SeededIdea:
    stack.run.snapshot_builder.item_ids.append(uid(snapshot_item))
    stack.run.snapshot_builder.content_hashes.append(f"{content_marker:064x}")
    result = await stack.run.executor.execute(
        request=request(key=key),
        run=command(),
    )
    assert result.status_code == 201
    run_id = UUID(json.loads(result.body)["data"]["run_id"])

    async with stack.run.database.read_session() as session:
        idea = await session.scalar(
            select(InspirationIdeaRow).where(InspirationIdeaRow.run_id == run_id)
        )
        item = await session.scalar(
            select(InspirationSnapshotItemRow).where(
                InspirationSnapshotItemRow.run_id == run_id
            )
        )
        occurrence = await session.scalar(
            select(IdeaOccurrenceRow).where(IdeaOccurrenceRow.run_id == run_id)
        )
        state = None if idea is None else await session.get(IdeaStateRow, idea.idea_id)
    assert idea is not None
    assert item is not None
    assert occurrence is not None
    assert state is not None
    return SeededIdea(
        idea_id=idea.idea_id,
        run_id=run_id,
        snapshot_item_id=item.snapshot_item_id,
        stable_evidence_key=item.stable_evidence_key,
        snapshot_hash=occurrence.snapshot_hash,
        mechanism_cluster_id=state.mechanism_cluster_id,
    )


def _evaluation_request(
    evaluation: IdeaEvaluation,
    *,
    key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{evaluation.evaluator_agent_id}",
        operation_scope="inspiration.idea.evaluate",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/ideas/{idea_id}:evaluate",
        path_parameters={
            "agent_id": evaluation.evaluator_agent_id,
            "idea_id": evaluation.idea_id,
        },
        body={
            "evaluated_at": evaluation.evaluated_at,
            "evidence": tuple(
                item.model_dump(mode="json") for item in evaluation.evidence
            ),
            "reason": (
                None
                if evaluation.reason is None
                else evaluation.reason.model_dump(mode="json")
            ),
            "verdict": evaluation.verdict.value,
        },
    )


async def _evaluate(
    stack: EvaluationStack,
    evaluation: IdeaEvaluation,
    *,
    key: str,
) -> CommandResult:
    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.evaluate(
            uow=uow,
            evaluation=evaluation,
            command_context=command_context,
        )

    return await stack.executor.execute(
        _evaluation_request(evaluation, key=key),
        handler,
    )


def _evaluation(
    idea: SeededIdea,
    *,
    verdict: EvaluationVerdict,
    evaluated_at_delta: timedelta,
    evaluator_agent_id: UUID = OWNER_ID,
    evidence: tuple[EvaluationEvidenceReference, ...] | None = None,
    reason: StructuredReason | None = None,
) -> IdeaEvaluation:
    return IdeaEvaluation(
        evaluator_agent_id=evaluator_agent_id,
        idea_id=idea.idea_id,
        verdict=verdict,
        reason=reason,
        evidence=evidence or (idea.snapshot_evidence,),
        evaluated_at=NOW + evaluated_at_delta,
    )


def _assert_success(
    result: CommandResult,
    *,
    idea_id: UUID,
    owner_decision: IdeaOwnerDecision,
    maturity: MechanismMaturity,
    revision: int,
) -> None:
    assert result.status_code == 200
    assert result.body == canonical_json_bytes(
        {
            "data": {
                "idea_id": idea_id,
                "maturity": maturity,
                "owner_decision": owner_decision,
                "revision": revision,
            }
        }
    )


def _error(result: CommandResult) -> dict[str, object]:
    assert canonical_json_bytes(json.loads(result.body)) == result.body
    decoded = json.loads(result.body)
    assert set(decoded) == {"error"}
    return cast(dict[str, object], decoded["error"])


async def _evaluated_events(
    stack: EvaluationStack,
    *,
    idea_id: UUID | None = None,
) -> tuple[tuple[DomainEventRow, InspirationIdeaEvaluatedV1], ...]:
    async with stack.run.database.read_session() as session:
        statement = select(DomainEventRow).where(
            DomainEventRow.event_type == InspirationIdeaEvaluatedV1.event_type
        )
        if idea_id is not None:
            statement = statement.where(DomainEventRow.aggregate_id == idea_id)
        rows = (
            await session.scalars(statement.order_by(DomainEventRow.event_id))
        ).all()
    decoded: list[tuple[DomainEventRow, InspirationIdeaEvaluatedV1]] = []
    for row in rows:
        payload = stack.registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
        assert isinstance(payload, InspirationIdeaEvaluatedV1)
        decoded.append((row, payload))
    return tuple(decoded)


def _projected_evaluation(
    evaluation: IdeaEvaluation,
    *,
    revision: int,
) -> dict[str, object]:
    return {
        "evaluated_at": evaluation.evaluated_at,
        "evaluator_agent_id": evaluation.evaluator_agent_id,
        "evidence": tuple(item.model_dump(mode="json") for item in evaluation.evidence),
        "reason": (
            None
            if evaluation.reason is None
            else evaluation.reason.model_dump(mode="json")
        ),
        "revision": revision,
        "verdict": evaluation.verdict,
    }


async def _run_projection(
    stack: EvaluationStack,
    *,
    run_id: UUID,
) -> dict[str, object]:
    async with stack.run.database.read_session() as session:
        row = await session.get(InspirationRunStateRow, run_id)
        assert row is not None
        return {
            column.name: getattr(row, column.name)
            for column in InspirationRunStateRow.__table__.columns
        }


async def _assert_event_matches_projections(
    stack: EvaluationStack,
    *,
    row: DomainEventRow,
    payload: InspirationIdeaEvaluatedV1,
    expected_evaluations: tuple[dict[str, object], ...],
    expected_sequence: int | None = None,
) -> None:
    async with stack.run.database.read_session() as session:
        idea = await session.get(IdeaStateRow, payload.idea_id)
        cluster = await session.get(
            MechanismIncubationRow,
            payload.mechanism_cluster_id,
        )
    assert idea is not None
    assert cluster is not None
    assert (
        row.aggregate_type,
        row.aggregate_id,
        row.sequence,
        row.actor_agent_id,
        row.occurred_at,
    ) == (
        "idea",
        payload.idea_id,
        payload.revision + 1 if expected_sequence is None else expected_sequence,
        payload.evaluator_agent_id,
        payload.last_signal_at_after,
    )
    assert idea.owner_decision == payload.owner_decision_after.value
    assert idea.mechanism_cluster_id == payload.mechanism_cluster_id
    assert idea.evaluations == canonical_json_bytes(expected_evaluations)
    assert idea.last_signal_at == payload.last_signal_at_after
    assert idea.projection_event_id == row.event_id
    assert (
        cluster.supported_count,
        cluster.refuted_count,
        cluster.maturity,
        cluster.candidate_since,
        cluster.last_signal_at,
        cluster.projection_event_id,
    ) == (
        payload.supported_count_after,
        payload.refuted_count_after,
        payload.maturity_after.value,
        payload.candidate_since_after,
        payload.last_signal_at_after,
        row.event_id,
    )


async def _seed_experience_version(
    stack: EvaluationStack,
    *,
    owner_agent_id: UUID,
    experience_id: UUID,
    version_id: UUID,
    created_at: datetime = NOW,
) -> None:
    content = VersionContent(
        body=f"Evidence body for {version_id}.",
        summary=f"Evidence {version_id}",
        mechanism="Observed acknowledgements release bounded capacity.",
        tags=("evaluation",),
        applicability=("bounded queue",),
        evidence=(),
        falsifiers=("Capacity stays blocked after acknowledgement.",),
    )
    encoded = encode_version_content(
        kind=ExperienceKind.SEMANTIC,
        content=content,
    )
    async with stack.run.database.transaction() as uow:
        if await uow.session.get(AgentRow, owner_agent_id) is None:
            agent = AgentRow(
                agent_id=owner_agent_id,
                name=f"Agent {owner_agent_id}",
                created_at=created_at,
            )
            uow.session.add(agent)
            await uow.session.flush((agent,))
        identity = ExperienceRow(
            experience_id=experience_id,
            owner_agent_id=owner_agent_id,
            kind=ExperienceKind.SEMANTIC,
            origin=ExperienceOrigin.LOCAL,
            created_at=created_at,
        )
        uow.session.add(identity)
        await uow.session.flush((identity,))
        version = ExperienceVersionRow(
            version_id=version_id,
            experience_id=experience_id,
            version_number=1,
            summary=content.summary,
            mechanism=content.mechanism,
            tags=canonical_json_bytes(content.tags),
            applicability=canonical_json_bytes(content.applicability),
            evidence=canonical_json_bytes(content.evidence),
            falsifiers=canonical_json_bytes(content.falsifiers),
            content_hash=encoded.content_hash,
            supersedes_version_id=None,
            created_at=created_at,
        )
        uow.session.add(version)
        await uow.session.flush((version,))
        uow.session.add(
            ExperiencePayloadRow(
                version_id=version_id,
                codec=PayloadCodec.PLAIN,
                payload=encoded.payload,
                payload_hash=encoded.payload_hash,
            )
        )


async def _archive_idea(
    stack: EvaluationStack,
    *,
    idea_id: UUID,
) -> None:
    reason = StructuredReason.from_user_text("Retain this idea outside the active set.")

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=idea_id,
                    event_type=InspirationIdeaArchivedV1.event_type,
                    payload=InspirationIdeaArchivedV1(
                        schema_version=1,
                        idea_id=idea_id,
                        owner_agent_id=OWNER_ID,
                        reason=reason,
                        owner_decision_before=IdeaOwnerDecision.ACTIVE,
                        owner_decision_after=IdeaOwnerDecision.ARCHIVED,
                        cycle_id=None,
                    ),
                    actor_agent_id=OWNER_ID,
                    occurred_at=NOW,
                ),
            ),
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": {"idea_id": idea_id}}),
        )

    result = await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_ID}",
            operation_scope="inspiration.idea.archive",
            idempotency_key=f"archive:{idea_id}",
            method="POST",
            route_template="/v1/agents/{agent_id}/ideas/{idea_id}:archive",
            path_parameters={
                "agent_id": OWNER_ID,
                "idea_id": idea_id,
            },
            body={"reason": reason.model_dump(mode="json")},
        ),
        handler,
    )
    assert result.status_code == 200


async def _arrange_owner_decision(
    stack: EvaluationStack,
    *,
    idea_id: UUID,
    decision: IdeaOwnerDecision,
) -> None:
    resulting_experience_id: UUID | None = None
    resulting_version_id: UUID | None = None
    if decision is IdeaOwnerDecision.ADOPTED:
        resulting_experience_id = uid(960)
        resulting_version_id = uid(961)
        await _seed_experience_version(
            stack,
            owner_agent_id=OWNER_ID,
            experience_id=resulting_experience_id,
            version_id=resulting_version_id,
        )
    async with stack.run.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == idea_id)
            .values(
                owner_decision=decision.value,
                resulting_experience_id=resulting_experience_id,
                resulting_version_id=resulting_version_id,
            )
        )
        assert changed.rowcount == 1


@pytest.mark.asyncio
async def test_revisions_replace_effective_verdict_and_match_all_projections(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="evaluation-first-occurrence",
        snapshot_item=701,
        content_marker=1,
    )
    await _seed_idea(
        evaluation_stack,
        key="evaluation-second-occurrence",
        snapshot_item=702,
        content_marker=2,
    )
    owned_version_id = uid(811)
    await _seed_experience_version(
        evaluation_stack,
        owner_agent_id=OWNER_ID,
        experience_id=uid(810),
        version_id=owned_version_id,
    )
    run_before = await _run_projection(
        evaluation_stack,
        run_id=idea.run_id,
    )
    reason = StructuredReason.from_user_text(
        "  Replayed measurements support the mechanism.  "
    )
    first = _evaluation(
        idea,
        verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at_delta=timedelta(minutes=1),
        evidence=(
            idea.snapshot_evidence,
            ExperienceVersionEvidenceReference(id=owned_version_id),
        ),
        reason=reason,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))

    first_result = await _evaluate(
        evaluation_stack,
        first,
        key="evaluation-revision-1",
    )

    _assert_success(
        first_result,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ACTIVE,
        maturity=MechanismMaturity.CANDIDATE,
        revision=1,
    )
    first_row, first_payload = (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
    )[0]
    assert (
        first_payload.revision,
        first_payload.previous_verdict,
        first_payload.current_verdict,
        first_payload.evidence,
        first_payload.reason,
        first_payload.owner_decision_before,
        first_payload.owner_decision_after,
        first_payload.supported_count_before,
        first_payload.supported_count_after,
        first_payload.refuted_count_before,
        first_payload.refuted_count_after,
        first_payload.maturity_before,
        first_payload.maturity_after,
        first_payload.candidate_since_before,
        first_payload.candidate_since_after,
        first_payload.last_signal_at_before,
        first_payload.last_signal_at_after,
    ) == (
        1,
        None,
        EvaluationVerdict.SUPPORTED,
        first.evidence,
        reason,
        IdeaOwnerDecision.ACTIVE,
        IdeaOwnerDecision.ACTIVE,
        0,
        1,
        0,
        0,
        MechanismMaturity.INCUBATING,
        MechanismMaturity.CANDIDATE,
        None,
        first.evaluated_at,
        NOW,
        first.evaluated_at,
    )
    projected = (_projected_evaluation(first, revision=1),)
    await _assert_event_matches_projections(
        evaluation_stack,
        row=first_row,
        payload=first_payload,
        expected_evaluations=projected,
    )

    second = _evaluation(
        idea,
        verdict=EvaluationVerdict.REFUTED,
        evaluated_at_delta=timedelta(minutes=2),
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    second_result = await _evaluate(
        evaluation_stack,
        second,
        key="evaluation-revision-2",
    )
    _assert_success(
        second_result,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ACTIVE,
        maturity=MechanismMaturity.INCUBATING,
        revision=2,
    )
    second_row, second_payload = (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
    )[1]
    assert (
        second_payload.revision,
        second_payload.previous_verdict,
        second_payload.current_verdict,
        second_payload.supported_count_before,
        second_payload.supported_count_after,
        second_payload.refuted_count_before,
        second_payload.refuted_count_after,
        second_payload.maturity_before,
        second_payload.maturity_after,
        second_payload.candidate_since_before,
        second_payload.candidate_since_after,
        second_payload.last_signal_at_before,
        second_payload.last_signal_at_after,
    ) == (
        2,
        EvaluationVerdict.SUPPORTED,
        EvaluationVerdict.REFUTED,
        1,
        0,
        0,
        1,
        MechanismMaturity.CANDIDATE,
        MechanismMaturity.INCUBATING,
        first.evaluated_at,
        None,
        first.evaluated_at,
        second.evaluated_at,
    )
    projected = (_projected_evaluation(second, revision=2),)
    await _assert_event_matches_projections(
        evaluation_stack,
        row=second_row,
        payload=second_payload,
        expected_evaluations=projected,
    )

    third = _evaluation(
        idea,
        verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at_delta=timedelta(minutes=3),
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    third_result = await _evaluate(
        evaluation_stack,
        third,
        key="evaluation-revision-3",
    )
    _assert_success(
        third_result,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ACTIVE,
        maturity=MechanismMaturity.CANDIDATE,
        revision=3,
    )
    third_row, third_payload = (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
    )[2]
    assert (
        third_payload.revision,
        third_payload.previous_verdict,
        third_payload.current_verdict,
        third_payload.supported_count_before,
        third_payload.supported_count_after,
        third_payload.refuted_count_before,
        third_payload.refuted_count_after,
        third_payload.maturity_before,
        third_payload.maturity_after,
        third_payload.candidate_since_before,
        third_payload.candidate_since_after,
        third_payload.last_signal_at_before,
        third_payload.last_signal_at_after,
    ) == (
        3,
        EvaluationVerdict.REFUTED,
        EvaluationVerdict.SUPPORTED,
        0,
        1,
        1,
        0,
        MechanismMaturity.INCUBATING,
        MechanismMaturity.CANDIDATE,
        None,
        third.evaluated_at,
        second.evaluated_at,
        third.evaluated_at,
    )
    projected = (_projected_evaluation(third, revision=3),)
    await _assert_event_matches_projections(
        evaluation_stack,
        row=third_row,
        payload=third_payload,
        expected_evaluations=projected,
    )
    assert (
        await _run_projection(
            evaluation_stack,
            run_id=idea.run_id,
        )
        == run_before
    )

    events = await _evaluated_events(
        evaluation_stack,
        idea_id=idea.idea_id,
    )
    async with evaluation_stack.run.database.read_session() as session:
        receipts = (
            await session.scalars(
                select(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == "inspiration.idea.evaluate")
                .order_by(IdempotencyRecordRow.created_at)
            )
        ).all()
    assert tuple(row.causation_id for row, _ in events) == tuple(
        receipt.receipt_id for receipt in receipts
    )
    assert {
        (receipt.result_resource_type, receipt.result_resource_id)
        for receipt in receipts
    } == {("idea", idea.idea_id)}


@pytest.mark.asyncio
async def test_archived_idea_remains_archived_when_evaluated(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="archived-evaluation-idea",
        snapshot_item=711,
        content_marker=11,
    )
    await _archive_idea(
        evaluation_stack,
        idea_id=idea.idea_id,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    evaluation = _evaluation(
        idea,
        verdict=EvaluationVerdict.INCONCLUSIVE,
        evaluated_at_delta=timedelta(minutes=1),
    )

    result = await _evaluate(
        evaluation_stack,
        evaluation,
        key="archived-evaluation",
    )

    _assert_success(
        result,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ARCHIVED,
        maturity=MechanismMaturity.SPECULATIVE,
        revision=1,
    )
    row, payload = (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
    )[0]
    assert (
        payload.owner_decision_before,
        payload.owner_decision_after,
        payload.supported_count_before,
        payload.supported_count_after,
        payload.refuted_count_before,
        payload.refuted_count_after,
        payload.maturity_before,
        payload.maturity_after,
        payload.candidate_since_before,
        payload.candidate_since_after,
    ) == (
        IdeaOwnerDecision.ARCHIVED,
        IdeaOwnerDecision.ARCHIVED,
        0,
        0,
        0,
        0,
        MechanismMaturity.SPECULATIVE,
        MechanismMaturity.SPECULATIVE,
        None,
        None,
    )
    await _assert_event_matches_projections(
        evaluation_stack,
        row=row,
        payload=payload,
        expected_evaluations=(_projected_evaluation(evaluation, revision=1),),
        expected_sequence=3,
    )


@pytest.mark.asyncio
async def test_archived_evaluation_rejects_a_corrupt_decision_reason(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="archived-corrupt-reason-idea",
        snapshot_item=712,
        content_marker=12,
    )
    await _archive_idea(
        evaluation_stack,
        idea_id=idea.idea_id,
    )
    corrupt_reason = StructuredReason.from_user_text(
        "This reason was not recorded by the archive event."
    )
    async with evaluation_stack.run.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == idea.idea_id)
            .values(decision_reason=canonical_json_bytes(corrupt_reason))
        )
        assert changed.rowcount == 1
    evaluation_stack.run.clock.advance(timedelta(minutes=1))

    with pytest.raises(
        InspirationProjectionIntegrityError,
        match="predecessor",
    ):
        await _evaluate(
            evaluation_stack,
            _evaluation(
                idea,
                verdict=EvaluationVerdict.INCONCLUSIVE,
                evaluated_at_delta=timedelta(minutes=1),
            ),
            key="archived-corrupt-reason-evaluation",
        )

    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.parametrize(
    "decision",
    (IdeaOwnerDecision.ADOPTED, IdeaOwnerDecision.REJECTED),
)
@pytest.mark.asyncio
async def test_terminal_owner_decisions_refuse_evaluation(
    evaluation_stack: EvaluationStack,
    decision: IdeaOwnerDecision,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key=f"{decision.value}-evaluation-idea",
        snapshot_item=720 + len(decision.value),
        content_marker=20 + len(decision.value),
    )
    await _arrange_owner_decision(
        evaluation_stack,
        idea_id=idea.idea_id,
        decision=decision,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))

    result = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(minutes=1),
        ),
        key=f"{decision.value}-cannot-evaluate",
    )

    assert result.status_code == 409
    assert _error(result)["code"] == "idea_not_evaluable"
    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )
    async with evaluation_stack.run.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == decision.value
    assert state.evaluations == canonical_json_bytes(())


@pytest.mark.asyncio
async def test_evaluation_is_owner_only_and_hides_a_private_idea(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="private-evaluation-idea",
        snapshot_item=731,
        content_marker=31,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    foreign = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            evaluator_agent_id=OTHER_AGENT_ID,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(minutes=1),
        ),
        key="foreign-private-evaluation",
    )
    unknown = await _evaluate(
        evaluation_stack,
        IdeaEvaluation(
            evaluator_agent_id=OTHER_AGENT_ID,
            idea_id=uid(999),
            verdict=EvaluationVerdict.SUPPORTED,
            reason=None,
            evidence=(
                SnapshotEvidenceReference(
                    id=uid(998),
                    stable_evidence_key="9" * 64,
                ),
            ),
            evaluated_at=NOW + timedelta(minutes=1),
        ),
        key="unknown-private-evaluation",
    )

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.body == unknown.body
    assert _error(foreign)["code"] == "resource_not_found"
    assert await _evaluated_events(evaluation_stack) == ()


@pytest.mark.asyncio
async def test_evaluation_command_is_bound_to_its_request_body(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="request-bound-evaluation-idea",
        snapshot_item=739,
        content_marker=39,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    actual = _evaluation(
        idea,
        verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at_delta=timedelta(minutes=1),
    )
    declared = actual.model_copy(update={"verdict": EvaluationVerdict.REFUTED})

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await evaluation_stack.service.evaluate(
            uow=uow,
            evaluation=actual,
            command_context=command_context,
        )

    result = await evaluation_stack.executor.execute(
        _evaluation_request(declared, key="request-bound-evaluation"),
        handler,
    )

    assert result.status_code == 404
    assert _error(result)["code"] == "resource_not_found"
    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_evidence_must_resolve_to_same_run_key_or_owned_version(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="evidence-target-idea",
        snapshot_item=741,
        content_marker=41,
    )
    other_run_idea = await _seed_idea(
        evaluation_stack,
        key="evidence-other-run-idea",
        snapshot_item=742,
        content_marker=42,
    )
    foreign_version_id = uid(821)
    await _seed_experience_version(
        evaluation_stack,
        owner_agent_id=OTHER_AGENT_ID,
        experience_id=uid(820),
        version_id=foreign_version_id,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    invalid_references: tuple[EvaluationEvidenceReference, ...] = (
        other_run_idea.snapshot_evidence,
        SnapshotEvidenceReference(
            id=uid(749),
            stable_evidence_key="7" * 64,
        ),
        SnapshotEvidenceReference(
            id=idea.snapshot_item_id,
            stable_evidence_key="8" * 64,
        ),
        ExperienceVersionEvidenceReference(id=foreign_version_id),
        ExperienceVersionEvidenceReference(id=uid(829)),
    )

    for index, evidence in enumerate(invalid_references, start=1):
        result = await _evaluate(
            evaluation_stack,
            _evaluation(
                idea,
                verdict=EvaluationVerdict.SUPPORTED,
                evaluated_at_delta=timedelta(minutes=1),
                evidence=(evidence,),
            ),
            key=f"invalid-evaluation-evidence-{index}",
        )
        assert result.status_code == 422
        assert _error(result)["code"] == "invalid_evidence"

    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )
    async with evaluation_stack.run.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
        cluster = await session.get(
            MechanismIncubationRow,
            idea.mechanism_cluster_id,
        )
    assert state is not None
    assert cluster is not None
    assert state.evaluations == canonical_json_bytes(())
    assert (
        cluster.supported_count,
        cluster.refuted_count,
        cluster.maturity,
        cluster.candidate_since,
    ) == (0, 0, MechanismMaturity.INCUBATING.value, None)

    with pytest.raises(ValidationError, match="evidence"):
        IdeaEvaluation(
            evaluator_agent_id=OWNER_ID,
            idea_id=idea.idea_id,
            verdict=EvaluationVerdict.SUPPORTED,
            reason=None,
            evidence=(),
            evaluated_at=NOW + timedelta(minutes=1),
        )


@pytest.mark.asyncio
async def test_evaluation_cannot_cite_an_owned_version_from_the_future(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="future-version-evidence-target",
        snapshot_item=743,
        content_marker=43,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=10))
    future_version_id = uid(831)
    await _seed_experience_version(
        evaluation_stack,
        owner_agent_id=OWNER_ID,
        experience_id=uid(830),
        version_id=future_version_id,
        created_at=evaluation_stack.run.clock.now(),
    )

    result = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(minutes=1),
            evidence=(
                ExperienceVersionEvidenceReference(
                    id=future_version_id,
                ),
            ),
        ),
        key="future-version-evidence",
    )

    assert result.status_code == 422
    assert _error(result)["code"] == "invalid_evidence"
    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_duplicate_evidence_is_a_replayable_validation_error(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="duplicate-evidence-target",
        snapshot_item=749,
        content_marker=49,
    )
    evaluation_stack.run.clock.advance(timedelta(minutes=1))
    alternate_snapshot_identity = SnapshotEvidenceReference(
        id=uid(759),
        stable_evidence_key=idea.stable_evidence_key,
    )
    duplicate_sets = (
        (idea.snapshot_evidence, idea.snapshot_evidence),
        (idea.snapshot_evidence, alternate_snapshot_identity),
    )

    for index, evidence in enumerate(duplicate_sets, start=1):
        invalid = _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(minutes=1),
        ).model_copy(update={"evidence": evidence})
        result = await _evaluate(
            evaluation_stack,
            invalid,
            key=f"duplicate-evaluation-evidence-{index}",
        )

        assert result.status_code == 422
        assert _error(result)["code"] == "invalid_evaluation"

    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_corrupt_cluster_counts_fail_closed_as_source_integrity(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="corrupt-cluster-counts-target",
        snapshot_item=758,
        content_marker=58,
    )
    async with evaluation_stack.run.database.transaction() as uow:
        changed = await uow.session.execute(
            update(MechanismIncubationRow)
            .where(MechanismIncubationRow.cluster_id == idea.mechanism_cluster_id)
            .values(supported_count=1)
        )
        assert changed.rowcount == 1
    evaluation_stack.run.clock.advance(timedelta(minutes=1))

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="evaluation transition",
    ):
        await _evaluate(
            evaluation_stack,
            _evaluation(
                idea,
                verdict=EvaluationVerdict.INCONCLUSIVE,
                evaluated_at_delta=timedelta(minutes=1),
            ),
            key="corrupt-cluster-counts-evaluation",
        )

    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_evaluation_rejects_a_future_domain_time(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="future-evaluation-target",
        snapshot_item=750,
        content_marker=50,
    )

    result = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(microseconds=1),
        ),
        key="future-evaluation",
    )

    assert result.status_code == 422
    assert _error(result)["code"] == "invalid_evaluated_at"
    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_evaluation_rejects_time_behind_idea_or_cluster_latest_signal(
    evaluation_stack: EvaluationStack,
) -> None:
    idea = await _seed_idea(
        evaluation_stack,
        key="clock-regression-target",
        snapshot_item=751,
        content_marker=51,
    )
    behind_idea = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(microseconds=-1),
        ),
        key="clock-behind-idea",
    )
    assert behind_idea.status_code == 409
    assert _error(behind_idea)["code"] == "clock_regression"

    evaluation_stack.run.clock.advance(timedelta(minutes=2))
    await _seed_idea(
        evaluation_stack,
        key="clock-regression-newer-cluster-signal",
        snapshot_item=752,
        content_marker=52,
    )
    behind_cluster = await _evaluate(
        evaluation_stack,
        _evaluation(
            idea,
            verdict=EvaluationVerdict.SUPPORTED,
            evaluated_at_delta=timedelta(minutes=1),
        ),
        key="clock-behind-cluster",
    )
    assert behind_cluster.status_code == 409
    assert _error(behind_cluster)["code"] == "clock_regression"
    assert (
        await _evaluated_events(
            evaluation_stack,
            idea_id=idea.idea_id,
        )
        == ()
    )
    async with evaluation_stack.run.database.read_session() as session:
        cluster = await session.get(
            MechanismIncubationRow,
            idea.mechanism_cluster_id,
        )
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == InspirationIdeaEvaluatedV1.event_type)
        )
    assert cluster is not None
    assert cluster.last_signal_at == NOW + timedelta(minutes=2)
    assert event_count == 0
