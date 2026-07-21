from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select, text
from tests.integration.test_inspiration_run import (
    NOW,
    FakeGenerator,
    FakeSnapshotBuilder,
    Stack,
    build_stack,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    StructuredReason,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.events import (
    InspirationIdeaArchivedV1,
    InspirationIdeaEvaluatedV1,
)
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.hashing import (
    hash_mechanism,
    mechanism_similarity,
)
from experience_hub.inspiration.models import (
    EvaluationVerdict,
    GeneratorKind,
    IdeaDraft,
    IdeaOwnerDecision,
    InspirationOperator,
    MechanismMaturity,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.projector import (
    InspirationProjectionIntegrityError,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    IdeaStateRow,
    InspirationIdeaRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)

OWNER_A = UUID("00000000-0000-0000-0000-000000000101")
OWNER_B = UUID("00000000-0000-0000-0000-000000009002")
BASE = "abcdefghijklmnopqrstuvwxyz0123456789"
BRIDGE_LEFT = "Xbcdefghijklmnopqrstuvwxyz0123456789"
BRIDGE_RIGHT = "aXcdefghijklmnopqrstuvwxyz0123456789"
NEAR_END = "abcdefghijklmnopqrstuvwxyz012345678XX"
BELOW_END = "abcdefghijklmnopqrstuvwxyz012345678XXX"
_PROJECTION_ORDER = {
    "inspiration_run_state": "run_id",
    "mechanism_incubation": "cluster_id",
    "idea_state": "idea_id",
}


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


@dataclass(slots=True)
class MechanismSequenceGenerator(FakeGenerator):
    mechanisms: list[str] = field(default_factory=list)

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        _ = (goal, context, branch_limit, output_token_limit)
        self.calls.append(operator)
        if not self.mechanisms:
            raise AssertionError("mechanism sequence is exhausted")
        mechanism = self.mechanisms.pop(0)
        item = frozen_items[0]
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title=f"{operator.value} mechanism {len(self.calls)}",
                    hypothesis="The selected mechanism remains testable.",
                    mechanism=mechanism,
                    predictions=("The measured capacity changes.",),
                    falsifiers=("The measured capacity is unchanged.",),
                    assumptions=("The queue remains bounded.",),
                    proposed_test="Compare capacity before and after the signal.",
                    evidence=(
                        SnapshotEvidenceReference(
                            id=item.snapshot_item_id,
                            stable_evidence_key=item.stable_evidence_key,
                        ),
                    ),
                ),
            ),
            output_tokens_consumed=0,
        )


def _run(owner_agent_id: UUID) -> StartInspirationRun:
    return StartInspirationRun(
        owner_agent_id=owner_agent_id,
        goal="Rebuild deterministic incubation",
        generator=GeneratorKind.DETERMINISTIC,
        operators=(InspirationOperator.CAUSAL_GAP,),
    )


def _request(
    *,
    owner_agent_id: UUID,
    key: str,
    run: StartInspirationRun,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "goal": run.goal,
            "context": run.context,
            "mode": run.mode.value,
            "generator": run.generator.value,
            "operators": tuple(operator.value for operator in run.operators),
            "include_inbox": run.include_inbox,
            "branches_per_operator": run.branches_per_operator,
            "output_tokens_per_operator": run.output_tokens_per_operator,
            "total_output_tokens": run.total_output_tokens,
            "operator_timeout_seconds": run.operator_timeout_seconds,
            "global_timeout_seconds": run.global_timeout_seconds,
        },
    )


def _manager(stack: Stack) -> ProjectionManager:
    manager = cast(Any, stack.database)._projection_applier
    assert isinstance(manager, ProjectionManager)
    return manager


async def _build_history(
    *,
    repository_root: Path,
    database_path: Path,
    mechanisms: Sequence[str],
    owners: Sequence[UUID] | None = None,
    content_hashes: Sequence[str] | None = None,
) -> Stack:
    retained_owners = tuple(owners or (OWNER_A,) * len(mechanisms))
    retained_hashes = tuple(
        content_hashes or (f"{index + 1:064x}" for index in range(len(mechanisms)))
    )
    if not (len(mechanisms) == len(retained_owners) == len(retained_hashes)):
        raise ValueError("history inputs must have equal lengths")
    generator = MechanismSequenceGenerator(mechanisms=list(mechanisms))
    builder = FakeSnapshotBuilder(
        item_ids=[_uuid(8_000 + index) for index in range(len(mechanisms))],
        content_hashes=list(retained_hashes),
    )
    stack = await build_stack(
        repository_root=repository_root,
        database_path=database_path,
        generator=generator,
        snapshot_builder=builder,
    )
    deliberately_nonmonotonic_ids: list[UUID] = []
    for index in range(len(mechanisms)):
        deliberately_nonmonotonic_ids.extend(
            (
                _uuid(50_000 + index),
                _uuid(70_000 + len(mechanisms) - index),
                _uuid(60_000 + index),
            )
        )
    cast(Any, stack.executor)._ids = SequenceIdGenerator(  # noqa: SLF001
        deliberately_nonmonotonic_ids
    )
    try:
        if OWNER_B in retained_owners:
            async with stack.database.transaction() as uow:
                uow.session.add(
                    AgentRow(
                        agent_id=OWNER_B,
                        name="Private peer",
                        created_at=stack.clock.now(),
                    )
                )
        for index, owner_agent_id in enumerate(retained_owners):
            selected = _run(owner_agent_id)
            response = await stack.executor.execute(
                request=_request(
                    owner_agent_id=owner_agent_id,
                    key=f"rebuild-{index}",
                    run=selected,
                ),
                run=selected,
            )
            assert response.status_code == 201
        return stack
    except BaseException:
        await stack.database.dispose()
        raise


async def _projection_rows(
    stack: Stack,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    async with stack.database.read_session() as session:
        return {
            table_name: tuple(
                tuple(row)
                for row in await session.execute(
                    text(f"SELECT * FROM {table_name} ORDER BY {ordering}")
                )
            )
            for table_name, ordering in _PROJECTION_ORDER.items()
        }


async def _rebuilt_rows(
    stack: Stack,
    *,
    prefix: str,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    manager = _manager(stack)
    async with stack.database.transaction() as uow:
        for reducer in manager.registry.reducers:
            await reducer.rebuild(uow.session, prefix)
        rebuilt = {
            table_name: tuple(
                tuple(row)
                for row in await uow.session.execute(
                    text(f"SELECT * FROM temp.{prefix}{table_name} ORDER BY {ordering}")
                )
            )
            for table_name, ordering in _PROJECTION_ORDER.items()
        }
        for table_name in reversed(tuple(_PROJECTION_ORDER)):
            await uow.session.execute(text(f"DROP TABLE temp.{prefix}{table_name}"))
    return rebuilt


async def _generated_payloads(
    stack: Stack,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == "inspiration.idea_generated")
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    return tuple(
        (row.event_id, cast(dict[str, Any], json.loads(row.payload))) for row in rows
    )


async def _append_evaluation_and_archive(
    stack: Stack,
) -> tuple[UUID, int, int, UUID, str]:
    evaluated_at = NOW + timedelta(minutes=1)
    archived_at = evaluated_at + timedelta(minutes=1)
    async with stack.database.transaction(immediate=True) as uow:
        generated = await uow.session.scalar(
            select(DomainEventRow)
            .where(DomainEventRow.event_type == "inspiration.idea_generated")
            .order_by(DomainEventRow.event_id)
            .limit(1)
        )
        assert generated is not None
        idea = await uow.session.get(InspirationIdeaRow, generated.aggregate_id)
        state = await uow.session.get(IdeaStateRow, generated.aggregate_id)
        assert idea is not None
        assert state is not None
        item = await uow.session.scalar(
            select(InspirationSnapshotItemRow).where(
                InspirationSnapshotItemRow.run_id == idea.run_id
            )
        )
        cluster = await uow.session.get(
            MechanismIncubationRow,
            state.mechanism_cluster_id,
        )
        assert item is not None
        assert cluster is not None
        assert cluster.maturity == MechanismMaturity.INCUBATING.value
        idea_id = idea.idea_id
        mechanism_cluster_id = state.mechanism_cluster_id
        snapshot_item_id = item.snapshot_item_id
        stable_evidence_key = item.stable_evidence_key
        cluster_last_signal_at = cluster.last_signal_at
        command_context = CommandContext(
            receipt_id=generated.causation_id,
            caller_scope=f"agent:{OWNER_A}",
            operation_scope="projection.rebuild.fixture",
            idempotency_key="evaluation-and-archive",
            request_hash="f" * 64,
        )
        [evaluated] = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=idea_id,
                    event_type=InspirationIdeaEvaluatedV1.event_type,
                    payload=InspirationIdeaEvaluatedV1(
                        schema_version=1,
                        idea_id=idea_id,
                        evaluator_agent_id=OWNER_A,
                        mechanism_cluster_id=mechanism_cluster_id,
                        revision=1,
                        previous_verdict=None,
                        current_verdict=EvaluationVerdict.SUPPORTED,
                        evidence=(
                            SnapshotEvidenceReference(
                                id=snapshot_item_id,
                                stable_evidence_key=stable_evidence_key,
                            ),
                        ),
                        reason=None,
                        owner_decision_before=IdeaOwnerDecision.ACTIVE,
                        owner_decision_after=IdeaOwnerDecision.ACTIVE,
                        supported_count_before=0,
                        supported_count_after=1,
                        refuted_count_before=0,
                        refuted_count_after=0,
                        maturity_before=MechanismMaturity.INCUBATING,
                        maturity_after=MechanismMaturity.CANDIDATE,
                        candidate_since_before=None,
                        candidate_since_after=evaluated_at,
                        last_signal_at_before=cluster_last_signal_at,
                        last_signal_at_after=evaluated_at,
                    ),
                    actor_agent_id=OWNER_A,
                    occurred_at=evaluated_at,
                ),
            ),
        )
        [archived] = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=idea_id,
                    event_type=InspirationIdeaArchivedV1.event_type,
                    payload=InspirationIdeaArchivedV1(
                        schema_version=1,
                        idea_id=idea_id,
                        owner_agent_id=OWNER_A,
                        reason=StructuredReason.policy_due(),
                        owner_decision_before=IdeaOwnerDecision.ACTIVE,
                        owner_decision_after=IdeaOwnerDecision.ARCHIVED,
                        cycle_id=None,
                    ),
                    actor_agent_id=OWNER_A,
                    occurred_at=archived_at,
                ),
            ),
        )
        retained = (
            idea_id,
            evaluated.event_id,
            archived.event_id,
            snapshot_item_id,
            stable_evidence_key,
        )
    return retained


@pytest.mark.asyncio
async def test_event_order_rebuild_matches_incremental_rows_and_preserves_privacy(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    same_snapshot = "a" * 64
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "event-order-rebuild.sqlite3",
        mechanisms=(BASE, NEAR_END, BASE),
        owners=(OWNER_A, OWNER_B, OWNER_A),
        content_hashes=(same_snapshot, same_snapshot, "b" * 64),
    )
    try:
        generated = await _generated_payloads(stack)
        assert tuple(event_id for event_id, _ in generated) == tuple(
            sorted(event_id for event_id, _ in generated)
        )
        assert [payload["idea_id"] for _, payload in generated] == sorted(
            (payload["idea_id"] for _, payload in generated),
            reverse=True,
        )
        assert [
            (
                payload["occurrence_count_before"],
                payload["occurrence_count_after"],
                payload["distinct_snapshot_count_before"],
                payload["distinct_snapshot_count_after"],
            )
            for _, payload in generated
        ] == [
            (0, 1, 0, 1),
            (1, 2, 1, 1),
            (2, 3, 1, 2),
        ]
        assert generated[1][1]["duplicate_relation"] is None
        assert generated[2][1]["duplicate_relation"] == generated[0][1]["idea_id"]
        assert generated[0][1]["cluster_id"] == hash_mechanism(BASE)
        assert generated[1][1]["cluster_id"] == hash_mechanism(BASE)
        assert generated[2][1]["cluster_id"] == hash_mechanism(BASE)

        online = await _projection_rows(stack)
        rebuilt = await _rebuilt_rows(stack, prefix="event_order_")

        assert rebuilt == online
        assert (await _manager(stack).verify(stack.database)).matches
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_evaluation_and_archive_rebuild_exact_latest_effective_state(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "evaluation-archive-rebuild.sqlite3",
        mechanisms=(BASE, BASE),
        content_hashes=("a" * 64, "b" * 64),
    )
    try:
        (
            idea_id,
            evaluation_event_id,
            archive_event_id,
            snapshot_item_id,
            stable_evidence_key,
        ) = await _append_evaluation_and_archive(stack)
        async with stack.database.read_session() as session:
            idea = await session.get(IdeaStateRow, idea_id)
            cluster = await session.scalar(select(MechanismIncubationRow))
        assert idea is not None
        assert cluster is not None
        assert idea.owner_decision == IdeaOwnerDecision.ARCHIVED.value
        assert idea.decision_reason == canonical_json_bytes(
            StructuredReason.policy_due()
        )
        assert idea.last_signal_at == NOW + timedelta(minutes=1)
        assert idea.projection_event_id == archive_event_id
        assert json.loads(idea.evaluations) == [
            {
                "evaluated_at": "2026-07-18T08:31:00.000000Z",
                "evaluator_agent_id": str(OWNER_A),
                "evidence": [
                    {
                        "id": str(snapshot_item_id),
                        "stable_evidence_key": stable_evidence_key,
                        "type": "snapshot_item",
                    }
                ],
                "reason": None,
                "revision": 1,
                "verdict": EvaluationVerdict.SUPPORTED.value,
            }
        ]
        assert (
            cluster.supported_count,
            cluster.refuted_count,
            cluster.maturity,
            cluster.candidate_since,
            cluster.last_signal_at,
            cluster.projection_event_id,
        ) == (
            1,
            0,
            MechanismMaturity.CANDIDATE.value,
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=1),
            evaluation_event_id,
        )

        online = await _projection_rows(stack)
        rebuilt = await _rebuilt_rows(stack, prefix="evaluated_archived_")

        assert rebuilt == online
        assert (await _manager(stack).verify(stack.database)).matches
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_cluster_members_remain_in_original_generation_order_after_updates(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "stable-member-order.sqlite3",
        mechanisms=(BASE, BASE),
        content_hashes=("a" * 64, "b" * 64),
    )
    try:
        generated = await _generated_payloads(stack)
        await _append_evaluation_and_archive(stack)

        async with stack.database.read_session() as session:
            clusters = await InspirationRepository.load_clusters(session=session)

        assert len(clusters) == 1
        assert tuple(member.idea_id for member in clusters[0].members) == tuple(
            UUID(payload["idea_id"]) for _, payload in generated
        )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_cluster_loading_rejects_a_missing_generated_source_event(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "missing-generated-source.sqlite3",
        mechanisms=(BASE,),
    )
    try:
        async with stack.database.transaction() as uow:
            generated = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.idea_generated"
                )
            )
            assert generated is not None
            await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
            await uow.session.execute(
                text(
                    "UPDATE domain_events SET event_type=:event_type "
                    "WHERE event_id=:event_id"
                ),
                {
                    "event_type": "corrupted.idea_generated",
                    "event_id": generated.event_id,
                },
            )
            await uow.session.execute(
                text(
                    "CREATE TRIGGER domain_events_reject_update "
                    "BEFORE UPDATE ON domain_events BEGIN "
                    "SELECT RAISE(ABORT, "
                    "'domain_events rows are immutable'); END"
                )
            )

        async with stack.database.read_session() as session:
            with pytest.raises(
                InspirationSourceIntegrityError,
                match="generated source",
            ):
                await InspirationRepository.load_clusters(session=session)
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_rebuild_uses_maximum_member_similarity_not_only_the_canonical(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    assert mechanism_similarity(BASE, NEAR_END) >= 0.82
    assert mechanism_similarity(BASE, BELOW_END) < 0.82
    assert mechanism_similarity(NEAR_END, BELOW_END) >= 0.82
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "maximum-member-rebuild.sqlite3",
        mechanisms=(BASE, NEAR_END, BELOW_END),
    )
    try:
        generated = await _generated_payloads(stack)
        assert [payload["cluster_id"] for _, payload in generated] == [
            hash_mechanism(BASE),
            hash_mechanism(BASE),
            hash_mechanism(BASE),
        ]
        assert generated[-1][1]["member_hashes_after"] == [
            hash_mechanism(BASE),
            hash_mechanism(NEAR_END),
            hash_mechanism(BELOW_END),
        ]

        assert await _rebuilt_rows(stack, prefix="maximum_member_") == (
            await _projection_rows(stack)
        )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_highest_similarity_then_canonical_hash_tie_break_never_merges_clusters(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    assert mechanism_similarity(NEAR_END, BASE) < mechanism_similarity(
        BRIDGE_LEFT,
        BASE,
    )
    assert mechanism_similarity(BRIDGE_LEFT, BRIDGE_RIGHT) < 0.82
    assert mechanism_similarity(BRIDGE_LEFT, BASE) == mechanism_similarity(
        BRIDGE_RIGHT,
        BASE,
    )
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "tie-break-rebuild.sqlite3",
        mechanisms=(NEAR_END, BRIDGE_LEFT, BRIDGE_RIGHT, BASE),
    )
    try:
        generated = await _generated_payloads(stack)
        eligible = (hash_mechanism(BRIDGE_LEFT), hash_mechanism(BRIDGE_RIGHT))
        expected_cluster = min(eligible)
        assert generated[-1][1]["cluster_id"] == expected_cluster
        assert generated[-1][1]["cluster_id"] != hash_mechanism(NEAR_END)

        async with stack.database.read_session() as session:
            clusters = tuple(
                await session.execute(
                    text(
                        "SELECT cluster_id, occurrence_count, member_hashes "
                        "FROM mechanism_incubation ORDER BY cluster_id"
                    )
                )
            )
        assert len(clusters) == 3
        assert sorted(int(row.occurrence_count) for row in clusters) == [1, 1, 2]
        untouched = next(
            row
            for row in clusters
            if row.cluster_id in eligible and row.cluster_id != expected_cluster
        )
        assert json.loads(untouched.member_hashes) == [untouched.cluster_id]

        assert await _rebuilt_rows(stack, prefix="tie_break_") == (
            await _projection_rows(stack)
        )
    finally:
        await stack.database.dispose()


async def _forge_second_generated_event_as_new_cluster(stack: Stack) -> None:
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow)
            .where(DomainEventRow.event_type == "inspiration.idea_generated")
            .order_by(DomainEventRow.event_id)
            .offset(1)
            .limit(1)
        )
        assert event is not None
        payload = cast(dict[str, Any], json.loads(event.payload))
        mechanism_hash = cast(str, payload["mechanism_hash"])
        payload.update(
            {
                "cluster_id": mechanism_hash,
                "canonical_mechanism_hash": mechanism_hash,
                "member_hashes_before": (),
                "member_hashes_after": (mechanism_hash,),
                "occurrence_count_before": 0,
                "occurrence_count_after": 1,
                "distinct_snapshot_count_before": 0,
                "distinct_snapshot_count_after": 1,
                "distinct_adopter_count_before": 0,
                "distinct_adopter_count_after": 0,
                "supported_count_before": 0,
                "supported_count_after": 0,
                "refuted_count_before": 0,
                "refuted_count_after": 0,
                "maturity_before": None,
                "maturity_after": "speculative",
                "candidate_since_before": None,
                "candidate_since_after": None,
                "last_signal_at_before": None,
            }
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text("UPDATE domain_events SET payload=:payload WHERE event_id=:event_id"),
            {
                "payload": canonical_json_bytes(payload),
                "event_id": event.event_id,
            },
        )
        await uow.session.execute(
            text(
                "CREATE TRIGGER domain_events_reject_update "
                "BEFORE UPDATE ON domain_events BEGIN "
                "SELECT RAISE(ABORT, 'domain_events rows are immutable'); END"
            )
        )


async def _forge_evaluation_before_state(stack: Stack) -> None:
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaEvaluatedV1.event_type
            )
        )
        assert event is not None
        payload = cast(dict[str, Any], json.loads(event.payload))
        payload.update(
            {
                "supported_count_before": 1,
                "supported_count_after": 2,
                "maturity_before": MechanismMaturity.CANDIDATE.value,
                "maturity_after": MechanismMaturity.CANDIDATE.value,
                "candidate_since_before": "2026-07-18T08:30:00.000000Z",
                "candidate_since_after": "2026-07-18T08:30:00.000000Z",
            }
        )
        encoded = canonical_json_bytes(payload)
        InspirationIdeaEvaluatedV1.model_validate_json(encoded)
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text("UPDATE domain_events SET payload=:payload WHERE event_id=:event_id"),
            {"payload": encoded, "event_id": event.event_id},
        )
        await uow.session.execute(
            text(
                "CREATE TRIGGER domain_events_reject_update "
                "BEFORE UPDATE ON domain_events BEGIN "
                "SELECT RAISE(ABORT, 'domain_events rows are immutable'); END"
            )
        )


@pytest.mark.asyncio
async def test_rebuild_rejects_a_valid_but_forged_declared_cluster_transition(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "forged-transition.sqlite3",
        mechanisms=(BASE, NEAR_END),
        owners=(OWNER_A, OWNER_B),
        content_hashes=("a" * 64, "a" * 64),
    )
    try:
        await _forge_second_generated_event_as_new_cluster(stack)
        manager = _manager(stack)
        run_projector = next(
            reducer
            for reducer in manager.registry.reducers
            if isinstance(reducer, InspirationRunProjector)
        )
        mechanism_projector = next(
            reducer
            for reducer in manager.registry.reducers
            if isinstance(reducer, MechanismIncubationProjector)
        )
        prefix = "forged_assignment_"
        async with stack.database.transaction() as uow:
            await run_projector.rebuild(uow.session, prefix)
            with pytest.raises(InspirationProjectionIntegrityError):
                await mechanism_projector.rebuild(uow.session, prefix)
            await uow.session.execute(
                text(f"DROP TABLE IF EXISTS temp.{prefix}mechanism_incubation")
            )
            await uow.session.execute(
                text(f"DROP TABLE temp.{prefix}inspiration_run_state")
            )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_rebuild_rejects_forged_evaluation_before_and_after_counts(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "forged-evaluation-transition.sqlite3",
        mechanisms=(BASE, BASE),
        content_hashes=("a" * 64, "b" * 64),
    )
    try:
        await _append_evaluation_and_archive(stack)
        await _forge_evaluation_before_state(stack)
        manager = _manager(stack)
        run_projector = next(
            reducer
            for reducer in manager.registry.reducers
            if isinstance(reducer, InspirationRunProjector)
        )
        mechanism_projector = next(
            reducer
            for reducer in manager.registry.reducers
            if isinstance(reducer, MechanismIncubationProjector)
        )
        prefix = "forged_evaluation_"
        async with stack.database.transaction() as uow:
            await run_projector.rebuild(uow.session, prefix)
            with pytest.raises(InspirationProjectionIntegrityError):
                await mechanism_projector.rebuild(uow.session, prefix)
            await uow.session.execute(
                text(f"DROP TABLE IF EXISTS temp.{prefix}mechanism_incubation")
            )
            await uow.session.execute(
                text(f"DROP TABLE temp.{prefix}inspiration_run_state")
            )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_projection_corruption_is_detected_and_repair_restores_exact_rows(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await _build_history(
        repository_root=repository_root,
        database_path=tmp_path / "projection-repair.sqlite3",
        mechanisms=(BASE, NEAR_END, BASE),
        content_hashes=("a" * 64, "a" * 64, "b" * 64),
    )
    try:
        manager = _manager(stack)
        golden = await _projection_rows(stack)
        cluster_id = hash_mechanism(BASE)
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text(
                    "UPDATE mechanism_incubation "
                    "SET occurrence_count=occurrence_count + 1 "
                    "WHERE cluster_id=:cluster_id"
                ),
                {"cluster_id": cluster_id},
            )

        with pytest.raises(ProjectionMismatch) as caught:
            await manager.verify(stack.database)
        assert tuple(
            difference.projection for difference in caught.value.report.differences
        ) == ("mechanism_incubation",)
        assert caught.value.report.differences[0].differing_keys == (cluster_id,)

        repaired = await manager.repair(stack.database)

        assert repaired.matches
        assert await _projection_rows(stack) == golden
        assert (await manager.verify(stack.database)).matches
    finally:
        await stack.database.dispose()
