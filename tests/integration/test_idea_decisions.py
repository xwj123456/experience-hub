from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import pytest
from sqlalchemy import func, select, update
from tests.integration.test_inspiration_run import (
    NOW,
    OWNER_ID,
)
from tests.integration.test_inspiration_run import (
    Stack as InspirationStack,
)
from tests.integration.test_inspiration_run import (
    build_stack as build_inspiration_stack,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    StructuredReason,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
)
from experience_hub.inspiration.commands import (
    ArchiveIdea,
    RejectIdea,
    StartInspirationRun,
)
from experience_hub.inspiration.events import (
    InspirationIdeaArchivedV1,
    InspirationIdeaRejectedV1,
    register_inspiration_events,
)
from experience_hub.inspiration.lifecycle import IdeaLifecycleService
from experience_hub.inspiration.models import (
    IdeaOwnerDecision,
    InspirationOperator,
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
    ExperienceRow,
    ExperienceVersionRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    InspirationIdeaRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000102")
UNKNOWN_IDEA_ID = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
MICROSECOND = timedelta(microseconds=1)
DecisionAction = Literal["reject", "archive"]
DecisionReason = str | StructuredReason


@dataclass(slots=True)
class RegressingClock:
    values: list[datetime]

    def now(self) -> datetime:
        if not self.values:
            raise AssertionError("regressing clock was sampled too many times")
        return self.values.pop(0)


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


@dataclass(frozen=True, slots=True)
class SeededIdea:
    idea_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    last_signal_at: datetime


@dataclass(slots=True)
class DecisionStack:
    inspiration: InspirationStack
    registry: EventRegistry
    receipts: ReceiptStore
    executor: CommandExecutor
    service: IdeaLifecycleService
    seed_number: int = 0


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[DecisionStack]:
    inspiration = await build_inspiration_stack(
        repository_root=repository_root,
        database_path=tmp_path / "idea-decisions.sqlite3",
    )
    async with inspiration.database.transaction() as uow:
        uow.session.add(
            AgentRow(
                agent_id=OTHER_OWNER_ID,
                name="Other idea owner",
                created_at=NOW,
            )
        )
    registry = EventRegistry()
    register_inspiration_events(registry)
    receipts = cast(
        ReceiptStore,
        inspiration.executor._receipt_store,  # noqa: SLF001 - shared test stack
    )
    value = DecisionStack(
        inspiration=inspiration,
        registry=registry,
        receipts=receipts,
        executor=CommandExecutor(
            database=inspiration.database,
            receipt_store=receipts,
            clock=inspiration.clock,
        ),
        service=IdeaLifecycleService(
            clock=inspiration.clock,
            receipt_store=receipts,
            repository=InspirationRepository(registry),
        ),
    )
    try:
        yield value
    finally:
        await inspiration.database.dispose()


def inspiration_request(
    run: StartInspirationRun,
    *,
    key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{run.owner_agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": run.owner_agent_id},
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


async def seed_idea(
    stack: DecisionStack,
    *,
    key: str,
    owner_agent_id: UUID = OWNER_ID,
) -> SeededIdea:
    stack.seed_number += 1
    stack.inspiration.snapshot_builder.item_ids.append(uid(700 + stack.seed_number))
    stack.inspiration.snapshot_builder.content_hashes.append(
        f"{stack.seed_number:064x}"
    )
    run = StartInspirationRun(
        owner_agent_id=owner_agent_id,
        goal=f"Decision fixture {key}",
        operators=(InspirationOperator.CAUSAL_GAP,),
    )
    result = await stack.inspiration.executor.execute(
        request=inspiration_request(run, key=key),
        run=run,
    )
    assert result.status_code == 201
    run_id = UUID(json.loads(result.body)["data"]["run_id"])
    async with stack.inspiration.database.read_session() as session:
        idea = await session.scalar(
            select(InspirationIdeaRow).where(InspirationIdeaRow.run_id == run_id)
        )
        assert idea is not None
        state = await session.get(IdeaStateRow, idea.idea_id)
        assert state is not None
        return SeededIdea(
            idea_id=idea.idea_id,
            run_id=run_id,
            owner_agent_id=owner_agent_id,
            last_signal_at=state.last_signal_at,
        )


def command_for(
    action: DecisionAction,
    *,
    owner_agent_id: UUID,
    idea_id: UUID,
    reason: DecisionReason,
) -> RejectIdea | ArchiveIdea:
    command_type = RejectIdea if action == "reject" else ArchiveIdea
    return command_type(
        owner_agent_id=owner_agent_id,
        idea_id=idea_id,
        reason=reason,
    )


def request_for(
    action: DecisionAction,
    command: RejectIdea | ArchiveIdea,
    *,
    key: str,
    caller_scope: str | None = None,
    operation_scope: str | None = None,
) -> CommandRequest:
    try:
        normalized_reason: str | StructuredReason = (
            StructuredReason.from_user_text(command.reason)
            if isinstance(command.reason, str)
            else command.reason
        )
    except ValueError:
        normalized_reason = command.reason
    return CommandRequest(
        caller_scope=caller_scope or f"agent:{command.owner_agent_id}",
        operation_scope=operation_scope or f"inspiration.idea.{action}",
        idempotency_key=key,
        method="POST",
        route_template=(f"/v1/agents/{{agent_id}}/ideas/{{idea_id}}:{action}"),
        path_parameters={
            "agent_id": command.owner_agent_id,
            "idea_id": command.idea_id,
        },
        body={
            "reason": (
                normalized_reason
                if isinstance(normalized_reason, str)
                else normalized_reason.model_dump(mode="json")
            )
        },
    )


async def decide(
    stack: DecisionStack,
    *,
    action: DecisionAction,
    key: str,
    owner_agent_id: UUID,
    idea_id: UUID,
    reason: DecisionReason,
    caller_scope: str | None = None,
    operation_scope: str | None = None,
) -> CommandResult:
    command = command_for(
        action,
        owner_agent_id=owner_agent_id,
        idea_id=idea_id,
        reason=reason,
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        method = stack.service.reject if action == "reject" else stack.service.archive
        return await method(
            uow=uow,
            command=command,
            command_context=command_context,
        )

    return await stack.executor.execute(
        request_for(
            action,
            command,
            key=key,
            caller_scope=caller_scope,
            operation_scope=operation_scope,
        ),
        handler,
    )


def error_code(result: CommandResult) -> str:
    assert canonical_json_bytes(json.loads(result.body)) == result.body
    decoded = json.loads(result.body)
    assert set(decoded) == {"error"}
    return cast(str, decoded["error"]["code"])


def assert_success(
    result: CommandResult,
    *,
    idea_id: UUID,
    owner_decision: IdeaOwnerDecision,
) -> None:
    assert result.status_code == 200
    assert result.body == canonical_json_bytes(
        {
            "data": {
                "idea_id": idea_id,
                "owner_decision": owner_decision,
            }
        }
    )


async def decision_rows(
    stack: DecisionStack,
    *,
    idea_id: UUID,
) -> tuple[DomainEventRow, ...]:
    async with stack.inspiration.database.read_session() as session:
        return tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_type == "idea",
                        DomainEventRow.aggregate_id == idea_id,
                        DomainEventRow.event_type.in_(
                            (
                                InspirationIdeaRejectedV1.event_type,
                                InspirationIdeaArchivedV1.event_type,
                            )
                        ),
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )


async def receipt_for(
    stack: DecisionStack,
    *,
    operation_scope: str,
    key: str,
) -> IdempotencyRecordRow:
    async with stack.inspiration.database.read_session() as session:
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == operation_scope,
                IdempotencyRecordRow.idempotency_key == key,
            )
        )
        assert receipt is not None
        return receipt


async def arrange_adopted(
    stack: DecisionStack,
    *,
    idea_id: UUID,
) -> None:
    experience_id = uid(980)
    version_id = uid(981)
    async with stack.inspiration.database.transaction(immediate=True) as uow:
        experience = ExperienceRow(
            experience_id=experience_id,
            owner_agent_id=OWNER_ID,
            kind=ExperienceKind.HYPOTHESIS,
            origin=ExperienceOrigin.ADOPTED_IDEA,
            created_at=NOW,
        )
        uow.session.add(experience)
        await uow.session.flush((experience,))
        version = ExperienceVersionRow(
            version_id=version_id,
            experience_id=experience_id,
            version_number=1,
            summary="Previously adopted idea",
            mechanism="Previously adopted mechanism",
            tags=canonical_json_bytes(("inspiration",)),
            applicability=canonical_json_bytes(()),
            evidence=canonical_json_bytes(()),
            falsifiers=canonical_json_bytes(("No effect was observed.",)),
            content_hash="a" * 64,
            supersedes_version_id=None,
            created_at=NOW,
        )
        uow.session.add(version)
        await uow.session.flush((version,))
        changed = await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == idea_id)
            .values(
                owner_decision=IdeaOwnerDecision.ADOPTED.value,
                resulting_experience_id=experience_id,
                resulting_version_id=version_id,
            )
        )
        assert changed.rowcount == 1


@pytest.mark.parametrize("command_type", (RejectIdea, ArchiveIdea))
def test_decision_commands_require_a_reason(
    command_type: type[RejectIdea] | type[ArchiveIdea],
) -> None:
    with pytest.raises(ValueError, match="reason"):
        command_type(
            owner_agent_id=OWNER_ID,
            idea_id=UNKNOWN_IDEA_ID,
            reason=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("action", ("reject", "archive"))
@pytest.mark.asyncio
async def test_blank_reason_is_replayable_422_without_a_decision_event(
    stack: DecisionStack,
    action: DecisionAction,
) -> None:
    idea = await seed_idea(stack, key=f"blank-{action}-idea")

    result = await decide(
        stack,
        action=action,
        key=f"blank-{action}",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="   ",
    )

    assert result.status_code == 422
    assert error_code(result) == "invalid_reason"
    assert await decision_rows(stack, idea_id=idea.idea_id) == ()
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_active_reject_normalizes_reason_projects_once_and_replays(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="reject-active-idea")
    raw_reason = "  当前证据不足，保留想法但拒绝进入采用流程。  "
    expected_reason = StructuredReason.from_user_text(raw_reason)
    decided_at = stack.inspiration.clock.advance(timedelta(minutes=1))

    first = await decide(
        stack,
        action="reject",
        key="reject-active",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason=raw_reason,
    )

    assert_success(
        first,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.REJECTED,
    )
    assert not first.replayed
    rows = await decision_rows(stack, idea_id=idea.idea_id)
    assert len(rows) == 1
    row = rows[0]
    payload = stack.registry.decode(
        event_type=row.event_type,
        payload=row.payload,
    )
    assert isinstance(payload, InspirationIdeaRejectedV1)
    assert (
        row.sequence,
        row.actor_agent_id,
        row.occurred_at,
        payload.idea_id,
        payload.owner_agent_id,
        payload.reason,
        payload.owner_decision_before,
        payload.owner_decision_after,
    ) == (
        2,
        OWNER_ID,
        decided_at,
        idea.idea_id,
        OWNER_ID,
        expected_reason,
        IdeaOwnerDecision.ACTIVE,
        IdeaOwnerDecision.REJECTED,
    )
    receipt = await receipt_for(
        stack,
        operation_scope="inspiration.idea.reject",
        key="reject-active",
    )
    assert (
        row.causation_id,
        receipt.state,
        receipt.result_resource_type,
        receipt.result_resource_id,
        receipt.response_status_code,
        receipt.response_body,
    ) == (
        receipt.receipt_id,
        "completed",
        "idea",
        idea.idea_id,
        200,
        first.body,
    )
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert (
        state.owner_decision,
        state.decision_reason,
        state.evaluations,
        state.resulting_experience_id,
        state.resulting_version_id,
        state.last_signal_at,
        state.projection_event_id,
    ) == (
        IdeaOwnerDecision.REJECTED.value,
        canonical_json_bytes(expected_reason),
        canonical_json_bytes(()),
        None,
        None,
        idea.last_signal_at,
        row.event_id,
    )

    replay = await decide(
        stack,
        action="reject",
        key="reject-active",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason=raw_reason,
    )
    reject_again = await decide(
        stack,
        action="reject",
        key="reject-terminal-again",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="A terminal rejection cannot be replaced.",
    )
    archive_rejected = await decide(
        stack,
        action="archive",
        key="archive-rejected",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="A rejected idea cannot be archived again.",
    )

    assert replay.replayed
    assert (replay.status_code, replay.body, replay.headers) == (
        first.status_code,
        first.body,
        first.headers,
    )
    assert reject_again.status_code == archive_rejected.status_code == 409
    assert (
        error_code(reject_again) == error_code(archive_rejected) == "idea_not_decidable"
    )
    assert len(await decision_rows(stack, idea_id=idea.idea_id)) == 1


@pytest.mark.asyncio
async def test_explicit_archive_has_owner_actor_no_cycle_and_is_replayable(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="archive-active-idea")
    reason = StructuredReason.from_user_text(
        "Keep this idea available for a later explicit adoption."
    )
    decided_at = stack.inspiration.clock.advance(timedelta(minutes=1))

    first = await decide(
        stack,
        action="archive",
        key="archive-active",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason=reason,
    )

    assert_success(
        first,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ARCHIVED,
    )
    rows = await decision_rows(stack, idea_id=idea.idea_id)
    assert len(rows) == 1
    row = rows[0]
    payload = stack.registry.decode(
        event_type=row.event_type,
        payload=row.payload,
    )
    assert isinstance(payload, InspirationIdeaArchivedV1)
    assert (
        row.sequence,
        row.actor_agent_id,
        row.occurred_at,
        payload.idea_id,
        payload.owner_agent_id,
        payload.reason,
        payload.owner_decision_before,
        payload.owner_decision_after,
        payload.cycle_id,
    ) == (
        2,
        OWNER_ID,
        decided_at,
        idea.idea_id,
        OWNER_ID,
        reason,
        IdeaOwnerDecision.ACTIVE,
        IdeaOwnerDecision.ARCHIVED,
        None,
    )
    receipt = await receipt_for(
        stack,
        operation_scope="inspiration.idea.archive",
        key="archive-active",
    )
    assert row.causation_id == receipt.receipt_id
    assert (
        receipt.state,
        receipt.result_resource_type,
        receipt.result_resource_id,
        receipt.response_status_code,
        receipt.response_body,
    ) == ("completed", "idea", idea.idea_id, 200, first.body)
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert (
        state.owner_decision,
        state.decision_reason,
        state.evaluations,
        state.resulting_experience_id,
        state.resulting_version_id,
        state.last_signal_at,
        state.projection_event_id,
    ) == (
        IdeaOwnerDecision.ARCHIVED.value,
        canonical_json_bytes(reason),
        canonical_json_bytes(()),
        None,
        None,
        idea.last_signal_at,
        row.event_id,
    )

    replay = await decide(
        stack,
        action="archive",
        key="archive-active",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason=reason,
    )
    assert replay.replayed
    assert (replay.status_code, replay.body, replay.headers) == (
        first.status_code,
        first.body,
        first.headers,
    )
    assert len(await decision_rows(stack, idea_id=idea.idea_id)) == 1


@pytest.mark.asyncio
async def test_decision_command_is_bound_to_its_request_body(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="request-bound-decision-idea")
    actual = ArchiveIdea(
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="Archive the idea represented by the executed command.",
    )
    declared = ArchiveIdea(
        owner_agent_id=actual.owner_agent_id,
        idea_id=actual.idea_id,
        reason="A different reason was declared in the HTTP request.",
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.archive(
            uow=uow,
            command=actual,
            command_context=command_context,
        )

    result = await stack.executor.execute(
        request_for(
            "archive",
            declared,
            key="request-bound-decision",
        ),
        handler,
    )

    assert result.status_code == 404
    assert error_code(result) == "resource_not_found"
    assert await decision_rows(stack, idea_id=idea.idea_id) == ()
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_archived_idea_can_be_rejected_without_reopening_it(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="archive-then-reject-idea")
    archived = await decide(
        stack,
        action="archive",
        key="archive-before-reject",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="Pause this idea before a final owner decision.",
    )
    stack.inspiration.clock.advance(timedelta(minutes=1))

    rejected = await decide(
        stack,
        action="reject",
        key="reject-archived",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="The owner has now rejected the archived idea.",
    )

    assert_success(
        archived,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.ARCHIVED,
    )
    assert_success(
        rejected,
        idea_id=idea.idea_id,
        owner_decision=IdeaOwnerDecision.REJECTED,
    )
    rows = await decision_rows(stack, idea_id=idea.idea_id)
    assert [row.event_type for row in rows] == [
        InspirationIdeaArchivedV1.event_type,
        InspirationIdeaRejectedV1.event_type,
    ]
    rejected_payload = stack.registry.decode(
        event_type=rows[1].event_type,
        payload=rows[1].payload,
    )
    assert isinstance(rejected_payload, InspirationIdeaRejectedV1)
    assert (
        rows[1].sequence,
        rejected_payload.owner_decision_before,
        rejected_payload.owner_decision_after,
    ) == (
        3,
        IdeaOwnerDecision.ARCHIVED,
        IdeaOwnerDecision.REJECTED,
    )
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.REJECTED.value
    assert state.last_signal_at == idea.last_signal_at


@pytest.mark.asyncio
async def test_foreign_missing_and_scope_mismatch_share_the_same_404(
    stack: DecisionStack,
) -> None:
    owned = await seed_idea(stack, key="owned-scope-idea")
    foreign = await seed_idea(
        stack,
        key="foreign-owner-idea",
        owner_agent_id=OTHER_OWNER_ID,
    )
    reason = "Do not reveal whether this private idea exists."

    foreign_result = await decide(
        stack,
        action="reject",
        key="reject-foreign-idea",
        owner_agent_id=OWNER_ID,
        idea_id=foreign.idea_id,
        reason=reason,
    )
    missing_result = await decide(
        stack,
        action="reject",
        key="reject-missing-idea",
        owner_agent_id=OWNER_ID,
        idea_id=UNKNOWN_IDEA_ID,
        reason=reason,
    )
    caller_mismatch = await decide(
        stack,
        action="reject",
        key="reject-caller-mismatch",
        owner_agent_id=OWNER_ID,
        idea_id=owned.idea_id,
        reason=reason,
        caller_scope=f"agent:{OTHER_OWNER_ID}",
    )
    operation_mismatch = await decide(
        stack,
        action="reject",
        key="reject-operation-mismatch",
        owner_agent_id=OWNER_ID,
        idea_id=owned.idea_id,
        reason=reason,
        operation_scope="inspiration.idea.archive",
    )

    results = (
        foreign_result,
        missing_result,
        caller_mismatch,
        operation_mismatch,
    )
    assert {result.status_code for result in results} == {404}
    assert {result.body for result in results} == {foreign_result.body}
    assert {error_code(result) for result in results} == {"resource_not_found"}
    assert await decision_rows(stack, idea_id=owned.idea_id) == ()
    assert await decision_rows(stack, idea_id=foreign.idea_id) == ()
    async with stack.inspiration.database.read_session() as session:
        owned_state = await session.get(IdeaStateRow, owned.idea_id)
        foreign_state = await session.get(IdeaStateRow, foreign.idea_id)
    assert owned_state is not None and foreign_state is not None
    assert {
        owned_state.owner_decision,
        foreign_state.owner_decision,
    } == {IdeaOwnerDecision.ACTIVE.value}


@pytest.mark.asyncio
async def test_foreign_corrupt_projection_is_still_hidden_as_the_same_404(
    stack: DecisionStack,
) -> None:
    foreign = await seed_idea(
        stack,
        key="foreign-corrupt-projection-idea",
        owner_agent_id=OTHER_OWNER_ID,
    )
    async with stack.inspiration.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == foreign.idea_id)
            .values(owner_agent_id=OWNER_ID)
        )
        assert changed.rowcount == 1

    foreign_result = await decide(
        stack,
        action="reject",
        key="reject-foreign-corrupt-projection",
        owner_agent_id=OWNER_ID,
        idea_id=foreign.idea_id,
        reason="Projection health must not reveal a foreign idea.",
    )
    missing_result = await decide(
        stack,
        action="reject",
        key="reject-missing-for-corrupt-projection",
        owner_agent_id=OWNER_ID,
        idea_id=UNKNOWN_IDEA_ID,
        reason="Projection health must not reveal a foreign idea.",
    )

    assert foreign_result.status_code == missing_result.status_code == 404
    assert foreign_result.body == missing_result.body
    assert error_code(foreign_result) == "resource_not_found"
    assert await decision_rows(stack, idea_id=foreign.idea_id) == ()


@pytest.mark.asyncio
async def test_decision_rejects_a_projection_checkpoint_from_another_idea(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="checkpoint-target-idea")
    other = await seed_idea(stack, key="checkpoint-source-idea")
    async with stack.inspiration.database.transaction(immediate=True) as uow:
        other_state = await uow.session.get(IdeaStateRow, other.idea_id)
        assert other_state is not None
        changed = await uow.session.execute(
            update(IdeaStateRow)
            .where(IdeaStateRow.idea_id == idea.idea_id)
            .values(projection_event_id=other_state.projection_event_id)
        )
        assert changed.rowcount == 1

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="projection checkpoint",
    ):
        await decide(
            stack,
            action="reject",
            key="reject-cross-idea-checkpoint",
            owner_agent_id=OWNER_ID,
            idea_id=idea.idea_id,
            reason="A projection checkpoint must belong to its own idea.",
        )

    assert await decision_rows(stack, idea_id=idea.idea_id) == ()


@pytest.mark.asyncio
async def test_adopted_idea_refuses_reject_and_archive_as_one_terminal_state(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="adopted-terminal-idea")
    await arrange_adopted(stack, idea_id=idea.idea_id)

    rejected = await decide(
        stack,
        action="reject",
        key="reject-adopted",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="An adopted idea is terminal.",
    )
    archived = await decide(
        stack,
        action="archive",
        key="archive-adopted",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="An adopted idea cannot be archived.",
    )

    assert rejected.status_code == archived.status_code == 409
    assert error_code(rejected) == error_code(archived) == "idea_not_decidable"
    assert await decision_rows(stack, idea_id=idea.idea_id) == ()
    async with stack.inspiration.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ADOPTED.value
    for action, key in (
        ("reject", "reject-adopted"),
        ("archive", "archive-adopted"),
    ):
        receipt = await receipt_for(
            stack,
            operation_scope=f"inspiration.idea.{action}",
            key=key,
        )
        assert receipt.response_status_code == 409
        assert receipt.result_resource_type is None
        assert receipt.result_resource_id is None


@pytest.mark.parametrize("action", ("reject", "archive"))
@pytest.mark.asyncio
async def test_decision_clock_regression_is_atomic(
    stack: DecisionStack,
    action: DecisionAction,
) -> None:
    idea = await seed_idea(stack, key=f"{action}-regression-idea")
    stack.inspiration.clock.advance(-MICROSECOND)
    async with stack.inspiration.database.read_session() as session:
        state_before = await session.get(IdeaStateRow, idea.idea_id)
        assert state_before is not None
        projection_event_id = state_before.projection_event_id
        event_count_before = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow)) or 0
        )

    result = await decide(
        stack,
        action=action,
        key=f"{action}-clock-regression",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="The command clock must not move behind the idea.",
    )

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    assert await decision_rows(stack, idea_id=idea.idea_id) == ()
    async with stack.inspiration.database.read_session() as session:
        state_after = await session.get(IdeaStateRow, idea.idea_id)
        event_count_after = int(
            await session.scalar(select(func.count()).select_from(DomainEventRow)) or 0
        )
    assert state_after is not None
    assert (
        state_after.owner_decision,
        state_after.decision_reason,
        state_after.last_signal_at,
        state_after.projection_event_id,
    ) == (
        IdeaOwnerDecision.ACTIVE.value,
        None,
        idea.last_signal_at,
        projection_event_id,
    )
    assert event_count_after == event_count_before
    receipt = await receipt_for(
        stack,
        operation_scope=f"inspiration.idea.{action}",
        key=f"{action}-clock-regression",
    )
    assert receipt.state == "completed"
    assert receipt.response_status_code == 409
    assert receipt.result_resource_type is None
    assert receipt.result_resource_id is None


@pytest.mark.asyncio
async def test_server_timed_decision_never_precedes_its_receipt(
    stack: DecisionStack,
) -> None:
    idea = await seed_idea(stack, key="decision-receipt-clock-idea")
    receipt_time = stack.inspiration.clock.now() + timedelta(minutes=10)
    clock = RegressingClock(
        values=[
            receipt_time,
            receipt_time - timedelta(minutes=1),
            receipt_time + timedelta(minutes=1),
        ]
    )
    stack.receipts._clock = clock  # noqa: SLF001
    stack.executor._clock = clock  # noqa: SLF001
    stack.service._clock = clock  # noqa: SLF001

    result = await decide(
        stack,
        action="archive",
        key="decision-receipt-clock",
        owner_agent_id=OWNER_ID,
        idea_id=idea.idea_id,
        reason="Archive at the command receipt boundary.",
    )

    assert result.status_code == 200
    [event] = await decision_rows(stack, idea_id=idea.idea_id)
    receipt = await receipt_for(
        stack,
        operation_scope="inspiration.idea.archive",
        key="decision-receipt-clock",
    )
    assert event.occurred_at == receipt.created_at == receipt_time
    assert receipt.completed_at == receipt_time + timedelta(minutes=1)
