from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_inspiration_run import (
    NOW,
    FakeGenerator,
    FakeSnapshotBuilder,
    Stack,
    build_stack,
    command,
    row_counts,
)

from experience_hub.clock import FrozenClock
from experience_hub.domain import CommandRequest
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.deadlines import DeadlineRunner
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.generators.openai_compatible import (
    GeneratorNotConfiguredError,
)
from experience_hub.inspiration.models import (
    GeneratorKind,
    IdeaDraft,
    InspirationOperator,
    InspirationRunStatus,
    SnapshotItem,
)
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import IdempotencyKeyConflict
from experience_hub.storage.tables import (
    DomainEventRow,
    IdempotencyRecordRow,
    InspirationRunRow,
    InspirationRunStateRow,
)


class InjectedFinalizationFailure(RuntimeError):
    """A deterministic final-transaction interruption."""


class InjectedTransactionFailure(RuntimeError):
    """A deterministic T1 or T2 transaction interruption."""


def _request_for(
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
            "operators": [operator.value for operator in run.operators],
            "include_inbox": run.include_inbox,
            "branches_per_operator": run.branches_per_operator,
            "output_tokens_per_operator": run.output_tokens_per_operator,
            "total_output_tokens": run.total_output_tokens,
            "operator_timeout_seconds": run.operator_timeout_seconds,
            "global_timeout_seconds": run.global_timeout_seconds,
        },
    )


@dataclass(slots=True)
class FailNthCheckpoint:
    checkpoint: FaultCheckpoint
    ordinal: int
    error: BaseException
    seen: int = 0

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is not self.checkpoint:
            return
        self.seen += 1
        if self.seen == self.ordinal:
            raise self.error


@dataclass(slots=True)
class ClockChangingSnapshotBuilder(FakeSnapshotBuilder):
    clock: FrozenClock | None = None
    delta: timedelta = timedelta()

    async def freeze(
        self,
        *,
        uow: Any,
        request: StartInspirationRun,
        run_id: Any,
        at: Any,
    ) -> Any:
        retained = await FakeSnapshotBuilder.freeze(
            self,
            uow=uow,
            request=request,
            run_id=run_id,
            at=at,
        )
        if self.clock is None:
            raise RuntimeError("test clock was not attached")
        self.clock.advance(self.delta)
        return retained


@dataclass(slots=True)
class ClockChangingDeadlineRunner(DeadlineRunner):
    delta: timedelta
    clock: FrozenClock | None = None

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        timeout_seconds: float,
    ) -> Any:
        _ = timeout_seconds
        if self.clock is None:
            raise RuntimeError("test clock was not attached")
        self.clock.advance(self.delta)
        return await operation()


@dataclass(slots=True)
class SameMechanismGenerator(FakeGenerator):
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
        generated = await FakeGenerator.generate(
            self,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operator=operator,
            branch_limit=branch_limit,
            output_token_limit=output_token_limit,
        )
        draft = generated.ideas[0]
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title=draft.title,
                    hypothesis=draft.hypothesis,
                    mechanism="Acknowledgement releases shared bounded capacity.",
                    predictions=draft.predictions,
                    falsifiers=draft.falsifiers,
                    assumptions=draft.assumptions,
                    proposed_test=draft.proposed_test,
                    evidence=draft.evidence,
                ),
            ),
            output_tokens_consumed=generated.output_tokens_consumed,
        )


@dataclass(slots=True)
class ReservingGenerator(FakeGenerator):
    @property
    def reserves_output_tokens(self) -> bool:
        return True

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
        generated = await FakeGenerator.generate(
            self,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operator=operator,
            branch_limit=branch_limit,
            output_token_limit=output_token_limit,
        )
        return GeneratorResult(
            ideas=generated.ideas,
            output_tokens_consumed=output_token_limit,
        )


@dataclass(slots=True)
class ReleasingReservationGenerator(FakeGenerator):
    @property
    def reserves_output_tokens(self) -> bool:
        return True


def _forbidden_factory(_: GeneratorKind) -> FakeGenerator:
    raise AssertionError("generator factory was consulted")


@pytest.fixture
async def run_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-adversarial.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def _assert_prepared_trace_is_retained(stack: Stack) -> None:
    async with stack.database.read_session() as session:
        events = (
            await session.scalars(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).all()
        receipt = await session.scalar(select(IdempotencyRecordRow))
        state = await session.scalar(select(InspirationRunStateRow))

    assert tuple(event.event_type for event in events) == (
        "inspiration.started",
        "inspiration.snapshot_frozen",
    )
    assert {event.occurred_at for event in events} == {NOW}
    assert receipt is not None
    assert receipt.state == "in_progress"
    assert receipt.response_body is None
    assert state is not None and state.status == InspirationRunStatus.RUNNING
    assert await row_counts(stack) == {
        "runs": 1,
        "snapshots": 1,
        "ideas": 0,
        "occurrences": 0,
        "run_state": 1,
        "idea_state": 0,
        "clusters": 0,
        "events": 2,
        "receipts": 1,
    }


async def _assert_started_trace_is_retained(stack: Stack) -> None:
    async with stack.database.read_session() as session:
        events = (
            await session.scalars(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).all()
        receipt = await session.scalar(select(IdempotencyRecordRow))
        state = await session.scalar(select(InspirationRunStateRow))

    assert tuple(event.event_type for event in events) == ("inspiration.started",)
    assert events[0].occurred_at == NOW
    assert receipt is not None
    assert receipt.state == "in_progress"
    assert receipt.response_body is None
    assert receipt.result_resource_type == "inspiration_run"
    assert receipt.result_resource_id == events[0].aggregate_id
    assert state is not None and state.status == InspirationRunStatus.RUNNING
    assert await row_counts(stack) == {
        "runs": 1,
        "snapshots": 0,
        "ideas": 0,
        "occurrences": 0,
        "run_state": 1,
        "idea_state": 0,
        "clusters": 0,
        "events": 1,
        "receipts": 1,
    }


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("goal", " forged goal"),
        (
            "operators",
            (
                InspirationOperator.COUNTERFACTUAL,
                InspirationOperator.CAUSAL_GAP,
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_forged_internal_start_is_rejected_before_receipt_reservation(
    run_stack: Stack,
    field_name: str,
    forged_value: object,
) -> None:
    forged = command()
    object.__setattr__(forged, field_name, forged_value)

    with pytest.raises(ValueError, match="valid StartInspirationRun"):
        await run_stack.executor.execute(
            request=_request_for(command(), key=f"forged-{field_name}"),
            run=forged,
        )

    counts = await row_counts(run_stack)
    assert counts["receipts"] == counts["runs"] == counts["events"] == 0
    assert run_stack.factory_calls == []


@pytest.mark.parametrize("unexpected_semantics", ["query", "header"])
@pytest.mark.asyncio
async def test_unexpected_request_semantics_are_rejected_before_reservation(
    run_stack: Stack,
    unexpected_semantics: str,
) -> None:
    selected = command()
    base = _request_for(
        selected,
        key=f"unexpected-{unexpected_semantics}",
    )
    invocation = CommandRequest(
        caller_scope=base.caller_scope,
        operation_scope=base.operation_scope,
        idempotency_key=base.idempotency_key,
        method=base.method,
        route_template=base.route_template,
        path_parameters=base.path_parameters,
        query_parameters=(
            (("unexpected", "1"),) if unexpected_semantics == "query" else ()
        ),
        body=base.body,
        semantic_headers=(
            {"x-unexpected": "1"} if unexpected_semantics == "header" else {}
        ),
    )

    with pytest.raises(ValueError, match="request"):
        await run_stack.executor.execute(request=invocation, run=selected)

    counts = await row_counts(run_stack)
    assert counts["receipts"] == counts["runs"] == counts["events"] == 0
    assert run_stack.factory_calls == []


@pytest.mark.asyncio
async def test_completed_in_progress_and_conflict_decisions_precede_factory(
    run_stack: Stack,
) -> None:
    selected = command()
    completed_request = _request_for(selected, key="decision-completed")
    completed = await run_stack.executor.execute(
        request=completed_request,
        run=selected,
    )

    run_stack.generator.cancellation = True
    run_stack.snapshot_builder.item_ids.append(
        UUID("00000000-0000-0000-0000-000000000402")
    )
    progress_request = _request_for(selected, key="decision-in-progress")
    with pytest.raises(asyncio.CancelledError):
        await run_stack.executor.execute(
            request=progress_request,
            run=selected,
        )

    factory_calls = tuple(run_stack.factory_calls)
    run_stack.executor._generator_factory = _forbidden_factory  # noqa: SLF001

    replay = await run_stack.executor.execute(
        request=completed_request,
        run=selected,
    )
    in_progress = await run_stack.executor.execute(
        request=progress_request,
        run=selected,
    )
    conflicting = command(goal="A different request")
    with pytest.raises(IdempotencyKeyConflict):
        await run_stack.executor.execute(
            request=_request_for(
                conflicting,
                key="decision-completed",
            ),
            run=conflicting,
        )

    assert replay == completed
    assert in_progress.status_code == 409
    assert json.loads(in_progress.body)["error"]["code"] == "operation_in_progress"
    assert tuple(run_stack.factory_calls) == factory_calls


@pytest.mark.asyncio
async def test_unconfigured_422_is_byte_exact_and_replays_without_factory(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "adversarial-unconfigured.sqlite3",
        factory_error=GeneratorNotConfiguredError(),
    )
    selected = command(generator=GeneratorKind.OPENAI_COMPATIBLE)
    invocation = _request_for(selected, key="adversarial-not-configured")
    expected = (
        b'{"error":{"code":"generator_not_configured","details":{},'
        b'"message":"The selected inspiration generator is not configured."}}'
    )
    try:
        first = await value.executor.execute(request=invocation, run=selected)
        value.executor._generator_factory = _forbidden_factory  # noqa: SLF001
        value.clock.advance(timedelta(days=1))
        replay = await value.executor.execute(request=invocation, run=selected)

        assert first == replay
        assert first.status_code == 422
        assert first.body == expected
        assert tuple(value.factory_calls) == (GeneratorKind.OPENAI_COMPATIBLE,)
        async with value.database.read_session() as session:
            receipt = await session.scalar(select(IdempotencyRecordRow))
        assert receipt is not None and receipt.state == "completed"
        assert receipt.response_body == expected
        counts = await row_counts(value)
        assert counts["runs"] == counts["events"] == 0
        assert counts["receipts"] == 1
    finally:
        await value.database.dispose()


@pytest.mark.parametrize(
    ("after_start", "after_snapshot"),
    [
        (timedelta(minutes=5), timedelta(minutes=-11)),
        (timedelta(minutes=-5), timedelta(minutes=11)),
    ],
)
@pytest.mark.asyncio
async def test_domain_clock_changes_after_t1_and_t2_do_not_change_run_time(
    repository_root: Path,
    tmp_path: Path,
    after_start: timedelta,
    after_snapshot: timedelta,
) -> None:
    builder = ClockChangingSnapshotBuilder(delta=after_start)
    deadline = ClockChangingDeadlineRunner(delta=after_snapshot)
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path
        / (
            f"clock-{after_start.total_seconds()}-"
            f"{after_snapshot.total_seconds()}.sqlite3"
        ),
        snapshot_builder=builder,
        deadline_runner=deadline,
    )
    builder.clock = value.clock
    deadline.clock = value.clock
    try:
        selected = command()
        result = await value.executor.execute(
            request=_request_for(
                selected,
                key=f"clock-{after_start.total_seconds()}",
            ),
            run=selected,
        )

        assert json.loads(result.body)["data"]["status"] == "completed"
        async with value.database.read_session() as session:
            events = (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
            run_row = await session.scalar(select(InspirationRunRow))
            run_state = await session.scalar(select(InspirationRunStateRow))
            receipt = await session.scalar(select(IdempotencyRecordRow))
        assert events and {event.occurred_at for event in events} == {NOW}
        assert run_row is not None
        assert run_row.created_at == NOW
        assert run_state is not None and run_state.completed_at == NOW
        assert receipt is not None
        assert receipt.created_at == receipt.completed_at == NOW
    finally:
        await value.database.dispose()


@pytest.mark.parametrize(
    "checkpoint",
    [
        FaultCheckpoint.AFTER_SOURCE_INSERT,
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
    ],
)
@pytest.mark.asyncio
async def test_t1_fault_rolls_back_reservation_and_run_and_closes_generator(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"t1-{checkpoint.value}.sqlite3",
    )
    fault = FailNthCheckpoint(
        checkpoint=checkpoint,
        ordinal=1,
        error=InjectedTransactionFailure(checkpoint.value),
    )
    value.database._fault_injector = fault  # noqa: SLF001
    selected = command()
    try:
        with pytest.raises(
            InjectedTransactionFailure,
            match=checkpoint.value,
        ):
            await value.executor.execute(
                request=_request_for(
                    selected,
                    key=f"t1-{checkpoint.value}",
                ),
                run=selected,
            )

        assert fault.seen == 1
        assert await row_counts(value) == {
            "runs": 0,
            "snapshots": 0,
            "ideas": 0,
            "occurrences": 0,
            "run_state": 0,
            "idea_state": 0,
            "clusters": 0,
            "events": 0,
            "receipts": 0,
        }
        assert value.factory_calls == [GeneratorKind.DETERMINISTIC]
        assert value.generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.parametrize(
    "checkpoint",
    [
        FaultCheckpoint.AFTER_SOURCE_INSERT,
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
    ],
)
@pytest.mark.asyncio
async def test_t2_fault_propagates_and_rolls_back_only_snapshot_transaction(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"t2-{checkpoint.value}.sqlite3",
    )
    fault = FailNthCheckpoint(
        checkpoint=checkpoint,
        ordinal=2,
        error=InjectedTransactionFailure(checkpoint.value),
    )
    value.database._fault_injector = fault  # noqa: SLF001
    selected = command()
    try:
        with pytest.raises(
            InjectedTransactionFailure,
            match=checkpoint.value,
        ):
            await value.executor.execute(
                request=_request_for(
                    selected,
                    key=f"t2-{checkpoint.value}",
                ),
                run=selected,
            )

        assert fault.seen == 2
        await _assert_started_trace_is_retained(value)
        assert value.generator.calls == []
        assert value.generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_snapshot_builder_external_cancellation_propagates_with_t1_trace(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    builder = FakeSnapshotBuilder(failure=asyncio.CancelledError())
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "t2-snapshot-cancelled.sqlite3",
        snapshot_builder=builder,
    )
    selected = command()
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=_request_for(selected, key="t2-snapshot-cancelled"),
                run=selected,
            )

        await _assert_started_trace_is_retained(value)
        assert len(builder.sessions) == 1
        assert value.generator.calls == []
        assert value.generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.parametrize(
    ("checkpoint", "ordinal"),
    [
        (FaultCheckpoint.AFTER_SOURCE_INSERT, 3),
        (FaultCheckpoint.AFTER_EVENT_APPEND, 3),
        (FaultCheckpoint.AFTER_PROJECTION_APPLY, 3),
        (FaultCheckpoint.AFTER_RECEIPT_COMPLETION, 1),
    ],
)
@pytest.mark.asyncio
async def test_final_transaction_fault_rolls_back_every_component(
    repository_root: Path,
    tmp_path: Path,
    checkpoint: FaultCheckpoint,
    ordinal: int,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"t3-{checkpoint.value}.sqlite3",
    )
    fault = FailNthCheckpoint(
        checkpoint=checkpoint,
        ordinal=ordinal,
        error=InjectedFinalizationFailure(checkpoint.value),
    )
    value.database._fault_injector = fault  # noqa: SLF001
    try:
        with pytest.raises(
            InjectedFinalizationFailure,
            match=checkpoint.value,
        ):
            selected = command()
            await value.executor.execute(
                request=_request_for(
                    selected,
                    key=f"t3-{checkpoint.value}",
                ),
                run=selected,
            )

        assert fault.seen == ordinal
        await _assert_prepared_trace_is_retained(value)
        assert value.generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_external_cancellation_in_final_transaction_propagates_and_rolls_back(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "t3-cancelled.sqlite3",
    )
    fault = FailNthCheckpoint(
        checkpoint=FaultCheckpoint.AFTER_PROJECTION_APPLY,
        ordinal=3,
        error=asyncio.CancelledError(),
    )
    value.database._fault_injector = fault  # noqa: SLF001
    try:
        with pytest.raises(asyncio.CancelledError):
            selected = command()
            await value.executor.execute(
                request=_request_for(selected, key="t3-cancelled"),
                run=selected,
            )

        await _assert_prepared_trace_is_retained(value)
        assert value.generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_all_provider_failures_produce_failed_terminal_trace(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    operators = (
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.COUNTERFACTUAL,
    )
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "all-failed.sqlite3",
        generator=FakeGenerator(fail_operators=frozenset(operators)),
    )
    try:
        selected = command(operators=operators)
        result = await value.executor.execute(
            request=_request_for(selected, key="all-failed"),
            run=selected,
        )
        data = json.loads(result.body)["data"]
        assert data["status"] == "failed"
        assert [outcome["error_code"] for outcome in data["operator_outcomes"]] == [
            "generator_error",
            "generator_error",
        ]

        async with value.database.read_session() as session:
            events = (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        assert tuple(event.event_type for event in events) == (
            "inspiration.started",
            "inspiration.snapshot_frozen",
            "inspiration.operator_failed",
            "inspiration.operator_failed",
            "inspiration.failed",
        )
        terminal = json.loads(events[-1].payload)
        assert terminal["failure_code"] == "all_operators_failed"
        assert b"raw provider secret" not in result.body
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_operator_with_only_cross_operator_duplicates_is_no_valid_branches(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    operators = (
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.COUNTERFACTUAL,
    )
    generator = SameMechanismGenerator()
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "no-valid-branches.sqlite3",
        generator=generator,
    )
    try:
        selected = command(operators=operators)
        result = await value.executor.execute(
            request=_request_for(selected, key="no-valid-branches"),
            run=selected,
        )
        data = json.loads(result.body)["data"]

        assert data["status"] == "completed_with_errors"
        assert data["operator_outcomes"][0]["succeeded"] is True
        assert data["operator_outcomes"][1] == {
            "operator": "counterfactual",
            "succeeded": False,
            "persisted_ideas": 0,
            "duplicate_count": 1,
            "error_code": "no_valid_branches",
            "output_tokens_consumed": 0,
        }
        async with value.database.read_session() as session:
            event_types = tuple(
                await session.scalars(
                    select(DomainEventRow.event_type).order_by(DomainEventRow.event_id)
                )
            )
        assert event_types == (
            "inspiration.started",
            "inspiration.snapshot_frozen",
            "inspiration.idea_generated",
            "inspiration.operator_completed",
            "inspiration.operator_failed",
            "inspiration.completed",
        )
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_unavailable_second_token_reservation_is_skipped_and_retained(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    operators = (
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.COUNTERFACTUAL,
    )
    generator = ReservingGenerator()
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "insufficient-reservation.sqlite3",
        generator=generator,
    )
    selected = StartInspirationRun(
        owner_agent_id=command().owner_agent_id,
        goal="Find a robust bridge",
        operators=operators,
        output_tokens_per_operator=800,
        total_output_tokens=800,
    )
    try:
        result = await value.executor.execute(
            request=_request_for(selected, key="insufficient-reservation"),
            run=selected,
        )
        data = json.loads(result.body)["data"]

        assert data["status"] == "completed_with_errors"
        assert data["output_tokens_reserved"] == 800
        assert data["output_tokens_consumed"] == 800
        assert data["operator_outcomes"][1]["error_code"] == (
            "insufficient_token_reservation"
        )
        assert generator.calls == [InspirationOperator.CAUSAL_GAP]

        async with value.database.read_session() as session:
            failed = await session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.operator_failed"
                )
            )
        assert failed is not None
        payload = json.loads(failed.payload)
        assert payload["output_tokens_reserved_before"] == 800
        assert payload["output_tokens_reserved_after"] == 800
        assert payload["outcome"]["error_code"] == ("insufficient_token_reservation")
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_released_unused_reservations_allow_all_operators_and_retain_peak_sum(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = ReleasingReservationGenerator()
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "released-reservations.sqlite3",
        generator=generator,
    )
    selected = StartInspirationRun(
        owner_agent_id=command().owner_agent_id,
        goal="Find a robust bridge",
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
            InspirationOperator.DISTANT_ANALOGY,
        ),
        output_tokens_per_operator=1_200,
        total_output_tokens=1_200,
    )
    try:
        result = await value.executor.execute(
            request=_request_for(selected, key="released-reservations"),
            run=selected,
        )
        data = json.loads(result.body)["data"]

        assert result.status_code == 201
        assert data["status"] == "completed"
        assert data["output_tokens_reserved"] == 3_600
        assert data["output_tokens_consumed"] == 0
        assert generator.calls == list(selected.operators)
        assert all(
            outcome["succeeded"] is True for outcome in data["operator_outcomes"]
        )
    finally:
        await value.database.dispose()
