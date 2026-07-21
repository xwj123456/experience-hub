from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import CommandRequest, EventRegistry
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
    register_experience_events,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.deadlines import (
    BoundedGenerationRunner,
    DeadlineExpired,
    DeadlineRunner,
)
from experience_hub.inspiration.events import register_inspiration_events
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.generators.openai_compatible import (
    GeneratorNotConfiguredError,
)
from experience_hub.inspiration.hashing import hash_snapshot, stable_evidence_key
from experience_hub.inspiration.models import (
    EvidenceSourceState,
    EvidenceSourceType,
    FrozenSnapshot,
    GeneratorKind,
    IdeaDraft,
    InspirationOperator,
    InspirationRunStatus,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.repository import InspirationRepository
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.service import InspirationRunExecutor
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import ReceiptStore
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
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
    InspirationRunRow,
    InspirationRunStateRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceValidator

NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
RUN_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(301, 321)
)
RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(201, 221)
)
SNAPSHOT_ITEM_ID = UUID("00000000-0000-0000-0000-000000000401")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000501")
SOURCE_VERSION_ID = UUID("00000000-0000-0000-0000-000000000502")
SOURCE_CONTENT = VersionContent(
    body="A bounded observation.",
    summary="The queue drains when work is acknowledged.",
    mechanism="Acknowledgement releases capacity.",
    tags=("queue",),
    applicability=("bounded queue",),
    evidence=(),
    falsifiers=("Capacity remains blocked after acknowledgement.",),
)


@dataclass(slots=True)
class RegressingClock:
    values: list[datetime]

    def now(self) -> datetime:
        if not self.values:
            raise AssertionError("regressing clock was sampled too many times")
        return self.values.pop(0)


ENCODED_SOURCE_CONTENT = encode_version_content(
    kind=ExperienceKind.SEMANTIC,
    content=SOURCE_CONTENT,
)
CONTENT_HASH = ENCODED_SOURCE_CONTENT.content_hash


def request(
    *,
    key: str,
    goal: str = "Find a robust bridge",
    run: StartInspirationRun | None = None,
) -> CommandRequest:
    selected = run or command(goal=goal)
    return CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope="inspiration.run.start",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": OWNER_ID},
        body={
            "goal": selected.goal,
            "context": selected.context,
            "mode": selected.mode.value,
            "generator": selected.generator.value,
            "operators": tuple(operator.value for operator in selected.operators),
            "include_inbox": selected.include_inbox,
            "branches_per_operator": selected.branches_per_operator,
            "output_tokens_per_operator": selected.output_tokens_per_operator,
            "total_output_tokens": selected.total_output_tokens,
            "operator_timeout_seconds": selected.operator_timeout_seconds,
            "global_timeout_seconds": selected.global_timeout_seconds,
        },
    )


def command(
    *,
    goal: str = "Find a robust bridge",
    operators: tuple[InspirationOperator, ...] = (InspirationOperator.CAUSAL_GAP,),
    generator: GeneratorKind = GeneratorKind.DETERMINISTIC,
) -> StartInspirationRun:
    return StartInspirationRun(
        owner_agent_id=OWNER_ID,
        goal=goal,
        generator=generator,
        operators=operators,
    )


def snapshot(
    *,
    run_id: UUID,
    at: datetime,
    snapshot_item_id: UUID = SNAPSHOT_ITEM_ID,
    content_hash: str = CONTENT_HASH,
    source_type: EvidenceSourceType = EvidenceSourceType.EXPERIENCE,
    source_id: UUID = SOURCE_ID,
    source_version_id: UUID = SOURCE_VERSION_ID,
) -> FrozenSnapshot:
    source_state = (
        EvidenceSourceState.WARM
        if source_type is EvidenceSourceType.EXPERIENCE
        else EvidenceSourceState.QUARANTINED
    )
    source_trust = 1.0 if source_type is EvidenceSourceType.EXPERIENCE else 0.25
    stable_key = stable_evidence_key(
        source_type=source_type,
        source_id=source_id,
        source_version_id=source_version_id,
        content_hash=content_hash,
    )
    item = SnapshotItem(
        snapshot_item_id=snapshot_item_id,
        stable_evidence_key=stable_key,
        run_id=run_id,
        source_type=source_type,
        source_id=source_id,
        source_version_id=source_version_id,
        source_state=source_state,
        source_trust=source_trust,
        rank=1,
        summary="The queue drains when work is acknowledged.",
        mechanism="Acknowledgement releases capacity.",
        applicability=("bounded queue",),
        tags=("queue",),
        falsifiers=("Capacity remains blocked after acknowledgement.",),
        excerpt="A bounded observation.",
        content_hash=content_hash,
        captured_at=at,
    )
    return FrozenSnapshot(
        run_id=run_id,
        items=(item,),
        snapshot_hash=hash_snapshot((item,)),
        frozen_at=at,
    )


@dataclass(slots=True)
class FakeSnapshotBuilder:
    failure: BaseException | None = None
    source_type: EvidenceSourceType = EvidenceSourceType.EXPERIENCE
    item_ids: list[UUID] = field(default_factory=list)
    source_ids: list[UUID] = field(default_factory=list)
    source_version_ids: list[UUID] = field(default_factory=list)
    content_hashes: list[str] = field(default_factory=list)
    sessions: list[object] = field(default_factory=list)

    async def freeze(
        self,
        *,
        uow: UnitOfWork,
        request: StartInspirationRun,
        run_id: UUID,
        at: datetime,
    ) -> FrozenSnapshot:
        _ = request
        self.sessions.append(uow.session)
        if self.failure is not None:
            raise self.failure
        item_id = self.item_ids.pop(0) if self.item_ids else SNAPSHOT_ITEM_ID
        source_id = self.source_ids.pop(0) if self.source_ids else SOURCE_ID
        source_version_id = (
            self.source_version_ids.pop(0)
            if self.source_version_ids
            else SOURCE_VERSION_ID
        )
        content_hash = (
            self.content_hashes.pop(0) if self.content_hashes else CONTENT_HASH
        )
        return snapshot(
            run_id=run_id,
            at=at,
            snapshot_item_id=item_id,
            content_hash=content_hash,
            source_type=self.source_type,
            source_id=source_id,
            source_version_id=source_version_id,
        )


@dataclass(slots=True)
class FakeGenerator:
    fail_operators: frozenset[InspirationOperator] = frozenset()
    cancellation: bool = False
    closed: bool = False
    calls: list[InspirationOperator] = field(default_factory=list)

    @property
    def reserves_output_tokens(self) -> bool:
        return False

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        self.closed = True

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
        if self.cancellation:
            raise asyncio.CancelledError
        if operator in self.fail_operators:
            raise RuntimeError("raw provider secret")
        item = frozen_items[0]
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title=f"{operator.value} branch",
                    hypothesis=f"{operator.value} can expose a testable bridge.",
                    mechanism=f"{operator.value} acknowledgement capacity bridge",
                    predictions=("Capacity changes after acknowledgement.",),
                    falsifiers=("Capacity is unchanged.",),
                    assumptions=("The queue is bounded.",),
                    proposed_test="Compare capacity before and after acknowledgement.",
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


class ImmediateDeadlineRunner(DeadlineRunner):
    async def run(
        self,
        operation: Callable[[], Any],
        *,
        timeout_seconds: float,
    ) -> Any:
        _ = timeout_seconds
        return await operation()


class ExpiringDeadlineRunner(DeadlineRunner):
    async def run(
        self,
        operation: Callable[[], Any],
        *,
        timeout_seconds: float,
    ) -> Any:
        _ = (operation, timeout_seconds)
        raise DeadlineExpired


@dataclass(slots=True)
class Stack:
    database: Database
    executor: InspirationRunExecutor
    generator: FakeGenerator
    snapshot_builder: FakeSnapshotBuilder
    clock: FrozenClock
    factory_calls: list[GeneratorKind]


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
    generator: FakeGenerator | None = None,
    snapshot_builder: FakeSnapshotBuilder | None = None,
    factory_error: BaseException | None = None,
    deadline_runner: DeadlineRunner | None = None,
    register_additional_events: Callable[[EventRegistry], None] | None = None,
    seed_experience_source_trace: bool = False,
) -> Stack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    register_inspiration_events(registry)
    if seed_experience_source_trace:
        register_experience_events(registry)
    if register_additional_events is not None:
        register_additional_events(registry)
    projection_registry = ProjectionRegistry(
        (
            InspirationRunProjector(registry),
            MechanismIncubationProjector(registry),
            IdeaStateProjector(registry),
        )
    )
    manager = ProjectionManager(
        projection_registry,
        source_validator=SourceValidator(registry),
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
    )
    async with database.transaction() as uow:
        uow.session.add(
            AgentRow(
                agent_id=OWNER_ID,
                name="Owner",
                created_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceRow(
                experience_id=SOURCE_ID,
                owner_agent_id=OWNER_ID,
                kind=ExperienceKind.SEMANTIC,
                origin=ExperienceOrigin.LOCAL,
                created_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceVersionRow(
                version_id=SOURCE_VERSION_ID,
                experience_id=SOURCE_ID,
                version_number=1,
                summary=SOURCE_CONTENT.summary,
                mechanism=SOURCE_CONTENT.mechanism,
                tags=canonical_json_bytes(SOURCE_CONTENT.tags),
                applicability=canonical_json_bytes(SOURCE_CONTENT.applicability),
                evidence=canonical_json_bytes(SOURCE_CONTENT.evidence),
                falsifiers=canonical_json_bytes(SOURCE_CONTENT.falsifiers),
                content_hash=CONTENT_HASH,
                supersedes_version_id=None,
                created_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperiencePayloadRow(
                version_id=SOURCE_VERSION_ID,
                codec=ENCODED_SOURCE_CONTENT.codec,
                payload=ENCODED_SOURCE_CONTENT.payload,
                payload_hash=ENCODED_SOURCE_CONTENT.payload_hash,
            )
        )
        if seed_experience_source_trace:
            source_receipt_id = UUID("00000000-0000-0000-0000-000000000590")
            uow.session.add(
                IdempotencyRecordRow(
                    receipt_id=source_receipt_id,
                    caller_scope=f"agent:{OWNER_ID}",
                    scope="experience.fixture.create",
                    idempotency_key="inspiration-source",
                    request_hash="d" * 64,
                    state="completed",
                    result_resource_type="experience",
                    result_resource_id=SOURCE_ID,
                    response_status_code=201,
                    response_body=canonical_json_bytes({}),
                    response_content_type="application/json",
                    response_headers=canonical_json_bytes({}),
                    created_at=NOW,
                    completed_at=NOW,
                )
            )
            await uow.session.flush()
            state = ExperienceStateSnapshotV1(
                experience_id=SOURCE_ID,
                owner_agent_id=OWNER_ID,
                current_version_id=SOURCE_VERSION_ID,
                current_content_hash=CONTENT_HASH,
                temperature=Temperature.WARM,
                importance=0.35,
                confidence=0.5,
                activation_score=0.5,
                source_trust=1.0,
                access_count=0,
                access_strength=0.0,
                strength_updated_at=NOW,
                last_accessed_at=None,
                last_transition_at=NOW,
                last_lifecycle_evaluated_at=None,
                consecutive_below_threshold=0,
                pinned=False,
            )
            uow.session.add(
                DomainEventRow(
                    aggregate_type="experience",
                    aggregate_id=SOURCE_ID,
                    sequence=1,
                    event_type=ExperienceCreatedV1.event_type,
                    payload=canonical_json_bytes(
                        ExperienceCreatedV1(
                            schema_version=1,
                            experience_id=SOURCE_ID,
                            version_id=SOURCE_VERSION_ID,
                            after=state,
                        ).model_dump(mode="json")
                    ),
                    actor_agent_id=OWNER_ID,
                    causation_id=source_receipt_id,
                    occurred_at=NOW,
                )
            )
            await uow.session.flush()
            uow.session.add(
                DomainEventRow(
                    aggregate_type="experience",
                    aggregate_id=SOURCE_ID,
                    sequence=2,
                    event_type=ExperienceVersionCreatedV1.event_type,
                    payload=canonical_json_bytes(
                        ExperienceVersionCreatedV1(
                            schema_version=1,
                            experience_id=SOURCE_ID,
                            version_id=SOURCE_VERSION_ID,
                            version_number=1,
                            supersedes_version_id=None,
                            links=(),
                            before=state,
                            after=state,
                        ).model_dump(mode="json")
                    ),
                    actor_agent_id=OWNER_ID,
                    causation_id=source_receipt_id,
                    occurred_at=NOW,
                )
            )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    selected = generator or FakeGenerator()
    builder = snapshot_builder or FakeSnapshotBuilder()
    factory_calls: list[GeneratorKind] = []

    def factory(kind: GeneratorKind) -> FakeGenerator:
        factory_calls.append(kind)
        if factory_error is not None:
            raise factory_error
        return selected

    executor = InspirationRunExecutor(
        database=database,
        receipt_store=receipts,
        repository=InspirationRepository(registry),
        snapshot_builder=builder,
        generator_factory=factory,
        generation_runner=BoundedGenerationRunner(
            deadline_runner=deadline_runner or ImmediateDeadlineRunner(),
        ),
        response_codec=InspirationResponseCodec(),
        clock=clock,
        id_generator=SequenceIdGenerator(RUN_IDS),
    )
    return Stack(
        database=database,
        executor=executor,
        generator=selected,
        snapshot_builder=builder,
        clock=clock,
        factory_calls=factory_calls,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-run.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def row_counts(stack: Stack) -> dict[str, int]:
    tables = {
        "runs": InspirationRunRow,
        "snapshots": InspirationSnapshotItemRow,
        "ideas": InspirationIdeaRow,
        "occurrences": IdeaOccurrenceRow,
        "run_state": InspirationRunStateRow,
        "idea_state": IdeaStateRow,
        "clusters": MechanismIncubationRow,
        "events": DomainEventRow,
        "receipts": IdempotencyRecordRow,
    }
    async with stack.database.read_session() as session:
        return {
            name: int(
                await session.scalar(select(func.count()).select_from(table)) or 0
            )
            for name, table in tables.items()
        }


@pytest.mark.asyncio
async def test_success_commits_golden_three_transaction_trace_and_replays(
    stack: Stack,
) -> None:
    invocation = request(key="success")
    first = await stack.executor.execute(request=invocation, run=command())

    assert first.status_code == 201
    assert first.content_type == "application/json"
    body = json.loads(first.body)
    assert body["data"]["status"] == InspirationRunStatus.COMPLETED.value
    assert body["data"]["created_at"] == "2026-07-18T08:30:00.000000Z"
    assert body["data"]["completed_at"] == "2026-07-18T08:30:00.000000Z"
    run_id = UUID(body["data"]["run_id"])
    assert first.headers == {
        "location": (f"/v1/agents/{OWNER_ID}/inspiration-runs/{run_id}"),
    }

    async with stack.database.read_session() as session:
        rows = (
            await session.scalars(
                select(DomainEventRow).order_by(DomainEventRow.event_id)
            )
        ).all()
    assert tuple(row.event_type for row in rows) == (
        "inspiration.started",
        "inspiration.snapshot_frozen",
        "inspiration.idea_generated",
        "inspiration.operator_completed",
        "inspiration.completed",
    )
    assert {row.occurred_at for row in rows} == {NOW}
    assert len({row.causation_id for row in rows}) == 1
    assert tuple(row.aggregate_type for row in rows) == (
        "inspiration_run",
        "inspiration_run",
        "idea",
        "inspiration_run",
        "inspiration_run",
    )
    assert await row_counts(stack) == {
        "runs": 1,
        "snapshots": 1,
        "ideas": 1,
        "occurrences": 1,
        "run_state": 1,
        "idea_state": 1,
        "clusters": 1,
        "events": 5,
        "receipts": 1,
    }

    stack.clock.advance(datetime(2026, 7, 18, 8, 31, tzinfo=UTC) - stack.clock.now())
    replay = await stack.executor.execute(request=invocation, run=command())
    assert replay == first
    assert stack.factory_calls == [GeneratorKind.DETERMINISTIC]
    assert await row_counts(stack) == {
        "runs": 1,
        "snapshots": 1,
        "ideas": 1,
        "occurrences": 1,
        "run_state": 1,
        "idea_state": 1,
        "clusters": 1,
        "events": 5,
        "receipts": 1,
    }


@pytest.mark.asyncio
async def test_run_time_never_precedes_its_receipt_when_clock_steps_back(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "run-receipt-clock-boundary.sqlite3",
    )
    receipt_time = NOW + timedelta(minutes=10)
    clock = RegressingClock(
        values=[
            receipt_time,
            receipt_time - timedelta(minutes=1),
        ]
    )
    value.executor._receipt_store._clock = clock  # noqa: SLF001
    value.executor._clock = clock  # noqa: SLF001
    try:
        result = await value.executor.execute(
            request=request(key="run-receipt-clock-boundary"),
            run=command(),
        )
        assert result.status_code == 201

        async with value.database.read_session() as session:
            run = await session.scalar(select(InspirationRunRow))
            receipt = await session.scalar(select(IdempotencyRecordRow))
            event_times = tuple(
                await session.scalars(
                    select(DomainEventRow.occurred_at).order_by(DomainEventRow.event_id)
                )
            )
        assert run is not None
        assert receipt is not None
        assert run.created_at == receipt.created_at == receipt_time
        assert receipt.completed_at == receipt_time
        assert set(event_times) == {receipt_time}
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_new_unconfigured_selection_replays_422_without_creating_run(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "unconfigured.sqlite3",
        factory_error=GeneratorNotConfiguredError(),
    )
    selected = command(generator=GeneratorKind.OPENAI_COMPATIBLE)
    invocation = request(key="not-configured", run=selected)
    try:
        first = await value.executor.execute(request=invocation, run=selected)
        replay = await value.executor.execute(request=invocation, run=selected)
        assert first == replay
        assert first.status_code == 422
        assert json.loads(first.body) == {
            "error": {
                "code": "generator_not_configured",
                "details": {},
                "message": "The selected inspiration generator is not configured.",
            }
        }
        assert value.factory_calls == [GeneratorKind.OPENAI_COMPATIBLE]
        counts = await row_counts(value)
        assert counts["runs"] == counts["events"] == 0
        assert counts["receipts"] == 1
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_preparation_failure_uses_one_logical_time_and_terminal_failed(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    builder = FakeSnapshotBuilder(failure=RuntimeError("raw evidence body"))
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "preparation-failed.sqlite3",
        snapshot_builder=builder,
    )
    try:
        result = await value.executor.execute(
            request=request(key="preparation"),
            run=command(),
        )
        assert result.status_code == 201
        assert json.loads(result.body)["data"]["status"] == "failed"
        async with value.database.read_session() as session:
            rows = (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        assert tuple(row.event_type for row in rows) == (
            "inspiration.started",
            "inspiration.failed",
        )
        assert {row.occurred_at for row in rows} == {NOW}
        assert b"raw evidence body" not in b"".join(row.payload for row in rows)
        counts = await row_counts(value)
        assert counts["snapshots"] == counts["ideas"] == 0
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_mixed_operator_result_retains_success_and_sanitizes_failure(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(
        fail_operators=frozenset({InspirationOperator.COUNTERFACTUAL})
    )
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "mixed.sqlite3",
        generator=generator,
    )
    selected = command(
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        )
    )
    try:
        result = await value.executor.execute(
            request=request(key="mixed", run=selected),
            run=selected,
        )
        data = json.loads(result.body)["data"]
        assert data["status"] == "completed_with_errors"
        assert [outcome["error_code"] for outcome in data["operator_outcomes"]] == [
            None,
            "generator_error",
        ]
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
        assert b"raw provider secret" not in result.body
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_provider_cancellation_propagates_and_leaves_durable_trace(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(cancellation=True)
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "cancelled.sqlite3",
        generator=generator,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="cancelled"),
                run=command(),
            )
        async with value.database.read_session() as session:
            events = tuple(
                await session.scalars(
                    select(DomainEventRow.event_type).order_by(DomainEventRow.event_id)
                )
            )
            receipt = await session.scalar(select(IdempotencyRecordRow))
            state = await session.scalar(select(InspirationRunStateRow))
        assert events == (
            "inspiration.started",
            "inspiration.snapshot_frozen",
        )
        assert receipt is not None and receipt.state == "in_progress"
        assert state is not None and state.status == "running"
        assert generator.closed
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_global_deadline_maps_to_timed_out_terminal(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "timed-out.sqlite3",
        deadline_runner=ExpiringDeadlineRunner(),
    )
    selected = StartInspirationRun(
        owner_agent_id=OWNER_ID,
        goal="Find a robust bridge",
        operators=(InspirationOperator.CAUSAL_GAP,),
        operator_timeout_seconds=1,
        global_timeout_seconds=1,
    )
    try:
        result = await value.executor.execute(
            request=request(key="timed-out", run=selected),
            run=selected,
        )
        assert json.loads(result.body)["data"]["status"] == "timed_out"
        async with value.database.read_session() as session:
            event_types = tuple(
                await session.scalars(
                    select(DomainEventRow.event_type).order_by(DomainEventRow.event_id)
                )
            )
        assert event_types[-2:] == (
            "inspiration.operator_failed",
            "inspiration.timed_out",
        )
    finally:
        await value.database.dispose()
