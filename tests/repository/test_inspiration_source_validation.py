from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from tests.integration import test_inspiration_run as inspiration_run_fixtures
from tests.integration.test_create_experience import create as create_experience_source
from tests.integration.test_idea_adoption import (
    OWNER_A,
    OWNER_B,
    AdoptionStack,
    adopt,
    build_adoption_stack,
    create_experience,
    experience_spec,
    generate_idea,
    mapped_content,
)
from tests.integration.test_idea_adoption import (
    SeededIdea as AdoptionSeededIdea,
)
from tests.integration.test_idea_archival import (
    ORIGIN,
    lifecycle_service,
    run_lifecycle,
)
from tests.integration.test_idea_archival import (
    SeededIdea as ArchivalSeededIdea,
)
from tests.integration.test_idea_archival import (
    Stack as ArchivalStack,
)
from tests.integration.test_idea_archival import (
    build_stack as build_archival_stack,
)
from tests.integration.test_idea_archival import (
    seed_idea as seed_archival_idea,
)
from tests.integration.test_inspiration_recovery import recovery
from tests.integration.test_inspiration_run import (
    CONTENT_HASH,
    NOW,
    OWNER_ID,
    SNAPSHOT_ITEM_ID,
    SOURCE_CONTENT,
    SOURCE_ID,
    SOURCE_VERSION_ID,
    ExpiringDeadlineRunner,
    FakeGenerator,
    FakeSnapshotBuilder,
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
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.events import register_experience_events
from experience_hub.experiences.models import (
    ExperienceKind,
    Temperature,
    VersionContent,
)
from experience_hub.inspiration.commands import (
    ArchiveIdea,
    RejectIdea,
    StartInspirationRun,
)
from experience_hub.inspiration.events import (
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
    register_inspiration_events,
)
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.generators.openai_compatible import (
    GeneratorNotConfiguredError,
)
from experience_hub.inspiration.hashing import (
    hash_snapshot,
    snapshot_canonical_bytes,
    stable_evidence_key,
)
from experience_hub.inspiration.models import (
    MAX_SNAPSHOT_UTF8_BYTES,
    EvaluationVerdict,
    EvidenceSourceState,
    EvidenceSourceType,
    FrozenSnapshot,
    GeneratorKind,
    IdeaEvaluation,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.request_hashing import (
    decision_command_request,
    evaluation_command_request,
)
from experience_hub.sharing.events import (
    CapsulePublishedV1,
    CapsuleReceivedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import CapsuleStatus, InboxState
from experience_hub.storage import validation
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceCapsuleRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdempotencyRecordRow,
    InboxItemRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationSnapshotItemRow,
    TopicRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


def _validator(
    registry: EventRegistry,
    *,
    experiences: bool = False,
) -> validation.SourceValidator:
    validator = validation.SourceValidator(registry)
    if experiences:
        validation.register_experience_source_validator(validator)
    validation.register_inspiration_source_validator(validator)
    return validator


async def _validate(
    stack: Stack | AdoptionStack | ArchivalStack,
    registry: EventRegistry,
    *,
    experiences: bool = False,
) -> None:
    async with stack.database.read_session() as session:
        await _validator(registry, experiences=experiences).validate(session)


async def _forge_terminal_reserved_budget(
    stack: Stack,
    *,
    amount: int,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.failed"
            )
        )
        assert terminal is not None
        document = json.loads(terminal.payload)
        document["output_tokens_reserved_before"] = amount
        document["output_tokens_reserved_after"] = amount
        terminal.payload = canonical_json_bytes(document)
        receipts = (
            await uow.session.scalars(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.result_resource_type == "inspiration_run",
                    IdempotencyRecordRow.result_resource_id == terminal.aggregate_id,
                    IdempotencyRecordRow.state == "completed",
                )
            )
        ).all()
        assert receipts
        for receipt in receipts:
            assert receipt.response_body is not None
            response = json.loads(receipt.response_body)
            response["data"]["output_tokens_reserved"] = amount
            receipt.response_body = canonical_json_bytes(response)


async def _swap_domain_event_contents(
    uow: UnitOfWork,
    first: DomainEventRow,
    second: DomainEventRow,
    *,
    preserve_aggregate_position: bool = False,
) -> None:
    aggregate_fields = (
        ()
        if preserve_aggregate_position
        else ("aggregate_type", "aggregate_id", "sequence")
    )
    fields = aggregate_fields + (
        "event_type",
        "payload",
        "actor_agent_id",
        "causation_id",
        "occurred_at",
    )
    first_values = {field: getattr(first, field) for field in fields}
    second_values = {field: getattr(second, field) for field in fields}
    if not preserve_aggregate_position:
        first.aggregate_type = "forged_first_slot"
        second.aggregate_type = "forged_second_slot"
        await uow.session.flush()
    for attribute, value in second_values.items():
        setattr(first, attribute, value)
    for attribute, value in first_values.items():
        setattr(second, attribute, value)
    await uow.session.flush()


async def _forge_second_operator_attempt_beyond_token_budget(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_failed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.completed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator.payload)
        operator_payload["outcome"]["error_code"] = "generator_error"
        operator_payload["output_tokens_reserved_after"] = 2_400
        operator.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["operator_outcomes"][1]["error_code"] = "generator_error"
        terminal_payload["output_tokens_reserved_before"] = 2_400
        terminal_payload["output_tokens_reserved_after"] = 2_400
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["operator_outcomes"][1]["error_code"] = "generator_error"
        response["data"]["output_tokens_reserved"] = 2_400
        receipt.response_body = canonical_json_bytes(response)


async def _forge_operator_execution_after_global_deadline(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operators = (
            await uow.session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == "inspiration.operator_failed")
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.timed_out"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert len(operators) == 2
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        second_payload = json.loads(operators[1].payload)
        second_payload["outcome"]["error_code"] = "generator_error"
        second_payload["outcome"]["output_tokens_consumed"] = 1_200
        second_payload["output_tokens_reserved_after"] = 2_400
        second_payload["output_tokens_consumed_after"] = 2_400
        operators[1].payload = canonical_json_bytes(second_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["operator_outcomes"][1]["error_code"] = "generator_error"
        terminal_payload["operator_outcomes"][1]["output_tokens_consumed"] = 1_200
        terminal_payload["output_tokens_reserved_before"] = 2_400
        terminal_payload["output_tokens_reserved_after"] = 2_400
        terminal_payload["output_tokens_consumed_before"] = 2_400
        terminal_payload["output_tokens_consumed_after"] = 2_400
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["operator_outcomes"][1]["error_code"] = "generator_error"
        response["data"]["operator_outcomes"][1]["output_tokens_consumed"] = 1_200
        response["data"]["output_tokens_reserved"] = 2_400
        response["data"]["output_tokens_consumed"] = 2_400
        receipt.response_body = canonical_json_bytes(response)


async def _forge_success_duplicate_count(
    stack: Stack,
    *,
    count: int,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_completed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.completed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator.payload)
        operator_payload["outcome"]["duplicate_count"] = count
        operator.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["operator_outcomes"][0]["duplicate_count"] = count
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["operator_outcomes"][0]["duplicate_count"] = count
        receipt.response_body = canonical_json_bytes(response)


async def _forge_global_deadline_as_failed_terminal(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.timed_out"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["failure_code"] = "all_operators_failed"
        terminal_payload["status_after"] = "failed"
        terminal.event_type = "inspiration.failed"
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["status"] = "failed"
        receipt.response_body = canonical_json_bytes(response)


async def _forge_global_deadline_before_timeout(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_failed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.timed_out"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator.payload)
        operator_payload["elapsed_milliseconds_after"] = 0
        operator.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["elapsed_milliseconds_before"] = 0
        terminal_payload["elapsed_milliseconds_after"] = 0
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["elapsed_milliseconds"] = 0
        receipt.response_body = canonical_json_bytes(response)


async def _forge_deterministic_failure_code(
    stack: Stack,
    *,
    error_code: str,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_failed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.failed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator.payload)
        operator_payload["outcome"]["error_code"] = error_code
        operator.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["operator_outcomes"][0]["error_code"] = error_code
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["operator_outcomes"][0]["error_code"] = error_code
        receipt.response_body = canonical_json_bytes(response)


async def _forge_provider_timeout_before_operator_deadline(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_failed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.failed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator.payload)
        operator_payload["elapsed_milliseconds_after"] = 0
        operator.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["elapsed_milliseconds_before"] = 0
        terminal_payload["elapsed_milliseconds_after"] = 0
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["elapsed_milliseconds"] = 0
        receipt.response_body = canonical_json_bytes(response)


async def _forge_empty_snapshot_operator_execution(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operators = (
            await uow.session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == "inspiration.operator_failed")
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.failed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert len(operators) == 2
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        second_payload = json.loads(operators[1].payload)
        second_payload["outcome"]["error_code"] = "generator_error"
        second_payload["outcome"]["output_tokens_consumed"] = 1_200
        second_payload["output_tokens_reserved_after"] = 1_200
        second_payload["output_tokens_consumed_after"] = 1_200
        operators[1].payload = canonical_json_bytes(second_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["operator_outcomes"][1]["error_code"] = "generator_error"
        terminal_payload["operator_outcomes"][1]["output_tokens_consumed"] = 1_200
        terminal_payload["output_tokens_reserved_before"] = 1_200
        terminal_payload["output_tokens_reserved_after"] = 1_200
        terminal_payload["output_tokens_consumed_before"] = 1_200
        terminal_payload["output_tokens_consumed_after"] = 1_200
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["operator_outcomes"][1]["error_code"] = "generator_error"
        response["data"]["operator_outcomes"][1]["output_tokens_consumed"] = 1_200
        response["data"]["output_tokens_reserved"] = 1_200
        response["data"]["output_tokens_consumed"] = 1_200
        receipt.response_body = canonical_json_bytes(response)


async def _insert_attached_receipt(
    stack: Stack,
    *,
    receipt_id: UUID,
    scope: str,
    idempotency_key: str,
    run_id: UUID,
) -> None:
    async with stack.database.transaction() as uow:
        original = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert original is not None
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=receipt_id,
                caller_scope=(
                    "system:local"
                    if scope == "inspiration.run.recover"
                    else original.caller_scope
                ),
                scope=scope,
                idempotency_key=idempotency_key,
                request_hash="f" * 64,
                state="completed",
                result_resource_type="inspiration_run",
                result_resource_id=run_id,
                response_status_code=201,
                response_body=canonical_json_bytes({}),
                response_content_type="application/json",
                response_headers=canonical_json_bytes({}),
                created_at=original.created_at,
                completed_at=original.created_at,
            )
        )


@dataclass(slots=True)
class ReservingFakeGenerator(FakeGenerator):
    consumed_by_operator: dict[InspirationOperator, int] = field(default_factory=dict)

    @property
    def reserves_output_tokens(self) -> bool:
        return True

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {
            "base_url": "https://generator.invalid/v1",
            "model": "source-validation-model",
        }

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
        result = await FakeGenerator.generate(
            self,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operator=operator,
            branch_limit=branch_limit,
            output_token_limit=output_token_limit,
        )
        return GeneratorResult(
            **{
                **result.model_dump(mode="python", warnings=False),
                "output_tokens_consumed": self.consumed_by_operator.get(operator, 0),
            }
        )


@dataclass(slots=True)
class EmptySnapshotBuilder(FakeSnapshotBuilder):
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
        return FrozenSnapshot(
            run_id=run_id,
            items=(),
            snapshot_hash=hash_snapshot(()),
            frozen_at=at,
        )


@dataclass(slots=True)
class StaticExperienceSnapshotBuilder:
    content: VersionContent
    content_hash: str
    sessions: list[object] = field(default_factory=list)
    frozen_items: list[SnapshotItem] = field(default_factory=list)

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
        item = SnapshotItem(
            snapshot_item_id=SNAPSHOT_ITEM_ID,
            stable_evidence_key=stable_evidence_key(
                source_type=EvidenceSourceType.EXPERIENCE,
                source_id=SOURCE_ID,
                source_version_id=SOURCE_VERSION_ID,
                content_hash=self.content_hash,
            ),
            run_id=run_id,
            source_type=EvidenceSourceType.EXPERIENCE,
            source_id=SOURCE_ID,
            source_version_id=SOURCE_VERSION_ID,
            source_state=EvidenceSourceState.WARM,
            source_trust=1.0,
            rank=1,
            summary=self.content.summary,
            mechanism=self.content.mechanism,
            applicability=self.content.applicability,
            tags=self.content.tags,
            falsifiers=self.content.falsifiers,
            excerpt=self.content.body[:2_048],
            content_hash=self.content_hash,
            captured_at=at,
        )
        self.frozen_items.append(item)
        return FrozenSnapshot(
            run_id=run_id,
            items=(item,),
            snapshot_hash=hash_snapshot((item,)),
            frozen_at=at,
        )


async def _downgrade_adoption_event_to_v1(
    stack: AdoptionStack,
) -> InspirationIdeaAdoptedV1:
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert event is not None
        document = json.loads(event.payload)
        document.pop("requested_importance")
        document.pop("requested_confidence")
        document["schema_version"] = 1
        payload = InspirationIdeaAdoptedV1.model_validate_json(
            canonical_json_bytes(document)
        )
        event.event_type = InspirationIdeaAdoptedV1.event_type
        event.payload = canonical_json_bytes(document)
        return payload


async def _seed_valid_archival_idea(
    stack: ArchivalStack,
    *,
    key: str,
    mechanism: str,
) -> ArchivalSeededIdea:
    status, source = await create_experience_source(
        stack,
        key=f"{key}-snapshot-source",
        value=SOURCE_CONTENT,
    )
    assert status == 201
    return await seed_archival_idea(
        stack,
        key=key,
        mechanism=mechanism,
        source_id=UUID(source["data"]["experience_id"]),
        source_version_id=UUID(source["data"]["version_id"]),
        source_content_hash=source["data"]["content_hash"],
    )


async def _execute_idea_decision(
    stack: AdoptionStack,
    command: ArchiveIdea | RejectIdea,
    *,
    key: str,
) -> CommandResult:
    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        if isinstance(command, ArchiveIdea):
            return await stack.service.archive(
                uow=uow,
                command=command,
                command_context=command_context,
            )
        return await stack.service.reject(
            uow=uow,
            command=command,
            command_context=command_context,
        )

    return await stack.executor.execute(
        decision_command_request(command, idempotency_key=key),
        handler,
    )


def _inspiration_registry(*, experiences: bool = True) -> EventRegistry:
    registry = EventRegistry()
    if experiences:
        register_experience_events(registry)
    register_inspiration_events(registry)
    return registry


@pytest.fixture
async def completed_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-validation.sqlite3",
        seed_experience_source_trace=True,
    )
    result = await value.executor.execute(
        request=request(key="source-complete"),
        run=command(),
    )
    assert result.status_code == 201
    try:
        yield value
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_inspiration_validator_accepts_a_complete_normal_trace(
    completed_stack: Stack,
) -> None:
    await _validate(completed_stack, _inspiration_registry())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    (
        "completed_with_errors",
        "all_operators_failed",
        "timed_out",
        "preparation_failed",
    ),
)
async def test_inspiration_validator_accepts_every_legal_terminal_shape(
    repository_root: Path,
    tmp_path: Path,
    scenario: str,
) -> None:
    selected: StartInspirationRun
    generator: FakeGenerator | None = None
    snapshot_builder: FakeSnapshotBuilder | None = None
    deadline_runner = None
    if scenario == "completed_with_errors":
        generator = FakeGenerator(
            fail_operators=frozenset({InspirationOperator.COUNTERFACTUAL})
        )
        selected = command(
            operators=(
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
            )
        )
    elif scenario == "all_operators_failed":
        generator = FakeGenerator(
            fail_operators=frozenset({InspirationOperator.CAUSAL_GAP})
        )
        selected = command()
    elif scenario == "timed_out":
        deadline_runner = ExpiringDeadlineRunner()
        selected = StartInspirationRun(
            owner_agent_id=OWNER_ID,
            goal="Prove the legal timed-out source trace",
            operators=(InspirationOperator.CAUSAL_GAP,),
            operator_timeout_seconds=1,
            global_timeout_seconds=1,
        )
    else:
        snapshot_builder = FakeSnapshotBuilder(
            failure=RuntimeError("private preparation failure")
        )
        selected = command()

    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"inspiration-source-{scenario}.sqlite3",
        generator=generator,
        snapshot_builder=snapshot_builder,
        deadline_runner=deadline_runner,
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key=f"source-{scenario}", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        await _validate(stack, _inspiration_registry())
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_inspiration_validator_accepts_a_started_only_interrupted_trace(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-started-only.sqlite3",
        snapshot_builder=FakeSnapshotBuilder(failure=asyncio.CancelledError()),
        seed_experience_source_trace=True,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await stack.executor.execute(
                request=request(key="source-started-only"),
                run=command(),
            )
        await _validate(stack, _inspiration_registry())
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("preparation_failed", "recovered"))
async def test_zero_operator_terminal_cannot_forge_reserved_budget(
    repository_root: Path,
    tmp_path: Path,
    scenario: str,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"inspiration-source-zero-budget-{scenario}.sqlite3",
        generator=(
            FakeGenerator(cancellation=True) if scenario == "recovered" else None
        ),
        snapshot_builder=(
            FakeSnapshotBuilder(failure=RuntimeError("private failure"))
            if scenario == "preparation_failed"
            else None
        ),
        seed_experience_source_trace=True,
    )
    try:
        if scenario == "recovered":
            with pytest.raises(asyncio.CancelledError):
                await stack.executor.execute(
                    request=request(key="source-zero-budget-recovered"),
                    run=command(),
                )
            stack.clock.advance(timedelta(minutes=1))
            assert len(await recovery(stack).recover()) == 1
        else:
            result = await stack.executor.execute(
                request=request(key="source-zero-budget-preparation"),
                run=command(),
            )
            assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_terminal_reserved_budget(stack, amount=1_200)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_reserving_generator_trace_honors_consumption_based_token_pool(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = ReservingFakeGenerator(
        consumed_by_operator={InspirationOperator.CAUSAL_GAP: 600}
    )
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-token-budget-positive.sqlite3",
        generator=generator,
        seed_experience_source_trace=True,
    )
    selected = replace(
        command(
            generator=GeneratorKind.OPENAI_COMPATIBLE,
            operators=(
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
            ),
        ),
        output_tokens_per_operator=1_200,
        total_output_tokens=1_200,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-token-budget-positive", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        assert generator.calls == [InspirationOperator.CAUSAL_GAP]
        await _validate(stack, _inspiration_registry())
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_reserving_generator_cannot_attempt_beyond_total_token_budget(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = ReservingFakeGenerator(
        consumed_by_operator={InspirationOperator.CAUSAL_GAP: 600}
    )
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-token-budget-negative.sqlite3",
        generator=generator,
        seed_experience_source_trace=True,
    )
    selected = replace(
        command(
            generator=GeneratorKind.OPENAI_COMPATIBLE,
            operators=(
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
            ),
        ),
        output_tokens_per_operator=1_200,
        total_output_tokens=1_200,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-token-budget-negative", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_second_operator_attempt_beyond_token_budget(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_no_operator_executes_after_global_deadline_exhaustion(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = ReservingFakeGenerator()
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-global-deadline-chain.sqlite3",
        generator=generator,
        deadline_runner=ExpiringDeadlineRunner(),
        seed_experience_source_trace=True,
    )
    selected = replace(
        command(
            generator=GeneratorKind.OPENAI_COMPATIBLE,
            operators=(
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
            ),
        ),
        operator_timeout_seconds=1,
        global_timeout_seconds=1,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-global-deadline-chain", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        assert json.loads(result.body)["data"]["status"] == "timed_out"
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_operator_execution_after_global_deadline(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_operator_duplicate_count_cannot_exceed_its_branch_limit(
    completed_stack: Stack,
) -> None:
    registry = _inspiration_registry()
    await _validate(completed_stack, registry)

    await _forge_success_duplicate_count(completed_stack, count=999)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, registry)
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_global_deadline_requires_a_timed_out_terminal(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-global-terminal.sqlite3",
        deadline_runner=ExpiringDeadlineRunner(),
        seed_experience_source_trace=True,
    )
    selected = replace(
        command(),
        operator_timeout_seconds=1,
        global_timeout_seconds=1,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-global-terminal", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        assert json.loads(result.body)["data"]["status"] == "timed_out"
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_global_deadline_as_failed_terminal(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_global_deadline_cannot_precede_configured_timeout(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-global-onset.sqlite3",
        deadline_runner=ExpiringDeadlineRunner(),
        seed_experience_source_trace=True,
    )
    selected = replace(
        command(),
        operator_timeout_seconds=1,
        global_timeout_seconds=1,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-global-onset", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_global_deadline_before_timeout(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_deterministic_generator_cannot_exhaust_a_token_reservation(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-deterministic-budget.sqlite3",
        generator=FakeGenerator(
            fail_operators=frozenset({InspirationOperator.CAUSAL_GAP})
        ),
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-deterministic-budget"),
            run=command(),
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_deterministic_failure_code(
            stack,
            error_code="insufficient_token_reservation",
        )

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_nonempty_snapshot_cannot_fail_for_insufficient_evidence(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-evidence-decision.sqlite3",
        generator=FakeGenerator(
            fail_operators=frozenset({InspirationOperator.CAUSAL_GAP})
        ),
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-evidence-decision"),
            run=command(),
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_deterministic_failure_code(
            stack,
            error_code="insufficient_evidence",
        )

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_provider_timeout_cannot_precede_operator_deadline(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-provider-timeout.sqlite3",
        deadline_runner=ExpiringDeadlineRunner(),
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-provider-timeout"),
            run=command(),
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_provider_timeout_before_operator_deadline(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_empty_snapshot_never_executes_a_generator(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    generator = ReservingFakeGenerator()
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-empty-snapshot.sqlite3",
        generator=generator,
        snapshot_builder=EmptySnapshotBuilder(),
        seed_experience_source_trace=True,
    )
    selected = command(
        generator=GeneratorKind.OPENAI_COMPATIBLE,
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        ),
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-empty-snapshot", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        assert generator.calls == []
        registry = _inspiration_registry()
        await _validate(stack, registry)

        await _forge_empty_snapshot_operator_execution(stack)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_snapshot_canonical_document_respects_total_byte_budget(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_values = tuple(f"{index:02d}-" + ("值" * 1_300) for index in range(32))
    content = VersionContent(
        body="b" * 2_048,
        summary="摘" * 1_000,
        mechanism="机制" * 1_000,
        tags=long_values,
        applicability=long_values,
        evidence=(),
        falsifiers=long_values,
    )
    encoded = encode_version_content(
        kind=ExperienceKind.SEMANTIC,
        content=content,
    )
    builder = StaticExperienceSnapshotBuilder(
        content=content,
        content_hash=encoded.content_hash,
    )
    monkeypatch.setattr(inspiration_run_fixtures, "SOURCE_CONTENT", content)
    monkeypatch.setattr(inspiration_run_fixtures, "ENCODED_SOURCE_CONTENT", encoded)
    monkeypatch.setattr(
        inspiration_run_fixtures,
        "CONTENT_HASH",
        encoded.content_hash,
    )
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-snapshot-byte-budget.sqlite3",
        snapshot_builder=builder,
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-snapshot-byte-budget"),
            run=command(),
        )
        assert result.status_code == 201
        assert len(builder.frozen_items) == 1
        item = builder.frozen_items[0]
        assert len(snapshot_canonical_bytes((item,))) > MAX_SNAPSHOT_UTF8_BYTES

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, _inspiration_registry())
        assert caught.value.mismatch_key.startswith("inspiration_snapshot:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_normal_run_receipt_completes_at_its_terminal_event(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert receipt is not None and receipt.completed_at is not None
        receipt.completed_at += timedelta(days=7)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", ("same_run", "unknown_run"))
async def test_start_receipts_are_bijective_with_runs(
    completed_stack: Stack,
    binding: str,
) -> None:
    async with completed_stack.database.read_session() as session:
        run_id = await session.scalar(select(InspirationRunRow.run_id))
    assert run_id is not None
    attached_id = (
        run_id
        if binding == "same_run"
        else UUID("00000000-0000-0000-0000-000000009999")
    )
    await _insert_attached_receipt(
        completed_stack,
        receipt_id=UUID("00000000-0000-0000-0000-000000009901"),
        scope="inspiration.run.start",
        idempotency_key=f"extra-{binding}",
        run_id=attached_id,
    )

    with pytest.raises(validation.SourceIntegrityError):
        await _validate(completed_stack, _inspiration_registry())


@pytest.mark.asyncio
async def test_unconfigured_generator_receipt_is_the_only_unattached_start_result(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-unconfigured.sqlite3",
        factory_error=GeneratorNotConfiguredError(),
    )
    selected = command(generator=GeneratorKind.OPENAI_COMPATIBLE)
    try:
        result = await stack.executor.execute(
            request=request(key="source-unconfigured", run=selected),
            run=selected,
        )
        assert result.status_code == 422
        await _validate(stack, _inspiration_registry())
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_unattached_start_receipt_cannot_return_a_forged_run(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        original = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert original is not None and original.response_body is not None
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=UUID("00000000-0000-0000-0000-000000009904"),
                caller_scope=original.caller_scope,
                scope="inspiration.run.start",
                idempotency_key="forged-unattached-run",
                request_hash="e" * 64,
                state="completed",
                result_resource_type=None,
                result_resource_id=None,
                response_status_code=201,
                response_body=original.response_body,
                response_content_type="application/json",
                response_headers=original.response_headers,
                created_at=original.created_at,
                completed_at=original.completed_at,
            )
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_receipt:")


@pytest.mark.asyncio
async def test_run_receipt_causation_is_closed_to_inspiration_effects(
    completed_stack: Stack,
) -> None:
    registry = _inspiration_registry(experiences=True)
    await _validate(completed_stack, registry, experiences=True)
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert receipt is not None
        await uow.session.execute(
            update(DomainEventRow)
            .where(
                DomainEventRow.event_type.in_(
                    (
                        "experience.created",
                        "experience.version_created",
                    )
                )
            )
            .values(causation_id=receipt.receipt_id)
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_nonterminal_run_cannot_have_a_recovery_receipt(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-premature-recovery.sqlite3",
        snapshot_builder=FakeSnapshotBuilder(failure=asyncio.CancelledError()),
        seed_experience_source_trace=True,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await stack.executor.execute(
                request=request(key="source-premature-recovery"),
                run=command(),
            )
        async with stack.database.read_session() as session:
            run_id = await session.scalar(select(InspirationRunRow.run_id))
        assert run_id is not None
        await _insert_attached_receipt(
            stack,
            receipt_id=UUID("00000000-0000-0000-0000-000000009902"),
            scope="inspiration.run.recover",
            idempotency_key=f"recovery:{run_id}",
            run_id=run_id,
        )

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, _inspiration_registry())
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration",
    (
        {
            "base_url": "https://user:secret@generator.invalid/v1",
            "model": "safe-model",
        },
        {
            "base_url": "https://generator.invalid/v1",
            "model": " unsafe-model ",
        },
    ),
)
async def test_openai_configuration_is_safe_and_credential_free(
    repository_root: Path,
    tmp_path: Path,
    configuration: dict[str, str],
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-generator-config.sqlite3",
        generator=ReservingFakeGenerator(),
        seed_experience_source_trace=True,
    )
    selected = command(generator=GeneratorKind.OPENAI_COMPATIBLE)
    try:
        result = await stack.executor.execute(
            request=request(key="source-generator-config", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text("DROP TRIGGER inspiration_runs_reject_update")
            )
            run = await uow.session.scalar(select(InspirationRunRow))
            assert run is not None
            run.generator_configuration = canonical_json_bytes(configuration)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_name",
    ("predictions", "falsifiers", "assumptions"),
)
async def test_idea_semantic_lists_are_sorted_and_unique(
    completed_stack: Stack,
    field_name: str,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER inspiration_ideas_reject_update"))
        idea = await uow.session.scalar(select(InspirationIdeaRow))
        assert idea is not None
        values = json.loads(getattr(idea, field_name))
        assert values
        setattr(
            idea,
            field_name,
            canonical_json_bytes([*values, values[0]]),
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_idea:")


@pytest.mark.asyncio
async def test_idea_evidence_cannot_repeat(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER inspiration_ideas_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        idea = await uow.session.scalar(select(InspirationIdeaRow))
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.idea_generated"
            )
        )
        assert idea is not None
        assert event is not None
        references = json.loads(idea.evidence_references)
        assert references
        duplicated = [*references, references[0]]
        idea.evidence_references = canonical_json_bytes(duplicated)
        payload = json.loads(event.payload)
        payload["evidence"] = duplicated
        event.payload = canonical_json_bytes(payload)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key in {
        "source_integrity",
        f"inspiration_idea:{idea.idea_id}",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_state", "hot"),
        ("source_trust", 0.5),
    ),
)
async def test_snapshot_experience_state_matches_its_historical_source(
    completed_stack: Stack,
    field: str,
    value: str | float,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER inspiration_snapshot_items_reject_update")
        )
        snapshot = await uow.session.scalar(select(InspirationSnapshotItemRow))
        assert snapshot is not None
        await uow.session.execute(
            update(InspirationSnapshotItemRow)
            .where(
                InspirationSnapshotItemRow.snapshot_item_id == snapshot.snapshot_item_id
            )
            .values(**{field: value})
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_snapshot_item:")


@pytest.mark.asyncio
async def test_capsule_snapshot_requires_an_available_owned_inbox_source(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-capsule-anchor.sqlite3",
        snapshot_builder=FakeSnapshotBuilder(
            source_type=EvidenceSourceType.CAPSULE,
        ),
    )
    try:
        result = await value.executor.execute(
            request=request(key="source-capsule-anchor"),
            run=command(),
        )
        assert result.status_code == 201

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(value, _inspiration_registry())
        assert caught.value.mismatch_key.startswith("inspiration_snapshot_item:")
    finally:
        await value.database.dispose()


async def _seed_available_capsule_source(stack: Stack) -> None:
    receipt_id = UUID("00000000-0000-0000-0000-000000009901")
    topic_id = UUID("00000000-0000-0000-0000-000000009902")
    inbox_item_id = UUID("00000000-0000-0000-0000-000000009903")
    root_fingerprint = "a" * 64
    capsule_hash = "b" * 64
    async with stack.database.transaction() as uow:
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=receipt_id,
                caller_scope=f"agent:{OWNER_ID}",
                scope="sharing.fixture",
                idempotency_key="available-capsule",
                request_hash="c" * 64,
                state="completed",
                result_resource_type=None,
                result_resource_id=None,
                response_status_code=204,
                response_body=canonical_json_bytes({}),
                response_content_type="application/json",
                response_headers=canonical_json_bytes({}),
                created_at=NOW,
                completed_at=NOW,
            )
        )
        uow.session.add(
            TopicRow(
                topic_id=topic_id,
                owner_agent_id=OWNER_ID,
                name="Available capsule fixture",
                description=None,
                created_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceCapsuleRow(
                capsule_id=SOURCE_ID,
                transport_schema_version=1,
                topic_id=topic_id,
                source_experience_id=SOURCE_ID,
                source_version_id=SOURCE_VERSION_ID,
                publisher_agent_id=OWNER_ID,
                kind=ExperienceKind.SEMANTIC,
                body=SOURCE_CONTENT.body,
                summary=SOURCE_CONTENT.summary,
                mechanism=SOURCE_CONTENT.mechanism,
                tags=canonical_json_bytes(SOURCE_CONTENT.tags),
                applicability=canonical_json_bytes(SOURCE_CONTENT.applicability),
                evidence=canonical_json_bytes(SOURCE_CONTENT.evidence),
                falsifiers=canonical_json_bytes(SOURCE_CONTENT.falsifiers),
                publisher_confidence=0.75,
                provenance_chain=canonical_json_bytes(()),
                root_fingerprint=root_fingerprint,
                source_content_hash=CONTENT_HASH,
                created_at=NOW,
                expires_at=NOW + timedelta(days=7),
                hop_count=0,
                capsule_hash=capsule_hash,
            )
        )
        await uow.session.flush()
        published = DomainEventRow(
            aggregate_type="capsule",
            aggregate_id=SOURCE_ID,
            sequence=1,
            event_type=CapsulePublishedV1.event_type,
            payload=canonical_json_bytes(
                CapsulePublishedV1(
                    schema_version=1,
                    capsule_id=SOURCE_ID,
                    topic_id=topic_id,
                    source_experience_id=SOURCE_ID,
                    source_version_id=SOURCE_VERSION_ID,
                    publisher_agent_id=OWNER_ID,
                    capsule_hash=capsule_hash,
                    root_fingerprint=root_fingerprint,
                    status_after=CapsuleStatus.ACTIVE,
                ).model_dump(mode="json")
            ),
            actor_agent_id=OWNER_ID,
            causation_id=receipt_id,
            occurred_at=NOW,
        )
        uow.session.add(published)
        await uow.session.flush()
        received = DomainEventRow(
            aggregate_type="inbox_item",
            aggregate_id=inbox_item_id,
            sequence=1,
            event_type=CapsuleReceivedV1.event_type,
            payload=canonical_json_bytes(
                CapsuleReceivedV1(
                    schema_version=1,
                    item_id=inbox_item_id,
                    capsule_id=SOURCE_ID,
                    recipient_agent_id=OWNER_ID,
                    state_after=InboxState.PENDING,
                ).model_dump(mode="json")
            ),
            actor_agent_id=OWNER_ID,
            causation_id=receipt_id,
            occurred_at=NOW,
        )
        uow.session.add(received)
        await uow.session.flush()
        uow.session.add(
            InboxItemRow(
                item_id=inbox_item_id,
                recipient_agent_id=OWNER_ID,
                capsule_id=SOURCE_ID,
                state=InboxState.PENDING,
                projection_event_id=received.event_id,
            )
        )


@pytest.mark.asyncio
async def test_capsule_snapshot_accepts_an_available_owned_inbox_source(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-valid-capsule.sqlite3",
        snapshot_builder=FakeSnapshotBuilder(
            source_type=EvidenceSourceType.CAPSULE,
        ),
        register_additional_events=register_sharing_events,
    )
    try:
        await _seed_available_capsule_source(value)
        run = replace(command(), include_inbox=True)
        result = await value.executor.execute(
            request=request(key="source-valid-capsule", run=run),
            run=run,
        )
        assert result.status_code == 201

        registry = _inspiration_registry()
        register_sharing_events(registry)
        await _validate(value, registry)
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_run_source_body_is_bound_to_its_start_request_hash(
    completed_stack: Stack,
) -> None:
    changed_goal = "Find a different but internally consistent bridge"
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER inspiration_runs_reject_update"))
        run = await uow.session.scalar(select(InspirationRunRow))
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert run is not None
        assert receipt is not None and receipt.response_body is not None
        run.goal = changed_goal
        response = json.loads(receipt.response_body)
        response["data"]["goal"] = changed_goal
        receipt.response_body = canonical_json_bytes(response)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_run_receipt_completion_cannot_precede_its_creation_or_terminal(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert receipt is not None
        receipt.completed_at = receipt.created_at - timedelta(minutes=1)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("caller_scope", "system:local"),
        ("scope", "forged.run"),
        ("result_resource_type", "idea"),
    ),
)
async def test_run_trace_requires_its_exact_attached_receipt(
    completed_stack: Stack,
    field: str,
    value: str,
) -> None:
    async with completed_stack.database.transaction() as uow:
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert receipt is not None
        setattr(receipt, field, value)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())

    assert caught.value.mismatch_key.startswith(
        ("inspiration_run:", "inspiration_receipt:")
    )


@pytest.mark.asyncio
async def test_normal_trace_uses_one_persisted_logical_timestamp(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.completed"
            )
        )
        assert terminal is not None
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == terminal.event_id)
            .values(occurred_at=terminal.occurred_at + timedelta(microseconds=1))
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())

    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_deterministic_run_cannot_forge_token_reservations(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        operator_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.operator_completed"
            )
        )
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.completed"
            )
        )
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.run.start"
            )
        )
        assert operator_event is not None
        assert terminal is not None
        assert receipt is not None and receipt.response_body is not None

        operator_payload = json.loads(operator_event.payload)
        operator_payload["output_tokens_reserved_after"] = 1
        operator_event.payload = canonical_json_bytes(operator_payload)

        terminal_payload = json.loads(terminal.payload)
        terminal_payload["output_tokens_reserved_before"] = 1
        terminal_payload["output_tokens_reserved_after"] = 1
        terminal.payload = canonical_json_bytes(terminal_payload)

        response = json.loads(receipt.response_body)
        response["data"]["output_tokens_reserved"] = 1
        receipt.response_body = canonical_json_bytes(response)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_each_operator_outcome_closes_its_own_idea_segment(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-operator-order.sqlite3",
        seed_experience_source_trace=True,
    )
    try:
        run = command(
            operators=(
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
            )
        )
        result = await stack.executor.execute(
            request=request(key="source-operator-order", run=run),
            run=run,
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
            ideas = (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == "inspiration.idea_generated")
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
            outcomes = (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            (
                                "inspiration.operator_completed",
                                "inspiration.operator_failed",
                            )
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
            assert len(ideas) == len(outcomes) == 2
            first_outcome = outcomes[0]
            second_idea = ideas[1]
            await _swap_domain_event_contents(uow, first_outcome, second_idea)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_run_cannot_retain_near_duplicate_mechanisms(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-run-local-dedup.sqlite3",
        seed_experience_source_trace=True,
    )
    selected = command(
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        )
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-run-local-dedup", run=selected),
            run=selected,
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        monkeypatch.setattr(
            validation,
            "mechanism_similarity",
            lambda _left, _right: 1.0,
        )
        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_frozen_snapshot_precedes_a_failed_operator_without_ideas(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-failed-operator-order.sqlite3",
        generator=FakeGenerator(
            fail_operators=frozenset({InspirationOperator.CAUSAL_GAP})
        ),
        seed_experience_source_trace=True,
    )
    try:
        result = await stack.executor.execute(
            request=request(key="source-failed-operator-order"),
            run=command(),
        )
        assert result.status_code == 201
        registry = _inspiration_registry()
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
            snapshot_event = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.snapshot_frozen"
                )
            )
            operator_event = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.operator_failed"
                )
            )
            assert snapshot_event is not None
            assert operator_event is not None
            await _swap_domain_event_contents(
                uow,
                snapshot_event,
                operator_event,
                preserve_aggregate_position=True,
            )

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    (
        "stable_key",
        "snapshot_hash",
        "idea_hash",
        "mechanism_hash",
        "occurrence",
        "cluster_transition",
    ),
)
async def test_inspiration_validator_recomputes_all_immutable_hash_anchors(
    completed_stack: Stack,
    corruption: str,
) -> None:
    async with completed_stack.database.transaction() as uow:
        if corruption in {"stable_key", "snapshot_hash"}:
            await uow.session.execute(
                text("DROP TRIGGER inspiration_snapshot_items_reject_update")
            )
            snapshot = await uow.session.scalar(select(InspirationSnapshotItemRow))
            assert snapshot is not None
            values = (
                {"stable_evidence_key": "f" * 64}
                if corruption == "stable_key"
                else {"content_hash": "e" * 64}
            )
            await uow.session.execute(
                update(InspirationSnapshotItemRow)
                .where(
                    InspirationSnapshotItemRow.snapshot_item_id
                    == snapshot.snapshot_item_id
                )
                .values(**values)
            )
        elif corruption in {
            "idea_hash",
            "mechanism_hash",
            "cluster_transition",
        }:
            await uow.session.execute(
                text("DROP TRIGGER inspiration_ideas_reject_update")
            )
            idea = await uow.session.scalar(select(InspirationIdeaRow))
            assert idea is not None
            if corruption == "cluster_transition":
                await uow.session.execute(
                    text("DROP TRIGGER domain_events_reject_update")
                )
                event = await uow.session.scalar(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type == "inspiration.idea_generated"
                    )
                )
                assert event is not None
                document = json.loads(event.payload)
                document["member_hashes_before"] = document["member_hashes_after"]
                document["occurrence_count_before"] = 1
                document["occurrence_count_after"] = 2
                document["distinct_snapshot_count_before"] = 1
                document["distinct_snapshot_count_after"] = 1
                document["maturity_before"] = "speculative"
                document["last_signal_at_before"] = document["last_signal_at_after"]
                event.payload = canonical_json_bytes(document)
            else:
                await uow.session.execute(
                    update(InspirationIdeaRow)
                    .where(InspirationIdeaRow.idea_id == idea.idea_id)
                    .values(
                        **{
                            (
                                "idea_content_hash"
                                if corruption == "idea_hash"
                                else "mechanism_hash"
                            ): "f" * 64
                        }
                    )
                )
        else:
            await uow.session.execute(
                text("DROP TRIGGER idea_occurrences_reject_update")
            )
            occurrence = await uow.session.scalar(select(IdeaOccurrenceRow))
            assert occurrence is not None
            await uow.session.execute(
                update(IdeaOccurrenceRow)
                .where(IdeaOccurrenceRow.occurrence_id == occurrence.occurrence_id)
                .values(snapshot_hash="f" * 64)
            )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())

    assert caught.value.mismatch_key.startswith("inspiration_")


@pytest.mark.asyncio
async def test_generated_ordinals_are_contiguous_per_operator(
    completed_stack: Stack,
) -> None:
    async with completed_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER inspiration_ideas_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        idea = await uow.session.scalar(select(InspirationIdeaRow))
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.idea_generated"
            )
        )
        assert idea is not None
        assert event is not None
        idea.ordinal = 2
        payload = json.loads(event.payload)
        payload["ordinal"] = 2
        event.payload = canonical_json_bytes(payload)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(completed_stack, _inspiration_registry())
    assert caught.value.mismatch_key.startswith("inspiration_run:")


@pytest.mark.asyncio
async def test_interrupted_and_recovered_traces_are_both_legal(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-recovery.sqlite3",
        generator=FakeGenerator(cancellation=True),
        seed_experience_source_trace=True,
    )
    registry = _inspiration_registry()
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="source-interrupted"),
                run=command(),
            )

        await _validate(value, registry)
        value.clock.advance(timedelta(minutes=5))
        assert len(await recovery(value).recover()) == 1
        await _validate(value, registry)
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_recovery_receipt_is_bound_to_its_canonical_request_hash(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-recovery-hash.sqlite3",
        generator=FakeGenerator(cancellation=True),
        seed_experience_source_trace=True,
    )
    registry = _inspiration_registry()
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="source-recovery-hash-interrupted"),
                run=command(),
            )
        value.clock.advance(timedelta(minutes=5))
        assert len(await recovery(value).recover()) == 1

        async with value.database.transaction() as uow:
            receipt = await uow.session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.scope == "inspiration.run.recover"
                )
            )
            assert receipt is not None
            receipt.request_hash = "f" * 64

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(value, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_recovery_terminal_uses_the_exact_recovery_clock_boundary(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-recovery-time.sqlite3",
        generator=FakeGenerator(cancellation=True),
        seed_experience_source_trace=True,
    )
    registry = _inspiration_registry()
    try:
        with pytest.raises(asyncio.CancelledError):
            await value.executor.execute(
                request=request(key="source-recovery-time-interrupted"),
                run=command(),
            )
        value.clock.advance(timedelta(minutes=5))
        assert len(await recovery(value).recover()) == 1
        await _validate(value, registry)

        moved_to = value.clock.advance(timedelta(days=1))
        async with value.database.transaction() as uow:
            await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
            terminal = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.failed"
                )
            )
            receipts = (
                await uow.session.scalars(
                    select(IdempotencyRecordRow).where(
                        IdempotencyRecordRow.result_resource_type == "inspiration_run",
                        IdempotencyRecordRow.state == "completed",
                    )
                )
            ).all()
            assert terminal is not None
            assert len(receipts) == 2
            terminal.occurred_at = moved_to
            for receipt in receipts:
                assert receipt.response_body is not None
                response = json.loads(receipt.response_body)
                response["data"]["completed_at"] = moved_to.isoformat().replace(
                    "+00:00",
                    "Z",
                )
                receipt.response_body = canonical_json_bytes(response)
                receipt.completed_at = moved_to

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(value, registry)
        assert caught.value.mismatch_key.startswith("inspiration_run:")
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("receipt", "owner"))
async def test_automatic_archive_requires_lifecycle_receipt_scope_and_resource(
    repository_root: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    stack = await build_archival_stack(
        repository_root=repository_root,
        database_path=tmp_path / f"inspiration-source-archive-{corruption}.sqlite3",
    )
    try:
        await _seed_valid_archival_idea(
            stack,
            key="source-archive-idea",
            mechanism="source archive receipt mechanism",
        )
        evaluated_at = stack.clock.advance(timedelta(days=365))
        result = await run_lifecycle(
            stack,
            lifecycle_service(stack),
            key="source-archive-cycle",
            evaluated_at=evaluated_at,
        )
        assert result.status_code == 200
        registry = _inspiration_registry(experiences=True)
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            if corruption == "receipt":
                receipt = await uow.session.scalar(
                    select(IdempotencyRecordRow).where(
                        IdempotencyRecordRow.scope == "lifecycle.run"
                    )
                )
                assert receipt is not None
                receipt.caller_scope = f"agent:{receipt.result_resource_id}"
            else:
                await uow.session.execute(
                    text("DROP TRIGGER domain_events_reject_update")
                )
                event = await uow.session.scalar(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type == "inspiration.idea_archived"
                    )
                )
                assert event is not None
                document = json.loads(event.payload)
                document["owner_agent_id"] = str(OWNER_B)
                event.payload = canonical_json_bytes(document)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        expected_prefix = (
            "inspiration_idea_history:"
            if corruption == "owner"
            else "inspiration_archive:"
        )
        assert caught.value.mismatch_key.startswith(expected_prefix)
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_automatic_archive_allows_backdated_evaluation_but_not_completion(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_archival_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-archive-time.sqlite3",
    )
    try:
        await _seed_valid_archival_idea(
            stack,
            key="source-archive-time-idea",
            mechanism="source archive receipt time mechanism",
        )
        command_time = stack.clock.advance(timedelta(days=366))
        evaluated_at = command_time - timedelta(days=1)
        result = await run_lifecycle(
            stack,
            lifecycle_service(stack),
            key="source-archive-time-cycle",
            evaluated_at=evaluated_at,
        )
        assert result.status_code == 200
        registry = _inspiration_registry(experiences=True)
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            receipt = await uow.session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.scope == "lifecycle.run"
                )
            )
            assert receipt is not None
            receipt.completed_at = receipt.created_at - timedelta(microseconds=1)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_archive:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_automatic_archive_receipt_matches_a_canonical_lifecycle_request(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_archival_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-archive-hash.sqlite3",
    )
    try:
        await _seed_valid_archival_idea(
            stack,
            key="source-archive-hash-idea",
            mechanism="source archive request hash mechanism",
        )
        evaluated_at = stack.clock.advance(timedelta(days=365))
        result = await run_lifecycle(
            stack,
            lifecycle_service(stack),
            key="source-archive-hash-cycle",
            evaluated_at=evaluated_at,
        )
        assert result.status_code == 200
        registry = _inspiration_registry(experiences=True)
        await _validate(stack, registry)

        async with stack.database.transaction() as uow:
            receipt = await uow.session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.scope == "lifecycle.run"
                )
            )
            assert receipt is not None
            receipt.request_hash = "f" * 64

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key.startswith("inspiration_archive:")
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_automatic_archive_must_be_due_in_its_historical_state(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_archival_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-archive-due.sqlite3",
    )
    try:
        idea = await _seed_valid_archival_idea(
            stack,
            key="source-early-archive-idea",
            mechanism="source historical due mechanism",
        )
        evaluated_at = stack.clock.advance(timedelta(days=365))
        result = await run_lifecycle(
            stack,
            lifecycle_service(stack),
            key="source-early-archive-cycle",
            evaluated_at=evaluated_at,
        )
        assert result.status_code == 200
        registry = _inspiration_registry(experiences=True)
        await _validate(stack, registry)

        forged_at = ORIGIN + timedelta(days=1)
        async with stack.database.transaction() as uow:
            await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
            event = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.idea_archived"
                )
            )
            assert event is not None
            await uow.session.execute(
                update(DomainEventRow)
                .where(DomainEventRow.event_id == event.event_id)
                .values(occurred_at=forged_at)
            )
            receipt = await uow.session.get(
                IdempotencyRecordRow,
                event.causation_id,
            )
            assert receipt is not None and receipt.response_body is not None
            body = json.loads(receipt.response_body)
            body["data"]["evaluated_at"] = "2026-01-02T12:00:00.000000Z"
            receipt.response_body = canonical_json_bytes(body)

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, registry)
        assert caught.value.mismatch_key == f"inspiration_archive:{idea.idea_id}"
    finally:
        await stack.database.dispose()


def _snapshot_source_content(marker: int) -> VersionContent:
    return VersionContent(
        body=(
            f"Frozen evidence used only by this run. Immutable source marker: {marker}."
        ),
        summary="Snapshot evidence 1.",
        mechanism="Acknowledgement releases bounded capacity.",
        tags=("adoption",),
        applicability=("bounded queue",),
        evidence=(
            TypedEvidence(
                type="experiment",
                id=f"source-validation-{marker}",
            ),
        ),
        falsifiers=("Capacity remains blocked.",),
    )


async def _generate_validated_idea(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
    key: str,
    marker: int,
) -> AdoptionSeededIdea:
    source = await create_experience(
        stack,
        owner_agent_id=owner_agent_id,
        content=_snapshot_source_content(marker),
        key=f"{key}-snapshot-source",
    )
    return await generate_idea(
        stack,
        owner_agent_id=owner_agent_id,
        key=key,
        specs=(
            experience_spec(
                marker=marker,
                experience_id=source.experience_id,
                version_id=source.version_id,
                content_hash=source.content_hash,
            ),
        ),
        marker=marker,
    )


@pytest.mark.asyncio
async def test_evaluation_receipt_is_bound_to_its_canonical_request_hash(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-evaluation-hash.sqlite3",
    )
    try:
        idea = await _generate_validated_idea(
            stack,
            owner_agent_id=OWNER_A,
            key="source-evaluation-hash-run",
            marker=89,
        )
        command_time = stack.clock.advance(timedelta(minutes=10))
        evaluated_at = command_time - timedelta(minutes=5)
        async with stack.database.read_session() as session:
            item = await session.scalar(
                select(InspirationSnapshotItemRow).where(
                    InspirationSnapshotItemRow.run_id == idea.run_id
                )
            )
        assert item is not None
        evaluation = IdeaEvaluation(
            evaluator_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            verdict=EvaluationVerdict.SUPPORTED,
            evidence=(
                SnapshotEvidenceReference(
                    id=item.snapshot_item_id,
                    stable_evidence_key=item.stable_evidence_key,
                ),
            ),
            evaluated_at=evaluated_at,
        )

        async def handler(
            uow: UnitOfWork,
            command_context: CommandContext,
        ) -> StoredResponse:
            return await stack.service.evaluate(
                uow=uow,
                evaluation=evaluation,
                command_context=command_context,
            )

        result = await stack.executor.execute(
            evaluation_command_request(
                evaluation,
                idempotency_key="source-evaluation-hash",
            ),
            handler,
        )
        assert result.status_code == 200
        await _validate(stack, stack.registry, experiences=True)

        async with stack.database.transaction() as uow:
            receipt = await uow.session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.scope == "inspiration.idea.evaluate"
                )
            )
            assert receipt is not None
            receipt.request_hash = "f" * 64

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, stack.registry, experiences=True)
        assert caught.value.mismatch_key.startswith("inspiration_evaluation:")
    finally:
        await stack.database.dispose()


@pytest.fixture
async def adopted_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[tuple[AdoptionStack, UUID]]:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-adoption.sqlite3",
    )
    idea = await _generate_validated_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="source-adoption-run",
        marker=91,
    )
    stack.clock.advance(timedelta(minutes=1))
    result = await adopt(
        stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="source-adopt",
    )
    assert result.status_code == 200
    try:
        yield stack, idea.idea_id
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_idea_decision_follows_its_run_terminal(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    await _validate(stack, stack.registry, experiences=True)

    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        terminal = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == "inspiration.completed"
            )
        )
        adoption = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert terminal is not None
        assert adoption is not None
        await _swap_domain_event_contents(uow, terminal, adoption)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, stack.registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_idea_history:")


@pytest.mark.asyncio
async def test_adoption_provenance_matches_idea_run_evidence_and_result(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    registry = stack.registry
    await _validate(stack, registry, experiences=True)
    event = await adopt(
        stack,
        owner_agent_id=OWNER_A,
        idea_id=adopted_stack[1],
        key="source-adopt-same-parameter-retry",
        importance=0.4,
        confidence=0.35,
    )
    assert event.status_code == 200
    await _validate(stack, registry, experiences=True)

    second = await _generate_validated_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="source-second-run",
        marker=92,
    )
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER idea_adoption_records_reject_update")
        )
        record = await uow.session.scalar(select(IdeaAdoptionRecordRow))
        assert record is not None
        await uow.session.execute(
            update(IdeaAdoptionRecordRow)
            .where(IdeaAdoptionRecordRow.adoption_id == record.adoption_id)
            .values(run_id=second.run_id)
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_legacy_v1_created_adoption_remains_source_valid(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    legacy = await _downgrade_adoption_event_to_v1(stack)

    assert legacy.created
    await _validate(stack, stack.registry, experiences=True)
    assert (await stack.manager.verify(stack.database)).matches

    same = await adopt(
        stack,
        owner_agent_id=legacy.owner_agent_id,
        idea_id=legacy.idea_id,
        key="source-legacy-created-same-retry",
        importance=0.4,
        confidence=0.35,
    )
    assert same.status_code == 200
    mismatch = await adopt(
        stack,
        owner_agent_id=legacy.owner_agent_id,
        idea_id=legacy.idea_id,
        key="source-legacy-created-mismatch",
        importance=0.9,
        confidence=0.8,
    )
    assert mismatch.status_code == 409
    await _validate(stack, stack.registry, experiences=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_decision", ("adopted", "rejected"))
async def test_archived_idea_may_be_explicitly_adopted_or_rejected(
    repository_root: Path,
    tmp_path: Path,
    terminal_decision: str,
) -> None:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path
        / f"inspiration-source-archived-{terminal_decision}.sqlite3",
    )
    try:
        idea = await _generate_validated_idea(
            stack,
            owner_agent_id=OWNER_A,
            key=f"source-archived-{terminal_decision}-run",
            marker=98,
        )
        reason = StructuredReason.from_user_text(
            "Keep this idea outside active incubation."
        )
        archived = await _execute_idea_decision(
            stack,
            ArchiveIdea(
                owner_agent_id=OWNER_A,
                idea_id=idea.idea_id,
                reason=reason,
            ),
            key=f"source-archived-{terminal_decision}-archive",
        )
        assert archived.status_code == 200

        if terminal_decision == "adopted":
            result = await adopt(
                stack,
                owner_agent_id=OWNER_A,
                idea_id=idea.idea_id,
                key="source-archived-adopt",
            )
            assert result.status_code == 200
        else:
            rejected = await _execute_idea_decision(
                stack,
                RejectIdea(
                    owner_agent_id=OWNER_A,
                    idea_id=idea.idea_id,
                    reason=reason,
                ),
                key="source-archived-reject",
            )
            assert rejected.status_code == 200

        await _validate(stack, stack.registry, experiences=True)
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_adoption_receipt_response_is_bound_to_the_adoption_event(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    registry = stack.registry
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert event is not None
        payload = InspirationIdeaAdoptedV2.model_validate_json(event.payload)
        receipt = await uow.session.get(
            IdempotencyRecordRow,
            event.causation_id,
        )
        assert receipt is not None
        receipt.response_body = canonical_json_bytes(
            {
                "data": {
                    "created": not payload.created,
                    "experience": {
                        "current_content_hash": "f" * 64,
                        "current_version_id": payload.resulting_version_id,
                        "experience_id": payload.resulting_experience_id,
                        "owner_agent_id": payload.owner_agent_id,
                        "temperature": "warm",
                    },
                }
            }
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_adoption_receipt_is_bound_to_its_canonical_request_hash(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    await _validate(stack, stack.registry, experiences=True)

    async with stack.database.transaction() as uow:
        receipt = await uow.session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt"
            )
        )
        assert receipt is not None
        receipt.request_hash = "f" * 64

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, stack.registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_adoption_receipt_temperature_matches_historical_result_state(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    registry = stack.registry
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert event is not None
        receipt = await uow.session.get(IdempotencyRecordRow, event.causation_id)
        assert receipt is not None and receipt.response_body is not None
        body = json.loads(receipt.response_body)
        body["data"]["experience"]["temperature"] = Temperature.HOT.value
        receipt.response_body = canonical_json_bytes(body)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("temperature", Temperature.HOT.value),
        ("source_trust", 0.5),
    ),
)
@pytest.mark.asyncio
async def test_created_adoption_result_keeps_its_exact_initial_lifecycle_mapping(
    adopted_stack: tuple[AdoptionStack, UUID],
    field: str,
    forged_value: str | float,
) -> None:
    stack, _ = adopted_stack
    await _validate(stack, stack.registry, experiences=True)

    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        adoption_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert adoption_event is not None
        adoption = InspirationIdeaAdoptedV2.model_validate_json(adoption_event.payload)
        assert adoption.created
        result_events = (
            await uow.session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "experience",
                    DomainEventRow.aggregate_id == adoption.resulting_experience_id,
                    DomainEventRow.event_id < adoption_event.event_id,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        assert len(result_events) == 2
        for result_event in result_events:
            document = json.loads(result_event.payload)
            for key in ("before", "after"):
                snapshot = document.get(key)
                if isinstance(snapshot, dict):
                    snapshot[field] = forged_value
            result_event.payload = canonical_json_bytes(document)
        if field == "temperature":
            receipt = await uow.session.get(
                IdempotencyRecordRow,
                adoption_event.causation_id,
            )
            assert receipt is not None and receipt.response_body is not None
            response = json.loads(receipt.response_body)
            response["data"]["experience"]["temperature"] = forged_value
            receipt.response_body = canonical_json_bytes(response)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, stack.registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_adoption_causal_receipt_time_encloses_its_event(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    registry = stack.registry
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert event is not None
        receipt = await uow.session.get(IdempotencyRecordRow, event.causation_id)
        assert receipt is not None
        receipt.created_at = event.occurred_at + timedelta(days=1)
        receipt.completed_at = event.occurred_at + timedelta(days=1)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, registry, experiences=True)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_adoption_command_cannot_hide_an_extra_experience_side_effect(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert event is not None
        payload = InspirationIdeaAdoptedV2.model_validate_json(event.payload)
        version_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.causation_id == event.causation_id,
                DomainEventRow.event_type == "experience.version_created",
            )
        )
        assert version_event is not None
        uow.session.add(
            DomainEventRow(
                aggregate_type="experience",
                aggregate_id=payload.resulting_experience_id,
                sequence=3,
                event_type=version_event.event_type,
                payload=version_event.payload,
                actor_agent_id=payload.owner_agent_id,
                causation_id=event.causation_id,
                occurred_at=event.occurred_at,
            )
        )

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, stack.registry)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.asyncio
async def test_adoption_source_rejects_archived_historical_result_state(
    adopted_stack: tuple[AdoptionStack, UUID],
) -> None:
    stack, _ = adopted_stack
    registry = stack.registry
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        adoption_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert adoption_event is not None
        adoption = InspirationIdeaAdoptedV2.model_validate_json(adoption_event.payload)
        state_event = await uow.session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "experience",
                DomainEventRow.aggregate_id == adoption.resulting_experience_id,
                DomainEventRow.event_id < adoption_event.event_id,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        assert state_event is not None
        document = json.loads(state_event.payload)
        historical_temperature = document["after"]["temperature"]
        document["after"]["temperature"] = Temperature.ARCHIVED.value
        if "before" in document:
            document["before"]["temperature"] = Temperature.ARCHIVED.value
        state_event.payload = canonical_json_bytes(document)
        receipt = await uow.session.get(
            IdempotencyRecordRow,
            adoption_event.causation_id,
        )
        assert receipt is not None and receipt.response_body is not None
        body = json.loads(receipt.response_body)
        assert body["data"]["experience"]["temperature"] == historical_temperature
        body["data"]["experience"]["temperature"] = Temperature.ARCHIVED.value
        receipt.response_body = canonical_json_bytes(body)

    with pytest.raises(validation.SourceIntegrityError) as caught:
        await _validate(stack, registry)
    assert caught.value.mismatch_key.startswith("inspiration_adoption:")


@pytest.mark.parametrize("corruption", ("time", "request_hash", "owner"))
@pytest.mark.asyncio
async def test_manual_decision_receipt_is_bound_to_event_and_request(
    repository_root: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=(tmp_path / f"inspiration-source-decision-{corruption}.sqlite3"),
    )
    try:
        idea = await _generate_validated_idea(
            stack,
            owner_agent_id=OWNER_A,
            key="source-decision-time-run",
            marker=95,
        )
        reason = StructuredReason.from_user_text(
            "Retain this proposal outside active incubation."
        )
        command = ArchiveIdea(
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            reason=reason,
        )

        async def handler(
            uow: UnitOfWork,
            command_context: CommandContext,
        ) -> StoredResponse:
            return await stack.service.archive(
                uow=uow,
                command=command,
                command_context=command_context,
            )

        result = await stack.executor.execute(
            CommandRequest(
                caller_scope=f"agent:{OWNER_A}",
                operation_scope="inspiration.idea.archive",
                idempotency_key="source-decision-time-archive",
                method="POST",
                route_template="/v1/agents/{agent_id}/ideas/{idea_id}:archive",
                path_parameters={
                    "agent_id": OWNER_A,
                    "idea_id": idea.idea_id,
                },
                body={"reason": reason.model_dump(mode="json")},
            ),
            handler,
        )
        assert result.status_code == 200
        await _validate(stack, stack.registry, experiences=True)

        async with stack.database.transaction() as uow:
            event = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == "inspiration.idea_archived"
                )
            )
            assert event is not None
            receipt = await uow.session.get(
                IdempotencyRecordRow,
                event.causation_id,
            )
            assert receipt is not None
            if corruption == "time":
                receipt.created_at = event.occurred_at + timedelta(days=1)
                receipt.completed_at = event.occurred_at + timedelta(days=1)
            elif corruption == "request_hash":
                receipt.request_hash = "f" * 64
            else:
                await uow.session.execute(
                    text("DROP TRIGGER domain_events_reject_update")
                )
                document = json.loads(event.payload)
                document["owner_agent_id"] = str(OWNER_B)
                event.payload = canonical_json_bytes(document)
                receipt.request_hash = decision_command_request(
                    ArchiveIdea(
                        owner_agent_id=OWNER_B,
                        idea_id=idea.idea_id,
                        reason=reason,
                    ),
                    idempotency_key=receipt.idempotency_key,
                ).request_hash

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, stack.registry, experiences=True)
        expected_prefix = (
            "inspiration_idea_history:"
            if corruption == "owner"
            else "inspiration_archive:"
        )
        assert caught.value.mismatch_key.startswith(expected_prefix)
    finally:
        await stack.database.dispose()


EVIDENCE_CONTENT = VersionContent(
    body="Owned observations remain immutable during inspiration.",
    summary="Owned inspiration evidence",
    mechanism="A bounded observation anchors one hypothesis.",
    tags=("inspiration", "owned"),
    applicability=("bounded observation",),
    evidence=(TypedEvidence(type="experiment", id="source-validation"),),
    falsifiers=("The immutable source changes after capture.",),
)


@pytest.mark.asyncio
async def test_reused_hypothesis_keeps_its_existing_links(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-reused-links.sqlite3",
    )
    try:
        source = await create_experience(
            stack,
            owner_agent_id=OWNER_A,
            content=EVIDENCE_CONTENT,
            key="source-reused-links-evidence",
        )
        idea = await generate_idea(
            stack,
            owner_agent_id=OWNER_A,
            key="source-reused-links-run",
            specs=(
                experience_spec(
                    marker=93,
                    experience_id=source.experience_id,
                    version_id=source.version_id,
                    content_hash=source.content_hash,
                ),
            ),
            marker=93,
        )
        await create_experience(
            stack,
            owner_agent_id=OWNER_A,
            content=mapped_content(idea),
            key="source-reused-links-target",
        )
        stack.clock.advance(timedelta(minutes=1))
        result = await adopt(
            stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="source-reused-links-adopt",
        )
        assert result.status_code == 200
        assert json.loads(result.body)["data"]["created"] is False

        await _validate(stack, stack.registry, experiences=True)
        legacy = await _downgrade_adoption_event_to_v1(stack)
        assert not legacy.created
        await _validate(stack, stack.registry, experiences=True)
        assert (await stack.manager.verify(stack.database)).matches

        retry = await adopt(
            stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="source-reused-links-legacy-retry",
            importance=0.91,
            confidence=0.83,
        )
        assert retry.status_code == 200
        assert retry.body == result.body
        await _validate(stack, stack.registry, experiences=True)
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_snapshot_experience_evidence_must_belong_to_run_owner(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-source-owner-isolation.sqlite3",
    )
    try:
        foreign = await create_experience(
            stack,
            owner_agent_id=OWNER_B,
            content=EVIDENCE_CONTENT,
            key="source-foreign-owner-evidence",
        )
        await generate_idea(
            stack,
            owner_agent_id=OWNER_A,
            key="source-foreign-owner-run",
            specs=(
                experience_spec(
                    marker=94,
                    experience_id=foreign.experience_id,
                    version_id=foreign.version_id,
                    content_hash=foreign.content_hash,
                ),
            ),
            marker=94,
        )

        with pytest.raises(validation.SourceIntegrityError) as caught:
            await _validate(stack, stack.registry, experiences=True)
        assert caught.value.mismatch_key.startswith("inspiration_snapshot_item:")
    finally:
        await stack.database.dispose()
