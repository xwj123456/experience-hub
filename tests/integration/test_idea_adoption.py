from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, text, update
from tests.integration.test_inspiration_run import (
    NOW,
    FakeGenerator,
    ImmediateDeadlineRunner,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    PendingEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences.contracts import (
    ExperienceCreation,
    ExperienceDraft,
    VersionLinkInput,
)
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
    register_experience_events,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
    decode_and_verify_version,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import AdoptIdea, StartInspirationRun
from experience_hub.inspiration.deadlines import BoundedGenerationRunner
from experience_hub.inspiration.events import (
    InspirationIdeaAdoptedV2,
    InspirationIdeaArchivedV1,
    InspirationIdeaRejectedV1,
    register_inspiration_events,
)
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.hashing import (
    hash_snapshot,
    stable_evidence_key,
    truncate_utf8,
)
from experience_hub.inspiration.lifecycle import IdeaLifecycleService
from experience_hub.inspiration.models import (
    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
    EvidenceSourceState,
    EvidenceSourceType,
    FrozenSnapshot,
    IdeaDraft,
    IdeaOwnerDecision,
    InspirationOperator,
    MechanismMaturity,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
)
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.service import InspirationRunExecutor
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandResult,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    InspirationIdeaRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceValidator,
    register_experience_source_validator,
    register_inspiration_source_validator,
)

OWNER_A = UUID("00000000-0000-0000-0000-000000000101")
OWNER_B = UUID("00000000-0000-0000-0000-000000000102")
OUTSIDER = UUID("00000000-0000-0000-0000-000000000103")

RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(1001, 1301)
)
RUN_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(2001, 2401)
)
WRITER_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(3001, 3401)
)
ADOPTION_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(4001, 4201)
)


class AdvancingClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        retained = self.current
        self.current += timedelta(microseconds=1)
        return retained


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
class SnapshotSpec:
    snapshot_item_id: UUID
    source_type: EvidenceSourceType
    source_id: UUID
    source_version_id: UUID
    source_state: EvidenceSourceState
    source_trust: float
    content_hash: str

    @property
    def stable_key(self) -> str:
        return stable_evidence_key(
            source_type=self.source_type,
            source_id=self.source_id,
            source_version_id=self.source_version_id,
            content_hash=self.content_hash,
        )


def capsule_spec(marker: int) -> SnapshotSpec:
    return SnapshotSpec(
        snapshot_item_id=uid(5000 + marker * 3),
        source_type=EvidenceSourceType.CAPSULE,
        source_id=uid(5001 + marker * 3),
        source_version_id=uid(5002 + marker * 3),
        source_state=EvidenceSourceState.QUARANTINED,
        source_trust=0.25,
        content_hash=f"{marker:064x}",
    )


def experience_spec(
    *,
    marker: int,
    experience_id: UUID,
    version_id: UUID,
    content_hash: str,
) -> SnapshotSpec:
    return SnapshotSpec(
        snapshot_item_id=uid(7000 + marker),
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=experience_id,
        source_version_id=version_id,
        source_state=EvidenceSourceState.WARM,
        source_trust=1.0,
        content_hash=content_hash,
    )


@dataclass(slots=True)
class QueuedSnapshotBuilder:
    queued: list[tuple[SnapshotSpec, ...]] = field(default_factory=list)

    def enqueue(self, specs: tuple[SnapshotSpec, ...]) -> None:
        if not specs:
            raise ValueError("a generated idea requires snapshot evidence")
        self.queued.append(specs)

    async def freeze(
        self,
        *,
        uow: UnitOfWork,
        request: StartInspirationRun,
        run_id: UUID,
        at: Any,
    ) -> FrozenSnapshot:
        _ = (uow, request)
        if not self.queued:
            raise RuntimeError("no queued snapshot")
        specs = self.queued.pop(0)
        items: list[SnapshotItem] = []
        for rank, spec in enumerate(specs, start=1):
            if spec.source_type is EvidenceSourceType.EXPERIENCE:
                identity = await uow.session.get(
                    ExperienceRow,
                    spec.source_id,
                )
                version = await uow.session.get(
                    ExperienceVersionRow,
                    spec.source_version_id,
                )
                payload = await uow.session.get(
                    ExperiencePayloadRow,
                    spec.source_version_id,
                )
                if identity is None or version is None or payload is None:
                    raise AssertionError(
                        "queued experience evidence must reference a real version"
                    )
                content = decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=payload,
                )
                summary = content.summary
                mechanism = content.mechanism
                applicability = content.applicability
                tags = content.tags
                falsifiers = content.falsifiers
                excerpt = truncate_utf8(
                    content.body,
                    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
                )
            else:
                summary = f"Snapshot evidence {rank}."
                mechanism = "Acknowledgement releases bounded capacity."
                applicability = ("bounded queue",)
                tags = ("adoption",)
                falsifiers = ("Capacity remains blocked.",)
                excerpt = "Frozen evidence used only by this run."
            items.append(
                SnapshotItem(
                    snapshot_item_id=spec.snapshot_item_id,
                    stable_evidence_key=spec.stable_key,
                    run_id=run_id,
                    source_type=spec.source_type,
                    source_id=spec.source_id,
                    source_version_id=spec.source_version_id,
                    source_state=spec.source_state,
                    source_trust=spec.source_trust,
                    rank=rank,
                    summary=summary,
                    mechanism=mechanism,
                    applicability=applicability,
                    tags=tags,
                    falsifiers=falsifiers,
                    excerpt=excerpt,
                    content_hash=spec.content_hash,
                    captured_at=at,
                )
            )
        frozen_items = tuple(items)
        return FrozenSnapshot(
            run_id=run_id,
            items=frozen_items,
            snapshot_hash=hash_snapshot(frozen_items),
            frozen_at=at,
        )


@dataclass(slots=True)
class AllEvidenceGenerator(FakeGenerator):
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
        source = generated.ideas[0]
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title=source.title,
                    hypothesis=source.hypothesis,
                    mechanism=source.mechanism,
                    predictions=source.predictions,
                    falsifiers=source.falsifiers,
                    assumptions=source.assumptions,
                    proposed_test=source.proposed_test,
                    evidence=tuple(
                        SnapshotEvidenceReference(
                            id=item.snapshot_item_id,
                            stable_evidence_key=item.stable_evidence_key,
                        )
                        for item in frozen_items
                    ),
                ),
            ),
            output_tokens_consumed=generated.output_tokens_consumed,
        )


class InjectedFailure(RuntimeError):
    pass


@dataclass(slots=True)
class FailAt:
    checkpoint: FaultCheckpoint | None = None
    ordinal: int = 1
    seen: int = 0

    def arm(self, checkpoint: FaultCheckpoint, *, ordinal: int) -> None:
        self.checkpoint = checkpoint
        self.ordinal = ordinal
        self.seen = 0

    def clear(self) -> None:
        self.checkpoint = None
        self.ordinal = 1
        self.seen = 0

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is not self.checkpoint:
            return
        self.seen += 1
        if self.seen == self.ordinal:
            raise InjectedFailure(checkpoint.value)


@dataclass(slots=True)
class AdoptionStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    run_executor: InspirationRunExecutor
    service: IdeaLifecycleService
    registry: EventRegistry
    receipts: ReceiptStore
    experience_writer: ExperienceWriter
    experience_repository: ExperienceRepository
    inspiration_repository: InspirationRepository
    snapshot_builder: QueuedSnapshotBuilder
    manager: ProjectionManager
    fault: FailAt


async def build_adoption_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> AdoptionStack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    register_experience_events(registry)
    register_inspiration_events(registry)
    lifecycle = LifecycleConfig()
    source_validator = SourceValidator(registry)
    register_experience_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry(
            (
                ExperienceProjector(registry, lifecycle),
                InspirationRunProjector(registry),
                MechanismIncubationProjector(registry),
                IdeaStateProjector(registry),
            )
        ),
        source_validator=source_validator,
    )
    fault = FailAt()
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
        fault_injector=fault,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            (
                AgentRow(agent_id=OWNER_A, name="Owner A", created_at=NOW),
                AgentRow(agent_id=OWNER_B, name="Owner B", created_at=NOW),
                AgentRow(agent_id=OUTSIDER, name="Outsider", created_at=NOW),
            )
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    experience_repository = ExperienceRepository(event_registry=registry)
    experience_writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(WRITER_IDS),
        repository=experience_repository,
        lifecycle_config=lifecycle,
    )
    inspiration_repository = InspirationRepository(registry)
    snapshot_builder = QueuedSnapshotBuilder()
    generator = AllEvidenceGenerator()
    run_executor = InspirationRunExecutor(
        database=database,
        receipt_store=receipts,
        repository=inspiration_repository,
        snapshot_builder=snapshot_builder,
        generator_factory=lambda _kind: generator,
        generation_runner=BoundedGenerationRunner(
            deadline_runner=ImmediateDeadlineRunner(),
        ),
        response_codec=InspirationResponseCodec(),
        clock=clock,
        id_generator=SequenceIdGenerator(RUN_IDS),
    )
    service = IdeaLifecycleService(
        clock=clock,
        receipt_store=receipts,
        repository=inspiration_repository,
        id_generator=SequenceIdGenerator(ADOPTION_IDS),
        experience_writer=experience_writer,
        experience_repository=experience_repository,
    )
    return AdoptionStack(
        database=database,
        clock=clock,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        run_executor=run_executor,
        service=service,
        registry=registry,
        receipts=receipts,
        experience_writer=experience_writer,
        experience_repository=experience_repository,
        inspiration_repository=inspiration_repository,
        snapshot_builder=snapshot_builder,
        manager=manager,
        fault=fault,
    )


@pytest.fixture
async def adoption_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "idea-adoption.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


@dataclass(frozen=True, slots=True)
class SeededIdea:
    idea_id: UUID
    run_id: UUID
    owner_agent_id: UUID
    operator: InspirationOperator
    title: str
    hypothesis: str
    mechanism: str
    predictions: tuple[str, ...]
    falsifiers: tuple[str, ...]
    assumptions: tuple[str, ...]
    proposed_test: str
    evidence: tuple[SnapshotEvidenceReference, ...]
    snapshot_hash: str
    mechanism_cluster_id: str


def run_request(run: StartInspirationRun, *, key: str) -> CommandRequest:
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


def _json_tuple(raw: bytes) -> tuple[str, ...]:
    values = json.loads(raw)
    assert isinstance(values, list)
    assert all(isinstance(value, str) for value in values)
    return tuple(values)


async def generate_idea(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
    key: str,
    specs: tuple[SnapshotSpec, ...] | None = None,
    marker: int = 1,
) -> SeededIdea:
    selected_specs = specs or (capsule_spec(marker),)
    stack.snapshot_builder.enqueue(selected_specs)
    run = StartInspirationRun(
        owner_agent_id=owner_agent_id,
        goal="Turn frozen evidence into a testable hypothesis",
        operators=(InspirationOperator.CAUSAL_GAP,),
    )
    result = await stack.run_executor.execute(
        request=run_request(run, key=key),
        run=run,
    )
    assert result.status_code == 201
    run_id = UUID(json.loads(result.body)["data"]["run_id"])
    async with stack.database.read_session() as session:
        idea = await session.scalar(
            select(InspirationIdeaRow).where(InspirationIdeaRow.run_id == run_id)
        )
        occurrence = await session.scalar(
            select(IdeaOccurrenceRow).where(IdeaOccurrenceRow.run_id == run_id)
        )
        state = None if idea is None else await session.get(IdeaStateRow, idea.idea_id)
    assert idea is not None
    assert occurrence is not None
    assert state is not None
    references = tuple(
        SnapshotEvidenceReference.model_validate_json(canonical_json_bytes(item))
        for item in json.loads(idea.evidence_references)
    )
    return SeededIdea(
        idea_id=idea.idea_id,
        run_id=idea.run_id,
        owner_agent_id=owner_agent_id,
        operator=InspirationOperator(idea.operator),
        title=idea.title,
        hypothesis=idea.hypothesis,
        mechanism=idea.mechanism,
        predictions=_json_tuple(idea.predictions),
        falsifiers=_json_tuple(idea.falsifiers),
        assumptions=_json_tuple(idea.assumptions),
        proposed_test=idea.proposed_test,
        evidence=references,
        snapshot_hash=occurrence.snapshot_hash,
        mechanism_cluster_id=state.mechanism_cluster_id,
    )


def mapped_content(idea: SeededIdea) -> VersionContent:
    return VersionContent(
        body=canonical_json_bytes(
            {
                "assumptions": idea.assumptions,
                "hypothesis": idea.hypothesis,
                "predictions": idea.predictions,
                "proposed_test": idea.proposed_test,
            }
        ).decode("utf-8"),
        summary=idea.title,
        mechanism=idea.mechanism,
        tags=("inspiration", f"operator:{idea.operator.value}"),
        applicability=idea.assumptions,
        evidence=tuple(
            TypedEvidence(
                type="inspiration_evidence",
                id=reference.stable_evidence_key,
            )
            for reference in idea.evidence
        ),
        falsifiers=idea.falsifiers,
    )


def adoption_request(command: AdoptIdea, *, key: str) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{command.owner_agent_id}",
        operation_scope="inspiration.idea.adopt",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/ideas/{idea_id}:adopt",
        path_parameters={
            "agent_id": command.owner_agent_id,
            "idea_id": command.idea_id,
        },
        body={
            "confidence": command.confidence,
            "importance": command.importance,
        },
    )


async def adopt(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
    idea_id: UUID,
    key: str,
    importance: float | None = None,
    confidence: float | None = None,
) -> CommandResult:
    values: dict[str, object] = {
        "owner_agent_id": owner_agent_id,
        "idea_id": idea_id,
    }
    if importance is not None:
        values["importance"] = importance
    if confidence is not None:
        values["confidence"] = confidence
    command = AdoptIdea(**values)

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.adopt(
            uow=uow,
            command=command,
            command_context=command_context,
        )

    return await stack.executor.execute(
        adoption_request(command, key=key),
        handler,
    )


def adoption_data(result: CommandResult) -> dict[str, Any]:
    assert result.status_code == 200
    assert canonical_json_bytes(json.loads(result.body)) == result.body
    decoded = json.loads(result.body)
    assert set(decoded) == {"data"}
    data = decoded["data"]
    assert set(data) == {"created", "experience"}
    assert set(data["experience"]) == {
        "current_content_hash",
        "current_version_id",
        "experience_id",
        "owner_agent_id",
        "temperature",
    }
    return cast(dict[str, Any], data)


def error_code(result: CommandResult) -> str:
    decoded = json.loads(result.body)
    assert canonical_json_bytes(decoded) == result.body
    return cast(str, decoded["error"]["code"])


async def receipt_for(
    stack: AdoptionStack,
    *,
    scope: str,
    key: str,
) -> IdempotencyRecordRow:
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == scope,
                IdempotencyRecordRow.idempotency_key == key,
            )
        )
    assert row is not None
    return row


async def create_experience(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
    content: VersionContent,
    key: str,
    kind: ExperienceKind = ExperienceKind.HYPOTHESIS,
    origin: ExperienceOrigin = ExperienceOrigin.LOCAL,
    importance: float = 0.40,
    confidence: float = 0.35,
    temperature: Temperature = Temperature.WARM,
    occurred_at: Any | None = None,
    links: tuple[VersionLinkInput, ...] = (),
) -> ExperienceCreation:
    draft = ExperienceDraft(
        owner_agent_id=owner_agent_id,
        actor_agent_id=owner_agent_id,
        kind=kind,
        origin=origin,
        content=content,
        importance=importance,
        confidence=confidence,
        source_trust=1.0,
        initial_temperature=temperature,
        links=links,
        occurred_at=occurred_at or stack.clock.now(),
    )
    retained: ExperienceCreation | None = None

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        nonlocal retained
        retained = await stack.experience_writer.create_from_draft(
            uow=uow,
            draft=draft,
            command=command_context,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "content_hash": retained.content_hash,
                        "experience_id": retained.experience_id,
                        "version_id": retained.version_id,
                    }
                }
            ),
        )

    result = await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{owner_agent_id}",
            operation_scope="test.experience.seed",
            idempotency_key=key,
            method="POST",
            route_template="/test/experiences",
            path_parameters={"agent_id": owner_agent_id},
            body={"content_hash_seed": content.model_dump(mode="json")},
        ),
        handler,
    )
    assert result.status_code == 201
    assert retained is not None
    return retained


async def apply_owner_decision(
    stack: AdoptionStack,
    *,
    idea: SeededIdea,
    decision: IdeaOwnerDecision,
    key: str,
) -> None:
    reason = StructuredReason.from_user_text(
        f"Move the idea to {decision.value} for this test."
    )
    if decision is IdeaOwnerDecision.ARCHIVED:
        payload = InspirationIdeaArchivedV1(
            schema_version=1,
            idea_id=idea.idea_id,
            owner_agent_id=idea.owner_agent_id,
            reason=reason,
            owner_decision_before=IdeaOwnerDecision.ACTIVE,
            owner_decision_after=IdeaOwnerDecision.ARCHIVED,
            cycle_id=None,
        )
    elif decision is IdeaOwnerDecision.REJECTED:
        payload = InspirationIdeaRejectedV1(
            schema_version=1,
            idea_id=idea.idea_id,
            owner_agent_id=idea.owner_agent_id,
            reason=reason,
            owner_decision_before=IdeaOwnerDecision.ACTIVE,
            owner_decision_after=IdeaOwnerDecision.REJECTED,
        )
    else:
        raise ValueError("only archive and reject are test arrangements")

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=idea.idea_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=idea.owner_agent_id,
                    occurred_at=stack.clock.now(),
                ),
            ),
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": {"idea_id": idea.idea_id}}),
        )

    result = await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{idea.owner_agent_id}",
            operation_scope=f"test.idea.{decision.value}",
            idempotency_key=key,
            method="POST",
            route_template=f"/test/ideas/{{idea_id}}:{decision.value}",
            path_parameters={"idea_id": idea.idea_id},
            body={"reason": reason.model_dump(mode="json")},
        ),
        handler,
    )
    assert result.status_code == 200


async def adoption_row_counts(
    stack: AdoptionStack,
) -> dict[str, int]:
    tables = {
        "experiences": ExperienceRow,
        "versions": ExperienceVersionRow,
        "payloads": ExperiencePayloadRow,
        "links": ExperienceLinkRow,
        "adoptions": IdeaAdoptionRecordRow,
        "adopted_events": DomainEventRow,
    }
    async with stack.database.read_session() as session:
        retained: dict[str, int] = {}
        for name, table in tables.items():
            statement = select(func.count()).select_from(table)
            if name == "adopted_events":
                statement = statement.where(
                    DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
                )
            retained[name] = int(await session.scalar(statement) or 0)
        return retained


def test_adopt_idea_defaults_are_strict_and_immutable() -> None:
    command = AdoptIdea(owner_agent_id=OWNER_A, idea_id=uid(9901))
    assert (command.importance, command.confidence) == (0.40, 0.35)
    with pytest.raises(FrozenInstanceError):
        command.importance = 0.90  # type: ignore[misc]

    invalid_values: tuple[object, ...] = (
        -0.01,
        1.01,
        math.nan,
        math.inf,
        True,
        "0.5",
    )
    for value in invalid_values:
        with pytest.raises(ValueError):
            AdoptIdea(
                owner_agent_id=OWNER_A,
                idea_id=uid(9902),
                importance=value,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError):
            AdoptIdea(
                owner_agent_id=OWNER_A,
                idea_id=uid(9903),
                confidence=value,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    (
        "importance",
        "confidence",
        "expected_importance",
        "expected_confidence",
        "temperature",
    ),
    (
        (None, None, 0.40, 0.35, Temperature.WARM),
        (0.90, 0.80, 0.90, 0.80, Temperature.HOT),
    ),
)
@pytest.mark.asyncio
async def test_active_adoption_maps_exact_hypothesis_and_lifecycle_defaults(
    adoption_stack: AdoptionStack,
    importance: float | None,
    confidence: float | None,
    expected_importance: float,
    expected_confidence: float,
    temperature: Temperature,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="mapping-run",
        marker=11,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))

    result = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="map-active-idea",
        importance=importance,
        confidence=confidence,
    )

    data = adoption_data(result)
    assert data["created"] is True
    experience_data = data["experience"]
    experience_id = UUID(experience_data["experience_id"])
    version_id = UUID(experience_data["current_version_id"])
    async with adoption_stack.database.read_session() as session:
        identity = await session.get(ExperienceRow, experience_id)
        version = await session.get(ExperienceVersionRow, version_id)
        payload = await session.get(ExperiencePayloadRow, version_id)
        state = await session.get(ExperienceStateRow, experience_id)
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
    assert identity is not None
    assert version is not None
    assert payload is not None
    assert state is not None
    assert idea_state is not None
    content = decode_and_verify_version(
        identity=identity,
        version=version,
        payload=payload,
    )
    assert (identity.owner_agent_id, identity.kind, identity.origin) == (
        OWNER_A,
        ExperienceKind.HYPOTHESIS,
        ExperienceOrigin.ADOPTED_IDEA,
    )
    assert content == mapped_content(idea)
    assert json.loads(content.body) == {
        "assumptions": list(idea.assumptions),
        "hypothesis": idea.hypothesis,
        "predictions": list(idea.predictions),
        "proposed_test": idea.proposed_test,
    }
    assert content.body.encode("utf-8") == canonical_json_bytes(
        json.loads(content.body)
    )
    assert (
        state.importance,
        state.confidence,
        state.source_trust,
        state.temperature,
    ) == (
        expected_importance,
        expected_confidence,
        1.0,
        temperature,
    )
    assert (
        experience_data["owner_agent_id"],
        experience_data["current_content_hash"],
        experience_data["temperature"],
    ) == (
        str(OWNER_A),
        version.content_hash,
        temperature.value,
    )
    assert (
        idea_state.owner_decision,
        idea_state.resulting_experience_id,
        idea_state.resulting_version_id,
    ) == (
        IdeaOwnerDecision.ADOPTED.value,
        experience_id,
        version_id,
    )

    receipt = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="map-active-idea",
    )
    async with adoption_stack.database.read_session() as session:
        events = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    assert tuple(event.event_type for event in events) == (
        ExperienceCreatedV1.event_type,
        ExperienceVersionCreatedV1.event_type,
        InspirationIdeaAdoptedV2.event_type,
    )


@pytest.mark.asyncio
async def test_owner_isolation_hides_private_and_unknown_ideas(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="private-adoption-run",
        marker=21,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))

    foreign = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_B,
        idea_id=idea.idea_id,
        key="foreign-adoption",
    )
    unknown = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_B,
        idea_id=uid(9999),
        key="unknown-adoption",
    )

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.body == unknown.body
    assert error_code(foreign) == "resource_not_found"
    counts = await adoption_row_counts(adoption_stack)
    assert counts["adoptions"] == counts["adopted_events"] == 0
    async with adoption_stack.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_adoption_command_is_bound_to_its_request_body(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="request-bound-adoption-run",
        marker=29,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    actual = AdoptIdea(
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
    )
    declared = AdoptIdea(
        owner_agent_id=actual.owner_agent_id,
        idea_id=actual.idea_id,
        importance=0.99,
        confidence=0.99,
    )

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        return await adoption_stack.service.adopt(
            uow=uow,
            command=actual,
            command_context=command_context,
        )

    result = await adoption_stack.executor.execute(
        adoption_request(
            declared,
            key="request-bound-adoption",
        ),
        handler,
    )

    assert result.status_code == 404
    assert error_code(result) == "resource_not_found"
    counts = await adoption_row_counts(adoption_stack)
    assert counts["adoptions"] == counts["adopted_events"] == 0
    async with adoption_stack.database.read_session() as session:
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_server_timed_adoption_never_precedes_its_receipt(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="adoption-receipt-clock-run",
        marker=30,
    )
    receipt_time = adoption_stack.clock.now() + timedelta(minutes=10)
    clock = RegressingClock(
        values=[
            receipt_time,
            receipt_time - timedelta(minutes=1),
            receipt_time + timedelta(minutes=1),
        ]
    )
    adoption_stack.receipts._clock = clock  # noqa: SLF001
    adoption_stack.executor._clock = clock  # noqa: SLF001
    adoption_stack.service._clock = clock  # noqa: SLF001

    result = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="adoption-receipt-clock",
    )

    assert result.status_code == 200
    receipt = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="adoption-receipt-clock",
    )
    async with adoption_stack.database.read_session() as session:
        adoption_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        event_times = tuple(
            await session.scalars(
                select(DomainEventRow.occurred_at)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )
    assert adoption_event is not None
    assert adoption_event.occurred_at == receipt.created_at == receipt_time
    assert set(event_times) == {receipt_time}
    assert receipt.completed_at == receipt_time + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_archived_idea_can_be_adopted_but_rejected_idea_cannot(
    adoption_stack: AdoptionStack,
) -> None:
    archived = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="archived-adoption-run",
        marker=31,
    )
    await apply_owner_decision(
        adoption_stack,
        idea=archived,
        decision=IdeaOwnerDecision.ARCHIVED,
        key="archive-before-adoption",
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    adopted = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=archived.idea_id,
        key="adopt-archived",
    )
    assert adoption_data(adopted)["created"] is True
    async with adoption_stack.database.read_session() as session:
        adopted_event_row = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type,
                DomainEventRow.aggregate_id == archived.idea_id,
            )
        )
    assert adopted_event_row is not None
    adopted_payload = adoption_stack.registry.decode(
        event_type=adopted_event_row.event_type,
        payload=adopted_event_row.payload,
    )
    assert isinstance(adopted_payload, InspirationIdeaAdoptedV2)
    assert adopted_payload.owner_decision_before is IdeaOwnerDecision.ARCHIVED

    rejected = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="rejected-adoption-run",
        marker=32,
    )
    await apply_owner_decision(
        adoption_stack,
        idea=rejected,
        decision=IdeaOwnerDecision.REJECTED,
        key="reject-before-adoption",
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    refused = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=rejected.idea_id,
        key="adopt-rejected",
    )
    assert refused.status_code == 409
    async with adoption_stack.database.read_session() as session:
        record = await session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == rejected.idea_id
            )
        )
    assert record is None


@pytest.mark.asyncio
async def test_already_adopted_returns_original_bytes_under_a_new_key(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="stable-adoption-run",
        marker=41,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="stable-adoption-first",
    )
    first_data = adoption_data(first)
    experience_id = UUID(first_data["experience"]["experience_id"])
    before = await adoption_row_counts(adoption_stack)
    async with adoption_stack.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(temperature=Temperature.HOT)
        )
        assert changed.rowcount == 1
    adoption_stack.clock.advance(timedelta(minutes=1))

    repeated = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="stable-adoption-second",
    )

    assert (
        repeated.status_code,
        repeated.body,
        repeated.content_type,
        dict(repeated.headers),
    ) == (
        first.status_code,
        first.body,
        first.content_type,
        dict(first.headers),
    )
    assert repeated.replayed is False
    assert await adoption_row_counts(adoption_stack) == before
    async with adoption_stack.database.read_session() as session:
        record = await session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == idea.idea_id
            )
        )
        receipts = (
            await session.scalars(
                select(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == "inspiration.idea.adopt")
                .order_by(IdempotencyRecordRow.created_at)
            )
        ).all()
    assert record is not None
    assert {
        (receipt.result_resource_type, receipt.result_resource_id)
        for receipt in receipts
    } == {("idea_adoption", record.adoption_id)}


@pytest.mark.asyncio
async def test_already_adopted_rejects_different_parameters(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="changed-adoption-run",
        marker=42,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="changed-adoption-first",
    )
    assert first.status_code == 200

    repeated = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="changed-adoption-second",
        importance=0.99,
        confidence=0.99,
    )

    assert repeated.status_code == 409
    assert json.loads(repeated.body)["error"]["code"] == ("adoption_request_mismatch")
    async with adoption_stack.database.read_session() as session:
        retry = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key == "changed-adoption-second",
            )
        )
    assert retry is not None
    assert retry.result_resource_type is None
    assert retry.result_resource_id is None


@pytest.mark.asyncio
async def test_already_adopted_rejects_a_regressed_retry_clock(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="regressed-retry-adoption-run",
        marker=45,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="regressed-retry-adoption-first",
    )
    assert first.status_code == 200
    adoption_stack.clock.advance(timedelta(minutes=-2))

    repeated = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="regressed-retry-adoption-second",
    )

    assert repeated.status_code == 409
    assert error_code(repeated) == "clock_regression"
    async with adoption_stack.database.read_session() as session:
        retry = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key
                == "regressed-retry-adoption-second",
            )
        )
    assert retry is not None
    assert retry.result_resource_type is None
    assert retry.result_resource_id is None


@pytest.mark.asyncio
async def test_already_adopted_uses_retry_receipt_time_at_clock_boundary(
    adoption_stack: AdoptionStack,
) -> None:
    source = await create_experience(
        adoption_stack,
        owner_agent_id=OWNER_A,
        content=VersionContent(
            body="A stable source for retry receipt boundary validation.",
            summary="Retry receipt boundary source",
            mechanism="A receipt timestamp anchors one retry command.",
            tags=("adoption", "retry"),
            applicability=("bounded retry",),
            evidence=(TypedEvidence(type="experiment", id="retry-boundary"),),
            falsifiers=("The receipt does not anchor the retry.",),
        ),
        key="boundary-retry-source",
    )
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="boundary-retry-adoption-run",
        specs=(
            experience_spec(
                marker=46,
                experience_id=source.experience_id,
                version_id=source.version_id,
                content_hash=source.content_hash,
            ),
        ),
        marker=46,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="boundary-retry-adoption-first",
    )
    assert first.status_code == 200
    async with adoption_stack.database.read_session() as session:
        adoption_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
    assert adoption_event is not None

    clock = AdvancingClock(adoption_event.occurred_at - timedelta(microseconds=1))
    adoption_stack.receipts._clock = clock  # noqa: SLF001
    adoption_stack.executor._clock = clock  # noqa: SLF001
    adoption_stack.service._clock = clock  # noqa: SLF001
    repeated = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="boundary-retry-adoption-second",
    )

    assert repeated.status_code == 409
    assert error_code(repeated) == "clock_regression"
    retry = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="boundary-retry-adoption-second",
    )
    assert retry.created_at < adoption_event.occurred_at
    assert retry.result_resource_type is None
    assert retry.result_resource_id is None
    validator = adoption_stack.manager._source_validator
    assert isinstance(validator, SourceValidator)
    register_inspiration_source_validator(validator)
    assert (await adoption_stack.manager.verify(adoption_stack.database)).matches


@pytest.mark.asyncio
async def test_already_adopted_rejects_a_tampered_original_receipt(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="tampered-receipt-adoption-run",
        marker=42,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="tampered-receipt-adoption-first",
    )
    assert first.status_code == 200
    original_receipt = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="tampered-receipt-adoption-first",
    )
    async with adoption_stack.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.receipt_id == original_receipt.receipt_id)
            .values(response_body=canonical_json_bytes({"data": {"corrupted": True}}))
        )
        assert changed.rowcount == 1

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="canonical response",
    ):
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="tampered-receipt-adoption-second",
        )

    async with adoption_stack.database.read_session() as session:
        second_receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key
                == "tampered-receipt-adoption-second",
            )
        )
    assert second_receipt is None


@pytest.mark.asyncio
async def test_already_adopted_rejects_a_receipt_created_after_its_event(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="late-receipt-adoption-run",
        marker=43,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="late-receipt-adoption-first",
    )
    assert first.status_code == 200
    original_receipt = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="late-receipt-adoption-first",
    )
    shifted = original_receipt.created_at + timedelta(days=1)
    async with adoption_stack.database.transaction(immediate=True) as uow:
        changed = await uow.session.execute(
            update(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.receipt_id == original_receipt.receipt_id)
            .values(
                created_at=shifted,
                completed_at=shifted,
            )
        )
        assert changed.rowcount == 1

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="canonical response",
    ):
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="late-receipt-adoption-second",
        )

    async with adoption_stack.database.read_session() as session:
        second_receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key == "late-receipt-adoption-second",
            )
        )
    assert second_receipt is None


@pytest.mark.asyncio
async def test_already_adopted_rejects_result_state_from_after_its_event(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="late-result-state-adoption-run",
        marker=44,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    first = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="late-result-state-adoption-first",
    )
    assert first.status_code == 200
    async with adoption_stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        adoption_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type
            )
        )
        assert adoption_event is not None
        adoption_payload = InspirationIdeaAdoptedV2.model_validate_json(
            adoption_event.payload
        )
        state_event = await uow.session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "experience",
                DomainEventRow.aggregate_id == adoption_payload.resulting_experience_id,
                DomainEventRow.event_id < adoption_event.event_id,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        assert state_event is not None
        state_event.occurred_at = adoption_event.occurred_at + timedelta(days=1)

    with pytest.raises(
        InspirationSourceIntegrityError,
        match="not current at adoption",
    ):
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="late-result-state-adoption-second",
        )

    async with adoption_stack.database.read_session() as session:
        second_receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key
                == "late-result-state-adoption-second",
            )
        )
    assert second_receipt is None


@pytest.mark.asyncio
async def test_current_equivalent_is_reused_without_confidence_change(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="equivalent-adoption-run",
        marker=51,
    )
    existing = await create_experience(
        adoption_stack,
        owner_agent_id=OWNER_A,
        content=mapped_content(idea),
        key="seed-current-equivalent",
        importance=0.62,
        confidence=0.73,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    before = await adoption_row_counts(adoption_stack)

    result = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="reuse-current-equivalent",
        importance=0.99,
        confidence=0.99,
    )

    data = adoption_data(result)
    assert data["created"] is False
    assert data["experience"]["experience_id"] == str(existing.experience_id)
    assert data["experience"]["current_version_id"] == str(existing.version_id)
    async with adoption_stack.database.read_session() as session:
        state = await session.get(
            ExperienceStateRow,
            existing.experience_id,
        )
    assert state is not None
    assert (state.importance, state.confidence) == (0.62, 0.73)
    after = await adoption_row_counts(adoption_stack)
    assert (
        after["experiences"],
        after["versions"],
        after["payloads"],
        after["links"],
        after["adoptions"],
        after["adopted_events"],
    ) == (
        before["experiences"],
        before["versions"],
        before["payloads"],
        before["links"],
        before["adoptions"] + 1,
        before["adopted_events"] + 1,
    )
    receipt = await receipt_for(
        adoption_stack,
        scope="inspiration.idea.adopt",
        key="reuse-current-equivalent",
    )
    async with adoption_stack.database.read_session() as session:
        event_types = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )
    assert event_types == (InspirationIdeaAdoptedV2.event_type,)


@pytest.mark.asyncio
async def test_archived_equivalent_requires_explicit_restore(
    adoption_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="archived-equivalent-run",
        marker=61,
    )
    equivalent = await create_experience(
        adoption_stack,
        owner_agent_id=OWNER_A,
        content=mapped_content(idea),
        key="seed-archived-equivalent",
        temperature=Temperature.ARCHIVED,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    before = await adoption_row_counts(adoption_stack)

    result = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="refuse-archived-equivalent",
    )

    assert result.status_code == 409
    assert error_code(result) == "restore_required"
    assert await adoption_row_counts(adoption_stack) == before
    async with adoption_stack.database.read_session() as session:
        experience_state = await session.get(
            ExperienceStateRow,
            equivalent.experience_id,
        )
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
    assert experience_state is not None
    assert idea_state is not None
    assert experience_state.temperature is Temperature.ARCHIVED
    assert idea_state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_same_owner_adoptions_count_once_per_cluster(
    adoption_stack: AdoptionStack,
) -> None:
    first = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="same-owner-first-run",
        marker=66,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    second = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="same-owner-second-run",
        marker=67,
    )
    assert first.mechanism_cluster_id == second.mechanism_cluster_id
    adoption_stack.clock.advance(timedelta(minutes=1))
    assert (
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_A,
            idea_id=first.idea_id,
            key="same-owner-adopts-first",
        )
    ).status_code == 200
    adoption_stack.clock.advance(timedelta(minutes=1))
    second_result = await adopt(
        adoption_stack,
        owner_agent_id=OWNER_A,
        idea_id=second.idea_id,
        key="same-owner-adopts-second",
    )

    assert second_result.status_code == 200
    async with adoption_stack.database.read_session() as session:
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        cluster = await session.get(
            MechanismIncubationRow,
            first.mechanism_cluster_id,
        )
    second_payload = adoption_stack.registry.decode(
        event_type=rows[-1].event_type,
        payload=rows[-1].payload,
    )
    assert isinstance(second_payload, InspirationIdeaAdoptedV2)
    assert (
        second_payload.distinct_adopter_count_before,
        second_payload.distinct_adopter_count_after,
        second_payload.maturity_after,
        second_payload.candidate_since_after,
    ) == (
        1,
        1,
        MechanismMaturity.INCUBATING,
        None,
    )
    assert cluster is not None
    assert (
        cluster.distinct_adopter_count,
        cluster.maturity,
        cluster.candidate_since,
    ) == (1, MechanismMaturity.INCUBATING.value, None)
    assert (await adoption_stack.manager.verify(adoption_stack.database)).matches


@pytest.mark.asyncio
async def test_two_distinct_owners_promote_cluster_on_second_adoption(
    adoption_stack: AdoptionStack,
) -> None:
    first = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_A,
        key="first-owner-cluster-run",
        marker=71,
    )
    adoption_stack.clock.advance(timedelta(minutes=1))
    second = await generate_idea(
        adoption_stack,
        owner_agent_id=OWNER_B,
        key="second-owner-cluster-run",
        marker=72,
    )
    assert first.mechanism_cluster_id == second.mechanism_cluster_id
    adoption_stack.clock.advance(timedelta(minutes=1))
    first_time = adoption_stack.clock.now()
    assert (
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_A,
            idea_id=first.idea_id,
            key="first-owner-adopts",
        )
    ).status_code == 200
    adoption_stack.clock.advance(timedelta(minutes=1))
    second_time = adoption_stack.clock.now()
    assert (
        await adopt(
            adoption_stack,
            owner_agent_id=OWNER_B,
            idea_id=second.idea_id,
            key="second-owner-adopts",
        )
    ).status_code == 200

    async with adoption_stack.database.read_session() as session:
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == InspirationIdeaAdoptedV2.event_type)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        cluster = await session.get(
            MechanismIncubationRow,
            first.mechanism_cluster_id,
        )
        states = (
            await session.scalars(
                select(IdeaStateRow).where(
                    IdeaStateRow.idea_id.in_((first.idea_id, second.idea_id))
                )
            )
        ).all()
    payloads = tuple(
        adoption_stack.registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
        for row in rows
    )
    assert all(isinstance(payload, InspirationIdeaAdoptedV2) for payload in payloads)
    first_payload, second_payload = cast(
        tuple[InspirationIdeaAdoptedV2, InspirationIdeaAdoptedV2],
        payloads,
    )
    assert (
        first_payload.distinct_adopter_count_before,
        first_payload.distinct_adopter_count_after,
        first_payload.maturity_before,
        first_payload.maturity_after,
        first_payload.candidate_since_before,
        first_payload.candidate_since_after,
        first_payload.last_signal_at_after,
    ) == (
        0,
        1,
        MechanismMaturity.INCUBATING,
        MechanismMaturity.INCUBATING,
        None,
        None,
        first_time,
    )
    assert (
        second_payload.distinct_adopter_count_before,
        second_payload.distinct_adopter_count_after,
        second_payload.maturity_before,
        second_payload.maturity_after,
        second_payload.candidate_since_before,
        second_payload.candidate_since_after,
        second_payload.last_signal_at_after,
    ) == (
        1,
        2,
        MechanismMaturity.INCUBATING,
        MechanismMaturity.CANDIDATE,
        None,
        second_time,
        second_time,
    )
    assert cluster is not None
    assert (
        cluster.distinct_adopter_count,
        cluster.maturity,
        cluster.candidate_since,
        cluster.last_signal_at,
    ) == (
        2,
        MechanismMaturity.CANDIDATE.value,
        second_time,
        second_time,
    )
    assert {state.owner_decision for state in states} == {
        IdeaOwnerDecision.ADOPTED.value
    }


__all__ = [
    "AdoptionStack",
    "InjectedFailure",
    "SeededIdea",
    "SnapshotSpec",
    "adopt",
    "adoption_data",
    "adoption_row_counts",
    "build_adoption_stack",
    "capsule_spec",
    "create_experience",
    "error_code",
    "experience_spec",
    "generate_idea",
    "mapped_content",
    "receipt_for",
    "uid",
]
