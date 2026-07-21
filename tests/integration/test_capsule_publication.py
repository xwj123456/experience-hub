from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import func, select, text, update

from experience_hub.agents import AgentCreated
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    EventRegistry,
    PendingEvent,
    StoredEvent,
)
from experience_hub.domain.values import TypedEvidence
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
    encode_version_content,
)
from experience_hub.experiences.contracts import (
    CreateExperienceVersion,
    ExperienceDraft,
    ShareableExperienceVersion,
)
from experience_hub.experiences.events import register_experience_events
from experience_hub.experiences.projector import ExperienceProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import ExperienceService
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.sharing.events import (
    CapsulePublishedV1,
    CapsuleReceivedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import PublishCapsule, Subscription
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.service import SharingService
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CommandExecutor,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    AgentRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceStateRow,
    IdempotencyRecordRow,
    TopicRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_experience_source_validator,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
OTHER_AGENT_ID = UUID("00000000-0000-0000-0000-000000000102")
TOPIC_ID = UUID("00000000-0000-0000-0000-000000000201")

EXPERIENCE_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(301, 321)
)
VERSION_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(401, 441)
)
WRITER_IDS = (
    tuple(
        item
        for pair in zip(EXPERIENCE_IDS, VERSION_IDS[: len(EXPERIENCE_IDS)], strict=True)
        for item in pair
    )
    + VERSION_IDS[len(EXPERIENCE_IDS) :]
)
CAPSULE_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(501, 561)
)
RECEIPT_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}") for value in range(601, 701)
)


def content(label: str) -> VersionContent:
    return VersionContent(
        body=f"Preserve exact transferable body for {label}.",
        summary=f"Summary {label}",
        mechanism=f"Mechanism {label}",
        tags=("distributed-memory", label),
        applicability=("explicit propagation",),
        evidence=(TypedEvidence(type="experiment", id=f"evidence:{label}"),),
        falsifiers=("the transfer changes semantic content",),
    )


class InjectedFailure(RuntimeError):
    pass


class FailAt:
    checkpoint: FaultCheckpoint | None = None

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is self.checkpoint:
            raise InjectedFailure(checkpoint.value)


@dataclass(slots=True)
class PublicationStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    receipts: ReceiptStore
    writer: ExperienceWriter
    experience_service: ExperienceService
    experience_query: ExperienceQuery
    sharing_ids: SequenceIdGenerator
    fault: FailAt
    registry: EventRegistry

    def sharing_service(
        self,
        *,
        experience_query: ExperienceQuery | Any | None = None,
    ) -> SharingService:
        return SharingService(
            clock=self.clock,
            id_generator=self.sharing_ids,
            receipt_store=self.receipts,
            repository=SharingRepository(),
            experience_query=experience_query or self.experience_query,
        )


async def build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> PublicationStack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    registry.register(AgentCreated)
    register_experience_events(registry)
    register_sharing_events(registry)
    lifecycle = LifecycleConfig()
    experience_projector = ExperienceProjector(registry, lifecycle)
    source_validator = SourceValidator(registry)
    register_experience_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry((experience_projector,)),
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
                AgentRow(agent_id=OWNER_ID, name="Owner", created_at=NOW),
                AgentRow(
                    agent_id=OTHER_AGENT_ID,
                    name="Other",
                    created_at=NOW,
                ),
                TopicRow(
                    topic_id=TOPIC_ID,
                    owner_agent_id=OWNER_ID,
                    name="publication-tests",
                    description=None,
                    created_at=NOW,
                ),
            )
        )

    clock = FrozenClock(NOW)
    receipts = ReceiptStore(
        clock=clock,
        id_generator=SequenceIdGenerator(RECEIPT_IDS),
    )
    repository = ExperienceRepository(event_registry=registry)
    writer = ExperienceWriter(
        id_generator=SequenceIdGenerator(WRITER_IDS),
        repository=repository,
        lifecycle_config=lifecycle,
    )
    experience_service = ExperienceService(
        clock=clock,
        receipt_store=receipts,
        writer=writer,
        lifecycle_config=lifecycle,
    )
    return PublicationStack(
        database=database,
        clock=clock,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        receipts=receipts,
        writer=writer,
        experience_service=experience_service,
        experience_query=ExperienceQuery(event_registry=registry),
        sharing_ids=SequenceIdGenerator(CAPSULE_IDS),
        fault=fault,
        registry=registry,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[PublicationStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-publication.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def command_request(
    *,
    key: str,
    operation_scope: str,
    owner_agent_id: UUID,
    caller_agent_id: UUID | None = None,
    body: dict[str, Any],
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{caller_agent_id or owner_agent_id}",
        operation_scope=operation_scope,
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/capsules",
        path_parameters={"agent_id": owner_agent_id},
        body=body,
    )


async def create_experience(
    stack: PublicationStack,
    *,
    key: str,
    value: VersionContent,
    owner_agent_id: UUID = OWNER_ID,
    origin: ExperienceOrigin = ExperienceOrigin.LOCAL,
) -> tuple[UUID, UUID]:
    request = command_request(
        key=key,
        operation_scope="experience.create",
        owner_agent_id=owner_agent_id,
        body={"label": key},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        creation = await stack.writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=owner_agent_id,
                actor_agent_id=owner_agent_id,
                kind=ExperienceKind.PROCEDURAL,
                origin=origin,
                content=value,
                importance=0.5,
                confidence=0.65,
                source_trust=1.0,
                initial_temperature=Temperature.WARM,
                links=(),
                occurred_at=stack.clock.now(),
            ),
            command=context,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "experience_id": creation.experience_id,
                        "version_id": creation.version_id,
                        "content_hash": creation.content_hash,
                    }
                }
            ),
        )

    result = await stack.executor.execute(request, handler)
    assert result.status_code == 201
    data = json.loads(result.body)["data"]
    return UUID(data["experience_id"]), UUID(data["version_id"])


async def correct_experience(
    stack: PublicationStack,
    *,
    key: str,
    experience_id: UUID,
    value: VersionContent,
) -> UUID:
    command = CreateExperienceVersion(
        owner_agent_id=OWNER_ID,
        experience_id=experience_id,
        content=value,
    )
    request = command_request(
        key=key,
        operation_scope="experience.create_version",
        owner_agent_id=OWNER_ID,
        body={"label": key},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.experience_service.create_version(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(request, handler)
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["version_id"])


async def publish_capsule(
    stack: PublicationStack,
    *,
    key: str,
    experience_id: UUID,
    version_id: UUID | None = None,
    owner_agent_id: UUID = OWNER_ID,
    caller_agent_id: UUID | None = None,
    topic_id: UUID = TOPIC_ID,
    expires_at: datetime | None = None,
    parent_adoption_id: UUID | None = None,
    operation_scope: str = "capsule.publish",
    service: SharingService | None = None,
) -> tuple[int, dict[str, Any], bool]:
    expiry = expires_at or stack.clock.now() + timedelta(days=7)
    command = PublishCapsule(
        owner_agent_id=owner_agent_id,
        topic_id=topic_id,
        experience_id=experience_id,
        version_id=version_id,
        expires_at=expiry,
        parent_adoption_id=parent_adoption_id,
    )
    request = command_request(
        key=key,
        operation_scope=operation_scope,
        owner_agent_id=owner_agent_id,
        caller_agent_id=caller_agent_id,
        body={
            "topic_id": topic_id,
            "experience_id": experience_id,
            "version_id": version_id,
            "expires_at": expiry,
            "parent_adoption_id": parent_adoption_id,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await (service or stack.sharing_service()).publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(request, handler)
    return result.status_code, json.loads(result.body), result.replayed


async def seed_parent_adoption(
    stack: PublicationStack,
    *,
    experience_id: UUID,
    version_id: UUID,
    adoption_id: UUID,
    adopter_agent_id: UUID = OWNER_ID,
    hop_count: int = 1,
    root_fingerprint: str = "a" * 64,
    adopted_at: datetime | None = None,
) -> tuple[UUID, tuple[dict[str, str], ...]]:
    """Create an immutable owned chain ending at one adopted experience."""
    if hop_count < 1:
        raise ValueError("A parent adoption must contain its capsule hop")
    capsule_id = UUID(int=adoption_id.int + 10_000)
    chain = tuple(
        {
            "capsule_id": str(
                capsule_id
                if index == hop_count - 1
                else UUID(int=adoption_id.int + 20_000 + index)
            ),
            "publisher_agent_id": str(OTHER_AGENT_ID),
        }
        for index in range(hop_count)
    )
    encoded = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content(f"parent-{adoption_id}"),
    )
    async with stack.database.transaction() as uow:
        uow.session.add(
            ExperienceCapsuleRow(
                capsule_id=capsule_id,
                transport_schema_version=1,
                topic_id=TOPIC_ID,
                source_experience_id=experience_id,
                source_version_id=version_id,
                publisher_agent_id=OTHER_AGENT_ID,
                kind=ExperienceKind.PROCEDURAL,
                body=f"Parent body {adoption_id}",
                summary=f"Parent summary {adoption_id}",
                mechanism=f"Parent mechanism {adoption_id}",
                tags=canonical_json_bytes(("parent",)),
                applicability=canonical_json_bytes(()),
                evidence=canonical_json_bytes(()),
                falsifiers=canonical_json_bytes(()),
                publisher_confidence=0.5,
                provenance_chain=canonical_json_bytes(chain[:-1]),
                root_fingerprint=root_fingerprint,
                source_content_hash=encoded.content_hash,
                created_at=NOW - timedelta(days=2),
                expires_at=NOW + timedelta(days=30),
                hop_count=len(chain) - 1,
                capsule_hash="b" * 64,
            )
        )
        await uow.session.flush()
        uow.session.add(
            AdoptionRecordRow(
                adoption_id=adoption_id,
                adopter_agent_id=adopter_agent_id,
                capsule_id=capsule_id,
                resulting_experience_id=experience_id,
                captured_trust=0.5,
                provenance_chain=canonical_json_bytes(chain),
                root_fingerprint=root_fingerprint,
                corroboration_applied=False,
                adopted_at=adopted_at or NOW - timedelta(days=1),
            )
        )
    return capsule_id, chain


async def capsule_rows(
    stack: PublicationStack,
) -> tuple[ExperienceCapsuleRow, ...]:
    async with stack.database.read_session() as session:
        return tuple(
            (
                await session.scalars(
                    select(ExperienceCapsuleRow).order_by(
                        ExperienceCapsuleRow.created_at,
                        ExperienceCapsuleRow.capsule_id,
                    )
                )
            ).all()
        )


@pytest.mark.asyncio
async def test_publish_selects_current_or_explicit_owned_version(
    stack: PublicationStack,
) -> None:
    original = content("history-v1")
    corrected = content("history-v2")
    experience_id, version_one_id = await create_experience(
        stack,
        key="history",
        value=original,
    )
    stack.clock.advance(timedelta(minutes=1))
    version_two_id = await correct_experience(
        stack,
        key="history-v2",
        experience_id=experience_id,
        value=corrected,
    )

    current_status, _, _ = await publish_capsule(
        stack,
        key="publish-current",
        experience_id=experience_id,
    )
    selected_status, _, _ = await publish_capsule(
        stack,
        key="publish-selected",
        experience_id=experience_id,
        version_id=version_one_id,
    )

    rows = await capsule_rows(stack)
    published = rows[-2:]
    assert current_status == selected_status == 201
    assert [row.source_version_id for row in published] == [
        version_two_id,
        version_one_id,
    ]
    assert [row.body for row in published] == [
        corrected.body,
        original.body,
    ]
    assert [row.source_content_hash for row in published] == [
        encode_version_content(
            kind=ExperienceKind.PROCEDURAL,
            content=corrected,
        ).content_hash,
        encode_version_content(
            kind=ExperienceKind.PROCEDURAL,
            content=original,
        ).content_hash,
    ]


@pytest.mark.asyncio
async def test_publish_hides_foreign_experience_and_refuses_archived_source(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="owned",
        value=content("owned"),
    )

    foreign_status, foreign, _ = await publish_capsule(
        stack,
        key="publish-foreign",
        experience_id=experience_id,
        owner_agent_id=OTHER_AGENT_ID,
    )
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(temperature=Temperature.ARCHIVED)
        )
    archived_status, archived, _ = await publish_capsule(
        stack,
        key="publish-archived",
        experience_id=experience_id,
    )

    assert foreign_status == 404
    assert foreign["error"]["code"] == "experience_not_found"
    assert archived_status == 409
    assert archived["error"]["code"] == "restore_required"
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delta",
    (timedelta(0), -timedelta(microseconds=1)),
)
async def test_expiry_must_be_strictly_future(
    stack: PublicationStack,
    delta: timedelta,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key=f"expiry-{delta.total_seconds()}",
        value=content(f"expiry-{delta.total_seconds()}"),
    )

    status, body, _ = await publish_capsule(
        stack,
        key=f"publish-expiry-{delta.total_seconds()}",
        experience_id=experience_id,
        expires_at=stack.clock.now() + delta,
    )

    assert status == 422
    assert body["error"]["code"] == "invalid_expiry"
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
async def test_publish_rejects_clock_regression_against_shareable_head(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="future-causal",
        value=content("future-causal"),
    )
    future = stack.clock.now() + timedelta(minutes=1)
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == experience_id)
            .values(strength_updated_at=future)
        )

    status, body, _ = await publish_capsule(
        stack,
        key="publish-before-causal-head",
        experience_id=experience_id,
    )

    assert status == 409
    assert body["error"]["code"] == "clock_regression"
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
async def test_publish_rejects_clock_regression_against_topic_creation(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="future-topic",
        value=content("future-topic"),
    )
    async with stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER topics_reject_update"))
        await uow.session.execute(
            update(TopicRow)
            .where(TopicRow.topic_id == TOPIC_ID)
            .values(created_at=NOW + timedelta(minutes=1))
        )

    status, body, _ = await publish_capsule(
        stack,
        key="publish-before-topic",
        experience_id=experience_id,
    )

    assert status == 409
    assert body["error"]["code"] == "clock_regression"
    assert not await capsule_rows(stack)


class RewritingExperienceQuery:
    def __init__(
        self,
        delegate: ExperienceQuery,
        rewrite: Callable[
            [ShareableExperienceVersion],
            ShareableExperienceVersion,
        ],
    ) -> None:
        self._delegate = delegate
        self._rewrite = rewrite

    async def get_owned_shareable_version(
        self,
        **values: Any,
    ) -> ShareableExperienceVersion:
        selected = await self._delegate.get_owned_shareable_version(**values)
        return self._rewrite(selected)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("content_hash", "evidence"))
async def test_publish_revalidates_evidence_and_reproduces_semantic_hash(
    stack: PublicationStack,
    corruption: str,
) -> None:
    value = content(f"integrity-{corruption}")
    experience_id, _ = await create_experience(
        stack,
        key=f"integrity-{corruption}",
        value=value,
    )

    def rewrite(selected: ShareableExperienceVersion) -> ShareableExperienceVersion:
        if corruption == "content_hash":
            return replace(selected, content_hash="f" * 64)
        invalid = selected.content.model_copy(
            update={"evidence": ({"type": "", "id": "not-typed"},)}
        )
        return replace(selected, content=invalid)

    service = stack.sharing_service(
        experience_query=RewritingExperienceQuery(
            stack.experience_query,
            rewrite,
        )
    )

    with pytest.raises(SourceIntegrityError):
        await publish_capsule(
            stack,
            key=f"publish-integrity-{corruption}",
            experience_id=experience_id,
            service=service,
        )
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
async def test_publish_revalidates_constructed_evidence_with_matching_hash(
    stack: PublicationStack,
) -> None:
    value = content("constructed-evidence")
    experience_id, _ = await create_experience(
        stack,
        key="constructed-evidence",
        value=value,
    )

    def rewrite(
        selected: ShareableExperienceVersion,
    ) -> ShareableExperienceVersion:
        invalid_evidence = TypedEvidence.model_construct(type="", id="")
        invalid_content = selected.content.model_copy(
            update={"evidence": (invalid_evidence,)}
        )
        matching_hash = encode_version_content(
            kind=selected.kind,
            content=invalid_content,
        ).content_hash
        return replace(
            selected,
            content=invalid_content,
            content_hash=matching_hash,
        )

    service = stack.sharing_service(
        experience_query=RewritingExperienceQuery(
            stack.experience_query,
            rewrite,
        )
    )

    with pytest.raises(SourceIntegrityError):
        await publish_capsule(
            stack,
            key="publish-constructed-evidence",
            experience_id=experience_id,
            service=service,
        )
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
async def test_original_publication_reproduces_transferable_fields_and_hashes(
    stack: PublicationStack,
) -> None:
    value = content("canonical")
    experience_id, version_id = await create_experience(
        stack,
        key="canonical",
        value=value,
    )

    status, body, _ = await publish_capsule(
        stack,
        key="publish-canonical",
        experience_id=experience_id,
        expires_at=NOW + timedelta(days=3),
    )

    rows = await capsule_rows(stack)
    assert status == 201
    assert len(rows) == 1
    capsule = rows[0]
    assert (
        capsule.source_experience_id,
        capsule.source_version_id,
        capsule.publisher_agent_id,
        capsule.kind,
        capsule.body,
        capsule.summary,
        capsule.mechanism,
        capsule.publisher_confidence,
        capsule.hop_count,
    ) == (
        experience_id,
        version_id,
        OWNER_ID,
        ExperienceKind.PROCEDURAL,
        value.body,
        value.summary,
        value.mechanism,
        0.65,
        0,
    )
    assert capsule.provenance_chain == canonical_json_bytes(())
    assert capsule.tags == canonical_json_bytes(value.tags)
    assert capsule.applicability == canonical_json_bytes(value.applicability)
    assert capsule.evidence == canonical_json_bytes(value.evidence)
    assert capsule.falsifiers == canonical_json_bytes(value.falsifiers)
    assert (
        capsule.source_content_hash
        == encode_version_content(
            kind=ExperienceKind.PROCEDURAL,
            content=value,
        ).content_hash
    )
    assert len(capsule.root_fingerprint) == len(capsule.capsule_hash) == 64
    assert body["data"]["root_fingerprint"] == capsule.root_fingerprint
    assert body["data"]["capsule_hash"] == capsule.capsule_hash

    async with stack.database.read_session() as session:
        event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsulePublishedV1.event_type
            )
        )
    assert event is not None
    payload = CapsulePublishedV1.model_validate_json(event.payload)
    assert payload.capsule_id == capsule.capsule_id
    assert payload.capsule_hash == capsule.capsule_hash
    assert b'"body"' not in event.payload


@pytest.mark.asyncio
async def test_capsule_origin_requires_one_owned_parent_for_selected_experience(
    stack: PublicationStack,
) -> None:
    adopted_id, adopted_version_id = await create_experience(
        stack,
        key="adopted",
        value=content("adopted"),
        origin=ExperienceOrigin.ADOPTED_CAPSULE,
    )
    other_id, other_version_id = await create_experience(
        stack,
        key="other",
        value=content("other"),
    )
    valid_adoption_id = UUID("00000000-0000-0000-0000-000000000701")
    wrong_result_id = UUID("00000000-0000-0000-0000-000000000702")
    foreign_adoption_id = UUID("00000000-0000-0000-0000-000000000703")
    await seed_parent_adoption(
        stack,
        experience_id=adopted_id,
        version_id=adopted_version_id,
        adoption_id=valid_adoption_id,
    )
    await seed_parent_adoption(
        stack,
        experience_id=other_id,
        version_id=other_version_id,
        adoption_id=wrong_result_id,
    )
    await seed_parent_adoption(
        stack,
        experience_id=adopted_id,
        version_id=adopted_version_id,
        adoption_id=foreign_adoption_id,
        adopter_agent_id=OTHER_AGENT_ID,
    )

    cases = (
        ("missing", None),
        ("wrong-result", wrong_result_id),
        ("foreign", foreign_adoption_id),
    )
    outcomes = [
        await publish_capsule(
            stack,
            key=f"publish-adopted-{label}",
            experience_id=adopted_id,
            parent_adoption_id=parent_id,
        )
        for label, parent_id in cases
    ]
    success = await publish_capsule(
        stack,
        key="publish-adopted-valid",
        experience_id=adopted_id,
        parent_adoption_id=valid_adoption_id,
    )

    assert [status for status, _, _ in outcomes] == [422, 404, 404]
    assert success[0] == 201
    published = [
        row for row in await capsule_rows(stack) if row.publisher_agent_id == OWNER_ID
    ]
    assert len(published) == 1


@pytest.mark.asyncio
async def test_publish_rejects_clock_regression_against_parent_adoption(
    stack: PublicationStack,
) -> None:
    experience_id, version_id = await create_experience(
        stack,
        key="future-parent-adoption",
        value=content("future-parent-adoption"),
    )
    adoption_id = UUID("00000000-0000-0000-0000-000000000704")
    await seed_parent_adoption(
        stack,
        experience_id=experience_id,
        version_id=version_id,
        adoption_id=adoption_id,
        adopted_at=NOW + timedelta(minutes=1),
    )

    status, body, _ = await publish_capsule(
        stack,
        key="publish-before-parent-adoption",
        experience_id=experience_id,
        parent_adoption_id=adoption_id,
    )

    assert status == 409
    assert body["error"]["code"] == "clock_regression"
    assert all(row.publisher_agent_id != OWNER_ID for row in await capsule_rows(stack))


@pytest.mark.asyncio
async def test_named_chain_extends_for_local_and_corrected_adopted_version(
    stack: PublicationStack,
) -> None:
    local_id, local_version_id = await create_experience(
        stack,
        key="local-chain",
        value=content("local-chain"),
    )
    local_parent_id = UUID("00000000-0000-0000-0000-000000000711")
    _, local_chain = await seed_parent_adoption(
        stack,
        experience_id=local_id,
        version_id=local_version_id,
        adoption_id=local_parent_id,
        hop_count=2,
        root_fingerprint="c" * 64,
    )
    adopted_id, adopted_version_id = await create_experience(
        stack,
        key="adopted-correction",
        value=content("adopted-correction-v1"),
        origin=ExperienceOrigin.ADOPTED_CAPSULE,
    )
    adopted_parent_id = UUID("00000000-0000-0000-0000-000000000712")
    _, adopted_chain = await seed_parent_adoption(
        stack,
        experience_id=adopted_id,
        version_id=adopted_version_id,
        adoption_id=adopted_parent_id,
        root_fingerprint="d" * 64,
    )
    stack.clock.advance(timedelta(minutes=1))
    corrected_version_id = await correct_experience(
        stack,
        key="adopted-correction-v2",
        experience_id=adopted_id,
        value=content("adopted-correction-v2"),
    )

    local = await publish_capsule(
        stack,
        key="publish-local-chain",
        experience_id=local_id,
        parent_adoption_id=local_parent_id,
    )
    corrected = await publish_capsule(
        stack,
        key="publish-corrected-adopted",
        experience_id=adopted_id,
        version_id=corrected_version_id,
        parent_adoption_id=adopted_parent_id,
    )

    assert local[0] == corrected[0] == 201
    published = [
        row for row in await capsule_rows(stack) if row.publisher_agent_id == OWNER_ID
    ]
    assert [json.loads(row.provenance_chain) for row in published] == [
        list(local_chain),
        list(adopted_chain),
    ]
    assert [row.hop_count for row in published] == [2, 1]
    assert [row.root_fingerprint for row in published] == [
        "c" * 64,
        "d" * 64,
    ]


@pytest.mark.asyncio
async def test_publication_allows_hop_four_and_rejects_longer_chain(
    stack: PublicationStack,
) -> None:
    allowed_id, allowed_version = await create_experience(
        stack,
        key="hop-four",
        value=content("hop-four"),
    )
    rejected_id, rejected_version = await create_experience(
        stack,
        key="hop-five",
        value=content("hop-five"),
    )
    allowed_parent = UUID("00000000-0000-0000-0000-000000000721")
    rejected_parent = UUID("00000000-0000-0000-0000-000000000722")
    await seed_parent_adoption(
        stack,
        experience_id=allowed_id,
        version_id=allowed_version,
        adoption_id=allowed_parent,
        hop_count=4,
    )
    await seed_parent_adoption(
        stack,
        experience_id=rejected_id,
        version_id=rejected_version,
        adoption_id=rejected_parent,
        hop_count=5,
    )

    allowed = await publish_capsule(
        stack,
        key="publish-hop-four",
        experience_id=allowed_id,
        parent_adoption_id=allowed_parent,
    )
    rejected = await publish_capsule(
        stack,
        key="publish-hop-five",
        experience_id=rejected_id,
        parent_adoption_id=rejected_parent,
    )

    assert allowed[0] == 201
    assert rejected[0] == 409
    assert rejected[1]["error"]["code"] == "max_provenance_hops"
    published = [
        row for row in await capsule_rows(stack) if row.publisher_agent_id == OWNER_ID
    ]
    assert len(published) == 1
    assert published[0].hop_count == 4


@pytest.mark.asyncio
async def test_publish_requires_matching_caller_and_operation_scope(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="scope",
        value=content("scope"),
    )

    foreign = await publish_capsule(
        stack,
        key="publish-foreign-caller",
        experience_id=experience_id,
        caller_agent_id=OTHER_AGENT_ID,
    )
    wrong_operation = await publish_capsule(
        stack,
        key="publish-wrong-operation",
        experience_id=experience_id,
        operation_scope="capsule.retract",
    )

    assert foreign[0] == wrong_operation[0] == 404
    assert foreign[1] == wrong_operation[1]
    assert foreign[1]["error"]["code"] == "resource_not_found"
    assert not await capsule_rows(stack)


@pytest.mark.asyncio
async def test_publish_is_idempotent_and_creates_one_source_and_event(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="idempotent",
        value=content("idempotent"),
    )

    first = await publish_capsule(
        stack,
        key="publish-idempotent",
        experience_id=experience_id,
    )
    replay = await publish_capsule(
        stack,
        key="publish-idempotent",
        experience_id=experience_id,
    )

    async with stack.database.read_session() as session:
        event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == CapsulePublishedV1.event_type)
        )
    assert first[0] == replay[0] == 201
    assert first[1] == replay[1]
    assert first[2] is False and replay[2] is True
    assert len(await capsule_rows(stack)) == 1
    assert event_count == 1


@pytest.mark.asyncio
async def test_publish_delivers_large_topics_in_bounded_ordered_pages(
    stack: PublicationStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="paged-delivery",
        value=content("paged-delivery"),
    )
    subscribers = tuple(UUID(int=20_000 + index) for index in range(101))
    subscriptions = tuple(
        Subscription(
            subscription_id=UUID(int=30_000 + index),
            subscriber_agent_id=subscriber_id,
            topic_id=TOPIC_ID,
            creation_event_id=1,
            created_at=NOW,
        )
        for index, subscriber_id in enumerate(subscribers)
    )
    capsule_id = UUID(int=40_000)
    item_ids = tuple(UUID(int=50_000 + index) for index in range(101))
    repository = SharingRepository()
    cursors: list[tuple[UUID, UUID] | None] = []

    async def paged_subscriptions(
        *,
        session: Any,
        topic_id: UUID,
        publication_event_id: int,
        after: tuple[UUID, UUID] | None = None,
        limit: int = 100,
        exclude_subscriber_agent_id: UUID | None = None,
    ) -> tuple[Subscription, ...]:
        del session, publication_event_id
        assert topic_id == TOPIC_ID
        assert exclude_subscriber_agent_id == OWNER_ID
        cursors.append(after)
        start = 0
        if after is not None:
            start = next(
                index + 1
                for index, item in enumerate(subscriptions)
                if (
                    item.subscriber_agent_id,
                    item.subscription_id,
                )
                == after
            )
        return subscriptions[start : start + limit]

    monkeypatch.setattr(
        repository,
        "list_eligible_subscriptions",
        paged_subscriptions,
    )
    batch_sizes: list[int] = []
    original_append = UnitOfWork.append_events

    async def recording_append(
        self: UnitOfWork,
        command: CommandContext,
        events: Sequence[PendingEvent],
    ) -> tuple[StoredEvent, ...]:
        if events and events[0].event_type == CapsuleReceivedV1.event_type:
            batch_sizes.append(len(events))
        return await original_append(self, command, events)

    monkeypatch.setattr(UnitOfWork, "append_events", recording_append)
    service = SharingService(
        clock=stack.clock,
        id_generator=SequenceIdGenerator((capsule_id, *item_ids)),
        receipt_store=stack.receipts,
        repository=repository,
        experience_query=stack.experience_query,
    )

    status, _, _ = await publish_capsule(
        stack,
        key="publish-paged-delivery",
        experience_id=experience_id,
        service=service,
    )

    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == CapsuleReceivedV1.event_type)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    payloads = tuple(
        stack.registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
        for row in rows
    )
    assert status == 201
    assert cursors == [
        None,
        (
            subscriptions[99].subscriber_agent_id,
            subscriptions[99].subscription_id,
        ),
    ]
    assert batch_sizes == [100, 1]
    assert [
        (
            payload.recipient_agent_id,
            payload.item_id,
        )
        for payload in payloads
        if isinstance(payload, CapsuleReceivedV1)
    ] == list(zip(subscribers, item_ids, strict=True))


@pytest.mark.asyncio
async def test_publish_rolls_back_source_event_and_receipt_atomically(
    stack: PublicationStack,
) -> None:
    experience_id, _ = await create_experience(
        stack,
        key="rollback",
        value=content("rollback"),
    )
    stack.fault.checkpoint = FaultCheckpoint.AFTER_SOURCE_INSERT

    with pytest.raises(InjectedFailure, match="after_source_insert"):
        await publish_capsule(
            stack,
            key="publish-rollback",
            experience_id=experience_id,
        )

    async with stack.database.read_session() as session:
        capsule_count = await session.scalar(
            select(func.count()).select_from(ExperienceCapsuleRow)
        )
        publication_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == CapsulePublishedV1.event_type)
        )
        receipt_count = await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.scope == "capsule.publish",
                IdempotencyRecordRow.idempotency_key == "publish-rollback",
            )
        )
    assert (capsule_count, publication_count, receipt_count) == (0, 0, 0)
