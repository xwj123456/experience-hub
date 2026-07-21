from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_idea_adoption import (
    OWNER_A,
    OWNER_B,
    AdoptionStack,
    InjectedFailure,
    SeededIdea,
    adopt,
    adoption_data,
    adoption_row_counts,
    build_adoption_stack,
    capsule_spec,
    create_experience,
    error_code,
    experience_spec,
    generate_idea,
    mapped_content,
    receipt_for,
)
from tests.integration.test_inspiration_run import NOW

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    TypedEvidence,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.inspiration.events import (
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
)
from experience_hub.inspiration.models import (
    IdeaOwnerDecision,
    MechanismMaturity,
)
from experience_hub.inspiration.projector import (
    InspirationProjectionIntegrityError,
)
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaStateRow,
    IdempotencyRecordRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


@pytest.fixture
async def provenance_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "idea-adoption-provenance.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


SOURCE_CONTENT = VersionContent(
    body="Owned evidence remains immutable and version scoped.",
    summary="Owned source evidence",
    mechanism="Acknowledgement releases bounded capacity.",
    tags=("owned", "source"),
    applicability=("bounded queue",),
    evidence=(TypedEvidence(type="experiment", id="owned-source"),),
    falsifiers=("The source version changes after capture.",),
)


async def _forge_reused_adoption(
    stack: AdoptionStack,
    *,
    idea: SeededIdea,
    target_experience_id: UUID,
    target_version_id: UUID,
    adoption_id: UUID,
    key: str,
) -> CommandResult:
    adopted_at = stack.clock.now()

    async def handler(
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> StoredResponse:
        uow.session.add(
            IdeaAdoptionRecordRow(
                adoption_id=adoption_id,
                owner_agent_id=OWNER_A,
                idea_id=idea.idea_id,
                run_id=idea.run_id,
                snapshot_hash=idea.snapshot_hash,
                evidence_snapshot_item_ids=canonical_json_bytes(
                    tuple(reference.id for reference in idea.evidence)
                ),
                evidence_stable_keys=canonical_json_bytes(
                    tuple(reference.stable_evidence_key for reference in idea.evidence)
                ),
                resulting_experience_id=target_experience_id,
                resulting_version_id=target_version_id,
                adopted_at=adopted_at,
            )
        )
        await uow.session.flush()
        await stack.receipts.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="idea_adoption",
            resource_id=adoption_id,
        )
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=idea.idea_id,
                    event_type=InspirationIdeaAdoptedV1.event_type,
                    payload=InspirationIdeaAdoptedV1(
                        schema_version=1,
                        adoption_id=adoption_id,
                        idea_id=idea.idea_id,
                        run_id=idea.run_id,
                        owner_agent_id=OWNER_A,
                        snapshot_hash=idea.snapshot_hash,
                        evidence=idea.evidence,
                        resulting_experience_id=target_experience_id,
                        resulting_version_id=target_version_id,
                        created=False,
                        mechanism_cluster_id=idea.mechanism_cluster_id,
                        owner_decision_before=IdeaOwnerDecision.ACTIVE,
                        owner_decision_after=IdeaOwnerDecision.ADOPTED,
                        distinct_adopter_count_before=0,
                        distinct_adopter_count_after=1,
                        maturity_before=MechanismMaturity.SPECULATIVE,
                        maturity_after=MechanismMaturity.SPECULATIVE,
                        candidate_since_before=None,
                        candidate_since_after=None,
                        last_signal_at_before=(adopted_at - timedelta(minutes=1)),
                        last_signal_at_after=adopted_at,
                    ),
                    actor_agent_id=OWNER_A,
                    occurred_at=adopted_at,
                ),
            ),
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": {"forged": True}}),
        )

    return await stack.executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_A}",
            operation_scope="inspiration.idea.adopt",
            idempotency_key=key,
            method="POST",
            route_template="/v1/agents/{agent_id}/ideas/{idea_id}:adopt",
            path_parameters={
                "agent_id": OWNER_A,
                "idea_id": idea.idea_id,
            },
            body={"importance": 0.40, "confidence": 0.35},
        ),
        handler,
    )


@pytest.mark.asyncio
async def test_adoption_event_cannot_redirect_an_idea_to_unrelated_content(
    provenance_stack: AdoptionStack,
) -> None:
    unrelated = await create_experience(
        provenance_stack,
        owner_agent_id=OWNER_A,
        content=SOURCE_CONTENT,
        key="seed-unrelated-adoption-target",
        kind=ExperienceKind.HYPOTHESIS,
        origin=ExperienceOrigin.LOCAL,
    )
    idea = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="redirected-adoption-run",
        marker=79,
    )
    provenance_stack.clock.advance(timedelta(minutes=1))
    adoption_id = UUID("00000000-0000-0000-0000-000000009999")

    with pytest.raises(
        InspirationProjectionIntegrityError,
        match="mapped hypothesis",
    ):
        await _forge_reused_adoption(
            provenance_stack,
            idea=idea,
            target_experience_id=unrelated.experience_id,
            target_version_id=unrelated.version_id,
            adoption_id=adoption_id,
            key="redirect-idea-adoption",
        )

    async with provenance_stack.database.read_session() as session:
        record = await session.get(IdeaAdoptionRecordRow, adoption_id)
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert record is None
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_adoption_event_cannot_reuse_an_archived_equivalent(
    provenance_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="archived-forged-adoption-run",
        marker=80,
    )
    archived = await create_experience(
        provenance_stack,
        owner_agent_id=OWNER_A,
        content=mapped_content(idea),
        key="seed-archived-forged-adoption-target",
        temperature=Temperature.ARCHIVED,
    )
    provenance_stack.clock.advance(timedelta(minutes=1))
    adoption_id = UUID("00000000-0000-0000-0000-000000009998")

    with pytest.raises(
        InspirationProjectionIntegrityError,
        match="archived",
    ):
        await _forge_reused_adoption(
            provenance_stack,
            idea=idea,
            target_experience_id=archived.experience_id,
            target_version_id=archived.version_id,
            adoption_id=adoption_id,
            key="reuse-archived-forged-adoption",
        )

    async with provenance_stack.database.read_session() as session:
        record = await session.get(IdeaAdoptionRecordRow, adoption_id)
        state = await session.get(IdeaStateRow, idea.idea_id)
    assert record is None
    assert state is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value


@pytest.mark.asyncio
async def test_mixed_snapshot_provenance_record_link_and_event_are_exact(
    provenance_stack: AdoptionStack,
) -> None:
    source = await create_experience(
        provenance_stack,
        owner_agent_id=OWNER_A,
        content=SOURCE_CONTENT,
        key="seed-owned-adoption-source",
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.LOCAL,
        importance=0.65,
        confidence=0.75,
    )
    provenance_stack.clock.advance(timedelta(minutes=1))
    owned_spec = experience_spec(
        marker=1,
        experience_id=source.experience_id,
        version_id=source.version_id,
        content_hash=source.content_hash,
    )
    quarantined_spec = capsule_spec(81)
    idea = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="mixed-provenance-run",
        specs=(owned_spec, quarantined_spec),
    )
    generated_at = provenance_stack.clock.now()
    provenance_stack.clock.advance(timedelta(minutes=1))
    adopted_at = provenance_stack.clock.now()

    result = await adopt(
        provenance_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="adopt-mixed-provenance",
    )

    data = adoption_data(result)
    assert data["created"] is True
    resulting_experience_id = UUID(data["experience"]["experience_id"])
    resulting_version_id = UUID(data["experience"]["current_version_id"])
    receipt = await receipt_for(
        provenance_stack,
        scope="inspiration.idea.adopt",
        key="adopt-mixed-provenance",
    )
    async with provenance_stack.database.read_session() as session:
        record = await session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == idea.idea_id
            )
        )
        events = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        links = (
            await session.scalars(
                select(ExperienceLinkRow).where(
                    ExperienceLinkRow.source_version_id == resulting_version_id
                )
            )
        ).all()
        identity = await session.get(
            ExperienceRow,
            resulting_experience_id,
        )
        version = await session.get(
            ExperienceVersionRow,
            resulting_version_id,
        )
        payload = await session.get(
            ExperiencePayloadRow,
            resulting_version_id,
        )
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
        cluster = await session.get(
            MechanismIncubationRow,
            idea.mechanism_cluster_id,
        )
    assert record is not None
    assert identity is not None
    assert version is not None
    assert payload is not None
    assert idea_state is not None
    assert cluster is not None
    assert (
        record.owner_agent_id,
        record.idea_id,
        record.run_id,
        record.snapshot_hash,
        record.evidence_snapshot_item_ids,
        record.evidence_stable_keys,
        record.resulting_experience_id,
        record.resulting_version_id,
        record.adopted_at,
    ) == (
        OWNER_A,
        idea.idea_id,
        idea.run_id,
        idea.snapshot_hash,
        canonical_json_bytes(tuple(reference.id for reference in idea.evidence)),
        canonical_json_bytes(
            tuple(reference.stable_evidence_key for reference in idea.evidence)
        ),
        resulting_experience_id,
        resulting_version_id,
        adopted_at,
    )
    assert (
        receipt.result_resource_type,
        receipt.result_resource_id,
    ) == ("idea_adoption", record.adoption_id)

    assert tuple(event.event_type for event in events) == (
        "experience.created",
        "experience.version_created",
        "inspiration.idea_adopted_v2",
    )
    assert {event.occurred_at for event in events} == {adopted_at}
    assert {event.actor_agent_id for event in events} == {OWNER_A}
    assert tuple(event.aggregate_type for event in events) == (
        "experience",
        "experience",
        "idea",
    )
    assert tuple(event.aggregate_id for event in events) == (
        resulting_experience_id,
        resulting_experience_id,
        idea.idea_id,
    )
    adopted_payload = provenance_stack.registry.decode(
        event_type=events[-1].event_type,
        payload=events[-1].payload,
    )
    assert isinstance(adopted_payload, InspirationIdeaAdoptedV2)
    assert (
        adopted_payload.adoption_id,
        adopted_payload.idea_id,
        adopted_payload.run_id,
        adopted_payload.owner_agent_id,
        adopted_payload.snapshot_hash,
        adopted_payload.evidence,
        adopted_payload.resulting_experience_id,
        adopted_payload.resulting_version_id,
        adopted_payload.created,
        adopted_payload.mechanism_cluster_id,
        adopted_payload.owner_decision_before,
        adopted_payload.owner_decision_after,
        adopted_payload.distinct_adopter_count_before,
        adopted_payload.distinct_adopter_count_after,
        adopted_payload.maturity_before,
        adopted_payload.maturity_after,
        adopted_payload.candidate_since_before,
        adopted_payload.candidate_since_after,
        adopted_payload.last_signal_at_before,
        adopted_payload.last_signal_at_after,
    ) == (
        record.adoption_id,
        idea.idea_id,
        idea.run_id,
        OWNER_A,
        idea.snapshot_hash,
        idea.evidence,
        resulting_experience_id,
        resulting_version_id,
        True,
        idea.mechanism_cluster_id,
        IdeaOwnerDecision.ACTIVE,
        IdeaOwnerDecision.ADOPTED,
        0,
        1,
        MechanismMaturity.SPECULATIVE,
        MechanismMaturity.SPECULATIVE,
        None,
        None,
        generated_at,
        adopted_at,
    )
    assert (
        idea_state.owner_decision,
        idea_state.resulting_experience_id,
        idea_state.resulting_version_id,
        idea_state.last_signal_at,
        idea_state.projection_event_id,
    ) == (
        IdeaOwnerDecision.ADOPTED.value,
        resulting_experience_id,
        resulting_version_id,
        adopted_at,
        events[-1].event_id,
    )
    assert (
        cluster.distinct_adopter_count,
        cluster.maturity,
        cluster.candidate_since,
        cluster.last_signal_at,
        cluster.projection_event_id,
    ) == (
        1,
        MechanismMaturity.SPECULATIVE.value,
        None,
        adopted_at,
        events[-1].event_id,
    )

    assert len(links) == 1
    assert (
        links[0].source_experience_id,
        links[0].source_version_id,
        links[0].target_experience_id,
        links[0].relation,
        links[0].source_event_id,
    ) == (
        resulting_experience_id,
        resulting_version_id,
        source.experience_id,
        LinkRelation.DERIVED_FROM,
        events[1].event_id,
    )
    adopted_content = decode_and_verify_version(
        identity=identity,
        version=version,
        payload=payload,
    )
    assert adopted_content == mapped_content(idea)
    assert adopted_content.evidence == tuple(
        sorted(
            (
                TypedEvidence(
                    type="inspiration_evidence",
                    id=owned_spec.stable_key,
                ),
                TypedEvidence(
                    type="inspiration_evidence",
                    id=quarantined_spec.stable_key,
                ),
            ),
            key=canonical_json_bytes,
        )
    )
    assert (await provenance_stack.manager.verify(provenance_stack.database)).matches


@pytest.mark.asyncio
async def test_failure_after_all_event_appends_rolls_back_every_adoption_write(
    provenance_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="atomic-adoption-run",
        marker=91,
    )
    provenance_stack.clock.advance(timedelta(minutes=1))
    before = await adoption_row_counts(provenance_stack)
    provenance_stack.fault.arm(
        FaultCheckpoint.AFTER_EVENT_APPEND,
        ordinal=2,
    )

    with pytest.raises(InjectedFailure):
        await adopt(
            provenance_stack,
            owner_agent_id=OWNER_A,
            idea_id=idea.idea_id,
            key="atomic-adoption",
        )

    provenance_stack.fault.clear()
    assert await adoption_row_counts(provenance_stack) == before
    async with provenance_stack.database.read_session() as session:
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
        cluster = await session.get(
            MechanismIncubationRow,
            idea.mechanism_cluster_id,
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "inspiration.idea.adopt",
                IdempotencyRecordRow.idempotency_key == "atomic-adoption",
            )
        )
    assert idea_state is not None
    assert cluster is not None
    assert receipt is None
    assert idea_state.owner_decision == IdeaOwnerDecision.ACTIVE.value
    assert (
        cluster.distinct_adopter_count,
        cluster.maturity,
        cluster.candidate_since,
    ) == (0, MechanismMaturity.SPECULATIVE.value, None)

    retried = await adopt(
        provenance_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="atomic-adoption",
    )
    assert adoption_data(retried)["created"] is True


@pytest.mark.asyncio
async def test_idea_and_cluster_clock_regressions_are_atomic(
    provenance_stack: AdoptionStack,
) -> None:
    target = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="idea-clock-target-run",
        marker=101,
    )
    provenance_stack.clock.advance(timedelta(microseconds=-1))
    before_idea = await adoption_row_counts(provenance_stack)
    idea_regression = await adopt(
        provenance_stack,
        owner_agent_id=OWNER_A,
        idea_id=target.idea_id,
        key="idea-clock-regression",
    )
    assert idea_regression.status_code == 409
    assert error_code(idea_regression) == "clock_regression"
    assert await adoption_row_counts(provenance_stack) == before_idea

    provenance_stack.clock.advance(timedelta(microseconds=1, minutes=2))
    recurrence = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_B,
        key="newer-cluster-signal-run",
        marker=102,
    )
    assert recurrence.mechanism_cluster_id == target.mechanism_cluster_id
    provenance_stack.clock.advance(timedelta(minutes=-1))
    before_cluster = await adoption_row_counts(provenance_stack)
    cluster_regression = await adopt(
        provenance_stack,
        owner_agent_id=OWNER_A,
        idea_id=target.idea_id,
        key="cluster-clock-regression",
    )
    assert cluster_regression.status_code == 409
    assert error_code(cluster_regression) == "clock_regression"
    assert await adoption_row_counts(provenance_stack) == before_cluster
    async with provenance_stack.database.read_session() as session:
        state = await session.get(IdeaStateRow, target.idea_id)
        cluster = await session.get(
            MechanismIncubationRow,
            target.mechanism_cluster_id,
        )
    assert state is not None
    assert cluster is not None
    assert state.owner_decision == IdeaOwnerDecision.ACTIVE.value
    assert cluster.distinct_adopter_count == 0
    assert cluster.last_signal_at == NOW + timedelta(minutes=2)


@pytest.mark.asyncio
async def test_equivalent_experience_clock_regression_prevents_provenance(
    provenance_stack: AdoptionStack,
) -> None:
    idea = await generate_idea(
        provenance_stack,
        owner_agent_id=OWNER_A,
        key="equivalent-clock-target-run",
        marker=111,
    )
    future_at = NOW + timedelta(minutes=2)
    equivalent = await create_experience(
        provenance_stack,
        owner_agent_id=OWNER_A,
        content=mapped_content(idea),
        key="future-equivalent-experience",
        confidence=0.67,
        occurred_at=future_at,
    )
    before = await adoption_row_counts(provenance_stack)

    result = await adopt(
        provenance_stack,
        owner_agent_id=OWNER_A,
        idea_id=idea.idea_id,
        key="equivalent-experience-clock-regression",
    )

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    assert await adoption_row_counts(provenance_stack) == before
    async with provenance_stack.database.read_session() as session:
        experience_state = await session.get(
            ExperienceStateRow,
            equivalent.experience_id,
        )
        idea_state = await session.get(IdeaStateRow, idea.idea_id)
        adoption = await session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == idea.idea_id
            )
        )
    assert experience_state is not None
    assert idea_state is not None
    assert adoption is None
    assert experience_state.confidence == 0.67
    assert experience_state.last_transition_at == future_at
    assert idea_state.owner_decision == IdeaOwnerDecision.ACTIVE.value
