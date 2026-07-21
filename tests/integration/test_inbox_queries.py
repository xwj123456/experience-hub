from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    OTHER_AGENT_ID,
    PUBLISHER_ID,
    SOURCE_CONTENT,
    SOURCE_EXPERIENCE_ID,
    SOURCE_VERSION_ID,
    TOPIC_ID,
    AdoptionStack,
    adopt,
    build_stack,
    create_source_experience,
    create_topic,
    request,
    subscribe_adopter,
)
from tests.integration.test_capsule_feedback import record_feedback
from tests.integration.test_capsule_rejection import reject

from experience_hub.domain import CommandContext
from experience_hub.experiences import ExperienceKind, VersionContent
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue, query_cues
from experience_hub.sharing.models import (
    Capsule,
    CapsuleStatus,
    CreateSubscription,
    EffectiveAvailability,
    InboxState,
    PublishCapsule,
    RetractCapsule,
)
from experience_hub.sharing.projector import AgentReputationProjector
from experience_hub.sharing.queries import InboxEvidenceReader, SharingQuery
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    AgentReputationRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceTermRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inbox-queries.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def _subscribe_other(stack: AdoptionStack) -> None:
    command = CreateSubscription(
        subscriber_agent_id=OTHER_AGENT_ID,
        topic_id=TOPIC_ID,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.create_subscription(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="subscribe-other-for-query",
            operation_scope="subscription.create",
            route_template="/v1/agents/{agent_id}/subscriptions",
            agent_id=OTHER_AGENT_ID,
            body={"topic_id": command.topic_id},
        ),
        handler,
    )
    assert result.status_code == 201


async def _create_experience(
    stack: AdoptionStack,
    *,
    key: str,
    content: VersionContent,
) -> tuple[UUID, UUID]:
    command = CreateExperience(
        owner_agent_id=PUBLISHER_ID,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        importance=0.70,
        confidence=0.80,
        links=(),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.experience_service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key=key,
            operation_scope="experience.create",
            route_template="/v1/experiences",
            agent_id=PUBLISHER_ID,
            body={"summary": content.summary},
        ),
        handler,
    )
    assert result.status_code == 201
    data = json.loads(result.body)["data"]
    return UUID(data["experience_id"]), UUID(data["version_id"])


async def _publish(
    stack: AdoptionStack,
    *,
    key: str,
    experience_id: UUID = SOURCE_EXPERIENCE_ID,
    version_id: UUID = SOURCE_VERSION_ID,
    expires_in: timedelta = timedelta(days=7),
) -> Capsule:
    command = PublishCapsule(
        owner_agent_id=PUBLISHER_ID,
        topic_id=TOPIC_ID,
        experience_id=experience_id,
        version_id=version_id,
        expires_at=stack.clock.now() + expires_in,
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
            route_template="/v1/agents/{agent_id}/capsules",
            agent_id=PUBLISHER_ID,
            body={
                "topic_id": command.topic_id,
                "experience_id": experience_id,
                "version_id": version_id,
                "expires_at": command.expires_at,
            },
        ),
        handler,
    )
    assert result.status_code == 201
    return Capsule.model_validate(json.loads(result.body)["data"], strict=False)


async def _retract(
    stack: AdoptionStack,
    *,
    key: str,
    capsule_id: UUID,
) -> CommandResult:
    command = RetractCapsule(
        publisher_agent_id=PUBLISHER_ID,
        capsule_id=capsule_id,
        reason="The transported procedure was superseded.",
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.retract_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.retract",
            route_template=(
                "/v1/agents/{agent_id}/capsules/{capsule_id}:retract"
            ),
            agent_id=PUBLISHER_ID,
            path_parameters={
                "agent_id": PUBLISHER_ID,
                "capsule_id": capsule_id,
            },
            body={"reason": command.reason},
        ),
        handler,
    )


async def _arrange_repeated_capsules(
    stack: AdoptionStack,
    *,
    expiries: Sequence[timedelta],
    include_other: bool,
) -> tuple[Capsule, ...]:
    await create_source_experience(stack)
    await create_topic(stack)
    await subscribe_adopter(stack)
    if include_other:
        await _subscribe_other(stack)
    return tuple(
        [
            await _publish(
                stack,
                key=f"publish-query-{index}",
                expires_in=expires_in,
            )
            for index, expires_in in enumerate(expiries)
        ]
    )


async def _owned_items(
    stack: AdoptionStack,
    *,
    owner_agent_id: UUID,
) -> dict[UUID, UUID]:
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(InboxItemRow).where(
                        InboxItemRow.recipient_agent_id == owner_agent_id
                    )
                )
            ).all()
        )
    return {row.capsule_id: row.item_id for row in rows}


def _content(
    *,
    body: str,
    summary: str,
    mechanism: str,
) -> VersionContent:
    return VersionContent(
        body=body,
        summary=summary,
        mechanism=mechanism,
        tags=("query-evidence",),
        applicability=("explicit inspiration opt-in",),
        evidence=(),
        falsifiers=(),
    )


def _repository(stack: AdoptionStack) -> SharingRepository:
    return SharingRepository(event_registry=stack.registry)


@pytest.mark.asyncio
async def test_list_inbox_is_owner_scoped_state_filtered_and_cursor_safe(
    stack: AdoptionStack,
) -> None:
    capsules = await _arrange_repeated_capsules(
        stack,
        expiries=(timedelta(days=7),) * 3,
        include_other=True,
    )
    adopter_items = await _owned_items(stack, owner_agent_id=ADOPTER_ID)
    ordered_capsules = sorted(
        capsules,
        key=lambda value: adopter_items[value.capsule_id],
    )
    adopted_capsule = ordered_capsules[1]
    adopted_item_id = adopter_items[adopted_capsule.capsule_id]
    assert (
        await adopt(
            stack,
            key="adopt-middle-query-item",
            item_id=adopted_item_id,
        )
    ).status_code == 200

    query = SharingQuery(repository=_repository(stack))
    async with stack.database.read_session() as session:
        first_page = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=None,
            cursor=None,
            limit=1,
            at=stack.clock.now(),
        )
        second_page = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=None,
            cursor=first_page.next_cursor,
            limit=2,
            at=stack.clock.now(),
        )
        pending = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=InboxState.PENDING,
            cursor=None,
            limit=100,
            at=stack.clock.now(),
        )
        adopted = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=InboxState.ADOPTED,
            cursor=None,
            limit=100,
            at=stack.clock.now(),
        )
        other_owner = await query.list_inbox(
            session=session,
            owner_agent_id=OTHER_AGENT_ID,
            state=None,
            cursor=None,
            limit=100,
            at=stack.clock.now(),
        )
        missing_owner = await query.list_inbox(
            session=session,
            owner_agent_id=UUID(int=9_999),
            state=None,
            cursor=None,
            limit=100,
            at=stack.clock.now(),
        )

    assert first_page.next_cursor is not None
    assert second_page.next_cursor is None
    combined = (*first_page.items, *second_page.items)
    expected_adopter_ids = tuple(sorted(adopter_items.values()))
    assert tuple(item.item_id for item in combined) == expected_adopter_ids
    assert all(item.recipient_agent_id == ADOPTER_ID for item in combined)
    assert {item.item_id for item in pending.items} == (
        set(expected_adopter_ids) - {adopted_item_id}
    )
    assert tuple(item.item_id for item in adopted.items) == (adopted_item_id,)
    assert all(
        item.recipient_agent_id == OTHER_AGENT_ID for item in other_owner.items
    )
    assert {item.capsule_id for item in other_owner.items} == {
        capsule.capsule_id for capsule in capsules
    }
    assert missing_owner.items == ()
    assert missing_owner.next_cursor is None

    # Raw quarantined content is deliberately exposed by this owner-scoped
    # query, while the body remains absent from every transport event.
    by_capsule = {item.capsule_id: item for item in combined}
    assert all(
        by_capsule[capsule.capsule_id].capsule.body == SOURCE_CONTENT.body
        for capsule in capsules
    )
    async with stack.database.read_session() as session:
        transport_payloads = tuple(
            (
                await session.scalars(
                    select(DomainEventRow.payload).where(
                        DomainEventRow.event_type.like("capsule.%")
                    )
                )
            ).all()
        )
    assert all(b'"body"' not in payload for payload in transport_payloads)

    async with stack.database.read_session() as session:
        with pytest.raises(ValueError, match="cursor"):
            await query.list_inbox(
                session=session,
                owner_agent_id=OTHER_AGENT_ID,
                state=None,
                cursor=first_page.next_cursor,
                limit=100,
                at=stack.clock.now(),
            )
        with pytest.raises(ValueError, match="cursor"):
            await query.list_inbox(
                session=session,
                owner_agent_id=ADOPTER_ID,
                state=InboxState.PENDING,
                cursor=first_page.next_cursor,
                limit=100,
                at=stack.clock.now(),
            )
        with pytest.raises(ValueError, match="limit"):
            await query.list_inbox(
                session=session,
                owner_agent_id=ADOPTER_ID,
                state=None,
                cursor=None,
                limit=0,
                at=stack.clock.now(),
            )
        with pytest.raises(ValueError, match="limit"):
            await query.list_inbox(
                session=session,
                owner_agent_id=ADOPTER_ID,
                state=None,
                cursor=None,
                limit=101,
                at=stack.clock.now(),
            )


@pytest.mark.asyncio
async def test_list_inbox_computes_expiry_at_the_exact_boundary_without_deletion(
    stack: AdoptionStack,
) -> None:
    capsules = await _arrange_repeated_capsules(
        stack,
        expiries=(timedelta(days=7), timedelta(days=7)),
        include_other=False,
    )
    retracted_capsule, expiring_capsule = capsules
    assert (
        await _retract(
            stack,
            key="retract-for-list-query",
            capsule_id=retracted_capsule.capsule_id,
        )
    ).status_code == 200
    query = SharingQuery(repository=_repository(stack))

    async with stack.database.read_session() as session:
        source_count_before = await session.scalar(
            select(func.count()).select_from(ExperienceCapsuleRow)
        )
        inbox_count_before = await session.scalar(
            select(func.count()).select_from(InboxItemRow)
        )
        just_before = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=None,
            cursor=None,
            limit=100,
            at=expiring_capsule.expires_at - timedelta(microseconds=1),
        )
        at_boundary = await query.list_inbox(
            session=session,
            owner_agent_id=ADOPTER_ID,
            state=None,
            cursor=None,
            limit=100,
            at=expiring_capsule.expires_at,
        )

    before_by_capsule = {
        item.capsule_id: item for item in just_before.items
    }
    boundary_by_capsule = {
        item.capsule_id: item for item in at_boundary.items
    }
    assert (
        before_by_capsule[retracted_capsule.capsule_id].effective_availability
        is EffectiveAvailability.RETRACTED
    )
    assert (
        boundary_by_capsule[retracted_capsule.capsule_id].effective_availability
        is EffectiveAvailability.RETRACTED
    )
    assert (
        before_by_capsule[expiring_capsule.capsule_id].effective_availability
        is EffectiveAvailability.ACTIVE
    )
    assert (
        boundary_by_capsule[expiring_capsule.capsule_id].effective_availability
        is EffectiveAvailability.EXPIRED
    )
    assert all(
        item.capsule.body == SOURCE_CONTENT.body
        for item in at_boundary.items
    )

    async with stack.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count()).select_from(ExperienceCapsuleRow)
            )
            == source_count_before
        )
        assert (
            await session.scalar(select(func.count()).select_from(InboxItemRow))
            == inbox_count_before
        )


@pytest.mark.asyncio
async def test_pending_evidence_filters_state_status_and_expiry_without_writes(
    stack: AdoptionStack,
) -> None:
    capsules = await _arrange_repeated_capsules(
        stack,
        expiries=(
            timedelta(days=7),
            timedelta(days=7),
            timedelta(days=7),
            timedelta(days=1),
        ),
        include_other=False,
    )
    available, adopted_capsule, retracted_capsule, expired_capsule = capsules
    items = await _owned_items(stack, owner_agent_id=ADOPTER_ID)
    assert (
        await adopt(
            stack,
            key="adopt-before-evidence-query",
            item_id=items[adopted_capsule.capsule_id],
        )
    ).status_code == 200
    assert (
        await _retract(
            stack,
            key="retract-before-evidence-query",
            capsule_id=retracted_capsule.capsule_id,
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        projection_event_id = await uow.session.scalar(
            select(func.min(DomainEventRow.event_id))
        )
        assert projection_event_id is not None
        uow.session.add(
            AgentReputationRow(
                subject_agent_id=PUBLISHER_ID,
                observer_agent_id=ADOPTER_ID,
                useful_count=1,
                refuted_count=0,
                harmful_count=0,
                alpha=3,
                beta=2,
                projection_event_id=projection_event_id,
            )
        )

    reader = InboxEvidenceReader(repository=_repository(stack))
    as_of = expired_capsule.expires_at
    cues = query_cues(SOURCE_CONTENT.summary)
    async with stack.database.read_session() as session:
        before = (
            await session.scalar(select(func.count()).select_from(DomainEventRow)),
            await session.scalar(
                select(func.count()).select_from(ExperienceTermRow)
            ),
            await session.scalar(select(func.count()).select_from(InboxItemRow)),
            await session.scalar(
                select(func.count()).select_from(CapsuleStateRow)
            ),
        )
        evidence = await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=as_of,
            query_cues=cues,
            mode=RetrievalMode.FOCUSED,
            limit=12,
        )
        after = (
            await session.scalar(select(func.count()).select_from(DomainEventRow)),
            await session.scalar(
                select(func.count()).select_from(ExperienceTermRow)
            ),
            await session.scalar(select(func.count()).select_from(InboxItemRow)),
            await session.scalar(
                select(func.count()).select_from(CapsuleStateRow)
            ),
        )

    assert tuple(item.item_id for item in evidence) == (
        items[available.capsule_id],
    )
    assert evidence[0].capsule_id == available.capsule_id
    assert evidence[0].excerpt == SOURCE_CONTENT.body
    assert evidence[0].falsifiers == SOURCE_CONTENT.falsifiers
    assert evidence[0].source_trust == pytest.approx(0.25)
    assert before == after
    assert items[adopted_capsule.capsule_id] not in {
        item.item_id for item in evidence
    }
    assert items[retracted_capsule.capsule_id] not in {
        item.item_id for item in evidence
    }
    assert items[expired_capsule.capsule_id] not in {
        item.item_id for item in evidence
    }

    # An uncommitted change is visible because the reader uses exactly the
    # supplied session. Rolling back the savepoint leaves persistent state
    # untouched.
    async with stack.database.transaction() as uow:
        savepoint = await uow.session.begin_nested()
        await uow.session.execute(
            delete(InboxItemRow).where(
                InboxItemRow.item_id == items[available.capsule_id]
            )
        )
        assert (
            await reader.list_available_pending(
                session=uow.session,
                recipient_agent_id=ADOPTER_ID,
                as_of=as_of,
                query_cues=cues,
                mode=RetrievalMode.FOCUSED,
                limit=12,
            )
            == ()
        )
        await savepoint.rollback()

    async with stack.database.read_session() as session:
        assert (
            await session.get(InboxItemRow, items[available.capsule_id])
            is not None
        )


@pytest.mark.asyncio
async def test_pending_evidence_refuses_rolled_back_retracted_capsule_projection(
    stack: AdoptionStack,
) -> None:
    capsule = (
        await _arrange_repeated_capsules(
            stack,
            expiries=(timedelta(days=7),),
            include_other=False,
        )
    )[0]
    assert (
        await _retract(
            stack,
            key="retract-before-capsule-projection-rollback",
            capsule_id=capsule.capsule_id,
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        publication_event_id = await uow.session.scalar(
            select(DomainEventRow.event_id).where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == capsule.capsule_id,
                DomainEventRow.event_type == "capsule.published",
            )
        )
        state = await uow.session.get(CapsuleStateRow, capsule.capsule_id)
        assert publication_event_id is not None
        assert state is not None
        state.status = CapsuleStatus.ACTIVE
        state.projection_event_id = publication_event_id

    reader = InboxEvidenceReader(repository=_repository(stack))
    async with stack.database.read_session() as session:
        with pytest.raises(
            SourceIntegrityError,
            match="state event head",
        ):
            await reader.list_available_pending(
                session=session,
                recipient_agent_id=ADOPTER_ID,
                as_of=stack.clock.now(),
                query_cues=query_cues(SOURCE_CONTENT.summary),
                mode=RetrievalMode.FOCUSED,
                limit=12,
            )


@pytest.mark.asyncio
async def test_pending_evidence_refuses_rolled_back_adopted_inbox_projection(
    stack: AdoptionStack,
) -> None:
    capsule = (
        await _arrange_repeated_capsules(
            stack,
            expiries=(timedelta(days=7),),
            include_other=False,
        )
    )[0]
    item_id = (await _owned_items(stack, owner_agent_id=ADOPTER_ID))[
        capsule.capsule_id
    ]
    assert (
        await adopt(
            stack,
            key="adopt-before-inbox-projection-rollback",
            item_id=item_id,
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        received_event_id = await uow.session.scalar(
            select(DomainEventRow.event_id).where(
                DomainEventRow.aggregate_type == "inbox_item",
                DomainEventRow.aggregate_id == item_id,
                DomainEventRow.event_type == "capsule.received",
            )
        )
        item = await uow.session.get(InboxItemRow, item_id)
        assert received_event_id is not None
        assert item is not None
        item.state = InboxState.PENDING
        item.projection_event_id = received_event_id

    reader = InboxEvidenceReader(repository=_repository(stack))
    async with stack.database.read_session() as session:
        with pytest.raises(
            SourceIntegrityError,
            match="state event head",
        ):
            await reader.list_available_pending(
                session=session,
                recipient_agent_id=ADOPTER_ID,
                as_of=stack.clock.now(),
                query_cues=query_cues(SOURCE_CONTENT.summary),
                mode=RetrievalMode.FOCUSED,
                limit=12,
            )


@pytest.mark.asyncio
async def test_pending_evidence_refuses_rolled_back_rejected_inbox_projection(
    stack: AdoptionStack,
) -> None:
    capsule = (
        await _arrange_repeated_capsules(
            stack,
            expiries=(timedelta(days=7),),
            include_other=False,
        )
    )[0]
    item_id = (await _owned_items(stack, owner_agent_id=ADOPTER_ID))[
        capsule.capsule_id
    ]
    assert (
        await reject(
            stack,
            key="reject-before-inbox-projection-rollback",
            item_id=item_id,
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        received_event_id = await uow.session.scalar(
            select(DomainEventRow.event_id).where(
                DomainEventRow.aggregate_type == "inbox_item",
                DomainEventRow.aggregate_id == item_id,
                DomainEventRow.event_type == "capsule.received",
            )
        )
        item = await uow.session.get(InboxItemRow, item_id)
        assert received_event_id is not None
        assert item is not None
        item.state = InboxState.PENDING
        item.projection_event_id = received_event_id

    reader = InboxEvidenceReader(repository=_repository(stack))
    async with stack.database.read_session() as session:
        with pytest.raises(
            SourceIntegrityError,
            match="state event head",
        ):
            await reader.list_available_pending(
                session=session,
                recipient_agent_id=ADOPTER_ID,
                as_of=stack.clock.now(),
                query_cues=query_cues(SOURCE_CONTENT.summary),
                mode=RetrievalMode.FOCUSED,
                limit=12,
            )


@pytest.mark.asyncio
async def test_pending_evidence_ignores_capsule_feedback_for_state_head(
    stack: AdoptionStack,
) -> None:
    manager = cast(
        ProjectionManager,
        stack.database._projection_applier,  # noqa: SLF001
    )
    manager.registry.register(AgentReputationProjector(stack.registry))
    capsule = (
        await _arrange_repeated_capsules(
            stack,
            expiries=(timedelta(days=7),),
            include_other=True,
        )
    )[0]
    items = await _owned_items(stack, owner_agent_id=ADOPTER_ID)
    assert (
        await adopt(
            stack,
            key="adopt-before-feedback-state-head-query",
            item_id=items[capsule.capsule_id],
        )
    ).status_code == 200
    assert (
        await record_feedback(
            stack,
            key="feedback-before-state-head-query",
            capsule_id=capsule.capsule_id,
        )
    ).status_code == 201

    reader = InboxEvidenceReader(repository=_repository(stack))
    async with stack.database.read_session() as session:
        evidence = await reader.list_available_pending(
            session=session,
            recipient_agent_id=OTHER_AGENT_ID,
            as_of=stack.clock.now(),
            query_cues=query_cues(SOURCE_CONTENT.summary),
            mode=RetrievalMode.FOCUSED,
            limit=12,
        )

    assert tuple(item.capsule_id for item in evidence) == (
        capsule.capsule_id,
    )


@pytest.mark.asyncio
async def test_pending_evidence_uses_shared_focused_and_associative_ranking(
    stack: AdoptionStack,
) -> None:
    await create_topic(stack)
    await subscribe_adopter(stack)
    contents = (
        _content(
            body="alpha beta direct evidence",
            summary="alpha beta",
            mechanism="ordinary lookup",
        ),
        _content(
            body="alpha partial evidence",
            summary="alpha",
            mechanism="ordinary lookup",
        ),
        _content(
            body="distant evidence",
            summary="unrelated observation",
            mechanism="bridge",
        ),
    )
    capsules: list[Capsule] = []
    for index, content in enumerate(contents):
        experience_id, version_id = await _create_experience(
            stack,
            key=f"create-ranked-source-{index}",
            content=content,
        )
        capsules.append(
            await _publish(
                stack,
                key=f"publish-ranked-source-{index}",
                experience_id=experience_id,
                version_id=version_id,
            )
        )
    items = await _owned_items(stack, owner_agent_id=ADOPTER_ID)

    reader = InboxEvidenceReader(repository=_repository(stack))
    focused_cues = (
        TermCue(term="alpha", term_kind="word", weight=1.0),
        TermCue(term="beta", term_kind="word", weight=1.0),
    )
    mechanism_only = (
        TermCue(term="bridge", term_kind="mechanism", weight=1.25),
    )
    async with stack.database.read_session() as session:
        focused = await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=stack.clock.now(),
            query_cues=focused_cues,
            mode=RetrievalMode.FOCUSED,
            limit=12,
        )
        focused_mechanism = await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=stack.clock.now(),
            query_cues=mechanism_only,
            mode=RetrievalMode.FOCUSED,
            limit=12,
        )
        associative_mechanism = await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=stack.clock.now(),
            query_cues=mechanism_only,
            mode=RetrievalMode.ASSOCIATIVE,
            limit=12,
        )
        top_only = await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=stack.clock.now(),
            query_cues=focused_cues,
            mode=RetrievalMode.FOCUSED,
            limit=1,
        )

    assert tuple(item.item_id for item in focused) == (
        items[capsules[0].capsule_id],
        items[capsules[1].capsule_id],
    )
    assert focused[0].ranking_relevance > focused[1].ranking_relevance
    assert focused_mechanism == ()
    assert tuple(item.item_id for item in associative_mechanism) == (
        items[capsules[2].capsule_id],
    )
    assert associative_mechanism[0].ranking_relevance == pytest.approx(0.80)
    assert top_only == focused[:1]
    assert all(item.source_trust == pytest.approx(0.25) for item in focused)
