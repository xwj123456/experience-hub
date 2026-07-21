from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from tests.integration.test_capsule_corroboration import (
    ADOPTER,
    PUBLISHER_A,
    CorroborationStack,
    adopt,
    adoption_row,
    build_stack,
    create_owned_experience,
    create_topic,
    publish,
    subscribe,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.sharing.models import InboxState, ProvenanceHop
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.validation import (
    register_sharing_source_validator,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceLinkRow,
    InboxItemRow,
)
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_experience_source_validator,
)


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[CorroborationStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "adoption-provenance.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_new_adoption_persists_exact_full_chain_without_links(
    stack: CorroborationStack,
) -> None:
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="provenance-source",
        confidence=0.76,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="provenance-publish",
    )

    result = await adopt(
        stack,
        adopter_agent_id=ADOPTER,
        item_id=published.item_ids[ADOPTER],
        key="provenance-adopt",
    )
    assert result.status_code == 200
    resulting_experience_id = UUID(
        json.loads(result.body)["data"]["experience"]["experience_id"]
    )

    row = await adoption_row(
        stack,
        adopter_agent_id=ADOPTER,
        capsule_id=published.capsule_id,
    )
    expected_chain = (
        ProvenanceHop(
            capsule_id=published.capsule_id,
            publisher_agent_id=PUBLISHER_A,
        ),
    )
    async with stack.database.read_session() as session:
        capsule = await session.get(
            ExperienceCapsuleRow,
            published.capsule_id,
        )
        parent = await SharingRepository().get_owned_parent_adoption(
            session=session,
            adopter_agent_id=ADOPTER,
            adoption_id=row.adoption_id,
        )
        root_is_represented = await SharingRepository.root_is_represented(
            session=session,
            resulting_experience_id=resulting_experience_id,
            root_fingerprint=row.root_fingerprint,
        )
        link_count = await session.scalar(
            select(func.count())
            .select_from(ExperienceLinkRow)
            .where(ExperienceLinkRow.source_experience_id == resulting_experience_id)
        )

    assert capsule is not None
    assert row.adopter_agent_id == ADOPTER
    assert row.capsule_id == published.capsule_id
    assert row.resulting_experience_id == resulting_experience_id
    assert row.captured_trust == 0.50
    assert row.provenance_chain == canonical_json_bytes(expected_chain)
    assert row.root_fingerprint == capsule.root_fingerprint
    assert row.corroboration_applied is False
    assert parent is not None
    assert parent.adoption_id == row.adoption_id
    assert parent.provenance_chain == expected_chain
    assert parent.root_fingerprint == row.root_fingerprint
    assert parent.resulting_experience_id == resulting_experience_id
    assert root_is_represented
    assert link_count == 0


@pytest.mark.asyncio
async def test_partial_unique_claim_retains_one_corroboration_per_result_root(
    stack: CorroborationStack,
) -> None:
    local = await create_owned_experience(
        stack,
        owner_agent_id=ADOPTER,
        key="claim-local",
        confidence=0.40,
    )
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="claim-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    capsules = [
        await publish(
            stack,
            publisher_agent_id=PUBLISHER_A,
            topic_id=topic_id,
            experience=source,
            key=f"claim-publish-{index}",
        )
        for index in range(3)
    ]

    first = await adopt(
        stack,
        adopter_agent_id=ADOPTER,
        item_id=capsules[0].item_ids[ADOPTER],
        key="claim-adopt-first",
    )
    second = await adopt(
        stack,
        adopter_agent_id=ADOPTER,
        item_id=capsules[1].item_ids[ADOPTER],
        key="claim-adopt-second",
    )
    assert first.status_code == second.status_code == 200
    first_row = await adoption_row(
        stack,
        adopter_agent_id=ADOPTER,
        capsule_id=capsules[0].capsule_id,
    )
    second_row = await adoption_row(
        stack,
        adopter_agent_id=ADOPTER,
        capsule_id=capsules[1].capsule_id,
    )
    assert first_row.root_fingerprint == second_row.root_fingerprint
    assert (first_row.corroboration_applied, second_row.corroboration_applied) == (
        True,
        False,
    )

    conflicting = AdoptionRecordRow(
        adoption_id=UUID(int=99_999),
        adopter_agent_id=ADOPTER,
        capsule_id=capsules[2].capsule_id,
        resulting_experience_id=local.experience_id,
        captured_trust=0.50,
        provenance_chain=canonical_json_bytes(
            (
                ProvenanceHop(
                    capsule_id=capsules[2].capsule_id,
                    publisher_agent_id=PUBLISHER_A,
                ),
            )
        ),
        root_fingerprint=first_row.root_fingerprint,
        corroboration_applied=True,
        adopted_at=stack.clock.now(),
    )
    with pytest.raises(
        IntegrityError,
        match="adoption_records identity already exists",
    ):
        async with stack.database.transaction(immediate=True) as uow:
            uow.session.add(conflicting)
            await uow.session.flush()

    async with stack.database.read_session() as session:
        retained = tuple(
            (
                await session.scalars(
                    select(AdoptionRecordRow).where(
                        AdoptionRecordRow.resulting_experience_id
                        == local.experience_id,
                        AdoptionRecordRow.root_fingerprint
                        == first_row.root_fingerprint,
                        AdoptionRecordRow.corroboration_applied.is_(True),
                    )
                )
            ).all()
        )
    assert [row.adoption_id for row in retained] == [first_row.adoption_id]


@pytest.mark.asyncio
async def test_source_validation_rejects_adoption_without_terminal_event(
    stack: CorroborationStack,
) -> None:
    source = await create_owned_experience(
        stack,
        owner_agent_id=PUBLISHER_A,
        key="orphan-source",
        confidence=0.80,
    )
    topic_id = await create_topic(stack)
    await subscribe(stack, subscriber_agent_id=ADOPTER, topic_id=topic_id)
    published = await publish(
        stack,
        publisher_agent_id=PUBLISHER_A,
        topic_id=topic_id,
        experience=source,
        key="orphan-publish",
    )
    item_id = published.item_ids[ADOPTER]
    assert (
        await adopt(
            stack,
            adopter_agent_id=ADOPTER,
            item_id=item_id,
            key="orphan-adopt",
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        received = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == item_id,
                DomainEventRow.event_type == "capsule.received",
            )
        )
        adopted = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == item_id,
                DomainEventRow.event_type == "capsule.adopted",
            )
        )
        item = await uow.session.get(InboxItemRow, item_id)
        assert received is not None and adopted is not None and item is not None
        item.state = InboxState.PENDING
        item.projection_event_id = received.event_id
        await uow.session.execute(
            text("DROP TRIGGER domain_events_reject_delete")
        )
        await uow.session.execute(
            delete(DomainEventRow).where(
                DomainEventRow.event_id == adopted.event_id
            )
        )

    validator = SourceValidator(stack.registry)
    register_experience_source_validator(validator)
    register_sharing_source_validator(validator)
    async with stack.database.read_session() as session:
        with pytest.raises(
            SourceIntegrityError,
            match="exactly one capsule.adopted event",
        ):
            await validator.validate(session)
