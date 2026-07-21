from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    ADOPTION_ID,
    CAPSULE_ID,
    ITEM_ID,
    OTHER_AGENT_ID,
    PUBLISHER_ID,
    TOPIC_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
)
from tests.integration.test_capsule_rejection import reject

from experience_hub.sharing.models import InboxState
from experience_hub.sharing.repository import SharingRepository

MISSING_TOPIC_ID = UUID("00000000-0000-0000-0000-000000000981")
MISSING_CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000982")
MISSING_ITEM_ID = UUID("00000000-0000-0000-0000-000000000983")
MISSING_ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000984")


@pytest.fixture(name="stack")
async def _stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "sharing-authorization.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_direct_sharing_lookups_are_owner_scoped_and_fail_closed(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="authorization-adopt")).status_code == 200
    repository = SharingRepository(event_registry=stack.registry)

    async with stack.database.read_session() as session:
        owned_topic = await repository.get_owned_topic(
            session=session,
            owner_agent_id=PUBLISHER_ID,
            topic_id=TOPIC_ID,
        )
        foreign_topic = await repository.get_owned_topic(
            session=session,
            owner_agent_id=OTHER_AGENT_ID,
            topic_id=TOPIC_ID,
        )
        missing_topic = await repository.get_owned_topic(
            session=session,
            owner_agent_id=PUBLISHER_ID,
            topic_id=MISSING_TOPIC_ID,
        )

        owned_capsule = await repository.get_owned_capsule(
            session=session,
            publisher_agent_id=PUBLISHER_ID,
            capsule_id=CAPSULE_ID,
        )
        foreign_capsule = await repository.get_owned_capsule(
            session=session,
            publisher_agent_id=ADOPTER_ID,
            capsule_id=CAPSULE_ID,
        )
        missing_capsule = await repository.get_owned_capsule(
            session=session,
            publisher_agent_id=PUBLISHER_ID,
            capsule_id=MISSING_CAPSULE_ID,
        )

        owned_item = await repository.get_owned_inbox_item(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            item_id=ITEM_ID,
        )
        foreign_item = await repository.get_owned_inbox_item(
            session=session,
            recipient_agent_id=PUBLISHER_ID,
            item_id=ITEM_ID,
        )
        missing_item = await repository.get_owned_inbox_item(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            item_id=MISSING_ITEM_ID,
        )

        owned_adoption = await repository.get_owned_adoption(
            session=session,
            adopter_agent_id=ADOPTER_ID,
            adoption_id=ADOPTION_ID,
        )
        foreign_adoption = await repository.get_owned_adoption(
            session=session,
            adopter_agent_id=PUBLISHER_ID,
            adoption_id=ADOPTION_ID,
        )
        missing_adoption = await repository.get_owned_adoption(
            session=session,
            adopter_agent_id=ADOPTER_ID,
            adoption_id=MISSING_ADOPTION_ID,
        )

    assert owned_topic is not None and owned_topic.topic_id == TOPIC_ID
    assert foreign_topic is missing_topic is None
    assert owned_capsule is not None and owned_capsule.capsule_id == CAPSULE_ID
    assert foreign_capsule is missing_capsule is None
    assert owned_item is not None and owned_item.item_id == ITEM_ID
    assert owned_item.state is InboxState.ADOPTED
    assert foreign_item is missing_item is None
    assert owned_adoption is not None and owned_adoption.adoption_id == ADOPTION_ID
    assert foreign_adoption is missing_adoption is None


@pytest.mark.asyncio
async def test_pending_recipient_and_cross_agent_caller_cannot_address_feedback(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    repository = SharingRepository(event_registry=stack.registry)

    async with stack.database.read_session() as session:
        pending = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=ADOPTER_ID,
            capsule_id=CAPSULE_ID,
        )
        publisher = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=PUBLISHER_ID,
            capsule_id=CAPSULE_ID,
        )
        unrelated = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=OTHER_AGENT_ID,
            capsule_id=CAPSULE_ID,
        )
        missing = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=ADOPTER_ID,
            capsule_id=MISSING_CAPSULE_ID,
        )

    assert pending is publisher is unrelated is missing is None


@pytest.mark.asyncio
async def test_adopted_inbox_owner_can_address_only_its_feedback_capsule(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="feedback-after-adopt")).status_code == 200
    repository = SharingRepository(event_registry=stack.registry)

    async with stack.database.read_session() as session:
        authorized = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=ADOPTER_ID,
            capsule_id=CAPSULE_ID,
        )
        unrelated = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=OTHER_AGENT_ID,
            capsule_id=CAPSULE_ID,
        )

    assert authorized is not None
    assert authorized.capsule_id == CAPSULE_ID
    assert unrelated is None


@pytest.mark.asyncio
async def test_rejected_inbox_owner_can_address_only_its_feedback_capsule(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="feedback-after-reject")).status_code == 200
    repository = SharingRepository(event_registry=stack.registry)

    async with stack.database.read_session() as session:
        authorized = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=ADOPTER_ID,
            capsule_id=CAPSULE_ID,
        )
        publisher = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=PUBLISHER_ID,
            capsule_id=CAPSULE_ID,
        )
        unrelated = await repository.get_feedback_authorized_capsule(
            session=session,
            observer_agent_id=OTHER_AGENT_ID,
            capsule_id=CAPSULE_ID,
        )

    assert authorized is not None
    assert authorized.capsule_id == CAPSULE_ID
    assert publisher is unrelated is None
