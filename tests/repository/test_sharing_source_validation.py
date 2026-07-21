from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from tests.integration.test_capsule_adoption import (
    ADOPTED_EXPERIENCE_ID,
    ADOPTED_VERSION_ID,
    ADOPTER_ID,
    ADOPTION_ID,
    CAPSULE_ID,
    ITEM_ID,
    OTHER_AGENT_ID,
    PUBLISHER_ID,
    SOURCE_EXPERIENCE_ID,
    TOPIC_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
    create_source_experience,
    create_topic,
    request,
    subscribe_adopter,
)
from tests.integration.test_capsule_adoption import (
    publish_capsule as publish_original_capsule,
)
from tests.integration.test_capsule_corroboration import (
    ADOPTER as MULTIHOP_ADOPTER,
)
from tests.integration.test_capsule_corroboration import (
    PUBLISHER_A as MULTIHOP_PUBLISHER,
)
from tests.integration.test_capsule_corroboration import (
    RELAY as MULTIHOP_RELAY,
)
from tests.integration.test_capsule_corroboration import (
    OwnedExperience as MultihopOwnedExperience,
)
from tests.integration.test_capsule_corroboration import (
    adopt as adopt_multihop,
)
from tests.integration.test_capsule_corroboration import (
    adoption_row as multihop_adoption_row,
)
from tests.integration.test_capsule_corroboration import (
    build_stack as build_multihop_stack,
)
from tests.integration.test_capsule_corroboration import (
    create_owned_experience as create_multihop_experience,
)
from tests.integration.test_capsule_corroboration import (
    create_topic as create_multihop_topic,
)
from tests.integration.test_capsule_corroboration import (
    publish as publish_multihop,
)
from tests.integration.test_capsule_corroboration import (
    subscribe as subscribe_multihop,
)
from tests.integration.test_capsule_feedback import publish_again, record_feedback
from tests.integration.test_capsule_rejection import reject
from tests.integration.test_capsule_retraction import retract

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandContext
from experience_hub.experiences.contracts import ConfirmExperience
from experience_hub.sharing.events import (
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    SubscriptionCreatedV1,
    TopicCreatedV1,
)
from experience_hub.sharing.hashing import (
    compute_capsule_hash,
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import (
    Capsule,
    CreateSubscription,
    FeedbackRevision,
    FeedbackVerdict,
    InboxState,
    ProvenanceHop,
    PublishCapsule,
)
from experience_hub.sharing.projector import AgentReputationProjector
from experience_hub.sharing.validation import (
    SharingSourceValidator,
    register_sharing_source_validator,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    CapsuleFeedbackRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    IdempotencyRecordRow,
    InboxItemRow,
    SubscriptionRow,
    TopicRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_experience_source_validator,
)

LATE_ITEM_ID = UUID("00000000-0000-0000-0000-00000000f701")
ORPHAN_SOURCE_ID = UUID("00000000-0000-0000-0000-00000000f702")
BAD_HASH = "f" * 64
BAD_ROOT = "0" * 64


@dataclass(frozen=True, slots=True)
class SharingHistory:
    original: Capsule
    relay: Capsule
    other_subscription_id: UUID
    relay_item_id: UUID
    feedback_ids: tuple[UUID, UUID]


def manager(stack: AdoptionStack) -> ProjectionManager:
    value = cast(Any, stack.database)._projection_applier
    assert isinstance(value, ProjectionManager)
    return value


def source_validator(stack: AdoptionStack) -> SourceValidator:
    validator = SourceValidator(stack.registry)
    register_experience_source_validator(validator)
    register_sharing_source_validator(validator)
    return validator


async def subscribe_other(stack: AdoptionStack) -> UUID:
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
            key="subscribe-other-after-original-publication",
            operation_scope="subscription.create",
            route_template="/v1/agents/{agent_id}/subscriptions",
            agent_id=OTHER_AGENT_ID,
            body={"topic_id": TOPIC_ID},
        ),
        handler,
    )
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["subscription_id"])


async def publish_relay(
    stack: AdoptionStack,
) -> tuple[Capsule, UUID]:
    command = PublishCapsule(
        owner_agent_id=ADOPTER_ID,
        topic_id=TOPIC_ID,
        experience_id=ADOPTED_EXPERIENCE_ID,
        version_id=ADOPTED_VERSION_ID,
        expires_at=stack.clock.now() + timedelta(days=7),
        parent_adoption_id=ADOPTION_ID,
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
            key="publish-relay-capsule",
            operation_scope="capsule.publish",
            route_template="/v1/capsules",
            agent_id=ADOPTER_ID,
            body={
                "topic_id": TOPIC_ID,
                "experience_id": ADOPTED_EXPERIENCE_ID,
                "version_id": ADOPTED_VERSION_ID,
                "expires_at": command.expires_at,
                "parent_adoption_id": ADOPTION_ID,
            },
        ),
        handler,
    )
    assert result.status_code == 201
    capsule = Capsule.model_validate(json.loads(result.body)["data"], strict=False)
    async with stack.database.read_session() as session:
        item_id = await session.scalar(
            select(InboxItemRow.item_id).where(
                InboxItemRow.recipient_agent_id == OTHER_AGENT_ID,
                InboxItemRow.capsule_id == capsule.capsule_id,
            )
        )
    assert item_id is not None
    return capsule, item_id


async def arrange_complete_history(
    stack: AdoptionStack,
) -> SharingHistory:
    original = await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-before-source-validation")).status_code == 200
    other_subscription_id = await subscribe_other(stack)

    first_result = await record_feedback(
        stack,
        key="source-validation-feedback-1",
        verdict=FeedbackVerdict.USEFUL,
    )
    assert first_result.status_code == 201
    first = FeedbackRevision.model_validate(
        json.loads(first_result.body)["data"],
        strict=False,
    )
    stack.clock.advance(timedelta(seconds=1))
    second_result = await record_feedback(
        stack,
        key="source-validation-feedback-2",
        verdict=FeedbackVerdict.REFUTED,
    )
    assert second_result.status_code == 201
    second = FeedbackRevision.model_validate(
        json.loads(second_result.body)["data"],
        strict=False,
    )
    relay, relay_item_id = await publish_relay(stack)
    return SharingHistory(
        original=original,
        relay=relay,
        other_subscription_id=other_subscription_id,
        relay_item_id=relay_item_id,
        feedback_ids=(first.feedback_id, second.feedback_id),
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "sharing-source-validation.sqlite3",
    )
    manager(value).registry.register(AgentReputationProjector(value.registry))
    try:
        yield value
    finally:
        await value.database.dispose()


async def assert_stable_source_error(
    stack: AdoptionStack,
    *,
    mismatch_prefix: str,
) -> None:
    errors: list[SourceIntegrityError] = []
    validator = source_validator(stack)
    for _ in range(2):
        with pytest.raises(SourceIntegrityError) as caught:
            async with stack.database.read_session() as session:
                await validator.validate(session)
        errors.append(caught.value)
    assert errors[0].mismatch_key.startswith(mismatch_prefix)
    assert (
        errors[1].mismatch_key,
        str(errors[1]),
    ) == (
        errors[0].mismatch_key,
        str(errors[0]),
    )


async def rewrite_event_identity(
    stack: AdoptionStack,
    *,
    event_type: str,
    identity_field: str,
    identity: UUID,
) -> None:
    async with stack.database.transaction() as uow:
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == event_type)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        selected: DomainEventRow | None = None
        selected_payload: Any = None
        for row in rows:
            payload = stack.registry.decode(
                event_type=row.event_type,
                payload=row.payload,
            )
            if getattr(payload, identity_field) == identity:
                selected = row
                selected_payload = payload
                break
        assert selected is not None
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == selected.event_id)
            .values(
                payload=canonical_json_bytes(
                    selected_payload.model_copy(
                        update={identity_field: ORPHAN_SOURCE_ID}
                    )
                )
            )
        )


def transport_hash(
    capsule: Capsule,
    **changes: Any,
) -> str:
    values: dict[str, Any] = {
        "transport_schema_version": capsule.transport_schema_version,
        "capsule_id": capsule.capsule_id,
        "topic_id": capsule.topic_id,
        "source_experience_id": capsule.source_experience_id,
        "source_version_id": capsule.source_version_id,
        "publisher_agent_id": capsule.publisher_agent_id,
        "kind": capsule.kind,
        "body": capsule.body,
        "summary": capsule.summary,
        "mechanism": capsule.mechanism,
        "tags": capsule.tags,
        "applicability": capsule.applicability,
        "evidence": capsule.evidence,
        "falsifiers": capsule.falsifiers,
        "publisher_confidence": capsule.publisher_confidence,
        "provenance_chain": capsule.provenance_chain,
        "root_fingerprint": capsule.root_fingerprint,
        "source_content_hash": capsule.source_content_hash,
        "created_at": capsule.created_at,
        "expires_at": capsule.expires_at,
        "hop_count": capsule.hop_count,
    }
    values.update(changes)
    return compute_capsule_hash(**values)


async def rewrite_capsule_semantics(
    stack: AdoptionStack,
    *,
    capsule: Capsule,
    changes: dict[str, Any],
) -> None:
    capsule_hash = transport_hash(capsule, **changes)
    source_values = dict(changes)
    if "provenance_chain" in source_values:
        source_values["provenance_chain"] = canonical_json_bytes(
            source_values["provenance_chain"]
        )
    source_values["capsule_hash"] = capsule_hash
    async with stack.database.transaction() as uow:
        publication = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == capsule.capsule_id,
                DomainEventRow.event_type == CapsulePublishedV1.event_type,
            )
        )
        assert publication is not None
        payload = stack.registry.decode(
            event_type=publication.event_type,
            payload=publication.payload,
        )
        assert isinstance(payload, CapsulePublishedV1)
        event_changes: dict[str, Any] = {"capsule_hash": capsule_hash}
        if "root_fingerprint" in changes:
            event_changes["root_fingerprint"] = changes["root_fingerprint"]
        await uow.session.execute(
            text("DROP TRIGGER experience_capsules_reject_update")
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        if "hop_count" in changes:
            await uow.session.execute(text("PRAGMA ignore_check_constraints = ON"))
        await uow.session.execute(
            update(ExperienceCapsuleRow)
            .where(ExperienceCapsuleRow.capsule_id == capsule.capsule_id)
            .values(**source_values)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == publication.event_id)
            .values(
                payload=canonical_json_bytes(payload.model_copy(update=event_changes))
            )
        )
        if "hop_count" in changes:
            await uow.session.execute(text("PRAGMA ignore_check_constraints = OFF"))


@pytest.mark.parametrize("corruption", ("source_version_hash", "transport_hash"))
@pytest.mark.asyncio
async def test_capsule_source_version_and_transport_hash_are_recomputed(
    stack: AdoptionStack,
    corruption: str,
) -> None:
    history = await arrange_complete_history(stack)
    if corruption == "source_version_hash":
        root = compute_original_root_fingerprint(
            root_publisher_id=PUBLISHER_ID,
            source_content_hash=BAD_HASH,
        )
        await rewrite_capsule_semantics(
            stack,
            capsule=history.original,
            changes={
                "source_content_hash": BAD_HASH,
                "root_fingerprint": root,
            },
        )
    else:
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text("DROP TRIGGER experience_capsules_reject_update")
            )
            await uow.session.execute(
                update(ExperienceCapsuleRow)
                .where(ExperienceCapsuleRow.capsule_id == CAPSULE_ID)
                .values(capsule_hash=BAD_HASH)
            )

    await assert_stable_source_error(stack, mismatch_prefix=f"capsule:{CAPSULE_ID}")


@pytest.mark.parametrize("corruption", ("chain", "hop_count", "root"))
@pytest.mark.asyncio
async def test_relay_provenance_chain_hop_and_root_are_continuous(
    stack: AdoptionStack,
    corruption: str,
) -> None:
    history = await arrange_complete_history(stack)
    changes: dict[str, Any]
    if corruption == "chain":
        changes = {
            "provenance_chain": (
                ProvenanceHop(
                    capsule_id=history.relay.capsule_id,
                    publisher_agent_id=ADOPTER_ID,
                ),
            )
        }
    elif corruption == "hop_count":
        changes = {"hop_count": 0}
    else:
        changes = {"root_fingerprint": BAD_ROOT}
    await rewrite_capsule_semantics(
        stack,
        capsule=history.relay,
        changes=changes,
    )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{history.relay.capsule_id}",
    )


@pytest.mark.asyncio
async def test_multihop_provenance_requires_every_exact_root_first_prefix(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_multihop_stack(
        repository_root=repository_root,
        database_path=tmp_path / "multihop-prefix-source-validation.sqlite3",
    )
    typed_stack = cast(AdoptionStack, cast(Any, stack))
    try:
        source = await create_multihop_experience(
            stack,
            owner_agent_id=MULTIHOP_PUBLISHER,
            key="multihop-prefix-source",
            confidence=0.80,
        )
        topic_id = await create_multihop_topic(stack)
        await subscribe_multihop(
            stack,
            subscriber_agent_id=MULTIHOP_RELAY,
            topic_id=topic_id,
        )
        await subscribe_multihop(
            stack,
            subscriber_agent_id=MULTIHOP_ADOPTER,
            topic_id=topic_id,
        )

        original = await publish_multihop(
            stack,
            publisher_agent_id=MULTIHOP_PUBLISHER,
            topic_id=topic_id,
            experience=source,
            key="multihop-prefix-original",
        )
        alternate_root = await publish_multihop(
            stack,
            publisher_agent_id=MULTIHOP_PUBLISHER,
            topic_id=topic_id,
            experience=source,
            key="multihop-prefix-alternate-root",
        )
        relay_result = await adopt_multihop(
            stack,
            adopter_agent_id=MULTIHOP_RELAY,
            item_id=original.item_ids[MULTIHOP_RELAY],
            key="multihop-prefix-relay-adopts",
        )
        assert relay_result.status_code == 200
        relay_view = cast(
            dict[str, Any],
            json.loads(relay_result.body)["data"]["experience"],
        )
        relay_experience = MultihopOwnedExperience(
            experience_id=UUID(relay_view["experience_id"]),
            version_id=UUID(relay_view["current_version_id"]),
            content_hash=cast(str, relay_view["current_content_hash"]),
        )
        relay_adoption = await multihop_adoption_row(
            stack,
            adopter_agent_id=MULTIHOP_RELAY,
            capsule_id=original.capsule_id,
        )

        first_relay = await publish_multihop(
            stack,
            publisher_agent_id=MULTIHOP_RELAY,
            topic_id=topic_id,
            experience=relay_experience,
            key="multihop-prefix-first-relay",
            parent_adoption_id=relay_adoption.adoption_id,
        )
        adopter_result = await adopt_multihop(
            stack,
            adopter_agent_id=MULTIHOP_ADOPTER,
            item_id=first_relay.item_ids[MULTIHOP_ADOPTER],
            key="multihop-prefix-adopter-adopts",
        )
        assert adopter_result.status_code == 200
        adopter_view = cast(
            dict[str, Any],
            json.loads(adopter_result.body)["data"]["experience"],
        )
        adopter_experience = MultihopOwnedExperience(
            experience_id=UUID(adopter_view["experience_id"]),
            version_id=UUID(adopter_view["current_version_id"]),
            content_hash=cast(str, adopter_view["current_content_hash"]),
        )
        adopter_adoption = await multihop_adoption_row(
            stack,
            adopter_agent_id=MULTIHOP_ADOPTER,
            capsule_id=first_relay.capsule_id,
        )
        second_relay = await publish_multihop(
            stack,
            publisher_agent_id=MULTIHOP_ADOPTER,
            topic_id=topic_id,
            experience=adopter_experience,
            key="multihop-prefix-second-relay",
            parent_adoption_id=adopter_adoption.adoption_id,
        )

        async with stack.database.read_session() as session:
            row = await session.get(ExperienceCapsuleRow, second_relay.capsule_id)
            await source_validator(typed_stack).validate(session)
        assert row is not None
        capsule = SharingSourceValidator(stack.registry)._capsule_from_row(row)
        assert len(capsule.provenance_chain) == 2
        await rewrite_capsule_semantics(
            typed_stack,
            capsule=capsule,
            changes={
                "provenance_chain": (
                    ProvenanceHop(
                        capsule_id=alternate_root.capsule_id,
                        publisher_agent_id=MULTIHOP_PUBLISHER,
                    ),
                    capsule.provenance_chain[1],
                )
            },
        )

        await assert_stable_source_error(
            typed_stack,
            mismatch_prefix=f"capsule:{second_relay.capsule_id}",
        )
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_delivery_requires_subscription_eligibility_at_publication_event(
    stack: AdoptionStack,
) -> None:
    history = await arrange_complete_history(stack)
    async with stack.database.transaction() as uow:
        publication = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == history.original.capsule_id,
                DomainEventRow.event_type == CapsulePublishedV1.event_type,
            )
        )
        subscription = await uow.session.get(
            SubscriptionRow,
            history.other_subscription_id,
        )
        assert publication is not None
        assert subscription is not None
        assert subscription.creation_event_id > publication.event_id
        received = DomainEventRow(
            aggregate_type="inbox_item",
            aggregate_id=LATE_ITEM_ID,
            sequence=1,
            event_type=CapsuleReceivedV1.event_type,
            payload=canonical_json_bytes(
                CapsuleReceivedV1(
                    schema_version=1,
                    item_id=LATE_ITEM_ID,
                    capsule_id=history.original.capsule_id,
                    recipient_agent_id=OTHER_AGENT_ID,
                    state_after=InboxState.PENDING,
                )
            ),
            actor_agent_id=PUBLISHER_ID,
            causation_id=publication.causation_id,
            occurred_at=publication.occurred_at,
        )
        uow.session.add(received)
        await uow.session.flush()
        uow.session.add(
            InboxItemRow(
                item_id=LATE_ITEM_ID,
                recipient_agent_id=OTHER_AGENT_ID,
                capsule_id=history.original.capsule_id,
                state=InboxState.PENDING,
                projection_event_id=received.event_id,
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"inbox_item:{LATE_ITEM_ID}",
    )


@pytest.mark.asyncio
async def test_every_eligible_subscription_requires_one_delivery_event(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    async with stack.database.transaction() as uow:
        received = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleReceivedV1.event_type,
                DomainEventRow.aggregate_id == ITEM_ID,
            )
        )
        assert received is not None
        await uow.session.execute(
            text("DELETE FROM inbox_items WHERE item_id = :item_id"),
            {"item_id": str(ITEM_ID)},
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_delete"))
        await uow.session.delete(received)

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{CAPSULE_ID}",
    )


@pytest.mark.asyncio
async def test_delivery_events_follow_deterministic_recipient_order(
    stack: AdoptionStack,
) -> None:
    await create_source_experience(stack)
    await create_topic(stack)
    await subscribe_adopter(stack)
    await subscribe_other(stack)
    capsule = await publish_original_capsule(stack)
    async with stack.database.transaction() as uow:
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type == CapsuleReceivedV1.event_type,
                        DomainEventRow.causation_id
                        == select(DomainEventRow.causation_id)
                        .where(
                            DomainEventRow.event_type == CapsulePublishedV1.event_type,
                            DomainEventRow.aggregate_id == capsule.capsule_id,
                        )
                        .scalar_subquery(),
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        assert len(rows) == 2
        decoded = tuple(
            cast(
                CapsuleReceivedV1,
                stack.registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                ),
            )
            for row in rows
        )
        assert tuple(item.recipient_agent_id for item in decoded) == (
            ADOPTER_ID,
            OTHER_AGENT_ID,
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_delete"))
        for item in decoded:
            await uow.session.execute(
                text("DELETE FROM inbox_items WHERE item_id = :item_id"),
                {"item_id": str(item.item_id)},
            )
        for row in rows:
            await uow.session.delete(row)
        await uow.session.flush()

        replacements: list[tuple[DomainEventRow, CapsuleReceivedV1]] = []
        for row, item in reversed(tuple(zip(rows, decoded, strict=True))):
            replacement = DomainEventRow(
                aggregate_type=row.aggregate_type,
                aggregate_id=row.aggregate_id,
                sequence=row.sequence,
                event_type=row.event_type,
                payload=row.payload,
                actor_agent_id=row.actor_agent_id,
                causation_id=row.causation_id,
                occurred_at=row.occurred_at,
            )
            uow.session.add(replacement)
            await uow.session.flush()
            replacements.append((replacement, item))
        for replacement, item in replacements:
            uow.session.add(
                InboxItemRow(
                    item_id=item.item_id,
                    recipient_agent_id=item.recipient_agent_id,
                    capsule_id=item.capsule_id,
                    state=InboxState.PENDING,
                    projection_event_id=replacement.event_id,
                )
            )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{capsule.capsule_id}",
    )


@pytest.mark.parametrize(
    ("entity", "event_type", "identity_field"),
    (
        ("topic", TopicCreatedV1.event_type, "topic_id"),
        (
            "subscription",
            SubscriptionCreatedV1.event_type,
            "subscription_id",
        ),
        ("capsule", CapsulePublishedV1.event_type, "capsule_id"),
        ("adoption", CapsuleAdoptedV1.event_type, "adoption_id"),
        (
            "feedback",
            CapsuleFeedbackRecordedV1.event_type,
            "feedback_id",
        ),
    ),
)
@pytest.mark.asyncio
async def test_every_authoritative_sharing_source_has_one_matching_event(
    stack: AdoptionStack,
    entity: str,
    event_type: str,
    identity_field: str,
) -> None:
    history = await arrange_complete_history(stack)
    identities = {
        "topic": TOPIC_ID,
        "subscription": history.other_subscription_id,
        "capsule": history.relay.capsule_id,
        "adoption": ADOPTION_ID,
        "feedback": history.feedback_ids[1],
    }
    await rewrite_event_identity(
        stack,
        event_type=event_type,
        identity_field=identity_field,
        identity=identities[entity],
    )

    mismatch_id = (
        ORPHAN_SOURCE_ID if entity == "adoption" else identities[entity]
    )
    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"{entity}:{mismatch_id}",
    )


@pytest.mark.parametrize(
    ("entity", "event_type", "identity_field", "aggregate_type"),
    (
        (
            "subscription",
            SubscriptionCreatedV1.event_type,
            "subscription_id",
            "subscription",
        ),
        ("capsule", CapsulePublishedV1.event_type, "capsule_id", "capsule"),
        ("adoption", CapsuleAdoptedV1.event_type, "adoption_id", "inbox_item"),
        (
            "feedback",
            CapsuleFeedbackRecordedV1.event_type,
            "feedback_id",
            "capsule",
        ),
    ),
)
@pytest.mark.asyncio
async def test_non_topic_sharing_event_without_source_is_rejected_independently(
    stack: AdoptionStack,
    entity: str,
    event_type: str,
    identity_field: str,
    aggregate_type: str,
) -> None:
    await arrange_complete_history(stack)
    async with stack.database.transaction() as uow:
        anchor = await uow.session.scalar(
            select(DomainEventRow)
            .where(DomainEventRow.event_type == event_type)
            .order_by(DomainEventRow.event_id)
        )
        topic_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == TopicCreatedV1.event_type
            )
        )
        assert anchor is not None
        assert topic_event is not None
        payload = stack.registry.decode(
            event_type=anchor.event_type,
            payload=anchor.payload,
        )
        payload_changes = {identity_field: ORPHAN_SOURCE_ID}
        aggregate_id = ORPHAN_SOURCE_ID
        sequence = 1

        if entity == "adoption":
            received = await uow.session.scalar(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == CapsuleReceivedV1.event_type)
                .order_by(DomainEventRow.event_id)
            )
            assert received is not None
            received_payload = stack.registry.decode(
                event_type=received.event_type,
                payload=received.payload,
            )
            assert isinstance(received_payload, CapsuleReceivedV1)
            uow.session.add(
                DomainEventRow(
                    aggregate_type="inbox_item",
                    aggregate_id=LATE_ITEM_ID,
                    sequence=1,
                    event_type=CapsuleReceivedV1.event_type,
                    payload=canonical_json_bytes(
                        received_payload.model_copy(
                            update={"item_id": LATE_ITEM_ID}
                        )
                    ),
                    actor_agent_id=received.actor_agent_id,
                    causation_id=topic_event.causation_id,
                    occurred_at=received.occurred_at,
                )
            )
            payload_changes["item_id"] = LATE_ITEM_ID
            aggregate_id = LATE_ITEM_ID
            sequence = 2

        uow.session.add(
            DomainEventRow(
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                sequence=sequence,
                event_type=event_type,
                payload=canonical_json_bytes(
                    payload.model_copy(update=payload_changes)
                ),
                actor_agent_id=anchor.actor_agent_id,
                causation_id=topic_event.causation_id,
                occurred_at=anchor.occurred_at,
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"{entity}:{ORPHAN_SOURCE_ID}",
    )


@pytest.mark.asyncio
async def test_adoption_source_without_event_is_rejected_independently(
    stack: AdoptionStack,
) -> None:
    history = await arrange_complete_history(stack)
    async with stack.database.transaction() as uow:
        uow.session.add(
            AdoptionRecordRow(
                adoption_id=ORPHAN_SOURCE_ID,
                adopter_agent_id=OTHER_AGENT_ID,
                capsule_id=history.relay.capsule_id,
                resulting_experience_id=ADOPTED_EXPERIENCE_ID,
                captured_trust=0.25,
                provenance_chain=canonical_json_bytes(
                    history.relay.provenance_chain
                ),
                root_fingerprint=history.relay.root_fingerprint,
                corroboration_applied=False,
                adopted_at=history.relay.created_at,
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"adoption:{ORPHAN_SOURCE_ID}",
    )


@pytest.mark.parametrize("orphan_side", ("source", "event"))
@pytest.mark.asyncio
async def test_topic_source_and_event_correspondence_is_bidirectional(
    stack: AdoptionStack,
    orphan_side: str,
) -> None:
    await arrange_pending_capsule(stack)
    async with stack.database.transaction() as uow:
        if orphan_side == "source":
            uow.session.add(
                TopicRow(
                    topic_id=ORPHAN_SOURCE_ID,
                    owner_agent_id=PUBLISHER_ID,
                    name="orphan-source-without-event",
                    description=None,
                    created_at=stack.clock.now(),
                )
            )
        else:
            anchor = await uow.session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.event_type == TopicCreatedV1.event_type
                )
            )
            assert anchor is not None
            uow.session.add(
                DomainEventRow(
                    aggregate_type="topic",
                    aggregate_id=ORPHAN_SOURCE_ID,
                    sequence=1,
                    event_type=TopicCreatedV1.event_type,
                    payload=canonical_json_bytes(
                        TopicCreatedV1(
                            schema_version=1,
                            topic_id=ORPHAN_SOURCE_ID,
                            owner_agent_id=PUBLISHER_ID,
                            name="orphan-event-without-source",
                            description=None,
                        )
                    ),
                    actor_agent_id=PUBLISHER_ID,
                    causation_id=anchor.causation_id,
                    occurred_at=stack.clock.now(),
                )
            )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"topic:{ORPHAN_SOURCE_ID}",
    )


@pytest.mark.asyncio
async def test_topic_creation_must_precede_subscription_and_publication(
    stack: AdoptionStack,
) -> None:
    capsule = await arrange_pending_capsule(stack)
    future_created_at = capsule.created_at + timedelta(seconds=10)
    async with stack.database.transaction() as uow:
        topic = await uow.session.get(TopicRow, TOPIC_ID)
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == TopicCreatedV1.event_type,
                DomainEventRow.aggregate_id == TOPIC_ID,
            )
        )
        assert topic is not None
        assert event is not None
        await uow.session.execute(text("DROP TRIGGER topics_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        topic.created_at = future_created_at
        event.occurred_at = future_created_at

    await assert_stable_source_error(stack, mismatch_prefix="subscription:")


@pytest.mark.parametrize(
    ("target_type", "lender_type", "mismatch_entity"),
    (
        (
            TopicCreatedV1.event_type,
            SubscriptionCreatedV1.event_type,
            "topic",
        ),
        (
            SubscriptionCreatedV1.event_type,
            TopicCreatedV1.event_type,
            "subscription",
        ),
    ),
)
@pytest.mark.asyncio
async def test_creation_event_cannot_borrow_another_commands_receipt(
    stack: AdoptionStack,
    target_type: str,
    lender_type: str,
    mismatch_entity: str,
) -> None:
    await arrange_pending_capsule(stack)
    async with stack.database.transaction() as uow:
        target = await uow.session.scalar(
            select(DomainEventRow).where(DomainEventRow.event_type == target_type)
        )
        lender = await uow.session.scalar(
            select(DomainEventRow).where(DomainEventRow.event_type == lender_type)
        )
        assert target is not None
        assert lender is not None
        target_id = target.aggregate_id
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        target.causation_id = lender.causation_id

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"{mismatch_entity}:{target_id}",
    )


@pytest.mark.asyncio
async def test_same_command_type_cannot_swap_completed_receipts(
    stack: AdoptionStack,
) -> None:
    await create_topic(stack)
    await subscribe_adopter(stack)
    await subscribe_other(stack)
    async with stack.database.transaction() as uow:
        events = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type
                        == SubscriptionCreatedV1.event_type
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        assert len(events) == 2
        first, second = events
        first_receipt = first.causation_id
        second_receipt = second.causation_id
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        first.causation_id = second_receipt
        second.causation_id = first_receipt

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"subscription:{first.aggregate_id}",
    )


@pytest.mark.asyncio
async def test_adoption_must_reference_the_adopters_result_and_owned_inbox(
    stack: AdoptionStack,
) -> None:
    await arrange_complete_history(stack)
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleAdoptedV1.event_type
            )
        )
        assert event is not None
        payload = stack.registry.decode(
            event_type=event.event_type,
            payload=event.payload,
        )
        assert isinstance(payload, CapsuleAdoptedV1)
        await uow.session.execute(text("DROP TRIGGER adoption_records_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(AdoptionRecordRow)
            .where(AdoptionRecordRow.adoption_id == ADOPTION_ID)
            .values(resulting_experience_id=SOURCE_EXPERIENCE_ID)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == event.event_id)
            .values(
                payload=canonical_json_bytes(
                    payload.model_copy(
                        update={"resulting_experience_id": SOURCE_EXPERIENCE_ID}
                    )
                )
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"adoption:{ADOPTION_ID}",
    )


@pytest.mark.asyncio
async def test_created_adoption_cannot_be_relabelled_as_corroboration(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-before-created-tamper")).status_code == 200
    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleAdoptedV1.event_type
            )
        )
        assert event is not None
        payload = stack.registry.decode(
            event_type=event.event_type,
            payload=event.payload,
        )
        assert isinstance(payload, CapsuleAdoptedV1)
        assert payload.created
        assert not payload.corroboration_applied
        await uow.session.execute(text("DROP TRIGGER adoption_records_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(AdoptionRecordRow)
            .where(AdoptionRecordRow.adoption_id == ADOPTION_ID)
            .values(corroboration_applied=True)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == event.event_id)
            .values(
                payload=canonical_json_bytes(
                    payload.model_copy(
                        update={
                            "created": False,
                            "corroboration_applied": True,
                        }
                    )
                )
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"adoption:{ADOPTION_ID}",
    )


@pytest.mark.asyncio
async def test_parent_adoption_event_must_precede_relay_publication(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-before-ledger-reorder")).status_code == 200
    await subscribe_other(stack)
    relay, _ = await publish_relay(stack)
    async with stack.database.transaction() as uow:
        adoption_event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleAdoptedV1.event_type,
                DomainEventRow.aggregate_id == ITEM_ID,
            )
        )
        publication = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsulePublishedV1.event_type,
                DomainEventRow.aggregate_id == relay.capsule_id,
            )
        )
        adoption = await uow.session.get(AdoptionRecordRow, ADOPTION_ID)
        assert adoption_event is not None
        assert publication is not None
        assert adoption is not None
        assert adoption_event.event_id < publication.event_id
        payload = stack.registry.decode(
            event_type=adoption_event.event_type,
            payload=adoption_event.payload,
        )
        assert isinstance(payload, CapsuleAdoptedV1)
        await uow.session.execute(text("DROP TRIGGER adoption_records_reject_delete"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_delete"))
        await uow.session.execute(
            text("DELETE FROM inbox_items WHERE item_id = :item_id"),
            {"item_id": str(ITEM_ID)},
        )
        await uow.session.execute(
            text("DELETE FROM adoption_records WHERE adoption_id = :adoption_id"),
            {"adoption_id": str(ADOPTION_ID)},
        )
        await uow.session.execute(
            text("DELETE FROM domain_events WHERE event_id = :event_id"),
            {"event_id": adoption_event.event_id},
        )
        future_event = DomainEventRow(
            aggregate_type="inbox_item",
            aggregate_id=ITEM_ID,
            sequence=2,
            event_type=CapsuleAdoptedV1.event_type,
            payload=canonical_json_bytes(
                payload.model_copy(update={"adoption_id": ORPHAN_SOURCE_ID})
            ),
            actor_agent_id=ADOPTER_ID,
            causation_id=adoption_event.causation_id,
            occurred_at=adoption_event.occurred_at,
        )
        uow.session.add(future_event)
        await uow.session.flush()
        assert future_event.event_id > publication.event_id
        uow.session.add_all(
            (
                AdoptionRecordRow(
                    adoption_id=ORPHAN_SOURCE_ID,
                    adopter_agent_id=adoption.adopter_agent_id,
                    capsule_id=adoption.capsule_id,
                    resulting_experience_id=adoption.resulting_experience_id,
                    captured_trust=adoption.captured_trust,
                    provenance_chain=adoption.provenance_chain,
                    root_fingerprint=adoption.root_fingerprint,
                    corroboration_applied=adoption.corroboration_applied,
                    adopted_at=adoption.adopted_at,
                ),
                InboxItemRow(
                    item_id=ITEM_ID,
                    recipient_agent_id=ADOPTER_ID,
                    capsule_id=CAPSULE_ID,
                    state=InboxState.ADOPTED,
                    projection_event_id=future_event.event_id,
                ),
            )
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{relay.capsule_id}",
    )


@pytest.mark.asyncio
async def test_terminal_inbox_clock_cannot_precede_prior_capsule_transition(
    stack: AdoptionStack,
) -> None:
    capsule = await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(seconds=10))
    assert (await retract(stack, key="retract-before-rejection")).status_code == 200
    stack.clock.advance(timedelta(seconds=10))
    assert (await reject(stack, key="reject-after-retraction")).status_code == 200

    async with stack.database.transaction() as uow:
        rejection = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleRejectedV1.event_type,
                DomainEventRow.aggregate_id == ITEM_ID,
            )
        )
        assert rejection is not None
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        rejection.occurred_at = capsule.created_at + timedelta(seconds=5)

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"inbox_item:{ITEM_ID}",
    )


@pytest.mark.asyncio
async def test_capsule_aggregate_clock_cannot_regress_at_retraction(
    stack: AdoptionStack,
) -> None:
    history = await arrange_complete_history(stack)
    stack.clock.advance(timedelta(seconds=10))
    assert (
        await retract(
            stack,
            key="retract-after-feedback-history",
            capsule_id=history.original.capsule_id,
        )
    ).status_code == 200

    async with stack.database.transaction() as uow:
        retraction = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type == CapsuleRetractedV1.event_type,
                DomainEventRow.aggregate_id == history.original.capsule_id,
            )
        )
        assert retraction is not None
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        retraction.occurred_at = history.original.created_at + timedelta(
            milliseconds=500
        )

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{history.original.capsule_id}",
    )


@pytest.mark.asyncio
async def test_capsule_aggregate_clock_cannot_regress_at_feedback(
    stack: AdoptionStack,
) -> None:
    capsule = await arrange_pending_capsule(stack)
    assert (await adopt(stack, key="adopt-before-clock-regression")).status_code == 200
    stack.clock.advance(timedelta(seconds=10))
    assert (await retract(stack, key="retract-before-feedback")).status_code == 200
    stack.clock.advance(timedelta(seconds=10))
    feedback_result = await record_feedback(
        stack,
        key="feedback-after-retraction",
        verdict=FeedbackVerdict.USEFUL,
    )
    assert feedback_result.status_code == 201
    feedback = FeedbackRevision.model_validate(
        json.loads(feedback_result.body)["data"],
        strict=False,
    )
    regressed_at = capsule.created_at + timedelta(seconds=5)

    async with stack.database.transaction() as uow:
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type
                == CapsuleFeedbackRecordedV1.event_type,
                DomainEventRow.aggregate_id == capsule.capsule_id,
            )
        )
        row = await uow.session.get(CapsuleFeedbackRow, feedback.feedback_id)
        assert event is not None
        assert row is not None
        await uow.session.execute(text("DROP TRIGGER capsule_feedback_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        row.created_at = regressed_at
        event.occurred_at = regressed_at

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"capsule:{capsule.capsule_id}",
    )


@pytest.mark.asyncio
async def test_reputation_pair_clock_cannot_regress_across_capsules(
    stack: AdoptionStack,
) -> None:
    first_capsule = await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-first-pair-clock")).status_code == 200
    second_capsule, second_item_id = await publish_again(
        stack,
        key="publish-second-pair-clock",
    )
    assert (
        await reject(
            stack,
            key="reject-second-pair-clock",
            item_id=second_item_id,
        )
    ).status_code == 200

    stack.clock.advance(timedelta(seconds=10))
    first_result = await record_feedback(
        stack,
        key="first-pair-clock-feedback",
        capsule_id=first_capsule.capsule_id,
        verdict=FeedbackVerdict.USEFUL,
    )
    assert first_result.status_code == 201
    stack.clock.advance(timedelta(seconds=10))
    second_result = await record_feedback(
        stack,
        key="second-pair-clock-feedback",
        capsule_id=second_capsule.capsule_id,
        verdict=FeedbackVerdict.REFUTED,
    )
    assert second_result.status_code == 201
    second_feedback = FeedbackRevision.model_validate(
        json.loads(second_result.body)["data"],
        strict=False,
    )
    regressed_at = first_capsule.created_at + timedelta(seconds=5)

    async with stack.database.transaction() as uow:
        row = await uow.session.get(
            CapsuleFeedbackRow,
            second_feedback.feedback_id,
        )
        event = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.event_type
                == CapsuleFeedbackRecordedV1.event_type,
                DomainEventRow.aggregate_id == second_capsule.capsule_id,
            )
        )
        assert row is not None
        assert event is not None
        receipt = await uow.session.get(
            IdempotencyRecordRow,
            event.causation_id,
        )
        assert receipt is not None
        await uow.session.execute(text("DROP TRIGGER capsule_feedback_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        row.created_at = regressed_at
        event.occurred_at = regressed_at
        receipt.created_at = regressed_at
        receipt.completed_at = regressed_at

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"feedback:{second_feedback.feedback_id}",
    )


@pytest.mark.asyncio
async def test_feedback_revisions_are_contiguous_and_match_ordered_events(
    stack: AdoptionStack,
) -> None:
    history = await arrange_complete_history(stack)
    second_feedback_id = history.feedback_ids[1]
    async with stack.database.transaction() as uow:
        events = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type
                        == CapsuleFeedbackRecordedV1.event_type
                    )
                )
            ).all()
        )
        event = next(
            (
                row
                for row in events
                if cast(
                    CapsuleFeedbackRecordedV1,
                    stack.registry.decode(
                        event_type=row.event_type,
                        payload=row.payload,
                    ),
                ).feedback_id
                == second_feedback_id
            ),
            None,
        )
        assert event is not None
        payload = stack.registry.decode(
            event_type=event.event_type,
            payload=event.payload,
        )
        assert isinstance(payload, CapsuleFeedbackRecordedV1)
        assert payload.feedback_id == second_feedback_id
        await uow.session.execute(text("DROP TRIGGER capsule_feedback_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(CapsuleFeedbackRow)
            .where(CapsuleFeedbackRow.feedback_id == second_feedback_id)
            .values(revision=3)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == event.event_id)
            .values(
                payload=canonical_json_bytes(payload.model_copy(update={"revision": 3}))
            )
        )

    await assert_stable_source_error(stack, mismatch_prefix="feedback:")


@pytest.mark.asyncio
async def test_feedback_clock_cannot_precede_its_terminal_inbox_event(
    stack: AdoptionStack,
) -> None:
    history = await arrange_complete_history(stack)
    first_feedback_id = history.feedback_ids[0]
    regressed_at = history.original.created_at - timedelta(seconds=1)
    async with stack.database.transaction() as uow:
        feedback = await uow.session.get(CapsuleFeedbackRow, first_feedback_id)
        events = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type
                        == CapsuleFeedbackRecordedV1.event_type
                    )
                )
            ).all()
        )
        event = next(
            row
            for row in events
            if cast(
                CapsuleFeedbackRecordedV1,
                stack.registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                ),
            ).feedback_id
            == first_feedback_id
        )
        assert feedback is not None
        await uow.session.execute(text("DROP TRIGGER capsule_feedback_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        feedback.created_at = regressed_at
        event.occurred_at = regressed_at

    await assert_stable_source_error(
        stack,
        mismatch_prefix=f"feedback:{first_feedback_id}",
    )


@pytest.mark.asyncio
async def test_non_sharing_event_cannot_occupy_sharing_aggregate_namespace(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    command = ConfirmExperience(
        owner_agent_id=PUBLISHER_ID,
        experience_id=SOURCE_EXPERIENCE_ID,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.experience_service.confirm(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await stack.executor.execute(
        request(
            key="confirm-before-sharing-namespace-corruption",
            operation_scope="experience.confirm",
            route_template="/v1/experiences/{experience_id}:confirm",
            agent_id=PUBLISHER_ID,
            path_parameters={"experience_id": SOURCE_EXPERIENCE_ID},
            body={"experience_id": SOURCE_EXPERIENCE_ID},
        ),
        handler,
    )
    assert result.status_code == 200

    async with stack.database.transaction() as uow:
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_type == "experience",
                        DomainEventRow.aggregate_id == SOURCE_EXPERIENCE_ID,
                        DomainEventRow.event_type.in_(
                            (
                                "experience.confirmed",
                                "experience.temperature_changed",
                            )
                        ),
                    )
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
        assert tuple(row.event_type for row in rows) == (
            "experience.confirmed",
            "experience.temperature_changed",
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        for sequence, row in enumerate(rows, start=2):
            await uow.session.execute(
                update(DomainEventRow)
                .where(DomainEventRow.event_id == row.event_id)
                .values(
                    aggregate_type="capsule",
                    aggregate_id=CAPSULE_ID,
                    sequence=sequence,
                )
            )

    await assert_stable_source_error(
        stack,
        mismatch_prefix="sharing_aggregate:capsule:",
    )


async def projection_snapshot(
    stack: AdoptionStack,
) -> tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]:
    statements = {
        "agent_reputation": (
            "SELECT subject_agent_id, observer_agent_id, useful_count, "
            "refuted_count, harmful_count, alpha, beta, projection_event_id "
            "FROM agent_reputation "
            "ORDER BY subject_agent_id, observer_agent_id"
        ),
        "capsule_state": (
            "SELECT capsule_id, status, projection_event_id "
            "FROM capsule_state ORDER BY capsule_id"
        ),
        "inbox_items": (
            "SELECT item_id, recipient_agent_id, capsule_id, state, "
            "projection_event_id FROM inbox_items ORDER BY item_id"
        ),
        "projection_versions": (
            "SELECT name, reducer_version, last_applied_event_id, "
            "last_verified_hash, last_verified_at "
            "FROM projection_versions ORDER BY name"
        ),
    }
    async with stack.database.read_session() as session:
        snapshot: list[tuple[str, tuple[tuple[Any, ...], ...]]] = []
        for name, statement in statements.items():
            rows = await session.execute(text(statement))
            snapshot.append((name, tuple(tuple(row) for row in rows)))
        return tuple(snapshot)


@pytest.mark.parametrize("operation", ("verify", "repair"))
@pytest.mark.parametrize("corruption", ("transport_hash", "chain"))
@pytest.mark.asyncio
async def test_corrupt_capsule_aborts_verify_or_repair_without_partial_projection_write(
    stack: AdoptionStack,
    operation: str,
    corruption: str,
) -> None:
    history = await arrange_complete_history(stack)
    assert (await manager(stack).verify(stack.database)).matches
    before = await projection_snapshot(stack)
    if corruption == "transport_hash":
        async with stack.database.transaction() as uow:
            await uow.session.execute(
                text("DROP TRIGGER experience_capsules_reject_update")
            )
            await uow.session.execute(
                update(ExperienceCapsuleRow)
                .where(ExperienceCapsuleRow.capsule_id == history.original.capsule_id)
                .values(capsule_hash=BAD_HASH)
            )
        corrupted_id = history.original.capsule_id
    else:
        await rewrite_capsule_semantics(
            stack,
            capsule=history.relay,
            changes={
                "provenance_chain": (
                    ProvenanceHop(
                        capsule_id=history.relay.capsule_id,
                        publisher_agent_id=ADOPTER_ID,
                    ),
                )
            },
        )
        corrupted_id = history.relay.capsule_id

    try:
        with pytest.raises(SourceIntegrityError) as caught:
            await getattr(manager(stack), operation)(stack.database)
        assert caught.value.mismatch_key.startswith(f"capsule:{corrupted_id}")
    finally:
        assert await projection_snapshot(stack) == before
        async with stack.database.read_session() as session:
            temp_count = await session.scalar(
                text(
                    "SELECT count(*) FROM sqlite_temp_master "
                    "WHERE name LIKE '_rebuild_%'"
                )
            )
        assert temp_count == 0
