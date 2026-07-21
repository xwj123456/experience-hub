from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.domain import EventPayload, EventRegistry, StructuredReason
from experience_hub.sharing.events import (
    SHARING_EVENT_AGGREGATE_TYPES,
    SHARING_EVENT_TYPES,
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    SubscriptionCreatedV1,
    TopicCreatedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import (
    CapsuleStatus,
    FeedbackVerdict,
    InboxState,
)

TOPIC_ID = UUID("00000000-0000-0000-0000-000000000101")
SUBSCRIPTION_ID = UUID("00000000-0000-0000-0000-000000000102")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000201")
RECIPIENT_ID = UUID("00000000-0000-0000-0000-000000000203")
CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000301")
ITEM_ID = UUID("00000000-0000-0000-0000-000000000302")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000303")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000304")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000305")
FEEDBACK_ID = UUID("00000000-0000-0000-0000-000000000306")


def _golden_payloads() -> tuple[EventPayload, ...]:
    return (
        TopicCreatedV1(
            schema_version=1,
            topic_id=TOPIC_ID,
            owner_agent_id=OWNER_ID,
            name="distributed-memory",
            description="Experiences suitable for deliberate propagation.",
        ),
        SubscriptionCreatedV1(
            schema_version=1,
            subscription_id=SUBSCRIPTION_ID,
            subscriber_agent_id=RECIPIENT_ID,
            topic_id=TOPIC_ID,
        ),
    )


def _compact_schema(payload_type: type[EventPayload]) -> dict[str, Any]:
    schema = payload_type.model_json_schema()
    return {
        "additionalProperties": schema.get("additionalProperties"),
        "properties": schema["properties"],
        "required": schema["required"],
        "type": schema["type"],
    }


def test_registry_has_exact_cumulative_vocabulary_and_aggregate_ownership() -> None:
    registry = EventRegistry()

    register_sharing_events(registry)

    expected_ownership = {
        "topic.created": "topic",
        "subscription.created": "subscription",
        "capsule.published": "capsule",
        "capsule.received": "inbox_item",
        "capsule.adopted": "inbox_item",
        "capsule.retracted": "capsule",
        "capsule.rejected": "inbox_item",
        "capsule.feedback_recorded": "capsule",
    }
    assert registry.event_types == SHARING_EVENT_TYPES == frozenset(expected_ownership)
    assert expected_ownership == SHARING_EVENT_AGGREGATE_TYPES

    assert registry.payload_type("capsule.published") is CapsulePublishedV1
    assert registry.payload_type("capsule.received") is CapsuleReceivedV1
    assert registry.payload_type("capsule.adopted") is CapsuleAdoptedV1
    assert registry.payload_type("capsule.retracted") is CapsuleRetractedV1
    assert registry.payload_type("capsule.rejected") is CapsuleRejectedV1
    assert (
        registry.payload_type("capsule.feedback_recorded")
        is CapsuleFeedbackRecordedV1
    )


def test_feedback_payload_freezes_revision_and_effective_count_transition() -> None:
    payload = CapsuleFeedbackRecordedV1(
        schema_version=1,
        feedback_id=FEEDBACK_ID,
        observer_agent_id=RECIPIENT_ID,
        capsule_id=CAPSULE_ID,
        publisher_agent_id=OWNER_ID,
        revision=1,
        previous_verdict=None,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )

    assert set(type(payload).model_fields) == {
        "schema_version",
        "feedback_id",
        "observer_agent_id",
        "capsule_id",
        "publisher_agent_id",
        "revision",
        "previous_verdict",
        "current_verdict",
        "alpha_before",
        "beta_before",
        "alpha_after",
        "beta_after",
    }
    values = payload.model_dump(mode="python")
    with pytest.raises(ValidationError):
        CapsuleFeedbackRecordedV1.model_validate({**values, "reason": "private"})
    with pytest.raises(ValidationError, match="after-values"):
        CapsuleFeedbackRecordedV1.model_validate(
            {**values, "alpha_after": 2}
        )
    with pytest.raises(ValidationError, match="previous_verdict"):
        CapsuleFeedbackRecordedV1.model_validate(
            {
                **values,
                "revision": 2,
                "previous_verdict": None,
            }
        )


def test_task2_payloads_match_exact_golden_fields() -> None:
    topic, subscription = _golden_payloads()

    assert set(type(topic).model_fields) == {
        "schema_version",
        "topic_id",
        "owner_agent_id",
        "name",
        "description",
    }
    assert set(type(subscription).model_fields) == {
        "schema_version",
        "subscription_id",
        "subscriber_agent_id",
        "topic_id",
    }


def test_task2_payload_json_schema_is_frozen() -> None:
    assert _compact_schema(TopicCreatedV1) == {
        "additionalProperties": False,
        "properties": {
            "schema_version": {
                "const": 1,
                "title": "Schema Version",
                "type": "integer",
            },
            "topic_id": {
                "format": "uuid",
                "title": "Topic Id",
                "type": "string",
            },
            "owner_agent_id": {
                "format": "uuid",
                "title": "Owner Agent Id",
                "type": "string",
            },
            "name": {"title": "Name", "type": "string"},
            "description": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Description",
            },
        },
        "required": [
            "schema_version",
            "topic_id",
            "owner_agent_id",
            "name",
            "description",
        ],
        "type": "object",
    }
    assert _compact_schema(SubscriptionCreatedV1) == {
        "additionalProperties": False,
        "properties": {
            "schema_version": {
                "const": 1,
                "title": "Schema Version",
                "type": "integer",
            },
            "subscription_id": {
                "format": "uuid",
                "title": "Subscription Id",
                "type": "string",
            },
            "subscriber_agent_id": {
                "format": "uuid",
                "title": "Subscriber Agent Id",
                "type": "string",
            },
            "topic_id": {
                "format": "uuid",
                "title": "Topic Id",
                "type": "string",
            },
        },
        "required": [
            "schema_version",
            "subscription_id",
            "subscriber_agent_id",
            "topic_id",
        ],
        "type": "object",
    }


@pytest.mark.parametrize(
    "payload",
    _golden_payloads(),
    ids=lambda payload: payload.event_type,
)
@pytest.mark.parametrize("forbidden_field", ["body", "query", "error"])
def test_task2_events_fail_closed_on_sensitive_or_unstructured_fields(
    payload: EventPayload,
    forbidden_field: str,
) -> None:
    values = payload.model_dump(mode="python")

    with pytest.raises(ValidationError):
        type(payload).model_validate({**values, forbidden_field: "private material"})


@pytest.mark.parametrize(
    "payload",
    _golden_payloads(),
    ids=lambda payload: payload.event_type,
)
def test_registry_rejects_wrong_version_extra_and_missing_fields(
    payload: EventPayload,
) -> None:
    registry = EventRegistry()
    register_sharing_events(registry)
    values = payload.model_dump(mode="json")

    with pytest.raises(ValidationError):
        registry.decode(
            event_type=payload.event_type,
            payload=payload.model_dump_json()
            .replace('"schema_version":1', '"schema_version":2')
            .encode(),
        )
    with pytest.raises(ValidationError):
        type(payload).model_validate({**values, "unexpected": True})
    with pytest.raises(ValidationError):
        type(payload).model_validate(
            {name: value for name, value in values.items() if name != "schema_version"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", " distributed-memory"),
        ("name", ""),
        ("name", "x" * 201),
        ("name", "\ud800"),
        ("description", " trailing "),
        ("description", ""),
        ("description", "x" * 2001),
        ("description", "\ud800"),
    ],
)
def test_topic_event_requires_canonical_transportable_text(
    field: str,
    value: str,
) -> None:
    values = _golden_payloads()[0].model_dump(mode="python")

    with pytest.raises(ValidationError):
        TopicCreatedV1.model_validate({**values, field: value})


def _published_capsule() -> CapsulePublishedV1:
    return CapsulePublishedV1(
        schema_version=1,
        capsule_id=CAPSULE_ID,
        topic_id=TOPIC_ID,
        source_experience_id=EXPERIENCE_ID,
        source_version_id=VERSION_ID,
        publisher_agent_id=OWNER_ID,
        capsule_hash="a" * 64,
        root_fingerprint="b" * 64,
        status_after=CapsuleStatus.ACTIVE,
    )


@pytest.mark.parametrize("invalid_hash", ["A" * 64, "a" * 63, "g" * 64])
@pytest.mark.parametrize("field", ["capsule_hash", "root_fingerprint"])
def test_published_capsule_requires_lowercase_sha256_hashes(
    field: str,
    invalid_hash: str,
) -> None:
    values = _published_capsule().model_dump(mode="python")

    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        CapsulePublishedV1.model_validate({**values, field: invalid_hash})


def test_published_capsule_can_only_create_active_state() -> None:
    values = _published_capsule().model_dump(mode="python")

    with pytest.raises(ValidationError, match="start active"):
        CapsulePublishedV1.model_validate(
            {
                **values,
                "status_after": CapsuleStatus.RETRACTED,
            }
        )


@pytest.mark.parametrize(
    "state",
    [InboxState.ADOPTED, InboxState.REJECTED],
)
def test_received_capsule_can_only_create_pending_state(
    state: InboxState,
) -> None:
    with pytest.raises(ValidationError, match="pending state"):
        CapsuleReceivedV1(
            schema_version=1,
            item_id=ITEM_ID,
            capsule_id=CAPSULE_ID,
            recipient_agent_id=RECIPIENT_ID,
            state_after=state,
        )


def _adopted_capsule() -> CapsuleAdoptedV1:
    return CapsuleAdoptedV1(
        schema_version=1,
        item_id=ITEM_ID,
        capsule_id=CAPSULE_ID,
        adopter_agent_id=RECIPIENT_ID,
        adoption_id=ADOPTION_ID,
        resulting_experience_id=EXPERIENCE_ID,
        root_fingerprint="c" * 64,
        created=False,
        corroboration_applied=True,
        state_before=InboxState.PENDING,
        state_after=InboxState.ADOPTED,
    )


def test_adopted_capsule_has_exact_strict_v1_fields() -> None:
    payload = _adopted_capsule()

    assert set(type(payload).model_fields) == {
        "schema_version",
        "item_id",
        "capsule_id",
        "adopter_agent_id",
        "adoption_id",
        "resulting_experience_id",
        "root_fingerprint",
        "created",
        "corroboration_applied",
        "state_before",
        "state_after",
    }
    with pytest.raises(ValidationError):
        CapsuleAdoptedV1.model_validate(
            {**payload.model_dump(mode="python"), "body": "forbidden"}
        )


@pytest.mark.parametrize(
    ("state_before", "state_after"),
    [
        (InboxState.ADOPTED, InboxState.ADOPTED),
        (InboxState.PENDING, InboxState.REJECTED),
    ],
)
def test_adopted_capsule_requires_pending_to_adopted_transition(
    state_before: InboxState,
    state_after: InboxState,
) -> None:
    payload = _adopted_capsule()

    with pytest.raises(ValidationError, match="pending to adopted"):
        CapsuleAdoptedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "state_before": state_before,
                "state_after": state_after,
            }
        )


def test_new_adopted_experience_cannot_also_be_corroborated() -> None:
    payload = _adopted_capsule()

    with pytest.raises(ValidationError, match="newly created"):
        CapsuleAdoptedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "created": True,
                "corroboration_applied": True,
            }
        )


def _reason() -> StructuredReason:
    return StructuredReason.from_user_text("The transported claim is obsolete.")


def _retracted_capsule() -> CapsuleRetractedV1:
    return CapsuleRetractedV1(
        schema_version=1,
        capsule_id=CAPSULE_ID,
        publisher_agent_id=OWNER_ID,
        reason=_reason(),
        status_before=CapsuleStatus.ACTIVE,
        status_after=CapsuleStatus.RETRACTED,
    )


def _rejected_capsule() -> CapsuleRejectedV1:
    return CapsuleRejectedV1(
        schema_version=1,
        item_id=ITEM_ID,
        capsule_id=CAPSULE_ID,
        recipient_agent_id=RECIPIENT_ID,
        reason=_reason(),
        state_before=InboxState.PENDING,
        state_after=InboxState.REJECTED,
    )


def test_governance_events_have_exact_strict_v1_fields() -> None:
    retracted = _retracted_capsule()
    rejected = _rejected_capsule()

    assert set(type(retracted).model_fields) == {
        "schema_version",
        "capsule_id",
        "publisher_agent_id",
        "reason",
        "status_before",
        "status_after",
    }
    assert set(type(rejected).model_fields) == {
        "schema_version",
        "item_id",
        "capsule_id",
        "recipient_agent_id",
        "reason",
        "state_before",
        "state_after",
    }
    for payload in (retracted, rejected):
        with pytest.raises(ValidationError):
            type(payload).model_validate(
                {**payload.model_dump(mode="python"), "body": "forbidden"}
            )


@pytest.mark.parametrize(
    ("payload", "before_field", "after_field", "before", "after", "message"),
    [
        (
            _retracted_capsule(),
            "status_before",
            "status_after",
            CapsuleStatus.RETRACTED,
            CapsuleStatus.RETRACTED,
            "active to retracted",
        ),
        (
            _rejected_capsule(),
            "state_before",
            "state_after",
            InboxState.ADOPTED,
            InboxState.REJECTED,
            "pending to rejected",
        ),
    ],
)
def test_governance_events_require_the_only_legal_transition(
    payload: EventPayload,
    before_field: str,
    after_field: str,
    before: CapsuleStatus | InboxState,
    after: CapsuleStatus | InboxState,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        type(payload).model_validate(
            {
                **payload.model_dump(mode="python"),
                before_field: before,
                after_field: after,
            }
        )
