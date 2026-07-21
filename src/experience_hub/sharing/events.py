"""Strict version-one events for sharing sources and capsule delivery.

Sharing events are registered progressively. An event's immutable V1 model is
introduced only with its first producer and complete reducer semantics.
Tasks 2-6 register topic/subscription creation, publication/delivery, explicit
adoption, retraction, rejection, and observer-relative feedback revisions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Self
from uuid import UUID

from pydantic import field_validator, model_validator

from experience_hub.domain import EventPayload, EventRegistry, StructuredReason
from experience_hub.sharing.models import (
    MAX_TOPIC_DESCRIPTION_CHARACTERS,
    MAX_TOPIC_NAME_CHARACTERS,
    CapsuleStatus,
    FeedbackVerdict,
    InboxState,
)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _require_transportable_text(label: str, value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must contain valid Unicode") from error
    return value


class TopicCreatedV1(EventPayload):
    """Record the immutable creation attributes of one topic."""

    event_type: ClassVar[str] = "topic.created"

    topic_id: UUID
    owner_agent_id: UUID
    name: str
    description: str | None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        _require_transportable_text("Topic event name", value)
        if value != value.strip() or not 1 <= len(value) <= MAX_TOPIC_NAME_CHARACTERS:
            raise ValueError("Topic event name must be canonical and non-empty")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _require_transportable_text("Topic event description", value)
        if (
            value != value.strip()
            or not 1 <= len(value) <= MAX_TOPIC_DESCRIPTION_CHARACTERS
        ):
            raise ValueError("Topic event description must be canonical and non-empty")
        return value


class SubscriptionCreatedV1(EventPayload):
    """Record one immutable, non-backfilling subscription."""

    event_type: ClassVar[str] = "subscription.created"

    subscription_id: UUID
    subscriber_agent_id: UUID
    topic_id: UUID


class CapsulePublishedV1(EventPayload):
    """Name the immutable capsule source consumed by capsule-state replay."""

    event_type: ClassVar[str] = "capsule.published"

    capsule_id: UUID
    topic_id: UUID
    source_experience_id: UUID
    source_version_id: UUID
    publisher_agent_id: UUID
    capsule_hash: str
    root_fingerprint: str
    status_after: CapsuleStatus

    @field_validator("capsule_hash", "root_fingerprint")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_initial_status(self) -> Self:
        if self.status_after is not CapsuleStatus.ACTIVE:
            raise ValueError("Published capsules must start active")
        return self


class CapsuleReceivedV1(EventPayload):
    """Allocate the stable inbox identity used by delivery replay."""

    event_type: ClassVar[str] = "capsule.received"

    item_id: UUID
    capsule_id: UUID
    recipient_agent_id: UUID
    state_after: InboxState

    @model_validator(mode="after")
    def validate_initial_state(self) -> Self:
        if self.state_after is not InboxState.PENDING:
            raise ValueError("Received capsules must enter the pending state")
        return self


class CapsuleAdoptedV1(EventPayload):
    """Record one immutable adoption result and pending-to-adopted transition."""

    event_type: ClassVar[str] = "capsule.adopted"

    item_id: UUID
    capsule_id: UUID
    adopter_agent_id: UUID
    adoption_id: UUID
    resulting_experience_id: UUID
    root_fingerprint: str
    created: bool
    corroboration_applied: bool
    state_before: InboxState
    state_after: InboxState

    @field_validator("root_fingerprint")
    @classmethod
    def validate_root_fingerprint(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("root_fingerprint must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_transition_and_result(self) -> Self:
        if (
            self.state_before is not InboxState.PENDING
            or self.state_after is not InboxState.ADOPTED
        ):
            raise ValueError(
                "Adoption must transition inbox state from pending to adopted"
            )
        if self.created and self.corroboration_applied:
            raise ValueError(
                "A newly created experience cannot receive corroboration"
            )
        return self


class CapsuleRetractedV1(EventPayload):
    """Record the only legal capsule-state withdrawal transition."""

    event_type: ClassVar[str] = "capsule.retracted"

    capsule_id: UUID
    publisher_agent_id: UUID
    reason: StructuredReason
    status_before: CapsuleStatus
    status_after: CapsuleStatus

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        if (
            self.status_before is not CapsuleStatus.ACTIVE
            or self.status_after is not CapsuleStatus.RETRACTED
        ):
            raise ValueError(
                "Retraction must transition capsule status from active to retracted"
            )
        return self


class CapsuleRejectedV1(EventPayload):
    """Record one recipient-owned pending-to-rejected inbox transition."""

    event_type: ClassVar[str] = "capsule.rejected"

    item_id: UUID
    capsule_id: UUID
    recipient_agent_id: UUID
    reason: StructuredReason
    state_before: InboxState
    state_after: InboxState

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        if (
            self.state_before is not InboxState.PENDING
            or self.state_after is not InboxState.REJECTED
        ):
            raise ValueError(
                "Rejection must transition inbox state from pending to rejected"
            )
        return self


class CapsuleFeedbackRecordedV1(EventPayload):
    """Apply one immutable feedback revision to observer-relative reputation."""

    event_type: ClassVar[str] = "capsule.feedback_recorded"

    feedback_id: UUID
    observer_agent_id: UUID
    capsule_id: UUID
    publisher_agent_id: UUID
    revision: int
    previous_verdict: FeedbackVerdict | None
    current_verdict: FeedbackVerdict
    alpha_before: int
    beta_before: int
    alpha_after: int
    beta_after: int

    @field_validator("revision", mode="before")
    @classmethod
    def validate_revision(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("revision must be a positive integer")
        return value

    @field_validator(
        "alpha_before",
        "beta_before",
        "alpha_after",
        "beta_after",
        mode="before",
    )
    @classmethod
    def validate_effective_count(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{info.field_name} must be an integer at least two")
        return value

    @model_validator(mode="after")
    def validate_revision_transition(self) -> Self:
        if self.publisher_agent_id == self.observer_agent_id:
            raise ValueError("Feedback subject and observer must be distinct")
        if (self.revision == 1) is not (self.previous_verdict is None):
            raise ValueError(
                "Only a first feedback revision may omit previous_verdict"
            )
        expected_alpha = self.alpha_before
        expected_beta = self.beta_before
        if self.previous_verdict is FeedbackVerdict.USEFUL:
            expected_alpha -= 1
        elif self.previous_verdict in (
            FeedbackVerdict.REFUTED,
            FeedbackVerdict.HARMFUL,
        ):
            expected_beta -= 1
        if self.current_verdict is FeedbackVerdict.USEFUL:
            expected_alpha += 1
        else:
            expected_beta += 1
        if (
            self.alpha_after != expected_alpha
            or self.beta_after != expected_beta
        ):
            raise ValueError(
                "Feedback alpha/beta after-values do not match the revision"
            )
        return self


_SHARING_EVENT_PAYLOAD_TYPES = (
    TopicCreatedV1,
    SubscriptionCreatedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleAdoptedV1,
    CapsuleRetractedV1,
    CapsuleRejectedV1,
    CapsuleFeedbackRecordedV1,
)

SHARING_EVENT_TYPES = frozenset(
    payload_type.event_type for payload_type in _SHARING_EVENT_PAYLOAD_TYPES
)

SHARING_EVENT_AGGREGATE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        TopicCreatedV1.event_type: "topic",
        SubscriptionCreatedV1.event_type: "subscription",
        CapsulePublishedV1.event_type: "capsule",
        CapsuleReceivedV1.event_type: "inbox_item",
        CapsuleAdoptedV1.event_type: "inbox_item",
        CapsuleRetractedV1.event_type: "capsule",
        CapsuleRejectedV1.event_type: "inbox_item",
        CapsuleFeedbackRecordedV1.event_type: "capsule",
    }
)


def register_sharing_events(registry: EventRegistry) -> None:
    """Register the immutable sharing protocol implemented through Task 6."""
    for payload_type in _SHARING_EVENT_PAYLOAD_TYPES:
        registry.register(payload_type)


__all__ = [
    "SHARING_EVENT_AGGREGATE_TYPES",
    "SHARING_EVENT_TYPES",
    "CapsulePublishedV1",
    "CapsuleReceivedV1",
    "CapsuleAdoptedV1",
    "CapsuleFeedbackRecordedV1",
    "CapsuleRejectedV1",
    "CapsuleRetractedV1",
    "SubscriptionCreatedV1",
    "TopicCreatedV1",
    "register_sharing_events",
]
