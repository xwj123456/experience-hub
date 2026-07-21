"""Public contracts for social experience propagation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from experience_hub.sharing.models import (
    MAX_PROVENANCE_HOPS,
    MAX_TOPIC_DESCRIPTION_CHARACTERS,
    MAX_TOPIC_NAME_CHARACTERS,
    AdoptCapsule,
    AdoptionResult,
    Capsule,
    CapsuleStatus,
    CreateSubscription,
    CreateTopic,
    EffectiveAvailability,
    FeedbackRevision,
    FeedbackVerdict,
    InboxItem,
    InboxState,
    ProvenanceHop,
    PublishCapsule,
    RecordCapsuleFeedback,
    RejectInboxItem,
    Reputation,
    RetractCapsule,
    SharingMutationReason,
    Subscription,
    Topic,
)

if TYPE_CHECKING:
    from experience_hub.sharing.queries import (
        InboxEvidenceReader,
        InboxPage,
        QuarantinedCapsuleEvidence,
        SharingQuery,
    )

_LAZY_QUERY_EXPORTS = frozenset(
    {
        "InboxEvidenceReader",
        "InboxPage",
        "QuarantinedCapsuleEvidence",
        "SharingQuery",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_QUERY_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("experience_hub.sharing.queries"), name)
    globals()[name] = value
    return value

__all__ = [
    "MAX_PROVENANCE_HOPS",
    "MAX_TOPIC_DESCRIPTION_CHARACTERS",
    "MAX_TOPIC_NAME_CHARACTERS",
    "AdoptionResult",
    "AdoptCapsule",
    "Capsule",
    "CapsuleStatus",
    "CreateSubscription",
    "CreateTopic",
    "EffectiveAvailability",
    "FeedbackRevision",
    "FeedbackVerdict",
    "InboxItem",
    "InboxEvidenceReader",
    "InboxPage",
    "InboxState",
    "ProvenanceHop",
    "PublishCapsule",
    "RecordCapsuleFeedback",
    "RejectInboxItem",
    "Reputation",
    "RetractCapsule",
    "SharingMutationReason",
    "QuarantinedCapsuleEvidence",
    "SharingQuery",
    "Subscription",
    "Topic",
]
