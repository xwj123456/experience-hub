"""Strict HTTP request contracts for social experience propagation."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import ConfigDict, Field, StrictFloat, field_validator

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import StrictModel, StructuredReason, TypedEvidence
from experience_hub.experiences.models import MAX_VERSION_LIST_ITEMS
from experience_hub.sharing import (
    MAX_TOPIC_DESCRIPTION_CHARACTERS,
    MAX_TOPIC_NAME_CHARACTERS,
    FeedbackVerdict,
    InboxState,
)

UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
_RFC3339_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})\Z"
)
_CANONICAL_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")


class _RequestModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _validate_unicode(value: str, *, field_name: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must contain valid Unicode") from error
    return value


def _trimmed_text(
    value: str,
    *,
    field_name: str,
    maximum: int,
) -> str:
    retained = value.strip()
    _validate_unicode(retained, field_name=field_name)
    if not 1 <= len(retained) <= maximum:
        raise ValueError(
            f"{field_name} must contain 1-{maximum:,} characters after trimming"
        )
    return retained


def _enforce_list_limit(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and len(values) > MAX_VERSION_LIST_ITEMS:
        raise ValueError(
            f"Lists may contain at most {MAX_VERSION_LIST_ITEMS} input items"
        )
    return values


def _canonical_evidence(
    values: tuple[TypedEvidence, ...],
) -> tuple[TypedEvidence, ...]:
    for item in values:
        _validate_unicode(item.type, field_name="Evidence type")
        _validate_unicode(item.id, field_name="Evidence id")
    by_bytes = {canonical_json_bytes(item): item for item in values}
    return tuple(by_bytes[key] for key in sorted(by_bytes))


class CreateTopicRequest(_RequestModel):
    """Create a topic for an explicit owner in the unauthenticated local API."""

    owner_agent_id: UUID
    name: str
    description: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _trimmed_text(
            value,
            field_name="Topic name",
            maximum=MAX_TOPIC_NAME_CHARACTERS,
        )

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _trimmed_text(
            value,
            field_name="Topic description",
            maximum=MAX_TOPIC_DESCRIPTION_CHARACTERS,
        )


class CreateSubscriptionRequest(_RequestModel):
    topic_id: UUID


class PublishCapsuleRequest(_RequestModel):
    topic_id: UUID
    experience_id: UUID
    version_id: UUID | None = None
    expires_at: datetime
    parent_adoption_id: UUID | None = None

    @field_validator("expires_at", mode="before")
    @classmethod
    def require_expiry_shape(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and _RFC3339_TIMESTAMP.fullmatch(value):
            return value
        raise ValueError("expires_at must be an RFC 3339 timestamp")

    @field_validator("expires_at", mode="after")
    @classmethod
    def normalize_expiry(cls, value: datetime) -> datetime:
        return require_utc(value)


class AdoptInboxItemRequest(_RequestModel):
    importance: UnitFloat = 0.50


class RequiredReasonRequest(_RequestModel):
    reason: str

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return _trimmed_text(
            value,
            field_name="Reason",
            maximum=2_000,
        )

    def to_reason(self) -> StructuredReason:
        return StructuredReason.from_user_text(self.reason)


class RecordFeedbackRequest(RequiredReasonRequest):
    verdict: FeedbackVerdict
    evidence: tuple[TypedEvidence, ...] = ()

    @field_validator("evidence", mode="before")
    @classmethod
    def enforce_raw_evidence_limit(cls, values: Any) -> Any:
        return _enforce_list_limit(values)

    @field_validator("evidence")
    @classmethod
    def canonicalize_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        return _canonical_evidence(values)


class InboxListQuery(StrictModel):
    limit: Annotated[int, Field(ge=1, le=100)] = 100
    cursor: str | None = None
    state: InboxState | None = None

    @field_validator("limit", mode="before")
    @classmethod
    def require_canonical_limit(cls, value: Any) -> int:
        if type(value) is int:
            return value
        if isinstance(value, str) and _CANONICAL_POSITIVE_INTEGER.fullmatch(value):
            return int(value)
        raise ValueError("limit must be a canonical positive integer")


__all__ = [
    "AdoptInboxItemRequest",
    "CreateSubscriptionRequest",
    "CreateTopicRequest",
    "InboxListQuery",
    "PublishCapsuleRequest",
    "RecordFeedbackRequest",
    "RequiredReasonRequest",
]
