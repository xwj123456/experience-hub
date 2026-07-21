"""Strict, versioned domain event contracts."""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EventPayload(BaseModel):
    """Base model for a registered, explicitly versioned event payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: ClassVar[str]
    schema_version: Literal[1]


class EventRegistry:
    """Bind each persisted event name to exactly one payload model."""

    def __init__(self) -> None:
        self._payload_types: dict[str, type[EventPayload]] = {}

    @property
    def event_types(self) -> frozenset[str]:
        """Return the immutable set of event names known by this registry."""
        return frozenset(self._payload_types)

    def register(self, payload_type: type[EventPayload]) -> None:
        try:
            event_type = payload_type.event_type
        except AttributeError as error:
            raise ValueError(
                "Event payload class must declare an event_type"
            ) from error
        if not event_type or event_type != event_type.strip():
            raise ValueError("Event type must be a non-empty trimmed string")
        if event_type in self._payload_types:
            raise ValueError(f"Event type {event_type!r} is already registered")
        self._payload_types[event_type] = payload_type

    def decode(self, *, event_type: str, payload: bytes) -> EventPayload:
        payload_type = self.payload_type(event_type)
        return payload_type.model_validate_json(payload)

    def payload_type(self, event_type: str) -> type[EventPayload]:
        try:
            return self._payload_types[event_type]
        except KeyError as error:
            raise ValueError(f"Unknown event type: {event_type}") from error


@dataclass(frozen=True, slots=True)
class PendingEvent:
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    payload: EventPayload
    actor_agent_id: UUID | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class StoredEvent:
    event_id: int
    aggregate_type: str
    aggregate_id: UUID
    sequence: int
    event_type: str
    payload: EventPayload
    actor_agent_id: UUID | None
    causation_id: UUID
    occurred_at: datetime
