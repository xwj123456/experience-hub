"""Strict domain events emitted by the agent aggregate."""

from typing import ClassVar
from uuid import UUID

from experience_hub.domain.events import EventPayload, EventRegistry


class AgentCreated(EventPayload):
    event_type: ClassVar[str] = "agent.created"

    agent_id: UUID
    name: str


def register_agent_events(registry: EventRegistry) -> None:
    registry.register(AgentCreated)
