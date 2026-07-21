"""Strict HTTP contracts for agent resources and list pagination."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, Field

from experience_hub.domain.values import StrictModel


def _trimmed_nonblank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("name must not be blank")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("name must contain valid Unicode") from error
    return normalized


class CreateAgentRequest(StrictModel):
    name: Annotated[str, AfterValidator(_trimmed_nonblank)]


class AgentResource(StrictModel):
    agent_id: UUID
    name: str


class AgentListQuery(StrictModel):
    limit: Annotated[int, Field(ge=1, le=100)] = 100
    cursor: str | None = None


__all__ = [
    "AgentListQuery",
    "AgentResource",
    "CreateAgentRequest",
]
