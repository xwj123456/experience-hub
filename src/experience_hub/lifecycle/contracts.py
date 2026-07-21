"""Strict lifecycle result and optional inspiration archive contracts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Protocol, Self
from uuid import UUID

from pydantic import ConfigDict, ValidationError, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import PendingEvent, StrictModel
from experience_hub.errors import CanonicalizationError


class LifecycleResult(StrictModel):
    """The one canonical success value shared by every lifecycle adapter."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    cycle_id: UUID
    evaluated_at: datetime
    evaluated_count: int
    transition_count: int
    archive_count: int
    idea_archive_count: int

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def normalize_evaluated_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator(
        "evaluated_count",
        "transition_count",
        "archive_count",
        "idea_archive_count",
        mode="before",
    )
    @classmethod
    def validate_count(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Lifecycle counts must be non-negative integers")
        return value

    @model_validator(mode="after")
    def validate_count_relationships(self) -> Self:
        if self.transition_count > self.evaluated_count:
            raise ValueError(
                "transition_count cannot exceed evaluated_count"
            )
        if self.archive_count > self.transition_count:
            raise ValueError("archive_count cannot exceed transition_count")
        return self


def encode_lifecycle_result(result: LifecycleResult) -> bytes:
    """Encode a lifecycle result in its canonical success envelope."""
    if not isinstance(result, LifecycleResult):
        raise ValueError("result must be a LifecycleResult")
    return canonical_json_bytes({"data": result})


def decode_lifecycle_result(body: bytes) -> LifecycleResult:
    """Decode only the exact canonical lifecycle success envelope."""
    if not isinstance(body, bytes):
        raise ValueError("Lifecycle result body must be bytes")
    try:
        decoded: Any = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Lifecycle result body must be valid JSON") from error
    try:
        canonical = canonical_json_bytes(decoded)
    except CanonicalizationError as error:
        raise ValueError("Lifecycle result body is not canonical") from error
    if canonical != body:
        raise ValueError("Lifecycle result body must use canonical JSON")
    if not isinstance(decoded, dict) or set(decoded) != {"data"}:
        raise ValueError("Lifecycle result body must contain only data")
    if not isinstance(decoded["data"], dict):
        raise ValueError("Lifecycle result data must be an object")
    try:
        result = LifecycleResult.model_validate_json(
            canonical_json_bytes(decoded["data"])
        )
    except ValidationError as error:
        raise ValueError("Lifecycle result data is invalid") from error
    if encode_lifecycle_result(result) != body:
        raise ValueError("Lifecycle result body is not canonical")
    return result


class IdeaArchivePlanner(Protocol):
    """Optional Plan 4 hook invoked after experience lifecycle events."""

    async def due_archive_events(
        self,
        *,
        session: AsyncSession,
        evaluated_at: datetime,
        cycle_id: UUID,
    ) -> tuple[PendingEvent, ...]: ...


class NullIdeaArchivePlanner:
    """Plan 2 default that has no inspiration tables or archive work."""

    async def due_archive_events(
        self,
        *,
        session: AsyncSession,
        evaluated_at: datetime,
        cycle_id: UUID,
    ) -> tuple[PendingEvent, ...]:
        _ = session
        if not isinstance(evaluated_at, datetime):
            raise ValueError(
                "evaluated_at must be a timezone-aware datetime"
            )
        require_utc(evaluated_at)
        if not isinstance(cycle_id, UUID):
            raise ValueError("cycle_id must be a UUID")
        return ()


__all__ = [
    "IdeaArchivePlanner",
    "LifecycleResult",
    "NullIdeaArchivePlanner",
    "decode_lifecycle_result",
    "encode_lifecycle_result",
]
