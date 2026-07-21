"""Strict HTTP request contract for manual lifecycle execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import field_validator

from experience_hub.clock import require_utc
from experience_hub.domain.values import StrictModel


class RunLifecycleRequest(StrictModel):
    evaluated_at: datetime | None = None

    @field_validator("evaluated_at", mode="before")
    @classmethod
    def require_timestamp_shape(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, datetime)):
            return value
        raise ValueError("evaluated_at must be an RFC 3339 timestamp")

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def normalize_evaluated_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else require_utc(value)


__all__ = ["RunLifecycleRequest"]
