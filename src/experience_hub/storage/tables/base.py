"""Declarative base and SQLite-safe value adapters."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import LargeBinary, String
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc


class Base(DeclarativeBase):
    pass


class UUIDString(TypeDecorator[UUID]):
    """Persist UUIDs as portable lowercase strings."""

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: UUID | None, dialect: Dialect) -> str | None:
        _ = dialect
        return None if value is None else str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> UUID | None:
        _ = dialect
        return None if value is None else UUID(value)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist aware datetimes as fixed-format UTC text."""

    impl = String(27)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> str | None:
        _ = dialect
        if value is None:
            return None
        normalized = require_utc(value)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> datetime | None:
        _ = dialect
        if value is None:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class CanonicalJSONBytes(TypeDecorator[bytes]):
    """Store canonical JSON as binary while preserving its exact bytes."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(
        self,
        value: bytes | None,
        dialect: Dialect,
    ) -> bytes | None:
        _ = dialect
        if value is None:
            return None
        encoded = bytes(value)
        decoded: Any = json.loads(encoded)
        if canonical_json_bytes(decoded) != encoded:
            raise ValueError("JSON bytes must use the canonical encoding")
        return encoded

    def process_result_value(
        self,
        value: bytes | None,
        dialect: Dialect,
    ) -> bytes | None:
        _ = dialect
        return None if value is None else bytes(value)
