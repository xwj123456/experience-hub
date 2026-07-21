"""Canonical UTF-8 JSON and digest helpers."""

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from experience_hub.errors import CanonicalizationError


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanonicalizationError(
            "Canonical datetimes must be timezone-aware UTC values"
        )
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc_timestamp(value)
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("Canonical JSON numbers must be finite")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("Canonical JSON object keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise CanonicalizationError(
        f"Unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize supported values as stable, compact, Unicode-preserving JSON."""
    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for bytes."""
    return sha256(value).hexdigest()
