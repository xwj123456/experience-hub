"""Injectable UTC clocks for deterministic domain behavior."""

from datetime import UTC, datetime, timedelta
from typing import Protocol


def require_utc(value: datetime) -> datetime:
    """Require an aware datetime and normalize it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime must be timezone-aware")
    return value.astimezone(UTC)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, current: datetime) -> None:
        self._current = require_utc(current)

    def now(self) -> datetime:
        return self._current

    def advance(self, delta: timedelta) -> datetime:
        self._current = require_utc(self._current + delta)
        return self._current
