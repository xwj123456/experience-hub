"""Projection application boundary shared by storage writers."""

from collections.abc import Sequence
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain.events import StoredEvent


class ProjectionApplier(Protocol):
    async def apply(
        self,
        *,
        session: AsyncSession,
        events: Sequence[StoredEvent],
    ) -> None:
        raise RuntimeError(
            "ProjectionApplier is an interface and cannot be called directly"
        )


class ProjectionReducer(Protocol):
    name: str
    version: int
    event_types: frozenset[str]

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        raise RuntimeError(
            "ProjectionReducer is an interface and cannot be called directly"
        )

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        raise RuntimeError(
            "ProjectionReducer is an interface and cannot be called directly"
        )


class NullProjectionApplier:
    async def apply(
        self,
        *,
        session: AsyncSession,
        events: Sequence[StoredEvent],
    ) -> None:
        _ = (session, events)
        return None
