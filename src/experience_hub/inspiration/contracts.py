"""Read-only evidence ports consumed by the snapshot boundary."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.retrieval.contracts import PeekExperiences, SearchResult
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue
from experience_hub.sharing.queries import QuarantinedCapsuleEvidence


class ExperienceEvidenceReader(Protocol):
    """Return owned evidence using only the caller's database session."""

    async def peek(
        self,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult: ...


class InboxEvidenceReader(Protocol):
    """Return explicitly opted-in quarantine evidence without side effects."""

    async def list_available_pending(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        as_of: datetime,
        query_cues: Iterable[TermCue],
        mode: RetrievalMode,
        limit: int,
    ) -> tuple[QuarantinedCapsuleEvidence, ...]: ...


__all__ = [
    "ExperienceEvidenceReader",
    "InboxEvidenceReader",
]
