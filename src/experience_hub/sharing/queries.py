"""Owner-scoped quarantine views and side-effect-free inbox evidence reads."""

from __future__ import annotations

import base64
import binascii
import heapq
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.errors import CanonicalizationError
from experience_hub.retrieval.ranking import (
    RetrievalMode,
    passes_relevance_threshold,
    relevance_components,
)
from experience_hub.retrieval.tokenizer import TermCue, index_version_terms
from experience_hub.sharing.models import (
    CapsuleStatus,
    EffectiveAvailability,
    InboxItem,
    InboxState,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.tables import InboxItemRow

_MAX_INBOX_PAGE_SIZE = 100
_MAX_EVIDENCE_LIMIT = 50
_MAX_CURSOR_CHARACTERS = 8_192
_QUARANTINED_SOURCE_TRUST = 0.25
_UNPADDED_BASE64URL = re.compile(r"[A-Za-z0-9_-]+\Z")


class InvalidInboxCursor(ValueError):
    """The opaque inbox cursor is malformed or bound to another context."""


@dataclass(frozen=True, slots=True)
class InboxPage:
    """One stable keyset page of owner-visible quarantined inbox items."""

    items: tuple[InboxItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class QuarantinedCapsuleEvidence:
    """Ranked capsule evidence suitable only for an explicit inspiration opt-in."""

    item_id: UUID
    capsule_id: UUID
    publisher_agent_id: UUID
    source_type: Literal["capsule"]
    source_id: UUID
    source_version_id: UUID
    source_state: Literal["quarantined"]
    content_hash: str
    summary: str
    mechanism: str
    applicability: tuple[str, ...]
    tags: tuple[str, ...]
    falsifiers: tuple[str, ...]
    excerpt: str
    ranking_relevance: float
    source_trust: float

    def __post_init__(self) -> None:
        if (
            self.source_type != "capsule"
            or self.source_state != "quarantined"
            or self.source_id != self.capsule_id
        ):
            raise ValueError("quarantined capsule evidence identity is invalid")
        if (
            not math.isfinite(self.ranking_relevance)
            or not 0.0 <= self.ranking_relevance <= 1.0
        ):
            raise ValueError("ranking_relevance must be between zero and one")
        if self.source_trust != _QUARANTINED_SOURCE_TRUST:
            raise ValueError("quarantined capsule source trust must be 0.25")


def _limit(value: int, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"limit must be an integer between 1 and {maximum}")
    return value


def _availability(
    *,
    status: CapsuleStatus,
    expires_at: datetime,
    at: datetime,
) -> EffectiveAvailability:
    if status is CapsuleStatus.RETRACTED:
        return EffectiveAvailability.RETRACTED
    if at >= expires_at:
        return EffectiveAvailability.EXPIRED
    return EffectiveAvailability.ACTIVE


def _encode_cursor(
    *,
    owner_agent_id: UUID,
    state: InboxState | None,
    item_id: UUID,
) -> str:
    encoded = base64.urlsafe_b64encode(
        canonical_json_bytes(
            {
                "v": 1,
                "owner_agent_id": owner_agent_id,
                "state": None if state is None else state.value,
                "item_id": item_id,
            }
        )
    ).decode("ascii")
    return encoded.rstrip("=")


def _decode_cursor(
    value: str | None,
    *,
    owner_agent_id: UUID,
    state: InboxState | None,
) -> UUID | None:
    if value is None:
        return None
    try:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_CURSOR_CHARACTERS
            or _UNPADDED_BASE64URL.fullmatch(value) is None
        ):
            raise ValueError("cursor token shape is invalid")
        padded = value + "=" * (-len(value) % 4)
        raw = base64.b64decode(
            padded,
            altchars=b"-_",
            validate=True,
        )
        decoded = json.loads(raw)
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"v", "owner_agent_id", "state", "item_id"}
            or type(decoded["v"]) is not int
            or decoded["v"] != 1
            or not isinstance(decoded["owner_agent_id"], str)
            or not isinstance(decoded["item_id"], str)
        ):
            raise ValueError("cursor context does not match")
        expected_state = None if state is None else state.value
        if expected_state is None:
            if decoded["state"] is not None:
                raise ValueError("cursor context does not match")
        elif (
            not isinstance(decoded["state"], str) or decoded["state"] != expected_state
        ):
            raise ValueError("cursor context does not match")
        if UUID(decoded["owner_agent_id"]) != owner_agent_id:
            raise ValueError("cursor context does not match")
        item_id = UUID(decoded["item_id"])
        if canonical_json_bytes(decoded) != raw:
            raise ValueError("cursor is not canonical")
        if (
            _encode_cursor(
                owner_agent_id=owner_agent_id,
                state=state,
                item_id=item_id,
            )
            != value
        ):
            raise ValueError("cursor is not canonical")
        return item_id
    except (
        binascii.Error,
        CanonicalizationError,
        RecursionError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise InvalidInboxCursor("cursor is invalid") from error


class SharingQuery:
    """Return full capsule content only through an owner-scoped inbox view."""

    def __init__(self, *, repository: SharingRepository | None = None) -> None:
        self._repository = repository or SharingRepository()

    async def list_inbox(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        state: InboxState | None = None,
        cursor: str | None = None,
        limit: int = 100,
        at: datetime,
    ) -> InboxPage:
        page_size = _limit(limit, maximum=_MAX_INBOX_PAGE_SIZE)
        observed_at = require_utc(at)
        if state is not None and not isinstance(state, InboxState):
            raise ValueError("state must be an InboxState or None")
        after_item_id = _decode_cursor(
            cursor,
            owner_agent_id=owner_agent_id,
            state=state,
        )
        statement = select(InboxItemRow.item_id).where(
            InboxItemRow.recipient_agent_id == owner_agent_id
        )
        if state is not None:
            statement = statement.where(InboxItemRow.state == state)
        if after_item_id is not None:
            statement = statement.where(InboxItemRow.item_id > after_item_id)
        item_ids = tuple(
            (
                await session.scalars(
                    statement.order_by(InboxItemRow.item_id).limit(page_size + 1)
                )
            ).all()
        )
        selected_ids = item_ids[:page_size]
        items: list[InboxItem] = []
        for item_id in selected_ids:
            source = await self._repository.get_owned_inbox_item(
                session=session,
                recipient_agent_id=owner_agent_id,
                item_id=item_id,
            )
            if source is None:
                raise RuntimeError("Owner-scoped inbox row disappeared during read")
            capsule = source.capsule
            items.append(
                InboxItem(
                    item_id=source.item_id,
                    recipient_agent_id=source.recipient_agent_id,
                    capsule_id=capsule.capsule_id,
                    capsule=capsule,
                    state=source.state,
                    effective_availability=_availability(
                        status=capsule.status,
                        expires_at=capsule.expires_at,
                        at=observed_at,
                    ),
                )
            )
        next_cursor = (
            None
            if len(item_ids) <= page_size or not selected_ids
            else _encode_cursor(
                owner_agent_id=owner_agent_id,
                state=state,
                item_id=selected_ids[-1],
            )
        )
        return InboxPage(items=tuple(items), next_cursor=next_cursor)


class InboxEvidenceReader:
    """Read relevant, available pending capsules without mutating retrieval."""

    def __init__(self, *, repository: SharingRepository | None = None) -> None:
        self._repository = repository or SharingRepository()

    async def list_available_pending(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        as_of: datetime,
        query_cues: Iterable[TermCue],
        mode: RetrievalMode,
        limit: int,
    ) -> tuple[QuarantinedCapsuleEvidence, ...]:
        observed_at = require_utc(as_of)
        requested = _limit(limit, maximum=_MAX_EVIDENCE_LIMIT)
        cues = tuple(query_cues)
        if any(not isinstance(cue, TermCue) for cue in cues):
            raise ValueError("query_cues must contain only TermCue values")
        if not isinstance(mode, RetrievalMode):
            raise ValueError("mode must be a RetrievalMode")
        candidates: list[tuple[float, int, QuarantinedCapsuleEvidence]] = []
        async for source in self._repository.stream_available_pending(
            session=session,
            recipient_agent_id=recipient_agent_id,
            as_of=observed_at,
        ):
            capsule = source.capsule
            relevance = relevance_components(
                cues,
                index_version_terms(capsule),
                mode,
            )
            if not passes_relevance_threshold(
                mode=mode,
                lexical_or_trigram_relevance=(relevance.lexical_or_trigram_relevance),
                mechanism_relevance=relevance.mechanism_relevance,
            ):
                continue
            candidate = QuarantinedCapsuleEvidence(
                item_id=source.item_id,
                capsule_id=capsule.capsule_id,
                publisher_agent_id=capsule.publisher_agent_id,
                source_type="capsule",
                source_id=capsule.capsule_id,
                source_version_id=capsule.source_version_id,
                source_state="quarantined",
                content_hash=capsule.source_content_hash,
                summary=capsule.summary,
                mechanism=capsule.mechanism,
                applicability=capsule.applicability,
                tags=capsule.tags,
                falsifiers=capsule.falsifiers,
                excerpt=capsule.body,
                ranking_relevance=relevance.ranking_relevance,
                source_trust=_QUARANTINED_SOURCE_TRUST,
            )
            entry = (
                candidate.ranking_relevance,
                -candidate.capsule_id.int,
                candidate,
            )
            if len(candidates) < requested:
                heapq.heappush(candidates, entry)
            elif entry[:2] > candidates[0][:2]:
                heapq.heapreplace(candidates, entry)
        selected = [entry[2] for entry in candidates]
        selected.sort(
            key=lambda candidate: (
                -candidate.ranking_relevance,
                candidate.capsule_id.int,
            )
        )
        return tuple(selected)


__all__ = [
    "InboxEvidenceReader",
    "InboxPage",
    "QuarantinedCapsuleEvidence",
    "SharingQuery",
]
