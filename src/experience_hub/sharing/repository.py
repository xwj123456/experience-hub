"""Transaction-bound persistence and deterministic sharing source queries."""

from __future__ import annotations

import json
import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select, tuple_
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import (
    EventRegistry,
    StoredEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences import VersionContent, encode_version_content
from experience_hub.sharing.confidence import observer_trust
from experience_hub.sharing.events import (
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    register_sharing_events,
)
from experience_hub.sharing.hashing import (
    compute_capsule_hash,
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import (
    MAX_PROVENANCE_HOPS,
    Capsule,
    CapsuleStatus,
    FeedbackRevision,
    FeedbackVerdict,
    InboxState,
    ProvenanceHop,
    Reputation,
    Subscription,
    Topic,
)
from experience_hub.sharing.projector import (
    SharingProjectionIntegrityError,
    replay_reputation,
    validate_feedback_source_event,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    AgentReputationRow,
    AgentRow,
    CapsuleFeedbackRow,
    CapsuleStateRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    InboxItemRow,
    SubscriptionRow,
    TopicRow,
)
from experience_hub.storage.validation import SourceIntegrityError

_MAX_PAGE_SIZE = 100
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ParentAdoption:
    """One owner-visible provenance bridge safe for capsule republishing."""

    adoption_id: UUID
    adopter_agent_id: UUID
    capsule_id: UUID
    resulting_experience_id: UUID
    provenance_chain: tuple[ProvenanceHop, ...]
    root_fingerprint: str
    adopted_at: datetime


@dataclass(frozen=True, slots=True)
class AdoptionInboxSource:
    """One owner-scoped inbox route with validated immutable capsule content."""

    item_id: UUID
    recipient_agent_id: UUID
    state: InboxState
    capsule: Capsule
    inbox_state_at: datetime
    capsule_state_at: datetime
    adopted_event: CapsuleAdoptedV1 | None

    @property
    def capsule_id(self) -> UUID:
        return self.capsule.capsule_id

    @property
    def latest_causal_at(self) -> datetime:
        return max(
            self.capsule.created_at,
            self.capsule.last_transition_at,
            self.inbox_state_at,
            self.capsule_state_at,
        )


@dataclass(frozen=True, slots=True)
class StoredAdoption:
    """One immutable provenance row decoded for command replay."""

    adoption_id: UUID
    adopter_agent_id: UUID
    capsule_id: UUID
    resulting_experience_id: UUID
    captured_trust: float
    provenance_chain: tuple[ProvenanceHop, ...]
    root_fingerprint: str
    corroboration_applied: bool
    adopted_at: datetime


def _page_size(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_PAGE_SIZE
    ):
        raise ValueError(f"limit must be an integer between 1 and {_MAX_PAGE_SIZE}")
    return limit


class SharingRepository:
    """Store immutable sharing rows and expose stable keyset scans."""

    def __init__(self, *, event_registry: EventRegistry | None = None) -> None:
        if event_registry is None:
            event_registry = EventRegistry()
            register_sharing_events(event_registry)
        self._event_registry = event_registry

    @staticmethod
    async def agent_exists(
        *,
        session: AsyncSession,
        agent_id: UUID,
    ) -> bool:
        return (
            await session.scalar(
                select(AgentRow.agent_id).where(AgentRow.agent_id == agent_id)
            )
            is not None
        )

    @staticmethod
    async def find_topic_by_name(
        *,
        session: AsyncSession,
        name: str,
    ) -> Topic | None:
        row = await session.scalar(select(TopicRow).where(TopicRow.name == name))
        return None if row is None else _topic(row)

    @staticmethod
    async def find_subscription(
        *,
        session: AsyncSession,
        subscriber_agent_id: UUID,
        topic_id: UUID,
    ) -> Subscription | None:
        row = await session.scalar(
            select(SubscriptionRow).where(
                SubscriptionRow.subscriber_agent_id == subscriber_agent_id,
                SubscriptionRow.topic_id == topic_id,
            )
        )
        return None if row is None else _subscription(row)

    @staticmethod
    async def get_topic(
        *,
        session: AsyncSession,
        topic_id: UUID,
    ) -> Topic | None:
        row = await session.get(TopicRow, topic_id)
        return None if row is None else _topic(row)

    @staticmethod
    async def get_owned_topic(
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        topic_id: UUID,
    ) -> Topic | None:
        row = await session.scalar(
            select(TopicRow).where(
                TopicRow.topic_id == topic_id,
                TopicRow.owner_agent_id == owner_agent_id,
            )
        )
        return None if row is None else _topic(row)

    @staticmethod
    async def get_subscription(
        *,
        session: AsyncSession,
        subscription_id: UUID,
    ) -> Subscription | None:
        row = await session.get(SubscriptionRow, subscription_id)
        return None if row is None else _subscription(row)

    async def _validated_capsule(
        self,
        *,
        session: AsyncSession,
        capsule_id: UUID,
        require_transition_event: bool = True,
    ) -> Capsule | None:
        """Decode one capsule and validate publication plus current checkpoint."""
        source = await session.get(ExperienceCapsuleRow, capsule_id)
        if source is None:
            return None
        try:
            state = await session.get(CapsuleStateRow, capsule_id)
            publication_row = await session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.aggregate_type == "capsule",
                    DomainEventRow.aggregate_id == capsule_id,
                    DomainEventRow.event_type == CapsulePublishedV1.event_type,
                    DomainEventRow.sequence == 1,
                )
            )
            current_row = (
                None
                if state is None
                else await session.get(
                    DomainEventRow,
                    state.projection_event_id,
                )
            )
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Capsule {capsule_id} has invalid source anchors",
                mismatch_key=f"capsule:{capsule_id}",
            ) from error
        if state is None or publication_row is None or current_row is None:
            raise SourceIntegrityError(
                f"Capsule {capsule_id} has incomplete source anchors",
                mismatch_key=f"capsule:{capsule_id}",
            )
        try:
            published = self._event_registry.decode(
                event_type=publication_row.event_type,
                payload=publication_row.payload,
            )
            current = self._event_registry.decode(
                event_type=current_row.event_type,
                payload=current_row.payload,
            )
            capsule = _capsule(
                source,
                status=state.status,
                last_transition_at=current_row.occurred_at,
            )
            _validate_capsule_hashes(capsule)
            if (
                not isinstance(published, CapsulePublishedV1)
                or publication_row.aggregate_type != "capsule"
                or publication_row.aggregate_id != capsule.capsule_id
                or publication_row.sequence != 1
                or publication_row.occurred_at != capsule.created_at
                or publication_row.actor_agent_id != capsule.publisher_agent_id
                or published.capsule_id != capsule.capsule_id
                or published.topic_id != capsule.topic_id
                or published.source_experience_id
                != capsule.source_experience_id
                or published.source_version_id != capsule.source_version_id
                or published.publisher_agent_id != capsule.publisher_agent_id
                or published.capsule_hash != capsule.capsule_hash
                or published.root_fingerprint != capsule.root_fingerprint
                or published.status_after is not CapsuleStatus.ACTIVE
            ):
                raise ValueError("capsule publication source is inconsistent")
            if state.status is CapsuleStatus.ACTIVE:
                if (
                    current_row.event_id != publication_row.event_id
                    or not isinstance(current, CapsulePublishedV1)
                ):
                    raise ValueError("active capsule checkpoint is inconsistent")
            elif state.status is CapsuleStatus.RETRACTED:
                invalid_retraction = (
                    not isinstance(current, CapsuleRetractedV1)
                    or current_row.aggregate_type != "capsule"
                    or current_row.aggregate_id != capsule.capsule_id
                    or current_row.sequence < 2
                    or current_row.actor_agent_id != capsule.publisher_agent_id
                    or current_row.occurred_at < capsule.created_at
                    or current.capsule_id != capsule.capsule_id
                    or current.publisher_agent_id != capsule.publisher_agent_id
                    or current.status_before is not CapsuleStatus.ACTIVE
                    or current.status_after is not CapsuleStatus.RETRACTED
                )
                projection_only_unavailable = (
                    not require_transition_event
                    and current_row.event_id == publication_row.event_id
                    and isinstance(current, CapsulePublishedV1)
                )
                if invalid_retraction and not projection_only_unavailable:
                    raise ValueError("retracted capsule checkpoint is inconsistent")
            else:
                raise ValueError("capsule status is invalid")
        except (TypeError, ValueError, ValidationError) as error:
            raise SourceIntegrityError(
                f"Capsule {capsule_id} failed semantic validation",
                mismatch_key=f"capsule:{capsule_id}",
            ) from error
        return capsule

    async def get_owned_capsule(
        self,
        *,
        session: AsyncSession,
        publisher_agent_id: UUID,
        capsule_id: UUID,
    ) -> Capsule | None:
        visible = await session.scalar(
            select(ExperienceCapsuleRow.capsule_id).where(
                ExperienceCapsuleRow.capsule_id == capsule_id,
                ExperienceCapsuleRow.publisher_agent_id == publisher_agent_id,
            )
        )
        if visible is None:
            return None
        return await self._validated_capsule(
            session=session,
            capsule_id=capsule_id,
        )

    @staticmethod
    async def latest_capsule_causal_at(
        *,
        session: AsyncSession,
        capsule_id: UUID,
    ) -> datetime:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == capsule_id,
            )
            .order_by(DomainEventRow.sequence.desc())
            .limit(1)
        )
        if row is None:
            raise SourceIntegrityError(
                f"Capsule {capsule_id} has no causal event",
                mismatch_key=f"capsule:{capsule_id}",
            )
        try:
            return require_utc(row.occurred_at)
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Capsule {capsule_id} has an invalid causal clock",
                mismatch_key=f"capsule:{capsule_id}",
            ) from error

    @staticmethod
    async def get_owned_parent_adoption(
        *,
        session: AsyncSession,
        adopter_agent_id: UUID,
        adoption_id: UUID,
    ) -> ParentAdoption | None:
        """Read one caller-owned adoption and verify its complete parent chain."""
        adoption = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.adoption_id == adoption_id,
                AdoptionRecordRow.adopter_agent_id == adopter_agent_id,
            )
        )
        if adoption is None:
            return None
        parent = await session.get(ExperienceCapsuleRow, adoption.capsule_id)
        if parent is None:
            raise SourceIntegrityError(
                f"Parent adoption {adoption_id} has no capsule source",
                mismatch_key=f"parent_adoption:{adoption_id}",
            )
        try:
            parent_chain = _decode_provenance(
                parent.provenance_chain,
                label=f"capsule {parent.capsule_id}",
                allow_empty=True,
            )
            adoption_chain = _decode_provenance(
                adoption.provenance_chain,
                label=f"adoption {adoption.adoption_id}",
                allow_empty=False,
            )
            if (
                isinstance(parent.hop_count, bool)
                or not isinstance(parent.hop_count, int)
                or not 0 <= parent.hop_count <= MAX_PROVENANCE_HOPS
                or parent.hop_count != len(parent_chain)
            ):
                raise ValueError(
                    "parent capsule hop count does not match its provenance"
                )
            if parent.capsule_id in {hop.capsule_id for hop in parent_chain}:
                raise ValueError("parent capsule occurs in its own provenance chain")
            expected_last = ProvenanceHop(
                capsule_id=parent.capsule_id,
                publisher_agent_id=parent.publisher_agent_id,
            )
            if adoption_chain != (*parent_chain, expected_last):
                raise ValueError(
                    "adoption provenance does not extend its parent capsule"
                )
            if (
                not isinstance(adoption.root_fingerprint, str)
                or not _SHA256_HEX.fullmatch(adoption.root_fingerprint)
                or not isinstance(parent.root_fingerprint, str)
                or not _SHA256_HEX.fullmatch(parent.root_fingerprint)
                or adoption.root_fingerprint != parent.root_fingerprint
            ):
                raise ValueError(
                    "adoption root fingerprint is invalid or differs "
                    "from its parent capsule"
                )
            adopted_at = require_utc(adoption.adopted_at)
            if adopted_at < require_utc(parent.created_at):
                raise ValueError("adoption time precedes its parent capsule creation")
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Parent adoption {adoption_id} has invalid provenance",
                mismatch_key=f"parent_adoption:{adoption_id}",
            ) from error
        return ParentAdoption(
            adoption_id=adoption.adoption_id,
            adopter_agent_id=adoption.adopter_agent_id,
            capsule_id=adoption.capsule_id,
            resulting_experience_id=adoption.resulting_experience_id,
            provenance_chain=adoption_chain,
            root_fingerprint=adoption.root_fingerprint,
            adopted_at=adopted_at,
        )

    async def get_owned_adoption_inbox(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        item_id: UUID,
        _require_transition_event: bool = False,
    ) -> AdoptionInboxSource | None:
        """Load an owner-visible inbox route and fail closed on source drift."""
        item = await session.scalar(
            select(InboxItemRow).where(
                InboxItemRow.item_id == item_id,
                InboxItemRow.recipient_agent_id == recipient_agent_id,
            )
        )
        if item is None:
            return None
        capsule = await self._validated_capsule(
            session=session,
            capsule_id=item.capsule_id,
            require_transition_event=_require_transition_event,
        )
        try:
            inbox_event = await session.get(
                DomainEventRow,
                item.projection_event_id,
            )
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Inbox item {item_id} has invalid source anchors",
                mismatch_key=f"inbox_item:{item_id}",
            ) from error
        if capsule is None or inbox_event is None:
            raise SourceIntegrityError(
                f"Inbox item {item_id} has incomplete source anchors",
                mismatch_key=f"inbox_item:{item_id}",
            )
        try:
            inbox_payload = self._event_registry.decode(
                event_type=inbox_event.event_type,
                payload=inbox_event.payload,
            )
            adopted_event: CapsuleAdoptedV1 | None = None
            if item.state is InboxState.PENDING:
                if (
                    not isinstance(inbox_payload, CapsuleReceivedV1)
                    or inbox_event.sequence != 1
                    or inbox_payload.item_id != item.item_id
                    or inbox_payload.capsule_id != item.capsule_id
                    or inbox_payload.recipient_agent_id
                    != item.recipient_agent_id
                    or inbox_payload.state_after is not InboxState.PENDING
                    or inbox_event.actor_agent_id
                    != capsule.publisher_agent_id
                ):
                    raise ValueError("pending inbox checkpoint is inconsistent")
            elif item.state is InboxState.ADOPTED:
                if (
                    not isinstance(inbox_payload, CapsuleAdoptedV1)
                    or inbox_event.sequence != 2
                    or inbox_payload.item_id != item.item_id
                    or inbox_payload.capsule_id != item.capsule_id
                    or inbox_payload.adopter_agent_id
                    != item.recipient_agent_id
                    or inbox_payload.state_after is not InboxState.ADOPTED
                    or inbox_event.actor_agent_id
                    != item.recipient_agent_id
                ):
                    raise ValueError("adopted inbox checkpoint is inconsistent")
                adopted_event = inbox_payload
            elif item.state is InboxState.REJECTED:
                valid_rejection = (
                    isinstance(inbox_payload, CapsuleRejectedV1)
                    and inbox_event.sequence == 2
                    and inbox_payload.item_id == item.item_id
                    and inbox_payload.capsule_id == item.capsule_id
                    and inbox_payload.recipient_agent_id
                    == item.recipient_agent_id
                    and inbox_payload.state_before is InboxState.PENDING
                    and inbox_payload.state_after is InboxState.REJECTED
                    and inbox_event.actor_agent_id
                    == item.recipient_agent_id
                )
                projection_only_terminal = (
                    not _require_transition_event
                    and isinstance(inbox_payload, CapsuleReceivedV1)
                    and inbox_event.sequence == 1
                    and inbox_payload.item_id == item.item_id
                    and inbox_payload.capsule_id == item.capsule_id
                    and inbox_payload.recipient_agent_id
                    == item.recipient_agent_id
                    and inbox_payload.state_after is InboxState.PENDING
                    and inbox_event.actor_agent_id
                    == capsule.publisher_agent_id
                )
                if not valid_rejection and not projection_only_terminal:
                    raise ValueError("rejected inbox checkpoint is inconsistent")
                adopted_event = None
            else:
                raise ValueError("inbox state is invalid")
            if (
                inbox_event.aggregate_type != "inbox_item"
                or inbox_event.aggregate_id != item.item_id
                or inbox_event.occurred_at < capsule.created_at
            ):
                raise ValueError("inbox checkpoint event is inconsistent")
        except (
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise SourceIntegrityError(
                f"Inbox item {item_id} failed semantic validation",
                mismatch_key=f"inbox_item:{item_id}",
            ) from error
        return AdoptionInboxSource(
            item_id=item.item_id,
            recipient_agent_id=item.recipient_agent_id,
            state=item.state,
            capsule=capsule,
            inbox_state_at=require_utc(inbox_event.occurred_at),
            capsule_state_at=capsule.last_transition_at,
            adopted_event=adopted_event,
        )

    async def get_adoption_for_capsule(
        self,
        *,
        session: AsyncSession,
        adopter_agent_id: UUID,
        capsule: Capsule,
    ) -> StoredAdoption | None:
        row = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.adopter_agent_id == adopter_agent_id,
                AdoptionRecordRow.capsule_id == capsule.capsule_id,
            )
        )
        if row is None:
            return None
        try:
            chain = _decode_provenance(
                row.provenance_chain,
                label=f"adoption {row.adoption_id}",
                allow_empty=False,
            )
            expected_chain = (
                *capsule.provenance_chain,
                ProvenanceHop(
                    capsule_id=capsule.capsule_id,
                    publisher_agent_id=capsule.publisher_agent_id,
                ),
            )
            captured_trust = float(row.captured_trust)
            adopted_at = require_utc(row.adopted_at)
            if (
                row.adopter_agent_id != adopter_agent_id
                or chain != expected_chain
                or row.root_fingerprint != capsule.root_fingerprint
                or not math.isfinite(captured_trust)
                or not 0.0 <= captured_trust <= 1.0
                or not isinstance(row.corroboration_applied, bool)
                or adopted_at < capsule.created_at
            ):
                raise ValueError("adoption row is inconsistent")
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Adoption {row.adoption_id} failed semantic validation",
                mismatch_key=f"adoption:{row.adoption_id}",
            ) from error
        return StoredAdoption(
            adoption_id=row.adoption_id,
            adopter_agent_id=row.adopter_agent_id,
            capsule_id=row.capsule_id,
            resulting_experience_id=row.resulting_experience_id,
            captured_trust=captured_trust,
            provenance_chain=chain,
            root_fingerprint=row.root_fingerprint,
            corroboration_applied=row.corroboration_applied,
            adopted_at=adopted_at,
        )

    async def get_owned_inbox_item(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        item_id: UUID,
    ) -> AdoptionInboxSource | None:
        return await self.get_owned_adoption_inbox(
            session=session,
            recipient_agent_id=recipient_agent_id,
            item_id=item_id,
            _require_transition_event=True,
        )

    async def get_owned_adoption(
        self,
        *,
        session: AsyncSession,
        adopter_agent_id: UUID,
        adoption_id: UUID,
    ) -> StoredAdoption | None:
        row = await session.scalar(
            select(AdoptionRecordRow).where(
                AdoptionRecordRow.adoption_id == adoption_id,
                AdoptionRecordRow.adopter_agent_id == adopter_agent_id,
            )
        )
        if row is None:
            return None
        capsule = await self._validated_capsule(
            session=session,
            capsule_id=row.capsule_id,
        )
        if capsule is None:
            raise SourceIntegrityError(
                f"Adoption {adoption_id} has no capsule source",
                mismatch_key=f"adoption:{adoption_id}",
            )
        adoption = await self.get_adoption_for_capsule(
            session=session,
            adopter_agent_id=adopter_agent_id,
            capsule=capsule,
        )
        if adoption is None or adoption.adoption_id != adoption_id:
            raise SourceIntegrityError(
                f"Adoption {adoption_id} is inconsistent",
                mismatch_key=f"adoption:{adoption_id}",
            )
        return adoption

    async def get_feedback_authorized_capsule(
        self,
        *,
        session: AsyncSession,
        observer_agent_id: UUID,
        capsule_id: UUID,
    ) -> AdoptionInboxSource | None:
        item_id = await session.scalar(
            select(InboxItemRow.item_id).where(
                InboxItemRow.recipient_agent_id == observer_agent_id,
                InboxItemRow.capsule_id == capsule_id,
                InboxItemRow.state.in_(
                    (InboxState.ADOPTED, InboxState.REJECTED)
                ),
            )
        )
        if item_id is None:
            return None
        source = await self.get_owned_inbox_item(
            session=session,
            recipient_agent_id=observer_agent_id,
            item_id=item_id,
        )
        if (
            source is None
            or source.state not in (InboxState.ADOPTED, InboxState.REJECTED)
            or source.capsule.publisher_agent_id == observer_agent_id
        ):
            return None
        return source

    @staticmethod
    async def get_latest_feedback(
        *,
        session: AsyncSession,
        observer_agent_id: UUID,
        capsule_id: UUID,
    ) -> FeedbackRevision | None:
        """Read and validate the contiguous immutable revision stream."""
        rows = (
            await session.scalars(
                select(CapsuleFeedbackRow)
                .where(
                    CapsuleFeedbackRow.observer_agent_id
                    == observer_agent_id,
                    CapsuleFeedbackRow.capsule_id == capsule_id,
                )
                .order_by(CapsuleFeedbackRow.revision)
            )
        ).all()
        if not rows:
            return None
        try:
            revisions = tuple(_feedback_revision(row) for row in rows)
            if tuple(item.revision for item in revisions) != tuple(
                range(1, len(revisions) + 1)
            ):
                raise ValueError("feedback revisions are not contiguous")
            if any(
                item.observer_agent_id != observer_agent_id
                or item.capsule_id != capsule_id
                for item in revisions
            ):
                raise ValueError("feedback revision identity changed")
            if any(
                later.created_at < earlier.created_at
                for earlier, later in zip(
                    revisions,
                    revisions[1:],
                    strict=False,
                )
            ):
                raise ValueError("feedback revision clock regressed")
        except (TypeError, ValueError, ValidationError) as error:
            raise SourceIntegrityError(
                "Feedback revision stream is invalid",
                mismatch_key=f"feedback:{observer_agent_id}:{capsule_id}",
            ) from error
        return revisions[-1]

    async def get_reputation(
        self,
        *,
        session: AsyncSession,
        subject_agent_id: UUID,
        observer_agent_id: UUID,
    ) -> Reputation | None:
        """Read one strict projection checkpoint for feedback mutation."""
        row = await session.get(
            AgentReputationRow,
            (subject_agent_id, observer_agent_id),
        )
        events = await self._reputation_feedback_events(
            session=session,
            subject_agent_id=subject_agent_id,
            observer_agent_id=observer_agent_id,
        )
        if row is None and not events:
            return None
        mismatch_key = f"reputation:{subject_agent_id}:{observer_agent_id}"
        if row is None or not events:
            raise SourceIntegrityError(
                "Observer-relative reputation checkpoint is missing",
                mismatch_key=mismatch_key,
            )
        try:
            expected, expected_event_id = replay_reputation(events)
            if expected is None or expected_event_id is None:
                raise ValueError("reputation replay unexpectedly produced no state")
            reputation = Reputation(
                subject_agent_id=row.subject_agent_id,
                observer_agent_id=row.observer_agent_id,
                useful_count=row.useful_count,
                refuted_count=row.refuted_count,
                harmful_count=row.harmful_count,
                alpha=row.alpha,
                beta=row.beta,
                last_feedback_at=events[-1].occurred_at,
            )
            if (
                expected_event_id != row.projection_event_id
                or expected.subject_agent_id != reputation.subject_agent_id
                or expected.observer_agent_id != reputation.observer_agent_id
                or expected.useful_count != reputation.useful_count
                or expected.refuted_count != reputation.refuted_count
                or expected.harmful_count != reputation.harmful_count
                or expected.alpha != reputation.alpha
                or expected.beta != reputation.beta
                or expected.last_feedback_at
                != reputation.last_feedback_at
            ):
                raise ValueError("reputation checkpoint is inconsistent")
        except (
            LookupError,
            SharingProjectionIntegrityError,
            StatementError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise SourceIntegrityError(
                "Observer-relative reputation projection is invalid",
                mismatch_key=mismatch_key,
            ) from error
        return reputation

    async def _reputation_feedback_events(
        self,
        *,
        session: AsyncSession,
        subject_agent_id: UUID,
        observer_agent_id: UUID,
    ) -> tuple[StoredEvent, ...]:
        """Read the ledger stream that must exactly reconstruct one row."""
        candidates = (
            await session.execute(
                select(
                    DomainEventRow,
                    ExperienceCapsuleRow.publisher_agent_id,
                )
                .outerjoin(
                    ExperienceCapsuleRow,
                    ExperienceCapsuleRow.capsule_id
                    == DomainEventRow.aggregate_id,
                )
                .where(
                    DomainEventRow.event_type
                    == CapsuleFeedbackRecordedV1.event_type,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        matching: list[StoredEvent] = []
        for row, source_publisher_id in candidates:
            try:
                payload = self._event_registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                )
                if not isinstance(payload, CapsuleFeedbackRecordedV1):
                    raise ValueError("feedback event payload type is invalid")
                require_utc(row.occurred_at)
            except (
                LookupError,
                StatementError,
                TypeError,
                ValueError,
                ValidationError,
            ) as error:
                raise SourceIntegrityError(
                    "Feedback event history is invalid",
                    mismatch_key=f"feedback_event:{row.event_id}",
                ) from error
            payload_targets_pair = (
                payload.publisher_agent_id == subject_agent_id
                and payload.observer_agent_id == observer_agent_id
            )
            if not (
                payload_targets_pair
                or row.actor_agent_id == observer_agent_id
                or source_publisher_id == subject_agent_id
            ):
                continue
            try:
                if (
                    source_publisher_id is None
                    or row.aggregate_type != "capsule"
                    or row.aggregate_id != payload.capsule_id
                    or row.sequence < 2
                    or row.actor_agent_id != payload.observer_agent_id
                    or source_publisher_id != payload.publisher_agent_id
                ):
                    raise ValueError("feedback event ledger anchor is invalid")
                event = StoredEvent(
                    event_id=row.event_id,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    sequence=row.sequence,
                    event_type=row.event_type,
                    payload=payload,
                    actor_agent_id=row.actor_agent_id,
                    causation_id=row.causation_id,
                    occurred_at=row.occurred_at,
                )
            except (StatementError, TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    "Feedback event history is invalid",
                    mismatch_key=f"feedback_event:{row.event_id}",
                ) from error
            if payload_targets_pair:
                try:
                    await validate_feedback_source_event(
                        session,
                        event_registry=self._event_registry,
                        event=event,
                        payload=payload,
                    )
                except (
                    LookupError,
                    SharingProjectionIntegrityError,
                    StatementError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ) as error:
                    raise SourceIntegrityError(
                        "Feedback source history is invalid",
                        mismatch_key=f"feedback:{payload.feedback_id}",
                    ) from error
                matching.append(event)
        source_ids = tuple(
            (
                await session.scalars(
                    select(CapsuleFeedbackRow.feedback_id)
                    .join(
                        ExperienceCapsuleRow,
                        ExperienceCapsuleRow.capsule_id
                        == CapsuleFeedbackRow.capsule_id,
                    )
                    .where(
                        CapsuleFeedbackRow.observer_agent_id
                        == observer_agent_id,
                        ExperienceCapsuleRow.publisher_agent_id
                        == subject_agent_id,
                    )
                    .order_by(CapsuleFeedbackRow.feedback_id)
                )
            ).all()
        )
        event_ids = tuple(
            cast(CapsuleFeedbackRecordedV1, event.payload).feedback_id
            for event in matching
        )
        if len(source_ids) != len(event_ids) or set(source_ids) != set(event_ids):
            raise SourceIntegrityError(
                "Feedback source/event correspondence is invalid",
                mismatch_key=(
                    f"feedback_stream:{subject_agent_id}:{observer_agent_id}"
                ),
            )
        return tuple(matching)

    async def strict_observer_trust(
        self,
        *,
        session: AsyncSession,
        subject_agent_id: UUID,
        observer_agent_id: UUID,
    ) -> float:
        """Read trust from the ledger head when feedback history exists."""
        reputation = await self.get_reputation(
            session=session,
            subject_agent_id=subject_agent_id,
            observer_agent_id=observer_agent_id,
        )
        if reputation is None:
            return observer_trust()
        return observer_trust(alpha=reputation.alpha, beta=reputation.beta)

    async def stream_available_pending(
        self,
        *,
        session: AsyncSession,
        recipient_agent_id: UUID,
        as_of: datetime,
    ) -> AsyncIterator[AdoptionInboxSource]:
        """Stream strict active pending sources in one database round trip."""
        observed_at = require_utc(as_of)
        capsule_event = aliased(DomainEventRow)
        inbox_event = aliased(DomainEventRow)
        capsule_state_head_sequence = (
            select(func.max(DomainEventRow.sequence))
            .where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id
                == ExperienceCapsuleRow.capsule_id,
                DomainEventRow.event_type.in_(
                    (
                        CapsulePublishedV1.event_type,
                        CapsuleRetractedV1.event_type,
                    )
                ),
            )
            .correlate(ExperienceCapsuleRow)
            .scalar_subquery()
        )
        inbox_state_head_sequence = (
            select(func.max(DomainEventRow.sequence))
            .where(
                DomainEventRow.aggregate_type == "inbox_item",
                DomainEventRow.aggregate_id == InboxItemRow.item_id,
                DomainEventRow.event_type.in_(
                    (
                        CapsuleReceivedV1.event_type,
                        CapsuleAdoptedV1.event_type,
                        CapsuleRejectedV1.event_type,
                    )
                ),
            )
            .correlate(InboxItemRow)
            .scalar_subquery()
        )
        statement = (
            select(
                InboxItemRow,
                ExperienceCapsuleRow,
                CapsuleStateRow,
                capsule_event,
                inbox_event,
                capsule_state_head_sequence,
                inbox_state_head_sequence,
            )
            .join(
                ExperienceCapsuleRow,
                ExperienceCapsuleRow.capsule_id == InboxItemRow.capsule_id,
            )
            .join(
                CapsuleStateRow,
                CapsuleStateRow.capsule_id == InboxItemRow.capsule_id,
            )
            .join(
                capsule_event,
                capsule_event.event_id == CapsuleStateRow.projection_event_id,
            )
            .join(
                inbox_event,
                inbox_event.event_id == InboxItemRow.projection_event_id,
            )
            .where(
                InboxItemRow.recipient_agent_id == recipient_agent_id,
                InboxItemRow.state == InboxState.PENDING,
                CapsuleStateRow.status == CapsuleStatus.ACTIVE,
                ExperienceCapsuleRow.expires_at > observed_at,
            )
            .order_by(InboxItemRow.item_id)
        )
        result = await session.stream(statement)
        async for (
            item,
            source,
            state,
            capsule_checkpoint,
            inbox_checkpoint,
            capsule_head_sequence,
            inbox_head_sequence,
        ) in result.tuples():
            if capsule_checkpoint.sequence != capsule_head_sequence:
                raise SourceIntegrityError(
                    f"Capsule {source.capsule_id} projection does not "
                    "reference its state event head",
                    mismatch_key=f"capsule:{source.capsule_id}",
                )
            if inbox_checkpoint.sequence != inbox_head_sequence:
                raise SourceIntegrityError(
                    f"Inbox item {item.item_id} projection does not "
                    "reference its state event head",
                    mismatch_key=f"inbox_item:{item.item_id}",
                )
            try:
                published = self._event_registry.decode(
                    event_type=capsule_checkpoint.event_type,
                    payload=capsule_checkpoint.payload,
                )
                received = self._event_registry.decode(
                    event_type=inbox_checkpoint.event_type,
                    payload=inbox_checkpoint.payload,
                )
                capsule = _capsule(
                    source,
                    status=state.status,
                    last_transition_at=capsule_checkpoint.occurred_at,
                )
                _validate_capsule_hashes(capsule)
                if (
                    not isinstance(published, CapsulePublishedV1)
                    or capsule_checkpoint.aggregate_type != "capsule"
                    or capsule_checkpoint.aggregate_id != capsule.capsule_id
                    or capsule_checkpoint.sequence != 1
                    or capsule_checkpoint.actor_agent_id
                    != capsule.publisher_agent_id
                    or capsule_checkpoint.occurred_at != capsule.created_at
                    or published.capsule_id != capsule.capsule_id
                    or published.topic_id != capsule.topic_id
                    or published.source_experience_id
                    != capsule.source_experience_id
                    or published.source_version_id
                    != capsule.source_version_id
                    or published.publisher_agent_id
                    != capsule.publisher_agent_id
                    or published.capsule_hash != capsule.capsule_hash
                    or published.root_fingerprint
                    != capsule.root_fingerprint
                    or published.status_after is not CapsuleStatus.ACTIVE
                ):
                    raise ValueError(
                        "active capsule checkpoint is inconsistent"
                    )
                if (
                    not isinstance(received, CapsuleReceivedV1)
                    or inbox_checkpoint.aggregate_type != "inbox_item"
                    or inbox_checkpoint.aggregate_id != item.item_id
                    or inbox_checkpoint.sequence != 1
                    or inbox_checkpoint.actor_agent_id
                    != capsule.publisher_agent_id
                    or inbox_checkpoint.occurred_at != capsule.created_at
                    or received.item_id != item.item_id
                    or received.capsule_id != capsule.capsule_id
                    or received.recipient_agent_id != recipient_agent_id
                    or received.state_after is not InboxState.PENDING
                ):
                    raise ValueError(
                        "pending inbox checkpoint is inconsistent"
                    )
            except (
                TypeError,
                ValueError,
                ValidationError,
            ) as error:
                raise SourceIntegrityError(
                    f"Inbox item {item.item_id} failed semantic validation",
                    mismatch_key=f"inbox_item:{item.item_id}",
                ) from error
            yield AdoptionInboxSource(
                item_id=item.item_id,
                recipient_agent_id=item.recipient_agent_id,
                state=item.state,
                capsule=capsule,
                inbox_state_at=require_utc(inbox_checkpoint.occurred_at),
                capsule_state_at=require_utc(
                    capsule_checkpoint.occurred_at
                ),
                adopted_event=None,
            )

    @staticmethod
    async def root_is_represented(
        *,
        session: AsyncSession,
        resulting_experience_id: UUID,
        root_fingerprint: str,
    ) -> bool:
        return (
            await session.scalar(
                select(AdoptionRecordRow.adoption_id).where(
                    AdoptionRecordRow.resulting_experience_id
                    == resulting_experience_id,
                    AdoptionRecordRow.root_fingerprint == root_fingerprint,
                )
            )
            is not None
        )

    @staticmethod
    async def observer_trust(
        *,
        session: AsyncSession,
        subject_agent_id: UUID,
        observer_agent_id: UUID,
    ) -> float:
        row = await session.get(
            AgentReputationRow,
            (subject_agent_id, observer_agent_id),
        )
        if row is None:
            return observer_trust()
        if (
            row.subject_agent_id != subject_agent_id
            or row.observer_agent_id != observer_agent_id
            or subject_agent_id == observer_agent_id
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    row.useful_count,
                    row.refuted_count,
                    row.harmful_count,
                )
            )
            or row.alpha != 2 + row.useful_count
            or row.beta != 2 + row.refuted_count + row.harmful_count
            or row.projection_event_id < 1
        ):
            raise SourceIntegrityError(
                "Observer-relative reputation projection is invalid",
                mismatch_key=(
                    f"reputation:{subject_agent_id}:{observer_agent_id}"
                ),
            )
        return observer_trust(alpha=row.alpha, beta=row.beta)

    @staticmethod
    def add_adoption(
        *,
        session: AsyncSession,
        adoption: StoredAdoption,
    ) -> None:
        session.add(
            AdoptionRecordRow(
                adoption_id=adoption.adoption_id,
                adopter_agent_id=adoption.adopter_agent_id,
                capsule_id=adoption.capsule_id,
                resulting_experience_id=adoption.resulting_experience_id,
                captured_trust=adoption.captured_trust,
                provenance_chain=canonical_json_bytes(
                    adoption.provenance_chain
                ),
                root_fingerprint=adoption.root_fingerprint,
                corroboration_applied=adoption.corroboration_applied,
                adopted_at=adoption.adopted_at,
            )
        )

    @staticmethod
    def add_feedback(
        *,
        session: AsyncSession,
        feedback: FeedbackRevision,
    ) -> None:
        session.add(
            CapsuleFeedbackRow(
                feedback_id=feedback.feedback_id,
                observer_agent_id=feedback.observer_agent_id,
                capsule_id=feedback.capsule_id,
                revision=feedback.revision,
                verdict=feedback.verdict,
                reason=canonical_json_bytes(feedback.reason),
                evidence=canonical_json_bytes(feedback.evidence),
                created_at=feedback.created_at,
            )
        )

    @staticmethod
    async def list_topics(
        *,
        session: AsyncSession,
        after_topic_id: UUID | None = None,
        limit: int = 100,
    ) -> tuple[Topic, ...]:
        statement = select(TopicRow)
        if after_topic_id is not None:
            statement = statement.where(TopicRow.topic_id > after_topic_id)
        rows = (
            (
                await session.scalars(
                    statement.order_by(TopicRow.topic_id).limit(_page_size(limit))
                )
            )
            .unique()
            .all()
        )
        return tuple(_topic(row) for row in rows)

    @staticmethod
    async def list_subscriptions(
        *,
        session: AsyncSession,
        subscriber_agent_id: UUID,
        after_subscription_id: UUID | None = None,
        limit: int = 100,
    ) -> tuple[Subscription, ...]:
        statement = select(SubscriptionRow).where(
            SubscriptionRow.subscriber_agent_id == subscriber_agent_id
        )
        if after_subscription_id is not None:
            statement = statement.where(
                SubscriptionRow.subscription_id > after_subscription_id
            )
        rows = (
            (
                await session.scalars(
                    statement.order_by(SubscriptionRow.subscription_id).limit(
                        _page_size(limit)
                    )
                )
            )
            .unique()
            .all()
        )
        return tuple(_subscription(row) for row in rows)

    @staticmethod
    async def list_eligible_subscriptions(
        *,
        session: AsyncSession,
        topic_id: UUID,
        publication_event_id: int,
        after: tuple[UUID, UUID] | None = None,
        limit: int = 100,
        exclude_subscriber_agent_id: UUID | None = None,
    ) -> tuple[Subscription, ...]:
        """List only subscriptions that predate a publication ledger event."""
        if (
            isinstance(publication_event_id, bool)
            or not isinstance(publication_event_id, int)
            or publication_event_id < 1
        ):
            raise ValueError("publication_event_id must be a positive integer")
        statement = select(SubscriptionRow).where(
            SubscriptionRow.topic_id == topic_id,
            SubscriptionRow.creation_event_id < publication_event_id,
        )
        if exclude_subscriber_agent_id is not None:
            statement = statement.where(
                SubscriptionRow.subscriber_agent_id != exclude_subscriber_agent_id
            )
        if after is not None:
            if (
                not isinstance(after, tuple)
                or len(after) != 2
                or not all(isinstance(value, UUID) for value in after)
            ):
                raise ValueError("after must contain subscriber and subscription UUIDs")
            statement = statement.where(
                tuple_(
                    SubscriptionRow.subscriber_agent_id,
                    SubscriptionRow.subscription_id,
                )
                > after
            )
        rows = (
            (
                await session.scalars(
                    statement.order_by(
                        SubscriptionRow.subscriber_agent_id,
                        SubscriptionRow.subscription_id,
                    ).limit(_page_size(limit))
                )
            )
            .unique()
            .all()
        )
        return tuple(_subscription(row) for row in rows)

    @staticmethod
    def add_topic(
        *,
        session: AsyncSession,
        topic: Topic,
    ) -> None:
        session.add(
            TopicRow(
                topic_id=topic.topic_id,
                owner_agent_id=topic.owner_agent_id,
                name=topic.name,
                description=topic.description,
                created_at=topic.created_at,
            )
        )

    @staticmethod
    def add_subscription(
        *,
        session: AsyncSession,
        subscription: Subscription,
    ) -> None:
        session.add(
            SubscriptionRow(
                subscription_id=subscription.subscription_id,
                subscriber_agent_id=subscription.subscriber_agent_id,
                topic_id=subscription.topic_id,
                creation_event_id=subscription.creation_event_id,
                created_at=subscription.created_at,
            )
        )

    @staticmethod
    def add_capsule(
        *,
        session: AsyncSession,
        capsule: Capsule,
    ) -> None:
        """Persist the immutable transport source using canonical arrays."""
        session.add(
            ExperienceCapsuleRow(
                capsule_id=capsule.capsule_id,
                transport_schema_version=capsule.transport_schema_version,
                topic_id=capsule.topic_id,
                source_experience_id=capsule.source_experience_id,
                source_version_id=capsule.source_version_id,
                publisher_agent_id=capsule.publisher_agent_id,
                kind=capsule.kind,
                body=capsule.body,
                summary=capsule.summary,
                mechanism=capsule.mechanism,
                tags=canonical_json_bytes(capsule.tags),
                applicability=canonical_json_bytes(capsule.applicability),
                evidence=canonical_json_bytes(capsule.evidence),
                falsifiers=canonical_json_bytes(capsule.falsifiers),
                publisher_confidence=capsule.publisher_confidence,
                provenance_chain=canonical_json_bytes(capsule.provenance_chain),
                root_fingerprint=capsule.root_fingerprint,
                source_content_hash=capsule.source_content_hash,
                created_at=capsule.created_at,
                expires_at=capsule.expires_at,
                hop_count=capsule.hop_count,
                capsule_hash=capsule.capsule_hash,
            )
        )


def _canonical_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid canonical JSON") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != raw:
        raise ValueError(f"{label} must be a canonical JSON object")
    return decoded


def _feedback_revision(row: CapsuleFeedbackRow) -> FeedbackRevision:
    reason = StructuredReason.model_validate(
        _canonical_object(
            row.reason,
            label=f"feedback {row.feedback_id} reason",
        )
    )
    evidence = tuple(
        TypedEvidence.model_validate(item)
        for item in _canonical_array(
            row.evidence,
            label=f"feedback {row.feedback_id} evidence",
        )
    )
    feedback = FeedbackRevision(
        feedback_id=row.feedback_id,
        observer_agent_id=row.observer_agent_id,
        capsule_id=row.capsule_id,
        revision=row.revision,
        verdict=FeedbackVerdict(row.verdict),
        reason=reason,
        evidence=evidence,
        created_at=row.created_at,
    )
    if (
        canonical_json_bytes(feedback.reason) != row.reason
        or canonical_json_bytes(feedback.evidence) != row.evidence
    ):
        raise ValueError("feedback source payload is not canonical")
    return feedback


def _canonical_array(raw: bytes, *, label: str) -> list[Any]:
    try:
        values: Any = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(values, list) or canonical_json_bytes(values) != raw:
        raise ValueError(f"{label} must be a canonical JSON array")
    return values


def _capsule(
    row: ExperienceCapsuleRow,
    *,
    status: CapsuleStatus,
    last_transition_at: datetime,
) -> Capsule:
    tags = _canonical_array(row.tags, label="capsule tags")
    applicability = _canonical_array(
        row.applicability,
        label="capsule applicability",
    )
    evidence_values = _canonical_array(
        row.evidence,
        label="capsule evidence",
    )
    falsifiers = _canonical_array(
        row.falsifiers,
        label="capsule falsifiers",
    )
    evidence = tuple(
        TypedEvidence.model_validate(value) for value in evidence_values
    )
    provenance = _decode_provenance(
        row.provenance_chain,
        label=f"capsule {row.capsule_id}",
        allow_empty=True,
    )
    capsule = Capsule.model_validate(
        {
            "capsule_id": row.capsule_id,
            "transport_schema_version": row.transport_schema_version,
            "topic_id": row.topic_id,
            "source_experience_id": row.source_experience_id,
            "source_version_id": row.source_version_id,
            "publisher_agent_id": row.publisher_agent_id,
            "kind": row.kind,
            "body": row.body,
            "summary": row.summary,
            "mechanism": row.mechanism,
            "tags": tuple(tags),
            "applicability": tuple(applicability),
            "evidence": evidence,
            "falsifiers": tuple(falsifiers),
            "publisher_confidence": row.publisher_confidence,
            "provenance_chain": provenance,
            "root_fingerprint": row.root_fingerprint,
            "source_content_hash": row.source_content_hash,
            "created_at": row.created_at,
            "expires_at": row.expires_at,
            "hop_count": row.hop_count,
            "capsule_hash": row.capsule_hash,
            "status": status,
            "last_transition_at": last_transition_at,
        }
    )
    if (
        canonical_json_bytes(capsule.tags) != row.tags
        or canonical_json_bytes(capsule.applicability) != row.applicability
        or canonical_json_bytes(capsule.evidence) != row.evidence
        or canonical_json_bytes(capsule.falsifiers) != row.falsifiers
        or canonical_json_bytes(capsule.provenance_chain)
        != row.provenance_chain
    ):
        raise ValueError("capsule metadata is not canonical")
    return capsule


def _validate_capsule_hashes(capsule: Capsule) -> None:
    content_hash = encode_version_content(
        kind=capsule.kind,
        content=VersionContent(
            body=capsule.body,
            summary=capsule.summary,
            mechanism=capsule.mechanism,
            tags=capsule.tags,
            applicability=capsule.applicability,
            evidence=capsule.evidence,
            falsifiers=capsule.falsifiers,
        ),
    ).content_hash
    if content_hash != capsule.source_content_hash:
        raise ValueError("capsule content does not match its semantic hash")
    expected_hash = compute_capsule_hash(
        transport_schema_version=capsule.transport_schema_version,
        capsule_id=capsule.capsule_id,
        topic_id=capsule.topic_id,
        source_experience_id=capsule.source_experience_id,
        source_version_id=capsule.source_version_id,
        publisher_agent_id=capsule.publisher_agent_id,
        kind=capsule.kind,
        body=capsule.body,
        summary=capsule.summary,
        mechanism=capsule.mechanism,
        tags=capsule.tags,
        applicability=capsule.applicability,
        evidence=capsule.evidence,
        falsifiers=capsule.falsifiers,
        publisher_confidence=capsule.publisher_confidence,
        provenance_chain=capsule.provenance_chain,
        root_fingerprint=capsule.root_fingerprint,
        source_content_hash=capsule.source_content_hash,
        created_at=capsule.created_at,
        expires_at=capsule.expires_at,
        hop_count=capsule.hop_count,
    )
    if expected_hash != capsule.capsule_hash:
        raise ValueError("capsule transport hash is invalid")
    if capsule.hop_count == 0:
        expected_root = compute_original_root_fingerprint(
            root_publisher_id=capsule.publisher_agent_id,
            source_content_hash=capsule.source_content_hash,
        )
        if expected_root != capsule.root_fingerprint:
            raise ValueError("original capsule root fingerprint is invalid")


def _decode_provenance(
    raw: bytes,
    *,
    label: str,
    allow_empty: bool,
) -> tuple[ProvenanceHop, ...]:
    try:
        values: Any = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} provenance is not JSON") from error
    if not isinstance(values, list):
        raise ValueError(f"{label} provenance must be an array")
    if not allow_empty and not values:
        raise ValueError(f"{label} provenance must not be empty")

    hops: list[ProvenanceHop] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "capsule_id",
            "publisher_agent_id",
        }:
            raise ValueError(f"{label} provenance has an invalid hop")
        capsule_id = value["capsule_id"]
        publisher_agent_id = value["publisher_agent_id"]
        if not isinstance(capsule_id, str) or not isinstance(
            publisher_agent_id,
            str,
        ):
            raise ValueError(f"{label} provenance IDs must be strings")
        try:
            hop = ProvenanceHop(
                capsule_id=UUID(capsule_id),
                publisher_agent_id=UUID(publisher_agent_id),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} provenance IDs must be canonical UUIDs"
            ) from error
        if str(hop.capsule_id) != capsule_id or (
            str(hop.publisher_agent_id) != publisher_agent_id
        ):
            raise ValueError(f"{label} provenance UUIDs must be canonical")
        hops.append(hop)

    result = tuple(hops)
    if canonical_json_bytes(result) != bytes(raw):
        raise ValueError(f"{label} provenance must use canonical JSON")
    capsule_ids = tuple(hop.capsule_id for hop in result)
    if len(capsule_ids) != len(set(capsule_ids)):
        raise ValueError(f"{label} provenance must not repeat a capsule")
    return result


def _topic(row: TopicRow) -> Topic:
    return Topic(
        topic_id=row.topic_id,
        owner_agent_id=row.owner_agent_id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
    )


def _subscription(row: SubscriptionRow) -> Subscription:
    return Subscription(
        subscription_id=row.subscription_id,
        subscriber_agent_id=row.subscriber_agent_id,
        topic_id=row.topic_id,
        creation_event_id=row.creation_event_id,
        created_at=row.created_at,
    )


__all__ = [
    "AdoptionInboxSource",
    "ParentAdoption",
    "SharingRepository",
    "StoredAdoption",
]
