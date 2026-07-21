"""Strict, independently rebuildable sharing projection reducers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import (
    EventRegistry,
    StoredEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences import VersionContent, encode_version_content
from experience_hub.experiences.events import (
    ExperienceCorroboratedV1,
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.sharing.confidence import initial_adoption_confidence
from experience_hub.sharing.events import (
    SHARING_EVENT_AGGREGATE_TYPES,
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    SubscriptionCreatedV1,
)
from experience_hub.sharing.hashing import (
    compute_capsule_hash,
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import (
    Capsule,
    CapsuleStatus,
    FeedbackRevision,
    FeedbackVerdict,
    InboxState,
    ProvenanceHop,
    Reputation,
)
from experience_hub.sharing.validation import (
    SharingAdoptionSourceValidator,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    CapsuleFeedbackRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceRow,
    ExperienceVersionRow,
    SubscriptionRow,
    TopicRow,
)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RECEIVED_CAPSULE_CACHE_KEY = "experience_hub.sharing.validated_received_capsules.v1"


class SharingProjectionIntegrityError(RuntimeError):
    """A sharing event cannot be reconciled with its source anchors."""

    code = "sharing_projection_integrity_error"


def _fail(message: str) -> SharingProjectionIntegrityError:
    return SharingProjectionIntegrityError(message)


def reduce_reputation(
    current: Reputation | None,
    *,
    event: StoredEvent,
    previous_verdict: FeedbackVerdict | None,
) -> Reputation:
    """Apply one latest-revision replacement to observer-relative counts."""
    payload = event.payload
    if (
        not isinstance(payload, CapsuleFeedbackRecordedV1)
        or event.event_type != CapsuleFeedbackRecordedV1.event_type
        or event.aggregate_type != "capsule"
        or event.aggregate_id != payload.capsule_id
        or event.event_id < 1
        or event.sequence < 2
        or event.actor_agent_id != payload.observer_agent_id
        or payload.publisher_agent_id == payload.observer_agent_id
    ):
        raise _fail("Feedback event has invalid reputation semantics")
    if previous_verdict is not payload.previous_verdict:
        raise _fail("Feedback previous verdict does not match its source revision")
    try:
        occurred_at = require_utc(event.occurred_at)
    except (TypeError, ValueError) as error:
        raise _fail("Feedback event has an invalid causal clock") from error

    if current is None:
        useful_count = 0
        refuted_count = 0
        harmful_count = 0
        alpha_before = 2
        beta_before = 2
    else:
        if (
            current.subject_agent_id != payload.publisher_agent_id
            or current.observer_agent_id != payload.observer_agent_id
        ):
            raise _fail("Feedback reputation identity does not match before-state")
        if occurred_at < current.last_feedback_at:
            raise _fail("Feedback event time precedes current reputation state")
        useful_count = current.useful_count
        refuted_count = current.refuted_count
        harmful_count = current.harmful_count
        alpha_before = current.alpha
        beta_before = current.beta
    if (
        payload.alpha_before != alpha_before
        or payload.beta_before != beta_before
    ):
        raise _fail("Feedback reputation before-state does not match locked state")

    if previous_verdict is FeedbackVerdict.USEFUL:
        useful_count -= 1
    elif previous_verdict is FeedbackVerdict.REFUTED:
        refuted_count -= 1
    elif previous_verdict is FeedbackVerdict.HARMFUL:
        harmful_count -= 1
    if min(useful_count, refuted_count, harmful_count) < 0:
        raise _fail("Feedback prior contribution is absent from reputation state")

    if payload.current_verdict is FeedbackVerdict.USEFUL:
        useful_count += 1
    elif payload.current_verdict is FeedbackVerdict.REFUTED:
        refuted_count += 1
    else:
        harmful_count += 1
    try:
        revised = Reputation(
            subject_agent_id=payload.publisher_agent_id,
            observer_agent_id=payload.observer_agent_id,
            useful_count=useful_count,
            refuted_count=refuted_count,
            harmful_count=harmful_count,
            alpha=2 + useful_count,
            beta=2 + refuted_count + harmful_count,
            last_feedback_at=occurred_at,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _fail("Feedback reputation transition is invalid") from error
    if (
        revised.alpha != payload.alpha_after
        or revised.beta != payload.beta_after
    ):
        raise _fail("Feedback reputation after-state does not match the event")
    return revised


def replay_reputation(
    events: tuple[StoredEvent, ...],
) -> tuple[Reputation | None, int | None]:
    """Replay one publisher/observer feedback stream in ledger order."""
    current: Reputation | None = None
    latest_by_capsule: dict[UUID, tuple[int, FeedbackVerdict]] = {}
    last_event_id: int | None = None
    for event in events:
        payload = event.payload
        if not isinstance(payload, CapsuleFeedbackRecordedV1):
            raise _fail("Reputation replay encountered a non-feedback event")
        if last_event_id is not None and event.event_id <= last_event_id:
            raise _fail("Reputation feedback events are not in ledger order")
        prior = latest_by_capsule.get(payload.capsule_id)
        expected_revision = 1 if prior is None else prior[0] + 1
        expected_verdict = None if prior is None else prior[1]
        if (
            payload.revision != expected_revision
            or payload.previous_verdict is not expected_verdict
        ):
            raise _fail("Reputation feedback revision history is not contiguous")
        current = reduce_reputation(
            current,
            event=event,
            previous_verdict=expected_verdict,
        )
        latest_by_capsule[payload.capsule_id] = (
            payload.revision,
            payload.current_verdict,
        )
        last_event_id = event.event_id
    return current, last_event_id


@dataclass(frozen=True, slots=True)
class _ValidatedPublication:
    capsule: Capsule
    event: StoredEvent


def require_sharing_event_anchor(
    event: StoredEvent,
    *,
    aggregate_id: UUID,
) -> None:
    """Fail closed when a sharing event is sequenced on the wrong resource."""
    try:
        expected_type = SHARING_EVENT_AGGREGATE_TYPES[event.event_type]
    except KeyError as error:
        raise _fail(f"Unknown sharing event type: {event.event_type}") from error
    if event.aggregate_type != expected_type or event.aggregate_id != aggregate_id:
        raise _fail(f"{event.event_type} has an invalid aggregate anchor")


def _target_table(target_prefix: str | None, projection_name: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(projection_name):
        raise ValueError("Unsafe sharing projection target")
    if target_prefix is None:
        return f'main."{projection_name}"'
    name = f"{target_prefix}{projection_name}"
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError("Unsafe sharing projection target")
    return f'temp."{name}"'


def _stored_event(
    registry: EventRegistry,
    row: DomainEventRow,
) -> StoredEvent:
    try:
        payload = registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
    except (TypeError, ValueError) as error:
        raise _fail("Sharing event payload is invalid") from error
    return StoredEvent(
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


def _json_array(value: bytes, *, label: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"Capsule {label} source is invalid") from error
    if not isinstance(decoded, list) or canonical_json_bytes(decoded) != value:
        raise _fail(f"Capsule {label} source is not a canonical array")
    return decoded


def _json_object(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"Feedback {label} source is invalid") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != value:
        raise _fail(f"Feedback {label} source is not a canonical object")
    return decoded


def _feedback_from_source(source: CapsuleFeedbackRow) -> FeedbackRevision:
    try:
        feedback = FeedbackRevision(
            feedback_id=source.feedback_id,
            observer_agent_id=source.observer_agent_id,
            capsule_id=source.capsule_id,
            revision=source.revision,
            verdict=source.verdict,
            reason=StructuredReason.model_validate(
                _json_object(source.reason, label="reason")
            ),
            evidence=tuple(
                TypedEvidence.model_validate(item)
                for item in _json_array(source.evidence, label="evidence")
            ),
            created_at=source.created_at,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _fail("Feedback source row is invalid") from error
    if (
        canonical_json_bytes(feedback.reason) != source.reason
        or canonical_json_bytes(feedback.evidence) != source.evidence
    ):
        raise _fail("Feedback source payloads are not canonical")
    return feedback


def _capsule_from_source(
    source: ExperienceCapsuleRow,
    *,
    event: StoredEvent,
) -> Capsule:
    try:
        evidence = tuple(
            TypedEvidence.model_validate(item)
            for item in _json_array(source.evidence, label="evidence")
        )
        provenance = tuple(
            ProvenanceHop(
                capsule_id=UUID(str(item["capsule_id"])),
                publisher_agent_id=UUID(str(item["publisher_agent_id"])),
            )
            for item in _json_array(
                source.provenance_chain,
                label="provenance chain",
            )
            if isinstance(item, dict)
            and set(item) == {"capsule_id", "publisher_agent_id"}
        )
        raw_provenance = _json_array(
            source.provenance_chain,
            label="provenance chain",
        )
        if len(provenance) != len(raw_provenance):
            raise ValueError("Every provenance hop must have the exact V1 shape")
        capsule = Capsule(
            capsule_id=source.capsule_id,
            transport_schema_version=cast(
                Literal[1],
                source.transport_schema_version,
            ),
            topic_id=source.topic_id,
            source_experience_id=source.source_experience_id,
            source_version_id=source.source_version_id,
            publisher_agent_id=source.publisher_agent_id,
            kind=source.kind,
            body=source.body,
            summary=source.summary,
            mechanism=source.mechanism,
            tags=tuple(_json_array(source.tags, label="tags")),
            applicability=tuple(
                _json_array(source.applicability, label="applicability")
            ),
            evidence=evidence,
            falsifiers=tuple(_json_array(source.falsifiers, label="falsifiers")),
            publisher_confidence=source.publisher_confidence,
            provenance_chain=provenance,
            root_fingerprint=source.root_fingerprint,
            source_content_hash=source.source_content_hash,
            created_at=source.created_at,
            expires_at=source.expires_at,
            hop_count=source.hop_count,
            capsule_hash=source.capsule_hash,
            status=CapsuleStatus.ACTIVE,
            last_transition_at=event.occurred_at,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise _fail("Capsule source row is invalid") from error
    if (
        canonical_json_bytes(capsule.tags) != source.tags
        or canonical_json_bytes(capsule.applicability) != source.applicability
        or canonical_json_bytes(capsule.evidence) != source.evidence
        or canonical_json_bytes(capsule.falsifiers) != source.falsifiers
        or canonical_json_bytes(capsule.provenance_chain) != source.provenance_chain
    ):
        raise _fail("Capsule source metadata is not canonical")
    return capsule


async def _validated_capsule_source(
    session: AsyncSession,
    *,
    event: StoredEvent,
    payload: CapsulePublishedV1,
) -> Capsule:
    try:
        source = await session.get(ExperienceCapsuleRow, payload.capsule_id)
        topic = await session.get(TopicRow, payload.topic_id)
        identity = await session.get(
            ExperienceRow,
            payload.source_experience_id,
        )
        version = await session.get(
            ExperienceVersionRow,
            payload.source_version_id,
        )
    except (LookupError, StatementError, TypeError, ValueError) as error:
        raise _fail("Published capsule source anchors are invalid") from error
    if source is None or topic is None or identity is None or version is None:
        raise _fail("Published capsule source anchor is missing")
    capsule = _capsule_from_source(source, event=event)
    if (
        event.event_type != CapsulePublishedV1.event_type
        or event.event_id < 1
        or event.sequence != 1
        or event.actor_agent_id != payload.publisher_agent_id
        or capsule.capsule_id != payload.capsule_id
        or capsule.topic_id != payload.topic_id
        or capsule.source_experience_id != payload.source_experience_id
        or capsule.source_version_id != payload.source_version_id
        or capsule.publisher_agent_id != payload.publisher_agent_id
        or capsule.capsule_hash != payload.capsule_hash
        or capsule.root_fingerprint != payload.root_fingerprint
        or payload.status_after is not CapsuleStatus.ACTIVE
        or capsule.created_at != event.occurred_at
        or topic.created_at > event.occurred_at
        or identity.owner_agent_id != payload.publisher_agent_id
        or identity.kind is not capsule.kind
        or version.experience_id != payload.source_experience_id
        or version.content_hash != capsule.source_content_hash
        or version.created_at > event.occurred_at
    ):
        raise _fail("Published capsule event does not match source anchors")
    try:
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
        expected_source_hash = encode_version_content(
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
        if capsule.source_content_hash != expected_source_hash:
            raise _fail("Published capsule semantic hash does not match its content")
        if capsule.hop_count == 0:
            expected_root = compute_original_root_fingerprint(
                root_publisher_id=capsule.publisher_agent_id,
                source_content_hash=capsule.source_content_hash,
            )
            if capsule.root_fingerprint != expected_root:
                raise _fail("Original capsule root fingerprint is invalid")
    except (TypeError, ValueError) as error:
        raise _fail("Published capsule hash inputs are invalid") from error
    if capsule.capsule_hash != expected_hash:
        raise _fail("Published capsule hash does not match its source")
    return capsule


async def _validated_received_publication(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    capsule_id: UUID,
) -> _ValidatedPublication:
    """Validate an immutable publication once per projection session."""
    cache = cast(
        dict[UUID, _ValidatedPublication],
        session.info.setdefault(_RECEIVED_CAPSULE_CACHE_KEY, {}),
    )
    cached = cache.get(capsule_id)
    if cached is not None:
        return cached
    publication_row = await session.scalar(
        select(DomainEventRow).where(
            DomainEventRow.aggregate_type == "capsule",
            DomainEventRow.aggregate_id == capsule_id,
            DomainEventRow.event_type == CapsulePublishedV1.event_type,
            DomainEventRow.sequence == 1,
        )
    )
    if publication_row is None:
        raise _fail("Received capsule publication event is missing")
    publication_event = _stored_event(event_registry, publication_row)
    if not isinstance(publication_event.payload, CapsulePublishedV1):
        raise _fail("Received capsule publication payload is invalid")
    require_sharing_event_anchor(
        publication_event,
        aggregate_id=capsule_id,
    )
    capsule = await _validated_capsule_source(
        session,
        event=publication_event,
        payload=publication_event.payload,
    )
    validated = _ValidatedPublication(
        capsule=capsule,
        event=publication_event,
    )
    cache[capsule_id] = validated
    return validated


async def validate_feedback_source_event(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    event: StoredEvent,
    payload: CapsuleFeedbackRecordedV1,
) -> tuple[FeedbackRevision, FeedbackVerdict | None]:
    """Bind one reputation transition to its immutable revision and history."""
    require_sharing_event_anchor(event, aggregate_id=payload.capsule_id)
    if (
        event.event_type != CapsuleFeedbackRecordedV1.event_type
        or event.event_id < 1
        or event.sequence < 2
        or event.actor_agent_id != payload.observer_agent_id
    ):
        raise _fail("Feedback event has invalid creation semantics")
    publication = await _validated_received_publication(
        session,
        event_registry=event_registry,
        capsule_id=payload.capsule_id,
    )
    await _require_feedback_authorization(
        session,
        event_registry=event_registry,
        event=event,
        payload=payload,
        publication=publication,
    )
    try:
        source = await session.get(CapsuleFeedbackRow, payload.feedback_id)
        previous_source = (
            None
            if payload.revision == 1
            else await session.scalar(
                select(CapsuleFeedbackRow).where(
                    CapsuleFeedbackRow.observer_agent_id
                    == payload.observer_agent_id,
                    CapsuleFeedbackRow.capsule_id == payload.capsule_id,
                    CapsuleFeedbackRow.revision == payload.revision - 1,
                )
            )
        )
        predecessor = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == payload.capsule_id,
                DomainEventRow.sequence == event.sequence - 1,
            )
        )
        prior_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "capsule",
                    DomainEventRow.aggregate_id == payload.capsule_id,
                    DomainEventRow.event_type
                    == CapsuleFeedbackRecordedV1.event_type,
                    DomainEventRow.event_id < event.event_id,
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    except (LookupError, StatementError, TypeError, ValueError) as error:
        raise _fail("Feedback source anchors are invalid") from error
    if source is None or predecessor is None:
        raise _fail("Feedback source anchor is missing")
    feedback = _feedback_from_source(source)
    previous_feedback = (
        None if previous_source is None else _feedback_from_source(previous_source)
    )
    matching_prior: list[CapsuleFeedbackRecordedV1] = []
    for row in prior_rows:
        prior_event = _stored_event(event_registry, row)
        if not isinstance(prior_event.payload, CapsuleFeedbackRecordedV1):
            raise _fail("Prior feedback payload is invalid")
        prior_payload = prior_event.payload
        if prior_payload.observer_agent_id != payload.observer_agent_id:
            continue
        if (
            prior_payload.capsule_id != payload.capsule_id
            or prior_payload.publisher_agent_id
            != payload.publisher_agent_id
            or prior_event.actor_agent_id != payload.observer_agent_id
        ):
            raise _fail("Prior observer feedback event is inconsistent")
        matching_prior.append(prior_payload)
    expected_revisions = tuple(range(1, payload.revision))
    if tuple(item.revision for item in matching_prior) != expected_revisions:
        raise _fail("Feedback event revision history is not contiguous")

    previous_verdict = (
        None if previous_feedback is None else previous_feedback.verdict
    )
    if (
        feedback.feedback_id != payload.feedback_id
        or feedback.observer_agent_id != payload.observer_agent_id
        or feedback.capsule_id != payload.capsule_id
        or feedback.revision != payload.revision
        or feedback.verdict is not payload.current_verdict
        or feedback.created_at != event.occurred_at
        or publication.capsule.publisher_agent_id
        != payload.publisher_agent_id
        or publication.event.event_id >= event.event_id
        or payload.publisher_agent_id == payload.observer_agent_id
        or previous_verdict is not payload.previous_verdict
    ):
        raise _fail("Feedback event does not match its immutable revision")
    if payload.revision == 1:
        if previous_feedback is not None or matching_prior:
            raise _fail("First feedback revision has unexpected prior state")
    else:
        if (
            previous_feedback is None
            or previous_feedback.observer_agent_id
            != payload.observer_agent_id
            or previous_feedback.capsule_id != payload.capsule_id
            or previous_feedback.revision != payload.revision - 1
            or previous_feedback.created_at > feedback.created_at
            or not matching_prior
            or matching_prior[-1].feedback_id
            != previous_feedback.feedback_id
            or matching_prior[-1].current_verdict
            is not previous_feedback.verdict
        ):
            raise _fail("Feedback previous revision source is inconsistent")
    if (
        predecessor.event_id >= event.event_id
        or predecessor.occurred_at > event.occurred_at
        or feedback.created_at < publication.capsule.created_at
    ):
        raise _fail("Feedback causal clock is inconsistent")
    causal_rows = (
        await session.scalars(
            select(DomainEventRow)
            .where(DomainEventRow.causation_id == event.causation_id)
            .order_by(DomainEventRow.event_id)
        )
    ).all()
    if (
        tuple(row.event_type for row in causal_rows)
        != (CapsuleFeedbackRecordedV1.event_type,)
        or causal_rows[0].event_id != event.event_id
    ):
        raise _fail("Feedback command event sequence is invalid")
    return feedback, previous_verdict


async def _validate_feedback_source_correspondence(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
) -> None:
    """Require every immutable feedback source to name exactly one ledger event."""
    try:
        sources = (
            await session.scalars(
                select(CapsuleFeedbackRow).order_by(
                    CapsuleFeedbackRow.observer_agent_id,
                    CapsuleFeedbackRow.capsule_id,
                    CapsuleFeedbackRow.revision,
                    CapsuleFeedbackRow.feedback_id,
                )
            )
        ).all()
        event_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.event_type
                    == CapsuleFeedbackRecordedV1.event_type
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
    except (LookupError, StatementError, TypeError, ValueError) as error:
        raise _fail("Feedback source/event correspondence is unreadable") from error

    events_by_feedback_id: dict[UUID, list[StoredEvent]] = {}
    for row in event_rows:
        event = _stored_event(event_registry, row)
        if not isinstance(event.payload, CapsuleFeedbackRecordedV1):
            raise _fail("Feedback source/event payload is invalid")
        events_by_feedback_id.setdefault(
            event.payload.feedback_id,
            [],
        ).append(event)

    for source in sources:
        feedback = _feedback_from_source(source)
        matches = events_by_feedback_id.get(feedback.feedback_id, [])
        if len(matches) != 1:
            raise _fail(
                "Feedback source has no unique event correspondence"
            )
        event = matches[0]
        payload = cast(CapsuleFeedbackRecordedV1, event.payload)
        if (
            payload.observer_agent_id != feedback.observer_agent_id
            or payload.capsule_id != feedback.capsule_id
            or payload.revision != feedback.revision
            or payload.current_verdict is not feedback.verdict
            or event.actor_agent_id != feedback.observer_agent_id
            or event.occurred_at != feedback.created_at
        ):
            raise _fail("Feedback source/event correspondence is inconsistent")


async def _require_feedback_authorization(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    event: StoredEvent,
    payload: CapsuleFeedbackRecordedV1,
    publication: _ValidatedPublication,
) -> None:
    """Prove adoption/rejection authorization from immutable inbox events."""
    received_rows = (
        await session.scalars(
            select(DomainEventRow)
            .where(
                DomainEventRow.event_type == CapsuleReceivedV1.event_type,
                DomainEventRow.event_id < event.event_id,
            )
            .order_by(DomainEventRow.event_id)
        )
    ).all()
    matches: list[tuple[StoredEvent, CapsuleReceivedV1]] = []
    for row in received_rows:
        received = _stored_event(event_registry, row)
        if not isinstance(received.payload, CapsuleReceivedV1):
            raise _fail("Feedback authorization receipt payload is invalid")
        received_payload = received.payload
        if (
            received_payload.capsule_id == payload.capsule_id
            and received_payload.recipient_agent_id
            == payload.observer_agent_id
        ):
            matches.append((received, received_payload))
    if len(matches) != 1:
        raise _fail(
            "Feedback requires exactly one observer-owned inbox receipt"
        )
    received, received_payload = matches[0]
    if (
        received.aggregate_type != "inbox_item"
        or received.aggregate_id != received_payload.item_id
        or received.sequence != 1
        or received.actor_agent_id
        != publication.capsule.publisher_agent_id
        or received.occurred_at != publication.capsule.created_at
        or received.event_id <= publication.event.event_id
        or received.causation_id != publication.event.causation_id
        or received_payload.state_after is not InboxState.PENDING
    ):
        raise _fail("Feedback inbox receipt history is inconsistent")
    terminal_row = await session.scalar(
        select(DomainEventRow).where(
            DomainEventRow.aggregate_type == "inbox_item",
            DomainEventRow.aggregate_id == received_payload.item_id,
            DomainEventRow.sequence == 2,
            DomainEventRow.event_id < event.event_id,
        )
    )
    if terminal_row is None:
        raise _fail("Feedback requires prior adoption or rejection")
    terminal = _stored_event(event_registry, terminal_row)
    terminal_payload = terminal.payload
    authorized = False
    if isinstance(terminal_payload, CapsuleAdoptedV1):
        authorized = (
            terminal.event_type == CapsuleAdoptedV1.event_type
            and terminal.actor_agent_id == payload.observer_agent_id
            and terminal_payload.item_id == received_payload.item_id
            and terminal_payload.capsule_id == payload.capsule_id
            and terminal_payload.adopter_agent_id
            == payload.observer_agent_id
            and terminal_payload.state_before is InboxState.PENDING
            and terminal_payload.state_after is InboxState.ADOPTED
        )
    elif isinstance(terminal_payload, CapsuleRejectedV1):
        authorized = (
            terminal.event_type == CapsuleRejectedV1.event_type
            and terminal.actor_agent_id == payload.observer_agent_id
            and terminal_payload.item_id == received_payload.item_id
            and terminal_payload.capsule_id == payload.capsule_id
            and terminal_payload.recipient_agent_id
            == payload.observer_agent_id
            and terminal_payload.state_before is InboxState.PENDING
            and terminal_payload.state_after is InboxState.REJECTED
        )
    if (
        not authorized
        or terminal.aggregate_type != "inbox_item"
        or terminal.aggregate_id != received_payload.item_id
        or terminal.sequence != 2
        or terminal.event_id <= received.event_id
        or terminal.event_id >= event.event_id
        or terminal.occurred_at < received.occurred_at
        or terminal.occurred_at > event.occurred_at
    ):
        raise _fail("Feedback authorization terminal history is inconsistent")


async def _replayed_reputation_before(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    event: StoredEvent,
    publisher_agent_id: UUID,
    observer_agent_id: UUID,
) -> tuple[Reputation | None, int | None]:
    rows = (
        await session.scalars(
            select(DomainEventRow)
            .where(
                DomainEventRow.event_type
                == CapsuleFeedbackRecordedV1.event_type,
                DomainEventRow.event_id < event.event_id,
            )
            .order_by(DomainEventRow.event_id)
        )
    ).all()
    matching: list[StoredEvent] = []
    for row in rows:
        prior = _stored_event(event_registry, row)
        if not isinstance(prior.payload, CapsuleFeedbackRecordedV1):
            raise _fail("Prior reputation feedback payload is invalid")
        prior_payload = prior.payload
        if (
            prior_payload.publisher_agent_id != publisher_agent_id
            or prior_payload.observer_agent_id != observer_agent_id
        ):
            continue
        require_sharing_event_anchor(
            prior,
            aggregate_id=prior_payload.capsule_id,
        )
        publication = await _validated_received_publication(
            session,
            event_registry=event_registry,
            capsule_id=prior_payload.capsule_id,
        )
        source = await session.get(
            CapsuleFeedbackRow,
            prior_payload.feedback_id,
        )
        if source is None:
            raise _fail("Prior reputation feedback source is missing")
        feedback = _feedback_from_source(source)
        if (
            publication.capsule.publisher_agent_id
            != publisher_agent_id
            or prior.actor_agent_id != observer_agent_id
            or prior.sequence < 2
            or feedback.observer_agent_id != observer_agent_id
            or feedback.capsule_id != prior_payload.capsule_id
            or feedback.revision != prior_payload.revision
            or feedback.verdict is not prior_payload.current_verdict
            or feedback.created_at != prior.occurred_at
        ):
            raise _fail("Prior reputation feedback anchor is inconsistent")
        matching.append(prior)
    return replay_reputation(tuple(matching))


async def _require_equivalent_experience_before_adoption(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    event: StoredEvent,
    experience_id: UUID,
    owner_agent_id: UUID,
    content_hash: str,
) -> None:
    """Bind an echo adoption to the result's historical current semantics."""
    checkpoint = await session.scalar(
        select(DomainEventRow)
        .where(
            DomainEventRow.aggregate_type == "experience",
            DomainEventRow.aggregate_id == experience_id,
            DomainEventRow.event_id < event.event_id,
        )
        .order_by(DomainEventRow.event_id.desc())
        .limit(1)
    )
    if checkpoint is None:
        raise _fail("Equivalent adopted experience checkpoint is missing")
    try:
        checkpoint_payload = event_registry.decode(
            event_type=checkpoint.event_type,
            payload=checkpoint.payload,
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            "Equivalent adopted experience checkpoint is invalid"
        ) from error
    after = getattr(checkpoint_payload, "after", None)
    if (
        not isinstance(after, ExperienceStateSnapshotV1)
        or checkpoint.aggregate_type != "experience"
        or checkpoint.aggregate_id != experience_id
        or checkpoint.sequence < 2
        or checkpoint.occurred_at > event.occurred_at
        or after.experience_id != experience_id
        or after.owner_agent_id != owner_agent_id
        or after.current_content_hash != content_hash
        or after.temperature is Temperature.ARCHIVED
    ):
        raise _fail("Equivalent adopted experience source is inconsistent")


async def _has_prior_matching_adoption_event(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    event: StoredEvent,
    resulting_experience_id: UUID,
    root_fingerprint: str,
) -> bool:
    """Determine root independence from immutable adoption event order."""
    rows = (
        await session.scalars(
            select(DomainEventRow)
            .where(
                DomainEventRow.event_type
                == CapsuleAdoptedV1.event_type,
                DomainEventRow.event_id < event.event_id,
            )
            .order_by(DomainEventRow.event_id)
        )
    ).all()
    for row in rows:
        try:
            decoded = event_registry.decode(
                event_type=row.event_type,
                payload=row.payload,
            )
        except (TypeError, ValueError) as error:
            raise _fail("Prior capsule adoption event is invalid") from error
        if (
            not isinstance(decoded, CapsuleAdoptedV1)
            or row.aggregate_type != "inbox_item"
            or row.aggregate_id != decoded.item_id
            or row.sequence != 2
            or row.actor_agent_id != decoded.adopter_agent_id
        ):
            raise _fail("Prior capsule adoption event is inconsistent")
        if (
            decoded.resulting_experience_id
            == resulting_experience_id
            and decoded.root_fingerprint == root_fingerprint
        ):
            return True
    return False


class CapsuleStateProjector:
    """Version-one reducer for the independently rebuildable capsule state."""

    name = "capsule_state"
    version = 1
    event_types = frozenset(
        {
            CapsulePublishedV1.event_type,
            CapsuleRetractedV1.event_type,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target_table(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "capsule_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "status VARCHAR(9) NOT NULL, "
                "projection_event_id INTEGER NOT NULL, "
                "CHECK (status IN ('active', 'retracted')), "
                "CHECK (projection_event_id > 0)"
                ")"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._event_registry, row),
                target_prefix=target_prefix,
            )
    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        if isinstance(event.payload, CapsuleRetractedV1):
            await self._apply_retracted(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if not isinstance(event.payload, CapsulePublishedV1):
            raise _fail(f"Unsupported capsule-state event {event.event_type!r}")
        payload = event.payload
        require_sharing_event_anchor(event, aggregate_id=payload.capsule_id)
        await _validated_capsule_source(
            session,
            event=event,
            payload=payload,
        )
        target = _target_table(target_prefix, self.name)
        existing = (
            await session.execute(
                text(f"SELECT capsule_id FROM {target} WHERE capsule_id = :capsule_id"),
                {"capsule_id": str(payload.capsule_id)},
            )
        ).one_or_none()
        if existing is not None:
            raise _fail("Published capsule projection already exists")
        await session.execute(
            text(
                f"INSERT INTO {target} "
                "(capsule_id, status, projection_event_id) "
                "VALUES (:capsule_id, :status, :projection_event_id)"
            ),
            {
                "capsule_id": str(payload.capsule_id),
                "status": payload.status_after.value,
                "projection_event_id": event.event_id,
            },
        )

    async def _apply_retracted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CapsuleRetractedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        require_sharing_event_anchor(event, aggregate_id=payload.capsule_id)
        if (
            event.event_type != CapsuleRetractedV1.event_type
            or event.event_id < 1
            or event.sequence < 2
            or event.actor_agent_id != payload.publisher_agent_id
            or payload.status_before is not CapsuleStatus.ACTIVE
            or payload.status_after is not CapsuleStatus.RETRACTED
        ):
            raise _fail("Retracted capsule event has invalid transition semantics")
        publication = await _validated_received_publication(
            session,
            event_registry=self._event_registry,
            capsule_id=payload.capsule_id,
        )
        if (
            publication.capsule.publisher_agent_id != payload.publisher_agent_id
            or publication.event.event_id >= event.event_id
        ):
            raise _fail("Retracted capsule publication source is inconsistent")
        prior_capsule_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "capsule",
                    DomainEventRow.aggregate_id == payload.capsule_id,
                    DomainEventRow.event_id < event.event_id,
                )
                .order_by(DomainEventRow.sequence)
            )
        ).all()
        if (
            not prior_capsule_rows
            or prior_capsule_rows[0].event_id != publication.event.event_id
            or prior_capsule_rows[-1].sequence != event.sequence - 1
            or event.occurred_at < prior_capsule_rows[-1].occurred_at
            or any(
                row.event_type == CapsuleRetractedV1.event_type
                for row in prior_capsule_rows
            )
        ):
            raise _fail("Retracted capsule aggregate history is inconsistent")
        causal_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == event.causation_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        if (
            tuple(row.event_type for row in causal_rows)
            != (CapsuleRetractedV1.event_type,)
            or causal_rows[0].event_id != event.event_id
        ):
            raise _fail("Retracted capsule command event sequence is invalid")
        target = _target_table(target_prefix, self.name)
        current = (
            await session.execute(
                text(
                    f"SELECT capsule_id, status, projection_event_id FROM {target} "
                    "WHERE capsule_id = :capsule_id"
                ),
                {"capsule_id": str(payload.capsule_id)},
            )
        ).mappings().one_or_none()
        if (
            current is None
            or str(current["status"]) != payload.status_before.value
            or int(current["projection_event_id"]) != publication.event.event_id
        ):
            raise _fail("Retracted capsule before-state is inconsistent")
        result = await session.execute(
            text(
                f"UPDATE {target} SET status = :status, "
                "projection_event_id = :projection_event_id "
                "WHERE capsule_id = :capsule_id AND status = :status_before"
            ),
            {
                "capsule_id": str(payload.capsule_id),
                "status": payload.status_after.value,
                "status_before": payload.status_before.value,
                "projection_event_id": event.event_id,
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("Retracted capsule projection update did not affect one row")


class AgentReputationProjector:
    """Reducer for latest-revision observer-relative publisher reputation."""

    name = "agent_reputation"
    version = 1
    event_types = frozenset({CapsuleFeedbackRecordedV1.event_type})

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        await _validate_feedback_source_correspondence(
            session,
            event_registry=self._event_registry,
        )
        target = _target_table(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "subject_agent_id VARCHAR(36) NOT NULL, "
                "observer_agent_id VARCHAR(36) NOT NULL, "
                "useful_count INTEGER NOT NULL, "
                "refuted_count INTEGER NOT NULL, "
                "harmful_count INTEGER NOT NULL, "
                "alpha INTEGER NOT NULL, "
                "beta INTEGER NOT NULL, "
                "projection_event_id INTEGER NOT NULL, "
                "PRIMARY KEY (subject_agent_id, observer_agent_id), "
                "CHECK (useful_count >= 0 AND refuted_count >= 0 "
                "AND harmful_count >= 0), "
                "CHECK (alpha = 2 + useful_count), "
                "CHECK (beta = 2 + refuted_count + harmful_count), "
                "CHECK (alpha > 0 AND beta > 0 "
                "AND subject_agent_id != observer_agent_id), "
                "CHECK (projection_event_id > 0)"
                ")"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._event_registry, row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        if not isinstance(event.payload, CapsuleFeedbackRecordedV1):
            raise _fail(f"Unsupported reputation event {event.event_type!r}")
        payload = event.payload
        _, previous_verdict = await validate_feedback_source_event(
            session,
            event_registry=self._event_registry,
            event=event,
            payload=payload,
        )
        target = _target_table(target_prefix, self.name)
        current_row = (
            await session.execute(
                text(
                    f"SELECT subject_agent_id, observer_agent_id, useful_count, "
                    f"refuted_count, harmful_count, alpha, beta, "
                    f"projection_event_id FROM {target} "
                    "WHERE subject_agent_id = :subject_agent_id "
                    "AND observer_agent_id = :observer_agent_id"
                ),
                {
                    "subject_agent_id": str(payload.publisher_agent_id),
                    "observer_agent_id": str(payload.observer_agent_id),
                },
            )
        ).mappings().one_or_none()
        current: Reputation | None = None
        current_event_id: int | None = None
        if current_row is not None:
            try:
                current_event_id = int(current_row["projection_event_id"])
                checkpoint_row = await session.get(
                    DomainEventRow,
                    current_event_id,
                )
                checkpoint = (
                    None
                    if checkpoint_row is None
                    else _stored_event(self._event_registry, checkpoint_row)
                )
                current = Reputation(
                    subject_agent_id=UUID(str(current_row["subject_agent_id"])),
                    observer_agent_id=UUID(str(current_row["observer_agent_id"])),
                    useful_count=int(current_row["useful_count"]),
                    refuted_count=int(current_row["refuted_count"]),
                    harmful_count=int(current_row["harmful_count"]),
                    alpha=int(current_row["alpha"]),
                    beta=int(current_row["beta"]),
                    last_feedback_at=(
                        event.occurred_at
                        if checkpoint is None
                        else checkpoint.occurred_at
                    ),
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise _fail("Reputation projection before-state is invalid") from error
            if (
                checkpoint is None
                or not isinstance(
                    checkpoint.payload,
                    CapsuleFeedbackRecordedV1,
                )
                or checkpoint.event_id != current_event_id
                or checkpoint.event_id >= event.event_id
                or checkpoint.payload.publisher_agent_id
                != payload.publisher_agent_id
                or checkpoint.payload.observer_agent_id
                != payload.observer_agent_id
                or checkpoint.payload.alpha_after != current.alpha
                or checkpoint.payload.beta_after != current.beta
            ):
                raise _fail(
                    "Reputation projection checkpoint does not match before-state"
                )
        if target_prefix is None:
            expected, expected_event_id = await _replayed_reputation_before(
                session,
                event_registry=self._event_registry,
                event=event,
                publisher_agent_id=payload.publisher_agent_id,
                observer_agent_id=payload.observer_agent_id,
            )
            if (
                (expected is None) is not (current is None)
                or expected_event_id != current_event_id
                or (
                    expected is not None
                    and current is not None
                    and (
                        expected.subject_agent_id
                        != current.subject_agent_id
                        or expected.observer_agent_id
                        != current.observer_agent_id
                        or expected.useful_count != current.useful_count
                        or expected.refuted_count != current.refuted_count
                        or expected.harmful_count != current.harmful_count
                        or expected.alpha != current.alpha
                        or expected.beta != current.beta
                        or expected.last_feedback_at
                        != current.last_feedback_at
                    )
                )
            ):
                raise _fail(
                    "Reputation projection does not match prior feedback replay"
                )
        revised = reduce_reputation(
            current,
            event=event,
            previous_verdict=previous_verdict,
        )
        values = {
            "subject_agent_id": str(revised.subject_agent_id),
            "observer_agent_id": str(revised.observer_agent_id),
            "useful_count": revised.useful_count,
            "refuted_count": revised.refuted_count,
            "harmful_count": revised.harmful_count,
            "alpha": revised.alpha,
            "beta": revised.beta,
            "projection_event_id": event.event_id,
        }
        if current_row is None:
            await session.execute(
                text(
                    f"INSERT INTO {target} "
                    "(subject_agent_id, observer_agent_id, useful_count, "
                    "refuted_count, harmful_count, alpha, beta, "
                    "projection_event_id) VALUES "
                    "(:subject_agent_id, :observer_agent_id, :useful_count, "
                    ":refuted_count, :harmful_count, :alpha, :beta, "
                    ":projection_event_id)"
                ),
                values,
            )
            return
        result = await session.execute(
            text(
                f"UPDATE {target} SET useful_count = :useful_count, "
                "refuted_count = :refuted_count, "
                "harmful_count = :harmful_count, alpha = :alpha, beta = :beta, "
                "projection_event_id = :projection_event_id "
                "WHERE subject_agent_id = :subject_agent_id "
                "AND observer_agent_id = :observer_agent_id "
                "AND projection_event_id = :current_event_id"
            ),
            {**values, "current_event_id": current_event_id},
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("Reputation projection update did not affect one row")


class InboxItemProjector:
    """Version-one reducer for stable event-allocated inbox identities."""

    name = "inbox_items"
    version = 1
    event_types = frozenset(
        {
            CapsuleReceivedV1.event_type,
            CapsuleAdoptedV1.event_type,
            CapsuleRejectedV1.event_type,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target_table(target_prefix, self.name)
        await session.execute(
            text(
                f"CREATE TEMP TABLE {target} ("
                "item_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "recipient_agent_id VARCHAR(36) NOT NULL, "
                "capsule_id VARCHAR(36) NOT NULL, "
                "state VARCHAR(8) NOT NULL, "
                "projection_event_id INTEGER NOT NULL, "
                "UNIQUE (recipient_agent_id, capsule_id), "
                "CHECK (state IN ('pending', 'adopted', 'rejected')), "
                "CHECK (projection_event_id > 0)"
                ")"
            )
        )
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.event_type.in_(self.event_types))
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(
                session,
                _stored_event(self._event_registry, row),
                target_prefix=target_prefix,
            )
        await SharingAdoptionSourceValidator(
            self._event_registry
        ).validate(session)

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.event_type not in self.event_types:
            return
        if isinstance(event.payload, CapsuleAdoptedV1):
            await self._apply_adopted(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if isinstance(event.payload, CapsuleRejectedV1):
            await self._apply_rejected(
                session,
                event,
                event.payload,
                target_prefix=target_prefix,
            )
            return
        if not isinstance(event.payload, CapsuleReceivedV1):
            raise _fail(f"Unsupported inbox event {event.event_type!r}")
        payload = event.payload
        require_sharing_event_anchor(event, aggregate_id=payload.item_id)
        if (
            event.event_type != CapsuleReceivedV1.event_type
            or event.event_id < 1
            or event.sequence != 1
            or payload.state_after is not InboxState.PENDING
        ):
            raise _fail("Received capsule event has invalid creation semantics")
        try:
            publication = await _validated_received_publication(
                session,
                event_registry=self._event_registry,
                capsule_id=payload.capsule_id,
            )
            capsule = publication.capsule
            publication_event = publication.event
            subscription = await session.scalar(
                select(SubscriptionRow).where(
                    SubscriptionRow.subscriber_agent_id == payload.recipient_agent_id,
                    SubscriptionRow.topic_id == capsule.topic_id,
                    SubscriptionRow.creation_event_id < publication_event.event_id,
                )
            )
            subscription_event_row = (
                None
                if subscription is None
                else await session.get(
                    DomainEventRow,
                    subscription.creation_event_id,
                )
            )
            subscription_event = (
                None
                if subscription_event_row is None
                else _stored_event(
                    self._event_registry,
                    subscription_event_row,
                )
            )
        except SharingProjectionIntegrityError:
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise _fail("Received capsule source anchors are invalid") from error
        if subscription is None:
            raise _fail("Received capsule has no eligible subscription")
        if subscription_event is None or not isinstance(
            subscription_event.payload,
            SubscriptionCreatedV1,
        ):
            raise _fail("Received capsule subscription event is missing")
        subscription_payload = subscription_event.payload
        require_sharing_event_anchor(
            subscription_event,
            aggregate_id=subscription.subscription_id,
        )
        if (
            subscription_event.event_id != subscription.creation_event_id
            or subscription_event.sequence != 1
            or subscription_event.actor_agent_id != subscription.subscriber_agent_id
            or subscription_event.occurred_at != subscription.created_at
            or subscription_payload.subscription_id != subscription.subscription_id
            or subscription_payload.subscriber_agent_id
            != subscription.subscriber_agent_id
            or subscription_payload.topic_id != subscription.topic_id
            or subscription_event.event_id >= publication_event.event_id
        ):
            raise _fail("Received capsule subscription source does not match its event")
        if (
            payload.recipient_agent_id == capsule.publisher_agent_id
            or event.actor_agent_id != capsule.publisher_agent_id
            or event.occurred_at != capsule.created_at
            or event.event_id <= publication_event.event_id
            or event.causation_id != publication_event.causation_id
        ):
            raise _fail("Received capsule event does not match publication")

        target = _target_table(target_prefix, self.name)
        existing = (
            await session.execute(
                text(
                    f"SELECT item_id FROM {target} "
                    "WHERE item_id = :item_id "
                    "OR (recipient_agent_id = :recipient_agent_id "
                    "AND capsule_id = :capsule_id) LIMIT 1"
                ),
                {
                    "item_id": str(payload.item_id),
                    "recipient_agent_id": str(payload.recipient_agent_id),
                    "capsule_id": str(payload.capsule_id),
                },
            )
        ).one_or_none()
        if existing is not None:
            raise _fail("Received inbox projection already exists")
        await session.execute(
            text(
                f"INSERT INTO {target} "
                "(item_id, recipient_agent_id, capsule_id, state, "
                "projection_event_id) VALUES "
                "(:item_id, :recipient_agent_id, :capsule_id, :state, "
                ":projection_event_id)"
            ),
            {
                "item_id": str(payload.item_id),
                "recipient_agent_id": str(payload.recipient_agent_id),
                "capsule_id": str(payload.capsule_id),
                "state": payload.state_after.value,
                "projection_event_id": event.event_id,
            },
        )

    async def _apply_rejected(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CapsuleRejectedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        require_sharing_event_anchor(event, aggregate_id=payload.item_id)
        if (
            event.event_type != CapsuleRejectedV1.event_type
            or event.event_id < 1
            or event.sequence != 2
            or event.actor_agent_id != payload.recipient_agent_id
            or payload.state_before is not InboxState.PENDING
            or payload.state_after is not InboxState.REJECTED
        ):
            raise _fail("Rejected capsule event has invalid transition semantics")
        target = _target_table(target_prefix, self.name)
        current = (
            await session.execute(
                text(
                    f"SELECT item_id, recipient_agent_id, capsule_id, state, "
                    f"projection_event_id FROM {target} "
                    "WHERE item_id = :item_id"
                ),
                {"item_id": str(payload.item_id)},
            )
        ).mappings().one_or_none()
        if current is None:
            raise _fail("Rejected inbox projection row is missing")
        prior_event_id = int(current["projection_event_id"])
        prior_row = await session.get(DomainEventRow, prior_event_id)
        if prior_row is None:
            raise _fail("Rejected inbox before-state event is missing")
        prior_event = _stored_event(self._event_registry, prior_row)
        if (
            str(current["recipient_agent_id"]) != str(payload.recipient_agent_id)
            or str(current["capsule_id"]) != str(payload.capsule_id)
            or str(current["state"]) != payload.state_before.value
            or not isinstance(prior_event.payload, CapsuleReceivedV1)
            or prior_event.event_id != prior_event_id
            or prior_event.sequence != 1
            or prior_event.aggregate_id != payload.item_id
            or prior_event.payload.item_id != payload.item_id
            or prior_event.payload.capsule_id != payload.capsule_id
            or prior_event.payload.recipient_agent_id
            != payload.recipient_agent_id
            or prior_event.event_id >= event.event_id
        ):
            raise _fail("Rejected capsule before-state is inconsistent")
        publication = await _validated_received_publication(
            session,
            event_registry=self._event_registry,
            capsule_id=payload.capsule_id,
        )
        latest_capsule_row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == payload.capsule_id,
                DomainEventRow.event_id < event.event_id,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        if (
            publication.event.event_id >= event.event_id
            or latest_capsule_row is None
            or event.occurred_at < prior_event.occurred_at
            or event.occurred_at < latest_capsule_row.occurred_at
        ):
            raise _fail("Rejected capsule causal clock is inconsistent")
        causal_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == event.causation_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        if (
            tuple(row.event_type for row in causal_rows)
            != (CapsuleRejectedV1.event_type,)
            or causal_rows[0].event_id != event.event_id
        ):
            raise _fail("Rejected capsule command event sequence is invalid")
        result = await session.execute(
            text(
                f"UPDATE {target} SET state = :state, "
                "projection_event_id = :projection_event_id "
                "WHERE item_id = :item_id AND state = :state_before"
            ),
            {
                "item_id": str(payload.item_id),
                "state": payload.state_after.value,
                "state_before": payload.state_before.value,
                "projection_event_id": event.event_id,
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("Rejected inbox projection update did not affect one row")

    async def _apply_adopted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CapsuleAdoptedV1,
        *,
        target_prefix: str | None,
    ) -> None:
        require_sharing_event_anchor(event, aggregate_id=payload.item_id)
        if (
            event.event_type != CapsuleAdoptedV1.event_type
            or event.event_id < 1
            or event.sequence != 2
            or event.actor_agent_id != payload.adopter_agent_id
            or payload.state_before is not InboxState.PENDING
            or payload.state_after is not InboxState.ADOPTED
        ):
            raise _fail("Adopted capsule event has invalid transition semantics")
        target = _target_table(target_prefix, self.name)
        current = (
            await session.execute(
                text(
                    f"SELECT item_id, recipient_agent_id, capsule_id, state, "
                    f"projection_event_id FROM {target} "
                    "WHERE item_id = :item_id"
                ),
                {"item_id": str(payload.item_id)},
            )
        ).mappings().one_or_none()
        if current is None:
            raise _fail("Adopted inbox projection row is missing")
        try:
            prior_event_id = int(current["projection_event_id"])
            prior_row = await session.get(DomainEventRow, prior_event_id)
            prior_event = (
                None
                if prior_row is None
                else _stored_event(self._event_registry, prior_row)
            )
            adoption = await session.get(
                AdoptionRecordRow,
                payload.adoption_id,
            )
            identity = await session.get(
                ExperienceRow,
                payload.resulting_experience_id,
            )
            publication = await _validated_received_publication(
                session,
                event_registry=self._event_registry,
                capsule_id=payload.capsule_id,
            )
        except SharingProjectionIntegrityError:
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise _fail("Adopted capsule source anchors are invalid") from error
        capsule = publication.capsule
        prior_retraction = await session.scalar(
            select(DomainEventRow.event_id)
            .where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == payload.capsule_id,
                DomainEventRow.event_type == CapsuleRetractedV1.event_type,
                DomainEventRow.event_id < event.event_id,
            )
            .limit(1)
        )
        if prior_retraction is not None:
            raise _fail("Adopted capsule was retracted before adoption")
        if (
            str(current["recipient_agent_id"]) != str(payload.adopter_agent_id)
            or str(current["capsule_id"]) != str(payload.capsule_id)
            or str(current["state"]) != InboxState.PENDING.value
            or prior_event is None
            or not isinstance(prior_event.payload, CapsuleReceivedV1)
            or prior_event.event_id != prior_event_id
            or prior_event.sequence != 1
            or prior_event.aggregate_id != payload.item_id
            or prior_event.payload.item_id != payload.item_id
            or prior_event.payload.capsule_id != payload.capsule_id
            or prior_event.payload.recipient_agent_id
            != payload.adopter_agent_id
            or prior_event.event_id >= event.event_id
        ):
            raise _fail("Adopted capsule before-state is inconsistent")
        if adoption is None or identity is None:
            raise _fail("Adopted capsule provenance source is missing")
        expected_chain = (
            *capsule.provenance_chain,
            ProvenanceHop(
                capsule_id=capsule.capsule_id,
                publisher_agent_id=capsule.publisher_agent_id,
            ),
        )
        if (
            adoption.adopter_agent_id != payload.adopter_agent_id
            or adoption.capsule_id != payload.capsule_id
            or adoption.resulting_experience_id
            != payload.resulting_experience_id
            or adoption.root_fingerprint != payload.root_fingerprint
            or adoption.root_fingerprint != capsule.root_fingerprint
            or adoption.corroboration_applied
            is not payload.corroboration_applied
            or adoption.adopted_at != event.occurred_at
            or canonical_json_bytes(expected_chain)
            != adoption.provenance_chain
            or identity.owner_agent_id != payload.adopter_agent_id
            or identity.created_at > event.occurred_at
            or event.occurred_at < capsule.created_at
            or event.occurred_at >= capsule.expires_at
        ):
            raise _fail("Adopted capsule provenance source is inconsistent")
        prior_matching_root = await _has_prior_matching_adoption_event(
            session,
            event_registry=self._event_registry,
            event=event,
            resulting_experience_id=payload.resulting_experience_id,
            root_fingerprint=payload.root_fingerprint,
        )
        if (
            (payload.created and prior_matching_root)
            or (
                not payload.created
                and payload.corroboration_applied is prior_matching_root
            )
        ):
            raise _fail("Adopted capsule root contribution is inconsistent")
        if payload.created:
            if (
                identity.origin is not ExperienceOrigin.ADOPTED_CAPSULE
                or identity.created_at != event.occurred_at
            ):
                raise _fail("New adopted experience source is inconsistent")
            created_events = (
                await session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.aggregate_type == "experience",
                        DomainEventRow.aggregate_id
                        == payload.resulting_experience_id,
                        DomainEventRow.causation_id == event.causation_id,
                        DomainEventRow.event_type.in_(
                            {
                                "experience.created",
                                "experience.version_created",
                            }
                        ),
                        DomainEventRow.event_id < event.event_id,
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
            if {row.event_type for row in created_events} != {
                "experience.created",
                "experience.version_created",
            }:
                raise _fail("New adopted experience events are incomplete")
            try:
                created_payload = self._event_registry.decode(
                    event_type=created_events[0].event_type,
                    payload=created_events[0].payload,
                )
                version_payload = self._event_registry.decode(
                    event_type=created_events[1].event_type,
                    payload=created_events[1].payload,
                )
            except (IndexError, TypeError, ValueError) as error:
                raise _fail(
                    "New adopted experience payloads are invalid"
                ) from error
            try:
                expected_confidence = initial_adoption_confidence(
                    capsule.publisher_confidence,
                    adoption.captured_trust,
                )
            except ValueError as error:
                raise _fail(
                    "New adopted experience confidence inputs are invalid"
                ) from error
            if (
                not isinstance(created_payload, ExperienceCreatedV1)
                or not isinstance(
                    version_payload,
                    ExperienceVersionCreatedV1,
                )
                or created_payload.experience_id
                != payload.resulting_experience_id
                or version_payload.experience_id
                != payload.resulting_experience_id
                or created_payload.after.owner_agent_id
                != payload.adopter_agent_id
                or created_payload.after.current_content_hash
                != capsule.source_content_hash
                or created_payload.after.temperature is not Temperature.HOT
                or abs(
                    created_payload.after.source_trust
                    - adoption.captured_trust
                )
                > 1e-12
                or abs(
                    created_payload.after.confidence
                    - expected_confidence
                )
                > 1e-12
                or version_payload.before != created_payload.after
                or version_payload.after != created_payload.after
                or version_payload.links
                or identity.kind is not capsule.kind
                or any(
                    row.actor_agent_id != payload.adopter_agent_id
                    or row.occurred_at != event.occurred_at
                    for row in created_events
                )
            ):
                raise _fail("New adopted experience source is inconsistent")
        corroborated_payload: ExperienceCorroboratedV1 | None = None
        if payload.corroboration_applied:
            corroborated = await session.scalar(
                select(DomainEventRow).where(
                    DomainEventRow.aggregate_type == "experience",
                    DomainEventRow.aggregate_id
                    == payload.resulting_experience_id,
                    DomainEventRow.event_type == "experience.corroborated",
                    DomainEventRow.causation_id == event.causation_id,
                    DomainEventRow.event_id < event.event_id,
                )
            )
            if corroborated is None:
                raise _fail("Adopted capsule corroboration event is missing")
            try:
                decoded_corroboration = self._event_registry.decode(
                    event_type=corroborated.event_type,
                    payload=corroborated.payload,
                )
            except (TypeError, ValueError) as error:
                raise _fail(
                    "Adopted capsule corroboration payload is invalid"
                ) from error
            if not isinstance(
                decoded_corroboration,
                ExperienceCorroboratedV1,
            ):
                raise _fail(
                    "Adopted capsule corroboration source is inconsistent"
                )
            corroborated_payload = decoded_corroboration
            if (
                corroborated_payload.experience_id
                != payload.resulting_experience_id
                or corroborated_payload.adoption_id != payload.adoption_id
                or corroborated_payload.capsule_id != payload.capsule_id
                or corroborated_payload.root_fingerprint
                != payload.root_fingerprint
                or abs(
                    corroborated_payload.captured_trust
                    - adoption.captured_trust
                )
                > 1e-12
                or corroborated.occurred_at != event.occurred_at
                or corroborated.actor_agent_id != payload.adopter_agent_id
            ):
                raise _fail(
                    "Adopted capsule corroboration source is inconsistent"
                )
        elif not payload.created:
            await _require_equivalent_experience_before_adoption(
                session,
                event_registry=self._event_registry,
                event=event,
                experience_id=payload.resulting_experience_id,
                owner_agent_id=payload.adopter_agent_id,
                content_hash=capsule.source_content_hash,
            )
        causal_rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(DomainEventRow.causation_id == event.causation_id)
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        causal_types = tuple(row.event_type for row in causal_rows)
        expected_sequences: tuple[tuple[str, ...], ...]
        if payload.created:
            expected_sequences = (
                (
                    "experience.created",
                    "experience.version_created",
                    "capsule.adopted",
                ),
            )
        elif payload.corroboration_applied:
            if corroborated_payload is None:
                raise _fail("Adopted capsule corroboration source is missing")
            if corroborated_payload.before.temperature is Temperature.COLD:
                expected_sequences = (
                    (
                        "experience.corroborated",
                        "experience.temperature_changed",
                        "capsule.adopted",
                    ),
                )
            else:
                expected_sequences = (
                    (
                        "experience.corroborated",
                        "capsule.adopted",
                    ),
                )
        else:
            expected_sequences = (("capsule.adopted",),)
        if causal_types not in expected_sequences:
            raise _fail("Adopted capsule command event sequence is invalid")
        if (
            len(causal_rows) < 1
            or causal_rows[-1].event_id != event.event_id
            or causal_rows[-1].aggregate_id != payload.item_id
        ):
            raise _fail("Adopted capsule command event order is invalid")
        if len(causal_rows) == 3 and payload.corroboration_applied:
            if corroborated_payload is None:
                raise _fail("Adopted capsule corroboration source is missing")
            try:
                decoded_transition = self._event_registry.decode(
                    event_type=causal_rows[1].event_type,
                    payload=causal_rows[1].payload,
                )
            except (TypeError, ValueError) as error:
                raise _fail(
                    "Adopted capsule promotion payload is invalid"
                ) from error
            if not isinstance(
                decoded_transition,
                ExperienceTemperatureChangedV1,
            ):
                raise _fail("Adopted capsule promotion event is inconsistent")
            transition = decoded_transition
            if (
                transition.cause != "capsule_corroboration"
                or transition.experience_id
                != payload.resulting_experience_id
                or transition.before != corroborated_payload.after
                or transition.after.temperature is not Temperature.HOT
                or causal_rows[1].actor_agent_id
                != payload.adopter_agent_id
                or causal_rows[1].occurred_at != event.occurred_at
            ):
                raise _fail("Adopted capsule promotion event is inconsistent")
        result = await session.execute(
            text(
                f"UPDATE {target} SET state = :state, "
                "projection_event_id = :projection_event_id "
                "WHERE item_id = :item_id AND state = :state_before"
            ),
            {
                "item_id": str(payload.item_id),
                "state": payload.state_after.value,
                "state_before": payload.state_before.value,
                "projection_event_id": event.event_id,
            },
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("Adopted inbox projection update did not affect one row")


__all__ = [
    "AgentReputationProjector",
    "CapsuleStateProjector",
    "InboxItemProjector",
    "SharingProjectionIntegrityError",
    "reduce_reputation",
    "replay_reputation",
    "require_sharing_event_anchor",
]
