"""Authoritative source validation for safe social experience propagation."""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    EventRegistry,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences.events import (
    ExperienceCorroboratedV1,
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.sharing.confidence import initial_adoption_confidence
from experience_hub.sharing.events import (
    SHARING_EVENT_AGGREGATE_TYPES,
    SHARING_EVENT_TYPES,
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    SubscriptionCreatedV1,
    TopicCreatedV1,
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
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    CapsuleFeedbackRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
    SubscriptionRow,
    TopicRow,
)
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
)


class SharingAdoptionSourceValidator:
    """Prove a bijection and root-order policy for adoption sources."""

    name = "sharing_adoptions"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        adoptions = tuple(
            (
                await session.scalars(
                    select(AdoptionRecordRow).order_by(AdoptionRecordRow.adoption_id)
                )
            ).all()
        )
        adoption_by_id = {row.adoption_id: row for row in adoptions}
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == CapsuleAdoptedV1.event_type)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        event_by_adoption: dict[
            UUID,
            tuple[DomainEventRow, CapsuleAdoptedV1],
        ] = {}
        seen_roots: set[tuple[UUID, str]] = set()
        for event in events:
            try:
                decoded = self._event_registry.decode(
                    event_type=event.event_type,
                    payload=event.payload,
                )
            except (TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    "Capsule adoption event cannot be decoded",
                    mismatch_key=f"adoption_event:{event.event_id}",
                ) from error
            if not isinstance(decoded, CapsuleAdoptedV1):
                raise SourceIntegrityError(
                    "Capsule adoption event decoded to the wrong schema",
                    mismatch_key=f"adoption_event:{event.event_id}",
                )
            row = adoption_by_id.get(decoded.adoption_id)
            key = f"adoption:{decoded.adoption_id}"
            if (
                row is None
                or decoded.adoption_id in event_by_adoption
                or event.aggregate_type != "inbox_item"
                or event.aggregate_id != decoded.item_id
                or event.sequence != 2
                or event.actor_agent_id != decoded.adopter_agent_id
                or event.occurred_at != row.adopted_at
                or decoded.state_before is not InboxState.PENDING
                or decoded.state_after is not InboxState.ADOPTED
                or decoded.adopter_agent_id != row.adopter_agent_id
                or decoded.capsule_id != row.capsule_id
                or decoded.resulting_experience_id != row.resulting_experience_id
                or decoded.root_fingerprint != row.root_fingerprint
                or decoded.corroboration_applied is not row.corroboration_applied
            ):
                raise SourceIntegrityError(
                    "Adoption row does not match exactly one adoption event",
                    mismatch_key=key,
                )
            root_key = (
                decoded.resulting_experience_id,
                decoded.root_fingerprint,
            )
            root_was_seen = root_key in seen_roots
            if (decoded.created and root_was_seen) or (
                not decoded.created and decoded.corroboration_applied is root_was_seen
            ):
                raise SourceIntegrityError(
                    "Adoption root contribution disagrees with event order",
                    mismatch_key=key,
                )
            seen_roots.add(root_key)
            event_by_adoption[decoded.adoption_id] = (event, decoded)

        unmatched = tuple(
            sorted(
                set(adoption_by_id) ^ set(event_by_adoption),
                key=lambda value: value.bytes,
            )
        )
        if unmatched:
            raise SourceIntegrityError(
                "Every adoption row requires exactly one capsule.adopted event",
                mismatch_key=f"adoption:{unmatched[0]}",
            )


type _DecodedEvent = tuple[DomainEventRow, Any]
type _ReceivedEvent = tuple[DomainEventRow, CapsuleReceivedV1]
type _TerminalEvent = tuple[
    DomainEventRow,
    CapsuleAdoptedV1 | CapsuleRejectedV1,
]
type _AdoptionAnchor = tuple[tuple[ProvenanceHop, ...], DomainEventRow]

_SHARING_AGGREGATE_TYPES = frozenset(SHARING_EVENT_AGGREGATE_TYPES.values())


def _ordered_uuid_rows(rows: tuple[Any, ...], attribute: str) -> tuple[Any, ...]:
    return tuple(sorted(rows, key=lambda row: getattr(row, attribute).bytes))


def _canonical_array(raw: bytes, *, label: str) -> tuple[Any, ...]:
    try:
        values = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(values, list) or canonical_json_bytes(values) != bytes(raw):
        raise ValueError(f"{label} must be a canonical JSON array")
    return tuple(values)


def _decode_provenance(
    raw: bytes,
    *,
    label: str,
    allow_empty: bool,
) -> tuple[ProvenanceHop, ...]:
    values = _canonical_array(raw, label=f"{label} provenance")
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
        hop = ProvenanceHop(
            capsule_id=UUID(capsule_id),
            publisher_agent_id=UUID(publisher_agent_id),
        )
        if (
            str(hop.capsule_id) != capsule_id
            or str(hop.publisher_agent_id) != publisher_agent_id
        ):
            raise ValueError(f"{label} provenance UUIDs must be canonical")
        hops.append(hop)
    result = tuple(hops)
    if canonical_json_bytes(result) != bytes(raw):
        raise ValueError(f"{label} provenance must use canonical values")
    return result


class SharingSourceValidator:
    """Prove the complete immutable sharing graph before any replay starts."""

    name = "sharing_graph"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        topics = tuple(
            (await session.scalars(select(TopicRow).order_by(TopicRow.topic_id))).all()
        )
        subscriptions = tuple(
            (
                await session.scalars(
                    select(SubscriptionRow).order_by(SubscriptionRow.subscription_id)
                )
            ).all()
        )
        capsule_rows = tuple(
            (
                await session.scalars(
                    select(ExperienceCapsuleRow).order_by(
                        ExperienceCapsuleRow.capsule_id
                    )
                )
            ).all()
        )
        adoption_rows = tuple(
            (
                await session.scalars(
                    select(AdoptionRecordRow).order_by(AdoptionRecordRow.adoption_id)
                )
            ).all()
        )
        feedback_rows = tuple(
            (
                await session.scalars(
                    select(CapsuleFeedbackRow).order_by(
                        CapsuleFeedbackRow.observer_agent_id,
                        CapsuleFeedbackRow.capsule_id,
                        CapsuleFeedbackRow.revision,
                        CapsuleFeedbackRow.feedback_id,
                    )
                )
            ).all()
        )
        identities = tuple(
            (
                await session.scalars(
                    select(ExperienceRow).order_by(ExperienceRow.experience_id)
                )
            ).all()
        )
        versions = tuple(
            (
                await session.scalars(
                    select(ExperienceVersionRow).order_by(
                        ExperienceVersionRow.experience_id,
                        ExperienceVersionRow.version_number,
                        ExperienceVersionRow.version_id,
                    )
                )
            ).all()
        )
        payloads = tuple(
            (
                await session.scalars(
                    select(ExperiencePayloadRow).order_by(
                        ExperiencePayloadRow.version_id
                    )
                )
            ).all()
        )
        all_event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        receipt_rows = tuple(
            (
                await session.scalars(
                    select(IdempotencyRecordRow).order_by(
                        IdempotencyRecordRow.receipt_id
                    )
                )
            ).all()
        )
        self._validate_aggregate_namespace(all_event_rows)
        event_rows = tuple(
            row for row in all_event_rows if row.event_type in SHARING_EVENT_TYPES
        )
        events = self._decode_events(event_rows)
        events_by_type: dict[str, list[_DecodedEvent]] = defaultdict(list)
        for event, payload in events:
            events_by_type[event.event_type].append((event, payload))
        events_by_causation: dict[UUID, list[DomainEventRow]] = defaultdict(list)
        events_by_aggregate: dict[
            tuple[str, UUID],
            list[DomainEventRow],
        ] = defaultdict(list)
        event_ids_by_aggregate: dict[tuple[str, UUID], list[int]] = defaultdict(list)
        experience_events: dict[UUID, list[DomainEventRow]] = defaultdict(list)
        for event in all_event_rows:
            events_by_causation[event.causation_id].append(event)
            aggregate_key = (event.aggregate_type, event.aggregate_id)
            events_by_aggregate[aggregate_key].append(event)
            event_ids_by_aggregate[aggregate_key].append(event.event_id)
            if event.aggregate_type == "experience":
                experience_events[event.aggregate_id].append(event)

        topic_anchors = self._validate_topics(topics, events_by_type)
        self._validate_subscriptions(
            subscriptions,
            events_by_type,
            topic_anchors=topic_anchors,
        )
        capsules, publications = self._validate_capsules(
            capsule_rows=capsule_rows,
            identities=identities,
            versions=versions,
            payloads=payloads,
            events_by_type=events_by_type,
            topic_anchors=topic_anchors,
        )
        received_by_item = self._validate_deliveries(
            subscriptions=subscriptions,
            capsules=capsules,
            publications=publications,
            events_by_type=events_by_type,
            events_by_causation=events_by_causation,
        )
        terminal_by_item = self._validate_terminal_events(
            capsules=capsules,
            received_by_item=received_by_item,
            events_by_type=events_by_type,
            events_by_causation=events_by_causation,
            events_by_aggregate=events_by_aggregate,
            event_ids_by_aggregate=event_ids_by_aggregate,
        )
        trust_history = self._validate_feedback(
            rows=feedback_rows,
            capsules=capsules,
            terminal_by_item=terminal_by_item,
            events_by_type=events_by_type,
            events_by_causation=events_by_causation,
        )
        adoption_anchors = self._validate_adoption_references(
            rows=adoption_rows,
            capsules=capsules,
            identities=identities,
            versions=versions,
            received_by_item=received_by_item,
            terminal_by_item=terminal_by_item,
            trust_history=trust_history,
            events_by_type=events_by_type,
            events_by_causation=events_by_causation,
            experience_events=experience_events,
        )
        self._validate_provenance(
            capsules=capsules,
            publications=publications,
            adoption_rows=adoption_rows,
            adoption_anchors=adoption_anchors,
        )
        self._validate_aggregate_clocks(event_rows)
        self._validate_command_receipts(
            events=events,
            receipts={
                receipt.receipt_id: receipt for receipt in receipt_rows
            },
            events_by_causation=events_by_causation,
        )

    @staticmethod
    def _validate_aggregate_namespace(
        rows: tuple[DomainEventRow, ...],
    ) -> None:
        for row in rows:
            expected_aggregate_type = SHARING_EVENT_AGGREGATE_TYPES.get(
                row.event_type
            )
            occupies_sharing_namespace = (
                row.aggregate_type in _SHARING_AGGREGATE_TYPES
            )
            if (
                expected_aggregate_type is not None
                or occupies_sharing_namespace
            ) and row.aggregate_type != expected_aggregate_type:
                raise SourceIntegrityError(
                    "Sharing aggregate namespace contains an incompatible event",
                    mismatch_key=(
                        f"sharing_aggregate:{row.aggregate_type}:"
                        f"{row.aggregate_id}:event:{row.event_id}"
                    ),
                )

    def _decode_events(
        self,
        rows: tuple[DomainEventRow, ...],
    ) -> tuple[_DecodedEvent, ...]:
        decoded: list[_DecodedEvent] = []
        for row in rows:
            try:
                payload = self._event_registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                )
            except (TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    "Sharing source event cannot be decoded",
                    mismatch_key=f"sharing_event:{row.event_id}",
                ) from error
            decoded.append((row, payload))
        return tuple(decoded)

    @staticmethod
    def _validate_aggregate_clocks(
        rows: tuple[DomainEventRow, ...],
    ) -> None:
        latest_by_aggregate: dict[tuple[str, UUID], DomainEventRow] = {}
        for row in rows:
            aggregate_key = (row.aggregate_type, row.aggregate_id)
            previous = latest_by_aggregate.get(aggregate_key)
            if previous is not None and row.occurred_at < previous.occurred_at:
                raise SourceIntegrityError(
                    "Sharing aggregate clock regresses in ledger order",
                    mismatch_key=f"{row.aggregate_type}:{row.aggregate_id}",
                )
            latest_by_aggregate[aggregate_key] = row

    def _validate_topics(
        self,
        rows: tuple[TopicRow, ...],
        events_by_type: dict[str, list[_DecodedEvent]],
    ) -> dict[UUID, tuple[TopicRow, DomainEventRow]]:
        events = events_by_type.get(TopicCreatedV1.event_type, [])
        by_id: dict[UUID, list[tuple[DomainEventRow, TopicCreatedV1]]] = defaultdict(
            list
        )
        for event, payload in events:
            if not isinstance(payload, TopicCreatedV1):
                raise SourceIntegrityError(
                    "Topic event decoded to the wrong schema",
                    mismatch_key=f"topic_event:{event.event_id}",
                )
            by_id[payload.topic_id].append((event, payload))

        row_ids = {row.topic_id for row in rows}
        anchors: dict[UUID, tuple[TopicRow, DomainEventRow]] = {}
        for row in _ordered_uuid_rows(rows, "topic_id"):
            key = f"topic:{row.topic_id}"
            matches = by_id.get(row.topic_id, [])
            if len(matches) != 1:
                raise SourceIntegrityError(
                    "Topic source requires exactly one creation event",
                    mismatch_key=key,
                )
            event, payload = matches[0]
            if (
                event.aggregate_type != "topic"
                or event.aggregate_id != row.topic_id
                or event.sequence != 1
                or event.actor_agent_id != row.owner_agent_id
                or event.occurred_at != row.created_at
                or payload.owner_agent_id != row.owner_agent_id
                or payload.name != row.name
                or payload.description != row.description
            ):
                raise SourceIntegrityError(
                    "Topic row and creation event disagree",
                    mismatch_key=key,
                )
            anchors[row.topic_id] = (row, event)
        extras = sorted(set(by_id) - row_ids, key=lambda value: value.bytes)
        if extras:
            raise SourceIntegrityError(
                "Topic creation event has no source row",
                mismatch_key=f"topic:{extras[0]}",
            )
        return anchors

    def _validate_subscriptions(
        self,
        rows: tuple[SubscriptionRow, ...],
        events_by_type: dict[str, list[_DecodedEvent]],
        *,
        topic_anchors: dict[UUID, tuple[TopicRow, DomainEventRow]],
    ) -> None:
        events = events_by_type.get(SubscriptionCreatedV1.event_type, [])
        by_id: dict[
            UUID,
            list[tuple[DomainEventRow, SubscriptionCreatedV1]],
        ] = defaultdict(list)
        for event, payload in events:
            if not isinstance(payload, SubscriptionCreatedV1):
                raise SourceIntegrityError(
                    "Subscription event decoded to the wrong schema",
                    mismatch_key=f"subscription_event:{event.event_id}",
                )
            by_id[payload.subscription_id].append((event, payload))

        row_ids = {row.subscription_id for row in rows}
        for row in _ordered_uuid_rows(rows, "subscription_id"):
            key = f"subscription:{row.subscription_id}"
            matches = by_id.get(row.subscription_id, [])
            if len(matches) != 1:
                raise SourceIntegrityError(
                    "Subscription source requires exactly one creation event",
                    mismatch_key=key,
                )
            event, payload = matches[0]
            topic_anchor = topic_anchors.get(row.topic_id)
            if (
                topic_anchor is None
                or topic_anchor[0].created_at > row.created_at
                or topic_anchor[1].event_id >= event.event_id
                or event.event_id != row.creation_event_id
                or event.aggregate_type != "subscription"
                or event.aggregate_id != row.subscription_id
                or event.sequence != 1
                or event.actor_agent_id != row.subscriber_agent_id
                or event.occurred_at != row.created_at
                or payload.subscriber_agent_id != row.subscriber_agent_id
                or payload.topic_id != row.topic_id
            ):
                raise SourceIntegrityError(
                    "Subscription row and creation event disagree",
                    mismatch_key=key,
                )
        extras = sorted(set(by_id) - row_ids, key=lambda value: value.bytes)
        if extras:
            raise SourceIntegrityError(
                "Subscription creation event has no source row",
                mismatch_key=f"subscription:{extras[0]}",
            )

    def _validate_capsules(
        self,
        *,
        capsule_rows: tuple[ExperienceCapsuleRow, ...],
        identities: tuple[ExperienceRow, ...],
        versions: tuple[ExperienceVersionRow, ...],
        payloads: tuple[ExperiencePayloadRow, ...],
        events_by_type: dict[str, list[_DecodedEvent]],
        topic_anchors: dict[UUID, tuple[TopicRow, DomainEventRow]],
    ) -> tuple[
        dict[UUID, Capsule],
        dict[UUID, DomainEventRow],
    ]:
        identity_by_id = {row.experience_id: row for row in identities}
        version_by_id = {row.version_id: row for row in versions}
        payload_by_id = {row.version_id: row for row in payloads}
        events = events_by_type.get(CapsulePublishedV1.event_type, [])
        by_id: dict[
            UUID,
            list[tuple[DomainEventRow, CapsulePublishedV1]],
        ] = defaultdict(list)
        for event, payload in events:
            if not isinstance(payload, CapsulePublishedV1):
                raise SourceIntegrityError(
                    "Capsule publication decoded to the wrong schema",
                    mismatch_key=f"capsule_event:{event.event_id}",
                )
            by_id[payload.capsule_id].append((event, payload))

        capsules: dict[UUID, Capsule] = {}
        publications: dict[UUID, DomainEventRow] = {}
        row_ids = {row.capsule_id for row in capsule_rows}
        for row in _ordered_uuid_rows(capsule_rows, "capsule_id"):
            key = f"capsule:{row.capsule_id}"
            matches = by_id.get(row.capsule_id, [])
            if len(matches) != 1:
                raise SourceIntegrityError(
                    "Capsule source requires exactly one publication event",
                    mismatch_key=key,
                )
            event, published = matches[0]
            try:
                capsule = self._capsule_from_row(row)
                identity = identity_by_id.get(row.source_experience_id)
                version = version_by_id.get(row.source_version_id)
                payload = payload_by_id.get(row.source_version_id)
                if identity is None or version is None or payload is None:
                    raise ValueError("capsule source experience version is missing")
                content = decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=payload,
                )
                if (
                    identity.owner_agent_id != row.publisher_agent_id
                    or identity.kind != row.kind
                    or identity.created_at > row.created_at
                    or version.experience_id != row.source_experience_id
                    or version.created_at > row.created_at
                    or version.content_hash != row.source_content_hash
                    or content.body != row.body
                    or content.summary != row.summary
                    or content.mechanism != row.mechanism
                    or content.tags != capsule.tags
                    or content.applicability != capsule.applicability
                    or content.evidence != capsule.evidence
                    or content.falsifiers != capsule.falsifiers
                    or (
                        identity.origin is ExperienceOrigin.ADOPTED_CAPSULE
                        and not capsule.provenance_chain
                    )
                ):
                    raise ValueError(
                        "capsule does not match its immutable source version"
                    )
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
                if expected_hash != row.capsule_hash:
                    raise ValueError("capsule transport hash is invalid")
                if not capsule.provenance_chain:
                    expected_root = compute_original_root_fingerprint(
                        root_publisher_id=capsule.publisher_agent_id,
                        source_content_hash=capsule.source_content_hash,
                    )
                    if expected_root != capsule.root_fingerprint:
                        raise ValueError("original capsule root fingerprint is invalid")
            except (
                SourceIntegrityError,
                TypeError,
                ValidationError,
                ValueError,
            ) as error:
                raise SourceIntegrityError(
                    "Capsule source content or hashes are invalid",
                    mismatch_key=key,
                ) from error
            if (
                event.aggregate_type != "capsule"
                or event.aggregate_id != row.capsule_id
                or event.sequence != 1
                or event.actor_agent_id != row.publisher_agent_id
                or event.occurred_at != row.created_at
                or published.topic_id != row.topic_id
                or published.source_experience_id != row.source_experience_id
                or published.source_version_id != row.source_version_id
                or published.publisher_agent_id != row.publisher_agent_id
                or published.capsule_hash != row.capsule_hash
                or published.root_fingerprint != row.root_fingerprint
                or published.status_after is not CapsuleStatus.ACTIVE
            ):
                raise SourceIntegrityError(
                    "Capsule row and publication event disagree",
                    mismatch_key=key,
                )
            topic_anchor = topic_anchors.get(row.topic_id)
            if (
                topic_anchor is None
                or topic_anchor[0].created_at > row.created_at
                or topic_anchor[1].event_id >= event.event_id
            ):
                raise SourceIntegrityError(
                    "Capsule publication does not follow topic creation",
                    mismatch_key=key,
                )
            capsules[row.capsule_id] = capsule
            publications[row.capsule_id] = event
        extras = sorted(set(by_id) - row_ids, key=lambda value: value.bytes)
        if extras:
            raise SourceIntegrityError(
                "Capsule publication event has no source row",
                mismatch_key=f"capsule:{extras[0]}",
            )
        return capsules, publications

    @staticmethod
    def _capsule_from_row(row: ExperienceCapsuleRow) -> Capsule:
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
        capsule = Capsule(
            capsule_id=row.capsule_id,
            transport_schema_version=cast(Any, row.transport_schema_version),
            topic_id=row.topic_id,
            source_experience_id=row.source_experience_id,
            source_version_id=row.source_version_id,
            publisher_agent_id=row.publisher_agent_id,
            kind=row.kind,
            body=row.body,
            summary=row.summary,
            mechanism=row.mechanism,
            tags=cast(Any, tags),
            applicability=cast(Any, applicability),
            evidence=evidence,
            falsifiers=cast(Any, falsifiers),
            publisher_confidence=row.publisher_confidence,
            provenance_chain=provenance,
            root_fingerprint=row.root_fingerprint,
            source_content_hash=row.source_content_hash,
            created_at=row.created_at,
            expires_at=row.expires_at,
            hop_count=row.hop_count,
            capsule_hash=row.capsule_hash,
            status=CapsuleStatus.ACTIVE,
            last_transition_at=row.created_at,
        )
        if (
            canonical_json_bytes(capsule.tags) != row.tags
            or canonical_json_bytes(capsule.applicability) != row.applicability
            or canonical_json_bytes(capsule.evidence) != row.evidence
            or canonical_json_bytes(capsule.falsifiers) != row.falsifiers
            or canonical_json_bytes(capsule.provenance_chain) != row.provenance_chain
        ):
            raise ValueError("capsule arrays are not canonical")
        return capsule

    def _validate_deliveries(
        self,
        *,
        subscriptions: tuple[SubscriptionRow, ...],
        capsules: dict[UUID, Capsule],
        publications: dict[UUID, DomainEventRow],
        events_by_type: dict[str, list[_DecodedEvent]],
        events_by_causation: dict[UUID, list[DomainEventRow]],
    ) -> dict[UUID, _ReceivedEvent]:
        received_by_item: dict[UUID, _ReceivedEvent] = {}
        received_by_capsule: dict[UUID, list[_ReceivedEvent]] = defaultdict(list)
        subscriptions_by_topic: dict[UUID, list[SubscriptionRow]] = defaultdict(list)
        subscription_by_route: dict[
            tuple[UUID, UUID],
            SubscriptionRow,
        ] = {}
        for subscription in subscriptions:
            subscriptions_by_topic[subscription.topic_id].append(subscription)
            subscription_by_route[
                (
                    subscription.topic_id,
                    subscription.subscriber_agent_id,
                )
            ] = subscription
        for topic_subscriptions in subscriptions_by_topic.values():
            topic_subscriptions.sort(
                key=lambda row: (
                    row.subscriber_agent_id.bytes,
                    row.subscription_id.bytes,
                )
            )
        events = events_by_type.get(CapsuleReceivedV1.event_type, [])
        for event, payload in events:
            if not isinstance(payload, CapsuleReceivedV1):
                raise SourceIntegrityError(
                    "Capsule receipt decoded to the wrong schema",
                    mismatch_key=f"inbox_item:{event.aggregate_id}",
                )
            key = f"inbox_item:{payload.item_id}"
            capsule = capsules.get(payload.capsule_id)
            publication = publications.get(payload.capsule_id)
            if capsule is None or publication is None:
                raise SourceIntegrityError(
                    "Inbox receipt references a missing capsule",
                    mismatch_key=key,
                )
            eligible_subscription = subscription_by_route.get(
                (
                    capsule.topic_id,
                    payload.recipient_agent_id,
                )
            )
            eligible = (
                eligible_subscription is not None
                and eligible_subscription.creation_event_id < publication.event_id
                and payload.recipient_agent_id != capsule.publisher_agent_id
            )
            if (
                not eligible
                or payload.item_id in received_by_item
                or event.aggregate_type != "inbox_item"
                or event.aggregate_id != payload.item_id
                or event.sequence != 1
                or event.actor_agent_id != capsule.publisher_agent_id
                or event.occurred_at != capsule.created_at
                or event.causation_id != publication.causation_id
                or event.event_id <= publication.event_id
            ):
                raise SourceIntegrityError(
                    "Inbox receipt was not eligible at publication",
                    mismatch_key=key,
                )
            received_by_item[payload.item_id] = (event, payload)
            received_by_capsule[payload.capsule_id].append((event, payload))

        for capsule_id, capsule in sorted(
            capsules.items(),
            key=lambda item: item[0].bytes,
        ):
            publication = publications[capsule_id]
            expected_subscriptions = tuple(
                row
                for row in subscriptions_by_topic.get(capsule.topic_id, ())
                if row.creation_event_id < publication.event_id
                and row.subscriber_agent_id != capsule.publisher_agent_id
            )
            expected_recipients = tuple(
                row.subscriber_agent_id for row in expected_subscriptions
            )
            actual_events = tuple(
                sorted(
                    received_by_capsule.get(capsule_id, []),
                    key=lambda item: item[0].event_id,
                )
            )
            actual_recipients = tuple(
                payload.recipient_agent_id for _, payload in actual_events
            )
            causal_rows = tuple(events_by_causation.get(publication.causation_id, ()))
            expected_event_ids = (
                publication.event_id,
                *(event.event_id for event, _ in actual_events),
            )
            if (
                actual_recipients != expected_recipients
                or tuple(row.event_id for row in causal_rows) != expected_event_ids
                or tuple(row.event_type for row in causal_rows)
                != (
                    CapsulePublishedV1.event_type,
                    *(CapsuleReceivedV1.event_type for _ in actual_events),
                )
            ):
                raise SourceIntegrityError(
                    "Capsule delivery order does not match prior subscriptions",
                    mismatch_key=f"capsule:{capsule_id}",
                )
        return received_by_item

    def _validate_terminal_events(
        self,
        *,
        capsules: dict[UUID, Capsule],
        received_by_item: dict[UUID, _ReceivedEvent],
        events_by_type: dict[str, list[_DecodedEvent]],
        events_by_causation: dict[UUID, list[DomainEventRow]],
        events_by_aggregate: dict[tuple[str, UUID], list[DomainEventRow]],
        event_ids_by_aggregate: dict[tuple[str, UUID], list[int]],
    ) -> dict[UUID, _TerminalEvent]:
        terminal_by_item: dict[UUID, _TerminalEvent] = {}
        for event_type in (
            CapsuleAdoptedV1.event_type,
            CapsuleRejectedV1.event_type,
        ):
            for event, payload in events_by_type.get(event_type, []):
                if not isinstance(payload, (CapsuleAdoptedV1, CapsuleRejectedV1)):
                    raise SourceIntegrityError(
                        "Inbox terminal event decoded to the wrong schema",
                        mismatch_key=f"inbox_item:{event.aggregate_id}",
                    )
                key = f"inbox_item:{payload.item_id}"
                received = received_by_item.get(payload.item_id)
                if received is None:
                    raise SourceIntegrityError(
                        "Inbox terminal event has no receipt",
                        mismatch_key=key,
                    )
                received_event, received_payload = received
                capsule_aggregate = ("capsule", received_payload.capsule_id)
                prior_index = (
                    bisect_left(
                        event_ids_by_aggregate.get(capsule_aggregate, ()),
                        event.event_id,
                    )
                    - 1
                )
                prior_capsule_events = events_by_aggregate.get(
                    capsule_aggregate,
                    (),
                )
                latest_capsule_event = (
                    prior_capsule_events[prior_index]
                    if prior_index >= 0
                    else None
                )
                actor_id = (
                    payload.adopter_agent_id
                    if isinstance(payload, CapsuleAdoptedV1)
                    else payload.recipient_agent_id
                )
                if (
                    payload.item_id in terminal_by_item
                    or event.aggregate_type != "inbox_item"
                    or event.aggregate_id != payload.item_id
                    or event.sequence != 2
                    or event.actor_agent_id != actor_id
                    or event.event_id <= received_event.event_id
                    or event.occurred_at < received_event.occurred_at
                    or latest_capsule_event is None
                    or event.occurred_at < latest_capsule_event.occurred_at
                    or payload.capsule_id != received_payload.capsule_id
                    or actor_id != received_payload.recipient_agent_id
                    or payload.capsule_id not in capsules
                    or (
                        isinstance(payload, CapsuleRejectedV1)
                        and tuple(
                            row.event_id
                            for row in events_by_causation.get(
                                event.causation_id,
                                (),
                            )
                        )
                        != (event.event_id,)
                    )
                ):
                    raise SourceIntegrityError(
                        "Inbox terminal event does not follow its receipt",
                        mismatch_key=key,
                    )
                terminal_by_item[payload.item_id] = (event, payload)

        retracted: set[UUID] = set()
        for event, payload in events_by_type.get(
            CapsuleRetractedV1.event_type,
            [],
        ):
            if not isinstance(payload, CapsuleRetractedV1):
                raise SourceIntegrityError(
                    "Capsule retraction decoded to the wrong schema",
                    mismatch_key=f"capsule:{event.aggregate_id}",
                )
            capsule = capsules.get(payload.capsule_id)
            if (
                capsule is None
                or payload.capsule_id in retracted
                or event.aggregate_type != "capsule"
                or event.aggregate_id != payload.capsule_id
                or event.sequence < 2
                or event.actor_agent_id != capsule.publisher_agent_id
                or payload.publisher_agent_id != capsule.publisher_agent_id
                or payload.status_before is not CapsuleStatus.ACTIVE
                or payload.status_after is not CapsuleStatus.RETRACTED
                or event.occurred_at < capsule.created_at
                or tuple(
                    row.event_id
                    for row in events_by_causation.get(
                        event.causation_id,
                        (),
                    )
                )
                != (event.event_id,)
            ):
                raise SourceIntegrityError(
                    "Capsule retraction does not match its source",
                    mismatch_key=f"capsule:{payload.capsule_id}",
                )
            retracted.add(payload.capsule_id)
        return terminal_by_item

    def _validate_feedback(
        self,
        *,
        rows: tuple[CapsuleFeedbackRow, ...],
        capsules: dict[UUID, Capsule],
        terminal_by_item: dict[UUID, _TerminalEvent],
        events_by_type: dict[str, list[_DecodedEvent]],
        events_by_causation: dict[UUID, list[DomainEventRow]],
    ) -> dict[tuple[UUID, UUID], list[tuple[int, int, int]]]:
        events = events_by_type.get(CapsuleFeedbackRecordedV1.event_type, [])
        by_id: dict[
            UUID,
            list[tuple[DomainEventRow, CapsuleFeedbackRecordedV1]],
        ] = defaultdict(list)
        for event, payload in events:
            if not isinstance(payload, CapsuleFeedbackRecordedV1):
                raise SourceIntegrityError(
                    "Feedback event decoded to the wrong schema",
                    mismatch_key=f"feedback_event:{event.event_id}",
                )
            by_id[payload.feedback_id].append((event, payload))

        rows_by_stream: dict[
            tuple[UUID, UUID],
            list[CapsuleFeedbackRow],
        ] = defaultdict(list)
        row_ids = {row.feedback_id for row in rows}
        for row in rows:
            rows_by_stream[(row.observer_agent_id, row.capsule_id)].append(row)
        for stream_rows in rows_by_stream.values():
            stream_rows.sort(key=lambda row: (row.revision, row.feedback_id.bytes))

        terminal_by_observer_capsule: dict[
            tuple[UUID, UUID],
            _TerminalEvent,
        ] = {}
        for terminal_value in terminal_by_item.values():
            event, payload = terminal_value
            observer = (
                payload.adopter_agent_id
                if isinstance(payload, CapsuleAdoptedV1)
                else payload.recipient_agent_id
            )
            terminal_by_observer_capsule[(observer, payload.capsule_id)] = (
                event,
                payload,
            )

        for stream_key, stream_rows in sorted(
            rows_by_stream.items(),
            key=lambda item: (item[0][0].bytes, item[0][1].bytes),
        ):
            prior_verdict: FeedbackVerdict | None = None
            prior_created: datetime | None = None
            prior_event_id = 0
            for expected_revision, row in enumerate(stream_rows, start=1):
                key = f"feedback:{row.feedback_id}"
                matches = by_id.get(row.feedback_id, [])
                if len(matches) != 1 or row.revision != expected_revision:
                    raise SourceIntegrityError(
                        "Feedback revisions must be contiguous and unique",
                        mismatch_key=key,
                    )
                event, payload = matches[0]
                capsule = capsules.get(row.capsule_id)
                terminal: _TerminalEvent | None = terminal_by_observer_capsule.get(
                    stream_key
                )
                try:
                    reason_value = json.loads(row.reason)
                    evidence_values = _canonical_array(
                        row.evidence,
                        label=f"feedback {row.feedback_id} evidence",
                    )
                    reason = StructuredReason.model_validate(reason_value)
                    evidence = tuple(
                        TypedEvidence.model_validate(value) for value in evidence_values
                    )
                    FeedbackRevision(
                        feedback_id=row.feedback_id,
                        observer_agent_id=row.observer_agent_id,
                        capsule_id=row.capsule_id,
                        revision=row.revision,
                        verdict=row.verdict,
                        reason=reason,
                        evidence=evidence,
                        created_at=row.created_at,
                    )
                    if (
                        canonical_json_bytes(reason) != row.reason
                        or canonical_json_bytes(evidence) != row.evidence
                    ):
                        raise ValueError("feedback values are not canonical")
                except (
                    TypeError,
                    ValidationError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    raise SourceIntegrityError(
                        "Feedback source payload is invalid",
                        mismatch_key=key,
                    ) from error
                if (
                    capsule is None
                    or terminal is None
                    or event.aggregate_type != "capsule"
                    or event.aggregate_id != row.capsule_id
                    or event.actor_agent_id != row.observer_agent_id
                    or event.occurred_at != row.created_at
                    or payload.observer_agent_id != row.observer_agent_id
                    or payload.capsule_id != row.capsule_id
                    or payload.publisher_agent_id != capsule.publisher_agent_id
                    or payload.revision != row.revision
                    or payload.previous_verdict is not prior_verdict
                    or payload.current_verdict is not row.verdict
                    or terminal[0].event_id >= event.event_id
                    or terminal[0].occurred_at > event.occurred_at
                    or tuple(
                        causal.event_id
                        for causal in events_by_causation.get(
                            event.causation_id,
                            (),
                        )
                    )
                    != (event.event_id,)
                    or event.event_id <= prior_event_id
                    or (prior_created is not None and row.created_at < prior_created)
                ):
                    raise SourceIntegrityError(
                        "Feedback row, authorization, and event disagree",
                        mismatch_key=key,
                    )
                prior_verdict = row.verdict
                prior_created = row.created_at
                prior_event_id = event.event_id

        extras = sorted(set(by_id) - row_ids, key=lambda value: value.bytes)
        if extras:
            raise SourceIntegrityError(
                "Feedback event has no source row",
                mismatch_key=f"feedback:{extras[0]}",
            )

        pair_state: dict[tuple[UUID, UUID], tuple[int, int]] = {}
        pair_clock: dict[tuple[UUID, UUID], datetime] = {}
        trust_history: dict[
            tuple[UUID, UUID],
            list[tuple[int, int, int]],
        ] = defaultdict(list)
        latest_verdict: dict[tuple[UUID, UUID], FeedbackVerdict] = {}
        ordered = sorted(
            (
                (matches[0][0], matches[0][1])
                for matches in by_id.values()
                if len(matches) == 1
            ),
            key=lambda item: item[0].event_id,
        )
        for event, payload in ordered:
            pair = (
                payload.publisher_agent_id,
                payload.observer_agent_id,
            )
            previous_pair_time = pair_clock.get(pair)
            if (
                previous_pair_time is not None
                and event.occurred_at < previous_pair_time
            ):
                raise SourceIntegrityError(
                    "Feedback reputation clock regresses in ledger order",
                    mismatch_key=f"feedback:{payload.feedback_id}",
                )
            alpha, beta = pair_state.get(pair, (2, 2))
            stream = (
                payload.observer_agent_id,
                payload.capsule_id,
            )
            previous = latest_verdict.get(stream)
            expected_alpha = alpha
            expected_beta = beta
            if previous is FeedbackVerdict.USEFUL:
                expected_alpha -= 1
            elif previous in (
                FeedbackVerdict.REFUTED,
                FeedbackVerdict.HARMFUL,
            ):
                expected_beta -= 1
            if payload.current_verdict is FeedbackVerdict.USEFUL:
                expected_alpha += 1
            else:
                expected_beta += 1
            if (
                payload.previous_verdict is not previous
                or payload.alpha_before != alpha
                or payload.beta_before != beta
                or payload.alpha_after != expected_alpha
                or payload.beta_after != expected_beta
            ):
                raise SourceIntegrityError(
                    "Feedback event reputation transition is discontinuous",
                    mismatch_key=f"feedback:{payload.feedback_id}",
                )
            pair_state[pair] = (expected_alpha, expected_beta)
            pair_clock[pair] = event.occurred_at
            latest_verdict[stream] = payload.current_verdict
            trust_history[pair].append((event.event_id, expected_alpha, expected_beta))
        return trust_history

    def _validate_adoption_references(
        self,
        *,
        rows: tuple[AdoptionRecordRow, ...],
        capsules: dict[UUID, Capsule],
        identities: tuple[ExperienceRow, ...],
        versions: tuple[ExperienceVersionRow, ...],
        received_by_item: dict[UUID, _ReceivedEvent],
        terminal_by_item: dict[UUID, _TerminalEvent],
        trust_history: dict[tuple[UUID, UUID], list[tuple[int, int, int]]],
        events_by_type: dict[str, list[_DecodedEvent]],
        events_by_causation: dict[UUID, list[DomainEventRow]],
        experience_events: dict[UUID, list[DomainEventRow]],
    ) -> dict[UUID, _AdoptionAnchor]:
        identity_by_id = {row.experience_id: row for row in identities}
        version_times_by_content: dict[
            tuple[UUID, str],
            list[datetime],
        ] = defaultdict(list)
        initial_version_anchors: set[tuple[UUID, str, datetime]] = set()
        for version in versions:
            version_times_by_content[
                (version.experience_id, version.content_hash)
            ].append(version.created_at)
            if version.version_number == 1:
                initial_version_anchors.add(
                    (
                        version.experience_id,
                        version.content_hash,
                        version.created_at,
                    )
                )
        for created_times in version_times_by_content.values():
            created_times.sort()
        trust_event_ids = {
            pair: tuple(item[0] for item in history)
            for pair, history in trust_history.items()
        }
        first_retraction_by_capsule: dict[UUID, int] = {}
        for retraction_event, retracted in events_by_type.get(
            CapsuleRetractedV1.event_type,
            (),
        ):
            if not isinstance(retracted, CapsuleRetractedV1):
                continue
            first_retraction_by_capsule.setdefault(
                retracted.capsule_id,
                retraction_event.event_id,
            )
        experience_event_ids = {
            experience_id: tuple(item.event_id for item in event_rows)
            for experience_id, event_rows in experience_events.items()
        }
        events = events_by_type.get(CapsuleAdoptedV1.event_type, [])
        by_id: dict[
            UUID,
            list[tuple[DomainEventRow, CapsuleAdoptedV1]],
        ] = defaultdict(list)
        for event, payload in events:
            if isinstance(payload, CapsuleAdoptedV1):
                by_id[payload.adoption_id].append((event, payload))

        anchors: dict[UUID, _AdoptionAnchor] = {}
        for row in _ordered_uuid_rows(rows, "adoption_id"):
            key = f"adoption:{row.adoption_id}"
            matches = by_id.get(row.adoption_id, [])
            if len(matches) != 1:
                raise SourceIntegrityError(
                    "Adoption source requires exactly one event",
                    mismatch_key=key,
                )
            event, payload = matches[0]
            capsule = capsules.get(row.capsule_id)
            identity = identity_by_id.get(row.resulting_experience_id)
            received = received_by_item.get(payload.item_id)
            terminal = terminal_by_item.get(payload.item_id)
            try:
                chain = _decode_provenance(
                    row.provenance_chain,
                    label=f"adoption {row.adoption_id}",
                    allow_empty=False,
                )
            except (TypeError, ValidationError, ValueError) as error:
                raise SourceIntegrityError(
                    "Adoption provenance is invalid",
                    mismatch_key=key,
                ) from error
            expected_chain = (
                ()
                if capsule is None
                else (
                    *capsule.provenance_chain,
                    ProvenanceHop(
                        capsule_id=capsule.capsule_id,
                        publisher_agent_id=capsule.publisher_agent_id,
                    ),
                )
            )
            matching_version_exists = False
            created_version_anchor_exists = False
            if identity is not None and capsule is not None:
                version_key = (
                    identity.experience_id,
                    capsule.source_content_hash,
                )
                created_times = version_times_by_content.get(version_key, [])
                matching_version_exists = (
                    bisect_right(created_times, row.adopted_at) > 0
                )
                created_version_anchor_exists = (
                    identity.experience_id,
                    capsule.source_content_hash,
                    row.adopted_at,
                ) in initial_version_anchors
            expected_trust = 0.5
            if capsule is not None:
                pair = (
                    capsule.publisher_agent_id,
                    row.adopter_agent_id,
                )
                history = trust_history.get(pair, [])
                position = bisect_left(
                    trust_event_ids.get(pair, ()),
                    event.event_id,
                )
                if position:
                    _, alpha, beta = history[position - 1]
                    expected_trust = alpha / (alpha + beta)
            retraction_event_id = first_retraction_by_capsule.get(row.capsule_id)
            prior_retraction = (
                retraction_event_id is not None and retraction_event_id < event.event_id
            )
            if (
                capsule is None
                or identity is None
                or identity.owner_agent_id != row.adopter_agent_id
                or identity.kind != capsule.kind
                or not matching_version_exists
                or received is None
                or terminal != (event, payload)
                or received[0].event_id >= event.event_id
                or row.adopted_at < capsule.created_at
                or row.adopted_at >= capsule.expires_at
                or prior_retraction
                or row.root_fingerprint != capsule.root_fingerprint
                or chain != expected_chain
                or not math.isfinite(row.captured_trust)
                or not math.isclose(
                    row.captured_trust,
                    expected_trust,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or (
                    payload.created
                    and (
                        identity.origin is not ExperienceOrigin.ADOPTED_CAPSULE
                        or identity.created_at != row.adopted_at
                        or not created_version_anchor_exists
                    )
                )
            ):
                raise SourceIntegrityError(
                    "Adoption does not reference an owned inbox result",
                    mismatch_key=key,
                )
            try:
                self._validate_adoption_causation(
                    row=row,
                    event=event,
                    payload=payload,
                    capsule=capsule,
                    identity=identity,
                    events_by_causation=events_by_causation,
                    experience_events=experience_events,
                    experience_event_ids=experience_event_ids,
                )
            except (TypeError, ValidationError, ValueError) as error:
                raise SourceIntegrityError(
                    "Adoption command causation is inconsistent",
                    mismatch_key=key,
                ) from error
            anchors[row.adoption_id] = (chain, event)
        return anchors

    def _validate_adoption_causation(
        self,
        *,
        row: AdoptionRecordRow,
        event: DomainEventRow,
        payload: CapsuleAdoptedV1,
        capsule: Capsule,
        identity: ExperienceRow,
        events_by_causation: dict[UUID, list[DomainEventRow]],
        experience_events: dict[UUID, list[DomainEventRow]],
        experience_event_ids: dict[UUID, tuple[int, ...]],
    ) -> None:
        causal_rows = tuple(events_by_causation.get(event.causation_id, ()))
        causal_types = tuple(item.event_type for item in causal_rows)
        if not causal_rows or causal_rows[-1].event_id != event.event_id:
            raise ValueError("adoption event must finish its command causation")

        if payload.created:
            if causal_types != (
                ExperienceCreatedV1.event_type,
                ExperienceVersionCreatedV1.event_type,
                CapsuleAdoptedV1.event_type,
            ):
                raise ValueError(
                    "created adoption requires creation, version, and adoption"
                )
            created_event, version_event, _ = causal_rows
            created = self._event_registry.decode(
                event_type=created_event.event_type,
                payload=created_event.payload,
            )
            version = self._event_registry.decode(
                event_type=version_event.event_type,
                payload=version_event.payload,
            )
            if not isinstance(created, ExperienceCreatedV1) or not isinstance(
                version,
                ExperienceVersionCreatedV1,
            ):
                raise ValueError("created adoption event schemas are invalid")
            expected_confidence = initial_adoption_confidence(
                capsule.publisher_confidence,
                row.captured_trust,
            )
            if (
                created_event.aggregate_type != "experience"
                or created_event.aggregate_id != payload.resulting_experience_id
                or created_event.sequence != 1
                or version_event.aggregate_type != "experience"
                or version_event.aggregate_id != payload.resulting_experience_id
                or version_event.sequence != 2
                or any(
                    item.actor_agent_id != payload.adopter_agent_id
                    or item.occurred_at != event.occurred_at
                    for item in (created_event, version_event)
                )
                or created.experience_id != payload.resulting_experience_id
                or version.experience_id != payload.resulting_experience_id
                or created.after.owner_agent_id != payload.adopter_agent_id
                or created.after.current_content_hash != capsule.source_content_hash
                or created.after.temperature is not Temperature.HOT
                or not math.isclose(
                    created.after.source_trust,
                    row.captured_trust,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    created.after.confidence,
                    expected_confidence,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or version.before != created.after
                or version.after != created.after
                or version.version_number != 1
                or version.supersedes_version_id is not None
                or version.links
                or identity.origin is not ExperienceOrigin.ADOPTED_CAPSULE
                or identity.kind is not capsule.kind
                or identity.created_at != event.occurred_at
            ):
                raise ValueError("created adoption state is inconsistent")
            return

        if payload.corroboration_applied:
            if causal_types not in (
                (
                    ExperienceCorroboratedV1.event_type,
                    CapsuleAdoptedV1.event_type,
                ),
                (
                    ExperienceCorroboratedV1.event_type,
                    ExperienceTemperatureChangedV1.event_type,
                    CapsuleAdoptedV1.event_type,
                ),
            ):
                raise ValueError("corroborating adoption has an invalid event sequence")
            corroboration_event = causal_rows[0]
            corroboration = self._event_registry.decode(
                event_type=corroboration_event.event_type,
                payload=corroboration_event.payload,
            )
            if not isinstance(corroboration, ExperienceCorroboratedV1):
                raise ValueError("corroboration event schema is invalid")
            self._validate_experience_checkpoint(
                experience_id=payload.resulting_experience_id,
                owner_agent_id=payload.adopter_agent_id,
                content_hash=capsule.source_content_hash,
                before=corroboration.before,
                before_event_id=corroboration_event.event_id,
                command_time=event.occurred_at,
                experience_events=experience_events,
                experience_event_ids=experience_event_ids,
            )
            if (
                corroboration_event.aggregate_type != "experience"
                or corroboration_event.aggregate_id != payload.resulting_experience_id
                or corroboration_event.actor_agent_id != payload.adopter_agent_id
                or corroboration_event.occurred_at != event.occurred_at
                or corroboration.experience_id != payload.resulting_experience_id
                or corroboration.adoption_id != payload.adoption_id
                or corroboration.capsule_id != payload.capsule_id
                or corroboration.root_fingerprint != payload.root_fingerprint
                or not math.isclose(
                    corroboration.captured_trust,
                    row.captured_trust,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("corroboration source anchor is inconsistent")
            if corroboration.before.temperature is Temperature.COLD:
                if len(causal_rows) != 3:
                    raise ValueError("cold corroboration must include hot promotion")
                transition_event = causal_rows[1]
                transition = self._event_registry.decode(
                    event_type=transition_event.event_type,
                    payload=transition_event.payload,
                )
                if (
                    not isinstance(
                        transition,
                        ExperienceTemperatureChangedV1,
                    )
                    or transition_event.aggregate_type != "experience"
                    or transition_event.aggregate_id != payload.resulting_experience_id
                    or transition_event.actor_agent_id != payload.adopter_agent_id
                    or transition_event.occurred_at != event.occurred_at
                    or transition.cause != "capsule_corroboration"
                    or transition.experience_id != payload.resulting_experience_id
                    or transition.before != corroboration.after
                    or transition.after.temperature is not Temperature.HOT
                ):
                    raise ValueError(
                        "corroboration temperature promotion is inconsistent"
                    )
            elif len(causal_rows) != 2:
                raise ValueError("non-cold corroboration must not include a promotion")
            return

        if causal_types != (CapsuleAdoptedV1.event_type,):
            raise ValueError("equivalent non-corroborating adoption must be event-only")
        event_rows = experience_events.get(
            payload.resulting_experience_id,
            (),
        )
        checkpoint_position = bisect_left(
            experience_event_ids.get(
                payload.resulting_experience_id,
                (),
            ),
            event.event_id,
        )
        if not checkpoint_position:
            raise ValueError("equivalent adoption has no prior experience checkpoint")
        checkpoint = event_rows[checkpoint_position - 1]
        checkpoint_payload = self._event_registry.decode(
            event_type=checkpoint.event_type,
            payload=checkpoint.payload,
        )
        after = getattr(checkpoint_payload, "after", None)
        if not isinstance(after, ExperienceStateSnapshotV1):
            raise ValueError("equivalent adoption checkpoint has no after-state")
        self._require_checkpoint_state(
            snapshot=after,
            experience_id=payload.resulting_experience_id,
            owner_agent_id=payload.adopter_agent_id,
            content_hash=capsule.source_content_hash,
            checkpoint_time=checkpoint.occurred_at,
            command_time=event.occurred_at,
        )

    def _validate_experience_checkpoint(
        self,
        *,
        experience_id: UUID,
        owner_agent_id: UUID,
        content_hash: str,
        before: ExperienceStateSnapshotV1,
        before_event_id: int,
        command_time: datetime,
        experience_events: dict[UUID, list[DomainEventRow]],
        experience_event_ids: dict[UUID, tuple[int, ...]],
    ) -> None:
        event_rows = experience_events.get(experience_id, ())
        checkpoint_position = bisect_left(
            experience_event_ids.get(experience_id, ()),
            before_event_id,
        )
        if not checkpoint_position:
            raise ValueError("corroboration has no prior experience checkpoint")
        checkpoint = event_rows[checkpoint_position - 1]
        checkpoint_payload = self._event_registry.decode(
            event_type=checkpoint.event_type,
            payload=checkpoint.payload,
        )
        after = getattr(checkpoint_payload, "after", None)
        if not isinstance(after, ExperienceStateSnapshotV1) or after != before:
            raise ValueError("corroboration before-state is not the current head")
        self._require_checkpoint_state(
            snapshot=before,
            experience_id=experience_id,
            owner_agent_id=owner_agent_id,
            content_hash=content_hash,
            checkpoint_time=checkpoint.occurred_at,
            command_time=command_time,
        )

    @staticmethod
    def _require_checkpoint_state(
        *,
        snapshot: ExperienceStateSnapshotV1,
        experience_id: UUID,
        owner_agent_id: UUID,
        content_hash: str,
        checkpoint_time: datetime,
        command_time: datetime,
    ) -> None:
        if (
            checkpoint_time > command_time
            or snapshot.experience_id != experience_id
            or snapshot.owner_agent_id != owner_agent_id
            or snapshot.current_content_hash != content_hash
            or snapshot.temperature is Temperature.ARCHIVED
        ):
            raise ValueError("equivalent experience state is inconsistent")

    def _validate_provenance(
        self,
        *,
        capsules: dict[UUID, Capsule],
        publications: dict[UUID, DomainEventRow],
        adoption_rows: tuple[AdoptionRecordRow, ...],
        adoption_anchors: dict[UUID, _AdoptionAnchor],
    ) -> None:
        parent_adoptions: dict[
            tuple[
                UUID,
                UUID,
                str,
                tuple[ProvenanceHop, ...],
            ],
            list[tuple[AdoptionRecordRow, DomainEventRow]],
        ] = defaultdict(list)
        for row in adoption_rows:
            anchor = adoption_anchors.get(row.adoption_id)
            if anchor is None:
                continue
            chain, event = anchor
            parent_adoptions[
                (
                    row.adopter_agent_id,
                    row.resulting_experience_id,
                    row.root_fingerprint,
                    chain,
                )
            ].append((row, event))

        for capsule_id, capsule in sorted(
            capsules.items(),
            key=lambda item: publications[item[0]].event_id,
        ):
            key = f"capsule:{capsule_id}"
            chain = capsule.provenance_chain
            for index, hop in enumerate(chain):
                referenced = capsules.get(hop.capsule_id)
                if (
                    referenced is None
                    or hop.publisher_agent_id != referenced.publisher_agent_id
                    or referenced.provenance_chain != chain[:index]
                    or referenced.root_fingerprint != capsule.root_fingerprint
                    or publications[hop.capsule_id].event_id
                    >= publications[capsule_id].event_id
                    or referenced.created_at > capsule.created_at
                ):
                    raise SourceIntegrityError(
                        "Capsule provenance chain is not root-first continuous",
                        mismatch_key=key,
                    )
            if not chain:
                continue
            parents = [
                row
                for row, event in parent_adoptions.get(
                    (
                        capsule.publisher_agent_id,
                        capsule.source_experience_id,
                        capsule.root_fingerprint,
                        chain,
                    ),
                    (),
                )
                if row.adopted_at <= capsule.created_at
                and event.event_id < publications[capsule_id].event_id
            ]
            if len(parents) != 1:
                raise SourceIntegrityError(
                    "Relayed capsule has no unique parent adoption",
                    mismatch_key=key,
                )

    @staticmethod
    def _validate_command_receipts(
        *,
        events: tuple[_DecodedEvent, ...],
        receipts: dict[UUID, IdempotencyRecordRow],
        events_by_causation: dict[UUID, list[DomainEventRow]],
    ) -> None:
        single_event_commands: list[tuple[DomainEventRow, str]] = []
        for event, payload in events:
            if isinstance(payload, CapsuleReceivedV1):
                continue
            if isinstance(payload, TopicCreatedV1):
                scope = "topic.create"
                resource_type = "topic"
                resource_id = payload.topic_id
                actor_id = payload.owner_agent_id
                mismatch_key = f"topic:{payload.topic_id}"
                requires_single_event = True
            elif isinstance(payload, SubscriptionCreatedV1):
                scope = "subscription.create"
                resource_type = "subscription"
                resource_id = payload.subscription_id
                actor_id = payload.subscriber_agent_id
                mismatch_key = f"subscription:{payload.subscription_id}"
                requires_single_event = True
            elif isinstance(payload, CapsulePublishedV1):
                scope = "capsule.publish"
                resource_type = "capsule"
                resource_id = payload.capsule_id
                actor_id = payload.publisher_agent_id
                mismatch_key = f"capsule:{payload.capsule_id}"
                requires_single_event = False
            elif isinstance(payload, CapsuleAdoptedV1):
                scope = "capsule.adopt"
                resource_type = "adoption"
                resource_id = payload.adoption_id
                actor_id = payload.adopter_agent_id
                mismatch_key = f"adoption:{payload.adoption_id}"
                requires_single_event = False
            elif isinstance(payload, CapsuleRetractedV1):
                scope = "capsule.retract"
                resource_type = "capsule"
                resource_id = payload.capsule_id
                actor_id = payload.publisher_agent_id
                mismatch_key = f"capsule:{payload.capsule_id}"
                requires_single_event = True
            elif isinstance(payload, CapsuleRejectedV1):
                scope = "capsule.reject"
                resource_type = "inbox_item"
                resource_id = payload.item_id
                actor_id = payload.recipient_agent_id
                mismatch_key = f"inbox_item:{payload.item_id}"
                requires_single_event = True
            elif isinstance(payload, CapsuleFeedbackRecordedV1):
                scope = "capsule.feedback"
                resource_type = "feedback"
                resource_id = payload.feedback_id
                actor_id = payload.observer_agent_id
                mismatch_key = f"feedback:{payload.feedback_id}"
                requires_single_event = True
            else:
                raise SourceIntegrityError(
                    "Sharing command event has an unsupported receipt contract",
                    mismatch_key=f"sharing_event:{event.event_id}",
                )

            receipt = receipts.get(event.causation_id)
            if (
                receipt is None
                or receipt.state != "completed"
                or receipt.scope != scope
                or receipt.caller_scope != f"agent:{actor_id}"
                or receipt.result_resource_type != resource_type
                or receipt.result_resource_id != resource_id
                or receipt.completed_at is None
                or receipt.created_at > event.occurred_at
                or receipt.completed_at < event.occurred_at
            ):
                raise SourceIntegrityError(
                    "Sharing event is not bound to its exact completed command receipt",
                    mismatch_key=mismatch_key,
                )
            if requires_single_event:
                single_event_commands.append((event, mismatch_key))

        for event, mismatch_key in single_event_commands:
            if tuple(
                row.event_id
                for row in events_by_causation.get(event.causation_id, ())
            ) != (event.event_id,):
                raise SourceIntegrityError(
                    "Sharing command receipt does not own exactly one event",
                    mismatch_key=mismatch_key,
                )


def register_sharing_source_validator(
    validator: SourceValidator,
) -> None:
    """Register adoption and complete sharing-graph source checks."""
    validator.register(SharingAdoptionSourceValidator(validator.event_registry))
    validator.register(SharingSourceValidator(validator.event_registry))


__all__ = [
    "SharingAdoptionSourceValidator",
    "SharingSourceValidator",
    "register_sharing_source_validator",
]
