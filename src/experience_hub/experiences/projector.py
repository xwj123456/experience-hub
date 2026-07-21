"""Fail-closed online and replay reducer for the experience-state projection."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry, StoredEvent, StructuredReason
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    TASK2_EXPERIENCE_EVENT_TYPES,
    ExperienceAccessedV1,
    ExperienceArchivedV1,
    ExperienceConfirmedV1,
    ExperienceCorroboratedV1,
    ExperienceCreatedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperiencePinnedV1,
    ExperienceReactivatedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
    ExperienceVersionCreatedV1,
    is_valid_version_event_sequence,
)
from experience_hub.experiences.models import Temperature, VersionContent
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
    record_access,
)
from experience_hub.retrieval.tokenizer import index_version_terms
from experience_hub.sharing.events import CapsuleAdoptedV1
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
)
from experience_hub.storage.validation import SourceIntegrityError

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ADOPTION_EVENT_INDEX_CACHE_KEY = (
    "experience_hub.experiences.adoption_event_index.v1"
)
_SEMANTIC_FIELDS = tuple(ExperienceStateSnapshotV1.model_fields)
_CORRECTION_FIELDS = frozenset(
    {
        "current_version_id",
        "current_content_hash",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_LIFECYCLE_EVALUATION_FIELDS = frozenset(
    {
        "access_strength",
        "strength_updated_at",
        "activation_score",
        "last_lifecycle_evaluated_at",
        "consecutive_below_threshold",
    }
)
_CONFIDENCE_EVENT_FIELDS = frozenset(
    {
        "confidence",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_PIN_EVENT_FIELDS = frozenset(
    {
        "pinned",
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)
_RESTORE_EVENT_FIELDS = frozenset(
    {
        "access_strength",
        "strength_updated_at",
        "activation_score",
    }
)


async def _adoption_event_index(
    session: AsyncSession,
    *,
    event_registry: EventRegistry,
    refresh: bool = False,
) -> dict[UUID, tuple[DomainEventRow, CapsuleAdoptedV1]]:
    if not refresh:
        cached = session.info.get(_ADOPTION_EVENT_INDEX_CACHE_KEY)
        if cached is not None:
            return cast(
                dict[UUID, tuple[DomainEventRow, CapsuleAdoptedV1]],
                cached,
            )
    rows = (
        await session.scalars(
            select(DomainEventRow)
            .where(
                DomainEventRow.event_type
                == CapsuleAdoptedV1.event_type
            )
            .order_by(DomainEventRow.event_id)
        )
    ).all()
    result: dict[UUID, tuple[DomainEventRow, CapsuleAdoptedV1]] = {}
    for row in rows:
        try:
            decoded = event_registry.decode(
                event_type=row.event_type,
                payload=row.payload,
            )
        except (TypeError, ValueError) as error:
            raise _fail("Capsule adoption event cannot be decoded") from error
        if (
            not isinstance(decoded, CapsuleAdoptedV1)
            or row.aggregate_type != "inbox_item"
            or row.aggregate_id != decoded.item_id
            or row.sequence != 2
            or row.actor_agent_id != decoded.adopter_agent_id
            or decoded.adoption_id in result
        ):
            raise _fail("Capsule adoption event index is inconsistent")
        result[decoded.adoption_id] = (row, decoded)
    session.info[_ADOPTION_EVENT_INDEX_CACHE_KEY] = result
    return result


class ExperienceProjectionIntegrityError(RuntimeError):
    """An event cannot be reconciled with its authoritative source anchors."""

    code = "experience_projection_integrity_error"


def _fail(message: str) -> ExperienceProjectionIntegrityError:
    return ExperienceProjectionIntegrityError(message)


def _utc(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return require_utc(value)
    return require_utc(
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return cast(int, value)


def _boolean(value: Any) -> bool:
    if value not in {0, 1} or isinstance(value, bool):
        raise ValueError("pinned must be stored as zero or one")
    return bool(value)


def _snapshot_from_mapping(
    row: Mapping[str, Any],
) -> ExperienceStateSnapshotV1:
    return ExperienceStateSnapshotV1(
        experience_id=UUID(str(row["experience_id"])),
        owner_agent_id=UUID(str(row["owner_agent_id"])),
        current_version_id=UUID(str(row["current_version_id"])),
        current_content_hash=str(row["current_content_hash"]),
        temperature=Temperature(str(row["temperature"])),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        activation_score=float(row["activation_score"]),
        source_trust=float(row["source_trust"]),
        access_count=_integer(row["access_count"], label="access_count"),
        access_strength=float(row["access_strength"]),
        strength_updated_at=_datetime(row["strength_updated_at"]),
        last_accessed_at=(
            None
            if row["last_accessed_at"] is None
            else _datetime(row["last_accessed_at"])
        ),
        last_transition_at=_datetime(row["last_transition_at"]),
        last_lifecycle_evaluated_at=(
            None
            if row["last_lifecycle_evaluated_at"] is None
            else _datetime(row["last_lifecycle_evaluated_at"])
        ),
        consecutive_below_threshold=_integer(
            row["consecutive_below_threshold"],
            label="consecutive_below_threshold",
        ),
        pinned=_boolean(row["pinned"]),
    )


def _snapshot_parameters(
    snapshot: ExperienceStateSnapshotV1,
    *,
    event_id: int,
) -> dict[str, Any]:
    return {
        "experience_id": str(snapshot.experience_id),
        "owner_agent_id": str(snapshot.owner_agent_id),
        "current_version_id": str(snapshot.current_version_id),
        "current_content_hash": snapshot.current_content_hash,
        "temperature": snapshot.temperature.value,
        "importance": snapshot.importance,
        "confidence": snapshot.confidence,
        "activation_score": snapshot.activation_score,
        "source_trust": snapshot.source_trust,
        "access_count": snapshot.access_count,
        "access_strength": snapshot.access_strength,
        "strength_updated_at": _utc(snapshot.strength_updated_at),
        "last_accessed_at": (
            None
            if snapshot.last_accessed_at is None
            else _utc(snapshot.last_accessed_at)
        ),
        "last_transition_at": _utc(snapshot.last_transition_at),
        "last_lifecycle_evaluated_at": (
            None
            if snapshot.last_lifecycle_evaluated_at is None
            else _utc(snapshot.last_lifecycle_evaluated_at)
        ),
        "consecutive_below_threshold": snapshot.consecutive_below_threshold,
        "pinned": int(snapshot.pinned),
        "projection_event_id": event_id,
    }


def _changed_snapshot_fields(
    before: ExperienceStateSnapshotV1,
    after: ExperienceStateSnapshotV1,
) -> frozenset[str]:
    return frozenset(
        name
        for name in _SEMANTIC_FIELDS
        if getattr(before, name) != getattr(after, name)
    )


def _target_table(
    target_prefix: str | None,
    projection_name: str = "experience_state",
) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(projection_name):
        raise ValueError("Unsafe experience projection target")
    if target_prefix is None:
        return f'main."{projection_name}"'
    name = f"{target_prefix}{projection_name}"
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError("Unsafe experience projection target")
    return f'temp."{name}"'


async def _create_state_rebuild_table(
    session: AsyncSession,
    target: str,
) -> None:
    """Create the v1 state target without reading the online projection."""
    await session.execute(
        text(
            f"CREATE TEMP TABLE {target} ("
            "experience_id VARCHAR(36) NOT NULL PRIMARY KEY, "
            "owner_agent_id VARCHAR(36) NOT NULL, "
            "current_version_id VARCHAR(36) NOT NULL, "
            "current_content_hash VARCHAR(64) NOT NULL, "
            "temperature VARCHAR(8) NOT NULL, "
            "importance FLOAT NOT NULL, "
            "confidence FLOAT NOT NULL, "
            "activation_score FLOAT NOT NULL, "
            "source_trust FLOAT NOT NULL, "
            "access_count INTEGER NOT NULL, "
            "access_strength FLOAT NOT NULL, "
            "strength_updated_at VARCHAR(27) NOT NULL, "
            "last_accessed_at VARCHAR(27), "
            "last_transition_at VARCHAR(27) NOT NULL, "
            "last_lifecycle_evaluated_at VARCHAR(27), "
            "consecutive_below_threshold INTEGER NOT NULL, "
            "pinned BOOLEAN NOT NULL, "
            "projection_event_id INTEGER NOT NULL, "
            "UNIQUE (owner_agent_id, current_content_hash), "
            "CHECK (temperature IN ('hot', 'warm', 'cold', 'archived')), "
            "CHECK (importance BETWEEN 0 AND 1), "
            "CHECK (confidence BETWEEN 0 AND 1), "
            "CHECK (activation_score BETWEEN 0 AND 1), "
            "CHECK (source_trust BETWEEN 0 AND 1), "
            "CHECK (access_count >= 0), "
            "CHECK (access_strength BETWEEN 0 AND 20), "
            "CHECK (consecutive_below_threshold >= 0), "
            "CHECK (pinned IN (0, 1)), "
            "CHECK (projection_event_id > 0), "
            "CHECK (length(current_content_hash) = 64 "
            "AND current_content_hash NOT GLOB '*[^0-9a-f]*')"
            ")"
        )
    )


async def _create_terms_rebuild_table(
    session: AsyncSession,
    target: str,
) -> None:
    """Create the v1 term target without reading the online projection."""
    await session.execute(
        text(
            f"CREATE TEMP TABLE {target} ("
            "experience_id VARCHAR(36) NOT NULL, "
            "term VARCHAR NOT NULL, "
            "term_kind VARCHAR(12) NOT NULL, "
            "weight FLOAT NOT NULL, "
            "PRIMARY KEY (experience_id, term, term_kind), "
            "CHECK (length(term) > 0), "
            "CHECK (term_kind IN "
            "('word', 'char_trigram', 'tag', 'mechanism')), "
            "CHECK (weight > 0 AND weight <= 1.5)"
            ")"
        )
    )


async def _load_source_anchors(
    session: AsyncSession,
    *,
    event: StoredEvent,
    experience_id: UUID,
    version_id: UUID,
) -> tuple[ExperienceRow, ExperienceVersionRow, VersionContent]:
    try:
        identity = await session.get(ExperienceRow, experience_id)
        version = await session.get(ExperienceVersionRow, version_id)
        payload = await session.get(ExperiencePayloadRow, version_id)
    except (LookupError, StatementError, TypeError, ValueError) as error:
        raise _fail("Experience event source anchor is invalid") from error
    if identity is None or version is None or payload is None:
        raise _fail("Experience event source anchor is missing")
    if (
        event.aggregate_id != experience_id
        or version.experience_id != experience_id
        or version.created_at != event.occurred_at
    ):
        raise _fail("Experience event aggregate or source anchor is inconsistent")
    try:
        content = decode_and_verify_version(
            identity=identity,
            version=version,
            payload=payload,
        )
    except SourceIntegrityError as error:
        raise _fail("Experience event source content is corrupt") from error
    return identity, version, content


class ExperienceProjector:
    name = "experience_state"
    version = 1
    event_types = STATE_EXPERIENCE_EVENT_TYPES

    def __init__(
        self,
        event_registry: EventRegistry,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._event_registry = event_registry
        self._lifecycle_config = lifecycle_config or LifecycleConfig()

    def stored_event_from_row(self, row: DomainEventRow) -> StoredEvent:
        payload = self._event_registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
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

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target_table(target_prefix)
        await _create_state_rebuild_table(session, target)
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
                self.stored_event_from_row(row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.aggregate_type != "experience":
            raise _fail("Experience event has wrong aggregate type")
        target = _target_table(target_prefix)
        if isinstance(event.payload, ExperienceCreatedV1):
            await self._apply_created(session, event, event.payload, target)
            return
        if isinstance(event.payload, ExperienceVersionCreatedV1):
            await self._apply_version(session, event, event.payload, target)
            return
        if isinstance(event.payload, ExperienceAccessedV1):
            await self._apply_accessed(session, event, event.payload, target)
            return
        if isinstance(event.payload, ExperienceReactivatedV1):
            await self._apply_reactivated(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceLifecycleEvaluatedV1):
            await self._apply_lifecycle_evaluated(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceConfirmedV1):
            await self._apply_confirmed(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceCorroboratedV1):
            await self._apply_corroborated(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceRefutedV1):
            await self._apply_refuted(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperiencePinnedV1):
            await self._apply_pinned(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceUnpinnedV1):
            await self._apply_unpinned(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceArchivedV1):
            await self._apply_archived(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceRestoredV1):
            await self._apply_restored(
                session,
                event,
                event.payload,
                target,
            )
            return
        if isinstance(event.payload, ExperienceTemperatureChangedV1):
            await self._apply_temperature_changed(
                session,
                event,
                event.payload,
                target,
            )
            return
        raise _fail(f"Unsupported experience event {event.event_type!r}")

    async def _source_anchors(
        self,
        session: AsyncSession,
        *,
        event: StoredEvent,
        experience_id: UUID,
        version_id: UUID,
    ) -> tuple[ExperienceRow, ExperienceVersionRow]:
        identity, version, _ = await _load_source_anchors(
            session,
            event=event,
            experience_id=experience_id,
            version_id=version_id,
        )
        return identity, version

    async def _current_snapshot(
        self,
        session: AsyncSession,
        target: str,
        experience_id: UUID,
    ) -> tuple[ExperienceStateSnapshotV1, int] | None:
        result = await session.execute(
            text(
                f"SELECT * FROM {target} WHERE experience_id = :experience_id"
            ),
            {"experience_id": str(experience_id)},
        )
        mapping = result.mappings().one_or_none()
        return (
            None
            if mapping is None
            else (
                _snapshot_from_mapping(cast(Mapping[str, Any], mapping)),
                _integer(
                    mapping["projection_event_id"],
                    label="projection_event_id",
                ),
            )
        )

    async def _state_event_context(
        self,
        session: AsyncSession,
        *,
        event: StoredEvent,
        target: str,
        experience_id: UUID,
        before: ExperienceStateSnapshotV1,
    ) -> tuple[ExperienceRow, ExperienceVersionRow, DomainEventRow]:
        if (
            event.aggregate_type != "experience"
            or event.aggregate_id != experience_id
            or event.sequence < 3
        ):
            raise _fail("State event has wrong aggregate or sequence")
        current_with_checkpoint = await self._current_snapshot(
            session,
            target,
            experience_id,
        )
        if current_with_checkpoint is None:
            raise _fail("State event projection row is missing")
        current, prior_event_id = current_with_checkpoint
        if current != before:
            raise _fail("State event before state does not match projection")

        identity = await session.get(ExperienceRow, experience_id)
        version = await session.get(
            ExperienceVersionRow,
            before.current_version_id,
        )
        if (
            identity is None
            or version is None
            or identity.owner_agent_id != before.owner_agent_id
            or version.experience_id != experience_id
            or version.content_hash != before.current_content_hash
        ):
            raise _fail("State event source anchor is inconsistent")

        prior_event = await session.get(DomainEventRow, prior_event_id)
        if (
            prior_event is None
            or prior_event.aggregate_type != "experience"
            or prior_event.aggregate_id != experience_id
            or prior_event.event_type not in self.event_types
            or prior_event.sequence != event.sequence - 1
            or prior_event.event_id >= event.event_id
        ):
            raise _fail("State event prior checkpoint is inconsistent")
        try:
            prior_payload = self._event_registry.decode(
                event_type=prior_event.event_type,
                payload=prior_event.payload,
            )
        except (TypeError, ValueError) as error:
            raise _fail("State event prior checkpoint payload is invalid") from error
        if getattr(prior_payload, "experience_id", None) != experience_id:
            raise _fail("State event prior checkpoint payload is invalid")

        causal_times = [
            identity.created_at,
            version.created_at,
            prior_event.occurred_at,
            before.strength_updated_at,
            before.last_transition_at,
        ]
        causal_times.extend(
            value
            for value in (
                before.last_accessed_at,
                before.last_lifecycle_evaluated_at,
            )
            if value is not None
        )
        if event.occurred_at < max(causal_times):
            raise _fail("State event command clock regresses state")
        return identity, version, prior_event

    async def _store_state_event(
        self,
        session: AsyncSession,
        *,
        target: str,
        event: StoredEvent,
        after: ExperienceStateSnapshotV1,
    ) -> None:
        parameters = _snapshot_parameters(after, event_id=event.event_id)
        assignments = ", ".join(
            f"{column} = :{column}"
            for column in (*_SEMANTIC_FIELDS, "projection_event_id")
            if column != "experience_id"
        )
        result = await session.execute(
            text(
                f"UPDATE {target} SET {assignments} "
                "WHERE experience_id = :experience_id"
            ),
            parameters,
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("State event projection update did not affect one row")

    def _prior_payload(self, prior_event: DomainEventRow) -> Any:
        try:
            return self._event_registry.decode(
                event_type=prior_event.event_type,
                payload=prior_event.payload,
            )
        except (TypeError, ValueError) as error:
            raise _fail("Lifecycle predecessor payload is invalid") from error

    @staticmethod
    def _require_allowed_changes(
        *,
        before: ExperienceStateSnapshotV1,
        after: ExperienceStateSnapshotV1,
        allowed: frozenset[str],
        label: str,
    ) -> None:
        if not _changed_snapshot_fields(before, after) <= allowed:
            raise _fail(f"{label} event changes unauthorized state")
        if before.temperature is not after.temperature:
            raise _fail(f"{label} event cannot change temperature")

    def _require_materialization(
        self,
        *,
        identity: ExperienceRow,
        before: ExperienceStateSnapshotV1,
        after: ExperienceStateSnapshotV1,
        occurred_at: datetime,
        confidence: float,
        label: str,
    ) -> None:
        materialized = activation_at(
            ActivationInputs(
                importance=before.importance,
                confidence=confidence,
                access_count=before.access_count,
                access_strength=before.access_strength,
                strength_updated_at=before.strength_updated_at,
                last_accessed_at=before.last_accessed_at,
                created_at=identity.created_at,
            ),
            occurred_at,
            self._lifecycle_config,
        )
        if (
            after.strength_updated_at != occurred_at
            or abs(
                after.access_strength - materialized.decayed_strength
            )
            > 1e-12
            or abs(after.activation_score - materialized.score) > 1e-12
        ):
            raise _fail(f"{label} event materialization is inconsistent")

    def _require_lifecycle_evaluation_policy(
        self,
        *,
        payload: ExperienceLifecycleEvaluatedV1,
        occurred_at: datetime,
    ) -> None:
        before = payload.before
        after = payload.after
        if (
            before.last_lifecycle_evaluated_at is not None
            and occurred_at - before.last_lifecycle_evaluated_at
            < self._lifecycle_config.minimum_cycle_interval
        ):
            raise _fail(
                "Lifecycle-evaluated event violates the minimum interval"
            )

        target = payload.threshold_target
        activation = after.activation_score
        expected_counter = 0
        target_is_valid = False
        if target == "none":
            if before.temperature is Temperature.HOT:
                target_is_valid = (
                    before.pinned
                    or activation
                    >= self._lifecycle_config.hot_to_warm_threshold
                )
            elif before.temperature is Temperature.WARM:
                target_is_valid = (
                    not before.pinned
                    and activation
                    < self._lifecycle_config.warm_to_hot_threshold
                    and activation
                    >= self._lifecycle_config.warm_to_cold_threshold
                )
            else:
                target_is_valid = before.temperature is Temperature.COLD
        elif target == "promote_hot":
            target_is_valid = (
                before.temperature is Temperature.WARM
                and (
                    before.pinned
                    or activation
                    >= self._lifecycle_config.warm_to_hot_threshold
                )
            )
        elif target == "demote_warm":
            target_is_valid = (
                before.temperature is Temperature.HOT
                and not before.pinned
                and activation
                < self._lifecycle_config.hot_to_warm_threshold
            )
            expected_counter = min(
                self._lifecycle_config.demotion_cycles,
                before.consecutive_below_threshold + 1,
            )
        elif target == "demote_cold":
            target_is_valid = (
                before.temperature is Temperature.WARM
                and not before.pinned
                and activation
                < self._lifecycle_config.warm_to_cold_threshold
            )
            expected_counter = min(
                self._lifecycle_config.demotion_cycles,
                before.consecutive_below_threshold + 1,
            )
        elif target == "archive":
            target_is_valid = (
                before.temperature is Temperature.COLD
                and occurred_at - before.last_transition_at
                >= timedelta(
                    days=self._lifecycle_config.archive_after_days
                )
                and before.importance
                < self._lifecycle_config.archive_importance_threshold
                and before.confidence
                < self._lifecycle_config.archive_confidence_threshold
                and after.access_strength
                < self._lifecycle_config.archive_strength_threshold
                and not before.pinned
            )
        if not target_is_valid:
            raise _fail(
                "Lifecycle-evaluated event threshold target is inconsistent"
            )
        if after.consecutive_below_threshold != expected_counter:
            raise _fail(
                "Lifecycle-evaluated event hysteresis counter is inconsistent"
            )

    @staticmethod
    def _require_paired_predecessor(
        *,
        event: StoredEvent,
        prior_event: DomainEventRow,
        label: str,
    ) -> None:
        if (
            prior_event.causation_id != event.causation_id
            or prior_event.occurred_at != event.occurred_at
        ):
            raise _fail(f"{label} event has inconsistent causal predecessor")

    async def _apply_created(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceCreatedV1,
        target: str,
    ) -> None:
        if (
            event.event_type != ExperienceCreatedV1.event_type
            or event.aggregate_id != payload.experience_id
            or event.sequence != 1
        ):
            raise _fail("Created event has wrong aggregate or event type")
        identity, version = await self._source_anchors(
            session,
            event=event,
            experience_id=payload.experience_id,
            version_id=payload.version_id,
        )
        after = payload.after
        if (
            identity.owner_agent_id != after.owner_agent_id
            or identity.created_at != event.occurred_at
            or version.version_number != 1
            or version.supersedes_version_id is not None
            or version.content_hash != after.current_content_hash
            or after.strength_updated_at != event.occurred_at
            or after.last_transition_at != event.occurred_at
            or after.last_accessed_at is not None
            or after.last_lifecycle_evaluated_at is not None
            or after.access_count != 0
            or after.access_strength != 0.0
            or after.consecutive_below_threshold != 0
            or after.pinned
        ):
            raise _fail("Created event does not match its initial source anchors")
        expected_activation = activation_at(
            ActivationInputs(
                importance=after.importance,
                confidence=after.confidence,
                access_count=0,
                access_strength=0.0,
                strength_updated_at=event.occurred_at,
                last_accessed_at=None,
                created_at=identity.created_at,
            ),
            event.occurred_at,
            self._lifecycle_config,
        ).score
        if abs(expected_activation - after.activation_score) > 1e-12:
            raise _fail("Created event has inconsistent initial activation")
        if (
            await self._current_snapshot(
                session,
                target,
                payload.experience_id,
            )
            is not None
        ):
            raise _fail("Created event would replace an existing projection")
        parameters = _snapshot_parameters(after, event_id=event.event_id)
        columns = (*_SEMANTIC_FIELDS, "projection_event_id")
        column_sql = ", ".join(columns)
        value_sql = ", ".join(f":{column}" for column in columns)
        await session.execute(
            text(f"INSERT INTO {target} ({column_sql}) VALUES ({value_sql})"),
            parameters,
        )

    async def _apply_version(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceVersionCreatedV1,
        target: str,
    ) -> None:
        if (
            event.event_type != ExperienceVersionCreatedV1.event_type
            or event.aggregate_id != payload.experience_id
            or event.sequence < 2
        ):
            raise _fail("Version event has wrong aggregate or event type")
        if not is_valid_version_event_sequence(
            version_number=payload.version_number,
            aggregate_sequence=event.sequence,
        ):
            raise _fail("Version event has an impossible aggregate sequence")
        identity, version = await self._source_anchors(
            session,
            event=event,
            experience_id=payload.experience_id,
            version_id=payload.version_id,
        )
        if (
            identity.owner_agent_id != payload.after.owner_agent_id
            or version.version_number != payload.version_number
            or version.supersedes_version_id != payload.supersedes_version_id
            or version.content_hash != payload.after.current_content_hash
        ):
            raise _fail("Version event does not match source anchors")
        superseded: ExperienceVersionRow | None = None
        if payload.version_number > 1:
            assert payload.supersedes_version_id is not None
            superseded = await session.get(
                ExperienceVersionRow,
                payload.supersedes_version_id,
            )
            if (
                superseded is None
                or superseded.experience_id != payload.experience_id
                or superseded.version_id != payload.before.current_version_id
                or superseded.version_number != payload.version_number - 1
                or superseded.content_hash
                != payload.before.current_content_hash
            ):
                raise _fail("Version event supersession source is inconsistent")
        current_with_checkpoint = await self._current_snapshot(
            session,
            target,
            payload.experience_id,
        )
        if current_with_checkpoint is None:
            raise _fail("Version event projection row is missing")
        current, prior_event_id = current_with_checkpoint
        if current != payload.before:
            raise _fail("Version event before state does not match projection")
        prior_event = await session.get(DomainEventRow, prior_event_id)
        if (
            prior_event is None
            or prior_event.aggregate_type != "experience"
            or prior_event.aggregate_id != payload.experience_id
            or prior_event.event_type not in self.event_types
            or prior_event.sequence != event.sequence - 1
            or prior_event.event_id >= event.event_id
        ):
            raise _fail("Version event prior checkpoint is inconsistent")
        try:
            prior_payload = self._event_registry.decode(
                event_type=prior_event.event_type,
                payload=prior_event.payload,
            )
        except (TypeError, ValueError) as error:
            raise _fail(
                "Version event prior checkpoint payload is invalid"
            ) from error
        if (
            getattr(prior_payload, "experience_id", None)
            != payload.experience_id
        ):
            raise _fail("Version event prior checkpoint payload is invalid")
        changed = frozenset(
            name
            for name in _SEMANTIC_FIELDS
            if getattr(payload.before, name) != getattr(payload.after, name)
        )
        if payload.version_number == 1:
            if changed:
                raise _fail("Initial version event must be a semantic no-op")
        else:
            assert superseded is not None
            if (
                not changed <= _CORRECTION_FIELDS
                or payload.supersedes_version_id
                != payload.before.current_version_id
                or payload.after.current_version_id != payload.version_id
            ):
                raise _fail("Version event changes unauthorized state")
            causal_times = [
                identity.created_at,
                superseded.created_at,
                prior_event.occurred_at,
                payload.before.strength_updated_at,
                payload.before.last_transition_at,
            ]
            causal_times.extend(
                value
                for value in (
                    payload.before.last_accessed_at,
                    payload.before.last_lifecycle_evaluated_at,
                )
                if value is not None
            )
            if event.occurred_at < max(causal_times):
                raise _fail("Version event command clock regresses state")
            materialized = activation_at(
                ActivationInputs(
                    importance=payload.before.importance,
                    confidence=payload.before.confidence,
                    access_count=payload.before.access_count,
                    access_strength=payload.before.access_strength,
                    strength_updated_at=payload.before.strength_updated_at,
                    last_accessed_at=payload.before.last_accessed_at,
                    created_at=identity.created_at,
                ),
                event.occurred_at,
                self._lifecycle_config,
            )
            if (
                payload.after.strength_updated_at != event.occurred_at
                or abs(
                    payload.after.access_strength
                    - materialized.decayed_strength
                )
                > 1e-12
                or abs(
                    payload.after.activation_score - materialized.score
                )
                > 1e-12
            ):
                raise _fail("Version event materialization is inconsistent")

        parameters = _snapshot_parameters(payload.after, event_id=event.event_id)
        assignments = ", ".join(
            f"{column} = :{column}"
            for column in (*_SEMANTIC_FIELDS, "projection_event_id")
            if column != "experience_id"
        )
        result = await session.execute(
            text(
                f"UPDATE {target} SET {assignments} "
                "WHERE experience_id = :experience_id"
            ),
            parameters,
        )
        if cast(Any, result).rowcount != 1:
            raise _fail("Version event projection update did not affect one row")

    async def _apply_accessed(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceAccessedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceAccessedV1.event_type:
            raise _fail("Accessed event has wrong event type")
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be accessed")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        inputs = ActivationInputs(
            importance=payload.before.importance,
            confidence=payload.before.confidence,
            access_count=payload.before.access_count,
            access_strength=payload.before.access_strength,
            strength_updated_at=payload.before.strength_updated_at,
            last_accessed_at=payload.before.last_accessed_at,
            created_at=identity.created_at,
        )
        try:
            access = record_access(
                inputs,
                event.occurred_at,
                self._lifecycle_config,
            )
        except ValueError as error:
            raise _fail("Accessed event command clock regresses state") from error
        materialized = activation_at(
            ActivationInputs(
                importance=payload.before.importance,
                confidence=payload.before.confidence,
                access_count=access.access_count,
                access_strength=access.access_strength,
                strength_updated_at=access.strength_updated_at,
                last_accessed_at=access.last_accessed_at,
                created_at=identity.created_at,
            ),
            event.occurred_at,
            self._lifecycle_config,
        )
        if (
            payload.after.access_count != access.access_count
            or abs(
                payload.after.access_strength - access.access_strength
            )
            > 1e-12
            or payload.after.strength_updated_at
            != access.strength_updated_at
            or payload.after.last_accessed_at != access.last_accessed_at
            or abs(
                payload.after.activation_score - materialized.score
            )
            > 1e-12
        ):
            raise _fail("Accessed event materialization is inconsistent")
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_reactivated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceReactivatedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceReactivatedV1.event_type:
            raise _fail("Reactivated event has wrong event type")
        _, _, prior_event = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        if (
            payload.before.temperature is not Temperature.COLD
            or prior_event.event_type != ExperienceAccessedV1.event_type
            or prior_event.causation_id != event.causation_id
            or prior_event.occurred_at != event.occurred_at
        ):
            raise _fail("Reactivated event does not follow cold access")
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_lifecycle_evaluated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceLifecycleEvaluatedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceLifecycleEvaluatedV1.event_type:
            raise _fail("Lifecycle-evaluated event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_LIFECYCLE_EVALUATION_FIELDS,
            label="Lifecycle-evaluated",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be lifecycle evaluated")
        if (
            payload.evaluated_at != event.occurred_at
            or payload.after.last_lifecycle_evaluated_at
            != event.occurred_at
        ):
            raise _fail(
                "Lifecycle-evaluated event evaluation time is inconsistent"
            )
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=payload.before.confidence,
            label="Lifecycle-evaluated",
        )
        self._require_lifecycle_evaluation_policy(
            payload=payload,
            occurred_at=event.occurred_at,
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_confirmed(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceConfirmedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceConfirmedV1.event_type:
            raise _fail("Confirmed event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_CONFIDENCE_EVENT_FIELDS,
            label="Confirmed",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be confirmed")
        expected_confidence = payload.before.confidence + (
            1.0 - payload.before.confidence
        ) * 0.20
        if (
            abs(payload.after.confidence - expected_confidence)
            > 1e-12
        ):
            raise _fail("Confirmed event confidence formula is inconsistent")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=expected_confidence,
            label="Confirmed",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_refuted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceRefutedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceRefutedV1.event_type:
            raise _fail("Refuted event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_CONFIDENCE_EVENT_FIELDS,
            label="Refuted",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be refuted")
        expected_confidence = payload.before.confidence * 0.65
        if (
            abs(payload.after.confidence - expected_confidence)
            > 1e-12
        ):
            raise _fail("Refuted event confidence formula is inconsistent")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=expected_confidence,
            label="Refuted",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_corroborated(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceCorroboratedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceCorroboratedV1.event_type:
            raise _fail("Corroborated event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_CONFIDENCE_EVENT_FIELDS,
            label="Corroborated",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be corroborated")
        adoption = await session.get(AdoptionRecordRow, payload.adoption_id)
        capsule = await session.get(ExperienceCapsuleRow, payload.capsule_id)
        if adoption is None or capsule is None:
            raise _fail("Corroborated event provenance source is missing")
        other_adoptions = tuple(
            (
                await session.scalars(
                    select(AdoptionRecordRow).where(
                        AdoptionRecordRow.resulting_experience_id
                        == payload.experience_id,
                        AdoptionRecordRow.root_fingerprint
                        == payload.root_fingerprint,
                        AdoptionRecordRow.adoption_id
                        != payload.adoption_id,
                    )
                )
            ).all()
        )
        adoption_events = await _adoption_event_index(
            session,
            event_registry=self._event_registry,
        )
        if any(
            row.adoption_id not in adoption_events
            for row in other_adoptions
        ):
            adoption_events = await _adoption_event_index(
                session,
                event_registry=self._event_registry,
                refresh=True,
            )
        overlapping_root = False
        prior_sources_valid = True
        for prior_adoption in other_adoptions:
            event_pair = adoption_events.get(prior_adoption.adoption_id)
            if event_pair is None:
                prior_sources_valid = False
                break
            prior_event, prior_payload = event_pair
            if (
                prior_payload.adopter_agent_id
                != prior_adoption.adopter_agent_id
                or prior_payload.capsule_id != prior_adoption.capsule_id
                or prior_payload.resulting_experience_id
                != prior_adoption.resulting_experience_id
                or prior_payload.root_fingerprint
                != prior_adoption.root_fingerprint
                or prior_payload.corroboration_applied
                is not prior_adoption.corroboration_applied
                or prior_event.occurred_at != prior_adoption.adopted_at
            ):
                prior_sources_valid = False
                break
            if prior_event.event_id < event.event_id:
                overlapping_root = True
                break
        try:
            parent_chain = json.loads(capsule.provenance_chain)
            expected_chain = (
                *parent_chain,
                {
                    "capsule_id": str(capsule.capsule_id),
                    "publisher_agent_id": str(capsule.publisher_agent_id),
                },
            )
            provenance_matches = (
                isinstance(parent_chain, list)
                and canonical_json_bytes(expected_chain)
                == adoption.provenance_chain
            )
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            provenance_matches = False
        if (
            adoption.adopter_agent_id != payload.before.owner_agent_id
            or adoption.capsule_id != payload.capsule_id
            or adoption.resulting_experience_id != payload.experience_id
            or adoption.root_fingerprint != payload.root_fingerprint
            or not adoption.corroboration_applied
            or adoption.adopted_at != event.occurred_at
            or abs(adoption.captured_trust - payload.captured_trust) > 1e-12
            or capsule.root_fingerprint != payload.root_fingerprint
            or capsule.source_content_hash
            != payload.before.current_content_hash
            or capsule.created_at > event.occurred_at
            or identity.owner_agent_id != adoption.adopter_agent_id
            or not provenance_matches
            or not prior_sources_valid
            or overlapping_root
        ):
            raise _fail("Corroborated event provenance source is inconsistent")
        expected_confidence = payload.before.confidence + (
            1.0 - payload.before.confidence
        ) * 0.20 * payload.captured_trust
        if abs(payload.after.confidence - expected_confidence) > 1e-12:
            raise _fail("Corroborated event confidence formula is inconsistent")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=expected_confidence,
            label="Corroborated",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_pinned(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperiencePinnedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperiencePinnedV1.event_type:
            raise _fail("Pinned event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_PIN_EVENT_FIELDS,
            label="Pinned",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be pinned")
        if payload.before.pinned or not payload.after.pinned:
            raise _fail("Pinned event pin flip is inconsistent")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=payload.before.confidence,
            label="Pinned",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_unpinned(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceUnpinnedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceUnpinnedV1.event_type:
            raise _fail("Unpinned event has wrong event type")
        identity, _, _ = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_PIN_EVENT_FIELDS,
            label="Unpinned",
        )
        if payload.before.temperature is Temperature.ARCHIVED:
            raise _fail("Archived experience cannot be unpinned")
        if not payload.before.pinned or payload.after.pinned:
            raise _fail("Unpinned event pin flip is inconsistent")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=payload.before.confidence,
            label="Unpinned",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_archived(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceArchivedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceArchivedV1.event_type:
            raise _fail("Archived event has wrong event type")
        _, _, prior_event = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=frozenset(),
            label="Archived",
        )
        prior_payload = self._prior_payload(prior_event)
        if (
            payload.before != payload.after
            or payload.before.temperature is not Temperature.COLD
            or payload.reason != StructuredReason.policy_due()
            or not isinstance(
                prior_payload,
                ExperienceLifecycleEvaluatedV1,
            )
            or prior_payload.cycle_id != payload.cycle_id
            or prior_payload.threshold_target != "archive"
            or prior_payload.after != payload.before
        ):
            raise _fail(
                "Archived event does not follow its lifecycle evaluation"
            )
        self._require_paired_predecessor(
            event=event,
            prior_event=prior_event,
            label="Archived",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_restored(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceRestoredV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceRestoredV1.event_type:
            raise _fail("Restored event has wrong event type")
        identity, _, prior_event = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        self._require_allowed_changes(
            before=payload.before,
            after=payload.after,
            allowed=_RESTORE_EVENT_FIELDS,
            label="Restored",
        )
        prior_payload = self._prior_payload(prior_event)
        if (
            payload.before.temperature is not Temperature.ARCHIVED
            or payload.after.temperature is not Temperature.ARCHIVED
            or not isinstance(
                prior_payload,
                ExperienceTemperatureChangedV1,
            )
            or prior_payload.cause != "policy_archive"
            or prior_payload.after != payload.before
        ):
            raise _fail("Restored event has inconsistent archived predecessor")
        self._require_materialization(
            identity=identity,
            before=payload.before,
            after=payload.after,
            occurred_at=event.occurred_at,
            confidence=payload.before.confidence,
            label="Restored",
        )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )

    async def _apply_temperature_changed(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceTemperatureChangedV1,
        target: str,
    ) -> None:
        if event.event_type != ExperienceTemperatureChangedV1.event_type:
            raise _fail("Temperature-changed event has wrong event type")
        _, _, prior_event = await self._state_event_context(
            session,
            event=event,
            target=target,
            experience_id=payload.experience_id,
            before=payload.before,
        )
        if payload.after.last_transition_at != event.occurred_at:
            raise _fail(
                "Temperature-changed event transition time is inconsistent"
            )
        prior_payload = self._prior_payload(prior_event)
        if payload.cause == "cold_reactivation":
            if (
                not isinstance(prior_payload, ExperienceReactivatedV1)
                or prior_payload.after != payload.before
            ):
                raise _fail(
                    "Cold-reactivation temperature event has wrong predecessor"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Cold-reactivation temperature",
            )
        elif payload.cause == "confirmation":
            if (
                not isinstance(prior_payload, ExperienceConfirmedV1)
                or prior_payload.after != payload.before
            ):
                raise _fail(
                    "Confirmation temperature event does not follow "
                    "confirmed event"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Confirmation temperature",
            )
        elif payload.cause == "pin":
            if (
                not isinstance(prior_payload, ExperiencePinnedV1)
                or prior_payload.after != payload.before
            ):
                raise _fail(
                    "Pin temperature event does not follow pinned event"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Pin temperature",
            )
        elif payload.cause == "capsule_corroboration":
            if (
                not isinstance(prior_payload, ExperienceCorroboratedV1)
                or prior_payload.after != payload.before
            ):
                raise _fail(
                    "Capsule-corroboration temperature event does not follow "
                    "a corroborated event"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Capsule-corroboration temperature",
            )
        elif payload.cause in {
            "lifecycle_activation",
            "lifecycle_demotion",
        }:
            expected_target = (
                "promote_hot"
                if payload.cause == "lifecycle_activation"
                else (
                    "demote_warm"
                    if payload.after.temperature is Temperature.WARM
                    else "demote_cold"
                )
            )
            if (
                not isinstance(
                    prior_payload,
                    ExperienceLifecycleEvaluatedV1,
                )
                or prior_payload.after != payload.before
                or prior_payload.cycle_id != payload.cycle_id
                or prior_payload.threshold_target != expected_target
                or (
                    payload.cause == "lifecycle_demotion"
                    and prior_payload.after.consecutive_below_threshold
                    < self._lifecycle_config.demotion_cycles
                )
            ):
                raise _fail(
                    "Lifecycle temperature event does not follow "
                    "its evaluation"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Lifecycle temperature",
            )
        elif payload.cause == "policy_archive":
            if (
                not isinstance(prior_payload, ExperienceArchivedV1)
                or prior_payload.after != payload.before
                or prior_payload.cycle_id != payload.cycle_id
            ):
                raise _fail(
                    "Policy-archive temperature event does not follow "
                    "archived event"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Policy-archive temperature",
            )
        elif payload.cause == "restore":
            if (
                not isinstance(prior_payload, ExperienceRestoredV1)
                or prior_payload.after != payload.before
            ):
                raise _fail(
                    "Restore temperature event does not follow restored event"
                )
            self._require_paired_predecessor(
                event=event,
                prior_event=prior_event,
                label="Restore temperature",
            )
        await self._store_state_event(
            session,
            target=target,
            event=event,
            after=payload.after,
        )


class ExperienceTermsProjector:
    """Version-one reducer for the current multilingual term projection."""

    name = "experience_terms"
    version = 1
    event_types = TASK2_EXPERIENCE_EVENT_TYPES

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    def stored_event_from_row(self, row: DomainEventRow) -> StoredEvent:
        payload = self._event_registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
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

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target_table(target_prefix, self.name)
        await _create_terms_rebuild_table(session, target)
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
                self.stored_event_from_row(row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.aggregate_type != "experience":
            raise _fail("Experience term event has wrong aggregate type")
        if isinstance(event.payload, ExperienceCreatedV1):
            content = await self._created_content(session, event, event.payload)
            experience_id = event.payload.experience_id
        elif isinstance(event.payload, ExperienceVersionCreatedV1):
            content = await self._version_content(session, event, event.payload)
            experience_id = event.payload.experience_id
        else:
            raise _fail(
                f"Unsupported experience term event {event.event_type!r}"
            )
        await self._replace_terms(
            session,
            target=_target_table(target_prefix, self.name),
            experience_id=experience_id,
            content=content,
        )

    async def _created_content(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceCreatedV1,
    ) -> VersionContent:
        if (
            event.event_type != ExperienceCreatedV1.event_type
            or event.aggregate_id != payload.experience_id
            or event.sequence != 1
        ):
            raise _fail("Created term event has wrong aggregate or event type")
        identity, version, content = await _load_source_anchors(
            session,
            event=event,
            experience_id=payload.experience_id,
            version_id=payload.version_id,
        )
        if (
            identity.owner_agent_id != payload.after.owner_agent_id
            or identity.created_at != event.occurred_at
            or version.version_number != 1
            or version.supersedes_version_id is not None
            or version.content_hash != payload.after.current_content_hash
            or payload.after.current_version_id != payload.version_id
        ):
            raise _fail("Created term event does not match source anchors")
        return content

    async def _version_content(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: ExperienceVersionCreatedV1,
    ) -> VersionContent:
        if (
            event.event_type != ExperienceVersionCreatedV1.event_type
            or event.aggregate_id != payload.experience_id
            or not is_valid_version_event_sequence(
                version_number=payload.version_number,
                aggregate_sequence=event.sequence,
            )
        ):
            raise _fail("Version term event has wrong aggregate or event type")
        identity, version, content = await _load_source_anchors(
            session,
            event=event,
            experience_id=payload.experience_id,
            version_id=payload.version_id,
        )
        if (
            identity.owner_agent_id != payload.after.owner_agent_id
            or version.version_number != payload.version_number
            or version.supersedes_version_id != payload.supersedes_version_id
            or version.content_hash != payload.after.current_content_hash
            or payload.after.current_version_id != payload.version_id
        ):
            raise _fail("Version term event does not match source anchors")
        if payload.version_number == 1:
            if payload.before != payload.after:
                raise _fail("Initial version term event must be a no-op")
            return content

        assert payload.supersedes_version_id is not None
        superseded = await session.get(
            ExperienceVersionRow,
            payload.supersedes_version_id,
        )
        if (
            superseded is None
            or superseded.experience_id != payload.experience_id
            or superseded.version_number != payload.version_number - 1
            or superseded.content_hash != payload.before.current_content_hash
            or superseded.version_id != payload.before.current_version_id
        ):
            raise _fail("Version term supersession source is inconsistent")
        return content

    async def _replace_terms(
        self,
        session: AsyncSession,
        *,
        target: str,
        experience_id: UUID,
        content: VersionContent,
    ) -> None:
        cues = tuple(
            sorted(
                index_version_terms(content),
                key=lambda cue: (cue.term, cue.term_kind),
            )
        )
        await session.execute(
            text(f"DELETE FROM {target} WHERE experience_id = :experience_id"),
            {"experience_id": str(experience_id)},
        )
        if not cues:
            return
        await session.execute(
            text(
                f"INSERT INTO {target} "
                "(experience_id, term, term_kind, weight) "
                "VALUES (:experience_id, :term, :term_kind, :weight)"
            ),
            [
                {
                    "experience_id": str(experience_id),
                    "term": cue.term,
                    "term_kind": cue.term_kind,
                    "weight": cue.weight,
                }
                for cue in cues
            ],
        )


__all__ = [
    "ExperienceProjectionIntegrityError",
    "ExperienceProjector",
    "ExperienceTermsProjector",
]
