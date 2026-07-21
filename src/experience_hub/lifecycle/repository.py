"""Transaction-bound lifecycle state and singleton lease persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    ExperienceStateSnapshotV1,
    register_experience_events,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    Temperature,
)
from experience_hub.experiences.repository import (
    require_current_aggregate_head,
    snapshot_from_state_row,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    LifecycleLeaseRow,
)
from experience_hub.storage.validation import SourceIntegrityError

_LEASE_NAME = "lifecycle"


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """Immutable current experience inputs required by lifecycle planning."""

    experience_id: UUID
    owner_agent_id: UUID
    kind: ExperienceKind
    origin: ExperienceOrigin
    created_at: datetime
    current_version_id: UUID
    current_version_number: int
    current_version_created_at: datetime
    current_content_hash: str
    state: ExperienceStateSnapshotV1
    projection_event_id: int
    latest_causal_at: datetime


class LifecycleRepository:
    """Persist lifecycle coordination state inside a caller-owned transaction."""

    @staticmethod
    async def ensure_lease(session: AsyncSession) -> None:
        if await session.get(LifecycleLeaseRow, _LEASE_NAME) is not None:
            return
        session.add(LifecycleLeaseRow(lease_name=_LEASE_NAME))
        await session.flush()

    async def claim_lease(
        self,
        session: AsyncSession,
        *,
        owner_id: UUID,
        at: datetime,
        ttl: timedelta,
    ) -> bool:
        if not isinstance(owner_id, UUID):
            raise ValueError("Lifecycle lease owner_id must be a UUID")
        acquired_at = require_utc(at)
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("Lifecycle lease ttl must be positive")
        await self.ensure_lease(session)
        lease = await session.get(LifecycleLeaseRow, _LEASE_NAME)
        if lease is None:
            raise RuntimeError("Lifecycle lease was not initialized")
        if (
            lease.owner_id is not None
            and lease.owner_id != owner_id
            and lease.expires_at is not None
            and lease.expires_at > acquired_at
        ):
            return False
        lease.owner_id = owner_id
        lease.acquired_at = acquired_at
        lease.expires_at = acquired_at + ttl
        await session.flush()
        return True

    @staticmethod
    async def release_lease(
        session: AsyncSession,
        *,
        owner_id: UUID,
    ) -> bool:
        if not isinstance(owner_id, UUID):
            raise ValueError("Lifecycle lease owner_id must be a UUID")
        lease = await session.get(LifecycleLeaseRow, _LEASE_NAME)
        if lease is None or lease.owner_id != owner_id:
            return False
        lease.owner_id = None
        lease.acquired_at = None
        lease.expires_at = None
        await session.flush()
        return True

    @staticmethod
    async def list_current(
        session: AsyncSession,
    ) -> tuple[LifecycleRecord, ...]:
        event_registry = EventRegistry()
        register_experience_events(event_registry)
        rows = (
            await session.execute(
                select(
                    ExperienceRow,
                    ExperienceVersionRow,
                    ExperienceStateRow,
                    DomainEventRow,
                )
                .select_from(ExperienceRow)
                .join(
                    ExperienceStateRow,
                    ExperienceStateRow.experience_id == ExperienceRow.experience_id,
                )
                .join(
                    ExperienceVersionRow,
                    ExperienceVersionRow.version_id
                    == ExperienceStateRow.current_version_id,
                )
                .join(
                    DomainEventRow,
                    DomainEventRow.event_id == ExperienceStateRow.projection_event_id,
                )
            )
        ).all()
        identity_count = await session.scalar(
            select(func.count()).select_from(ExperienceRow)
        )
        if identity_count != len(rows):
            raise SourceIntegrityError(
                "Experience lifecycle inputs have incomplete current state"
            )

        records: list[LifecycleRecord] = []
        for identity, version, state, projection_event in rows:
            experience_id = identity.experience_id
            if (
                state.experience_id != experience_id
                or state.owner_agent_id != identity.owner_agent_id
                or version.experience_id != experience_id
                or state.current_version_id != version.version_id
                or state.current_content_hash != version.content_hash
                or projection_event.aggregate_type != "experience"
                or projection_event.aggregate_id != experience_id
                or projection_event.event_id != state.projection_event_id
            ):
                raise SourceIntegrityError(
                    f"Experience {experience_id} has inconsistent lifecycle state"
                )
            await require_current_aggregate_head(
                session=session,
                experience_id=experience_id,
                projection_event=projection_event,
                event_registry=event_registry,
                handled_event_types=STATE_EXPERIENCE_EVENT_TYPES,
            )
            snapshot = snapshot_from_state_row(state)
            causal_times = [
                identity.created_at,
                version.created_at,
                projection_event.occurred_at,
                state.strength_updated_at,
                state.last_transition_at,
            ]
            causal_times.extend(
                value
                for value in (
                    state.last_accessed_at,
                    state.last_lifecycle_evaluated_at,
                )
                if value is not None
            )
            records.append(
                LifecycleRecord(
                    experience_id=experience_id,
                    owner_agent_id=identity.owner_agent_id,
                    kind=identity.kind,
                    origin=identity.origin,
                    created_at=require_utc(identity.created_at),
                    current_version_id=version.version_id,
                    current_version_number=version.version_number,
                    current_version_created_at=require_utc(version.created_at),
                    current_content_hash=version.content_hash,
                    state=snapshot,
                    projection_event_id=projection_event.event_id,
                    latest_causal_at=max(require_utc(value) for value in causal_times),
                )
            )
        return tuple(sorted(records, key=lambda record: record.experience_id.bytes))

    @staticmethod
    async def active_dependent_target_ids(
        session: AsyncSession,
    ) -> frozenset[UUID]:
        target_ids = await session.scalars(
            select(ExperienceLinkRow.target_experience_id)
            .distinct()
            .join(
                ExperienceStateRow,
                ExperienceStateRow.experience_id
                == ExperienceLinkRow.source_experience_id,
            )
            .where(
                ExperienceStateRow.current_version_id
                == ExperienceLinkRow.source_version_id,
                ExperienceStateRow.temperature != Temperature.ARCHIVED,
                ExperienceLinkRow.relation.in_(
                    (
                        LinkRelation.DERIVED_FROM,
                        LinkRelation.SUPPORTS,
                        LinkRelation.TESTS,
                    )
                ),
            )
        )
        return frozenset(target_ids.all())
