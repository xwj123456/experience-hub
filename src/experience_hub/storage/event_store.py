"""Append-only persistence for strict domain events."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain.events import (
    EventRegistry,
    PendingEvent,
    StoredEvent,
)
from experience_hub.storage.tables import DomainEventRow, IdempotencyRecordRow


class EventStore:
    """Allocate aggregate sequences and append events in declared order."""

    def __init__(self, registry: EventRegistry) -> None:
        self._registry = registry

    async def append(
        self,
        *,
        session: AsyncSession,
        causation_id: UUID,
        events: Sequence[PendingEvent],
        immediate_transaction: bool,
    ) -> tuple[StoredEvent, ...]:
        if not immediate_transaction:
            raise RuntimeError("Event append requires an immediate transaction")

        receipt_exists = await session.scalar(
            select(IdempotencyRecordRow.receipt_id).where(
                IdempotencyRecordRow.receipt_id == causation_id
            )
        )
        if receipt_exists is None:
            raise ValueError("Event causation receipt does not exist")

        next_sequences: dict[tuple[str, UUID], int] = {}
        rows: list[tuple[DomainEventRow, PendingEvent]] = []
        for event in events:
            registered_type = self._registry.payload_type(event.event_type)
            payload_type = type(event.payload)
            try:
                payload_event_type = payload_type.event_type
            except AttributeError as error:
                raise ValueError(
                    "Event payload class must declare an event type"
                ) from error
            if payload_event_type != event.event_type:
                raise ValueError(
                    "Pending event type does not match its payload event type"
                )
            if registered_type is not payload_type:
                raise ValueError(
                    "Pending event payload class does not match its registered "
                    "event type"
                )

            aggregate = (event.aggregate_type, event.aggregate_id)
            sequence = next_sequences.get(aggregate)
            if sequence is None:
                current = await session.scalar(
                    select(func.max(DomainEventRow.sequence)).where(
                        DomainEventRow.aggregate_type == event.aggregate_type,
                        DomainEventRow.aggregate_id == event.aggregate_id,
                    )
                )
                sequence = (current or 0) + 1
            next_sequences[aggregate] = sequence + 1

            row = DomainEventRow(
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                sequence=sequence,
                event_type=event.event_type,
                payload=canonical_json_bytes(event.payload),
                actor_agent_id=event.actor_agent_id,
                causation_id=causation_id,
                occurred_at=event.occurred_at,
            )
            session.add(row)
            rows.append((row, event))

        await session.flush()
        return tuple(
            StoredEvent(
                event_id=row.event_id,
                aggregate_type=row.aggregate_type,
                aggregate_id=row.aggregate_id,
                sequence=row.sequence,
                event_type=row.event_type,
                payload=event.payload,
                actor_agent_id=row.actor_agent_id,
                causation_id=row.causation_id,
                occurred_at=row.occurred_at,
            )
            for row, event in rows
        )
