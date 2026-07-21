"""Transaction-bound repository access."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain.commands import CommandContext
from experience_hub.domain.events import PendingEvent, StoredEvent
from experience_hub.storage.event_store import EventStore
from experience_hub.storage.faults import FaultCheckpoint, FaultInjector
from experience_hub.storage.projection_contracts import ProjectionApplier


@dataclass(slots=True)
class UnitOfWork:
    session: AsyncSession
    event_store: EventStore
    projection_applier: ProjectionApplier
    immediate: bool
    fault_injector: FaultInjector

    def inject_fault(self, checkpoint: FaultCheckpoint) -> None:
        self.fault_injector(checkpoint)

    async def append_events(
        self,
        command: CommandContext,
        events: Sequence[PendingEvent],
    ) -> tuple[StoredEvent, ...]:
        stored = await self.event_store.append(
            session=self.session,
            causation_id=command.receipt_id,
            events=events,
            immediate_transaction=self.immediate,
        )
        self.inject_fault(FaultCheckpoint.AFTER_EVENT_APPEND)
        await self.projection_applier.apply(session=self.session, events=stored)
        self.inject_fault(FaultCheckpoint.AFTER_PROJECTION_APPLY)
        return stored
