"""Agent command service."""

from sqlalchemy import select

from experience_hub.agents.events import AgentCreated
from experience_hub.agents.models import CreateAgent
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock
from experience_hub.domain.commands import CommandContext, ReplayableCommandError
from experience_hub.domain.events import PendingEvent
from experience_hub.ids import IdGenerator
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import ReceiptStore, StoredResponse
from experience_hub.storage.tables import AgentRow
from experience_hub.storage.unit_of_work import UnitOfWork


class AgentService:
    def __init__(
        self,
        *,
        clock: Clock,
        id_generator: IdGenerator,
        receipt_store: ReceiptStore,
    ) -> None:
        self._clock = clock
        self._id_generator = id_generator
        self._receipt_store = receipt_store

    async def create(
        self,
        *,
        uow: UnitOfWork,
        command: CreateAgent,
        command_context: CommandContext,
    ) -> StoredResponse:
        name = command.name.strip()
        if not name:
            raise ReplayableCommandError(
                code="agent_name_required",
                message="Agent name must not be empty",
                status_code=422,
            )
        duplicate = await uow.session.scalar(
            select(AgentRow.agent_id).where(AgentRow.name == name)
        )
        if duplicate is not None:
            raise ReplayableCommandError(
                code="agent_name_conflict",
                message="An agent with this name already exists",
                details={"name": name},
                status_code=409,
            )

        created_at = self._clock.now()
        agent_id = self._id_generator.new()
        uow.session.add(AgentRow(agent_id=agent_id, name=name, created_at=created_at))
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="agent",
            resource_id=agent_id,
        )
        await uow.append_events(
            command=command_context,
            events=[
                PendingEvent(
                    aggregate_type="agent",
                    aggregate_id=agent_id,
                    event_type=AgentCreated.event_type,
                    payload=AgentCreated(
                        schema_version=1,
                        agent_id=agent_id,
                        name=name,
                    ),
                    actor_agent_id=None,
                    occurred_at=created_at,
                )
            ],
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {"data": {"agent_id": str(agent_id), "name": name}}
            ),
            headers={"location": f"/v1/agents/{agent_id}"},
        )
