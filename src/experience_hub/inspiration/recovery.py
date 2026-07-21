"""Ledger-driven startup recovery for interrupted inspiration runs."""

from __future__ import annotations

from uuid import UUID

from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandRequest, PendingEvent
from experience_hub.inspiration.events import (
    InspirationFailedV1,
    InspirationRunFailureCode,
)
from experience_hub.inspiration.models import InspirationRunStatus
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
    RunningInspirationTrace,
)
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import (
    CompletedReceipt,
    InProgressReceipt,
    NewReceipt,
    ReceiptRecord,
    ReceiptStore,
)


def _recovery_request(run_id: UUID) -> CommandRequest:
    return CommandRequest(
        caller_scope="system:local",
        operation_scope="inspiration.run.recover",
        idempotency_key=f"recovery:{run_id}",
        method="POST",
        route_template="/internal/inspiration-runs/{run_id}:recover",
        path_parameters={"run_id": run_id},
        body={"failure_code": "process_interrupted"},
    )


def _validate_original_receipt(
    trace: RunningInspirationTrace,
    record: ReceiptRecord | None,
) -> ReceiptRecord:
    if (
        record is None
        or record.receipt_id != trace.receipt_id
        or record.caller_scope != f"agent:{trace.owner_agent_id}"
        or record.operation_scope != "inspiration.run.start"
        or record.request_hash != trace.request_hash
        or record.state != "in_progress"
        or record.result_resource_type != "inspiration_run"
        or record.result_resource_id != trace.run_id
        or record.response is not None
    ):
        raise InspirationSourceIntegrityError(
            "running trace lacks its original attached in-progress receipt"
        )
    return record


class InspirationRunRecovery:
    """Convert only legal retained phase traces into process interruption."""

    def __init__(
        self,
        *,
        database: Database,
        receipt_store: ReceiptStore,
        repository: InspirationRepository,
        response_codec: InspirationResponseCodec,
        clock: Clock,
    ) -> None:
        self.database = database
        self._receipt_store = receipt_store
        self._repository = repository
        self._codec = response_codec
        self._clock = clock

    async def recover(self) -> tuple[UUID, ...]:
        """Recover every legal running trace atomically, without generation."""
        async with self.database.transaction(immediate=True) as uow:
            traces = await self._repository.running_traces(session=uow.session)
            originals: dict[UUID, ReceiptRecord] = {}
            for trace in traces:
                originals[trace.run_id] = _validate_original_receipt(
                    trace,
                    await self._receipt_store.get_by_id(
                        session=uow.session,
                        receipt_id=trace.receipt_id,
                    ),
                )

            recovered: list[UUID] = []
            for trace in traces:
                decision = await self._receipt_store.reserve(
                    uow=uow,
                    request=_recovery_request(trace.run_id),
                )
                if isinstance(decision, CompletedReceipt):
                    raise InspirationSourceIntegrityError(
                        "completed recovery receipt still has a running trace"
                    )
                if isinstance(decision, InProgressReceipt):
                    raise InspirationSourceIntegrityError(
                        "recovery receipt is unexpectedly in progress"
                    )
                if not isinstance(decision, NewReceipt):
                    raise RuntimeError("unknown recovery receipt decision")
                reservation = decision.reservation
                await self._receipt_store.attach_resource(
                    uow=uow,
                    receipt_id=reservation.receipt_id,
                    resource_type="inspiration_run",
                    resource_id=trace.run_id,
                )
                occurred_at = max(
                    require_utc(reservation.created_at),
                    require_utc(trace.last_event_at),
                )
                payload = InspirationFailedV1(
                    schema_version=1,
                    run_id=trace.run_id,
                    status_before=InspirationRunStatus.RUNNING,
                    status_after=InspirationRunStatus.FAILED,
                    operator_outcomes=(),
                    output_tokens_reserved_before=0,
                    output_tokens_reserved_after=0,
                    output_tokens_consumed_before=0,
                    output_tokens_consumed_after=0,
                    elapsed_milliseconds_before=0,
                    elapsed_milliseconds_after=0,
                    failure_code=InspirationRunFailureCode.PROCESS_INTERRUPTED,
                )
                command = reservation.command_context()
                await uow.append_events(
                    command,
                    (
                        PendingEvent(
                            aggregate_type="inspiration_run",
                            aggregate_id=trace.run_id,
                            event_type=payload.event_type,
                            payload=payload,
                            actor_agent_id=None,
                            occurred_at=occurred_at,
                        ),
                    ),
                )
                run = await self._repository.get_run(
                    session=uow.session,
                    run_id=trace.run_id,
                )
                if run is None:
                    raise InspirationSourceIntegrityError(
                        "recovered run source is missing"
                    )
                response = self._codec.terminal(run)
                await self._receipt_store.complete_existing(
                    uow=uow,
                    receipt_id=originals[trace.run_id].receipt_id,
                    response=response,
                    completed_at=occurred_at,
                )
                await self._receipt_store.complete(
                    uow=uow,
                    reservation=reservation,
                    response=response,
                    completed_at=occurred_at,
                )
                recovered.append(trace.run_id)
            return tuple(recovered)


__all__ = ["InspirationRunRecovery"]
