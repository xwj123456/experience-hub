"""Manual lifecycle command route."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import Response

from experience_hub.api.dependencies import (
    ContainerDependency,
    RequiredIdempotencyKey,
    command_result_response,
    reject_unknown_query_parameters,
)
from experience_hub.api.schemas.lifecycle import RunLifecycleRequest
from experience_hub.clock import require_utc
from experience_hub.domain.commands import (
    CommandContext,
    CommandRequest,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import IdempotencyRecordRow
from experience_hub.storage.unit_of_work import UnitOfWork

_RUN_ROUTE = "/v1/lifecycle:run"


router = APIRouter(prefix="/v1", tags=["lifecycle"])


@router.post("/lifecycle:run", status_code=200)
async def run_lifecycle(
    body: RunLifecycleRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    requested_evaluated_at = body.evaluated_at
    command_request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=_RUN_ROUTE,
        body={
            "evaluated_at": requested_evaluated_at,
            "mode": "manual",
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        evaluated_at = requested_evaluated_at
        if evaluated_at is None:
            receipt = await uow.session.get(
                IdempotencyRecordRow,
                context.receipt_id,
            )
            if receipt is None:
                raise RuntimeError("Lifecycle receipt disappeared after reservation")
            evaluated_at = require_utc(receipt.created_at)
        return await container.lifecycle_service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode="manual",
            evaluated_at_was_omitted=requested_evaluated_at is None,
        )

    result = await container.command_executor.execute(command_request, handler)
    return command_result_response(result, request=request)


__all__ = ["router"]
