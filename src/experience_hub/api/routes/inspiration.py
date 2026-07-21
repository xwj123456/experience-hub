"""Bounded inspiration runs, owner-scoped ideas, and explicit decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request
from starlette.responses import Response

from experience_hub.api.dependencies import (
    ContainerDependency,
    RequiredIdempotencyKey,
    command_result_response,
    reject_unknown_query_parameters,
)
from experience_hub.api.inflight import InFlightRunRegistry
from experience_hub.api.pagination import CursorCodec
from experience_hub.api.schemas.inspiration import (
    AdoptIdeaRequest,
    EvaluateIdeaRequest,
    IdeaListQuery,
    IdeaReasonRequest,
    StartInspirationRunRequest,
)
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.inspiration import InspirationOperator
from experience_hub.inspiration.request_hashing import (
    adoption_command_request,
    decision_command_request,
    evaluation_command_request,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import AgentRow
from experience_hub.storage.unit_of_work import UnitOfWork

_START_ROUTE = "/v1/agents/{agent_id}/inspiration-runs"
_IDEAS_ROUTE = "/v1/agents/{agent_id}/inspiration-runs/{run_id}/ideas"
type CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]

router = APIRouter(prefix="/v1", tags=["inspiration"])


def _not_found() -> DomainError:
    return DomainError(
        code="resource_not_found",
        message="The command resource was not found",
        status_code=404,
    )


def _inflight_registry(request: Request) -> InFlightRunRegistry:
    value = getattr(request.app.state, "inflight_runs", None)
    if not isinstance(value, InFlightRunRegistry):
        raise DomainError(
            code="service_not_ready",
            message="The service is not ready.",
            status_code=503,
        )
    return value


def _stored_response(value: StoredResponse, *, request: Request) -> Response:
    headers = dict(value.headers or {})
    if value.status_code >= 400:
        request_id = getattr(request.state, "request_id", None)
        if isinstance(request_id, UUID):
            headers["X-Request-ID"] = str(request_id)
    return Response(
        content=value.body,
        status_code=value.status_code,
        media_type=value.content_type,
        headers=headers,
    )


async def _existing_start_response(
    *,
    container: ApplicationContainer,
    foundation_request: CommandRequest,
) -> StoredResponse | None:
    async with container.database.read_session() as session:
        record = await container.receipt_store.find_for_request(
            session=session,
            request=foundation_request,
        )
    if record is None:
        return None
    if record.state == "completed":
        assert record.response is not None
        return record.response
    if (
        record.result_resource_type != "inspiration_run"
        or record.result_resource_id is None
    ):
        raise RuntimeError("in-progress inspiration receipt has no run attachment")
    return container.inspiration_response_codec.in_progress(
        receipt_id=record.receipt_id,
        run_id=record.result_resource_id,
    )


async def _execute(
    *,
    request: Request,
    container: ApplicationContainer,
    foundation_request: CommandRequest,
    handler: CommandHandler,
) -> Response:
    result = await container.command_executor.execute(
        foundation_request,
        handler,
    )
    return command_result_response(result, request=request)


@router.post("/agents/{agent_id}/inspiration-runs", status_code=201)
async def start_inspiration_run(
    agent_id: UUID,
    payload: StartInspirationRunRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    async with container.database.read_session() as session:
        if await session.get(AgentRow, agent_id) is None:
            raise _not_found()
    run = payload.to_command(owner_agent_id=agent_id)
    foundation_request = CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=_START_ROUTE,
        path_parameters={"agent_id": agent_id},
        body=payload.command_body(),
    )
    existing = await _existing_start_response(
        container=container,
        foundation_request=foundation_request,
    )
    if existing is not None:
        return _stored_response(existing, request=request)
    registry = _inflight_registry(request)
    stored = await registry.execute(
        lambda: container.inspiration_run_executor.execute(
            request=foundation_request,
            run=run,
        ),
        shutdown_timeout_seconds=run.global_timeout_seconds,
    )
    return _stored_response(stored, request=request)


@router.get("/agents/{agent_id}/inspiration-runs/{run_id}")
async def get_inspiration_run(
    agent_id: UUID,
    run_id: UUID,
    request: Request,
    container: ContainerDependency,
) -> Response:
    reject_unknown_query_parameters(request)
    async with container.database.read_session() as session:
        if not await container.inspiration_repository.owns_run(
            session=session,
            owner_agent_id=agent_id,
            run_id=run_id,
        ):
            raise _not_found()
        run = await container.inspiration_repository.get_run(
            session=session,
            run_id=run_id,
        )
        if run is None or run.owner_agent_id != agent_id:
            raise RuntimeError("owned inspiration run source changed during read")
    return Response(
        content=canonical_json_bytes({"data": run}),
        media_type="application/json",
    )


def _decode_idea_cursor(
    *,
    codec: CursorCodec,
    cursor: str | None,
) -> tuple[InspirationOperator, int, UUID] | None:
    if cursor is None:
        return None
    sort = codec.decode(cursor)
    if (
        len(sort) != 3
        or not isinstance(sort[0], str)
        or type(sort[1]) is not int
        or not isinstance(sort[2], str)
        or not 1 <= sort[1] <= 3
    ):
        raise DomainError(
            code="invalid_cursor",
            message="The cursor is invalid.",
            status_code=400,
        )
    try:
        operator = InspirationOperator(sort[0])
        idea_id = UUID(sort[2])
    except ValueError as error:
        raise DomainError(
            code="invalid_cursor",
            message="The cursor is invalid.",
            status_code=400,
        ) from error
    if str(idea_id) != sort[2]:
        raise DomainError(
            code="invalid_cursor",
            message="The cursor is invalid.",
            status_code=400,
        )
    return operator, sort[1], idea_id


@router.get("/agents/{agent_id}/inspiration-runs/{run_id}/ideas")
async def list_inspiration_ideas(
    agent_id: UUID,
    run_id: UUID,
    query: Annotated[IdeaListQuery, Query()],
    request: Request,
    container: ContainerDependency,
) -> Response:
    reject_unknown_query_parameters(
        request,
        allowed=frozenset({"cursor", "limit"}),
    )
    codec = CursorCodec(
        route=_IDEAS_ROUTE,
        context={
            "agent_id": agent_id,
            "run_id": run_id,
        },
    )
    after = _decode_idea_cursor(codec=codec, cursor=query.cursor)
    async with container.database.read_session() as session:
        if not await container.inspiration_repository.owns_run(
            session=session,
            owner_agent_id=agent_id,
            run_id=run_id,
        ):
            raise _not_found()
        run = await container.inspiration_repository.get_run(
            session=session,
            run_id=run_id,
        )
        if run is None or run.owner_agent_id != agent_id:
            raise RuntimeError("owned inspiration run source changed during read")
        values = await container.inspiration_repository.list_owned_ideas(
            session=session,
            owner_agent_id=agent_id,
            run_id=run_id,
            after=after,
            limit=query.limit + 1,
        )
    selected = values[: query.limit]
    next_cursor = (
        None
        if len(values) <= query.limit or not selected
        else codec.encode(
            (
                selected[-1].operator.value,
                selected[-1].ordinal,
                selected[-1].idea_id,
            )
        )
    )
    return Response(
        content=canonical_json_bytes(
            {
                "data": selected,
                "page": {"next_cursor": next_cursor},
            }
        ),
        media_type="application/json",
    )


@router.post("/agents/{agent_id}/ideas/{idea_id}:adopt", status_code=200)
async def adopt_idea(
    agent_id: UUID,
    idea_id: UUID,
    payload: AdoptIdeaRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = payload.to_command(
        owner_agent_id=agent_id,
        idea_id=idea_id,
    )
    foundation_request = adoption_command_request(
        command,
        idempotency_key=idempotency_key,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.idea_lifecycle_service.adopt(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await _execute(
        request=request,
        container=container,
        foundation_request=foundation_request,
        handler=handler,
    )


@router.post("/agents/{agent_id}/ideas/{idea_id}:reject", status_code=200)
async def reject_idea(
    agent_id: UUID,
    idea_id: UUID,
    payload: IdeaReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = payload.to_reject(
        owner_agent_id=agent_id,
        idea_id=idea_id,
    )
    foundation_request = decision_command_request(
        command,
        idempotency_key=idempotency_key,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.idea_lifecycle_service.reject(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await _execute(
        request=request,
        container=container,
        foundation_request=foundation_request,
        handler=handler,
    )


@router.post("/agents/{agent_id}/ideas/{idea_id}:evaluate", status_code=200)
async def evaluate_idea(
    agent_id: UUID,
    idea_id: UUID,
    payload: EvaluateIdeaRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    evaluation = payload.to_evaluation(
        evaluator_agent_id=agent_id,
        idea_id=idea_id,
    )
    foundation_request = evaluation_command_request(
        evaluation,
        idempotency_key=idempotency_key,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.idea_lifecycle_service.evaluate(
            uow=uow,
            evaluation=evaluation,
            command_context=context,
        )

    return await _execute(
        request=request,
        container=container,
        foundation_request=foundation_request,
        handler=handler,
    )


@router.post("/agents/{agent_id}/ideas/{idea_id}:archive", status_code=200)
async def archive_idea(
    agent_id: UUID,
    idea_id: UUID,
    payload: IdeaReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = payload.to_archive(
        owner_agent_id=agent_id,
        idea_id=idea_id,
    )
    foundation_request = decision_command_request(
        command,
        idempotency_key=idempotency_key,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.idea_lifecycle_service.archive(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await _execute(
        request=request,
        container=container,
        foundation_request=foundation_request,
        handler=handler,
    )


__all__ = ["router"]
