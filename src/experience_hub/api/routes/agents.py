"""Agent creation and stable read-only listing routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request
from sqlalchemy import and_, or_, select
from starlette.responses import Response

from experience_hub.agents import CreateAgent
from experience_hub.api.dependencies import (
    ContainerDependency,
    RequiredIdempotencyKey,
    command_result_response,
    reject_unknown_query_parameters,
)
from experience_hub.api.pagination import CursorCodec
from experience_hub.api.schemas.agents import (
    AgentListQuery,
    AgentResource,
    CreateAgentRequest,
)
from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain.commands import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import AgentRow
from experience_hub.storage.unit_of_work import UnitOfWork

_CREATE_ROUTE = "/v1/agents"
_LIST_CURSOR_ROUTE = "agents.list"


router = APIRouter(prefix="/v1", tags=["agents"])


def _invalid_cursor() -> DomainError:
    return DomainError(
        code="invalid_cursor",
        message="The cursor is invalid.",
        status_code=400,
    )


def _decode_agent_sort(
    codec: CursorCodec,
    cursor: str,
) -> tuple[datetime, UUID]:
    sort = codec.decode(cursor)
    if len(sort) != 2 or not isinstance(sort[0], str) or not isinstance(sort[1], str):
        raise _invalid_cursor()
    try:
        created_at = datetime.fromisoformat(sort[0].replace("Z", "+00:00"))
        if (
            created_at.tzinfo is None
            or created_at.utcoffset() is None
            or created_at.utcoffset() != UTC.utcoffset(created_at)
        ):
            raise ValueError("cursor timestamp is not UTC")
        created_at = created_at.astimezone(UTC)
        agent_id = UUID(sort[1])
        if codec.encode((created_at, agent_id)) != cursor:
            raise ValueError("cursor sort is not canonical")
    except (TypeError, ValueError) as error:
        raise _invalid_cursor() from error
    return created_at, agent_id


@router.post("/agents", status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command_request = CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=_CREATE_ROUTE,
        body={"name": body.name},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=body.name),
            command_context=context,
        )

    result = await container.command_executor.execute(command_request, handler)
    return command_result_response(result, request=request)


@router.get("/agents")
async def list_agents(
    query: Annotated[AgentListQuery, Query()],
    container: ContainerDependency,
) -> Response:
    codec = CursorCodec(route=_LIST_CURSOR_ROUTE)
    statement = select(AgentRow).order_by(
        AgentRow.created_at.asc(),
        AgentRow.agent_id.asc(),
    )
    if query.cursor is not None:
        created_at, agent_id = _decode_agent_sort(codec, query.cursor)
        statement = statement.where(
            or_(
                AgentRow.created_at > created_at,
                and_(
                    AgentRow.created_at == created_at,
                    AgentRow.agent_id > agent_id,
                ),
            )
        )
    statement = statement.limit(query.limit + 1)
    async with container.database.read_session() as session:
        rows = tuple((await session.scalars(statement)).all())

    page_rows = rows[: query.limit]
    next_cursor = (
        codec.encode((page_rows[-1].created_at, page_rows[-1].agent_id))
        if len(rows) > query.limit and page_rows
        else None
    )
    resources = tuple(
        AgentResource(agent_id=row.agent_id, name=row.name) for row in page_rows
    )
    return Response(
        content=canonical_json_bytes(
            {
                "data": resources,
                "page": {"next_cursor": next_cursor},
            }
        ),
        media_type="application/json",
    )


__all__ = ["router"]
