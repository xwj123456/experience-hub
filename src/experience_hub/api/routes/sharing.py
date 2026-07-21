"""Social experience propagation HTTP routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
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
from experience_hub.api.schemas.sharing import (
    AdoptInboxItemRequest,
    CreateSubscriptionRequest,
    CreateTopicRequest,
    InboxListQuery,
    PublishCapsuleRequest,
    RecordFeedbackRequest,
    RequiredReasonRequest,
)
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.sharing import (
    AdoptCapsule,
    CreateSubscription,
    CreateTopic,
    PublishCapsule,
    RecordCapsuleFeedback,
    RejectInboxItem,
    RetractCapsule,
)
from experience_hub.sharing.queries import InvalidInboxCursor
from experience_hub.storage.idempotency import (
    ReservationPreflight,
    StoredResponse,
)
from experience_hub.storage.tables import AgentRow
from experience_hub.storage.unit_of_work import UnitOfWork

_TOPIC_ROUTE = "/v1/topics"
_SUBSCRIPTION_ROUTE = "/v1/agents/{agent_id}/subscriptions"
_PUBLISH_ROUTE = "/v1/agents/{agent_id}/capsules"
_RETRACT_ROUTE = "/v1/agents/{agent_id}/capsules/{capsule_id}:retract"
_ADOPT_ROUTE = "/v1/agents/{agent_id}/inbox/{item_id}:adopt"
_REJECT_ROUTE = "/v1/agents/{agent_id}/inbox/{item_id}:reject"
_FEEDBACK_ROUTE = "/v1/agents/{agent_id}/capsules/{capsule_id}:feedback"

type CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]

router = APIRouter(prefix="/v1", tags=["sharing"])


def _agent_request(
    *,
    agent_id: UUID,
    operation_scope: str,
    idempotency_key: str,
    route_template: str,
    path_parameters: dict[str, UUID],
    body: dict[str, object],
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope=operation_scope,
        idempotency_key=idempotency_key,
        method="POST",
        route_template=route_template,
        path_parameters=path_parameters,
        body=body,
    )


async def _execute(
    *,
    request: Request,
    container: ApplicationContainer,
    foundation_request: CommandRequest,
    handler: CommandHandler,
    reservation_preflight: ReservationPreflight | None = None,
) -> Response:
    result = await container.command_executor.execute(
        foundation_request,
        handler,
        reservation_preflight=reservation_preflight,
    )
    return command_result_response(result, request=request)


def _invalid_cursor() -> DomainError:
    return DomainError(
        code="invalid_cursor",
        message="The cursor is invalid.",
        status_code=400,
    )


@router.post("/topics", status_code=201)
async def create_topic(
    payload: CreateTopicRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = CreateTopic(
        owner_agent_id=payload.owner_agent_id,
        name=payload.name,
        description=payload.description,
    )
    foundation_request = CommandRequest(
        caller_scope=f"agent:{payload.owner_agent_id}",
        operation_scope="topic.create",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=_TOPIC_ROUTE,
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_topic(
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


@router.post("/agents/{agent_id}/subscriptions", status_code=201)
async def create_subscription(
    agent_id: UUID,
    payload: CreateSubscriptionRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = CreateSubscription(
        subscriber_agent_id=agent_id,
        topic_id=payload.topic_id,
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="subscription.create",
        idempotency_key=idempotency_key,
        route_template=_SUBSCRIPTION_ROUTE,
        path_parameters={"agent_id": agent_id},
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_subscription(
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


@router.post("/agents/{agent_id}/capsules", status_code=201)
async def publish_capsule(
    agent_id: UUID,
    payload: PublishCapsuleRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = PublishCapsule(
        owner_agent_id=agent_id,
        topic_id=payload.topic_id,
        experience_id=payload.experience_id,
        version_id=payload.version_id,
        expires_at=payload.expires_at,
        parent_adoption_id=payload.parent_adoption_id,
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="capsule.publish",
        idempotency_key=idempotency_key,
        route_template=_PUBLISH_ROUTE,
        path_parameters={"agent_id": agent_id},
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    async def validate_expiry(
        _: UnitOfWork,
        receipt_created_at: datetime,
    ) -> None:
        if payload.expires_at <= receipt_created_at:
            raise DomainError(
                code="invalid_expiry",
                message="Capsule expiry must be strictly in the future",
                status_code=422,
            )

    return await _execute(
        request=request,
        container=container,
        foundation_request=foundation_request,
        handler=handler,
        reservation_preflight=validate_expiry,
    )


@router.post(
    "/agents/{agent_id}/capsules/{capsule_id}:retract",
    status_code=200,
)
async def retract_capsule(
    agent_id: UUID,
    capsule_id: UUID,
    payload: RequiredReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = RetractCapsule(
        publisher_agent_id=agent_id,
        capsule_id=capsule_id,
        reason=payload.to_reason(),
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="capsule.retract",
        idempotency_key=idempotency_key,
        route_template=_RETRACT_ROUTE,
        path_parameters={
            "agent_id": agent_id,
            "capsule_id": capsule_id,
        },
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.retract_capsule(
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


@router.get("/agents/{agent_id}/inbox")
async def list_inbox(
    agent_id: UUID,
    query: Annotated[InboxListQuery, Query()],
    request: Request,
    container: ContainerDependency,
) -> Response:
    reject_unknown_query_parameters(
        request,
        allowed=frozenset({"cursor", "limit", "state"}),
    )
    async with container.database.read_session() as session:
        if await session.get(AgentRow, agent_id) is None:
            raise DomainError(
                code="agent_not_found",
                message="Agent was not found",
                status_code=404,
            )
        try:
            page = await container.sharing_query.list_inbox(
                session=session,
                owner_agent_id=agent_id,
                state=query.state,
                cursor=query.cursor,
                limit=query.limit,
                at=require_utc(container.clock.now()),
            )
        except InvalidInboxCursor as error:
            raise _invalid_cursor() from error
    return Response(
        content=canonical_json_bytes(
            {
                "data": page.items,
                "page": {"next_cursor": page.next_cursor},
            }
        ),
        media_type="application/json",
    )


@router.post("/agents/{agent_id}/inbox/{item_id}:adopt", status_code=200)
async def adopt_inbox_item(
    agent_id: UUID,
    item_id: UUID,
    payload: AdoptInboxItemRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = AdoptCapsule(
        adopter_agent_id=agent_id,
        item_id=item_id,
        importance=payload.importance,
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="capsule.adopt",
        idempotency_key=idempotency_key,
        route_template=_ADOPT_ROUTE,
        path_parameters={
            "agent_id": agent_id,
            "item_id": item_id,
        },
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.adopt_capsule(
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


@router.post("/agents/{agent_id}/inbox/{item_id}:reject", status_code=200)
async def reject_inbox_item(
    agent_id: UUID,
    item_id: UUID,
    payload: RequiredReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = RejectInboxItem(
        recipient_agent_id=agent_id,
        item_id=item_id,
        reason=payload.to_reason(),
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="capsule.reject",
        idempotency_key=idempotency_key,
        route_template=_REJECT_ROUTE,
        path_parameters={
            "agent_id": agent_id,
            "item_id": item_id,
        },
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.reject_inbox_item(
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


@router.post(
    "/agents/{agent_id}/capsules/{capsule_id}:feedback",
    status_code=201,
)
async def record_capsule_feedback(
    agent_id: UUID,
    capsule_id: UUID,
    payload: RecordFeedbackRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = RecordCapsuleFeedback(
        observer_agent_id=agent_id,
        capsule_id=capsule_id,
        verdict=payload.verdict,
        reason=payload.to_reason(),
        evidence=payload.evidence,
    )
    foundation_request = _agent_request(
        agent_id=agent_id,
        operation_scope="capsule.feedback",
        idempotency_key=idempotency_key,
        route_template=_FEEDBACK_ROUTE,
        path_parameters={
            "agent_id": agent_id,
            "capsule_id": capsule_id,
        },
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.record_capsule_feedback(
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
