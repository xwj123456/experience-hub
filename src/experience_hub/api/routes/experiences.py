"""Owner-scoped experience command and retrieval routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from experience_hub.api.dependencies import (
    ContainerDependency,
    OptionalIdempotencyKey,
    RequiredIdempotencyKey,
    command_result_response,
    reject_unknown_query_parameters,
)
from experience_hub.api.schemas.experiences import (
    CreateExperienceRequest,
    CreateExperienceVersionRequest,
    ExperienceEvidenceReasonRequest,
    ExperienceReasonRequest,
    SearchExperiencesRequest,
)
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    StructuredReason,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import decode_version_content
from experience_hub.experiences.contracts import (
    ConfirmExperience,
    CreateExperience,
    CreateExperienceVersion,
    PinExperience,
    RefuteExperience,
    RestoreExperience,
    UnpinExperience,
)
from experience_hub.experiences.queries import ExperienceNotFoundError
from experience_hub.retrieval.contracts import (
    ExperienceView,
    SearchExperiences,
)
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.tables import AgentRow, IdempotencyRecordRow
from experience_hub.storage.unit_of_work import UnitOfWork

router = APIRouter(prefix="/v1/agents/{agent_id}", tags=["experiences"])

type _StoredHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]


def _validation_error(
    *,
    location: tuple[str | int, ...],
    message: str,
    error_type: str,
) -> ReplayableCommandError:
    return ReplayableCommandError(
        code="validation_error",
        message="Request validation failed.",
        details={
            "errors": [
                {
                    "location": list(location),
                    "message": message,
                    "type": error_type,
                }
            ]
        },
        status_code=422,
    )


async def _require_no_body(request: Request) -> None:
    if await request.body():
        raise _validation_error(
            location=("body",),
            message="A request body is not permitted",
            error_type="body_not_permitted",
        )


async def _require_agent(session: AsyncSession, agent_id: UUID) -> None:
    if await session.get(AgentRow, agent_id) is None:
        raise ReplayableCommandError(
            code="agent_not_found",
            message="Agent was not found",
            status_code=404,
        )


def _reason(value: str | None) -> StructuredReason | None:
    return None if value is None else StructuredReason.from_user_text(value)


async def _full_experience_view(
    *,
    session: AsyncSession,
    container: ApplicationContainer,
    owner_agent_id: UUID,
    experience_id: UUID,
) -> ExperienceView:
    record = await container.experience_query.get_owned_retrieval_record(
        session=session,
        owner_agent_id=owner_agent_id,
        experience_id=experience_id,
    )
    if record is None:
        raise ExperienceNotFoundError
    decoded = await container.experience_query.load_decoded_payloads(
        session=session,
        owner_agent_id=owner_agent_id,
        version_ids=(record.current_version_id,),
    )
    content = decode_version_content(
        body_payload=decoded[record.current_version_id],
        summary=record.summary,
        mechanism=record.mechanism,
        tags=record.tags,
        applicability=record.applicability,
        evidence=record.evidence,
        falsifiers=record.falsifiers,
    )
    state = record.state
    return ExperienceView(
        experience_id=record.experience_id,
        owner_agent_id=record.owner_agent_id,
        kind=record.kind,
        origin=record.origin,
        created_at=record.created_at,
        version_id=record.current_version_id,
        version_number=record.current_version_number,
        version_created_at=record.current_version_created_at,
        content_hash=record.current_content_hash,
        temperature=state.temperature,
        importance=state.importance,
        confidence=state.confidence,
        activation_score=state.activation_score,
        source_trust=state.source_trust,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        last_transition_at=state.last_transition_at,
        last_lifecycle_evaluated_at=state.last_lifecycle_evaluated_at,
        consecutive_below_threshold=state.consecutive_below_threshold,
        pinned=state.pinned,
        summary=record.summary,
        mechanism=record.mechanism,
        tags=record.tags,
        applicability=record.applicability,
        evidence=record.evidence,
        falsifiers=record.falsifiers,
        blurred=False,
        body=content.body,
        body_is_excerpt=False,
    )


async def _creation_response(
    *,
    uow: UnitOfWork,
    context: CommandContext,
    container: ApplicationContainer,
    owner_agent_id: UUID,
    service_call: Awaitable[StoredResponse],
) -> StoredResponse:
    service_response = await service_call
    if service_response.status_code != 201:
        raise RuntimeError("Experience creation returned an invalid status")
    receipt = await uow.session.get(IdempotencyRecordRow, context.receipt_id)
    if (
        receipt is None
        or receipt.result_resource_type != "experience"
        or receipt.result_resource_id is None
    ):
        raise RuntimeError("Experience creation did not attach its resource")
    view = await _full_experience_view(
        session=uow.session,
        container=container,
        owner_agent_id=owner_agent_id,
        experience_id=receipt.result_resource_id,
    )
    return StoredResponse(
        status_code=201,
        body=canonical_json_bytes({"data": view}),
        headers={
            "location": (
                f"/v1/agents/{owner_agent_id}/experiences/{receipt.result_resource_id}"
            )
        },
    )


async def _execute(
    *,
    container: ApplicationContainer,
    request: CommandRequest,
    handler: _StoredHandler,
) -> CommandResult:
    return await container.command_executor.execute(request, handler)


@router.post("/experiences", status_code=201)
async def create_experience(
    agent_id: UUID,
    payload: CreateExperienceRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    content = payload.to_content()
    links = payload.to_links()
    command = CreateExperience(
        owner_agent_id=agent_id,
        kind=payload.kind,
        content=content,
        importance=payload.importance,
        confidence=payload.confidence,
        links=links,
    )
    foundation_request = CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope="experience.create",
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": agent_id},
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        await _require_agent(uow.session, agent_id)
        return await _creation_response(
            uow=uow,
            context=context,
            container=container,
            owner_agent_id=agent_id,
            service_call=container.experience_service.create(
                uow=uow,
                command=command,
                command_context=context,
            ),
        )

    result = await _execute(
        container=container,
        request=foundation_request,
        handler=handler,
    )
    return command_result_response(result, request=request)


@router.post("/experiences/{experience_id}/versions", status_code=201)
async def create_experience_version(
    agent_id: UUID,
    experience_id: UUID,
    payload: CreateExperienceVersionRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    command = CreateExperienceVersion(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        content=payload.to_content(),
        links=payload.to_links(),
    )
    foundation_request = CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope="experience.create_version",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=("/v1/agents/{agent_id}/experiences/{experience_id}/versions"),
        path_parameters={
            "agent_id": agent_id,
            "experience_id": experience_id,
        },
        body=payload.model_dump(mode="python"),
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await _creation_response(
            uow=uow,
            context=context,
            container=container,
            owner_agent_id=agent_id,
            service_call=container.experience_service.create_version(
                uow=uow,
                command=command,
                command_context=context,
            ),
        )

    result = await _execute(
        container=container,
        request=foundation_request,
        handler=handler,
    )
    return command_result_response(result, request=request)


@router.get("/experiences/{experience_id}")
async def get_experience(
    agent_id: UUID,
    experience_id: UUID,
    request: Request,
    container: ContainerDependency,
    idempotency_key: OptionalIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    await _require_no_body(request)
    result = await container.retrieval_adapter.get(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        idempotency_key=idempotency_key,
    )
    return command_result_response(result, request=request)


@router.post("/experiences:search")
async def search_experiences(
    agent_id: UUID,
    payload: SearchExperiencesRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: OptionalIdempotencyKey,
) -> Response:
    reject_unknown_query_parameters(request)
    query = SearchExperiences(
        owner_agent_id=agent_id,
        query=payload.query,
        mode=payload.mode,
        tags=payload.tags,
        mechanism_cues=payload.mechanism_cues,
        limit=payload.limit,
        content_budget_bytes=payload.content_budget_bytes,
        expand_cold=payload.expand_cold,
    )
    result = await container.retrieval_adapter.search(
        query=query,
        idempotency_key=idempotency_key,
    )
    return command_result_response(result, request=request)


def _mutation_request(
    *,
    agent_id: UUID,
    experience_id: UUID,
    suffix: str,
    idempotency_key: str,
    body: dict[str, Any],
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope=f"experience.{suffix}",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=("/v1/agents/{agent_id}/experiences/{experience_id}:" + suffix),
        path_parameters={
            "agent_id": agent_id,
            "experience_id": experience_id,
        },
        body=body,
    )


async def _execute_mutation(
    *,
    request: Request,
    container: ApplicationContainer,
    foundation_request: CommandRequest,
    service: Callable[..., Awaitable[StoredResponse]],
    command: object,
) -> Response:
    reject_unknown_query_parameters(request)

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await service(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await _execute(
        container=container,
        request=foundation_request,
        handler=handler,
    )
    return command_result_response(result, request=request)


@router.post("/experiences/{experience_id}:confirm")
async def confirm_experience(
    agent_id: UUID,
    experience_id: UUID,
    payload: ExperienceEvidenceReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    command = ConfirmExperience(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        reason=_reason(payload.reason),
        evidence=payload.evidence,
    )
    return await _execute_mutation(
        request=request,
        container=container,
        foundation_request=_mutation_request(
            agent_id=agent_id,
            experience_id=experience_id,
            suffix="confirm",
            idempotency_key=idempotency_key,
            body=payload.model_dump(mode="python"),
        ),
        service=container.experience_service.confirm,
        command=command,
    )


@router.post("/experiences/{experience_id}:refute")
async def refute_experience(
    agent_id: UUID,
    experience_id: UUID,
    payload: ExperienceEvidenceReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    command = RefuteExperience(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        reason=_reason(payload.reason),
        evidence=payload.evidence,
    )
    return await _execute_mutation(
        request=request,
        container=container,
        foundation_request=_mutation_request(
            agent_id=agent_id,
            experience_id=experience_id,
            suffix="refute",
            idempotency_key=idempotency_key,
            body=payload.model_dump(mode="python"),
        ),
        service=container.experience_service.refute,
        command=command,
    )


@router.post("/experiences/{experience_id}:pin")
async def pin_experience(
    agent_id: UUID,
    experience_id: UUID,
    payload: ExperienceReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    command = PinExperience(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        reason=_reason(payload.reason),
    )
    return await _execute_mutation(
        request=request,
        container=container,
        foundation_request=_mutation_request(
            agent_id=agent_id,
            experience_id=experience_id,
            suffix="pin",
            idempotency_key=idempotency_key,
            body=payload.model_dump(mode="python"),
        ),
        service=container.experience_service.pin,
        command=command,
    )


@router.post("/experiences/{experience_id}:unpin")
async def unpin_experience(
    agent_id: UUID,
    experience_id: UUID,
    payload: ExperienceReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    command = UnpinExperience(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        reason=_reason(payload.reason),
    )
    return await _execute_mutation(
        request=request,
        container=container,
        foundation_request=_mutation_request(
            agent_id=agent_id,
            experience_id=experience_id,
            suffix="unpin",
            idempotency_key=idempotency_key,
            body=payload.model_dump(mode="python"),
        ),
        service=container.experience_service.unpin,
        command=command,
    )


@router.post("/experiences/{experience_id}:restore")
async def restore_experience(
    agent_id: UUID,
    experience_id: UUID,
    payload: ExperienceReasonRequest,
    request: Request,
    container: ContainerDependency,
    idempotency_key: RequiredIdempotencyKey,
) -> Response:
    command = RestoreExperience(
        owner_agent_id=agent_id,
        experience_id=experience_id,
        reason=_reason(payload.reason),
    )
    return await _execute_mutation(
        request=request,
        container=container,
        foundation_request=_mutation_request(
            agent_id=agent_id,
            experience_id=experience_id,
            suffix="restore",
            idempotency_key=idempotency_key,
            body=payload.model_dump(mode="python"),
        ),
        service=container.experience_service.restore,
        command=command,
    )


__all__ = ["router"]
