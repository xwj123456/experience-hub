"""Shared FastAPI dependency accessors and transport-only command values."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Request
from starlette.responses import Response

from experience_hub.bootstrap import ApplicationContainer
from experience_hub.errors import DomainError
from experience_hub.storage.idempotency import CommandResult


def get_container(request: Request) -> ApplicationContainer:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, ApplicationContainer) or container.closed:
        raise DomainError(
            code="service_not_ready",
            message="The service is not ready.",
            status_code=503,
        )
    return container


ContainerDependency = Annotated[
    ApplicationContainer,
    Depends(get_container),
]


def _normalize_idempotency_key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Idempotency-Key must contain 1 to 128 nonblank characters")
    normalized = value.strip()
    if not 1 <= len(normalized) <= 128:
        raise ValueError("Idempotency-Key must contain 1 to 128 nonblank characters")
    return normalized


def _ambiguous_idempotency_key() -> DomainError:
    return DomainError(
        code="validation_error",
        message="Request validation failed.",
        details={
            "errors": [
                {
                    "location": ["header", "Idempotency-Key"],
                    "message": "Exactly one Idempotency-Key header is permitted",
                    "type": "value_error",
                }
            ]
        },
        status_code=422,
    )


def _required_idempotency_key(
    request: Request,
    value: Annotated[str, Header(alias="Idempotency-Key")],
) -> str:
    if len(request.headers.getlist("Idempotency-Key")) != 1:
        raise _ambiguous_idempotency_key()
    try:
        return _normalize_idempotency_key(value)
    except ValueError as error:
        raise _ambiguous_idempotency_key() from error


def _optional_idempotency_key(
    request: Request,
    value: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> str | None:
    values = request.headers.getlist("Idempotency-Key")
    if not values:
        return None
    if len(values) != 1:
        raise _ambiguous_idempotency_key()
    assert value is not None
    try:
        return _normalize_idempotency_key(value)
    except ValueError as error:
        raise _ambiguous_idempotency_key() from error


RequiredIdempotencyKey = Annotated[
    str,
    Depends(_required_idempotency_key),
]
OptionalIdempotencyKey = Annotated[
    str | None,
    Depends(_optional_idempotency_key),
]


def reject_unknown_query_parameters(
    request: Request,
    *,
    allowed: frozenset[str] = frozenset(),
) -> None:
    """Reject transport fields FastAPI would otherwise silently ignore."""
    unknown = sorted(set(request.query_params) - allowed)
    duplicate = sorted(
        name for name in allowed if len(request.query_params.getlist(name)) > 1
    )
    if not unknown and not duplicate:
        return
    name = unknown[0] if unknown else duplicate[0]
    is_duplicate = not unknown
    raise DomainError(
        code="validation_error",
        message="Request validation failed.",
        details={
            "errors": [
                {
                    "location": ["query", name],
                    "message": (
                        "Exactly one query value is permitted"
                        if is_duplicate
                        else "Extra inputs are not permitted"
                    ),
                    "type": ("value_error" if is_duplicate else "extra_forbidden"),
                }
            ]
        },
        status_code=422,
    )


def command_result_response(
    result: CommandResult,
    *,
    request: Request | None = None,
) -> Response:
    """Return durable status, bytes, and approved headers without re-encoding."""
    headers = dict(result.headers)
    if result.status_code >= 400 and request is not None:
        request_id = getattr(request.state, "request_id", None)
        if isinstance(request_id, UUID):
            headers["X-Request-ID"] = str(request_id)
    return Response(
        content=result.body,
        status_code=result.status_code,
        media_type=result.content_type,
        headers=headers,
    )


__all__ = [
    "ContainerDependency",
    "OptionalIdempotencyKey",
    "RequiredIdempotencyKey",
    "command_result_response",
    "get_container",
    "reject_unknown_query_parameters",
]
