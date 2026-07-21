"""Canonical, user-safe FastAPI error handling."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from experience_hub.api.schemas.common import ErrorBody, ErrorEnvelope
from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import CanonicalizationError, DomainError
from experience_hub.storage.database import DatabaseBusy

_LOGGER = logging.getLogger(__name__)
_REQUEST_ID_HEADER = "X-Request-ID"
type RequestIdFactory = Callable[[], UUID]


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, UUID):
        return str(value)
    generated = uuid4()
    request.state.request_id = generated
    return str(generated)


def _json_details(details: Mapping[str, Any]) -> dict[str, Any]:
    try:
        decoded = json.loads(canonical_json_bytes(details))
    except (CanonicalizationError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return decoded


def error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> Response:
    envelope = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            details=_json_details(details or {}),
        )
    )
    response_headers = dict(headers or {})
    response_headers[_REQUEST_ID_HEADER] = _request_id(request)
    return Response(
        content=canonical_json_bytes(envelope),
        status_code=status_code,
        media_type="application/json",
        headers=response_headers,
    )


def _validation_details(error: RequestValidationError) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    for issue in error.errors():
        location = issue.get("loc", ())
        retained.append(
            {
                "location": [
                    value if isinstance(value, (str, int)) else str(value)
                    for value in location
                ],
                "message": str(issue.get("msg", "Invalid input")),
                "type": str(issue.get("type", "value_error")),
            }
        )
    retained.sort(
        key=lambda item: (
            tuple(str(value) for value in item["location"]),
            item["type"],
            item["message"],
        )
    )
    return {"errors": retained}


def install_error_handlers(
    app: FastAPI,
    *,
    request_id_factory: RequestIdFactory | None = None,
) -> None:
    """Install request identity and deterministic exception mappings."""
    factory = request_id_factory or uuid4

    @app.middleware("http")
    async def assign_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = factory()
        return await call_next(request)

    @app.exception_handler(DatabaseBusy)
    async def database_busy_handler(
        request: Request,
        error: DatabaseBusy,
    ) -> Response:
        return error_response(
            request=request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
            headers={"Retry-After": str(error.retry_after)},
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(
        request: Request,
        error: DomainError,
    ) -> Response:
        return error_response(
            request=request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> Response:
        return error_response(
            request=request,
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details=_validation_details(error),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        error: StarletteHTTPException,
    ) -> Response:
        if error.status_code == 404:
            code, message = "route_not_found", "The requested route was not found."
        elif error.status_code == 405:
            code, message = (
                "method_not_allowed",
                "The request method is not allowed.",
            )
        else:
            code, message = "http_error", "The HTTP request could not be completed."
        return error_response(
            request=request,
            status_code=error.status_code,
            code=code,
            message=message,
            details={},
            headers=error.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ) -> Response:
        request_id = _request_id(request)
        _LOGGER.exception("Unhandled request error (request_id=%s)", request_id)
        return error_response(
            request=request,
            status_code=500,
            code="internal_error",
            message="An unexpected error occurred.",
            details={"request_id": request_id},
        )


__all__ = ["error_response", "install_error_handlers"]
