from typing import Annotated, Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Query

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import DomainError
from experience_hub.inspiration import InspirationErrorResponseV1
from experience_hub.storage import DatabaseBusy


def _assert_canonical_error(
    response: httpx.Response,
    expected: dict[str, Any],
) -> None:
    parsed = InspirationErrorResponseV1.model_validate(expected, strict=True)
    assert response.content == canonical_json_bytes(parsed)
    assert response.headers["content-type"] == "application/json"


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


def _app_with_failure_routes() -> FastAPI:
    app = create_app()

    @app.get("/_contract/domain")
    async def raise_domain_error() -> None:
        raise DomainError(
            code="experience_not_found",
            message="The experience was not found.",
            details={"experience_id": UUID("00000000-0000-0000-0000-000000000001")},
            status_code=404,
        )

    @app.get("/_contract/database-busy")
    async def raise_database_busy() -> None:
        raise DatabaseBusy

    @app.get("/_contract/validation")
    async def validate_limit(
        limit: Annotated[int, Query(ge=1, le=100)],
    ) -> dict[str, int]:
        return {"limit": limit}

    @app.get("/_contract/unexpected")
    async def raise_unexpected_error() -> None:
        raise RuntimeError("secret provider credential must never escape")

    return app


@pytest.mark.asyncio
async def test_domain_errors_use_the_canonical_shared_envelope() -> None:
    response = await _get(_app_with_failure_routes(), "/_contract/domain")

    assert response.status_code == 404
    _assert_canonical_error(
        response,
        {
            "error": {
                "code": "experience_not_found",
                "message": "The experience was not found.",
                "details": {"experience_id": ("00000000-0000-0000-0000-000000000001")},
            }
        },
    )
    UUID(response.headers["x-request-id"])


@pytest.mark.asyncio
async def test_request_validation_errors_are_stable_and_sanitized() -> None:
    response = await _get(
        _app_with_failure_routes(),
        "/_contract/validation?limit=0",
    )

    assert response.status_code == 422
    _assert_canonical_error(
        response,
        {
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": {
                    "errors": [
                        {
                            "location": ["query", "limit"],
                            "message": ("Input should be greater than or equal to 1"),
                            "type": "greater_than_equal",
                        }
                    ]
                },
            }
        },
    )
    UUID(response.headers["x-request-id"])


@pytest.mark.asyncio
async def test_database_busy_is_retryable_after_five_seconds() -> None:
    response = await _get(
        _app_with_failure_routes(),
        "/_contract/database-busy",
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    _assert_canonical_error(
        response,
        {
            "error": {
                "code": "database_busy",
                "message": "The database is busy; retry the request",
                "details": {},
            }
        },
    )
    UUID(response.headers["x-request-id"])


@pytest.mark.asyncio
async def test_unexpected_errors_expose_only_a_request_id() -> None:
    response = await _get(
        _app_with_failure_routes(),
        "/_contract/unexpected",
    )

    assert response.status_code == 500
    request_id = response.headers["x-request-id"]
    UUID(request_id)
    _assert_canonical_error(
        response,
        {
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "details": {"request_id": request_id},
            }
        },
    )
    assert b"secret" not in response.content
    assert b"credential" not in response.content
