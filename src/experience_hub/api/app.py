"""FastAPI application factory and shared runtime lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request
from starlette.responses import Response

from experience_hub.api.errors import install_error_handlers
from experience_hub.api.inflight import InFlightRunRegistry
from experience_hub.api.routes import api_router
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock
from experience_hub.config import Settings
from experience_hub.errors import DomainError
from experience_hub.ids import IdGenerator
from experience_hub.runtime import ApplicationRuntime

try:
    _APPLICATION_VERSION = version("experience-hub")
except PackageNotFoundError:  # pragma: no cover - source-only import fallback
    _APPLICATION_VERSION = "0.1.0"


def create_app(
    *,
    runtime: ApplicationRuntime | None = None,
    settings: Settings | None = None,
    clock: Clock | None = None,
    ids: IdGenerator | None = None,
    request_id_factory: Callable[[], UUID] | None = None,
) -> FastAPI:
    """Create an app whose readiness boundary is the shared runtime."""
    if runtime is not None and any(
        value is not None for value in (settings, clock, ids)
    ):
        raise ValueError("runtime cannot be combined with settings, clock, or ids")
    retained_runtime = runtime or ApplicationRuntime(
        settings=settings or Settings(),
        clock=clock,
        ids=ids,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.ready = False
        async with retained_runtime.initialize(
            start_lifecycle_worker=True,
            recover_interrupted=True,
        ) as container:
            inflight_runs = InFlightRunRegistry()
            container.register_shutdown_hook(inflight_runs.shutdown)
            application.state.container = container
            application.state.inflight_runs = inflight_runs
            application.state.ready = True
            try:
                yield
            finally:
                application.state.ready = False
                application.state.container = None
                application.state.inflight_runs = None

    application = FastAPI(
        title="Experience Hub",
        version=_APPLICATION_VERSION,
        lifespan=lifespan,
    )
    application.state.ready = False
    application.state.container = None
    application.state.inflight_runs = None
    application.state.runtime = retained_runtime
    install_error_handlers(
        application,
        request_id_factory=request_id_factory,
    )
    application.include_router(api_router)

    @application.get("/health")
    async def health(request: Request) -> Response:
        container = getattr(request.app.state, "container", None)
        if (
            not getattr(request.app.state, "ready", False)
            or not isinstance(container, ApplicationContainer)
            or container.closed
            or container.schema_revision is None
        ):
            raise DomainError(
                code="service_not_ready",
                message="The service is not ready.",
                status_code=503,
            )
        body: dict[str, Any] = {
            "data": {
                "status": "ready",
                "version": _APPLICATION_VERSION,
                "schema_revision": container.schema_revision,
                "reducer_versions": container.reducer_versions,
            }
        }
        return Response(
            content=canonical_json_bytes(body),
            media_type="application/json",
        )

    return application


app = create_app()

__all__ = ["app", "create_app"]
