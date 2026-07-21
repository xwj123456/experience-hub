"""Operational command-line adapter over the shared application runtime."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
import uvicorn
from sqlalchemy.engine import make_url

from experience_hub.api.app import create_app
from experience_hub.benchmark.runner import run_benchmark
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli.demo import build_demo_report
from experience_hub.clock import require_utc
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.experiences.reconcile import PayloadReconcileReport
from experience_hub.lifecycle import (
    decode_lifecycle_result,
    encode_lifecycle_result,
)
from experience_hub.runtime import ApplicationRuntime, SchemaRevisionError
from experience_hub.storage import DatabaseBusy
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.projections import (
    EventHeadChanged,
    MaintenanceBlockedByInflight,
    ProjectionDiff,
    ProjectionMismatch,
    ReducerVersionMismatch,
    SourceValidatorRequired,
    VerificationReport,
)
from experience_hub.storage.tables import IdempotencyRecordRow
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    PayloadReconcileValidationError,
    SourceIntegrityError,
)

app = typer.Typer(
    name="experience-hub",
    help="Experience Hub operations.",
    no_args_is_help=True,
    add_completion=False,
)
lifecycle_app = typer.Typer(
    help="Run memory lifecycle operations.",
    no_args_is_help=True,
    add_completion=False,
)
projections_app = typer.Typer(
    help="Verify or repair replayable projections.",
    no_args_is_help=True,
    add_completion=False,
)
payloads_app = typer.Typer(
    help="Maintain physical experience payloads.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(lifecycle_app, name="lifecycle")
app.add_typer(projections_app, name="projections")
app.add_typer(payloads_app, name="payloads")

_RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


def _settings(database: Path | None) -> Settings:
    if database is None:
        return Settings()
    database_path = str(database)
    candidate = f"sqlite+aiosqlite:///{database_path}"
    parsed = make_url(candidate)
    if parsed.database != database_path or parsed.query or database_path == ":memory:":
        raise typer.BadParameter(
            "database must be a file path without URL query syntax",
            param_hint="--database",
        )
    return Settings(database_url=candidate)


def _emit_bytes(body: bytes) -> None:
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CLI response body must be canonical JSON") from error
    if canonical_json_bytes(decoded) != body:
        raise ValueError("CLI response body must be canonical JSON")
    typer.echo(body.decode("utf-8"))


def _emit_document(document: Mapping[str, Any]) -> None:
    _emit_bytes(canonical_json_bytes(document))


def _error_document(
    *,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "details": dict(details or {}),
            "message": message,
        }
    }


def _projection_difference(value: ProjectionDiff) -> dict[str, Any]:
    return {
        "differing_keys": list(value.differing_keys),
        "online_hash": value.online_hash,
        "projection": value.projection,
        "rebuilt_hash": value.rebuilt_hash,
    }


def _projection_report(
    report: VerificationReport,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "data": {
            "differences": [
                _projection_difference(difference) for difference in report.differences
            ],
            "event_head": report.event_head,
            "matches": report.matches,
            "mode": mode,
        }
    }


def _projection_mismatch_error(
    report: VerificationReport,
) -> dict[str, Any]:
    return _error_document(
        code=ProjectionMismatch.code,
        message="Projection verification failed",
        details={
            "differences": [
                _projection_difference(difference) for difference in report.differences
            ],
            "event_head": report.event_head,
        },
    )


def _maintenance_error(error: BaseException) -> dict[str, Any]:
    if isinstance(error, DatabaseBusy):
        return _error_document(
            code=error.code,
            message=error.message,
            details={"retry_after": error.retry_after},
        )
    if isinstance(error, MaintenanceBlockedByInflight):
        return _error_document(
            code=error.code,
            message="Projection repair is blocked by in-progress work",
        )
    if isinstance(error, SourceIntegrityError):
        return _error_document(
            code=error.code,
            message="Authoritative source integrity validation failed",
            details={"mismatch_key": error.mismatch_key},
        )
    stable_errors: tuple[tuple[type[BaseException], str, str], ...] = (
        (
            ReducerVersionMismatch,
            ReducerVersionMismatch.code,
            "Stored projection reducer version is incompatible",
        ),
        (
            EventHeadChanged,
            EventHeadChanged.code,
            "Event head changed during projection maintenance",
        ),
        (
            SourceValidatorRequired,
            SourceValidatorRequired.code,
            "Projection source validation is unavailable",
        ),
        (
            SchemaRevisionError,
            SchemaRevisionError.code,
            "Database schema revision is not supported",
        ),
    )
    for error_type, code, message in stable_errors:
        if isinstance(error, error_type):
            return _error_document(code=code, message=message)
    if isinstance(error, DomainError):
        return _error_document(
            code=error.code,
            message=error.message,
            details=error.details,
        )
    return _error_document(
        code="internal_error",
        message="The operation failed unexpectedly",
    )


def _parse_evaluated_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    try:
        if _RFC3339_TIMESTAMP.fullmatch(normalized) is None or normalized.endswith(
            "-00:00"
        ):
            raise ValueError("timestamp is not in the supported RFC 3339 form")
        parsed = datetime.fromisoformat(
            normalized[:-1] + "+00:00"
            if normalized.endswith(("Z", "z"))
            else normalized
        )
        return require_utc(parsed)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(
            "evaluated-at must be an RFC 3339 timezone-aware timestamp",
            param_hint="--evaluated-at",
        ) from error


def _idempotency_key(value: str | None) -> str:
    if value is None:
        return f"lifecycle:manual:{uuid4()}"
    retained = value.strip()
    if not 1 <= len(retained) <= 128:
        raise typer.BadParameter(
            "idempotency-key must contain 1 to 128 nonblank characters",
            param_hint="--idempotency-key",
        )
    return retained


async def _run_lifecycle(
    *,
    settings: Settings,
    requested_evaluated_at: datetime | None,
    idempotency_key: str,
) -> tuple[int, bytes]:
    runtime = ApplicationRuntime(settings)
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        command_request = CommandRequest(
            caller_scope="system:local",
            operation_scope="lifecycle.run",
            idempotency_key=idempotency_key,
            method="POST",
            route_template="/v1/lifecycle:run",
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
                    raise RuntimeError(
                        "Lifecycle receipt disappeared after reservation"
                    )
                evaluated_at = require_utc(receipt.created_at)
            return await container.lifecycle_service.run(
                uow=uow,
                evaluated_at=evaluated_at,
                command=context,
                mode="manual",
                evaluated_at_was_omitted=requested_evaluated_at is None,
            )

        result = await container.command_executor.execute(
            command_request,
            handler,
        )
        if result.status_code < 200 or result.status_code >= 300:
            return 1, result.body
        decoded = decode_lifecycle_result(result.body)
        return 0, encode_lifecycle_result(decoded)


async def _run_projection_rebuild(
    *,
    settings: Settings,
    mode: str,
) -> tuple[int, dict[str, Any]]:
    runtime = ApplicationRuntime(settings)
    try:
        async with runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container:
            if mode == "verify":
                report = await container.projection_manager.verify(container.database)
            else:
                report = await container.projection_manager.repair(container.database)
    except ProjectionMismatch as error:
        return 1, _projection_mismatch_error(error.report)
    except Exception as error:
        return 1, _maintenance_error(error)
    return (0 if report.matches else 1), _projection_report(report, mode=mode)


def _payload_report(report: PayloadReconcileReport) -> dict[str, Any]:
    return {
        "data": {
            "changed_count": report.changed_count,
            "error_count": report.error_count,
            "errors": [
                {
                    "code": issue.code,
                    "experience_id": issue.experience_id,
                    "version_id": issue.version_id,
                    "version_number": issue.version_number,
                }
                for issue in report.errors
            ],
            "skipped_count": report.skipped_count,
        }
    }


async def _run_payload_reconcile(
    *,
    settings: Settings,
) -> tuple[int, dict[str, Any]]:
    runtime = ApplicationRuntime(settings)
    try:
        async with (
            runtime.initialize(
                start_lifecycle_worker=False,
                recover_interrupted=False,
            ) as container,
            container.database.transaction(immediate=True) as uow,
        ):
            report = await container.payload_reconciler.run(uow=uow)
    except PayloadReconcileValidationError as error:
        return 1, _payload_report(error.report)
    except Exception as error:
        return 1, _maintenance_error(error)
    return (0 if report.error_count == 0 else 1), _payload_report(report)


@app.callback()
def main() -> None:
    """Operate the local-first Experience Hub."""


@app.command("serve")
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Interface to bind."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65_535, help="TCP port to bind."),
    ] = 8_000,
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Serve the HTTP API; its lifespan owns startup and shutdown."""
    settings = _settings(database)

    def factory() -> Any:
        return create_app(settings=settings)

    uvicorn.run(
        factory,
        factory=True,
        host=host,
        port=port,
    )


@lifecycle_app.command("run")
def lifecycle_run(
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
    evaluated_at: Annotated[
        str | None,
        typer.Option("--evaluated-at", help="RFC 3339 evaluation timestamp."),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        typer.Option(
            "--idempotency-key",
            help="Optional stable key for exact command replay.",
        ),
    ] = None,
) -> None:
    """Run one manual lifecycle cycle through the ordinary command executor."""
    requested = _parse_evaluated_at(evaluated_at)
    key = _idempotency_key(idempotency_key)
    settings = _settings(database)
    try:
        exit_code, body = asyncio.run(
            _run_lifecycle(
                settings=settings,
                requested_evaluated_at=requested,
                idempotency_key=key,
            )
        )
    except Exception as error:
        exit_code = 1
        body = canonical_json_bytes(_maintenance_error(error))
    _emit_bytes(body)
    if exit_code:
        raise typer.Exit(exit_code)


@projections_app.command("rebuild")
def projections_rebuild(
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Compare projections without changing them."),
    ] = False,
    repair: Annotated[
        bool,
        typer.Option("--repair", help="Rebuild and atomically replace projections."),
    ] = False,
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Verify or repair every registered replayable projection."""
    if verify == repair:
        raise typer.BadParameter("Exactly one of --verify or --repair is required")
    mode = "verify" if verify else "repair"
    exit_code, document = asyncio.run(
        _run_projection_rebuild(
            settings=_settings(database),
            mode=mode,
        )
    )
    _emit_document(document)
    if exit_code:
        raise typer.Exit(exit_code)


@payloads_app.command("reconcile")
def payloads_reconcile(
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Restore preferred payload codecs without changing semantic hashes."""
    exit_code, document = asyncio.run(
        _run_payload_reconcile(settings=_settings(database))
    )
    _emit_document(document)
    if exit_code:
        raise typer.Exit(exit_code)


@app.command("demo")
def demo(
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset the isolated deterministic demo."),
    ] = False,
) -> None:
    """Run the deterministic local two-agent memory demonstration."""
    try:
        document = asyncio.run(build_demo_report(reset=reset))
    except Exception as error:
        _emit_document(_maintenance_error(error))
        raise typer.Exit(1) from error
    _emit_document(document)


@app.command("benchmark")
def benchmark() -> None:
    """Run the isolated deterministic offline effectiveness benchmark."""
    try:
        execution = asyncio.run(run_benchmark())
    except Exception as error:
        _emit_document(_maintenance_error(error))
        raise typer.Exit(1) from error
    _emit_bytes(execution.body)
    if not execution.passed:
        raise typer.Exit(1)


__all__ = ["ApplicationRuntime", "app"]
