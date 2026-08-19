"""CLI-first trajectory capture and candidate quarantine commands."""

from __future__ import annotations

import asyncio
import stat
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import UUID

import typer

import experience_hub.runtime as runtime_module
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.capture.extraction import DeterministicSignalExtractor
from experience_hub.capture.jsonl import MAX_JSONL_INPUT_BYTES, GenericJsonlAdapter
from experience_hub.capture.models import PreparedCaptureV1
from experience_hub.capture.sanitization import DefaultSecretScanner
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.capture.service import CapturePreparer
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, StructuredReason
from experience_hub.errors import DomainError
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
    AdoptCandidate,
    CandidateDecision,
    RejectCandidate,
)
from experience_hub.experiences.candidate_service import (
    CandidatePageV1,
    CandidateViewV1,
)
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork

capture_app = typer.Typer(
    help="Inspect or import sanitized agent trajectories.",
    no_args_is_help=True,
    add_completion=False,
)
candidate_app = typer.Typer(
    help="Inspect and decide quarantined experience candidates.",
    no_args_is_help=True,
    add_completion=False,
)

_CAPTURE_PREPARER = CapturePreparer(
    adapter=GenericJsonlAdapter(),
    scanner=DefaultSecretScanner(),
    extractor=DeterministicSignalExtractor(),
)


def _input_error(
    *,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> DomainError:
    return DomainError(
        code=code,
        message=message,
        details=details,
        status_code=400,
    )


def _read_capture_input(path: Path) -> bytes:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        raise _input_error(
            code="capture_input_not_file",
            message="Capture input must be a regular file",
        ) from None
    except OSError:
        raise _input_error(
            code="capture_input_read_failed",
            message="Capture input could not be read",
        ) from None
    if not stat.S_ISREG(metadata.st_mode):
        raise _input_error(
            code="capture_input_not_file",
            message="Capture input must be a regular file",
        )
    if metadata.st_size > MAX_JSONL_INPUT_BYTES:
        raise _input_error(
            code="capture_input_too_large",
            message="Capture input exceeds the size limit",
            details={"max_bytes": MAX_JSONL_INPUT_BYTES},
        )
    try:
        with path.open("rb") as source:
            data = source.read(MAX_JSONL_INPUT_BYTES + 1)
    except OSError:
        raise _input_error(
            code="capture_input_read_failed",
            message="Capture input could not be read",
        ) from None
    if len(data) > MAX_JSONL_INPUT_BYTES:
        raise _input_error(
            code="capture_input_too_large",
            message="Capture input exceeds the size limit",
            details={"max_bytes": MAX_JSONL_INPUT_BYTES},
        )
    return data


def _prepare_capture(path: Path) -> PreparedCaptureV1:
    data = _read_capture_input(path)
    try:
        return _CAPTURE_PREPARER.prepare_jsonl(data)
    except DomainError:
        raise
    except (TypeError, ValueError):
        raise _input_error(
            code="capture_input_invalid",
            message="Capture input is invalid",
        ) from None


def _settings(database: Path | None) -> Settings:
    from experience_hub.cli.app import _settings as shared_settings

    return shared_settings(database)


def _required_idempotency_key(value: str) -> str:
    retained = value.strip()
    if not 1 <= len(retained) <= 128:
        raise typer.BadParameter(
            "idempotency-key must contain 1 to 128 nonblank characters",
            param_hint="--idempotency-key",
        )
    return retained


def _structured_reason(value: str) -> StructuredReason:
    try:
        return StructuredReason.from_user_text(value)
    except ValueError:
        raise typer.BadParameter(
            "reason must contain 1 to 2,000 characters after trimming",
            param_hint="--reason",
        ) from None


def _candidate_score(value: float, option: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise typer.BadParameter(
            "candidate scores must be finite values from 0 to 1",
            param_hint=option,
        )
    return value


def _exit_with_error(error: BaseException) -> NoReturn:
    from experience_hub.cli.app import _emit_document, _maintenance_error

    _emit_document(_maintenance_error(error))
    raise typer.Exit(1) from error


def _emit_result(result: CommandResult) -> None:
    from experience_hub.cli.app import _emit_bytes

    _emit_bytes(result.body)
    if not 200 <= result.status_code < 300:
        raise typer.Exit(1)


@asynccontextmanager
async def _application_container(
    settings: Settings,
) -> AsyncIterator[ApplicationContainer]:
    runtime = runtime_module.ApplicationRuntime(settings)
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        yield container


async def _run_capture_import(
    *,
    settings: Settings,
    prepared: PreparedCaptureV1,
    idempotency_key: str,
) -> CommandResult:
    async with _application_container(settings) as container:
        owner_agent_id = prepared.bundle.owner_agent_id
        request = CommandRequest(
            caller_scope=f"agent:{owner_agent_id}",
            operation_scope=TRAJECTORY_IMPORT_SCOPE,
            idempotency_key=idempotency_key,
            method="POST",
            route_template="/v1/agents/{agent_id}/trajectory-bundles",
            path_parameters={"agent_id": owner_agent_id},
            body={
                "adapter": prepared.bundle.adapter.kind,
                "candidate_count": len(prepared.candidates),
                "manifest_hash": prepared.bundle.manifest_hash,
            },
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            return await container.capture_service.capture(
                uow=uow,
                prepared=prepared,
                command=command,
            )

        return await container.command_executor.execute(request, handler)


async def _list_candidates(
    *,
    settings: Settings,
    owner_agent_id: UUID,
    decision: CandidateDecision | None,
    limit: int,
    cursor: str | None,
) -> CandidatePageV1:
    async with (
        _application_container(settings) as container,
        container.database.read_session() as session,
    ):
        return await container.candidate_service.list_owned(
            session=session,
            owner_agent_id=owner_agent_id,
            decision=decision,
            limit=limit,
            cursor=cursor,
        )


async def _show_candidate(
    *,
    settings: Settings,
    owner_agent_id: UUID,
    candidate_id: UUID,
) -> CandidateViewV1:
    async with (
        _application_container(settings) as container,
        container.database.read_session() as session,
    ):
        return await container.candidate_service.get_owned(
            session=session,
            owner_agent_id=owner_agent_id,
            candidate_id=candidate_id,
        )


async def _run_candidate_adoption(
    *,
    settings: Settings,
    request: AdoptCandidate,
    idempotency_key: str,
) -> CommandResult:
    async with _application_container(settings) as container:
        command_request = CommandRequest(
            caller_scope=f"agent:{request.owner_agent_id}",
            operation_scope=CANDIDATE_ADOPT_SCOPE,
            idempotency_key=idempotency_key,
            method="POST",
            route_template=(
                "/v1/agents/{agent_id}/candidates/{candidate_id}:adopt"
            ),
            path_parameters={
                "agent_id": request.owner_agent_id,
                "candidate_id": request.candidate_id,
            },
            body={
                "confidence": request.confidence,
                "importance": request.importance,
            },
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            return await container.candidate_service.adopt(
                uow=uow,
                request=request,
                command=command,
            )

        return await container.command_executor.execute(command_request, handler)


async def _run_candidate_rejection(
    *,
    settings: Settings,
    request: RejectCandidate,
    idempotency_key: str,
) -> CommandResult:
    async with _application_container(settings) as container:
        command_request = CommandRequest(
            caller_scope=f"agent:{request.owner_agent_id}",
            operation_scope=CANDIDATE_REJECT_SCOPE,
            idempotency_key=idempotency_key,
            method="POST",
            route_template=(
                "/v1/agents/{agent_id}/candidates/{candidate_id}:reject"
            ),
            path_parameters={
                "agent_id": request.owner_agent_id,
                "candidate_id": request.candidate_id,
            },
            body={"reason": request.reason.model_dump(mode="json")},
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            return await container.candidate_service.reject(
                uow=uow,
                request=request,
                command=command,
            )

        return await container.command_executor.execute(command_request, handler)


@capture_app.command("inspect")
def capture_inspect(
    path: Annotated[
        Path,
        typer.Argument(help="Canonical sanitized trajectory JSONL file."),
    ],
) -> None:
    """Inspect sanitized trajectory input without opening the database."""
    from experience_hub.cli.app import _emit_document, _maintenance_error

    try:
        prepared = _prepare_capture(path)
    except Exception as error:
        _emit_document(_maintenance_error(error))
        raise typer.Exit(1) from error
    bundle = prepared.bundle
    _emit_document(
        {
            "data": {
                "adapter": bundle.adapter.kind,
                "candidate_count": len(prepared.candidates),
                "manifest_hash": bundle.manifest_hash,
                "owner_agent_id": bundle.owner_agent_id,
                "persisted": False,
                "step_count": len(bundle.steps),
                "trajectory_id": bundle.trajectory_id,
            }
        }
    )


@capture_app.command("import")
def capture_import(
    path: Annotated[
        Path,
        typer.Argument(help="Canonical sanitized trajectory JSONL file."),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option(
            "--idempotency-key",
            help="Stable key required for exact command replay.",
        ),
    ],
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Import one sanitized trajectory into quarantine."""
    key = _required_idempotency_key(idempotency_key)
    settings = _settings(database)
    try:
        prepared = _prepare_capture(path)
        result = asyncio.run(
            _run_capture_import(
                settings=settings,
                prepared=prepared,
                idempotency_key=key,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_result(result)


@candidate_app.command("list")
def candidates_list(
    owner_agent_id: Annotated[
        UUID,
        typer.Argument(help="Owner agent UUID."),
    ],
    decision: Annotated[
        CandidateDecision | None,
        typer.Option("--decision", help="Filter by candidate decision."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Page size."),
    ] = 50,
    cursor: Annotated[
        str | None,
        typer.Option("--cursor", help="Opaque page cursor."),
    ] = None,
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """List owner-scoped quarantined candidates."""
    from experience_hub.cli.app import _emit_document

    settings = _settings(database)
    try:
        page = asyncio.run(
            _list_candidates(
                settings=settings,
                owner_agent_id=owner_agent_id,
                decision=decision,
                limit=limit,
                cursor=cursor,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_document({"data": page.model_dump(mode="json")})


@candidate_app.command("show")
def candidates_show(
    owner_agent_id: Annotated[
        UUID,
        typer.Argument(help="Owner agent UUID."),
    ],
    candidate_id: Annotated[
        UUID,
        typer.Argument(help="Candidate UUID."),
    ],
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Show one owner-scoped quarantined candidate."""
    from experience_hub.cli.app import _emit_document

    settings = _settings(database)
    try:
        candidate = asyncio.run(
            _show_candidate(
                settings=settings,
                owner_agent_id=owner_agent_id,
                candidate_id=candidate_id,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_document({"data": candidate.model_dump(mode="json")})


@candidate_app.command("adopt")
def candidates_adopt(
    owner_agent_id: Annotated[
        UUID,
        typer.Argument(help="Owner agent UUID."),
    ],
    candidate_id: Annotated[
        UUID,
        typer.Argument(help="Candidate UUID."),
    ],
    importance: Annotated[
        float,
        typer.Option(
            "--importance",
            min=0.0,
            max=1.0,
            help="Adopted experience importance from 0 to 1.",
        ),
    ],
    confidence: Annotated[
        float,
        typer.Option(
            "--confidence",
            min=0.0,
            max=1.0,
            help="Adopted experience confidence from 0 to 1.",
        ),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option(
            "--idempotency-key",
            help="Stable key required for exact command replay.",
        ),
    ],
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Adopt one pending candidate into ordinary experience storage."""
    key = _required_idempotency_key(idempotency_key)
    settings = _settings(database)
    request = AdoptCandidate(
        owner_agent_id=owner_agent_id,
        candidate_id=candidate_id,
        importance=_candidate_score(importance, "--importance"),
        confidence=_candidate_score(confidence, "--confidence"),
    )
    try:
        result = asyncio.run(
            _run_candidate_adoption(
                settings=settings,
                request=request,
                idempotency_key=key,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_result(result)


@candidate_app.command("reject")
def candidates_reject(
    owner_agent_id: Annotated[
        UUID,
        typer.Argument(help="Owner agent UUID."),
    ],
    candidate_id: Annotated[
        UUID,
        typer.Argument(help="Candidate UUID."),
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Human-readable rejection reason."),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option(
            "--idempotency-key",
            help="Stable key required for exact command replay.",
        ),
    ],
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite database path."),
    ] = None,
) -> None:
    """Reject one pending candidate with a structured reason."""
    key = _required_idempotency_key(idempotency_key)
    settings = _settings(database)
    request = RejectCandidate(
        owner_agent_id=owner_agent_id,
        candidate_id=candidate_id,
        reason=_structured_reason(reason),
    )
    try:
        result = asyncio.run(
            _run_candidate_rejection(
                settings=settings,
                request=request,
                idempotency_key=key,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_result(result)


__all__ = ["candidate_app", "capture_app"]
