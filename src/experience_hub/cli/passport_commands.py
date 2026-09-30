"""Offline Passport commands with fixed, canonical responses."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from math import isfinite
from pathlib import Path
from typing import Annotated, NoReturn
from uuid import UUID

import typer
from sqlalchemy.engine import URL
from typer._click.core import Context as ClickContext
from typer.core import TyperCommand, TyperGroup

import experience_hub.runtime as runtime_module
from experience_hub import canonical_json_bytes
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, StructuredReason
from experience_hub.errors import DomainError
from experience_hub.passports.contracts import (
    ImportPassport,
    PassportDeclarationV1,
    PassportState,
)
from experience_hub.passports.errors import PassportError
from experience_hub.passports.export import PassportExportService
from experience_hub.passports.files import publish_passport_file, read_passport_file
from experience_hub.passports.requests import passport_import_request
from experience_hub.runtime import SchemaRevisionError
from experience_hub.storage import (
    CommandResult,
    DatabaseBusy,
    StoredResponse,
    UnitOfWork,
)
from experience_hub.storage.readonly import readonly_sqlite_session
from experience_hub.storage.validation import SourceIntegrityError


def _emit(document: Mapping[str, object]) -> None:
    typer.echo(canonical_json_bytes(document).decode("utf-8"))


def _fail(error: Exception) -> NoReturn:
    if isinstance(error, DomainError):
        code, message, details = error.code, error.message, error.details
    elif isinstance(error, DatabaseBusy):
        code, message, details = error.code, error.message, {}
    elif isinstance(error, SchemaRevisionError):
        code, message, details = (
            error.code,
            "Database schema revision is not supported",
            {},
        )
    elif isinstance(error, SourceIntegrityError):
        code, message, details = (
            error.code,
            "Authoritative source integrity validation failed",
            {},
        )
    else:
        code, message, details = (
            "internal_error",
            "The operation failed unexpectedly",
            {},
        )
    _emit({"error": {"code": code, "message": message, "details": details}})
    raise typer.Exit(1) from None


class _PassportCommand(TyperCommand):
    # Match Typer's vendored Click base, not its narrower public Context subclass.
    def parse_args(self, ctx: ClickContext, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except typer.Exit:
            raise
        except Exception:
            _fail(PassportError("invalid"))


class _PassportGroup(TyperGroup):
    def invoke(self, ctx: ClickContext) -> None:
        try:
            super().invoke(ctx)
        except typer.Exit:
            raise
        except Exception:
            _fail(PassportError("invalid"))


passport_app = typer.Typer(
    cls=_PassportGroup,
    help="Transfer one experience through local quarantine.",
    no_args_is_help=True,
    add_completion=False,
)


def _required(value: str | None) -> str:
    if value is None or not value.strip():
        raise PassportError("invalid")
    return value


def _uuid(value: str | None) -> UUID:
    try:
        return UUID(_required(value))
    except (ValueError, TypeError):
        raise PassportError("invalid") from None


def _database(value: Path | None) -> Settings:
    if value is None or str(value) == ":memory:":
        raise PassportError("invalid")
    return Settings(database_url=URL.create("sqlite+aiosqlite", database=str(value)))


def _key(value: str | None) -> str:
    retained = _required(value).strip()
    if not 1 <= len(retained) <= 128:
        raise PassportError("invalid")
    return retained


def _score(value: str | None) -> float:
    try:
        score = float(_required(value))
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError
        return score
    except (ValueError, TypeError):
        raise PassportError("invalid") from None


def _run[T](operation: Coroutine[object, object, T]) -> T:
    try:
        return asyncio.run(operation)
    except Exception as error:
        _fail(error)


def _result(result: CommandResult) -> None:
    typer.echo(result.body.decode("utf-8"))
    if not 200 <= result.status_code < 300:
        raise typer.Exit(1)


@passport_app.command("inspect", cls=_PassportCommand)
def inspect_passport(path: Annotated[Path | None, typer.Argument()] = None) -> None:
    try:
        if path is None:
            raise PassportError("invalid")
        prepared = read_passport_file(path)
    except Exception as error:
        _fail(error)
    _emit({"data": prepared.report.model_dump(mode="json", warnings="error")})


async def _export(
    database: Path,
    owner: UUID,
    experience: UUID,
    version: UUID | None,
    declaration: PassportDeclarationV1,
    parent: UUID | None,
) -> bytes:
    async with readonly_sqlite_session(database) as session:
        prepared = await PassportExportService().export(
            session=session,
            owner_agent_id=owner,
            experience_id=experience,
            version_id=version,
            declaration=declaration,
            parent_adoption_id=parent,
        )
    return prepared.canonical_bytes


@passport_app.command("export", cls=_PassportCommand)
def export_passport(
    owner: Annotated[str | None, typer.Argument()] = None,
    experience: Annotated[str | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
    input_sanitized: Annotated[bool, typer.Option("--input-sanitized")] = False,
    sanitization_profile: Annotated[
        str | None, typer.Option("--sanitization-profile")
    ] = None,
    sharing_authorized: Annotated[bool, typer.Option("--sharing-authorized")] = False,
    parent_adoption: Annotated[str | None, typer.Option("--parent-adoption")] = None,
) -> None:
    try:
        _database(database)
        if (
            database is None
            or output is None
            or not input_sanitized
            or not sharing_authorized
        ):
            raise PassportError("invalid")
        declaration = PassportDeclarationV1(
            input_sanitized=True,
            profile_id=_required(sanitization_profile),
            sharing_authorized=True,
        )
        owner_id, experience_id = _uuid(owner), _uuid(experience)
        version_id = None if version is None else _uuid(version)
        parent_id = None if parent_adoption is None else _uuid(parent_adoption)
    except Exception:
        _fail(PassportError("invalid"))
    body = _run(
        _export(database, owner_id, experience_id, version_id, declaration, parent_id)
    )
    try:
        publish_passport_file(output, body)
        from experience_hub.passports.codec import verify_passport_bytes

        prepared = verify_passport_bytes(body)
    except Exception as error:
        _fail(error)
    _emit({"data": prepared.report.model_dump(mode="json", warnings="error")})


async def _import(
    settings: Settings, request: ImportPassport, key: str
) -> CommandResult:
    async with runtime_module.ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:

        async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
            return await container.passport_service.import_passport(
                uow=uow, request=request, command=command
            )

        return await container.command_executor.execute(
            passport_import_request(
                owner_agent_id=request.owner_agent_id,
                passport_hash=request.prepared.document.passport_hash,
                idempotency_key=key,
            ),
            handler,
        )


@passport_app.command("import", cls=_PassportCommand)
def import_passport(
    owner: Annotated[str | None, typer.Argument()] = None,
    path: Annotated[Path | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    idempotency_key: Annotated[str | None, typer.Option("--idempotency-key")] = None,
) -> None:
    try:
        settings, owner_id, key = (
            _database(database),
            _uuid(owner),
            _key(idempotency_key),
        )
        if path is None:
            raise PassportError("invalid")
        prepared = read_passport_file(path)
        request = ImportPassport(owner_agent_id=owner_id, prepared=prepared)
    except Exception as error:
        _fail(error)
    _result(_run(_import(settings, request, key)))


async def _show(
    settings: Settings, owner: UUID, import_id: UUID
) -> Mapping[str, object]:
    async with (
        runtime_module.ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False, recover_interrupted=False
        ) as container,
        container.database.read_session() as session,
    ):
        view = await container.passport_query.get_owned(
            session=session, owner_agent_id=owner, import_id=import_id
        )
        return view.model_dump(mode="json", warnings="error")


@passport_app.command("show", cls=_PassportCommand)
def show_passport(
    owner: Annotated[str | None, typer.Argument()] = None,
    import_id: Annotated[str | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
) -> None:
    try:
        settings, owner_id, item_id = (
            _database(database),
            _uuid(owner),
            _uuid(import_id),
        )
    except Exception as error:
        _fail(error)
    _emit({"data": _run(_show(settings, owner_id, item_id))})


async def _list(
    settings: Settings,
    owner: UUID,
    state: PassportState | None,
    limit: int,
    cursor: str | None,
) -> Mapping[str, object]:
    async with (
        runtime_module.ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False, recover_interrupted=False
        ) as container,
        container.database.read_session() as session,
    ):
        page = await container.passport_query.list_owned(
            session=session,
            owner_agent_id=owner,
            state=state,
            limit=limit,
            cursor=cursor,
        )
        return page.model_dump(mode="json", warnings="error")


@passport_app.command("list", cls=_PassportCommand)
def list_passports(
    owner: Annotated[str | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    state: Annotated[str | None, typer.Option("--state")] = None,
    limit: Annotated[str, typer.Option("--limit")] = "50",
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    try:
        settings, owner_id = _database(database), _uuid(owner)
        retained_state = None if state is None else PassportState(state)
        size = int(limit)
        if not 1 <= size <= 100 or (cursor is not None and len(cursor) > 8192):
            raise ValueError
    except Exception:
        _fail(PassportError("invalid"))
    _emit({"data": _run(_list(settings, owner_id, retained_state, size, cursor))})


async def _adopt(
    settings: Settings,
    owner: UUID,
    import_id: UUID,
    importance: float,
    confidence: float,
    key: str,
) -> CommandResult:
    from experience_hub.passports.contracts import AdoptPassport
    from experience_hub.passports.requests import passport_adopt_request

    request = AdoptPassport(
        owner_agent_id=owner,
        import_id=import_id,
        importance=importance,
        confidence=confidence,
    )
    async with runtime_module.ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:

        async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
            return await container.passport_service.adopt(
                uow=uow, request=request, command=command
            )

        return await container.command_executor.execute(
            passport_adopt_request(
                owner_agent_id=owner,
                import_id=import_id,
                importance=importance,
                confidence=confidence,
                idempotency_key=key,
            ),
            handler,
        )


@passport_app.command("adopt", cls=_PassportCommand)
def adopt_passport(
    owner: Annotated[str | None, typer.Argument()] = None,
    import_id: Annotated[str | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    importance: Annotated[str | None, typer.Option("--importance")] = None,
    confidence: Annotated[str | None, typer.Option("--confidence")] = None,
    idempotency_key: Annotated[str | None, typer.Option("--idempotency-key")] = None,
) -> None:
    try:
        settings, owner_id, item_id = (
            _database(database),
            _uuid(owner),
            _uuid(import_id),
        )
        retained_importance, retained_confidence, key = (
            _score(importance),
            _score(confidence),
            _key(idempotency_key),
        )
    except Exception as error:
        _fail(error)
    _result(
        _run(
            _adopt(
                settings,
                owner_id,
                item_id,
                retained_importance,
                retained_confidence,
                key,
            )
        )
    )


async def _reject(
    settings: Settings,
    owner: UUID,
    import_id: UUID,
    reason: StructuredReason,
    key: str,
) -> CommandResult:
    from experience_hub.passports.contracts import RejectPassport
    from experience_hub.passports.requests import passport_reject_request

    request = RejectPassport(owner_agent_id=owner, import_id=import_id, reason=reason)
    async with runtime_module.ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:

        async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
            return await container.passport_service.reject(
                uow=uow, request=request, command=command
            )

        return await container.command_executor.execute(
            passport_reject_request(
                owner_agent_id=owner,
                import_id=import_id,
                reason=reason,
                idempotency_key=key,
            ),
            handler,
        )


@passport_app.command("reject", cls=_PassportCommand)
def reject_passport(
    owner: Annotated[str | None, typer.Argument()] = None,
    import_id: Annotated[str | None, typer.Argument()] = None,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    idempotency_key: Annotated[str | None, typer.Option("--idempotency-key")] = None,
) -> None:
    try:
        settings, owner_id, item_id = (
            _database(database),
            _uuid(owner),
            _uuid(import_id),
        )
        key = _key(idempotency_key)
        retained_reason = StructuredReason.from_user_text(_required(reason))
    except Exception:
        _fail(PassportError("invalid"))
    _result(_run(_reject(settings, owner_id, item_id, retained_reason, key)))
