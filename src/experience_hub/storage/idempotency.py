"""Durable idempotency receipts and their atomic state transitions."""

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import Select, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock, require_utc
from experience_hub.domain.commands import (
    CommandContext,
    CommandRequest,
    ReplayableCommandError,
)
from experience_hub.errors import DomainError
from experience_hub.ids import IdGenerator
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import IdempotencyRecordRow
from experience_hub.storage.unit_of_work import UnitOfWork

if False:  # pragma: no cover - imports used only by static type checkers
    from experience_hub.storage.database import Database


class IdempotencyIntegrityError(RuntimeError):
    """A receipt transition conflicts with already durable state."""


class IdempotencyKeyConflict(DomainError):
    def __init__(self) -> None:
        super().__init__(
            "idempotency_key_conflict",
            "The idempotency key was already used for a different request",
            status_code=409,
        )


@dataclass(frozen=True, slots=True)
class StoredResponse:
    status_code: int
    body: bytes
    content_type: str = "application/json"
    headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not 100 <= self.status_code <= 599:
            raise ValueError("Response status code must be between 100 and 599")
        content_type = self.content_type.strip()
        if not content_type:
            raise ValueError("Response content type must not be blank")
        headers = {
            str(name).strip().lower(): str(value)
            for name, value in (self.headers or {}).items()
        }
        if any(not name for name in headers):
            raise ValueError("Response header names must not be blank")
        object.__setattr__(self, "body", bytes(self.body))
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "headers", MappingProxyType(headers))


@dataclass(frozen=True, slots=True)
class CommandResult:
    status_code: int
    body: bytes
    content_type: str
    headers: Mapping[str, str]
    replayed: bool


@dataclass(frozen=True, slots=True)
class ReceiptReservation:
    receipt_id: UUID
    caller_scope: str
    operation_scope: str
    idempotency_key: str
    request_hash: str
    created_at: datetime

    def command_context(self) -> CommandContext:
        return CommandContext(
            receipt_id=self.receipt_id,
            caller_scope=self.caller_scope,
            operation_scope=self.operation_scope,
            idempotency_key=self.idempotency_key,
            request_hash=self.request_hash,
        )


@dataclass(frozen=True, slots=True)
class ReceiptRecord:
    receipt_id: UUID
    caller_scope: str
    operation_scope: str
    idempotency_key: str
    request_hash: str
    state: Literal["in_progress", "completed"]
    result_resource_type: str | None
    result_resource_id: UUID | None
    response: StoredResponse | None
    created_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        if self.state not in {"in_progress", "completed"}:
            raise ValueError(f"Unknown receipt state: {self.state}")
        has_resource = self.result_resource_type is not None
        if has_resource != (self.result_resource_id is not None):
            raise ValueError("Receipt resource type and ID must be present together")
        if self.state == "completed":
            if self.response is None or self.completed_at is None:
                raise ValueError(
                    "Completed receipt requires a response and completion time"
                )
        elif self.response is not None or self.completed_at is not None:
            raise ValueError("In-progress receipt cannot contain completion data")


@dataclass(frozen=True, slots=True)
class NewReceipt:
    reservation: ReceiptReservation


@dataclass(frozen=True, slots=True)
class CompletedReceipt:
    record: ReceiptRecord


@dataclass(frozen=True, slots=True)
class InProgressReceipt:
    record: ReceiptRecord


type ReceiptDecision = NewReceipt | CompletedReceipt | InProgressReceipt
type CommandHandler = Callable[[UnitOfWork, CommandContext], Awaitable[StoredResponse]]
type ReservationPreflight = Callable[
    [UnitOfWork, datetime],
    Awaitable[None],
]


class CommandExecutor:
    """Execute a command and its receipt in one serialized writer transaction."""

    _ALLOWED_RESPONSE_HEADERS = frozenset(
        {"content-location", "etag", "last-modified", "location", "retry-after"}
    )

    def __init__(
        self,
        *,
        database: "Database",
        receipt_store: "ReceiptStore",
        clock: Clock,
    ) -> None:
        self.database = database
        self._receipt_store = receipt_store
        self._clock = clock

    async def execute(
        self,
        request: CommandRequest,
        handler: CommandHandler,
        *,
        reservation_preflight: ReservationPreflight | None = None,
    ) -> CommandResult:
        async with self.database.transaction(immediate=True) as uow:
            decision = await self._receipt_store.reserve(
                uow=uow,
                request=request,
                reservation_preflight=reservation_preflight,
            )
            if isinstance(decision, CompletedReceipt):
                assert decision.record.response is not None
                return self._result(decision.record.response, replayed=True)
            if isinstance(decision, InProgressReceipt):
                return self._in_progress_result(decision.record)

            context = decision.reservation.command_context()
            try:
                async with uow.session.begin_nested():
                    response = await handler(uow, context)
            except ReplayableCommandError as error:
                response = StoredResponse(
                    status_code=error.status_code,
                    body=canonical_json_bytes(
                        {
                            "error": {
                                "code": error.code,
                                "details": error.details,
                                "message": error.message,
                            }
                        }
                    ),
                )

            stored = self._allowed_response(response)
            await self._receipt_store.complete(
                uow=uow,
                reservation=decision.reservation,
                response=stored,
                completed_at=max(
                    require_utc(decision.reservation.created_at),
                    require_utc(self._clock.now()),
                ),
            )
            uow.inject_fault(FaultCheckpoint.AFTER_RECEIPT_COMPLETION)
            return self._result(stored, replayed=False)

    @classmethod
    def _allowed_response(cls, response: StoredResponse) -> StoredResponse:
        return StoredResponse(
            status_code=response.status_code,
            body=response.body,
            content_type=response.content_type,
            headers={
                name: value
                for name, value in (response.headers or {}).items()
                if name in cls._ALLOWED_RESPONSE_HEADERS
            },
        )

    @staticmethod
    def _result(response: StoredResponse, *, replayed: bool) -> CommandResult:
        return CommandResult(
            status_code=response.status_code,
            body=response.body,
            content_type=response.content_type,
            headers=MappingProxyType(dict(response.headers or {})),
            replayed=replayed,
        )

    @staticmethod
    def _in_progress_result(record: ReceiptRecord) -> CommandResult:
        resource = None
        if record.result_resource_type is not None:
            assert record.result_resource_id is not None
            resource = {
                "id": str(record.result_resource_id),
                "type": record.result_resource_type,
            }
        response = StoredResponse(
            status_code=409,
            body=canonical_json_bytes(
                {
                    "error": {
                        "code": "operation_in_progress",
                        "details": {
                            "receipt_id": str(record.receipt_id),
                            "resource": resource,
                        },
                        "message": "The operation is still in progress",
                    }
                }
            ),
            headers={"retry-after": "1"},
        )
        return CommandExecutor._result(response, replayed=False)


class ReceiptStore:
    def __init__(self, *, clock: Clock, id_generator: IdGenerator) -> None:
        self._clock = clock
        self._id_generator = id_generator

    async def reserve(
        self,
        *,
        uow: UnitOfWork,
        request: CommandRequest,
        reservation_preflight: ReservationPreflight | None = None,
    ) -> ReceiptDecision:
        row = await uow.session.scalar(self._scope_query(request))
        if row is not None:
            if row.request_hash != request.request_hash:
                raise IdempotencyKeyConflict
            record = self._record(row)
            if record.state == "completed":
                return CompletedReceipt(record)
            return InProgressReceipt(record)

        created_at = require_utc(self._clock.now())
        if reservation_preflight is not None:
            await reservation_preflight(uow, created_at)
        reservation = ReceiptReservation(
            receipt_id=self._id_generator.new(),
            caller_scope=request.caller_scope,
            operation_scope=request.operation_scope,
            idempotency_key=request.idempotency_key,
            request_hash=request.request_hash,
            created_at=created_at,
        )
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=reservation.receipt_id,
                caller_scope=reservation.caller_scope,
                scope=reservation.operation_scope,
                idempotency_key=reservation.idempotency_key,
                request_hash=reservation.request_hash,
                state="in_progress",
                created_at=created_at,
            )
        )
        await uow.session.flush()
        return NewReceipt(reservation)

    async def attach_resource(
        self,
        *,
        uow: UnitOfWork,
        receipt_id: UUID,
        resource_type: str,
        resource_id: UUID,
    ) -> None:
        normalized_type = resource_type.strip()
        if not normalized_type:
            raise ValueError("Receipt resource type must not be blank")
        requested = (normalized_type, resource_id)
        result = cast(
            CursorResult[Any],
            await uow.session.execute(
                update(IdempotencyRecordRow)
                .where(
                    IdempotencyRecordRow.receipt_id == receipt_id,
                    IdempotencyRecordRow.state == "in_progress",
                    IdempotencyRecordRow.result_resource_type.is_(None),
                    IdempotencyRecordRow.result_resource_id.is_(None),
                )
                .values(
                    result_resource_type=normalized_type,
                    result_resource_id=resource_id,
                )
            ),
        )
        if result.rowcount == 1:
            return
        row = await self._required_row(uow.session, receipt_id, refresh=True)
        existing = (row.result_resource_type, row.result_resource_id)
        if existing != requested:
            raise IdempotencyIntegrityError("Receipt resource is immutable")

    async def get_by_id(
        self,
        *,
        session: AsyncSession,
        receipt_id: UUID,
    ) -> ReceiptRecord | None:
        row = await session.get(IdempotencyRecordRow, receipt_id)
        return None if row is None else self._record(row)

    async def find_for_request(
        self,
        *,
        session: AsyncSession,
        request: CommandRequest,
    ) -> ReceiptRecord | None:
        """Inspect an existing canonical command receipt without reserving one."""
        if not isinstance(request, CommandRequest):
            raise TypeError("request must be a CommandRequest")
        row = await session.scalar(self._scope_query(request))
        if row is None:
            return None
        if row.request_hash != request.request_hash:
            raise IdempotencyKeyConflict
        return self._record(row)

    async def find_completed_resource(
        self,
        *,
        session: AsyncSession,
        resource_type: str,
        resource_id: UUID,
    ) -> ReceiptRecord | None:
        row = await session.scalar(
            select(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.result_resource_type == resource_type,
                IdempotencyRecordRow.result_resource_id == resource_id,
                IdempotencyRecordRow.state == "completed",
                IdempotencyRecordRow.response_status_code >= 200,
                IdempotencyRecordRow.response_status_code < 300,
            )
            .order_by(
                IdempotencyRecordRow.completed_at,
                IdempotencyRecordRow.created_at,
                IdempotencyRecordRow.receipt_id,
            )
            .limit(1)
        )
        return None if row is None else self._record(row)

    async def complete(
        self,
        *,
        uow: UnitOfWork,
        reservation: ReceiptReservation,
        response: StoredResponse,
        completed_at: datetime,
    ) -> None:
        await self._complete_cas(
            uow,
            receipt_id=reservation.receipt_id,
            response=response,
            completed_at=completed_at,
            reservation=reservation,
        )

    async def complete_existing(
        self,
        *,
        uow: UnitOfWork,
        receipt_id: UUID,
        response: StoredResponse,
        completed_at: datetime,
    ) -> None:
        await self._complete_cas(
            uow,
            receipt_id=receipt_id,
            response=response,
            completed_at=completed_at,
        )

    async def _complete_cas(
        self,
        uow: UnitOfWork,
        *,
        receipt_id: UUID,
        response: StoredResponse,
        completed_at: datetime,
        reservation: ReceiptReservation | None = None,
    ) -> None:
        conditions = [
            IdempotencyRecordRow.receipt_id == receipt_id,
            IdempotencyRecordRow.state == "in_progress",
        ]
        if reservation is not None:
            conditions.extend(
                [
                    IdempotencyRecordRow.caller_scope == reservation.caller_scope,
                    IdempotencyRecordRow.scope == reservation.operation_scope,
                    IdempotencyRecordRow.idempotency_key == reservation.idempotency_key,
                    IdempotencyRecordRow.request_hash == reservation.request_hash,
                ]
            )
        result = cast(
            CursorResult[Any],
            await uow.session.execute(
                update(IdempotencyRecordRow)
                .where(*conditions)
                .values(
                    state="completed",
                    response_status_code=response.status_code,
                    response_body=response.body,
                    response_content_type=response.content_type,
                    response_headers=canonical_json_bytes(dict(response.headers or {})),
                    completed_at=require_utc(completed_at),
                )
            ),
        )
        if result.rowcount == 1:
            return
        row = await self._required_row(uow.session, receipt_id, refresh=True)
        if reservation is not None and (
            row.caller_scope,
            row.scope,
            row.idempotency_key,
            row.request_hash,
        ) != (
            reservation.caller_scope,
            reservation.operation_scope,
            reservation.idempotency_key,
            reservation.request_hash,
        ):
            raise IdempotencyIntegrityError("Receipt reservation identity changed")
        if row.state == "completed" and self._response(row) == response:
            return
        if row.state == "completed":
            raise IdempotencyIntegrityError("Receipt response is immutable")
        raise IdempotencyIntegrityError("Receipt completion compare-and-set failed")

    @staticmethod
    async def _required_row(
        session: AsyncSession,
        receipt_id: UUID,
        *,
        refresh: bool = False,
    ) -> IdempotencyRecordRow:
        statement = select(IdempotencyRecordRow).where(
            IdempotencyRecordRow.receipt_id == receipt_id
        )
        if refresh:
            statement = statement.execution_options(populate_existing=True)
        row = await session.scalar(statement)
        if row is None:
            raise LookupError(f"Unknown idempotency receipt: {receipt_id}")
        return row

    @staticmethod
    def _scope_query(request: CommandRequest) -> Select[tuple[IdempotencyRecordRow]]:
        return select(IdempotencyRecordRow).where(
            IdempotencyRecordRow.caller_scope == request.caller_scope,
            IdempotencyRecordRow.scope == request.operation_scope,
            IdempotencyRecordRow.idempotency_key == request.idempotency_key,
        )

    @staticmethod
    def _response(row: IdempotencyRecordRow) -> StoredResponse | None:
        if row.response_status_code is None:
            return None
        assert row.response_body is not None
        assert row.response_content_type is not None
        assert row.response_headers is not None
        headers = json.loads(row.response_headers)
        if not isinstance(headers, dict):
            raise IdempotencyIntegrityError("Stored response headers are invalid")
        return StoredResponse(
            status_code=row.response_status_code,
            body=row.response_body,
            content_type=row.response_content_type,
            headers={str(name): str(value) for name, value in headers.items()},
        )

    @classmethod
    def _record(cls, row: IdempotencyRecordRow) -> ReceiptRecord:
        state: Literal["in_progress", "completed"]
        if row.state not in {"in_progress", "completed"}:
            raise IdempotencyIntegrityError(f"Unknown receipt state: {row.state}")
        state = cast(Literal["in_progress", "completed"], row.state)
        return ReceiptRecord(
            receipt_id=row.receipt_id,
            caller_scope=row.caller_scope,
            operation_scope=row.scope,
            idempotency_key=row.idempotency_key,
            request_hash=row.request_hash,
            state=state,
            result_resource_type=row.result_resource_type,
            result_resource_id=row.result_resource_id,
            response=cls._response(row),
            created_at=row.created_at,
            completed_at=row.completed_at,
        )
