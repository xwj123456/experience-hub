"""Interruptible background execution for deterministic lifecycle cycles."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.errors import DomainError
from experience_hub.lifecycle.service import LifecycleService
from experience_hub.storage.database import DatabaseBusy
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork

_MAX_RECORDED_FAILURES = 100
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")


class LifecycleTicker(Protocol):
    """Release lifecycle work and provide an interruptible shutdown wake-up."""

    async def wait_for_tick(self) -> bool:
        """Return true for work or false after the ticker has closed."""

    def close(self) -> None:
        """Wake a waiter and permanently close this ticker."""


class ProductionLifecycleTicker:
    """Release a tick after each interval unless shutdown interrupts the wait."""

    def __init__(self, interval: timedelta) -> None:
        if not isinstance(interval, timedelta) or interval <= timedelta(0):
            raise ValueError("Lifecycle ticker interval must be positive")
        self._interval_seconds = interval.total_seconds()
        self._wake = asyncio.Event()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def wait_for_tick(self) -> bool:
        if self._closed:
            return False
        try:
            await asyncio.wait_for(
                self._wake.wait(),
                timeout=self._interval_seconds,
            )
        except TimeoutError:
            return not self._closed
        self._wake.clear()
        return not self._closed

    def close(self) -> None:
        self._closed = True
        self._wake.set()


@dataclass(slots=True)
class _ManualTick:
    processed: asyncio.Future[None]


_MANUAL_CLOSE = object()


class ManualLifecycleTicker:
    """Release explicit test ticks and acknowledge completed worker handling."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_ManualTick | object] = asyncio.Queue()
        self._active: _ManualTick | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def tick(self) -> None:
        """Release one tick and return after its worker handling completes."""
        if self._closed:
            raise RuntimeError("Lifecycle ticker is closed")
        processed = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_ManualTick(processed=processed))
        await asyncio.shield(processed)

    async def wait_for_tick(self) -> bool:
        self._acknowledge_active()
        if self._closed:
            return False
        item = await self._queue.get()
        if item is _MANUAL_CLOSE:
            return False
        if not isinstance(item, _ManualTick):
            raise RuntimeError("Manual lifecycle ticker queue is corrupt")
        if self._closed:
            if not item.processed.done():
                item.processed.set_exception(
                    RuntimeError("Lifecycle ticker closed before processing")
                )
            return False
        self._active = item
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _ManualTick) and not item.processed.done():
                item.processed.set_exception(
                    RuntimeError("Lifecycle ticker closed before processing")
                )
        self._queue.put_nowait(_MANUAL_CLOSE)

    def _acknowledge_active(self) -> None:
        active = self._active
        self._active = None
        if active is not None and not active.processed.done():
            active.processed.set_result(None)


type LifecycleHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]


class LifecycleCommandExecutor(Protocol):
    async def execute(
        self,
        request: CommandRequest,
        handler: LifecycleHandler,
    ) -> CommandResult: ...


@dataclass(frozen=True, slots=True)
class LifecycleWorkerFailure:
    """Bounded, user-safe failure metadata without raw exception details."""

    cycle_id: UUID
    evaluated_at: datetime
    code: str
    status_code: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, UUID):
            raise ValueError("cycle_id must be a UUID")
        object.__setattr__(
            self,
            "evaluated_at",
            require_utc(self.evaluated_at),
        )
        if _SAFE_ERROR_CODE.fullmatch(self.code) is None:
            raise ValueError("Failure code is not safe")
        if self.status_code is not None and not (
            isinstance(self.status_code, int)
            and not isinstance(self.status_code, bool)
            and 400 <= self.status_code <= 599
        ):
            raise ValueError("Failure status code must be an HTTP error")


class LifecycleWorker:
    """Run serialized background lifecycle commands for ticker releases."""

    def __init__(
        self,
        *,
        clock: Clock,
        ticker: LifecycleTicker,
        executor: LifecycleCommandExecutor,
        service: LifecycleService,
    ) -> None:
        self._clock = clock
        self._ticker = ticker
        self._executor = executor
        self._service = service
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._failures: deque[LifecycleWorkerFailure] = deque(
            maxlen=_MAX_RECORDED_FAILURES
        )

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def failures(self) -> tuple[LifecycleWorkerFailure, ...]:
        return tuple(self._failures)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Lifecycle worker has already been started")
        self._started = True
        self._task = asyncio.create_task(
            self._run(),
            name="experience-hub-lifecycle-worker",
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._ticker.close()
        await self._task

    async def _run(self) -> None:
        while await self._ticker.wait_for_tick():
            await self._run_cycle()

    async def _run_cycle(self) -> None:
        evaluated_at = require_utc(self._clock.now())
        cycle_id = self._service.cycle_id(evaluated_at)
        request = CommandRequest(
            caller_scope="system:local",
            operation_scope="lifecycle.run",
            idempotency_key=f"lifecycle:{cycle_id}",
            method="POST",
            route_template="/v1/lifecycle:run",
            body={
                "evaluated_at": evaluated_at,
                "mode": "background",
            },
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            return await self._service.run(
                uow=uow,
                evaluated_at=evaluated_at,
                command=command,
                mode="background",
            )

        try:
            result = await self._executor.execute(request, handler)
        except asyncio.CancelledError:
            raise
        except DatabaseBusy:
            self._record_failure(
                cycle_id=cycle_id,
                evaluated_at=evaluated_at,
                code="database_busy",
                status_code=503,
            )
        except DomainError as error:
            self._record_failure(
                cycle_id=cycle_id,
                evaluated_at=evaluated_at,
                code=_sanitized_code(error.code, "command_failed"),
                status_code=_sanitized_status(error.status_code),
            )
        except Exception:
            self._record_failure(
                cycle_id=cycle_id,
                evaluated_at=evaluated_at,
                code="internal_error",
                status_code=None,
            )
        else:
            if not 200 <= result.status_code <= 299:
                self._record_failure(
                    cycle_id=cycle_id,
                    evaluated_at=evaluated_at,
                    code=_response_error_code(result.body),
                    status_code=_sanitized_status(result.status_code),
                )

    def _record_failure(
        self,
        *,
        cycle_id: UUID,
        evaluated_at: datetime,
        code: str,
        status_code: int | None,
    ) -> None:
        self._failures.append(
            LifecycleWorkerFailure(
                cycle_id=cycle_id,
                evaluated_at=evaluated_at,
                code=code,
                status_code=status_code,
            )
        )


def _response_error_code(body: bytes) -> str:
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "command_failed"
    if not isinstance(decoded, dict):
        return "command_failed"
    error = decoded.get("error")
    if not isinstance(error, dict):
        return "command_failed"
    return _sanitized_code(error.get("code"), "command_failed")


def _sanitized_code(value: object, fallback: str) -> str:
    if isinstance(value, str) and _SAFE_ERROR_CODE.fullmatch(value) is not None:
        return value
    return fallback


def _sanitized_status(value: object) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 400 <= value <= 599
    ):
        return value
    return None


__all__ = [
    "LifecycleCommandExecutor",
    "LifecycleTicker",
    "LifecycleWorker",
    "LifecycleWorkerFailure",
    "ManualLifecycleTicker",
    "ProductionLifecycleTicker",
]
