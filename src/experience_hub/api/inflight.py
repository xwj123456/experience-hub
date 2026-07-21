"""Retain inspiration execution independently of one HTTP request task."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Coroutine
from typing import Any


class InFlightRunRegistry:
    """Shield bounded run tasks and drain them before dependencies close."""

    def __init__(self) -> None:
        self._tasks: dict[asyncio.Task[Any], float] = {}
        self._accepting = True

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    async def execute[Result](
        self,
        factory: Callable[[], Coroutine[Any, Any, Result]],
        *,
        shutdown_timeout_seconds: float,
    ) -> Result:
        if not self._accepting:
            raise RuntimeError("In-flight inspiration registry is shutting down")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if (
            isinstance(shutdown_timeout_seconds, bool)
            or not isinstance(shutdown_timeout_seconds, (int, float))
            or not math.isfinite(shutdown_timeout_seconds)
            or not 0 < shutdown_timeout_seconds <= 90
        ):
            raise ValueError(
                "shutdown_timeout_seconds must be greater than zero and at most 90"
            )

        awaitable = factory()
        if not inspect.iscoroutine(awaitable):
            raise TypeError("factory must return a coroutine")
        loop = asyncio.get_running_loop()
        task: asyncio.Task[Result] = asyncio.create_task(
            awaitable,
            name="experience-hub-inspiration-run",
        )
        self._tasks[task] = loop.time() + float(shutdown_timeout_seconds)
        task.add_done_callback(self._completed)
        return await asyncio.shield(task)

    def _completed(self, task: asyncio.Task[Any]) -> None:
        self._tasks.pop(task, None)
        if not task.cancelled():
            task.exception()

    async def shutdown(self) -> None:
        """Wait through retained deadlines, then cancel and gather leftovers."""
        self._accepting = False
        deadlines = dict(self._tasks)
        if not deadlines:
            return
        loop = asyncio.get_running_loop()
        pending = set(deadlines)
        while pending:
            now = loop.time()
            expired = {task for task in pending if deadlines[task] <= now}
            for task in expired:
                task.cancel()
            pending.difference_update(expired)
            if not pending:
                break
            timeout = max(
                0.0,
                min(deadlines[task] for task in pending) - loop.time(),
            )
            _, retained = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending = set(retained)
        await asyncio.gather(*deadlines, return_exceptions=True)
        for task in deadlines:
            self._tasks.pop(task, None)


__all__ = ["InFlightRunRegistry"]
