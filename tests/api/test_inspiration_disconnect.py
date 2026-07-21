from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import pytest

from experience_hub.api.app import create_app
from experience_hub.api.inflight import InFlightRunRegistry
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest
from experience_hub.inspiration import (
    GeneratorKind,
    GeneratorResult,
    IdeaDraft,
    InspirationOperator,
    InspirationRunExecutor,
    SnapshotEvidenceReference,
    SnapshotItem,
    StartInspirationRun,
)
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import StoredResponse

NOW = datetime(2026, 7, 21, 8, 30, tzinfo=UTC)


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


@dataclass(slots=True)
class _ControlledGenerator:
    started: asyncio.Event
    release: asyncio.Event | None
    calls: list[InspirationOperator] = field(default_factory=list)

    @property
    def reserves_output_tokens(self) -> bool:
        return False

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        return None

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        _ = (goal, context, branch_limit, output_token_limit)
        self.calls.append(operator)
        self.started.set()
        if self.release is None:
            await asyncio.Event().wait()
        else:
            await self.release.wait()
        item = frozen_items[0]
        return GeneratorResult(
            ideas=(
                IdeaDraft(
                    title="Retain work after request cancellation",
                    hypothesis="Shielded run work reaches a durable terminal receipt.",
                    mechanism="The application registry owns the executor task.",
                    predictions=("A replay returns the first stored response.",),
                    falsifiers=("A replay starts the generator a second time.",),
                    assumptions=("The application remains alive.",),
                    proposed_test="Cancel the ASGI request and replay its key.",
                    evidence=(
                        SnapshotEvidenceReference(
                            id=item.snapshot_item_id,
                            stable_evidence_key=item.stable_evidence_key,
                        ),
                    ),
                ),
            ),
            output_tokens_consumed=0,
        )


@dataclass(slots=True)
class _CountingRunExecutor:
    delegate: InspirationRunExecutor
    calls: int = 0

    async def execute(
        self,
        *,
        request: CommandRequest,
        run: StartInspirationRun,
    ) -> StoredResponse:
        self.calls += 1
        return await self.delegate.execute(request=request, run=run)


async def _create_agent_and_evidence(client: httpx.AsyncClient) -> UUID:
    created = await client.post(
        "/v1/agents",
        headers={"Idempotency-Key": "disconnect-owner"},
        json={"name": "Disconnect owner"},
    )
    assert created.status_code == 201, created.text
    owner_id = UUID(created.json()["data"]["agent_id"])
    evidence = await client.post(
        f"/v1/agents/{owner_id}/experiences",
        headers={"Idempotency-Key": "disconnect-evidence"},
        json={
            "applicability": ["client disconnect"],
            "body": "A retained task outlives the cancelled request task.",
            "confidence": 0.8,
            "evidence": [{"id": "asgi-cancel", "type": "test"}],
            "falsifiers": ["The generator is invoked twice for one key."],
            "importance": 0.8,
            "kind": "procedural",
            "mechanism": "asyncio.shield separates request and run cancellation.",
            "summary": "Retain bounded work across a client disconnect.",
            "tags": ["asgi", "recovery"],
        },
    )
    assert evidence.status_code == 201, evidence.text
    return owner_id


async def _wait_for_empty_registry(
    registry: InFlightRunRegistry,
    *,
    timeout_seconds: float = 5,
) -> None:
    async with asyncio.timeout(timeout_seconds):
        while registry.active_count:
            await asyncio.sleep(0.01)


def _original_run_receipt(database_path: Path, *, key: str) -> sqlite3.Row:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT result_resource_id, state, response_status_code, response_body "
                "FROM idempotency_records "
                "WHERE scope = 'inspiration.run.start' AND idempotency_key = ?",
                (key,),
            ).fetchone(),
        )
    assert row is not None
    return row


@pytest.mark.asyncio
async def test_request_cancellation_does_not_cancel_retained_run() -> None:
    registry = InFlightRunRegistry()
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def run() -> str:
        started.set()
        await release.wait()
        completed.set()
        return "terminal"

    request = asyncio.create_task(
        registry.execute(
            run,
            shutdown_timeout_seconds=1,
        )
    )
    await started.wait()

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert registry.active_count == 1
    assert not completed.is_set()
    release.set()
    await registry.shutdown()

    assert completed.is_set()
    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_work_past_its_deadline() -> None:
    registry = InFlightRunRegistry()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def run() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    request = asyncio.create_task(
        registry.execute(
            run,
            shutdown_timeout_seconds=0.01,
        )
    )
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    await registry.shutdown()

    assert cancelled.is_set()
    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_shutdown_enforces_each_retained_run_deadline_independently() -> None:
    registry = InFlightRunRegistry()
    short_started = asyncio.Event()
    short_cancelled = asyncio.Event()
    long_started = asyncio.Event()
    long_release = asyncio.Event()
    long_cancelled = asyncio.Event()

    async def short_run() -> None:
        short_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            short_cancelled.set()

    async def long_run() -> None:
        long_started.set()
        try:
            await long_release.wait()
        except asyncio.CancelledError:
            long_cancelled.set()
            raise

    short_request = asyncio.create_task(
        registry.execute(
            short_run,
            shutdown_timeout_seconds=0.01,
        )
    )
    long_request = asyncio.create_task(
        registry.execute(
            long_run,
            shutdown_timeout_seconds=1,
        )
    )
    await asyncio.gather(short_started.wait(), long_started.wait())
    short_request.cancel()
    long_request.cancel()
    for request in (short_request, long_request):
        with pytest.raises(asyncio.CancelledError):
            await request

    shutdown = asyncio.create_task(registry.shutdown())
    async with asyncio.timeout(0.25):
        await short_cancelled.wait()
    assert not long_cancelled.is_set()
    long_release.set()
    await shutdown

    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_closed_registry_rejects_new_work_without_starting_it() -> None:
    registry = InFlightRunRegistry()
    await registry.shutdown()
    called = False

    async def run() -> None:
        nonlocal called
        called = True

    factory: Callable[[], Coroutine[Any, Any, None]] = run
    with pytest.raises(RuntimeError, match="shutting down"):
        await registry.execute(
            factory,
            shutdown_timeout_seconds=1,
        )

    assert not called


@pytest.mark.asyncio
async def test_cancelled_asgi_request_finishes_once_and_replays_stored_bytes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "asgi-disconnect.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    generator = _ControlledGenerator(started=started, release=release)
    body = {
        "goal": "Keep bounded generation alive after a client disconnect.",
        "operators": ["causal_gap"],
        "branches_per_operator": 1,
        "operator_timeout_seconds": 5,
        "global_timeout_seconds": 5,
    }
    key = "cancelled-asgi-run"

    async with app.router.lifespan_context(app):
        app.state.container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
            lambda _: generator
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            owner_id = await _create_agent_and_evidence(client)
            path = f"/v1/agents/{owner_id}/inspiration-runs"
            executor = _CountingRunExecutor(
                delegate=app.state.container.inspiration_run_executor
            )
            app.state.container.inspiration_run_executor = executor
            request_task = asyncio.create_task(
                client.post(
                    path,
                    headers={"Idempotency-Key": key},
                    json=body,
                )
            )
            await started.wait()
            active_receipt = _original_run_receipt(database_path, key=key)
            run_id = UUID(active_receipt["result_resource_id"])
            assert active_receipt["state"] == "in_progress"

            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task

            registry = app.state.inflight_runs
            assert registry.active_count == 1
            in_progress = await client.post(
                path,
                headers={"Idempotency-Key": key},
                json=body,
            )
            assert in_progress.status_code == 409
            assert in_progress.headers["retry-after"] == "1"
            UUID(in_progress.headers["x-request-id"])
            assert in_progress.json()["error"]["code"] == "operation_in_progress"
            assert in_progress.json()["error"]["details"]["resource"] == {
                "id": str(run_id),
                "type": "inspiration_run",
            }
            assert generator.calls == [InspirationOperator.CAUSAL_GAP]
            assert executor.calls == 1
            release.set()
            await _wait_for_empty_registry(registry)

            completed_receipt = _original_run_receipt(database_path, key=key)
            stored_body = bytes(completed_receipt["response_body"])
            replay = await client.post(
                path,
                headers={"Idempotency-Key": key},
                json=body,
            )

    assert completed_receipt["state"] == "completed"
    assert completed_receipt["response_status_code"] == 201
    assert UUID(completed_receipt["result_resource_id"]) == run_id
    assert replay.status_code == 201
    assert replay.content == stored_body
    assert replay.json()["data"]["run_id"] == str(run_id)
    assert generator.calls == [InspirationOperator.CAUSAL_GAP]


@pytest.mark.asyncio
async def test_shutdown_cancellation_is_recovered_once_on_next_startup(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "asgi-shutdown-recovery.sqlite3"
    settings = _settings(database_path)
    app = create_app(settings=settings, clock=FrozenClock(NOW))
    started = asyncio.Event()
    generator = _ControlledGenerator(started=started, release=None)
    body = {
        "goal": "Recover work interrupted by application shutdown.",
        "operators": ["causal_gap"],
        "branches_per_operator": 1,
        "operator_timeout_seconds": 30,
        "global_timeout_seconds": 90,
    }
    key = "shutdown-interrupted-run"

    async with app.router.lifespan_context(app):
        app.state.container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
            lambda _: generator
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            owner_id = await _create_agent_and_evidence(client)
            path = f"/v1/agents/{owner_id}/inspiration-runs"
            request_task = asyncio.create_task(
                client.post(
                    path,
                    headers={"Idempotency-Key": key},
                    json=body,
                )
            )
            await started.wait()
            active_receipt = _original_run_receipt(database_path, key=key)
            run_id = UUID(active_receipt["result_resource_id"])

            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task

            registry = app.state.inflight_runs
            assert registry.active_count == 1
            loop = asyncio.get_running_loop()
            for task in registry._tasks:  # noqa: SLF001
                registry._tasks[task] = loop.time() - 1  # noqa: SLF001

    interrupted_receipt = _original_run_receipt(database_path, key=key)
    assert interrupted_receipt["state"] == "in_progress"
    assert generator.calls == [InspirationOperator.CAUSAL_GAP]

    startup_generator_calls: list[GeneratorKind] = []

    def container_factory(**kwargs: Any) -> ApplicationContainer:
        container = ApplicationContainer.build(**kwargs)

        def fail_if_generated(kind: GeneratorKind) -> _ControlledGenerator:
            startup_generator_calls.append(kind)
            raise AssertionError("startup recovery must not restart generation")

        container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
            fail_if_generated
        )
        return container

    runtime = ApplicationRuntime(
        settings=settings,
        clock=FrozenClock(NOW),
        container_factory=container_factory,
    )
    recovered_app = create_app(runtime=runtime)
    async with recovered_app.router.lifespan_context(recovered_app):
        transport = httpx.ASGITransport(
            app=recovered_app,
            raise_app_exceptions=True,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            replay = await client.post(
                path,
                headers={"Idempotency-Key": key},
                json=body,
            )

        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            receipts = connection.execute(
                "SELECT scope, state, response_status_code, response_body "
                "FROM idempotency_records "
                "WHERE result_resource_id = ? "
                "AND scope IN ('inspiration.run.start', 'inspiration.run.recover') "
                "ORDER BY scope",
                (str(run_id),),
            ).fetchall()
            failures = connection.execute(
                "SELECT payload FROM domain_events "
                "WHERE aggregate_id = ? AND event_type = 'inspiration.failed'",
                (str(run_id),),
            ).fetchall()

    assert startup_generator_calls == []
    assert len(receipts) == 2
    assert {row["scope"] for row in receipts} == {
        "inspiration.run.start",
        "inspiration.run.recover",
    }
    assert all(row["state"] == "completed" for row in receipts)
    assert all(row["response_status_code"] == 201 for row in receipts)
    assert receipts[0]["response_body"] == receipts[1]["response_body"]
    assert len(failures) == 1
    assert json.loads(failures[0]["payload"])["failure_code"] == "process_interrupted"
    stored_body = bytes(receipts[0]["response_body"])
    assert replay.status_code == 201
    assert replay.content == stored_body
    assert replay.json()["data"]["status"] == "failed"
