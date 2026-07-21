from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from typer.testing import CliRunner

import experience_hub.runtime as runtime_module
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli.app import app
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.lifecycle import LifecycleResult, encode_lifecycle_result
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import lifecycle_cycle_id
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)

RUNNER = CliRunner()
cli_module = import_module("experience_hub.cli.app")
EVALUATED_AT = datetime(2000, 1, 2, 3, 4, 5, tzinfo=UTC)
CYCLE_ID = UUID("00000000-0000-0000-0000-00000000c001")
RECEIPT_ID = UUID("00000000-0000-0000-0000-00000000c002")
RESULT = LifecycleResult(
    cycle_id=CYCLE_ID,
    evaluated_at=EVALUATED_AT,
    evaluated_count=4,
    transition_count=2,
    archive_count=1,
    idea_archive_count=1,
)
RESULT_BODY = encode_lifecycle_result(RESULT)
type LifecycleHandler = Callable[
    [object, CommandContext],
    Awaitable[StoredResponse],
]


@dataclass(slots=True)
class _ServiceProbe:
    calls: list[dict[str, Any]]

    async def run(self, **kwargs: Any) -> StoredResponse:
        self.calls.append(kwargs)
        return StoredResponse(status_code=200, body=RESULT_BODY)


@dataclass(slots=True)
class _ExecutorProbe:
    requests: list[CommandRequest]
    uow: object

    async def execute(
        self,
        request: CommandRequest,
        handler: LifecycleHandler,
    ) -> CommandResult:
        self.requests.append(request)
        context = CommandContext(
            receipt_id=RECEIPT_ID,
            caller_scope=request.caller_scope,
            operation_scope=request.operation_scope,
            idempotency_key=request.idempotency_key,
            request_hash=request.request_hash,
        )
        response = await handler(self.uow, context)
        return CommandResult(
            status_code=response.status_code,
            body=response.body,
            content_type=response.content_type,
            headers=response.headers or {},
            replayed=False,
        )


@dataclass(slots=True)
class _ErrorExecutorProbe:
    body: bytes
    requests: list[CommandRequest]

    async def execute(
        self,
        request: CommandRequest,
        handler: LifecycleHandler,
    ) -> CommandResult:
        del handler
        self.requests.append(request)
        return CommandResult(
            status_code=409,
            body=self.body,
            content_type="application/json",
            headers={},
            replayed=False,
        )


@dataclass(slots=True)
class _RuntimeObservation:
    settings: Settings
    initialize_calls: list[tuple[bool, bool]]
    entered: bool = False
    exited: bool = False


def _install_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    executor: _ExecutorProbe | _ErrorExecutorProbe,
    service: _ServiceProbe,
) -> list[_RuntimeObservation]:
    observations: list[_RuntimeObservation] = []

    class FakeApplicationRuntime:
        def __init__(self, settings: Settings, **_: object) -> None:
            self.observation = _RuntimeObservation(
                settings=settings,
                initialize_calls=[],
            )
            observations.append(self.observation)

        @asynccontextmanager
        async def initialize(
            self,
            *,
            start_lifecycle_worker: bool,
            recover_interrupted: bool,
        ) -> AsyncIterator[SimpleNamespace]:
            self.observation.initialize_calls.append(
                (start_lifecycle_worker, recover_interrupted)
            )
            self.observation.entered = True
            try:
                yield SimpleNamespace(
                    command_executor=executor,
                    lifecycle_service=service,
                )
            finally:
                self.observation.exited = True

    monkeypatch.setattr(
        cli_module,
        "ApplicationRuntime",
        FakeApplicationRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "ApplicationRuntime",
        FakeApplicationRuntime,
    )
    return observations


def test_lifecycle_help_exposes_run_database_and_evaluation_time() -> None:
    group = RUNNER.invoke(app, ["lifecycle", "--help"])
    command = RUNNER.invoke(app, ["lifecycle", "run", "--help"])

    assert group.exit_code == 0, group.output
    assert "run" in group.output
    assert command.exit_code == 0, command.output
    assert "--database" in command.output
    assert "--evaluated-at" in command.output


def test_lifecycle_run_uses_shared_runtime_and_the_ordinary_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "lifecycle-probe.sqlite3"
    uow = object()
    executor = _ExecutorProbe(requests=[], uow=uow)
    service = _ServiceProbe(calls=[])
    runtimes = _install_runtime_probe(
        monkeypatch,
        executor=executor,
        service=service,
    )

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
            "--evaluated-at",
            "2000-01-02T03:04:05Z",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == RESULT_BODY.decode("utf-8") + "\n"
    assert len(runtimes) == 1
    runtime = runtimes[0]
    assert runtime.settings.database_url == f"sqlite+aiosqlite:///{database_path}"
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.entered
    assert runtime.exited

    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.caller_scope == "system:local"
    assert request.operation_scope == "lifecycle.run"
    assert request.method == "POST"
    assert request.route_template == "/v1/lifecycle:run"
    assert request.idempotency_key
    assert dict(request.body) == {
        "evaluated_at": EVALUATED_AT,
        "mode": "manual",
    }

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["uow"] is uow
    context = call["command"]
    assert isinstance(context, CommandContext)
    assert context.request_hash == request.request_hash
    assert call["evaluated_at"] == EVALUATED_AT
    assert call["mode"] == "manual"
    assert call["evaluated_at_was_omitted"] is False


def test_lifecycle_nonzero_result_still_closes_the_shared_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "lifecycle-error.sqlite3"
    error_body = canonical_json_bytes(
        {
            "error": {
                "code": "lifecycle_in_progress",
                "details": {"mode": "manual"},
                "message": "Another lifecycle cycle is already in progress",
            }
        }
    )
    executor = _ErrorExecutorProbe(body=error_body, requests=[])
    service = _ServiceProbe(calls=[])
    runtimes = _install_runtime_probe(
        monkeypatch,
        executor=executor,
        service=service,
    )

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
            "--evaluated-at",
            "2000-01-02T03:04:05Z",
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == error_body.decode("utf-8") + "\n"
    assert len(runtimes) == 1
    assert runtimes[0].initialize_calls == [(False, False)]
    assert runtimes[0].entered
    assert runtimes[0].exited
    assert len(executor.requests) == 1
    assert service.calls == []


def test_lifecycle_run_migrates_a_fresh_database_and_keeps_one_caller_receipt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-real.sqlite3"
    expected = LifecycleResult(
        cycle_id=lifecycle_cycle_id(
            evaluated_at=EVALUATED_AT,
            config=LifecycleConfig(),
        ),
        evaluated_at=EVALUATED_AT,
        evaluated_count=0,
        transition_count=0,
        archive_count=0,
        idea_archive_count=0,
    )
    expected_body = encode_lifecycle_result(expected)

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
            "--evaluated-at",
            "2000-01-02T03:04:05Z",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == expected_body.decode("utf-8") + "\n"
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        receipts = connection.execute(
            "SELECT caller_scope, scope, idempotency_key, request_hash, state, "
            "result_resource_type, result_resource_id, response_status_code, "
            "response_body FROM idempotency_records"
        ).fetchall()
        revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()

    assert revision is not None
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["caller_scope"] == "system:local"
    assert receipt["scope"] == "lifecycle.run"
    assert receipt["state"] == "completed"
    assert receipt["result_resource_type"] == "lifecycle_cycle"
    assert receipt["result_resource_id"] == str(expected.cycle_id)
    assert receipt["response_status_code"] == 200
    assert bytes(receipt["response_body"]) == expected_body
    expected_request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=receipt["idempotency_key"],
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": EVALUATED_AT, "mode": "manual"},
    )
    assert receipt["request_hash"] == expected_request.request_hash


def test_lifecycle_run_omits_time_from_the_request_but_uses_receipt_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-omitted-time.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
            "--idempotency-key",
            "omitted-time",
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        receipt = connection.execute(
            "SELECT receipt_id, created_at, request_hash, response_body "
            "FROM idempotency_records"
        ).fetchone()

    assert receipt is not None
    created_at = datetime.fromisoformat(receipt["created_at"]).replace(tzinfo=UTC)
    assert output["data"]["evaluated_at"] == (
        created_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    assert result.stdout == bytes(receipt["response_body"]).decode("utf-8") + "\n"
    expected_request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key="omitted-time",
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": None, "mode": "manual"},
    )
    assert receipt["request_hash"] == expected_request.request_hash


def test_lifecycle_run_replays_the_exact_result_for_a_retained_key(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-replay.sqlite3"
    arguments = [
        "lifecycle",
        "run",
        "--database",
        str(database_path),
        "--evaluated-at",
        "2000-01-02T03:04:05Z",
        "--idempotency-key",
        "retained-cli-cycle",
    ]

    first = RUNNER.invoke(app, arguments)
    replay = RUNNER.invoke(app, arguments)

    assert first.exit_code == replay.exit_code == 0
    assert replay.stdout == first.stdout
    with sqlite3.connect(database_path) as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) FROM idempotency_records"
        ).fetchone()
    assert receipt_count == (1,)


@pytest.mark.parametrize(
    "invalid_arguments",
    (
        ("--evaluated-at", "2000-01-02T03:04:05"),
        ("--evaluated-at", "20000102T030405Z"),
        ("--evaluated-at", "2000-W01-7T03:04:05Z"),
        ("--evaluated-at", "2000-01-02 03:04:05Z"),
        ("--evaluated-at", "2000-01-02T03:04:05+00:00:30"),
        ("--idempotency-key", " \t "),
        ("--idempotency-key", "x" * 129),
    ),
)
def test_lifecycle_cli_validation_precedes_database_creation(
    tmp_path: Path,
    invalid_arguments: tuple[str, str],
) -> None:
    database_path = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
            *invalid_arguments,
        ],
    )

    assert result.exit_code == 2
    assert not database_path.exists()


def test_lifecycle_rejects_database_url_syntax_as_a_usage_error(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle?timeout=5"
    silently_truncated_path = tmp_path / "lifecycle"

    result = RUNNER.invoke(
        app,
        [
            "lifecycle",
            "run",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 2
    assert not database_path.exists()
    assert not silently_truncated_path.exists()
