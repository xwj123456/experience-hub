from __future__ import annotations

import json
import logging
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import experience_hub.config as config
from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli import app
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest

PROJECT_ROOT = Path(__file__).parents[2]
RUNNER = CliRunner()


class _LogRecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _canonical_cli_document(result: Any) -> dict[str, Any]:
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    body = result.stdout[:-1].encode("utf-8")
    document = cast(dict[str, Any], json.loads(body))
    assert canonical_json_bytes(document) == body
    return document


def _assert_release_documentation() -> None:
    assert (PROJECT_ROOT / "uv.lock").is_file(), "missing committed uv.lock"
    requirements = {
        PROJECT_ROOT / "README.md": (
            "## Setup",
            "uv sync --all-groups --frozen",
            "experience-hub demo --reset",
            "experience-hub benchmark",
        ),
        PROJECT_ROOT / "docs" / "architecture" / "system-overview.md": (
            "## Architecture",
            "projection",
            "quarantine",
            "inspiration",
        ),
        PROJECT_ROOT / "docs" / "operations" / "local-runbook.md": (
            "## Operations",
            "backup",
            "projections rebuild --verify",
            "projections rebuild --repair",
            "payloads reconcile",
        ),
    }
    for path, expected_fragments in requirements.items():
        assert path.is_file(), f"missing release document: {path}"
        body = path.read_text(encoding="utf-8")
        lowered = body.lower()
        for fragment in expected_fragments:
            assert fragment.lower() in lowered, (
                f"{path} is missing release section or command: {fragment}"
            )


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


def test_runtime_migration_preserves_existing_loggers(
    tmp_path: Path,
) -> None:
    logger_names = (
        "experience_hub.api.errors",
        "uvicorn.access",
        "uvicorn.error",
    )
    for name in logger_names:
        logging.getLogger(name).disabled = False

    root_logger = logging.getLogger()
    collector = _LogRecordCollector()
    root_logger.addHandler(collector)
    try:
        application = create_app(
            settings=_settings(tmp_path / "logger-preservation.sqlite3")
        )

        @application.get("/_release-acceptance/error-log")
        async def error_log_probe() -> None:
            raise RuntimeError("release acceptance log probe")

        with TestClient(application, raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200
            assert all(
                not logging.getLogger(name).disabled for name in logger_names
            )
            assert collector in root_logger.handlers
            failure = client.get("/_release-acceptance/error-log")
    finally:
        if collector in root_logger.handlers:
            root_logger.removeHandler(collector)

    request_id = failure.headers["x-request-id"]
    assert failure.status_code == 500
    assert failure.json()["error"]["details"]["request_id"] == request_id
    assert any(request_id in record.getMessage() for record in collector.records)
    assert all(not logging.getLogger(name).disabled for name in logger_names)


def _prepare_benchmark_fixtures(root: Path) -> None:
    destination = root / "benchmarks"
    destination.mkdir(parents=True)
    for name in ("seed.json", "cases.jsonl"):
        shutil.copy2(PROJECT_ROOT / "benchmarks" / name, destination / name)


def test_release_candidate_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _assert_release_documentation()
    sandbox_root = tmp_path / "release-root"
    sandbox_root.mkdir()
    _prepare_benchmark_fixtures(sandbox_root)
    monkeypatch.setattr(config, "repository_root", lambda: sandbox_root)

    demo = _canonical_cli_document(RUNNER.invoke(app, ["demo", "--reset"]))
    demo_data = cast(dict[str, Any], demo["data"])
    assert demo_data["all_invariants_hold"] is True
    assert len(cast(list[dict[str, Any]], demo_data["stages"])) == 11
    demo_database = sandbox_root / ".data" / "demo.db"
    assert demo_database.is_file()

    benchmark = _canonical_cli_document(RUNNER.invoke(app, ["benchmark"]))
    benchmark_data = cast(dict[str, Any], benchmark["data"])
    assert benchmark_data["passed"] is True
    assert benchmark_data["failed_gates"] == []
    assert all(
        gate["passed"] is True
        for gate in cast(list[dict[str, Any]], benchmark_data["gates"])
    )
    assert (
        cast(dict[str, Any], benchmark_data["metrics"])[
            "byte_identical_replay"
        ]
        is True
    )

    api_database = tmp_path / "api-lifespan.sqlite3"
    application = create_app(settings=_settings(api_database))
    replay_key = "release-agent-hash-replay"
    replay_name = "Release replay agent"
    execution_results: list[Any] = []
    handler_calls = 0
    with TestClient(application) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["data"]["status"] == "ready"
        assert application.state.ready is True
        container = application.state.container
        original_execute = container.command_executor.execute
        original_create = container.agent_service.create

        async def observed_execute(*args: Any, **kwargs: Any) -> Any:
            result = await original_execute(*args, **kwargs)
            execution_results.append(result)
            return result

        async def observed_create(*args: Any, **kwargs: Any) -> Any:
            nonlocal handler_calls
            handler_calls += 1
            return await original_create(*args, **kwargs)

        monkeypatch.setattr(
            container.command_executor,
            "execute",
            observed_execute,
        )
        monkeypatch.setattr(
            container.agent_service,
            "create",
            observed_create,
        )
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": replay_key,
        }
        first_command = client.post(
            "/v1/agents",
            headers=headers,
            content=f'{{"name":"{replay_name}"}}'.encode(),
        )
        replayed_command = client.post(
            "/v1/agents",
            headers=headers,
            content=f'{{ "name" : "{replay_name}" }}'.encode(),
        )

        assert first_command.status_code == replayed_command.status_code == 201
        assert first_command.content == replayed_command.content
        assert first_command.headers["location"] == (
            replayed_command.headers["location"]
        )
        assert [result.replayed for result in execution_results] == [
            False,
            True,
        ]
        assert handler_calls == 1
    assert application.state.ready is False
    assert api_database.is_file()
    expected_request_hash = CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=replay_key,
        method="POST",
        route_template="/v1/agents",
        body={"name": replay_name},
    ).request_hash
    with closing(sqlite3.connect(api_database)) as connection:
        receipt = connection.execute(
            "SELECT request_hash, state, response_status_code, response_body "
            "FROM idempotency_records WHERE idempotency_key = ?",
            (replay_key,),
        ).fetchone()
        assert receipt == (
            expected_request_hash,
            "completed",
            201,
            first_command.content,
        )
        assert connection.execute(
            "SELECT count(*) FROM agents WHERE name = ?",
            (replay_name,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM domain_events "
            "WHERE event_type = 'agent.created'",
        ).fetchone() == (1,)

    verified = _canonical_cli_document(
        RUNNER.invoke(
            app,
            [
                "projections",
                "rebuild",
                "--verify",
                "--database",
                str(demo_database),
            ],
        )
    )
    assert verified["data"]["matches"] is True

    with closing(sqlite3.connect(demo_database)) as connection:
        experience_id = connection.execute(
            "SELECT experience_id FROM experience_state "
            "ORDER BY experience_id LIMIT 1"
        ).fetchone()
        assert experience_id is not None
        connection.execute(
            "UPDATE experience_state "
            "SET confidence = CASE confidence WHEN 0.123 THEN 0.124 ELSE 0.123 END "
            "WHERE experience_id = ?",
            (experience_id[0],),
        )
        connection.commit()

    mismatch = RUNNER.invoke(
        app,
        [
            "projections",
            "rebuild",
            "--verify",
            "--database",
            str(demo_database),
        ],
    )
    assert mismatch.exit_code == 1
    mismatch_document = cast(dict[str, Any], json.loads(mismatch.stdout))
    assert mismatch_document["error"]["code"] == "projection_mismatch"

    repaired = _canonical_cli_document(
        RUNNER.invoke(
            app,
            [
                "projections",
                "rebuild",
                "--repair",
                "--database",
                str(demo_database),
            ],
        )
    )
    assert repaired["data"]["matches"] is True
    assert repaired["data"]["mode"] == "repair"

    post_repair = _canonical_cli_document(
        RUNNER.invoke(
            app,
            [
                "projections",
                "rebuild",
                "--verify",
                "--database",
                str(demo_database),
            ],
        )
    )
    assert post_repair["data"]["matches"] is True
    assert post_repair["data"]["differences"] == []

    reconciled = _canonical_cli_document(
        RUNNER.invoke(
            app,
            [
                "payloads",
                "reconcile",
                "--database",
                str(demo_database),
            ],
        )
    )
    reconcile_data = cast(dict[str, Any], reconciled["data"])
    assert reconcile_data["error_count"] == 0
    assert reconcile_data["errors"] == []

def _unused_loopback_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def test_serve_health_acceptance(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("experience-hub")
    assert executable.is_file()
    port = _unused_loopback_port()
    database = tmp_path / "served.sqlite3"
    process = subprocess.Popen(
        [
            str(executable),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database",
            str(database),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready_response: httpx.Response | None = None
    stdout = ""
    stderr = ""
    forced_kill = False
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and process.poll() is None:
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/health",
                    timeout=0.25,
                )
            except httpx.HTTPError:
                time.sleep(0.05)
                continue
            if response.status_code == 200:
                ready_response = response
                break
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            forced_kill = True
            process.kill()
            stdout, stderr = process.communicate(timeout=5)

    assert ready_response is not None, (
        f"serve never became ready\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert ready_response.json()["data"]["status"] == "ready"
    assert forced_kill is False
    # Uvicorn restores and re-raises the original signal after its graceful
    # shutdown sequence, so Popen may expose the terminating SIGTERM.
    assert process.returncode in (0, -signal.SIGTERM), (
        f"serve did not stop cleanly\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "Application shutdown complete." in stderr
    assert database.is_file()
