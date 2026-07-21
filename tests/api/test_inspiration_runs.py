from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest
from experience_hub.inspiration import (
    INSPIRATION_OPERATOR_ORDER,
    GeneratorKind,
    InspirationOperator,
    InspirationRunStatus,
    StartInspirationRun,
)
from experience_hub.retrieval import RetrievalMode
from experience_hub.storage.idempotency import StoredResponse

NOW = datetime(2026, 7, 20, 8, 30, tzinfo=UTC)


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


@contextmanager
def _client(database_path: Path) -> Iterator[TestClient]:
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    with TestClient(app) as client:
        yield client


def _count(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _create_agent(client: TestClient, *, name: str, key: str) -> UUID:
    response = client.post(
        "/v1/agents",
        headers={"Idempotency-Key": key},
        json={"name": name},
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["agent_id"])


def _create_evidence(client: TestClient, *, owner_id: UUID) -> None:
    response = client.post(
        f"/v1/agents/{owner_id}/experiences",
        headers={"Idempotency-Key": "inspiration-evidence"},
        json={
            "applicability": ["incident response"],
            "body": "Durable receipts bind retries to one causal operation.",
            "confidence": 0.8,
            "evidence": [{"id": "run-trace", "type": "test"}],
            "falsifiers": ["A retry creates a second causal event."],
            "importance": 0.8,
            "kind": "procedural",
            "mechanism": "A canonical request hash selects one durable receipt.",
            "summary": "Durable receipts make command retries safe.",
            "tags": ["idempotency", "recovery"],
        },
    )
    assert response.status_code == 201, response.text


def test_inspiration_run_routes_have_locked_methods_and_statuses() -> None:
    paths = create_app().openapi()["paths"]
    expected = {
        "/v1/agents/{agent_id}/inspiration-runs": ("post", "201"),
        "/v1/agents/{agent_id}/inspiration-runs/{run_id}": ("get", "200"),
        "/v1/agents/{agent_id}/inspiration-runs/{run_id}/ideas": (
            "get",
            "200",
        ),
    }

    for path, (method, success_status) in expected.items():
        assert path in paths
        assert set(paths[path]) == {method}
        assert success_status in paths[path][method]["responses"]


def test_deterministic_run_is_terminal_queryable_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-run.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            name="Inspiration owner",
            key="inspiration-owner",
        )
        _create_evidence(client, owner_id=owner_id)
        body = {
            "goal": "Find retry and recovery improvements.",
            "mode": "associative",
        }
        created = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "start-inspiration"},
            json=body,
        )
        replay = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "start-inspiration"},
            json=body,
        )
        conflict = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "start-inspiration"},
            json={**body, "goal": "A different inspiration goal."},
        )
        assert created.status_code == 201, created.text
        run = created.json()["data"]
        run_id = UUID(run["run_id"])
        fetched = client.get(f"/v1/agents/{owner_id}/inspiration-runs/{run_id}")

    assert replay.status_code == created.status_code
    assert replay.content == created.content
    assert replay.headers["location"] == created.headers["location"]
    assert created.content == canonical_json_bytes({"data": run})
    assert created.headers["location"] == (
        f"/v1/agents/{owner_id}/inspiration-runs/{run_id}"
    )
    assert run["owner_agent_id"] == str(owner_id)
    assert run["goal"] == body["goal"]
    assert run["context"] == ""
    assert run["generator"] == "deterministic"
    assert run["operators"] == [value.value for value in INSPIRATION_OPERATOR_ORDER]
    assert run["include_inbox"] is False
    assert run["branches_per_operator"] == 3
    assert run["output_tokens_per_operator"] == 1_200
    assert run["total_output_tokens"] == 3_600
    assert run["operator_timeout_seconds"] == 30
    assert run["global_timeout_seconds"] == 90
    assert run["status"] in {
        InspirationRunStatus.COMPLETED,
        InspirationRunStatus.COMPLETED_WITH_ERRORS,
        InspirationRunStatus.FAILED,
        InspirationRunStatus.TIMED_OUT,
    }
    assert len(run["request_hash"]) == 64
    assert fetched.status_code == 200
    assert fetched.content == canonical_json_bytes({"data": run})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert _count(database_path, "inspiration_runs") == 1


def test_start_route_calls_only_the_split_transaction_executor_and_keeps_bytes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-spy.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )

    class SpyRunExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[CommandRequest, StartInspirationRun]] = []

        async def execute(
            self,
            *,
            request: CommandRequest,
            run: StartInspirationRun,
        ) -> StoredResponse:
            self.calls.append((request, run))
            return StoredResponse(
                status_code=201,
                body=b'{"data":{"sentinel":"stored bytes"}}',
                headers={
                    "location": "/v1/sentinel-run",
                    "retry-after": "7",
                },
            )

    class ForbiddenCommandExecutor:
        async def execute(self, *_: Any, **__: Any) -> None:
            raise AssertionError("ordinary CommandExecutor must not run")

    with TestClient(app) as client:
        owner_id = _create_agent(
            client,
            name="Executor spy owner",
            key="executor-spy-owner",
        )
        spy = SpyRunExecutor()
        app.state.container.inspiration_run_executor = spy
        app.state.container.command_executor = ForbiddenCommandExecutor()
        response = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "executor-spy-run"},
            json={
                "branches_per_operator": 2,
                "context": "  Bounded generation context.  ",
                "generator": "deterministic",
                "global_timeout_seconds": 20,
                "goal": "  Explore bounded generation.  ",
                "include_inbox": True,
                "mode": "focused",
                "operator_timeout_seconds": 10,
                "operators": [
                    "distant_analogy",
                    "causal_gap",
                ],
                "output_tokens_per_operator": 500,
                "total_output_tokens": 900,
            },
        )

    assert response.status_code == 201
    assert response.content == b'{"data":{"sentinel":"stored bytes"}}'
    assert response.headers["location"] == "/v1/sentinel-run"
    assert response.headers["retry-after"] == "7"
    assert len(spy.calls) == 1
    request, run = spy.calls[0]
    assert run == StartInspirationRun(
        owner_agent_id=owner_id,
        goal="Explore bounded generation.",
        context="Bounded generation context.",
        mode=RetrievalMode.FOCUSED,
        generator=GeneratorKind.DETERMINISTIC,
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.DISTANT_ANALOGY,
        ),
        include_inbox=True,
        branches_per_operator=2,
        output_tokens_per_operator=500,
        total_output_tokens=900,
        operator_timeout_seconds=10,
        global_timeout_seconds=20,
    )
    assert request == CommandRequest(
        caller_scope=f"agent:{owner_id}",
        operation_scope="inspiration.run.start",
        idempotency_key="executor-spy-run",
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": owner_id},
        body={
            "goal": run.goal,
            "context": run.context,
            "mode": run.mode.value,
            "generator": run.generator.value,
            "operators": tuple(value.value for value in run.operators),
            "include_inbox": run.include_inbox,
            "branches_per_operator": run.branches_per_operator,
            "output_tokens_per_operator": run.output_tokens_per_operator,
            "total_output_tokens": run.total_output_tokens,
            "operator_timeout_seconds": run.operator_timeout_seconds,
            "global_timeout_seconds": run.global_timeout_seconds,
        },
    )


@pytest.mark.parametrize(
    "status",
    (
        InspirationRunStatus.COMPLETED,
        InspirationRunStatus.COMPLETED_WITH_ERRORS,
        InspirationRunStatus.FAILED,
        InspirationRunStatus.TIMED_OUT,
    ),
)
def test_start_route_preserves_every_terminal_status_from_the_run_executor(
    tmp_path: Path,
    status: InspirationRunStatus,
) -> None:
    database_path = tmp_path / f"inspiration-terminal-{status.value}.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    run_id = uuid4()
    stored_body = canonical_json_bytes(
        {
            "data": {
                "run_id": run_id,
                "status": status,
            }
        }
    )

    class TerminalRunExecutor:
        calls = 0

        async def execute(
            self,
            *,
            request: CommandRequest,
            run: StartInspirationRun,
        ) -> StoredResponse:
            _ = (request, run)
            self.calls += 1
            return StoredResponse(
                status_code=201,
                body=stored_body,
                headers={
                    "location": (
                        f"/v1/agents/{run.owner_agent_id}/inspiration-runs/{run_id}"
                    )
                },
            )

    with TestClient(app) as client:
        owner_id = _create_agent(
            client,
            name=f"{status.value} owner",
            key=f"{status.value}-owner",
        )
        executor = TerminalRunExecutor()
        app.state.container.inspiration_run_executor = executor
        response = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": f"{status.value}-run"},
            json={"goal": f"Return a {status.value} run."},
        )

    assert response.status_code == 201
    assert response.content == stored_body
    assert response.json()["data"]["status"] == status.value
    assert executor.calls == 1


def test_unconfigured_selected_provider_is_stored_without_a_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-provider.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            name="Provider owner",
            key="provider-owner",
        )
        baseline_receipts = _count(database_path, "idempotency_records")
        body = {
            "generator": "openai_compatible",
            "goal": "Use an explicitly selected provider.",
        }
        first = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "provider-run"},
            json=body,
        )
        replay = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "provider-run"},
            json=body,
        )

    assert first.status_code == replay.status_code == 422
    assert first.content == replay.content
    UUID(first.headers["x-request-id"])
    UUID(replay.headers["x-request-id"])
    assert first.json() == {
        "error": {
            "code": "generator_not_configured",
            "details": {},
            "message": "The selected inspiration generator is not configured.",
        }
    }
    assert _count(database_path, "idempotency_records") == baseline_receipts + 1
    assert _count(database_path, "inspiration_runs") == 0


def test_lifespan_is_ready_without_optional_provider_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "EXPERIENCE_HUB_OPENAI_COMPATIBLE_BASE_URL",
        "EXPERIENCE_HUB_OPENAI_COMPATIBLE_MODEL",
        "EXPERIENCE_HUB_OPENAI_COMPATIBLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    database_path = tmp_path / "inspiration-no-provider-environment.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        owner_id = _create_agent(
            client,
            name="No provider environment owner",
            key="no-provider-environment-owner",
        )
        _create_evidence(client, owner_id=owner_id)
        created = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "no-provider-environment-run"},
            json={
                "goal": "Run deterministically without optional provider settings.",
                "operators": ["counterfactual"],
                "branches_per_operator": 1,
            },
        )

    assert created.status_code == 201
    assert created.json()["data"]["status"] == "completed"


def test_run_reads_hide_foreign_and_missing_resources_identically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-run-owner.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, name="Owner", key="run-owner")
        outsider_id = _create_agent(client, name="Outsider", key="run-outsider")
        created = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "owner-run"},
            json={"goal": "Create one private run."},
        )
        assert created.status_code == 201, created.text
        run_id = UUID(created.json()["data"]["run_id"])
        foreign = client.get(f"/v1/agents/{outsider_id}/inspiration-runs/{run_id}")
        missing = client.get(f"/v1/agents/{outsider_id}/inspiration-runs/{uuid4()}")
        receipts = _count(database_path, "idempotency_records")

    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json()["error"]["code"] == "resource_not_found"
    assert _count(database_path, "idempotency_records") == receipts


def test_foreign_run_reads_hide_corrupt_projection_before_validating_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-run-corrupt-owner.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, name="Owner", key="corrupt-run-owner")
        outsider_id = _create_agent(
            client,
            name="Outsider",
            key="corrupt-run-outsider",
        )
        created = client.post(
            f"/v1/agents/{owner_id}/inspiration-runs",
            headers={"Idempotency-Key": "corrupt-owner-run"},
            json={"goal": "Create a run whose projection will be removed."},
        )
        assert created.status_code == 201, created.text
        run_id = UUID(created.json()["data"]["run_id"])
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "DELETE FROM inspiration_run_state WHERE run_id = ?",
                (str(run_id),),
            )

        foreign_run = client.get(f"/v1/agents/{outsider_id}/inspiration-runs/{run_id}")
        foreign_ideas = client.get(
            f"/v1/agents/{outsider_id}/inspiration-runs/{run_id}/ideas"
        )

    assert foreign_run.status_code == foreign_ideas.status_code == 404
    assert foreign_run.content == foreign_ideas.content
    assert foreign_run.json()["error"]["code"] == "resource_not_found"
