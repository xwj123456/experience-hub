from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)


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


def _receipt_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.parametrize("key", (None, "", " \t ", "k" * 129))
def test_required_key_shapes_fail_before_agent_mutation(
    tmp_path: Path,
    key: str | None,
) -> None:
    database_path = tmp_path / "required-key-shape.sqlite3"
    headers = {} if key is None else {"Idempotency-Key": key}
    with _client(database_path) as client:
        response = client.post(
            "/v1/agents",
            headers=headers,
            json={"name": "Alice"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0


def test_same_key_with_different_canonical_request_conflicts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idempotency-conflict.sqlite3"
    with _client(database_path) as client:
        first = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "same-key"},
            json={"name": "Alice"},
        )
        conflict = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "same-key"},
            json={"name": "Bob"},
        )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json() == {
        "error": {
            "code": "idempotency_key_conflict",
            "details": {},
            "message": ("The idempotency key was already used for a different request"),
        }
    }
    assert _receipt_count(database_path) == 1


def test_surrounding_key_whitespace_normalizes_to_the_foundation_key(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idempotency-normalized-key.sqlite3"
    with _client(database_path) as client:
        first = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "  normalized-key\t"},
            json={"name": "Alice"},
        )
        replay = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "normalized-key"},
            json={"name": "Alice"},
        )

    assert first.status_code == replay.status_code == 201
    assert replay.content == first.content
    assert replay.headers["location"] == first.headers["location"]
    assert _receipt_count(database_path) == 1


def test_duplicate_idempotency_headers_are_rejected_as_ambiguous(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idempotency-duplicate-headers.sqlite3"
    with _client(database_path) as client:
        response = client.post(
            "/v1/agents",
            headers=[
                ("Idempotency-Key", "key-a"),
                ("Idempotency-Key", "key-b"),
            ],
            json={"name": "Alice"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agents").fetchone() == (0,)


def test_same_key_is_isolated_by_agent_caller_scope(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idempotency-caller-scope.sqlite3"
    with _client(database_path) as client:
        alice = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "alice"},
            json={"name": "Alice"},
        )
        bob = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "bob"},
            json={"name": "Bob"},
        )
        alice_id = UUID(alice.json()["data"]["agent_id"])
        bob_id = UUID(bob.json()["data"]["agent_id"])
        payload = {
            "body": "Caller scope isolates an idempotency key.",
            "kind": "semantic",
            "mechanism": "The owner agent is part of receipt identity.",
            "summary": "Keys are owner scoped.",
        }
        first = client.post(
            f"/v1/agents/{alice_id}/experiences",
            headers={"Idempotency-Key": "shared-key"},
            json=payload,
        )
        second = client.post(
            f"/v1/agents/{bob_id}/experiences",
            headers={"Idempotency-Key": "shared-key"},
            json=payload,
        )

    assert first.status_code == second.status_code == 201
    assert first.json()["data"]["owner_agent_id"] == str(alice_id)
    assert second.json()["data"]["owner_agent_id"] == str(bob_id)
    assert _receipt_count(database_path) == 4


def test_locked_task_four_route_table_and_success_statuses() -> None:
    schema = create_app().openapi()
    expected = {
        "/health": {"get": "200"},
        "/v1/agents": {"get": "200", "post": "201"},
        "/v1/agents/{agent_id}/experiences": {"post": "201"},
        "/v1/agents/{agent_id}/experiences/{experience_id}/versions": {"post": "201"},
        "/v1/agents/{agent_id}/experiences/{experience_id}": {"get": "200"},
        "/v1/agents/{agent_id}/experiences:search": {"post": "200"},
        "/v1/agents/{agent_id}/experiences/{experience_id}:confirm": {"post": "200"},
        "/v1/agents/{agent_id}/experiences/{experience_id}:refute": {"post": "200"},
        "/v1/agents/{agent_id}/experiences/{experience_id}:pin": {"post": "200"},
        "/v1/agents/{agent_id}/experiences/{experience_id}:unpin": {"post": "200"},
        "/v1/agents/{agent_id}/experiences/{experience_id}:restore": {"post": "200"},
        "/v1/lifecycle:run": {"post": "200"},
        "/v1/topics": {"post": "201"},
        "/v1/agents/{agent_id}/subscriptions": {"post": "201"},
        "/v1/agents/{agent_id}/capsules": {"post": "201"},
        "/v1/agents/{agent_id}/capsules/{capsule_id}:retract": {"post": "200"},
        "/v1/agents/{agent_id}/inbox": {"get": "200"},
        "/v1/agents/{agent_id}/inbox/{item_id}:adopt": {"post": "200"},
        "/v1/agents/{agent_id}/inbox/{item_id}:reject": {"post": "200"},
        "/v1/agents/{agent_id}/capsules/{capsule_id}:feedback": {"post": "201"},
        "/v1/agents/{agent_id}/inspiration-runs": {"post": "201"},
        "/v1/agents/{agent_id}/inspiration-runs/{run_id}": {"get": "200"},
        "/v1/agents/{agent_id}/inspiration-runs/{run_id}/ideas": {"get": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:adopt": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:reject": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:evaluate": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:archive": {"post": "200"},
    }
    domain_paths = {
        path: operations
        for path, operations in schema["paths"].items()
        if path == "/health" or path.startswith("/v1")
    }

    assert set(domain_paths) == set(expected)
    for path, methods in expected.items():
        assert set(domain_paths[path]) == set(methods)
        for method, success_status in methods.items():
            assert success_status in domain_paths[path][method]["responses"]


def test_all_twenty_one_external_commands_require_a_key_before_execution(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "all-required-keys.sqlite3"
    agent_id = UUID("00000000-0000-0000-0000-000000000101")
    experience_id = UUID("00000000-0000-0000-0000-000000000201")
    content = {
        "body": "Required keys are checked before command execution.",
        "kind": "semantic",
        "mechanism": "Header validation precedes receipt reservation.",
        "summary": "Every external command requires a key.",
    }
    version = dict(content)
    version.pop("kind")

    with _client(database_path) as client:
        responses = [
            client.post("/v1/agents", json={"name": "Alice"}),
            client.post(
                f"/v1/agents/{agent_id}/experiences",
                json=content,
            ),
            client.post(
                (f"/v1/agents/{agent_id}/experiences/{experience_id}/versions"),
                json=version,
            ),
            *[
                client.post(
                    (f"/v1/agents/{agent_id}/experiences/{experience_id}:{suffix}"),
                    json={},
                )
                for suffix in ("confirm", "refute", "pin", "unpin", "restore")
            ],
            client.post("/v1/lifecycle:run", json={}),
            client.post(
                "/v1/topics",
                json={
                    "owner_agent_id": str(agent_id),
                    "name": "Required key topic",
                },
            ),
            client.post(
                f"/v1/agents/{agent_id}/subscriptions",
                json={"topic_id": str(experience_id)},
            ),
            client.post(
                f"/v1/agents/{agent_id}/capsules",
                json={
                    "topic_id": str(experience_id),
                    "experience_id": str(experience_id),
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                },
            ),
            client.post(
                f"/v1/agents/{agent_id}/capsules/{experience_id}:retract",
                json={"reason": "No longer suitable for circulation."},
            ),
            client.post(
                f"/v1/agents/{agent_id}/inbox/{experience_id}:adopt",
                json={},
            ),
            client.post(
                f"/v1/agents/{agent_id}/inbox/{experience_id}:reject",
                json={"reason": "The evidence is not sufficient."},
            ),
            client.post(
                f"/v1/agents/{agent_id}/capsules/{experience_id}:feedback",
                json={
                    "verdict": "useful",
                    "reason": "The capsule worked in the observed setting.",
                },
            ),
            client.post(
                f"/v1/agents/{agent_id}/inspiration-runs",
                json={"goal": "Find a safer migration strategy."},
            ),
            client.post(
                f"/v1/agents/{agent_id}/ideas/{experience_id}:adopt",
                json={},
            ),
            client.post(
                f"/v1/agents/{agent_id}/ideas/{experience_id}:reject",
                json={"reason": "The idea lacks sufficient evidence."},
            ),
            client.post(
                f"/v1/agents/{agent_id}/ideas/{experience_id}:evaluate",
                json={
                    "verdict": "supported",
                    "evidence": [
                        {
                            "type": "experience_version",
                            "id": str(experience_id),
                        }
                    ],
                    "evaluated_at": NOW.isoformat(),
                },
            ),
            client.post(
                f"/v1/agents/{agent_id}/ideas/{experience_id}:archive",
                json={"reason": "The idea is no longer relevant."},
            ),
        ]

    assert len(responses) == 21
    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _receipt_count(database_path) == 0


@pytest.mark.parametrize("key", ("", " ", "x" * 129))
@pytest.mark.parametrize(
    ("path", "body"),
    (
        (
            "/v1/topics",
            {
                "owner_agent_id": "00000000-0000-0000-0000-000000000101",
                "name": "Strict sharing keys",
            },
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/subscriptions",
            {"topic_id": "00000000-0000-0000-0000-000000000201"},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/capsules",
            {
                "topic_id": "00000000-0000-0000-0000-000000000201",
                "experience_id": "00000000-0000-0000-0000-000000000202",
                "expires_at": "2026-07-20T08:30:00Z",
            },
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "capsules/00000000-0000-0000-0000-000000000201:retract",
            {"reason": "Retraction requires a strict idempotency key."},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "inbox/00000000-0000-0000-0000-000000000201:adopt",
            {},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "inbox/00000000-0000-0000-0000-000000000201:reject",
            {"reason": "Rejection requires a strict idempotency key."},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "capsules/00000000-0000-0000-0000-000000000201:feedback",
            {
                "verdict": "useful",
                "reason": "Feedback requires a strict idempotency key.",
            },
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/inspiration-runs",
            {"goal": "Strict run keys"},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "ideas/00000000-0000-0000-0000-000000000201:adopt",
            {},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "ideas/00000000-0000-0000-0000-000000000201:reject",
            {"reason": "Rejection requires a strict idempotency key."},
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "ideas/00000000-0000-0000-0000-000000000201:evaluate",
            {
                "verdict": "supported",
                "evidence": [
                    {
                        "type": "experience_version",
                        "id": "00000000-0000-0000-0000-000000000201",
                    }
                ],
                "evaluated_at": "2026-07-19T08:30:00Z",
            },
        ),
        (
            "/v1/agents/00000000-0000-0000-0000-000000000101/"
            "ideas/00000000-0000-0000-0000-000000000201:archive",
            {"reason": "Archival requires a strict idempotency key."},
        ),
    ),
)
def test_every_later_command_rejects_invalid_key_shapes_before_execution(
    tmp_path: Path,
    key: str,
    path: str,
    body: dict[str, object],
) -> None:
    database_path = tmp_path / "later-invalid-key-shape.sqlite3"
    with _client(database_path) as client:
        response = client.post(
            path,
            headers={"Idempotency-Key": key},
            json=body,
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0


@pytest.mark.parametrize("key", ("", " ", "x" * 129))
def test_optional_direct_get_key_is_strict_when_supplied(
    tmp_path: Path,
    key: str,
) -> None:
    database_path = tmp_path / "optional-get-invalid-key.sqlite3"
    with _client(database_path) as client:
        response = client.get(
            (
                "/v1/agents/00000000-0000-0000-0000-000000000101/"
                "experiences/00000000-0000-0000-0000-000000000201"
            ),
            headers={"Idempotency-Key": key},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0
