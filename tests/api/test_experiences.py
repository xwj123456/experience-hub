from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
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


def _create_agent(client: TestClient, *, name: str = "Alice") -> UUID:
    response = client.post(
        "/v1/agents",
        headers={"Idempotency-Key": f"agent-{name}"},
        json={"name": name},
    )
    assert response.status_code == 201
    return UUID(response.json()["data"]["agent_id"])


def _experience_body(
    *,
    body: str = "Retries must reuse the same command receipt.",
    summary: str = "Retries are receipt scoped.",
    mechanism: str = "A canonical request hash binds one durable receipt.",
) -> dict[str, object]:
    return {
        "applicability": ["network retry"],
        "body": body,
        "confidence": 0.7,
        "evidence": [{"id": "trace-1", "type": "test"}],
        "falsifiers": ["A duplicate event is appended."],
        "importance": 0.6,
        "kind": "semantic",
        "links": [],
        "mechanism": mechanism,
        "summary": summary,
        "tags": ["idempotency", "retries"],
    }


def _create_experience(
    client: TestClient,
    agent_id: UUID,
    *,
    key: str = "experience-create",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    response = client.post(
        f"/v1/agents/{agent_id}/experiences",
        headers={"Idempotency-Key": key},
        json=payload or _experience_body(),
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


def _table_count(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_create_and_version_return_full_current_experience(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-create-version.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        experience_id = UUID(str(created["experience_id"]))

        version = _experience_body(
            body="A replay returns the original response bytes.",
            summary="Replays preserve response bytes.",
            mechanism="The completed receipt stores status, headers, and body.",
        )
        version.pop("kind")
        version.pop("importance")
        version.pop("confidence")
        response = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}/versions",
            headers={"Idempotency-Key": "experience-version"},
            json=version,
        )

    assert created["owner_agent_id"] == str(agent_id)
    assert created["kind"] == "semantic"
    assert created["origin"] == "local"
    assert created["body"] == _experience_body()["body"]
    assert created["blurred"] is False
    assert created["body_is_excerpt"] is False
    assert created["version_number"] == 1
    assert created["access_count"] == 0

    assert response.status_code == 201
    updated = response.json()["data"]
    assert updated["experience_id"] == str(experience_id)
    assert updated["owner_agent_id"] == str(agent_id)
    assert updated["version_number"] == 2
    assert updated["body"] == version["body"]
    assert updated["blurred"] is False
    assert updated["content_hash"] != created["content_hash"]
    assert response.headers["location"] == (
        f"/v1/agents/{agent_id}/experiences/{experience_id}"
    )


def test_direct_get_accepts_an_optional_key_and_replays_one_access(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-get-replay.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        experience_id = created["experience_id"]
        path = f"/v1/agents/{agent_id}/experiences/{experience_id}"

        first = client.get(path, headers={"Idempotency-Key": "get-once"})
        replay = client.get(path, headers={"Idempotency-Key": "get-once"})

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.content == first.content
    assert first.json()["data"]["body"] == _experience_body()["body"]
    assert first.json()["data"]["blurred"] is False
    assert first.json()["data"]["access_count"] == 1
    # agent create + experience create + one retained get
    assert _table_count(database_path, "idempotency_records") == 3
    with sqlite3.connect(database_path) as connection:
        access_events = connection.execute(
            "SELECT COUNT(*) FROM domain_events "
            "WHERE event_type = 'experience.accessed'"
        ).fetchone()
    assert access_events == (1,)


def test_direct_get_without_a_key_is_a_new_retained_access(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-get-no-key.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        path = f"/v1/agents/{agent_id}/experiences/{created['experience_id']}"

        first = client.get(path)
        second = client.get(path)

    assert first.status_code == second.status_code == 200
    assert first.json()["data"]["access_count"] == 1
    assert second.json()["data"]["access_count"] == 2
    assert _table_count(database_path, "idempotency_records") == 4


@pytest.mark.parametrize(
    ("suffix", "payload", "assertion", "event_type"),
    [
        (
            "confirm",
            {
                "reason": "  independently reproduced  ",
                "evidence": [
                    {"type": "test", "id": "trace-b"},
                    {"id": "trace-a", "type": "test"},
                    {"id": "trace-a", "type": "test"},
                ],
            },
            ("temperature", "hot"),
            "experience.confirmed",
        ),
        (
            "refute",
            {
                "reason": "counterexample observed",
                "evidence": [{"id": "counterexample-1", "type": "test"}],
            },
            ("blurred", True),
            "experience.refuted",
        ),
        (
            "pin",
            {"reason": "retain for incident response"},
            ("pinned", True),
            "experience.pinned",
        ),
        (
            "unpin",
            {"reason": "incident resolved"},
            ("pinned", False),
            "experience.unpinned",
        ),
    ],
)
def test_mutations_return_metadata_only(
    tmp_path: Path,
    suffix: str,
    payload: dict[str, object],
    assertion: tuple[str, object],
    event_type: str,
) -> None:
    database_path = tmp_path / f"experience-{suffix}.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        experience_id = created["experience_id"]
        if suffix == "unpin":
            pinned = client.post(
                f"/v1/agents/{agent_id}/experiences/{experience_id}:pin",
                headers={"Idempotency-Key": "pin-first"},
                json={},
            )
            assert pinned.status_code == 200

        response = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:{suffix}",
            headers={"Idempotency-Key": f"mutation-{suffix}"},
            json=payload,
        )
        replay = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:{suffix}",
            headers={"Idempotency-Key": f"mutation-{suffix}"},
            json=payload,
        )
        conflict = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:{suffix}",
            headers={"Idempotency-Key": f"mutation-{suffix}"},
            json={**payload, "reason": "different canonical command"},
        )

    assert response.status_code == 200
    assert replay.status_code == response.status_code
    assert replay.content == response.content
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
    experience = response.json()["data"]
    assert experience[assertion[0]] == assertion[1]
    assert experience["blurred"] is True
    assert experience["body"] is None
    assert experience["body_is_excerpt"] is False
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            (event_type,),
        ).fetchone() == (1,)


def test_restore_active_error_replays_and_conflicting_reason_is_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-restore-replay.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        path = f"/v1/agents/{agent_id}/experiences/{created['experience_id']}:restore"
        first = client.post(
            path,
            headers={"Idempotency-Key": "restore-active"},
            json={"reason": "recover memory"},
        )
        replay = client.post(
            path,
            headers={"Idempotency-Key": "restore-active"},
            json={"reason": "recover memory"},
        )
        conflict = client.post(
            path,
            headers={"Idempotency-Key": "restore-active"},
            json={"reason": "different restore command"},
        )

    assert first.status_code == replay.status_code == 409
    assert replay.content == first.content
    assert first.json()["error"]["code"] == "experience_not_archived"
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert _table_count(database_path, "idempotency_records") == 3


def test_archived_experience_can_be_restored_through_the_public_route(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-restore-archived.sqlite3"
    clock = FrozenClock(NOW)
    app = create_app(settings=_settings(database_path), clock=clock)
    with TestClient(app) as client:
        agent_id = _create_agent(client)
        created = _create_experience(
            client,
            agent_id,
            payload={
                **_experience_body(),
                "confidence": 0.1,
                "importance": 0.1,
            },
        )
        experience_id = created["experience_id"]

        clock.advance(timedelta(days=30))
        first_cycle = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "restore-cycle-1"},
            json={},
        )
        clock.advance(timedelta(minutes=16))
        cold_cycle = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "restore-cycle-2"},
            json={},
        )
        clock.advance(timedelta(days=91))
        archive_cycle = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "restore-cycle-3"},
            json={},
        )
        restored = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:restore",
            headers={"Idempotency-Key": "restore-archived"},
            json={"reason": "context made this experience relevant again"},
        )
        replay = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:restore",
            headers={"Idempotency-Key": "restore-archived"},
            json={"reason": "context made this experience relevant again"},
        )

    assert first_cycle.status_code == cold_cycle.status_code == 200
    assert archive_cycle.status_code == 200
    assert archive_cycle.json()["data"]["archive_count"] == 1
    assert restored.status_code == replay.status_code == 200
    assert replay.content == restored.content
    experience = restored.json()["data"]
    assert experience["temperature"] == "warm"
    assert experience["blurred"] is True
    assert experience["body"] is None
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events "
            "WHERE event_type = 'experience.restored'"
        ).fetchone() == (1,)


def test_cross_agent_and_missing_direct_get_are_indistinguishable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-owner-isolation.sqlite3"
    with _client(database_path) as client:
        alice_id = _create_agent(client, name="Alice")
        bob_id = _create_agent(client, name="Bob")
        created = _create_experience(client, alice_id)
        experience_id = created["experience_id"]

        foreign = client.get(
            f"/v1/agents/{bob_id}/experiences/{experience_id}",
            headers={"Idempotency-Key": "foreign-get"},
        )
        missing = client.get(
            f"/v1/agents/{bob_id}/experiences/{uuid4()}",
            headers={"Idempotency-Key": "missing-get"},
        )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json()["error"]["code"] == "experience_not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {**_experience_body(), "unknown": True},
        {**_experience_body(), "body": "界" * 21_846},
        {**_experience_body(), "summary": "s" * 1_001},
        {**_experience_body(), "mechanism": "m" * 2_001},
        {**_experience_body(), "tags": [f"tag-{index}" for index in range(33)]},
        {**_experience_body(), "importance": True},
        {**_experience_body(), "confidence": "0.7"},
        {
            **_experience_body(),
            "links": [
                {
                    "relation": "supports",
                    "target_experience_id": ("00000000-0000-0000-0000-000000000001"),
                },
                {
                    "relation": "supports",
                    "target_experience_id": ("00000000-0000-0000-0000-000000000001"),
                },
            ],
        },
    ],
)
def test_create_rejects_invalid_transport_values_before_mutation(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    database_path = tmp_path / "experience-invalid.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        response = client.post(
            f"/v1/agents/{agent_id}/experiences",
            headers={"Idempotency-Key": "invalid-experience"},
            json=payload,
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _table_count(database_path, "experiences") == 0
    # Only the prior agent-create receipt exists.
    assert _table_count(database_path, "idempotency_records") == 1


def test_create_under_a_missing_agent_is_a_stable_not_found(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-missing-owner.sqlite3"
    missing_agent = uuid4()
    with _client(database_path) as client:
        response = client.post(
            f"/v1/agents/{missing_agent}/experiences",
            headers={"Idempotency-Key": "missing-owner"},
            json=_experience_body(),
        )

    assert response.status_code == 404
    assert response.content == canonical_json_bytes(
        {
            "error": {
                "code": "agent_not_found",
                "details": {},
                "message": "Agent was not found",
            }
        }
    )


def test_api_command_traces_pass_startup_source_validation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "experience-api-restart.sqlite3"
    with _client(database_path) as client:
        agent_id = _create_agent(client)
        created = _create_experience(client, agent_id)
        experience_id = created["experience_id"]
        searched = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            headers={"Idempotency-Key": "validated-search"},
            json={
                "mode": "focused",
                "query": "durable receipt retry",
            },
        )
        confirmed = client.post(
            f"/v1/agents/{agent_id}/experiences/{experience_id}:confirm",
            headers={"Idempotency-Key": "validated-confirm"},
            json={"reason": "restart validation"},
        )
        lifecycle = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "validated-lifecycle"},
            json={},
        )

    assert searched.status_code == confirmed.status_code == lifecycle.status_code == 200

    # Entering lifespan again re-runs source/reducer validation before readiness.
    with _client(database_path) as restarted:
        health = restarted.get("/health")

    assert health.status_code == 200
    assert health.json()["data"]["status"] == "ready"
