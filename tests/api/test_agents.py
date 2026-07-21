from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.ids import SequenceIdGenerator

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
RECEIPT_IDS = (
    UUID("00000000-0000-0000-0000-000000005001"),
    UUID("00000000-0000-0000-0000-000000005002"),
    UUID("00000000-0000-0000-0000-000000005003"),
)
AGENT_IDS = (
    UUID("00000000-0000-0000-0000-000000006001"),
    UUID("00000000-0000-0000-0000-000000006002"),
    UUID("00000000-0000-0000-0000-000000006003"),
)


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


@contextmanager
def _client(
    database_path: Path,
    *,
    ids: tuple[UUID, ...],
) -> Iterator[TestClient]:
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(ids),
    )
    with TestClient(app) as client:
        yield client


def _resource(agent_id: UUID, name: str) -> dict[str, str]:
    return {"agent_id": str(agent_id), "name": name}


def _database_count(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_create_agent_returns_exact_resource_and_list_exposes_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-create.sqlite3"
    with _client(
        database_path,
        ids=(RECEIPT_IDS[0], AGENT_IDS[0]),
    ) as client:
        created = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "create-alice"},
            json={"name": "  Alice  "},
        )
        listed = client.get("/v1/agents")

    agent = _resource(AGENT_IDS[0], "Alice")
    assert created.status_code == 201
    assert created.content == canonical_json_bytes({"data": agent})
    assert created.headers["location"] == f"/v1/agents/{AGENT_IDS[0]}"
    assert listed.status_code == 200
    assert listed.content == canonical_json_bytes(
        {"data": [agent], "page": {"next_cursor": None}}
    )


def test_create_agent_requires_an_idempotency_key_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-required-key.sqlite3"
    with _client(database_path, ids=()) as client:
        response = client.post("/v1/agents", json={"name": "Alice"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _database_count(database_path, "agents") == 0
    assert _database_count(database_path, "idempotency_records") == 0


def test_create_agent_rejects_blank_names_and_unknown_fields_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-strict-input.sqlite3"
    with _client(database_path, ids=()) as client:
        blank = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "blank-agent"},
            json={"name": " \t "},
        )
        unknown = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "unknown-agent-field"},
            json={"name": "Alice", "role": "administrator"},
        )

    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "validation_error"
    assert _database_count(database_path, "agents") == 0
    assert _database_count(database_path, "idempotency_records") == 0


def test_create_agent_rejects_invalid_unicode_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-invalid-unicode.sqlite3"
    with _client(database_path, ids=()) as client:
        response = client.post(
            "/v1/agents",
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": "invalid-unicode-agent",
            },
            content=b'{"name":"\\ud800"}',
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _database_count(database_path, "agents") == 0
    assert _database_count(database_path, "idempotency_records") == 0
    assert _database_count(database_path, "domain_events") == 0


def test_create_agent_replays_the_exact_stored_response(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-replay.sqlite3"
    with _client(
        database_path,
        ids=(RECEIPT_IDS[0], AGENT_IDS[0]),
    ) as client:
        first = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "create-alice"},
            json={"name": "Alice"},
        )
        replay = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "create-alice"},
            json={"name": "Alice"},
        )

    assert (replay.status_code, replay.content) == (
        first.status_code,
        first.content,
    )
    assert replay.headers["location"] == first.headers["location"]
    assert _database_count(database_path, "agents") == 1
    assert _database_count(database_path, "idempotency_records") == 1
    assert _database_count(database_path, "domain_events") == 1


def test_list_agents_uses_a_stable_opaque_cursor(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "agent-pagination.sqlite3"
    ids = (
        RECEIPT_IDS[0],
        AGENT_IDS[0],
        RECEIPT_IDS[1],
        AGENT_IDS[1],
        RECEIPT_IDS[2],
        AGENT_IDS[2],
    )
    with _client(database_path, ids=ids) as client:
        for index, name in enumerate(("Alice", "Bob", "Carol")):
            response = client.post(
                "/v1/agents",
                headers={"Idempotency-Key": f"create-agent-{index}"},
                json={"name": name},
            )
            assert response.status_code == 201

        first = client.get("/v1/agents", params={"limit": 2})
        first_body = first.json()
        cursor = first_body["page"]["next_cursor"]
        second = client.get(
            "/v1/agents",
            params={"limit": 2, "cursor": cursor},
        )

    assert first.status_code == 200
    assert first_body["data"] == [
        _resource(AGENT_IDS[0], "Alice"),
        _resource(AGENT_IDS[1], "Bob"),
    ]
    assert isinstance(cursor, str) and cursor
    assert "=" not in cursor
    assert second.status_code == 200
    assert second.json() == {
        "data": [_resource(AGENT_IDS[2], "Carol")],
        "page": {"next_cursor": None},
    }


@pytest.mark.parametrize("query", ({"limit": 0}, {"limit": 101}, {"extra": "x"}))
def test_list_agents_rejects_non_contract_query_parameters(
    tmp_path: Path,
    query: dict[str, int | str],
) -> None:
    database_path = tmp_path / "agent-list-validation.sqlite3"
    with _client(database_path, ids=()) as client:
        response = client.get("/v1/agents", params=query)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
