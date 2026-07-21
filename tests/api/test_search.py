from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

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


def _seed(client: TestClient) -> tuple[UUID, UUID]:
    agent = client.post(
        "/v1/agents",
        headers={"Idempotency-Key": "search-agent"},
        json={"name": "Search owner"},
    )
    assert agent.status_code == 201
    agent_id = UUID(agent.json()["data"]["agent_id"])
    experience = client.post(
        f"/v1/agents/{agent_id}/experiences",
        headers={"Idempotency-Key": "search-experience"},
        json={
            "body": "A durable receipt prevents duplicated retry side effects.",
            "confidence": 0.8,
            "importance": 0.7,
            "kind": "procedural",
            "mechanism": "Canonical request hashes bind retries to one receipt.",
            "summary": "Durable receipts make retries safe.",
            "tags": ["idempotency", "retry"],
        },
    )
    assert experience.status_code == 201, experience.text
    return agent_id, UUID(experience.json()["data"]["experience_id"])


def _search_body(**updates: object) -> dict[str, object]:
    body: dict[str, object] = {
        "content_budget_bytes": 65_536,
        "expand_cold": True,
        "limit": 10,
        "mechanism_cues": ["canonical request hash"],
        "mode": "focused",
        "query": "durable receipt retry",
        "tags": ["idempotency"],
    }
    body.update(updates)
    return body


def _event_count(database_path: Path, event_type: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            (event_type,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _receipt_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone()
    assert row is not None
    return int(row[0])


def test_search_returns_final_scoring_evidence_and_remaining_budget(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "search-result.sqlite3"
    with _client(database_path) as client:
        agent_id, experience_id = _seed(client)
        response = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            json=_search_body(),
        )

    assert response.status_code == 200
    result = response.json()["data"]
    assert isinstance(result["remaining_content_budget_bytes"], int)
    assert len(result["hits"]) == 1
    hit = result["hits"][0]
    assert hit["experience"]["experience_id"] == str(experience_id)
    assert hit["experience"]["owner_agent_id"] == str(agent_id)
    assert hit["experience"]["body"].startswith("A durable receipt")
    assert hit["expanded"] is True
    assert hit["reactivated"] is False
    for field in (
        "score",
        "ranking_relevance",
        "lexical_or_trigram_relevance",
        "mechanism_relevance",
        "activation",
    ):
        assert 0.0 <= hit[field] <= 1.0
    for field in ("confidence", "importance", "source_trust"):
        assert 0.0 <= hit["experience"][field] <= 1.0


def test_search_with_zero_content_budget_returns_a_blurred_hit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "search-zero-budget.sqlite3"
    with _client(database_path) as client:
        agent_id, _ = _seed(client)
        response = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            json=_search_body(content_budget_bytes=0),
        )

    assert response.status_code == 200
    result = response.json()["data"]
    assert result["remaining_content_budget_bytes"] == 0
    hit = result["hits"][0]
    assert hit["experience"]["blurred"] is True
    assert hit["experience"]["body"] is None
    assert hit["expanded"] is False


def test_search_optional_key_replays_without_another_access_event(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "search-replay.sqlite3"
    with _client(database_path) as client:
        agent_id, _ = _seed(client)
        first = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            headers={"Idempotency-Key": "search-once"},
            json=_search_body(),
        )
        replay = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            headers={"Idempotency-Key": "search-once"},
            json=_search_body(),
        )

    assert first.status_code == replay.status_code == 200
    assert replay.content == first.content
    assert _event_count(database_path, "experience.accessed") == 1


def test_search_under_a_missing_agent_is_not_an_empty_success(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "search-missing-agent.sqlite3"
    with _client(database_path) as client:
        first = client.post(
            f"/v1/agents/{uuid4()}/experiences:search",
            headers={"Idempotency-Key": "missing-agent-search"},
            json=_search_body(),
        )
        replay = client.post(
            first.request.url.path,
            headers={"Idempotency-Key": "missing-agent-search"},
            json=_search_body(),
        )

    assert first.status_code == replay.status_code == 404
    assert replay.content == first.content
    assert first.json()["error"] == {
        "code": "agent_not_found",
        "details": {},
        "message": "Agent was not found",
    }
    assert _receipt_count(database_path) == 1


@pytest.mark.parametrize(
    "payload",
    [
        _search_body(peek=True),
        _search_body(query="q" * 2_001),
        _search_body(limit=0),
        _search_body(limit=51),
        _search_body(content_budget_bytes=-1),
        _search_body(content_budget_bytes=65_537),
        _search_body(expand_cold=1),
        _search_body(tags=[f"tag-{index}" for index in range(33)]),
    ],
)
def test_search_rejects_private_or_out_of_range_fields(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    database_path = tmp_path / "search-invalid.sqlite3"
    with _client(database_path) as client:
        agent_id, _ = _seed(client)
        response = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            headers={"Idempotency-Key": "invalid-search"},
            json=payload,
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _event_count(database_path, "experience.accessed") == 0


@pytest.mark.parametrize("key", ("", " ", "x" * 129))
def test_optional_search_key_is_strict_when_supplied(
    tmp_path: Path,
    key: str,
) -> None:
    database_path = tmp_path / "search-invalid-key.sqlite3"
    with _client(database_path) as client:
        agent_id, _ = _seed(client)
        response = client.post(
            f"/v1/agents/{agent_id}/experiences:search",
            headers={"Idempotency-Key": key},
            json=_search_body(),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _event_count(database_path, "experience.accessed") == 0
