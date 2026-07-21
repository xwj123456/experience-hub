from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

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


def _table_count(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _event_count(database_path: Path, event_type: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            (event_type,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _create_agent(client: TestClient, *, key: str, name: str) -> UUID:
    response = client.post(
        "/v1/agents",
        headers={"Idempotency-Key": key},
        json={"name": name},
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["agent_id"])


def test_topic_and_subscription_routes_have_locked_methods_and_statuses() -> None:
    paths = create_app().openapi()["paths"]
    expected = {
        "/v1/topics": ("post", "201"),
        "/v1/agents/{agent_id}/subscriptions": ("post", "201"),
    }

    for path, (method, success_status) in expected.items():
        assert path in paths
        assert set(paths[path]) == {method}
        assert success_status in paths[path][method]["responses"]


def test_create_topic_and_subscription_return_exact_resources(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "topic-subscription-create.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, key="topic-owner", name="Owner")
        subscriber_id = _create_agent(
            client,
            key="topic-subscriber",
            name="Subscriber",
        )
        topic_response = client.post(
            "/v1/topics",
            headers={"Idempotency-Key": "topic-create"},
            json={
                "owner_agent_id": str(owner_id),
                "name": "  Operational Memory  ",
                "description": "  Durable incident experience.  ",
            },
        )
        topic_data = topic_response.json()["data"]
        topic_id = UUID(topic_data["topic_id"])
        subscription_response = client.post(
            f"/v1/agents/{subscriber_id}/subscriptions",
            headers={"Idempotency-Key": "subscription-create"},
            json={"topic_id": str(topic_id)},
        )
        subscription_data = subscription_response.json()["data"]

    expected_topic = {
        "topic_id": str(topic_id),
        "owner_agent_id": str(owner_id),
        "name": "Operational Memory",
        "description": "Durable incident experience.",
        "created_at": "2026-07-19T08:30:00.000000Z",
    }
    assert topic_response.status_code == 201
    assert topic_response.content == canonical_json_bytes({"data": expected_topic})
    assert topic_response.headers["location"] == f"/v1/topics/{topic_id}"

    assert subscription_response.status_code == 201
    assert set(subscription_data) == {
        "subscription_id",
        "subscriber_agent_id",
        "topic_id",
        "creation_event_id",
        "created_at",
    }
    assert subscription_data["subscriber_agent_id"] == str(subscriber_id)
    assert subscription_data["topic_id"] == str(topic_id)
    assert (
        isinstance(subscription_data["creation_event_id"], int)
        and subscription_data["creation_event_id"] > 0
    )
    assert subscription_data["created_at"] == "2026-07-19T08:30:00.000000Z"
    assert subscription_response.content == canonical_json_bytes(
        {"data": subscription_data}
    )
    assert subscription_response.headers["location"] == (
        f"/v1/subscriptions/{subscription_data['subscription_id']}"
    )
    assert _event_count(database_path, "topic.created") == 1
    assert _event_count(database_path, "subscription.created") == 1


def test_topic_and_subscription_requests_are_strict_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "topic-subscription-strict.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, key="strict-owner", name="Owner")
        invalid_topics = (
            {"name": "Missing owner"},
            {
                "owner_agent_id": str(owner_id),
                "name": "Unknown field",
                "visibility": "public",
            },
            {"owner_agent_id": str(owner_id), "name": " \t "},
            {"owner_agent_id": str(owner_id), "name": "n" * 201},
            {
                "owner_agent_id": str(owner_id),
                "name": "Blank description",
                "description": " ",
            },
            {
                "owner_agent_id": str(owner_id),
                "name": "Long description",
                "description": "d" * 2_001,
            },
        )
        topic_responses = tuple(
            client.post(
                "/v1/topics",
                headers={"Idempotency-Key": f"invalid-topic-{index}"},
                json=body,
            )
            for index, body in enumerate(invalid_topics)
        )
        subscription_responses = (
            client.post(
                f"/v1/agents/{owner_id}/subscriptions",
                headers={"Idempotency-Key": "missing-topic-field"},
                json={},
            ),
            client.post(
                f"/v1/agents/{owner_id}/subscriptions",
                headers={"Idempotency-Key": "extra-subscription-field"},
                json={"topic_id": str(uuid4()), "backfill": True},
            ),
        )

    responses = (*topic_responses, *subscription_responses)
    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _table_count(database_path, "topics") == 0
    assert _table_count(database_path, "subscriptions") == 0
    assert _table_count(database_path, "idempotency_records") == 1


def test_topic_rejects_invalid_unicode_before_mutation(tmp_path: Path) -> None:
    database_path = tmp_path / "topic-invalid-unicode.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, key="unicode-owner", name="Owner")
        response = client.post(
            "/v1/topics",
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": "invalid-unicode-topic",
            },
            content=(
                b'{"owner_agent_id":"'
                + str(owner_id).encode("ascii")
                + b'","name":"\\ud800"}'
            ),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _table_count(database_path, "topics") == 0
    assert _table_count(database_path, "idempotency_records") == 1


def test_topic_and_subscription_require_idempotency_keys(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "topic-subscription-required-key.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, key="required-key-owner", name="Owner")
        topic = client.post(
            "/v1/topics",
            json={
                "owner_agent_id": str(owner_id),
                "name": "Missing receipt key",
            },
        )
        subscription = client.post(
            f"/v1/agents/{owner_id}/subscriptions",
            json={"topic_id": str(uuid4())},
        )

    assert topic.status_code == subscription.status_code == 422
    assert topic.json()["error"]["code"] == "validation_error"
    assert subscription.json()["error"]["code"] == "validation_error"
    assert _table_count(database_path, "topics") == 0
    assert _table_count(database_path, "subscriptions") == 0
    assert _table_count(database_path, "idempotency_records") == 1


def test_topic_and_subscription_replay_exactly_and_conflict_by_body(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "topic-subscription-idempotency.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(client, key="idempotent-owner", name="Owner")
        subscriber_id = _create_agent(
            client,
            key="idempotent-subscriber",
            name="Subscriber",
        )
        topic_body = {
            "owner_agent_id": str(owner_id),
            "name": "Idempotent Topic",
            "description": "One durable topic.",
        }
        topic_first = client.post(
            "/v1/topics",
            headers={"Idempotency-Key": "same-topic"},
            json=topic_body,
        )
        topic_replay = client.post(
            "/v1/topics",
            headers={"Idempotency-Key": "same-topic"},
            json=topic_body,
        )
        topic_conflict = client.post(
            "/v1/topics",
            headers={"Idempotency-Key": "same-topic"},
            json={**topic_body, "description": "A different command."},
        )
        topic_id = UUID(topic_first.json()["data"]["topic_id"])

        subscription_body = {"topic_id": str(topic_id)}
        subscription_first = client.post(
            f"/v1/agents/{subscriber_id}/subscriptions",
            headers={"Idempotency-Key": "same-subscription"},
            json=subscription_body,
        )
        subscription_replay = client.post(
            f"/v1/agents/{subscriber_id}/subscriptions",
            headers={"Idempotency-Key": "same-subscription"},
            json=subscription_body,
        )
        subscription_conflict = client.post(
            f"/v1/agents/{subscriber_id}/subscriptions",
            headers={"Idempotency-Key": "same-subscription"},
            json={"topic_id": str(uuid4())},
        )

    for first, replay in (
        (topic_first, topic_replay),
        (subscription_first, subscription_replay),
    ):
        assert first.status_code == replay.status_code == 201
        assert replay.content == first.content
        assert replay.headers["location"] == first.headers["location"]
    for conflict in (topic_conflict, subscription_conflict):
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_conflict"

    assert _table_count(database_path, "topics") == 1
    assert _table_count(database_path, "subscriptions") == 1
    assert _event_count(database_path, "topic.created") == 1
    assert _event_count(database_path, "subscription.created") == 1
    assert _table_count(database_path, "idempotency_records") == 4
