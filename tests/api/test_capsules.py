from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from httpx import Response

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
FUTURE = NOW + timedelta(days=7)

EXPERIENCE_BODY: dict[str, object] = {
    "applicability": ["incident response"],
    "body": "A bounded retry must reuse its durable receipt.",
    "confidence": 0.7,
    "evidence": [{"id": "trace-1", "type": "test"}],
    "falsifiers": ["A replay appends another event."],
    "importance": 0.6,
    "kind": "semantic",
    "links": [],
    "mechanism": "The canonical request identifies one retained result.",
    "summary": "Retries reuse one command result.",
    "tags": ["idempotency", "sharing"],
}

CAPSULE_FIELDS = {
    "capsule_id",
    "transport_schema_version",
    "topic_id",
    "source_experience_id",
    "source_version_id",
    "publisher_agent_id",
    "kind",
    "body",
    "summary",
    "mechanism",
    "tags",
    "applicability",
    "evidence",
    "falsifiers",
    "publisher_confidence",
    "provenance_chain",
    "root_fingerprint",
    "source_content_hash",
    "created_at",
    "expires_at",
    "hop_count",
    "capsule_hash",
    "status",
    "last_transition_at",
}


class BoundaryAdvancingClock:
    """Advance only after a publication preflight observes the boundary."""

    def __init__(self) -> None:
        self._armed = False
        self._calls_after_arm = 0

    def arm(self) -> None:
        self._armed = True

    def now(self) -> datetime:
        if not self._armed:
            return NOW
        self._calls_after_arm += 1
        if self._calls_after_arm == 1:
            return NOW
        return NOW + timedelta(minutes=2)


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


def _create_topic(
    client: TestClient,
    *,
    owner_agent_id: UUID,
    key: str,
    name: str,
) -> UUID:
    response = client.post(
        "/v1/topics",
        headers={"Idempotency-Key": key},
        json={
            "owner_agent_id": str(owner_agent_id),
            "name": name,
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["topic_id"])


def _create_experience(
    client: TestClient,
    *,
    owner_agent_id: UUID,
    key: str,
) -> dict[str, object]:
    response = client.post(
        f"/v1/agents/{owner_agent_id}/experiences",
        headers={"Idempotency-Key": key},
        json=EXPERIENCE_BODY,
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, object], response.json()["data"])


def _publish_body(
    *,
    topic_id: UUID,
    experience_id: UUID,
    expires_at: datetime | str | int | bool = FUTURE,
    version_id: UUID | None = None,
    parent_adoption_id: UUID | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "topic_id": str(topic_id),
        "experience_id": str(experience_id),
        "expires_at": (
            expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
        ),
    }
    if version_id is not None:
        body["version_id"] = str(version_id)
    if parent_adoption_id is not None:
        body["parent_adoption_id"] = str(parent_adoption_id)
    return body


def _publish(
    client: TestClient,
    *,
    publisher_agent_id: UUID,
    key: str,
    body: dict[str, object],
) -> Response:
    return cast(
        Response,
        client.post(
            f"/v1/agents/{publisher_agent_id}/capsules",
            headers={"Idempotency-Key": key},
            json=body,
        ),
    )


def _inbox_item_id(
    database_path: Path,
    *,
    recipient_agent_id: UUID,
    capsule_id: UUID,
) -> UUID:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT item_id FROM inbox_items "
            "WHERE recipient_agent_id = ? AND capsule_id = ?",
            (str(recipient_agent_id), str(capsule_id)),
        ).fetchone()
    assert row is not None
    return UUID(row[0])


def _setup_publishable_experience(
    client: TestClient,
) -> tuple[UUID, UUID, dict[str, object]]:
    publisher_id = _create_agent(client, key="publisher", name="Publisher")
    topic_id = _create_topic(
        client,
        owner_agent_id=publisher_id,
        key="publication-topic",
        name="Publication Topic",
    )
    experience = _create_experience(
        client,
        owner_agent_id=publisher_id,
        key="publication-experience",
    )
    return publisher_id, topic_id, experience


def test_capsule_routes_have_locked_methods_and_statuses() -> None:
    paths = create_app().openapi()["paths"]
    expected = {
        "/v1/agents/{agent_id}/capsules": ("post", "201"),
        "/v1/agents/{agent_id}/capsules/{capsule_id}:retract": (
            "post",
            "200",
        ),
    }

    for path, (method, success_status) in expected.items():
        assert path in paths
        assert set(paths[path]) == {method}
        assert success_status in paths[path][method]["responses"]


def test_publish_and_retract_replay_exactly_and_conflict_by_body(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-publish-retract.sqlite3"
    with _client(database_path) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        publish_body = _publish_body(
            topic_id=topic_id,
            experience_id=UUID(str(experience["experience_id"])),
        )
        published = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-once",
            body=publish_body,
        )
        publish_replay = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-once",
            body=publish_body,
        )
        publish_conflict = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-once",
            body={
                **publish_body,
                "expires_at": (FUTURE + timedelta(days=1)).isoformat(),
            },
        )
        capsule = published.json()["data"]
        capsule_id = UUID(capsule["capsule_id"])

        retract_path = f"/v1/agents/{publisher_id}/capsules/{capsule_id}:retract"
        retract_body = {"reason": "The published guidance is superseded."}
        retracted = client.post(
            retract_path,
            headers={"Idempotency-Key": "retract-once"},
            json=retract_body,
        )
        retract_replay = client.post(
            retract_path,
            headers={"Idempotency-Key": "retract-once"},
            json=retract_body,
        )
        retract_conflict = client.post(
            retract_path,
            headers={"Idempotency-Key": "retract-once"},
            json={"reason": "A different retraction command."},
        )

    assert published.status_code == publish_replay.status_code == 201
    assert publish_replay.content == published.content
    assert publish_replay.headers["location"] == published.headers["location"]
    assert published.content == canonical_json_bytes({"data": capsule})
    assert set(capsule) == CAPSULE_FIELDS
    assert capsule["transport_schema_version"] == 1
    assert capsule["topic_id"] == str(topic_id)
    assert capsule["source_experience_id"] == experience["experience_id"]
    assert capsule["source_version_id"] == experience["version_id"]
    assert capsule["publisher_agent_id"] == str(publisher_id)
    assert capsule["kind"] == EXPERIENCE_BODY["kind"]
    assert capsule["body"] == EXPERIENCE_BODY["body"]
    assert capsule["summary"] == EXPERIENCE_BODY["summary"]
    assert capsule["mechanism"] == EXPERIENCE_BODY["mechanism"]
    assert capsule["publisher_confidence"] == EXPERIENCE_BODY["confidence"]
    assert capsule["provenance_chain"] == []
    assert capsule["hop_count"] == 0
    assert capsule["source_content_hash"] == experience["content_hash"]
    assert capsule["created_at"] == "2026-07-19T08:30:00.000000Z"
    assert capsule["expires_at"] == "2026-07-26T08:30:00.000000Z"
    assert capsule["status"] == "active"
    assert len(capsule["root_fingerprint"]) == 64
    assert len(capsule["capsule_hash"]) == 64
    assert published.headers["location"] == f"/v1/capsules/{capsule_id}"
    assert publish_conflict.status_code == 409
    assert publish_conflict.json()["error"]["code"] == "idempotency_key_conflict"

    retracted_capsule = retracted.json()["data"]
    assert retracted.status_code == retract_replay.status_code == 200
    assert retract_replay.content == retracted.content
    assert retract_replay.headers["location"] == retracted.headers["location"]
    assert retracted.content == canonical_json_bytes({"data": retracted_capsule})
    assert set(retracted_capsule) == CAPSULE_FIELDS
    assert retracted_capsule["capsule_id"] == str(capsule_id)
    assert retracted_capsule["status"] == "retracted"
    assert retracted_capsule["last_transition_at"] == ("2026-07-19T08:30:00.000000Z")
    assert retract_conflict.status_code == 409
    assert retract_conflict.json()["error"]["code"] == "idempotency_key_conflict"

    assert _table_count(database_path, "experience_capsules") == 1
    assert _event_count(database_path, "capsule.published") == 1
    assert _event_count(database_path, "capsule.retracted") == 1


def test_publish_replays_its_stored_result_after_expiry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-publish-replay-after-expiry.sqlite3"
    clock = FrozenClock(NOW)
    app = create_app(
        settings=_settings(database_path),
        clock=clock,
    )
    with TestClient(app) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        publish_body = _publish_body(
            topic_id=topic_id,
            experience_id=UUID(str(experience["experience_id"])),
            expires_at=NOW + timedelta(minutes=1),
        )
        published = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-before-expiry",
            body=publish_body,
        )
        clock.advance(timedelta(minutes=2))
        replay = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-before-expiry",
            body=publish_body,
        )

    assert published.status_code == replay.status_code == 201
    assert replay.content == published.content
    assert replay.headers["location"] == published.headers["location"]
    assert _table_count(database_path, "experience_capsules") == 1
    assert _event_count(database_path, "capsule.published") == 1


def test_new_publish_binds_expiry_validation_to_its_receipt_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-publish-expiry-receipt-time.sqlite3"
    clock = BoundaryAdvancingClock()
    app = create_app(
        settings=_settings(database_path),
        clock=clock,
    )
    with TestClient(app) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        baseline_receipts = _table_count(database_path, "idempotency_records")
        clock.arm()
        published = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-at-receipt-boundary",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(experience["experience_id"])),
                expires_at=NOW + timedelta(minutes=1),
            ),
        )

    assert published.status_code == 201
    assert _table_count(database_path, "idempotency_records") == baseline_receipts + 1
    assert _table_count(database_path, "experience_capsules") == 1
    assert _event_count(database_path, "capsule.published") == 1


def test_publish_requires_strict_future_expiry_and_known_fields(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-publish-validation.sqlite3"
    with _client(database_path) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        experience_id = UUID(str(experience["experience_id"]))
        baseline_receipts = _table_count(database_path, "idempotency_records")
        invalid_bodies = (
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at=NOW,
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at=NOW - timedelta(microseconds=1),
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at="2026-07-20T08:30:00",
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at="1784536200",
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at="2026-07-20 08:30:00Z",
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at="2026-07-20T08:30:00+0000",
            ),
            _publish_body(
                topic_id=topic_id,
                experience_id=experience_id,
                expires_at=0,
            ),
            {
                **_publish_body(
                    topic_id=topic_id,
                    experience_id=experience_id,
                ),
                "status": "active",
            },
        )
        responses = tuple(
            _publish(
                client,
                publisher_agent_id=publisher_id,
                key=f"invalid-publication-{index}",
                body=body,
            )
            for index, body in enumerate(invalid_bodies)
        )

    assert all(response.status_code == 422 for response in responses)
    assert tuple(response.json()["error"]["code"] for response in responses) == (
        "invalid_expiry",
        "invalid_expiry",
        "validation_error",
        "validation_error",
        "validation_error",
        "validation_error",
        "validation_error",
        "validation_error",
    )
    assert _table_count(database_path, "experience_capsules") == 0
    assert _event_count(database_path, "capsule.published") == 0
    assert _table_count(database_path, "idempotency_records") == baseline_receipts


def test_retract_requires_a_strict_nonblank_reason_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-retract-validation.sqlite3"
    with _client(database_path) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        published = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publish-before-invalid-retract",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(experience["experience_id"])),
            ),
        )
        assert published.status_code == 201, published.text
        capsule_id = UUID(published.json()["data"]["capsule_id"])
        baseline_receipts = _table_count(database_path, "idempotency_records")
        path = f"/v1/agents/{publisher_id}/capsules/{capsule_id}:retract"
        responses = (
            client.post(
                path,
                headers={"Idempotency-Key": "missing-retract-reason"},
                json={},
            ),
            client.post(
                path,
                headers={"Idempotency-Key": "blank-retract-reason"},
                json={"reason": " \t "},
            ),
            client.post(
                path,
                headers={"Idempotency-Key": "extra-retract-field"},
                json={"reason": "superseded", "delete": True},
            ),
        )

    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _event_count(database_path, "capsule.retracted") == 0
    assert _table_count(database_path, "idempotency_records") == baseline_receipts


def test_publish_and_retract_require_idempotency_keys(tmp_path: Path) -> None:
    database_path = tmp_path / "capsule-required-key.sqlite3"
    with _client(database_path) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        publish_without_key = client.post(
            f"/v1/agents/{publisher_id}/capsules",
            json=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(experience["experience_id"])),
            ),
        )
        retract_without_key = client.post(
            f"/v1/agents/{publisher_id}/capsules/{uuid4()}:retract",
            json={"reason": "superseded"},
        )

    assert publish_without_key.status_code == retract_without_key.status_code == 422
    assert publish_without_key.json()["error"]["code"] == "validation_error"
    assert retract_without_key.json()["error"]["code"] == "validation_error"
    assert _table_count(database_path, "experience_capsules") == 0
    assert _event_count(database_path, "capsule.retracted") == 0
    assert _table_count(database_path, "idempotency_records") == 3


def test_publisher_owned_resources_are_indistinguishable_from_missing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-publisher-isolation.sqlite3"
    with _client(database_path) as client:
        publisher_id, topic_id, experience = _setup_publishable_experience(client)
        outsider_id = _create_agent(client, key="outsider", name="Outsider")
        published = _publish(
            client,
            publisher_agent_id=publisher_id,
            key="publisher-publication",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(experience["experience_id"])),
            ),
        )
        assert published.status_code == 201, published.text
        capsule_id = UUID(published.json()["data"]["capsule_id"])

        foreign_publish = _publish(
            client,
            publisher_agent_id=outsider_id,
            key="foreign-experience-publication",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(experience["experience_id"])),
            ),
        )
        missing_publish = _publish(
            client,
            publisher_agent_id=outsider_id,
            key="missing-experience-publication",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=uuid4(),
            ),
        )
        foreign_retract = client.post(
            f"/v1/agents/{outsider_id}/capsules/{capsule_id}:retract",
            headers={"Idempotency-Key": "foreign-retraction"},
            json={"reason": "unauthorized"},
        )
        missing_retract = client.post(
            f"/v1/agents/{outsider_id}/capsules/{uuid4()}:retract",
            headers={"Idempotency-Key": "missing-retraction"},
            json={"reason": "not found"},
        )

    assert foreign_publish.status_code == missing_publish.status_code == 404
    assert foreign_publish.content == missing_publish.content
    assert foreign_retract.status_code == missing_retract.status_code == 404
    assert foreign_retract.content == missing_retract.content
    assert _table_count(database_path, "experience_capsules") == 1
    assert _event_count(database_path, "capsule.retracted") == 0


def test_adopted_capsule_requires_its_owned_parent_adoption_to_republish(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capsule-parent-adoption.sqlite3"
    with _client(database_path) as client:
        root_publisher_id = _create_agent(
            client,
            key="root-publisher",
            name="Root Publisher",
        )
        relay_id = _create_agent(client, key="relay", name="Relay")
        topic_id = _create_topic(
            client,
            owner_agent_id=root_publisher_id,
            key="parent-chain-topic",
            name="Parent Chain",
        )
        subscribed = client.post(
            f"/v1/agents/{relay_id}/subscriptions",
            headers={"Idempotency-Key": "relay-subscription"},
            json={"topic_id": str(topic_id)},
        )
        assert subscribed.status_code == 201, subscribed.text
        source = _create_experience(
            client,
            owner_agent_id=root_publisher_id,
            key="root-experience",
        )
        root_published = _publish(
            client,
            publisher_agent_id=root_publisher_id,
            key="root-publication",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=UUID(str(source["experience_id"])),
            ),
        )
        assert root_published.status_code == 201, root_published.text
        root_capsule = root_published.json()["data"]
        root_capsule_id = UUID(root_capsule["capsule_id"])
        item_id = _inbox_item_id(
            database_path,
            recipient_agent_id=relay_id,
            capsule_id=root_capsule_id,
        )
        adopted = client.post(
            f"/v1/agents/{relay_id}/inbox/{item_id}:adopt",
            headers={"Idempotency-Key": "relay-adoption"},
            json={},
        )
        assert adopted.status_code == 200, adopted.text
        adopted_experience_id = UUID(
            adopted.json()["data"]["experience"]["experience_id"]
        )
        parent_adoption_id = UUID(
            adopted.headers["location"].rsplit("/", maxsplit=1)[-1]
        )

        missing_parent = _publish(
            client,
            publisher_agent_id=relay_id,
            key="relay-publication-without-parent",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=adopted_experience_id,
            ),
        )
        wrong_parent = _publish(
            client,
            publisher_agent_id=relay_id,
            key="relay-publication-wrong-parent",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=adopted_experience_id,
                parent_adoption_id=uuid4(),
            ),
        )
        relayed = _publish(
            client,
            publisher_agent_id=relay_id,
            key="relay-publication",
            body=_publish_body(
                topic_id=topic_id,
                experience_id=adopted_experience_id,
                parent_adoption_id=parent_adoption_id,
            ),
        )

    assert missing_parent.status_code == 422
    assert missing_parent.json()["error"]["code"] == "parent_adoption_required"
    assert wrong_parent.status_code == 404
    assert wrong_parent.json()["error"]["code"] == "parent_adoption_not_found"
    assert relayed.status_code == 201
    relay_capsule = relayed.json()["data"]
    assert set(relay_capsule) == CAPSULE_FIELDS
    assert relay_capsule["publisher_agent_id"] == str(relay_id)
    assert relay_capsule["source_experience_id"] == str(adopted_experience_id)
    assert relay_capsule["hop_count"] == 1
    assert relay_capsule["provenance_chain"] == [
        {
            "capsule_id": str(root_capsule_id),
            "publisher_agent_id": str(root_publisher_id),
        }
    ]
    assert relay_capsule["root_fingerprint"] == root_capsule["root_fingerprint"]
    assert relay_capsule["source_content_hash"] == root_capsule["source_content_hash"]
