from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

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


def _create_agent(client: TestClient, *, name: str, key: str) -> UUID:
    response = client.post(
        "/v1/agents",
        headers={"Idempotency-Key": key},
        json={"name": name},
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["agent_id"])


def _create_experience(
    client: TestClient,
    *,
    publisher_id: UUID,
    key: str,
) -> tuple[UUID, UUID]:
    response = client.post(
        f"/v1/agents/{publisher_id}/experiences",
        headers={"Idempotency-Key": key},
        json={
            "applicability": ["network operations"],
            "body": "A durable receipt prevents duplicate retry side effects.",
            "confidence": 0.8,
            "evidence": [{"id": "trace-sharing", "type": "test"}],
            "falsifiers": ["A replay appends a second event."],
            "importance": 0.7,
            "kind": "procedural",
            "mechanism": "Canonical request hashes bind retries to one receipt.",
            "summary": "Durable receipts make retries safe.",
            "tags": ["idempotency", "sharing"],
        },
    )
    assert response.status_code == 201, response.text
    data = response.json()["data"]
    return UUID(data["experience_id"]), UUID(data["version_id"])


def _create_topic(
    client: TestClient,
    *,
    publisher_id: UUID,
    suffix: str,
) -> UUID:
    response = client.post(
        "/v1/topics",
        headers={"Idempotency-Key": f"topic-{suffix}"},
        json={
            "description": "Owner-scoped propagation tests.",
            "name": f"api-sharing-{suffix}",
            "owner_agent_id": str(publisher_id),
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["topic_id"])


def _subscribe(
    client: TestClient,
    *,
    recipient_id: UUID,
    topic_id: UUID,
    key: str,
) -> None:
    response = client.post(
        f"/v1/agents/{recipient_id}/subscriptions",
        headers={"Idempotency-Key": key},
        json={"topic_id": str(topic_id)},
    )
    assert response.status_code == 201, response.text


def _publish(
    client: TestClient,
    *,
    publisher_id: UUID,
    topic_id: UUID,
    experience_id: UUID,
    version_id: UUID,
    key: str,
) -> UUID:
    response = client.post(
        f"/v1/agents/{publisher_id}/capsules",
        headers={"Idempotency-Key": key},
        json={
            "experience_id": str(experience_id),
            "expires_at": (NOW + timedelta(days=7)).isoformat(),
            "topic_id": str(topic_id),
            "version_id": str(version_id),
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["capsule_id"])


def _list_inbox(
    client: TestClient,
    *,
    recipient_id: UUID,
    params: dict[str, str | int] | None = None,
) -> Any:
    return client.get(
        f"/v1/agents/{recipient_id}/inbox",
        params=params,
    )


def _cursor_with_updates(cursor: str, **updates: object) -> str:
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    payload.update(updates)
    return (
        base64.urlsafe_b64encode(canonical_json_bytes(payload))
        .decode("ascii")
        .rstrip("=")
    )


@dataclass(frozen=True, slots=True)
class _Delivery:
    publisher_id: UUID
    recipient_id: UUID
    outsider_id: UUID
    experience_id: UUID
    version_id: UUID
    topic_id: UUID
    capsule_id: UUID
    item_id: UUID


def _seed_delivery(client: TestClient, *, suffix: str) -> _Delivery:
    publisher_id = _create_agent(
        client,
        name=f"Publisher {suffix}",
        key=f"publisher-{suffix}",
    )
    recipient_id = _create_agent(
        client,
        name=f"Recipient {suffix}",
        key=f"recipient-{suffix}",
    )
    outsider_id = _create_agent(
        client,
        name=f"Outsider {suffix}",
        key=f"outsider-{suffix}",
    )
    experience_id, version_id = _create_experience(
        client,
        publisher_id=publisher_id,
        key=f"experience-{suffix}",
    )
    topic_id = _create_topic(
        client,
        publisher_id=publisher_id,
        suffix=suffix,
    )
    _subscribe(
        client,
        recipient_id=recipient_id,
        topic_id=topic_id,
        key=f"subscription-{suffix}",
    )
    capsule_id = _publish(
        client,
        publisher_id=publisher_id,
        topic_id=topic_id,
        experience_id=experience_id,
        version_id=version_id,
        key=f"capsule-{suffix}",
    )
    inbox = _list_inbox(client, recipient_id=recipient_id)
    assert inbox.status_code == 200, inbox.text
    items = inbox.json()["data"]
    assert len(items) == 1
    assert items[0]["capsule_id"] == str(capsule_id)
    return _Delivery(
        publisher_id=publisher_id,
        recipient_id=recipient_id,
        outsider_id=outsider_id,
        experience_id=experience_id,
        version_id=version_id,
        topic_id=topic_id,
        capsule_id=capsule_id,
        item_id=UUID(items[0]["item_id"]),
    )


def test_inbox_is_keyless_paginated_state_filtered_and_cursor_bound(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inbox-list.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="inbox-list")
        extra_capsules = [
            _publish(
                client,
                publisher_id=seed.publisher_id,
                topic_id=seed.topic_id,
                experience_id=seed.experience_id,
                version_id=seed.version_id,
                key=f"capsule-inbox-list-{index}",
            )
            for index in (2, 3)
        ]
        receipt_count = _table_count(database_path, "idempotency_records")

        complete = _list_inbox(client, recipient_id=seed.recipient_id)
        first = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"limit": 1},
        )
        cursor = first.json()["page"]["next_cursor"]
        second = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"cursor": cursor, "limit": 2},
        )

        all_items = complete.json()["data"]
        first_items = first.json()["data"]
        second_items = second.json()["data"]
        assert complete.status_code == first.status_code == second.status_code == 200
        assert len(all_items) == 3
        assert len(first_items) == 1
        assert len(second_items) == 2
        assert isinstance(cursor, str) and cursor and "=" not in cursor
        assert {item["item_id"] for item in (*first_items, *second_items)} == {
            item["item_id"] for item in all_items
        }
        assert {item["capsule_id"] for item in all_items} == {
            str(seed.capsule_id),
            *(str(value) for value in extra_capsules),
        }
        assert all(item["state"] == "pending" for item in all_items)
        assert all(item["effective_availability"] == "active" for item in all_items)
        assert all(
            item["recipient_agent_id"] == str(seed.recipient_id) for item in all_items
        )
        assert _table_count(database_path, "idempotency_records") == receipt_count

        invalid_limits = tuple(
            _list_inbox(
                client,
                recipient_id=seed.recipient_id,
                params={"limit": value},
            )
            for value in ("1.0", "01", "true", "+1", " 1", "1١", "1０")
        )
        duplicate_limit = client.get(
            f"/v1/agents/{seed.recipient_id}/inbox",
            params=(("limit", "1"), ("limit", "2")),
        )
        invalid_cursors = tuple(
            _list_inbox(
                client,
                recipient_id=seed.recipient_id,
                params={"cursor": value},
            )
            for value in (
                f"{cursor}=",
                f"{cursor}+",
                _cursor_with_updates(cursor, v=True),
                _cursor_with_updates(cursor, v=1.0),
                _cursor_with_updates(
                    cursor,
                    owner_agent_id=f"{{{seed.recipient_id}}}",
                ),
                "A" * 8_193,
            )
        )

        assert all(response.status_code == 422 for response in invalid_limits)
        assert duplicate_limit.status_code == 422
        assert all(
            response.json()["error"]["code"] == "validation_error"
            for response in (*invalid_limits, duplicate_limit)
        )
        assert all(response.status_code == 400 for response in invalid_cursors)
        assert all(
            response.json()["error"]["code"] == "invalid_cursor"
            for response in invalid_cursors
        )
        assert _table_count(database_path, "idempotency_records") == receipt_count

        adopted_item_id = all_items[0]["item_id"]
        rejected_item_id = all_items[1]["item_id"]
        adopted = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{adopted_item_id}:adopt"),
            headers={"Idempotency-Key": "adopt-for-inbox-filter"},
            json={"importance": 0.55},
        )
        rejected = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{rejected_item_id}:reject"),
            headers={"Idempotency-Key": "reject-for-inbox-filter"},
            json={"reason": "The evidence is not applicable here."},
        )
        assert adopted.status_code == rejected.status_code == 200

        filtered_receipts = _table_count(database_path, "idempotency_records")
        adopted_page = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"state": "adopted"},
        )
        rejected_page = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"state": "rejected"},
        )
        pending_page = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"state": "pending"},
        )
        mismatched_cursor = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"cursor": cursor, "state": "pending"},
        )

    assert [item["item_id"] for item in adopted_page.json()["data"]] == [
        adopted_item_id
    ]
    assert [item["item_id"] for item in rejected_page.json()["data"]] == [
        rejected_item_id
    ]
    assert len(pending_page.json()["data"]) == 1
    assert mismatched_cursor.status_code == 400
    assert mismatched_cursor.json()["error"]["code"] == "invalid_cursor"
    assert _table_count(database_path, "idempotency_records") == filtered_receipts


def test_inbox_does_not_misreport_internal_value_errors_as_invalid_cursors(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inbox-internal-value-error.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        owner_id = _create_agent(
            client,
            name="Corrupt inbox owner",
            key="corrupt-inbox-owner",
        )

        async def fail_inbox_read(**_: object) -> None:
            raise ValueError("simulated source decoding failure")

        app.state.container.sharing_query.list_inbox = fail_inbox_read
        response = _list_inbox(client, recipient_id=owner_id)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


def test_adopt_and_reject_replay_exactly_and_conflict_without_duplication(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inbox-decisions.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="inbox-decisions")
        second_capsule_id = _publish(
            client,
            publisher_id=seed.publisher_id,
            topic_id=seed.topic_id,
            experience_id=seed.experience_id,
            version_id=seed.version_id,
            key="capsule-inbox-decisions-2",
        )
        inbox = _list_inbox(client, recipient_id=seed.recipient_id)
        by_capsule = {UUID(item["capsule_id"]): item for item in inbox.json()["data"]}
        adopt_path = (
            f"/v1/agents/{seed.recipient_id}/inbox/"
            f"{by_capsule[seed.capsule_id]['item_id']}:adopt"
        )
        reject_source = by_capsule[second_capsule_id]
        reject_path = (
            f"/v1/agents/{seed.recipient_id}/inbox/{reject_source['item_id']}:reject"
        )

        adopted = client.post(
            adopt_path,
            headers={"Idempotency-Key": "adopt-replay"},
            json={"importance": 0.6},
        )
        adopted_replay = client.post(
            adopt_path,
            headers={"Idempotency-Key": "adopt-replay"},
            json={"importance": 0.6},
        )
        adopt_conflict = client.post(
            adopt_path,
            headers={"Idempotency-Key": "adopt-replay"},
            json={"importance": 0.7},
        )

        rejected = client.post(
            reject_path,
            headers={"Idempotency-Key": "reject-replay"},
            json={"reason": "  Not applicable to this deployment.  "},
        )
        rejected_replay = client.post(
            reject_path,
            headers={"Idempotency-Key": "reject-replay"},
            json={"reason": "Not applicable to this deployment."},
        )
        reject_conflict = client.post(
            reject_path,
            headers={"Idempotency-Key": "reject-replay"},
            json={"reason": "A different retained reason."},
        )

    assert adopted.status_code == adopted_replay.status_code == 200
    assert adopted.content == adopted_replay.content
    assert adopted.headers["location"] == adopted_replay.headers["location"]
    adoption = adopted.json()["data"]
    assert adoption["experience"]["owner_agent_id"] == str(seed.recipient_id)
    assert adoption["created"] is True
    assert adoption["corroboration_applied"] is False
    assert "body" not in adoption["experience"]
    assert adopt_conflict.status_code == 409
    assert adopt_conflict.json()["error"]["code"] == "idempotency_key_conflict"

    assert rejected.status_code == rejected_replay.status_code == 200
    assert rejected.content == rejected_replay.content
    assert rejected.headers["location"] == rejected_replay.headers["location"]
    assert rejected.json()["data"] == {
        **reject_source,
        "state": "rejected",
    }
    assert reject_conflict.status_code == 409
    assert reject_conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert _table_count(database_path, "adoption_records") == 1
    assert _event_count(database_path, "capsule.adopted") == 1
    assert _event_count(database_path, "capsule.rejected") == 1


def test_reject_requires_one_strict_reason_before_receipt_or_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inbox-reject-validation.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="reject-validation")
        path = f"/v1/agents/{seed.recipient_id}/inbox/{seed.item_id}:reject"
        receipt_count = _table_count(database_path, "idempotency_records")
        invalid_payloads: tuple[dict[str, object], ...] = (
            {},
            {"reason": " \t "},
            {"reason": "r" * 2_001},
            {"reason": "Valid reason.", "unknown": True},
            {"reason": 7},
        )
        responses = [
            client.post(
                path,
                headers={"Idempotency-Key": f"invalid-reject-{index}"},
                json=payload,
            )
            for index, payload in enumerate(invalid_payloads)
        ]
        pending = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"state": "pending"},
        )

    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _table_count(database_path, "idempotency_records") == receipt_count
    assert _event_count(database_path, "capsule.rejected") == 0
    assert [item["item_id"] for item in pending.json()["data"]] == [str(seed.item_id)]
