from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from tests.api.test_inbox import (
    _client,
    _event_count,
    _list_inbox,
    _seed_delivery,
    _table_count,
)


def test_foreign_and_missing_inbox_decisions_share_the_same_404(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cross-inbox-decisions.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="cross-inbox-decisions")
        missing_item_id = uuid4()

        foreign_adopt = client.post(
            (f"/v1/agents/{seed.outsider_id}/inbox/{seed.item_id}:adopt"),
            headers={"Idempotency-Key": "foreign-adopt"},
            json={"importance": 0.5},
        )
        foreign_adopt_replay = client.post(
            (f"/v1/agents/{seed.outsider_id}/inbox/{seed.item_id}:adopt"),
            headers={"Idempotency-Key": "foreign-adopt"},
            json={"importance": 0.5},
        )
        missing_adopt = client.post(
            (f"/v1/agents/{seed.outsider_id}/inbox/{missing_item_id}:adopt"),
            headers={"Idempotency-Key": "missing-adopt"},
            json={"importance": 0.5},
        )
        foreign_reject = client.post(
            (f"/v1/agents/{seed.outsider_id}/inbox/{seed.item_id}:reject"),
            headers={"Idempotency-Key": "foreign-reject"},
            json={"reason": "Owner isolation must hide this item."},
        )
        missing_reject = client.post(
            (f"/v1/agents/{seed.outsider_id}/inbox/{missing_item_id}:reject"),
            headers={"Idempotency-Key": "missing-reject"},
            json={"reason": "Owner isolation must hide this item."},
        )
        recipient_pending = _list_inbox(
            client,
            recipient_id=seed.recipient_id,
            params={"state": "pending"},
        )
        outsider_inbox = _list_inbox(
            client,
            recipient_id=seed.outsider_id,
        )

    assert (
        foreign_adopt.status_code
        == foreign_adopt_replay.status_code
        == missing_adopt.status_code
        == foreign_reject.status_code
        == missing_reject.status_code
        == 404
    )
    assert foreign_adopt.content == foreign_adopt_replay.content
    assert foreign_adopt.content == missing_adopt.content
    assert foreign_reject.content == missing_reject.content
    assert (
        foreign_adopt.json()["error"]
        == foreign_reject.json()["error"]
        == {
            "code": "resource_not_found",
            "details": {},
            "message": "The command resource was not found",
        }
    )
    assert [item["item_id"] for item in recipient_pending.json()["data"]] == [
        str(seed.item_id)
    ]
    assert outsider_inbox.status_code == 200
    assert outsider_inbox.json() == {
        "data": [],
        "page": {"next_cursor": None},
    }
    assert _table_count(database_path, "adoption_records") == 0
    assert _event_count(database_path, "capsule.adopted") == 0
    assert _event_count(database_path, "capsule.rejected") == 0


def test_foreign_and_missing_capsule_retractions_share_the_same_404(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cross-capsule-retraction.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="cross-capsule-retraction")
        payload = {"reason": "Only the publisher can retract a capsule."}
        foreign = client.post(
            (f"/v1/agents/{seed.outsider_id}/capsules/{seed.capsule_id}:retract"),
            headers={"Idempotency-Key": "foreign-retraction"},
            json=payload,
        )
        missing = client.post(
            (f"/v1/agents/{seed.outsider_id}/capsules/{uuid4()}:retract"),
            headers={"Idempotency-Key": "missing-retraction"},
            json=payload,
        )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json()["error"] == {
        "code": "resource_not_found",
        "details": {},
        "message": "The command resource was not found",
    }
    assert _event_count(database_path, "capsule.retracted") == 0


def test_pending_capsule_never_leaks_into_recall_and_adoption_makes_it_owned(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "quarantine-recall-boundary.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="quarantine-recall")
        search_path = f"/v1/agents/{seed.recipient_id}/experiences:search"
        search_body = {
            "mode": "focused",
            "query": "canonical request hashes durable receipt retries",
        }
        pending_search = client.post(
            search_path,
            headers={"Idempotency-Key": "search-before-adoption"},
            json=search_body,
        )
        adopted = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{seed.item_id}:adopt"),
            headers={"Idempotency-Key": "adopt-for-recall"},
            json={},
        )
        owned_search = client.post(
            search_path,
            headers={"Idempotency-Key": "search-after-adoption"},
            json=search_body,
        )

    assert pending_search.status_code == 200
    assert pending_search.json()["data"]["hits"] == []
    assert adopted.status_code == 200
    adopted_experience_id = adopted.json()["data"]["experience"]["experience_id"]
    assert owned_search.status_code == 200
    hits = owned_search.json()["data"]["hits"]
    assert any(
        hit["experience"]["experience_id"] == adopted_experience_id
        and hit["experience"]["owner_agent_id"] == str(seed.recipient_id)
        and hit["experience"]["origin"] == "adopted_capsule"
        for hit in hits
    )
