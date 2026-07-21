from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from tests.api.test_inbox import (
    _client,
    _event_count,
    _seed_delivery,
    _table_count,
)


def test_feedback_revisions_are_authorized_canonical_and_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "feedback-revisions.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="feedback-revisions")
        rejected = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{seed.item_id}:reject"),
            headers={"Idempotency-Key": "reject-before-feedback"},
            json={"reason": "The capsule was reviewed outside quarantine."},
        )
        assert rejected.status_code == 200

        path = f"/v1/agents/{seed.recipient_id}/capsules/{seed.capsule_id}:feedback"
        first_payload = {
            "evidence": [
                {"type": "test", "id": "trace-b"},
                {"id": "trace-a", "type": "test"},
                {"id": "trace-a", "type": "test"},
            ],
            "reason": "  Reproduced independently in staging.  ",
            "verdict": "useful",
        }
        first = client.post(
            path,
            headers={"Idempotency-Key": "feedback-revision-1"},
            json=first_payload,
        )
        replay = client.post(
            path,
            headers={"Idempotency-Key": "feedback-revision-1"},
            json=first_payload,
        )
        second = client.post(
            path,
            headers={"Idempotency-Key": "feedback-revision-2"},
            json={
                "evidence": [{"id": "counterexample", "type": "test"}],
                "reason": "A later counterexample refuted the mechanism.",
                "verdict": "refuted",
            },
        )
        conflict = client.post(
            path,
            headers={"Idempotency-Key": "feedback-revision-2"},
            json={
                "evidence": [{"id": "counterexample", "type": "test"}],
                "reason": "The same key cannot rewrite its revision.",
                "verdict": "harmful",
            },
        )

    assert first.status_code == replay.status_code == 201
    assert first.content == replay.content
    assert first.headers["location"] == replay.headers["location"]
    first_data = first.json()["data"]
    assert first_data["observer_agent_id"] == str(seed.recipient_id)
    assert first_data["capsule_id"] == str(seed.capsule_id)
    assert first_data["revision"] == 1
    assert first_data["verdict"] == "useful"
    assert first_data["reason"]["code"] == "user_provided"
    assert first_data["reason"]["text"] == "Reproduced independently in staging."
    assert first_data["evidence"] == [
        {"id": "trace-a", "type": "test"},
        {"id": "trace-b", "type": "test"},
    ]

    assert second.status_code == 201
    assert second.json()["data"]["revision"] == 2
    assert second.json()["data"]["verdict"] == "refuted"
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert _table_count(database_path, "capsule_feedback") == 2
    assert _event_count(database_path, "capsule.feedback_recorded") == 2


def test_pending_foreign_and_missing_feedback_are_one_stable_404(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "feedback-authorization.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="feedback-authorization")
        payload = {
            "evidence": [],
            "reason": "This feedback must be authorization scoped.",
            "verdict": "useful",
        }
        pending = client.post(
            (f"/v1/agents/{seed.recipient_id}/capsules/{seed.capsule_id}:feedback"),
            headers={"Idempotency-Key": "feedback-pending"},
            json=payload,
        )
        foreign = client.post(
            (f"/v1/agents/{seed.outsider_id}/capsules/{seed.capsule_id}:feedback"),
            headers={"Idempotency-Key": "feedback-foreign"},
            json=payload,
        )
        missing = client.post(
            (f"/v1/agents/{seed.outsider_id}/capsules/{uuid4()}:feedback"),
            headers={"Idempotency-Key": "feedback-missing"},
            json=payload,
        )

        rejected = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{seed.item_id}:reject"),
            headers={"Idempotency-Key": "reject-to-authorize-feedback"},
            json={"reason": "Reviewed and rejected from quarantine."},
        )
        authorized = client.post(
            (f"/v1/agents/{seed.recipient_id}/capsules/{seed.capsule_id}:feedback"),
            headers={"Idempotency-Key": "feedback-after-rejection"},
            json=payload,
        )

    assert pending.status_code == foreign.status_code == missing.status_code == 404
    assert pending.content == foreign.content == missing.content
    assert pending.json()["error"] == {
        "code": "resource_not_found",
        "details": {},
        "message": "The command resource was not found",
    }
    assert rejected.status_code == 200
    assert authorized.status_code == 201
    assert authorized.json()["data"]["revision"] == 1
    assert _table_count(database_path, "capsule_feedback") == 1


def test_feedback_rejects_invalid_reason_evidence_and_verdict_pre_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "feedback-validation.sqlite3"
    with _client(database_path) as client:
        seed = _seed_delivery(client, suffix="feedback-validation")
        rejected = client.post(
            (f"/v1/agents/{seed.recipient_id}/inbox/{seed.item_id}:reject"),
            headers={"Idempotency-Key": "reject-before-feedback-validation"},
            json={"reason": "Complete quarantine review first."},
        )
        assert rejected.status_code == 200
        path = f"/v1/agents/{seed.recipient_id}/capsules/{seed.capsule_id}:feedback"
        valid = {
            "evidence": [],
            "reason": "A valid reason.",
            "verdict": "useful",
        }
        invalid_payloads: tuple[dict[str, object], ...] = (
            {},
            {**valid, "reason": " \t "},
            {**valid, "reason": "r" * 2_001},
            {**valid, "verdict": "unknown"},
            {**valid, "verdict": 1},
            {
                **valid,
                "evidence": [
                    {"id": f"trace-{index}", "type": "test"} for index in range(33)
                ],
            },
            {
                **valid,
                "evidence": [{"id": "trace", "type": "test", "unknown": True}],
            },
            {**valid, "unknown": True},
        )
        receipt_count = _table_count(database_path, "idempotency_records")
        responses = [
            client.post(
                path,
                headers={"Idempotency-Key": f"invalid-feedback-{index}"},
                json=payload,
            )
            for index, payload in enumerate(invalid_payloads)
        ]

    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _table_count(database_path, "idempotency_records") == receipt_count
    assert _table_count(database_path, "capsule_feedback") == 0
    assert _event_count(database_path, "capsule.feedback_recorded") == 0
