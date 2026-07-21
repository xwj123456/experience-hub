from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.inspiration.repository import InspirationSourceIntegrityError

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_IDEA_FIELDS = {
    "draft",
    "duplicate_relation",
    "idea_content_hash",
    "idea_id",
    "last_signal_at",
    "maturity",
    "mechanism_cluster_id",
    "mechanism_hash",
    "operator",
    "ordinal",
    "owner_agent_id",
    "owner_decision",
    "resulting_experience_id",
    "resulting_version_id",
    "run_id",
}
_DRAFT_FIELDS = {
    "assumptions",
    "evidence",
    "falsifiers",
    "hypothesis",
    "mechanism",
    "predictions",
    "proposed_test",
    "title",
}
_MUTATION_EVENTS = {
    "adopt": "inspiration.idea_adopted_v2",
    "archive": "inspiration.idea_archived",
    "evaluate": "inspiration.idea_evaluated",
    "reject": "inspiration.idea_rejected",
}


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


def _event_count(database_path: Path, event_type: str) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            (event_type,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _create_agent(
    client: TestClient,
    *,
    key: str,
    name: str,
) -> UUID:
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
    owner_agent_id: UUID,
    key: str,
    marker: str,
) -> UUID:
    response = client.post(
        f"/v1/agents/{owner_agent_id}/experiences",
        headers={"Idempotency-Key": key},
        json={
            "applicability": [f"boundary condition {marker}"],
            "body": (
                f"Inspiration pagination signal {marker} remains observable "
                "under a bounded experiment."
            ),
            "confidence": 0.70,
            "evidence": [{"id": f"trace-{marker}", "type": "test"}],
            "falsifiers": [
                f"The inspiration signal {marker} disappears under repetition."
            ],
            "importance": 0.60,
            "kind": "semantic",
            "links": [],
            "mechanism": (
                f"Inspiration signal {marker} maps one boundary condition "
                "to one observable outcome."
            ),
            "summary": f"Inspiration pagination evidence signal {marker}.",
            "tags": ["inspiration", "pagination", "signal"],
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["data"]["experience_id"])


def _create_run(
    client: TestClient,
    *,
    owner_agent_id: UUID,
    key: str,
    goal_marker: str,
    branches: int,
    operators: tuple[str, ...] = ("counterfactual",),
) -> UUID:
    response = client.post(
        f"/v1/agents/{owner_agent_id}/inspiration-runs",
        headers={"Idempotency-Key": key},
        json={
            "branches_per_operator": branches,
            "context": "",
            "generator": "deterministic",
            "global_timeout_seconds": 90,
            "goal": (
                "Find counterfactual inspiration pagination signal branches "
                f"for {goal_marker}"
            ),
            "include_inbox": False,
            "mode": "associative",
            "operator_timeout_seconds": 30,
            "operators": list(operators),
            "output_tokens_per_operator": 1200,
            "total_output_tokens": 3600,
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["data"]["status"] == "completed"
    return UUID(response.json()["data"]["run_id"])


def _list_ideas(
    client: TestClient,
    *,
    owner_agent_id: UUID,
    run_id: UUID,
    params: dict[str, object] | None = None,
) -> Response:
    return cast(
        Response,
        client.get(
            f"/v1/agents/{owner_agent_id}/inspiration-runs/{run_id}/ideas",
            params=params,
        ),
    )


def _assert_hash(value: object) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    assert set(value) <= _HASH_CHARACTERS


def _assert_full_owner_idea(
    idea: dict[str, Any],
    *,
    owner_agent_id: UUID,
    run_id: UUID,
) -> None:
    assert set(idea) == _IDEA_FIELDS
    assert UUID(idea["idea_id"])
    assert UUID(idea["run_id"]) == run_id
    assert UUID(idea["owner_agent_id"]) == owner_agent_id
    assert idea["operator"] in {
        "causal_gap",
        "counterfactual",
        "distant_analogy",
    }
    assert type(idea["ordinal"]) is int
    assert 1 <= idea["ordinal"] <= 3
    _assert_hash(idea["idea_content_hash"])
    _assert_hash(idea["mechanism_hash"])
    _assert_hash(idea["mechanism_cluster_id"])
    assert idea["owner_decision"] in {
        "active",
        "adopted",
        "archived",
        "rejected",
    }
    assert idea["maturity"] in {"speculative", "incubating", "candidate"}
    datetime.fromisoformat(idea["last_signal_at"].replace("Z", "+00:00"))

    draft = idea["draft"]
    assert isinstance(draft, dict)
    assert set(draft) == _DRAFT_FIELDS
    assert all(
        isinstance(draft[field], str) and draft[field]
        for field in (
            "title",
            "hypothesis",
            "mechanism",
            "proposed_test",
        )
    )
    assert all(
        isinstance(draft[field], list) and draft[field]
        for field in ("predictions", "falsifiers", "assumptions", "evidence")
    )
    for reference in draft["evidence"]:
        assert set(reference) == {"id", "stable_evidence_key", "type"}
        assert reference["type"] == "snapshot_item"
        UUID(reference["id"])
        _assert_hash(reference["stable_evidence_key"])

    serialized = canonical_json_bytes(idea).decode("utf-8")
    assert "evaluator_agent_id" not in serialized
    assert "distinct_adopter" not in serialized


def _seed_one_idea(
    client: TestClient,
    *,
    owner_agent_id: UUID,
    key: str,
) -> tuple[UUID, dict[str, Any]]:
    run_id = _create_run(
        client,
        owner_agent_id=owner_agent_id,
        key=f"run-{key}",
        goal_marker=key,
        branches=1,
    )
    response = _list_ideas(
        client,
        owner_agent_id=owner_agent_id,
        run_id=run_id,
    )
    assert response.status_code == 200, response.text
    ideas = response.json()["data"]
    assert len(ideas) == 1
    idea = ideas[0]
    _assert_full_owner_idea(
        idea,
        owner_agent_id=owner_agent_id,
        run_id=run_id,
    )
    return run_id, idea


def _mutation_body(
    action: str,
    *,
    idea: dict[str, Any],
) -> dict[str, object]:
    if action == "adopt":
        return {}
    if action in {"archive", "reject"}:
        return {"reason": f"Owner explicitly chose to {action} this idea."}
    if action == "evaluate":
        return {
            "evaluated_at": NOW.isoformat(),
            "evidence": [idea["draft"]["evidence"][0]],
            "reason": "The frozen evidence supports this bounded evaluation.",
            "verdict": "supported",
        }
    raise AssertionError(f"unsupported mutation action: {action}")


def _conflicting_mutation_body(
    action: str,
    *,
    body: dict[str, object],
) -> dict[str, object]:
    if action == "adopt":
        return {"importance": 0.90}
    if action in {"archive", "reject"}:
        return {"reason": f"A different {action} reason changes the command hash."}
    if action == "evaluate":
        return {**body, "verdict": "refuted"}
    raise AssertionError(f"unsupported mutation action: {action}")


def test_inspiration_idea_routes_have_locked_methods_and_statuses() -> None:
    paths = create_app().openapi()["paths"]
    expected = {
        "/v1/agents/{agent_id}/inspiration-runs/{run_id}/ideas": {"get": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:adopt": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:reject": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:evaluate": {"post": "200"},
        "/v1/agents/{agent_id}/ideas/{idea_id}:archive": {"post": "200"},
    }

    for path, methods in expected.items():
        assert path in paths
        assert set(paths[path]) == set(methods)
        for method, success_status in methods.items():
            assert success_status in paths[path][method]["responses"]


def test_owner_list_returns_complete_ideas_with_stable_bound_cursor(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idea-list-pagination.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            key="idea-list-owner",
            name="Idea list owner",
        )
        other_id = _create_agent(
            client,
            key="idea-list-other",
            name="Other idea owner",
        )
        for marker in ("alpha", "beta", "gamma"):
            _create_experience(
                client,
                owner_agent_id=owner_id,
                key=f"idea-list-evidence-{marker}",
                marker=marker,
            )
        run_id = _create_run(
            client,
            owner_agent_id=owner_id,
            key="idea-list-run",
            goal_marker="stable owner list",
            branches=1,
            operators=("causal_gap", "counterfactual"),
        )
        other_run_id = _create_run(
            client,
            owner_agent_id=owner_id,
            key="idea-list-other-run",
            goal_marker="cursor binding",
            branches=1,
        )

        first = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=run_id,
            params={"limit": 1},
        )
        first_again = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=run_id,
            params={"limit": 1},
        )
        assert first.status_code == first_again.status_code == 200
        assert first.content == first_again.content

        pages = [first]
        cursor = first.json()["page"]["next_cursor"]
        while cursor is not None:
            page = _list_ideas(
                client,
                owner_agent_id=owner_id,
                run_id=run_id,
                params={"cursor": cursor, "limit": 1},
            )
            assert page.status_code == 200, page.text
            pages.append(page)
            cursor = page.json()["page"]["next_cursor"]

        first_cursor = first.json()["page"]["next_cursor"]
        assert isinstance(first_cursor, str) and first_cursor
        cross_run_cursor = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=other_run_id,
            params={"cursor": first_cursor, "limit": 1},
        )
        malformed_cursor = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=run_id,
            params={"cursor": f"{first_cursor}x", "limit": 1},
        )
        foreign = _list_ideas(
            client,
            owner_agent_id=other_id,
            run_id=run_id,
        )
        missing = _list_ideas(
            client,
            owner_agent_id=other_id,
            run_id=uuid4(),
        )

    ideas = [page.json()["data"][0] for page in pages]
    assert len(ideas) == 2
    assert len({idea["idea_id"] for idea in ideas}) == 2
    assert [idea["operator"] for idea in ideas] == [
        "causal_gap",
        "counterfactual",
    ]
    assert [idea["ordinal"] for idea in ideas] == [1, 1]
    for idea in ideas:
        _assert_full_owner_idea(
            idea,
            owner_agent_id=owner_id,
            run_id=run_id,
        )
    assert pages[-1].json()["page"]["next_cursor"] is None
    assert cross_run_cursor.status_code == malformed_cursor.status_code == 400
    assert cross_run_cursor.json()["error"]["code"] == "invalid_cursor"
    assert malformed_cursor.json()["error"]["code"] == "invalid_cursor"
    assert foreign.status_code == missing.status_code == 404
    assert foreign.content == missing.content
    assert foreign.json()["error"]["code"] == "resource_not_found"


@pytest.mark.parametrize("projection", ("state", "cluster"))
def test_owner_idea_list_fails_closed_when_a_projection_row_is_missing(
    tmp_path: Path,
    projection: str,
) -> None:
    database_path = tmp_path / f"idea-list-missing-{projection}.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            key="missing-projection-owner",
            name="Missing projection owner",
        )
        _create_experience(
            client,
            owner_agent_id=owner_id,
            key="missing-projection-evidence",
            marker="missing projection",
        )
        run_id = _create_run(
            client,
            owner_agent_id=owner_id,
            key="missing-projection-run",
            goal_marker="projection integrity",
            branches=1,
        )
        before = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=run_id,
        )
        assert before.status_code == 200, before.text
        idea_id = UUID(before.json()["data"][0]["idea_id"])
        with sqlite3.connect(database_path) as connection:
            if projection == "state":
                connection.execute(
                    "DELETE FROM idea_state WHERE idea_id = ?",
                    (str(idea_id),),
                )
            else:
                row = connection.execute(
                    "SELECT mechanism_cluster_id FROM idea_state WHERE idea_id = ?",
                    (str(idea_id),),
                ).fetchone()
                assert row is not None
                connection.execute(
                    "DELETE FROM mechanism_incubation WHERE cluster_id = ?",
                    (row[0],),
                )

        with pytest.raises(
            InspirationSourceIntegrityError,
            match="idea source or projection",
        ):
            _list_ideas(
                client,
                owner_agent_id=owner_id,
                run_id=run_id,
            )


def test_adopt_reject_evaluate_and_archive_replay_exactly_and_conflict(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idea-mutations.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            key="idea-mutation-owner",
            name="Idea mutation owner",
        )
        _create_experience(
            client,
            owner_agent_id=owner_id,
            key="idea-mutation-evidence",
            marker="mutation",
        )

        observations: dict[str, tuple[dict[str, Any], Response]] = {}
        for action in ("adopt", "reject", "evaluate", "archive"):
            run_id, idea = _seed_one_idea(
                client,
                owner_agent_id=owner_id,
                key=f"mutation-{action}",
            )
            idea_id = UUID(idea["idea_id"])
            path = f"/v1/agents/{owner_id}/ideas/{idea_id}:{action}"
            body = _mutation_body(action, idea=idea)
            headers = {"Idempotency-Key": f"idea-{action}-once"}

            first = client.post(path, headers=headers, json=body)
            replay = client.post(path, headers=headers, json=body)
            conflict = client.post(
                path,
                headers=headers,
                json=_conflicting_mutation_body(action, body=body),
            )

            assert first.status_code == replay.status_code == 200, first.text
            assert replay.content == first.content
            assert replay.headers["content-type"] == first.headers["content-type"]
            assert canonical_json_bytes(first.json()) == first.content
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "idempotency_key_conflict"

            listed = _list_ideas(
                client,
                owner_agent_id=owner_id,
                run_id=run_id,
            )
            assert listed.status_code == 200
            listed_idea = listed.json()["data"][0]
            _assert_full_owner_idea(
                listed_idea,
                owner_agent_id=owner_id,
                run_id=run_id,
            )
            observations[action] = (listed_idea, first)

    adopted_idea, adopted = observations["adopt"]
    assert adopted_idea["owner_decision"] == "adopted"
    assert set(adopted.json()["data"]) == {"created", "experience"}
    assert adopted.json()["data"]["created"] is True
    adopted_experience = adopted.json()["data"]["experience"]
    assert set(adopted_experience) == {
        "current_content_hash",
        "current_version_id",
        "experience_id",
        "owner_agent_id",
        "temperature",
    }
    assert adopted_experience["owner_agent_id"] == str(owner_id)
    assert adopted_experience["temperature"] == "warm"
    assert (
        adopted_idea["resulting_experience_id"] == (adopted_experience["experience_id"])
    )
    assert (
        adopted_idea["resulting_version_id"]
        == (adopted_experience["current_version_id"])
    )

    rejected_idea, rejected = observations["reject"]
    assert rejected_idea["owner_decision"] == "rejected"
    assert rejected.json() == {
        "data": {
            "idea_id": rejected_idea["idea_id"],
            "owner_decision": "rejected",
        }
    }

    evaluated_idea, evaluated = observations["evaluate"]
    assert evaluated_idea["owner_decision"] == "active"
    assert evaluated.json()["data"] == {
        "idea_id": evaluated_idea["idea_id"],
        "maturity": evaluated_idea["maturity"],
        "owner_decision": "active",
        "revision": 1,
    }

    archived_idea, archived = observations["archive"]
    assert archived_idea["owner_decision"] == "archived"
    assert archived.json() == {
        "data": {
            "idea_id": archived_idea["idea_id"],
            "owner_decision": "archived",
        }
    }

    for event_type in _MUTATION_EVENTS.values():
        assert _event_count(database_path, event_type) == 1


@pytest.mark.parametrize(
    "action",
    ("adopt", "reject", "evaluate", "archive"),
)
def test_every_idea_mutation_hides_foreign_and_missing_ideas(
    tmp_path: Path,
    action: str,
) -> None:
    database_path = tmp_path / f"idea-{action}-owner-isolation.sqlite3"
    with _client(database_path) as client:
        owner_id = _create_agent(
            client,
            key=f"{action}-isolation-owner",
            name=f"{action.title()} idea owner",
        )
        outsider_id = _create_agent(
            client,
            key=f"{action}-isolation-outsider",
            name=f"{action.title()} outsider",
        )
        _create_experience(
            client,
            owner_agent_id=owner_id,
            key=f"{action}-isolation-evidence",
            marker=f"{action}-isolation",
        )
        run_id, idea = _seed_one_idea(
            client,
            owner_agent_id=owner_id,
            key=f"{action}-private",
        )
        body = _mutation_body(action, idea=idea)
        foreign_path = f"/v1/agents/{outsider_id}/ideas/{idea['idea_id']}:{action}"
        missing_path = f"/v1/agents/{outsider_id}/ideas/{uuid4()}:{action}"
        foreign = client.post(
            foreign_path,
            headers={"Idempotency-Key": f"{action}-foreign"},
            json=body,
        )
        foreign_replay = client.post(
            foreign_path,
            headers={"Idempotency-Key": f"{action}-foreign"},
            json=body,
        )
        missing = client.post(
            missing_path,
            headers={"Idempotency-Key": f"{action}-missing"},
            json=body,
        )
        retained = _list_ideas(
            client,
            owner_agent_id=owner_id,
            run_id=run_id,
        )

    assert (
        foreign.status_code == foreign_replay.status_code == missing.status_code == 404
    )
    assert foreign.content == foreign_replay.content == missing.content
    assert foreign.json()["error"] == {
        "code": "resource_not_found",
        "details": {},
        "message": "The command resource was not found",
    }
    assert retained.status_code == 200
    assert retained.json()["data"][0]["owner_decision"] == "active"
    assert _event_count(database_path, _MUTATION_EVENTS[action]) == 0


def test_every_idea_mutation_requires_one_strict_idempotency_key(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idea-required-keys.sqlite3"
    agent_id = uuid4()
    idea_id = uuid4()
    evidence = {
        "id": str(uuid4()),
        "stable_evidence_key": "a" * 64,
        "type": "snapshot_item",
    }
    bodies = {
        "adopt": {},
        "archive": {"reason": "Archive requires one strict key."},
        "evaluate": {
            "evaluated_at": NOW.isoformat(),
            "evidence": [evidence],
            "verdict": "supported",
        },
        "reject": {"reason": "Reject requires one strict key."},
    }

    with _client(database_path) as client:
        responses = []
        for action, body in bodies.items():
            path = f"/v1/agents/{agent_id}/ideas/{idea_id}:{action}"
            for key in (None, "", " \t ", "k" * 129):
                headers = {} if key is None else {"Idempotency-Key": key}
                responses.append(client.post(path, headers=headers, json=body))
            responses.append(
                client.post(
                    path,
                    headers=[
                        ("Idempotency-Key", f"{action}-a"),
                        ("Idempotency-Key", f"{action}-b"),
                    ],
                    json=body,
                )
            )

    assert len(responses) == 20
    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _receipt_count(database_path) == 0


def test_idea_mutation_schemas_reject_invalid_values_before_receipts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "idea-strict-schemas.sqlite3"
    agent_id = uuid4()
    idea_id = uuid4()
    valid_reference = {
        "id": str(uuid4()),
        "stable_evidence_key": "b" * 64,
        "type": "snapshot_item",
    }
    cases: tuple[tuple[str, dict[str, object]], ...] = (
        ("adopt", {"importance": -0.01}),
        ("adopt", {"importance": 1.01}),
        ("adopt", {"importance": True}),
        ("adopt", {"confidence": "0.5"}),
        ("adopt", {"unexpected": "field"}),
        ("reject", {"reason": ""}),
        ("reject", {"reason": " \t "}),
        ("reject", {"reason": "r" * 2_001}),
        ("reject", {"reason": "valid", "unexpected": "field"}),
        ("archive", {"reason": ""}),
        ("archive", {"reason": " \t "}),
        ("archive", {"reason": "a" * 2_001}),
        ("archive", {"reason": "valid", "unexpected": "field"}),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [],
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [valid_reference],
                "verdict": "not_a_verdict",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [
                    {
                        "id": str(uuid4()),
                        "type": "snapshot_item",
                    }
                ],
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [valid_reference, valid_reference],
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [valid_reference],
                "reason": "",
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [valid_reference],
                "reason": "e" * 2_001,
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": NOW.isoformat(),
                "evidence": [valid_reference],
                "unexpected": "field",
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evidence": [valid_reference],
                "verdict": "supported",
            },
        ),
        (
            "evaluate",
            {
                "evaluated_at": "2026-07-19T08:30:00",
                "evidence": [valid_reference],
                "verdict": "supported",
            },
        ),
    )

    with _client(database_path) as client:
        responses = [
            client.post(
                f"/v1/agents/{agent_id}/ideas/{idea_id}:{action}",
                headers={"Idempotency-Key": f"invalid-{index}"},
                json=body,
            )
            for index, (action, body) in enumerate(cases)
        ]

    assert len(responses) == len(cases)
    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["error"]["code"] == "validation_error" for response in responses
    )
    assert _receipt_count(database_path) == 0


@pytest.mark.parametrize("limit", (0, 101, True, "01"))
def test_idea_list_rejects_noncanonical_limits(
    tmp_path: Path,
    limit: object,
) -> None:
    database_path = tmp_path / f"idea-invalid-limit-{str(limit)}.sqlite3"
    with _client(database_path) as client:
        response = _list_ideas(
            client,
            owner_agent_id=uuid4(),
            run_id=uuid4(),
            params={"limit": limit},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0
