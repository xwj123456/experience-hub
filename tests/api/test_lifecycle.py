from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import UUID

from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import lifecycle_cycle_id

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
EARLIER = NOW - timedelta(hours=1)
RECEIPT_IDS = (
    UUID("00000000-0000-0000-0000-000000007001"),
    UUID("00000000-0000-0000-0000-000000007002"),
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


def _result(evaluated_at: datetime) -> dict[str, int | str]:
    return {
        "archive_count": 0,
        "cycle_id": str(
            lifecycle_cycle_id(
                evaluated_at=evaluated_at,
                config=LifecycleConfig(),
            )
        ),
        "evaluated_at": evaluated_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "evaluated_count": 0,
        "idea_archive_count": 0,
        "transition_count": 0,
    }


def _receipt_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone()
    assert row is not None
    return int(row[0])


def test_lifecycle_run_uses_the_application_clock_when_time_is_omitted(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-default-time.sqlite3"
    with _client(database_path, ids=(RECEIPT_IDS[0],)) as client:
        response = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-now"},
            json={},
        )

    assert response.status_code == 200
    assert response.content == canonical_json_bytes({"data": _result(NOW)})
    assert _receipt_count(database_path) == 1


def test_lifecycle_run_accepts_an_explicit_evaluation_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-explicit-time.sqlite3"
    with _client(database_path, ids=(RECEIPT_IDS[0],)) as client:
        response = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-earlier"},
            json={"evaluated_at": EARLIER.isoformat()},
        )

    assert response.status_code == 200
    assert response.content == canonical_json_bytes({"data": _result(EARLIER)})


def test_lifecycle_run_requires_an_idempotency_key_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-required-key.sqlite3"
    with _client(database_path, ids=()) as client:
        response = client.post("/v1/lifecycle:run", json={})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0


def test_lifecycle_run_rejects_a_future_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-future.sqlite3"
    future = NOW + timedelta(microseconds=1)
    with _client(database_path, ids=(RECEIPT_IDS[0],)) as client:
        response = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-future"},
            json={"evaluated_at": future.isoformat()},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_evaluated_at",
            "details": {},
            "message": (
                "Lifecycle evaluation time must be UTC-aware and not in the future"
            ),
        }
    }


def test_lifecycle_run_replays_the_exact_canonical_result(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-replay.sqlite3"
    with _client(database_path, ids=(RECEIPT_IDS[0],)) as client:
        first = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-once"},
            json={},
        )
        replay = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-once"},
            json={},
        )

    assert first.status_code == 200
    assert replay.status_code == first.status_code
    assert replay.content == first.content
    assert replay.content == canonical_json_bytes({"data": _result(NOW)})
    assert _receipt_count(database_path) == 1


def test_omitted_time_replays_after_the_application_clock_advances(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-replay-after-clock-advance.sqlite3"
    clock = FrozenClock(NOW)
    app = create_app(
        settings=_settings(database_path),
        clock=clock,
    )
    with TestClient(app) as client:
        first = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "stable-omitted-time"},
            json={},
        )
        clock.advance(timedelta(hours=1))
        replay = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "stable-omitted-time"},
            json={},
        )

    assert first.status_code == replay.status_code == 200
    assert replay.content == first.content
    assert replay.content == canonical_json_bytes({"data": _result(NOW)})
    assert _receipt_count(database_path) == 1


def test_concurrent_omitted_time_requests_share_one_stable_receipt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-concurrent-omitted-time.sqlite3"
    barrier = Barrier(2)
    with _client(database_path, ids=(RECEIPT_IDS[0],)) as client:

        def invoke() -> tuple[int, bytes]:
            barrier.wait()
            response = client.post(
                "/v1/lifecycle:run",
                headers={"Idempotency-Key": "concurrent-omitted-time"},
                json={},
            )
            return response.status_code, response.content

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _: invoke(), range(2)))

    assert results[0][0] == results[1][0] == 200
    assert results[0][1] == results[1][1]
    assert _receipt_count(database_path) == 1
    expected = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key="concurrent-omitted-time",
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": None, "mode": "manual"},
    )
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT request_hash FROM idempotency_records"
        ).fetchone()
    assert row == (expected.request_hash,)


def test_lifecycle_run_rejects_unknown_request_fields_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-strict-body.sqlite3"
    with _client(database_path, ids=()) as client:
        response = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "run-lifecycle-extra"},
            json={"mode": "background"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0


def test_lifecycle_run_rejects_numeric_or_boolean_timestamps_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lifecycle-strict-timestamp.sqlite3"
    with _client(database_path, ids=()) as client:
        numeric = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "numeric-lifecycle-time"},
            json={"evaluated_at": 0},
        )
        boolean = client.post(
            "/v1/lifecycle:run",
            headers={"Idempotency-Key": "boolean-lifecycle-time"},
            json={"evaluated_at": False},
        )

    assert numeric.status_code == boolean.status_code == 422
    assert numeric.json()["error"]["code"] == "validation_error"
    assert boolean.json()["error"]["code"] == "validation_error"
    assert _receipt_count(database_path) == 0
