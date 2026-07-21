from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from typer.testing import CliRunner

from experience_hub.api.app import create_app
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli.app import ApplicationRuntime, app
from experience_hub.config import Settings
from experience_hub.experiences.content import decode_payload
from experience_hub.experiences.models import PayloadCodec
from experience_hub.experiences.reconcile import (
    PayloadReconcileIssue,
    PayloadReconcileReport,
)
from experience_hub.runtime import ApplicationRuntime as SharedApplicationRuntime
from experience_hub.storage.database import Database, payload_rewrite_guard
from experience_hub.storage.payload_rewrite import rewrite_payload_codec
from experience_hub.storage.tables import ExperiencePayloadRow

RUNNER = CliRunner()
cli_app = import_module("experience_hub.cli.app")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000101")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000201")


def _canonical_output(result: Any) -> dict[str, Any]:
    decoded = json.loads(result.stdout)
    assert result.stdout == canonical_json_bytes(decoded).decode("utf-8") + "\n"
    assert isinstance(decoded, dict)
    return decoded


@dataclass(slots=True)
class _RuntimeProbe:
    container: Any
    settings: Any | None = None
    initialize_calls: list[tuple[bool, bool]] = field(default_factory=list)
    exited: bool = False

    def __call__(self, settings: Any, *args: Any, **kwargs: Any) -> _RuntimeProbe:
        assert not args
        assert not kwargs
        self.settings = settings
        return self

    @asynccontextmanager
    async def initialize(
        self,
        *,
        start_lifecycle_worker: bool,
        recover_interrupted: bool,
    ) -> Any:
        self.initialize_calls.append((start_lifecycle_worker, recover_interrupted))
        try:
            yield self.container
        finally:
            self.exited = True


@dataclass(slots=True)
class _Transaction:
    uow: Any
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> Any:
        self.entered = True
        return self.uow

    async def __aexit__(self, *exc_info: Any) -> None:
        self.exited = True


@dataclass(slots=True)
class _Database:
    uow: Any = field(default_factory=object)
    transaction_calls: list[dict[str, Any]] = field(default_factory=list)
    retained_transaction: _Transaction | None = None

    def transaction(self, **kwargs: Any) -> _Transaction:
        self.transaction_calls.append(kwargs)
        self.retained_transaction = _Transaction(self.uow)
        return self.retained_transaction


@dataclass(slots=True)
class _PayloadReconciler:
    report: PayloadReconcileReport
    uows: list[Any] = field(default_factory=list)

    async def run(self, *, uow: Any) -> PayloadReconcileReport:
        self.uows.append(uow)
        return self.report


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    report: PayloadReconcileReport,
) -> tuple[Any, _RuntimeProbe, _Database, _PayloadReconciler]:
    database = _Database()
    reconciler = _PayloadReconciler(report)
    runtime = _RuntimeProbe(
        SimpleNamespace(
            database=database,
            payload_reconciler=reconciler,
        )
    )
    monkeypatch.setattr(cli_app, "ApplicationRuntime", runtime)
    result = RUNNER.invoke(app, ["payloads", "reconcile"])
    return result, runtime, database, reconciler


def test_cli_uses_the_shared_application_runtime_boundary() -> None:
    assert ApplicationRuntime is SharedApplicationRuntime


def test_payload_reconcile_prints_exact_counts_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = PayloadReconcileReport(
        changed_count=2,
        skipped_count=3,
        error_count=0,
    )

    result, runtime, database, reconciler = _invoke(monkeypatch, report)

    assert result.exit_code == 0
    assert _canonical_output(result) == {
        "data": {
            "changed_count": 2,
            "error_count": 0,
            "errors": [],
            "skipped_count": 3,
        }
    }
    assert database.transaction_calls == [{"immediate": True}]
    assert database.retained_transaction is not None
    assert database.retained_transaction.entered
    assert database.retained_transaction.exited
    assert reconciler.uows == [database.uow]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


def test_payload_semantic_hash_error_is_reported_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue = PayloadReconcileIssue(
        experience_id=EXPERIENCE_ID,
        version_number=1,
        version_id=VERSION_ID,
        code="semantic_validation_failed",
    )
    report = PayloadReconcileReport(
        changed_count=0,
        skipped_count=4,
        error_count=1,
        errors=(issue,),
    )

    result, runtime, database, reconciler = _invoke(monkeypatch, report)

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "data": {
            "changed_count": 0,
            "error_count": 1,
            "errors": [
                {
                    "code": "semantic_validation_failed",
                    "experience_id": str(EXPERIENCE_ID),
                    "version_id": str(VERSION_ID),
                    "version_number": 1,
                }
            ],
            "skipped_count": 4,
        }
    }
    assert database.transaction_calls == [{"immediate": True}]
    assert database.retained_transaction is not None
    assert database.retained_transaction.entered
    assert database.retained_transaction.exited
    assert reconciler.uows == [database.uow]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


def test_payload_reconcile_operates_on_a_fresh_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "payload-reconcile.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert _canonical_output(result) == {
        "data": {
            "changed_count": 0,
            "error_count": 0,
            "errors": [],
            "skipped_count": 0,
        }
    }
    with sqlite3.connect(database_path) as connection:
        revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
    assert revision is not None


async def _corrupt_payload(
    *,
    settings: Settings,
    version_id: UUID,
) -> bytes:
    assert settings.database_url is not None
    database = Database.create(settings.database_url)
    try:
        async with database.transaction(immediate=True) as uow:
            original = await uow.session.scalar(
                select(ExperiencePayloadRow.payload).where(
                    ExperiencePayloadRow.version_id == version_id
                )
            )
            assert original is not None
            connection = await uow.session.connection()
            with payload_rewrite_guard(connection):
                await uow.session.execute(
                    update(ExperiencePayloadRow)
                    .where(ExperiencePayloadRow.version_id == version_id)
                    .values(payload=b"corrupt canonical body")
                )
        return bytes(original)
    finally:
        await database.dispose()


async def _rewrite_codec(
    *,
    settings: Settings,
    version_id: UUID,
    codec: PayloadCodec,
) -> None:
    assert settings.database_url is not None
    database = Database.create(settings.database_url)
    try:
        async with database.transaction(immediate=True) as uow:
            await rewrite_payload_codec(
                session=uow.session,
                version_id=version_id,
                codec=codec,
            )
    finally:
        await database.dispose()


def _seed_experiences(
    database_path: Path,
    *,
    count: int,
) -> tuple[Settings, tuple[dict[str, Any], ...]]:
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    api = create_app(settings=settings)
    with TestClient(api) as client:
        agent_response = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "payload-agent"},
            json={"name": "Payload owner"},
        )
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["data"]["agent_id"]
        experiences: list[dict[str, Any]] = []
        for index in range(count):
            experience_response = client.post(
                f"/v1/agents/{agent_id}/experiences",
                headers={"Idempotency-Key": f"payload-experience-{index}"},
                json={
                    "applicability": ["local maintenance"],
                    "body": f"Semantic payload {index} must fail closed.",
                    "confidence": 0.8,
                    "evidence": [{"id": f"payload-test-{index}", "type": "test"}],
                    "falsifiers": ["The corrupt payload is accepted."],
                    "importance": 0.7,
                    "kind": "semantic",
                    "links": [],
                    "mechanism": (
                        f"Recompute retained semantic hashes for version {index}."
                    ),
                    "summary": f"Payload hash {index} preserves meaning.",
                    "tags": ["payload", "integrity"],
                },
            )
            assert experience_response.status_code == 201
            experiences.append(experience_response.json()["data"])
    return settings, tuple(experiences)


def test_payload_reconcile_aggregates_real_corruption_without_rewriting(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "payload-semantic-corruption.sqlite3"
    settings, experiences = _seed_experiences(database_path, count=3)

    corrupted = tuple(
        (
            UUID(experience["experience_id"]),
            UUID(experience["version_id"]),
        )
        for experience in experiences[:2]
    )
    for _, version_id in corrupted:
        asyncio.run(_corrupt_payload(settings=settings, version_id=version_id))
    with sqlite3.connect(database_path) as connection:
        snapshot_before = (
            connection.execute(
                "SELECT version_id, codec, payload, payload_hash "
                "FROM experience_payloads ORDER BY version_id"
            ).fetchall(),
            connection.execute("SELECT COUNT(*) FROM domain_events").fetchone(),
            connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone(),
        )

    result = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "data": {
            "changed_count": 0,
            "error_count": 2,
            "errors": [
                {
                    "code": "semantic_validation_failed",
                    "experience_id": str(experience_id),
                    "version_id": str(version_id),
                    "version_number": 1,
                }
                for experience_id, version_id in sorted(
                    corrupted,
                    key=lambda item: item[0].int,
                )
            ],
            "skipped_count": 1,
        }
    }
    with sqlite3.connect(database_path) as connection:
        snapshot_after = (
            connection.execute(
                "SELECT version_id, codec, payload, payload_hash "
                "FROM experience_payloads ORDER BY version_id"
            ).fetchall(),
            connection.execute("SELECT COUNT(*) FROM domain_events").fetchone(),
            connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone(),
        )
    assert snapshot_after == snapshot_before
    assert (
        sum(
            payload == b"corrupt canonical body"
            for _, _, payload, _ in snapshot_after[0]
        )
        == 2
    )


def test_payload_reconcile_repairs_a_real_codec_mismatch_idempotently(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "payload-codec-repair.sqlite3"
    settings, experiences = _seed_experiences(database_path, count=1)
    version_id = UUID(experiences[0]["version_id"])
    asyncio.run(
        _rewrite_codec(
            settings=settings,
            version_id=version_id,
            codec=PayloadCodec.ZLIB,
        )
    )
    with sqlite3.connect(database_path) as connection:
        forced = connection.execute(
            "SELECT codec, payload, payload_hash FROM experience_payloads "
            "WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        content_hash = connection.execute(
            "SELECT content_hash FROM experience_versions WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        counts_before = (
            connection.execute("SELECT COUNT(*) FROM domain_events").fetchone(),
            connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone(),
        )
    assert forced is not None
    assert forced[0] == PayloadCodec.ZLIB.value
    decoded_before = decode_payload(PayloadCodec(forced[0]), bytes(forced[1]))

    first = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )
    second = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )

    assert first.exit_code == second.exit_code == 0
    assert _canonical_output(first) == {
        "data": {
            "changed_count": 1,
            "error_count": 0,
            "errors": [],
            "skipped_count": 0,
        }
    }
    assert _canonical_output(second) == {
        "data": {
            "changed_count": 0,
            "error_count": 0,
            "errors": [],
            "skipped_count": 1,
        }
    }
    with sqlite3.connect(database_path) as connection:
        repaired = connection.execute(
            "SELECT codec, payload, payload_hash FROM experience_payloads "
            "WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        retained_content_hash = connection.execute(
            "SELECT content_hash FROM experience_versions WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        counts_after = (
            connection.execute("SELECT COUNT(*) FROM domain_events").fetchone(),
            connection.execute("SELECT COUNT(*) FROM idempotency_records").fetchone(),
        )
    assert repaired is not None
    assert repaired[0] == PayloadCodec.PLAIN.value
    assert repaired[2] == forced[2]
    assert (
        decode_payload(PayloadCodec(repaired[0]), bytes(repaired[1])) == decoded_before
    )
    assert retained_content_hash == content_hash
    assert counts_after == counts_before


@pytest.mark.parametrize(
    ("damage", "expected_code"),
    (("payload", "missing_payload"), ("state", "missing_state")),
)
def test_payload_reconcile_uses_one_report_for_missing_rows(
    tmp_path: Path,
    damage: str,
    expected_code: str,
) -> None:
    database_path = tmp_path / f"payload-missing-{damage}.sqlite3"
    _, experiences = _seed_experiences(database_path, count=1)
    experience_id = UUID(experiences[0]["experience_id"])
    version_id = UUID(experiences[0]["version_id"])
    with sqlite3.connect(database_path) as connection:
        if damage == "payload":
            connection.execute("DROP TRIGGER experience_payloads_reject_delete")
            connection.execute(
                "DELETE FROM experience_payloads WHERE version_id = ?",
                (str(version_id),),
            )
        else:
            connection.execute(
                "DELETE FROM experience_state WHERE experience_id = ?",
                (str(experience_id),),
            )

    result = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "data": {
            "changed_count": 0,
            "error_count": 1,
            "errors": [
                {
                    "code": expected_code,
                    "experience_id": str(experience_id),
                    "version_id": str(version_id),
                    "version_number": 1,
                }
            ],
            "skipped_count": 0,
        }
    }


def test_non_payload_source_damage_keeps_the_generic_integrity_error(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "payload-noncanonical-metadata.sqlite3"
    _, experiences = _seed_experiences(database_path, count=1)
    version_id = UUID(experiences[0]["version_id"])
    with sqlite3.connect(database_path) as connection:
        payload_before = connection.execute(
            "SELECT codec, payload, payload_hash FROM experience_payloads "
            "WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        tags = connection.execute(
            "SELECT tags FROM experience_versions WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
        assert tags is not None
        connection.execute("DROP TRIGGER experience_versions_reject_update")
        connection.execute(
            "UPDATE experience_versions SET tags = ? WHERE version_id = ?",
            (b" " + bytes(tags[0]), str(version_id)),
        )

    result = RUNNER.invoke(
        app,
        [
            "payloads",
            "reconcile",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "error": {
            "code": "source_integrity_error",
            "details": {"mismatch_key": f"experience_version_payload:{version_id}"},
            "message": "Authoritative source integrity validation failed",
        }
    }
    with sqlite3.connect(database_path) as connection:
        payload_after = connection.execute(
            "SELECT codec, payload, payload_hash FROM experience_payloads "
            "WHERE version_id = ?",
            (str(version_id),),
        ).fetchone()
    assert payload_after == payload_before
