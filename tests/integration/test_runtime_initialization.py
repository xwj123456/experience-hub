from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text

from experience_hub.bootstrap import ApplicationContainer
from experience_hub.config import Settings
from experience_hub.experiences.reconcile_contracts import (
    PayloadReconcileIssue,
    PayloadReconcileReport,
)
from experience_hub.runtime import ApplicationRuntime, SchemaRevisionError
from experience_hub.storage.projections import ReducerVersionMismatch
from experience_hub.storage.tables import ProjectionVersionRow
from experience_hub.storage.validation import (
    PayloadReconcileValidationError,
)


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


@pytest.mark.asyncio
async def test_runtime_migrates_and_initializes_a_fresh_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "fresh.sqlite3"
    runtime = ApplicationRuntime(settings=_settings(database_path))
    retained: ApplicationContainer | None = None

    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        retained = container
        assert container.schema_revision == "0005_inspiration_falsifiers"
        assert container.lifecycle_worker.running is False
        async with container.database.read_session() as session:
            version = await session.scalar(
                text("SELECT version_num FROM alembic_version")
            )
            agent_table = await session.scalar(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'agents'"
                )
            )
        assert version == container.schema_revision
        assert agent_table == "agents"

    assert retained is not None
    assert retained.closed


@pytest.mark.asyncio
async def test_relative_database_url_uses_one_file_for_migrations_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        database_url="sqlite+aiosqlite:///relative-experience-hub.sqlite3"
    )

    async with (
        ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container,
        container.database.read_session() as session,
    ):
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version"))
            == container.schema_revision
        )

    assert (tmp_path / "relative-experience-hub.sqlite3").is_file()


@pytest.mark.asyncio
async def test_runtime_closes_every_resource_when_command_body_fails(
    tmp_path: Path,
) -> None:
    runtime = ApplicationRuntime(settings=_settings(tmp_path / "failure.sqlite3"))
    retained: ApplicationContainer | None = None

    with pytest.raises(RuntimeError, match="command failed"):
        async with runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container:
            retained = container
            raise RuntimeError("command failed")

    assert retained is not None
    assert retained.closed


@pytest.mark.asyncio
async def test_shutdown_stops_worker_then_runs_hooks_before_engine_disposal(
    tmp_path: Path,
) -> None:
    observations: list[tuple[bool, str | None]] = []
    runtime = ApplicationRuntime(_settings(tmp_path / "shutdown-order.sqlite3"))

    async with runtime.initialize(
        start_lifecycle_worker=True,
        recover_interrupted=False,
    ) as container:
        assert container.lifecycle_worker.running

        async def observe_shutdown() -> None:
            async with container.database.read_session() as session:
                revision = await session.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
            observations.append((container.lifecycle_worker.running, revision))

        container.register_shutdown_hook(observe_shutdown)

    assert observations == [(False, "0005_inspiration_falsifiers")]


@pytest.mark.asyncio
async def test_runtime_checks_reducer_compatibility_without_projection_verify(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "projection-health.sqlite3")
    async with (
        ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container,
        container.database.transaction(immediate=True) as uow,
    ):
        uow.session.add(
            ProjectionVersionRow(
                name="experience_state",
                reducer_version=1,
                last_applied_event_id=999,
            )
        )

    # A stale projection checkpoint must not prevent the maintenance command
    # from entering the runtime and repairing it.
    async with ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        report = await container.projection_manager.repair(container.database)
        assert report.matches

    async with (
        ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container,
        container.database.transaction(immediate=True) as uow,
    ):
        version = await uow.session.get(
            ProjectionVersionRow,
            "experience_state",
        )
        assert version is not None
        version.reducer_version = 99

    with pytest.raises(ReducerVersionMismatch):
        async with ApplicationRuntime(settings).initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ):
            pytest.fail("an unsupported reducer version must prevent readiness")


@pytest.mark.asyncio
async def test_runtime_refuses_an_unknown_newer_schema_without_rewriting_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "future.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version "
            "(version_num VARCHAR(255) NOT NULL PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO alembic_version(version_num) VALUES (?)",
            ("9999_future",),
        )

    runtime = ApplicationRuntime(settings=_settings(database_path))
    with pytest.raises(SchemaRevisionError) as raised:
        async with runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ):
            pytest.fail("a future schema must never become ready")

    assert raised.value.current_revision == "9999_future"
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone() == ("9999_future",)


class _RecordingWorker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.running = False

    def start(self) -> None:
        self.calls.append("worker.start")
        self.running = True


class _RecordingProjectionManager:
    def __init__(
        self,
        calls: list[str],
        *,
        fail: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.calls = calls
        self.fail = fail
        self.error = error

    async def validate_startup(self, database: object) -> None:
        assert database is not None
        self.calls.append("projections.validate")
        if self.error is not None:
            raise self.error
        if self.fail:
            raise RuntimeError("invalid sources")


class _RecordingRecovery:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def recover(self) -> tuple[object, ...]:
        self.calls.append("recovery.run")
        return ()


class _RecordingContainer:
    def __init__(
        self,
        calls: list[str],
        *,
        fail_validation: bool = False,
        validation_error: BaseException | None = None,
    ) -> None:
        self.calls = calls
        self.database = object()
        self.projection_manager = _RecordingProjectionManager(
            calls,
            fail=fail_validation,
            error=validation_error,
        )
        self.inspiration_recovery = _RecordingRecovery(calls)
        self.lifecycle_worker = _RecordingWorker(calls)
        self.schema_revision: str | None = None

    async def close(self) -> None:
        self.calls.append("container.close")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recover_interrupted", "start_worker", "expected"),
    [
        (
            False,
            False,
            [
                "container.build",
                "schema.migrate",
                "projections.validate",
                "command.body",
                "container.close",
            ],
        ),
        (
            True,
            True,
            [
                "container.build",
                "schema.migrate",
                "projections.validate",
                "recovery.run",
                "worker.start",
                "command.body",
                "container.close",
            ],
        ),
    ],
)
async def test_runtime_applies_selected_recovery_and_worker_policy_before_body(
    tmp_path: Path,
    recover_interrupted: bool,
    start_worker: bool,
    expected: list[str],
) -> None:
    calls: list[str] = []

    def factory(**_: Any) -> _RecordingContainer:
        calls.append("container.build")
        return _RecordingContainer(calls)

    async def migrator(_: Settings) -> str:
        calls.append("schema.migrate")
        return "test_head"

    runtime = ApplicationRuntime(
        settings=_settings(tmp_path / "policy.sqlite3"),
        container_factory=factory,
        migrator=migrator,
    )
    async with runtime.initialize(
        start_lifecycle_worker=start_worker,
        recover_interrupted=recover_interrupted,
    ) as container:
        assert container.schema_revision == "test_head"
        calls.append("command.body")

    assert calls == expected


@pytest.mark.asyncio
async def test_runtime_closes_container_when_startup_validation_fails(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def factory(**_: Any) -> _RecordingContainer:
        calls.append("container.build")
        return _RecordingContainer(calls, fail_validation=True)

    async def migrator(_: Settings) -> str:
        calls.append("schema.migrate")
        return "test_head"

    runtime = ApplicationRuntime(
        settings=_settings(tmp_path / "invalid.sqlite3"),
        container_factory=factory,
        migrator=migrator,
    )
    with pytest.raises(RuntimeError, match="invalid sources"):
        async with runtime.initialize(
            start_lifecycle_worker=True,
            recover_interrupted=True,
        ):
            pytest.fail("invalid sources must prevent readiness")

    assert calls == [
        "container.build",
        "schema.migrate",
        "projections.validate",
        "container.close",
    ]


@pytest.mark.asyncio
async def test_payload_validation_report_still_closes_before_runtime_yields(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    report = PayloadReconcileReport(
        changed_count=0,
        skipped_count=1,
        error_count=1,
        errors=(
            PayloadReconcileIssue(
                experience_id=UUID("00000000-0000-0000-0000-000000000101"),
                version_number=1,
                version_id=UUID("00000000-0000-0000-0000-000000000201"),
                code="semantic_validation_failed",
            ),
        ),
    )

    def factory(**_: Any) -> Any:
        calls.append("container.build")
        return _RecordingContainer(
            calls,
            validation_error=PayloadReconcileValidationError(report),
        )

    async def migrator(_: Settings) -> str:
        calls.append("schema.migrate")
        return "test_head"

    runtime = ApplicationRuntime(
        settings=_settings(tmp_path / "payload-invalid.sqlite3"),
        container_factory=factory,
        migrator=migrator,
    )
    with pytest.raises(PayloadReconcileValidationError) as caught:
        async with runtime.initialize(
            start_lifecycle_worker=True,
            recover_interrupted=True,
        ):
            pytest.fail("payload source damage must prevent readiness")

    assert caught.value.report is report
    assert calls == [
        "container.build",
        "schema.migrate",
        "projections.validate",
        "container.close",
    ]
