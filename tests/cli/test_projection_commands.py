from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli.app import ApplicationRuntime, app
from experience_hub.config import Settings, repository_root
from experience_hub.runtime import ApplicationRuntime as SharedApplicationRuntime
from experience_hub.storage.database import DatabaseBusy
from experience_hub.storage.projections import (
    MaintenanceBlockedByInflight,
    ProjectionDiff,
    ProjectionMismatch,
    VerificationReport,
)

RUNNER = CliRunner()
cli_app = import_module("experience_hub.cli.app")
EVENT_HEAD = 19
DIFFERENCE = ProjectionDiff(
    projection="experience_state",
    online_hash="a" * 64,
    rebuilt_hash="b" * 64,
    differing_keys=("00000000-0000-0000-0000-000000000101",),
)


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
class _ProjectionManager:
    verify_outcome: VerificationReport | BaseException
    repair_outcome: VerificationReport | BaseException
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def verify(self, database: Any) -> VerificationReport:
        self.calls.append(("verify", database))
        if isinstance(self.verify_outcome, BaseException):
            raise self.verify_outcome
        return self.verify_outcome

    async def repair(self, database: Any) -> VerificationReport:
        self.calls.append(("repair", database))
        if isinstance(self.repair_outcome, BaseException):
            raise self.repair_outcome
        return self.repair_outcome


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    manager: _ProjectionManager,
    *arguments: str,
) -> tuple[Any, _RuntimeProbe, Any]:
    database = object()
    runtime = _RuntimeProbe(
        SimpleNamespace(
            database=database,
            projection_manager=manager,
        )
    )
    monkeypatch.setattr(cli_app, "ApplicationRuntime", runtime)
    result = RUNNER.invoke(app, ["projections", "rebuild", *arguments])
    return result, runtime, database


def test_cli_uses_the_shared_application_runtime_boundary() -> None:
    assert ApplicationRuntime is SharedApplicationRuntime


@pytest.mark.parametrize(
    ("flag", "operation"),
    (("--verify", "verify"), ("--repair", "repair")),
)
def test_projection_rebuild_success_has_canonical_report_and_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    operation: str,
) -> None:
    report = VerificationReport(event_head=EVENT_HEAD, differences=())
    manager = _ProjectionManager(report, report)

    result, runtime, database = _invoke(monkeypatch, manager, flag)

    assert result.exit_code == 0
    assert _canonical_output(result) == {
        "data": {
            "differences": [],
            "event_head": EVENT_HEAD,
            "matches": True,
            "mode": operation,
        }
    }
    assert manager.calls == [(operation, database)]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


def test_projection_verify_mismatch_reports_bounded_keys_and_exits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = VerificationReport(
        event_head=EVENT_HEAD,
        differences=(DIFFERENCE,),
    )
    manager = _ProjectionManager(
        ProjectionMismatch(report),
        VerificationReport(event_head=EVENT_HEAD, differences=()),
    )

    result, runtime, database = _invoke(monkeypatch, manager, "--verify")

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "error": {
            "code": "projection_mismatch",
            "details": {
                "differences": [
                    {
                        "differing_keys": list(DIFFERENCE.differing_keys),
                        "online_hash": DIFFERENCE.online_hash,
                        "projection": DIFFERENCE.projection,
                        "rebuilt_hash": DIFFERENCE.rebuilt_hash,
                    }
                ],
                "event_head": EVENT_HEAD,
            },
            "message": "Projection verification failed",
        }
    }
    assert manager.calls == [("verify", database)]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


def test_projection_repair_maps_inflight_refusal_to_stable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = VerificationReport(event_head=EVENT_HEAD, differences=())
    manager = _ProjectionManager(
        report,
        MaintenanceBlockedByInflight("sensitive implementation detail"),
    )

    result, runtime, database = _invoke(monkeypatch, manager, "--repair")

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "error": {
            "code": "maintenance_blocked_by_inflight",
            "details": {},
            "message": "Projection repair is blocked by in-progress work",
        }
    }
    assert "sensitive implementation detail" not in result.output
    assert manager.calls == [("repair", database)]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


def test_projection_repair_maps_database_busy_to_stable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = VerificationReport(event_head=EVENT_HEAD, differences=())
    manager = _ProjectionManager(report, DatabaseBusy())

    result, runtime, database = _invoke(monkeypatch, manager, "--repair")

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "error": {
            "code": "database_busy",
            "details": {"retry_after": 5},
            "message": "The database is busy; retry the request",
        }
    }
    assert manager.calls == [("repair", database)]
    assert runtime.initialize_calls == [(False, False)]
    assert runtime.exited


@pytest.mark.parametrize(
    "arguments",
    ((), ("--verify", "--repair")),
)
def test_projection_rebuild_requires_exactly_one_mode(
    arguments: tuple[str, ...],
) -> None:
    result = RUNNER.invoke(app, ["projections", "rebuild", *arguments])

    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("flag", "mode"),
    (("--verify", "verify"), ("--repair", "repair")),
)
def test_projection_rebuild_operates_on_a_fresh_database(
    tmp_path: Path,
    flag: str,
    mode: str,
) -> None:
    database_path = tmp_path / f"projection-{mode}.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "projections",
            "rebuild",
            flag,
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert _canonical_output(result) == {
        "data": {
            "differences": [],
            "event_head": 0,
            "matches": True,
            "mode": mode,
        }
    }
    with sqlite3.connect(database_path) as connection:
        revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
    assert revision is not None


def test_database_path_with_url_query_syntax_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory?timeout=5"
    silently_truncated_path = tmp_path / "memory"

    result = RUNNER.invoke(
        app,
        [
            "projections",
            "rebuild",
            "--verify",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 2
    assert not database_path.exists()
    assert not silently_truncated_path.exists()


def test_migration_lock_is_reported_as_database_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration-busy.sqlite3"
    config = Config(str(repository_root() / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "0004_inspiration")
    locked_settings = Settings(
        database_url=(f"sqlite+aiosqlite:///{database_path}?timeout=0.01")
    )
    monkeypatch.setattr(cli_app, "_settings", lambda _: locked_settings)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        result = RUNNER.invoke(
            app,
            [
                "projections",
                "rebuild",
                "--verify",
                "--database",
                str(database_path),
            ],
        )
    finally:
        connection.rollback()
        connection.close()

    assert result.exit_code == 1
    assert _canonical_output(result) == {
        "error": {
            "code": "database_busy",
            "details": {"retry_after": 5},
            "message": "The database is busy; retry the request",
        }
    }
