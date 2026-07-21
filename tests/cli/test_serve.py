from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import uvicorn
from fastapi import FastAPI
from typer import Typer
from typer.testing import CliRunner

from experience_hub.cli import app as package_app
from experience_hub.cli.app import app

RUNNER = CliRunner()


def test_package_level_app_export_remains_the_typer_application() -> None:
    assert isinstance(package_app, Typer)
    assert package_app is app


def test_root_help_lists_the_complete_operational_command_tree() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Experience Hub operations." in result.output
    for command in (
        "serve",
        "lifecycle",
        "projections",
        "payloads",
        "demo",
        "benchmark",
    ):
        assert command in result.output


@pytest.mark.parametrize(
    "arguments",
    (
        ("lifecycle", "run", "--help"),
        ("projections", "rebuild", "--help"),
        ("payloads", "reconcile", "--help"),
        ("demo", "--help"),
        ("benchmark", "--help"),
    ),
)
def test_every_planned_command_has_help(arguments: tuple[str, ...]) -> None:
    result = RUNNER.invoke(app, list(arguments))

    assert result.exit_code == 0, result.output


def test_demo_help_exposes_reset_option() -> None:
    result = RUNNER.invoke(app, ["demo", "--help"])

    assert result.exit_code == 0, result.output
    assert "--reset" in result.output


def test_serve_help_locks_host_port_and_database_options() -> None:
    result = RUNNER.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--database" in result.output


def test_serve_passes_a_lazy_database_bound_factory_to_uvicorn(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    database_path = tmp_path / "custom serve.sqlite3"
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_uvicorn_run(application: object, **kwargs: object) -> None:
        calls.append((application, kwargs))
        assert callable(application)
        factory = cast(Callable[[], FastAPI], application)
        created = factory()
        assert isinstance(created, FastAPI)
        assert created.state.runtime.settings.database_url == (
            f"sqlite+aiosqlite:///{database_path}"
        )
        assert not database_path.exists()

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    result = RUNNER.invoke(
        app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    application, kwargs = calls[0]
    assert callable(application)
    assert kwargs["factory"] is True
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8765
    assert not database_path.exists()
