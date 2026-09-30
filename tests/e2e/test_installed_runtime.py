from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["UV_OFFLINE"] = "true"
    return environment


def _run(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.fixture(scope="module")
def installed_environment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workspace = tmp_path_factory.mktemp("installed-runtime")
    wheel_directory = workspace / "wheels"
    requirements = workspace / "requirements.txt"
    environment = workspace / "environment"
    commands = (
        ["uv", "build", "--wheel", "--out-dir", str(wheel_directory)],
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements),
        ],
        ["uv", "venv", "--python", sys.executable, str(environment)],
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(environment / "bin/python"),
            str(requirements),
        ],
    )
    for arguments in commands:
        result = _run(arguments, cwd=PROJECT_ROOT)
        assert result.returncode == 0, result.stderr
    (wheel,) = wheel_directory.glob("*.whl")
    result = _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment / "bin/python"),
            "--no-deps",
            str(wheel),
        ],
        cwd=workspace,
    )
    assert result.returncode == 0, result.stderr
    return environment


def test_installed_console_ignores_unrelated_project(
    installed_environment: Path, tmp_path: Path
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "unrelated-project"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )

    result = _run(
        [str(installed_environment / "bin/experience-hub"), "--help"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "Experience Hub" in result.stdout
    assert not (tmp_path / ".data").exists()


def test_installed_runtime_ignores_unrelated_project(
    installed_environment: Path, tmp_path: Path
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "unrelated-project"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    program = dedent(
        """
        import asyncio
        import json
        import sys
        from sqlalchemy import select
        from sqlalchemy.engine import URL
        from experience_hub.agents import CreateAgent
        from experience_hub.config import Settings
        from experience_hub.domain import CommandRequest
        from experience_hub.runtime import ApplicationRuntime, installed_schema_head
        from experience_hub.storage.tables import AgentRow

        async def main():
            settings = Settings(database_url=URL.create(
                "sqlite+aiosqlite", database=sys.argv[1]
            ))
            async with ApplicationRuntime(settings).initialize(
                start_lifecycle_worker=False, recover_interrupted=False
            ) as container:
                async def handler(uow, command):
                    return await container.agent_service.create(
                        uow=uow, command=CreateAgent(name="Portable owner"),
                        command_context=command
                    )
                request = CommandRequest(
                    caller_scope="system:local", operation_scope="agent.create",
                    idempotency_key="portable-owner", method="POST",
                    route_template="/v1/agents", body={"name": "Portable owner"}
                )
                result = await container.command_executor.execute(request, handler)
                assert result.status_code == 201
                async with container.database.read_session() as session:
                    names = list(await session.scalars(select(AgentRow.name)))
                print(json.dumps({
                    "names": names,
                    "schema_matches_head": (
                        container.schema_revision == installed_schema_head()
                    ),
                }))

        asyncio.run(main())
        """
    )
    database = tmp_path / "portable?exact.sqlite3"

    result = _run(
        [str(installed_environment / "bin/python"), "-c", program, str(database)],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "names": ["Portable owner"],
        "schema_matches_head": True,
    }
    assert database.is_file()
    assert not (tmp_path / "portable").exists()
