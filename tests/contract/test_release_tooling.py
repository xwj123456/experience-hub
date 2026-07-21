from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).parents[2]
CHECKOUT_ACTION = (
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
)
SETUP_UV_ACTION = (
    "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990"
)
UPLOAD_ARTIFACT_ACTION = (
    "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
)


def _project_configuration() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        ),
    )


def test_test_dependencies_use_patched_compatible_majors() -> None:
    configuration = _project_configuration()
    development = set(
        cast(
            list[str],
            cast(dict[str, Any], configuration["dependency-groups"])["dev"],
        )
    )

    assert "pytest>=9.0.3,<10" in development
    assert "pytest-asyncio>=1.4,<2" in development
    assert "pytest-cov>=7,<8" in development


def test_default_pytest_options_do_not_enable_coverage() -> None:
    configuration = _project_configuration()
    pytest_options = cast(
        dict[str, Any],
        cast(dict[str, Any], configuration["tool"])["pytest"]["ini_options"],
    )

    assert pytest_options["addopts"] == "-ra"
    assert "--cov" not in cast(str, pytest_options["addopts"])


def _workflow(name: str) -> str:
    path = PROJECT_ROOT / ".github" / "workflows" / name
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_required_ci_is_sharded_without_coverage() -> None:
    workflow = _workflow("ci.yml")

    assert "pull_request:" in workflow
    assert "quality:" in workflow
    assert "tests:" in workflow
    assert "demo:" in workflow
    assert "benchmark:" in workflow
    assert "build:" in workflow
    assert "uv run pytest --no-cov -q" in workflow
    assert "--cov=" not in workflow
    for paths in (
        "tests/unit tests/integration tests/contract",
        "tests/repository",
        "tests/api tests/cli",
        "tests/e2e tests/benchmark",
    ):
        assert paths in workflow
    assert CHECKOUT_ACTION in workflow
    assert SETUP_UV_ACTION in workflow
    assert UPLOAD_ARTIFACT_ACTION in workflow
    assert "sha256sum --check dist/SHA256SUMS" in workflow


def test_coverage_runs_outside_pull_requests() -> None:
    workflow = _workflow("coverage.yml")

    assert "pull_request:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "--cov=experience_hub" in workflow
    assert "--cov-branch" in workflow
    assert "--cov-report=xml" in workflow
    assert CHECKOUT_ACTION in workflow
    assert SETUP_UV_ACTION in workflow
    assert UPLOAD_ARTIFACT_ACTION in workflow
