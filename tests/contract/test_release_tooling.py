from __future__ import annotations

import posixpath
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
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
_MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _project_configuration() -> dict[str, Any]:
    return tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
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


def _workflow_job(workflow: str, job_name: str) -> str:
    marker = f"\n  {job_name}:\n"
    start = workflow.index(marker) + len(marker)
    remaining = workflow[start:]
    next_job = re.search(r"(?m)^  [^\s][^:\n]*:\n", remaining)
    return remaining if next_job is None else remaining[: next_job.start()]


def _broken_local_markdown_links(files: dict[str, bytes]) -> tuple[str, ...]:
    broken: list[str] = []
    for source_path, raw in files.items():
        if not source_path.endswith(".md"):
            continue
        for match in _MARKDOWN_LINK.finditer(raw.decode("utf-8")):
            target = match.group(1)
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_without_fragment = target.split("#", maxsplit=1)[0]
            resolved = posixpath.normpath(
                posixpath.join(
                    posixpath.dirname(source_path),
                    path_without_fragment,
                )
            )
            if resolved not in files:
                broken.append(f"{source_path} -> {target}")
    return tuple(broken)


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
    quality_job = _workflow_job(workflow, "quality")
    assert "uv run experience-hub release verify" in quality_job
    assert quality_job.index("uv run ruff check .") < quality_job.index(
        "uv run mypy src"
    ) < quality_job.index("uv run experience-hub release verify")


def test_quality_job_verifies_release_evidence_after_static_checks() -> None:
    quality_job = _workflow_job(_workflow("ci.yml"), "quality")

    assert quality_job.index("uv run ruff check .") < quality_job.index(
        "uv run mypy src"
    ) < quality_job.index("uv run experience-hub release verify")


def test_quality_job_does_not_match_release_verify_in_another_job() -> None:
    workflow = _workflow("ci.yml")
    release_step = (
        "      - name: Verify release evidence\n"
        "        run: uv run experience-hub release verify\n"
    )
    without_release_step = workflow.replace(release_step, "")
    tests_start = without_release_step.index("\n  tests:\n")
    tests_steps = without_release_step.index("\n    steps:\n", tests_start)
    relocated = (
        without_release_step[: tests_steps + len("\n    steps:\n")]
        + release_step
        + without_release_step[tests_steps + len("\n    steps:\n") :]
    )

    assert "uv run experience-hub release verify" not in _workflow_job(
        relocated, "quality"
    )


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


def test_build_artifacts_exclude_private_files_and_retain_public_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        PROJECT_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".data",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".internal",
            ".venv",
            ".worktrees",
            "__pycache__",
            "build",
            "dist",
        ),
    )
    sentinels = (
        ".data/private.sqlite3",
        ".internal/internal-review.md",
        ".worktrees/private-worktree.txt",
        "build/generated.txt",
        "dist/stale.whl",
        "htmlcov/index.html",
        "local.db",
        "local.sqlite",
        "local.sqlite3",
        "AGENTS.md",
        ".coverage",
        ".coverage.local",
        "coverage.xml",
    )
    for relative_path in sentinels:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private build sentinel\n", encoding="utf-8")

    private_path_sentinel = "/pri" + "vate/release-owner/source.sqlite3"
    credential_sentinel = "api_" + "key=release-package-credential"
    sensitive_body = (
        f"{private_path_sentinel}\n{credential_sentinel}\n"
    ).encode()
    sensitive_sentinels = (
        ".env",
        ".env.local",
        ".secrets/provider.env",
        ".local/session.json",
        ".cache/provider/result.json",
        "provider-secrets.txt",
    )
    for relative_path in sensitive_sentinels:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sensitive_body)
    (source / ".gitignore").unlink()

    output = tmp_path / "artifacts"
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    sdists = tuple(output.glob("experience_hub-*.tar.gz"))
    wheels = tuple(output.glob("experience_hub-*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1

    with tarfile.open(sdists[0], mode="r:gz") as tar_archive:
        sdist_files = {
            "/".join(Path(member.name).parts[1:]): extracted.read()
            for member in tar_archive.getmembers()
            if member.isfile()
            for extracted in (tar_archive.extractfile(member),)
            if extracted is not None
        }
        sdist_paths = tuple(sdist_files)
        retained_text = b"".join(sdist_files.values())

    with zipfile.ZipFile(wheels[0]) as wheel_archive:
        wheel_paths = tuple(wheel_archive.namelist())
        wheel_text = b"".join(wheel_archive.read(path) for path in wheel_paths)

    forbidden_prefixes = (
        ".cache/",
        ".data/",
        ".local/",
        ".secrets/",
        ".internal/",
        ".worktrees/",
        "build/",
        "dist/",
        "docs/internal/",
        "htmlcov/",
    )
    forbidden_names = {
        "AGENTS.md",
        ".coverage",
        ".env",
        ".env.local",
        "coverage.xml",
        "provider-secrets.txt",
    }

    def leaked(paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            path
            for path in paths
            if path in forbidden_names
            or path.startswith(forbidden_prefixes)
            or path.startswith(".coverage.")
            or path.endswith((".db", ".sqlite", ".sqlite3"))
        )

    assert leaked(sdist_paths) == ()
    assert leaked(wheel_paths) == ()
    assert set(
        (
            "README.md",
            "alembic.ini",
            "benchmarks/cases.jsonl",
            "benchmarks/seed.json",
            "docs/architecture/capture-contracts.md",
            "docs/architecture/replay-contracts.md",
            "docs/evidence/release-evidence.json",
            "docs/rfcs/2026-07-22-experience-replay-lab.md",
            "examples/replay/smoke-cases.jsonl",
            "examples/replay/smoke-manifest.json",
            "examples/trajectories/coding-agent-recovery.jsonl",
            "pyproject.toml",
            "src/experience_hub/capture/source_integrity.py",
            "src/experience_hub/storage/migrations/versions/"
            "0007_capture_evidence_hashes.py",
            "tests/contract/test_release_tooling.py",
            "uv.lock",
        )
    ).issubset(sdist_paths)
    assert str(PROJECT_ROOT).encode() not in retained_text
    assert str(source).encode() not in retained_text
    assert str(PROJECT_ROOT).encode() not in wheel_text
    assert str(source).encode() not in wheel_text
    assert private_path_sentinel.encode() not in retained_text
    assert private_path_sentinel.encode() not in wheel_text
    assert credential_sentinel.encode() not in retained_text
    assert credential_sentinel.encode() not in wheel_text
    assert _broken_local_markdown_links(sdist_files) == ()

    wheel_root = "experience_hub-0.1.0.dist-info"
    assert {path.split("/", maxsplit=1)[0] for path in wheel_paths} == {
        "experience_hub",
        wheel_root,
    }
    package_paths = {
        path for path in wheel_paths if path.startswith("experience_hub/")
    }
    assert all(path.endswith((".py", ".mako")) for path in package_paths)
    expected_package_paths = {
        path.relative_to(source / "src").as_posix()
        for path in (source / "src" / "experience_hub").rglob("*")
        if path.is_file() and path.suffix in {".py", ".mako"}
    }
    assert package_paths == expected_package_paths
    assert {
        "experience_hub/capture/source_integrity.py",
        "experience_hub/cli/capture_commands.py",
        "experience_hub/cli/release_commands.py",
        "experience_hub/release_evidence/__init__.py",
        "experience_hub/release_evidence/collection.py",
        "experience_hub/release_evidence/contracts.py",
        "experience_hub/release_evidence/errors.py",
        "experience_hub/release_evidence/service.py",
        "experience_hub/release_evidence/source_tree.py",
        "experience_hub/release_evidence/storage.py",
        "experience_hub/storage/migrations/versions/"
        "0007_capture_evidence_hashes.py",
        f"{wheel_root}/METADATA",
        f"{wheel_root}/RECORD",
        f"{wheel_root}/WHEEL",
        f"{wheel_root}/entry_points.txt",
    }.issubset(wheel_paths)
