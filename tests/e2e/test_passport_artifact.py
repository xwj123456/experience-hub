"""Build real archives with synthetic forbidden files to prove exclusion rules."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_FILES = (
    "AGENTS.md",
    ".private-workflow/local-plan.md",
    "work/team/local-plan.md",
    "outputs/passport-transfer/seed.json",
    "docs/internal/local-design.md",
    ".internal/local-designs/plan.md",
    ".local/passport-plan.md",
    ".env",
    ".env.secret",
    "provider-secrets.json",
    ".secrets/auth.json",
    ".cache/credential-cache.json",
    ".data/private.db",
    "private.sqlite",
    "private.sqlite3",
    "src/experience_hub/private.sqlite3",
    "src/experience_hub/.env",
    "src/experience_hub/provider-secrets.json",
)


@pytest.fixture(scope="module")
def passport_archives(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("passport-artifact")
    project, output = root / "project", root / "artifacts"
    project.mkdir()
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        source = PROJECT_ROOT / name
        if source.exists():
            shutil.copy2(source, project / name)
    for name in ("src", "docs", "examples", "tests"):
        if (PROJECT_ROOT / name).exists():
            shutil.copytree(
                PROJECT_ROOT / name,
                project / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    marker = "synthetic-private-" + "artifact-sentinel-v1"
    for name in FORBIDDEN_FILES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n")
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(Path.home()),
        "UV_OFFLINE": "true",
        "UV_NO_CONFIG": "true",
    }
    result = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(output)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = output.glob("*.whl")
    (sdist,) = output.glob("*.tar.gz")
    return wheel, sdist


def _members(archive: Path) -> dict[str, bytes]:
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            return {
                name: wheel.read(name)
                for name in wheel.namelist()
                if not name.endswith("/")
            }
    with tarfile.open(archive) as sdist:
        result = {}
        for member in sdist.getmembers():
            if member.isfile():
                stream = sdist.extractfile(member)
                assert stream is not None
                result[member.name.split("/", 1)[1]] = stream.read()
        return result


@pytest.mark.parametrize("index", [0, 1], ids=["wheel", "sdist"])
def test_passport_archives_exclude_private_synthetic_material(
    passport_archives: tuple[Path, Path], index: int
) -> None:
    members = _members(passport_archives[index])
    unexpected = [
        name
        for name in members
        if name in FORBIDDEN_FILES or "src/" + name in FORBIDDEN_FILES
    ]
    assert unexpected == [], f"Private synthetic files entered archive: {unexpected}"
    marker = ("synthetic-private-" + "artifact-sentinel-v1").encode()
    assert all(marker not in body for body in members.values())
    home = str(Path.home()).encode()
    project = str(PROJECT_ROOT).encode()
    assert all(home not in body and project not in body for body in members.values())
    assert not any(
        name.endswith((".db", ".sqlite", ".sqlite3", ".pyc")) or "/__pycache__/" in name
        for name in members
    )


@pytest.mark.parametrize("index", [0, 1], ids=["wheel", "sdist"])
def test_passport_archives_include_current_modules_and_migrations(
    passport_archives: tuple[Path, Path], index: int
) -> None:
    members = _members(passport_archives[index])
    prefix = "" if index == 0 else "src/"
    for module in (
        "contracts",
        "codec",
        "events",
        "validation",
        "projector",
        "service",
        "queries",
        "repository",
        "export",
        "files",
    ):
        assert f"{prefix}experience_hub/passports/{module}.py" in members
    assert f"{prefix}experience_hub/storage/migrations/env.py" in members
    assert (
        f"{prefix}experience_hub/storage/migrations/versions/0008_evidence_passports.py"
    ) in members
