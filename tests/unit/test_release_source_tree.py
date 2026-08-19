"""Tests for release evidence source-tree closure."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from experience_hub.release_evidence.errors import ReleaseEvidenceError
from experience_hub.release_evidence.source_tree import (
    inspect_source_tree,
    verify_source_tree,
)


def run_git(repository: Path, *arguments: str) -> str:
    """Run Git in a temporary test repository."""
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=repository,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
        text=True,
    )
    return completed.stdout


def committed_repository(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Create a committed temporary repository with deterministic identity."""
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    run_git(repository, "init", "--quiet")
    run_git(repository, "config", "user.name", "Release Evidence Test")
    run_git(repository, "config", "user.email", "release-evidence@example.test")
    for relative_path, content in files.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    run_git(repository, "add", ".")
    run_git(repository, "commit", "--quiet", "-m", "test repository")
    return repository


def commit_all(repository: Path, message: str) -> None:
    """Commit the current temporary repository state."""
    run_git(repository, "add", ".")
    run_git(repository, "commit", "--quiet", "-m", message)


def test_identical_tracked_content_has_identical_digest(tmp_path: Path) -> None:
    first = committed_repository(tmp_path / "first", {"source.py": b"answer = 42\n"})
    second = committed_repository(tmp_path / "second", {"source.py": b"answer = 42\n"})

    assert inspect_source_tree(first).source_tree_sha256 == (
        inspect_source_tree(second).source_tree_sha256
    )


@pytest.mark.parametrize(
    ("first_files", "second_files"),
    (
        ({"one.py": b"same\n"}, {"two.py": b"same\n"}),
        ({"source.py": b"first\n"}, {"source.py": b"second\n"}),
    ),
)
def test_path_names_and_bytes_affect_digest(
    tmp_path: Path,
    first_files: dict[str, bytes],
    second_files: dict[str, bytes],
) -> None:
    first = committed_repository(tmp_path / "first", first_files)
    second = committed_repository(tmp_path / "second", second_files)

    assert inspect_source_tree(first).source_tree_sha256 != (
        inspect_source_tree(second).source_tree_sha256
    )


def test_git_mode_affects_digest(tmp_path: Path) -> None:
    first = committed_repository(tmp_path / "first", {"script.py": b"print('ok')\n"})
    second = committed_repository(
        tmp_path / "second", {"script.py": b"print('ok')\n"}
    )
    (second / "script.py").chmod(0o755)
    run_git(second, "add", "script.py")
    run_git(second, "commit", "--quiet", "-m", "make executable")

    assert inspect_source_tree(first).source_tree_sha256 != (
        inspect_source_tree(second).source_tree_sha256
    )


@pytest.mark.parametrize(
    "changed_path",
    ("source.py", "README.md", ".github/workflows/check.yml", "tests/fixture.json"),
)
def test_source_related_tracked_edits_stale_the_digest(
    tmp_path: Path, changed_path: str
) -> None:
    repository = committed_repository(
        tmp_path,
        {
            "source.py": b"before\n",
            "README.md": b"before\n",
            ".github/workflows/check.yml": b"before\n",
            "tests/fixture.json": b"before\n",
        },
    )
    before = inspect_source_tree(repository)
    (repository / changed_path).write_bytes(b"after\n")

    with pytest.raises(ReleaseEvidenceError) as raised:
        verify_source_tree(repository, before)

    assert raised.value.code == "stale_source_tree"


@pytest.mark.parametrize(
    "generated_path",
    (
        "docs/evidence/release-evidence.json",
        "docs/evidence/launch-assets.sha256",
        "docs/assets/launch/hero.png",
    ),
)
def test_generated_evidence_changes_do_not_stale_digest(
    tmp_path: Path, generated_path: str
) -> None:
    repository = committed_repository(tmp_path, {"source.py": b"stable\n"})
    before = inspect_source_tree(repository)
    path = repository / generated_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"generated\n")

    verify_source_tree(repository, before)


def test_tracked_edits_participate_while_untracked_public_file_rejects(
    tmp_path: Path,
) -> None:
    repository = committed_repository(tmp_path, {"source.py": b"before\n"})
    before = inspect_source_tree(repository)
    (repository / "source.py").write_bytes(b"after\n")

    changed = inspect_source_tree(repository)

    assert changed.source_tree_sha256 != before.source_tree_sha256
    (repository / "public.txt").write_bytes(b"untracked\n")
    with pytest.raises(ReleaseEvidenceError) as raised:
        inspect_source_tree(repository)

    assert raised.value.code == "dirty_source_tree"
    assert "public.txt" not in raised.value.message


def test_internal_and_ignored_cache_files_do_not_enter_digest(tmp_path: Path) -> None:
    repository = committed_repository(
        tmp_path,
        {
            "source.py": b"stable\n",
            ".gitignore": b".cache/\n",
            "docs/internal/private.md": b"before\n",
            ".internal/scratch.txt": b"before\n",
            ".data/state.sqlite": b"before\n",
            "dist/package.whl": b"before\n",
        },
    )
    before = inspect_source_tree(repository)
    for changed_path in (
        "docs/internal/private.md",
        ".internal/scratch.txt",
        ".data/state.sqlite",
        "dist/package.whl",
        ".cache/runtime.bin",
    ):
        path = repository / changed_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"after\n")

    verify_source_tree(repository, before)


def test_recorded_commit_must_remain_an_ancestor_of_head(tmp_path: Path) -> None:
    repository = committed_repository(tmp_path, {"source.py": b"before\n"})
    identity = inspect_source_tree(repository)
    (repository / "source.py").write_bytes(b"after\n")
    commit_all(repository, "replace history")
    run_git(repository, "checkout", "--quiet", "--orphan", "replacement")
    run_git(repository, "rm", "--quiet", "--cached", "-r", ".")
    run_git(repository, "add", "source.py")
    run_git(repository, "commit", "--quiet", "-m", "replacement root")

    with pytest.raises(ReleaseEvidenceError) as raised:
        verify_source_tree(repository, identity)

    assert raised.value.code == "stale_source_tree"


def test_tracked_symlink_fails_closed_without_private_path(tmp_path: Path) -> None:
    repository = committed_repository(tmp_path, {"source.py": b"stable\n"})
    (repository / "linked-source.py").symlink_to("source.py")
    run_git(repository, "add", "linked-source.py")
    run_git(repository, "commit", "--quiet", "-m", "add symlink")

    with pytest.raises(ReleaseEvidenceError) as raised:
        inspect_source_tree(repository)

    assert raised.value.code == "invalid_source_tree"
    assert "linked-source.py" not in raised.value.message


def test_missing_repository_fails_closed_without_repository_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "private-missing-repository"

    with pytest.raises(ReleaseEvidenceError) as raised:
        inspect_source_tree(repository)

    assert raised.value.code == "invalid_source_tree"
    assert str(repository) not in raised.value.message
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_source_inspection_ignores_inherited_git_repository_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = committed_repository(
        tmp_path / "target",
        {"source.py": b"target source\n"},
    )
    redirect = committed_repository(
        tmp_path / "redirect",
        {"source.py": b"redirected source\n"},
    )
    expected = inspect_source_tree(repository)
    assert inspect_source_tree(redirect) != expected
    monkeypatch.setenv("GIT_DIR", str(redirect / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(redirect))
    monkeypatch.setenv("GIT_INDEX_FILE", str(redirect / ".git" / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(redirect / ".git" / "objects"))

    assert inspect_source_tree(repository) == expected
