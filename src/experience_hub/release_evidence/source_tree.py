"""Deterministic source-tree identity for release evidence."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Protocol

from experience_hub.release_evidence.errors import ReleaseEvidenceError

GENERATED_EXACT = frozenset(
    {
        "docs/evidence/release-evidence.json",
        "docs/evidence/launch-assets.sha256",
    }
)
GENERATED_PREFIX = "docs/assets/launch/"

_INTERNAL_PREFIXES = (
    "docs/internal/",
    ".internal/",
    ".data/",
    "dist/",
)
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})


class _HashUpdater(Protocol):
    """The digest operation needed while enumerating source entries."""

    def update(self, data: bytes, /) -> None:
        """Add bytes to the digest."""


@dataclass(frozen=True, slots=True)
class SourceTreeIdentity:
    """The commit and canonical current-content digest of a source tree."""

    verified_commit: str
    source_tree_sha256: str


def inspect_source_tree(repository: Path) -> SourceTreeIdentity:
    """Return the source identity after rejecting unsafe working-tree state."""
    verified_commit = _head_commit(repository)
    _reject_untracked_public_files(repository)
    return SourceTreeIdentity(
        verified_commit=verified_commit,
        source_tree_sha256=_source_tree_digest(repository),
    )


def verify_source_tree(repository: Path, expected: SourceTreeIdentity) -> None:
    """Raise a stable error unless the recorded source identity still holds."""
    _require_valid_commit(expected.verified_commit)
    if not _is_ancestor(repository, expected.verified_commit):
        raise ReleaseEvidenceError(
            code="stale_source_tree",
            message="release source tree is stale",
        )

    actual = inspect_source_tree(repository)
    if actual.source_tree_sha256 != expected.source_tree_sha256:
        raise ReleaseEvidenceError(
            code="stale_source_tree",
            message="release source tree is stale",
        )


def _head_commit(repository: Path) -> str:
    output = _git_output(repository, "rev-parse", "--verify", "HEAD")
    try:
        commit = output.rstrip(b"\n").decode("ascii")
    except UnicodeDecodeError as error:
        raise _invalid_tree() from error
    _require_valid_commit(commit)
    return commit


def _source_tree_digest(repository: Path) -> str:
    digest = sha256()
    output = _git_output(repository, "ls-files", "--stage", "-z")
    for record in _nul_records(output):
        mode, path = _stage_record(record)
        if _is_excluded(path):
            continue
        _hash_regular_file(digest, repository, mode, path)
    return digest.hexdigest()


def _reject_untracked_public_files(repository: Path) -> None:
    output = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    for record in _nul_records(output):
        if len(record) < 4 or record[2:3] != b" ":
            raise _invalid_tree()
        status = record[:2]
        path = _decode_path(record[3:])
        if status == b"??":
            if not _is_excluded(path):
                raise ReleaseEvidenceError(
                    code="dirty_source_tree",
                    message="release source tree is dirty",
                )
            continue
        if status == b"!!":
            raise _invalid_tree()


def _nul_records(output: bytes) -> tuple[bytes, ...]:
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise _invalid_tree()
    return tuple(record for record in output[:-1].split(b"\0") if record)


def _stage_record(record: bytes) -> tuple[str, str]:
    metadata, separator, encoded_path = record.partition(b"\t")
    if not separator or not encoded_path:
        raise _invalid_tree()
    fields = metadata.split(b" ")
    if len(fields) != 3 or fields[2] != b"0":
        raise _invalid_tree()
    try:
        mode = fields[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise _invalid_tree() from error
    if mode not in _REGULAR_GIT_MODES:
        raise _invalid_tree()
    return mode, _decode_path(encoded_path)


def _decode_path(encoded_path: bytes) -> str:
    try:
        path = encoded_path.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _invalid_tree() from error
    posix_path = PurePosixPath(path)
    if (
        not path
        or posix_path.is_absolute()
        or path == "."
        or ".." in posix_path.parts
    ):
        raise _invalid_tree()
    return path


def _hash_regular_file(
    digest: _HashUpdater,
    repository: Path,
    mode: str,
    relative_path: str,
) -> None:
    path = repository / relative_path
    try:
        file_mode = path.lstat().st_mode
        if not stat.S_ISREG(file_mode):
            raise _invalid_tree()
        contents_digest = sha256(path.read_bytes()).digest()
    except OSError as error:
        raise _invalid_tree() from error
    entry = (
        mode.encode("ascii")
        + b"\0"
        + relative_path.encode("utf-8")
        + b"\0"
        + contents_digest
    )
    digest.update(len(entry).to_bytes(8, "big"))
    digest.update(entry)


def _is_excluded(path: str) -> bool:
    return (
        path in GENERATED_EXACT
        or path.startswith(GENERATED_PREFIX)
        or path.startswith(_INTERNAL_PREFIXES)
    )


def _is_ancestor(repository: Path, commit: str) -> bool:
    completed = _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise _invalid_tree()


def _git_output(repository: Path, *arguments: str) -> bytes:
    completed = _git(repository, *arguments)
    if completed.returncode != 0:
        raise _invalid_tree()
    return completed.stdout


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError:
        pass
    raise _invalid_tree()


def _git_environment() -> dict[str, str]:
    """Return a deterministic environment without repository redirects."""
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }


def _require_valid_commit(commit: str) -> None:
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise _invalid_tree()


def _invalid_tree() -> ReleaseEvidenceError:
    return ReleaseEvidenceError(
        code="invalid_source_tree",
        message="release source tree cannot be inspected",
    )
