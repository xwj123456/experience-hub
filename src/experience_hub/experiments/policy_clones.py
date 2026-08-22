"""Shared safe local SQLite clone boundary for closed policy arms."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine import URL

from experience_hub.config import Settings
from experience_hub.experiments.errors import ExperimentIsolationError

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_WRITABLE_FILE_FLAGS = (
    os.O_RDWR
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True, slots=True)
class PolicyCloneIdentity:
    """One retained regular, unlinked local clone identity."""

    path: Path
    device: int
    inode: int


@dataclass(slots=True)
class PolicyCloneLease:
    """Descriptor-retained clone authority for one synchronous SQLite operation."""

    identity: PolicyCloneIdentity
    descriptor: int

    @property
    def sqlite_uri(self) -> str:
        if self.descriptor < 0:
            raise _clone_error()
        return f"file:/dev/fd/{self.descriptor}?mode=rw&cache=private"

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            try:
                os.close(descriptor)
            except OSError:
                raise _clone_error() from None


def _clone_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_policy_clone_invalid",
        "Replay policy clone is not a valid current-schema database",
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_clone_parent(path: Path) -> int:
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:-1]:
            retained = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(retained.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not _same_identity(retained, opened)
            ):
                os.close(child)
                raise _clone_error()
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _retain_safe_policy_clone(
    path: Path,
    *,
    file_flags: int,
) -> tuple[PolicyCloneIdentity, int]:
    if not isinstance(path, Path):
        raise _clone_error()
    absolute = path.absolute()
    if not absolute.name:
        raise _clone_error()
    parent_descriptor = -1
    clone_descriptor = -1
    try:
        parent_descriptor = _open_clone_parent(absolute)
        parent_retained = os.stat(absolute.parent, follow_symlinks=False)
        parent_opened = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_retained.st_mode)
            or not _same_identity(parent_retained, parent_opened)
        ):
            raise _clone_error()

        retained = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(retained.st_mode) or retained.st_nlink != 1:
            raise _clone_error()
        clone_descriptor = os.open(
            absolute.name,
            file_flags,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(clone_descriptor)
        declared = os.stat(absolute, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_identity(retained, opened)
            or not _same_identity(opened, declared)
        ):
            raise _clone_error()
        identity = PolicyCloneIdentity(
            path=absolute, device=opened.st_dev, inode=opened.st_ino
        )
        retained_descriptor = clone_descriptor
        clone_descriptor = -1
        return identity, retained_descriptor
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _clone_error() from None
    finally:
        if clone_descriptor >= 0:
            os.close(clone_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def require_safe_policy_clone(path: Path) -> PolicyCloneIdentity:
    """Retain one regular single-link clone without following symlinks."""
    identity, descriptor = _retain_safe_policy_clone(path, file_flags=_FILE_FLAGS)
    try:
        return identity
    finally:
        try:
            os.close(descriptor)
        except OSError:
            raise _clone_error() from None


def acquire_policy_clone_lease(path: Path) -> PolicyCloneLease:
    """Keep a writable clone descriptor bound through one SQLite operation."""
    identity, descriptor = _retain_safe_policy_clone(
        path,
        file_flags=_WRITABLE_FILE_FLAGS,
    )
    return PolicyCloneLease(identity=identity, descriptor=descriptor)


def require_same_policy_clone(identity: PolicyCloneIdentity) -> None:
    """Fail when the retained clone path or identity changed."""
    retained = require_safe_policy_clone(identity.path)
    if retained.device != identity.device or retained.inode != identity.inode:
        raise _clone_error()


def policy_clone_settings(path: Path) -> Settings:
    """Build one local SQLite setting for a validated clone."""
    database_url = URL.create("sqlite+aiosqlite", database=str(path))
    return Settings(database_url=database_url)
