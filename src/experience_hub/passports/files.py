"""Bounded, no-follow Passport files with cooperating-publisher locking.

Retained descriptors and no-clobber links detect ordinary path races. They do not
provide an absolute sandbox against arbitrary malicious processes with the same
UID. A failed cleanup leaves a private artifact rather than deleting an entry
whose identity cannot be established.
"""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from experience_hub.passports.codec import VerifiedPassportV1, verify_passport_bytes
from experience_hub.passports.contracts import MAX_PASSPORT_BYTES
from experience_hub.passports.errors import PassportError

_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True, slots=True)
class _Directory:
    descriptor: int
    parent_descriptor: int | None
    name: str
    status: os.stat_result


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_identity(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _safe_directory(status: os.stat_result) -> bool:
    writable_by_others = status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return stat.S_ISDIR(status.st_mode) and (
        not writable_by_others or bool(status.st_mode & stat.S_ISVTX)
    )


def _require_directories(directories: tuple[_Directory, ...]) -> None:
    for directory in directories:
        current = os.stat(
            directory.name,
            dir_fd=directory.parent_descriptor,
            follow_symlinks=False,
        )
        opened = os.fstat(directory.descriptor)
        if (
            not _safe_directory(current)
            or not _safe_directory(opened)
            or not _same_identity(directory.status, current)
            or not _same_identity(current, opened)
        ):
            raise PassportError("file_invalid")


@contextmanager
def _parent(path: Path) -> Iterator[tuple[str, tuple[_Directory, ...]]]:
    if not isinstance(path, Path) or ".." in path.parts:
        raise PassportError("file_invalid")
    directories: list[_Directory] = []
    try:
        absolute = path.absolute()
        if not absolute.name:
            raise PassportError("file_invalid")
        names = (absolute.anchor, *absolute.parts[1:-1])
        parent: int | None = None
        for name in names:
            retained = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not _safe_directory(retained):
                raise PassportError("file_invalid")
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
            try:
                opened = os.fstat(descriptor)
                if not _safe_directory(opened) or not _same_identity(retained, opened):
                    raise PassportError("file_invalid")
            except BaseException:
                os.close(descriptor)
                raise
            directories.append(_Directory(descriptor, parent, name, opened))
            parent = descriptor
        chain = tuple(directories)
        _require_directories(chain)
        yield absolute.name, chain
    except (OSError, ValueError):
        raise PassportError("file_invalid") from None
    finally:
        for directory in reversed(directories):
            with suppress(OSError):
                os.close(directory.descriptor)


def _entry_status(name: str, parent: int) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _read_bounded(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _require_file(
    name: str, parent: int, descriptor: int, expected: os.stat_result
) -> None:
    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or not _same_file(expected, current)
        or not _same_file(expected, opened)
    ):
        raise PassportError("file_invalid")


def _read_regular(
    name: str, parent: int, *, expected: os.stat_result | None = None
) -> bytes:
    retained = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(retained.st_mode) or (
        expected is not None and not _same_file(expected, retained)
    ):
        raise PassportError("file_invalid")
    descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(retained, opened):
            raise PassportError("file_invalid")
        body = _read_bounded(descriptor, MAX_PASSPORT_BYTES + 1)
        _require_file(name, parent, descriptor, opened)
        if len(body) > MAX_PASSPORT_BYTES:
            raise PassportError("size_limit")
        if len(body) != opened.st_size:
            raise PassportError("file_invalid")
        return body
    finally:
        with suppress(OSError):
            os.close(descriptor)


def read_passport_file(path: Path) -> VerifiedPassportV1:
    """Read one stable regular file, then verify its bounded bytes offline."""
    with _parent(path) as (name, directories):
        body = _read_regular(name, directories[-1].descriptor)
        _require_directories(directories)
        return verify_passport_bytes(body)


def _write_and_read_back(descriptor: int, body: bytes) -> os.stat_result:
    view = memoryview(body)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise PassportError("file_invalid")
        written += count
    os.fsync(descriptor)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_size != len(body):
        raise PassportError("file_invalid")
    os.lseek(descriptor, 0, os.SEEK_SET)
    if _read_bounded(descriptor, len(body) + 1) != body or not _same_file(
        status, os.fstat(descriptor)
    ):
        raise PassportError("file_invalid")
    return status


def _cleanup_temporary(
    name: str, parent: int, descriptor: int, expected: os.stat_result
) -> None:
    opened = os.fstat(descriptor)
    current = _entry_status(name, parent)
    if current is None:
        return
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or not _same_identity(expected, opened)
        or not _same_identity(opened, current)
    ):
        raise PassportError("file_invalid")
    os.unlink(name, dir_fd=parent)


def _publish_locked(
    name: str, directories: tuple[_Directory, ...], body: bytes
) -> None:
    parent = directories[-1].descriptor
    _require_directories(directories)
    existing = _entry_status(name, parent)
    if existing is not None:
        try:
            retained = _read_regular(name, parent, expected=existing)
        except PassportError as error:
            if error.code == "passport_size_limit":
                raise PassportError("output_conflict") from None
            raise
        _require_directories(directories)
        if retained != body:
            raise PassportError("output_conflict")
        return

    temporary_name = f".passport-{uuid4().hex}.tmp"
    temporary_descriptor = -1
    temporary_status: os.stat_result | None = None
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent,
        )
        temporary_status = os.fstat(temporary_descriptor)
        os.fchmod(temporary_descriptor, 0o600)
        temporary_status = _write_and_read_back(temporary_descriptor, body)
        if stat.S_IMODE(temporary_status.st_mode) != 0o600:
            raise PassportError("file_invalid")
        _require_file(temporary_name, parent, temporary_descriptor, temporary_status)
        _require_directories(directories)
        if _entry_status(name, parent) is not None:
            raise PassportError("output_conflict")
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise PassportError("output_conflict") from None
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not _same_identity(temporary_status, linked)
            or _read_regular(name, parent, expected=linked) != body
        ):
            raise PassportError("file_invalid")
        _require_file(name, parent, temporary_descriptor, linked)
        _require_directories(directories)
        os.fsync(parent)
    finally:
        try:
            if temporary_status is not None:
                _cleanup_temporary(
                    temporary_name, parent, temporary_descriptor, temporary_status
                )
        finally:
            if temporary_descriptor >= 0:
                with suppress(OSError):
                    os.close(temporary_descriptor)


def publish_passport_file(path: Path, body: bytes) -> None:
    """Publish bounded caller-prepared bytes without ever replacing a target.

    A directory fsync or cleanup failure after the no-clobber link can leave the
    complete output in place. Retrying identical bytes is safe and idempotent.
    """
    if not isinstance(body, bytes):
        raise PassportError("file_invalid")
    if len(body) > MAX_PASSPORT_BYTES:
        raise PassportError("size_limit")
    with _parent(path) as (name, directories):
        parent = directories[-1].descriptor
        fcntl.flock(parent, fcntl.LOCK_EX)
        try:
            _publish_locked(name, directories, body)
        finally:
            with suppress(OSError):
                fcntl.flock(parent, fcntl.LOCK_UN)
