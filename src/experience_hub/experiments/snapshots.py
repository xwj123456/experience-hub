"""Immutable, non-migrating SQLite snapshots for replay experiments."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import stat
from contextlib import closing, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.experiments.errors import ExperimentIsolationError
from experience_hub.ids import SequenceIdGenerator
from experience_hub.runtime import (
    ApplicationRuntime,
    SchemaRevisionError,
    require_current_schema,
)
from experience_hub.storage.projections import (
    ProjectionMismatch,
    ReducerVersionMismatch,
)
from experience_hub.storage.validation import SourceIntegrityError

_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_READ_CHUNK_SIZE = 1024 * 1024
_CLONE_TEMPORARY_SUFFIX = ".experience-hub-clone.tmp"
_CLONE_BACKUP_PREFIX = ".experience-hub-clone-backup-"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)

type _SidecarStatuses = tuple[tuple[str, os.stat_result | None], ...]


@dataclass(frozen=True, slots=True)
class FrozenSqliteSnapshot:
    """Exact bytes and filesystem identity of one closed SQLite source."""

    source_path: Path
    database_bytes: bytes
    database_sha256: str
    _device: int
    _inode: int
    _size: int
    _mtime_ns: int
    _ctime_ns: int
    _sidecars: _SidecarStatuses


@dataclass(frozen=True, slots=True)
class _ReadSource:
    path: Path
    body: bytes
    sha256: str
    status: os.stat_result
    sidecars: _SidecarStatuses


@dataclass(frozen=True, slots=True)
class _StagedCloneEntry:
    original_name: str
    backup_name: str
    status: os.stat_result


def _snapshot_invalid() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_snapshot_invalid",
        "Replay source must be a closed regular SQLite file",
    )


def _snapshot_changed() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_snapshot_changed",
        "Replay source changed after it was frozen",
    )


def _clone_invalid() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_clone_invalid",
        "Replay clone destination is not safe",
    )


def _clone_failed() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_clone_failed",
        "Replay clone bytes could not be retained exactly",
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_status(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_identity(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _same_sidecar_statuses(
    left: _SidecarStatuses,
    right: _SidecarStatuses,
) -> bool:
    if len(left) != len(right):
        return False
    for (left_suffix, left_status), (right_suffix, right_status) in zip(
        left,
        right,
        strict=True,
    ):
        if left_suffix != right_suffix:
            return False
        if left_status is None or right_status is None:
            if left_status is not right_status:
                return False
        elif not _same_status(left_status, right_status):
            return False
    return True


def _require_safe_sidecars(
    path: Path,
    *,
    error: ExperimentIsolationError,
) -> _SidecarStatuses:
    retained: list[tuple[str, os.stat_result | None]] = []
    for suffix in _SIDECAR_SUFFIXES:
        sidecar = Path(f"{path}{suffix}")
        try:
            status = sidecar.lstat()
        except FileNotFoundError:
            retained.append((suffix, None))
            continue
        except OSError:
            raise error from None
        if not stat.S_ISREG(status.st_mode) or status.st_size != 0:
            raise error
        retained.append((suffix, status))
    return tuple(retained)


def _read_from_descriptor(
    path: Path,
    *,
    initial_error: ExperimentIsolationError,
    changed_error: ExperimentIsolationError,
) -> _ReadSource:
    initial_sidecars = _require_safe_sidecars(path, error=initial_error)
    try:
        path_status = path.lstat()
    except OSError:
        raise initial_error from None
    if not stat.S_ISREG(path_status.st_mode):
        raise initial_error

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or not _same_identity(path_status, opened_status)
        ):
            raise changed_error

        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)

        final_descriptor_status = os.fstat(descriptor)
        if not _same_status(opened_status, final_descriptor_status):
            raise changed_error
        final_sidecars = _require_safe_sidecars(path, error=changed_error)
        if not _same_sidecar_statuses(initial_sidecars, final_sidecars):
            raise changed_error
        final_descriptor_status = os.fstat(descriptor)
        try:
            final_path_status = path.lstat()
        except OSError:
            raise changed_error from None
        if (
            not _same_status(opened_status, final_descriptor_status)
            or not _same_status(final_descriptor_status, final_path_status)
        ):
            raise changed_error
        return _ReadSource(
            path=path,
            body=b"".join(chunks),
            sha256=digest.hexdigest(),
            status=final_descriptor_status,
            sidecars=final_sidecars,
        )
    except ExperimentIsolationError:
        raise
    except OSError:
        raise changed_error from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def freeze_closed_sqlite(source: Path) -> FrozenSqliteSnapshot:
    """Freeze one already-closed SQLite file without invoking SQLite."""
    if not isinstance(source, Path):
        raise _snapshot_invalid()
    retained = _read_from_descriptor(
        source.absolute(),
        initial_error=_snapshot_invalid(),
        changed_error=_snapshot_changed(),
    )
    status = retained.status
    return FrozenSqliteSnapshot(
        source_path=retained.path,
        database_bytes=retained.body,
        database_sha256=retained.sha256,
        _device=status.st_dev,
        _inode=status.st_ino,
        _size=status.st_size,
        _mtime_ns=status.st_mtime_ns,
        _ctime_ns=status.st_ctime_ns,
        _sidecars=retained.sidecars,
    )


def verify_source_unchanged(snapshot: FrozenSqliteSnapshot) -> None:
    """Re-read the source by descriptor and require its frozen identity and bytes."""
    if not isinstance(snapshot, FrozenSqliteSnapshot):
        raise _snapshot_changed()
    retained = _read_from_descriptor(
        snapshot.source_path,
        initial_error=_snapshot_changed(),
        changed_error=_snapshot_changed(),
    )
    status = retained.status
    if (
        status.st_dev != snapshot._device
        or status.st_ino != snapshot._inode
        or status.st_size != snapshot._size
        or status.st_mtime_ns != snapshot._mtime_ns
        or status.st_ctime_ns != snapshot._ctime_ns
        or not _same_sidecar_statuses(retained.sidecars, snapshot._sidecars)
        or retained.sha256 != snapshot.database_sha256
        or retained.body != snapshot.database_bytes
    ):
        raise _snapshot_changed()


def _entry_status(name: str, parent_fd: int) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_or_create_clone_parent(path: Path) -> int:
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_parent_identity(path: Path, parent_fd: int) -> None:
    opened = os.fstat(parent_fd)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(opened.st_mode) or not _same_identity(opened, current):
        raise _clone_invalid()


def _require_entry_unchanged(
    name: str,
    parent_fd: int,
    expected: os.stat_result | None,
) -> None:
    current = _entry_status(name, parent_fd)
    if expected is None:
        if current is not None:
            raise _clone_invalid()
        return
    if current is None or not _same_status(expected, current):
        raise _clone_invalid()


def _safe_destination_sidecars(
    target_name: str,
    parent_fd: int,
) -> tuple[tuple[str, os.stat_result | None], ...]:
    retained: list[tuple[str, os.stat_result | None]] = []
    for suffix in _SIDECAR_SUFFIXES:
        name = f"{target_name}{suffix}"
        status = _entry_status(name, parent_fd)
        if status is not None and (
            not stat.S_ISREG(status.st_mode) or status.st_size
        ):
            raise _clone_invalid()
        retained.append((name, status))
    return tuple(retained)


def _lock_clone_parent(parent_fd: int) -> None:
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
    except OSError:
        raise _clone_invalid() from None


def _unlock_clone_parent(parent_fd: int) -> None:
    with suppress(OSError):
        fcntl.flock(parent_fd, fcntl.LOCK_UN)


def _clone_backup_name(target_name: str) -> str:
    digest = hashlib.sha256(target_name.encode("utf-8")).hexdigest()[:16]
    return f"{_CLONE_BACKUP_PREFIX}{digest}"


def _create_clone_backup(
    name: str,
    parent_fd: int,
) -> tuple[int, os.stat_result]:
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    status = os.fstat(descriptor)
    current = _entry_status(name, parent_fd)
    if current is None or not _same_identity(current, status):
        os.close(descriptor)
        raise _clone_invalid()
    if os.listdir(descriptor):
        os.close(descriptor)
        raise _clone_invalid()
    return descriptor, status


def _stage_clone_entry(
    *,
    original_name: str,
    backup_name: str,
    parent_fd: int,
    backup_fd: int,
    status: os.stat_result,
    staged: list[_StagedCloneEntry],
) -> None:
    entry = _StagedCloneEntry(
        original_name=original_name,
        backup_name=backup_name,
        status=status,
    )
    os.replace(
        original_name,
        backup_name,
        src_dir_fd=parent_fd,
        dst_dir_fd=backup_fd,
    )
    staged.append(entry)
    retained = _entry_status(backup_name, backup_fd)
    if retained is None or not _same_identity(retained, status):
        raise _clone_failed()


def _require_staged_entries(
    entries: list[_StagedCloneEntry],
    backup_fd: int,
) -> None:
    for entry in entries:
        retained = _entry_status(entry.backup_name, backup_fd)
        if retained is None or not _same_identity(retained, entry.status):
            raise _clone_failed()


def _restore_staged_entries(
    entries: list[_StagedCloneEntry],
    *,
    parent_fd: int,
    backup_fd: int,
) -> None:
    for entry in reversed(entries):
        if _entry_status(entry.original_name, parent_fd) is not None:
            raise _clone_failed()
        retained = _entry_status(entry.backup_name, backup_fd)
        if retained is None or not _same_identity(retained, entry.status):
            raise _clone_failed()
        os.replace(
            entry.backup_name,
            entry.original_name,
            src_dir_fd=backup_fd,
            dst_dir_fd=parent_fd,
        )
        restored = _entry_status(entry.original_name, parent_fd)
        if restored is None or not _same_identity(restored, entry.status):
            raise _clone_failed()


def _remove_empty_clone_backup(
    name: str,
    parent_fd: int,
    backup_fd: int,
    status: os.stat_result,
) -> None:
    if os.listdir(backup_fd):
        raise _clone_failed()
    current = _entry_status(name, parent_fd)
    if current is None or not _same_identity(current, status):
        raise _clone_failed()
    os.rmdir(name, dir_fd=parent_fd)


def _discard_committed_clone_backup(
    name: str,
    parent_fd: int,
    backup_fd: int,
    status: os.stat_result,
    entries: list[_StagedCloneEntry],
) -> None:
    """Best-effort cleanup after the final clone name has committed."""
    try:
        for entry in entries:
            retained = _entry_status(entry.backup_name, backup_fd)
            if retained is None or not _same_identity(retained, entry.status):
                return
            os.unlink(entry.backup_name, dir_fd=backup_fd)
        if os.listdir(backup_fd):
            return
        current = _entry_status(name, parent_fd)
        if current is None or not _same_identity(current, status):
            return
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        # Publication is the linearization point. A recognizable private
        # backup is safer than reporting a correct final clone as a failure.
        return


def _cleanup_clone_temporary(
    name: str,
    parent_fd: int,
    expected: os.stat_result,
) -> None:
    current = _entry_status(name, parent_fd)
    if current is None:
        return
    if not _same_identity(current, expected):
        raise _clone_failed()
    try:
        os.unlink(name, dir_fd=parent_fd)
    except OSError:
        raise _clone_failed() from None


def _create_temporary_clone(
    name: str,
    parent_fd: int,
) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    return descriptor, os.fstat(descriptor)


def _write_temporary_clone(
    descriptor: int,
    body: bytes,
) -> os.stat_result:
    view = memoryview(body)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise _clone_failed()
        written += count
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
    retained = b"".join(chunks)
    if retained != body or digest.hexdigest() != hashlib.sha256(body).hexdigest():
        raise _clone_failed()
    return os.fstat(descriptor)


def _publish_temporary_clone(
    *,
    snapshot: FrozenSqliteSnapshot,
    target: Path,
    parent_fd: int,
    temporary_name: str,
    temporary_status: os.stat_result,
    target_status: os.stat_result | None,
    sidecars: tuple[tuple[str, os.stat_result | None], ...],
    backup_name: str,
) -> None:
    present = [
        (target.name, "target", target_status),
        *(
            (name, f"sidecar-{index}", status)
            for index, (name, status) in enumerate(sidecars)
        ),
    ]
    retained = [
        (original_name, slot_name, status)
        for original_name, slot_name, status in present
        if status is not None
    ]
    backup_fd = -1
    backup_status: os.stat_result | None = None
    staged: list[_StagedCloneEntry] = []
    committed = False
    try:
        if retained:
            backup_fd, backup_status = _create_clone_backup(
                backup_name,
                parent_fd,
            )
            for original_name, slot_name, status in retained:
                assert status is not None
                _stage_clone_entry(
                    original_name=original_name,
                    backup_name=slot_name,
                    parent_fd=parent_fd,
                    backup_fd=backup_fd,
                    status=status,
                    staged=staged,
                )

        verify_source_unchanged(snapshot)
        _require_parent_identity(target.parent, parent_fd)
        _require_entry_unchanged(temporary_name, parent_fd, temporary_status)
        _require_entry_unchanged(target.name, parent_fd, None)
        for sidecar_name, _ in sidecars:
            _require_entry_unchanged(sidecar_name, parent_fd, None)
        if backup_fd >= 0:
            _require_staged_entries(staged, backup_fd)

        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        committed = True
        if backup_fd >= 0 and backup_status is not None:
            _discard_committed_clone_backup(
                backup_name,
                parent_fd,
                backup_fd,
                backup_status,
                staged,
            )
    except BaseException:
        if not committed and backup_fd >= 0 and backup_status is not None:
            _restore_staged_entries(
                staged,
                parent_fd=parent_fd,
                backup_fd=backup_fd,
            )
            _remove_empty_clone_backup(
                backup_name,
                parent_fd,
                backup_fd,
                backup_status,
            )
        raise
    finally:
        if backup_fd >= 0:
            os.close(backup_fd)


def clone_frozen_sqlite(
    snapshot: FrozenSqliteSnapshot,
    destination: Path,
) -> Path:
    """Atomically publish frozen bytes to a caller-owned destination.

    The parent descriptor's advisory lock serializes cooperating clone
    publishers. All byte, hash, source, identity, and sidecar checks finish
    before the same-parent ``os.replace`` linearization point. A successful
    replace is therefore the commit: cleanup of recognizable private backups
    is best effort and cannot turn a correct final clone into a reported
    failure. Same-UID processes that ignore the lock are outside this portable
    contract.
    """
    if not isinstance(snapshot, FrozenSqliteSnapshot):
        raise _clone_invalid()
    if not isinstance(destination, Path):
        raise _clone_invalid()
    verify_source_unchanged(snapshot)
    target = destination.absolute()
    if not target.name or target == snapshot.source_path:
        raise _clone_invalid()
    parent_fd = -1
    temporary_fd = -1
    temporary_status: os.stat_result | None = None
    temporary_name = f"{target.name}{_CLONE_TEMPORARY_SUFFIX}"
    backup_name = _clone_backup_name(target.name)
    locked = False
    published = False
    try:
        try:
            parent_fd = _open_or_create_clone_parent(target.parent)
            _lock_clone_parent(parent_fd)
            locked = True
            _require_parent_identity(target.parent, parent_fd)
            target_status = _entry_status(target.name, parent_fd)
            if _entry_status(temporary_name, parent_fd) is not None:
                raise _clone_invalid()
            if _entry_status(backup_name, parent_fd) is not None:
                raise _clone_invalid()
            sidecars = _safe_destination_sidecars(target.name, parent_fd)
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _clone_invalid() from None

        if (
            target_status is not None
            and (
                not stat.S_ISREG(target_status.st_mode)
                or (
                    target_status.st_dev == snapshot._device
                    and target_status.st_ino == snapshot._inode
                )
            )
        ):
            raise _clone_invalid()

        try:
            temporary_fd, temporary_status = _create_temporary_clone(
                temporary_name,
                parent_fd,
            )
            temporary_status = _write_temporary_clone(
                temporary_fd,
                snapshot.database_bytes,
            )
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _clone_failed() from None

        verify_source_unchanged(snapshot)
        try:
            _require_parent_identity(target.parent, parent_fd)
            _require_entry_unchanged(target.name, parent_fd, target_status)
            _require_entry_unchanged(
                temporary_name,
                parent_fd,
                temporary_status,
            )
            for sidecar_name, sidecar_status in sidecars:
                _require_entry_unchanged(
                    sidecar_name,
                    parent_fd,
                    sidecar_status,
                )
            assert temporary_status is not None
            _publish_temporary_clone(
                snapshot=snapshot,
                target=target,
                parent_fd=parent_fd,
                temporary_name=temporary_name,
                temporary_status=temporary_status,
                target_status=target_status,
                sidecars=sidecars,
                backup_name=backup_name,
            )
            published = True
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _clone_failed() from None
        return target
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if (
            parent_fd >= 0
            and temporary_status is not None
            and not published
        ):
            _cleanup_clone_temporary(
                temporary_name,
                parent_fd,
                temporary_status,
            )
        if locked:
            _unlock_clone_parent(parent_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _sqlite_url(path: Path) -> URL:
    return URL.create(
        "sqlite+aiosqlite",
        database=str(path.absolute()),
    )


def _replay_ids(*, seed: int, scope: str) -> SequenceIdGenerator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not scope or scope != scope.strip():
        raise ValueError("scope must be a non-empty trimmed string")
    values = tuple(
        UUID(
            bytes=hashlib.sha256(
                f"experience-hub:{scope}:{seed}:{ordinal}".encode()
            ).digest()[:16],
            version=4,
        )
        for ordinal in range(1, 4_097)
    )
    return SequenceIdGenerator(values)


def _schema_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_schema_unsupported",
        "Replay source schema is not supported",
    )


def _source_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_source_invalid",
        "Replay authoritative source is invalid",
    )


def _projection_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_projection_mismatch",
        "Replay source projections do not match authoritative replay",
    )


async def validate_frozen_snapshot(
    snapshot: FrozenSqliteSnapshot,
    *,
    validation_path: Path,
    frozen_at: datetime,
    seed: int,
) -> str:
    """Validate schema, source, and projections only on a disposable clone."""
    try:
        clone = clone_frozen_sqlite(snapshot, validation_path)
        runtime = ApplicationRuntime(
            Settings(database_url=_sqlite_url(clone)),
            clock=FrozenClock(frozen_at),
            ids=_replay_ids(seed=seed, scope="validation"),
            migrator=require_current_schema,
        )
        async with runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container:
            report = await container.projection_manager.verify(container.database)
            if not report.matches:
                raise _projection_error()
            return cast(str, container.schema_revision)
    except SchemaRevisionError:
        raise _schema_error() from None
    except (ProjectionMismatch, ReducerVersionMismatch):
        raise _projection_error() from None
    except SourceIntegrityError:
        raise _source_error() from None
    except ExperimentIsolationError as error:
        if error.code == "replay_projection_mismatch":
            raise _projection_error() from None
        raise _source_error() from None
    except (sqlite3.DatabaseError, SQLAlchemyError):
        raise _source_error() from None


def checkpoint_owned_sqlite(path: Path) -> tuple[int, int, int]:
    """Checkpoint a closed SQLite database that is owned by the caller."""
    try:
        with closing(
            sqlite3.connect(path, isolation_level=None, timeout=5)
        ) as connection:
            row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if (
            row is None
            or len(row) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in row
            )
        ):
            raise ExperimentIsolationError(
                "replay_checkpoint_failed",
                "Owned SQLite checkpoint returned an invalid result",
            )
        result = cast(tuple[int, int, int], tuple(row))
        if result != (0, 0, 0):
            raise ExperimentIsolationError(
                "replay_checkpoint_failed",
                "Owned SQLite checkpoint did not fully truncate",
            )
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = Path(f"{path}{suffix}")
            try:
                status = sidecar.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(status.st_mode) or status.st_size:
                raise ExperimentIsolationError(
                    "replay_checkpoint_failed",
                    "Owned SQLite checkpoint left an unsafe sidecar",
                )
            sidecar.unlink()
        return result
    except ExperimentIsolationError:
        raise
    except (OSError, sqlite3.Error):
        raise ExperimentIsolationError(
            "replay_checkpoint_failed",
            "Owned SQLite checkpoint failed",
        ) from None


__all__ = [
    "FrozenSqliteSnapshot",
    "checkpoint_owned_sqlite",
    "clone_frozen_sqlite",
    "freeze_closed_sqlite",
    "validate_frozen_snapshot",
    "verify_source_unchanged",
]
