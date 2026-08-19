"""Marker-protected workspaces for replay-generated artifacts.

Operations hold an exclusive advisory lock on the trusted workspace directory
descriptor. This serializes cooperating Experience Hub processes. It does not
protect against a malicious same-UID process that ignores the lock and swaps a
directory entry between system calls; that behavior is outside this portable
workspace contract.
"""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from experience_hub.experiments.errors import ExperimentIsolationError

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TEMPORARY_SUFFIX = ".experience-hub.tmp"


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    """The marker and exact top-level entries an operation owns."""

    marker_name: str
    marker_body: bytes
    owned_entries: frozenset[str]


@dataclass(frozen=True, slots=True)
class _OwnedEntry:
    name: str
    status: os.stat_result


REPLAY_WORKSPACE_POLICY = WorkspacePolicy(
    marker_name=".experience-hub-replay-workspace",
    marker_body=b"experience-hub replay workspace v1\n",
    owned_entries=frozenset({"snapshot", "validation", "arms", "artifacts"}),
)


@dataclass(frozen=True, slots=True)
class OwnedWorkspace:
    """A validated workspace anchored to one immutable directory identity."""

    root: Path
    _policy: WorkspacePolicy
    _device: int
    _inode: int

    def atomic_write(self, relative: PurePosixPath, body: bytes) -> Path:
        """Atomically replace one policy-owned artifact with exact bytes."""
        parts = _safe_relative_parts(relative)
        if parts[0] not in self._policy.owned_entries or not isinstance(body, bytes):
            raise _path_invalid()

        parent_fd = -1
        root_fd = -1
        opened_directories: list[tuple[int, str, int]] = []
        temporary_created = False
        temporary_status: os.stat_result | None = None
        temporary_name = f"{parts[-1]}{_TEMPORARY_SUFFIX}"
        locked = False
        try:
            root_fd = _open_existing_root(self.root)
            _lock_workspace(root_fd)
            locked = True
            _require_identity(root_fd, self._device, self._inode)
            _require_safe_relative_path(root_fd, parts)
            _validate_current_ownership(root_fd, self._policy)
            parent_fd = root_fd
            for part in parts[:-1]:
                child_fd = _open_or_create_directory(part, parent_fd)
                opened_directories.append((parent_fd, part, child_fd))
                parent_fd = child_fd
            _validate_regular_or_missing(parts[-1], parent_fd)
            temporary_fd = _create_temporary(temporary_name, parent_fd)
            temporary_created = True
            try:
                _write_and_read_back(temporary_fd, body)
                temporary_status = os.fstat(temporary_fd)
            finally:
                os.close(temporary_fd)
            if temporary_status is None:
                raise _write_failed()
            _validate_regular_or_missing(parts[-1], parent_fd)
            _require_linked_directories(opened_directories)
            _require_path_identity(self.root, root_fd)
            _commit_temporary(
                temporary_name,
                parts[-1],
                parent_fd,
                temporary_status,
            )
            temporary_created = False
            try:
                _require_linked_directories(opened_directories)
                _require_path_identity(self.root, root_fd)
            except ExperimentIsolationError:
                _rollback_created_entry(parts[-1], parent_fd, temporary_status)
                raise
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _write_failed() from None
        finally:
            if temporary_created and parent_fd >= 0:
                _unlink_if_owned_temporary(temporary_name, parent_fd)
            for _, _, directory_fd in reversed(opened_directories):
                os.close(directory_fd)
            if locked:
                _unlock_workspace(root_fd)
            if root_fd >= 0:
                os.close(root_fd)
        return self.root.joinpath(*parts)


def prepare_owned_workspace(
    path: Path,
    *,
    policy: WorkspacePolicy,
    replace_owned: bool,
    allow_unmarked_empty: bool,
) -> OwnedWorkspace:
    """Prepare a workspace without deleting data outside the stated policy."""
    if not isinstance(path, Path) or not isinstance(policy, WorkspacePolicy):
        raise _unowned()
    if not isinstance(replace_owned, bool) or not isinstance(
        allow_unmarked_empty, bool
    ):
        raise _unowned()
    _validate_policy(policy)

    root = path.absolute()
    root_fd = -1
    locked = False
    try:
        root_fd, created = _open_or_create_root(root)
        _lock_workspace(root_fd)
        locked = True
        if created:
            _adopt_empty_workspace(root_fd, policy)
            return _owned_workspace(root, policy, root_fd)

        entries = _list_entries(root_fd)
        if policy.marker_name not in entries:
            if entries or not allow_unmarked_empty:
                raise _unowned()
            _adopt_empty_workspace(root_fd, policy)
            return _owned_workspace(root, policy, root_fd)

        _validate_marked_entries(root_fd, entries, policy)
        _read_marker(root_fd, policy)
        owned = tuple(name for name in entries if name in policy.owned_entries)
        if owned and not replace_owned:
            raise ExperimentIsolationError(
                "replay_workspace_exists", "workspace already contains owned output"
            )
        owned_entries = _validate_owned_trees(root_fd, owned)
        for entry in owned_entries:
            _remove_owned_entry(root_fd, entry)
        return _owned_workspace(root, policy, root_fd)
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _unowned() from None
    finally:
        if locked:
            _unlock_workspace(root_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _open_or_create_root(root: Path) -> tuple[int, bool]:
    parent_fd = _open_directory_path(root.parent)
    try:
        try:
            return os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd), False
        except FileNotFoundError:
            os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
            return os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd), True
    finally:
        os.close(parent_fd)


def _open_existing_root(root: Path) -> int:
    parent_fd = _open_directory_path(root.parent)
    try:
        return os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        raise _unowned() from None
    finally:
        os.close(parent_fd)


def _lock_workspace(directory_fd: int) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
    except OSError:
        raise _unowned() from None


def _unlock_workspace(directory_fd: int) -> None:
    with suppress(OSError):
        fcntl.flock(directory_fd, fcntl.LOCK_UN)


def _open_directory_path(path: Path) -> int:
    if path == Path(path.anchor):
        return os.open(path.anchor, _DIRECTORY_FLAGS)
    current_fd = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except OSError:
        os.close(current_fd)
        raise _unowned() from None


def _owned_workspace(
    root: Path, policy: WorkspacePolicy, root_fd: int
) -> OwnedWorkspace:
    _require_path_identity(root, root_fd)
    status = os.fstat(root_fd)
    return OwnedWorkspace(
        root=root,
        _policy=policy,
        _device=status.st_dev,
        _inode=status.st_ino,
    )


def _validate_policy(policy: WorkspacePolicy) -> None:
    if (
        not policy.marker_name
        or "/" in policy.marker_name
        or "\\" in policy.marker_name
        or policy.marker_name in {".", ".."}
        or not policy.marker_body
        or not policy.owned_entries
    ):
        raise _unowned()
    if any(
        not entry
        or "/" in entry
        or "\\" in entry
        or entry in {".", "..", policy.marker_name}
        for entry in policy.owned_entries
    ):
        raise _unowned()


def _list_entries(directory_fd: int) -> tuple[str, ...]:
    try:
        return tuple(sorted(os.listdir(directory_fd)))
    except OSError:
        raise _unowned() from None


def _validate_marked_entries(
    directory_fd: int, entries: tuple[str, ...], policy: WorkspacePolicy
) -> None:
    allowed = policy.owned_entries | frozenset({policy.marker_name})
    for name in entries:
        if name not in allowed:
            raise _unowned()
        status = _lstat(name, directory_fd)
        if stat.S_ISLNK(status.st_mode):
            raise _unowned()
        if name == policy.marker_name and not stat.S_ISREG(status.st_mode):
            raise _unowned()


def _validate_current_ownership(directory_fd: int, policy: WorkspacePolicy) -> None:
    entries = _list_entries(directory_fd)
    if policy.marker_name not in entries:
        raise _unowned()
    _validate_marked_entries(directory_fd, entries, policy)
    _read_marker(directory_fd, policy)


def _read_marker(directory_fd: int, policy: WorkspacePolicy) -> None:
    marker_fd = -1
    try:
        marker_fd = os.open(
            policy.marker_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        status = os.fstat(marker_fd)
        if not stat.S_ISREG(status.st_mode):
            raise _unowned()
        body = os.read(marker_fd, len(policy.marker_body) + 1)
        _require_entry_identity(policy.marker_name, directory_fd, status)
        if body != policy.marker_body:
            raise _unowned()
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _unowned() from None
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)


def _adopt_empty_workspace(directory_fd: int, policy: WorkspacePolicy) -> None:
    temporary_name = f"{policy.marker_name}.creating"
    temporary_status: os.stat_result | None = None
    marker_status: os.stat_result | None = None
    try:
        if _list_entries(directory_fd):
            raise _unowned()
        temporary_status = _create_marker_file(
            directory_fd, temporary_name, policy.marker_body
        )
        if _list_entries(directory_fd) != (temporary_name,):
            raise _unowned()
        os.link(
            temporary_name,
            policy.marker_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        marker_status = _lstat(policy.marker_name, directory_fd)
        _require_identity_status(marker_status, temporary_status)
        _unlink_same_entry(temporary_name, directory_fd, temporary_status)
        temporary_status = None
        if _list_entries(directory_fd) != (policy.marker_name,):
            raise _unowned()
        _read_marker(directory_fd, policy)
        marker_status = None
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _unowned() from None
    finally:
        if marker_status is not None:
            _rollback_created_entry(policy.marker_name, directory_fd, marker_status)
        if temporary_status is not None:
            _rollback_created_entry(temporary_name, directory_fd, temporary_status)


def _create_marker_file(
    directory_fd: int, name: str, body: bytes
) -> os.stat_result:
    marker_fd = -1
    try:
        marker_fd = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(marker_fd, body)
        os.fsync(marker_fd)
        os.lseek(marker_fd, 0, os.SEEK_SET)
        if os.read(marker_fd, len(body) + 1) != body:
            raise _unowned()
        status = os.fstat(marker_fd)
        _require_entry_identity(name, directory_fd, status)
        return status
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _unowned() from None
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)


def _rollback_created_entry(
    name: str, directory_fd: int, expected: os.stat_result
) -> None:
    with suppress(ExperimentIsolationError):
        _unlink_same_entry(name, directory_fd, expected)


def _unlink_same_entry(
    name: str, directory_fd: int, expected: os.stat_result
) -> None:
    _require_entry_identity(name, directory_fd, expected)
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        raise _unowned() from None


def _validate_owned_trees(
    directory_fd: int, names: tuple[str, ...]
) -> tuple[_OwnedEntry, ...]:
    entries: list[_OwnedEntry] = []
    for name in names:
        status = _lstat(name, directory_fd)
        if stat.S_ISDIR(status.st_mode):
            child_fd = _open_directory(name, directory_fd)
            try:
                _validate_tree(child_fd)
                _require_entry_identity(name, directory_fd, status)
            finally:
                os.close(child_fd)
        elif not stat.S_ISREG(status.st_mode):
            raise _unowned()
        entries.append(_OwnedEntry(name=name, status=status))
    return tuple(entries)


def _validate_tree(directory_fd: int) -> None:
    for name in _list_entries(directory_fd):
        status = _lstat(name, directory_fd)
        if stat.S_ISLNK(status.st_mode):
            raise _unowned()
        if stat.S_ISDIR(status.st_mode):
            child_fd = _open_directory(name, directory_fd)
            try:
                _validate_tree(child_fd)
                _require_entry_identity(name, directory_fd, status)
            finally:
                os.close(child_fd)
        elif not stat.S_ISREG(status.st_mode):
            raise _unowned()


def _remove_owned_entry(directory_fd: int, entry: _OwnedEntry) -> None:
    if stat.S_ISREG(entry.status.st_mode):
        _require_entry_identity(entry.name, directory_fd, entry.status)
        try:
            os.unlink(entry.name, dir_fd=directory_fd)
        except OSError:
            raise _unowned() from None
        return
    if not stat.S_ISDIR(entry.status.st_mode):
        raise _unowned()
    child_fd = _open_directory(entry.name, directory_fd)
    try:
        _require_identity_status(os.fstat(child_fd), entry.status)
        _remove_tree(child_fd)
        _require_entry_identity(entry.name, directory_fd, entry.status)
        os.rmdir(entry.name, dir_fd=directory_fd)
    finally:
        os.close(child_fd)


def _remove_tree(directory_fd: int) -> None:
    for name in _list_entries(directory_fd):
        status = _lstat(name, directory_fd)
        if stat.S_ISREG(status.st_mode):
            os.unlink(name, dir_fd=directory_fd)
        elif stat.S_ISDIR(status.st_mode):
            child_fd = _open_directory(name, directory_fd)
            try:
                _remove_tree(child_fd)
                _require_entry_identity(name, directory_fd, status)
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            raise _unowned()


def _safe_relative_parts(relative: PurePosixPath) -> tuple[str, ...]:
    if not isinstance(relative, PurePosixPath) or relative.is_absolute():
        raise _path_invalid()
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _path_invalid()
    return parts


def _open_or_create_directory(name: str, parent_fd: int) -> int:
    try:
        return _open_directory(name, parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            return _open_directory(name, parent_fd)
        except OSError:
            raise _path_invalid() from None
    except OSError:
        raise _path_invalid() from None


def _open_directory(name: str, parent_fd: int) -> int:
    return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)


def _validate_regular_or_missing(name: str, parent_fd: int) -> None:
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise _path_invalid() from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise _path_invalid()


def _require_safe_relative_path(root_fd: int, parts: tuple[str, ...]) -> None:
    parent_fd = root_fd
    opened: list[int] = []
    try:
        for index, name in enumerate(parts):
            try:
                status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISLNK(status.st_mode):
                raise _path_invalid()
            if index == len(parts) - 1:
                return
            if not stat.S_ISDIR(status.st_mode):
                raise _path_invalid()
            child_fd = _open_directory(name, parent_fd)
            opened.append(child_fd)
            parent_fd = child_fd
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _path_invalid() from None
    finally:
        for directory_fd in reversed(opened):
            os.close(directory_fd)


def _create_temporary(name: str, parent_fd: int) -> int:
    try:
        return os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError:
        raise _path_invalid() from None
    except OSError:
        raise _write_failed() from None


def _commit_temporary(
    temporary_name: str,
    target_name: str,
    parent_fd: int,
    expected: os.stat_result,
) -> None:
    try:
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        _require_entry_identity(target_name, parent_fd, expected)
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _write_failed() from None


def _write_and_read_back(file_fd: int, body: bytes) -> None:
    try:
        _write_all(file_fd, body)
        os.fsync(file_fd)
        os.lseek(file_fd, 0, os.SEEK_SET)
        if os.read(file_fd, len(body) + 1) != body:
            raise _write_failed()
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _write_failed() from None


def _write_all(file_fd: int, body: bytes) -> None:
    offset = 0
    while offset < len(body):
        written = os.write(file_fd, body[offset:])
        if written <= 0:
            raise _write_failed()
        offset += written


def _require_linked_directories(
    directories: list[tuple[int, str, int]],
) -> None:
    for parent_fd, name, child_fd in directories:
        _require_entry_identity(name, parent_fd, os.fstat(child_fd))


def _require_path_identity(path: Path, directory_fd: int) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        raise _unowned() from None
    _require_identity(directory_fd, status.st_dev, status.st_ino)


def _require_entry_identity(name: str, parent_fd: int, status: os.stat_result) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise _unowned() from None
    if current.st_dev != status.st_dev or current.st_ino != status.st_ino:
        raise _unowned()


def _require_identity(directory_fd: int, device: int, inode: int) -> None:
    status = os.fstat(directory_fd)
    if status.st_dev != device or status.st_ino != inode:
        raise _unowned()


def _require_identity_status(
    current: os.stat_result, expected: os.stat_result
) -> None:
    if current.st_dev != expected.st_dev or current.st_ino != expected.st_ino:
        raise _unowned()


def _lstat(name: str, directory_fd: int) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        raise _unowned() from None


def _unlink_if_owned_temporary(name: str, parent_fd: int) -> None:
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISREG(status.st_mode):
            os.unlink(name, dir_fd=parent_fd)
    except OSError:
        pass


def _unowned() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_workspace_unowned", "workspace is not safely owned by this policy"
    )


def _path_invalid() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_workspace_path_invalid", "artifact path is not safely contained"
    )


def _write_failed() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_workspace_write_failed", "artifact could not be written atomically"
    )
