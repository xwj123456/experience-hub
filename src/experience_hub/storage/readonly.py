"""Non-mutating sessions over retained, quiescent SQLite source files."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import stat
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from aiosqlite import Connection as SqliteConnection
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from experience_hub.errors import DomainError
from experience_hub.runtime import SchemaRevisionError, installed_schema_head

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_SIDECARS = ("-wal", "-shm", "-journal")
_READ_CHUNK_SIZE = 1024 * 1024
_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_RECURSIVE,
    }
)
_READ_PRAGMAS = frozenset(
    {
        "compile_options",
        "database_list",
        "foreign_keys",
        "journal_mode",
        "query_only",
        "schema_version",
        "user_version",
    }
)
_PARAMETERIZED_READ_PRAGMAS = frozenset(
    {
        "foreign_key_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "integrity_check",
        "quick_check",
        "table_info",
        "table_xinfo",
    }
)


class ReadonlyDatabaseError(DomainError):
    """The requested file cannot safely provide a non-mutating SQLite read."""

    def __init__(self) -> None:
        super().__init__(
            "readonly_database_invalid",
            "The database cannot be opened safely for read-only access",
        )


@dataclass(frozen=True, slots=True)
class _Directory:
    descriptor: int
    parent_descriptor: int | None
    name: str
    status: os.stat_result


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    descriptor: int
    status: os.stat_result
    digest: bytes
    directories: tuple[_Directory, ...]
    sidecars: tuple[os.stat_result | None, ...]


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_file_status(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_identity(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _file_digest(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
        digest.update(chunk)
    return digest.digest()


def _sidecar_statuses(name: str, parent: int) -> tuple[os.stat_result | None, ...]:
    statuses: list[os.stat_result | None] = []
    for suffix in _SIDECARS:
        try:
            status = os.stat(f"{name}{suffix}", dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            statuses.append(None)
            continue
        # immutable SQLite ignores WAL/journal state. Nonempty sidecars therefore
        # cannot be accepted as evidence of a fully checkpointed source.
        if not stat.S_ISREG(status.st_mode) or status.st_size != 0:
            raise ReadonlyDatabaseError
        statuses.append(status)
    return tuple(statuses)


def _close_source(source: _Source) -> None:
    with suppress(OSError):
        os.close(source.descriptor)
    for directory in reversed(source.directories):
        with suppress(OSError):
            os.close(directory.descriptor)


def _safe_directory(status: os.stat_result) -> bool:
    # Permit sticky system ancestors such as /tmp. This is not an absolute
    # defense against administrators or arbitrary same-UID competing processes.
    writable_by_others = status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return stat.S_ISDIR(status.st_mode) and (
        not writable_by_others or bool(status.st_mode & stat.S_ISVTX)
    )


def _open_directory(name: str, parent: int | None) -> _Directory:
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    try:
        status = os.fstat(descriptor)
        if not _safe_directory(status):
            raise ReadonlyDatabaseError
        return _Directory(descriptor, parent, name, status)
    except BaseException:
        os.close(descriptor)
        raise


def _open_source(path: Path) -> _Source:
    if not isinstance(path, Path) or ".." in path.parts:
        raise ReadonlyDatabaseError
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ReadonlyDatabaseError
    directories: list[_Directory] = []
    descriptor = -1
    try:
        root = _open_directory(absolute.anchor, None)
        directories.append(root)
        parent = root.descriptor
        for name in absolute.parts[1:-1]:
            retained = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(retained.st_mode):
                raise ReadonlyDatabaseError
            child = _open_directory(name, parent)
            directories.append(child)
            if not stat.S_ISDIR(child.status.st_mode) or not _same_identity(
                retained, child.status
            ):
                raise ReadonlyDatabaseError
            parent = child.descriptor

        retained = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(retained.st_mode):
            raise ReadonlyDatabaseError
        descriptor = os.open(absolute.name, _FILE_FLAGS, dir_fd=parent)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_status(retained, opened):
            raise ReadonlyDatabaseError
        sidecars = _sidecar_statuses(absolute.name, parent)
        digest = _file_digest(descriptor)
        source = _Source(
            absolute, descriptor, opened, digest, tuple(directories), sidecars
        )
        _validate_source(source)
        return source
    except (OSError, ValueError, ReadonlyDatabaseError):
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        for directory in reversed(directories):
            with suppress(OSError):
                os.close(directory.descriptor)
        raise ReadonlyDatabaseError from None


def _validate_source(source: _Source) -> None:
    try:
        for directory in source.directories:
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
                raise ReadonlyDatabaseError
        parent = source.directories[-1].descriptor
        current = os.stat(source.path.name, dir_fd=parent, follow_symlinks=False)
        opened = os.fstat(source.descriptor)
        if not _same_file_status(source.status, current) or not _same_file_status(
            source.status, opened
        ):
            raise ReadonlyDatabaseError
        if _file_digest(source.descriptor) != source.digest:
            raise ReadonlyDatabaseError
        if not _same_file_status(source.status, os.fstat(source.descriptor)):
            raise ReadonlyDatabaseError
        sidecars = _sidecar_statuses(source.path.name, parent)
        for original, retained in zip(source.sidecars, sidecars, strict=True):
            if original is None or retained is None:
                if original is not retained:
                    raise ReadonlyDatabaseError
            elif not _same_file_status(original, retained):
                raise ReadonlyDatabaseError
    except (OSError, ValueError):
        raise ReadonlyDatabaseError from None


async def _validate_schema(connection: AsyncConnection, head: str) -> None:
    exists = await connection.scalar(
        text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='alembic_version'"
        )
    )
    if exists != 1:
        raise SchemaRevisionError(current_revision=None, expected_revision=head)
    matches = await connection.execute(
        text(
            "SELECT version_num = :head COLLATE BINARY "
            "FROM alembic_version LIMIT 2"
        ),
        {"head": head},
    )
    if matches.scalars().all() != [1]:
        # A corrupt revision may itself contain private data. Compare inside
        # SQLite without retaining or exposing the source-controlled value.
        raise SchemaRevisionError(current_revision=None, expected_revision=head)


def _authorize_readonly(
    action: int,
    first: str | None,
    second: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    if action == sqlite3.SQLITE_PRAGMA:
        name = (first or "").lower()
        if name in _PARAMETERIZED_READ_PRAGMAS or (
            name in _READ_PRAGMAS and second is None
        ):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in {
        "load_extension",
        "readfile",
        "writefile",
    }:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


async def _enable_authorizer(connection: AsyncConnection) -> None:
    raw_connection = await connection.get_raw_connection()
    driver = raw_connection.driver_connection
    if not isinstance(driver, SqliteConnection) or not hasattr(
        driver, "set_authorizer"
    ):
        raise ReadonlyDatabaseError
    # mode=ro protects the main file, not ATTACH targets. The SQLite authorizer
    # rejects those operations before SQLite can create another database file.
    await driver.set_authorizer(_authorize_readonly)


def readonly_sqlite_session(path: Path) -> AbstractAsyncContextManager[AsyncSession]:
    """Read an existing current-schema SQLite file without changing any bytes.

    The source must remain quiescent and have no nonempty write sidecars. It is
    fingerprinted before and after reading; concurrent changes fail closed. This
    does not checkpoint, migrate, recover, or stop a live source automatically.
    """
    return _readonly_sqlite_session(path)


@asynccontextmanager
async def _readonly_sqlite_session(path: Path) -> AsyncIterator[AsyncSession]:
    opening = asyncio.create_task(asyncio.to_thread(_open_source, path))
    try:
        source = await asyncio.shield(opening)
    except asyncio.CancelledError:
        # A cancelled to_thread await does not stop the worker. Retain its result
        # until descriptors can be closed rather than abandoning opened files.
        with suppress(ReadonlyDatabaseError):
            source = await opening
            await asyncio.to_thread(_close_source, source)
        raise
    engine = create_async_engine(
        URL.create(
            "sqlite+aiosqlite",
            database=source.path.as_uri(),
            query={"mode": "ro", "immutable": "1", "uri": "true"},
        ),
        poolclass=NullPool,
    )
    try:
        head = await asyncio.to_thread(installed_schema_head)
        async with engine.connect() as connection:
            await connection.execute(text("PRAGMA foreign_keys=ON"))
            await connection.execute(text("PRAGMA query_only=ON"))
            await _enable_authorizer(connection)
            await connection.execute(text("BEGIN"))
            await _validate_schema(connection, head)
            await asyncio.to_thread(_validate_source, source)
            async with AsyncSession(bind=connection, expire_on_commit=False) as session:
                yield session
    except SQLAlchemyError:
        raise ReadonlyDatabaseError from None
    finally:
        try:
            await engine.dispose()
            await asyncio.to_thread(_validate_source, source)
        finally:
            await asyncio.to_thread(_close_source, source)


__all__ = ["ReadonlyDatabaseError", "readonly_sqlite_session"]
