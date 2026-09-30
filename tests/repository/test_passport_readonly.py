from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiosqlite import Connection as SqliteConnection
from sqlalchemy import text
from sqlalchemy.engine import URL

from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.errors import DomainError
from experience_hub.ids import SequenceIdGenerator
from experience_hub.runtime import (
    ApplicationRuntime,
    SchemaRevisionError,
    installed_schema_head,
)
from experience_hub.storage import readonly as readonly_module
from experience_hub.storage.readonly import readonly_sqlite_session as _readonly_session

_SIDECARS = ("-wal", "-shm", "-journal")


def _retained_bytes(path: Path) -> dict[str, bytes | None]:
    return {
        suffix: candidate.read_bytes() if candidate.exists() else None
        for suffix in ("", *_SIDECARS)
        for candidate in (Path(f"{path}{suffix}"),)
    }


def _execute_sql(
    path: Path, statement: str, parameters: tuple[str, ...] = ()
) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(statement, parameters)


async def _create_database(path: Path, *, journal_mode: str = "wal") -> None:
    settings = Settings(
        database_url=URL.create("sqlite+aiosqlite", database=str(path))
    )
    runtime = ApplicationRuntime(
        settings,
        clock=FrozenClock(datetime(2026, 9, 30, tzinfo=UTC)),
        ids=SequenceIdGenerator([]),
    )
    async with (
        runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ) as container,
        container.database.transaction() as uow,
    ):
        await uow.session.execute(text("CREATE TABLE readonly_probe(value TEXT)"))
        await uow.session.execute(
            text("INSERT INTO readonly_probe VALUES ('original')")
        )
    if journal_mode == "delete":
        await asyncio.to_thread(_execute_sql, path, "PRAGMA journal_mode=DELETE")


@pytest.fixture
async def source_database(tmp_path: Path) -> Path:
    path = tmp_path.resolve() / "source.sqlite3"
    await _create_database(path)
    return path


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
@pytest.mark.asyncio
async def test_readonly_current_schema_preserves_database_and_sidecar_bytes(
    tmp_path: Path, journal_mode: str
) -> None:
    path = tmp_path.resolve() / "source.sqlite3"
    await _create_database(path, journal_mode=journal_mode)
    before = await asyncio.to_thread(_retained_bytes, path)
    database_bytes = before[""]
    assert database_bytes is not None
    expected_journal_header = b"\x02\x02" if journal_mode == "wal" else b"\x01\x01"
    assert database_bytes[18:20] == expected_journal_header

    async with _readonly_session(path) as session:
        assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
            "original"
        )
        assert await session.scalar(text("PRAGMA query_only")) == 1
        assert await session.scalar(text("PRAGMA foreign_keys")) == 1
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version"))
            == installed_schema_head()
        )

    assert await asyncio.to_thread(_retained_bytes, path) == before


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE forbidden(value TEXT)",
        "INSERT INTO readonly_probe VALUES ('forbidden')",
        "PRAGMA user_version=19",
    ],
)
@pytest.mark.asyncio
async def test_readonly_session_rejects_persistent_mutations(
    source_database: Path, statement: str
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)

    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database) as session:
            await session.execute(text(statement))

    assert caught.value.code == "readonly_database_invalid"
    assert str(source_database) not in str(caught.value)
    assert caught.value.details == {}
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_query_guard_cannot_be_disabled(
    source_database: Path,
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(DomainError):
        async with _readonly_session(source_database) as session:
            await session.execute(text("PRAGMA query_only=OFF"))
            await session.execute(
                text("INSERT INTO readonly_probe VALUES ('forbidden')")
            )
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_journal_mode_assignment_is_refused_without_changing_source_bytes(
    source_database: Path,
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(DomainError):
        async with _readonly_session(source_database) as session:
            await session.execute(text("PRAGMA journal_mode=DELETE"))
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.parametrize(
    "filename", ["exact?db.sqlite3", "exact#db.sqlite3", "数据 %25 空格.sqlite3"]
)
@pytest.mark.asyncio
async def test_readonly_sqlite_uses_exact_encoded_filename(
    tmp_path: Path, filename: str
) -> None:
    root = tmp_path.resolve()
    path = root / filename
    await _create_database(path)
    before_names = set(root.iterdir())
    before = await asyncio.to_thread(_retained_bytes, path)

    async with _readonly_session(path) as session:
        assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
            "original"
        )

    assert set(root.iterdir()) == before_names
    assert await asyncio.to_thread(_retained_bytes, path) == before


@pytest.mark.asyncio
async def test_readonly_missing_file_creates_no_directory_or_database(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    path = root / "missing" / "database.sqlite3"
    before = tuple(root.iterdir())

    with pytest.raises(DomainError) as caught:
        async with _readonly_session(path):
            pytest.fail("missing databases must not enter a session")

    assert caught.value.code == "readonly_database_invalid"
    assert caught.value.details == {}
    assert str(path) not in str(caught.value)
    assert tuple(root.iterdir()) == before


@pytest.mark.parametrize("revision", ["missing", "empty", "old", "new", "multiple"])
@pytest.mark.asyncio
async def test_readonly_refuses_noncurrent_or_ambiguous_schema_without_migration(
    source_database: Path, revision: str
) -> None:
    if revision == "missing":
        statement = "DROP TABLE alembic_version"
    elif revision == "empty":
        statement = "DELETE FROM alembic_version"
    elif revision == "old":
        statement = "UPDATE alembic_version SET version_num='0001_core'"
    elif revision == "new":
        statement = "UPDATE alembic_version SET version_num='future_schema_head'"
    else:
        statement = "INSERT INTO alembic_version VALUES ('other_schema_head')"
    await asyncio.to_thread(_execute_sql, source_database, statement)
    before = await asyncio.to_thread(_retained_bytes, source_database)

    with pytest.raises(SchemaRevisionError) as caught:
        async with _readonly_session(source_database):
            pytest.fail("unsupported schema must not enter a session")

    assert caught.value.code == "schema_version_unsupported"
    assert str(source_database) not in str(caught.value)
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.parametrize("node", ["symlink", "symlink_parent", "directory", "fifo"])
@pytest.mark.asyncio
async def test_readonly_refuses_unsafe_file_or_directory_chain(
    tmp_path: Path, source_database: Path, node: str
) -> None:
    root = tmp_path.resolve()
    path = root / "unsafe.sqlite3"
    if node == "symlink":
        path.symlink_to(source_database)
    elif node == "symlink_parent":
        parent = root / "linked"
        parent.symlink_to(root, target_is_directory=True)
        path = parent / source_database.name
    elif node == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    before = await asyncio.to_thread(_retained_bytes, source_database)

    with pytest.raises(DomainError) as caught:
        async with _readonly_session(path):
            pytest.fail("unsafe paths must not enter a session")

    assert caught.value.code == "readonly_database_invalid"
    assert str(path) not in str(caught.value)
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
@pytest.mark.asyncio
async def test_readonly_refuses_nonempty_write_sidecars(
    source_database: Path, suffix: str
) -> None:
    Path(f"{source_database}{suffix}").write_bytes(b"retained write-sidecar bytes")
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database):
            pytest.fail("immutable reads must not ignore write sidecars")
    assert caught.value.code == "readonly_database_invalid"
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_refuses_symlinked_sidecar(source_database: Path) -> None:
    Path(f"{source_database}-wal").symlink_to(source_database)
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(DomainError):
        async with _readonly_session(source_database):
            pytest.fail("unsafe sidecars must not enter a session")
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_detects_database_change_during_session(
    tmp_path: Path,
) -> None:
    path = tmp_path.resolve() / "changing.sqlite3"
    await _create_database(path, journal_mode="delete")
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(path) as session:
            assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
                "original"
            )
            await asyncio.to_thread(
                _execute_sql, path, "UPDATE readonly_probe SET value='competing'"
            )
    assert caught.value.code == "readonly_database_invalid"
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT value FROM readonly_probe").fetchone() == (
            "competing",
        )


@pytest.mark.asyncio
async def test_readonly_detects_database_replacement_during_session(
    source_database: Path,
) -> None:
    replacement = source_database.with_name("replacement.sqlite3")
    replacement.write_bytes(source_database.read_bytes())
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database):
            replacement.replace(source_database)
    assert caught.value.code == "readonly_database_invalid"


@pytest.mark.asyncio
async def test_readonly_closes_session_and_preserves_body_failure(
    source_database: Path,
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(ValueError, match="consumer failed"):
        async with _readonly_session(source_database) as session:
            assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
                "original"
            )
            raise ValueError("consumer failed")
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_filesystem_boundary_does_not_block_event_loop(
    source_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_thread = threading.current_thread()
    real_open = os.open
    real_stat = os.stat

    def guarded_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        assert threading.current_thread() is not main_thread
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def guarded_stat(
        path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        assert threading.current_thread() is not main_thread
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", guarded_open)
        patch.setattr(os, "stat", guarded_stat)
        async with _readonly_session(source_database) as session:
            assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
                "original"
            )


@pytest.mark.asyncio
async def test_readonly_refuses_corrupt_sqlite_without_changing_it(
    tmp_path: Path,
) -> None:
    path = tmp_path.resolve() / "corrupt.sqlite3"
    path.write_bytes(b"not a SQLite database")
    before = await asyncio.to_thread(_retained_bytes, path)
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(path):
            pytest.fail("corrupt SQLite must not enter a session")
    assert caught.value.code == "readonly_database_invalid"
    assert str(path) not in str(caught.value)
    assert await asyncio.to_thread(_retained_bytes, path) == before


@pytest.mark.asyncio
async def test_readonly_detects_sidecar_appearing_during_session(
    source_database: Path,
) -> None:
    sidecar = Path(f"{source_database}-wal")
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database) as session:
            assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
                "original"
            )
            sidecar.write_bytes(b"competing writer")
    assert caught.value.code == "readonly_database_invalid"
    assert sidecar.read_bytes() == b"competing writer"


@pytest.mark.parametrize("failure_index", [1, 2])
@pytest.mark.asyncio
async def test_readonly_closes_opened_descriptors_when_directory_stat_fails(
    source_database: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    retained: set[int] = set()
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat

    def tracked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        retained.add(descriptor)
        return descriptor

    def failing_fstat(descriptor: int) -> os.stat_result:
        if len(retained) == failure_index:
            raise OSError("injected filesystem diagnostic")
        return real_fstat(descriptor)

    def tracked_close(descriptor: int) -> None:
        real_close(descriptor)
        retained.discard(descriptor)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", tracked_open)
            patch.setattr(os, "fstat", failing_fstat)
            patch.setattr(os, "close", tracked_close)
            with pytest.raises(DomainError) as caught:
                async with _readonly_session(source_database):
                    pytest.fail("failed filesystem checks must not enter a session")
        assert caught.value.code == "readonly_database_invalid"
        assert "injected filesystem diagnostic" not in str(caught.value)
        assert retained == set()
    finally:
        for descriptor in retained:
            real_close(descriptor)


@pytest.mark.asyncio
async def test_readonly_cancellation_during_open_closes_retained_descriptors(
    source_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    opened = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()
    retained: list[readonly_module._Source] = []
    real_open = readonly_module._open_source

    def delayed_open(path: Path) -> readonly_module._Source:
        source = real_open(path)
        retained.append(source)
        loop.call_soon_threadsafe(opened.set)
        release.wait()
        loop.call_soon_threadsafe(finished.set)
        return source

    async def consume() -> None:
        async with _readonly_session(source_database):
            pytest.fail("cancelled opens must not enter a session")

    monkeypatch.setattr(readonly_module, "_open_source", delayed_open)
    task = asyncio.create_task(consume())
    try:
        await opened.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await finished.wait()
        for source in retained:
            with pytest.raises(OSError):
                os.fstat(source.descriptor)
            for directory in source.directories:
                with pytest.raises(OSError):
                    os.fstat(directory.descriptor)
        retained.clear()
    finally:
        release.set()
        for source in retained:
            readonly_module._close_source(source)


@pytest.mark.parametrize("operation", ["attach", "vacuum_into"])
@pytest.mark.asyncio
async def test_readonly_forbids_sql_operations_that_create_other_database_files(
    source_database: Path, operation: str
) -> None:
    destination = source_database.with_name("must-not-exist.sqlite3")
    before = await asyncio.to_thread(_retained_bytes, source_database)
    statement = (
        "ATTACH DATABASE :destination AS other"
        if operation == "attach"
        else "VACUUM INTO :destination"
    )
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database) as session:
            await session.execute(text(statement), {"destination": str(destination)})
    assert caught.value.code == "readonly_database_invalid"
    assert not destination.exists()
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_allows_parameterized_schema_introspection(
    source_database: Path,
) -> None:
    async with _readonly_session(source_database) as session:
        rows = (await session.execute(text("PRAGMA table_info(readonly_probe)"))).all()
    assert [row[1] for row in rows] == ["value"]


@pytest.mark.asyncio
async def test_readonly_refuses_driver_without_authorizer_support(
    source_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)
    monkeypatch.delattr(SqliteConnection, "set_authorizer")
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database):
            pytest.fail("drivers without authorization must not yield weak read-only")
    assert caught.value.code == "readonly_database_invalid"
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.parametrize("mode", [0o770, 0o777])
@pytest.mark.asyncio
async def test_readonly_refuses_nonsticky_shared_writable_parent(
    tmp_path: Path, mode: int
) -> None:
    parent = tmp_path.resolve() / "unsafe-parent"
    parent.mkdir(mode=0o700)
    path = parent / "source.sqlite3"
    await _create_database(path)
    parent.chmod(mode)
    before = await asyncio.to_thread(_retained_bytes, path)
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(path):
            pytest.fail("world-writable parent must not enter a session")
    assert caught.value.code == "readonly_database_invalid"
    assert await asyncio.to_thread(_retained_bytes, path) == before


@pytest.mark.asyncio
async def test_readonly_allows_sticky_shared_directory_ancestor(tmp_path: Path) -> None:
    parent = tmp_path.resolve() / "sticky-parent"
    parent.mkdir(mode=0o700)
    path = parent / "source.sqlite3"
    await _create_database(path)
    parent.chmod(0o1777)
    before = await asyncio.to_thread(_retained_bytes, path)
    async with _readonly_session(path) as session:
        assert await session.scalar(text("SELECT value FROM readonly_probe")) == (
            "original"
        )
    assert await asyncio.to_thread(_retained_bytes, path) == before


@pytest.mark.asyncio
async def test_readonly_detects_parent_becoming_unsafe_during_session(
    source_database: Path,
) -> None:
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(DomainError) as caught:
        async with _readonly_session(source_database):
            source_database.parent.chmod(0o770)
    assert caught.value.code == "readonly_database_invalid"
    assert await asyncio.to_thread(_retained_bytes, source_database) == before


@pytest.mark.asyncio
async def test_readonly_schema_error_does_not_echo_corrupt_revision_data(
    source_database: Path,
) -> None:
    private_revision = str(source_database.with_name("foreign-private.sqlite3"))
    await asyncio.to_thread(
        _execute_sql,
        source_database,
        "UPDATE alembic_version SET version_num=?",
        (private_revision,),
    )
    before = await asyncio.to_thread(_retained_bytes, source_database)
    with pytest.raises(SchemaRevisionError) as caught:
        async with _readonly_session(source_database):
            pytest.fail("corrupt schema revisions must not enter a session")
    assert caught.value.current_revision is None
    assert private_revision not in str(caught.value)
    assert await asyncio.to_thread(_retained_bytes, source_database) == before
