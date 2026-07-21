from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from experience_hub.storage.database import Database


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database.create(f"sqlite+aiosqlite:///{tmp_path / 'hub.sqlite3'}")
    try:
        yield database
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_database_enables_required_sqlite_pragmas(database: Database) -> None:
    async with database.read_session() as session:
        journal = await session.scalar(text("PRAGMA journal_mode"))
        foreign_keys = await session.scalar(text("PRAGMA foreign_keys"))
        busy_timeout = await session.scalar(text("PRAGMA busy_timeout"))

    assert str(journal).lower() == "wal"
    assert foreign_keys == 1
    assert busy_timeout == 5000


@pytest.mark.asyncio
async def test_transaction_commits_once_context_exits(database: Database) -> None:
    async with database.transaction() as uow:
        await uow.session.execute(text("CREATE TABLE committed (value INTEGER)"))
        await uow.session.execute(text("INSERT INTO committed VALUES (1)"))

    async with database.read_session() as session:
        assert await session.scalar(text("SELECT value FROM committed")) == 1


@pytest.mark.asyncio
async def test_transaction_rolls_back_when_context_raises(database: Database) -> None:
    async with database.transaction() as uow:
        await uow.session.execute(text("CREATE TABLE rolled_back (value INTEGER)"))

    with pytest.raises(RuntimeError, match="stop"):
        async with database.transaction() as uow:
            await uow.session.execute(text("INSERT INTO rolled_back VALUES (1)"))
            raise RuntimeError("stop")

    async with database.read_session() as session:
        count = await session.scalar(text("SELECT count(*) FROM rolled_back"))

    assert count == 0


@pytest.mark.parametrize("mode", ["immediate", "exclusive"])
@pytest.mark.asyncio
async def test_explicit_lock_mode_transaction_commits(
    database: Database,
    mode: str,
) -> None:
    async with database.transaction(**{mode: True}) as uow:
        await uow.session.execute(text("CREATE TABLE locked (value INTEGER)"))
        await uow.session.execute(text("INSERT INTO locked VALUES (1)"))

    async with database.read_session() as session:
        assert await session.scalar(text("SELECT value FROM locked")) == 1


def test_transaction_rejects_multiple_explicit_lock_modes(database: Database) -> None:
    with pytest.raises(
        ValueError,
        match="immediate and exclusive transaction modes are mutually exclusive",
    ):
        database.transaction(immediate=True, exclusive=True)
