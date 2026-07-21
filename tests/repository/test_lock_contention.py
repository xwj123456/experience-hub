from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from experience_hub.storage.database import Database, DatabaseBusy


@pytest.fixture
async def database(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Database]:
    database_path = tmp_path / "lock-contention.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        busy_timeout_ms=0,
    )
    try:
        yield database
    finally:
        await database.dispose()


@pytest.mark.parametrize("holder_mode", ["immediate", "exclusive"])
@pytest.mark.asyncio
async def test_explicit_writer_lock_exhaustion_maps_to_database_busy(
    database: Database,
    holder_mode: str,
) -> None:
    async with database.transaction(**{holder_mode: True}):
        with pytest.raises(DatabaseBusy) as caught:
            async with database.transaction(immediate=True):
                pass

    error = caught.value
    assert error.code == "database_busy"
    assert error.status_code == 503
    assert error.retry_after == 5


@pytest.mark.asyncio
async def test_non_lock_operational_error_is_not_mapped(
    database: Database,
) -> None:
    with pytest.raises(OperationalError) as caught:
        async with database.transaction(immediate=True) as uow:
            await uow.session.execute(text("SELECT * FROM table_that_does_not_exist"))

    assert not isinstance(caught.value, DatabaseBusy)


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_busy_timeout_requires_a_nonnegative_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        Database.create(
            "sqlite+aiosqlite://",
            busy_timeout_ms=invalid,  # type: ignore[arg-type]
        )
