"""A failed migration must leave a database safe for an ordinary retry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from tests.repository.test_candidate_schema import _seed_capture_sources
from tests.repository.test_passport_schema import (
    EXPERIENCE,
    NOW,
    OWNER,
    _config,
)


def _snapshot(engine: Engine) -> tuple[tuple, dict[str, tuple]]:
    with engine.connect() as connection:
        schema = tuple(
            connection.execute(
                text(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
                )
            )
        )
        rows = {
            table: tuple(connection.execute(text(f'SELECT * FROM "{table}"')))
            for table in sorted(inspect(engine).get_table_names())
        }
    return schema, rows


@pytest.mark.parametrize(
    "direction,failed_statement",
    (
        ("upgrade", "CREATE TABLE _alembic_tmp_experiences"),
        ("upgrade", "DROP TABLE experiences"),
        ("upgrade", "ALTER TABLE _alembic_tmp_experiences RENAME"),
        ("upgrade", "CREATE UNIQUE INDEX ux_experiences_id_owner"),
        ("upgrade", "CREATE TABLE passport_imports"),
        ("upgrade", "CREATE TRIGGER passport_imports_reject_update"),
        ("upgrade", "UPDATE alembic_version"),
        ("downgrade", "DROP TABLE passport_adoptions"),
        ("downgrade", "CREATE TABLE _alembic_tmp_experiences"),
        ("downgrade", "CREATE UNIQUE INDEX ux_experiences_id_owner"),
        ("downgrade", "UPDATE alembic_version"),
    ),
)
def test_failed_passport_migration_rolls_back_schema_and_retries_directly(
    tmp_path: Path, direction: str, failed_statement: str
) -> None:
    # Without real transactional DDL, failures lose immutable guards and indexes.
    database = tmp_path / "migration-recovery.sqlite3"
    config = _config(database)
    command.upgrade(config, "0007_capture_evidence_hashes")
    engine = create_engine(f"sqlite:///{database}")
    try:
        _seed_capture_sources(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO agents VALUES (:owner,'Prior Owner',:now)"),
                {"owner": str(OWNER), "now": NOW},
            )
            connection.execute(
                text(
                    "INSERT INTO experiences VALUES "
                    "(:id,:owner,'semantic','local',:now)"
                ),
                {"id": str(EXPERIENCE), "owner": str(OWNER), "now": NOW},
            )
            connection.execute(
                text("CREATE INDEX ix_prior_expression ON experiences(lower(origin))")
            )
        if direction == "downgrade":
            command.upgrade(config, "head")
        before = _snapshot(engine)
        fired = False

        def fail_once(connection, cursor, statement, parameters, context, executemany):
            nonlocal fired
            normalized = " ".join(statement.split())
            if not fired and normalized.startswith(failed_statement):
                fired = True
                raise OperationalError(
                    statement,
                    parameters,
                    sqlite3.OperationalError("synthetic migration interruption"),
                )

        def migrate() -> None:
            if direction == "upgrade":
                command.upgrade(config, "head")
            else:
                command.downgrade(config, "0007_capture_evidence_hashes")

        event.listen(Engine, "before_cursor_execute", fail_once)
        try:
            with pytest.raises(OperationalError, match="synthetic migration"):
                migrate()
        finally:
            event.remove(Engine, "before_cursor_execute", fail_once)
        assert fired
        assert _snapshot(engine) == before

        # Retry without repairing schema or touching the retained sources.
        migrate()
        after = _snapshot(engine)
        before_rows, after_rows = before[1], after[1]
        assert {
            table: rows
            for table, rows in after_rows.items()
            if table in before_rows and table != "alembic_version"
        } == {
            table: rows
            for table, rows in before_rows.items()
            if table in after_rows and table != "alembic_version"
        }
        assert {
            row for row in after[0] if row[2] == "experiences" and row[0] != "table"
        }
        assert {
            row for row in before[0] if row[2] == "experiences" and row[0] != "table"
        } == {row for row in after[0] if row[2] == "experiences" and row[0] != "table"}
        with engine.connect() as connection:
            assert tuple(connection.execute(text("PRAGMA foreign_key_check"))) == ()
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == (
                "0008_evidence_passports"
                if direction == "upgrade"
                else "0007_capture_evidence_hashes"
            )
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(text("UPDATE experiences SET origin=origin"))
    finally:
        engine.dispose()
