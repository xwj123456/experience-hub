from __future__ import annotations

from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from experience_hub.storage.tables import Base

NOW = "2026-09-30T00:00:00.000000Z"
OWNER = UUID("30000000-0000-4000-8000-000000000001")
OTHER = UUID("30000000-0000-4000-8000-000000000002")
IMPORT = UUID("30000000-0000-4000-8000-000000000003")
ADOPTION = UUID("30000000-0000-4000-8000-000000000004")
EXPERIENCE = UUID("30000000-0000-4000-8000-000000000005")
VERSION = UUID("30000000-0000-4000-8000-000000000006")
HASH = "a" * 64
TABLES = {"passport_imports", "passport_adoptions", "passport_state"}


def _config(path: Path) -> Config:
    config = Config()
    config.attributes["configure_logger"] = False
    config.set_main_option(
        "script_location", str(files("experience_hub.storage") / "migrations")
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    return config


@pytest.fixture(params=("migration", "metadata"))
def engine(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Engine]:
    path = tmp_path / f"passport-{request.param}.sqlite3"
    if request.param == "migration":
        command.upgrade(_config(path), "head")
    result = create_engine(f"sqlite:///{path}")
    if request.param == "metadata":
        Base.metadata.create_all(result)
    with result.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    try:
        yield result
    finally:
        result.dispose()


def _seed(engine: Engine, *, target_owner: UUID = OWNER) -> None:
    assert set(inspect(engine).get_table_names()) >= TABLES
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO agents VALUES (:id, :name, :now)"),
            [
                {"id": str(value), "name": f"Synthetic Owner {index}", "now": NOW}
                for index, value in enumerate((OWNER, OTHER))
            ],
        )
        connection.execute(
            text(
                "INSERT INTO experiences VALUES "
                "(:id, :owner, 'semantic', 'local', :now)"
            ),
            {"id": str(EXPERIENCE), "owner": str(target_owner), "now": NOW},
        )
        connection.execute(
            text(
                "INSERT INTO experience_versions "
                "(version_id,experience_id,version_number,summary,mechanism,"
                "tags,applicability,evidence,falsifiers,content_hash,created_at) "
                "VALUES (:version,:experience,1,'Summary','Mechanism',"
                ":empty,:empty,:empty,:empty,:hash,:now)"
            ),
            {
                "version": str(VERSION),
                "experience": str(EXPERIENCE),
                "empty": b"[]",
                "hash": HASH,
                "now": NOW,
            },
        )


def _import(engine: Engine, **overrides: object) -> None:
    values: dict[str, object] = {
        "id": str(IMPORT),
        "owner": str(OWNER),
        "hash": HASH,
        "bytes": b'{"foreign_source":"not-a-local-agent"}',
        "now": NOW,
    }
    values.update(overrides)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO passport_imports "
                "(import_id,owner_agent_id,passport_hash,canonical_bytes,imported_at) "
                "VALUES (:id,:owner,:hash,:bytes,:now)"
            ),
            values,
        )


def _adopt(engine: Engine, **overrides: object) -> None:
    values: dict[str, object] = {
        "id": str(ADOPTION),
        "import": str(IMPORT),
        "owner": str(OWNER),
        "experience": str(EXPERIENCE),
        "version": str(VERSION),
        "hash": HASH,
        "created": 0,
        "importance": 0.7,
        "confidence": 0.6,
        "now": NOW,
    }
    values.update(overrides)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO passport_adoptions "
                "(adoption_id,import_id,owner_agent_id,resulting_experience_id,"
                "resulting_version_id,resulting_content_hash,created,importance,"
                "confidence,adopted_at) VALUES (:id,:import,:owner,:experience,"
                ":version,:hash,:created,:importance,:confidence,:now)"
            ),
            values,
        )


def test_schema_installs_sources_and_all_owner_foreign_keys(engine: Engine) -> None:
    _seed(engine)
    _import(engine)
    _adopt(engine)
    targets = inspect(engine).get_foreign_keys("passport_adoptions")
    assert any(
        item["constrained_columns"] == ["import_id", "owner_agent_id"]
        and item["referred_table"] == "passport_imports"
        for item in targets
    )
    assert any(
        item["constrained_columns"] == ["resulting_experience_id", "owner_agent_id"]
        and item["referred_table"] == "experiences"
        for item in targets
    )
    assert any(
        item["constrained_columns"]
        == ["resulting_version_id", "resulting_experience_id", "resulting_content_hash"]
        for item in targets
    )


def test_import_hash_uniqueness_is_owner_scoped(engine: Engine) -> None:
    _seed(engine)
    _import(engine)
    with pytest.raises(IntegrityError):
        _import(engine, id=str(UUID(int=31)))
    _import(engine, id=str(UUID(int=32)), owner=str(OTHER))


@pytest.mark.parametrize(
    "bad",
    (
        b"",
        b"[]",
        b'{ "a":1}',
        b'{"z":1,"a":2}',
        b'{"a":{"z":1,"a":2}}',
        b'{"a":1,"a":1}',
        b'{"a":1}\n',
        b'{"a":"' + b"x" * 524281 + b'"}',
    ),
)
def test_source_rejects_noncanonical_or_unbounded_bytes(
    engine: Engine, bad: bytes
) -> None:
    _seed(engine)
    with pytest.raises(IntegrityError):
        _import(engine, bytes=bad)


def test_source_accepts_exact_512kib_boundary(engine: Engine) -> None:
    _seed(engine)
    body = b'{"a":"' + b"x" * 524280 + b'"}'
    assert len(body) == 524288
    _import(engine, bytes=body)


@pytest.mark.parametrize(
    "field,value",
    (
        ("owner", str(OTHER)),
        ("hash", "b" * 64),
        ("version", str(UUID(int=41))),
        ("confidence", 1.01),
        ("importance", -0.1),
        ("created", 2),
    ),
)
def test_adoption_rejects_wrong_owner_target_hash_and_scores(
    engine: Engine, field: str, value: object
) -> None:
    _seed(engine)
    _import(engine)
    with pytest.raises(IntegrityError):
        _adopt(engine, **{field: value})


def test_adoption_cannot_point_at_another_owners_target(engine: Engine) -> None:
    _seed(engine, target_owner=OTHER)
    _import(engine)
    with pytest.raises(IntegrityError):
        _adopt(engine)


def test_adoption_is_unique_per_import(engine: Engine) -> None:
    _seed(engine)
    _import(engine)
    _adopt(engine)
    with pytest.raises(IntegrityError):
        _adopt(engine, id=str(UUID(int=42)))


@pytest.mark.parametrize("table", ("passport_imports", "passport_adoptions"))
@pytest.mark.parametrize(
    "operation", ("UPDATE", "DELETE", "INSERT OR REPLACE", "INSERT OR IGNORE")
)
def test_source_refuses_updates_deletes_and_replacement(
    engine: Engine, table: str, operation: str
) -> None:
    _seed(engine)
    _import(engine)
    _adopt(engine)
    statement = {
        "UPDATE": f"UPDATE {table} SET owner_agent_id=owner_agent_id",
        "DELETE": f"DELETE FROM {table}",
        "INSERT OR REPLACE": f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}",
        "INSERT OR IGNORE": f"INSERT OR IGNORE INTO {table} SELECT * FROM {table}",
    }[operation]
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text(statement))


def _event(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO idempotency_records "
                "(receipt_id,caller_scope,scope,idempotency_key,request_hash,"
                "state,created_at) VALUES (:id,:caller,'passport.import',"
                "'schema-fixture',:hash,'in_progress',:now)"
            ),
            {
                "id": str(UUID(int=71)),
                "caller": f"agent:{OWNER}",
                "hash": HASH,
                "now": NOW,
            },
        )
        connection.execute(
            text(
                "INSERT INTO domain_events "
                "(event_id,aggregate_type,aggregate_id,sequence,event_type,payload,"
                "actor_agent_id,causation_id,occurred_at) VALUES "
                "(1,'passport_import',:id,1,'passport.imported',:payload,:owner,:cause,:now)"
            ),
            {
                "id": str(IMPORT),
                "payload": b"{}",
                "owner": str(OWNER),
                "cause": str(UUID(int=71)),
                "now": NOW,
            },
        )


def _state(engine: Engine, **overrides: object) -> None:
    values: dict[str, object] = {
        "id": str(IMPORT),
        "owner": str(OWNER),
        "state": "pending",
        "adoption": None,
        "experience": None,
        "version": None,
        "code": None,
        "reason": None,
        "hash": None,
        "when": None,
        "event": 1,
    }
    values.update(overrides)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO passport_state "
                "(import_id,owner_agent_id,state,adoption_id,resulting_experience_id,"
                "resulting_version_id,reason_code,reason_text,reason_text_hash,decided_at,"
                "projection_event_id) VALUES (:id,:owner,:state,:adoption,:experience,"
                ":version,:code,:reason,:hash,:when,:event)"
            ),
            values,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"owner": str(OTHER)},
        {"event": 0},
        {"state": "unknown"},
        {"adoption": str(ADOPTION)},
        {"when": NOW},
        {"state": "adopted"},
        {"state": "rejected"},
        {
            "state": "rejected",
            "code": "reason",
            "reason": "text",
            "hash": "A" * 64,
            "when": NOW,
        },
    ),
)
def test_projection_rejects_wrong_owner_or_incomplete_terminal_shape(
    engine: Engine,
    overrides: dict[str, object],
) -> None:
    _seed(engine)
    _import(engine)
    _event(engine)
    with pytest.raises(IntegrityError):
        _state(engine, **overrides)


def test_adopted_projection_binds_exact_owned_lineage(engine: Engine) -> None:
    _seed(engine)
    _import(engine)
    _adopt(engine)
    _event(engine)
    values = {
        "state": "adopted",
        "adoption": str(ADOPTION),
        "experience": str(EXPERIENCE),
        "version": str(VERSION),
        "when": NOW,
    }
    with pytest.raises(IntegrityError):
        _state(engine, **{**values, "version": str(UUID(int=72))})
    _state(engine, **values)


@pytest.mark.parametrize("table", ("imports", "adoptions", "state"))
@pytest.mark.parametrize(
    "when",
    (
        "2026-09-30T00:00:00",
        "2026-09-30T00:00:00.000000+00:00",
        "2026-09-30T08:00:00.000000+08:00",
        "2026-09-30T00:00:00Z",
        "2026-09-30T00:00:00.000000z",
        "2026-02-30T00:00:00.000000Z",
        "2026-09-30T24:00:00.000000Z",
        "invalid timestamp",
    ),
)
def test_sources_and_projection_require_fixed_valid_utc_timestamps(
    engine: Engine,
    table: str,
    when: str,
) -> None:
    _seed(engine)
    if table != "imports":
        _import(engine)
    if table == "state":
        _event(engine)
    with pytest.raises(IntegrityError):
        if table == "imports":
            _import(engine, now=when)
        elif table == "adoptions":
            _adopt(engine, now=when)
        else:
            _state(
                engine,
                state="rejected",
                code="not_applicable",
                reason="not needed",
                hash=HASH,
                when=when,
            )


def test_upgrade_preserves_prior_sources_indexes_triggers_and_foreign_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "upgrade.sqlite3"
    config = _config(path)
    command.upgrade(config, "0007_capture_evidence_hashes")
    engine = create_engine(f"sqlite:///{path}")
    from tests.repository.test_candidate_schema import _seed_capture_sources

    _seed_capture_sources(engine)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO agents VALUES (:owner,'Prior Owner',:now)"),
            {"owner": str(OWNER), "now": NOW},
        )
        connection.execute(
            text("INSERT INTO experiences VALUES (:id,:owner,'semantic','local',:now)"),
            {"id": str(EXPERIENCE), "owner": str(OWNER), "now": NOW},
        )
        connection.execute(text("CREATE INDEX ix_prior_origin ON experiences(origin)"))
        connection.execute(
            text(
                "CREATE INDEX ix_prior_origin_expression ON experiences(lower(origin))"
            )
        )
        connection.execute(
            text(
                "CREATE TRIGGER prior_experience_guard BEFORE UPDATE ON experiences "
                "BEGIN SELECT RAISE(ABORT,'retained prior guard'); END"
            )
        )
    with engine.connect() as connection:
        previous_tables = tuple(
            table
            for table in sorted(inspect(engine).get_table_names())
            if table != "alembic_version"
        )
        before = {
            table: tuple(
                connection.execute(text(f'SELECT * FROM "{table}" ORDER BY rowid'))
            )
            for table in previous_tables
        }
        schema_before = tuple(
            connection.execute(
                text(
                    "SELECT type,name,sql FROM sqlite_master "
                    "WHERE tbl_name='experiences' AND type IN ('trigger','index') "
                    "AND sql IS NOT NULL ORDER BY type,name"
                )
            )
        )
    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert {
            table: tuple(
                connection.execute(text(f'SELECT * FROM "{table}" ORDER BY rowid'))
            )
            for table in previous_tables
        } == before
        assert (
            tuple(
                connection.execute(
                    text(
                        "SELECT type,name,sql FROM sqlite_master "
                        "WHERE tbl_name='experiences' AND type IN ('trigger','index') "
                        "AND sql IS NOT NULL ORDER BY type,name"
                    )
                )
            )
            == schema_before
        )
        assert not tuple(connection.execute(text("PRAGMA foreign_key_check")))
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0008_evidence_passports"
        )
    engine.dispose()


def test_empty_downgrade_restores_prior_origin_constraint(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite3"
    config = _config(path)
    command.upgrade(config, "head")
    command.downgrade(config, "0007_capture_evidence_hashes")
    engine = create_engine(f"sqlite:///{path}")
    assert not TABLES.intersection(inspect(engine).get_table_names())
    checks = inspect(engine).get_check_constraints("experiences")
    assert all("adopted_passport" not in item["sqltext"] for item in checks)
    engine.dispose()


@pytest.mark.parametrize("retained", ("source", "ledger", "origin"))
def test_downgrade_refuses_retained_passport_data(
    tmp_path: Path, retained: str
) -> None:
    path = tmp_path / "retained.sqlite3"
    config = _config(path)
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    _seed(engine)
    if retained == "source":
        _import(engine)
    else:
        with engine.begin() as connection:
            if retained == "origin":
                connection.execute(
                    text(
                        "INSERT INTO experiences VALUES "
                        "(:id,:owner,'semantic','adopted_passport',:now)"
                    ),
                    {"id": str(UUID(int=61)), "owner": str(OWNER), "now": NOW},
                )
            else:
                connection.execute(
                    text(
                        "INSERT INTO domain_events "
                        "(aggregate_type,aggregate_id,sequence,"
                        "event_type,payload,causation_id,occurred_at) "
                        "VALUES ('passport_import',:id,1,'passport.imported',"
                        ":payload,:causation,:now)"
                    ),
                    {
                        "id": str(IMPORT),
                        "payload": b"{}",
                        "causation": str(UUID(int=62)),
                        "now": NOW,
                    },
                )
    with pytest.raises(RuntimeError, match="Cannot downgrade"):
        command.downgrade(config, "0007_capture_evidence_hashes")
    assert set(inspect(engine).get_table_names()) >= TABLES
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0008_evidence_passports"
        )
    engine.dispose()
