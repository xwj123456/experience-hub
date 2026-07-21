from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from experience_hub.storage.tables import Base

EXPERIENCE_TABLES = {
    "experiences",
    "experience_versions",
    "experience_payloads",
    "experience_links",
    "experience_state",
    "experience_terms",
}


@pytest.fixture
def migrated_database_path(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Path]:
    database_path = tmp_path / "experiences.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    yield database_path


@pytest.fixture
def migrated_engine(migrated_database_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        yield engine
    finally:
        engine.dispose()


def _column_names(engine: Engine, table_name: str) -> tuple[str, ...]:
    return tuple(
        str(column["name"]) for column in inspect(engine).get_columns(table_name)
    )


def _index_keys(engine: Engine, table_name: str) -> set[tuple[object, ...]]:
    return {
        (
            index["name"],
            tuple(index["column_names"]),
            bool(index["unique"]),
        )
        for index in inspect(engine).get_indexes(table_name)
    }


def _metadata_index_keys(table_name: str) -> set[tuple[object, ...]]:
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            bool(index.unique),
        )
        for index in Base.metadata.tables[table_name].indexes
    }


def test_migration_declares_all_six_experience_tables(
    migrated_engine: Engine,
) -> None:
    assert set(inspect(migrated_engine).get_table_names()) >= EXPERIENCE_TABLES
    assert set(Base.metadata.tables) >= EXPERIENCE_TABLES


@pytest.mark.parametrize(
    ("table_name", "columns"),
    [
        (
            "experiences",
            (
                "experience_id",
                "owner_agent_id",
                "kind",
                "origin",
                "created_at",
            ),
        ),
        (
            "experience_versions",
            (
                "version_id",
                "experience_id",
                "version_number",
                "summary",
                "mechanism",
                "tags",
                "applicability",
                "evidence",
                "falsifiers",
                "content_hash",
                "supersedes_version_id",
                "created_at",
            ),
        ),
        (
            "experience_payloads",
            ("version_id", "codec", "payload", "payload_hash"),
        ),
        (
            "experience_links",
            (
                "source_experience_id",
                "source_version_id",
                "target_experience_id",
                "relation",
                "source_event_id",
            ),
        ),
        (
            "experience_state",
            (
                "experience_id",
                "owner_agent_id",
                "current_version_id",
                "current_content_hash",
                "temperature",
                "importance",
                "confidence",
                "activation_score",
                "source_trust",
                "access_count",
                "access_strength",
                "strength_updated_at",
                "last_accessed_at",
                "last_transition_at",
                "last_lifecycle_evaluated_at",
                "consecutive_below_threshold",
                "pinned",
                "projection_event_id",
            ),
        ),
        (
            "experience_terms",
            ("experience_id", "term", "term_kind", "weight"),
        ),
    ],
)
def test_experience_table_columns_match_design(
    migrated_engine: Engine,
    table_name: str,
    columns: tuple[str, ...],
) -> None:
    assert _column_names(migrated_engine, table_name) == columns


def test_migration_indexes_match_declared_metadata(migrated_engine: Engine) -> None:
    for table_name in EXPERIENCE_TABLES:
        assert _index_keys(migrated_engine, table_name) == _metadata_index_keys(
            table_name
        )

    assert (
        "ux_experience_versions_experience_number",
        ("experience_id", "version_number"),
        True,
    ) in _index_keys(migrated_engine, "experience_versions")
    assert (
        "ux_experience_state_owner_content",
        ("owner_agent_id", "current_content_hash"),
        True,
    ) in _index_keys(migrated_engine, "experience_state")
    assert (
        "ix_experience_terms_lookup",
        ("term_kind", "term", "experience_id"),
        False,
    ) in _index_keys(migrated_engine, "experience_terms")
    assert (
        "ix_experience_versions_supersedes",
        ("supersedes_version_id",),
        False,
    ) in _index_keys(migrated_engine, "experience_versions")
    assert (
        "ix_experience_state_current_version",
        ("current_version_id",),
        False,
    ) in _index_keys(migrated_engine, "experience_state")
    assert (
        "ix_experience_links_source_event",
        ("source_event_id",),
        False,
    ) in _index_keys(migrated_engine, "experience_links")


def test_experience_primary_and_foreign_keys_match_design(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    expected_primary_keys = {
        "experiences": ("experience_id",),
        "experience_versions": ("version_id",),
        "experience_payloads": ("version_id",),
        "experience_links": (
            "source_version_id",
            "target_experience_id",
            "relation",
        ),
        "experience_state": ("experience_id",),
        "experience_terms": ("experience_id", "term", "term_kind"),
    }
    for table_name, expected in expected_primary_keys.items():
        primary_key = inspector.get_pk_constraint(table_name)
        assert tuple(primary_key["constrained_columns"]) == expected

    foreign_key_targets = {
        table_name: {
            (
                tuple(foreign_key["constrained_columns"]),
                foreign_key["referred_table"],
                tuple(foreign_key["referred_columns"]),
            )
            for foreign_key in inspector.get_foreign_keys(table_name)
        }
        for table_name in EXPERIENCE_TABLES
    }
    assert (
        ("owner_agent_id",),
        "agents",
        ("agent_id",),
    ) in foreign_key_targets["experiences"]
    assert (
        ("experience_id",),
        "experiences",
        ("experience_id",),
    ) in foreign_key_targets["experience_versions"]
    assert (
        ("version_id",),
        "experience_versions",
        ("version_id",),
    ) in foreign_key_targets["experience_payloads"]
    assert (
        ("projection_event_id",),
        "domain_events",
        ("event_id",),
    ) in foreign_key_targets["experience_state"]
    assert (
        ("source_event_id",),
        "domain_events",
        ("event_id",),
    ) in foreign_key_targets["experience_links"]

    check_names = {
        table_name: {
            constraint["name"]
            for constraint in inspector.get_check_constraints(table_name)
        }
        for table_name in EXPERIENCE_TABLES
    }
    assert (
        "ck_experience_versions_supersession"
        in check_names["experience_versions"]
    )
    assert (
        "ck_experience_state_projection_event"
        in check_names["experience_state"]
    )


def test_experience_enum_range_unique_and_foreign_key_constraints(
    migrated_engine: Engine,
) -> None:
    owner_id = str(uuid4())
    experience_id = str(uuid4())
    target_id = str(uuid4())
    version_id = str(uuid4())
    receipt_id = str(uuid4())
    now = "2026-07-17T12:00:00.000000Z"
    content_hash = "a" * 64
    payload_hash = "b" * 64

    with migrated_engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
        connection.execute(
            text(
                "INSERT INTO agents (agent_id, name, created_at) "
                "VALUES (:owner_id, 'Owner', :now)"
            ),
            {"owner_id": owner_id, "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, created_at) VALUES "
                "(:receipt_id, 'system:local', 'schema.seed', 'seed', :hash, "
                "'in_progress', :now)"
            ),
            {
                "receipt_id": receipt_id,
                "hash": "c" * 64,
                "now": now,
            },
        )
        event_id = connection.execute(
            text(
                "INSERT INTO domain_events "
                "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                "actor_agent_id, causation_id, occurred_at) VALUES "
                "('experience', :experience_id, 1, 'experience.created', "
                ":payload, :owner_id, :receipt_id, :now) RETURNING event_id"
            ),
            {
                "experience_id": experience_id,
                "payload": b'{"schema_version":1}',
                "owner_id": owner_id,
                "receipt_id": receipt_id,
                "now": now,
            },
        ).scalar_one()
        for identity in (experience_id, target_id):
            connection.execute(
                text(
                    "INSERT INTO experiences "
                    "(experience_id, owner_agent_id, kind, origin, created_at) "
                    "VALUES (:experience_id, :owner_id, 'procedural', 'local', :now)"
                ),
                {
                    "experience_id": identity,
                    "owner_id": owner_id,
                    "now": now,
                },
            )
        connection.execute(
            text(
                "INSERT INTO experience_versions "
                "(version_id, experience_id, version_number, summary, mechanism, "
                "tags, applicability, evidence, falsifiers, content_hash, "
                "supersedes_version_id, created_at) VALUES "
                "(:version_id, :experience_id, 1, 'Summary', 'Mechanism', "
                ":empty, :empty, :empty, :empty, :content_hash, NULL, :now)"
            ),
            {
                "version_id": version_id,
                "experience_id": experience_id,
                "empty": b"[]",
                "content_hash": content_hash,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_payloads "
                "(version_id, codec, payload, payload_hash) "
                "VALUES (:version_id, 'plain', :payload, :payload_hash)"
            ),
            {
                "version_id": version_id,
                "payload": b'{"body":"body"}',
                "payload_hash": payload_hash,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_state "
                "(experience_id, owner_agent_id, current_version_id, "
                "current_content_hash, temperature, importance, confidence, "
                "activation_score, source_trust, access_count, access_strength, "
                "strength_updated_at, last_accessed_at, last_transition_at, "
                "last_lifecycle_evaluated_at, consecutive_below_threshold, pinned, "
                "projection_event_id) VALUES "
                "(:experience_id, :owner_id, :version_id, :content_hash, 'warm', "
                "0, 1, 0.3, 1, 0, 20, :now, NULL, :now, NULL, 0, 0, :event_id)"
            ),
            {
                "experience_id": experience_id,
                "owner_id": owner_id,
                "version_id": version_id,
                "content_hash": content_hash,
                "now": now,
                "event_id": event_id,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_links "
                "(source_experience_id, source_version_id, target_experience_id, "
                "relation, source_event_id) VALUES "
                "(:experience_id, :version_id, :target_id, 'supports', :event_id)"
            ),
            {
                "experience_id": experience_id,
                "version_id": version_id,
                "target_id": target_id,
                "event_id": event_id,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_terms "
                "(experience_id, term, term_kind, weight) "
                "VALUES (:experience_id, ' boundary ', 'char_trigram', 0.35)"
            ),
            {"experience_id": experience_id},
        )

    invalid_statements = (
        (
            "INSERT INTO experiences "
            "(experience_id, owner_agent_id, kind, origin, created_at) "
            "VALUES (:new_id, :owner_id, 'memory', 'local', :now)",
            {"new_id": str(uuid4()), "owner_id": owner_id, "now": now},
        ),
        (
            "INSERT INTO experiences "
            "(experience_id, owner_agent_id, kind, origin, created_at) "
            "VALUES (:new_id, :missing_owner, 'semantic', 'local', :now)",
            {
                "new_id": str(uuid4()),
                "missing_owner": str(uuid4()),
                "now": now,
            },
        ),
        (
            "INSERT INTO experience_versions "
            "(version_id, experience_id, version_number, summary, mechanism, "
            "tags, applicability, evidence, falsifiers, content_hash, created_at) "
            "VALUES (:new_id, :experience_id, 1, 'Other', 'Other', :empty, :empty, "
            ":empty, :empty, :hash, :now)",
            {
                "new_id": str(uuid4()),
                "experience_id": experience_id,
                "empty": b"[]",
                "hash": "d" * 64,
                "now": now,
            },
        ),
        (
            "INSERT INTO experience_payloads "
            "(version_id, codec, payload, payload_hash) "
            "VALUES (:missing_version, 'gzip', X'00', :hash)",
            {"missing_version": str(uuid4()), "hash": "e" * 64},
        ),
        (
            "UPDATE experience_state SET activation_score = 1.1 "
            "WHERE experience_id = :experience_id",
            {"experience_id": experience_id},
        ),
        (
            "UPDATE experience_state SET access_strength = 20.1 "
            "WHERE experience_id = :experience_id",
            {"experience_id": experience_id},
        ),
        (
            "INSERT INTO experience_terms "
            "(experience_id, term, term_kind, weight) "
            "VALUES (:experience_id, 'bad', 'phrase', 2)",
            {"experience_id": experience_id},
        ),
        (
            "INSERT INTO experience_links "
            "(source_experience_id, source_version_id, target_experience_id, "
            "relation, source_event_id) VALUES "
            "(:experience_id, :version_id, :target_id, 'supports', :event_id)",
            {
                "experience_id": experience_id,
                "version_id": version_id,
                "target_id": target_id,
                "event_id": event_id,
            },
        ),
    )
    for statement, parameters in invalid_statements:
        with (
            pytest.raises(IntegrityError),
            migrated_engine.begin() as connection,
        ):
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(text(statement), parameters)


def test_downgrade_to_core_removes_only_experience_schema(
    repository_root: Path,
    migrated_database_path: Path,
) -> None:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{migrated_database_path}")

    command.downgrade(config, "0001_core")

    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        tables = set(inspect(engine).get_table_names())
        assert not tables & EXPERIENCE_TABLES
        assert {
            "agents",
            "domain_events",
            "idempotency_records",
            "lifecycle_lease",
            "projection_versions",
        } <= tables
    finally:
        engine.dispose()


def test_populated_experience_downgrade_fails_closed_before_ddl(
    repository_root: Path,
    migrated_database_path: Path,
) -> None:
    owner_id = str(uuid4())
    experience_id = str(uuid4())
    now = "2026-07-17T12:00:00.000000Z"
    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(
                text(
                    "INSERT INTO agents (agent_id, name, created_at) "
                    "VALUES (:owner_id, 'Retained Owner', :now)"
                ),
                {"owner_id": owner_id, "now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO experiences "
                    "(experience_id, owner_agent_id, kind, origin, created_at) "
                    "VALUES (:experience_id, :owner_id, 'semantic', 'local', :now)"
                ),
                {
                    "experience_id": experience_id,
                    "owner_id": owner_id,
                    "now": now,
                },
            )
    finally:
        engine.dispose()

    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{migrated_database_path}")
    with pytest.raises(RuntimeError, match="experience source or ledger data"):
        command.downgrade(config, "0001_core")

    verification_engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        assert inspect(verification_engine).has_table("experiences")
        with verification_engine.connect() as connection:
            assert (
                connection.scalar(
                    text(
                        "SELECT count(*) FROM experiences "
                        "WHERE experience_id = :experience_id"
                    ),
                    {"experience_id": experience_id},
                )
                == 1
            )
    finally:
        verification_engine.dispose()


@pytest.mark.parametrize(
    ("aggregate_type", "event_type"),
    [
        ("experience", "legacy.experience_event"),
        ("legacy", "experience.created"),
    ],
)
def test_ledger_only_experience_downgrade_stops_after_empty_newer_schema(
    repository_root: Path,
    migrated_database_path: Path,
    aggregate_type: str,
    event_type: str,
) -> None:
    owner_id = str(uuid4())
    receipt_id = str(uuid4())
    aggregate_id = str(uuid4())
    now = "2026-07-17T12:00:00.000000Z"
    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(
                text(
                    "INSERT INTO agents (agent_id, name, created_at) "
                    "VALUES (:owner_id, 'Ledger Owner', :now)"
                ),
                {"owner_id": owner_id, "now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO idempotency_records "
                    "(receipt_id, caller_scope, scope, idempotency_key, "
                    "request_hash, state, created_at) VALUES "
                    "(:receipt_id, 'system:local', 'ledger.seed', 'seed', "
                    ":request_hash, 'in_progress', :now)"
                ),
                {
                    "receipt_id": receipt_id,
                    "request_hash": "f" * 64,
                    "now": now,
                },
            )
            event_id = connection.execute(
                text(
                    "INSERT INTO domain_events "
                    "(aggregate_type, aggregate_id, sequence, event_type, "
                    "payload, actor_agent_id, causation_id, occurred_at) VALUES "
                    "(:aggregate_type, :aggregate_id, 1, :event_type, :payload, "
                    ":owner_id, :receipt_id, :now) RETURNING event_id"
                ),
                {
                    "aggregate_type": aggregate_type,
                    "aggregate_id": aggregate_id,
                    "event_type": event_type,
                    "payload": b'{"schema_version":1}',
                    "owner_id": owner_id,
                    "receipt_id": receipt_id,
                    "now": now,
                },
            ).scalar_one()
            before_version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
            before_event = tuple(
                connection.execute(
                    text(
                        "SELECT * FROM domain_events "
                        "WHERE event_id = :event_id"
                    ),
                    {"event_id": event_id},
                ).one()
            )
        before_tables = (
            set(inspect(engine).get_table_names()) & EXPERIENCE_TABLES
        )
        assert before_tables == EXPERIENCE_TABLES
    finally:
        engine.dispose()

    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{migrated_database_path}")
    with pytest.raises(RuntimeError, match="experience source or ledger data"):
        command.downgrade(config, "0001_core")

    verification_engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        after_tables = (
            set(inspect(verification_engine).get_table_names())
            & EXPERIENCE_TABLES
        )
        assert after_tables == before_tables == EXPERIENCE_TABLES
        with verification_engine.connect() as connection:
            after_event = tuple(
                connection.execute(
                    text(
                        "SELECT * FROM domain_events "
                        "WHERE event_id = :event_id"
                    ),
                    {"event_id": event_id},
                ).one()
            )
            after_version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
        assert after_event == before_event
        assert before_version == "0005_inspiration_falsifiers"
        assert after_version == "0002_experiences"
    finally:
        verification_engine.dispose()
