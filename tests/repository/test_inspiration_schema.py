from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    CheckConstraint,
    Engine,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError

from experience_hub import canonical_json_bytes
from experience_hub.storage.tables import Base

NOW = datetime(2026, 7, 18, 8, tzinfo=UTC)
NOW_TEXT = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000201")
RUN_ID = UUID("00000000-0000-0000-0000-000000000202")
ITEM_ID = UUID("00000000-0000-0000-0000-000000000203")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000204")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000205")
IDEA_ID = UUID("00000000-0000-0000-0000-000000000206")
OCCURRENCE_ID = UUID("00000000-0000-0000-0000-000000000207")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64

INSPIRATION_COLUMNS = {
    "inspiration_runs": (
        "run_id",
        "owner_agent_id",
        "goal",
        "context",
        "mode",
        "generator_kind",
        "generator_configuration",
        "operators",
        "include_inbox",
        "branches_per_operator",
        "output_tokens_per_operator",
        "total_output_tokens",
        "operator_timeout_seconds",
        "global_timeout_seconds",
        "request_hash",
        "created_at",
    ),
    "inspiration_snapshot_items": (
        "snapshot_item_id",
        "run_id",
        "stable_evidence_key",
        "source_type",
        "source_id",
        "source_version_id",
        "source_state",
        "rank",
        "summary",
        "mechanism",
        "applicability",
        "tags",
        "excerpt",
        "source_trust",
        "content_hash",
        "falsifiers",
    ),
    "inspiration_ideas": (
        "idea_id",
        "run_id",
        "operator",
        "ordinal",
        "title",
        "hypothesis",
        "mechanism",
        "predictions",
        "falsifiers",
        "assumptions",
        "proposed_test",
        "evidence_references",
        "idea_content_hash",
        "mechanism_hash",
        "duplicate_relation",
    ),
    "idea_occurrences": (
        "occurrence_id",
        "idea_id",
        "mechanism_hash",
        "run_id",
        "snapshot_hash",
        "owner_agent_id",
        "occurred_at",
    ),
    "idea_adoption_records": (
        "adoption_id",
        "owner_agent_id",
        "idea_id",
        "run_id",
        "snapshot_hash",
        "evidence_snapshot_item_ids",
        "evidence_stable_keys",
        "resulting_experience_id",
        "resulting_version_id",
        "adopted_at",
    ),
    "inspiration_run_state": (
        "run_id",
        "status",
        "snapshot_hash",
        "operator_outcomes",
        "output_tokens_reserved",
        "output_tokens_consumed",
        "elapsed_milliseconds",
        "started_at",
        "completed_at",
        "projection_event_id",
    ),
    "mechanism_incubation": (
        "cluster_id",
        "canonical_mechanism_hash",
        "member_hashes",
        "occurrence_count",
        "distinct_snapshot_count",
        "distinct_adopter_count",
        "supported_count",
        "refuted_count",
        "maturity",
        "candidate_since",
        "last_signal_at",
        "projection_event_id",
    ),
    "idea_state": (
        "idea_id",
        "owner_agent_id",
        "mechanism_cluster_id",
        "owner_decision",
        "evaluations",
        "decision_reason",
        "resulting_experience_id",
        "resulting_version_id",
        "last_signal_at",
        "projection_event_id",
    ),
}

AUTHORITATIVE_TABLES = (
    "inspiration_runs",
    "inspiration_snapshot_items",
    "inspiration_ideas",
    "idea_occurrences",
    "idea_adoption_records",
)


@pytest.fixture
def migrated_engine(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Engine]:
    database_path = tmp_path / "inspiration.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
        yield engine
    finally:
        engine.dispose()


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


def _foreign_key_targets(
    engine: Engine,
    table_name: str,
) -> set[tuple[tuple[str, ...], str, tuple[str, ...]]]:
    return {
        (
            tuple(foreign_key["constrained_columns"]),
            str(foreign_key["referred_table"]),
            tuple(foreign_key["referred_columns"]),
        )
        for foreign_key in inspect(engine).get_foreign_keys(table_name)
    }


def _seed_run(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agents (agent_id, name, created_at) "
                "VALUES (:owner_id, 'Inspiration Owner', :now)"
            ),
            {"owner_id": str(OWNER_ID), "now": NOW_TEXT},
        )
        connection.execute(
            text(
                "INSERT INTO inspiration_runs "
                "(run_id, owner_agent_id, goal, context, mode, generator_kind, "
                "generator_configuration, operators, include_inbox, "
                "branches_per_operator, output_tokens_per_operator, "
                "total_output_tokens, operator_timeout_seconds, "
                "global_timeout_seconds, request_hash, created_at) VALUES "
                "(:run_id, :owner_id, 'Explain stale reads', 'service=ledger', "
                "'associative', 'deterministic', :empty_object, :operators, "
                "0, 3, 1200, 3600, 30, 90, :request_hash, :now)"
            ),
            {
                "run_id": str(RUN_ID),
                "owner_id": str(OWNER_ID),
                "empty_object": canonical_json_bytes({}),
                "operators": canonical_json_bytes(
                    ["causal_gap", "counterfactual", "distant_analogy"]
                ),
                "request_hash": HASH_A,
                "now": NOW_TEXT,
            },
        )


def _replace_with_pre_fix_snapshot_schema(engine: Engine) -> None:
    """Reproduce the short-lived 99ecacd 0004 table shape exactly enough."""
    table_name = "inspiration_snapshot_items"
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE tbl_name = :table_name AND sql IS NOT NULL "
                "ORDER BY CASE type WHEN 'index' THEN 1 ELSE 2 END, name"
            ),
            {"table_name": table_name},
        ).mappings()
        objects = tuple(
            (str(row["type"]), str(row["name"]), str(row["sql"]))
            for row in rows
        )
        table_sql = connection.scalar(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = :table_name"
            ),
            {"table_name": table_name},
        )
        assert isinstance(table_sql, str)
        old_table = f"{table_name}_canonical"
        connection.execute(text(f"ALTER TABLE {table_name} RENAME TO {old_table}"))
        legacy_sql = table_sql.replace(
            "tags BLOB NOT NULL,",
            "tags BLOB NOT NULL, falsifiers BLOB NOT NULL,",
            1,
        ).replace(
            "json_array_length(CAST(tags AS TEXT)) <= 32)",
            "json_array_length(CAST(tags AS TEXT)) <= 32 "
            "AND length(falsifiers) > 0 "
            "AND json_valid(CAST(falsifiers AS TEXT)) "
            "AND json_type(CAST(falsifiers AS TEXT)) = 'array' "
            "AND json_array_length(CAST(falsifiers AS TEXT)) <= 32)",
            1,
        )
        connection.execute(text(legacy_sql))
        connection.execute(text(f"DROP TABLE {old_table}"))
        for object_type, _, sql in objects:
            if object_type in {"index", "trigger"}:
                connection.execute(text(sql))


def _seed_snapshot_idea_and_occurrence(engine: Engine) -> None:
    _seed_run(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO inspiration_snapshot_items "
                "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
                "source_id, source_version_id, source_state, rank, summary, "
                "mechanism, applicability, tags, falsifiers, excerpt, source_trust, "
                "content_hash) VALUES "
                "(:item_id, :run_id, :stable_key, 'experience', :source_id, "
                ":version_id, 'warm', 1, 'A frozen observation', "
                "'Commit then invalidate', :one, :one, :one, 'Bounded excerpt', "
                "1.0, :content_hash)"
            ),
            {
                "item_id": str(ITEM_ID),
                "run_id": str(RUN_ID),
                "stable_key": HASH_B,
                "source_id": str(SOURCE_ID),
                "version_id": str(VERSION_ID),
                "one": canonical_json_bytes(["one"]),
                "content_hash": HASH_C,
            },
        )
        connection.execute(
            text(
                "INSERT INTO inspiration_ideas "
                "(idea_id, run_id, operator, ordinal, title, hypothesis, "
                "mechanism, predictions, falsifiers, assumptions, "
                "proposed_test, evidence_references, idea_content_hash, "
                "mechanism_hash, duplicate_relation) VALUES "
                "(:idea_id, :run_id, 'causal_gap', 1, 'Commit boundary', "
                "'A rollback exposes stale reads', 'Commit then invalidate', "
                ":one, :one, :one, 'Inject one rollback', :references, "
                ":idea_hash, :mechanism_hash, NULL)"
            ),
            {
                "idea_id": str(IDEA_ID),
                "run_id": str(RUN_ID),
                "one": canonical_json_bytes(["one"]),
                "references": canonical_json_bytes(
                    [
                        {
                            "type": "snapshot_item",
                            "id": str(ITEM_ID),
                            "stable_evidence_key": HASH_B,
                        }
                    ]
                ),
                "idea_hash": HASH_D,
                "mechanism_hash": HASH_C,
            },
        )
        connection.execute(
            text(
                "INSERT INTO idea_occurrences "
                "(occurrence_id, idea_id, mechanism_hash, run_id, "
                "snapshot_hash, owner_agent_id, occurred_at) VALUES "
                "(:occurrence_id, :idea_id, :mechanism_hash, :run_id, "
                ":snapshot_hash, :owner_id, :now)"
            ),
            {
                "occurrence_id": str(OCCURRENCE_ID),
                "idea_id": str(IDEA_ID),
                "mechanism_hash": HASH_C,
                "run_id": str(RUN_ID),
                "snapshot_hash": HASH_B,
                "owner_id": str(OWNER_ID),
                "now": NOW_TEXT,
            },
        )


def test_inspiration_migration_and_metadata_declare_exact_tables_and_columns(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    assert set(inspector.get_table_names()) >= set(INSPIRATION_COLUMNS)
    assert set(Base.metadata.tables) >= set(INSPIRATION_COLUMNS)
    for table_name, expected_columns in INSPIRATION_COLUMNS.items():
        assert tuple(
            str(column["name"]) for column in inspector.get_columns(table_name)
        ) == expected_columns
        assert (
            tuple(Base.metadata.tables[table_name].columns.keys())
            == expected_columns
        )


def test_inspiration_primary_and_foreign_keys_match_ownership(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    expected_primary_keys = {
        "inspiration_runs": ("run_id",),
        "inspiration_snapshot_items": ("snapshot_item_id",),
        "inspiration_ideas": ("idea_id",),
        "idea_occurrences": ("occurrence_id",),
        "idea_adoption_records": ("adoption_id",),
        "inspiration_run_state": ("run_id",),
        "mechanism_incubation": ("cluster_id",),
        "idea_state": ("idea_id",),
    }
    for table_name, expected in expected_primary_keys.items():
        assert tuple(
            inspector.get_pk_constraint(table_name)["constrained_columns"]
        ) == expected

    expected_foreign_keys = {
        "inspiration_runs": {
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "inspiration_snapshot_items": {
            (("run_id",), "inspiration_runs", ("run_id",)),
        },
        "inspiration_ideas": {
            (("run_id",), "inspiration_runs", ("run_id",)),
            (("duplicate_relation",), "inspiration_ideas", ("idea_id",)),
        },
        "idea_occurrences": {
            (("idea_id",), "inspiration_ideas", ("idea_id",)),
            (("run_id",), "inspiration_runs", ("run_id",)),
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "idea_adoption_records": {
            (("owner_agent_id",), "agents", ("agent_id",)),
            (("idea_id",), "inspiration_ideas", ("idea_id",)),
            (("run_id",), "inspiration_runs", ("run_id",)),
            (("resulting_experience_id",), "experiences", ("experience_id",)),
            (
                ("resulting_version_id",),
                "experience_versions",
                ("version_id",),
            ),
        },
        "inspiration_run_state": {
            (("run_id",), "inspiration_runs", ("run_id",)),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
        "mechanism_incubation": {
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
        "idea_state": {
            (("idea_id",), "inspiration_ideas", ("idea_id",)),
            (("owner_agent_id",), "agents", ("agent_id",)),
            (
                ("mechanism_cluster_id",),
                "mechanism_incubation",
                ("cluster_id",),
            ),
            (("resulting_experience_id",), "experiences", ("experience_id",)),
            (
                ("resulting_version_id",),
                "experience_versions",
                ("version_id",),
            ),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
    }
    for table_name, expected in expected_foreign_keys.items():
        assert _foreign_key_targets(migrated_engine, table_name) == expected


def test_inspiration_indexes_match_metadata_and_locked_uniqueness(
    migrated_engine: Engine,
) -> None:
    for table_name in INSPIRATION_COLUMNS:
        assert _index_keys(migrated_engine, table_name) == _metadata_index_keys(
            table_name
        )

    expected_unique_indexes = {
        (
            "ux_inspiration_snapshot_items_run_rank",
            ("run_id", "rank"),
            True,
        ),
        (
            "ux_inspiration_snapshot_items_run_source",
            ("run_id", "source_type", "source_id", "source_version_id"),
            True,
        ),
        (
            "ux_inspiration_ideas_run_operator_ordinal",
            ("run_id", "operator", "ordinal"),
            True,
        ),
        (
            "ux_idea_occurrences_run_mechanism",
            ("run_id", "mechanism_hash"),
            True,
        ),
        ("ux_idea_occurrences_idea", ("idea_id",), True),
        (
            "ux_idea_adoption_records_owner_idea",
            ("owner_agent_id", "idea_id"),
            True,
        ),
    }
    actual = {
        index
        for table_name in INSPIRATION_COLUMNS
        for index in _index_keys(migrated_engine, table_name)
    }
    assert expected_unique_indexes <= actual


def test_inspiration_checks_and_unique_constraints_match_metadata(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    for table_name in INSPIRATION_COLUMNS:
        reflected_checks = {
            str(constraint["name"]): " ".join(
                str(constraint["sqltext"]).split()
            )
            for constraint in inspector.get_check_constraints(table_name)
        }
        metadata_checks = {
            str(constraint.name): " ".join(str(constraint.sqltext).split())
            for constraint in Base.metadata.tables[table_name].constraints
            if isinstance(constraint, CheckConstraint)
        }
        assert reflected_checks == metadata_checks

        reflected_uniques = {
            (
                str(constraint["name"]),
                tuple(constraint["column_names"]),
            )
            for constraint in inspector.get_unique_constraints(table_name)
        }
        metadata_uniques = {
            (
                str(constraint.name),
                tuple(column.name for column in constraint.columns),
            )
            for constraint in Base.metadata.tables[table_name].constraints
            if isinstance(constraint, UniqueConstraint)
        }
        assert reflected_uniques == metadata_uniques


def test_authoritative_tables_are_immutable_and_projections_are_rebuildable(
    migrated_engine: Engine,
) -> None:
    triggers = {
        str(row[0])
        for row in migrated_engine.connect().execute(
            text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        )
    }
    for table_name in AUTHORITATIVE_TABLES:
        assert {
            f"{table_name}_reject_update",
            f"{table_name}_reject_delete",
            f"{table_name}_reject_conflicting_insert",
        } <= triggers
    for table_name in (
        "inspiration_run_state",
        "mechanism_incubation",
        "idea_state",
    ):
        assert not any(name.startswith(f"{table_name}_reject_") for name in triggers)

    _seed_run(migrated_engine)
    for statement in (
        "UPDATE inspiration_runs SET goal = 'Changed' WHERE run_id = :run_id",
        "DELETE FROM inspiration_runs WHERE run_id = :run_id",
    ):
        with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
            connection.execute(text(statement), {"run_id": str(RUN_ID)})


def test_logical_unique_keys_reject_duplicate_snapshot_and_occurrence_sources(
    migrated_engine: Engine,
) -> None:
    _seed_snapshot_idea_and_occurrence(migrated_engine)
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO inspiration_snapshot_items "
                "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
                "source_id, source_version_id, source_state, rank, summary, "
                "mechanism, applicability, tags, falsifiers, excerpt, source_trust, "
                "content_hash) SELECT :new_id, run_id, stable_evidence_key, "
                "source_type, source_id, source_version_id, source_state, "
                "2, summary, mechanism, applicability, tags, falsifiers, excerpt, "
                "source_trust, content_hash FROM inspiration_snapshot_items "
                "WHERE snapshot_item_id = :item_id"
            ),
            {"new_id": str(UUID(int=999)), "item_id": str(ITEM_ID)},
        )
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT OR REPLACE INTO idea_occurrences "
                "(occurrence_id, idea_id, mechanism_hash, run_id, "
                "snapshot_hash, owner_agent_id, occurred_at) VALUES "
                "(:new_id, :idea_id, :mechanism_hash, :run_id, "
                ":snapshot_hash, :owner_id, :now)"
            ),
            {
                "new_id": str(UUID(int=1_000)),
                "idea_id": str(IDEA_ID),
                "mechanism_hash": HASH_C,
                "run_id": str(RUN_ID),
                "snapshot_hash": HASH_B,
                "owner_id": str(OWNER_ID),
                "now": NOW_TEXT,
            },
        )


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("branches_per_operator", 0),
        ("branches_per_operator", 4),
        ("output_tokens_per_operator", 0),
        ("output_tokens_per_operator", 1_201),
        ("total_output_tokens", 0),
        ("total_output_tokens", 3_601),
        ("operator_timeout_seconds", 0),
        ("operator_timeout_seconds", 31),
        ("global_timeout_seconds", 0),
        ("global_timeout_seconds", 91),
    ),
)
def test_run_database_checks_lock_budget_boundaries(
    migrated_engine: Engine,
    column: str,
    value: int,
) -> None:
    _seed_run(migrated_engine)
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(text("DROP TRIGGER inspiration_runs_reject_update"))
        connection.execute(
            text(f"UPDATE inspiration_runs SET {column} = :value"),
            {"value": value},
        )


def test_snapshot_database_checks_lock_source_state_trust_and_utf8_limit(
    migrated_engine: Engine,
) -> None:
    _seed_run(migrated_engine)
    common = {
        "run_id": str(RUN_ID),
        "stable_key": HASH_B,
        "source_id": str(SOURCE_ID),
        "version_id": str(VERSION_ID),
        "one": canonical_json_bytes(["one"]),
        "content_hash": HASH_C,
    }
    statement = text(
        "INSERT INTO inspiration_snapshot_items "
        "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
        "source_id, source_version_id, source_state, rank, summary, "
        "mechanism, applicability, tags, falsifiers, excerpt, source_trust, "
        "content_hash) VALUES "
        "(:item_id, :run_id, :stable_key, :source_type, :source_id, "
        ":version_id, :source_state, 1, 'Summary', 'Mechanism', :one, "
        ":one, :one, :excerpt, :source_trust, :content_hash)"
    )
    invalid_rows = (
        {
            "item_id": str(UUID(int=2_001)),
            "source_type": "capsule",
            "source_state": "quarantined",
            "source_trust": 0.5,
            "excerpt": "bounded",
        },
        {
            "item_id": str(UUID(int=2_002)),
            "source_type": "experience",
            "source_state": "quarantined",
            "source_trust": 1.0,
            "excerpt": "bounded",
        },
        {
            "item_id": str(UUID(int=2_003)),
            "source_type": "experience",
            "source_state": "warm",
            "source_trust": 1.0,
            "excerpt": "知" * 683,
        },
    )
    for values in invalid_rows:
        with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
            connection.execute(statement, {**common, **values})


def test_empty_inspiration_migration_can_downgrade_cleanly(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "empty-inspiration-downgrade.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    command.downgrade(config, "0003_sharing")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        inspector = inspect(engine)
        assert not set(INSPIRATION_COLUMNS) & set(inspector.get_table_names())
        with engine.connect() as connection:
            version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
            assert version == "0003_sharing"
    finally:
        engine.dispose()


def test_falsifiers_upgrade_preserves_existing_0004_snapshot_rows(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspiration-falsifiers-upgrade.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "0004_inspiration")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        assert "falsifiers" not in {
            str(column["name"])
            for column in inspect(engine).get_columns(
                "inspiration_snapshot_items"
            )
        }
        _seed_run(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO inspiration_snapshot_items "
                    "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
                    "source_id, source_version_id, source_state, rank, summary, "
                    "mechanism, applicability, tags, excerpt, source_trust, "
                    "content_hash) VALUES "
                    "(:item_id, :run_id, :stable_key, 'experience', :source_id, "
                    ":version_id, 'warm', 1, 'Legacy snapshot', "
                    "'Legacy mechanism', :empty, :empty, '', 1.0, :content_hash)"
                ),
                {
                    "item_id": str(ITEM_ID),
                    "run_id": str(RUN_ID),
                    "stable_key": HASH_B,
                    "source_id": str(SOURCE_ID),
                    "version_id": str(VERSION_ID),
                    "empty": canonical_json_bytes([]),
                    "content_hash": HASH_C,
                },
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    upgraded = create_engine(f"sqlite:///{database_path}")
    try:
        assert "falsifiers" in {
            str(column["name"])
            for column in inspect(upgraded).get_columns(
                "inspiration_snapshot_items"
            )
        }
        with upgraded.connect() as connection:
            assert connection.scalar(
                text(
                    "SELECT falsifiers FROM inspiration_snapshot_items "
                    "WHERE snapshot_item_id = :item_id"
                ),
                {"item_id": str(ITEM_ID)},
            ) == canonical_json_bytes([])
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == "0005_inspiration_falsifiers"
            )
    finally:
        upgraded.dispose()


def test_falsifiers_upgrade_accepts_the_pre_fix_0004_column(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pre-fix-inspiration-falsifiers.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "0004_inspiration")
    engine = create_engine(f"sqlite:///{database_path}")
    legacy_falsifiers = canonical_json_bytes(["legacy falsifier"])
    try:
        _replace_with_pre_fix_snapshot_schema(engine)
        _seed_run(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO inspiration_snapshot_items "
                    "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
                    "source_id, source_version_id, source_state, rank, summary, "
                    "mechanism, applicability, tags, excerpt, source_trust, "
                    "content_hash, falsifiers) VALUES "
                    "(:item_id, :run_id, :stable_key, 'experience', :source_id, "
                    ":version_id, 'warm', 1, 'Pre-fix snapshot', "
                    "'Pre-fix mechanism', :empty, :empty, '', 1.0, "
                    ":content_hash, :falsifiers)"
                ),
                {
                    "item_id": str(ITEM_ID),
                    "run_id": str(RUN_ID),
                    "stable_key": HASH_B,
                    "source_id": str(SOURCE_ID),
                    "version_id": str(VERSION_ID),
                    "empty": canonical_json_bytes([]),
                    "content_hash": HASH_C,
                    "falsifiers": legacy_falsifiers,
                },
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    upgraded = create_engine(f"sqlite:///{database_path}")
    try:
        assert tuple(
            str(column["name"])
            for column in inspect(upgraded).get_columns(
                "inspiration_snapshot_items"
            )
        ) == INSPIRATION_COLUMNS["inspiration_snapshot_items"]
        checks = {
            str(check["name"]): str(check["sqltext"])
            for check in inspect(upgraded).get_check_constraints(
                "inspiration_snapshot_items"
            )
        }
        assert "falsifiers" not in checks["ck_inspiration_snapshot_items_arrays"]
        assert "falsifiers" in checks[
            "ck_inspiration_snapshot_items_falsifiers"
        ]
        with upgraded.connect() as connection:
            assert connection.scalar(
                text(
                    "SELECT falsifiers FROM inspiration_snapshot_items "
                    "WHERE snapshot_item_id = :item_id"
                ),
                {"item_id": str(ITEM_ID)},
            ) == legacy_falsifiers
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == "0005_inspiration_falsifiers"
            )
            assert (
                connection.scalar(
                    text(
                        "SELECT count(*) FROM sqlite_master "
                        "WHERE type = 'trigger' "
                        "AND tbl_name = 'inspiration_snapshot_items'"
                    )
                )
                == 3
            )
    finally:
        upgraded.dispose()


def test_pre_fix_empty_schema_can_upgrade_and_downgrade_to_canonical_0004(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pre-fix-empty-round-trip.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "0004_inspiration")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        _replace_with_pre_fix_snapshot_schema(engine)
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    command.downgrade(config, "0004_inspiration")

    downgraded = create_engine(f"sqlite:///{database_path}")
    try:
        assert "falsifiers" not in {
            str(column["name"])
            for column in inspect(downgraded).get_columns(
                "inspiration_snapshot_items"
            )
        }
        checks = {
            str(check["name"]): str(check["sqltext"])
            for check in inspect(downgraded).get_check_constraints(
                "inspiration_snapshot_items"
            )
        }
        assert "falsifiers" not in checks["ck_inspiration_snapshot_items_arrays"]
        with downgraded.connect() as connection:
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == "0004_inspiration"
            )
    finally:
        downgraded.dispose()


def test_falsifiers_migration_supports_only_safe_offline_upgrade_sql(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite:///{tmp_path / 'offline-falsifiers.sqlite3'}",
    )
    upgrade_sql = StringIO()
    config.output_buffer = upgrade_sql
    command.upgrade(config, "head", sql=True)
    assert "ADD COLUMN falsifiers" in upgrade_sql.getvalue()

    downgrade_sql = StringIO()
    config.output_buffer = downgrade_sql
    with pytest.raises(
        RuntimeError,
        match="offline downgrade cannot verify frozen inspiration data",
    ):
        command.downgrade(
            config,
            "0005_inspiration_falsifiers:0004_inspiration",
            sql=True,
        )
    assert "_alembic_tmp_inspiration_snapshot_items" not in (
        downgrade_sql.getvalue()
    )


def test_falsifiers_downgrade_refuses_to_discard_frozen_values(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "retained-falsifiers-downgrade.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    retained_falsifiers = canonical_json_bytes(["retained falsifier"])
    try:
        _seed_run(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO inspiration_snapshot_items "
                    "(snapshot_item_id, run_id, stable_evidence_key, source_type, "
                    "source_id, source_version_id, source_state, rank, summary, "
                    "mechanism, applicability, tags, excerpt, source_trust, "
                    "content_hash, falsifiers) VALUES "
                    "(:item_id, :run_id, :stable_key, 'experience', :source_id, "
                    ":version_id, 'warm', 1, 'Retained snapshot', "
                    "'Retained mechanism', :empty, :empty, '', 1.0, "
                    ":content_hash, :falsifiers)"
                ),
                {
                    "item_id": str(ITEM_ID),
                    "run_id": str(RUN_ID),
                    "stable_key": HASH_B,
                    "source_id": str(SOURCE_ID),
                    "version_id": str(VERSION_ID),
                    "empty": canonical_json_bytes([]),
                    "content_hash": HASH_C,
                    "falsifiers": retained_falsifiers,
                },
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="frozen inspiration falsifiers"):
        command.downgrade(config, "0004_inspiration")

    preserved = create_engine(f"sqlite:///{database_path}")
    try:
        assert "falsifiers" in {
            str(column["name"])
            for column in inspect(preserved).get_columns(
                "inspiration_snapshot_items"
            )
        }
        with preserved.connect() as connection:
            assert connection.scalar(
                text(
                    "SELECT falsifiers FROM inspiration_snapshot_items "
                    "WHERE snapshot_item_id = :item_id"
                ),
                {"item_id": str(ITEM_ID)},
            ) == retained_falsifiers
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == "0005_inspiration_falsifiers"
            )
    finally:
        preserved.dispose()


@pytest.mark.parametrize("retained_kind", ("source", "ledger"))
def test_inspiration_downgrade_refuses_retained_authority_before_ddl(
    repository_root: Path,
    tmp_path: Path,
    retained_kind: str,
) -> None:
    database_path = tmp_path / f"retained-{retained_kind}.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        if retained_kind == "source":
            _seed_run(engine)
        else:
            receipt_id = UUID(int=3_001)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO agents (agent_id, name, created_at) "
                        "VALUES (:owner_id, 'Ledger Owner', :now)"
                    ),
                    {"owner_id": str(OWNER_ID), "now": NOW_TEXT},
                )
                connection.execute(
                    text(
                        "INSERT INTO idempotency_records "
                        "(receipt_id, caller_scope, scope, idempotency_key, "
                        "request_hash, state, result_resource_type, "
                        "result_resource_id, response_status_code, "
                        "response_body, response_content_type, response_headers, "
                        "created_at, completed_at) VALUES "
                        "(:receipt_id, 'agent', 'inspiration.test', "
                        "'retained-ledger', :request_hash, 'in_progress', "
                        "NULL, NULL, NULL, NULL, NULL, NULL, :now, NULL)"
                    ),
                    {
                        "receipt_id": str(receipt_id),
                        "request_hash": HASH_A,
                        "now": NOW_TEXT,
                    },
                )
                connection.execute(
                    text(
                        "INSERT INTO domain_events "
                        "(aggregate_type, aggregate_id, sequence, event_type, "
                        "payload, actor_agent_id, causation_id, occurred_at) "
                        "VALUES ('inspiration_run', :run_id, 1, "
                        "'inspiration.started', :payload, :owner_id, "
                        ":receipt_id, :now)"
                    ),
                    {
                        "run_id": str(RUN_ID),
                        "payload": canonical_json_bytes({}),
                        "owner_id": str(OWNER_ID),
                        "receipt_id": str(receipt_id),
                        "now": NOW_TEXT,
                    },
                )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError):
        command.downgrade(config, "0003_sharing")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        assert set(INSPIRATION_COLUMNS) <= set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
            assert version == "0004_inspiration"
    finally:
        engine.dispose()
