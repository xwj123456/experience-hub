from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import CheckConstraint, Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.experiences.models import ExperienceOrigin
from experience_hub.storage.tables import Base
from experience_hub.storage.tables.base import CanonicalJSONBytes

NOW = datetime(2026, 7, 22, 8, tzinfo=UTC)
NOW_TEXT = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000301")
BUNDLE_ID = UUID("00000000-0000-0000-0000-000000000302")
EVIDENCE_ID = UUID("00000000-0000-0000-0000-000000000303")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000304")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000305")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000306")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000307")
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000308")
OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000309")
OTHER_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000310")
OTHER_VERSION_ID = UUID("00000000-0000-0000-0000-000000000311")
SECOND_CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000312")
SECOND_ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000313")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64

SOURCE_TABLES = (
    "trajectory_bundles",
    "trajectory_evidence",
    "experience_candidates",
    "candidate_adoptions",
)
CAPTURE_COLUMNS = {
    "trajectory_bundles": (
        "bundle_id",
        "owner_agent_id",
        "trajectory_id",
        "adapter_kind",
        "adapter_version",
        "sanitization_profile",
        "manifest",
        "manifest_hash",
        "source_started_at",
        "source_completed_at",
        "captured_at",
    ),
    "trajectory_evidence": (
        "evidence_id",
        "bundle_id",
        "owner_agent_id",
        "step_id",
        "field",
        "ordinal",
        "excerpt",
        "excerpt_hash",
        "source_hash",
    ),
    "experience_candidates": (
        "candidate_id",
        "bundle_id",
        "owner_agent_id",
        "candidate_ordinal",
        "kind",
        "body",
        "summary",
        "mechanism",
        "tags",
        "applicability",
        "evidence",
        "evidence_refs",
        "falsifiers",
        "content_hash",
        "extractor_kind",
        "extractor_configuration_hash",
        "created_at",
    ),
    "candidate_adoptions": (
        "adoption_id",
        "candidate_id",
        "owner_agent_id",
        "resulting_experience_id",
        "resulting_version_id",
        "resulting_content_hash",
        "created",
        "adopted_at",
    ),
    "candidate_state": (
        "candidate_id",
        "owner_agent_id",
        "decision",
        "adoption_id",
        "resulting_experience_id",
        "resulting_version_id",
        "reason_code",
        "reason_text",
        "reason_text_hash",
        "decided_at",
        "projection_event_id",
    ),
}


def _manifest_document() -> dict[str, object]:
    return {
        "adapter": {"kind": "generic_jsonl", "version": 1},
        "owner_agent_id": str(OWNER_ID),
        "sanitization": {
            "input_sanitized": True,
            "profile_id": "trusted-v1",
        },
        "schema_version": 1,
        "source_completed_at": NOW_TEXT,
        "source_started_at": NOW_TEXT,
        "steps": [
            {
                "action_hash": HASH_A,
                "candidate_signal_hash": None,
                "observation_hash": HASH_B,
                "occurred_at": NOW_TEXT,
                "ordinal": 1,
                "outcome_hash": HASH_C,
                "status": "succeeded",
                "step_id": "step-1",
            }
        ],
        "trajectory_id": "trajectory-1",
    }


def _config(repository_root: Path, database_path: Path) -> Config:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


@pytest.fixture
def migrated_engine(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Engine]:
    database_path = tmp_path / "capture.sqlite3"
    command.upgrade(_config(repository_root, database_path), "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(params=("migration", "metadata"))
def capture_schema_engine(
    request: pytest.FixtureRequest,
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Engine]:
    database_path = tmp_path / f"capture-{request.param}.sqlite3"
    if request.param == "migration":
        command.upgrade(_config(repository_root, database_path), "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
        if request.param == "metadata":
            Base.metadata.create_all(engine)
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


def _seed_agent(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agents (agent_id, name, created_at) "
                "VALUES (:owner_id, 'Capture Owner', :now)"
            ),
            {"owner_id": str(OWNER_ID), "now": NOW_TEXT},
        )


def _insert_experience(engine: Engine, *, origin: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO experiences "
                "(experience_id, owner_agent_id, kind, origin, created_at) "
                "VALUES (:experience_id, :owner_id, 'semantic', :origin, :now)"
            ),
            {
                "experience_id": str(EXPERIENCE_ID),
                "owner_id": str(OWNER_ID),
                "origin": origin,
                "now": NOW_TEXT,
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
                "version_id": str(VERSION_ID),
                "experience_id": str(EXPERIENCE_ID),
                "empty": canonical_json_bytes([]),
                "content_hash": HASH_C,
                "now": NOW_TEXT,
            },
        )


def _seed_other_owner_experience(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agents (agent_id, name, created_at) "
                "VALUES (:owner_id, 'Other Capture Owner', :now)"
            ),
            {"owner_id": str(OTHER_OWNER_ID), "now": NOW_TEXT},
        )
        connection.execute(
            text(
                "INSERT INTO experiences "
                "(experience_id, owner_agent_id, kind, origin, created_at) "
                "VALUES (:experience_id, :owner_id, 'semantic', 'local', :now)"
            ),
            {
                "experience_id": str(OTHER_EXPERIENCE_ID),
                "owner_id": str(OTHER_OWNER_ID),
                "now": NOW_TEXT,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_versions "
                "(version_id, experience_id, version_number, summary, mechanism, "
                "tags, applicability, evidence, falsifiers, content_hash, "
                "supersedes_version_id, created_at) VALUES "
                "(:version_id, :experience_id, 1, 'Other summary', "
                "'Other mechanism', :empty, :empty, :empty, :empty, "
                ":content_hash, NULL, :now)"
            ),
            {
                "version_id": str(OTHER_VERSION_ID),
                "experience_id": str(OTHER_EXPERIENCE_ID),
                "empty": canonical_json_bytes([]),
                "content_hash": HASH_D,
                "now": NOW_TEXT,
            },
        )


def _insert_candidate_row(
    engine: Engine,
    *,
    candidate_id: UUID = SECOND_CANDIDATE_ID,
    owner_agent_id: UUID = OWNER_ID,
    ordinal: int = 2,
    evidence: bytes | None = None,
    evidence_refs: bytes | None = None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO experience_candidates "
                "(candidate_id, bundle_id, owner_agent_id, candidate_ordinal, "
                "kind, body, summary, mechanism, tags, applicability, evidence, "
                "evidence_refs, falsifiers, content_hash, extractor_kind, "
                "extractor_configuration_hash, created_at) VALUES "
                "(:candidate_id, :bundle_id, :owner_id, :ordinal, 'semantic', "
                "'Second body', 'Second summary', 'Second mechanism', :empty, "
                ":empty, :evidence, :evidence_refs, :empty, :content_hash, "
                "'deterministic_signal_v1', :configuration_hash, :now)"
            ),
            {
                "candidate_id": str(candidate_id),
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(owner_agent_id),
                "ordinal": ordinal,
                "empty": canonical_json_bytes([]),
                "evidence": (
                    canonical_json_bytes([]) if evidence is None else evidence
                ),
                "evidence_refs": (
                    canonical_json_bytes([])
                    if evidence_refs is None
                    else evidence_refs
                ),
                "content_hash": HASH_D,
                "configuration_hash": HASH_C,
                "now": NOW_TEXT,
            },
        )


def _insert_adoption_row(
    engine: Engine,
    *,
    candidate_id: UUID = SECOND_CANDIDATE_ID,
    owner_agent_id: UUID = OWNER_ID,
    experience_id: UUID = EXPERIENCE_ID,
    version_id: UUID = VERSION_ID,
    content_hash: str = HASH_C,
    adoption_id: UUID = SECOND_ADOPTION_ID,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO candidate_adoptions "
                "(adoption_id, candidate_id, owner_agent_id, "
                "resulting_experience_id, resulting_version_id, "
                "resulting_content_hash, created, adopted_at) VALUES "
                "(:adoption_id, :candidate_id, :owner_id, :experience_id, "
                ":version_id, :content_hash, 1, :now)"
            ),
            {
                "adoption_id": str(adoption_id),
                "candidate_id": str(candidate_id),
                "owner_id": str(owner_agent_id),
                "experience_id": str(experience_id),
                "version_id": str(version_id),
                "content_hash": content_hash,
                "now": NOW_TEXT,
            },
        )


def _insert_candidate_event(engine: Engine) -> int:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, created_at) VALUES "
                "(:receipt_id, 'agent', 'candidate.test', 'seed', :hash, "
                "'in_progress', :now)"
            ),
            {
                "receipt_id": str(RECEIPT_ID),
                "hash": HASH_A,
                "now": NOW_TEXT,
            },
        )
        return int(
            connection.execute(
                text(
                    "INSERT INTO domain_events "
                    "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                    "actor_agent_id, causation_id, occurred_at) VALUES "
                    "('experience_candidate', :candidate_id, 1, "
                    "'candidate.created', :payload, :owner_id, :receipt_id, :now) "
                    "RETURNING event_id"
                ),
                {
                    "candidate_id": str(CANDIDATE_ID),
                    "payload": canonical_json_bytes({}),
                    "owner_id": str(OWNER_ID),
                    "receipt_id": str(RECEIPT_ID),
                    "now": NOW_TEXT,
                },
            ).scalar_one()
        )


def _insert_trajectory_event(engine: Engine) -> int:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, request_hash, "
                "state, created_at) VALUES "
                "(:receipt_id, 'agent', 'trajectory.test', 'seed', :hash, "
                "'in_progress', :now)"
            ),
            {
                "receipt_id": str(RECEIPT_ID),
                "hash": HASH_A,
                "now": NOW_TEXT,
            },
        )
        return int(
            connection.execute(
                text(
                    "INSERT INTO domain_events "
                    "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                    "actor_agent_id, causation_id, occurred_at) VALUES "
                    "('trajectory_bundle', :bundle_id, 1, "
                    "'trajectory.captured', :payload, :owner_id, :receipt_id, :now) "
                    "RETURNING event_id"
                ),
                {
                    "bundle_id": str(BUNDLE_ID),
                    "payload": canonical_json_bytes({}),
                    "owner_id": str(OWNER_ID),
                    "receipt_id": str(RECEIPT_ID),
                    "now": NOW_TEXT,
                },
            ).scalar_one()
        )


def _seed_capture_sources(engine: Engine) -> None:
    _seed_agent(engine)
    _insert_experience(engine, origin="adopted_candidate")
    event_id = _insert_candidate_event(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO trajectory_bundles "
                "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                "adapter_version, sanitization_profile, manifest, manifest_hash, "
                "source_started_at, source_completed_at, captured_at) VALUES "
                "(:bundle_id, :owner_id, 'trajectory-1', 'generic_jsonl', 1, "
                "'trusted-v1', :manifest, :manifest_hash, :now, :now, :now)"
            ),
            {
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "manifest": canonical_json_bytes(_manifest_document()),
                "manifest_hash": HASH_A,
                "now": NOW_TEXT,
            },
        )
        connection.execute(
            text(
                "INSERT INTO trajectory_evidence "
                "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                "ordinal, excerpt, source_hash, excerpt_hash) VALUES "
                "(:evidence_id, :bundle_id, :owner_id, 'step-1', 'observation', "
                "1, 'Observed outcome', :source_hash, :excerpt_hash)"
            ),
            {
                "evidence_id": str(EVIDENCE_ID),
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "source_hash": HASH_B,
                "excerpt_hash": HASH_B,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_candidates "
                "(candidate_id, bundle_id, owner_agent_id, candidate_ordinal, "
                "kind, body, summary, mechanism, tags, applicability, evidence, "
                "evidence_refs, falsifiers, content_hash, extractor_kind, "
                "extractor_configuration_hash, created_at) VALUES "
                "(:candidate_id, :bundle_id, :owner_id, 1, 'semantic', 'Body', "
                "'Summary', 'Mechanism', :empty, :empty, :evidence, :refs, "
                ":empty, :content_hash, 'deterministic_signal_v1', "
                ":configuration_hash, :now)"
            ),
            {
                "candidate_id": str(CANDIDATE_ID),
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "empty": canonical_json_bytes([]),
                "evidence": canonical_json_bytes(
                    [
                        {
                            "id": f"{HASH_A}:step-1:observation",
                            "type": "trajectory_field",
                        }
                    ]
                ),
                "refs": canonical_json_bytes([str(EVIDENCE_ID)]),
                "content_hash": HASH_C,
                "configuration_hash": HASH_D,
                "now": NOW_TEXT,
            },
        )
        connection.execute(
            text(
                "INSERT INTO candidate_adoptions "
                "(adoption_id, candidate_id, owner_agent_id, "
                "resulting_experience_id, resulting_version_id, "
                "resulting_content_hash, created, adopted_at) VALUES "
                "(:adoption_id, :candidate_id, :owner_id, :experience_id, "
                ":version_id, :content_hash, 1, :now)"
            ),
            {
                "adoption_id": str(ADOPTION_ID),
                "candidate_id": str(CANDIDATE_ID),
                "owner_id": str(OWNER_ID),
                "experience_id": str(EXPERIENCE_ID),
                "version_id": str(VERSION_ID),
                "content_hash": HASH_C,
                "now": NOW_TEXT,
            },
        )
        connection.execute(
            text(
                "INSERT INTO candidate_state "
                "(candidate_id, owner_agent_id, decision, adoption_id, "
                "resulting_experience_id, resulting_version_id, reason_code, "
                "reason_text, reason_text_hash, decided_at, projection_event_id) "
                "VALUES (:candidate_id, :owner_id, 'adopted', :adoption_id, "
                ":experience_id, :version_id, NULL, NULL, NULL, :now, :event_id)"
            ),
            {
                "candidate_id": str(CANDIDATE_ID),
                "owner_id": str(OWNER_ID),
                "adoption_id": str(ADOPTION_ID),
                "experience_id": str(EXPERIENCE_ID),
                "version_id": str(VERSION_ID),
                "now": NOW_TEXT,
                "event_id": event_id,
            },
        )


def test_capture_tables_columns_and_origin_are_declared(
    migrated_engine: Engine,
) -> None:
    assert ExperienceOrigin.ADOPTED_CANDIDATE.value == "adopted_candidate"
    assert set(CAPTURE_COLUMNS) <= set(inspect(migrated_engine).get_table_names())
    assert set(CAPTURE_COLUMNS) <= set(Base.metadata.tables)
    for table_name, columns in CAPTURE_COLUMNS.items():
        assert _column_names(migrated_engine, table_name) == columns
    for table_name, column_name in (
        ("trajectory_bundles", "manifest"),
        ("experience_candidates", "tags"),
        ("experience_candidates", "applicability"),
        ("experience_candidates", "evidence"),
        ("experience_candidates", "evidence_refs"),
        ("experience_candidates", "falsifiers"),
    ):
        assert isinstance(
            Base.metadata.tables[table_name].c[column_name].type,
            CanonicalJSONBytes,
        )


def test_capture_foreign_keys_preserve_ownership_and_lineage(
    migrated_engine: Engine,
) -> None:
    expected = {
        "trajectory_bundles": {
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "trajectory_evidence": {
            (
                ("bundle_id", "owner_agent_id"),
                "trajectory_bundles",
                ("bundle_id", "owner_agent_id"),
            ),
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "experience_candidates": {
            (
                ("bundle_id", "owner_agent_id"),
                "trajectory_bundles",
                ("bundle_id", "owner_agent_id"),
            ),
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "candidate_adoptions": {
            (
                ("candidate_id", "owner_agent_id"),
                "experience_candidates",
                ("candidate_id", "owner_agent_id"),
            ),
            (("owner_agent_id",), "agents", ("agent_id",)),
            (
                ("resulting_experience_id", "owner_agent_id"),
                "experiences",
                ("experience_id", "owner_agent_id"),
            ),
            (
                (
                    "resulting_version_id",
                    "resulting_experience_id",
                    "resulting_content_hash",
                ),
                "experience_versions",
                ("version_id", "experience_id", "content_hash"),
            ),
        },
        "candidate_state": {
            (
                ("candidate_id", "owner_agent_id"),
                "experience_candidates",
                ("candidate_id", "owner_agent_id"),
            ),
            (("owner_agent_id",), "agents", ("agent_id",)),
            (
                (
                    "adoption_id",
                    "candidate_id",
                    "owner_agent_id",
                    "resulting_experience_id",
                    "resulting_version_id",
                ),
                "candidate_adoptions",
                (
                    "adoption_id",
                    "candidate_id",
                    "owner_agent_id",
                    "resulting_experience_id",
                    "resulting_version_id",
                ),
            ),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
    }
    for table_name, foreign_keys in expected.items():
        assert _foreign_key_targets(migrated_engine, table_name) == foreign_keys


def test_evidence_and_candidate_reject_a_different_bundle_owner(
    migrated_engine: Engine,
) -> None:
    _seed_capture_sources(migrated_engine)
    _seed_other_owner_experience(migrated_engine)
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO trajectory_evidence "
                "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                "ordinal, excerpt, source_hash, excerpt_hash) VALUES "
                "(:evidence_id, :bundle_id, :owner_id, 'cross-owner-step', "
                "'action', 2, 'Cross-owner excerpt', :source_hash, :excerpt_hash)"
            ),
            {
                "evidence_id": str(UUID(int=1_101)),
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OTHER_OWNER_ID),
                "source_hash": HASH_D,
                "excerpt_hash": HASH_D,
            },
        )
    with pytest.raises(IntegrityError):
        _insert_candidate_row(
            migrated_engine,
            owner_agent_id=OTHER_OWNER_ID,
        )


@pytest.mark.parametrize(
    ("owner_id", "experience_id", "version_id", "content_hash"),
    (
        (OTHER_OWNER_ID, OTHER_EXPERIENCE_ID, OTHER_VERSION_ID, HASH_D),
        (OWNER_ID, OTHER_EXPERIENCE_ID, OTHER_VERSION_ID, HASH_D),
        (OWNER_ID, EXPERIENCE_ID, OTHER_VERSION_ID, HASH_D),
        (OWNER_ID, EXPERIENCE_ID, VERSION_ID, HASH_D),
    ),
    ids=(
        "candidate-owner",
        "experience-owner",
        "version-experience",
        "version-content-hash",
    ),
)
def test_adoption_rejects_cross_owner_or_mismatched_result_lineage(
    migrated_engine: Engine,
    owner_id: UUID,
    experience_id: UUID,
    version_id: UUID,
    content_hash: str,
) -> None:
    _seed_capture_sources(migrated_engine)
    _seed_other_owner_experience(migrated_engine)
    _insert_candidate_row(migrated_engine)
    with pytest.raises(IntegrityError):
        _insert_adoption_row(
            migrated_engine,
            owner_agent_id=owner_id,
            experience_id=experience_id,
            version_id=version_id,
            content_hash=content_hash,
        )


def test_candidate_state_rejects_owner_and_adoption_tuple_mismatches(
    migrated_engine: Engine,
) -> None:
    _seed_capture_sources(migrated_engine)
    _seed_other_owner_experience(migrated_engine)
    _insert_candidate_row(migrated_engine)
    with migrated_engine.connect() as connection:
        event_id = connection.scalar(text("SELECT event_id FROM domain_events LIMIT 1"))
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO candidate_state "
                "(candidate_id, owner_agent_id, decision, adoption_id, "
                "resulting_experience_id, resulting_version_id, reason_code, "
                "reason_text, reason_text_hash, decided_at, projection_event_id) "
                "VALUES (:candidate_id, :owner_id, 'pending', NULL, NULL, NULL, "
                "NULL, NULL, NULL, NULL, :event_id)"
            ),
            {
                "candidate_id": str(SECOND_CANDIDATE_ID),
                "owner_id": str(OTHER_OWNER_ID),
                "event_id": event_id,
            },
        )
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO candidate_state "
                "(candidate_id, owner_agent_id, decision, adoption_id, "
                "resulting_experience_id, resulting_version_id, reason_code, "
                "reason_text, reason_text_hash, decided_at, projection_event_id) "
                "VALUES (:candidate_id, :owner_id, 'adopted', :adoption_id, "
                ":experience_id, :version_id, NULL, NULL, NULL, :now, :event_id)"
            ),
            {
                "candidate_id": str(SECOND_CANDIDATE_ID),
                "owner_id": str(OWNER_ID),
                "adoption_id": str(ADOPTION_ID),
                "experience_id": str(EXPERIENCE_ID),
                "version_id": str(VERSION_ID),
                "now": NOW_TEXT,
                "event_id": event_id,
            },
        )


def test_capture_indexes_match_metadata_and_lock_logical_identity(
    migrated_engine: Engine,
) -> None:
    for table_name in CAPTURE_COLUMNS:
        assert _index_keys(migrated_engine, table_name) == _metadata_index_keys(
            table_name
        )
    actual = {
        index
        for table_name in CAPTURE_COLUMNS
        for index in _index_keys(migrated_engine, table_name)
    }
    assert {
        (
            "ux_trajectory_bundles_owner_manifest",
            ("owner_agent_id", "manifest_hash"),
            True,
        ),
        (
            "ux_trajectory_bundles_id_owner",
            ("bundle_id", "owner_agent_id"),
            True,
        ),
        (
            "ix_trajectory_bundles_owner_trajectory",
            ("owner_agent_id", "trajectory_id", "captured_at", "bundle_id"),
            False,
        ),
        (
            "ux_trajectory_evidence_bundle_step_field",
            ("bundle_id", "step_id", "field"),
            True,
        ),
        (
            "ux_experience_candidates_bundle_ordinal",
            ("bundle_id", "candidate_ordinal"),
            True,
        ),
        (
            "ux_experience_candidates_id_owner",
            ("candidate_id", "owner_agent_id"),
            True,
        ),
        (
            "ux_candidate_adoptions_candidate",
            ("candidate_id",),
            True,
        ),
        (
            "ux_candidate_adoptions_lineage",
            (
                "adoption_id",
                "candidate_id",
                "owner_agent_id",
                "resulting_experience_id",
                "resulting_version_id",
            ),
            True,
        ),
    } <= actual
    assert (
        "ux_experiences_id_owner",
        ("experience_id", "owner_agent_id"),
        True,
    ) in _index_keys(migrated_engine, "experiences")
    assert (
        "ux_experience_versions_id_experience_content",
        ("version_id", "experience_id", "content_hash"),
        True,
    ) in _index_keys(migrated_engine, "experience_versions")


def test_capture_checks_match_metadata(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    for table_name in CAPTURE_COLUMNS:
        reflected = {
            str(constraint["name"]): " ".join(
                str(constraint["sqltext"]).split()
            )
            for constraint in inspector.get_check_constraints(table_name)
        }
        metadata = {
            str(constraint.name): " ".join(str(constraint.sqltext).split())
            for constraint in Base.metadata.tables[table_name].constraints
            if isinstance(constraint, CheckConstraint)
        }
        assert reflected == metadata


def test_capture_source_rows_are_immutable_and_projection_is_rebuildable(
    migrated_engine: Engine,
) -> None:
    _seed_capture_sources(migrated_engine)
    identities = {
        "trajectory_bundles": ("bundle_id", str(BUNDLE_ID)),
        "trajectory_evidence": ("evidence_id", str(EVIDENCE_ID)),
        "experience_candidates": ("candidate_id", str(CANDIDATE_ID)),
        "candidate_adoptions": ("adoption_id", str(ADOPTION_ID)),
    }
    for table_name in SOURCE_TABLES:
        identity_column, identity = identities[table_name]
        with (
            pytest.raises(IntegrityError, match="immutable"),
            migrated_engine.begin() as connection,
        ):
            connection.execute(
                text(
                    f"UPDATE {table_name} SET owner_agent_id = owner_agent_id "
                    f"WHERE {identity_column} = :identity"
                ),
                {"identity": identity},
            )
        with (
            pytest.raises(IntegrityError, match="immutable"),
            migrated_engine.begin() as connection,
        ):
            connection.execute(
                text(
                    f"DELETE FROM {table_name} "
                    f"WHERE {identity_column} = :identity"
                ),
                {"identity": identity},
            )

    with migrated_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE candidate_state SET decision = 'rejected', "
                "adoption_id = NULL, resulting_experience_id = NULL, "
                "resulting_version_id = NULL, reason_code = 'not_useful', "
                "reason_text = 'Not useful now', reason_text_hash = :reason_hash, "
                "decided_at = :now WHERE candidate_id = :candidate_id"
            ),
            {
                "reason_hash": HASH_D,
                "now": NOW_TEXT,
                "candidate_id": str(CANDIDATE_ID),
            },
        )
        decision = connection.scalar(
            text(
                "SELECT decision FROM candidate_state "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(CANDIDATE_ID)},
        )
    assert decision == "rejected"


def test_capture_conflicting_insert_triggers_protect_logical_identity(
    migrated_engine: Engine,
) -> None:
    _seed_capture_sources(migrated_engine)
    statements = (
        (
            "trajectory_bundles",
            "INSERT OR REPLACE INTO trajectory_bundles SELECT "
            f"'{UUID(int=901)}', owner_agent_id, trajectory_id, adapter_kind, "
            "adapter_version, sanitization_profile, manifest, manifest_hash, "
            "source_started_at, source_completed_at, captured_at "
            "FROM trajectory_bundles WHERE bundle_id = :identity",
            str(BUNDLE_ID),
        ),
        (
            "trajectory_evidence",
            "INSERT OR REPLACE INTO trajectory_evidence SELECT "
            f"'{UUID(int=902)}', bundle_id, owner_agent_id, step_id, field, "
            "ordinal, excerpt, source_hash, excerpt_hash FROM trajectory_evidence "
            "WHERE evidence_id = :identity",
            str(EVIDENCE_ID),
        ),
        (
            "experience_candidates",
            "INSERT OR REPLACE INTO experience_candidates SELECT "
            f"'{UUID(int=903)}', bundle_id, owner_agent_id, candidate_ordinal, "
            "kind, body, summary, mechanism, tags, applicability, evidence, "
            "evidence_refs, falsifiers, content_hash, extractor_kind, "
            "extractor_configuration_hash, created_at FROM experience_candidates "
            "WHERE candidate_id = :identity",
            str(CANDIDATE_ID),
        ),
        (
            "candidate_adoptions",
            "INSERT OR REPLACE INTO candidate_adoptions SELECT "
            f"'{UUID(int=904)}', candidate_id, owner_agent_id, "
            "resulting_experience_id, resulting_version_id, "
            "resulting_content_hash, created, adopted_at "
            "FROM candidate_adoptions WHERE adoption_id = :identity",
            str(ADOPTION_ID),
        ),
    )
    for table_name, statement, identity in statements:
        with (
            pytest.raises(IntegrityError, match=f"{table_name} identity"),
            migrated_engine.begin() as connection,
        ):
            connection.execute(text(statement), {"identity": identity})


@pytest.mark.parametrize(
    ("table_name", "column"),
    (
        ("trajectory_bundles", "manifest_hash"),
        ("trajectory_evidence", "source_hash"),
        ("trajectory_evidence", "excerpt_hash"),
        ("experience_candidates", "content_hash"),
        ("experience_candidates", "extractor_configuration_hash"),
        ("candidate_adoptions", "resulting_content_hash"),
        ("candidate_state", "reason_text_hash"),
    ),
)
def test_capture_sha256_checks_reject_invalid_digests(
    migrated_engine: Engine,
    table_name: str,
    column: str,
) -> None:
    _seed_capture_sources(migrated_engine)
    if table_name in SOURCE_TABLES:
        with migrated_engine.begin() as connection:
            connection.execute(text(f"DROP TRIGGER {table_name}_reject_update"))
    update = f"UPDATE {table_name} SET {column} = 'INVALID'"
    if table_name == "candidate_state":
        update += (
            ", decision = 'rejected', adoption_id = NULL, "
            "resulting_experience_id = NULL, resulting_version_id = NULL, "
            "reason_code = 'not_useful', reason_text = 'Not useful now', "
            f"decided_at = '{NOW_TEXT}'"
        )
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(text(update))


@pytest.mark.parametrize(
    ("table_name", "column"),
    (
        ("trajectory_bundles", "manifest"),
        ("experience_candidates", "tags"),
        ("experience_candidates", "applicability"),
        ("experience_candidates", "evidence"),
        ("experience_candidates", "evidence_refs"),
        ("experience_candidates", "falsifiers"),
    ),
)
def test_capture_json_checks_require_canonical_json(
    migrated_engine: Engine,
    table_name: str,
    column: str,
) -> None:
    _seed_capture_sources(migrated_engine)
    with migrated_engine.begin() as connection:
        connection.execute(text(f"DROP TRIGGER {table_name}_reject_update"))
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(f"UPDATE {table_name} SET {column} = :noncanonical"),
            {"noncanonical": b"[ ]" if column != "manifest" else b"{ }"},
        )


@pytest.mark.parametrize("case", ("unsorted", "duplicate"))
def test_manifest_rejects_noncanonical_object_key_order_and_duplicates(
    migrated_engine: Engine,
    case: str,
) -> None:
    _seed_agent(migrated_engine)
    document = _manifest_document()
    canonical = canonical_json_bytes(document)
    if case == "unsorted":
        unsorted = {
            "schema_version": document["schema_version"],
            **{
                key: value
                for key, value in document.items()
                if key != "schema_version"
            },
        }
        manifest = json.dumps(
            unsorted,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        manifest = canonical[:-1] + b',"trajectory_id":"duplicate"}'
    with (
        pytest.raises(IntegrityError, match="canonical"),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO trajectory_bundles "
                "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                "adapter_version, sanitization_profile, manifest, manifest_hash, "
                "source_started_at, source_completed_at, captured_at) VALUES "
                "(:bundle_id, :owner_id, 'trajectory-invalid', 'generic_jsonl', "
                "1, 'trusted-v1', :manifest, :manifest_hash, :now, :now, :now)"
            ),
            {
                "bundle_id": str(UUID(int=1_201)),
                "owner_id": str(OWNER_ID),
                "manifest": manifest,
                "manifest_hash": HASH_B,
                "now": NOW_TEXT,
            },
        )


@pytest.mark.parametrize("case", ("unsorted", "duplicate"))
def test_candidate_evidence_rejects_noncanonical_nested_object_keys(
    migrated_engine: Engine,
    case: str,
) -> None:
    _seed_capture_sources(migrated_engine)
    evidence_id = f"{HASH_A}:step-1:observation"
    if case == "unsorted":
        evidence = json.dumps(
            [{"type": "trajectory_field", "id": evidence_id}],
            separators=(",", ":"),
        ).encode()
    else:
        evidence = (
            f'[{{"id":"{evidence_id}","id":"duplicate",'
            '"type":"trajectory_field"}]'
        ).encode()
    with pytest.raises(IntegrityError, match="canonical"):
        _insert_candidate_row(
            migrated_engine,
            evidence=evidence,
            evidence_refs=canonical_json_bytes([str(EVIDENCE_ID)]),
        )


@pytest.mark.parametrize("field_path", ("trajectory_id", "step_id"))
def test_manifest_rejects_unicode_escape_spelling_in_known_text_fields(
    capture_schema_engine: Engine,
    field_path: str,
) -> None:
    _seed_agent(capture_schema_engine)
    document = _manifest_document()
    if field_path == "trajectory_id":
        document["trajectory_id"] = "轨迹-1"
    else:
        steps = document["steps"]
        assert isinstance(steps, list)
        step = steps[0]
        assert isinstance(step, dict)
        step["step_id"] = "步骤-1"
    noncanonical = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert noncanonical != canonical_json_bytes(document)
    assert b"\\u" in noncanonical

    with (
        pytest.raises(IntegrityError, match="canonical"),
        capture_schema_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO trajectory_bundles "
                "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                "adapter_version, sanitization_profile, manifest, manifest_hash, "
                "source_started_at, source_completed_at, captured_at) VALUES "
                "(:bundle_id, :owner_id, 'trajectory-escaped', 'generic_jsonl', "
                "1, 'trusted-v1', :manifest, :manifest_hash, :now, :now, :now)"
            ),
            {
                "bundle_id": str(UUID(int=1_301)),
                "owner_id": str(OWNER_ID),
                "manifest": noncanonical,
                "manifest_hash": HASH_B,
                "now": NOW_TEXT,
            },
        )


@pytest.mark.parametrize("field", ("id", "type"))
def test_candidate_evidence_rejects_unicode_escape_spelling_in_known_text_fields(
    capture_schema_engine: Engine,
    field: str,
) -> None:
    _seed_capture_sources(capture_schema_engine)
    evidence_item = {
        "id": f"{HASH_A}:step-1:observation",
        "type": "trajectory_field",
    }
    evidence_item[field] = f"字段-{field}"
    noncanonical = json.dumps(
        [evidence_item],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert noncanonical != canonical_json_bytes([evidence_item])
    assert b"\\u" in noncanonical

    with pytest.raises(IntegrityError, match="canonical"):
        _insert_candidate_row(
            capture_schema_engine,
            evidence=noncanonical,
            evidence_refs=canonical_json_bytes([str(EVIDENCE_ID)]),
        )


def test_known_capture_json_accepts_canonical_scalars_arrays_and_special_strings(
    capture_schema_engine: Engine,
) -> None:
    _seed_agent(capture_schema_engine)
    document = _manifest_document()
    document["trajectory_id"] = '轨迹"\\u4f60\n'
    sanitization = document["sanitization"]
    assert isinstance(sanitization, dict)
    sanitization["profile_id"] = '策略"\\u597d\t'
    steps = document["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["step_id"] = '步骤"\\u5b57\r'
    manifest = canonical_json_bytes(document)
    assert b"\\\\u" in manifest
    assert b"\\u8f68" not in manifest

    with capture_schema_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO trajectory_bundles "
                "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                "adapter_version, sanitization_profile, manifest, manifest_hash, "
                "source_started_at, source_completed_at, captured_at) VALUES "
                "(:bundle_id, :owner_id, :trajectory_id, 'generic_jsonl', 1, "
                ":profile, :manifest, :manifest_hash, :now, :now, :now)"
            ),
            {
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "trajectory_id": document["trajectory_id"],
                "profile": sanitization["profile_id"],
                "manifest": manifest,
                "manifest_hash": HASH_A,
                "now": NOW_TEXT,
            },
        )

    evidence = canonical_json_bytes(
        [{"id": '证据"\\u0041\n', "type": "trajectory_field"}]
    )
    assert b"\\\\u0041" in evidence
    assert b"\\u8bc1" not in evidence
    _insert_candidate_row(
        capture_schema_engine,
        evidence=evidence,
        evidence_refs=canonical_json_bytes([str(EVIDENCE_ID)]),
    )


@pytest.mark.parametrize(
    (
        "decision",
        "adoption_id",
        "experience_id",
        "version_id",
        "reason_code",
        "reason_text",
        "reason_hash",
        "decided_at",
    ),
    (
        ("pending", None, None, None, None, None, None, NOW_TEXT),
        (
            "adopted",
            str(ADOPTION_ID),
            str(EXPERIENCE_ID),
            str(VERSION_ID),
            "reason",
            None,
            None,
            NOW_TEXT,
        ),
        (
            "rejected",
            None,
            None,
            None,
            None,
            "Reason",
            HASH_D,
            NOW_TEXT,
        ),
        (
            "rejected",
            str(ADOPTION_ID),
            str(EXPERIENCE_ID),
            str(VERSION_ID),
            "reason",
            "Reason",
            HASH_D,
            NOW_TEXT,
        ),
        ("unknown", None, None, None, None, None, None, None),
    ),
)
def test_candidate_state_rejects_invalid_decision_shapes(
    migrated_engine: Engine,
    decision: str,
    adoption_id: str | None,
    experience_id: str | None,
    version_id: str | None,
    reason_code: str | None,
    reason_text: str | None,
    reason_hash: str | None,
    decided_at: str | None,
) -> None:
    _seed_capture_sources(migrated_engine)
    with migrated_engine.begin() as connection:
        connection.execute(text("DELETE FROM candidate_state"))
    with (
        pytest.raises(IntegrityError),
        migrated_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "INSERT INTO candidate_state "
                "(candidate_id, owner_agent_id, decision, adoption_id, "
                "resulting_experience_id, resulting_version_id, reason_code, "
                "reason_text, reason_text_hash, decided_at, "
                "projection_event_id) VALUES "
                "(:candidate_id, :owner_id, :decision, :adoption_id, "
                ":experience_id, :version_id, :reason_code, :reason_text, "
                ":reason_hash, :decided_at, "
                "(SELECT event_id FROM domain_events LIMIT 1))"
            ),
            {
                "candidate_id": str(CANDIDATE_ID),
                "owner_id": str(OWNER_ID),
                "decision": decision,
                "adoption_id": adoption_id,
                "experience_id": experience_id,
                "version_id": version_id,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "reason_hash": reason_hash,
                "decided_at": decided_at,
            },
        )


def test_candidate_state_accepts_each_exact_decision_shape(
    migrated_engine: Engine,
) -> None:
    _seed_capture_sources(migrated_engine)
    with migrated_engine.begin() as connection:
        connection.execute(text("DELETE FROM candidate_state"))
        event_id = connection.scalar(text("SELECT event_id FROM domain_events LIMIT 1"))
        for values in (
            {
                "decision": "pending",
                "adoption_id": None,
                "experience_id": None,
                "version_id": None,
                "reason_code": None,
                "reason_text": None,
                "reason_hash": None,
                "decided_at": None,
            },
            {
                "decision": "adopted",
                "adoption_id": str(ADOPTION_ID),
                "experience_id": str(EXPERIENCE_ID),
                "version_id": str(VERSION_ID),
                "reason_code": None,
                "reason_text": None,
                "reason_hash": None,
                "decided_at": NOW_TEXT,
            },
            {
                "decision": "rejected",
                "adoption_id": None,
                "experience_id": None,
                "version_id": None,
                "reason_code": "not_useful",
                "reason_text": "Not useful now",
                "reason_hash": HASH_D,
                "decided_at": NOW_TEXT,
            },
        ):
            connection.execute(
                text(
                    "INSERT INTO candidate_state "
                    "(candidate_id, owner_agent_id, decision, adoption_id, "
                    "resulting_experience_id, resulting_version_id, reason_code, "
                    "reason_text, reason_text_hash, decided_at, "
                    "projection_event_id) VALUES "
                    "(:candidate_id, :owner_id, :decision, :adoption_id, "
                    ":experience_id, :version_id, :reason_code, :reason_text, "
                    ":reason_hash, :decided_at, :event_id)"
                ),
                {
                    **values,
                    "candidate_id": str(CANDIDATE_ID),
                    "owner_id": str(OWNER_ID),
                    "event_id": event_id,
                },
            )
            connection.execute(text("DELETE FROM candidate_state"))


def test_upgrade_from_0005_preserves_experiences_and_immutability(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "populated-0005.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "0005_inspiration_falsifiers")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        _seed_agent(engine)
        _insert_experience(engine, origin="local")
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    upgraded = create_engine(f"sqlite:///{database_path}")
    try:
        with upgraded.connect() as connection:
            assert connection.scalar(text("SELECT origin FROM experiences")) == "local"
            triggers = {
                str(row[0])
                for row in connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'trigger' AND tbl_name = 'experiences'"
                    )
                )
            }
        assert triggers == {
            "experiences_reject_update",
            "experiences_reject_delete",
            "experiences_reject_conflicting_insert",
        }
        assert (
            "ix_experiences_owner_created",
            ("owner_agent_id", "created_at", "experience_id"),
            False,
        ) in _index_keys(upgraded, "experiences")
        for statement in (
            "UPDATE experiences SET origin = 'adopted_candidate'",
            "DELETE FROM experiences",
        ):
            with (
                pytest.raises(IntegrityError, match="immutable"),
                upgraded.begin() as connection,
            ):
                connection.execute(text(statement))
    finally:
        upgraded.dispose()


def test_empty_capture_migration_downgrades_to_0005(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "empty-capture-downgrade.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "head")
    command.downgrade(config, "0005_inspiration_falsifiers")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        assert not set(CAPTURE_COLUMNS) & set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == "0005_inspiration_falsifiers"
        _seed_agent(engine)
        with pytest.raises(IntegrityError):
            _insert_experience(engine, origin="adopted_candidate")
    finally:
        engine.dispose()


def test_offline_capture_downgrade_fails_closed(
    repository_root: Path,
) -> None:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", "sqlite:///offline-capture.sqlite3")
    with pytest.raises(RuntimeError, match="offline downgrade cannot verify"):
        command.downgrade(
            config,
            "0006_capture_candidates:0005_inspiration_falsifiers",
            sql=True,
        )


def test_offline_evidence_hash_upgrade_fails_closed(
    repository_root: Path,
) -> None:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", "sqlite:///offline-hashes.sqlite3")
    with pytest.raises(RuntimeError, match="offline upgrade cannot recompute"):
        command.upgrade(
            config,
            "0006_capture_candidates:0007_capture_evidence_hashes",
            sql=True,
        )


def _seed_authenticated_head_evidence(engine: Engine) -> str:
    excerpt = "trusted dynamic-type evidence"
    source_hash = sha256_hex(excerpt.encode())
    manifest = _manifest_document()
    steps = manifest["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["observation_hash"] = source_hash
    manifest_bytes = canonical_json_bytes(manifest)
    _seed_agent(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO trajectory_bundles "
                "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                "adapter_version, sanitization_profile, manifest, "
                "manifest_hash, source_started_at, source_completed_at, "
                "captured_at) VALUES (:bundle_id, :owner_id, 'trajectory-1', "
                "'generic_jsonl', 1, 'trusted-v1', :manifest, "
                ":manifest_hash, :now, :now, :now)"
            ),
            {
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "manifest": manifest_bytes,
                "manifest_hash": sha256_hex(manifest_bytes),
                "now": NOW_TEXT,
            },
        )
        connection.execute(
            text(
                "INSERT INTO trajectory_evidence "
                "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                "ordinal, excerpt, source_hash, excerpt_hash) VALUES "
                "(:evidence_id, :bundle_id, :owner_id, 'step-1', "
                "'observation', 1, :excerpt, :source_hash, :excerpt_hash)"
            ),
            {
                "evidence_id": str(EVIDENCE_ID),
                "bundle_id": str(BUNDLE_ID),
                "owner_id": str(OWNER_ID),
                "excerpt": excerpt,
                "source_hash": source_hash,
                "excerpt_hash": source_hash,
            },
        )
    return source_hash


def _capture_evidence_schema_snapshot(engine: Engine) -> dict[str, object]:
    inspector = inspect(engine)
    with engine.connect() as connection:
        version = connection.scalar(text("SELECT version_num FROM alembic_version"))
        table_sql = connection.scalar(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'trajectory_evidence'"
            )
        )
        triggers = tuple(
            tuple(row)
            for row in connection.execute(
                text(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND tbl_name = 'trajectory_evidence' ORDER BY name"
                )
            )
        )
        rows = tuple(
            tuple(row)
            for row in connection.execute(
                text(
                    "SELECT evidence_id, typeof(evidence_id), bundle_id, "
                    "typeof(bundle_id), owner_agent_id, typeof(owner_agent_id), "
                    "step_id, typeof(step_id), field, typeof(field), ordinal, "
                    "typeof(ordinal), excerpt, typeof(excerpt), excerpt_hash, "
                    "typeof(excerpt_hash) FROM trajectory_evidence "
                    "ORDER BY evidence_id"
                )
            )
        )
        bundle_rows = tuple(
            tuple(row)
            for row in connection.execute(
                text(
                    "SELECT bundle_id, typeof(bundle_id), owner_agent_id, "
                    "typeof(owner_agent_id), trajectory_id, "
                    "typeof(trajectory_id), adapter_kind, "
                    "typeof(adapter_kind), adapter_version, "
                    "typeof(adapter_version), sanitization_profile, "
                    "typeof(sanitization_profile), manifest, typeof(manifest), "
                    "manifest_hash, typeof(manifest_hash), source_started_at, "
                    "typeof(source_started_at), source_completed_at, "
                    "typeof(source_completed_at) FROM trajectory_bundles "
                    "ORDER BY bundle_id"
                )
            )
        )
    columns = tuple(
        (
            str(column["name"]),
            str(column["type"]),
            bool(column["nullable"]),
            column["default"],
            int(column["primary_key"]),
        )
        for column in inspector.get_columns("trajectory_evidence")
    )
    checks = tuple(
        sorted(
            (
                str(constraint["name"]),
                " ".join(str(constraint["sqltext"]).split()),
            )
            for constraint in inspector.get_check_constraints(
                "trajectory_evidence"
            )
        )
    )
    foreign_keys = tuple(
        sorted(
            (
                tuple(constraint["constrained_columns"]),
                str(constraint["referred_table"]),
                tuple(constraint["referred_columns"]),
            )
            for constraint in inspector.get_foreign_keys("trajectory_evidence")
        )
    )
    return {
        "version": version,
        "table_sql": table_sql,
        "columns": columns,
        "indexes": _index_keys(engine, "trajectory_evidence"),
        "checks": checks,
        "foreign_keys": foreign_keys,
        "triggers": triggers,
        "rows": rows,
        "bundle_rows": bundle_rows,
    }


def _replace_immutable_trigger(
    engine: Engine,
    *,
    table_name: str,
    statements: tuple[str, ...],
) -> None:
    trigger_name = f"{table_name}_reject_update"
    with engine.begin() as connection:
        trigger_sql = connection.scalar(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
                "AND name = :trigger_name"
            ),
            {"trigger_name": trigger_name},
        )
        assert isinstance(trigger_sql, str)
        connection.execute(text(f"DROP TRIGGER {trigger_name}"))
        connection.execute(text("PRAGMA ignore_check_constraints = ON"))
        for statement in statements:
            connection.execute(text(statement))
        connection.execute(text("PRAGMA ignore_check_constraints = OFF"))
        connection.execute(text(trigger_sql))


def test_evidence_hash_upgrade_rejects_blob_identity_before_any_schema_change(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-blob-evidence-id.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "head")
    head_engine = create_engine(f"sqlite:///{database_path}")
    try:
        source_hash = _seed_authenticated_head_evidence(head_engine)
    finally:
        head_engine.dispose()
    command.downgrade(config, "0006_capture_candidates")

    corrupted = create_engine(f"sqlite:///{database_path}")
    try:
        _replace_immutable_trigger(
            corrupted,
            table_name="trajectory_evidence",
            statements=(
                "UPDATE trajectory_evidence SET evidence_id = x'010203'",
            ),
        )
        before = _capture_evidence_schema_snapshot(corrupted)
    finally:
        corrupted.dispose()

    with pytest.raises(RuntimeError, match="cannot be migrated safely"):
        command.upgrade(config, "head")

    unchanged = create_engine(f"sqlite:///{database_path}")
    try:
        assert _capture_evidence_schema_snapshot(unchanged) == before
        row = before["rows"]
        assert isinstance(row, tuple)
        assert row[0][0:2] == (b"\x01\x02\x03", "blob")
        with (
            pytest.raises(IntegrityError, match="immutable"),
            unchanged.begin() as connection,
        ):
            connection.execute(
                text("UPDATE trajectory_evidence SET excerpt = 'changed'")
            )
        _replace_immutable_trigger(
            unchanged,
            table_name="trajectory_evidence",
            statements=(
                "UPDATE trajectory_evidence SET evidence_id = "
                f"'{EVIDENCE_ID}' WHERE evidence_id = x'010203'",
            ),
        )
    finally:
        unchanged.dispose()

    command.upgrade(config, "head")
    retried = create_engine(f"sqlite:///{database_path}")
    try:
        with retried.connect() as connection:
            migrated = connection.execute(
                text(
                    "SELECT source_hash, excerpt_hash FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            ).one()
        assert migrated == (source_hash, source_hash)
    finally:
        retried.dispose()


@pytest.mark.parametrize(
    "corruption",
    (
        "evidence_id_number",
        "bundle_id_blob",
        "bundle_id_number",
        "owner_id_blob",
        "owner_id_number",
        "step_id_blob",
        "field_number",
        "ordinal_blob",
        "trajectory_id_blob",
        "manifest_hash_number",
    ),
)
def test_evidence_hash_upgrade_rejects_non_text_identity_types_before_ddl(
    repository_root: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    database_path = tmp_path / f"legacy-{corruption}.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "head")
    head_engine = create_engine(f"sqlite:///{database_path}")
    try:
        _seed_authenticated_head_evidence(head_engine)
    finally:
        head_engine.dispose()
    command.downgrade(config, "0006_capture_candidates")

    corrupted = create_engine(f"sqlite:///{database_path}")
    try:
        if corruption == "evidence_id_number":
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=("UPDATE trajectory_evidence SET evidence_id = 17",),
            )
        elif corruption.startswith("bundle_id"):
            value = "x'010203'" if corruption.endswith("blob") else "17"
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=(
                    f"UPDATE trajectory_evidence SET bundle_id = {value}",
                ),
            )
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_bundles",
                statements=(
                    f"UPDATE trajectory_bundles SET bundle_id = {value}",
                ),
            )
        elif corruption.startswith("owner_id"):
            value = "x'010203'" if corruption.endswith("blob") else "17"
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=(
                    f"UPDATE trajectory_evidence SET owner_agent_id = {value}",
                ),
            )
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_bundles",
                statements=(
                    f"UPDATE trajectory_bundles SET owner_agent_id = {value}",
                ),
            )
        elif corruption == "step_id_blob":
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=(
                    "UPDATE trajectory_evidence SET step_id = x'010203'",
                ),
            )
        elif corruption == "field_number":
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=("UPDATE trajectory_evidence SET field = 17",),
            )
        elif corruption == "ordinal_blob":
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_evidence",
                statements=("UPDATE trajectory_evidence SET ordinal = x'01'",),
            )
        elif corruption == "trajectory_id_blob":
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_bundles",
                statements=(
                    "UPDATE trajectory_bundles SET trajectory_id = x'010203'",
                ),
            )
        else:
            assert corruption == "manifest_hash_number"
            _replace_immutable_trigger(
                corrupted,
                table_name="trajectory_bundles",
                statements=("UPDATE trajectory_bundles SET manifest_hash = 17",),
            )
        before = _capture_evidence_schema_snapshot(corrupted)
    finally:
        corrupted.dispose()

    with pytest.raises(RuntimeError, match="cannot be migrated safely"):
        command.upgrade(config, "head")

    unchanged = create_engine(f"sqlite:///{database_path}")
    try:
        assert _capture_evidence_schema_snapshot(unchanged) == before
    finally:
        unchanged.dispose()


@pytest.mark.parametrize("excerpt", ("trusted short evidence", "x" * 512))
def test_evidence_hash_migration_splits_and_restores_authenticated_legacy_digest(
    repository_root: Path,
    tmp_path: Path,
    excerpt: str,
) -> None:
    database_path = tmp_path / "legacy-evidence-hash.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "0006_capture_candidates")
    engine = create_engine(f"sqlite:///{database_path}")
    source_hash = sha256_hex(excerpt.encode())
    manifest = _manifest_document()
    steps = manifest["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["observation_hash"] = source_hash
    manifest_bytes = canonical_json_bytes(manifest)
    try:
        _seed_agent(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO trajectory_bundles "
                    "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                    "adapter_version, sanitization_profile, manifest, "
                    "manifest_hash, source_started_at, source_completed_at, "
                    "captured_at) VALUES (:bundle_id, :owner_id, 'trajectory-1', "
                    "'generic_jsonl', 1, 'trusted-v1', :manifest, "
                    ":manifest_hash, :now, :now, :now)"
                ),
                {
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "manifest": manifest_bytes,
                    "manifest_hash": sha256_hex(manifest_bytes),
                    "now": NOW_TEXT,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO trajectory_evidence "
                    "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                    "ordinal, excerpt, excerpt_hash) VALUES (:evidence_id, "
                    ":bundle_id, :owner_id, 'step-1', 'observation', 1, "
                    ":excerpt, :legacy_hash)"
                ),
                {
                    "evidence_id": str(EVIDENCE_ID),
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "excerpt": excerpt,
                    "legacy_hash": source_hash,
                },
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    upgraded = create_engine(f"sqlite:///{database_path}")
    try:
        with upgraded.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT source_hash, excerpt_hash FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            ).mappings().one()
        assert row["source_hash"] == source_hash
        assert row["excerpt_hash"] == source_hash
    finally:
        upgraded.dispose()

    command.downgrade(config, "0006_capture_candidates")
    downgraded = create_engine(f"sqlite:///{database_path}")
    try:
        assert "source_hash" not in set(
            _column_names(downgraded, "trajectory_evidence")
        )
        with downgraded.connect() as connection:
            restored = connection.scalar(
                text(
                    "SELECT excerpt_hash FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            )
        assert restored == source_hash
    finally:
        downgraded.dispose()


def test_evidence_hash_upgrade_refuses_unauthenticated_truncated_excerpt_before_ddl(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-truncated-evidence.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "0006_capture_candidates")
    engine = create_engine(f"sqlite:///{database_path}")
    full_source = "x" * 513
    retained_excerpt = full_source[:512]
    source_hash = sha256_hex(full_source.encode())
    manifest = _manifest_document()
    steps = manifest["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["observation_hash"] = source_hash
    manifest_bytes = canonical_json_bytes(manifest)
    try:
        _seed_agent(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO trajectory_bundles "
                    "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                    "adapter_version, sanitization_profile, manifest, "
                    "manifest_hash, source_started_at, source_completed_at, "
                    "captured_at) VALUES (:bundle_id, :owner_id, 'trajectory-1', "
                    "'generic_jsonl', 1, 'trusted-v1', :manifest, "
                    ":manifest_hash, :now, :now, :now)"
                ),
                {
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "manifest": manifest_bytes,
                    "manifest_hash": sha256_hex(manifest_bytes),
                    "now": NOW_TEXT,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO trajectory_evidence "
                    "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                    "ordinal, excerpt, excerpt_hash) VALUES (:evidence_id, "
                    ":bundle_id, :owner_id, 'step-1', 'observation', 1, "
                    ":excerpt, :legacy_hash)"
                ),
                {
                    "evidence_id": str(EVIDENCE_ID),
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "excerpt": retained_excerpt,
                    "legacy_hash": source_hash,
                },
            )
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="recapture or trusted backfill"):
        command.upgrade(config, "head")

    unchanged = create_engine(f"sqlite:///{database_path}")
    try:
        assert "source_hash" not in set(
            _column_names(unchanged, "trajectory_evidence")
        )
        with unchanged.connect() as connection:
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == "0006_capture_candidates"
            row = connection.execute(
                text(
                    "SELECT excerpt, excerpt_hash FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            ).one()
            triggers = {
                str(item[0])
                for item in connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                        "AND tbl_name = 'trajectory_evidence'"
                    )
                )
            }
        assert row == (retained_excerpt, source_hash)
        assert triggers == {
            "trajectory_evidence_reject_update",
            "trajectory_evidence_reject_delete",
            "trajectory_evidence_reject_conflicting_insert",
        }
        with (
            pytest.raises(IntegrityError, match="immutable"),
            unchanged.begin() as connection,
        ):
            connection.execute(
                text("UPDATE trajectory_evidence SET excerpt = 'changed'")
            )
    finally:
        unchanged.dispose()


def test_evidence_hash_upgrade_rejects_downgraded_tampering_without_washing_it(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "downgraded-tampered-evidence.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    trusted_excerpt = "trusted evidence"
    source_hash = sha256_hex(trusted_excerpt.encode())
    manifest = _manifest_document()
    steps = manifest["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["observation_hash"] = source_hash
    manifest_bytes = canonical_json_bytes(manifest)
    try:
        _seed_agent(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO trajectory_bundles "
                    "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                    "adapter_version, sanitization_profile, manifest, "
                    "manifest_hash, source_started_at, source_completed_at, "
                    "captured_at) VALUES (:bundle_id, :owner_id, 'trajectory-1', "
                    "'generic_jsonl', 1, 'trusted-v1', :manifest, "
                    ":manifest_hash, :now, :now, :now)"
                ),
                {
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "manifest": manifest_bytes,
                    "manifest_hash": sha256_hex(manifest_bytes),
                    "now": NOW_TEXT,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO trajectory_evidence "
                    "(evidence_id, bundle_id, owner_agent_id, step_id, field, "
                    "ordinal, excerpt, source_hash, excerpt_hash) VALUES "
                    "(:evidence_id, :bundle_id, :owner_id, 'step-1', "
                    "'observation', 1, :excerpt, :source_hash, :excerpt_hash)"
                ),
                {
                    "evidence_id": str(EVIDENCE_ID),
                    "bundle_id": str(BUNDLE_ID),
                    "owner_id": str(OWNER_ID),
                    "excerpt": trusted_excerpt,
                    "source_hash": source_hash,
                    "excerpt_hash": source_hash,
                },
            )
    finally:
        engine.dispose()

    command.downgrade(config, "0006_capture_candidates")
    downgraded = create_engine(f"sqlite:///{database_path}")
    try:
        with downgraded.begin() as connection:
            connection.execute(
                text("DROP TRIGGER trajectory_evidence_reject_update")
            )
            connection.execute(
                text(
                    "UPDATE trajectory_evidence SET excerpt = 'tampered evidence' "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            )
            connection.execute(
                text(
                    "CREATE TRIGGER trajectory_evidence_reject_update "
                    "BEFORE UPDATE ON trajectory_evidence BEGIN SELECT "
                    "RAISE(ABORT, 'trajectory_evidence rows are immutable'); END"
                )
            )
    finally:
        downgraded.dispose()

    with pytest.raises(RuntimeError, match="recapture or trusted backfill"):
        command.upgrade(config, "head")

    unchanged = create_engine(f"sqlite:///{database_path}")
    try:
        assert "source_hash" not in set(
            _column_names(unchanged, "trajectory_evidence")
        )
        with unchanged.connect() as connection:
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == "0006_capture_candidates"
            assert connection.scalar(
                text(
                    "SELECT excerpt FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_ID)},
            ) == "tampered evidence"
    finally:
        unchanged.dispose()


@pytest.mark.parametrize(
    "retained_kind",
    ("source", "event", "trajectory_event", "experience"),
)
def test_capture_downgrade_refuses_retained_authority_before_ddl(
    repository_root: Path,
    tmp_path: Path,
    retained_kind: str,
) -> None:
    database_path = tmp_path / f"retained-capture-{retained_kind}.sqlite3"
    config = _config(repository_root, database_path)
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        _seed_agent(engine)
        if retained_kind == "source":
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO trajectory_bundles "
                        "(bundle_id, owner_agent_id, trajectory_id, adapter_kind, "
                        "adapter_version, sanitization_profile, manifest, "
                        "manifest_hash, source_started_at, source_completed_at, "
                        "captured_at) VALUES (:bundle_id, :owner_id, "
                        "'trajectory-1', 'generic_jsonl', 1, 'trusted-v1', "
                        ":manifest, :manifest_hash, :now, :now, :now)"
                    ),
                    {
                        "bundle_id": str(BUNDLE_ID),
                        "owner_id": str(OWNER_ID),
                        "manifest": canonical_json_bytes(_manifest_document()),
                        "manifest_hash": HASH_A,
                        "now": NOW_TEXT,
                    },
                )
        elif retained_kind == "event":
            _insert_candidate_event(engine)
        elif retained_kind == "trajectory_event":
            _insert_trajectory_event(engine)
        else:
            _insert_experience(engine, origin="adopted_candidate")
    finally:
        engine.dispose()

    with pytest.raises(RuntimeError, match="candidate source or ledger data"):
        command.downgrade(config, "0005_inspiration_falsifiers")

    preserved = create_engine(f"sqlite:///{database_path}")
    try:
        assert set(CAPTURE_COLUMNS) <= set(inspect(preserved).get_table_names())
        with preserved.connect() as connection:
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == "0006_capture_candidates"
    finally:
        preserved.dispose()
