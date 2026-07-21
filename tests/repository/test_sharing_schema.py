from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from experience_hub import canonical_json_bytes
from experience_hub.domain import StructuredReason, TypedEvidence
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.models import ExperienceKind, Temperature
from experience_hub.sharing.models import (
    AdoptionResult,
    Capsule,
    CapsuleStatus,
    EffectiveAvailability,
    FeedbackVerdict,
    InboxItem,
    InboxState,
    ProvenanceHop,
    Reputation,
)
from experience_hub.storage.tables import Base

NOW = datetime(2026, 7, 18, 8, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

SHARING_COLUMNS = {
    "topics": (
        "topic_id",
        "owner_agent_id",
        "name",
        "description",
        "created_at",
    ),
    "subscriptions": (
        "subscription_id",
        "subscriber_agent_id",
        "topic_id",
        "creation_event_id",
        "created_at",
    ),
    "experience_capsules": (
        "capsule_id",
        "transport_schema_version",
        "topic_id",
        "source_experience_id",
        "source_version_id",
        "publisher_agent_id",
        "kind",
        "body",
        "summary",
        "mechanism",
        "tags",
        "applicability",
        "evidence",
        "falsifiers",
        "publisher_confidence",
        "provenance_chain",
        "root_fingerprint",
        "source_content_hash",
        "created_at",
        "expires_at",
        "hop_count",
        "capsule_hash",
    ),
    "adoption_records": (
        "adoption_id",
        "adopter_agent_id",
        "capsule_id",
        "resulting_experience_id",
        "captured_trust",
        "provenance_chain",
        "root_fingerprint",
        "corroboration_applied",
        "adopted_at",
    ),
    "capsule_feedback": (
        "feedback_id",
        "observer_agent_id",
        "capsule_id",
        "revision",
        "verdict",
        "reason",
        "evidence",
        "created_at",
    ),
    "capsule_state": (
        "capsule_id",
        "status",
        "projection_event_id",
    ),
    "inbox_items": (
        "item_id",
        "recipient_agent_id",
        "capsule_id",
        "state",
        "projection_event_id",
    ),
    "agent_reputation": (
        "subject_agent_id",
        "observer_agent_id",
        "useful_count",
        "refuted_count",
        "harmful_count",
        "alpha",
        "beta",
        "projection_event_id",
    ),
}


@pytest.fixture
def migrated_engine(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Engine]:
    database_path = tmp_path / "sharing.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        yield engine
    finally:
        engine.dispose()


def _index_keys(engine: Engine, table_name: str) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for index in inspect(engine).get_indexes(table_name):
        reflected_where = index.get("dialect_options", {}).get("sqlite_where")
        keys.add(
            (
                index["name"],
                tuple(index["column_names"]),
                bool(index["unique"]),
                str(reflected_where) if reflected_where is not None else None,
            )
        )
    return keys


def _metadata_index_keys(table_name: str) -> set[tuple[object, ...]]:
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            bool(index.unique),
            (
                str(index.dialect_options["sqlite"].get("where"))
                if index.dialect_options["sqlite"].get("where") is not None
                else None
            ),
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


def test_sharing_migration_and_metadata_declare_exact_tables_and_columns(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    assert set(inspector.get_table_names()) >= set(SHARING_COLUMNS)
    assert set(Base.metadata.tables) >= set(SHARING_COLUMNS)

    for table_name, columns in SHARING_COLUMNS.items():
        migrated = tuple(
            str(column["name"]) for column in inspector.get_columns(table_name)
        )
        metadata = tuple(Base.metadata.tables[table_name].columns.keys())
        assert migrated == columns
        assert metadata == columns


def test_sharing_primary_and_foreign_keys_match_ownership(
    migrated_engine: Engine,
) -> None:
    inspector = inspect(migrated_engine)
    expected_primary_keys = {
        "topics": ("topic_id",),
        "subscriptions": ("subscription_id",),
        "experience_capsules": ("capsule_id",),
        "adoption_records": ("adoption_id",),
        "capsule_feedback": ("feedback_id",),
        "capsule_state": ("capsule_id",),
        "inbox_items": ("item_id",),
        "agent_reputation": ("subject_agent_id", "observer_agent_id"),
    }
    for table_name, expected in expected_primary_keys.items():
        assert (
            tuple(
                inspector.get_pk_constraint(table_name)["constrained_columns"]
            )
            == expected
        )

    expected_foreign_keys = {
        "topics": {
            (("owner_agent_id",), "agents", ("agent_id",)),
        },
        "subscriptions": {
            (("subscriber_agent_id",), "agents", ("agent_id",)),
            (("topic_id",), "topics", ("topic_id",)),
            (("creation_event_id",), "domain_events", ("event_id",)),
        },
        "experience_capsules": {
            (("topic_id",), "topics", ("topic_id",)),
            (("source_experience_id",), "experiences", ("experience_id",)),
            (("source_version_id",), "experience_versions", ("version_id",)),
            (("publisher_agent_id",), "agents", ("agent_id",)),
        },
        "adoption_records": {
            (("adopter_agent_id",), "agents", ("agent_id",)),
            (("capsule_id",), "experience_capsules", ("capsule_id",)),
            (
                ("resulting_experience_id",),
                "experiences",
                ("experience_id",),
            ),
        },
        "capsule_feedback": {
            (("observer_agent_id",), "agents", ("agent_id",)),
            (("capsule_id",), "experience_capsules", ("capsule_id",)),
        },
        "capsule_state": {
            (("capsule_id",), "experience_capsules", ("capsule_id",)),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
        "inbox_items": {
            (("recipient_agent_id",), "agents", ("agent_id",)),
            (("capsule_id",), "experience_capsules", ("capsule_id",)),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
        "agent_reputation": {
            (("subject_agent_id",), "agents", ("agent_id",)),
            (("observer_agent_id",), "agents", ("agent_id",)),
            (("projection_event_id",), "domain_events", ("event_id",)),
        },
    }
    for table_name, expected in expected_foreign_keys.items():
        assert _foreign_key_targets(migrated_engine, table_name) == expected


def test_sharing_indexes_match_metadata_and_locked_uniqueness(
    migrated_engine: Engine,
) -> None:
    for table_name in SHARING_COLUMNS:
        assert _index_keys(migrated_engine, table_name) == _metadata_index_keys(
            table_name
        )

    expected_indexes = {
        "topics": {
            ("ux_topics_name", ("name",), True),
        },
        "subscriptions": {
            (
                "ux_subscriptions_subscriber_topic",
                ("subscriber_agent_id", "topic_id"),
                True,
            ),
        },
        "inbox_items": {
            (
                "ux_inbox_items_recipient_capsule",
                ("recipient_agent_id", "capsule_id"),
                True,
            ),
        },
        "adoption_records": {
            (
                "ux_adoption_records_adopter_capsule",
                ("adopter_agent_id", "capsule_id"),
                True,
            ),
        },
        "capsule_feedback": {
            (
                "ux_capsule_feedback_observer_capsule_revision",
                ("observer_agent_id", "capsule_id", "revision"),
                True,
            ),
        },
    }
    for table_name, expected in expected_indexes.items():
        actual = {
            (name, columns, unique)
            for name, columns, unique, _ in _index_keys(
                migrated_engine,
                table_name,
            )
        }
        assert expected <= actual

    partial = next(
        index
        for index in _index_keys(migrated_engine, "adoption_records")
        if index[0] == "ux_adoption_records_corroborated_root"
    )
    assert partial[:3] == (
        "ux_adoption_records_corroborated_root",
        ("resulting_experience_id", "root_fingerprint"),
        True,
    )
    assert "corroboration_applied = 1" in str(partial[3])


def test_every_sharing_table_has_named_checks(
    migrated_engine: Engine,
) -> None:
    expected = {
        "topics": {"ck_topics_name", "ck_topics_description"},
        "subscriptions": {"ck_subscriptions_creation_event"},
        "experience_capsules": {
            "ck_experience_capsules_transport_version",
            "ck_experience_capsules_content",
            "ck_experience_capsules_arrays",
            "ck_experience_capsules_confidence",
            "ck_experience_capsules_hashes",
            "ck_experience_capsules_expiry",
            "ck_experience_capsules_hop_count",
        },
        "adoption_records": {
            "ck_adoption_records_captured_trust",
            "ck_adoption_records_provenance",
            "ck_adoption_records_root_fingerprint",
            "ck_adoption_records_corroboration",
        },
        "capsule_feedback": {
            "ck_capsule_feedback_revision",
            "ck_capsule_feedback_verdict",
            "ck_capsule_feedback_payloads",
        },
        "capsule_state": {
            "ck_capsule_state_status",
            "ck_capsule_state_projection_event",
        },
        "inbox_items": {
            "ck_inbox_items_state",
            "ck_inbox_items_projection_event",
        },
        "agent_reputation": {
            "ck_agent_reputation_counts",
            "ck_agent_reputation_beta_prior",
            "ck_agent_reputation_alpha_prior",
            "ck_agent_reputation_trust",
            "ck_agent_reputation_projection_event",
        },
    }
    inspector = inspect(migrated_engine)
    for table_name, names in expected.items():
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
        assert names <= reflected.keys()
        assert reflected == metadata


def test_locked_unique_keys_and_partial_corroboration_key_are_enforced(
    migrated_engine: Engine,
) -> None:
    seed = _seed_sharing_graph(migrated_engine)

    duplicates = (
        (
            "INSERT INTO topics "
            "(topic_id, owner_agent_id, name, description, created_at) "
            "VALUES (:new_id, :publisher_id, 'Operations', NULL, :now)",
            {},
        ),
        (
            "INSERT INTO subscriptions "
            "(subscription_id, subscriber_agent_id, topic_id, "
            "creation_event_id, created_at) VALUES "
            "(:new_id, :recipient_id, :topic_id, :event_id, :now)",
            {},
        ),
        (
            "INSERT INTO inbox_items "
            "(item_id, recipient_agent_id, capsule_id, state, "
            "projection_event_id) VALUES "
            "(:new_id, :recipient_id, :capsule_id, 'pending', :event_id)",
            {},
        ),
        (
            "INSERT INTO adoption_records "
            "(adoption_id, adopter_agent_id, capsule_id, "
            "resulting_experience_id, captured_trust, provenance_chain, "
            "root_fingerprint, corroboration_applied, adopted_at) VALUES "
            "(:new_id, :recipient_id, :capsule_id, :experience_id, 0.5, "
            ":chain, :root, 0, :now)",
            {},
        ),
        (
            "INSERT INTO capsule_feedback "
            "(feedback_id, observer_agent_id, capsule_id, revision, verdict, "
            "reason, evidence, created_at) VALUES "
            "(:new_id, :recipient_id, :capsule_id, 1, 'useful', "
            ":reason, :empty, :now)",
            {},
        ),
        (
            "INSERT INTO agent_reputation "
            "(subject_agent_id, observer_agent_id, useful_count, "
            "refuted_count, harmful_count, alpha, beta, "
            "projection_event_id) VALUES "
            "(:publisher_id, :recipient_id, 0, 0, 0, 2, 2, :event_id)",
            {},
        ),
        (
            "INSERT INTO agent_reputation "
            "(subject_agent_id, observer_agent_id, useful_count, "
            "refuted_count, harmful_count, alpha, beta, "
            "projection_event_id) VALUES "
            "(:recipient_id, :recipient_id, 0, 0, 0, 2, 2, :event_id)",
            {},
        ),
    )
    parameters = {
        **seed,
        "new_id": str(uuid4()),
        "now": NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "chain": canonical_json_bytes(
            [
                {
                    "capsule_id": seed["capsule_id"],
                    "publisher_agent_id": seed["publisher_id"],
                }
            ]
        ),
        "root": HASH_B,
        "reason": canonical_json_bytes(
            StructuredReason.from_user_text("Useful").model_dump(mode="json")
        ),
        "empty": b"[]",
    }
    for statement, overrides in duplicates:
        with (
            pytest.raises(IntegrityError),
            migrated_engine.begin() as connection,
        ):
            connection.execute(
                text(statement),
                {**parameters, **overrides, "new_id": str(uuid4())},
            )

    for statement, overrides in (
        duplicates[0],
        duplicates[1],
        duplicates[3],
        duplicates[4],
    ):
        replacement = statement.replace(
            "INSERT INTO",
            "INSERT OR REPLACE INTO",
            1,
        )
        with (
            pytest.raises(IntegrityError),
            migrated_engine.begin() as connection,
        ):
            connection.execute(
                text(replacement),
                {**parameters, **overrides, "new_id": str(uuid4())},
            )

    capsule_ids = [
        _insert_capsule(
            migrated_engine,
            seed=seed,
            capsule_id=str(uuid4()),
            capsule_hash=character * 64,
        )
        for character in ("d", "e", "f")
    ]
    provenance_only = {
        **parameters,
        "root": HASH_C,
        "corroboration": 0,
    }
    corroborated_adoption_id = str(uuid4())
    adoption_sql = text(
        "INSERT INTO adoption_records "
        "(adoption_id, adopter_agent_id, capsule_id, "
        "resulting_experience_id, captured_trust, provenance_chain, "
        "root_fingerprint, corroboration_applied, adopted_at) VALUES "
        "(:new_id, :recipient_id, :target_capsule_id, :experience_id, 0.5, "
        ":chain, :root, :corroboration, :now)"
    )
    with migrated_engine.begin() as connection:
        for capsule_id in capsule_ids[:2]:
            connection.execute(
                adoption_sql,
                {
                    **provenance_only,
                    "new_id": str(uuid4()),
                    "target_capsule_id": capsule_id,
                },
            )
        connection.execute(
            adoption_sql,
            {
                **provenance_only,
                "new_id": corroborated_adoption_id,
                "target_capsule_id": capsule_ids[2],
                "corroboration": 1,
            },
        )

    fourth_capsule = _insert_capsule(
        migrated_engine,
        seed=seed,
        capsule_id=str(uuid4()),
        capsule_hash="1" * 64,
    )
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            adoption_sql,
            {
                **provenance_only,
                "new_id": str(uuid4()),
                "target_capsule_id": fourth_capsule,
                "corroboration": 1,
            },
        )

    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT OR REPLACE INTO adoption_records "
                "(adoption_id, adopter_agent_id, capsule_id, "
                "resulting_experience_id, captured_trust, provenance_chain, "
                "root_fingerprint, corroboration_applied, adopted_at) VALUES "
                "(:new_id, :recipient_id, :target_capsule_id, "
                ":experience_id, 0.5, :chain, :root, 1, :now)"
            ),
            {
                **provenance_only,
                "new_id": str(uuid4()),
                "target_capsule_id": fourth_capsule,
            },
        )
    with migrated_engine.connect() as connection:
        retained = connection.scalar(
            text(
                "SELECT adoption_id FROM adoption_records "
                "WHERE resulting_experience_id = :experience_id "
                "AND root_fingerprint = :root "
                "AND corroboration_applied = 1"
            ),
            provenance_only,
        )
    assert retained == corroborated_adoption_id


def test_capsule_body_database_check_uses_utf8_bytes_and_rejects_blanks(
    migrated_engine: Engine,
) -> None:
    seed = _seed_sharing_graph(migrated_engine)
    for body, capsule_hash in (
        ("   ", "2" * 64),
        ("知" * 21_846, "3" * 64),
    ):
        with pytest.raises(IntegrityError):
            _insert_capsule(
                migrated_engine,
                seed=seed,
                capsule_id=str(uuid4()),
                capsule_hash=capsule_hash,
                body=body,
            )


def test_capsule_provenance_database_check_requires_a_json_array(
    migrated_engine: Engine,
) -> None:
    seed = _seed_sharing_graph(migrated_engine)
    with pytest.raises(IntegrityError):
        _insert_capsule(
            migrated_engine,
            seed=seed,
            capsule_id=str(uuid4()),
            capsule_hash="4" * 64,
            provenance=canonical_json_bytes({}),
        )


def test_capsule_and_reputation_models_enforce_canonical_invariants() -> None:
    publisher_id = uuid4()
    capsule_id = uuid4()
    topic_id = uuid4()
    experience_id = uuid4()
    version_id = uuid4()
    hop = ProvenanceHop(
        capsule_id=uuid4(),
        publisher_agent_id=uuid4(),
    )
    capsule = Capsule(
        capsule_id=capsule_id,
        transport_schema_version=1,
        topic_id=topic_id,
        source_experience_id=experience_id,
        source_version_id=version_id,
        publisher_agent_id=publisher_id,
        kind=ExperienceKind.PROCEDURAL,
        body="Preserve the fencing token.",
        summary="Lease handoff",
        mechanism="single-writer fencing",
        tags=("ops", "ops"),
        applicability=("failover",),
        evidence=(
            TypedEvidence(type="log", id="case-1"),
            TypedEvidence(type="log", id="case-1"),
        ),
        falsifiers=("overlap observed",),
        publisher_confidence=0.8,
        provenance_chain=(hop,),
        root_fingerprint=HASH_A,
        source_content_hash=HASH_B,
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
        hop_count=1,
        capsule_hash=HASH_C,
        status=CapsuleStatus.ACTIVE,
        last_transition_at=NOW,
    )
    assert capsule.tags == ("ops",)
    assert capsule.evidence == (TypedEvidence(type="log", id="case-1"),)
    inbox_item = InboxItem(
        item_id=uuid4(),
        recipient_agent_id=uuid4(),
        capsule_id=capsule.capsule_id,
        capsule=capsule,
        state=InboxState.PENDING,
        effective_availability=EffectiveAvailability.ACTIVE,
    )
    assert inbox_item.capsule is capsule

    retracted = Capsule.model_validate(
        {
            **capsule.model_dump(mode="python"),
            "status": CapsuleStatus.RETRACTED,
        }
    )
    with pytest.raises(ValidationError):
        InboxItem(
            item_id=uuid4(),
            recipient_agent_id=uuid4(),
            capsule_id=retracted.capsule_id,
            capsule=retracted,
            state=InboxState.PENDING,
            effective_availability=EffectiveAvailability.EXPIRED,
        )

    experience = ExperienceRecord(
        experience_id=experience_id,
        owner_agent_id=uuid4(),
        current_version_id=version_id,
        current_content_hash=HASH_B,
        temperature=Temperature.HOT,
    )
    adoption = AdoptionResult(
        experience=experience,
        created=True,
        corroboration_applied=False,
    )
    assert adoption.experience is experience
    with pytest.raises(ValidationError):
        AdoptionResult(
            experience=ExperienceRecord(
                experience_id=experience_id,
                owner_agent_id=uuid4(),
                current_version_id=version_id,
                current_content_hash="not-a-content-hash",
                temperature=Temperature.HOT,
            ),
            created=False,
            corroboration_applied=False,
        )
    with pytest.raises(ValidationError):
        AdoptionResult(
            experience=ExperienceRecord(
                experience_id=experience_id,
                owner_agent_id=uuid4(),
                current_version_id=version_id,
                current_content_hash=cast(str, None),
                temperature=Temperature.HOT,
            ),
            created=False,
            corroboration_applied=False,
        )

    with pytest.raises(ValidationError):
        Capsule.model_validate(
            {
                **capsule.model_dump(mode="python"),
                "hop_count": 0,
            }
        )
    with pytest.raises(ValidationError):
        Capsule.model_validate(
            {
                **capsule.model_dump(mode="python"),
                "unknown": "rejected",
            }
        )
    with pytest.raises(ValidationError):
        Reputation(
            subject_agent_id=publisher_id,
            observer_agent_id=uuid4(),
            useful_count=1,
            refuted_count=0,
            harmful_count=0,
            alpha=2.0,
            beta=2.0,
            last_feedback_at=NOW,
        )
    reputation = Reputation(
        subject_agent_id=publisher_id,
        observer_agent_id=uuid4(),
        useful_count=1,
        refuted_count=0,
        harmful_count=0,
        alpha=3,
        beta=2,
        last_feedback_at=NOW,
    )
    assert reputation.trust == pytest.approx(0.6)
    with pytest.raises(ValidationError):
        Reputation(
            subject_agent_id=publisher_id,
            observer_agent_id=uuid4(),
            useful_count=0,
            refuted_count=0,
            harmful_count=0,
            alpha=2.0,
            beta=2,
            last_feedback_at=NOW,
        )
    with pytest.raises(ValidationError):
        Reputation(
            subject_agent_id=publisher_id,
            observer_agent_id=publisher_id,
            useful_count=0,
            refuted_count=0,
            harmful_count=0,
            alpha=2.0,
            beta=2.0,
            last_feedback_at=NOW,
        )

    assert InboxState.PENDING.value == "pending"
    assert FeedbackVerdict.HARMFUL.value == "harmful"


def test_sharing_value_import_does_not_load_retrieval_contracts() -> None:
    script = "\n".join(
        (
            "import sys",
            "import experience_hub.sharing.models",
            "assert 'experience_hub.retrieval.contracts' not in sys.modules",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_empty_sharing_migration_downgrades_cleanly_to_experience_schema(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sharing-empty-downgrade.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")

    command.downgrade(config, "0002_experiences")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        table_names = set(inspect(engine).get_table_names())
        assert set(SHARING_COLUMNS).isdisjoint(table_names)
        assert "experiences" in table_names
        with engine.connect() as connection:
            version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
        assert version == "0002_experiences"
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "retained_data",
    ("source", "capsule_ledger", "corroboration_ledger"),
)
def test_populated_or_ledger_only_sharing_downgrade_fails_before_ddl(
    repository_root: Path,
    tmp_path: Path,
    retained_data: str,
) -> None:
    database_path = tmp_path / f"sharing-{retained_data}-downgrade.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    agent_id = str(uuid4())
    receipt_id = str(uuid4())
    now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(
                text(
                    "INSERT INTO agents (agent_id, name, created_at) "
                    "VALUES (:agent, 'Downgrade Owner', :now)"
                ),
                {"agent": agent_id, "now": now},
            )
            if retained_data == "source":
                connection.execute(
                    text(
                        "INSERT INTO topics "
                        "(topic_id, owner_agent_id, name, description, created_at) "
                        "VALUES (:topic, :agent, 'Retained', NULL, :now)"
                    ),
                    {
                        "topic": str(uuid4()),
                        "agent": agent_id,
                        "now": now,
                    },
                )
            else:
                aggregate_type = (
                    "experience"
                    if retained_data == "corroboration_ledger"
                    else "capsule"
                )
                event_type = (
                    "experience.corroborated"
                    if retained_data == "corroboration_ledger"
                    else "capsule.published"
                )
                connection.execute(
                    text(
                        "INSERT INTO idempotency_records "
                        "(receipt_id, caller_scope, scope, idempotency_key, "
                        "request_hash, state, created_at) VALUES "
                        "(:receipt, 'system:local', 'sharing.seed', 'ledger', "
                        ":hash, 'in_progress', :now)"
                    ),
                    {
                        "receipt": receipt_id,
                        "hash": HASH_A,
                        "now": now,
                    },
                )
                connection.execute(
                    text(
                        "INSERT INTO domain_events "
                        "(aggregate_type, aggregate_id, sequence, event_type, "
                        "payload, actor_agent_id, causation_id, occurred_at) "
                        "VALUES (:aggregate_type, :aggregate_id, 1, :event_type, "
                        ":payload, :agent, :receipt, :now)"
                    ),
                    {
                        "aggregate_type": aggregate_type,
                        "aggregate_id": str(uuid4()),
                        "event_type": event_type,
                        "payload": b'{"schema_version":1}',
                        "agent": agent_id,
                        "receipt": receipt_id,
                        "now": now,
                    },
                )

        with pytest.raises(
            RuntimeError,
            match="Cannot downgrade while sharing source or ledger data exists",
        ):
            command.downgrade(config, "0002_experiences")

        assert set(SHARING_COLUMNS) <= set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            version = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
        assert version == "0003_sharing"
    finally:
        engine.dispose()


def _seed_sharing_graph(engine: Engine) -> dict[str, str | int]:
    publisher_id = str(uuid4())
    recipient_id = str(uuid4())
    experience_id = str(uuid4())
    version_id = str(uuid4())
    receipt_id = str(uuid4())
    topic_id = str(uuid4())
    subscription_id = str(uuid4())
    capsule_id = str(uuid4())
    item_id = str(uuid4())
    adoption_id = str(uuid4())
    feedback_id = str(uuid4())
    now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
        connection.execute(
            text(
                "INSERT INTO agents (agent_id, name, created_at) VALUES "
                "(:publisher, 'Publisher', :now), "
                "(:recipient, 'Recipient', :now)"
            ),
            {
                "publisher": publisher_id,
                "recipient": recipient_id,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO idempotency_records "
                "(receipt_id, caller_scope, scope, idempotency_key, "
                "request_hash, state, created_at) VALUES "
                "(:receipt, 'system:local', 'sharing.seed', 'seed', "
                ":hash, 'in_progress', :now)"
            ),
            {"receipt": receipt_id, "hash": HASH_C, "now": now},
        )
        event_id = connection.execute(
            text(
                "INSERT INTO domain_events "
                "(aggregate_type, aggregate_id, sequence, event_type, payload, "
                "actor_agent_id, causation_id, occurred_at) VALUES "
                "('subscription', :subscription, 1, 'subscription.created', "
                ":payload, :recipient, :receipt, :now) RETURNING event_id"
            ),
            {
                "subscription": subscription_id,
                "payload": b'{"schema_version":1}',
                "recipient": recipient_id,
                "receipt": receipt_id,
                "now": now,
            },
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO experiences "
                "(experience_id, owner_agent_id, kind, origin, created_at) "
                "VALUES (:experience, :publisher, 'procedural', 'local', :now)"
            ),
            {
                "experience": experience_id,
                "publisher": publisher_id,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO experience_versions "
                "(version_id, experience_id, version_number, summary, "
                "mechanism, tags, applicability, evidence, falsifiers, "
                "content_hash, supersedes_version_id, created_at) VALUES "
                "(:version, :experience, 1, 'Lease handoff', 'fencing', "
                ":empty, :empty, :empty, :empty, :hash, NULL, :now)"
            ),
            {
                "version": version_id,
                "experience": experience_id,
                "empty": b"[]",
                "hash": HASH_A,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO topics "
                "(topic_id, owner_agent_id, name, description, created_at) "
                "VALUES (:topic, :publisher, 'Operations', NULL, :now)"
            ),
            {"topic": topic_id, "publisher": publisher_id, "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO subscriptions "
                "(subscription_id, subscriber_agent_id, topic_id, "
                "creation_event_id, created_at) VALUES "
                "(:subscription, :recipient, :topic, :event, :now)"
            ),
            {
                "subscription": subscription_id,
                "recipient": recipient_id,
                "topic": topic_id,
                "event": event_id,
                "now": now,
            },
        )
    _insert_capsule(
        engine,
        seed={
            "publisher_id": publisher_id,
            "experience_id": experience_id,
            "version_id": version_id,
            "topic_id": topic_id,
        },
        capsule_id=capsule_id,
        capsule_hash=HASH_B,
    )
    chain = canonical_json_bytes(
        [
            {
                "capsule_id": capsule_id,
                "publisher_agent_id": publisher_id,
            }
        ]
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO inbox_items "
                "(item_id, recipient_agent_id, capsule_id, state, "
                "projection_event_id) VALUES "
                "(:item, :recipient, :capsule, 'pending', :event)"
            ),
            {
                "item": item_id,
                "recipient": recipient_id,
                "capsule": capsule_id,
                "now": now,
                "event": event_id,
            },
        )
        connection.execute(
            text(
                "INSERT INTO adoption_records "
                "(adoption_id, adopter_agent_id, capsule_id, "
                "resulting_experience_id, captured_trust, provenance_chain, "
                "root_fingerprint, corroboration_applied, adopted_at) VALUES "
                "(:adoption, :recipient, :capsule, :experience, 0.5, "
                ":chain, :root, 0, :now)"
            ),
            {
                "adoption": adoption_id,
                "recipient": recipient_id,
                "capsule": capsule_id,
                "experience": experience_id,
                "chain": chain,
                "root": HASH_B,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO capsule_feedback "
                "(feedback_id, observer_agent_id, capsule_id, revision, "
                "verdict, reason, evidence, created_at) VALUES "
                "(:feedback, :recipient, :capsule, 1, 'useful', "
                ":reason, :empty, :now)"
            ),
            {
                "feedback": feedback_id,
                "recipient": recipient_id,
                "capsule": capsule_id,
                "reason": canonical_json_bytes(
                    StructuredReason.from_user_text("Useful").model_dump(
                        mode="json"
                    )
                ),
                "empty": b"[]",
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO agent_reputation "
                "(subject_agent_id, observer_agent_id, useful_count, "
                "refuted_count, harmful_count, alpha, beta, "
                "projection_event_id) VALUES "
                "(:publisher, :recipient, 1, 0, 0, 3, 2, :event)"
            ),
            {
                "publisher": publisher_id,
                "recipient": recipient_id,
                "now": now,
                "event": event_id,
            },
        )
    return {
        "publisher_id": publisher_id,
        "recipient_id": recipient_id,
        "experience_id": experience_id,
        "version_id": version_id,
        "topic_id": topic_id,
        "subscription_id": subscription_id,
        "capsule_id": capsule_id,
        "item_id": item_id,
        "adoption_id": adoption_id,
        "feedback_id": feedback_id,
        "event_id": event_id,
    }


def _insert_capsule(
    engine: Engine,
    *,
    seed: dict[str, str | int],
    capsule_id: str,
    capsule_hash: str,
    body: str = "Preserve the token.",
    provenance: bytes = b"[]",
) -> str:
    now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO experience_capsules "
                "(capsule_id, transport_schema_version, topic_id, "
                "source_experience_id, source_version_id, publisher_agent_id, "
                "kind, body, summary, mechanism, tags, applicability, evidence, "
                "falsifiers, publisher_confidence, provenance_chain, "
                "root_fingerprint, source_content_hash, created_at, expires_at, "
                "hop_count, capsule_hash) VALUES "
                "(:capsule, 1, :topic, :experience, :version, :publisher, "
                "'procedural', :body, 'Lease handoff', "
                "'fencing', :empty, :empty, :empty, :empty, 0.8, :provenance, "
                ":root, :content_hash, :now, :expires, 0, :capsule_hash)"
            ),
            {
                "capsule": capsule_id,
                "topic": seed["topic_id"],
                "experience": seed["experience_id"],
                "version": seed["version_id"],
                "publisher": seed["publisher_id"],
                "body": body,
                "empty": b"[]",
                "provenance": provenance,
                "root": HASH_A,
                "content_hash": HASH_A,
                "now": now,
                "expires": (
                    NOW + timedelta(days=7)
                ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "capsule_hash": capsule_hash,
            },
        )
    return capsule_id
