from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError

from experience_hub import canonical_json_bytes
from experience_hub.domain import StructuredReason

NOW = datetime(2026, 7, 18, 8, tzinfo=UTC)


@pytest.fixture
def seeded_engine(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[tuple[Engine, dict[str, str | int | bytes]]]:
    database_path = tmp_path / "sharing-immutability.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path}")
    seed = _seed_sources(engine)
    try:
        yield engine, seed
    finally:
        engine.dispose()


def test_sharing_sources_reject_update_delete_and_replace(
    seeded_engine: tuple[Engine, dict[str, str | int | bytes]],
) -> None:
    engine, seed = seeded_engine
    cases = (
        (
            "UPDATE topics SET name = 'Changed' WHERE topic_id = :id",
            seed["topic_id"],
            "topics rows are immutable",
        ),
        (
            "DELETE FROM topics WHERE topic_id = :id",
            seed["topic_id"],
            "topics rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO topics SELECT * FROM topics "
            "WHERE topic_id = :id",
            seed["topic_id"],
            "topics identity already exists",
        ),
        (
            "UPDATE subscriptions SET created_at = :later "
            "WHERE subscription_id = :id",
            seed["subscription_id"],
            "subscriptions rows are immutable",
        ),
        (
            "DELETE FROM subscriptions WHERE subscription_id = :id",
            seed["subscription_id"],
            "subscriptions rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO subscriptions SELECT * FROM subscriptions "
            "WHERE subscription_id = :id",
            seed["subscription_id"],
            "subscriptions identity already exists",
        ),
        (
            "UPDATE experience_capsules SET body = 'Changed' "
            "WHERE capsule_id = :id",
            seed["capsule_id"],
            "experience_capsules rows are immutable",
        ),
        (
            "DELETE FROM experience_capsules WHERE capsule_id = :id",
            seed["capsule_id"],
            "experience_capsules rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO experience_capsules "
            "SELECT * FROM experience_capsules WHERE capsule_id = :id",
            seed["capsule_id"],
            "experience_capsules identity already exists",
        ),
        (
            "UPDATE adoption_records SET captured_trust = 0.6 "
            "WHERE adoption_id = :id",
            seed["adoption_id"],
            "adoption_records rows are immutable",
        ),
        (
            "DELETE FROM adoption_records WHERE adoption_id = :id",
            seed["adoption_id"],
            "adoption_records rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO adoption_records "
            "SELECT * FROM adoption_records WHERE adoption_id = :id",
            seed["adoption_id"],
            "adoption_records identity already exists",
        ),
        (
            "UPDATE capsule_feedback SET verdict = 'harmful' "
            "WHERE feedback_id = :id",
            seed["feedback_id"],
            "capsule_feedback rows are immutable",
        ),
        (
            "DELETE FROM capsule_feedback WHERE feedback_id = :id",
            seed["feedback_id"],
            "capsule_feedback rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO capsule_feedback "
            "SELECT * FROM capsule_feedback WHERE feedback_id = :id",
            seed["feedback_id"],
            "capsule_feedback identity already exists",
        ),
    )
    for statement, identity, message in cases:
        with (
            pytest.raises(IntegrityError, match=message),
            engine.begin() as connection,
        ):
            connection.execute(
                text(statement),
                {
                    "id": identity,
                    "later": (NOW + timedelta(hours=1))
                    .isoformat(timespec="microseconds")
                    .replace(
                        "+00:00",
                        "Z",
                    ),
                },
            )


def test_reducer_owned_projection_rows_remain_mutable(
    seeded_engine: tuple[Engine, dict[str, str | int | bytes]],
) -> None:
    engine, seed = seeded_engine
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE capsule_state SET status = 'retracted' "
                "WHERE capsule_id = :capsule"
            ),
            {"capsule": seed["capsule_id"]},
        )
        connection.execute(
            text(
                "UPDATE inbox_items SET state = 'rejected' "
                "WHERE item_id = :item"
            ),
            {"item": seed["item_id"]},
        )
        connection.execute(
            text(
                "UPDATE agent_reputation SET useful_count = 1, alpha = 3, "
                "beta = 2 WHERE subject_agent_id = :publisher "
                "AND observer_agent_id = :recipient"
            ),
            {
                "publisher": seed["publisher_id"],
                "recipient": seed["recipient_id"],
            },
        )


def _seed_sources(engine: Engine) -> dict[str, str | int | bytes]:
    publisher_id = str(uuid4())
    recipient_id = str(uuid4())
    experience_id = str(uuid4())
    version_id = str(uuid4())
    receipt_id = str(uuid4())
    topic_id = str(uuid4())
    subscription_id = str(uuid4())
    capsule_id = str(uuid4())
    adoption_id = str(uuid4())
    feedback_id = str(uuid4())
    item_id = str(uuid4())
    now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    empty = b"[]"
    reason = canonical_json_bytes(
        StructuredReason.from_user_text("Useful").model_dump(mode="json")
    )
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
                "(:receipt, 'system:local', 'sharing.seed', 'seed', :hash, "
                "'in_progress', :now)"
            ),
            {
                "receipt": receipt_id,
                "hash": "f" * 64,
                "now": now,
            },
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
                "(version_id, experience_id, version_number, summary, mechanism, "
                "tags, applicability, evidence, falsifiers, content_hash, "
                "supersedes_version_id, created_at) VALUES "
                "(:version, :experience, 1, 'Lease handoff', 'fencing', "
                ":empty, :empty, :empty, :empty, :hash, NULL, :now)"
            ),
            {
                "version": version_id,
                "experience": experience_id,
                "empty": empty,
                "hash": "a" * 64,
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
                "'procedural', 'Preserve the token.', 'Lease handoff', "
                "'fencing', :empty, :empty, :empty, :empty, 0.8, :empty, "
                ":root, :content, :now, :expires, 0, :capsule_hash)"
            ),
            {
                "capsule": capsule_id,
                "topic": topic_id,
                "experience": experience_id,
                "version": version_id,
                "publisher": publisher_id,
                "empty": empty,
                "root": "b" * 64,
                "content": "a" * 64,
                "now": now,
                "expires": (
                    NOW + timedelta(days=7)
                ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "capsule_hash": "c" * 64,
            },
        )
        connection.execute(
            text(
                "INSERT INTO adoption_records "
                "(adoption_id, adopter_agent_id, capsule_id, "
                "resulting_experience_id, captured_trust, provenance_chain, "
                "root_fingerprint, corroboration_applied, adopted_at) VALUES "
                "(:adoption, :recipient, :capsule, :experience, 0.5, :chain, "
                ":root, 0, :now)"
            ),
            {
                "adoption": adoption_id,
                "recipient": recipient_id,
                "capsule": capsule_id,
                "experience": experience_id,
                "chain": canonical_json_bytes(
                    [
                        {
                            "capsule_id": capsule_id,
                            "publisher_agent_id": publisher_id,
                        }
                    ]
                ),
                "root": "b" * 64,
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
                "reason": reason,
                "empty": empty,
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO capsule_state "
                "(capsule_id, status, projection_event_id) "
                "VALUES (:capsule, 'active', :event)"
            ),
            {
                "capsule": capsule_id,
                "now": now,
                "event": event_id,
            },
        )
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
                "INSERT INTO agent_reputation "
                "(subject_agent_id, observer_agent_id, useful_count, "
                "refuted_count, harmful_count, alpha, beta, "
                "projection_event_id) VALUES "
                "(:publisher, :recipient, 0, 0, 0, 2, 2, :event)"
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
        "topic_id": topic_id,
        "subscription_id": subscription_id,
        "capsule_id": capsule_id,
        "adoption_id": adoption_id,
        "feedback_id": feedback_id,
        "item_id": item_id,
        "event_id": event_id,
        "empty": empty,
    }
