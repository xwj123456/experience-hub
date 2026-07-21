"""Create immutable sharing sources and replayable sharing projections."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_sharing"
down_revision: str | None = "0002_experiences"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _sha256_check(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _create_immutable_triggers(
    table_name: str,
    *,
    conflict_when: str,
) -> None:
    op.execute(
        f"CREATE TRIGGER {table_name}_reject_update "
        f"BEFORE UPDATE ON {table_name} "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{table_name} rows are immutable'); "
        "END"
    )
    op.execute(
        f"CREATE TRIGGER {table_name}_reject_delete "
        f"BEFORE DELETE ON {table_name} "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{table_name} rows are immutable'); "
        "END"
    )
    op.execute(
        f"CREATE TRIGGER {table_name}_reject_conflicting_insert "
        f"BEFORE INSERT ON {table_name} "
        f"WHEN EXISTS (SELECT 1 FROM {table_name} WHERE {conflict_when}) "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{table_name} identity already exists'); "
        "END"
    )


def upgrade() -> None:
    op.create_table(
        "topics",
        sa.Column("topic_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "length(name) BETWEEN 1 AND 200 "
            "AND length(trim(name)) > 0 AND name = trim(name)",
            name="ck_topics_name",
        ),
        sa.CheckConstraint(
            "description IS NULL OR "
            "(length(description) BETWEEN 1 AND 2000 "
            "AND length(trim(description)) > 0 "
            "AND description = trim(description))",
            name="ck_topics_description",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("topic_id"),
    )
    op.create_index("ux_topics_name", "topics", ["name"], unique=True)
    op.create_index(
        "ix_topics_owner_created",
        "topics",
        ["owner_agent_id", "created_at", "topic_id"],
        unique=False,
    )

    op.create_table(
        "subscriptions",
        sa.Column("subscription_id", sa.String(length=36), nullable=False),
        sa.Column("subscriber_agent_id", sa.String(length=36), nullable=False),
        sa.Column("topic_id", sa.String(length=36), nullable=False),
        sa.Column("creation_event_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "creation_event_id > 0",
            name="ck_subscriptions_creation_event",
        ),
        sa.ForeignKeyConstraint(
            ["subscriber_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.topic_id"]),
        sa.ForeignKeyConstraint(
            ["creation_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("subscription_id"),
    )
    op.create_index(
        "ux_subscriptions_subscriber_topic",
        "subscriptions",
        ["subscriber_agent_id", "topic_id"],
        unique=True,
    )
    op.create_index(
        "ix_subscriptions_topic_delivery",
        "subscriptions",
        [
            "topic_id",
            "creation_event_id",
            "subscriber_agent_id",
            "subscription_id",
        ],
        unique=False,
    )

    op.create_table(
        "experience_capsules",
        sa.Column("capsule_id", sa.String(length=36), nullable=False),
        sa.Column(
            "transport_schema_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column("topic_id", sa.String(length=36), nullable=False),
        sa.Column(
            "source_experience_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column("source_version_id", sa.String(length=36), nullable=False),
        sa.Column("publisher_agent_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("summary", sa.String(length=1000), nullable=False),
        sa.Column("mechanism", sa.String(length=2000), nullable=False),
        sa.Column("tags", sa.LargeBinary(), nullable=False),
        sa.Column("applicability", sa.LargeBinary(), nullable=False),
        sa.Column("evidence", sa.LargeBinary(), nullable=False),
        sa.Column("falsifiers", sa.LargeBinary(), nullable=False),
        sa.Column("publisher_confidence", sa.Float(), nullable=False),
        sa.Column("provenance_chain", sa.LargeBinary(), nullable=False),
        sa.Column("root_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("source_content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.Column("expires_at", sa.String(length=27), nullable=False),
        sa.Column("hop_count", sa.Integer(), nullable=False),
        sa.Column("capsule_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "transport_schema_version = 1",
            name="ck_experience_capsules_transport_version",
        ),
        sa.CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis') "
            "AND length(trim(body)) > 0 "
            "AND length(CAST(body AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0 "
            "AND length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_capsules_content",
        ),
        sa.CheckConstraint(
            "length(tags) > 0 AND length(applicability) > 0 "
            "AND length(evidence) > 0 AND length(falsifiers) > 0 "
            "AND length(provenance_chain) > 0",
            name="ck_experience_capsules_arrays",
        ),
        sa.CheckConstraint(
            "publisher_confidence BETWEEN 0 AND 1",
            name="ck_experience_capsules_confidence",
        ),
        sa.CheckConstraint(
            f"{_sha256_check('root_fingerprint')} "
            f"AND {_sha256_check('source_content_hash')} "
            f"AND {_sha256_check('capsule_hash')}",
            name="ck_experience_capsules_hashes",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_experience_capsules_expiry",
        ),
        sa.CheckConstraint(
            "hop_count BETWEEN 0 AND 4 "
            "AND json_type(CAST(provenance_chain AS TEXT)) = 'array' "
            "AND json_array_length(CAST(provenance_chain AS TEXT)) = hop_count",
            name="ck_experience_capsules_hop_count",
        ),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.topic_id"]),
        sa.ForeignKeyConstraint(
            ["source_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.ForeignKeyConstraint(
            ["publisher_agent_id"],
            ["agents.agent_id"],
        ),
        sa.PrimaryKeyConstraint("capsule_id"),
    )
    op.create_index(
        "ix_experience_capsules_topic_created",
        "experience_capsules",
        ["topic_id", "created_at", "capsule_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_capsules_publisher_created",
        "experience_capsules",
        ["publisher_agent_id", "created_at", "capsule_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_capsules_source",
        "experience_capsules",
        ["source_experience_id", "source_version_id"],
        unique=False,
    )

    op.create_table(
        "adoption_records",
        sa.Column("adoption_id", sa.String(length=36), nullable=False),
        sa.Column("adopter_agent_id", sa.String(length=36), nullable=False),
        sa.Column("capsule_id", sa.String(length=36), nullable=False),
        sa.Column(
            "resulting_experience_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column("captured_trust", sa.Float(), nullable=False),
        sa.Column("provenance_chain", sa.LargeBinary(), nullable=False),
        sa.Column("root_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("corroboration_applied", sa.Boolean(), nullable=False),
        sa.Column("adopted_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "captured_trust BETWEEN 0 AND 1",
            name="ck_adoption_records_captured_trust",
        ),
        sa.CheckConstraint(
            "length(provenance_chain) > 0 "
            "AND json_type(CAST(provenance_chain AS TEXT)) = 'array' "
            "AND json_array_length(CAST(provenance_chain AS TEXT)) > 0",
            name="ck_adoption_records_provenance",
        ),
        sa.CheckConstraint(
            _sha256_check("root_fingerprint"),
            name="ck_adoption_records_root_fingerprint",
        ),
        sa.CheckConstraint(
            "corroboration_applied IN (0, 1)",
            name="ck_adoption_records_corroboration",
        ),
        sa.ForeignKeyConstraint(
            ["adopter_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(
            ["capsule_id"],
            ["experience_capsules.capsule_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resulting_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.PrimaryKeyConstraint("adoption_id"),
    )
    op.create_index(
        "ux_adoption_records_adopter_capsule",
        "adoption_records",
        ["adopter_agent_id", "capsule_id"],
        unique=True,
    )
    op.create_index(
        "ix_adoption_records_resulting_root",
        "adoption_records",
        ["resulting_experience_id", "root_fingerprint"],
        unique=False,
    )
    op.create_index(
        "ux_adoption_records_corroborated_root",
        "adoption_records",
        ["resulting_experience_id", "root_fingerprint"],
        unique=True,
        sqlite_where=sa.text("corroboration_applied = 1"),
    )

    op.create_table(
        "capsule_feedback",
        sa.Column("feedback_id", sa.String(length=36), nullable=False),
        sa.Column("observer_agent_id", sa.String(length=36), nullable=False),
        sa.Column("capsule_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.String(length=7), nullable=False),
        sa.Column("reason", sa.LargeBinary(), nullable=False),
        sa.Column("evidence", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "revision > 0",
            name="ck_capsule_feedback_revision",
        ),
        sa.CheckConstraint(
            "verdict IN ('useful', 'refuted', 'harmful')",
            name="ck_capsule_feedback_verdict",
        ),
        sa.CheckConstraint(
            "length(reason) > 0 AND length(evidence) > 0",
            name="ck_capsule_feedback_payloads",
        ),
        sa.ForeignKeyConstraint(
            ["observer_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(
            ["capsule_id"],
            ["experience_capsules.capsule_id"],
        ),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index(
        "ux_capsule_feedback_observer_capsule_revision",
        "capsule_feedback",
        ["observer_agent_id", "capsule_id", "revision"],
        unique=True,
    )
    op.create_index(
        "ix_capsule_feedback_capsule_observer_revision",
        "capsule_feedback",
        ["capsule_id", "observer_agent_id", "revision"],
        unique=False,
    )

    op.create_table(
        "capsule_state",
        sa.Column("capsule_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=9), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'retracted')",
            name="ck_capsule_state_status",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_capsule_state_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["capsule_id"],
            ["experience_capsules.capsule_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("capsule_id"),
    )

    op.create_table(
        "inbox_items",
        sa.Column("item_id", sa.String(length=36), nullable=False),
        sa.Column("recipient_agent_id", sa.String(length=36), nullable=False),
        sa.Column("capsule_id", sa.String(length=36), nullable=False),
        sa.Column("state", sa.String(length=8), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending', 'adopted', 'rejected')",
            name="ck_inbox_items_state",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_inbox_items_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["recipient_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(
            ["capsule_id"],
            ["experience_capsules.capsule_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("item_id"),
    )
    op.create_index(
        "ux_inbox_items_recipient_capsule",
        "inbox_items",
        ["recipient_agent_id", "capsule_id"],
        unique=True,
    )
    op.create_index(
        "ix_inbox_items_recipient_state",
        "inbox_items",
        ["recipient_agent_id", "state", "item_id"],
        unique=False,
    )

    op.create_table(
        "agent_reputation",
        sa.Column("subject_agent_id", sa.String(length=36), nullable=False),
        sa.Column("observer_agent_id", sa.String(length=36), nullable=False),
        sa.Column("useful_count", sa.Integer(), nullable=False),
        sa.Column("refuted_count", sa.Integer(), nullable=False),
        sa.Column("harmful_count", sa.Integer(), nullable=False),
        sa.Column("alpha", sa.Integer(), nullable=False),
        sa.Column("beta", sa.Integer(), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "useful_count >= 0 AND refuted_count >= 0 AND harmful_count >= 0",
            name="ck_agent_reputation_counts",
        ),
        sa.CheckConstraint(
            "alpha = 2 + useful_count",
            name="ck_agent_reputation_alpha_prior",
        ),
        sa.CheckConstraint(
            "beta = 2 + refuted_count + harmful_count",
            name="ck_agent_reputation_beta_prior",
        ),
        sa.CheckConstraint(
            "alpha > 0 AND beta > 0 "
            "AND subject_agent_id != observer_agent_id",
            name="ck_agent_reputation_trust",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_agent_reputation_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["subject_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(
            ["observer_agent_id"],
            ["agents.agent_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("subject_agent_id", "observer_agent_id"),
    )

    _create_immutable_triggers(
        "topics",
        conflict_when="topic_id = NEW.topic_id OR name = NEW.name",
    )
    _create_immutable_triggers(
        "subscriptions",
        conflict_when=(
            "subscription_id = NEW.subscription_id "
            "OR (subscriber_agent_id = NEW.subscriber_agent_id "
            "AND topic_id = NEW.topic_id)"
        ),
    )
    _create_immutable_triggers(
        "experience_capsules",
        conflict_when="capsule_id = NEW.capsule_id",
    )
    _create_immutable_triggers(
        "adoption_records",
        conflict_when=(
            "adoption_id = NEW.adoption_id "
            "OR (adopter_agent_id = NEW.adopter_agent_id "
            "AND capsule_id = NEW.capsule_id) "
            "OR (NEW.corroboration_applied = 1 "
            "AND corroboration_applied = 1 "
            "AND resulting_experience_id = NEW.resulting_experience_id "
            "AND root_fingerprint = NEW.root_fingerprint)"
        ),
    )
    _create_immutable_triggers(
        "capsule_feedback",
        conflict_when=(
            "feedback_id = NEW.feedback_id "
            "OR (observer_agent_id = NEW.observer_agent_id "
            "AND capsule_id = NEW.capsule_id "
            "AND revision = NEW.revision)"
        ),
    )


def _drop_immutable_triggers(table_name: str) -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_conflicting_insert")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_delete")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_update")


def _refuse_populated_sharing_downgrade() -> None:
    connection = op.get_bind()
    for table_name in (
        "topics",
        "subscriptions",
        "experience_capsules",
        "adoption_records",
        "capsule_feedback",
    ):
        populated = connection.execute(
            sa.text(f"SELECT 1 FROM {table_name} LIMIT 1")
        ).first()
        if populated is not None:
            raise RuntimeError(
                "Cannot downgrade while sharing source or ledger data exists"
            )

    ledger_entry = connection.execute(
        sa.text(
            "SELECT 1 FROM domain_events "
            "WHERE aggregate_type IN "
            "('topic', 'subscription', 'capsule', 'inbox_item') "
            "OR event_type LIKE 'topic.%' "
            "OR event_type LIKE 'subscription.%' "
            "OR event_type LIKE 'capsule.%' "
            "OR event_type = 'experience.corroborated' "
            "LIMIT 1"
        )
    ).first()
    if ledger_entry is not None:
        raise RuntimeError(
            "Cannot downgrade while sharing source or ledger data exists"
        )


def downgrade() -> None:
    _refuse_populated_sharing_downgrade()

    _drop_immutable_triggers("capsule_feedback")
    _drop_immutable_triggers("adoption_records")
    _drop_immutable_triggers("experience_capsules")
    _drop_immutable_triggers("subscriptions")
    _drop_immutable_triggers("topics")

    op.drop_table("agent_reputation")
    op.drop_index(
        "ix_inbox_items_recipient_state",
        table_name="inbox_items",
    )
    op.drop_index(
        "ux_inbox_items_recipient_capsule",
        table_name="inbox_items",
    )
    op.drop_table("inbox_items")
    op.drop_table("capsule_state")
    op.drop_index(
        "ix_capsule_feedback_capsule_observer_revision",
        table_name="capsule_feedback",
    )
    op.drop_index(
        "ux_capsule_feedback_observer_capsule_revision",
        table_name="capsule_feedback",
    )
    op.drop_table("capsule_feedback")
    op.drop_index(
        "ux_adoption_records_corroborated_root",
        table_name="adoption_records",
    )
    op.drop_index(
        "ix_adoption_records_resulting_root",
        table_name="adoption_records",
    )
    op.drop_index(
        "ux_adoption_records_adopter_capsule",
        table_name="adoption_records",
    )
    op.drop_table("adoption_records")
    op.drop_index(
        "ix_experience_capsules_source",
        table_name="experience_capsules",
    )
    op.drop_index(
        "ix_experience_capsules_publisher_created",
        table_name="experience_capsules",
    )
    op.drop_index(
        "ix_experience_capsules_topic_created",
        table_name="experience_capsules",
    )
    op.drop_table("experience_capsules")
    op.drop_index(
        "ix_subscriptions_topic_delivery",
        table_name="subscriptions",
    )
    op.drop_index(
        "ux_subscriptions_subscriber_topic",
        table_name="subscriptions",
    )
    op.drop_table("subscriptions")
    op.drop_index("ix_topics_owner_created", table_name="topics")
    op.drop_index("ux_topics_name", table_name="topics")
    op.drop_table("topics")
