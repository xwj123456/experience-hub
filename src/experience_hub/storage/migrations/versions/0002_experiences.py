"""Create immutable experience sources and lifecycle/search projections."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_experiences"
down_revision: str | None = "0001_core"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _sha256_check(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _create_immutable_triggers(
    table_name: str,
    *,
    conflict_when: str,
    conflict_message: str,
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
        f"SELECT RAISE(ABORT, '{conflict_message}'); "
        "END"
    )


def upgrade() -> None:
    op.create_table(
        "experiences",
        sa.Column("experience_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("origin", sa.String(length=15), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis')",
            name="ck_experiences_kind",
        ),
        sa.CheckConstraint(
            "origin IN ('local', 'adopted_capsule', 'adopted_idea')",
            name="ck_experiences_origin",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("experience_id"),
    )
    op.create_index(
        "ix_experiences_owner_created",
        "experiences",
        ["owner_agent_id", "created_at", "experience_id"],
        unique=False,
    )

    op.create_table(
        "experience_versions",
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("experience_id", sa.String(length=36), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("summary", sa.String(length=1000), nullable=False),
        sa.Column("mechanism", sa.String(length=2000), nullable=False),
        sa.Column("tags", sa.LargeBinary(), nullable=False),
        sa.Column("applicability", sa.LargeBinary(), nullable=False),
        sa.Column("evidence", sa.LargeBinary(), nullable=False),
        sa.Column("falsifiers", sa.LargeBinary(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("supersedes_version_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "version_number > 0",
            name="ck_experience_versions_number",
        ),
        sa.CheckConstraint(
            "(version_number = 1 AND supersedes_version_id IS NULL) "
            "OR (version_number > 1 AND supersedes_version_id IS NOT NULL)",
            name="ck_experience_versions_supersession",
        ),
        sa.CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0",
            name="ck_experience_versions_summary",
        ),
        sa.CheckConstraint(
            "length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_versions_mechanism",
        ),
        sa.CheckConstraint(
            "length(tags) > 0 AND length(applicability) > 0 "
            "AND length(evidence) > 0 AND length(falsifiers) > 0",
            name="ck_experience_versions_arrays",
        ),
        sa.CheckConstraint(
            _sha256_check("content_hash"),
            name="ck_experience_versions_content_hash",
        ),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index(
        "ux_experience_versions_experience_number",
        "experience_versions",
        ["experience_id", "version_number"],
        unique=True,
    )
    op.create_index(
        "ix_experience_versions_experience_created",
        "experience_versions",
        ["experience_id", "created_at", "version_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_versions_supersedes",
        "experience_versions",
        ["supersedes_version_id"],
        unique=False,
    )

    op.create_table(
        "experience_payloads",
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("codec", sa.String(length=5), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "codec IN ('plain', 'zlib')",
            name="ck_experience_payloads_codec",
        ),
        sa.CheckConstraint(
            "length(payload) > 0",
            name="ck_experience_payloads_payload",
        ),
        sa.CheckConstraint(
            _sha256_check("payload_hash"),
            name="ck_experience_payloads_payload_hash",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["experience_versions.version_id"],
        ),
        sa.PrimaryKeyConstraint("version_id"),
    )

    op.create_table(
        "experience_state",
        sa.Column("experience_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=False),
        sa.Column("current_content_hash", sa.String(length=64), nullable=False),
        sa.Column("temperature", sa.String(length=8), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("activation_score", sa.Float(), nullable=False),
        sa.Column("source_trust", sa.Float(), nullable=False),
        sa.Column("access_count", sa.Integer(), nullable=False),
        sa.Column("access_strength", sa.Float(), nullable=False),
        sa.Column("strength_updated_at", sa.String(length=27), nullable=False),
        sa.Column("last_accessed_at", sa.String(length=27), nullable=True),
        sa.Column("last_transition_at", sa.String(length=27), nullable=False),
        sa.Column(
            "last_lifecycle_evaluated_at",
            sa.String(length=27),
            nullable=True,
        ),
        sa.Column("consecutive_below_threshold", sa.Integer(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "temperature IN ('hot', 'warm', 'cold', 'archived')",
            name="ck_experience_state_temperature",
        ),
        sa.CheckConstraint(
            "importance BETWEEN 0 AND 1",
            name="ck_experience_state_importance",
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name="ck_experience_state_confidence",
        ),
        sa.CheckConstraint(
            "activation_score BETWEEN 0 AND 1",
            name="ck_experience_state_activation",
        ),
        sa.CheckConstraint(
            "source_trust BETWEEN 0 AND 1",
            name="ck_experience_state_source_trust",
        ),
        sa.CheckConstraint(
            "access_count >= 0",
            name="ck_experience_state_access_count",
        ),
        sa.CheckConstraint(
            "access_strength BETWEEN 0 AND 20",
            name="ck_experience_state_access_strength",
        ),
        sa.CheckConstraint(
            "consecutive_below_threshold >= 0",
            name="ck_experience_state_below_threshold",
        ),
        sa.CheckConstraint(
            "pinned IN (0, 1)",
            name="ck_experience_state_pinned",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_experience_state_projection_event",
        ),
        sa.CheckConstraint(
            _sha256_check("current_content_hash"),
            name="ck_experience_state_content_hash",
        ),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["current_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("experience_id"),
    )
    op.create_index(
        "ux_experience_state_owner_content",
        "experience_state",
        ["owner_agent_id", "current_content_hash"],
        unique=True,
    )
    op.create_index(
        "ix_experience_state_owner_temperature",
        "experience_state",
        ["owner_agent_id", "temperature", "experience_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_state_current_version",
        "experience_state",
        ["current_version_id"],
        unique=False,
    )

    op.create_table(
        "experience_links",
        sa.Column("source_experience_id", sa.String(length=36), nullable=False),
        sa.Column("source_version_id", sa.String(length=36), nullable=False),
        sa.Column("target_experience_id", sa.String(length=36), nullable=False),
        sa.Column("relation", sa.String(length=12), nullable=False),
        sa.Column("source_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "relation IN ('derived_from', 'supports', 'contradicts', 'tests')",
            name="ck_experience_links_relation",
        ),
        sa.ForeignKeyConstraint(
            ["source_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.ForeignKeyConstraint(
            ["target_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint(
            "source_version_id",
            "target_experience_id",
            "relation",
        ),
    )
    op.create_index(
        "ix_experience_links_target_relation",
        "experience_links",
        [
            "target_experience_id",
            "relation",
            "source_experience_id",
            "source_version_id",
        ],
        unique=False,
    )
    op.create_index(
        "ix_experience_links_source_experience",
        "experience_links",
        ["source_experience_id", "source_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_links_source_event",
        "experience_links",
        ["source_event_id"],
        unique=False,
    )

    op.create_table(
        "experience_terms",
        sa.Column("experience_id", sa.String(length=36), nullable=False),
        sa.Column("term", sa.String(), nullable=False),
        sa.Column("term_kind", sa.String(length=12), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "length(term) > 0",
            name="ck_experience_terms_term",
        ),
        sa.CheckConstraint(
            "term_kind IN ('word', 'char_trigram', 'tag', 'mechanism')",
            name="ck_experience_terms_kind",
        ),
        sa.CheckConstraint(
            "weight > 0 AND weight <= 1.5",
            name="ck_experience_terms_weight",
        ),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.experience_id"],
        ),
        sa.PrimaryKeyConstraint("experience_id", "term", "term_kind"),
    )
    op.create_index(
        "ix_experience_terms_lookup",
        "experience_terms",
        ["term_kind", "term", "experience_id"],
        unique=False,
    )

    _create_immutable_triggers(
        "experiences",
        conflict_when="experience_id = NEW.experience_id",
        conflict_message="experiences identity already exists",
    )
    _create_immutable_triggers(
        "experience_versions",
        conflict_when=(
            "version_id = NEW.version_id "
            "OR (experience_id = NEW.experience_id "
            "AND version_number = NEW.version_number)"
        ),
        conflict_message="experience_versions identity already exists",
    )
    _create_immutable_triggers(
        "experience_links",
        conflict_when=(
            "source_version_id = NEW.source_version_id "
            "AND target_experience_id = NEW.target_experience_id "
            "AND relation = NEW.relation"
        ),
        conflict_message="experience_links identity already exists",
    )
    op.execute(
        "CREATE TRIGGER experience_payloads_reject_delete "
        "BEFORE DELETE ON experience_payloads "
        "BEGIN "
        "SELECT RAISE(ABORT, 'experience_payloads rows are immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER experience_payloads_reject_semantic_update "
        "BEFORE UPDATE ON experience_payloads "
        "WHEN OLD.version_id <> NEW.version_id "
        "OR OLD.payload_hash <> NEW.payload_hash "
        "BEGIN "
        "SELECT RAISE(ABORT, 'payload semantic identity is immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER experience_payloads_reject_unguarded_rewrite "
        "BEFORE UPDATE ON experience_payloads "
        "WHEN (OLD.codec <> NEW.codec OR OLD.payload <> NEW.payload) "
        "AND experience_hub_payload_rewrite_allowed() <> 1 "
        "BEGIN "
        "SELECT RAISE(ABORT, 'payload rewrite is not allowed'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER experience_payloads_reject_conflicting_insert "
        "BEFORE INSERT ON experience_payloads "
        "WHEN EXISTS ("
        "SELECT 1 FROM experience_payloads WHERE version_id = NEW.version_id"
        ") "
        "BEGIN "
        "SELECT RAISE(ABORT, 'experience_payloads identity already exists'); "
        "END"
    )


def _drop_immutable_triggers(table_name: str) -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_conflicting_insert")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_delete")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_update")


def _refuse_populated_experience_downgrade() -> None:
    connection = op.get_bind()
    for table_name in (
        "experiences",
        "experience_versions",
        "experience_payloads",
        "experience_links",
    ):
        populated = connection.execute(
            sa.text(f"SELECT 1 FROM {table_name} LIMIT 1")
        ).first()
        if populated is not None:
            raise RuntimeError(
                "Cannot downgrade while experience source or ledger data exists"
            )
    ledger_entry = connection.execute(
        sa.text(
            "SELECT 1 FROM domain_events "
            "WHERE aggregate_type = 'experience' "
            "OR event_type LIKE 'experience.%' "
            "LIMIT 1"
        )
    ).first()
    if ledger_entry is not None:
        raise RuntimeError(
            "Cannot downgrade while experience source or ledger data exists"
        )


def downgrade() -> None:
    _refuse_populated_experience_downgrade()
    op.execute(
        "DROP TRIGGER IF EXISTS "
        "experience_payloads_reject_conflicting_insert"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS "
        "experience_payloads_reject_unguarded_rewrite"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS experience_payloads_reject_semantic_update"
    )
    op.execute("DROP TRIGGER IF EXISTS experience_payloads_reject_delete")
    _drop_immutable_triggers("experience_links")
    _drop_immutable_triggers("experience_versions")
    _drop_immutable_triggers("experiences")

    op.drop_index(
        "ix_experience_terms_lookup",
        table_name="experience_terms",
    )
    op.drop_table("experience_terms")
    op.drop_index(
        "ix_experience_links_source_event",
        table_name="experience_links",
    )
    op.drop_index(
        "ix_experience_links_source_experience",
        table_name="experience_links",
    )
    op.drop_index(
        "ix_experience_links_target_relation",
        table_name="experience_links",
    )
    op.drop_table("experience_links")
    op.drop_index(
        "ix_experience_state_current_version",
        table_name="experience_state",
    )
    op.drop_index(
        "ix_experience_state_owner_temperature",
        table_name="experience_state",
    )
    op.drop_index(
        "ux_experience_state_owner_content",
        table_name="experience_state",
    )
    op.drop_table("experience_state")
    op.drop_table("experience_payloads")
    op.drop_index(
        "ix_experience_versions_supersedes",
        table_name="experience_versions",
    )
    op.drop_index(
        "ix_experience_versions_experience_created",
        table_name="experience_versions",
    )
    op.drop_index(
        "ux_experience_versions_experience_number",
        table_name="experience_versions",
    )
    op.drop_table("experience_versions")
    op.drop_index(
        "ix_experiences_owner_created",
        table_name="experiences",
    )
    op.drop_table("experiences")
