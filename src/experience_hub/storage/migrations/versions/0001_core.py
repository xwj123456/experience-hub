"""Create the core authoritative, ledger, and operational tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_core"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("agent_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "length(trim(name)) > 0 AND name = trim(name)",
            name="ck_agents_name_trimmed",
        ),
        sa.PrimaryKeyConstraint("agent_id"),
    )
    op.create_index("ux_agents_name", "agents", ["name"], unique=True)

    op.create_table(
        "idempotency_records",
        sa.Column("receipt_id", sa.String(length=36), nullable=False),
        sa.Column("caller_scope", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("result_resource_type", sa.String(), nullable=True),
        sa.Column("result_resource_id", sa.String(length=36), nullable=True),
        sa.Column("response_status_code", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.LargeBinary(), nullable=True),
        sa.Column("response_content_type", sa.String(), nullable=True),
        sa.Column("response_headers", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.Column("completed_at", sa.String(length=27), nullable=True),
        sa.CheckConstraint(
            "length(trim(caller_scope)) > 0 AND caller_scope = trim(caller_scope)",
            name="ck_idempotency_records_caller_scope",
        ),
        sa.CheckConstraint(
            "length(trim(scope)) > 0 AND scope = trim(scope)",
            name="ck_idempotency_records_scope",
        ),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128 "
            "AND idempotency_key = trim(idempotency_key)",
            name="ck_idempotency_records_key",
        ),
        sa.CheckConstraint(
            "length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_idempotency_records_request_hash",
        ),
        sa.CheckConstraint(
            "state IN ('in_progress', 'completed')",
            name="ck_idempotency_records_state",
        ),
        sa.CheckConstraint(
            "(result_resource_type IS NULL AND result_resource_id IS NULL) OR "
            "(length(trim(result_resource_type)) > 0 "
            "AND result_resource_type = trim(result_resource_type) "
            "AND result_resource_id IS NOT NULL)",
            name="ck_idempotency_records_resource",
        ),
        sa.CheckConstraint(
            "(state = 'in_progress' "
            "AND response_status_code IS NULL "
            "AND response_body IS NULL "
            "AND response_content_type IS NULL "
            "AND response_headers IS NULL "
            "AND completed_at IS NULL) "
            "OR (state = 'completed' "
            "AND response_status_code BETWEEN 100 AND 599 "
            "AND response_body IS NOT NULL "
            "AND length(trim(response_content_type)) > 0 "
            "AND response_headers IS NOT NULL "
            "AND completed_at IS NOT NULL)",
            name="ck_idempotency_records_completion",
        ),
        sa.PrimaryKeyConstraint("receipt_id"),
    )
    op.create_index(
        "ux_idempotency_records_scope_key",
        "idempotency_records",
        ["caller_scope", "scope", "idempotency_key"],
        unique=True,
    )
    op.create_index(
        "ix_idempotency_records_resource_state",
        "idempotency_records",
        ["result_resource_type", "result_resource_id", "state"],
        unique=False,
    )

    op.create_table(
        "domain_events",
        sa.Column("event_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("aggregate_type", sa.String(), nullable=False),
        sa.Column("aggregate_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("actor_agent_id", sa.String(length=36), nullable=True),
        sa.Column("causation_id", sa.String(length=36), nullable=False),
        sa.Column("occurred_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "length(trim(aggregate_type)) > 0 "
            "AND aggregate_type = trim(aggregate_type)",
            name="ck_domain_events_aggregate_type",
        ),
        sa.CheckConstraint(
            "length(trim(event_type)) > 0 AND event_type = trim(event_type)",
            name="ck_domain_events_event_type",
        ),
        sa.CheckConstraint(
            "length(payload) > 0",
            name="ck_domain_events_payload",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_domain_events_sequence"),
        sa.ForeignKeyConstraint(["actor_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["causation_id"],
            ["idempotency_records.receipt_id"],
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ux_domain_events_aggregate_sequence",
        "domain_events",
        ["aggregate_type", "aggregate_id", "sequence"],
        unique=True,
    )
    op.execute(
        "CREATE TRIGGER domain_events_reject_update "
        "BEFORE UPDATE ON domain_events "
        "BEGIN "
        "SELECT RAISE(ABORT, 'domain_events rows are immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER domain_events_reject_delete "
        "BEFORE DELETE ON domain_events "
        "BEGIN "
        "SELECT RAISE(ABORT, 'domain_events rows are immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER domain_events_reject_conflicting_insert "
        "BEFORE INSERT ON domain_events "
        "WHEN EXISTS ("
        "SELECT 1 FROM domain_events "
        "WHERE event_id = NEW.event_id "
        "OR (aggregate_type = NEW.aggregate_type "
        "AND aggregate_id = NEW.aggregate_id "
        "AND sequence = NEW.sequence)"
        ") "
        "BEGIN "
        "SELECT RAISE("
        "ABORT, 'domain_events identity or sequence already exists'"
        "); "
        "END"
    )

    op.create_table(
        "projection_versions",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("reducer_version", sa.Integer(), nullable=False),
        sa.Column("last_applied_event_id", sa.Integer(), nullable=False),
        sa.Column("last_verified_hash", sa.String(length=64), nullable=True),
        sa.Column("last_verified_at", sa.String(length=27), nullable=True),
        sa.CheckConstraint(
            "last_applied_event_id >= 0",
            name="ck_projection_versions_event_id",
        ),
        sa.CheckConstraint(
            "length(trim(name)) > 0 AND name = trim(name)",
            name="ck_projection_versions_name",
        ),
        sa.CheckConstraint(
            "reducer_version > 0",
            name="ck_projection_versions_reducer_version",
        ),
        sa.CheckConstraint(
            "(last_verified_hash IS NULL AND last_verified_at IS NULL) OR "
            "(length(last_verified_hash) = 64 "
            "AND last_verified_hash NOT GLOB '*[^0-9a-f]*' "
            "AND last_verified_at IS NOT NULL)",
            name="ck_projection_versions_verification",
        ),
        sa.PrimaryKeyConstraint("name"),
    )

    op.create_table(
        "lifecycle_lease",
        sa.Column("lease_name", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=True),
        sa.Column("acquired_at", sa.String(length=27), nullable=True),
        sa.Column("expires_at", sa.String(length=27), nullable=True),
        sa.CheckConstraint(
            "lease_name = 'lifecycle'",
            name="ck_lifecycle_lease_singleton",
        ),
        sa.CheckConstraint(
            "(owner_id IS NULL AND acquired_at IS NULL AND expires_at IS NULL) OR "
            "(owner_id IS NOT NULL AND acquired_at IS NOT NULL "
            "AND expires_at IS NOT NULL AND expires_at > acquired_at)",
            name="ck_lifecycle_lease_state",
        ),
        sa.PrimaryKeyConstraint("lease_name"),
    )

    op.execute(
        "CREATE TRIGGER agents_reject_update "
        "BEFORE UPDATE ON agents "
        "BEGIN "
        "SELECT RAISE(ABORT, 'agents rows are immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER agents_reject_delete "
        "BEFORE DELETE ON agents "
        "BEGIN "
        "SELECT RAISE(ABORT, 'agents rows are immutable'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER agents_reject_conflicting_insert "
        "BEFORE INSERT ON agents "
        "WHEN EXISTS ("
        "SELECT 1 FROM agents "
        "WHERE agent_id = NEW.agent_id OR name = NEW.name"
        ") "
        "BEGIN "
        "SELECT RAISE(ABORT, 'agents identity or name already exists'); "
        "END"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS agents_reject_conflicting_insert")
    op.execute("DROP TRIGGER IF EXISTS agents_reject_delete")
    op.execute("DROP TRIGGER IF EXISTS agents_reject_update")
    op.drop_table("lifecycle_lease")
    op.drop_table("projection_versions")
    op.execute("DROP TRIGGER IF EXISTS domain_events_reject_conflicting_insert")
    op.execute("DROP TRIGGER IF EXISTS domain_events_reject_delete")
    op.execute("DROP TRIGGER IF EXISTS domain_events_reject_update")
    op.drop_index(
        "ux_domain_events_aggregate_sequence",
        table_name="domain_events",
    )
    op.drop_table("domain_events")
    op.drop_index(
        "ux_idempotency_records_scope_key",
        table_name="idempotency_records",
    )
    op.drop_index(
        "ix_idempotency_records_resource_state",
        table_name="idempotency_records",
    )
    op.drop_table("idempotency_records")
    op.drop_index("ux_agents_name", table_name="agents")
    op.drop_table("agents")
