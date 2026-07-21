"""Create immutable inspiration sources and replayable projections."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_inspiration"
down_revision: str | None = "0003_sharing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _sha256_check(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _json_array_check(column: str, *, nonempty: bool) -> str:
    cardinality = (
        f" AND json_array_length(CAST({column} AS TEXT)) > 0"
        if nonempty
        else ""
    )
    return (
        f"length({column}) > 0 "
        f"AND json_valid(CAST({column} AS TEXT)) "
        f"AND json_type(CAST({column} AS TEXT)) = 'array'"
        f"{cardinality}"
    )


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
        "inspiration_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("goal", sa.String(length=2000), nullable=False),
        sa.Column("context", sa.String(length=4000), nullable=True),
        sa.Column("mode", sa.String(length=11), nullable=False),
        sa.Column("generator_kind", sa.String(length=17), nullable=False),
        sa.Column("generator_configuration", sa.LargeBinary(), nullable=False),
        sa.Column("operators", sa.LargeBinary(), nullable=False),
        sa.Column("include_inbox", sa.Boolean(), nullable=False),
        sa.Column("branches_per_operator", sa.Integer(), nullable=False),
        sa.Column("output_tokens_per_operator", sa.Integer(), nullable=False),
        sa.Column("total_output_tokens", sa.Integer(), nullable=False),
        sa.Column("operator_timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("global_timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "length(goal) BETWEEN 1 AND 2000 "
            "AND length(trim(goal)) > 0 AND goal = trim(goal)",
            name="ck_inspiration_runs_goal",
        ),
        sa.CheckConstraint(
            "context IS NULL OR "
            "(length(context) BETWEEN 1 AND 4000 "
            "AND length(trim(context)) > 0 AND context = trim(context))",
            name="ck_inspiration_runs_context",
        ),
        sa.CheckConstraint(
            "mode IN ('focused', 'associative')",
            name="ck_inspiration_runs_mode",
        ),
        sa.CheckConstraint(
            "generator_kind IN ('deterministic', 'openai_compatible')",
            name="ck_inspiration_runs_generator",
        ),
        sa.CheckConstraint(
            "length(generator_configuration) > 0 "
            "AND json_valid(CAST(generator_configuration AS TEXT)) "
            "AND json_type(CAST(generator_configuration AS TEXT)) = 'object'",
            name="ck_inspiration_runs_generator_configuration",
        ),
        sa.CheckConstraint(
            "CAST(operators AS TEXT) IN "
            "('[\"causal_gap\"]', "
            "'[\"counterfactual\"]', "
            "'[\"distant_analogy\"]', "
            "'[\"causal_gap\",\"counterfactual\"]', "
            "'[\"causal_gap\",\"distant_analogy\"]', "
            "'[\"counterfactual\",\"distant_analogy\"]', "
            "'[\"causal_gap\",\"counterfactual\",\"distant_analogy\"]')",
            name="ck_inspiration_runs_operators",
        ),
        sa.CheckConstraint(
            "include_inbox IN (0, 1)",
            name="ck_inspiration_runs_include_inbox",
        ),
        sa.CheckConstraint(
            "branches_per_operator BETWEEN 1 AND 3",
            name="ck_inspiration_runs_branches",
        ),
        sa.CheckConstraint(
            "output_tokens_per_operator BETWEEN 1 AND 1200 "
            "AND total_output_tokens BETWEEN 1 AND 3600",
            name="ck_inspiration_runs_token_budgets",
        ),
        sa.CheckConstraint(
            "operator_timeout_seconds BETWEEN 1 AND 30 "
            "AND global_timeout_seconds BETWEEN 1 AND 90 "
            "AND global_timeout_seconds >= operator_timeout_seconds",
            name="ck_inspiration_runs_time_budgets",
        ),
        sa.CheckConstraint(
            _sha256_check("request_hash"),
            name="ck_inspiration_runs_request_hash",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "ix_inspiration_runs_owner_created",
        "inspiration_runs",
        ["owner_agent_id", "created_at", "run_id"],
        unique=False,
    )

    op.create_table(
        "inspiration_snapshot_items",
        sa.Column("snapshot_item_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("stable_evidence_key", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=10), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("source_version_id", sa.String(length=36), nullable=False),
        sa.Column("source_state", sa.String(length=11), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("summary", sa.String(length=1000), nullable=False),
        sa.Column("mechanism", sa.String(length=2000), nullable=False),
        sa.Column("applicability", sa.LargeBinary(), nullable=False),
        sa.Column("tags", sa.LargeBinary(), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("source_trust", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            f"{_sha256_check('stable_evidence_key')} "
            f"AND {_sha256_check('content_hash')}",
            name="ck_inspiration_snapshot_items_hashes",
        ),
        sa.CheckConstraint(
            "source_type IN ('experience', 'capsule') "
            "AND source_state IN ('hot', 'warm', 'cold', 'quarantined') "
            "AND ((source_type = 'experience' "
            "AND source_state IN ('hot', 'warm', 'cold') "
            "AND source_trust BETWEEN 0 AND 1) "
            "OR (source_type = 'capsule' "
            "AND source_state = 'quarantined' "
            "AND source_trust = 0.25))",
            name="ck_inspiration_snapshot_items_source",
        ),
        sa.CheckConstraint(
            "rank BETWEEN 1 AND 12",
            name="ck_inspiration_snapshot_items_rank",
        ),
        sa.CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0 "
            "AND length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_inspiration_snapshot_items_text",
        ),
        sa.CheckConstraint(
            f"{_json_array_check('applicability', nonempty=False)} "
            "AND json_array_length(CAST(applicability AS TEXT)) <= 32 "
            f"AND {_json_array_check('tags', nonempty=False)} "
            "AND json_array_length(CAST(tags AS TEXT)) <= 32",
            name="ck_inspiration_snapshot_items_arrays",
        ),
        sa.CheckConstraint(
            "length(CAST(excerpt AS BLOB)) <= 2048",
            name="ck_inspiration_snapshot_items_excerpt",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inspiration_runs.run_id"],
        ),
        sa.PrimaryKeyConstraint("snapshot_item_id"),
    )
    op.create_index(
        "ux_inspiration_snapshot_items_run_rank",
        "inspiration_snapshot_items",
        ["run_id", "rank"],
        unique=True,
    )
    op.create_index(
        "ux_inspiration_snapshot_items_run_source",
        "inspiration_snapshot_items",
        ["run_id", "source_type", "source_id", "source_version_id"],
        unique=True,
    )
    op.create_index(
        "ix_inspiration_snapshot_items_stable_key",
        "inspiration_snapshot_items",
        ["stable_evidence_key", "snapshot_item_id"],
        unique=False,
    )

    op.create_table(
        "inspiration_ideas",
        sa.Column("idea_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("operator", sa.String(length=16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=1000), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("mechanism", sa.String(length=2000), nullable=False),
        sa.Column("predictions", sa.LargeBinary(), nullable=False),
        sa.Column("falsifiers", sa.LargeBinary(), nullable=False),
        sa.Column("assumptions", sa.LargeBinary(), nullable=False),
        sa.Column("proposed_test", sa.Text(), nullable=False),
        sa.Column("evidence_references", sa.LargeBinary(), nullable=False),
        sa.Column("idea_content_hash", sa.String(length=64), nullable=False),
        sa.Column("mechanism_hash", sa.String(length=64), nullable=False),
        sa.Column("duplicate_relation", sa.String(length=36), nullable=True),
        sa.CheckConstraint(
            "operator IN ('causal_gap', 'counterfactual', 'distant_analogy')",
            name="ck_inspiration_ideas_operator",
        ),
        sa.CheckConstraint(
            "ordinal BETWEEN 1 AND 3",
            name="ck_inspiration_ideas_ordinal",
        ),
        sa.CheckConstraint(
            "length(title) BETWEEN 1 AND 1000 "
            "AND length(trim(title)) > 0 "
            "AND length(CAST(hypothesis AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(trim(hypothesis)) > 0 "
            "AND length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0 "
            "AND length(CAST(proposed_test AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(trim(proposed_test)) > 0",
            name="ck_inspiration_ideas_text",
        ),
        sa.CheckConstraint(
            f"{_json_array_check('predictions', nonempty=True)} "
            "AND json_array_length(CAST(predictions AS TEXT)) <= 32 "
            f"AND {_json_array_check('falsifiers', nonempty=True)} "
            "AND json_array_length(CAST(falsifiers AS TEXT)) <= 32 "
            f"AND {_json_array_check('assumptions', nonempty=True)} "
            "AND json_array_length(CAST(assumptions AS TEXT)) <= 32 "
            f"AND {_json_array_check('evidence_references', nonempty=True)} "
            "AND json_array_length(CAST(evidence_references AS TEXT)) <= 12",
            name="ck_inspiration_ideas_arrays",
        ),
        sa.CheckConstraint(
            f"{_sha256_check('idea_content_hash')} "
            f"AND {_sha256_check('mechanism_hash')}",
            name="ck_inspiration_ideas_hashes",
        ),
        sa.CheckConstraint(
            "duplicate_relation IS NULL OR duplicate_relation != idea_id",
            name="ck_inspiration_ideas_duplicate_relation",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inspiration_runs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["duplicate_relation"],
            ["inspiration_ideas.idea_id"],
        ),
        sa.PrimaryKeyConstraint("idea_id"),
    )
    op.create_index(
        "ux_inspiration_ideas_run_operator_ordinal",
        "inspiration_ideas",
        ["run_id", "operator", "ordinal"],
        unique=True,
    )
    op.create_index(
        "ix_inspiration_ideas_mechanism_hash",
        "inspiration_ideas",
        ["mechanism_hash", "idea_id"],
        unique=False,
    )
    op.create_index(
        "ix_inspiration_ideas_duplicate_relation",
        "inspiration_ideas",
        ["duplicate_relation"],
        unique=False,
    )

    op.create_table(
        "idea_occurrences",
        sa.Column("occurrence_id", sa.String(length=36), nullable=False),
        sa.Column("idea_id", sa.String(length=36), nullable=False),
        sa.Column("mechanism_hash", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("occurred_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            f"{_sha256_check('mechanism_hash')} "
            f"AND {_sha256_check('snapshot_hash')}",
            name="ck_idea_occurrences_hashes",
        ),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["inspiration_ideas.idea_id"],
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inspiration_runs.run_id"],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("occurrence_id"),
    )
    op.create_index(
        "ux_idea_occurrences_run_mechanism",
        "idea_occurrences",
        ["run_id", "mechanism_hash"],
        unique=True,
    )
    op.create_index(
        "ux_idea_occurrences_idea",
        "idea_occurrences",
        ["idea_id"],
        unique=True,
    )
    op.create_index(
        "ix_idea_occurrences_mechanism_time",
        "idea_occurrences",
        ["mechanism_hash", "occurred_at", "occurrence_id"],
        unique=False,
    )

    op.create_table(
        "idea_adoption_records",
        sa.Column("adoption_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("idea_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("evidence_snapshot_item_ids", sa.LargeBinary(), nullable=False),
        sa.Column("evidence_stable_keys", sa.LargeBinary(), nullable=False),
        sa.Column("resulting_experience_id", sa.String(length=36), nullable=False),
        sa.Column("resulting_version_id", sa.String(length=36), nullable=False),
        sa.Column("adopted_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            _sha256_check("snapshot_hash"),
            name="ck_idea_adoption_records_snapshot_hash",
        ),
        sa.CheckConstraint(
            f"{_json_array_check('evidence_snapshot_item_ids', nonempty=True)} "
            f"AND {_json_array_check('evidence_stable_keys', nonempty=True)} "
            "AND json_array_length(CAST(evidence_snapshot_item_ids AS TEXT)) "
            "= json_array_length(CAST(evidence_stable_keys AS TEXT))",
            name="ck_idea_adoption_records_evidence",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["inspiration_ideas.idea_id"],
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inspiration_runs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resulting_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resulting_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.PrimaryKeyConstraint("adoption_id"),
    )
    op.create_index(
        "ux_idea_adoption_records_owner_idea",
        "idea_adoption_records",
        ["owner_agent_id", "idea_id"],
        unique=True,
    )

    op.create_table(
        "inspiration_run_state",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=21), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=True),
        sa.Column("operator_outcomes", sa.LargeBinary(), nullable=False),
        sa.Column("output_tokens_reserved", sa.Integer(), nullable=False),
        sa.Column("output_tokens_consumed", sa.Integer(), nullable=False),
        sa.Column("elapsed_milliseconds", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.String(length=27), nullable=False),
        sa.Column("completed_at", sa.String(length=27), nullable=True),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'completed_with_errors', "
            "'failed', 'timed_out')",
            name="ck_inspiration_run_state_status",
        ),
        sa.CheckConstraint(
            "snapshot_hash IS NULL OR "
            f"({_sha256_check('snapshot_hash')})",
            name="ck_inspiration_run_state_snapshot_hash",
        ),
        sa.CheckConstraint(
            _json_array_check("operator_outcomes", nonempty=False),
            name="ck_inspiration_run_state_outcomes",
        ),
        sa.CheckConstraint(
            "output_tokens_reserved >= 0 "
            "AND output_tokens_consumed >= 0 "
            "AND output_tokens_consumed <= output_tokens_reserved "
            "AND elapsed_milliseconds >= 0",
            name="ck_inspiration_run_state_budgets",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) "
            "OR (status != 'running' AND completed_at IS NOT NULL "
            "AND completed_at >= started_at)",
            name="ck_inspiration_run_state_terminality",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_inspiration_run_state_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inspiration_runs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "mechanism_incubation",
        sa.Column("cluster_id", sa.String(length=64), nullable=False),
        sa.Column(
            "canonical_mechanism_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("member_hashes", sa.LargeBinary(), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("distinct_snapshot_count", sa.Integer(), nullable=False),
        sa.Column("distinct_adopter_count", sa.Integer(), nullable=False),
        sa.Column("supported_count", sa.Integer(), nullable=False),
        sa.Column("refuted_count", sa.Integer(), nullable=False),
        sa.Column("maturity", sa.String(length=11), nullable=False),
        sa.Column("candidate_since", sa.String(length=27), nullable=True),
        sa.Column("last_signal_at", sa.String(length=27), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            f"{_sha256_check('cluster_id')} "
            f"AND {_sha256_check('canonical_mechanism_hash')} "
            "AND cluster_id = canonical_mechanism_hash",
            name="ck_mechanism_incubation_identity",
        ),
        sa.CheckConstraint(
            f"{_json_array_check('member_hashes', nonempty=True)}",
            name="ck_mechanism_incubation_members",
        ),
        sa.CheckConstraint(
            "occurrence_count >= 1 "
            "AND distinct_snapshot_count BETWEEN 1 AND occurrence_count "
            "AND distinct_adopter_count >= 0 "
            "AND supported_count >= 0 "
            "AND refuted_count >= 0",
            name="ck_mechanism_incubation_counts",
        ),
        sa.CheckConstraint(
            "maturity IN ('speculative', 'incubating', 'candidate') "
            "AND ((maturity = 'candidate' AND candidate_since IS NOT NULL) "
            "OR (maturity != 'candidate' AND candidate_since IS NULL))",
            name="ck_mechanism_incubation_maturity",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_mechanism_incubation_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("cluster_id"),
        sa.UniqueConstraint(
            "canonical_mechanism_hash",
            name="ux_mechanism_incubation_canonical_hash",
        ),
    )
    op.create_index(
        "ix_mechanism_incubation_maturity_signal",
        "mechanism_incubation",
        ["maturity", "last_signal_at", "cluster_id"],
        unique=False,
    )

    op.create_table(
        "idea_state",
        sa.Column("idea_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("mechanism_cluster_id", sa.String(length=64), nullable=False),
        sa.Column("owner_decision", sa.String(length=8), nullable=False),
        sa.Column("evaluations", sa.LargeBinary(), nullable=False),
        sa.Column("decision_reason", sa.LargeBinary(), nullable=True),
        sa.Column("resulting_experience_id", sa.String(length=36), nullable=True),
        sa.Column("resulting_version_id", sa.String(length=36), nullable=True),
        sa.Column("last_signal_at", sa.String(length=27), nullable=False),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "owner_decision IN ('active', 'adopted', 'rejected', 'archived')",
            name="ck_idea_state_decision",
        ),
        sa.CheckConstraint(
            _json_array_check("evaluations", nonempty=False),
            name="ck_idea_state_evaluations",
        ),
        sa.CheckConstraint(
            "decision_reason IS NULL OR "
            "(length(decision_reason) > 0 "
            "AND json_valid(CAST(decision_reason AS TEXT)) "
            "AND json_type(CAST(decision_reason AS TEXT)) = 'object')",
            name="ck_idea_state_reason",
        ),
        sa.CheckConstraint(
            "(owner_decision = 'adopted' "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL) "
            "OR (owner_decision != 'adopted' "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL)",
            name="ck_idea_state_result",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_idea_state_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["inspiration_ideas.idea_id"],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["mechanism_cluster_id"],
            ["mechanism_incubation.cluster_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resulting_experience_id"],
            ["experiences.experience_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resulting_version_id"],
            ["experience_versions.version_id"],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("idea_id"),
    )
    op.create_index(
        "ix_idea_state_owner_decision",
        "idea_state",
        ["owner_agent_id", "owner_decision", "idea_id"],
        unique=False,
    )
    op.create_index(
        "ix_idea_state_cluster",
        "idea_state",
        ["mechanism_cluster_id", "idea_id"],
        unique=False,
    )
    op.create_index(
        "ix_idea_state_archive_due",
        "idea_state",
        ["owner_decision", "last_signal_at", "idea_id"],
        unique=False,
    )

    _create_immutable_triggers(
        "inspiration_runs",
        conflict_when="run_id = NEW.run_id",
    )
    _create_immutable_triggers(
        "inspiration_snapshot_items",
        conflict_when=(
            "snapshot_item_id = NEW.snapshot_item_id "
            "OR (run_id = NEW.run_id AND rank = NEW.rank) "
            "OR (run_id = NEW.run_id "
            "AND source_type = NEW.source_type "
            "AND source_id = NEW.source_id "
            "AND source_version_id = NEW.source_version_id)"
        ),
    )
    _create_immutable_triggers(
        "inspiration_ideas",
        conflict_when=(
            "idea_id = NEW.idea_id "
            "OR (run_id = NEW.run_id "
            "AND operator = NEW.operator "
            "AND ordinal = NEW.ordinal)"
        ),
    )
    _create_immutable_triggers(
        "idea_occurrences",
        conflict_when=(
            "occurrence_id = NEW.occurrence_id "
            "OR idea_id = NEW.idea_id "
            "OR (run_id = NEW.run_id "
            "AND mechanism_hash = NEW.mechanism_hash)"
        ),
    )
    _create_immutable_triggers(
        "idea_adoption_records",
        conflict_when=(
            "adoption_id = NEW.adoption_id "
            "OR (owner_agent_id = NEW.owner_agent_id "
            "AND idea_id = NEW.idea_id)"
        ),
    )


def _drop_immutable_triggers(table_name: str) -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_conflicting_insert")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_delete")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_update")


def _refuse_populated_inspiration_downgrade() -> None:
    connection = op.get_bind()
    for table_name in (
        "inspiration_runs",
        "inspiration_snapshot_items",
        "inspiration_ideas",
        "idea_occurrences",
        "idea_adoption_records",
    ):
        if (
            connection.execute(
                sa.text(f"SELECT 1 FROM {table_name} LIMIT 1")
            ).first()
            is not None
        ):
            raise RuntimeError(
                "Cannot downgrade while inspiration source or ledger data exists"
            )
    if (
        connection.execute(
            sa.text(
                "SELECT 1 FROM domain_events "
                "WHERE aggregate_type IN ('inspiration_run', 'idea') "
                "OR event_type LIKE 'inspiration.%' LIMIT 1"
            )
        ).first()
        is not None
    ):
        raise RuntimeError(
            "Cannot downgrade while inspiration source or ledger data exists"
        )


def downgrade() -> None:
    _refuse_populated_inspiration_downgrade()

    for table_name in reversed(
        (
            "inspiration_runs",
            "inspiration_snapshot_items",
            "inspiration_ideas",
            "idea_occurrences",
            "idea_adoption_records",
        )
    ):
        _drop_immutable_triggers(table_name)

    op.drop_table("idea_state")
    op.drop_table("mechanism_incubation")
    op.drop_table("inspiration_run_state")
    op.drop_table("idea_adoption_records")
    op.drop_table("idea_occurrences")
    op.drop_table("inspiration_ideas")
    op.drop_table("inspiration_snapshot_items")
    op.drop_table("inspiration_runs")
