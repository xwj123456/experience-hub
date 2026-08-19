"""Add immutable trajectory capture and candidate quarantine storage."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision: str = "0006_capture_candidates"
down_revision: str | None = "0005_inspiration_falsifiers"
branch_labels: str | None = None
depends_on: str | None = None

_SOURCE_TABLES = (
    "trajectory_bundles",
    "trajectory_evidence",
    "experience_candidates",
    "candidate_adoptions",
)


def _sha256_check(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _canonical_json_check(column: str, *, json_type: str) -> str:
    return (
        f"length({column}) > 0 "
        f"AND json_valid(CAST({column} AS TEXT)) "
        f"AND json_type(CAST({column} AS TEXT)) = '{json_type}' "
        f"AND CAST({column} AS TEXT) = json(CAST({column} AS TEXT))"
    )


def _bounded_json_array_check(column: str, *, limit: int) -> str:
    return (
        f"{_canonical_json_check(column, json_type='array')} "
        f"AND json_array_length(CAST({column} AS TEXT)) <= {limit}"
    )


def _manifest_canonical_json(column: str) -> str:
    document = f"CAST({column} AS TEXT)"
    return (
        "json_object("
        "'adapter',json_object("
        f"'kind',json_extract({document},'$.adapter.kind'),"
        f"'version',json_extract({document},'$.adapter.version')),"
        f"'owner_agent_id',json_extract({document},'$.owner_agent_id'),"
        "'sanitization',json_object("
        "'input_sanitized',json(CASE "
        f"json_type({document},'$.sanitization.input_sanitized') "
        "WHEN 'true' THEN 'true' WHEN 'false' THEN 'false' END),"
        f"'profile_id',json_extract({document},'$.sanitization.profile_id')),"
        f"'schema_version',json_extract({document},'$.schema_version'),"
        "'source_completed_at',"
        f"json_extract({document},'$.source_completed_at'),"
        "'source_started_at',"
        f"json_extract({document},'$.source_started_at'),"
        "'steps',json((SELECT json_group_array(json(canonical_step)) FROM ("
        "SELECT json_object("
        "'action_hash',json_extract(value,'$.action_hash'),"
        "'candidate_signal_hash',"
        "json_extract(value,'$.candidate_signal_hash'),"
        "'observation_hash',json_extract(value,'$.observation_hash'),"
        "'occurred_at',json_extract(value,'$.occurred_at'),"
        "'ordinal',json_extract(value,'$.ordinal'),"
        "'outcome_hash',json_extract(value,'$.outcome_hash'),"
        "'status',json_extract(value,'$.status'),"
        "'step_id',json_extract(value,'$.step_id')) AS canonical_step "
        f"FROM json_each({document},'$.steps') "
        "ORDER BY CAST(key AS INTEGER)))),"
        f"'trajectory_id',json_extract({document},'$.trajectory_id'))"
    )


def _evidence_canonical_json(column: str) -> str:
    document = f"CAST({column} AS TEXT)"
    return (
        "(SELECT json_group_array(json(canonical_item)) FROM ("
        "SELECT json_object("
        "'id',json_extract(value,'$.id'),"
        "'type',json_extract(value,'$.type')) AS canonical_item "
        f"FROM json_each({document}) ORDER BY CAST(key AS INTEGER)))"
    )


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


def _drop_immutable_triggers(table_name: str) -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_conflicting_insert")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_delete")
    op.execute(f"DROP TRIGGER IF EXISTS {table_name}_reject_update")


def _experiences_table(*, expanded: bool) -> sa.Table:
    metadata = sa.MetaData()
    sa.Table(
        "agents",
        metadata,
        sa.Column("agent_id", sa.String(length=36), primary_key=True),
    )
    origin_check = (
        "origin IN ('local','adopted_capsule','adopted_idea',"
        "'adopted_candidate')"
        if expanded
        else "origin IN ('local', 'adopted_capsule', 'adopted_idea')"
    )
    return sa.Table(
        "experiences",
        metadata,
        sa.Column("experience_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column(
            "origin",
            sa.String(length=17 if expanded else 15),
            nullable=False,
        ),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis')",
            name="ck_experiences_kind",
        ),
        sa.CheckConstraint(origin_check, name="ck_experiences_origin"),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("experience_id"),
    )


def _rebuild_experiences_origin(*, expanded: bool) -> None:
    _drop_immutable_triggers("experiences")
    if not expanded:
        op.drop_index("ux_experiences_id_owner", table_name="experiences")
    op.drop_index("ix_experiences_owner_created", table_name="experiences")
    target_length = 17 if expanded else 15
    allowed_origins = (
        "origin IN ('local','adopted_capsule','adopted_idea',"
        "'adopted_candidate')"
        if expanded
        else "origin IN ('local', 'adopted_capsule', 'adopted_idea')"
    )
    with op.batch_alter_table(
        "experiences",
        recreate="always",
        copy_from=_experiences_table(expanded=not expanded),
    ) as batch:
        batch.alter_column(
            "origin",
            existing_type=sa.String(length=15 if expanded else 17),
            type_=sa.String(length=target_length),
            existing_nullable=False,
        )
        batch.drop_constraint("ck_experiences_origin", type_="check")
        batch.create_check_constraint(
            "ck_experiences_origin",
            allowed_origins,
        )
    op.create_index(
        "ix_experiences_owner_created",
        "experiences",
        ["owner_agent_id", "created_at", "experience_id"],
        unique=False,
    )
    if expanded:
        op.create_index(
            "ux_experiences_id_owner",
            "experiences",
            ["experience_id", "owner_agent_id"],
            unique=True,
        )
    _create_immutable_triggers(
        "experiences",
        conflict_when="experience_id = NEW.experience_id",
        conflict_message="experiences identity already exists",
    )


def _create_capture_tables() -> None:
    op.create_table(
        "trajectory_bundles",
        sa.Column("bundle_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("trajectory_id", sa.String(), nullable=False),
        sa.Column("adapter_kind", sa.String(length=13), nullable=False),
        sa.Column("adapter_version", sa.Integer(), nullable=False),
        sa.Column("sanitization_profile", sa.String(), nullable=False),
        sa.Column("manifest", sa.LargeBinary(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("source_started_at", sa.String(length=27), nullable=False),
        sa.Column("source_completed_at", sa.String(length=27), nullable=False),
        sa.Column("captured_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "length(trim(trajectory_id)) > 0",
            name="ck_trajectory_bundles_trajectory_id",
        ),
        sa.CheckConstraint(
            "adapter_kind = 'generic_jsonl' AND adapter_version = 1",
            name="ck_trajectory_bundles_adapter",
        ),
        sa.CheckConstraint(
            "length(trim(sanitization_profile)) > 0",
            name="ck_trajectory_bundles_sanitization_profile",
        ),
        sa.CheckConstraint(
            _canonical_json_check("manifest", json_type="object"),
            name="ck_trajectory_bundles_manifest",
        ),
        sa.CheckConstraint(
            _sha256_check("manifest_hash"),
            name="ck_trajectory_bundles_manifest_hash",
        ),
        sa.CheckConstraint(
            "source_started_at <= source_completed_at",
            name="ck_trajectory_bundles_source_time",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("bundle_id"),
    )
    op.create_index(
        "ux_trajectory_bundles_owner_manifest",
        "trajectory_bundles",
        ["owner_agent_id", "manifest_hash"],
        unique=True,
    )
    op.create_index(
        "ux_trajectory_bundles_id_owner",
        "trajectory_bundles",
        ["bundle_id", "owner_agent_id"],
        unique=True,
    )
    op.create_index(
        "ix_trajectory_bundles_owner_trajectory",
        "trajectory_bundles",
        ["owner_agent_id", "trajectory_id", "captured_at", "bundle_id"],
        unique=False,
    )

    op.create_table(
        "trajectory_evidence",
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("bundle_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("field", sa.String(length=11), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("excerpt", sa.String(), nullable=False),
        sa.Column("excerpt_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "length(trim(step_id)) > 0",
            name="ck_trajectory_evidence_step_id",
        ),
        sa.CheckConstraint(
            "field IN ('observation', 'action', 'outcome')",
            name="ck_trajectory_evidence_field",
        ),
        sa.CheckConstraint(
            "ordinal > 0",
            name="ck_trajectory_evidence_ordinal",
        ),
        sa.CheckConstraint(
            "length(CAST(excerpt AS BLOB)) <= 512",
            name="ck_trajectory_evidence_excerpt",
        ),
        sa.CheckConstraint(
            _sha256_check("excerpt_hash"),
            name="ck_trajectory_evidence_excerpt_hash",
        ),
        sa.ForeignKeyConstraint(
            ["bundle_id", "owner_agent_id"],
            [
                "trajectory_bundles.bundle_id",
                "trajectory_bundles.owner_agent_id",
            ],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index(
        "ux_trajectory_evidence_bundle_step_field",
        "trajectory_evidence",
        ["bundle_id", "step_id", "field"],
        unique=True,
    )
    op.create_index(
        "ix_trajectory_evidence_bundle_ordinal",
        "trajectory_evidence",
        ["bundle_id", "ordinal", "field", "evidence_id"],
        unique=False,
    )

    op.create_table(
        "experience_candidates",
        sa.Column("candidate_id", sa.String(length=36), nullable=False),
        sa.Column("bundle_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("candidate_ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("body", sa.String(), nullable=False),
        sa.Column("summary", sa.String(length=1000), nullable=False),
        sa.Column("mechanism", sa.String(length=2000), nullable=False),
        sa.Column("tags", sa.LargeBinary(), nullable=False),
        sa.Column("applicability", sa.LargeBinary(), nullable=False),
        sa.Column("evidence", sa.LargeBinary(), nullable=False),
        sa.Column("evidence_refs", sa.LargeBinary(), nullable=False),
        sa.Column("falsifiers", sa.LargeBinary(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("extractor_kind", sa.String(length=23), nullable=False),
        sa.Column(
            "extractor_configuration_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("created_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            "candidate_ordinal > 0",
            name="ck_experience_candidates_ordinal",
        ),
        sa.CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis')",
            name="ck_experience_candidates_kind",
        ),
        sa.CheckConstraint(
            "length(CAST(body AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(trim(body)) > 0",
            name="ck_experience_candidates_body",
        ),
        sa.CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0",
            name="ck_experience_candidates_summary",
        ),
        sa.CheckConstraint(
            "length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_candidates_mechanism",
        ),
        sa.CheckConstraint(
            f"{_bounded_json_array_check('tags', limit=32)} "
            f"AND {_bounded_json_array_check('applicability', limit=32)} "
            f"AND {_bounded_json_array_check('falsifiers', limit=32)}",
            name="ck_experience_candidates_content_arrays",
        ),
        sa.CheckConstraint(
            f"{_bounded_json_array_check('evidence', limit=8)} "
            f"AND {_bounded_json_array_check('evidence_refs', limit=8)} "
            "AND json_array_length(CAST(evidence AS TEXT)) "
            "= json_array_length(CAST(evidence_refs AS TEXT))",
            name="ck_experience_candidates_evidence",
        ),
        sa.CheckConstraint(
            f"{_sha256_check('content_hash')} "
            f"AND {_sha256_check('extractor_configuration_hash')}",
            name="ck_experience_candidates_hashes",
        ),
        sa.CheckConstraint(
            "extractor_kind = 'deterministic_signal_v1'",
            name="ck_experience_candidates_extractor",
        ),
        sa.ForeignKeyConstraint(
            ["bundle_id", "owner_agent_id"],
            [
                "trajectory_bundles.bundle_id",
                "trajectory_bundles.owner_agent_id",
            ],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.PrimaryKeyConstraint("candidate_id"),
    )
    op.create_index(
        "ux_experience_candidates_bundle_ordinal",
        "experience_candidates",
        ["bundle_id", "candidate_ordinal"],
        unique=True,
    )
    op.create_index(
        "ix_experience_candidates_owner_created",
        "experience_candidates",
        ["owner_agent_id", "created_at", "candidate_id"],
        unique=False,
    )
    op.create_index(
        "ux_experience_candidates_id_owner",
        "experience_candidates",
        ["candidate_id", "owner_agent_id"],
        unique=True,
    )

    op.create_table(
        "candidate_adoptions",
        sa.Column("adoption_id", sa.String(length=36), nullable=False),
        sa.Column("candidate_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("resulting_experience_id", sa.String(length=36), nullable=False),
        sa.Column("resulting_version_id", sa.String(length=36), nullable=False),
        sa.Column("resulting_content_hash", sa.String(length=64), nullable=False),
        sa.Column("created", sa.Boolean(), nullable=False),
        sa.Column("adopted_at", sa.String(length=27), nullable=False),
        sa.CheckConstraint(
            _sha256_check("resulting_content_hash"),
            name="ck_candidate_adoptions_content_hash",
        ),
        sa.CheckConstraint(
            "created IN (0, 1)",
            name="ck_candidate_adoptions_created",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id", "owner_agent_id"],
            [
                "experience_candidates.candidate_id",
                "experience_candidates.owner_agent_id",
            ],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["resulting_experience_id", "owner_agent_id"],
            ["experiences.experience_id", "experiences.owner_agent_id"],
        ),
        sa.ForeignKeyConstraint(
            [
                "resulting_version_id",
                "resulting_experience_id",
                "resulting_content_hash",
            ],
            [
                "experience_versions.version_id",
                "experience_versions.experience_id",
                "experience_versions.content_hash",
            ],
        ),
        sa.PrimaryKeyConstraint("adoption_id"),
    )
    op.create_index(
        "ux_candidate_adoptions_candidate",
        "candidate_adoptions",
        ["candidate_id"],
        unique=True,
    )
    op.create_index(
        "ix_candidate_adoptions_resulting_experience",
        "candidate_adoptions",
        ["resulting_experience_id", "adoption_id"],
        unique=False,
    )
    op.create_index(
        "ux_candidate_adoptions_lineage",
        "candidate_adoptions",
        [
            "adoption_id",
            "candidate_id",
            "owner_agent_id",
            "resulting_experience_id",
            "resulting_version_id",
        ],
        unique=True,
    )


def _create_candidate_state() -> None:
    op.create_table(
        "candidate_state",
        sa.Column("candidate_id", sa.String(length=36), nullable=False),
        sa.Column("owner_agent_id", sa.String(length=36), nullable=False),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("adoption_id", sa.String(length=36), nullable=True),
        sa.Column("resulting_experience_id", sa.String(length=36), nullable=True),
        sa.Column("resulting_version_id", sa.String(length=36), nullable=True),
        sa.Column("reason_code", sa.String(), nullable=True),
        sa.Column("reason_text", sa.String(), nullable=True),
        sa.Column("reason_text_hash", sa.String(length=64), nullable=True),
        sa.Column("decided_at", sa.String(length=27), nullable=True),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "decision IN ('pending', 'adopted', 'rejected')",
            name="ck_candidate_state_decision",
        ),
        sa.CheckConstraint(
            "reason_text_hash IS NULL OR "
            f"({_sha256_check('reason_text_hash')})",
            name="ck_candidate_state_reason_hash",
        ),
        sa.CheckConstraint(
            "(decision = 'pending' "
            "AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL "
            "AND reason_code IS NULL "
            "AND reason_text IS NULL "
            "AND reason_text_hash IS NULL "
            "AND decided_at IS NULL) "
            "OR (decision = 'adopted' "
            "AND adoption_id IS NOT NULL "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL "
            "AND reason_code IS NULL "
            "AND reason_text IS NULL "
            "AND reason_text_hash IS NULL "
            "AND decided_at IS NOT NULL) "
            "OR (decision = 'rejected' "
            "AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL "
            "AND reason_code IS NOT NULL "
            "AND reason_text IS NOT NULL "
            "AND length(trim(reason_code)) > 0 "
            "AND length(trim(reason_text)) > 0 "
            "AND reason_text_hash IS NOT NULL "
            "AND decided_at IS NOT NULL)",
            name="ck_candidate_state_shape",
        ),
        sa.CheckConstraint(
            "projection_event_id > 0",
            name="ck_candidate_state_projection_event",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id", "owner_agent_id"],
            [
                "experience_candidates.candidate_id",
                "experience_candidates.owner_agent_id",
            ],
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            [
                "adoption_id",
                "candidate_id",
                "owner_agent_id",
                "resulting_experience_id",
                "resulting_version_id",
            ],
            [
                "candidate_adoptions.adoption_id",
                "candidate_adoptions.candidate_id",
                "candidate_adoptions.owner_agent_id",
                "candidate_adoptions.resulting_experience_id",
                "candidate_adoptions.resulting_version_id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["projection_event_id"],
            ["domain_events.event_id"],
        ),
        sa.PrimaryKeyConstraint("candidate_id"),
    )
    op.create_index(
        "ix_candidate_state_owner_decision",
        "candidate_state",
        ["owner_agent_id", "decision", "candidate_id"],
        unique=False,
    )


def _install_capture_triggers() -> None:
    _create_immutable_triggers(
        "trajectory_bundles",
        conflict_when=(
            "bundle_id = NEW.bundle_id "
            "OR (owner_agent_id = NEW.owner_agent_id "
            "AND manifest_hash = NEW.manifest_hash)"
        ),
        conflict_message="trajectory_bundles identity already exists",
    )
    _create_immutable_triggers(
        "trajectory_evidence",
        conflict_when=(
            "evidence_id = NEW.evidence_id "
            "OR (bundle_id = NEW.bundle_id "
            "AND step_id = NEW.step_id AND field = NEW.field)"
        ),
        conflict_message="trajectory_evidence identity already exists",
    )
    _create_immutable_triggers(
        "experience_candidates",
        conflict_when=(
            "candidate_id = NEW.candidate_id "
            "OR (bundle_id = NEW.bundle_id "
            "AND candidate_ordinal = NEW.candidate_ordinal)"
        ),
        conflict_message="experience_candidates identity already exists",
    )
    _create_immutable_triggers(
        "candidate_adoptions",
        conflict_when=(
            "adoption_id = NEW.adoption_id OR candidate_id = NEW.candidate_id"
        ),
        conflict_message="candidate_adoptions identity already exists",
    )


def _install_canonical_json_triggers() -> None:
    op.execute(
        "CREATE TRIGGER trajectory_bundles_reject_noncanonical_manifest "
        "BEFORE INSERT ON trajectory_bundles "
        "WHEN COALESCE((SELECT group_concat(key, ',') "
        "FROM json_each(CAST(NEW.manifest AS TEXT))), '') <> "
        "'adapter,owner_agent_id,sanitization,schema_version,"
        "source_completed_at,source_started_at,steps,trajectory_id' "
        "OR COALESCE(json_type(CAST(NEW.manifest AS TEXT), '$.adapter'), '') "
        "<> 'object' "
        "OR COALESCE((SELECT group_concat(key, ',') "
        "FROM json_each(CAST(NEW.manifest AS TEXT), '$.adapter')), '') "
        "<> 'kind,version' "
        "OR COALESCE(json_type(CAST(NEW.manifest AS TEXT), '$.sanitization'), '') "
        "<> 'object' "
        "OR COALESCE((SELECT group_concat(key, ',') "
        "FROM json_each(CAST(NEW.manifest AS TEXT), '$.sanitization')), '') "
        "<> 'input_sanitized,profile_id' "
        "OR COALESCE(json_type(CAST(NEW.manifest AS TEXT), '$.steps'), '') "
        "<> 'array' "
        "OR EXISTS ("
        "SELECT 1 FROM json_each(CAST(NEW.manifest AS TEXT), '$.steps') "
        "AS manifest_step "
        "WHERE COALESCE(json_type(manifest_step.value), '') <> 'object' "
        "OR COALESCE((SELECT group_concat(step_key.key, ',') "
        "FROM json_each(manifest_step.value) AS step_key), '') <> "
        "'action_hash,candidate_signal_hash,observation_hash,occurred_at,"
        "ordinal,outcome_hash,status,step_id'"
        ") "
        "OR COALESCE(CAST(NEW.manifest AS TEXT) <> "
        f"{_manifest_canonical_json('NEW.manifest')}, 1) "
        "BEGIN "
        "SELECT RAISE(ABORT, "
        "'trajectory_bundles manifest must use canonical object keys'); "
        "END"
    )
    op.execute(
        "CREATE TRIGGER experience_candidates_reject_noncanonical_evidence "
        "BEFORE INSERT ON experience_candidates "
        "WHEN COALESCE(json_type(CAST(NEW.evidence AS TEXT)), '') <> 'array' "
        "OR EXISTS ("
        "SELECT 1 FROM json_each(CAST(NEW.evidence AS TEXT)) AS evidence_item "
        "WHERE COALESCE(json_type(evidence_item.value), '') <> 'object' "
        "OR COALESCE((SELECT group_concat(evidence_key.key, ',') "
        "FROM json_each(evidence_item.value) AS evidence_key), '') <> 'id,type'"
        ") "
        "OR COALESCE(CAST(NEW.evidence AS TEXT) <> "
        f"{_evidence_canonical_json('NEW.evidence')}, 1) "
        "BEGIN "
        "SELECT RAISE(ABORT, "
        "'experience_candidates evidence must use canonical object keys'); "
        "END"
    )


def upgrade() -> None:
    _rebuild_experiences_origin(expanded=True)
    op.create_index(
        "ux_experience_versions_id_experience_content",
        "experience_versions",
        ["version_id", "experience_id", "content_hash"],
        unique=True,
    )
    _create_capture_tables()
    _create_candidate_state()
    _install_capture_triggers()
    _install_canonical_json_triggers()


def _refuse_populated_candidate_downgrade() -> None:
    connection = op.get_bind()
    for table_name in _SOURCE_TABLES:
        retained = connection.execute(
            sa.text(f"SELECT 1 FROM {table_name} LIMIT 1")
        ).first()
        if retained is not None:
            raise RuntimeError(
                "Cannot downgrade while candidate source or ledger data exists"
            )
    retained_event = connection.execute(
        sa.text(
            "SELECT 1 FROM domain_events "
            "WHERE aggregate_type IN "
            "('candidate', 'experience_candidate', 'trajectory_bundle') "
            "OR event_type LIKE 'candidate.%' "
            "OR event_type = 'trajectory.captured' LIMIT 1"
        )
    ).first()
    retained_experience = connection.execute(
        sa.text(
            "SELECT 1 FROM experiences "
            "WHERE origin = 'adopted_candidate' LIMIT 1"
        )
    ).first()
    if retained_event is not None or retained_experience is not None:
        raise RuntimeError(
            "Cannot downgrade while candidate source or ledger data exists"
        )


def downgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError(
            "offline downgrade cannot verify candidate data; "
            "run the downgrade online"
        )
    _refuse_populated_candidate_downgrade()
    for table_name in reversed(_SOURCE_TABLES):
        _drop_immutable_triggers(table_name)
    op.drop_table("candidate_state")
    op.drop_table("candidate_adoptions")
    op.drop_table("experience_candidates")
    op.drop_table("trajectory_evidence")
    op.drop_table("trajectory_bundles")
    op.drop_index(
        "ux_experience_versions_id_experience_content",
        table_name="experience_versions",
    )
    _rebuild_experiences_origin(expanded=False)
