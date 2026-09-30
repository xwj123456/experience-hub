"""Add immutable owner-bound Passport imports, adoptions and decision state."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision: str = "0008_evidence_passports"
down_revision: str | None = "0007_capture_evidence_hashes"
branch_labels: str | None = None
depends_on: str | None = None

_OLD_ORIGIN = "origin IN ('local','adopted_capsule','adopted_idea','adopted_candidate')"
_NEW_ORIGIN = (
    "origin IN ('local','adopted_capsule','adopted_idea',"
    "'adopted_candidate','adopted_passport')"
)


def _sha256(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _utc_check(column: str) -> str:
    pattern = (
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T"
        "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]."
        "[0-9][0-9][0-9][0-9][0-9][0-9]Z"
    )
    return (
        f"typeof({column}) = 'text' AND length({column}) = 27 "
        f"AND {column} GLOB '{pattern}' AND substr({column},1,4) != '0000' "
        f"AND COALESCE(strftime('%Y-%m-%dT%H:%M:%S',substr({column},1,19),"
        f"'+0 seconds') = substr({column},1,19),0)"
    )


def _guards(table: str, conflict: str) -> None:
    for operation in ("UPDATE", "DELETE"):
        op.execute(
            f"CREATE TRIGGER {table}_reject_{operation.lower()} "
            f"BEFORE {operation} ON {table} BEGIN "
            f"SELECT RAISE(ABORT,'{table} rows are immutable'); END"
        )
    op.execute(
        f"CREATE TRIGGER {table}_reject_conflicting_insert "
        f"BEFORE INSERT ON {table} "
        f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {conflict}) "
        f"BEGIN SELECT RAISE(ABORT,'{table} identity already exists'); END"
    )


def _rebuild_origin(*, expanded: bool) -> None:
    # Keep exact index SQL too: reflected expression indexes are not portable
    # through a SQLite batch rebuild. Implicit constraint indexes remain intact.
    connection = op.get_bind()
    retained = tuple(
        connection.execute(
            sa.text(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name='experiences' ORDER BY name"
            )
        )
    )
    indexes = tuple(
        connection.execute(
            sa.text(
                "SELECT name,sql FROM sqlite_master WHERE type='index' "
                "AND tbl_name='experiences' AND sql IS NOT NULL ORDER BY name"
            )
        )
    )
    for name, _ in retained:
        op.execute(sa.text(f'DROP TRIGGER "{str(name).replace(chr(34), chr(34) * 2)}"'))
    for name, _ in indexes:
        op.execute(sa.text(f'DROP INDEX "{str(name).replace(chr(34), chr(34) * 2)}"'))
    with op.batch_alter_table("experiences", recreate="always") as batch:
        batch.drop_constraint("ck_experiences_origin", type_="check")
        batch.create_check_constraint(
            "ck_experiences_origin",
            _NEW_ORIGIN if expanded else _OLD_ORIGIN,
        )
    for _, sql in (*indexes, *retained):
        op.execute(sa.text(str(sql)))


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("Passport migration must preserve live SQLite schema online")
    _rebuild_origin(expanded=True)
    op.create_table(
        "passport_imports",
        sa.Column("import_id", sa.String(36), primary_key=True),
        sa.Column("owner_agent_id", sa.String(36), nullable=False),
        sa.Column("passport_hash", sa.String(64), nullable=False),
        sa.Column("canonical_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("imported_at", sa.String(27), nullable=False),
        sa.CheckConstraint(_utc_check("imported_at"), name="ck_passport_imports_time"),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.CheckConstraint(_sha256("passport_hash"), name="ck_passport_imports_hash"),
        sa.CheckConstraint(
            "typeof(canonical_bytes) = 'blob' AND "
            "length(canonical_bytes) BETWEEN 1 AND 524288 AND "
            "CASE WHEN json_valid(CAST(canonical_bytes AS TEXT)) THEN "
            "json_type(CAST(canonical_bytes AS TEXT)) = 'object' AND "
            "CAST(canonical_bytes AS TEXT) = json(CAST(canonical_bytes AS TEXT)) "
            "ELSE 0 END",
            name="ck_passport_imports_bytes",
        ),
    )
    op.create_index(
        "ux_passport_imports_owner_hash",
        "passport_imports",
        ["owner_agent_id", "passport_hash"],
        unique=True,
    )
    op.create_index(
        "ux_passport_imports_id_owner",
        "passport_imports",
        ["import_id", "owner_agent_id"],
        unique=True,
    )
    op.create_index(
        "ix_passport_imports_owner_time",
        "passport_imports",
        ["owner_agent_id", "imported_at", "import_id"],
    )
    op.create_table(
        "passport_adoptions",
        sa.Column("adoption_id", sa.String(36), primary_key=True),
        sa.Column("import_id", sa.String(36), nullable=False),
        sa.Column("owner_agent_id", sa.String(36), nullable=False),
        sa.Column("resulting_experience_id", sa.String(36), nullable=False),
        sa.Column("resulting_version_id", sa.String(36), nullable=False),
        sa.Column("resulting_content_hash", sa.String(64), nullable=False),
        sa.Column("created", sa.Boolean(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("adopted_at", sa.String(27), nullable=False),
        sa.CheckConstraint(_utc_check("adopted_at"), name="ck_passport_adoptions_time"),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(
            ["import_id", "owner_agent_id"],
            ["passport_imports.import_id", "passport_imports.owner_agent_id"],
        ),
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
        sa.CheckConstraint(
            _sha256("resulting_content_hash"), name="ck_passport_adoptions_hash"
        ),
        sa.CheckConstraint("created IN (0,1)", name="ck_passport_adoptions_created"),
        sa.CheckConstraint(
            "importance >= 0 AND importance <= 1",
            name="ck_passport_adoptions_importance",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_passport_adoptions_confidence",
        ),
    )
    op.create_index(
        "ux_passport_adoptions_import", "passport_adoptions", ["import_id"], unique=True
    )
    op.create_index(
        "ix_passport_adoptions_result",
        "passport_adoptions",
        ["resulting_experience_id", "adoption_id"],
    )
    op.create_index(
        "ux_passport_adoptions_lineage",
        "passport_adoptions",
        [
            "adoption_id",
            "import_id",
            "owner_agent_id",
            "resulting_experience_id",
            "resulting_version_id",
        ],
        unique=True,
    )
    op.create_table(
        "passport_state",
        sa.Column("import_id", sa.String(36), primary_key=True),
        sa.Column("owner_agent_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(8), nullable=False),
        sa.Column("adoption_id", sa.String(36)),
        sa.Column("resulting_experience_id", sa.String(36)),
        sa.Column("resulting_version_id", sa.String(36)),
        sa.Column("reason_code", sa.String()),
        sa.Column("reason_text", sa.String()),
        sa.Column("reason_text_hash", sa.String(64)),
        sa.Column("decided_at", sa.String(27)),
        sa.Column("projection_event_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "decided_at IS NULL OR (" + _utc_check("decided_at") + ")",
            name="ck_passport_state_time",
        ),
        sa.ForeignKeyConstraint(["owner_agent_id"], ["agents.agent_id"]),
        sa.ForeignKeyConstraint(["projection_event_id"], ["domain_events.event_id"]),
        sa.ForeignKeyConstraint(
            ["import_id", "owner_agent_id"],
            ["passport_imports.import_id", "passport_imports.owner_agent_id"],
        ),
        sa.ForeignKeyConstraint(
            [
                "adoption_id",
                "import_id",
                "owner_agent_id",
                "resulting_experience_id",
                "resulting_version_id",
            ],
            [
                "passport_adoptions.adoption_id",
                "passport_adoptions.import_id",
                "passport_adoptions.owner_agent_id",
                "passport_adoptions.resulting_experience_id",
                "passport_adoptions.resulting_version_id",
            ],
        ),
        sa.CheckConstraint(
            "state IN ('pending','adopted','rejected')", name="ck_passport_state_state"
        ),
        sa.CheckConstraint(
            "projection_event_id > 0", name="ck_passport_state_projection_event"
        ),
        sa.CheckConstraint(
            "reason_text_hash IS NULL OR " + _sha256("reason_text_hash"),
            name="ck_passport_state_reason_hash",
        ),
        sa.CheckConstraint(
            "(state='pending' AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NULL) "
            "OR (state='adopted' AND adoption_id IS NOT NULL "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NOT NULL) "
            "OR (state='rejected' AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
            "AND reason_code IS NOT NULL AND length(trim(reason_code)) > 0 "
            "AND reason_text IS NOT NULL AND length(trim(reason_text)) > 0 "
            "AND reason_text_hash IS NOT NULL AND decided_at IS NOT NULL)",
            name="ck_passport_state_shape",
        ),
    )
    op.create_index(
        "ix_passport_state_owner_state",
        "passport_state",
        ["owner_agent_id", "state", "import_id"],
    )
    _guards(
        "passport_imports",
        "import_id=NEW.import_id OR "
        "(owner_agent_id=NEW.owner_agent_id AND passport_hash=NEW.passport_hash)",
    )
    _guards(
        "passport_adoptions", "adoption_id=NEW.adoption_id OR import_id=NEW.import_id"
    )
    op.execute(
        "CREATE TRIGGER passport_imports_reject_noncanonical_bytes "
        "BEFORE INSERT ON passport_imports WHEN CASE "
        "WHEN json_valid(CAST(NEW.canonical_bytes AS TEXT)) THEN EXISTS ("
        "WITH nodes AS MATERIALIZED (SELECT id,parent,key,type FROM "
        "json_tree(CAST(NEW.canonical_bytes AS TEXT))) "
        "SELECT 1 FROM (SELECT a.key AS object_key, "
        "lag(a.key) OVER(PARTITION BY a.parent ORDER BY a.id) AS previous_key "
        "FROM nodes a JOIN nodes p ON p.id=a.parent WHERE p.type='object') "
        "WHERE object_key<=previous_key COLLATE BINARY"
        ") ELSE 1 END BEGIN SELECT RAISE(ABORT,"
        "'passport_imports bytes must use canonical object keys'); END"
    )


def downgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("Offline downgrade cannot verify retained Passport data")
    connection = op.get_bind()
    for table in ("passport_imports", "passport_adoptions", "passport_state"):
        if connection.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1")).first():
            raise RuntimeError(
                "Cannot downgrade while Passport source or ledger data exists"
            )
    if (
        connection.execute(
            sa.text(
                "SELECT 1 FROM domain_events WHERE aggregate_type='passport_import' "
                "OR event_type LIKE 'passport.%' LIMIT 1"
            )
        ).first()
        or connection.execute(
            sa.text("SELECT 1 FROM experiences WHERE origin='adopted_passport' LIMIT 1")
        ).first()
    ):
        raise RuntimeError(
            "Cannot downgrade while Passport source or ledger data exists"
        )
    for table in ("passport_state", "passport_adoptions", "passport_imports"):
        op.drop_table(table)
    _rebuild_origin(expanded=False)
