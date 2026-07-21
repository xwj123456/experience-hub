"""Add replayable falsifiers to frozen inspiration evidence."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision: str = "0005_inspiration_falsifiers"
down_revision: str | None = "0004_inspiration"
branch_labels: str | None = None
depends_on: str | None = None

_TABLE_NAME = "inspiration_snapshot_items"
_COLUMN_ORDER = (
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
)
_LEGACY_COLUMN_ORDER = _COLUMN_ORDER[:-1]


def _json_array_check(column: str) -> str:
    return (
        f"length({column}) > 0 "
        f"AND json_valid(CAST({column} AS TEXT)) "
        f"AND json_type(CAST({column} AS TEXT)) = 'array'"
    )


def _add_falsifiers_column() -> None:
    op.add_column(
        _TABLE_NAME,
        sa.Column(
            "falsifiers",
            sa.LargeBinary(),
            sa.CheckConstraint(
                f"{_json_array_check('falsifiers')} "
                "AND json_array_length(CAST(falsifiers AS TEXT)) <= 32",
                name="ck_inspiration_snapshot_items_falsifiers",
            ),
            nullable=False,
            server_default=sa.text("X'5B5D'"),
        ),
    )


def _drop_immutable_triggers() -> None:
    for suffix in ("conflicting_insert", "delete", "update"):
        op.execute(f"DROP TRIGGER IF EXISTS {_TABLE_NAME}_reject_{suffix}")


def _create_immutable_triggers() -> None:
    op.execute(
        f"CREATE TRIGGER {_TABLE_NAME}_reject_update "
        f"BEFORE UPDATE ON {_TABLE_NAME} "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{_TABLE_NAME} rows are immutable'); "
        "END"
    )
    op.execute(
        f"CREATE TRIGGER {_TABLE_NAME}_reject_delete "
        f"BEFORE DELETE ON {_TABLE_NAME} "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{_TABLE_NAME} rows are immutable'); "
        "END"
    )
    op.execute(
        f"CREATE TRIGGER {_TABLE_NAME}_reject_conflicting_insert "
        f"BEFORE INSERT ON {_TABLE_NAME} "
        "WHEN EXISTS ("
        f"SELECT 1 FROM {_TABLE_NAME} "
        "WHERE snapshot_item_id = NEW.snapshot_item_id "
        "OR (run_id = NEW.run_id AND rank = NEW.rank) "
        "OR (run_id = NEW.run_id "
        "AND source_type = NEW.source_type "
        "AND source_id = NEW.source_id "
        "AND source_version_id = NEW.source_version_id)"
        ") "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{_TABLE_NAME} identity already exists'); "
        "END"
    )


def _normalize_pre_fix_schema() -> None:
    inspector = sa.inspect(op.get_bind())
    check_names = {
        str(check["name"])
        for check in inspector.get_check_constraints(_TABLE_NAME)
        if check["name"] is not None
    }
    _drop_immutable_triggers()
    with op.batch_alter_table(
        _TABLE_NAME,
        recreate="always",
        partial_reordering=[_COLUMN_ORDER],
    ) as batch:
        if "ck_inspiration_snapshot_items_arrays" in check_names:
            batch.drop_constraint(
                "ck_inspiration_snapshot_items_arrays",
                type_="check",
            )
        if "ck_inspiration_snapshot_items_falsifiers" in check_names:
            batch.drop_constraint(
                "ck_inspiration_snapshot_items_falsifiers",
                type_="check",
            )
        batch.alter_column(
            "falsifiers",
            existing_type=sa.LargeBinary(),
            existing_nullable=False,
            server_default=sa.text("X'5B5D'"),
        )
        batch.create_check_constraint(
            "ck_inspiration_snapshot_items_arrays",
            f"{_json_array_check('applicability')} "
            "AND json_array_length(CAST(applicability AS TEXT)) <= 32 "
            f"AND {_json_array_check('tags')} "
            "AND json_array_length(CAST(tags AS TEXT)) <= 32",
        )
        batch.create_check_constraint(
            "ck_inspiration_snapshot_items_falsifiers",
            f"{_json_array_check('falsifiers')} "
            "AND json_array_length(CAST(falsifiers AS TEXT)) <= 32",
        )
    _create_immutable_triggers()


def upgrade() -> None:
    """Backfill legacy snapshots with an explicit empty falsifier list."""
    if context.is_offline_mode():
        _add_falsifiers_column()
        return
    columns = {
        str(column["name"])
        for column in sa.inspect(op.get_bind()).get_columns(_TABLE_NAME)
    }
    if "falsifiers" in columns:
        # Normalize databases created while the unreleased 0004 briefly
        # carried this column under the same revision identifier.
        _normalize_pre_fix_schema()
        return
    _add_falsifiers_column()


def _refuse_populated_falsifier_downgrade() -> None:
    retained = op.get_bind().execute(
        sa.text("SELECT 1 FROM inspiration_snapshot_items LIMIT 1")
    ).first()
    if retained is not None:
        raise RuntimeError(
            "Cannot discard frozen inspiration falsifiers during downgrade"
        )


def _drop_falsifiers_column_online() -> None:
    check_names = {
        str(check["name"])
        for check in sa.inspect(op.get_bind()).get_check_constraints(_TABLE_NAME)
        if check["name"] is not None
    }
    _drop_immutable_triggers()
    with op.batch_alter_table(
        _TABLE_NAME,
        recreate="always",
        partial_reordering=[_LEGACY_COLUMN_ORDER],
    ) as batch:
        if "ck_inspiration_snapshot_items_falsifiers" in check_names:
            batch.drop_constraint(
                "ck_inspiration_snapshot_items_falsifiers",
                type_="check",
            )
        batch.drop_column("falsifiers")
    _create_immutable_triggers()


def downgrade() -> None:
    """Remove the additive frozen-falsifier field."""
    if context.is_offline_mode():
        raise RuntimeError(
            "offline downgrade cannot verify frozen inspiration data; "
            "run the downgrade online"
        )
    _refuse_populated_falsifier_downgrade()
    _drop_falsifiers_column_online()
