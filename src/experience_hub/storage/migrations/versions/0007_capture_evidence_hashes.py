"""Split full source and retained excerpt evidence hashes."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from alembic import context, op

revision: str = "0007_capture_evidence_hashes"
down_revision: str | None = "0006_capture_candidates"
branch_labels: str | None = None
depends_on: str | None = None

_TABLE = "trajectory_evidence"
_MANIFEST_KEYS = {
    "adapter",
    "owner_agent_id",
    "sanitization",
    "schema_version",
    "source_completed_at",
    "source_started_at",
    "steps",
    "trajectory_id",
}
_STEP_KEYS = {
    "action_hash",
    "candidate_signal_hash",
    "observation_hash",
    "occurred_at",
    "ordinal",
    "outcome_hash",
    "status",
    "step_id",
}
_EVIDENCE_FIELDS = {"action", "observation", "outcome"}
_MAX_EXCERPT_UTF8_BYTES = 512
_MAX_TRAJECTORY_STEPS = 2_000


def _sha256_check(column: str) -> str:
    return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_uuid_text(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Stored trajectory identity cannot be migrated safely")
    try:
        identifier = UUID(value)
    except ValueError:
        raise RuntimeError(
            "Stored trajectory identity cannot be migrated safely"
        ) from None
    if str(identifier) != value:
        raise RuntimeError("Stored trajectory identity cannot be migrated safely")
    return value


def _nonblank_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Stored trajectory text cannot be migrated safely")
    return value


def _manifest_steps(row: sa.RowMapping) -> dict[tuple[str, int], dict[str, Any]]:
    raw_manifest = row["manifest"]
    if not isinstance(raw_manifest, bytes):
        raise RuntimeError("Stored trajectory manifest cannot be migrated safely")
    try:
        document: Any = json.loads(raw_manifest)
        canonical_manifest = _canonical_json_bytes(document)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError(
            "Stored trajectory manifest cannot be migrated safely"
        ) from None
    adapter = document.get("adapter") if isinstance(document, dict) else None
    sanitization = (
        document.get("sanitization") if isinstance(document, dict) else None
    )
    steps = document.get("steps") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != _MANIFEST_KEYS
        or canonical_manifest != raw_manifest
        or not _is_sha256(row["manifest_hash"])
        or sha256(raw_manifest).hexdigest() != row["manifest_hash"]
        or not isinstance(adapter, dict)
        or set(adapter) != {"kind", "version"}
        or not isinstance(adapter["kind"], str)
        or adapter["kind"] != row["adapter_kind"]
        or type(adapter["version"]) is not int
        or adapter["version"] != row["adapter_version"]
        or not isinstance(document["owner_agent_id"], str)
        or document["owner_agent_id"] != row["bundle_owner_agent_id"]
        or not isinstance(sanitization, dict)
        or set(sanitization) != {"input_sanitized", "profile_id"}
        or sanitization["input_sanitized"] is not True
        or not isinstance(sanitization["profile_id"], str)
        or sanitization["profile_id"] != row["sanitization_profile"]
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["source_started_at"], str)
        or document["source_started_at"] != row["source_started_at"]
        or not isinstance(document["source_completed_at"], str)
        or document["source_completed_at"] != row["source_completed_at"]
        or not isinstance(document["trajectory_id"], str)
        or document["trajectory_id"] != row["trajectory_id"]
        or not isinstance(steps, list)
        or not 1 <= len(steps) <= _MAX_TRAJECTORY_STEPS
    ):
        raise RuntimeError("Stored trajectory manifest cannot be migrated safely")

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    step_ids: set[str] = set()
    for expected_ordinal, step in enumerate(steps, start=1):
        if (
            not isinstance(step, dict)
            or set(step) != _STEP_KEYS
            or type(step["ordinal"]) is not int
            or step["ordinal"] != expected_ordinal
            or not isinstance(step["step_id"], str)
            or not step["step_id"].strip()
            or step["step_id"] in step_ids
            or not isinstance(step["occurred_at"], str)
            or not isinstance(step["status"], str)
            or step["status"] not in {"succeeded", "failed", "unknown"}
            or any(
                not _is_sha256(step[f"{field}_hash"])
                for field in _EVIDENCE_FIELDS
            )
            or (
                step["candidate_signal_hash"] is not None
                and not _is_sha256(step["candidate_signal_hash"])
            )
        ):
            raise RuntimeError("Stored trajectory manifest cannot be migrated safely")
        step_ids.add(step["step_id"])
        indexed[(step["step_id"], expected_ordinal)] = step
    return indexed


def _prepare_evidence(row: sa.RowMapping) -> dict[str, str]:
    evidence_id = _canonical_uuid_text(row["evidence_id"])
    evidence_bundle_id = _canonical_uuid_text(row["bundle_id"])
    evidence_owner_id = _canonical_uuid_text(row["owner_agent_id"])
    bundle_id = _canonical_uuid_text(row["bundle_manifest_id"])
    bundle_owner_id = _canonical_uuid_text(row["bundle_owner_agent_id"])
    step_id = _nonblank_text(row["step_id"])
    field = _nonblank_text(row["field"])
    _nonblank_text(row["adapter_kind"])
    _nonblank_text(row["trajectory_id"])
    _nonblank_text(row["sanitization_profile"])
    _nonblank_text(row["source_started_at"])
    _nonblank_text(row["source_completed_at"])
    if (
        evidence_bundle_id != bundle_id
        or evidence_owner_id != bundle_owner_id
        or field not in _EVIDENCE_FIELDS
        or type(row["ordinal"]) is not int
        or row["ordinal"] <= 0
        or type(row["adapter_version"]) is not int
    ):
        raise RuntimeError("Stored trajectory evidence cannot be migrated safely")
    steps = _manifest_steps(row)
    step = steps.get((step_id, row["ordinal"]))
    source_hash = row["legacy_source_hash"]
    if step is None or not _is_sha256(source_hash):
        raise RuntimeError("Stored trajectory evidence cannot be migrated safely")
    field_hash = step[f"{field}_hash"]
    if source_hash != field_hash:
        raise RuntimeError("Stored trajectory evidence cannot be migrated safely")

    excerpt = row["excerpt"]
    if not isinstance(excerpt, str):
        raise RuntimeError("Stored trajectory evidence cannot be migrated safely")
    try:
        excerpt_bytes = excerpt.encode("utf-8")
    except UnicodeEncodeError:
        raise RuntimeError(
            "Stored trajectory evidence cannot be migrated safely"
        ) from None
    if len(excerpt_bytes) > _MAX_EXCERPT_UTF8_BYTES:
        raise RuntimeError("Stored trajectory evidence cannot be migrated safely")
    excerpt_hash = sha256(excerpt_bytes).hexdigest()
    if excerpt_hash != source_hash:
        raise RuntimeError(
            "Stored trajectory evidence is unauthenticated; "
            "recapture or trusted backfill is required"
        )
    return {
        "evidence_id": evidence_id,
        "source_hash": source_hash,
        "excerpt_hash": excerpt_hash,
    }


def _drop_immutable_triggers() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {_TABLE}_reject_conflicting_insert")
    op.execute(f"DROP TRIGGER IF EXISTS {_TABLE}_reject_delete")
    op.execute(f"DROP TRIGGER IF EXISTS {_TABLE}_reject_update")


def _create_immutable_triggers() -> None:
    op.execute(
        f"CREATE TRIGGER {_TABLE}_reject_update "
        f"BEFORE UPDATE ON {_TABLE} BEGIN "
        f"SELECT RAISE(ABORT, '{_TABLE} rows are immutable'); END"
    )
    op.execute(
        f"CREATE TRIGGER {_TABLE}_reject_delete "
        f"BEFORE DELETE ON {_TABLE} BEGIN "
        f"SELECT RAISE(ABORT, '{_TABLE} rows are immutable'); END"
    )
    op.execute(
        f"CREATE TRIGGER {_TABLE}_reject_conflicting_insert "
        f"BEFORE INSERT ON {_TABLE} "
        "WHEN EXISTS (SELECT 1 FROM trajectory_evidence "
        "WHERE evidence_id = NEW.evidence_id "
        "OR (bundle_id = NEW.bundle_id "
        "AND step_id = NEW.step_id AND field = NEW.field)) BEGIN "
        "SELECT RAISE(ABORT, 'trajectory_evidence identity already exists'); END"
    )


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError(
            "offline upgrade cannot recompute retained evidence hashes; "
            "run the upgrade online"
        )
    connection = op.get_bind()
    rows = tuple(
        connection.execute(
            sa.text(
                "SELECT evidence.evidence_id, evidence.bundle_id, "
                "evidence.owner_agent_id, evidence.step_id, evidence.field, "
                "evidence.ordinal, evidence.excerpt, "
                "evidence.excerpt_hash AS legacy_source_hash, "
                "bundle.bundle_id AS bundle_manifest_id, "
                "bundle.owner_agent_id AS bundle_owner_agent_id, "
                "bundle.trajectory_id, bundle.adapter_kind, "
                "bundle.adapter_version, bundle.sanitization_profile, "
                "bundle.manifest, bundle.manifest_hash, "
                "bundle.source_started_at, bundle.source_completed_at "
                "FROM trajectory_evidence AS evidence LEFT JOIN "
                "trajectory_bundles AS bundle "
                "ON bundle.bundle_id = evidence.bundle_id "
                "ORDER BY evidence.evidence_id"
            )
        ).mappings()
    )
    prepared = [_prepare_evidence(row) for row in rows]
    _drop_immutable_triggers()
    op.add_column(
        _TABLE,
        sa.Column("source_hash", sa.String(length=64), nullable=True),
    )
    for values in prepared:
        connection.execute(
            sa.text(
                "UPDATE trajectory_evidence SET source_hash = :source_hash, "
                "excerpt_hash = :excerpt_hash WHERE evidence_id = :evidence_id"
            ),
            values,
        )
    with op.batch_alter_table(_TABLE, recreate="always") as batch:
        batch.alter_column(
            "source_hash",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch.create_check_constraint(
            "ck_trajectory_evidence_source_hash",
            _sha256_check("source_hash"),
        )
    _create_immutable_triggers()


def downgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError(
            "offline downgrade cannot restore legacy evidence hashes; "
            "run the downgrade online"
        )
    _drop_immutable_triggers()
    connection = op.get_bind()
    connection.execute(
        sa.text("UPDATE trajectory_evidence SET excerpt_hash = source_hash")
    )
    with op.batch_alter_table(_TABLE, recreate="always") as batch:
        batch.drop_constraint(
            "ck_trajectory_evidence_source_hash",
            type_="check",
        )
        batch.drop_column("source_hash")
    _create_immutable_triggers()
