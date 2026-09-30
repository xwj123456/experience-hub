"""Immutable Passport import and adoption sources with rebuildable state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Connection,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Table,
    event,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.passports.contracts import PassportState
from experience_hub.storage.tables.base import (
    Base,
    CanonicalJSONBytes,
    UTCDateTime,
    UUIDString,
)

_SHA256_CHECK = "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"
_SOURCE_JSON_CHECK = (
    "typeof(canonical_bytes) = 'blob' AND "
    "length(canonical_bytes) BETWEEN 1 AND 524288 AND "
    "CASE WHEN json_valid(CAST(canonical_bytes AS TEXT)) THEN "
    "json_type(CAST(canonical_bytes AS TEXT)) = 'object' AND "
    "CAST(canonical_bytes AS TEXT) = json(CAST(canonical_bytes AS TEXT)) "
    "ELSE 0 END"
)
_STATE_SHAPE = (
    "(state = 'pending' AND adoption_id IS NULL "
    "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
    "AND reason_code IS NULL AND reason_text IS NULL "
    "AND reason_text_hash IS NULL AND decided_at IS NULL) "
    "OR (state = 'adopted' AND adoption_id IS NOT NULL "
    "AND resulting_experience_id IS NOT NULL AND resulting_version_id IS NOT NULL "
    "AND reason_code IS NULL AND reason_text IS NULL "
    "AND reason_text_hash IS NULL AND decided_at IS NOT NULL) "
    "OR (state = 'rejected' AND adoption_id IS NULL "
    "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
    "AND reason_code IS NOT NULL AND length(trim(reason_code)) > 0 "
    "AND reason_text IS NOT NULL AND length(trim(reason_text)) > 0 "
    "AND reason_text_hash IS NOT NULL AND decided_at IS NOT NULL)"
)


def _utc_check(column: str) -> str:
    pattern = (
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T"
        "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]."
        "[0-9][0-9][0-9][0-9][0-9][0-9]Z"
    )
    # Validate calendar/time without rounding the six retained fractional digits.
    return (
        f"typeof({column}) = 'text' AND length({column}) = 27 "
        f"AND {column} GLOB '{pattern}' AND substr({column},1,4) != '0000' "
        f"AND COALESCE(strftime('%Y-%m-%dT%H:%M:%S',substr({column},1,19),"
        f"'+0 seconds') = substr({column},1,19),0)"
    )


class PassportImportRow(Base):
    """Exact external bytes scoped only to the explicitly selected local owner."""

    __tablename__ = "passport_imports"
    __table_args__ = (
        CheckConstraint(_utc_check("imported_at"), name="ck_passport_imports_time"),
        CheckConstraint(_SOURCE_JSON_CHECK, name="ck_passport_imports_bytes"),
        CheckConstraint(
            _SHA256_CHECK.format(column="passport_hash"),
            name="ck_passport_imports_hash",
        ),
        Index(
            "ux_passport_imports_owner_hash",
            "owner_agent_id",
            "passport_hash",
            unique=True,
        ),
        Index(
            "ux_passport_imports_id_owner", "import_id", "owner_agent_id", unique=True
        ),
        Index(
            "ix_passport_imports_owner_time",
            "owner_agent_id",
            "imported_at",
            "import_id",
        ),
    )

    import_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    passport_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_bytes: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class PassportAdoptionRow(Base):
    """Immutable local decision parameters and exact owned target version."""

    __tablename__ = "passport_adoptions"
    __table_args__ = (
        CheckConstraint(_utc_check("adopted_at"), name="ck_passport_adoptions_time"),
        CheckConstraint(
            _SHA256_CHECK.format(column="resulting_content_hash"),
            name="ck_passport_adoptions_hash",
        ),
        CheckConstraint("created IN (0,1)", name="ck_passport_adoptions_created"),
        CheckConstraint(
            "importance >= 0 AND importance <= 1",
            name="ck_passport_adoptions_importance",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_passport_adoptions_confidence",
        ),
        ForeignKeyConstraint(
            ["import_id", "owner_agent_id"],
            ["passport_imports.import_id", "passport_imports.owner_agent_id"],
        ),
        ForeignKeyConstraint(
            ["resulting_experience_id", "owner_agent_id"],
            ["experiences.experience_id", "experiences.owner_agent_id"],
        ),
        ForeignKeyConstraint(
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
        Index("ux_passport_adoptions_import", "import_id", unique=True),
        Index("ix_passport_adoptions_result", "resulting_experience_id", "adoption_id"),
        Index(
            "ux_passport_adoptions_lineage",
            "adoption_id",
            "import_id",
            "owner_agent_id",
            "resulting_experience_id",
            "resulting_version_id",
            unique=True,
        ),
    )

    adoption_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    import_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    resulting_experience_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    resulting_version_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    resulting_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created: Mapped[bool] = mapped_column(Boolean, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    adopted_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class PassportStateRow(Base):
    """Discardable owner-scoped decision projection, never a fact source."""

    __tablename__ = "passport_state"
    __table_args__ = (
        CheckConstraint(
            "decided_at IS NULL OR (" + _utc_check("decided_at") + ")",
            name="ck_passport_state_time",
        ),
        CheckConstraint(
            "state IN ('pending','adopted','rejected')", name="ck_passport_state_state"
        ),
        CheckConstraint(_STATE_SHAPE, name="ck_passport_state_shape"),
        CheckConstraint(
            "reason_text_hash IS NULL OR "
            + _SHA256_CHECK.format(column="reason_text_hash"),
            name="ck_passport_state_reason_hash",
        ),
        CheckConstraint(
            "projection_event_id > 0", name="ck_passport_state_projection_event"
        ),
        ForeignKeyConstraint(
            ["import_id", "owner_agent_id"],
            ["passport_imports.import_id", "passport_imports.owner_agent_id"],
        ),
        ForeignKeyConstraint(
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
        Index("ix_passport_state_owner_state", "owner_agent_id", "state", "import_id"),
    )

    import_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    state: Mapped[PassportState] = mapped_column(
        SqlEnum(
            PassportState,
            values_callable=lambda values: [v.value for v in values],
            native_enum=False,
            create_constraint=False,
            length=8,
        ),
        nullable=False,
    )
    adoption_id: Mapped[UUID | None] = mapped_column(UUIDString(), nullable=True)
    resulting_experience_id: Mapped[UUID | None] = mapped_column(
        UUIDString(), nullable=True
    )
    resulting_version_id: Mapped[UUID | None] = mapped_column(
        UUIDString(), nullable=True
    )
    reason_code: Mapped[str | None] = mapped_column(String, nullable=True)
    reason_text: Mapped[str | None] = mapped_column(String, nullable=True)
    reason_text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


def _source_guards(table: Table, connection: Connection, **_kwargs: object) -> None:
    if connection.dialect.name != "sqlite":
        return
    name = table.name
    conflict = (
        "import_id=NEW.import_id OR "
        "(owner_agent_id=NEW.owner_agent_id AND passport_hash=NEW.passport_hash)"
        if name == "passport_imports"
        else "adoption_id=NEW.adoption_id OR import_id=NEW.import_id"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.exec_driver_sql(
            f"CREATE TRIGGER {name}_reject_{operation.lower()} "
            f"BEFORE {operation} ON {name} BEGIN "
            f"SELECT RAISE(ABORT,'{name} rows are immutable'); END"
        )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {name}_reject_conflicting_insert BEFORE INSERT ON {name} "
        f"WHEN EXISTS (SELECT 1 FROM {name} WHERE {conflict}) "
        f"BEGIN SELECT RAISE(ABORT,'{name} identity already exists'); END"
    )
    if name == "passport_imports":
        # Every object must have unique, increasing keys. Full protocol/hash
        # authentication remains the domain source validator's responsibility.
        connection.exec_driver_sql(
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


event.listen(PassportImportRow.__table__, "after_create", _source_guards)
event.listen(PassportAdoptionRow.__table__, "after_create", _source_guards)

__all__ = ["PassportImportRow", "PassportAdoptionRow", "PassportStateRow"]
