"""Immutable trajectory capture sources and rebuildable candidate state."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Table,
    event,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.experiences.models import ExperienceKind
from experience_hub.storage.tables.base import (
    Base,
    CanonicalJSONBytes,
    UTCDateTime,
    UUIDString,
)

_SHA256_CHECK = "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def _enum_type(enum_type: type[StrEnum]) -> SqlEnum:
    return SqlEnum(
        enum_type,
        values_callable=lambda members: [member.value for member in members],
        native_enum=False,
        create_constraint=False,
        length=max(len(member.value) for member in enum_type),
    )


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
        f"'action_hash',json_extract(value,'$.action_hash'),"
        "'candidate_signal_hash',"
        "json_extract(value,'$.candidate_signal_hash'),"
        f"'observation_hash',json_extract(value,'$.observation_hash'),"
        f"'occurred_at',json_extract(value,'$.occurred_at'),"
        f"'ordinal',json_extract(value,'$.ordinal'),"
        f"'outcome_hash',json_extract(value,'$.outcome_hash'),"
        f"'status',json_extract(value,'$.status'),"
        f"'step_id',json_extract(value,'$.step_id')) AS canonical_step "
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


def _canonical_json_trigger(
    *,
    name: str,
    table: str,
    column: str,
    canonical_json: str,
    operation: str,
) -> str:
    return (
        f"CREATE TRIGGER {name} BEFORE {operation} ON {table} "
        "WHEN CASE "
        f"WHEN json_valid(CAST(NEW.{column} AS TEXT)) "
        f"THEN COALESCE(CAST(NEW.{column} AS TEXT) <> {canonical_json}, 1) "
        "ELSE 0 END "
        "BEGIN "
        f"SELECT RAISE(ABORT, '{table} {column} must be canonical JSON'); "
        "END"
    )


class TrajectoryBundleRow(Base):
    """One immutable normalized trajectory manifest."""

    __tablename__ = "trajectory_bundles"
    __table_args__ = (
        CheckConstraint(
            "length(trim(trajectory_id)) > 0",
            name="ck_trajectory_bundles_trajectory_id",
        ),
        CheckConstraint(
            "adapter_kind = 'generic_jsonl' AND adapter_version = 1",
            name="ck_trajectory_bundles_adapter",
        ),
        CheckConstraint(
            "length(trim(sanitization_profile)) > 0",
            name="ck_trajectory_bundles_sanitization_profile",
        ),
        CheckConstraint(
            _canonical_json_check("manifest", json_type="object"),
            name="ck_trajectory_bundles_manifest",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="manifest_hash"),
            name="ck_trajectory_bundles_manifest_hash",
        ),
        CheckConstraint(
            "source_started_at <= source_completed_at",
            name="ck_trajectory_bundles_source_time",
        ),
        Index(
            "ux_trajectory_bundles_owner_manifest",
            "owner_agent_id",
            "manifest_hash",
            unique=True,
        ),
        Index(
            "ux_trajectory_bundles_id_owner",
            "bundle_id",
            "owner_agent_id",
            unique=True,
        ),
        Index(
            "ix_trajectory_bundles_owner_trajectory",
            "owner_agent_id",
            "trajectory_id",
            "captured_at",
            "bundle_id",
        ),
    )

    bundle_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    trajectory_id: Mapped[str] = mapped_column(String, nullable=False)
    adapter_kind: Mapped[str] = mapped_column(String(13), nullable=False)
    adapter_version: Mapped[int] = mapped_column(Integer, nullable=False)
    sanitization_profile: Mapped[str] = mapped_column(String, nullable=False)
    manifest: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_started_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    source_completed_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class TrajectoryEvidenceRow(Base):
    """One immutable bounded excerpt from a captured trajectory field."""

    __tablename__ = "trajectory_evidence"
    __table_args__ = (
        CheckConstraint(
            "length(trim(step_id)) > 0",
            name="ck_trajectory_evidence_step_id",
        ),
        CheckConstraint(
            "field IN ('observation', 'action', 'outcome')",
            name="ck_trajectory_evidence_field",
        ),
        CheckConstraint(
            "ordinal > 0",
            name="ck_trajectory_evidence_ordinal",
        ),
        CheckConstraint(
            "length(CAST(excerpt AS BLOB)) <= 512",
            name="ck_trajectory_evidence_excerpt",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="source_hash"),
            name="ck_trajectory_evidence_source_hash",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="excerpt_hash"),
            name="ck_trajectory_evidence_excerpt_hash",
        ),
        ForeignKeyConstraint(
            ["bundle_id", "owner_agent_id"],
            ["trajectory_bundles.bundle_id", "trajectory_bundles.owner_agent_id"],
        ),
        Index(
            "ux_trajectory_evidence_bundle_step_field",
            "bundle_id",
            "step_id",
            "field",
            unique=True,
        ),
        Index(
            "ix_trajectory_evidence_bundle_ordinal",
            "bundle_id",
            "ordinal",
            "field",
            "evidence_id",
        ),
    )

    evidence_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    bundle_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        nullable=False,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    step_id: Mapped[str] = mapped_column(String, nullable=False)
    field: Mapped[str] = mapped_column(String(11), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    excerpt: Mapped[str] = mapped_column(String, nullable=False)
    excerpt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ExperienceCandidateRow(Base):
    """Immutable extracted experience content held in quarantine."""

    __tablename__ = "experience_candidates"
    __table_args__ = (
        CheckConstraint(
            "candidate_ordinal > 0",
            name="ck_experience_candidates_ordinal",
        ),
        CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis')",
            name="ck_experience_candidates_kind",
        ),
        CheckConstraint(
            "length(CAST(body AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(trim(body)) > 0",
            name="ck_experience_candidates_body",
        ),
        CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0",
            name="ck_experience_candidates_summary",
        ),
        CheckConstraint(
            "length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_candidates_mechanism",
        ),
        CheckConstraint(
            f"{_bounded_json_array_check('tags', limit=32)} "
            f"AND {_bounded_json_array_check('applicability', limit=32)} "
            f"AND {_bounded_json_array_check('falsifiers', limit=32)}",
            name="ck_experience_candidates_content_arrays",
        ),
        CheckConstraint(
            f"{_bounded_json_array_check('evidence', limit=8)} "
            f"AND {_bounded_json_array_check('evidence_refs', limit=8)} "
            "AND json_array_length(CAST(evidence AS TEXT)) "
            "= json_array_length(CAST(evidence_refs AS TEXT))",
            name="ck_experience_candidates_evidence",
        ),
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='content_hash')} "
            f"AND {_SHA256_CHECK.format(column='extractor_configuration_hash')}",
            name="ck_experience_candidates_hashes",
        ),
        CheckConstraint(
            "extractor_kind = 'deterministic_signal_v1'",
            name="ck_experience_candidates_extractor",
        ),
        ForeignKeyConstraint(
            ["bundle_id", "owner_agent_id"],
            ["trajectory_bundles.bundle_id", "trajectory_bundles.owner_agent_id"],
        ),
        Index(
            "ux_experience_candidates_bundle_ordinal",
            "bundle_id",
            "candidate_ordinal",
            unique=True,
        ),
        Index(
            "ix_experience_candidates_owner_created",
            "owner_agent_id",
            "created_at",
            "candidate_id",
        ),
        Index(
            "ux_experience_candidates_id_owner",
            "candidate_id",
            "owner_agent_id",
            unique=True,
        ),
    )

    candidate_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    bundle_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        nullable=False,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    candidate_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[ExperienceKind] = mapped_column(
        _enum_type(ExperienceKind),
        nullable=False,
    )
    body: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    mechanism: Mapped[str] = mapped_column(String(2000), nullable=False)
    tags: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    applicability: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    evidence: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    evidence_refs: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    falsifiers: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extractor_kind: Mapped[str] = mapped_column(String(23), nullable=False)
    extractor_configuration_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class CandidateAdoptionRow(Base):
    """Immutable provenance from a candidate to its adopted experience."""

    __tablename__ = "candidate_adoptions"
    __table_args__ = (
        CheckConstraint(
            _SHA256_CHECK.format(column="resulting_content_hash"),
            name="ck_candidate_adoptions_content_hash",
        ),
        CheckConstraint(
            "created IN (0, 1)",
            name="ck_candidate_adoptions_created",
        ),
        ForeignKeyConstraint(
            ["candidate_id", "owner_agent_id"],
            [
                "experience_candidates.candidate_id",
                "experience_candidates.owner_agent_id",
            ],
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
        Index(
            "ux_candidate_adoptions_candidate",
            "candidate_id",
            unique=True,
        ),
        Index(
            "ix_candidate_adoptions_resulting_experience",
            "resulting_experience_id",
            "adoption_id",
        ),
        Index(
            "ux_candidate_adoptions_lineage",
            "adoption_id",
            "candidate_id",
            "owner_agent_id",
            "resulting_experience_id",
            "resulting_version_id",
            unique=True,
        ),
    )

    adoption_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    candidate_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        nullable=False,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    resulting_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        nullable=False,
    )
    resulting_version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        nullable=False,
    )
    resulting_content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created: Mapped[bool] = mapped_column(Boolean, nullable=False)
    adopted_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class CandidateStateRow(Base):
    """Rebuildable owner decision for a quarantined candidate."""

    __tablename__ = "candidate_state"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('pending', 'adopted', 'rejected')",
            name="ck_candidate_state_decision",
        ),
        CheckConstraint(
            "reason_text_hash IS NULL OR "
            f"({_SHA256_CHECK.format(column='reason_text_hash')})",
            name="ck_candidate_state_reason_hash",
        ),
        CheckConstraint(
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
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_candidate_state_projection_event",
        ),
        ForeignKeyConstraint(
            ["candidate_id", "owner_agent_id"],
            [
                "experience_candidates.candidate_id",
                "experience_candidates.owner_agent_id",
            ],
        ),
        ForeignKeyConstraint(
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
        Index(
            "ix_candidate_state_owner_decision",
            "owner_agent_id",
            "decision",
            "candidate_id",
        ),
    )

    candidate_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        primary_key=True,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    decision: Mapped[str] = mapped_column(String(8), nullable=False)
    adoption_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        nullable=True,
    )
    resulting_experience_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        nullable=True,
    )
    resulting_version_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        nullable=True,
    )
    reason_code: Mapped[str | None] = mapped_column(String, nullable=True)
    reason_text: Mapped[str | None] = mapped_column(String, nullable=True)
    reason_text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


def _create_manifest_canonical_triggers(
    _target: Table,
    connection: Connection,
    **_kwargs: object,
) -> None:
    if connection.dialect.name != "sqlite":
        return
    for operation, suffix in (("INSERT", ""), ("UPDATE", "_update")):
        connection.exec_driver_sql(
            _canonical_json_trigger(
                name=f"trajectory_bundles_reject_noncanonical_manifest{suffix}",
                table="trajectory_bundles",
                column="manifest",
                canonical_json=_manifest_canonical_json("NEW.manifest"),
                operation=operation,
            )
        )


def _create_evidence_canonical_triggers(
    _target: Table,
    connection: Connection,
    **_kwargs: object,
) -> None:
    if connection.dialect.name != "sqlite":
        return
    for operation, suffix in (("INSERT", ""), ("UPDATE", "_update")):
        connection.exec_driver_sql(
            _canonical_json_trigger(
                name=(
                    "experience_candidates_reject_noncanonical_evidence"
                    f"{suffix}"
                ),
                table="experience_candidates",
                column="evidence",
                canonical_json=_evidence_canonical_json("NEW.evidence"),
                operation=operation,
            )
        )


event.listen(
    TrajectoryBundleRow.__table__,
    "after_create",
    _create_manifest_canonical_triggers,
)
event.listen(
    ExperienceCandidateRow.__table__,
    "after_create",
    _create_evidence_canonical_triggers,
)
