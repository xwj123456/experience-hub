"""Experience authoritative source rows and rebuildable projections."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    PayloadCodec,
    Temperature,
)
from experience_hub.storage.tables.base import (
    Base,
    CanonicalJSONBytes,
    UTCDateTime,
    UUIDString,
)


def _enum_type(enum_type: type[StrEnum]) -> SqlEnum:
    return SqlEnum(
        enum_type,
        values_callable=lambda members: [member.value for member in members],
        native_enum=False,
        create_constraint=False,
        length=max(len(member.value) for member in enum_type),
    )


_SHA256_CHECK = (
    "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"
)


class ExperienceRow(Base):
    __tablename__ = "experiences"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis')",
            name="ck_experiences_kind",
        ),
        CheckConstraint(
            "origin IN ('local', 'adopted_capsule', 'adopted_idea')",
            name="ck_experiences_origin",
        ),
        Index(
            "ix_experiences_owner_created",
            "owner_agent_id",
            "created_at",
            "experience_id",
        ),
    )

    experience_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    kind: Mapped[ExperienceKind] = mapped_column(
        _enum_type(ExperienceKind),
        nullable=False,
    )
    origin: Mapped[ExperienceOrigin] = mapped_column(
        _enum_type(ExperienceOrigin),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class ExperienceVersionRow(Base):
    __tablename__ = "experience_versions"
    __table_args__ = (
        CheckConstraint(
            "version_number > 0",
            name="ck_experience_versions_number",
        ),
        CheckConstraint(
            "(version_number = 1 AND supersedes_version_id IS NULL) "
            "OR (version_number > 1 AND supersedes_version_id IS NOT NULL)",
            name="ck_experience_versions_supersession",
        ),
        CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0",
            name="ck_experience_versions_summary",
        ),
        CheckConstraint(
            "length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_versions_mechanism",
        ),
        CheckConstraint(
            "length(tags) > 0 AND length(applicability) > 0 "
            "AND length(evidence) > 0 AND length(falsifiers) > 0",
            name="ck_experience_versions_arrays",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="content_hash"),
            name="ck_experience_versions_content_hash",
        ),
        Index(
            "ux_experience_versions_experience_number",
            "experience_id",
            "version_number",
            unique=True,
        ),
        Index(
            "ix_experience_versions_experience_created",
            "experience_id",
            "created_at",
            "version_id",
        ),
        Index(
            "ix_experience_versions_supersedes",
            "supersedes_version_id",
        ),
    )

    version_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    mechanism: Mapped[str] = mapped_column(String(2000), nullable=False)
    tags: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    applicability: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    evidence: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    falsifiers: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    supersedes_version_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class ExperiencePayloadRow(Base):
    __tablename__ = "experience_payloads"
    __table_args__ = (
        CheckConstraint(
            "codec IN ('plain', 'zlib')",
            name="ck_experience_payloads_codec",
        ),
        CheckConstraint(
            "length(payload) > 0",
            name="ck_experience_payloads_payload",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="payload_hash"),
            name="ck_experience_payloads_payload_hash",
        ),
    )

    version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        primary_key=True,
    )
    codec: Mapped[PayloadCodec] = mapped_column(
        _enum_type(PayloadCodec),
        nullable=False,
    )
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ExperienceLinkRow(Base):
    __tablename__ = "experience_links"
    __table_args__ = (
        CheckConstraint(
            "relation IN ('derived_from', 'supports', 'contradicts', 'tests')",
            name="ck_experience_links_relation",
        ),
        Index(
            "ix_experience_links_target_relation",
            "target_experience_id",
            "relation",
            "source_experience_id",
            "source_version_id",
        ),
        Index(
            "ix_experience_links_source_experience",
            "source_experience_id",
            "source_version_id",
        ),
        Index(
            "ix_experience_links_source_event",
            "source_event_id",
        ),
    )

    source_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=False,
    )
    source_version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        primary_key=True,
    )
    target_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        primary_key=True,
    )
    relation: Mapped[LinkRelation] = mapped_column(
        _enum_type(LinkRelation),
        primary_key=True,
    )
    source_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class ExperienceStateRow(Base):
    __tablename__ = "experience_state"
    __table_args__ = (
        CheckConstraint(
            "temperature IN ('hot', 'warm', 'cold', 'archived')",
            name="ck_experience_state_temperature",
        ),
        CheckConstraint(
            "importance BETWEEN 0 AND 1",
            name="ck_experience_state_importance",
        ),
        CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name="ck_experience_state_confidence",
        ),
        CheckConstraint(
            "activation_score BETWEEN 0 AND 1",
            name="ck_experience_state_activation",
        ),
        CheckConstraint(
            "source_trust BETWEEN 0 AND 1",
            name="ck_experience_state_source_trust",
        ),
        CheckConstraint(
            "access_count >= 0",
            name="ck_experience_state_access_count",
        ),
        CheckConstraint(
            "access_strength BETWEEN 0 AND 20",
            name="ck_experience_state_access_strength",
        ),
        CheckConstraint(
            "consecutive_below_threshold >= 0",
            name="ck_experience_state_below_threshold",
        ),
        CheckConstraint(
            "pinned IN (0, 1)",
            name="ck_experience_state_pinned",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_experience_state_projection_event",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="current_content_hash"),
            name="ck_experience_state_content_hash",
        ),
        Index(
            "ux_experience_state_owner_content",
            "owner_agent_id",
            "current_content_hash",
            unique=True,
        ),
        Index(
            "ix_experience_state_owner_temperature",
            "owner_agent_id",
            "temperature",
            "experience_id",
        ),
        Index(
            "ix_experience_state_current_version",
            "current_version_id",
        ),
    )

    experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        primary_key=True,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    current_version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        nullable=False,
    )
    current_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    temperature: Mapped[Temperature] = mapped_column(
        _enum_type(Temperature),
        nullable=False,
    )
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    activation_score: Mapped[float] = mapped_column(Float, nullable=False)
    source_trust: Mapped[float] = mapped_column(Float, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False)
    access_strength: Mapped[float] = mapped_column(Float, nullable=False)
    strength_updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    last_transition_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    last_lifecycle_evaluated_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    consecutive_below_threshold: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False)
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class ExperienceTermRow(Base):
    __tablename__ = "experience_terms"
    __table_args__ = (
        CheckConstraint(
            "length(term) > 0",
            name="ck_experience_terms_term",
        ),
        CheckConstraint(
            "term_kind IN ('word', 'char_trigram', 'tag', 'mechanism')",
            name="ck_experience_terms_kind",
        ),
        CheckConstraint(
            "weight > 0 AND weight <= 1.5",
            name="ck_experience_terms_weight",
        ),
        Index(
            "ix_experience_terms_lookup",
            "term_kind",
            "term",
            "experience_id",
        ),
    )

    experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        primary_key=True,
    )
    term: Mapped[str] = mapped_column(String, primary_key=True)
    term_kind: Mapped[str] = mapped_column(String(12), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
