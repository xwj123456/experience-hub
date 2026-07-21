"""Sharing authoritative source rows and rebuildable projections."""

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
    String,
    Text,
    text,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.experiences.models import ExperienceKind
from experience_hub.sharing.models import (
    CapsuleStatus,
    FeedbackVerdict,
    InboxState,
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


_SHA256_CHECK = "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


class TopicRow(Base):
    __tablename__ = "topics"
    __table_args__ = (
        CheckConstraint(
            "length(name) BETWEEN 1 AND 200 "
            "AND length(trim(name)) > 0 AND name = trim(name)",
            name="ck_topics_name",
        ),
        CheckConstraint(
            "description IS NULL OR "
            "(length(description) BETWEEN 1 AND 2000 "
            "AND length(trim(description)) > 0 "
            "AND description = trim(description))",
            name="ck_topics_description",
        ),
        Index("ux_topics_name", "name", unique=True),
        Index(
            "ix_topics_owner_created",
            "owner_agent_id",
            "created_at",
            "topic_id",
        ),
    )

    topic_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class SubscriptionRow(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint(
            "creation_event_id > 0",
            name="ck_subscriptions_creation_event",
        ),
        Index(
            "ux_subscriptions_subscriber_topic",
            "subscriber_agent_id",
            "topic_id",
            unique=True,
        ),
        Index(
            "ix_subscriptions_topic_delivery",
            "topic_id",
            "creation_event_id",
            "subscriber_agent_id",
            "subscription_id",
        ),
    )

    subscription_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        primary_key=True,
    )
    subscriber_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    topic_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("topics.topic_id"),
        nullable=False,
    )
    creation_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class ExperienceCapsuleRow(Base):
    __tablename__ = "experience_capsules"
    __table_args__ = (
        CheckConstraint(
            "transport_schema_version = 1",
            name="ck_experience_capsules_transport_version",
        ),
        CheckConstraint(
            "kind IN ('episodic', 'semantic', 'procedural', 'hypothesis') "
            "AND length(trim(body)) > 0 "
            "AND length(CAST(body AS BLOB)) BETWEEN 1 AND 65536 "
            "AND length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0 "
            "AND length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_experience_capsules_content",
        ),
        CheckConstraint(
            "length(tags) > 0 AND length(applicability) > 0 "
            "AND length(evidence) > 0 AND length(falsifiers) > 0 "
            "AND length(provenance_chain) > 0",
            name="ck_experience_capsules_arrays",
        ),
        CheckConstraint(
            "publisher_confidence BETWEEN 0 AND 1",
            name="ck_experience_capsules_confidence",
        ),
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='root_fingerprint')} "
            f"AND {_SHA256_CHECK.format(column='source_content_hash')} "
            f"AND {_SHA256_CHECK.format(column='capsule_hash')}",
            name="ck_experience_capsules_hashes",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_experience_capsules_expiry",
        ),
        CheckConstraint(
            "hop_count BETWEEN 0 AND 4 "
            "AND json_type(CAST(provenance_chain AS TEXT)) = 'array' "
            "AND json_array_length(CAST(provenance_chain AS TEXT)) = hop_count",
            name="ck_experience_capsules_hop_count",
        ),
        Index(
            "ix_experience_capsules_topic_created",
            "topic_id",
            "created_at",
            "capsule_id",
        ),
        Index(
            "ix_experience_capsules_publisher_created",
            "publisher_agent_id",
            "created_at",
            "capsule_id",
        ),
        Index(
            "ix_experience_capsules_source",
            "source_experience_id",
            "source_version_id",
        ),
    )

    capsule_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    transport_schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    topic_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("topics.topic_id"),
        nullable=False,
    )
    source_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=False,
    )
    source_version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        nullable=False,
    )
    publisher_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    kind: Mapped[ExperienceKind] = mapped_column(
        _enum_type(ExperienceKind),
        nullable=False,
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
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
    publisher_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    provenance_chain: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    root_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    hop_count: Mapped[int] = mapped_column(Integer, nullable=False)
    capsule_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class AdoptionRecordRow(Base):
    __tablename__ = "adoption_records"
    __table_args__ = (
        CheckConstraint(
            "captured_trust BETWEEN 0 AND 1",
            name="ck_adoption_records_captured_trust",
        ),
        CheckConstraint(
            "length(provenance_chain) > 0 "
            "AND json_type(CAST(provenance_chain AS TEXT)) = 'array' "
            "AND json_array_length(CAST(provenance_chain AS TEXT)) > 0",
            name="ck_adoption_records_provenance",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="root_fingerprint"),
            name="ck_adoption_records_root_fingerprint",
        ),
        CheckConstraint(
            "corroboration_applied IN (0, 1)",
            name="ck_adoption_records_corroboration",
        ),
        Index(
            "ux_adoption_records_adopter_capsule",
            "adopter_agent_id",
            "capsule_id",
            unique=True,
        ),
        Index(
            "ix_adoption_records_resulting_root",
            "resulting_experience_id",
            "root_fingerprint",
        ),
        Index(
            "ux_adoption_records_corroborated_root",
            "resulting_experience_id",
            "root_fingerprint",
            unique=True,
            sqlite_where=text("corroboration_applied = 1"),
        ),
    )

    adoption_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    adopter_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    capsule_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_capsules.capsule_id"),
        nullable=False,
    )
    resulting_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=False,
    )
    captured_trust: Mapped[float] = mapped_column(Float, nullable=False)
    provenance_chain: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    root_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    corroboration_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    adopted_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class CapsuleFeedbackRow(Base):
    __tablename__ = "capsule_feedback"
    __table_args__ = (
        CheckConstraint(
            "revision > 0",
            name="ck_capsule_feedback_revision",
        ),
        CheckConstraint(
            "verdict IN ('useful', 'refuted', 'harmful')",
            name="ck_capsule_feedback_verdict",
        ),
        CheckConstraint(
            "length(reason) > 0 AND length(evidence) > 0",
            name="ck_capsule_feedback_payloads",
        ),
        Index(
            "ux_capsule_feedback_observer_capsule_revision",
            "observer_agent_id",
            "capsule_id",
            "revision",
            unique=True,
        ),
        Index(
            "ix_capsule_feedback_capsule_observer_revision",
            "capsule_id",
            "observer_agent_id",
            "revision",
        ),
    )

    feedback_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    observer_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    capsule_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_capsules.capsule_id"),
        nullable=False,
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[FeedbackVerdict] = mapped_column(
        _enum_type(FeedbackVerdict),
        nullable=False,
    )
    reason: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    evidence: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class CapsuleStateRow(Base):
    __tablename__ = "capsule_state"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'retracted')",
            name="ck_capsule_state_status",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_capsule_state_projection_event",
        ),
    )

    capsule_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_capsules.capsule_id"),
        primary_key=True,
    )
    status: Mapped[CapsuleStatus] = mapped_column(
        _enum_type(CapsuleStatus),
        nullable=False,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class InboxItemRow(Base):
    __tablename__ = "inbox_items"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'adopted', 'rejected')",
            name="ck_inbox_items_state",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_inbox_items_projection_event",
        ),
        Index(
            "ux_inbox_items_recipient_capsule",
            "recipient_agent_id",
            "capsule_id",
            unique=True,
        ),
        Index(
            "ix_inbox_items_recipient_state",
            "recipient_agent_id",
            "state",
            "item_id",
        ),
    )

    item_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    recipient_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    capsule_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_capsules.capsule_id"),
        nullable=False,
    )
    state: Mapped[InboxState] = mapped_column(
        _enum_type(InboxState),
        nullable=False,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class AgentReputationRow(Base):
    __tablename__ = "agent_reputation"
    __table_args__ = (
        CheckConstraint(
            "useful_count >= 0 AND refuted_count >= 0 AND harmful_count >= 0",
            name="ck_agent_reputation_counts",
        ),
        CheckConstraint(
            "alpha = 2 + useful_count",
            name="ck_agent_reputation_alpha_prior",
        ),
        CheckConstraint(
            "beta = 2 + refuted_count + harmful_count",
            name="ck_agent_reputation_beta_prior",
        ),
        CheckConstraint(
            "alpha > 0 AND beta > 0 "
            "AND subject_agent_id != observer_agent_id",
            name="ck_agent_reputation_trust",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_agent_reputation_projection_event",
        ),
    )

    subject_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        primary_key=True,
    )
    observer_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        primary_key=True,
    )
    useful_count: Mapped[int] = mapped_column(Integer, nullable=False)
    refuted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    harmful_count: Mapped[int] = mapped_column(Integer, nullable=False)
    alpha: Mapped[int] = mapped_column(Integer, nullable=False)
    beta: Mapped[int] = mapped_column(Integer, nullable=False)
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


__all__ = [
    "AdoptionRecordRow",
    "AgentReputationRow",
    "CapsuleFeedbackRow",
    "CapsuleStateRow",
    "ExperienceCapsuleRow",
    "InboxItemRow",
    "SubscriptionRow",
    "TopicRow",
]
