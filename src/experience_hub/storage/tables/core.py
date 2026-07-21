"""Core authoritative, ledger, and operational storage rows."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.storage.tables.base import (
    Base,
    CanonicalJSONBytes,
    UTCDateTime,
    UUIDString,
)


class AgentRow(Base):
    __tablename__ = "agents"
    __table_args__ = (
        CheckConstraint(
            "length(trim(name)) > 0 AND name = trim(name)",
            name="ck_agents_name_trimmed",
        ),
        Index("ux_agents_name", "name", unique=True),
    )

    agent_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class IdempotencyRecordRow(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        CheckConstraint(
            "length(trim(caller_scope)) > 0 AND caller_scope = trim(caller_scope)",
            name="ck_idempotency_records_caller_scope",
        ),
        CheckConstraint(
            "length(trim(scope)) > 0 AND scope = trim(scope)",
            name="ck_idempotency_records_scope",
        ),
        CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128 "
            "AND idempotency_key = trim(idempotency_key)",
            name="ck_idempotency_records_key",
        ),
        CheckConstraint(
            "length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_idempotency_records_request_hash",
        ),
        CheckConstraint(
            "state IN ('in_progress', 'completed')",
            name="ck_idempotency_records_state",
        ),
        CheckConstraint(
            "(result_resource_type IS NULL AND result_resource_id IS NULL) OR "
            "(length(trim(result_resource_type)) > 0 "
            "AND result_resource_type = trim(result_resource_type) "
            "AND result_resource_id IS NOT NULL)",
            name="ck_idempotency_records_resource",
        ),
        CheckConstraint(
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
        Index(
            "ux_idempotency_records_scope_key",
            "caller_scope",
            "scope",
            "idempotency_key",
            unique=True,
        ),
        Index(
            "ix_idempotency_records_resource_state",
            "result_resource_type",
            "result_resource_id",
            "state",
        ),
    )

    receipt_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    caller_scope: Mapped[str] = mapped_column(String, nullable=False)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    result_resource_type: Mapped[str | None] = mapped_column(String, nullable=True)
    result_resource_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        nullable=True,
    )
    response_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[bytes | None] = mapped_column(
        CanonicalJSONBytes(),
        nullable=True,
    )
    response_content_type: Mapped[str | None] = mapped_column(String, nullable=True)
    response_headers: Mapped[bytes | None] = mapped_column(
        CanonicalJSONBytes(),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )


class DomainEventRow(Base):
    __tablename__ = "domain_events"
    __table_args__ = (
        CheckConstraint(
            "length(trim(aggregate_type)) > 0 "
            "AND aggregate_type = trim(aggregate_type)",
            name="ck_domain_events_aggregate_type",
        ),
        CheckConstraint("sequence > 0", name="ck_domain_events_sequence"),
        CheckConstraint(
            "length(trim(event_type)) > 0 AND event_type = trim(event_type)",
            name="ck_domain_events_event_type",
        ),
        CheckConstraint("length(payload) > 0", name="ck_domain_events_payload"),
        Index(
            "ux_domain_events_aggregate_sequence",
            "aggregate_type",
            "aggregate_id",
            "sequence",
            unique=True,
        ),
    )

    event_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    aggregate_type: Mapped[str] = mapped_column(String, nullable=False)
    aggregate_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    actor_agent_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=True,
    )
    causation_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("idempotency_records.receipt_id"),
        nullable=False,
    )
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class ProjectionVersionRow(Base):
    __tablename__ = "projection_versions"
    __table_args__ = (
        CheckConstraint(
            "length(trim(name)) > 0 AND name = trim(name)",
            name="ck_projection_versions_name",
        ),
        CheckConstraint(
            "reducer_version > 0",
            name="ck_projection_versions_reducer_version",
        ),
        CheckConstraint(
            "last_applied_event_id >= 0",
            name="ck_projection_versions_event_id",
        ),
        CheckConstraint(
            "(last_verified_hash IS NULL AND last_verified_at IS NULL) OR "
            "(length(last_verified_hash) = 64 "
            "AND last_verified_hash NOT GLOB '*[^0-9a-f]*' "
            "AND last_verified_at IS NOT NULL)",
            name="ck_projection_versions_verification",
        ),
    )

    name: Mapped[str] = mapped_column(String, primary_key=True)
    reducer_version: Mapped[int] = mapped_column(Integer, nullable=False)
    last_applied_event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    last_verified_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )


class LifecycleLeaseRow(Base):
    __tablename__ = "lifecycle_lease"
    __table_args__ = (
        CheckConstraint(
            "lease_name = 'lifecycle'",
            name="ck_lifecycle_lease_singleton",
        ),
        CheckConstraint(
            "(owner_id IS NULL AND acquired_at IS NULL AND expires_at IS NULL) OR "
            "(owner_id IS NOT NULL AND acquired_at IS NOT NULL "
            "AND expires_at IS NOT NULL AND expires_at > acquired_at)",
            name="ck_lifecycle_lease_state",
        ),
    )

    lease_name: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[UUID | None] = mapped_column(UUIDString(), nullable=True)
    acquired_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
