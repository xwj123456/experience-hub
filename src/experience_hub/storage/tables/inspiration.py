"""Inspiration authoritative source rows and rebuildable projections."""

from __future__ import annotations

from datetime import datetime
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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from experience_hub.storage.tables.base import (
    Base,
    CanonicalJSONBytes,
    UTCDateTime,
    UUIDString,
)

_SHA256_CHECK = "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"


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


class InspirationRunRow(Base):
    """Immutable identity, configuration, and budgets for one run."""

    __tablename__ = "inspiration_runs"
    __table_args__ = (
        CheckConstraint(
            "length(goal) BETWEEN 1 AND 2000 "
            "AND length(trim(goal)) > 0 AND goal = trim(goal)",
            name="ck_inspiration_runs_goal",
        ),
        CheckConstraint(
            "context IS NULL OR (length(context) BETWEEN 1 AND 4000 "
            "AND length(trim(context)) > 0 AND context = trim(context))",
            name="ck_inspiration_runs_context",
        ),
        CheckConstraint(
            "mode IN ('focused', 'associative')",
            name="ck_inspiration_runs_mode",
        ),
        CheckConstraint(
            "generator_kind IN ('deterministic', 'openai_compatible')",
            name="ck_inspiration_runs_generator",
        ),
        CheckConstraint(
            "length(generator_configuration) > 0 "
            "AND json_valid(CAST(generator_configuration AS TEXT)) "
            "AND json_type(CAST(generator_configuration AS TEXT)) = 'object'",
            name="ck_inspiration_runs_generator_configuration",
        ),
        CheckConstraint(
            "CAST(operators AS TEXT) IN "
            "('[\"causal_gap\"]', "
            "'[\"counterfactual\"]', "
            "'[\"distant_analogy\"]', "
            "'[\"causal_gap\",\"counterfactual\"]', "
            "'[\"causal_gap\",\"distant_analogy\"]', "
            "'[\"counterfactual\",\"distant_analogy\"]', "
            "'[\"causal_gap\",\"counterfactual\",\"distant_analogy\"]'"
            ")",
            name="ck_inspiration_runs_operators",
        ),
        CheckConstraint(
            "include_inbox IN (0, 1)",
            name="ck_inspiration_runs_include_inbox",
        ),
        CheckConstraint(
            "branches_per_operator BETWEEN 1 AND 3",
            name="ck_inspiration_runs_branches",
        ),
        CheckConstraint(
            "output_tokens_per_operator BETWEEN 1 AND 1200 "
            "AND total_output_tokens BETWEEN 1 AND 3600",
            name="ck_inspiration_runs_token_budgets",
        ),
        CheckConstraint(
            "operator_timeout_seconds BETWEEN 1 AND 30 "
            "AND global_timeout_seconds BETWEEN 1 AND 90 "
            "AND global_timeout_seconds >= operator_timeout_seconds",
            name="ck_inspiration_runs_time_budgets",
        ),
        CheckConstraint(
            _SHA256_CHECK.format(column="request_hash"),
            name="ck_inspiration_runs_request_hash",
        ),
        Index(
            "ix_inspiration_runs_owner_created",
            "owner_agent_id",
            "created_at",
            "run_id",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    goal: Mapped[str] = mapped_column(String(2000), nullable=False)
    context: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    mode: Mapped[str] = mapped_column(String(11), nullable=False)
    generator_kind: Mapped[str] = mapped_column(String(17), nullable=False)
    generator_configuration: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    operators: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    include_inbox: Mapped[bool] = mapped_column(Boolean, nullable=False)
    branches_per_operator: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens_per_operator: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    operator_timeout_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    global_timeout_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class InspirationSnapshotItemRow(Base):
    """Immutable bounded evidence captured before generation begins."""

    __tablename__ = "inspiration_snapshot_items"
    __table_args__ = (
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='stable_evidence_key')} "
            f"AND {_SHA256_CHECK.format(column='content_hash')}",
            name="ck_inspiration_snapshot_items_hashes",
        ),
        CheckConstraint(
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
        CheckConstraint(
            "rank BETWEEN 1 AND 12",
            name="ck_inspiration_snapshot_items_rank",
        ),
        CheckConstraint(
            "length(summary) BETWEEN 1 AND 1000 "
            "AND length(trim(summary)) > 0 "
            "AND length(mechanism) BETWEEN 1 AND 2000 "
            "AND length(trim(mechanism)) > 0",
            name="ck_inspiration_snapshot_items_text",
        ),
        CheckConstraint(
            f"{_json_array_check('applicability', nonempty=False)} "
            "AND json_array_length(CAST(applicability AS TEXT)) <= 32 "
            f"AND {_json_array_check('tags', nonempty=False)} "
            "AND json_array_length(CAST(tags AS TEXT)) <= 32",
            name="ck_inspiration_snapshot_items_arrays",
        ),
        CheckConstraint(
            f"{_json_array_check('falsifiers', nonempty=False)} "
            "AND json_array_length(CAST(falsifiers AS TEXT)) <= 32",
            name="ck_inspiration_snapshot_items_falsifiers",
        ),
        CheckConstraint(
            "length(CAST(excerpt AS BLOB)) <= 2048",
            name="ck_inspiration_snapshot_items_excerpt",
        ),
        Index(
            "ux_inspiration_snapshot_items_run_rank",
            "run_id",
            "rank",
            unique=True,
        ),
        Index(
            "ux_inspiration_snapshot_items_run_source",
            "run_id",
            "source_type",
            "source_id",
            "source_version_id",
            unique=True,
        ),
        Index(
            "ix_inspiration_snapshot_items_stable_key",
            "stable_evidence_key",
            "snapshot_item_id",
        ),
    )

    snapshot_item_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        primary_key=True,
    )
    run_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_runs.run_id"),
        nullable=False,
    )
    stable_evidence_key: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    source_version_id: Mapped[UUID] = mapped_column(UUIDString(), nullable=False)
    source_state: Mapped[str] = mapped_column(String(11), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    mechanism: Mapped[str] = mapped_column(String(2000), nullable=False)
    applicability: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    tags: Mapped[bytes] = mapped_column(CanonicalJSONBytes(), nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    source_trust: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    falsifiers: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )


class InspirationIdeaRow(Base):
    """Immutable validated idea body produced by one operator."""

    __tablename__ = "inspiration_ideas"
    __table_args__ = (
        CheckConstraint(
            "operator IN ('causal_gap', 'counterfactual', 'distant_analogy')",
            name="ck_inspiration_ideas_operator",
        ),
        CheckConstraint(
            "ordinal BETWEEN 1 AND 3",
            name="ck_inspiration_ideas_ordinal",
        ),
        CheckConstraint(
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
        CheckConstraint(
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
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='idea_content_hash')} "
            f"AND {_SHA256_CHECK.format(column='mechanism_hash')}",
            name="ck_inspiration_ideas_hashes",
        ),
        CheckConstraint(
            "duplicate_relation IS NULL OR duplicate_relation != idea_id",
            name="ck_inspiration_ideas_duplicate_relation",
        ),
        Index(
            "ux_inspiration_ideas_run_operator_ordinal",
            "run_id",
            "operator",
            "ordinal",
            unique=True,
        ),
        Index(
            "ix_inspiration_ideas_mechanism_hash",
            "mechanism_hash",
            "idea_id",
        ),
        Index(
            "ix_inspiration_ideas_duplicate_relation",
            "duplicate_relation",
        ),
    )

    idea_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_runs.run_id"),
        nullable=False,
    )
    operator: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[str] = mapped_column(String(2000), nullable=False)
    predictions: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    falsifiers: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    assumptions: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    proposed_test: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_references: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    idea_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mechanism_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    duplicate_relation: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_ideas.idea_id"),
        nullable=True,
    )


class IdeaOccurrenceRow(Base):
    """Immutable recurrence signal for an idea mechanism and snapshot."""

    __tablename__ = "idea_occurrences"
    __table_args__ = (
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='mechanism_hash')} "
            f"AND {_SHA256_CHECK.format(column='snapshot_hash')}",
            name="ck_idea_occurrences_hashes",
        ),
        Index(
            "ux_idea_occurrences_run_mechanism",
            "run_id",
            "mechanism_hash",
            unique=True,
        ),
        Index(
            "ux_idea_occurrences_idea",
            "idea_id",
            unique=True,
        ),
        Index(
            "ix_idea_occurrences_mechanism_time",
            "mechanism_hash",
            "occurred_at",
            "occurrence_id",
        ),
    )

    occurrence_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    idea_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_ideas.idea_id"),
        nullable=False,
    )
    mechanism_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_runs.run_id"),
        nullable=False,
    )
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class IdeaAdoptionRecordRow(Base):
    """Immutable provenance from an idea to an adopted hypothesis."""

    __tablename__ = "idea_adoption_records"
    __table_args__ = (
        CheckConstraint(
            _SHA256_CHECK.format(column="snapshot_hash"),
            name="ck_idea_adoption_records_snapshot_hash",
        ),
        CheckConstraint(
            f"{_json_array_check('evidence_snapshot_item_ids', nonempty=True)} "
            f"AND {_json_array_check('evidence_stable_keys', nonempty=True)} "
            "AND json_array_length(CAST(evidence_snapshot_item_ids AS TEXT)) "
            "= json_array_length(CAST(evidence_stable_keys AS TEXT))",
            name="ck_idea_adoption_records_evidence",
        ),
        Index(
            "ux_idea_adoption_records_owner_idea",
            "owner_agent_id",
            "idea_id",
            unique=True,
        ),
    )

    adoption_id: Mapped[UUID] = mapped_column(UUIDString(), primary_key=True)
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    idea_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_ideas.idea_id"),
        nullable=False,
    )
    run_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_runs.run_id"),
        nullable=False,
    )
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_snapshot_item_ids: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    evidence_stable_keys: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    resulting_experience_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=False,
    )
    resulting_version_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        nullable=False,
    )
    adopted_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class InspirationRunStateRow(Base):
    """Rebuildable run status and bounded generation accounting."""

    __tablename__ = "inspiration_run_state"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'completed_with_errors', "
            "'failed', 'timed_out')",
            name="ck_inspiration_run_state_status",
        ),
        CheckConstraint(
            "snapshot_hash IS NULL OR "
            f"({_SHA256_CHECK.format(column='snapshot_hash')})",
            name="ck_inspiration_run_state_snapshot_hash",
        ),
        CheckConstraint(
            _json_array_check("operator_outcomes", nonempty=False),
            name="ck_inspiration_run_state_outcomes",
        ),
        CheckConstraint(
            "output_tokens_reserved >= 0 "
            "AND output_tokens_consumed >= 0 "
            "AND output_tokens_consumed <= output_tokens_reserved "
            "AND elapsed_milliseconds >= 0",
            name="ck_inspiration_run_state_budgets",
        ),
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) OR "
            "(status != 'running' AND completed_at IS NOT NULL "
            "AND completed_at >= started_at)",
            name="ck_inspiration_run_state_terminality",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_inspiration_run_state_projection_event",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_runs.run_id"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String(21), nullable=False)
    snapshot_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    operator_outcomes: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    output_tokens_reserved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    output_tokens_consumed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    elapsed_milliseconds: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class MechanismIncubationRow(Base):
    """Rebuildable cross-run recurrence and maturity cluster."""

    __tablename__ = "mechanism_incubation"
    __table_args__ = (
        CheckConstraint(
            f"{_SHA256_CHECK.format(column='cluster_id')} "
            f"AND {_SHA256_CHECK.format(column='canonical_mechanism_hash')} "
            "AND cluster_id = canonical_mechanism_hash",
            name="ck_mechanism_incubation_identity",
        ),
        CheckConstraint(
            _json_array_check("member_hashes", nonempty=True),
            name="ck_mechanism_incubation_members",
        ),
        CheckConstraint(
            "occurrence_count >= 1 "
            "AND distinct_snapshot_count BETWEEN 1 AND occurrence_count "
            "AND distinct_adopter_count >= 0 "
            "AND supported_count >= 0 "
            "AND refuted_count >= 0",
            name="ck_mechanism_incubation_counts",
        ),
        CheckConstraint(
            "maturity IN ('speculative', 'incubating', 'candidate') "
            "AND ((maturity = 'candidate' AND candidate_since IS NOT NULL) "
            "OR "
            "(maturity != 'candidate' AND candidate_since IS NULL))",
            name="ck_mechanism_incubation_maturity",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_mechanism_incubation_projection_event",
        ),
        UniqueConstraint(
            "canonical_mechanism_hash",
            name="ux_mechanism_incubation_canonical_hash",
        ),
        Index(
            "ix_mechanism_incubation_maturity_signal",
            "maturity",
            "last_signal_at",
            "cluster_id",
        ),
    )

    cluster_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_mechanism_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    member_hashes: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_snapshot_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_adopter_count: Mapped[int] = mapped_column(Integer, nullable=False)
    supported_count: Mapped[int] = mapped_column(Integer, nullable=False)
    refuted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    maturity: Mapped[str] = mapped_column(String(11), nullable=False)
    candidate_since: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    last_signal_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


class IdeaStateRow(Base):
    """Rebuildable owner decision and effective evaluation state."""

    __tablename__ = "idea_state"
    __table_args__ = (
        CheckConstraint(
            "owner_decision IN ('active', 'adopted', 'rejected', 'archived')",
            name="ck_idea_state_decision",
        ),
        CheckConstraint(
            _json_array_check("evaluations", nonempty=False),
            name="ck_idea_state_evaluations",
        ),
        CheckConstraint(
            "decision_reason IS NULL OR "
            "(length(decision_reason) > 0 "
            "AND json_valid(CAST(decision_reason AS TEXT)) "
            "AND json_type(CAST(decision_reason AS TEXT)) = 'object')",
            name="ck_idea_state_reason",
        ),
        CheckConstraint(
            "(owner_decision = 'adopted' "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL) OR "
            "(owner_decision != 'adopted' "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL)",
            name="ck_idea_state_result",
        ),
        CheckConstraint(
            "projection_event_id > 0",
            name="ck_idea_state_projection_event",
        ),
        Index(
            "ix_idea_state_owner_decision",
            "owner_agent_id",
            "owner_decision",
            "idea_id",
        ),
        Index(
            "ix_idea_state_cluster",
            "mechanism_cluster_id",
            "idea_id",
        ),
        Index(
            "ix_idea_state_archive_due",
            "owner_decision",
            "last_signal_at",
            "idea_id",
        ),
    )

    idea_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("inspiration_ideas.idea_id"),
        primary_key=True,
    )
    owner_agent_id: Mapped[UUID] = mapped_column(
        UUIDString(),
        ForeignKey("agents.agent_id"),
        nullable=False,
    )
    mechanism_cluster_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("mechanism_incubation.cluster_id"),
        nullable=False,
    )
    owner_decision: Mapped[str] = mapped_column(String(8), nullable=False)
    evaluations: Mapped[bytes] = mapped_column(
        CanonicalJSONBytes(),
        nullable=False,
    )
    decision_reason: Mapped[bytes | None] = mapped_column(
        CanonicalJSONBytes(),
        nullable=True,
    )
    resulting_experience_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        ForeignKey("experiences.experience_id"),
        nullable=True,
    )
    resulting_version_id: Mapped[UUID | None] = mapped_column(
        UUIDString(),
        ForeignKey("experience_versions.version_id"),
        nullable=True,
    )
    last_signal_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )
    projection_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("domain_events.event_id"),
        nullable=False,
    )


__all__ = [
    "IdeaAdoptionRecordRow",
    "IdeaOccurrenceRow",
    "IdeaStateRow",
    "InspirationIdeaRow",
    "InspirationRunRow",
    "InspirationRunStateRow",
    "InspirationSnapshotItemRow",
    "MechanismIncubationRow",
]
