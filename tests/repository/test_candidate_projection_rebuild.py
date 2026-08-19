from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.hashing import extractor_configuration_hash
from experience_hub.capture.models import CapturedEvidenceV1, TrajectoryField
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.capture.validation import register_capture_source_validator
from experience_hub.domain import EventRegistry, StoredEvent, StructuredReason
from experience_hub.experiences.candidate_events import (
    CandidateAdoptedV1,
    CandidateCreatedV1,
    CandidateRejectedV1,
    TrajectoryCapturedV1,
    register_candidate_events,
)
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
    CandidateDecision,
)
from experience_hub.experiences.candidate_projector import (
    CandidateProjectionIntegrityError,
    CandidateStateProjector,
)
from experience_hub.experiences.candidate_service import CandidateViewV1
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
    register_experience_events,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.storage.database import Database
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
    ProjectionRegistry,
)
from experience_hub.storage.tables import (
    AgentRow,
    CandidateAdoptionRow,
    DomainEventRow,
    ExperienceCandidateRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
    TrajectoryBundleRow,
    TrajectoryEvidenceRow,
)
from experience_hub.storage.validation import SourceValidator

NOW = datetime(2026, 7, 22, 9, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000421")
BUNDLE_ID = UUID("00000000-0000-0000-0000-000000000422")
PENDING_ID = UUID("00000000-0000-0000-0000-000000000423")
ADOPTED_ID = UUID("00000000-0000-0000-0000-000000000424")
REJECTED_ID = UUID("00000000-0000-0000-0000-000000000425")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000426")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000427")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000428")
CAPTURE_RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000429")
ADOPTION_RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000430")
REJECTION_RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000431")
REUSED_ID = UUID("00000000-0000-0000-0000-000000000435")
REUSED_ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000436")
REUSED_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000437")
REUSED_VERSION_ID = UUID("00000000-0000-0000-0000-000000000438")
REUSED_ADOPTION_RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000439")
REUSED_TARGET_RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000440")
EVIDENCE_IDS = (
    UUID("00000000-0000-0000-0000-000000000432"),
    UUID("00000000-0000-0000-0000-000000000433"),
    UUID("00000000-0000-0000-0000-000000000434"),
    UUID("00000000-0000-0000-0000-000000000441"),
)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _receipt(
    *,
    receipt_id: UUID,
    scope: str,
    resource_type: str,
    resource_id: UUID,
    completed_at: datetime,
    created_at: datetime | None = None,
    response_status_code: int = 200,
    response_body: bytes | None = None,
    response_headers: bytes | None = None,
) -> IdempotencyRecordRow:
    return IdempotencyRecordRow(
        receipt_id=receipt_id,
        caller_scope=f"agent:{OWNER_ID}",
        scope=scope,
        idempotency_key=str(receipt_id),
        request_hash=HASH_A,
        state="completed",
        result_resource_type=resource_type,
        result_resource_id=resource_id,
        response_status_code=response_status_code,
        response_body=(
            canonical_json_bytes({}) if response_body is None else response_body
        ),
        response_content_type="application/json",
        response_headers=(
            canonical_json_bytes({})
            if response_headers is None
            else response_headers
        ),
        created_at=completed_at if created_at is None else created_at,
        completed_at=completed_at + timedelta(seconds=1),
    )


def _step_id(candidate_id: UUID) -> str:
    return f"step-{candidate_id}"


def _excerpt(candidate_id: UUID) -> str:
    if candidate_id == PENDING_ID:
        return "界" * 170
    return f"Observation {candidate_id}"


def _source_observation(candidate_id: UUID) -> str:
    if candidate_id == PENDING_ID:
        return "界" * 171 + "tail"
    return _excerpt(candidate_id)


def _candidate_content(
    candidate_id: UUID,
    manifest_hash: str,
) -> tuple[VersionContent, str, bytes]:
    content = VersionContent(
        body=f"Body {candidate_id}",
        summary=f"Summary {candidate_id}",
        mechanism=f"Mechanism {candidate_id}",
        tags=("capture",),
        applicability=("tests",),
        evidence=(
            {
                "id": (
                    f"{manifest_hash}:{_step_id(candidate_id)}:observation"
                ),
                "type": "trajectory_field",
            },
        ),
        falsifiers=("Counterexample",),
    )
    encoded = encode_version_content(kind=ExperienceKind.SEMANTIC, content=content)
    return content, encoded.content_hash, encoded.payload


def _candidate_row(
    candidate_id: UUID,
    *,
    ordinal: int,
    manifest_hash: str,
    evidence_id: UUID,
) -> ExperienceCandidateRow:
    content, content_hash, _ = _candidate_content(candidate_id, manifest_hash)
    return ExperienceCandidateRow(
        candidate_id=candidate_id,
        bundle_id=BUNDLE_ID,
        owner_agent_id=OWNER_ID,
        candidate_ordinal=ordinal,
        kind=ExperienceKind.SEMANTIC,
        body=content.body,
        summary=content.summary,
        mechanism=content.mechanism,
        tags=canonical_json_bytes(content.tags),
        applicability=canonical_json_bytes(content.applicability),
        evidence=canonical_json_bytes(content.evidence),
        evidence_refs=canonical_json_bytes([evidence_id]),
        falsifiers=canonical_json_bytes(content.falsifiers),
        content_hash=content_hash,
        extractor_kind="deterministic_signal_v1",
        extractor_configuration_hash=extractor_configuration_hash(),
        created_at=NOW,
    )


def _candidate_response_body(
    candidate_id: UUID,
    manifest_hash: str,
    *,
    decision: CandidateDecision,
    resulting_experience_id: UUID | None = None,
    resulting_version_id: UUID | None = None,
    reason: StructuredReason | None = None,
    decided_at: datetime | None = None,
) -> bytes:
    content, content_hash, _ = _candidate_content(candidate_id, manifest_hash)
    excerpt = _excerpt(candidate_id)
    view = CandidateViewV1(
        candidate_id=candidate_id,
        bundle_id=BUNDLE_ID,
        owner_agent_id=OWNER_ID,
        decision=decision,
        kind=ExperienceKind.SEMANTIC,
        content=content,
        content_hash=content_hash,
        evidence=(
            CapturedEvidenceV1(
                step_id=_step_id(candidate_id),
                field=TrajectoryField.OBSERVATION,
                excerpt=excerpt,
                source_hash=sha256_hex(
                    _source_observation(candidate_id).encode()
                ),
                excerpt_hash=sha256_hex(excerpt.encode()),
            ),
        ),
        extractor_kind="deterministic_signal_v1",
        extractor_configuration_hash=extractor_configuration_hash(),
        resulting_experience_id=resulting_experience_id,
        resulting_version_id=resulting_version_id,
        reason=reason,
        created_at=NOW,
        decided_at=decided_at,
    )
    return canonical_json_bytes({"data": view.model_dump(mode="json")})


def _experience_snapshot(
    *,
    experience_id: UUID,
    version_id: UUID,
    content_hash: str,
    occurred_at: datetime,
) -> ExperienceStateSnapshotV1:
    return ExperienceStateSnapshotV1(
        experience_id=experience_id,
        owner_agent_id=OWNER_ID,
        current_version_id=version_id,
        current_content_hash=content_hash,
        temperature=Temperature.HOT,
        importance=0.5,
        confidence=0.5,
        activation_score=0.5,
        source_trust=1.0,
        access_count=0,
        access_strength=0.0,
        strength_updated_at=occurred_at,
        last_accessed_at=None,
        last_transition_at=occurred_at,
        last_lifecycle_evaluated_at=None,
        consecutive_below_threshold=0,
        pinned=False,
    )


def _stored(registry: EventRegistry, row: DomainEventRow) -> StoredEvent:
    return StoredEvent(
        event_id=row.event_id,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        sequence=row.sequence,
        event_type=row.event_type,
        payload=registry.decode(event_type=row.event_type, payload=row.payload),
        actor_agent_id=row.actor_agent_id,
        causation_id=row.causation_id,
        occurred_at=row.occurred_at,
    )


async def seed_candidate_graph(
    database: Database,
    manager: ProjectionManager,
    registry: EventRegistry,
) -> None:
    candidate_ids = (PENDING_ID, ADOPTED_ID, REJECTED_ID, REUSED_ID)
    manifest = canonical_json_bytes(
        {
            "adapter": {"kind": "generic_jsonl", "version": 1},
            "owner_agent_id": str(OWNER_ID),
            "sanitization": {
                "input_sanitized": True,
                "profile_id": "trusted-v1",
            },
            "schema_version": 1,
            "source_completed_at": NOW,
            "source_started_at": NOW,
            "steps": [
                {
                    "action_hash": HASH_A,
                    "candidate_signal_hash": None,
                    "observation_hash": sha256_hex(
                        _source_observation(candidate_id).encode()
                    ),
                    "occurred_at": NOW,
                    "ordinal": ordinal,
                    "outcome_hash": HASH_C,
                    "status": "succeeded",
                    "step_id": _step_id(candidate_id),
                }
                for ordinal, candidate_id in enumerate(candidate_ids, start=1)
            ],
            "trajectory_id": "projection-rebuild",
        }
    )
    manifest_hash = sha256_hex(manifest)
    candidates = tuple(
        _candidate_row(
            candidate_id,
            ordinal=ordinal,
            manifest_hash=manifest_hash,
            evidence_id=evidence_id,
        )
        for ordinal, (candidate_id, evidence_id) in enumerate(
            zip(candidate_ids, EVIDENCE_IDS, strict=True),
            start=1,
        )
    )
    adopted_content, adopted_hash, adopted_payload = _candidate_content(
        ADOPTED_ID,
        manifest_hash,
    )
    reused_content, reused_hash, reused_payload = _candidate_content(
        REUSED_ID,
        manifest_hash,
    )
    reused_target_at = NOW - timedelta(seconds=10)
    adopted_at = NOW + timedelta(seconds=1)
    reused_at = NOW + timedelta(seconds=2)
    rejected_at = NOW + timedelta(seconds=3)
    reason = StructuredReason.from_user_text("Not applicable.")
    capture_response_body = canonical_json_bytes(
        {
            "data": {
                "bundle_id": BUNDLE_ID,
                "owner_agent_id": OWNER_ID,
                "manifest_hash": manifest_hash,
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "captured_at": NOW,
            }
        }
    )
    capture_response_headers = canonical_json_bytes(
        {
            "location": (
                f"/v1/agents/{OWNER_ID}/trajectory-bundles/{BUNDLE_ID}"
            )
        }
    )

    async with database.transaction() as uow:
        uow.session.add(
            AgentRow(agent_id=OWNER_ID, name="Candidate Owner", created_at=NOW)
        )
        uow.session.add_all(
            (
                _receipt(
                    receipt_id=CAPTURE_RECEIPT_ID,
                    scope=TRAJECTORY_IMPORT_SCOPE,
                    resource_type="trajectory_bundle",
                    resource_id=BUNDLE_ID,
                    completed_at=NOW,
                    response_status_code=201,
                    response_body=capture_response_body,
                    response_headers=capture_response_headers,
                ),
                _receipt(
                    receipt_id=ADOPTION_RECEIPT_ID,
                    scope=CANDIDATE_ADOPT_SCOPE,
                    resource_type="candidate_adoption",
                    resource_id=ADOPTION_ID,
                    completed_at=adopted_at,
                    response_body=_candidate_response_body(
                        ADOPTED_ID,
                        manifest_hash,
                        decision=CandidateDecision.ADOPTED,
                        resulting_experience_id=EXPERIENCE_ID,
                        resulting_version_id=VERSION_ID,
                        decided_at=adopted_at,
                    ),
                ),
                _receipt(
                    receipt_id=REJECTION_RECEIPT_ID,
                    scope=CANDIDATE_REJECT_SCOPE,
                    resource_type="experience_candidate",
                    resource_id=REJECTED_ID,
                    completed_at=rejected_at,
                    response_body=_candidate_response_body(
                        REJECTED_ID,
                        manifest_hash,
                        decision=CandidateDecision.REJECTED,
                        reason=reason,
                        decided_at=rejected_at,
                    ),
                ),
                _receipt(
                    receipt_id=REUSED_ADOPTION_RECEIPT_ID,
                    scope=CANDIDATE_ADOPT_SCOPE,
                    resource_type="candidate_adoption",
                    resource_id=REUSED_ADOPTION_ID,
                    completed_at=reused_at,
                    response_body=_candidate_response_body(
                        REUSED_ID,
                        manifest_hash,
                        decision=CandidateDecision.ADOPTED,
                        resulting_experience_id=REUSED_EXPERIENCE_ID,
                        resulting_version_id=REUSED_VERSION_ID,
                        decided_at=reused_at,
                    ),
                ),
                _receipt(
                    receipt_id=REUSED_TARGET_RECEIPT_ID,
                    scope="experience.create",
                    resource_type="experience",
                    resource_id=REUSED_EXPERIENCE_ID,
                    completed_at=reused_target_at,
                    created_at=reused_target_at - timedelta(seconds=1),
                ),
            )
        )
        await uow.session.flush()
        uow.session.add(
            TrajectoryBundleRow(
                bundle_id=BUNDLE_ID,
                owner_agent_id=OWNER_ID,
                trajectory_id="projection-rebuild",
                adapter_kind="generic_jsonl",
                adapter_version=1,
                sanitization_profile="trusted-v1",
                manifest=manifest,
                manifest_hash=manifest_hash,
                source_started_at=NOW,
                source_completed_at=NOW,
                captured_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add_all(
            tuple(
                TrajectoryEvidenceRow(
                    evidence_id=evidence_id,
                    bundle_id=BUNDLE_ID,
                    owner_agent_id=OWNER_ID,
                    step_id=_step_id(candidate_id),
                    field="observation",
                    ordinal=ordinal,
                    excerpt=_excerpt(candidate_id),
                    source_hash=sha256_hex(
                        _source_observation(candidate_id).encode()
                    ),
                    excerpt_hash=sha256_hex(_excerpt(candidate_id).encode()),
                )
                for ordinal, (candidate_id, evidence_id) in enumerate(
                    zip(candidate_ids, EVIDENCE_IDS, strict=True),
                    start=1,
                )
            )
        )
        await uow.session.flush()
        uow.session.add_all(candidates)
        await uow.session.flush()
        uow.session.add_all(
            (
                ExperienceRow(
                    experience_id=EXPERIENCE_ID,
                    owner_agent_id=OWNER_ID,
                    kind=ExperienceKind.SEMANTIC,
                    origin=ExperienceOrigin.ADOPTED_CANDIDATE,
                    created_at=adopted_at,
                ),
                ExperienceRow(
                    experience_id=REUSED_EXPERIENCE_ID,
                    owner_agent_id=OWNER_ID,
                    kind=ExperienceKind.SEMANTIC,
                    origin=ExperienceOrigin.LOCAL,
                    created_at=reused_target_at,
                ),
            )
        )
        await uow.session.flush()
        uow.session.add_all(
            (
                ExperienceVersionRow(
                    version_id=VERSION_ID,
                    experience_id=EXPERIENCE_ID,
                    version_number=1,
                    summary=adopted_content.summary,
                    mechanism=adopted_content.mechanism,
                    tags=canonical_json_bytes(adopted_content.tags),
                    applicability=canonical_json_bytes(
                        adopted_content.applicability
                    ),
                    evidence=canonical_json_bytes(adopted_content.evidence),
                    falsifiers=canonical_json_bytes(adopted_content.falsifiers),
                    content_hash=adopted_hash,
                    supersedes_version_id=None,
                    created_at=adopted_at,
                ),
                ExperienceVersionRow(
                    version_id=REUSED_VERSION_ID,
                    experience_id=REUSED_EXPERIENCE_ID,
                    version_number=1,
                    summary=reused_content.summary,
                    mechanism=reused_content.mechanism,
                    tags=canonical_json_bytes(reused_content.tags),
                    applicability=canonical_json_bytes(
                        reused_content.applicability
                    ),
                    evidence=canonical_json_bytes(reused_content.evidence),
                    falsifiers=canonical_json_bytes(reused_content.falsifiers),
                    content_hash=reused_hash,
                    supersedes_version_id=None,
                    created_at=reused_target_at,
                ),
            )
        )
        await uow.session.flush()
        uow.session.add_all(
            (
                ExperiencePayloadRow(
                    version_id=VERSION_ID,
                    codec=PayloadCodec.PLAIN,
                    payload=adopted_payload,
                    payload_hash=sha256_hex(adopted_payload),
                ),
                ExperiencePayloadRow(
                    version_id=REUSED_VERSION_ID,
                    codec=PayloadCodec.PLAIN,
                    payload=reused_payload,
                    payload_hash=sha256_hex(reused_payload),
                ),
            )
        )
        await uow.session.flush()
        uow.session.add_all(
            (
                CandidateAdoptionRow(
                    adoption_id=ADOPTION_ID,
                    candidate_id=ADOPTED_ID,
                    owner_agent_id=OWNER_ID,
                    resulting_experience_id=EXPERIENCE_ID,
                    resulting_version_id=VERSION_ID,
                    resulting_content_hash=adopted_hash,
                    created=True,
                    adopted_at=adopted_at,
                ),
                CandidateAdoptionRow(
                    adoption_id=REUSED_ADOPTION_ID,
                    candidate_id=REUSED_ID,
                    owner_agent_id=OWNER_ID,
                    resulting_experience_id=REUSED_EXPERIENCE_ID,
                    resulting_version_id=REUSED_VERSION_ID,
                    resulting_content_hash=reused_hash,
                    created=False,
                    adopted_at=reused_at,
                ),
            )
        )
        await uow.session.flush()

        captured = TrajectoryCapturedV1(
            schema_version=1,
            bundle_id=BUNDLE_ID,
            owner_agent_id=OWNER_ID,
            manifest_hash=manifest_hash,
            evidence_ids=EVIDENCE_IDS,
            candidate_ids=candidate_ids,
        )
        pending_payload = CandidateCreatedV1(
            schema_version=1,
            candidate_id=PENDING_ID,
            bundle_id=BUNDLE_ID,
            owner_agent_id=OWNER_ID,
            content_hash=candidates[0].content_hash,
            evidence_ids=(EVIDENCE_IDS[0],),
            decision_after=CandidateDecision.PENDING,
        )
        adopted_created = pending_payload.model_copy(
            update={
                "candidate_id": ADOPTED_ID,
                "content_hash": candidates[1].content_hash,
                "evidence_ids": (EVIDENCE_IDS[1],),
            }
        )
        rejected_created = pending_payload.model_copy(
            update={
                "candidate_id": REJECTED_ID,
                "content_hash": candidates[2].content_hash,
                "evidence_ids": (EVIDENCE_IDS[2],),
            }
        )
        reused_created = pending_payload.model_copy(
            update={
                "candidate_id": REUSED_ID,
                "content_hash": candidates[3].content_hash,
                "evidence_ids": (EVIDENCE_IDS[3],),
            }
        )
        adopted_snapshot = _experience_snapshot(
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            content_hash=adopted_hash,
            occurred_at=adopted_at,
        )
        reused_snapshot = _experience_snapshot(
            experience_id=REUSED_EXPERIENCE_ID,
            version_id=REUSED_VERSION_ID,
            content_hash=reused_hash,
            occurred_at=reused_target_at,
        )
        payloads = (
            (
                "experience",
                REUSED_EXPERIENCE_ID,
                1,
                ExperienceCreatedV1(
                    schema_version=1,
                    experience_id=REUSED_EXPERIENCE_ID,
                    version_id=REUSED_VERSION_ID,
                    after=reused_snapshot,
                ),
                REUSED_TARGET_RECEIPT_ID,
                reused_target_at,
            ),
            (
                "experience",
                REUSED_EXPERIENCE_ID,
                2,
                ExperienceVersionCreatedV1(
                    schema_version=1,
                    experience_id=REUSED_EXPERIENCE_ID,
                    version_id=REUSED_VERSION_ID,
                    version_number=1,
                    supersedes_version_id=None,
                    links=(),
                    before=reused_snapshot,
                    after=reused_snapshot,
                ),
                REUSED_TARGET_RECEIPT_ID,
                reused_target_at,
            ),
            ("trajectory_bundle", BUNDLE_ID, 1, captured, CAPTURE_RECEIPT_ID, NOW),
            (
                "experience_candidate",
                PENDING_ID,
                1,
                pending_payload,
                CAPTURE_RECEIPT_ID,
                NOW,
            ),
            (
                "experience_candidate",
                ADOPTED_ID,
                1,
                adopted_created,
                CAPTURE_RECEIPT_ID,
                NOW,
            ),
            (
                "experience_candidate",
                REJECTED_ID,
                1,
                rejected_created,
                CAPTURE_RECEIPT_ID,
                NOW,
            ),
            (
                "experience_candidate",
                REUSED_ID,
                1,
                reused_created,
                CAPTURE_RECEIPT_ID,
                NOW,
            ),
            (
                "experience",
                EXPERIENCE_ID,
                1,
                ExperienceCreatedV1(
                    schema_version=1,
                    experience_id=EXPERIENCE_ID,
                    version_id=VERSION_ID,
                    after=adopted_snapshot,
                ),
                ADOPTION_RECEIPT_ID,
                adopted_at,
            ),
            (
                "experience",
                EXPERIENCE_ID,
                2,
                ExperienceVersionCreatedV1(
                    schema_version=1,
                    experience_id=EXPERIENCE_ID,
                    version_id=VERSION_ID,
                    version_number=1,
                    supersedes_version_id=None,
                    links=(),
                    before=adopted_snapshot,
                    after=adopted_snapshot,
                ),
                ADOPTION_RECEIPT_ID,
                adopted_at,
            ),
            (
                "experience_candidate",
                ADOPTED_ID,
                2,
                CandidateAdoptedV1(
                    schema_version=1,
                    candidate_id=ADOPTED_ID,
                    owner_agent_id=OWNER_ID,
                    decision_before=CandidateDecision.PENDING,
                    decision_after=CandidateDecision.ADOPTED,
                    adoption_id=ADOPTION_ID,
                    resulting_experience_id=EXPERIENCE_ID,
                    resulting_version_id=VERSION_ID,
                    resulting_content_hash=adopted_hash,
                    created=True,
                ),
                ADOPTION_RECEIPT_ID,
                adopted_at,
            ),
            (
                "experience_candidate",
                REUSED_ID,
                2,
                CandidateAdoptedV1(
                    schema_version=1,
                    candidate_id=REUSED_ID,
                    owner_agent_id=OWNER_ID,
                    decision_before=CandidateDecision.PENDING,
                    decision_after=CandidateDecision.ADOPTED,
                    adoption_id=REUSED_ADOPTION_ID,
                    resulting_experience_id=REUSED_EXPERIENCE_ID,
                    resulting_version_id=REUSED_VERSION_ID,
                    resulting_content_hash=reused_hash,
                    created=False,
                ),
                REUSED_ADOPTION_RECEIPT_ID,
                reused_at,
            ),
            (
                "experience_candidate",
                REJECTED_ID,
                2,
                CandidateRejectedV1(
                    schema_version=1,
                    candidate_id=REJECTED_ID,
                    owner_agent_id=OWNER_ID,
                    decision_before=CandidateDecision.PENDING,
                    decision_after=CandidateDecision.REJECTED,
                    reason=reason,
                ),
                REJECTION_RECEIPT_ID,
                rejected_at,
            ),
        )
        event_rows = tuple(
            DomainEventRow(
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                sequence=sequence,
                event_type=payload.event_type,
                payload=canonical_json_bytes(payload),
                actor_agent_id=OWNER_ID,
                causation_id=receipt_id,
                occurred_at=occurred_at,
            )
            for (
                aggregate_type,
                aggregate_id,
                sequence,
                payload,
                receipt_id,
                occurred_at,
            ) in payloads
        )
        uow.session.add_all(event_rows)
        await uow.session.flush()
        await manager.apply(
            session=uow.session,
            events=[_stored(registry, row) for row in event_rows],
        )


async def _rows(
    database: Database,
    table: str,
    ordering: str,
) -> tuple[tuple[Any, ...], ...]:
    async with database.read_session() as session:
        return tuple(
            tuple(row)
            for row in await session.execute(
                text(f"SELECT * FROM {table} ORDER BY {ordering}")
            )
        )


@pytest.fixture
async def candidate_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[tuple[Database, ProjectionManager]]:
    path = tmp_path / "candidate-projection.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")
    registry = EventRegistry()
    register_experience_events(registry)
    register_candidate_events(registry)
    source_validator = SourceValidator(registry)
    register_capture_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry([CandidateStateProjector(registry)]),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{path}",
        event_registry=registry,
        projection_applier=manager,
    )
    await seed_candidate_graph(database, manager, registry)
    assert (await manager.verify(database)).matches
    try:
        yield database, manager
    finally:
        await database.dispose()


@pytest.mark.parametrize(
    ("candidate_id", "damage"),
    (
        (PENDING_ID, "decision = 'rejected', reason_code = 'damaged', "
         "reason_text = 'damaged', reason_text_hash = :hash, decided_at = :now"),
        (ADOPTED_ID, "decided_at = :now"),
        (REJECTED_ID, "reason_text = 'damaged'"),
    ),
)
@pytest.mark.asyncio
async def test_candidate_state_verify_reports_stable_key_and_repair_is_exact(
    candidate_stack: tuple[Database, ProjectionManager],
    candidate_id: UUID,
    damage: str,
) -> None:
    database, manager = candidate_stack
    golden = await _rows(database, "candidate_state", "candidate_id")
    sources = await _rows(database, "experience_candidates", "candidate_id")
    async with database.transaction() as uow:
        await uow.session.execute(
            text(f"UPDATE candidate_state SET {damage} WHERE candidate_id = :id"),
            {"id": str(candidate_id), "hash": HASH_C, "now": NOW.isoformat()},
        )

    observed: list[tuple[str, ...]] = []
    for _ in range(2):
        with pytest.raises(ProjectionMismatch) as caught:
            await manager.verify(database)
        observed.append(caught.value.report.differences[0].differing_keys)
    assert observed == [(str(candidate_id),), (str(candidate_id),)]

    assert (await manager.repair(database)).matches
    assert await _rows(database, "candidate_state", "candidate_id") == golden
    assert await _rows(database, "experience_candidates", "candidate_id") == sources


@pytest.mark.asyncio
async def test_candidate_rebuild_uses_only_sources_and_candidate_events(
    candidate_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = candidate_stack
    golden = await _rows(database, "candidate_state", "candidate_id")
    async with database.transaction() as uow:
        await uow.session.execute(text("DELETE FROM candidate_state"))

    assert (await manager.repair(database)).matches
    assert await _rows(database, "candidate_state", "candidate_id") == golden
    async with database.read_session() as session:
        decisions = tuple(
            await session.scalars(
                select(text("decision")).select_from(text("candidate_state"))
            )
        )
    assert set(decisions) == {"pending", "adopted", "rejected"}


@pytest.mark.asyncio
async def test_live_apply_accepts_strictly_anchored_in_progress_receipt(
    candidate_stack: tuple[Database, ProjectionManager],
) -> None:
    database, _ = candidate_stack
    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)

    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE idempotency_records SET state = 'in_progress', "
                "response_status_code = NULL, response_body = NULL, "
                "response_content_type = NULL, response_headers = NULL, "
                "completed_at = NULL WHERE receipt_id = :receipt_id"
            ),
            {"receipt_id": str(CAPTURE_RECEIPT_ID)},
        )
        await uow.session.execute(
            text("DELETE FROM candidate_state WHERE candidate_id = :candidate_id"),
            {"candidate_id": str(PENDING_ID)},
        )
        event_row = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == PENDING_ID,
                DomainEventRow.sequence == 1,
            )
        )
        assert event_row is not None

        await projector.apply(uow.session, _stored(registry, event_row))

        state = await uow.session.scalar(
            text(
                "SELECT decision FROM candidate_state "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(PENDING_ID)},
        )
        assert state == CandidateDecision.PENDING.value


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("caller_scope", "agent:00000000-0000-0000-0000-000000000999"),
        ("scope", CANDIDATE_ADOPT_SCOPE),
        ("result_resource_type", "experience_candidate"),
        ("result_resource_id", str(REJECTED_ID)),
        (
            "created_at",
            (NOW + timedelta(seconds=1))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
        ),
    ),
)
@pytest.mark.asyncio
async def test_live_apply_rejects_misanchored_in_progress_receipt(
    candidate_stack: tuple[Database, ProjectionManager],
    field: str,
    value: object,
) -> None:
    database, _ = candidate_stack
    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)

    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                f"UPDATE idempotency_records SET state = 'in_progress', "
                "response_status_code = NULL, response_body = NULL, "
                "response_content_type = NULL, response_headers = NULL, "
                f"completed_at = NULL, {field} = :value "
                "WHERE receipt_id = :receipt_id"
            ),
            {"receipt_id": str(CAPTURE_RECEIPT_ID), "value": value},
        )
        await uow.session.execute(
            text("DELETE FROM candidate_state WHERE candidate_id = :candidate_id"),
            {"candidate_id": str(PENDING_ID)},
        )
        event_row = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == PENDING_ID,
                DomainEventRow.sequence == 1,
            )
        )
        assert event_row is not None

        with pytest.raises(CandidateProjectionIntegrityError):
            await projector.apply(uow.session, _stored(registry, event_row))


@pytest.mark.asyncio
async def test_live_apply_rejects_completed_receipt_before_event(
    candidate_stack: tuple[Database, ProjectionManager],
) -> None:
    database, _ = candidate_stack
    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)

    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE idempotency_records SET completed_at = :completed_at "
                "WHERE receipt_id = :receipt_id"
            ),
            {
                "receipt_id": str(CAPTURE_RECEIPT_ID),
                "completed_at": (NOW - timedelta(seconds=2))
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
            },
        )
        await uow.session.execute(
            text("DELETE FROM candidate_state WHERE candidate_id = :candidate_id"),
            {"candidate_id": str(PENDING_ID)},
        )
        event_row = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == PENDING_ID,
                DomainEventRow.sequence == 1,
            )
        )
        assert event_row is not None

        with pytest.raises(CandidateProjectionIntegrityError):
            await projector.apply(uow.session, _stored(registry, event_row))


@pytest.mark.asyncio
async def test_live_apply_rejects_tampered_completed_receipt_response(
    candidate_stack: tuple[Database, ProjectionManager],
) -> None:
    database, _ = candidate_stack
    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)

    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE idempotency_records SET response_status_code = 202 "
                "WHERE receipt_id = :receipt_id"
            ),
            {"receipt_id": str(CAPTURE_RECEIPT_ID)},
        )
        await uow.session.execute(
            text("DELETE FROM candidate_state WHERE candidate_id = :candidate_id"),
            {"candidate_id": str(PENDING_ID)},
        )
        event_row = await uow.session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == PENDING_ID,
                DomainEventRow.sequence == 1,
            )
        )
        assert event_row is not None

        with pytest.raises(CandidateProjectionIntegrityError):
            await projector.apply(uow.session, _stored(registry, event_row))


@pytest.mark.asyncio
async def test_completed_decision_response_authenticates_manifest_once(
    candidate_stack: tuple[Database, ProjectionManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _ = candidate_stack
    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)

    async with database.transaction() as uow:
        await uow.session.execute(
            text("DELETE FROM candidate_state WHERE candidate_id = :candidate_id"),
            {"candidate_id": str(ADOPTED_ID)},
        )
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.aggregate_id == ADOPTED_ID)
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
        events = tuple(projector.stored_event_from_row(row) for row in rows)
        assert len(events) == 2
        await projector.apply(uow.session, events[0])

        calls: list[UUID] = []
        original = capture_source_integrity.authenticate_trajectory_manifest

        def counting_authenticate(
            bundle: TrajectoryBundleRow,
        ) -> capture_source_integrity.AuthenticatedTrajectoryManifest:
            calls.append(bundle.bundle_id)
            return original(bundle)

        monkeypatch.setattr(
            capture_source_integrity,
            "authenticate_trajectory_manifest",
            counting_authenticate,
        )
        await projector.apply(uow.session, events[1])

    assert calls == [BUNDLE_ID]
