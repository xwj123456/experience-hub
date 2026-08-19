"""Pure capture preparation and transaction-bound quarantine persistence."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from uuid import UUID

from sqlalchemy import select

from experience_hub import canonical_json_bytes
from experience_hub.capture.extraction import CandidateExtractor
from experience_hub.capture.hashing import trajectory_manifest_document
from experience_hub.capture.jsonl import GenericJsonlAdapter
from experience_hub.capture.models import (
    CandidateDraftV1,
    CapturedEvidenceV1,
    PreparedCaptureV1,
    SensitiveMatchV1,
    TrajectoryField,
)
from experience_hub.capture.responses import capture_stored_response
from experience_hub.capture.sanitization import SecretScanner
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandContext, PendingEvent, ReplayableCommandError
from experience_hub.errors import DomainError
from experience_hub.experiences.candidate_events import (
    CandidateCreatedV1,
    TrajectoryCapturedV1,
)
from experience_hub.experiences.candidate_models import CandidateDecision
from experience_hub.ids import IdGenerator
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import ReceiptStore, StoredResponse
from experience_hub.storage.tables import (
    AgentRow,
    ExperienceCandidateRow,
    TrajectoryBundleRow,
    TrajectoryEvidenceRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


class SensitiveInputError(DomainError):
    """Reject sensitive input without retaining the matched material."""

    def __init__(self, matches: tuple[SensitiveMatchV1, ...]) -> None:
        super().__init__(
            code="sensitive_input_detected",
            message="Sensitive input was detected",
            details={
                "matches": [match.model_dump(mode="json") for match in matches]
            },
            status_code=422,
        )


class CapturePreparer:
    """Parse, scan, and deterministically extract before any mutation begins."""

    def __init__(
        self,
        *,
        adapter: GenericJsonlAdapter,
        scanner: SecretScanner,
        extractor: CandidateExtractor,
    ) -> None:
        self._adapter = adapter
        self._scanner = scanner
        self._extractor = extractor

    def prepare_jsonl(self, data: bytes) -> PreparedCaptureV1:
        bundle = self._adapter.parse(data)
        matches = self._scanner.scan(bundle)
        if matches:
            raise SensitiveInputError(matches)
        candidates = self._extractor.extract(bundle)
        return PreparedCaptureV1(
            bundle=bundle,
            manifest_json=canonical_json_bytes(
                trajectory_manifest_document(bundle)
            ),
            candidates=candidates,
        )


class CaptureService:
    """Persist one prepared capture and its quarantine ledger atomically."""

    def __init__(
        self,
        *,
        clock: Clock,
        id_generator: IdGenerator,
        receipt_store: ReceiptStore,
    ) -> None:
        self._clock = clock
        self._id_generator = id_generator
        self._receipt_store = receipt_store

    async def capture(
        self,
        *,
        uow: UnitOfWork,
        prepared: PreparedCaptureV1,
        command: CommandContext,
    ) -> StoredResponse:
        if not uow.immediate:
            raise RuntimeError("Trajectory capture requires an immediate transaction")
        if not isinstance(prepared, PreparedCaptureV1):
            raise TypeError("prepared must be PreparedCaptureV1")

        bundle = prepared.bundle
        expected_caller = f"agent:{bundle.owner_agent_id}"
        if (
            command.caller_scope != expected_caller
            or command.operation_scope != TRAJECTORY_IMPORT_SCOPE
        ):
            raise ReplayableCommandError(
                code="capture_owner_invalid",
                message="Capture owner does not match the command caller",
                status_code=403,
            )
        owner_exists = await uow.session.scalar(
            select(AgentRow.agent_id).where(
                AgentRow.agent_id == bundle.owner_agent_id
            )
        )
        if owner_exists is None:
            raise ReplayableCommandError(
                code="agent_not_found",
                message="Agent was not found",
                status_code=404,
            )

        receipt = await self._receipt_store.get_by_id(
            session=uow.session,
            receipt_id=command.receipt_id,
        )
        if (
            receipt is None
            or receipt.state != "in_progress"
            or receipt.caller_scope != command.caller_scope
            or receipt.operation_scope != command.operation_scope
            or receipt.idempotency_key != command.idempotency_key
            or receipt.request_hash != command.request_hash
            or receipt.result_resource_type is not None
            or receipt.result_resource_id is not None
        ):
            raise RuntimeError("Capture command receipt is inconsistent")
        captured_at = require_utc(receipt.created_at)
        if captured_at > require_utc(self._clock.now()):
            raise RuntimeError("Capture command receipt is from the future")

        duplicate = await uow.session.scalar(
            select(TrajectoryBundleRow.bundle_id).where(
                TrajectoryBundleRow.owner_agent_id == bundle.owner_agent_id,
                TrajectoryBundleRow.manifest_hash == bundle.manifest_hash,
            )
        )
        if duplicate is not None:
            raise ReplayableCommandError(
                code="trajectory_already_captured",
                message="Trajectory was already captured",
                status_code=409,
            )

        bundle_id = self._id_generator.new()
        await self._insert_source(
            uow,
            TrajectoryBundleRow(
                bundle_id=bundle_id,
                owner_agent_id=bundle.owner_agent_id,
                trajectory_id=bundle.trajectory_id,
                adapter_kind=bundle.adapter.kind,
                adapter_version=bundle.adapter.version,
                sanitization_profile=bundle.sanitization.profile_id,
                manifest=prepared.manifest_json,
                manifest_hash=bundle.manifest_hash,
                source_started_at=bundle.source_started_at,
                source_completed_at=bundle.source_completed_at,
                captured_at=captured_at,
            ),
        )

        unique_evidence = _unique_evidence(prepared.candidates)
        step_ordinals = {step.step_id: step.ordinal for step in bundle.steps}
        evidence_ids = {
            _evidence_key(item): self._id_generator.new()
            for item in unique_evidence
        }
        for item in unique_evidence:
            row = TrajectoryEvidenceRow(
                evidence_id=evidence_ids[_evidence_key(item)],
                bundle_id=bundle_id,
                owner_agent_id=bundle.owner_agent_id,
                step_id=item.step_id,
                field=item.field.value,
                ordinal=step_ordinals[item.step_id],
                excerpt=item.excerpt,
                source_hash=item.source_hash,
                excerpt_hash=item.excerpt_hash,
            )
            await self._insert_source(uow, row)

        candidate_ids = tuple(
            self._id_generator.new() for _ in prepared.candidates
        )
        candidate_evidence_ids: list[tuple[UUID, ...]] = []
        for ordinal, (candidate_id, draft) in enumerate(
            zip(candidate_ids, prepared.candidates, strict=True),
            start=1,
        ):
            referenced_ids = tuple(
                evidence_ids[_evidence_key(item)] for item in draft.evidence
            )
            candidate_evidence_ids.append(referenced_ids)
            await self._insert_source(
                uow,
                _candidate_row(
                    candidate_id=candidate_id,
                    bundle_id=bundle_id,
                    owner_agent_id=bundle.owner_agent_id,
                    ordinal=ordinal,
                    draft=draft,
                    evidence_ids=referenced_ids,
                    created_at=captured_at,
                ),
            )

        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="trajectory_bundle",
            resource_id=bundle_id,
        )
        capture_evidence_ids = tuple(
            evidence_ids[_evidence_key(item)] for item in unique_evidence
        )
        events = [
            PendingEvent(
                aggregate_type="trajectory_bundle",
                aggregate_id=bundle_id,
                event_type=TrajectoryCapturedV1.event_type,
                payload=TrajectoryCapturedV1(
                    schema_version=1,
                    bundle_id=bundle_id,
                    owner_agent_id=bundle.owner_agent_id,
                    manifest_hash=bundle.manifest_hash,
                    evidence_ids=capture_evidence_ids,
                    candidate_ids=candidate_ids,
                ),
                actor_agent_id=bundle.owner_agent_id,
                occurred_at=captured_at,
            ),
            *(
                PendingEvent(
                    aggregate_type="experience_candidate",
                    aggregate_id=candidate_id,
                    event_type=CandidateCreatedV1.event_type,
                    payload=CandidateCreatedV1(
                        schema_version=1,
                        candidate_id=candidate_id,
                        bundle_id=bundle_id,
                        owner_agent_id=bundle.owner_agent_id,
                        content_hash=draft.content_hash,
                        evidence_ids=referenced_ids,
                        decision_after=CandidateDecision.PENDING,
                    ),
                    actor_agent_id=bundle.owner_agent_id,
                    occurred_at=captured_at,
                )
                for candidate_id, draft, referenced_ids in zip(
                    candidate_ids,
                    prepared.candidates,
                    candidate_evidence_ids,
                    strict=True,
                )
            ),
        ]
        await uow.append_events(command=command, events=events)
        return capture_stored_response(
            bundle_id=bundle_id,
            owner_agent_id=bundle.owner_agent_id,
            manifest_hash=bundle.manifest_hash,
            candidate_ids=candidate_ids,
            captured_at=captured_at,
        )

    @staticmethod
    async def _insert_source(uow: UnitOfWork, row: object) -> None:
        uow.session.add(row)
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)


def _evidence_key(item: CapturedEvidenceV1) -> tuple[str, TrajectoryField]:
    return (item.step_id, item.field)


def _unique_evidence(
    candidates: Iterable[CandidateDraftV1],
) -> tuple[CapturedEvidenceV1, ...]:
    retained: dict[tuple[str, TrajectoryField], CapturedEvidenceV1] = {}
    for candidate in candidates:
        for evidence in candidate.evidence:
            key = _evidence_key(evidence)
            existing = retained.get(key)
            if existing is not None and existing != evidence:
                raise ValueError("Evidence location resolved to conflicting excerpts")
            retained.setdefault(key, evidence)
    return tuple(retained.values())


def _candidate_row(
    *,
    candidate_id: UUID,
    bundle_id: UUID,
    owner_agent_id: UUID,
    ordinal: int,
    draft: CandidateDraftV1,
    evidence_ids: tuple[UUID, ...],
    created_at: datetime,
) -> ExperienceCandidateRow:
    content = draft.content
    return ExperienceCandidateRow(
        candidate_id=candidate_id,
        bundle_id=bundle_id,
        owner_agent_id=owner_agent_id,
        candidate_ordinal=ordinal,
        kind=draft.kind,
        body=content.body,
        summary=content.summary,
        mechanism=content.mechanism,
        tags=canonical_json_bytes(content.tags),
        applicability=canonical_json_bytes(content.applicability),
        evidence=canonical_json_bytes(content.evidence),
        evidence_refs=canonical_json_bytes(tuple(str(item) for item in evidence_ids)),
        falsifiers=canonical_json_bytes(content.falsifiers),
        content_hash=draft.content_hash,
        extractor_kind=draft.extractor_kind,
        extractor_configuration_hash=draft.extractor_configuration_hash,
        created_at=created_at,
    )


__all__ = ["CapturePreparer", "CaptureService", "SensitiveInputError"]
