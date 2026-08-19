"""Immutable owner-scoped candidate read models and query service."""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes
from experience_hub.capture.hashing import extractor_configuration_hash
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import (
    CommandContext,
    PendingEvent,
    ReplayableCommandError,
)
from experience_hub.domain.values import StructuredReason, TypedEvidence
from experience_hub.errors import DomainError
from experience_hub.experiences.candidate_events import (
    CandidateAdoptedV1,
    CandidateRejectedV1,
)
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
    AdoptCandidate,
    CandidateDecision,
    CandidateDecisionScope,
    CandidatePageV1,
    CandidateViewV1,
    RejectCandidate,
)
from experience_hub.experiences.candidate_repository import (
    CandidateRecord,
    CandidateRepository,
)
from experience_hub.experiences.candidate_responses import (
    candidate_stored_response,
)
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.contracts import ExperienceDraft
from experience_hub.experiences.models import (
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.repository import ExperienceWriter
from experience_hub.ids import IdGenerator
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import ReceiptStore, StoredResponse
from experience_hub.storage.tables import (
    CandidateAdoptionRow,
    TrajectoryEvidenceRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

_CURSOR_TEXT = re.compile(r"[A-Za-z0-9_-]+\Z")
_CURSOR_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z"
)


class CandidateService:
    """Read or explicitly decide owner-scoped quarantined candidates."""

    def __init__(
        self,
        *,
        repository: CandidateRepository,
        experience_writer: ExperienceWriter | None = None,
        receipt_store: ReceiptStore | None = None,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self._repository = repository
        self._experience_writer = experience_writer
        self._receipt_store = receipt_store
        self._clock = clock
        self._id_generator = id_generator

    async def adopt(
        self,
        *,
        uow: UnitOfWork,
        request: AdoptCandidate,
        command: CommandContext,
    ) -> StoredResponse:
        writer, receipt_store, clock, id_generator = self._decision_dependencies()
        record, decided_at = await self._pending_decision(
            uow=uow,
            owner_agent_id=request.owner_agent_id,
            candidate_id=request.candidate_id,
            command=command,
            operation_scope=CANDIDATE_ADOPT_SCOPE,
            receipt_store=receipt_store,
            clock=clock,
        )
        try:
            manifest = capture_source_integrity.authenticate_trajectory_manifest(
                record.bundle
            )
            pending = await self._view(
                session=uow.session,
                owner_agent_id=request.owner_agent_id,
                record=record,
                manifest=manifest,
            )
        except (
            SourceIntegrityError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _lineage_invalid() from None

        equivalent = await writer.find_current_equivalent(
            session=uow.session,
            owner_agent_id=request.owner_agent_id,
            content_hash=pending.content_hash,
        )
        adoption_id = id_generator.new()
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="candidate_adoption",
            resource_id=adoption_id,
        )
        created = equivalent is None
        if equivalent is None:
            creation = await writer.create_from_draft(
                uow=uow,
                draft=ExperienceDraft(
                    owner_agent_id=request.owner_agent_id,
                    actor_agent_id=request.owner_agent_id,
                    kind=pending.kind,
                    origin=ExperienceOrigin.ADOPTED_CANDIDATE,
                    content=pending.content,
                    importance=request.importance,
                    confidence=request.confidence,
                    source_trust=0.5,
                    initial_temperature=Temperature.WARM,
                    links=(),
                    occurred_at=decided_at,
                ),
                command=command,
            )
            experience_id = creation.experience_id
            version_id = creation.version_id
            content_hash = creation.content_hash
        else:
            experience_id = equivalent.experience_id
            version_id = equivalent.current_version_id
            content_hash = equivalent.current_content_hash
        if content_hash != pending.content_hash:
            raise _lineage_invalid()

        uow.session.add(
            CandidateAdoptionRow(
                adoption_id=adoption_id,
                candidate_id=request.candidate_id,
                owner_agent_id=request.owner_agent_id,
                resulting_experience_id=experience_id,
                resulting_version_id=version_id,
                resulting_content_hash=content_hash,
                created=created,
                adopted_at=decided_at,
            )
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="experience_candidate",
                    aggregate_id=request.candidate_id,
                    event_type=CandidateAdoptedV1.event_type,
                    payload=CandidateAdoptedV1(
                        schema_version=1,
                        candidate_id=request.candidate_id,
                        owner_agent_id=request.owner_agent_id,
                        decision_before=CandidateDecision.PENDING,
                        decision_after=CandidateDecision.ADOPTED,
                        adoption_id=adoption_id,
                        resulting_experience_id=experience_id,
                        resulting_version_id=version_id,
                        resulting_content_hash=content_hash,
                        created=created,
                    ),
                    actor_agent_id=request.owner_agent_id,
                    occurred_at=decided_at,
                ),
            ),
        )
        return await self._decision_response(
            uow=uow,
            owner_agent_id=request.owner_agent_id,
            candidate_id=request.candidate_id,
            manifest=manifest,
        )

    async def reject(
        self,
        *,
        uow: UnitOfWork,
        request: RejectCandidate,
        command: CommandContext,
    ) -> StoredResponse:
        _, receipt_store, clock, _ = self._decision_dependencies()
        record, decided_at = await self._pending_decision(
            uow=uow,
            owner_agent_id=request.owner_agent_id,
            candidate_id=request.candidate_id,
            command=command,
            operation_scope=CANDIDATE_REJECT_SCOPE,
            receipt_store=receipt_store,
            clock=clock,
        )
        try:
            manifest = capture_source_integrity.authenticate_trajectory_manifest(
                record.bundle
            )
            await self._view(
                session=uow.session,
                owner_agent_id=request.owner_agent_id,
                record=record,
                manifest=manifest,
            )
        except (
            SourceIntegrityError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _lineage_invalid() from None
        await receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="experience_candidate",
            resource_id=request.candidate_id,
        )
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="experience_candidate",
                    aggregate_id=request.candidate_id,
                    event_type=CandidateRejectedV1.event_type,
                    payload=CandidateRejectedV1(
                        schema_version=1,
                        candidate_id=request.candidate_id,
                        owner_agent_id=request.owner_agent_id,
                        decision_before=CandidateDecision.PENDING,
                        decision_after=CandidateDecision.REJECTED,
                        reason=request.reason,
                    ),
                    actor_agent_id=request.owner_agent_id,
                    occurred_at=decided_at,
                ),
            ),
        )
        return await self._decision_response(
            uow=uow,
            owner_agent_id=request.owner_agent_id,
            candidate_id=request.candidate_id,
            manifest=manifest,
        )

    def _decision_dependencies(
        self,
    ) -> tuple[ExperienceWriter, ReceiptStore, Clock, IdGenerator]:
        if (
            self._experience_writer is None
            or self._receipt_store is None
            or self._clock is None
            or self._id_generator is None
        ):
            raise RuntimeError("Candidate decision dependencies are unavailable")
        return (
            self._experience_writer,
            self._receipt_store,
            self._clock,
            self._id_generator,
        )

    async def _pending_decision(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        candidate_id: UUID,
        command: CommandContext,
        operation_scope: CandidateDecisionScope,
        receipt_store: ReceiptStore,
        clock: Clock,
    ) -> tuple[CandidateRecord, datetime]:
        if not uow.immediate:
            raise RuntimeError("Candidate decisions require an immediate transaction")
        if (
            command.caller_scope != f"agent:{owner_agent_id}"
            or command.operation_scope != operation_scope
        ):
            raise _candidate_not_found()
        record = await self._repository.lock_owned(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            candidate_id=candidate_id,
        )
        if record is None:
            raise _candidate_not_found()
        if record.state.decision != CandidateDecision.PENDING.value:
            raise ReplayableCommandError(
                code="candidate_already_decided",
                message="Candidate already has a terminal decision",
                status_code=409,
            )
        receipt = await receipt_store.get_by_id(
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
            raise RuntimeError("Candidate decision receipt is inconsistent")
        decided_at = require_utc(receipt.created_at)
        if decided_at > require_utc(clock.now()):
            raise RuntimeError("Candidate decision receipt is from the future")
        return record, decided_at

    async def _decision_response(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        candidate_id: UUID,
        manifest: capture_source_integrity.AuthenticatedTrajectoryManifest,
    ) -> StoredResponse:
        record = await self._repository.find_owned(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            candidate_id=candidate_id,
        )
        if record is None:
            raise RuntimeError("Candidate decision projection is missing")
        view = await self._view(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            record=record,
            manifest=manifest,
        )
        return candidate_stored_response(view)

    async def list_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        decision: CandidateDecision | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> CandidatePageV1:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if decision is not None and not isinstance(decision, CandidateDecision):
            raise ValueError("decision must be CandidateDecision or None")
        after = None if cursor is None else _decode_cursor(cursor)
        records = await self._repository.list_owned(
            session=session,
            owner_agent_id=owner_agent_id,
            decision=decision,
            limit=limit + 1,
            after=after,
        )
        retained = records[:limit]
        evidence_by_candidate: dict[UUID, tuple[TrajectoryEvidenceRow, ...]] = {}
        evidence_ids_by_candidate: dict[UUID, tuple[UUID, ...]] = {}
        manifests_by_bundle: dict[
            tuple[UUID, UUID, str],
            capture_source_integrity.AuthenticatedTrajectoryManifest,
        ] = {}
        for record in retained:
            try:
                evidence_ids_by_candidate[record.candidate.candidate_id] = (
                    _uuid_array(record.candidate.evidence_refs)
                )
                bundle_key = (
                    record.bundle.bundle_id,
                    record.bundle.owner_agent_id,
                    record.bundle.manifest_hash,
                )
                if bundle_key not in manifests_by_bundle:
                    manifests_by_bundle[bundle_key] = (
                        capture_source_integrity.authenticate_trajectory_manifest(
                            record.bundle
                        )
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                raise _invalid_source(record.candidate.candidate_id) from None
        all_evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for evidence_ids in evidence_ids_by_candidate.values()
                for evidence_id in evidence_ids
            )
        )
        evidence_rows = await self._repository.evidence_owned_batch(
            session=session,
            owner_agent_id=owner_agent_id,
            evidence_ids=all_evidence_ids,
        )
        rows_by_id = {row.evidence_id: row for row in evidence_rows}
        if len(rows_by_id) != len(evidence_rows):
            raise SourceIntegrityError(
                "Candidate source could not be reconstructed",
                mismatch_key="trajectory_evidence",
            )
        for record in retained:
            evidence_by_candidate[record.candidate.candidate_id] = tuple(
                rows_by_id[evidence_id]
                for evidence_id in evidence_ids_by_candidate[
                    record.candidate.candidate_id
                ]
                if evidence_id in rows_by_id
            )
        items = tuple(
            [
                await self._view(
                    session=session,
                    owner_agent_id=owner_agent_id,
                    record=record,
                    evidence_rows=evidence_by_candidate[
                        record.candidate.candidate_id
                    ],
                    manifest=manifests_by_bundle[
                        (
                            record.bundle.bundle_id,
                            record.bundle.owner_agent_id,
                            record.bundle.manifest_hash,
                        )
                    ],
                )
                for record in retained
            ]
        )
        next_cursor = None
        if len(records) > limit:
            last = retained[-1].candidate
            next_cursor = _encode_cursor(last.created_at, last.candidate_id)
        return CandidatePageV1(items=items, next_cursor=next_cursor)

    async def get_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        candidate_id: UUID,
    ) -> CandidateViewV1:
        record = await self._repository.find_owned(
            session=session,
            owner_agent_id=owner_agent_id,
            candidate_id=candidate_id,
        )
        if record is None:
            raise _candidate_not_found()
        return await self._view(
            session=session,
            owner_agent_id=owner_agent_id,
            record=record,
        )

    async def _view(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        record: CandidateRecord,
        evidence_rows: tuple[TrajectoryEvidenceRow, ...] | None = None,
        manifest: (
            capture_source_integrity.AuthenticatedTrajectoryManifest | None
        ) = None,
    ) -> CandidateViewV1:
        candidate = record.candidate
        try:
            tags = _string_array(candidate.tags)
            applicability = _string_array(candidate.applicability)
            falsifiers = _string_array(candidate.falsifiers)
            typed_evidence = tuple(
                TypedEvidence.model_validate(item)
                for item in _object_array(candidate.evidence)
            )
            evidence_ids = _uuid_array(candidate.evidence_refs)
            if len(evidence_ids) != len(typed_evidence):
                raise ValueError("Candidate evidence lengths differ")
            content = VersionContent(
                body=candidate.body,
                summary=candidate.summary,
                mechanism=candidate.mechanism,
                tags=tags,
                applicability=applicability,
                evidence=typed_evidence,
                falsifiers=falsifiers,
            )
            if (
                candidate.extractor_kind != "deterministic_signal_v1"
                or candidate.extractor_configuration_hash
                != extractor_configuration_hash()
                or canonical_json_bytes(content.tags) != candidate.tags
                or canonical_json_bytes(content.applicability)
                != candidate.applicability
                or canonical_json_bytes(content.evidence) != candidate.evidence
                or canonical_json_bytes(content.falsifiers) != candidate.falsifiers
                or encode_version_content(
                    kind=candidate.kind,
                    content=content,
                ).content_hash
                != candidate.content_hash
            ):
                raise ValueError("Candidate content is not canonical")
        except (TypeError, ValueError, json.JSONDecodeError):
            raise _invalid_source(candidate.candidate_id) from None

        if evidence_rows is None:
            evidence_rows = await self._repository.evidence_owned(
                session=session,
                owner_agent_id=owner_agent_id,
                bundle_id=candidate.bundle_id,
                evidence_ids=evidence_ids,
            )
        try:
            authenticated_manifest = (
                capture_source_integrity.authenticate_trajectory_manifest(
                    record.bundle
                )
                if manifest is None
                else manifest
            )
            evidence = capture_source_integrity.reconstruct_captured_evidence(
                bundle=record.bundle,
                rows=evidence_rows,
                evidence_ids=evidence_ids,
                owner_agent_id=owner_agent_id,
                manifest=authenticated_manifest,
            )
            expected = VersionContent(
                body=content.body,
                summary=content.summary,
                mechanism=content.mechanism,
                tags=content.tags,
                applicability=content.applicability,
                evidence=tuple(
                    TypedEvidence(
                        type="trajectory_field",
                        id=(
                            f"{record.bundle.manifest_hash}:{item.step_id}:"
                            f"{item.field.value}"
                        ),
                    )
                    for item in evidence
                ),
                falsifiers=content.falsifiers,
            ).evidence
            if content.evidence != expected:
                raise ValueError("Candidate evidence lineage differs")
            decision, reason = _decision(record)
            return CandidateViewV1(
                candidate_id=candidate.candidate_id,
                bundle_id=candidate.bundle_id,
                owner_agent_id=candidate.owner_agent_id,
                decision=decision,
                kind=candidate.kind,
                content=content,
                content_hash=candidate.content_hash,
                evidence=evidence,
                extractor_kind=candidate.extractor_kind,
                extractor_configuration_hash=(
                    candidate.extractor_configuration_hash
                ),
                resulting_experience_id=record.state.resulting_experience_id,
                resulting_version_id=record.state.resulting_version_id,
                reason=reason,
                created_at=require_utc(candidate.created_at),
                decided_at=(
                    None
                    if record.state.decided_at is None
                    else require_utc(record.state.decided_at)
                ),
            )
        except (TypeError, ValueError):
            raise _invalid_source(candidate.candidate_id) from None


def _invalid_source(candidate_id: UUID) -> SourceIntegrityError:
    return SourceIntegrityError(
        "Candidate source could not be reconstructed",
        mismatch_key=f"experience_candidate:{candidate_id}",
    )


def _candidate_not_found() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="candidate_not_found",
        message="Candidate was not found",
        status_code=404,
    )


def _lineage_invalid() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="candidate_lineage_invalid",
        message="Candidate lineage could not be preserved",
        status_code=409,
    )


def _json_array(raw: bytes) -> list[object]:
    value = json.loads(raw)
    if not isinstance(value, list) or canonical_json_bytes(value) != bytes(raw):
        raise ValueError("Candidate array is not canonical")
    return value


def _string_array(raw: bytes) -> tuple[str, ...]:
    values = _json_array(raw)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Candidate string array is invalid")
    return tuple(value for value in values if isinstance(value, str))


def _object_array(raw: bytes) -> tuple[dict[str, object], ...]:
    values = _json_array(raw)
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("Candidate object array is invalid")
    return tuple(value for value in values if isinstance(value, dict))


def _uuid_array(raw: bytes) -> tuple[UUID, ...]:
    values = _string_array(raw)
    result = tuple(UUID(value) for value in values)
    if (
        tuple(str(value) for value in result) != values
        or len(result) != len(set(result))
    ):
        raise ValueError("Candidate UUID array is invalid")
    return result


def _decision(
    record: CandidateRecord,
) -> tuple[CandidateDecision, StructuredReason | None]:
    state = record.state
    decision = CandidateDecision(state.decision)
    has_result = (
        state.resulting_experience_id is not None
        and state.resulting_version_id is not None
    )
    has_no_result = (
        state.resulting_experience_id is None
        and state.resulting_version_id is None
    )
    has_reason = (
        state.reason_code is not None
        and state.reason_text is not None
        and state.reason_text_hash is not None
    )
    has_no_reason = (
        state.reason_code is None
        and state.reason_text is None
        and state.reason_text_hash is None
    )
    if decision is CandidateDecision.PENDING:
        if not (
            state.adoption_id is None
            and has_no_result
            and has_no_reason
            and state.decided_at is None
        ):
            raise ValueError("Pending candidate state is inconsistent")
    elif decision is CandidateDecision.ADOPTED:
        if not (
            state.adoption_id is not None
            and has_result
            and has_no_reason
            and state.decided_at is not None
        ):
            raise ValueError("Adopted candidate state is inconsistent")
    elif not (
        state.adoption_id is None
        and has_no_result
        and has_reason
        and state.decided_at is not None
    ):
        raise ValueError("Rejected candidate state is inconsistent")
    reason = None
    if decision is CandidateDecision.REJECTED:
        assert state.reason_code is not None
        assert state.reason_text is not None
        assert state.reason_text_hash is not None
        reason = StructuredReason(
            code=state.reason_code,
            text=state.reason_text,
            text_hash=state.reason_text_hash,
        )
    return decision, reason


def _timestamp(value: datetime) -> str:
    return require_utc(value).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _encode_cursor(created_at: datetime, candidate_id: UUID) -> str:
    raw = canonical_json_bytes(
        {
            "candidate_id": str(candidate_id),
            "created_at": _timestamp(created_at),
        }
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _invalid_cursor() -> DomainError:
    return DomainError(
        code="invalid_cursor",
        message="The cursor is invalid.",
        status_code=400,
    )


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        if (
            not isinstance(cursor, str)
            or not cursor
            or _CURSOR_TEXT.fullmatch(cursor) is None
        ):
            raise ValueError("Cursor text is invalid")
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        document = json.loads(raw)
        if (
            not isinstance(document, dict)
            or set(document) != {"candidate_id", "created_at"}
            or not isinstance(document["candidate_id"], str)
            or not isinstance(document["created_at"], str)
            or canonical_json_bytes(document) != raw
            or _CURSOR_TIMESTAMP.fullmatch(document["created_at"]) is None
        ):
            raise ValueError("Cursor document is invalid")
        candidate_id = UUID(document["candidate_id"])
        created_at = datetime.fromisoformat(
            document["created_at"].replace("Z", "+00:00")
        ).astimezone(UTC)
        if (
            str(candidate_id) != document["candidate_id"]
            or _timestamp(created_at) != document["created_at"]
            or _encode_cursor(created_at, candidate_id) != cursor
        ):
            raise ValueError("Cursor values are not canonical")
        return created_at, candidate_id
    except (
        binascii.Error,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        raise _invalid_cursor() from None


__all__ = ["CandidatePageV1", "CandidateService", "CandidateViewV1"]
