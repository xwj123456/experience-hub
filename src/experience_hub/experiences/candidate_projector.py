"""Fail-closed replay reducer for quarantined candidate state."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes
from experience_hub.capture.responses import capture_stored_response
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry, StoredEvent
from experience_hub.experiences.candidate_events import (
    CandidateAdoptedV1,
    CandidateCreatedV1,
    CandidateRejectedV1,
)
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
    CandidateDecision,
    CandidateViewV1,
)
from experience_hub.experiences.candidate_responses import (
    candidate_stored_response,
    reconstruct_candidate_content,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    CandidateAdoptionRow,
    DomainEventRow,
    ExperienceCandidateRow,
    IdempotencyRecordRow,
    TrajectoryBundleRow,
    TrajectoryEvidenceRow,
)

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class CandidateProjectionIntegrityError(RuntimeError):
    """A candidate event cannot be reconciled with immutable sources."""

    code = "candidate_projection_integrity_error"


def _fail(message: str) -> CandidateProjectionIntegrityError:
    return CandidateProjectionIntegrityError(message)


def _utc(value: datetime) -> str:
    timestamp = require_utc(value)
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _target_table(target_prefix: str | None) -> str:
    if target_prefix is None:
        return 'main."candidate_state"'
    name = f"{target_prefix}candidate_state"
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError("Unsafe candidate projection target")
    return f'temp."{name}"'


async def _create_rebuild_table(
    session: AsyncSession,
    target: str,
) -> None:
    await session.execute(
        text(
            f"CREATE TEMP TABLE {target} ("
            "candidate_id VARCHAR(36) NOT NULL PRIMARY KEY, "
            "owner_agent_id VARCHAR(36) NOT NULL, "
            "decision VARCHAR(8) NOT NULL, "
            "adoption_id VARCHAR(36), "
            "resulting_experience_id VARCHAR(36), "
            "resulting_version_id VARCHAR(36), "
            "reason_code VARCHAR, "
            "reason_text VARCHAR, "
            "reason_text_hash VARCHAR(64), "
            "decided_at VARCHAR(27), "
            "projection_event_id INTEGER NOT NULL, "
            "CHECK (decision IN ('pending', 'adopted', 'rejected')), "
            "CHECK (projection_event_id > 0), "
            "CHECK ((decision = 'pending' "
            "AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NULL) "
            "OR (decision = 'adopted' "
            "AND adoption_id IS NOT NULL "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NOT NULL) "
            "OR (decision = 'rejected' "
            "AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL "
            "AND resulting_version_id IS NULL "
            "AND reason_code IS NOT NULL AND reason_text IS NOT NULL "
            "AND reason_text_hash IS NOT NULL AND decided_at IS NOT NULL))"
            ")"
        )
    )


def _decode_uuid_array(raw: bytes) -> tuple[UUID, ...]:
    try:
        values: Any = json.loads(raw)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or canonical_json_bytes(values) != bytes(raw)
        ):
            raise ValueError("not a canonical UUID array")
        result = tuple(UUID(value) for value in values)
    except (TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise _fail("Candidate evidence references are invalid") from None
    if tuple(str(value) for value in result) != tuple(values):
        raise _fail("Candidate evidence references are invalid")
    return result


async def _require_receipt_anchor(
    session: AsyncSession,
    event: StoredEvent,
    *,
    owner_agent_id: UUID,
    scope: str,
    resource_type: str,
    resource_id: UUID,
) -> IdempotencyRecordRow:
    receipt = await session.get(IdempotencyRecordRow, event.causation_id)
    if (
        receipt is None
        or receipt.caller_scope != f"agent:{owner_agent_id}"
        or receipt.scope != scope
        or receipt.result_resource_type != resource_type
        or receipt.result_resource_id != resource_id
        or receipt.created_at != event.occurred_at
    ):
        raise _fail("Candidate event command receipt is inconsistent")
    in_progress = (
        receipt.state == "in_progress"
        and receipt.completed_at is None
        and receipt.response_status_code is None
        and receipt.response_body is None
        and receipt.response_content_type is None
        and receipt.response_headers is None
    )
    completed = (
        receipt.state == "completed"
        and receipt.completed_at is not None
        and receipt.completed_at >= event.occurred_at
        and receipt.response_status_code is not None
        and receipt.response_body is not None
        and receipt.response_content_type is not None
        and receipt.response_headers is not None
    )
    if not (in_progress or completed):
        raise _fail("Candidate event command receipt is inconsistent")
    return receipt


def _require_completed_response(
    receipt: IdempotencyRecordRow,
    expected_response: StoredResponse,
) -> None:
    if receipt.state == "in_progress":
        return
    if (
        receipt.response_status_code != expected_response.status_code
        or receipt.response_body != expected_response.body
        or receipt.response_content_type != expected_response.content_type
        or receipt.response_headers
        != canonical_json_bytes(dict(expected_response.headers or {}))
    ):
        raise _fail("Candidate event command receipt is inconsistent")


async def _expected_capture_response(
    session: AsyncSession,
    bundle_id: UUID,
    owner_agent_id: UUID,
) -> StoredResponse:
    bundle = await session.get(TrajectoryBundleRow, bundle_id)
    if bundle is None or bundle.owner_agent_id != owner_agent_id:
        raise _fail("Candidate capture response source is missing")
    candidates = tuple(
        (
            await session.scalars(
                select(ExperienceCandidateRow)
                .where(
                    ExperienceCandidateRow.bundle_id == bundle_id,
                    ExperienceCandidateRow.owner_agent_id == owner_agent_id,
                )
                .order_by(
                    ExperienceCandidateRow.candidate_ordinal,
                    ExperienceCandidateRow.candidate_id,
                )
            )
        ).all()
    )
    if any(
        candidate.candidate_ordinal != ordinal
        for ordinal, candidate in enumerate(candidates, start=1)
    ):
        raise _fail("Candidate capture response source is invalid")
    return capture_stored_response(
        bundle_id=bundle.bundle_id,
        owner_agent_id=bundle.owner_agent_id,
        manifest_hash=bundle.manifest_hash,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        captured_at=bundle.captured_at,
    )


async def _expected_decision_response(
    session: AsyncSession,
    event: StoredEvent,
    payload: CandidateAdoptedV1 | CandidateRejectedV1,
) -> StoredResponse:
    candidate = await session.get(ExperienceCandidateRow, payload.candidate_id)
    if candidate is None or candidate.owner_agent_id != payload.owner_agent_id:
        raise _fail("Candidate decision response source is missing")
    bundle = await session.get(TrajectoryBundleRow, candidate.bundle_id)
    if bundle is None or bundle.owner_agent_id != payload.owner_agent_id:
        raise _fail("Candidate decision response source is missing")
    try:
        evidence_ids = _decode_uuid_array(candidate.evidence_refs)
        rows = tuple(
            (
                await session.scalars(
                    select(TrajectoryEvidenceRow).where(
                        TrajectoryEvidenceRow.owner_agent_id
                        == payload.owner_agent_id,
                        TrajectoryEvidenceRow.bundle_id == candidate.bundle_id,
                        TrajectoryEvidenceRow.evidence_id.in_(evidence_ids),
                    )
                )
            ).all()
        )
        manifest = capture_source_integrity.authenticate_trajectory_manifest(bundle)
        evidence = capture_source_integrity.reconstruct_captured_evidence(
            bundle=bundle,
            rows=rows,
            evidence_ids=evidence_ids,
            owner_agent_id=payload.owner_agent_id,
            manifest=manifest,
        )
        content = reconstruct_candidate_content(candidate)
        if isinstance(payload, CandidateAdoptedV1):
            return candidate_stored_response(
                CandidateViewV1(
                    candidate_id=candidate.candidate_id,
                    bundle_id=candidate.bundle_id,
                    owner_agent_id=candidate.owner_agent_id,
                    decision=CandidateDecision.ADOPTED,
                    kind=candidate.kind,
                    content=content,
                    content_hash=candidate.content_hash,
                    evidence=evidence,
                    extractor_kind=candidate.extractor_kind,
                    extractor_configuration_hash=(
                        candidate.extractor_configuration_hash
                    ),
                    resulting_experience_id=payload.resulting_experience_id,
                    resulting_version_id=payload.resulting_version_id,
                    reason=None,
                    created_at=candidate.created_at,
                    decided_at=event.occurred_at,
                )
            )
        return candidate_stored_response(
            CandidateViewV1(
                candidate_id=candidate.candidate_id,
                bundle_id=candidate.bundle_id,
                owner_agent_id=candidate.owner_agent_id,
                decision=CandidateDecision.REJECTED,
                kind=candidate.kind,
                content=content,
                content_hash=candidate.content_hash,
                evidence=evidence,
                extractor_kind=candidate.extractor_kind,
                extractor_configuration_hash=(
                    candidate.extractor_configuration_hash
                ),
                resulting_experience_id=None,
                resulting_version_id=None,
                reason=payload.reason,
                created_at=candidate.created_at,
                decided_at=event.occurred_at,
            )
        )
    except (KeyError, TypeError, ValueError):
        raise _fail("Candidate decision response source is invalid") from None


class CandidateStateProjector:
    """Replay candidate decisions without trusting the online projection."""

    name = "candidate_state"
    version = 1
    event_types = frozenset(
        {
            CandidateCreatedV1.event_type,
            CandidateAdoptedV1.event_type,
            CandidateRejectedV1.event_type,
        }
    )

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    def stored_event_from_row(self, row: DomainEventRow) -> StoredEvent:
        try:
            payload = self._event_registry.decode(
                event_type=row.event_type,
                payload=row.payload,
            )
        except (TypeError, ValueError):
            raise _fail("Candidate event payload is invalid") from None
        return StoredEvent(
            event_id=row.event_id,
            aggregate_type=row.aggregate_type,
            aggregate_id=row.aggregate_id,
            sequence=row.sequence,
            event_type=row.event_type,
            payload=payload,
            actor_agent_id=row.actor_agent_id,
            causation_id=row.causation_id,
            occurred_at=row.occurred_at,
        )

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target_prefix=None)
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        await _create_rebuild_table(session, _target_table(target_prefix))
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type.in_(self.event_types))
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        for row in rows:
            await self._apply(
                session,
                self.stored_event_from_row(row),
                target_prefix=target_prefix,
            )

    async def _apply(
        self,
        session: AsyncSession,
        event: StoredEvent,
        *,
        target_prefix: str | None,
    ) -> None:
        if event.aggregate_type != "experience_candidate":
            raise _fail("Candidate event has an invalid aggregate anchor")
        target = _target_table(target_prefix)
        if isinstance(event.payload, CandidateCreatedV1):
            await self._apply_created(session, event, event.payload, target)
            return
        if isinstance(event.payload, CandidateAdoptedV1):
            await self._apply_adopted(session, event, event.payload, target)
            return
        if isinstance(event.payload, CandidateRejectedV1):
            await self._apply_rejected(session, event, event.payload, target)
            return
        raise _fail("Candidate reducer received an unsupported event")

    async def _apply_created(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CandidateCreatedV1,
        target: str,
    ) -> None:
        candidate = await session.get(ExperienceCandidateRow, payload.candidate_id)
        if (
            event.event_type != CandidateCreatedV1.event_type
            or event.aggregate_id != payload.candidate_id
            or event.sequence != 1
            or event.actor_agent_id != payload.owner_agent_id
            or candidate is None
            or candidate.bundle_id != payload.bundle_id
            or candidate.owner_agent_id != payload.owner_agent_id
            or candidate.content_hash != payload.content_hash
            or candidate.created_at != event.occurred_at
            or _decode_uuid_array(candidate.evidence_refs) != payload.evidence_ids
        ):
            raise _fail("Candidate created event source anchor is inconsistent")
        receipt = await _require_receipt_anchor(
            session,
            event,
            owner_agent_id=payload.owner_agent_id,
            scope=TRAJECTORY_IMPORT_SCOPE,
            resource_type="trajectory_bundle",
            resource_id=payload.bundle_id,
        )
        # Source validation authenticates the ordered, same-causation candidate
        # closure, so the bundle-scoped completed response is checked once.
        if receipt.state == "completed" and candidate.candidate_ordinal == 1:
            _require_completed_response(
                receipt,
                await _expected_capture_response(
                    session,
                    payload.bundle_id,
                    payload.owner_agent_id,
                ),
            )
        try:
            result = await session.execute(
                text(
                    f"INSERT INTO {target} ("
                    "candidate_id, owner_agent_id, decision, adoption_id, "
                    "resulting_experience_id, resulting_version_id, reason_code, "
                    "reason_text, reason_text_hash, decided_at, projection_event_id"
                    ") VALUES ("
                    ":candidate_id, :owner_agent_id, 'pending', NULL, NULL, NULL, "
                    "NULL, NULL, NULL, NULL, :event_id)"
                ),
                {
                    "candidate_id": str(payload.candidate_id),
                    "owner_agent_id": str(payload.owner_agent_id),
                    "event_id": event.event_id,
                },
            )
        except SQLAlchemyError:
            raise _fail("Candidate pending state could not be created") from None
        if cast(Any, result).rowcount != 1:
            raise _fail("Candidate pending state was not created exactly once")

    async def _pending_event_id(
        self,
        session: AsyncSession,
        *,
        candidate_id: UUID,
        owner_agent_id: UUID,
    ) -> int:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.aggregate_type == "experience_candidate",
                        DomainEventRow.aggregate_id == candidate_id,
                        DomainEventRow.sequence == 1,
                        DomainEventRow.event_type == CandidateCreatedV1.event_type,
                    )
                )
            ).all()
        )
        if len(rows) != 1:
            raise _fail("Candidate decision has no unique created event")
        created = self.stored_event_from_row(rows[0])
        payload = created.payload
        if (
            not isinstance(payload, CandidateCreatedV1)
            or payload.candidate_id != candidate_id
            or payload.owner_agent_id != owner_agent_id
        ):
            raise _fail("Candidate decision created event is inconsistent")
        return created.event_id

    async def _require_pending(
        self,
        session: AsyncSession,
        *,
        event: StoredEvent,
        candidate_id: UUID,
        owner_agent_id: UUID,
        target: str,
    ) -> int:
        if (
            event.aggregate_id != candidate_id
            or event.sequence != 2
            or event.actor_agent_id != owner_agent_id
        ):
            raise _fail("Candidate decision event anchor is inconsistent")
        pending_event_id = await self._pending_event_id(
            session,
            candidate_id=candidate_id,
            owner_agent_id=owner_agent_id,
        )
        row = (
            await session.execute(
                text(
                    f"SELECT owner_agent_id, decision, projection_event_id "
                    f"FROM {target} WHERE candidate_id = :candidate_id"
                ),
                {"candidate_id": str(candidate_id)},
            )
        ).mappings().one_or_none()
        if (
            row is None
            or row["owner_agent_id"] != str(owner_agent_id)
            or row["decision"] != CandidateDecision.PENDING.value
            or row["projection_event_id"] != pending_event_id
        ):
            raise _fail("Candidate decision does not match pending state")
        return pending_event_id

    async def _apply_adopted(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CandidateAdoptedV1,
        target: str,
    ) -> None:
        if event.event_type != CandidateAdoptedV1.event_type:
            raise _fail("Candidate adoption has an invalid event type")
        before_event_id = await self._require_pending(
            session,
            event=event,
            candidate_id=payload.candidate_id,
            owner_agent_id=payload.owner_agent_id,
            target=target,
        )
        adoption = await session.get(CandidateAdoptionRow, payload.adoption_id)
        if (
            adoption is None
            or adoption.candidate_id != payload.candidate_id
            or adoption.owner_agent_id != payload.owner_agent_id
            or adoption.resulting_experience_id != payload.resulting_experience_id
            or adoption.resulting_version_id != payload.resulting_version_id
            or adoption.resulting_content_hash != payload.resulting_content_hash
            or adoption.created is not payload.created
            or adoption.adopted_at != event.occurred_at
        ):
            raise _fail("Candidate adoption source anchor is inconsistent")
        receipt = await _require_receipt_anchor(
            session,
            event,
            owner_agent_id=payload.owner_agent_id,
            scope=CANDIDATE_ADOPT_SCOPE,
            resource_type="candidate_adoption",
            resource_id=payload.adoption_id,
        )
        if receipt.state == "completed":
            _require_completed_response(
                receipt,
                await _expected_decision_response(session, event, payload),
            )
        try:
            result = await session.execute(
                text(
                    f"UPDATE {target} SET decision = 'adopted', "
                    "adoption_id = :adoption_id, "
                    "resulting_experience_id = :experience_id, "
                    "resulting_version_id = :version_id, decided_at = :decided_at, "
                    "projection_event_id = :event_id "
                    "WHERE candidate_id = :candidate_id "
                    "AND owner_agent_id = :owner_agent_id "
                    "AND decision = 'pending' "
                    "AND projection_event_id = :before_event_id"
                ),
                {
                    "adoption_id": str(payload.adoption_id),
                    "experience_id": str(payload.resulting_experience_id),
                    "version_id": str(payload.resulting_version_id),
                    "decided_at": _utc(event.occurred_at),
                    "event_id": event.event_id,
                    "candidate_id": str(payload.candidate_id),
                    "owner_agent_id": str(payload.owner_agent_id),
                    "before_event_id": before_event_id,
                },
            )
        except SQLAlchemyError:
            raise _fail("Candidate adoption compare-and-set failed") from None
        if cast(Any, result).rowcount != 1:
            raise _fail("Candidate adoption compare-and-set failed")

    async def _apply_rejected(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: CandidateRejectedV1,
        target: str,
    ) -> None:
        if event.event_type != CandidateRejectedV1.event_type:
            raise _fail("Candidate rejection has an invalid event type")
        before_event_id = await self._require_pending(
            session,
            event=event,
            candidate_id=payload.candidate_id,
            owner_agent_id=payload.owner_agent_id,
            target=target,
        )
        receipt = await _require_receipt_anchor(
            session,
            event,
            owner_agent_id=payload.owner_agent_id,
            scope=CANDIDATE_REJECT_SCOPE,
            resource_type="experience_candidate",
            resource_id=payload.candidate_id,
        )
        if receipt.state == "completed":
            _require_completed_response(
                receipt,
                await _expected_decision_response(session, event, payload),
            )
        try:
            result = await session.execute(
                text(
                    f"UPDATE {target} SET decision = 'rejected', "
                    "reason_code = :reason_code, reason_text = :reason_text, "
                    "reason_text_hash = :reason_text_hash, "
                    "decided_at = :decided_at, projection_event_id = :event_id "
                    "WHERE candidate_id = :candidate_id "
                    "AND owner_agent_id = :owner_agent_id "
                    "AND decision = 'pending' "
                    "AND projection_event_id = :before_event_id"
                ),
                {
                    "reason_code": payload.reason.code,
                    "reason_text": payload.reason.text,
                    "reason_text_hash": payload.reason.text_hash,
                    "decided_at": _utc(event.occurred_at),
                    "event_id": event.event_id,
                    "candidate_id": str(payload.candidate_id),
                    "owner_agent_id": str(payload.owner_agent_id),
                    "before_event_id": before_event_id,
                },
            )
        except SQLAlchemyError:
            raise _fail("Candidate rejection compare-and-set failed") from None
        if cast(Any, result).rowcount != 1:
            raise _fail("Candidate rejection compare-and-set failed")


__all__ = ["CandidateProjectionIntegrityError", "CandidateStateProjector"]
