"""Fail-closed validation for immutable capture and candidate sources."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes
from experience_hub.capture.hashing import extractor_configuration_hash
from experience_hub.capture.models import (
    CapturedEvidenceV1,
)
from experience_hub.capture.responses import capture_stored_response
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.domain import (
    EventPayload,
    EventRegistry,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.experiences.candidate_events import (
    CANDIDATE_EVENT_TYPES,
    CandidateAdoptedV1,
    CandidateCreatedV1,
    CandidateRejectedV1,
    TrajectoryCapturedV1,
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
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import ExperienceOrigin, VersionContent
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
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
from experience_hub.storage.validation import SourceIntegrityError, SourceValidator


def _fail(kind: str, identifier: UUID) -> SourceIntegrityError:
    return SourceIntegrityError(
        "Capture source graph is inconsistent",
        mismatch_key=f"{kind}:{identifier}",
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_array(raw: bytes) -> list[Any]:
    value: Any = json.loads(raw)
    if not isinstance(value, list) or canonical_json_bytes(value) != bytes(raw):
        raise ValueError("not a canonical array")
    return value


def _uuid_array(raw: bytes) -> tuple[UUID, ...]:
    values = _canonical_array(raw)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("not a UUID array")
    result = tuple(UUID(value) for value in values)
    if tuple(str(value) for value in result) != tuple(values):
        raise ValueError("UUID array is not canonical")
    if len(result) != len(set(result)):
        raise ValueError("UUID array repeats an identifier")
    return result


def _manifest_document(
    row: TrajectoryBundleRow,
) -> capture_source_integrity.AuthenticatedTrajectoryManifest:
    return capture_source_integrity.authenticate_trajectory_manifest(row)


async def _require_receipt(
    session: AsyncSession,
    event: DomainEventRow,
    *,
    owner_agent_id: UUID,
    scope: str,
    resource_type: str,
    resource_id: UUID,
    expected_response: StoredResponse,
) -> None:
    receipt = await session.get(IdempotencyRecordRow, event.causation_id)
    if (
        receipt is None
        or receipt.state != "completed"
        or receipt.caller_scope != f"agent:{owner_agent_id}"
        or receipt.scope != scope
        or receipt.result_resource_type != resource_type
        or receipt.result_resource_id != resource_id
        or receipt.completed_at is None
        or receipt.created_at != event.occurred_at
        or receipt.completed_at < event.occurred_at
        or receipt.response_status_code != expected_response.status_code
        or receipt.response_body != expected_response.body
        or receipt.response_content_type != expected_response.content_type
        or receipt.response_headers
        != canonical_json_bytes(dict(expected_response.headers or {}))
    ):
        raise _fail("capture_receipt", event.causation_id)


def _candidate_view(
    *,
    candidate: ExperienceCandidateRow,
    content: VersionContent,
    evidence: tuple[CapturedEvidenceV1, ...],
    decision: CandidateDecision,
    resulting_experience_id: UUID | None,
    resulting_version_id: UUID | None,
    reason: StructuredReason | None,
    decided_at: datetime | None,
) -> CandidateViewV1:
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
        extractor_configuration_hash=candidate.extractor_configuration_hash,
        resulting_experience_id=resulting_experience_id,
        resulting_version_id=resulting_version_id,
        reason=reason,
        created_at=candidate.created_at,
        decided_at=decided_at,
    )


@dataclass(frozen=True, slots=True)
class _CaptureEventIndex:
    captured_by_bundle: dict[
        UUID,
        tuple[tuple[DomainEventRow, TrajectoryCapturedV1], ...],
    ]
    candidate_events: dict[
        UUID,
        tuple[
            tuple[
                DomainEventRow,
                CandidateCreatedV1 | CandidateAdoptedV1 | CandidateRejectedV1,
            ],
            ...,
        ],
    ]


def _index_capture_events(
    decoded_events: list[tuple[DomainEventRow, EventPayload]],
) -> _CaptureEventIndex:
    captured: dict[
        UUID,
        list[tuple[DomainEventRow, TrajectoryCapturedV1]],
    ] = defaultdict(list)
    by_candidate: dict[
        UUID,
        list[
            tuple[
                DomainEventRow,
                CandidateCreatedV1 | CandidateAdoptedV1 | CandidateRejectedV1,
            ]
        ],
    ] = defaultdict(list)
    for event, payload in decoded_events:
        if isinstance(payload, TrajectoryCapturedV1):
            captured[payload.bundle_id].append((event, payload))
        elif isinstance(
            payload,
            (CandidateCreatedV1, CandidateAdoptedV1, CandidateRejectedV1),
        ):
            by_candidate[payload.candidate_id].append((event, payload))
    return _CaptureEventIndex(
        captured_by_bundle={key: tuple(value) for key, value in captured.items()},
        candidate_events={key: tuple(value) for key, value in by_candidate.items()},
    )


class CaptureSourceValidator:
    """Validate immutable capture lineage without trusting its projection."""

    name = "capture_graph"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        bundles = tuple(
            (await session.scalars(select(TrajectoryBundleRow))).all()
        )
        evidence_rows = tuple(
            (await session.scalars(select(TrajectoryEvidenceRow))).all()
        )
        candidates = tuple(
            (
                await session.scalars(
                    select(ExperienceCandidateRow).order_by(
                        ExperienceCandidateRow.bundle_id,
                        ExperienceCandidateRow.candidate_ordinal,
                    )
                )
            ).all()
        )
        adoptions = tuple(
            (await session.scalars(select(CandidateAdoptionRow))).all()
        )
        all_events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        events = tuple(
            event
            for event in all_events
            if event.event_type in CANDIDATE_EVENT_TYPES
        )
        events_by_causation: dict[UUID, list[DomainEventRow]] = defaultdict(list)
        for event in all_events:
            events_by_causation[event.causation_id].append(event)

        evidence_by_bundle: dict[UUID, list[TrajectoryEvidenceRow]] = defaultdict(list)
        for evidence in evidence_rows:
            evidence_by_bundle[evidence.bundle_id].append(evidence)
        candidates_by_bundle: dict[
            UUID, list[ExperienceCandidateRow]
        ] = defaultdict(list)
        for candidate in candidates:
            candidates_by_bundle[candidate.bundle_id].append(candidate)
        adoptions_by_candidate = {row.candidate_id: row for row in adoptions}
        if len(adoptions_by_candidate) != len(adoptions):
            raise SourceIntegrityError(mismatch_key="candidate_adoption")

        decoded_events: list[tuple[DomainEventRow, EventPayload]] = []
        for event in events:
            try:
                payload = self._event_registry.decode(
                    event_type=event.event_type,
                    payload=event.payload,
                )
            except (TypeError, ValueError):
                raise _fail("capture_event", event.aggregate_id) from None
            decoded_events.append((event, payload))

        known_bundles = {row.bundle_id for row in bundles}
        known_candidates = {row.candidate_id for row in candidates}
        event_index = _index_capture_events(decoded_events)
        for event, payload in decoded_events:
            identifier = getattr(payload, "candidate_id", None)
            if isinstance(payload, TrajectoryCapturedV1):
                if payload.bundle_id not in known_bundles:
                    raise _fail("trajectory_bundle", payload.bundle_id)
            elif not isinstance(identifier, UUID) or identifier not in known_candidates:
                raise _fail("capture_event", event.aggregate_id)

        for bundle in bundles:
            await self._validate_bundle(
                session,
                bundle,
                evidence_by_bundle[bundle.bundle_id],
                candidates_by_bundle[bundle.bundle_id],
                event_index,
                events_by_causation,
                adoptions_by_candidate,
            )
        if set(evidence_by_bundle) - known_bundles:
            raise SourceIntegrityError(mismatch_key="trajectory_evidence")
        if set(candidates_by_bundle) - known_bundles:
            raise SourceIntegrityError(mismatch_key="experience_candidate")

    async def _validate_bundle(
        self,
        session: AsyncSession,
        bundle: TrajectoryBundleRow,
        evidence_rows: list[TrajectoryEvidenceRow],
        candidates: list[ExperienceCandidateRow],
        event_index: _CaptureEventIndex,
        events_by_causation: dict[UUID, list[DomainEventRow]],
        adoptions: dict[UUID, CandidateAdoptionRow],
    ) -> None:
        evidence_by_id = {row.evidence_id: row for row in evidence_rows}
        try:
            manifest = capture_source_integrity.authenticate_trajectory_manifest(
                bundle
            )
            stored_evidence_ids = tuple(row.evidence_id for row in evidence_rows)
            captured_evidence = capture_source_integrity.reconstruct_captured_evidence(
                bundle=bundle,
                rows=tuple(evidence_rows),
                evidence_ids=stored_evidence_ids,
                owner_agent_id=bundle.owner_agent_id,
                manifest=manifest,
            )
            captured_evidence_by_id = dict(
                zip(stored_evidence_ids, captured_evidence, strict=True)
            )
        except (
            AttributeError,
            TypeError,
            UnicodeEncodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _fail("trajectory_bundle", bundle.bundle_id) from None

        captured = event_index.captured_by_bundle.get(bundle.bundle_id, ())
        if len(captured) != 1:
            raise _fail("trajectory_bundle", bundle.bundle_id)
        capture_event, capture_payload = captured[0]
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        evidence_ids: list[UUID] = []
        seen_evidence_ids: set[UUID] = set()
        try:
            # Candidate and pointer order define the capture ledger. Evidence
            # row ordinals describe trajectory steps, not a global ID order.
            for candidate in candidates:
                for evidence_id in _uuid_array(candidate.evidence_refs):
                    if evidence_id not in evidence_by_id:
                        raise KeyError(evidence_id)
                    if evidence_id not in seen_evidence_ids:
                        evidence_ids.append(evidence_id)
                        seen_evidence_ids.add(evidence_id)
            if seen_evidence_ids != set(evidence_by_id):
                raise ValueError("stored evidence is not referenced")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise _fail("trajectory_bundle", bundle.bundle_id) from None
        if (
            capture_event.aggregate_type != "trajectory_bundle"
            or capture_event.aggregate_id != bundle.bundle_id
            or capture_event.sequence != 1
            or capture_event.actor_agent_id != bundle.owner_agent_id
            or capture_event.occurred_at != bundle.captured_at
            or capture_payload.owner_agent_id != bundle.owner_agent_id
            or capture_payload.manifest_hash != bundle.manifest_hash
            or capture_payload.evidence_ids != tuple(evidence_ids)
            or capture_payload.candidate_ids != candidate_ids
        ):
            raise _fail("trajectory_bundle", bundle.bundle_id)
        capture_response = capture_stored_response(
            bundle_id=bundle.bundle_id,
            owner_agent_id=bundle.owner_agent_id,
            manifest_hash=bundle.manifest_hash,
            candidate_ids=tuple(
                candidate.candidate_id for candidate in candidates
            ),
            captured_at=bundle.captured_at,
        )
        await _require_receipt(
            session,
            capture_event,
            owner_agent_id=bundle.owner_agent_id,
            scope=TRAJECTORY_IMPORT_SCOPE,
            resource_type="trajectory_bundle",
            resource_id=bundle.bundle_id,
            expected_response=capture_response,
        )

        created_event_ids: list[int] = []
        for candidate in candidates:
            created_matches = [
                event.event_id
                for event, payload in event_index.candidate_events.get(
                    candidate.candidate_id,
                    (),
                )
                if isinstance(payload, CandidateCreatedV1)
            ]
            if len(created_matches) != 1:
                raise _fail("experience_candidate", candidate.candidate_id)
            created_event_ids.append(created_matches[0])
        ordered_event_ids = (capture_event.event_id, *created_event_ids)
        if any(
            current <= previous
            for previous, current in zip(
                ordered_event_ids[:-1],
                ordered_event_ids[1:],
                strict=True,
            )
        ):
            raise _fail("trajectory_bundle", bundle.bundle_id)
        causal_event_ids = tuple(
            event.event_id
            for event in events_by_causation.get(capture_event.causation_id, ())
        )
        if causal_event_ids != ordered_event_ids:
            raise _fail("trajectory_bundle", bundle.bundle_id)

        for ordinal, candidate in enumerate(candidates, start=1):
            await self._validate_candidate(
                session,
                bundle,
                candidate,
                ordinal,
                evidence_by_id,
                event_index,
                events_by_causation,
                adoptions.get(candidate.candidate_id),
                capture_event.causation_id,
                capture_response,
                captured_evidence_by_id,
            )

    async def _validate_candidate(
        self,
        session: AsyncSession,
        bundle: TrajectoryBundleRow,
        candidate: ExperienceCandidateRow,
        ordinal: int,
        evidence_by_id: dict[UUID, TrajectoryEvidenceRow],
        event_index: _CaptureEventIndex,
        events_by_causation: dict[UUID, list[DomainEventRow]],
        adoption: CandidateAdoptionRow | None,
        capture_receipt_id: UUID,
        capture_response: StoredResponse,
        captured_evidence_by_id: dict[UUID, CapturedEvidenceV1],
    ) -> None:
        try:
            content = reconstruct_candidate_content(candidate)
            evidence_ids = _uuid_array(candidate.evidence_refs)
            referenced = tuple(
                evidence_by_id[evidence_id] for evidence_id in evidence_ids
            )
            captured_evidence = tuple(
                captured_evidence_by_id[evidence_id]
                for evidence_id in evidence_ids
            )
            expected_evidence = VersionContent(
                body=content.body,
                summary=content.summary,
                mechanism=content.mechanism,
                tags=content.tags,
                applicability=content.applicability,
                evidence=tuple(
                    TypedEvidence(
                        type="trajectory_field",
                        id=(
                            f"{bundle.manifest_hash}:{evidence.step_id}:"
                            f"{evidence.field}"
                        ),
                    )
                    for evidence in referenced
                ),
                falsifiers=content.falsifiers,
            ).evidence
            encoded = encode_version_content(kind=candidate.kind, content=content)
            if (
                candidate.owner_agent_id != bundle.owner_agent_id
                or candidate.candidate_ordinal != ordinal
                or candidate.created_at != bundle.captured_at
                or content.evidence != expected_evidence
                or encoded.content_hash != candidate.content_hash
                or candidate.extractor_kind != "deterministic_signal_v1"
                or candidate.extractor_configuration_hash
                != extractor_configuration_hash()
            ):
                raise ValueError("candidate source is invalid")
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _fail("experience_candidate", candidate.candidate_id) from None

        candidate_events = event_index.candidate_events.get(
            candidate.candidate_id,
            (),
        )
        created = [
            (event, payload)
            for event, payload in candidate_events
            if isinstance(payload, CandidateCreatedV1)
        ]
        decisions = [
            (event, payload)
            for event, payload in candidate_events
            if isinstance(payload, (CandidateAdoptedV1, CandidateRejectedV1))
        ]
        if len(created) != 1 or len(decisions) > 1:
            raise _fail("experience_candidate", candidate.candidate_id)
        created_event, created_payload = created[0]
        if (
            created_event.aggregate_type != "experience_candidate"
            or created_event.aggregate_id != candidate.candidate_id
            or created_event.sequence != 1
            or created_event.actor_agent_id != candidate.owner_agent_id
            or created_event.causation_id != capture_receipt_id
            or created_event.occurred_at != candidate.created_at
            or created_payload.bundle_id != candidate.bundle_id
            or created_payload.owner_agent_id != candidate.owner_agent_id
            or created_payload.content_hash != candidate.content_hash
            or created_payload.evidence_ids != evidence_ids
        ):
            raise _fail("experience_candidate", candidate.candidate_id)
        await _require_receipt(
            session,
            created_event,
            owner_agent_id=candidate.owner_agent_id,
            scope=TRAJECTORY_IMPORT_SCOPE,
            resource_type="trajectory_bundle",
            resource_id=candidate.bundle_id,
            expected_response=capture_response,
        )

        if not decisions:
            if adoption is not None:
                raise _fail("candidate_adoption", adoption.adoption_id)
            return
        decision_event, decision = decisions[0]
        if (
            decision_event.aggregate_type != "experience_candidate"
            or decision_event.aggregate_id != candidate.candidate_id
            or decision_event.sequence != 2
            or decision_event.actor_agent_id != candidate.owner_agent_id
            or decision_event.event_id <= created_event.event_id
            or decision.owner_agent_id != candidate.owner_agent_id
        ):
            raise _fail("experience_candidate", candidate.candidate_id)
        if isinstance(decision, CandidateAdoptedV1):
            await self._validate_adoption(
                session,
                candidate,
                adoption,
                decision_event,
                decision,
                content,
                captured_evidence,
                events_by_causation,
            )
            return
        if adoption is not None or not isinstance(decision.reason, StructuredReason):
            raise _fail("experience_candidate", candidate.candidate_id)
        if tuple(events_by_causation.get(decision_event.causation_id, ())) != (
            decision_event,
        ):
            raise _fail("experience_candidate", candidate.candidate_id)
        await _require_receipt(
            session,
            decision_event,
            owner_agent_id=candidate.owner_agent_id,
            scope=CANDIDATE_REJECT_SCOPE,
            resource_type="experience_candidate",
            resource_id=candidate.candidate_id,
            expected_response=candidate_stored_response(
                _candidate_view(
                    candidate=candidate,
                    content=content,
                    evidence=captured_evidence,
                    decision=CandidateDecision.REJECTED,
                    resulting_experience_id=None,
                    resulting_version_id=None,
                    reason=decision.reason,
                    decided_at=decision_event.occurred_at,
                )
            ),
        )

    async def _validate_adoption(
        self,
        session: AsyncSession,
        candidate: ExperienceCandidateRow,
        adoption: CandidateAdoptionRow | None,
        event: DomainEventRow,
        payload: CandidateAdoptedV1,
        candidate_content: VersionContent,
        captured_evidence: tuple[CapturedEvidenceV1, ...],
        events_by_causation: dict[UUID, list[DomainEventRow]],
    ) -> None:
        if (
            adoption is None
            or adoption.candidate_id != candidate.candidate_id
            or adoption.owner_agent_id != candidate.owner_agent_id
            or adoption.adoption_id != payload.adoption_id
            or adoption.resulting_experience_id != payload.resulting_experience_id
            or adoption.resulting_version_id != payload.resulting_version_id
            or adoption.resulting_content_hash != payload.resulting_content_hash
            or adoption.resulting_content_hash != candidate.content_hash
            or adoption.created is not payload.created
            or adoption.adopted_at != event.occurred_at
        ):
            raise _fail("experience_candidate", candidate.candidate_id)
        identity = await session.get(ExperienceRow, adoption.resulting_experience_id)
        version = await session.get(
            ExperienceVersionRow, adoption.resulting_version_id
        )
        source_payload = await session.get(
            ExperiencePayloadRow, adoption.resulting_version_id
        )
        try:
            if (
                identity is None
                or version is None
                or source_payload is None
                or identity.owner_agent_id != candidate.owner_agent_id
                or identity.kind != candidate.kind
                or version.experience_id != identity.experience_id
                or version.content_hash != adoption.resulting_content_hash
                or (
                    adoption.created
                    and (
                        identity.origin is not ExperienceOrigin.ADOPTED_CANDIDATE
                        or identity.created_at != adoption.adopted_at
                        or version.version_number != 1
                        or version.supersedes_version_id is not None
                        or version.created_at != adoption.adopted_at
                    )
                )
                or (
                    not adoption.created
                    and (
                        identity.created_at > adoption.adopted_at
                        or version.created_at > adoption.adopted_at
                    )
                )
                or decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=source_payload,
                )
                != candidate_content
            ):
                raise ValueError("candidate adoption target is invalid")
            self._validate_adoption_causation(
                candidate=candidate,
                adoption=adoption,
                event=event,
                payload=payload,
                identity=identity,
                version=version,
                causal_rows=tuple(events_by_causation.get(event.causation_id, ())),
            )
        except (SourceIntegrityError, TypeError, ValueError):
            raise _fail("candidate_adoption", adoption.adoption_id) from None
        await _require_receipt(
            session,
            event,
            owner_agent_id=candidate.owner_agent_id,
            scope=CANDIDATE_ADOPT_SCOPE,
            resource_type="candidate_adoption",
            resource_id=adoption.adoption_id,
            expected_response=candidate_stored_response(
                _candidate_view(
                    candidate=candidate,
                    content=candidate_content,
                    evidence=captured_evidence,
                    decision=CandidateDecision.ADOPTED,
                    resulting_experience_id=adoption.resulting_experience_id,
                    resulting_version_id=adoption.resulting_version_id,
                    reason=None,
                    decided_at=event.occurred_at,
                )
            ),
        )

    def _validate_adoption_causation(
        self,
        *,
        candidate: ExperienceCandidateRow,
        adoption: CandidateAdoptionRow,
        event: DomainEventRow,
        payload: CandidateAdoptedV1,
        identity: ExperienceRow,
        version: ExperienceVersionRow,
        causal_rows: tuple[DomainEventRow, ...],
    ) -> None:
        causal_types = tuple(row.event_type for row in causal_rows)
        if not causal_rows or causal_rows[-1].event_id != event.event_id:
            raise ValueError("candidate adoption must finish its causation")
        if not payload.created:
            if causal_types != (CandidateAdoptedV1.event_type,):
                raise ValueError("reused candidate adoption must be event-only")
            return
        if causal_types != (
            ExperienceCreatedV1.event_type,
            ExperienceVersionCreatedV1.event_type,
            CandidateAdoptedV1.event_type,
        ):
            raise ValueError("created candidate adoption causation is invalid")
        created_event, version_event, _ = causal_rows
        created = self._event_registry.decode(
            event_type=created_event.event_type,
            payload=created_event.payload,
        )
        version_created = self._event_registry.decode(
            event_type=version_event.event_type,
            payload=version_event.payload,
        )
        if not isinstance(created, ExperienceCreatedV1) or not isinstance(
            version_created,
            ExperienceVersionCreatedV1,
        ):
            raise ValueError("created candidate adoption payloads are invalid")
        if (
            created_event.aggregate_type != "experience"
            or created_event.aggregate_id != adoption.resulting_experience_id
            or created_event.sequence != 1
            or version_event.aggregate_type != "experience"
            or version_event.aggregate_id != adoption.resulting_experience_id
            or version_event.sequence != 2
            or any(
                row.actor_agent_id != candidate.owner_agent_id
                or row.occurred_at != event.occurred_at
                for row in (created_event, version_event)
            )
            or created.experience_id != adoption.resulting_experience_id
            or created.version_id != adoption.resulting_version_id
            or created.after.owner_agent_id != candidate.owner_agent_id
            or created.after.current_version_id != adoption.resulting_version_id
            or created.after.current_content_hash != candidate.content_hash
            or version_created.experience_id != adoption.resulting_experience_id
            or version_created.version_id != adoption.resulting_version_id
            or version_created.version_number != 1
            or version_created.supersedes_version_id is not None
            or version_created.links
            or version_created.before != created.after
            or version_created.after != created.after
            or identity.experience_id != created.experience_id
            or version.version_id != version_created.version_id
        ):
            raise ValueError("created candidate adoption anchors are invalid")


def register_capture_source_validator(validator: SourceValidator) -> None:
    """Register the capture graph validator once at composition time."""
    validator.register(CaptureSourceValidator(validator.event_registry))


__all__ = ["CaptureSourceValidator", "register_capture_source_validator"]
