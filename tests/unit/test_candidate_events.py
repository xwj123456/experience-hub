from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes
from experience_hub.domain import StructuredReason
from experience_hub.domain.events import EventPayload, EventRegistry
from experience_hub.experiences.candidate_events import (
    CandidateAdoptedV1,
    CandidateCreatedV1,
    CandidateRejectedV1,
    TrajectoryCapturedV1,
    register_candidate_events,
)
from experience_hub.experiences.candidate_models import CandidateDecision

OWNER_ID = UUID("00000000-0000-0000-0000-000000000401")
BUNDLE_ID = UUID("00000000-0000-0000-0000-000000000402")
EVIDENCE_A = UUID("00000000-0000-0000-0000-000000000403")
EVIDENCE_B = UUID("00000000-0000-0000-0000-000000000404")
CANDIDATE_A = UUID("00000000-0000-0000-0000-000000000405")
CANDIDATE_B = UUID("00000000-0000-0000-0000-000000000406")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000407")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000408")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000409")
CONTENT_HASH = "a" * 64
MANIFEST_HASH = "b" * 64


def event_fixtures() -> Iterator[EventPayload]:
    yield TrajectoryCapturedV1(
        schema_version=1,
        bundle_id=BUNDLE_ID,
        owner_agent_id=OWNER_ID,
        manifest_hash=MANIFEST_HASH,
        evidence_ids=(EVIDENCE_B, EVIDENCE_A),
        candidate_ids=(CANDIDATE_B, CANDIDATE_A),
    )
    yield CandidateCreatedV1(
        schema_version=1,
        candidate_id=CANDIDATE_A,
        bundle_id=BUNDLE_ID,
        owner_agent_id=OWNER_ID,
        content_hash=CONTENT_HASH,
        evidence_ids=(EVIDENCE_B, EVIDENCE_A),
        decision_after=CandidateDecision.PENDING,
    )
    yield CandidateAdoptedV1(
        schema_version=1,
        candidate_id=CANDIDATE_A,
        owner_agent_id=OWNER_ID,
        decision_before=CandidateDecision.PENDING,
        decision_after=CandidateDecision.ADOPTED,
        adoption_id=ADOPTION_ID,
        resulting_experience_id=EXPERIENCE_ID,
        resulting_version_id=VERSION_ID,
        resulting_content_hash=CONTENT_HASH,
        created=True,
    )
    yield CandidateRejectedV1(
        schema_version=1,
        candidate_id=CANDIDATE_B,
        owner_agent_id=OWNER_ID,
        decision_before=CandidateDecision.PENDING,
        decision_after=CandidateDecision.REJECTED,
        reason=StructuredReason.from_user_text("Not generally applicable."),
    )


def test_candidate_event_registry_round_trips_all_v1_payloads() -> None:
    registry = EventRegistry()
    register_candidate_events(registry)

    assert registry.event_types == {
        "trajectory.captured",
        "candidate.created",
        "candidate.adopted",
        "candidate.rejected",
    }
    for payload in event_fixtures():
        encoded = canonical_json_bytes(payload)
        assert registry.decode(
            event_type=payload.event_type,
            payload=encoded,
        ) == payload


def test_capture_event_preserves_declared_identifier_order() -> None:
    captured, created, *_ = event_fixtures()

    assert isinstance(captured, TrajectoryCapturedV1)
    assert captured.evidence_ids == (EVIDENCE_B, EVIDENCE_A)
    assert captured.candidate_ids == (CANDIDATE_B, CANDIDATE_A)
    assert isinstance(created, CandidateCreatedV1)
    assert created.evidence_ids == (EVIDENCE_B, EVIDENCE_A)


@pytest.mark.parametrize(
    ("payload_type", "overrides"),
    (
        (TrajectoryCapturedV1, {"manifest_hash": "A" * 64}),
        (CandidateCreatedV1, {"content_hash": "a" * 63}),
        (CandidateAdoptedV1, {"resulting_content_hash": "not-a-hash"}),
        (
            CandidateCreatedV1,
            {"decision_after": CandidateDecision.ADOPTED},
        ),
        (
            CandidateAdoptedV1,
            {"decision_before": CandidateDecision.REJECTED},
        ),
        (
            CandidateAdoptedV1,
            {"decision_after": CandidateDecision.REJECTED},
        ),
        (
            CandidateRejectedV1,
            {"decision_before": CandidateDecision.ADOPTED},
        ),
        (
            CandidateRejectedV1,
            {"decision_after": CandidateDecision.ADOPTED},
        ),
    ),
)
def test_candidate_events_reject_invalid_hashes_and_decisions(
    payload_type: type[EventPayload],
    overrides: dict[str, object],
) -> None:
    fixture = next(
        item for item in event_fixtures() if type(item) is payload_type
    )

    with pytest.raises(ValidationError):
        payload_type.model_validate({**fixture.model_dump(), **overrides})


@pytest.mark.parametrize(
    ("payload_type", "field", "identifier"),
    (
        (TrajectoryCapturedV1, "evidence_ids", EVIDENCE_A),
        (TrajectoryCapturedV1, "candidate_ids", CANDIDATE_A),
        (CandidateCreatedV1, "evidence_ids", EVIDENCE_A),
    ),
)
def test_candidate_events_reject_repeated_identifiers(
    payload_type: type[EventPayload],
    field: str,
    identifier: UUID,
) -> None:
    fixture = next(
        item for item in event_fixtures() if type(item) is payload_type
    )

    with pytest.raises(ValidationError):
        payload_type.model_validate(
            {**fixture.model_dump(), field: (identifier, identifier)}
        )
