from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.domain import TypedEvidence
from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    ExperienceView,
    PeekExperiences,
    RetrievalCandidate,
    RetrievalRecord,
    SearchExperiences,
    SearchHit,
    SearchResult,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue

NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000201")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000301")


def state(
    *,
    temperature: Temperature = Temperature.WARM,
) -> ExperienceStateSnapshotV1:
    return ExperienceStateSnapshotV1(
        experience_id=EXPERIENCE_ID,
        owner_agent_id=OWNER_ID,
        current_version_id=VERSION_ID,
        current_content_hash="a" * 64,
        temperature=temperature,
        importance=0.4,
        confidence=0.6,
        activation_score=0.5,
        source_trust=1.0,
        access_count=2,
        access_strength=1.5,
        strength_updated_at=NOW,
        last_accessed_at=NOW,
        last_transition_at=NOW,
        last_lifecycle_evaluated_at=None,
        consecutive_below_threshold=0,
        pinned=False,
    )


def record(
    *,
    temperature: Temperature = Temperature.WARM,
) -> RetrievalRecord:
    return RetrievalRecord(
        experience_id=EXPERIENCE_ID,
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.LOCAL,
        created_at=NOW,
        current_version_id=VERSION_ID,
        current_version_number=1,
        current_version_created_at=NOW,
        current_content_hash="a" * 64,
        summary="Lease handoff",
        mechanism="single writer",
        tags=("memory",),
        applicability=("local runtime",),
        evidence=(TypedEvidence(type="test", id="case-1"),),
        falsifiers=("two writers overlap",),
        state=state(temperature=temperature),
        projection_event_id=2,
        latest_causal_at=NOW,
    )


def view(*, blurred: bool, body: str | None) -> ExperienceView:
    return ExperienceView(
        experience_id=EXPERIENCE_ID,
        owner_agent_id=OWNER_ID,
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.LOCAL,
        created_at=NOW,
        version_id=VERSION_ID,
        version_number=1,
        version_created_at=NOW,
        content_hash="a" * 64,
        temperature=Temperature.WARM,
        importance=0.4,
        confidence=0.6,
        activation_score=0.5,
        source_trust=1.0,
        access_count=2,
        access_strength=1.5,
        strength_updated_at=NOW,
        last_accessed_at=NOW,
        last_transition_at=NOW,
        last_lifecycle_evaluated_at=None,
        consecutive_below_threshold=0,
        pinned=False,
        summary="Lease handoff",
        mechanism="single writer",
        tags=("memory",),
        applicability=("local runtime",),
        evidence=(TypedEvidence(type="test", id="case-1"),),
        falsifiers=("two writers overlap",),
        blurred=blurred,
        body=body,
        body_is_excerpt=False,
    )


def test_retrieval_record_is_strict_and_normalizes_aware_times_to_utc() -> None:
    non_utc = NOW.astimezone(timezone(timedelta(hours=8)))
    value = record()
    shifted = RetrievalRecord(
        experience_id=value.experience_id,
        owner_agent_id=value.owner_agent_id,
        kind=value.kind,
        origin=value.origin,
        created_at=non_utc,
        current_version_id=value.current_version_id,
        current_version_number=value.current_version_number,
        current_version_created_at=non_utc,
        current_content_hash=value.current_content_hash,
        summary=value.summary,
        mechanism=value.mechanism,
        tags=value.tags,
        applicability=value.applicability,
        evidence=value.evidence,
        falsifiers=value.falsifiers,
        state=value.state,
        projection_event_id=value.projection_event_id,
        latest_causal_at=non_utc,
    )

    assert shifted.created_at == NOW
    assert shifted.current_version_created_at == NOW
    assert shifted.latest_causal_at == NOW
    arguments = {
        field.name: getattr(value, field.name)
        for field in fields(RetrievalRecord)
    }
    arguments["current_content_hash"] = "b" * 64
    with pytest.raises(ValueError, match="state anchors"):
        RetrievalRecord(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_version_number", True),
        ("projection_event_id", 0),
        ("tags", ["memory"]),
        ("evidence", [{"type": "test", "id": "case-1"}]),
    ],
)
def test_retrieval_record_rejects_coercion_and_mutable_sequences(
    field: str,
    value: object,
) -> None:
    valid = record()
    arguments = {
        name: getattr(valid, name)
        for name in valid.__dataclass_fields__
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        RetrievalRecord(**arguments)


def test_candidate_values_require_cues_overlap_and_active_state() -> None:
    cue = TermCue(term="memory", term_kind="word", weight=1.0)
    selection = CandidateSelection(
        owner_agent_id=OWNER_ID,
        query_cues=(cue,),
        mode=RetrievalMode.FOCUSED,
        requested_limit=10,
    )
    candidate = RetrievalCandidate(
        record=record(),
        terms=(cue,),
        raw_overlap=1.0,
    )

    assert selection.query_cues == (cue,)
    assert candidate.raw_overlap == 1.0
    with pytest.raises(ValueError, match="must not be empty"):
        CandidateSelection(
            owner_agent_id=OWNER_ID,
            query_cues=(),
            mode=RetrievalMode.FOCUSED,
            requested_limit=10,
        )
    with pytest.raises(ValueError, match="positive"):
        RetrievalCandidate(record=record(), terms=(cue,), raw_overlap=0.0)
    with pytest.raises(ValueError, match="archived"):
        RetrievalCandidate(
            record=record(temperature=Temperature.ARCHIVED),
            terms=(cue,),
            raw_overlap=1.0,
        )


def test_search_and_peek_contracts_apply_public_and_internal_byte_limits() -> None:
    search = SearchExperiences(
        owner_agent_id=OWNER_ID,
        query="  lease handoff  ",
        mode=RetrievalMode.FOCUSED,
        tags=("memory",),
        mechanism_cues=("single writer",),
        limit=50,
        content_budget_bytes=0,
        expand_cold=True,
    )
    peek = PeekExperiences(
        owner_agent_id=OWNER_ID,
        query="lease handoff",
        mode=RetrievalMode.ASSOCIATIVE,
    )

    assert search.query == "lease handoff"
    assert search.content_budget_bytes == 0
    assert peek.limit == 12
    assert peek.content_budget_bytes == 24_576
    assert peek.per_hit_excerpt_bytes == 2_048
    assert peek.expand_cold is True

    assert (
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="q" * 2_000,
            mode=RetrievalMode.FOCUSED,
        ).query
        == "q" * 2_000
    )
    with pytest.raises(ValueError, match="2,000"):
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="q" * 2_001,
            mode=RetrievalMode.FOCUSED,
        )
    assert (
        PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="q" * 6_001,
            mode=RetrievalMode.FOCUSED,
        ).query
        == "q" * 6_001
    )
    with pytest.raises(ValueError, match="6,001"):
        PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="q" * 6_002,
            mode=RetrievalMode.FOCUSED,
        )

    with pytest.raises(ValueError, match="1 and 50"):
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="memory",
            mode=RetrievalMode.FOCUSED,
            limit=0,
        )
    with pytest.raises(ValueError, match="positive"):
        PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="memory",
            mode=RetrievalMode.FOCUSED,
            content_budget_bytes=0,
        )
    with pytest.raises(ValueError, match="1 and 12"):
        PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="memory",
            mode=RetrievalMode.FOCUSED,
            limit=13,
        )
    with pytest.raises(ValueError, match="2,048"):
        PeekExperiences(
            owner_agent_id=OWNER_ID,
            query="memory",
            mode=RetrievalMode.FOCUSED,
            per_hit_excerpt_bytes=2_049,
        )
    with pytest.raises(ValueError, match="boolean"):
        SearchExperiences(
            owner_agent_id=OWNER_ID,
            query="memory",
            mode=RetrievalMode.FOCUSED,
            expand_cold=1,  # type: ignore[arg-type]
        )


def test_view_and_search_result_enforce_blur_and_budget_invariants() -> None:
    full = view(blurred=False, body="full body")
    blurred = view(blurred=True, body=None)
    hit = SearchHit(
        experience=full,
        score=0.8,
        ranking_relevance=0.75,
        lexical_or_trigram_relevance=0.75,
        mechanism_relevance=0.5,
        activation=0.4,
        expanded=True,
        reactivated=False,
    )
    result = SearchResult(
        hits=(hit,),
        remaining_content_budget_bytes=100,
    )

    assert result.hits[0].experience.body == "full body"
    assert result.hits[0].experience.body_is_excerpt is False
    assert blurred.body is None
    with pytest.raises(ValidationError, match="blurred"):
        view(blurred=True, body="leak")
    with pytest.raises(ValidationError, match="body"):
        view(blurred=False, body=None)
    with pytest.raises(ValidationError):
        SearchResult(
            hits=(hit,),
            remaining_content_budget_bytes=-1,
        )
