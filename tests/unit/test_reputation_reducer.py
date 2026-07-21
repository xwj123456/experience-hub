from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import EventRegistry, StoredEvent
from experience_hub.sharing.events import (
    CapsuleFeedbackRecordedV1,
    register_sharing_events,
)
from experience_hub.sharing.models import FeedbackVerdict, Reputation
from experience_hub.sharing.projector import (
    AgentReputationProjector,
    SharingProjectionIntegrityError,
    reduce_reputation,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    AgentReputationRow,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
PUBLISHER_ID = UUID("10000000-0000-0000-0000-000000000001")
OBSERVER_A_ID = UUID("10000000-0000-0000-0000-000000000002")
OBSERVER_B_ID = UUID("10000000-0000-0000-0000-000000000003")
CAPSULE_A_ID = UUID("20000000-0000-0000-0000-000000000001")
CAPSULE_B_ID = UUID("20000000-0000-0000-0000-000000000002")
FEEDBACK_IDS = tuple(
    UUID(f"30000000-0000-0000-0000-{value:012d}") for value in range(1, 21)
)
CAUSATION_ID = UUID("40000000-0000-0000-0000-000000000001")
ADOPTION_ID = UUID("50000000-0000-0000-0000-000000000001")
EXPERIENCE_ID = UUID("50000000-0000-0000-0000-000000000002")


def feedback_event(
    *,
    event_id: int,
    feedback_id: UUID,
    capsule_id: UUID = CAPSULE_A_ID,
    publisher_agent_id: UUID = PUBLISHER_ID,
    observer_agent_id: UUID = OBSERVER_A_ID,
    revision: int = 1,
    previous_verdict: FeedbackVerdict | None = None,
    current_verdict: FeedbackVerdict,
    alpha_before: int,
    beta_before: int,
    alpha_after: int,
    beta_after: int,
    occurred_at: datetime = NOW,
) -> StoredEvent:
    payload = CapsuleFeedbackRecordedV1(
        schema_version=1,
        feedback_id=feedback_id,
        capsule_id=capsule_id,
        publisher_agent_id=publisher_agent_id,
        observer_agent_id=observer_agent_id,
        revision=revision,
        previous_verdict=previous_verdict,
        current_verdict=current_verdict,
        alpha_before=alpha_before,
        beta_before=beta_before,
        alpha_after=alpha_after,
        beta_after=beta_after,
    )
    return StoredEvent(
        event_id=event_id,
        aggregate_type="capsule",
        aggregate_id=capsule_id,
        sequence=event_id + 1,
        event_type=CapsuleFeedbackRecordedV1.event_type,
        payload=payload,
        actor_agent_id=observer_agent_id,
        causation_id=CAUSATION_ID,
        occurred_at=occurred_at,
    )


def test_agent_reputation_projector_owns_only_feedback_events() -> None:
    registry = EventRegistry()
    register_sharing_events(registry)

    projector = AgentReputationProjector(registry)

    assert projector.name == "agent_reputation"
    assert projector.version == 1
    assert projector.event_types == frozenset({CapsuleFeedbackRecordedV1.event_type})


def test_reducer_uses_locked_two_two_prior_and_rejects_before_drift() -> None:
    valid = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )

    reduced = reduce_reputation(
        None,
        event=valid,
        previous_verdict=None,
    )

    assert isinstance(reduced, Reputation)
    assert (
        reduced.useful_count,
        reduced.refuted_count,
        reduced.harmful_count,
        reduced.alpha,
        reduced.beta,
    ) == (1, 0, 0, 3, 2)
    assert reduced.trust == pytest.approx(0.6, abs=1e-12)

    drifted = feedback_event(
        event_id=2,
        feedback_id=FEEDBACK_IDS[1],
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=3,
        beta_before=2,
        alpha_after=4,
        beta_after=2,
    )
    with pytest.raises(
        SharingProjectionIntegrityError,
        match="before|prior",
    ):
        reduce_reputation(
            None,
            event=drifted,
            previous_verdict=None,
        )


@pytest.mark.parametrize(
    ("verdict", "counts", "alpha", "beta", "trust"),
    (
        (FeedbackVerdict.USEFUL, (1, 0, 0), 3, 2, 0.6),
        (FeedbackVerdict.REFUTED, (0, 1, 0), 2, 3, 0.4),
        (FeedbackVerdict.HARMFUL, (0, 0, 1), 2, 3, 0.4),
    ),
)
def test_first_feedback_applies_exactly_one_effective_increment(
    verdict: FeedbackVerdict,
    counts: tuple[int, int, int],
    alpha: int,
    beta: int,
    trust: float,
) -> None:
    event = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        current_verdict=verdict,
        alpha_before=2,
        beta_before=2,
        alpha_after=alpha,
        beta_after=beta,
    )

    reduced = reduce_reputation(
        None,
        event=event,
        previous_verdict=None,
    )

    assert (
        reduced.useful_count,
        reduced.refuted_count,
        reduced.harmful_count,
    ) == counts
    assert (reduced.alpha, reduced.beta) == (alpha, beta)
    assert reduced.trust == pytest.approx(trust, abs=1e-12)
    assert reduced.last_feedback_at == NOW


@pytest.mark.parametrize(
    (
        "first_verdict",
        "second_verdict",
        "first_alpha",
        "first_beta",
        "second_alpha",
        "second_beta",
        "expected_counts",
        "expected_trust",
    ),
    (
        (
            FeedbackVerdict.USEFUL,
            FeedbackVerdict.REFUTED,
            3,
            2,
            2,
            3,
            (0, 1, 0),
            0.4,
        ),
        (
            FeedbackVerdict.REFUTED,
            FeedbackVerdict.USEFUL,
            2,
            3,
            3,
            2,
            (1, 0, 0),
            0.6,
        ),
    ),
)
def test_revision_replaces_the_prior_effective_verdict(
    first_verdict: FeedbackVerdict,
    second_verdict: FeedbackVerdict,
    first_alpha: int,
    first_beta: int,
    second_alpha: int,
    second_beta: int,
    expected_counts: tuple[int, int, int],
    expected_trust: float,
) -> None:
    first_event = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        current_verdict=first_verdict,
        alpha_before=2,
        beta_before=2,
        alpha_after=first_alpha,
        beta_after=first_beta,
    )
    first = reduce_reputation(
        None,
        event=first_event,
        previous_verdict=None,
    )
    second_event = feedback_event(
        event_id=2,
        feedback_id=FEEDBACK_IDS[1],
        revision=2,
        previous_verdict=first_verdict,
        current_verdict=second_verdict,
        alpha_before=first_alpha,
        beta_before=first_beta,
        alpha_after=second_alpha,
        beta_after=second_beta,
        occurred_at=NOW + timedelta(minutes=1),
    )

    revised = reduce_reputation(
        first,
        event=second_event,
        previous_verdict=first_verdict,
    )

    assert (
        revised.useful_count,
        revised.refuted_count,
        revised.harmful_count,
    ) == expected_counts
    assert (revised.alpha, revised.beta) == (
        second_alpha,
        second_beta,
    )
    assert revised.trust == pytest.approx(expected_trust, abs=1e-12)
    assert revised.last_feedback_at == NOW + timedelta(minutes=1)


def test_same_publisher_reputation_is_independent_for_two_observers() -> None:
    observer_a_event = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        observer_agent_id=OBSERVER_A_ID,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )
    observer_b_event = feedback_event(
        event_id=2,
        feedback_id=FEEDBACK_IDS[1],
        observer_agent_id=OBSERVER_B_ID,
        current_verdict=FeedbackVerdict.HARMFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=2,
        beta_after=3,
    )

    observer_a = reduce_reputation(
        None,
        event=observer_a_event,
        previous_verdict=None,
    )
    observer_b = reduce_reputation(
        None,
        event=observer_b_event,
        previous_verdict=None,
    )

    assert (
        observer_a.subject_agent_id,
        observer_a.observer_agent_id,
    ) == (PUBLISHER_ID, OBSERVER_A_ID)
    assert observer_a.trust == pytest.approx(0.6, abs=1e-12)
    assert (
        observer_b.subject_agent_id,
        observer_b.observer_agent_id,
    ) == (PUBLISHER_ID, OBSERVER_B_ID)
    assert observer_b.trust == pytest.approx(0.4, abs=1e-12)


def test_two_capsules_from_one_publisher_accumulate_for_one_observer() -> None:
    first_event = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        capsule_id=CAPSULE_A_ID,
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
    )
    first = reduce_reputation(
        None,
        event=first_event,
        previous_verdict=None,
    )
    second_event = feedback_event(
        event_id=2,
        feedback_id=FEEDBACK_IDS[1],
        capsule_id=CAPSULE_B_ID,
        current_verdict=FeedbackVerdict.REFUTED,
        alpha_before=3,
        beta_before=2,
        alpha_after=3,
        beta_after=3,
        occurred_at=NOW + timedelta(minutes=1),
    )

    combined = reduce_reputation(
        first,
        event=second_event,
        previous_verdict=None,
    )

    assert (
        combined.subject_agent_id,
        combined.observer_agent_id,
        combined.useful_count,
        combined.refuted_count,
        combined.harmful_count,
        combined.alpha,
        combined.beta,
    ) == (
        PUBLISHER_ID,
        OBSERVER_A_ID,
        1,
        1,
        0,
        3,
        3,
    )
    assert combined.trust == pytest.approx(0.5, abs=1e-12)


def test_reducer_rejects_feedback_clock_regression() -> None:
    first_event = feedback_event(
        event_id=1,
        feedback_id=FEEDBACK_IDS[0],
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
        occurred_at=NOW + timedelta(minutes=1),
    )
    current = reduce_reputation(
        None,
        event=first_event,
        previous_verdict=None,
    )
    old_event = feedback_event(
        event_id=2,
        feedback_id=FEEDBACK_IDS[1],
        capsule_id=CAPSULE_B_ID,
        current_verdict=FeedbackVerdict.REFUTED,
        alpha_before=3,
        beta_before=2,
        alpha_after=3,
        beta_after=3,
        occurred_at=NOW,
    )

    with pytest.raises(
        SharingProjectionIntegrityError,
        match="clock|time|preced",
    ):
        reduce_reputation(
            current,
            event=old_event,
            previous_verdict=None,
        )


class ReputationLookupSession:
    def __init__(self) -> None:
        self.row: AgentReputationRow | None = None

    async def get(
        self,
        model: type[AgentReputationRow],
        identity: tuple[UUID, UUID],
    ) -> AgentReputationRow | None:
        assert model is AgentReputationRow
        if self.row is not None:
            assert identity == (
                self.row.subject_agent_id,
                self.row.observer_agent_id,
            )
        return self.row


@pytest.mark.asyncio
async def test_later_feedback_changes_only_future_trust_lookup() -> None:
    lookup_session = ReputationLookupSession()
    early_trust = await SharingRepository.observer_trust(
        session=cast(AsyncSession, lookup_session),
        subject_agent_id=PUBLISHER_ID,
        observer_agent_id=OBSERVER_A_ID,
    )
    early_adoption = AdoptionRecordRow(
        adoption_id=ADOPTION_ID,
        adopter_agent_id=OBSERVER_A_ID,
        capsule_id=CAPSULE_A_ID,
        resulting_experience_id=EXPERIENCE_ID,
        captured_trust=early_trust,
        provenance_chain=canonical_json_bytes(
            (
                {
                    "capsule_id": CAPSULE_A_ID,
                    "publisher_agent_id": PUBLISHER_ID,
                },
            )
        ),
        root_fingerprint="a" * 64,
        corroboration_applied=False,
        adopted_at=NOW,
    )
    event = feedback_event(
        event_id=10,
        feedback_id=FEEDBACK_IDS[0],
        current_verdict=FeedbackVerdict.USEFUL,
        alpha_before=2,
        beta_before=2,
        alpha_after=3,
        beta_after=2,
        occurred_at=NOW + timedelta(minutes=1),
    )
    reputation = reduce_reputation(
        None,
        event=event,
        previous_verdict=None,
    )
    lookup_session.row = AgentReputationRow(
        subject_agent_id=reputation.subject_agent_id,
        observer_agent_id=reputation.observer_agent_id,
        useful_count=reputation.useful_count,
        refuted_count=reputation.refuted_count,
        harmful_count=reputation.harmful_count,
        alpha=reputation.alpha,
        beta=reputation.beta,
        projection_event_id=event.event_id,
    )

    future_trust = await SharingRepository.observer_trust(
        session=cast(AsyncSession, lookup_session),
        subject_agent_id=PUBLISHER_ID,
        observer_agent_id=OBSERVER_A_ID,
    )

    assert early_trust == pytest.approx(0.5, abs=1e-12)
    assert future_trust == pytest.approx(0.6, abs=1e-12)
    assert early_adoption.captured_trust == pytest.approx(0.5, abs=1e-12)
