from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from experience_hub.inspiration.hashing import (
    hash_mechanism,
    mechanism_similarity,
)
from experience_hub.inspiration.incubation import (
    ClusterTransition,
    EvaluationTransition,
    IncubationCluster,
    IncubationMember,
    OccurrencePlan,
    plan_evaluation_transition,
    plan_occurrence,
)
from experience_hub.inspiration.models import (
    EvaluationVerdict,
    MechanismIncubation,
    MechanismMaturity,
)

NOW = datetime(2026, 7, 18, 14, tzinfo=UTC)
LATER = NOW + timedelta(hours=2)
OWNER_A = UUID("00000000-0000-0000-0000-000000003001")
OWNER_B = UUID("00000000-0000-0000-0000-000000003002")
BASE = "abcdefghijklmnopqrstuvwxyz0123456789"
BRIDGE_LEFT = "Xbcdefghijklmnopqrstuvwxyz0123456789"
BRIDGE_RIGHT = "aXcdefghijklmnopqrstuvwxyz0123456789"
NEAR_END = "abcdefghijklmnopqrstuvwxyz012345678XX"
BELOW_END = "abcdefghijklmnopqrstuvwxyz012345678XXX"


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def _snapshot(value: int) -> str:
    return f"{value:064x}"


def _member(
    ordinal: int,
    mechanism: str,
    *,
    owner: UUID = OWNER_A,
) -> IncubationMember:
    return IncubationMember(
        idea_id=_uuid(4_000 + ordinal),
        owner_agent_id=owner,
        mechanism=mechanism,
        mechanism_hash=hash_mechanism(mechanism),
    )


def _unique_hashes(members: tuple[IncubationMember, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(member.mechanism_hash for member in members))


def _cluster(
    *mechanisms: str,
    owners: tuple[UUID, ...] | None = None,
    snapshot_hashes: frozenset[str] = frozenset({_snapshot(1)}),
    last_signal_at: datetime = NOW,
    supported_count: int = 0,
    refuted_count: int = 0,
    distinct_adopter_count: int = 0,
    maturity: MechanismMaturity | None = None,
    candidate_since: datetime | None = None,
) -> IncubationCluster:
    retained_owners = owners or (OWNER_A,) * len(mechanisms)
    members = tuple(
        _member(index, mechanism, owner=retained_owners[index - 1])
        for index, mechanism in enumerate(mechanisms, start=1)
    )
    member_hashes = _unique_hashes(members)
    inferred_maturity = maturity or (
        MechanismMaturity.INCUBATING
        if len(snapshot_hashes) >= 2
        else MechanismMaturity.SPECULATIVE
    )
    state = MechanismIncubation(
        cluster_id=members[0].mechanism_hash,
        canonical_mechanism_hash=members[0].mechanism_hash,
        member_hashes=member_hashes,
        occurrence_count=len(members),
        distinct_snapshot_count=len(snapshot_hashes),
        distinct_adopter_count=distinct_adopter_count,
        supported_count=supported_count,
        refuted_count=refuted_count,
        maturity=inferred_maturity,
        candidate_since=candidate_since,
        last_signal_at=last_signal_at,
    )
    return IncubationCluster(
        state=state,
        members=members,
        snapshot_hashes=snapshot_hashes,
    )


def test_first_occurrence_requests_new_rows_and_exact_transition() -> None:
    mechanism = "first immutable mechanism"
    mechanism_hash = hash_mechanism(mechanism)

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=mechanism,
        snapshot_hash=_snapshot(10),
        run_occurred_at=NOW,
        clusters=(),
    )

    assert isinstance(plan, OccurrencePlan)
    assert plan.create_idea is True
    assert plan.create_occurrence is True
    assert plan.mechanism_hash == mechanism_hash
    assert plan.duplicate_relation is None
    assert plan.maximum_similarity is None
    assert plan.transition == ClusterTransition(
        cluster_id=mechanism_hash,
        canonical_mechanism_hash=mechanism_hash,
        member_hashes_before=(),
        member_hashes_after=(mechanism_hash,),
        occurrence_count_before=0,
        occurrence_count_after=1,
        distinct_snapshot_count_before=0,
        distinct_snapshot_count_after=1,
        distinct_adopter_count_before=0,
        distinct_adopter_count_after=0,
        supported_count_before=0,
        supported_count_after=0,
        refuted_count_before=0,
        refuted_count_after=0,
        maturity_before=None,
        maturity_after=MechanismMaturity.SPECULATIVE,
        candidate_since_before=None,
        candidate_since_after=None,
        last_signal_at_before=None,
        last_signal_at_after=NOW,
    )


def test_same_snapshot_recurrence_does_not_promote_maturity() -> None:
    cluster = _cluster(BASE)

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(1),
        run_occurred_at=LATER,
        clusters=(cluster,),
    )

    transition = plan.transition
    assert transition.occurrence_count_before == 1
    assert transition.occurrence_count_after == 2
    assert transition.distinct_snapshot_count_before == 1
    assert transition.distinct_snapshot_count_after == 1
    assert transition.maturity_before is MechanismMaturity.SPECULATIVE
    assert transition.maturity_after is MechanismMaturity.SPECULATIVE
    assert transition.last_signal_at_before == NOW
    assert transition.last_signal_at_after == LATER
    assert transition.member_hashes_before == transition.member_hashes_after


def test_distinct_snapshot_recurrence_becomes_incubating() -> None:
    cluster = _cluster(BASE)

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(2),
        run_occurred_at=LATER,
        clusters=(cluster,),
    )

    transition = plan.transition
    assert transition.distinct_snapshot_count_before == 1
    assert transition.distinct_snapshot_count_after == 2
    assert transition.maturity_before is MechanismMaturity.SPECULATIVE
    assert transition.maturity_after is MechanismMaturity.INCUBATING
    assert transition.candidate_since_before is None
    assert transition.candidate_since_after is None


def test_cluster_similarity_uses_the_maximum_member_not_the_canonical_member() -> None:
    maximum_member_cluster = _cluster(
        "a deliberately unrelated canonical mechanism",
        BASE,
    )
    canonical_near_cluster = _cluster(NEAR_END)
    assert mechanism_similarity(
        maximum_member_cluster.members[0].mechanism,
        BASE,
    ) < 0.82
    assert mechanism_similarity(
        maximum_member_cluster.members[1].mechanism,
        BASE,
    ) == 1.0

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(3),
        run_occurred_at=LATER,
        clusters=(canonical_near_cluster, maximum_member_cluster),
    )

    assert (
        plan.transition.cluster_id
        == maximum_member_cluster.state.cluster_id
    )
    assert plan.maximum_similarity == 1.0


def test_highest_maximum_similarity_wins() -> None:
    lower = _cluster(NEAR_END)
    higher = _cluster(BRIDGE_LEFT)
    assert mechanism_similarity(NEAR_END, BASE) < mechanism_similarity(
        BRIDGE_LEFT,
        BASE,
    )

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(4),
        run_occurred_at=LATER,
        clusters=(lower, higher),
    )

    assert plan.transition.cluster_id == higher.state.cluster_id


def test_cluster_threshold_is_inclusive_and_below_starts_a_new_cluster() -> None:
    cluster = _cluster(BASE)
    assert mechanism_similarity(BASE, NEAR_END) >= 0.82
    assert mechanism_similarity(BASE, BELOW_END) < 0.82

    near = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=NEAR_END,
        snapshot_hash=_snapshot(4),
        run_occurred_at=LATER,
        clusters=(cluster,),
    )
    below = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BELOW_END,
        snapshot_hash=_snapshot(4),
        run_occurred_at=LATER,
        clusters=(cluster,),
    )

    assert near.transition.cluster_id == cluster.state.cluster_id
    assert below.transition.cluster_id == hash_mechanism(BELOW_END)
    assert below.maximum_similarity is None


def test_similarity_tie_uses_canonical_hash_and_never_merges_prior_clusters() -> None:
    left = _cluster(BRIDGE_LEFT)
    right = _cluster(BRIDGE_RIGHT)
    assert mechanism_similarity(BRIDGE_LEFT, BRIDGE_RIGHT) < 0.82
    assert mechanism_similarity(BRIDGE_LEFT, BASE) == mechanism_similarity(
        BRIDGE_RIGHT,
        BASE,
    )
    expected = min(
        (left, right),
        key=lambda cluster: cluster.state.canonical_mechanism_hash,
    )
    untouched = right if expected is left else left

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(5),
        run_occurred_at=LATER,
        clusters=(right, left),
    )

    assert plan.transition.cluster_id == expected.state.cluster_id
    assert plan.transition.member_hashes_before == expected.state.member_hashes
    assert set(plan.transition.member_hashes_after) == {
        *expected.state.member_hashes,
        hash_mechanism(BASE),
    }
    assert not set(untouched.state.member_hashes) & {
        hash_mechanism(BASE),
    }


def test_duplicate_relation_is_the_earliest_member_visible_to_owner() -> None:
    foreign_only = _cluster(BASE, owners=(OWNER_B,))
    private_plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(6),
        run_occurred_at=LATER,
        clusters=(foreign_only,),
    )
    assert private_plan.duplicate_relation is None

    mixed = _cluster(
        BRIDGE_LEFT,
        BASE,
        owners=(OWNER_B, OWNER_A),
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
    )
    visible_plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(6),
        run_occurred_at=LATER,
        clusters=(mixed,),
    )
    assert visible_plan.duplicate_relation == mixed.members[1].idea_id
    assert visible_plan.duplicate_relation != mixed.members[0].idea_id


def test_out_of_order_finish_never_moves_signal_or_candidate_time_backward() -> None:
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
        last_signal_at=LATER,
        supported_count=1,
        maturity=MechanismMaturity.CANDIDATE,
        candidate_since=LATER,
    )

    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(3),
        run_occurred_at=NOW,
        clusters=(cluster,),
    )

    transition = plan.transition
    assert transition.maturity_before is MechanismMaturity.CANDIDATE
    assert transition.maturity_after is MechanismMaturity.CANDIDATE
    assert transition.last_signal_at_before == LATER
    assert transition.last_signal_at_after == LATER
    assert transition.candidate_since_before == LATER
    assert transition.candidate_since_after == LATER


def test_cluster_rejects_maturity_that_disagrees_with_its_counts() -> None:
    with pytest.raises(ValueError, match="maturity must match"):
        _cluster(
            BASE,
            BASE,
            snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
            maturity=MechanismMaturity.SPECULATIVE,
        )


def test_cluster_rejects_candidate_time_after_last_signal() -> None:
    with pytest.raises(ValueError, match="candidate_since"):
        _cluster(
            BASE,
            last_signal_at=NOW,
            supported_count=1,
            maturity=MechanismMaturity.CANDIDATE,
            candidate_since=LATER,
        )


def test_transition_and_plan_reject_impossible_member_relationships() -> None:
    plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(8),
        run_occurred_at=NOW,
        clusters=(),
    )

    with pytest.raises(ValueError, match="prefix"):
        replace(plan.transition, member_hashes_after=())
    with pytest.raises(ValueError, match="mechanism_hash"):
        replace(plan, mechanism_hash="f" * 64)


def test_transition_rejects_canonical_or_occurrence_member_contradictions() -> None:
    new_plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(9),
        run_occurred_at=NOW,
        clusters=(),
    )
    with pytest.raises(ValueError, match="canonical"):
        replace(
            new_plan.transition,
            cluster_id="f" * 64,
            canonical_mechanism_hash="f" * 64,
        )

    existing_plan = plan_occurrence(
        owner_agent_id=OWNER_A,
        mechanism=BASE,
        snapshot_hash=_snapshot(2),
        run_occurred_at=LATER,
        clusters=(_cluster(BASE),),
    )
    with pytest.raises(ValueError, match="occurrence"):
        replace(
            existing_plan.transition,
            member_hashes_before=(
                existing_plan.transition.canonical_mechanism_hash,
                "e" * 64,
            ),
            member_hashes_after=(
                existing_plan.transition.canonical_mechanism_hash,
                "e" * 64,
            ),
        )


def test_supported_evaluation_enters_candidate_at_the_signal_time() -> None:
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
        last_signal_at=NOW,
    )

    transition = plan_evaluation_transition(
        cluster=cluster.state,
        previous_verdict=None,
        current_verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at=LATER,
    )

    assert transition == EvaluationTransition(
        previous_verdict=None,
        current_verdict=EvaluationVerdict.SUPPORTED,
        supported_count_before=0,
        supported_count_after=1,
        refuted_count_before=0,
        refuted_count_after=0,
        maturity_before=MechanismMaturity.INCUBATING,
        maturity_after=MechanismMaturity.CANDIDATE,
        candidate_since_before=None,
        candidate_since_after=LATER,
        last_signal_at_before=NOW,
        last_signal_at_after=LATER,
    )


def test_evaluation_transition_is_frozen_and_slotted() -> None:
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
    )
    transition = plan_evaluation_transition(
        cluster=cluster.state,
        previous_verdict=None,
        current_verdict=EvaluationVerdict.INCONCLUSIVE,
        evaluated_at=LATER,
    )

    assert not hasattr(transition, "__dict__")
    with pytest.raises(FrozenInstanceError):
        transition.supported_count_after = 99  # type: ignore[misc]


def test_refutation_demotes_candidate_and_clears_candidate_since() -> None:
    candidate_since = NOW - timedelta(hours=1)
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
        last_signal_at=NOW,
        supported_count=1,
        maturity=MechanismMaturity.CANDIDATE,
        candidate_since=candidate_since,
    )

    transition = plan_evaluation_transition(
        cluster=cluster.state,
        previous_verdict=EvaluationVerdict.SUPPORTED,
        current_verdict=EvaluationVerdict.REFUTED,
        evaluated_at=LATER,
    )

    assert transition.supported_count_before == 1
    assert transition.supported_count_after == 0
    assert transition.refuted_count_before == 0
    assert transition.refuted_count_after == 1
    assert transition.maturity_before is MechanismMaturity.CANDIDATE
    assert transition.maturity_after is MechanismMaturity.INCUBATING
    assert transition.candidate_since_before == candidate_since
    assert transition.candidate_since_after is None
    assert transition.last_signal_at_after == LATER


def test_candidate_reentry_resets_candidate_since_to_the_new_signal() -> None:
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
        last_signal_at=NOW,
        refuted_count=1,
    )

    transition = plan_evaluation_transition(
        cluster=cluster.state,
        previous_verdict=EvaluationVerdict.REFUTED,
        current_verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at=LATER,
    )

    assert transition.supported_count_after == 1
    assert transition.refuted_count_after == 0
    assert transition.maturity_before is MechanismMaturity.INCUBATING
    assert transition.maturity_after is MechanismMaturity.CANDIDATE
    assert transition.candidate_since_before is None
    assert transition.candidate_since_after == LATER


def test_repeated_effective_verdict_does_not_double_count() -> None:
    cluster = _cluster(
        BASE,
        BASE,
        snapshot_hashes=frozenset({_snapshot(1), _snapshot(2)}),
        last_signal_at=NOW,
        supported_count=1,
        maturity=MechanismMaturity.CANDIDATE,
        candidate_since=NOW,
    )

    transition = plan_evaluation_transition(
        cluster=cluster.state,
        previous_verdict=EvaluationVerdict.SUPPORTED,
        current_verdict=EvaluationVerdict.SUPPORTED,
        evaluated_at=LATER,
    )

    assert transition.supported_count_before == 1
    assert transition.supported_count_after == 1
    assert transition.refuted_count_before == 0
    assert transition.refuted_count_after == 0
    assert transition.candidate_since_before == NOW
    assert transition.candidate_since_after == NOW


def test_evaluation_rejects_a_clock_behind_the_cluster_signal() -> None:
    cluster = _cluster(BASE, last_signal_at=LATER)

    with pytest.raises(ValueError, match="clock|earlier|last_signal"):
        plan_evaluation_transition(
            cluster=cluster.state,
            previous_verdict=None,
            current_verdict=EvaluationVerdict.INCONCLUSIVE,
            evaluated_at=NOW,
        )
