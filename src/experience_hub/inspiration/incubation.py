"""Pure online clustering and recurrence transition planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from experience_hub.clock import require_utc
from experience_hub.inspiration.hashing import (
    NEAR_DUPLICATE_THRESHOLD,
    hash_mechanism,
    mechanism_similarity,
)
from experience_hub.inspiration.models import (
    EvaluationVerdict,
    MechanismIncubation,
    MechanismMaturity,
)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _require_hash(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a timezone-aware datetime")
    try:
        return require_utc(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timezone-aware datetime") from error


def _maturity(
    *,
    distinct_snapshot_count: int,
    distinct_adopter_count: int,
    supported_count: int,
    refuted_count: int,
) -> MechanismMaturity:
    if (
        supported_count >= 1
        and refuted_count == 0
        or distinct_adopter_count >= 2
    ):
        return MechanismMaturity.CANDIDATE
    if distinct_snapshot_count >= 2:
        return MechanismMaturity.INCUBATING
    return MechanismMaturity.SPECULATIVE


@dataclass(frozen=True, slots=True)
class IncubationMember:
    """One private idea member loaded in ascending generation-event order."""

    idea_id: UUID
    owner_agent_id: UUID
    mechanism: str
    mechanism_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.idea_id, UUID):
            raise TypeError("idea_id must be a UUID")
        if not isinstance(self.owner_agent_id, UUID):
            raise TypeError("owner_agent_id must be a UUID")
        if not isinstance(self.mechanism, str):
            raise TypeError("mechanism must be a string")
        if hash_mechanism(self.mechanism) != self.mechanism_hash:
            raise ValueError("mechanism_hash must match mechanism")


@dataclass(frozen=True, slots=True)
class IncubationCluster:
    """Projection plus private source facts needed by the pure planner."""

    state: MechanismIncubation
    members: tuple[IncubationMember, ...]
    snapshot_hashes: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.state, MechanismIncubation):
            raise TypeError("state must be a MechanismIncubation")
        if not isinstance(self.members, tuple) or not self.members or any(
            not isinstance(member, IncubationMember) for member in self.members
        ):
            raise TypeError("members must be a nonempty immutable member tuple")
        if not isinstance(self.snapshot_hashes, frozenset):
            raise TypeError("snapshot_hashes must be a frozenset")
        for snapshot_hash in self.snapshot_hashes:
            _require_hash("snapshot_hash", snapshot_hash)
        member_hashes = tuple(
            dict.fromkeys(member.mechanism_hash for member in self.members)
        )
        if self.state.occurrence_count != len(self.members):
            raise ValueError("occurrence_count must equal loaded member count")
        if self.state.distinct_snapshot_count != len(self.snapshot_hashes):
            raise ValueError(
                "distinct_snapshot_count must equal loaded snapshot hash count"
            )
        if self.state.distinct_snapshot_count > self.state.occurrence_count:
            raise ValueError("distinct snapshots cannot exceed occurrences")
        if self.state.member_hashes != member_hashes:
            raise ValueError("member_hashes must follow first occurrence order")
        if self.state.canonical_mechanism_hash != self.members[0].mechanism_hash:
            raise ValueError("canonical mechanism must be the earliest member")
        if self.state.cluster_id != self.state.canonical_mechanism_hash:
            raise ValueError("cluster identity must equal its canonical hash")
        if len({member.idea_id for member in self.members}) != len(self.members):
            raise ValueError("members must not repeat an idea identity")
        if self.state.distinct_adopter_count > self.state.occurrence_count:
            raise ValueError("distinct adopters cannot exceed occurrences")
        expected_maturity = _maturity(
            distinct_snapshot_count=self.state.distinct_snapshot_count,
            distinct_adopter_count=self.state.distinct_adopter_count,
            supported_count=self.state.supported_count,
            refuted_count=self.state.refuted_count,
        )
        if self.state.maturity is not expected_maturity:
            raise ValueError("maturity must match the loaded cluster counts")
        if (
            self.state.candidate_since is not None
            and self.state.candidate_since > self.state.last_signal_at
        ):
            raise ValueError("candidate_since cannot follow last_signal_at")


@dataclass(frozen=True, slots=True)
class ClusterTransition:
    """Complete event-ready before/after values for one occurrence."""

    cluster_id: str
    canonical_mechanism_hash: str
    member_hashes_before: tuple[str, ...]
    member_hashes_after: tuple[str, ...]
    occurrence_count_before: int
    occurrence_count_after: int
    distinct_snapshot_count_before: int
    distinct_snapshot_count_after: int
    distinct_adopter_count_before: int
    distinct_adopter_count_after: int
    supported_count_before: int
    supported_count_after: int
    refuted_count_before: int
    refuted_count_after: int
    maturity_before: MechanismMaturity | None
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime | None
    last_signal_at_after: datetime

    def __post_init__(self) -> None:
        _require_hash("cluster_id", self.cluster_id)
        _require_hash(
            "canonical_mechanism_hash",
            self.canonical_mechanism_hash,
        )
        if self.cluster_id != self.canonical_mechanism_hash:
            raise ValueError("cluster identity must equal its canonical hash")
        for values in (self.member_hashes_before, self.member_hashes_after):
            if not isinstance(values, tuple):
                raise TypeError("member hashes must be immutable tuples")
            for value in values:
                _require_hash("member_hash", value)
            if len(values) != len(set(values)):
                raise ValueError("member hashes must be unique")
        count_pairs = (
            (self.occurrence_count_before, self.occurrence_count_after),
            (
                self.distinct_snapshot_count_before,
                self.distinct_snapshot_count_after,
            ),
            (
                self.distinct_adopter_count_before,
                self.distinct_adopter_count_after,
            ),
            (self.supported_count_before, self.supported_count_after),
            (self.refuted_count_before, self.refuted_count_after),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for pair in count_pairs
            for value in pair
        ):
            raise ValueError("cluster counters must be non-negative strict integers")
        if self.occurrence_count_after != self.occurrence_count_before + 1:
            raise ValueError("an occurrence transition must increment occurrence count")
        if self.distinct_snapshot_count_after not in {
            self.distinct_snapshot_count_before,
            self.distinct_snapshot_count_before + 1,
        }:
            raise ValueError("distinct snapshot count may increase by at most one")
        if (
            self.distinct_adopter_count_before
            != self.distinct_adopter_count_after
            or self.supported_count_before != self.supported_count_after
            or self.refuted_count_before != self.refuted_count_after
        ):
            raise ValueError("an occurrence cannot change evaluation/adoption counts")
        if (
            self.member_hashes_after[: len(self.member_hashes_before)]
            != self.member_hashes_before
            or len(self.member_hashes_after)
            not in {
                len(self.member_hashes_before),
                len(self.member_hashes_before) + 1,
            }
            or not self.member_hashes_after
        ):
            raise ValueError(
                "member hashes after must retain and extend the before prefix"
            )
        if (
            self.member_hashes_after[0] != self.canonical_mechanism_hash
            or (
                self.member_hashes_before
                and self.member_hashes_before[0]
                != self.canonical_mechanism_hash
            )
        ):
            raise ValueError("canonical mechanism must be the first member")
        if (
            len(self.member_hashes_before) > self.occurrence_count_before
            or len(self.member_hashes_after) > self.occurrence_count_after
        ):
            raise ValueError("unique members cannot exceed occurrence count")
        if self.occurrence_count_before == 0:
            if (
                self.member_hashes_before
                or self.distinct_snapshot_count_before != 0
                or self.distinct_adopter_count_before != 0
                or self.supported_count_before != 0
                or self.refuted_count_before != 0
                or self.maturity_before is not None
                or self.candidate_since_before is not None
                or self.last_signal_at_before is not None
            ):
                raise ValueError("a first occurrence requires an empty before-state")
        elif (
            not self.member_hashes_before
            or not 1
            <= self.distinct_snapshot_count_before
            <= self.occurrence_count_before
            or self.maturity_before is None
            or self.last_signal_at_before is None
        ):
            raise ValueError("an existing occurrence requires a complete before-state")
        if not 1 <= self.distinct_snapshot_count_after <= self.occurrence_count_after:
            raise ValueError("after snapshot count must fit occurrence count")
        if (
            self.distinct_adopter_count_before > self.occurrence_count_before
            or self.distinct_adopter_count_after > self.occurrence_count_after
        ):
            raise ValueError("distinct adopters cannot exceed occurrences")
        if not isinstance(self.maturity_after, MechanismMaturity):
            raise TypeError("maturity_after must be a MechanismMaturity")
        if self.maturity_before is not None and not isinstance(
            self.maturity_before,
            MechanismMaturity,
        ):
            raise TypeError("maturity_before must be a MechanismMaturity or None")
        if self.candidate_since_before is not None:
            _require_utc("candidate_since_before", self.candidate_since_before)
        if self.candidate_since_after is not None:
            _require_utc("candidate_since_after", self.candidate_since_after)
        if self.last_signal_at_before is not None:
            _require_utc("last_signal_at_before", self.last_signal_at_before)
        _require_utc("last_signal_at_after", self.last_signal_at_after)
        if (
            self.last_signal_at_before is not None
            and self.last_signal_at_after < self.last_signal_at_before
        ):
            raise ValueError("last_signal_at must not move backward")
        if (
            self.maturity_before is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_before is None):
            raise ValueError("candidate_since_before must match candidate maturity")
        if (
            self.maturity_after is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_after is None):
            raise ValueError("candidate_since_after must match candidate maturity")
        if (
            self.candidate_since_before is not None
            and self.last_signal_at_before is not None
            and self.candidate_since_before > self.last_signal_at_before
        ):
            raise ValueError("candidate_since_before cannot follow last signal")
        if (
            self.candidate_since_after is not None
            and self.candidate_since_after > self.last_signal_at_after
        ):
            raise ValueError("candidate_since_after cannot follow last signal")
        if self.occurrence_count_before > 0:
            expected_before = _maturity(
                distinct_snapshot_count=self.distinct_snapshot_count_before,
                distinct_adopter_count=self.distinct_adopter_count_before,
                supported_count=self.supported_count_before,
                refuted_count=self.refuted_count_before,
            )
            if self.maturity_before is not expected_before:
                raise ValueError("maturity_before must match before counters")
        expected_after = _maturity(
            distinct_snapshot_count=self.distinct_snapshot_count_after,
            distinct_adopter_count=self.distinct_adopter_count_after,
            supported_count=self.supported_count_after,
            refuted_count=self.refuted_count_after,
        )
        if self.maturity_after is not expected_after:
            raise ValueError("maturity_after must match after counters")
        if (
            self.maturity_before is MechanismMaturity.CANDIDATE
            and self.candidate_since_after != self.candidate_since_before
        ):
            raise ValueError("candidate recurrence must retain candidate_since")


@dataclass(frozen=True, slots=True)
class EvaluationTransition:
    """Complete effective-count and maturity change for one evaluation revision."""

    previous_verdict: EvaluationVerdict | None
    current_verdict: EvaluationVerdict
    supported_count_before: int
    supported_count_after: int
    refuted_count_before: int
    refuted_count_after: int
    maturity_before: MechanismMaturity
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime
    last_signal_at_after: datetime

    def __post_init__(self) -> None:
        if self.previous_verdict is not None and not isinstance(
            self.previous_verdict,
            EvaluationVerdict,
        ):
            raise TypeError(
                "previous_verdict must be an EvaluationVerdict or None"
            )
        if not isinstance(self.current_verdict, EvaluationVerdict):
            raise TypeError("current_verdict must be an EvaluationVerdict")
        for name, value in (
            ("supported_count_before", self.supported_count_before),
            ("supported_count_after", self.supported_count_after),
            ("refuted_count_before", self.refuted_count_before),
            ("refuted_count_after", self.refuted_count_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative strict integer")
        if not isinstance(self.maturity_before, MechanismMaturity) or not isinstance(
            self.maturity_after,
            MechanismMaturity,
        ):
            raise TypeError("maturity values must be MechanismMaturity members")
        if self.candidate_since_before is not None:
            _require_utc("candidate_since_before", self.candidate_since_before)
        if self.candidate_since_after is not None:
            _require_utc("candidate_since_after", self.candidate_since_after)
        before_signal = _require_utc(
            "last_signal_at_before",
            self.last_signal_at_before,
        )
        after_signal = _require_utc(
            "last_signal_at_after",
            self.last_signal_at_after,
        )
        if after_signal < before_signal:
            raise ValueError("evaluation signal time cannot move backward")
        if (
            self.maturity_before is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_before is None):
            raise ValueError(
                "candidate_since_before must match candidate maturity"
            )
        if (
            self.maturity_after is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_after is None):
            raise ValueError(
                "candidate_since_after must match candidate maturity"
            )
        if (
            self.candidate_since_before is not None
            and self.candidate_since_before > before_signal
        ):
            raise ValueError("candidate_since_before cannot follow last signal")
        if (
            self.candidate_since_after is not None
            and self.candidate_since_after > after_signal
        ):
            raise ValueError("candidate_since_after cannot follow last signal")
        expected_supported = self.supported_count_before
        expected_refuted = self.refuted_count_before
        if self.previous_verdict is EvaluationVerdict.SUPPORTED:
            expected_supported -= 1
        elif self.previous_verdict is EvaluationVerdict.REFUTED:
            expected_refuted -= 1
        if self.current_verdict is EvaluationVerdict.SUPPORTED:
            expected_supported += 1
        elif self.current_verdict is EvaluationVerdict.REFUTED:
            expected_refuted += 1
        if (
            expected_supported < 0
            or expected_refuted < 0
            or self.supported_count_after != expected_supported
            or self.refuted_count_after != expected_refuted
        ):
            raise ValueError(
                "effective counts must match the evaluation verdict revision"
            )
        if self.maturity_before is MechanismMaturity.CANDIDATE:
            if (
                self.maturity_after is MechanismMaturity.CANDIDATE
                and self.candidate_since_after != self.candidate_since_before
            ):
                raise ValueError(
                    "candidate maturity must retain candidate_since"
                )
        elif (
            self.maturity_after is MechanismMaturity.CANDIDATE
            and self.candidate_since_after != after_signal
        ):
            raise ValueError(
                "candidate re-entry must start at the current signal"
            )


@dataclass(frozen=True, slots=True)
class AdoptionTransition:
    """Complete distinct-adopter and maturity change for one idea adoption."""

    distinct_adopter_count_before: int
    distinct_adopter_count_after: int
    maturity_before: MechanismMaturity
    maturity_after: MechanismMaturity
    candidate_since_before: datetime | None
    candidate_since_after: datetime | None
    last_signal_at_before: datetime
    last_signal_at_after: datetime

    def __post_init__(self) -> None:
        for name, value in (
            (
                "distinct_adopter_count_before",
                self.distinct_adopter_count_before,
            ),
            (
                "distinct_adopter_count_after",
                self.distinct_adopter_count_after,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative strict integer")
        if self.distinct_adopter_count_after not in {
            self.distinct_adopter_count_before,
            self.distinct_adopter_count_before + 1,
        }:
            raise ValueError(
                "one adoption may add at most one distinct adopter"
            )
        if not isinstance(self.maturity_before, MechanismMaturity) or not isinstance(
            self.maturity_after,
            MechanismMaturity,
        ):
            raise TypeError("maturity values must be MechanismMaturity members")
        before_signal = _require_utc(
            "last_signal_at_before",
            self.last_signal_at_before,
        )
        after_signal = _require_utc(
            "last_signal_at_after",
            self.last_signal_at_after,
        )
        if after_signal < before_signal:
            raise ValueError("adoption signal time cannot move backward")
        if self.candidate_since_before is not None:
            before_candidate = _require_utc(
                "candidate_since_before",
                self.candidate_since_before,
            )
            if before_candidate > before_signal:
                raise ValueError(
                    "candidate_since_before cannot follow last signal"
                )
        if self.candidate_since_after is not None:
            after_candidate = _require_utc(
                "candidate_since_after",
                self.candidate_since_after,
            )
            if after_candidate > after_signal:
                raise ValueError(
                    "candidate_since_after cannot follow last signal"
                )
        if (
            self.maturity_before is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_before is None):
            raise ValueError(
                "candidate_since_before must match candidate maturity"
            )
        if (
            self.maturity_after is MechanismMaturity.CANDIDATE
        ) is (self.candidate_since_after is None):
            raise ValueError(
                "candidate_since_after must match candidate maturity"
            )
        if self.maturity_before is MechanismMaturity.CANDIDATE:
            if (
                self.maturity_after is MechanismMaturity.CANDIDATE
                and self.candidate_since_after
                != self.candidate_since_before
            ):
                raise ValueError(
                    "candidate maturity must retain candidate_since"
                )
        elif (
            self.maturity_after is MechanismMaturity.CANDIDATE
            and self.candidate_since_after != after_signal
        ):
            raise ValueError(
                "candidate entry must start at the adoption signal"
            )


@dataclass(frozen=True, slots=True)
class OccurrencePlan:
    """A recurrence always retained as a new immutable idea and occurrence."""

    mechanism_hash: str
    duplicate_relation: UUID | None
    maximum_similarity: float | None
    transition: ClusterTransition
    create_idea: bool = True
    create_occurrence: bool = True

    def __post_init__(self) -> None:
        _require_hash("mechanism_hash", self.mechanism_hash)
        if self.duplicate_relation is not None and not isinstance(
            self.duplicate_relation,
            UUID,
        ):
            raise TypeError("duplicate_relation must be a UUID or None")
        if self.maximum_similarity is not None and (
            isinstance(self.maximum_similarity, bool)
            or not isinstance(self.maximum_similarity, float)
            or not 0.0 <= self.maximum_similarity <= 1.0
        ):
            raise ValueError("maximum_similarity must be a float from zero to one")
        if not isinstance(self.transition, ClusterTransition):
            raise TypeError("transition must be a ClusterTransition")
        if self.create_idea is not True or self.create_occurrence is not True:
            raise ValueError("occurrences must always request both immutable rows")
        transition = self.transition
        before = transition.member_hashes_before
        after = transition.member_hashes_after
        if self.mechanism_hash not in after:
            raise ValueError("mechanism_hash must belong to transition members")
        if len(after) == len(before) + 1 and after[-1] != self.mechanism_hash:
            raise ValueError("an appended member must match mechanism_hash")
        if transition.occurrence_count_before == 0:
            if (
                transition.cluster_id != self.mechanism_hash
                or self.duplicate_relation is not None
                or self.maximum_similarity is not None
            ):
                raise ValueError("a new cluster must be bound to its mechanism")
        elif (
            self.maximum_similarity is None
            or self.maximum_similarity < NEAR_DUPLICATE_THRESHOLD
        ):
            raise ValueError(
                "an existing cluster requires qualifying maximum_similarity"
            )


def _maximum_similarity(
    mechanism: str,
    cluster: IncubationCluster,
) -> float:
    return max(
        mechanism_similarity(mechanism, member.mechanism)
        for member in cluster.members
    )


def _new_cluster_transition(
    *,
    mechanism_hash: str,
    run_occurred_at: datetime,
) -> ClusterTransition:
    return ClusterTransition(
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
        last_signal_at_after=run_occurred_at,
    )


def _existing_cluster_transition(
    *,
    cluster: IncubationCluster,
    mechanism_hash: str,
    snapshot_hash: str,
    run_occurred_at: datetime,
) -> ClusterTransition:
    state = cluster.state
    member_hashes_after = state.member_hashes
    if mechanism_hash not in member_hashes_after:
        member_hashes_after = (*member_hashes_after, mechanism_hash)
    distinct_snapshot_count = state.distinct_snapshot_count + (
        snapshot_hash not in cluster.snapshot_hashes
    )
    maturity_after = _maturity(
        distinct_snapshot_count=distinct_snapshot_count,
        distinct_adopter_count=state.distinct_adopter_count,
        supported_count=state.supported_count,
        refuted_count=state.refuted_count,
    )
    last_signal_at = max(state.last_signal_at, run_occurred_at)
    if maturity_after is MechanismMaturity.CANDIDATE:
        candidate_since = (
            state.candidate_since
            if state.maturity is MechanismMaturity.CANDIDATE
            else last_signal_at
        )
    else:
        candidate_since = None
    return ClusterTransition(
        cluster_id=state.cluster_id,
        canonical_mechanism_hash=state.canonical_mechanism_hash,
        member_hashes_before=state.member_hashes,
        member_hashes_after=member_hashes_after,
        occurrence_count_before=state.occurrence_count,
        occurrence_count_after=state.occurrence_count + 1,
        distinct_snapshot_count_before=state.distinct_snapshot_count,
        distinct_snapshot_count_after=distinct_snapshot_count,
        distinct_adopter_count_before=state.distinct_adopter_count,
        distinct_adopter_count_after=state.distinct_adopter_count,
        supported_count_before=state.supported_count,
        supported_count_after=state.supported_count,
        refuted_count_before=state.refuted_count,
        refuted_count_after=state.refuted_count,
        maturity_before=state.maturity,
        maturity_after=maturity_after,
        candidate_since_before=state.candidate_since,
        candidate_since_after=candidate_since,
        last_signal_at_before=state.last_signal_at,
        last_signal_at_after=last_signal_at,
    )


def plan_evaluation_transition(
    *,
    cluster: MechanismIncubation,
    previous_verdict: EvaluationVerdict | None,
    current_verdict: EvaluationVerdict,
    evaluated_at: datetime,
) -> EvaluationTransition:
    """Revise one evaluator's effective signal without double counting."""
    if not isinstance(cluster, MechanismIncubation):
        raise TypeError("cluster must be a MechanismIncubation")
    if previous_verdict is not None and not isinstance(
        previous_verdict,
        EvaluationVerdict,
    ):
        raise TypeError(
            "previous_verdict must be an EvaluationVerdict or None"
        )
    if not isinstance(current_verdict, EvaluationVerdict):
        raise TypeError("current_verdict must be an EvaluationVerdict")
    retained_at = _require_utc("evaluated_at", evaluated_at)
    if retained_at < cluster.last_signal_at:
        raise ValueError("evaluation time cannot precede cluster last_signal_at")
    expected_before = _maturity(
        distinct_snapshot_count=cluster.distinct_snapshot_count,
        distinct_adopter_count=cluster.distinct_adopter_count,
        supported_count=cluster.supported_count,
        refuted_count=cluster.refuted_count,
    )
    if cluster.maturity is not expected_before:
        raise ValueError("cluster maturity does not match its effective counts")

    supported_after = cluster.supported_count
    refuted_after = cluster.refuted_count
    if previous_verdict is EvaluationVerdict.SUPPORTED:
        supported_after -= 1
    elif previous_verdict is EvaluationVerdict.REFUTED:
        refuted_after -= 1
    if current_verdict is EvaluationVerdict.SUPPORTED:
        supported_after += 1
    elif current_verdict is EvaluationVerdict.REFUTED:
        refuted_after += 1
    if supported_after < 0 or refuted_after < 0:
        raise ValueError(
            "previous evaluation contribution is absent from cluster counts"
        )

    maturity_after = _maturity(
        distinct_snapshot_count=cluster.distinct_snapshot_count,
        distinct_adopter_count=cluster.distinct_adopter_count,
        supported_count=supported_after,
        refuted_count=refuted_after,
    )
    candidate_since_after = (
        cluster.candidate_since
        if (
            cluster.maturity is MechanismMaturity.CANDIDATE
            and maturity_after is MechanismMaturity.CANDIDATE
        )
        else (
            retained_at
            if maturity_after is MechanismMaturity.CANDIDATE
            else None
        )
    )
    return EvaluationTransition(
        previous_verdict=previous_verdict,
        current_verdict=current_verdict,
        supported_count_before=cluster.supported_count,
        supported_count_after=supported_after,
        refuted_count_before=cluster.refuted_count,
        refuted_count_after=refuted_after,
        maturity_before=cluster.maturity,
        maturity_after=maturity_after,
        candidate_since_before=cluster.candidate_since,
        candidate_since_after=candidate_since_after,
        last_signal_at_before=cluster.last_signal_at,
        last_signal_at_after=retained_at,
    )


def plan_adoption_transition(
    *,
    cluster: MechanismIncubation,
    owner_already_adopted: bool,
    adopted_at: datetime,
) -> AdoptionTransition:
    """Apply one owner's first cluster adoption as an effective signal."""
    if not isinstance(cluster, MechanismIncubation):
        raise TypeError("cluster must be a MechanismIncubation")
    if not isinstance(owner_already_adopted, bool):
        raise TypeError("owner_already_adopted must be a bool")
    retained_at = _require_utc("adopted_at", adopted_at)
    if retained_at < cluster.last_signal_at:
        raise ValueError("adoption time cannot precede cluster last_signal_at")
    if cluster.distinct_adopter_count > cluster.occurrence_count:
        raise ValueError("distinct adopters cannot exceed occurrences")
    expected_before = _maturity(
        distinct_snapshot_count=cluster.distinct_snapshot_count,
        distinct_adopter_count=cluster.distinct_adopter_count,
        supported_count=cluster.supported_count,
        refuted_count=cluster.refuted_count,
    )
    if cluster.maturity is not expected_before:
        raise ValueError("cluster maturity does not match its effective counts")

    distinct_after = cluster.distinct_adopter_count + (
        not owner_already_adopted
    )
    if distinct_after > cluster.occurrence_count:
        raise ValueError("distinct adopters cannot exceed occurrences")
    maturity_after = _maturity(
        distinct_snapshot_count=cluster.distinct_snapshot_count,
        distinct_adopter_count=distinct_after,
        supported_count=cluster.supported_count,
        refuted_count=cluster.refuted_count,
    )
    candidate_since_after = (
        cluster.candidate_since
        if (
            cluster.maturity is MechanismMaturity.CANDIDATE
            and maturity_after is MechanismMaturity.CANDIDATE
        )
        else (
            retained_at
            if maturity_after is MechanismMaturity.CANDIDATE
            else None
        )
    )
    return AdoptionTransition(
        distinct_adopter_count_before=cluster.distinct_adopter_count,
        distinct_adopter_count_after=distinct_after,
        maturity_before=cluster.maturity,
        maturity_after=maturity_after,
        candidate_since_before=cluster.candidate_since,
        candidate_since_after=candidate_since_after,
        last_signal_at_before=cluster.last_signal_at,
        last_signal_at_after=retained_at,
    )


def plan_occurrence(
    *,
    owner_agent_id: object,
    mechanism: object,
    snapshot_hash: object,
    run_occurred_at: object,
    clusters: object,
) -> OccurrencePlan:
    """Assign one recurrence by maximum-member similarity without merging."""
    if not isinstance(owner_agent_id, UUID):
        raise TypeError("owner_agent_id must be a UUID")
    if not isinstance(mechanism, str):
        raise TypeError("mechanism must be a string")
    retained_snapshot_hash = _require_hash("snapshot_hash", snapshot_hash)
    retained_occurred_at = _require_utc("run_occurred_at", run_occurred_at)
    if not isinstance(clusters, tuple) or any(
        not isinstance(cluster, IncubationCluster) for cluster in clusters
    ):
        raise TypeError("clusters must be an immutable IncubationCluster tuple")
    retained_clusters: tuple[IncubationCluster, ...] = clusters
    cluster_ids = tuple(cluster.state.cluster_id for cluster in retained_clusters)
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("clusters must not repeat an identity")
    all_member_hashes = tuple(
        member_hash
        for cluster in retained_clusters
        for member_hash in cluster.state.member_hashes
    )
    if len(set(all_member_hashes)) != len(all_member_hashes):
        raise ValueError("member hashes must belong to only one cluster")
    mechanism_hash = hash_mechanism(mechanism)

    eligible = tuple(
        (cluster, _maximum_similarity(mechanism, cluster))
        for cluster in retained_clusters
    )
    eligible = tuple(
        (cluster, similarity)
        for cluster, similarity in eligible
        if similarity >= NEAR_DUPLICATE_THRESHOLD
    )
    if not eligible:
        return OccurrencePlan(
            mechanism_hash=mechanism_hash,
            duplicate_relation=None,
            maximum_similarity=None,
            transition=_new_cluster_transition(
                mechanism_hash=mechanism_hash,
                run_occurred_at=retained_occurred_at,
            ),
        )

    cluster, maximum_similarity = min(
        eligible,
        key=lambda match: (
            -match[1],
            match[0].state.canonical_mechanism_hash,
        ),
    )
    duplicate_relation = next(
        (
            member.idea_id
            for member in cluster.members
            if member.owner_agent_id == owner_agent_id
        ),
        None,
    )
    return OccurrencePlan(
        mechanism_hash=mechanism_hash,
        duplicate_relation=duplicate_relation,
        maximum_similarity=maximum_similarity,
        transition=_existing_cluster_transition(
            cluster=cluster,
            mechanism_hash=mechanism_hash,
            snapshot_hash=retained_snapshot_hash,
            run_occurred_at=retained_occurred_at,
        ),
    )


__all__ = [
    "NEAR_DUPLICATE_THRESHOLD",
    "AdoptionTransition",
    "ClusterTransition",
    "EvaluationTransition",
    "IncubationCluster",
    "IncubationMember",
    "OccurrencePlan",
    "plan_adoption_transition",
    "plan_evaluation_transition",
    "plan_occurrence",
]
