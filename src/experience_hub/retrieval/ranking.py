"""Pure candidate selection and deterministic experience ranking."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from experience_hub.clock import require_utc
from experience_hub.experiences.models import Temperature
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.retrieval.tokenizer import TermCue, TermKind

FOCUSED_RELEVANCE_THRESHOLD = 0.05
ASSOCIATIVE_RELEVANCE_THRESHOLD = 0.02
MAX_RETRIEVAL_LIMIT = 50

_TERM_KINDS = frozenset({"word", "char_trigram", "tag", "mechanism"})
_COMPATIBLE_KINDS: dict[str, frozenset[str]] = {
    "word": frozenset({"word", "tag", "mechanism"}),
    "tag": frozenset({"tag"}),
    "mechanism": frozenset({"mechanism"}),
    "char_trigram": frozenset({"char_trigram"}),
}


class RetrievalMode(StrEnum):
    FOCUSED = "focused"
    ASSOCIATIVE = "associative"


_POOL_MULTIPLIERS: dict[
    RetrievalMode,
    dict[Temperature, int],
] = {
    RetrievalMode.FOCUSED: {
        Temperature.HOT: 4,
        Temperature.WARM: 4,
        Temperature.COLD: 2,
    },
    RetrievalMode.ASSOCIATIVE: {
        Temperature.HOT: 3,
        Temperature.WARM: 3,
        Temperature.COLD: 5,
    },
}


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite float")
    return converted


def _unit_float(name: str, value: float) -> float:
    converted = _finite_float(name, value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return converted


def _uuid(name: str, value: UUID) -> UUID:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} must be a UUID")
    return value


def _temperature(value: Temperature) -> Temperature:
    if not isinstance(value, Temperature):
        raise ValueError("temperature must be a Temperature")
    return value


def _mode(value: RetrievalMode) -> RetrievalMode:
    if not isinstance(value, RetrievalMode):
        raise ValueError("mode must be a RetrievalMode")
    return value


def _requested_limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_RETRIEVAL_LIMIT
    ):
        raise ValueError("requested_limit must be between 1 and 50")
    return value


def _timestamp(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return require_utc(value)


def _term_tuple(name: str, values: Sequence[TermCue]) -> tuple[TermCue, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of TermCue values")
    terms = tuple(values)
    if any(not isinstance(value, TermCue) for value in terms):
        raise ValueError(f"{name} must contain only TermCue values")
    return terms


def _query_cue_tuple(
    query_cues: Iterable[TermCue],
) -> tuple[TermCue, ...]:
    if isinstance(query_cues, (str, bytes)):
        raise ValueError("query_cues must contain only TermCue values")
    try:
        query = tuple(query_cues)
    except TypeError as error:
        raise ValueError(
            "query_cues must contain only TermCue values"
        ) from error
    if not query:
        raise ValueError("query_cues must not be empty")
    if any(not isinstance(value, TermCue) for value in query):
        raise ValueError("query_cues must contain only TermCue values")
    return query


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """One positive or zero-overlap candidate before temperature quotas."""

    experience_id: UUID
    temperature: Temperature
    raw_overlap: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experience_id",
            _uuid("experience_id", self.experience_id),
        )
        object.__setattr__(
            self,
            "temperature",
            _temperature(self.temperature),
        )
        overlap = _finite_float("raw_overlap", self.raw_overlap)
        if overlap < 0.0:
            raise ValueError("raw_overlap must be non-negative")
        object.__setattr__(self, "raw_overlap", overlap)


@dataclass(frozen=True, slots=True)
class RelevanceComponents:
    """Coverage and mode-adjusted relevance used by final scoring."""

    word_tag_coverage: float
    trigram_coverage: float
    lexical_or_trigram_relevance: float
    mechanism_relevance: float
    ranking_relevance: float

    def __post_init__(self) -> None:
        for field_name in (
            "word_tag_coverage",
            "trigram_coverage",
            "lexical_or_trigram_relevance",
            "mechanism_relevance",
            "ranking_relevance",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(field_name, getattr(self, field_name)),
            )


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    """Storage-independent values required to rank one experience."""

    experience_id: UUID
    temperature: Temperature
    current_version_created_at: datetime
    terms: tuple[TermCue, ...]
    activation_inputs: ActivationInputs
    source_trust: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experience_id",
            _uuid("experience_id", self.experience_id),
        )
        object.__setattr__(
            self,
            "temperature",
            _temperature(self.temperature),
        )
        object.__setattr__(
            self,
            "current_version_created_at",
            _timestamp(
                "current_version_created_at",
                self.current_version_created_at,
            ),
        )
        if not isinstance(self.terms, tuple):
            raise ValueError("terms must be an immutable tuple")
        object.__setattr__(self, "terms", _term_tuple("terms", self.terms))
        if not isinstance(self.activation_inputs, ActivationInputs):
            raise ValueError("activation_inputs must be ActivationInputs")
        object.__setattr__(
            self,
            "source_trust",
            _unit_float("source_trust", self.source_trust),
        )


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A candidate with every component of its final deterministic score."""

    experience_id: UUID
    temperature: Temperature
    current_version_created_at: datetime
    score: float
    ranking_relevance: float
    lexical_or_trigram_relevance: float
    mechanism_relevance: float
    word_tag_coverage: float
    trigram_coverage: float
    activation: float
    confidence: float
    importance: float
    source_trust: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experience_id",
            _uuid("experience_id", self.experience_id),
        )
        object.__setattr__(
            self,
            "temperature",
            _temperature(self.temperature),
        )
        object.__setattr__(
            self,
            "current_version_created_at",
            _timestamp(
                "current_version_created_at",
                self.current_version_created_at,
            ),
        )
        for field_name in (
            "score",
            "ranking_relevance",
            "lexical_or_trigram_relevance",
            "mechanism_relevance",
            "word_tag_coverage",
            "trigram_coverage",
            "activation",
            "confidence",
            "importance",
            "source_trust",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(field_name, getattr(self, field_name)),
            )


def _term_kind(name: str, value: TermKind) -> str:
    if not isinstance(value, str) or value not in _TERM_KINDS:
        raise ValueError(f"{name} is not a supported term kind")
    return value


def cue_kinds_compatible(
    query_kind: TermKind,
    candidate_kind: TermKind,
) -> bool:
    """Return whether a query cue kind may match a candidate term kind."""
    query = _term_kind("query_kind", query_kind)
    candidate = _term_kind("candidate_kind", candidate_kind)
    return candidate in _COMPATIBLE_KINDS[query]


def _max_term_weights(
    cues: Iterable[TermCue],
    *,
    name: str,
) -> dict[tuple[str, str], float]:
    maximums: dict[tuple[str, str], float] = {}
    for cue in cues:
        if not isinstance(cue, TermCue):
            raise ValueError(f"{name} must contain only TermCue values")
        key = (cue.term, cue.term_kind)
        maximums[key] = max(cue.weight, maximums.get(key, 0.0))
    return maximums


def _matched_weight(
    *,
    term: str,
    query_kind: str,
    candidate_weights: dict[tuple[str, str], float],
    candidate_kinds: frozenset[str],
) -> float:
    return max(
        (
            weight
            for (candidate_term, candidate_kind), weight
            in candidate_weights.items()
            if candidate_term == term
            and candidate_kind in candidate_kinds
            and candidate_kind in _COMPATIBLE_KINDS[query_kind]
        ),
        default=0.0,
    )


def raw_overlap(
    query_cues: Iterable[TermCue],
    candidate_terms: Iterable[TermCue],
) -> float:
    """Sum each unique query cue's maximum compatible exact-term overlap."""
    query_weights = _max_term_weights(query_cues, name="query_cues")
    candidate_weights = _max_term_weights(
        candidate_terms,
        name="candidate_terms",
    )
    ordered_query_weights = sorted(query_weights.items())
    return math.fsum(
        min(
            query_weight,
            _matched_weight(
                term=term,
                query_kind=query_kind,
                candidate_weights=candidate_weights,
                candidate_kinds=_TERM_KINDS,
            ),
        )
        for (term, query_kind), query_weight in ordered_query_weights
    )


def _kind_set(
    name: str,
    kinds: Iterable[TermKind],
) -> frozenset[str]:
    values = frozenset(_term_kind(name, kind) for kind in kinds)
    return values


def coverage(
    query_cues: Iterable[TermCue],
    candidate_terms: Iterable[TermCue],
    *,
    query_kinds: Iterable[TermKind],
    candidate_kinds: Iterable[TermKind] | None = None,
) -> float:
    """Compute compatible matched weight over one query cue family."""
    included_query_kinds = _kind_set("query_kinds", query_kinds)
    included_candidate_kinds = (
        _TERM_KINDS
        if candidate_kinds is None
        else _kind_set("candidate_kinds", candidate_kinds)
    )
    query_weights = {
        key: weight
        for key, weight in _max_term_weights(
            query_cues,
            name="query_cues",
        ).items()
        if key[1] in included_query_kinds
    }
    ordered_query_weights = sorted(query_weights.items())
    denominator = math.fsum(
        query_weight for _, query_weight in ordered_query_weights
    )
    if denominator == 0.0:
        return 0.0
    candidate_weights = _max_term_weights(
        candidate_terms,
        name="candidate_terms",
    )
    matched = math.fsum(
        min(
            query_weight,
            _matched_weight(
                term=term,
                query_kind=query_kind,
                candidate_weights=candidate_weights,
                candidate_kinds=included_candidate_kinds,
            ),
        )
        for (term, query_kind), query_weight in ordered_query_weights
    )
    return min(1.0, max(0.0, matched / denominator))


def relevance_components(
    query_cues: Iterable[TermCue],
    candidate_terms: Iterable[TermCue],
    mode: RetrievalMode,
) -> RelevanceComponents:
    """Compute all deterministic relevance components for one candidate."""
    selected_mode = _mode(mode)
    query = tuple(query_cues)
    candidate = tuple(candidate_terms)
    word_tag = coverage(
        query,
        candidate,
        query_kinds=("word", "tag"),
    )
    trigram = coverage(
        query,
        candidate,
        query_kinds=("char_trigram",),
    )
    lexical = max(word_tag, trigram)
    mechanism = coverage(
        query,
        candidate,
        query_kinds=("word", "mechanism"),
        candidate_kinds=("mechanism",),
    )
    if selected_mode is RetrievalMode.FOCUSED:
        ranking = lexical
    else:
        ranking = max(lexical, 0.80 * mechanism + 0.20 * lexical)
    return RelevanceComponents(
        word_tag_coverage=word_tag,
        trigram_coverage=trigram,
        lexical_or_trigram_relevance=lexical,
        mechanism_relevance=mechanism,
        ranking_relevance=min(1.0, max(0.0, ranking)),
    )


def passes_relevance_threshold(
    *,
    mode: RetrievalMode,
    lexical_or_trigram_relevance: float,
    mechanism_relevance: float,
) -> bool:
    """Apply the locked focused or associative candidate threshold."""
    selected_mode = _mode(mode)
    lexical = _unit_float(
        "lexical_or_trigram_relevance",
        lexical_or_trigram_relevance,
    )
    mechanism = _unit_float("mechanism_relevance", mechanism_relevance)
    if selected_mode is RetrievalMode.FOCUSED:
        return lexical >= FOCUSED_RELEVANCE_THRESHOLD
    return (
        lexical >= ASSOCIATIVE_RELEVANCE_THRESHOLD
        or mechanism >= ASSOCIATIVE_RELEVANCE_THRESHOLD
    )


def temperature_pool_quota(
    mode: RetrievalMode,
    temperature: Temperature,
    requested_limit: int,
) -> int:
    """Return one independent mode-and-temperature candidate quota."""
    selected_mode = _mode(mode)
    selected_temperature = _temperature(temperature)
    limit = _requested_limit(requested_limit)
    if selected_temperature is Temperature.ARCHIVED:
        return 0
    return max(
        10,
        _POOL_MULTIPLIERS[selected_mode][selected_temperature] * limit,
    )


def select_temperature_pools(
    candidates: Iterable[CandidateMatch],
    *,
    mode: RetrievalMode,
    requested_limit: int,
) -> tuple[CandidateMatch, ...]:
    """Take independent quotas while preserving global overlap/UUID order."""
    selected_mode = _mode(mode)
    limit = _requested_limit(requested_limit)
    values = tuple(candidates)
    if any(not isinstance(value, CandidateMatch) for value in values):
        raise ValueError("candidates must contain only CandidateMatch values")
    experience_ids = [value.experience_id for value in values]
    if len(experience_ids) != len(set(experience_ids)):
        raise ValueError("candidates must not repeat an experience_id")
    ordered = sorted(
        (
            value
            for value in values
            if value.raw_overlap > 0.0
            and value.temperature is not Temperature.ARCHIVED
        ),
        key=lambda value: (-value.raw_overlap, value.experience_id.bytes),
    )
    counts = {
        Temperature.HOT: 0,
        Temperature.WARM: 0,
        Temperature.COLD: 0,
    }
    selected: list[CandidateMatch] = []
    for value in ordered:
        quota = temperature_pool_quota(
            selected_mode,
            value.temperature,
            limit,
        )
        if counts[value.temperature] >= quota:
            continue
        counts[value.temperature] += 1
        selected.append(value)
    return tuple(selected)


def final_score(
    *,
    ranking_relevance: float,
    activation: float,
    confidence: float,
    importance: float,
    source_trust: float,
) -> float:
    """Combine the five locked deterministic ranking components."""
    relevance = _unit_float("ranking_relevance", ranking_relevance)
    current_activation = _unit_float("activation", activation)
    current_confidence = _unit_float("confidence", confidence)
    current_importance = _unit_float("importance", importance)
    current_source_trust = _unit_float("source_trust", source_trust)
    score = (
        0.50 * relevance
        + 0.20 * current_activation
        + 0.15 * current_confidence
        + 0.10 * current_importance
        + 0.05 * current_source_trust
    )
    return min(1.0, max(0.0, score))


def rank_candidate(
    candidate: RankingCandidate,
    *,
    query_cues: Iterable[TermCue],
    mode: RetrievalMode,
    at: datetime,
    lifecycle_config: LifecycleConfig,
) -> RankedCandidate:
    """Rank one candidate using activation recomputed at the query clock."""
    if not isinstance(candidate, RankingCandidate):
        raise ValueError("candidate must be a RankingCandidate")
    selected_mode = _mode(mode)
    query = _query_cue_tuple(query_cues)
    if not isinstance(lifecycle_config, LifecycleConfig):
        raise ValueError("lifecycle_config must be LifecycleConfig")
    query_at = _timestamp("at", at)
    relevance = relevance_components(
        query,
        candidate.terms,
        selected_mode,
    )
    activation = activation_at(
        candidate.activation_inputs,
        query_at,
        lifecycle_config,
    ).score
    confidence = candidate.activation_inputs.confidence
    importance = candidate.activation_inputs.importance
    score = final_score(
        ranking_relevance=relevance.ranking_relevance,
        activation=activation,
        confidence=confidence,
        importance=importance,
        source_trust=candidate.source_trust,
    )
    return RankedCandidate(
        experience_id=candidate.experience_id,
        temperature=candidate.temperature,
        current_version_created_at=candidate.current_version_created_at,
        score=score,
        ranking_relevance=relevance.ranking_relevance,
        lexical_or_trigram_relevance=(
            relevance.lexical_or_trigram_relevance
        ),
        mechanism_relevance=relevance.mechanism_relevance,
        word_tag_coverage=relevance.word_tag_coverage,
        trigram_coverage=relevance.trigram_coverage,
        activation=activation,
        confidence=confidence,
        importance=importance,
        source_trust=candidate.source_trust,
    )


def sort_ranked_candidates(
    candidates: Iterable[RankedCandidate],
) -> tuple[RankedCandidate, ...]:
    """Sort by score, relevance, version time, then UUID bytes."""
    values = tuple(candidates)
    if any(not isinstance(value, RankedCandidate) for value in values):
        raise ValueError(
            "candidates must contain only RankedCandidate values"
        )
    experience_ids = [value.experience_id for value in values]
    if len(experience_ids) != len(set(experience_ids)):
        raise ValueError("candidates must not repeat an experience_id")

    ordered = sorted(values, key=lambda value: value.experience_id.bytes)
    ordered.sort(
        key=lambda value: value.current_version_created_at,
        reverse=True,
    )
    ordered.sort(key=lambda value: value.ranking_relevance, reverse=True)
    ordered.sort(key=lambda value: value.score, reverse=True)
    return tuple(ordered)


def rank_candidates(
    candidates: Iterable[RankingCandidate],
    *,
    query_cues: Iterable[TermCue],
    mode: RetrievalMode,
    at: datetime,
    lifecycle_config: LifecycleConfig,
) -> tuple[RankedCandidate, ...]:
    """Rank non-archived candidates and discard those below mode threshold."""
    selected_mode = _mode(mode)
    query = _query_cue_tuple(query_cues)
    query_at = _timestamp("at", at)
    if not isinstance(lifecycle_config, LifecycleConfig):
        raise ValueError("lifecycle_config must be LifecycleConfig")
    values = tuple(candidates)
    if any(not isinstance(value, RankingCandidate) for value in values):
        raise ValueError(
            "candidates must contain only RankingCandidate values"
        )
    experience_ids = [value.experience_id for value in values]
    if len(experience_ids) != len(set(experience_ids)):
        raise ValueError("candidates must not repeat an experience_id")
    ranked = []
    for candidate in values:
        if candidate.temperature is Temperature.ARCHIVED:
            continue
        value = rank_candidate(
            candidate,
            query_cues=query,
            mode=selected_mode,
            at=query_at,
            lifecycle_config=lifecycle_config,
        )
        if passes_relevance_threshold(
            mode=selected_mode,
            lexical_or_trigram_relevance=(
                value.lexical_or_trigram_relevance
            ),
            mechanism_relevance=value.mechanism_relevance,
        ):
            ranked.append(value)
    return sort_ranked_candidates(ranked)


__all__ = [
    "ASSOCIATIVE_RELEVANCE_THRESHOLD",
    "FOCUSED_RELEVANCE_THRESHOLD",
    "MAX_RETRIEVAL_LIMIT",
    "CandidateMatch",
    "RankedCandidate",
    "RankingCandidate",
    "RelevanceComponents",
    "RetrievalMode",
    "coverage",
    "cue_kinds_compatible",
    "final_score",
    "passes_relevance_threshold",
    "rank_candidate",
    "rank_candidates",
    "raw_overlap",
    "relevance_components",
    "select_temperature_pools",
    "sort_ranked_candidates",
    "temperature_pool_quota",
]
