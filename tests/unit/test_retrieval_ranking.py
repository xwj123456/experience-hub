from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import permutations
from typing import Any, cast
from uuid import UUID

import pytest

from experience_hub.experiences import Temperature
from experience_hub.lifecycle import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.retrieval.ranking import (
    CandidateMatch,
    RankedCandidate,
    RankingCandidate,
    RetrievalMode,
    coverage,
    cue_kinds_compatible,
    final_score,
    passes_relevance_threshold,
    rank_candidate,
    rank_candidates,
    raw_overlap,
    relevance_components,
    select_temperature_pools,
    sort_ranked_candidates,
    temperature_pool_quota,
)
from experience_hub.retrieval.tokenizer import TermCue, TermKind

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)
CONFIG = LifecycleConfig()


def cue(
    term: str,
    term_kind: TermKind,
    weight: float = 1.0,
) -> TermCue:
    return TermCue(term=term, term_kind=term_kind, weight=weight)


@pytest.mark.parametrize(
    ("query_kind", "candidate_kind", "expected"),
    [
        ("word", "word", True),
        ("word", "tag", True),
        ("word", "mechanism", True),
        ("word", "char_trigram", False),
        ("tag", "word", False),
        ("tag", "tag", True),
        ("tag", "mechanism", False),
        ("tag", "char_trigram", False),
        ("mechanism", "word", False),
        ("mechanism", "tag", False),
        ("mechanism", "mechanism", True),
        ("mechanism", "char_trigram", False),
        ("char_trigram", "word", False),
        ("char_trigram", "tag", False),
        ("char_trigram", "mechanism", False),
        ("char_trigram", "char_trigram", True),
    ],
)
def test_cue_kind_compatibility_matrix(
    query_kind: TermKind,
    candidate_kind: TermKind,
    expected: bool,
) -> None:
    assert cue_kinds_compatible(query_kind, candidate_kind) is expected


def test_raw_overlap_uses_exact_terms_and_max_compatible_weight_per_query_cue(
) -> None:
    query = (
        cue("cache", "word", 1.0),
        cue("cache", "word", 0.4),
        cue("ops", "tag", 1.5),
        cue("lease", "mechanism", 1.25),
        cue("cac", "char_trigram", 0.35),
        cue("exact-only", "word", 1.0),
    )
    candidate = (
        cue("cache", "word", 0.6),
        cue("cache", "tag", 1.5),
        cue("cache", "mechanism", 1.25),
        cue("ops", "tag", 1.0),
        cue("lease", "mechanism", 0.75),
        cue("cac", "char_trigram", 0.35),
        cue("exact only", "word", 1.0),
    )

    assert raw_overlap(query, candidate) == pytest.approx(
        1.0 + 1.0 + 0.75 + 0.35,
        abs=1e-12,
    )


def test_overlap_and_coverage_are_bit_stable_across_cue_permutations() -> None:
    weights = (
        float.fromhex("0x1.20adcb5194bcap-144"),
        float.fromhex("0x1.394ef17178fcdp-972"),
        float.fromhex("0x1.68457f4f2e722p-142"),
    )
    query = (
        cue("alpha", "word", weights[0]),
        cue("beta", "word", weights[1]),
        cue("gamma", "word", weights[2]),
    )
    coverage_candidate = (query[0],)
    expected_overlap = math.fsum(weights)
    expected_coverage = weights[0] / expected_overlap

    overlap_values = {
        raw_overlap(query_order, candidate_order).hex()
        for query_order in permutations(query)
        for candidate_order in permutations(query)
    }
    coverage_values = {
        coverage(
            query_order,
            coverage_candidate,
            query_kinds=("word",),
        ).hex()
        for query_order in permutations(query)
    }

    assert overlap_values == {expected_overlap.hex()}
    assert coverage_values == {expected_coverage.hex()}


def test_coverage_returns_zero_for_an_empty_query_family() -> None:
    assert (
        coverage(
            (),
            (cue("memory", "word"),),
            query_kinds=("word", "tag"),
        )
        == 0.0
    )


def test_coverage_combines_word_and_tag_denominator() -> None:
    query = (
        cue("memory", "word", 1.0),
        cue("ops", "tag", 1.5),
        cue("lease", "mechanism", 1.25),
    )
    candidate = (
        cue("memory", "word", 0.5),
        cue("ops", "tag", 1.0),
        cue("lease", "mechanism", 1.25),
    )

    assert coverage(
        query,
        candidate,
        query_kinds=("word", "tag"),
    ) == pytest.approx(1.5 / 2.5, abs=1e-12)


def test_mechanism_coverage_uses_word_and_mechanism_query_family(
) -> None:
    query = (
        cue("handoff", "word", 1.0),
        cue("lease", "mechanism", 1.25),
    )
    candidate = (
        cue("handoff", "word", 1.0),
        cue("handoff", "mechanism", 0.5),
        cue("lease", "mechanism", 0.75),
    )

    value = coverage(
        query,
        candidate,
        query_kinds=("word", "mechanism"),
        candidate_kinds=("mechanism",),
    )

    assert value == pytest.approx(5.0 / 9.0, abs=1e-12)


def test_focused_and_associative_relevance_use_locked_formulas() -> None:
    query = (
        cue("direct", "word", 1.0),
        cue("ops", "tag", 1.5),
        cue("lease", "mechanism", 1.25),
    )
    candidate = (
        cue("direct", "mechanism", 1.0),
        cue("lease", "mechanism", 1.25),
    )

    focused = relevance_components(query, candidate, RetrievalMode.FOCUSED)
    associative = relevance_components(
        query,
        candidate,
        RetrievalMode.ASSOCIATIVE,
    )

    assert focused.word_tag_coverage == pytest.approx(0.4, abs=1e-12)
    assert focused.trigram_coverage == 0.0
    assert focused.lexical_or_trigram_relevance == pytest.approx(
        0.4,
        abs=1e-12,
    )
    assert focused.mechanism_relevance == pytest.approx(1.0, abs=1e-12)
    assert focused.ranking_relevance == pytest.approx(0.4, abs=1e-12)
    assert associative.ranking_relevance == pytest.approx(0.88, abs=1e-12)


@pytest.mark.parametrize(
    (
        "mode",
        "lexical_or_trigram",
        "mechanism",
        "expected",
    ),
    [
        (RetrievalMode.FOCUSED, 0.05, 0.0, True),
        (
            RetrievalMode.FOCUSED,
            math.nextafter(0.05, 0.0),
            1.0,
            False,
        ),
        (RetrievalMode.ASSOCIATIVE, 0.02, 0.0, True),
        (RetrievalMode.ASSOCIATIVE, 0.0, 0.02, True),
        (
            RetrievalMode.ASSOCIATIVE,
            math.nextafter(0.02, 0.0),
            math.nextafter(0.02, 0.0),
            False,
        ),
    ],
)
def test_mode_thresholds_retain_exact_equality(
    mode: RetrievalMode,
    lexical_or_trigram: float,
    mechanism: float,
    expected: bool,
) -> None:
    assert (
        passes_relevance_threshold(
            mode=mode,
            lexical_or_trigram_relevance=lexical_or_trigram,
            mechanism_relevance=mechanism,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("mode", "temperature", "limit", "expected"),
    [
        (RetrievalMode.FOCUSED, Temperature.HOT, 1, 10),
        (RetrievalMode.FOCUSED, Temperature.WARM, 6, 24),
        (RetrievalMode.FOCUSED, Temperature.COLD, 6, 12),
        (RetrievalMode.ASSOCIATIVE, Temperature.HOT, 6, 18),
        (RetrievalMode.ASSOCIATIVE, Temperature.WARM, 6, 18),
        (RetrievalMode.ASSOCIATIVE, Temperature.COLD, 6, 30),
        (RetrievalMode.ASSOCIATIVE, Temperature.ARCHIVED, 6, 0),
    ],
)
def test_temperature_pool_quotas_are_locked(
    mode: RetrievalMode,
    temperature: Temperature,
    limit: int,
    expected: int,
) -> None:
    assert temperature_pool_quota(mode, temperature, limit) == expected


def _pool_candidates(
    *,
    per_temperature: int,
) -> tuple[CandidateMatch, ...]:
    values: list[CandidateMatch] = []
    serial = 1
    for temperature in (
        Temperature.HOT,
        Temperature.WARM,
        Temperature.COLD,
    ):
        for index in range(per_temperature):
            values.append(
                CandidateMatch(
                    experience_id=UUID(
                        f"00000000-0000-0000-0000-{serial:012d}"
                    ),
                    temperature=temperature,
                    raw_overlap=float(per_temperature - index),
                )
            )
            serial += 1
    return tuple(values)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            RetrievalMode.FOCUSED,
            {
                Temperature.HOT: 24,
                Temperature.WARM: 24,
                Temperature.COLD: 12,
            },
        ),
        (
            RetrievalMode.ASSOCIATIVE,
            {
                Temperature.HOT: 18,
                Temperature.WARM: 18,
                Temperature.COLD: 30,
            },
        ),
    ],
)
def test_temperature_pools_apply_independent_quotas_and_preserve_global_order(
    mode: RetrievalMode,
    expected: dict[Temperature, int],
) -> None:
    selected = select_temperature_pools(
        _pool_candidates(per_temperature=30),
        mode=mode,
        requested_limit=6,
    )

    assert Counter(item.temperature for item in selected) == expected
    assert selected == tuple(
        sorted(
            selected,
            key=lambda item: (-item.raw_overlap, item.experience_id.bytes),
        )
    )


def test_empty_temperature_pool_does_not_lend_quota() -> None:
    warm_only = tuple(
        CandidateMatch(
            experience_id=UUID(
                f"10000000-0000-0000-0000-{index:012d}"
            ),
            temperature=Temperature.WARM,
            raw_overlap=float(100 - index),
        )
        for index in range(30)
    )

    selected = select_temperature_pools(
        warm_only,
        mode=RetrievalMode.FOCUSED,
        requested_limit=6,
    )

    assert len(selected) == 24
    assert {item.temperature for item in selected} == {Temperature.WARM}


def test_nonpositive_overlap_and_archived_candidates_are_excluded() -> None:
    candidates = (
        CandidateMatch(
            experience_id=UUID("20000000-0000-0000-0000-000000000001"),
            temperature=Temperature.HOT,
            raw_overlap=0.0,
        ),
        CandidateMatch(
            experience_id=UUID("20000000-0000-0000-0000-000000000002"),
            temperature=Temperature.ARCHIVED,
            raw_overlap=1.0,
        ),
    )

    assert (
        select_temperature_pools(
            candidates,
            mode=RetrievalMode.FOCUSED,
            requested_limit=1,
        )
        == ()
    )


def test_pool_quota_cutoff_prefers_uuid_bytes_for_equal_overlap() -> None:
    candidates = tuple(
        CandidateMatch(
            experience_id=UUID(
                f"21000000-0000-0000-0000-{serial:012d}"
            ),
            temperature=Temperature.WARM,
            raw_overlap=1.0,
        )
        for serial in range(11, 0, -1)
    )

    selected = select_temperature_pools(
        candidates,
        mode=RetrievalMode.FOCUSED,
        requested_limit=1,
    )

    assert tuple(item.experience_id for item in selected) == tuple(
        sorted(
            (item.experience_id for item in candidates),
            key=lambda value: value.bytes,
        )[:10]
    )


def test_final_score_uses_all_five_locked_components() -> None:
    assert final_score(
        ranking_relevance=0.8,
        activation=0.5,
        confidence=0.4,
        importance=0.5,
        source_trust=0.8,
    ) == pytest.approx(0.65, abs=1e-12)


def _activation_inputs(
    *,
    importance: float = 0.4,
    confidence: float = 0.6,
) -> ActivationInputs:
    return ActivationInputs(
        importance=importance,
        confidence=confidence,
        access_count=0,
        access_strength=0.0,
        strength_updated_at=NOW,
        last_accessed_at=None,
        created_at=NOW,
    )


def _ranking_candidate(
    *,
    serial: int = 1,
    terms: tuple[TermCue, ...] = (TermCue("memory", "word", 1.0),),
    current_version_created_at: datetime = NOW,
    source_trust: float = 0.8,
) -> RankingCandidate:
    return RankingCandidate(
        experience_id=UUID(
            f"30000000-0000-0000-0000-{serial:012d}"
        ),
        temperature=Temperature.WARM,
        current_version_created_at=current_version_created_at,
        terms=terms,
        activation_inputs=_activation_inputs(),
        source_trust=source_trust,
    )


def test_rank_candidate_has_no_cached_activation_and_uses_query_clock(
) -> None:
    candidate = _ranking_candidate()
    query_at = NOW + timedelta(
        hours=2 * CONFIG.recency_half_life_hours
    )
    query = (cue("memory", "word"),)

    ranked = rank_candidate(
        candidate,
        query_cues=query,
        mode=RetrievalMode.FOCUSED,
        at=query_at,
        lifecycle_config=CONFIG,
    )
    expected_activation = activation_at(
        candidate.activation_inputs,
        query_at,
        CONFIG,
    ).score

    assert expected_activation == pytest.approx(0.285, abs=1e-12)
    assert ranked.activation == pytest.approx(
        expected_activation,
        abs=1e-12,
    )
    assert not hasattr(candidate, "cached_activation_score")
    assert ranked.ranking_relevance == pytest.approx(1.0, abs=1e-12)
    assert ranked.confidence == pytest.approx(0.6, abs=1e-12)
    assert ranked.importance == pytest.approx(0.4, abs=1e-12)
    assert ranked.source_trust == pytest.approx(0.8, abs=1e-12)
    assert ranked.score == pytest.approx(0.727, abs=1e-12)


def test_rank_candidate_rejects_an_empty_overall_query() -> None:
    with pytest.raises(ValueError, match="query_cues must not be empty"):
        rank_candidate(
            _ranking_candidate(),
            query_cues=(),
            mode=RetrievalMode.FOCUSED,
            at=NOW,
            lifecycle_config=CONFIG,
        )


def test_rank_candidates_rejects_an_empty_query_with_nonempty_candidates(
) -> None:
    with pytest.raises(ValueError, match="query_cues must not be empty"):
        rank_candidates(
            (_ranking_candidate(),),
            query_cues=(),
            mode=RetrievalMode.FOCUSED,
            at=NOW,
            lifecycle_config=CONFIG,
        )


def test_zero_candidates_still_validate_query_elements() -> None:
    with pytest.raises(
        ValueError,
        match="query_cues must contain only TermCue values",
    ):
        rank_candidates(
            (),
            query_cues=(cast(Any, "not-a-cue"),),
            mode=RetrievalMode.FOCUSED,
            at=NOW,
            lifecycle_config=CONFIG,
        )


def test_zero_candidates_still_validate_query_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rank_candidates(
            (),
            query_cues=(cue("memory", "word"),),
            mode=RetrievalMode.FOCUSED,
            at=NOW.replace(tzinfo=None),
            lifecycle_config=CONFIG,
        )


def test_zero_candidates_still_validate_lifecycle_config() -> None:
    with pytest.raises(ValueError, match="LifecycleConfig"):
        rank_candidates(
            (),
            query_cues=(cue("memory", "word"),),
            mode=RetrievalMode.FOCUSED,
            at=NOW,
            lifecycle_config=cast(Any, object()),
        )


def test_rank_candidates_discards_below_mode_threshold() -> None:
    candidates = (
        _ranking_candidate(
            serial=1,
            terms=(cue("memory", "word", 0.05),),
        ),
        _ranking_candidate(
            serial=2,
            terms=(
                cue(
                    "memory",
                    "word",
                    math.nextafter(0.05, 0.0),
                ),
            ),
        ),
    )

    ranked = rank_candidates(
        candidates,
        query_cues=(cue("memory", "word"),),
        mode=RetrievalMode.FOCUSED,
        at=NOW,
        lifecycle_config=CONFIG,
    )

    assert tuple(item.experience_id for item in ranked) == (
        candidates[0].experience_id,
    )


def _ranked_candidate(
    *,
    serial: int,
    score: float = 0.5,
    ranking_relevance: float = 0.5,
    created_at: datetime = NOW,
) -> RankedCandidate:
    return RankedCandidate(
        experience_id=UUID(
            f"40000000-0000-0000-0000-{serial:012d}"
        ),
        temperature=Temperature.WARM,
        current_version_created_at=created_at,
        score=score,
        ranking_relevance=ranking_relevance,
        lexical_or_trigram_relevance=ranking_relevance,
        mechanism_relevance=0.0,
        word_tag_coverage=ranking_relevance,
        trigram_coverage=0.0,
        activation=0.5,
        confidence=0.5,
        importance=0.5,
        source_trust=0.5,
    )


def test_final_sort_prefers_score_before_other_fields() -> None:
    lower = _ranked_candidate(
        serial=1,
        score=0.7,
        ranking_relevance=1.0,
        created_at=NOW + timedelta(days=1),
    )
    higher = _ranked_candidate(
        serial=2,
        score=0.8,
        ranking_relevance=0.0,
        created_at=NOW,
    )

    assert sort_ranked_candidates((lower, higher)) == (higher, lower)


def test_final_sort_uses_ranking_relevance_as_second_tie_break() -> None:
    lower = _ranked_candidate(
        serial=1,
        ranking_relevance=0.6,
        created_at=NOW + timedelta(days=1),
    )
    higher = _ranked_candidate(
        serial=2,
        ranking_relevance=0.7,
        created_at=NOW,
    )

    assert sort_ranked_candidates((lower, higher)) == (higher, lower)


def test_final_sort_uses_newer_current_version_as_third_tie_break() -> None:
    older = _ranked_candidate(serial=1, created_at=NOW)
    newer = _ranked_candidate(
        serial=2,
        created_at=NOW + timedelta(microseconds=1),
    )

    assert sort_ranked_candidates((older, newer)) == (newer, older)


def test_final_sort_uses_experience_uuid_bytes_as_last_tie_break() -> None:
    lower_uuid = _ranked_candidate(serial=1)
    higher_uuid = _ranked_candidate(serial=2)

    assert sort_ranked_candidates((higher_uuid, lower_uuid)) == (
        lower_uuid,
        higher_uuid,
    )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: CandidateMatch(
                experience_id=cast(Any, "not-a-uuid"),
                temperature=Temperature.HOT,
                raw_overlap=1.0,
            ),
            "experience_id",
        ),
        (
            lambda: CandidateMatch(
                experience_id=UUID(
                    "50000000-0000-0000-0000-000000000001"
                ),
                temperature=cast(Any, "warm"),
                raw_overlap=1.0,
            ),
            "temperature",
        ),
        (
            lambda: RankingCandidate(
                experience_id=UUID(
                    "50000000-0000-0000-0000-000000000002"
                ),
                temperature=Temperature.WARM,
                current_version_created_at=NOW.replace(tzinfo=None),
                terms=(),
                activation_inputs=_activation_inputs(),
                source_trust=0.5,
            ),
            "timezone-aware",
        ),
        (
            lambda: _ranking_candidate(source_trust=math.nan),
            "source_trust",
        ),
        (
            lambda: _ranked_candidate(serial=1, score=math.inf),
            "score",
        ),
    ],
)
def test_ranking_values_reject_invalid_runtime_state(
    factory: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize("limit", [0, -1, True, 51])
def test_pool_quota_rejects_invalid_requested_limit(limit: Any) -> None:
    with pytest.raises(ValueError, match="requested_limit"):
        temperature_pool_quota(
            RetrievalMode.FOCUSED,
            Temperature.WARM,
            limit,
        )
