from __future__ import annotations

from decimal import Decimal

import pytest

from experience_hub.benchmark.runner import (
    EffectivenessMetrics,
    evaluate_effectiveness_gates,
)


def _passing_metrics(**updates: object) -> EffectivenessMetrics:
    values: dict[str, object] = {
        "focused_macro_recall_at_five": Decimal("0.90"),
        "cold_macro_recall_at_five": Decimal("0.85"),
        "cold_baseline_macro_recall_at_five": Decimal("0.60"),
        "distractor_false_reactivations": 0,
        "pending_leakage_count": 0,
        "adopted_provenance_complete": 1,
        "adopted_provenance_total": 1,
        "valid_idea_count": 12,
        "idea_schema_evidence_valid_count": 12,
        "distinct_mechanism_count": 9,
        "same_snapshot_incubation_promotions": 0,
        "byte_identical_replay": True,
    }
    values.update(updates)
    return EffectivenessMetrics(**values)  # type: ignore[arg-type]


def test_effectiveness_gate_boundaries_are_inclusive_and_named() -> None:
    gates = evaluate_effectiveness_gates(_passing_metrics())

    assert all(gate.passed for gate in gates)
    assert tuple(gate.name for gate in gates) == (
        "focused_macro_recall_at_5",
        "cold_macro_recall_at_5",
        "cold_recall_gain_over_hot_warm_baseline",
        "distractor_false_reactivations",
        "pending_capsule_leakage",
        "adopted_provenance_completeness",
        "valid_idea_count",
        "idea_schema_and_evidence_validity",
        "unique_mechanism_ratio",
        "same_snapshot_incubation_promotion",
        "byte_identical_replay",
    )


@pytest.mark.parametrize(
    ("updates", "failed_name"),
    (
        (
            {"focused_macro_recall_at_five": Decimal("0.8999")},
            "focused_macro_recall_at_5",
        ),
        (
            {"cold_macro_recall_at_five": Decimal("0.8499")},
            "cold_macro_recall_at_5",
        ),
        (
            {"cold_baseline_macro_recall_at_five": Decimal("0.6001")},
            "cold_recall_gain_over_hot_warm_baseline",
        ),
        ({"distractor_false_reactivations": 1}, "distractor_false_reactivations"),
        ({"pending_leakage_count": 1}, "pending_capsule_leakage"),
        ({"adopted_provenance_complete": 0}, "adopted_provenance_completeness"),
        (
            {
                "valid_idea_count": 11,
                "idea_schema_evidence_valid_count": 11,
                "distinct_mechanism_count": 8,
            },
            "valid_idea_count",
        ),
        (
            {"idea_schema_evidence_valid_count": 11},
            "idea_schema_and_evidence_validity",
        ),
        ({"distinct_mechanism_count": 8}, "unique_mechanism_ratio"),
        (
            {"same_snapshot_incubation_promotions": 1},
            "same_snapshot_incubation_promotion",
        ),
        ({"byte_identical_replay": False}, "byte_identical_replay"),
    ),
)
def test_each_failed_gate_is_reported_independently(
    updates: dict[str, object],
    failed_name: str,
) -> None:
    failed = tuple(
        gate.name
        for gate in evaluate_effectiveness_gates(_passing_metrics(**updates))
        if not gate.passed
    )
    assert failed_name in failed


def test_metric_domains_reject_impossible_counts() -> None:
    with pytest.raises(ValueError):
        _passing_metrics(
            adopted_provenance_complete=2,
            adopted_provenance_total=1,
        )
    with pytest.raises(ValueError):
        _passing_metrics(
            idea_schema_evidence_valid_count=13,
            valid_idea_count=12,
        )
    with pytest.raises(ValueError):
        _passing_metrics(
            distinct_mechanism_count=13,
            valid_idea_count=12,
        )


def test_missing_curated_evidence_fails_only_the_evidence_validity_gate() -> None:
    gates = {
        gate.name: gate
        for gate in evaluate_effectiveness_gates(
            _passing_metrics(inspiration_evidence_coverage_failures=1)
        )
    }

    assert gates["idea_schema_and_evidence_validity"].passed is False
    assert gates["idea_schema_and_evidence_validity"].actual == "0.000000000000"
    assert gates["valid_idea_count"].passed is True
    assert gates["unique_mechanism_ratio"].passed is True
