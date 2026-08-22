"""Paired all-integer ExperienceBench-S comparison metrics."""

from __future__ import annotations

from collections import Counter

from pydantic import ValidationError

from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    BENCHMARK_STRATUM_ORDER,
    BenchmarkAggregateV1,
    BenchmarkArmEvidenceV1,
    BenchmarkArmKind,
    BenchmarkCaseEvidenceV1,
    BenchmarkCaseV1,
    BenchmarkDeltaAggregateV1,
)
from experience_hub.experiments.errors import ExperimentInputError


def _invalid_metric_input() -> ExperimentInputError:
    return ExperimentInputError(
        "benchmark_metric_invalid",
        "Benchmark comparisons cannot be derived from the supplied evidence",
    )


def _validate_case(case: BenchmarkCaseV1) -> BenchmarkCaseV1:
    if not isinstance(case, BenchmarkCaseV1):
        raise _invalid_metric_input()
    try:
        return BenchmarkCaseV1.model_validate(case, strict=True)
    except ValidationError:
        raise _invalid_metric_input() from None


def _validate_arms(
    arms: tuple[BenchmarkArmEvidenceV1, ...],
) -> tuple[BenchmarkArmEvidenceV1, ...]:
    if not isinstance(arms, tuple):
        raise _invalid_metric_input()
    try:
        validated = tuple(
            BenchmarkArmEvidenceV1.model_validate(arm, strict=True) for arm in arms
        )
    except ValidationError:
        raise _invalid_metric_input() from None
    if tuple(arm.arm_id for arm in validated) != tuple(
        item.value for item in BENCHMARK_ARM_ORDER
    ):
        raise _invalid_metric_input()
    return validated


def derive_case_comparison(
    case: BenchmarkCaseV1,
    arms: tuple[BenchmarkArmEvidenceV1, ...],
) -> BenchmarkCaseEvidenceV1:
    """Pair Experience Hub with the strongest fixed-order baseline for one case."""
    case = _validate_case(case)
    arms = _validate_arms(arms)
    if any(arm.status != "complete" for arm in arms):
        return BenchmarkCaseEvidenceV1(
            schema_version=1,
            case=case,
            case_id=case.case_id,
            source_class=case.source_class,
            stratum=case.stratum,
            status="incomplete",
            arms=arms,
            comparator_arm_id=None,
            comparator_utility_micros=None,
            experience_hub_utility_micros=None,
            delta_utility_micros=None,
        )
    utilities = tuple(
        arm.oracle.scores.utility_micros if arm.oracle is not None else None
        for arm in arms
    )
    if any(value is None for value in utilities):
        raise _invalid_metric_input()
    baseline_utilities = tuple(
        value for value in utilities[:3] if value is not None
    )
    if len(baseline_utilities) != 3:
        raise _invalid_metric_input()
    comparator_utility = max(baseline_utilities)
    comparator_index = baseline_utilities.index(comparator_utility)
    experience_hub_utility = utilities[3]
    if experience_hub_utility is None:
        raise _invalid_metric_input()
    comparator_kind = BENCHMARK_ARM_ORDER[comparator_index]
    if comparator_kind is BenchmarkArmKind.EXPERIENCE_HUB:
        raise _invalid_metric_input()
    return BenchmarkCaseEvidenceV1(
        schema_version=1,
        case=case,
        case_id=case.case_id,
        source_class=case.source_class,
        stratum=case.stratum,
        status="complete",
        arms=arms,
        comparator_arm_id=comparator_kind.value,
        comparator_utility_micros=comparator_utility,
        experience_hub_utility_micros=experience_hub_utility,
        delta_utility_micros=experience_hub_utility - comparator_utility,
    )


def aggregate_benchmark_cases(
    cases: tuple[BenchmarkCaseEvidenceV1, ...],
) -> BenchmarkAggregateV1 | None:
    """Aggregate a complete pilot only; incomplete cases intentionally have no mean."""
    if not isinstance(cases, tuple):
        raise _invalid_metric_input()
    try:
        validated = tuple(
            BenchmarkCaseEvidenceV1.model_validate(case, strict=True) for case in cases
        )
    except ValidationError:
        raise _invalid_metric_input() from None
    if any(case.status != "complete" for case in validated):
        return None
    if len(validated) != 30 or len({case.case_id for case in validated}) != 30:
        raise _invalid_metric_input()
    strata_counts = Counter(case.stratum for case in validated)
    if any(strata_counts[stratum] != 6 for stratum in BENCHMARK_STRATUM_ORDER):
        raise _invalid_metric_input()
    deltas = tuple(case.delta_utility_micros for case in validated)
    if any(delta is None for delta in deltas):
        raise _invalid_metric_input()
    signed_deltas = tuple(delta for delta in deltas if delta is not None)
    overall_sum = sum(signed_deltas)
    strata = tuple(
        BenchmarkDeltaAggregateV1(
            schema_version=1,
            scope=stratum.value,
            sum_delta_micros=sum(
                case.delta_utility_micros
                for case in validated
                if case.stratum is stratum and case.delta_utility_micros is not None
            ),
            case_count=6,
            mean_delta_micros=sum(
                case.delta_utility_micros
                for case in validated
                if case.stratum is stratum and case.delta_utility_micros is not None
            )
            // 6,
        )
        for stratum in BENCHMARK_STRATUM_ORDER
    )
    return BenchmarkAggregateV1(
        schema_version=1,
        overall=BenchmarkDeltaAggregateV1(
            schema_version=1,
            scope="overall",
            sum_delta_micros=overall_sum,
            case_count=30,
            mean_delta_micros=overall_sum // 30,
        ),
        strata=strata,
    )


__all__ = ["aggregate_benchmark_cases", "derive_case_comparison"]
