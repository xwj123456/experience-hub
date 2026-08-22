"""Exact, integer-only expansion gates for the ExperienceBench-S pilot."""

from __future__ import annotations

from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    BENCHMARK_STRATUM_ORDER,
    BenchmarkGateResultV1,
    BenchmarkPassPayloadV1,
)


def _gate(gate_id: str, passed: bool) -> BenchmarkGateResultV1:
    return BenchmarkGateResultV1(schema_version=1, gate_id=gate_id, passed=passed)


def _has_complete_pilot(payload: BenchmarkPassPayloadV1) -> bool:
    aggregate = payload.aggregate
    return (
        payload.comparison_complete
        and aggregate is not None
        and len(payload.cases) == 30
        and all(case.status == "complete" for case in payload.cases)
        and all(
            tuple(arm.arm_id for arm in case.arms)
            == tuple(arm_kind.value for arm_kind in BENCHMARK_ARM_ORDER)
            for case in payload.cases
        )
        and all(
            arm.status == "complete"
            for case in payload.cases
            for arm in case.arms
        )
        and all(
            sum(case.stratum is stratum for case in payload.cases) == 6
            for stratum in BENCHMARK_STRATUM_ORDER
        )
        and aggregate.overall.case_count == 30
        and len(aggregate.strata) == len(BENCHMARK_STRATUM_ORDER)
        and tuple(item.scope for item in aggregate.strata)
        == tuple(item.value for item in BENCHMARK_STRATUM_ORDER)
        and all(item.case_count == 6 for item in aggregate.strata)
    )


def evaluate_pilot_gates(
    first_pass: BenchmarkPassPayloadV1,
    second_pass: BenchmarkPassPayloadV1,
) -> tuple[BenchmarkGateResultV1, ...]:
    """Evaluate safety and effectiveness only from two complete pilot payloads."""
    first_complete = _has_complete_pilot(first_pass)
    second_complete = _has_complete_pilot(second_pass)
    complete = first_complete and second_complete
    gates = [
        _gate("comparison_complete", complete),
        _gate("complete_arms", complete),
        _gate("deterministic_replay", first_pass == second_pass),
        _gate("safety", first_pass.safety.is_safe and second_pass.safety.is_safe),
    ]
    if not complete:
        return tuple(gates)
    aggregate = first_pass.aggregate
    if aggregate is None:
        return tuple(gates)
    overall_passed = (
        aggregate.overall.sum_delta_micros
        >= 50_000 * aggregate.overall.case_count
    )
    stratum_passed = all(
        item.sum_delta_micros >= -20_000 * item.case_count
        for item in aggregate.strata
    )
    return (
        *gates,
        _gate("overall_effectiveness", overall_passed),
        _gate("stratum_effectiveness", stratum_passed),
    )


__all__ = ["evaluate_pilot_gates"]
