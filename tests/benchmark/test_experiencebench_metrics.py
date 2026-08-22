from __future__ import annotations

from collections.abc import Iterable

from tests.benchmark.experiencebench_factories import valid_case_document

from experience_hub.canonical import canonical_json_bytes
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    BENCHMARK_STRATUM_ORDER,
    BenchmarkArmDescriptorV1,
    BenchmarkArmEvidenceV1,
    BenchmarkArmObservationV1,
    BenchmarkCaseEvidenceV1,
    BenchmarkCaseV1,
    BenchmarkCategoryScoresV1,
    BenchmarkOracleEvidenceV1,
    BenchmarkPassPayloadV1,
    BenchmarkSafetyEvidenceV1,
    BenchmarkStratum,
    ResolvedBenchmarkManifestV1,
)
from experience_hub.experiments.benchmarks.gates import evaluate_pilot_gates
from experience_hub.experiments.benchmarks.metrics import (
    aggregate_benchmark_cases,
    derive_case_comparison,
)


def _case(*, case_id: str, stratum: BenchmarkStratum) -> BenchmarkCaseV1:
    document = valid_case_document()
    document["case_id"] = case_id
    document["stratum"] = stratum.value
    return BenchmarkCaseV1.model_validate_json(canonical_json_bytes(document))


def _arm(
    arm_id: str,
    utility_micros: int,
    *,
    complete: bool = True,
) -> BenchmarkArmEvidenceV1:
    if not complete:
        return BenchmarkArmEvidenceV1(
            schema_version=1,
            arm_id=arm_id,
            status="failed",
            observation=None,
            oracle=None,
            error_code="benchmark_policy_failed",
            error_stage="policy",
        )
    scores = BenchmarkCategoryScoresV1(
        schema_version=1,
        required_coverage_micros=utility_micros,
        avoidance_micros=0,
        recovery_order_micros=0,
        evidence_efficiency_micros=0,
        utility_micros=utility_micros,
    )
    return BenchmarkArmEvidenceV1(
        schema_version=1,
        arm_id=arm_id,
        status="complete",
        observation=BenchmarkArmObservationV1(
            schema_version=1, returned_labels=(), selected_content_bytes=0
        ),
        oracle=BenchmarkOracleEvidenceV1(
            schema_version=1,
            returned_labels=(),
            required_labels=(),
            optional_labels=(),
            violation_labels=(),
            satisfied_checkpoint_indexes=(),
            scores=scores,
        ),
        error_code=None,
        error_stage=None,
    )


def _comparison(
    case: BenchmarkCaseV1,
    *,
    baselines: tuple[int, int, int],
    experience_hub: int,
    complete: bool = True,
) -> BenchmarkCaseEvidenceV1:
    arms = tuple(
        _arm(kind.value, utility, complete=complete)
        for kind, utility in zip(
            BENCHMARK_ARM_ORDER, (*baselines, experience_hub), strict=True
        )
    )
    return derive_case_comparison(case, arms)


def test_case_comparison_uses_strongest_baseline_and_earliest_fixed_order_tie() -> None:
    case = _case(case_id="metric-case", stratum=BenchmarkStratum.FAILURE_RECOVERY)
    result = _comparison(
        case, baselines=(700_000, 800_000, 800_000), experience_hub=900_000
    )

    assert result.comparator_arm_id == "recent_notes"
    assert result.comparator_utility_micros == 800_000
    assert result.delta_utility_micros == 100_000


def test_case_comparison_keeps_negative_signed_delta() -> None:
    result = _comparison(
        _case(case_id="negative-case", stratum=BenchmarkStratum.STATE_CHANGE),
        baselines=(750_000, 800_000, 700_000),
        experience_hub=600_000,
    )

    assert result.delta_utility_micros == -200_000


def test_incomplete_case_has_no_comparison_and_prevents_aggregate() -> None:
    case = _case(case_id="incomplete-case", stratum=BenchmarkStratum.STATE_CHANGE)
    incomplete = _comparison(
        case, baselines=(1, 2, 3), experience_hub=4, complete=False
    )

    assert incomplete.status == "incomplete"
    assert incomplete.delta_utility_micros is None
    assert aggregate_benchmark_cases((incomplete,)) is None


def _thirty_cases(
    deltas: Iterable[int],
) -> tuple[BenchmarkCaseEvidenceV1, ...]:
    values = tuple(deltas)
    assert len(values) == 30
    cases: list[BenchmarkCaseEvidenceV1] = []
    for index, (stratum, delta) in enumerate(
        zip(
            (item for item in BENCHMARK_STRATUM_ORDER for _ in range(6)),
            values,
            strict=True,
        )
    ):
        cases.append(
            _comparison(
                _case(case_id=f"case-{index}", stratum=stratum),
                baselines=(500_000, 500_000, 500_000),
                experience_hub=500_000 + delta,
            )
        )
    return tuple(cases)


def test_aggregate_has_signed_sums_exact_counts_and_five_ordered_strata() -> None:
    aggregate = aggregate_benchmark_cases(_thirty_cases((10_000,) * 29 + (-5_000,)))

    assert aggregate is not None
    assert aggregate.overall.sum_delta_micros == 285_000
    assert aggregate.overall.case_count == 30
    assert tuple(item.scope for item in aggregate.strata) == tuple(
        item.value for item in BENCHMARK_STRATUM_ORDER
    )
    assert tuple(item.case_count for item in aggregate.strata) == (6, 6, 6, 6, 6)


def _manifest() -> ResolvedBenchmarkManifestV1:
    return ResolvedBenchmarkManifestV1(
        schema_version=1,
        pack_id="experiencebench-s-pilot",
        maturity="pilot-30",
        manifest_sha256="a" * 64,
        cases_sha256="b" * 64,
        source_fixture_sha256="c" * 64,
        snapshot_sha256="d" * 64,
        source_schema_revision=1,
        frozen_at="2026-08-20T00:00:00Z",
        seed=20260820,
        arms=tuple(
            BenchmarkArmDescriptorV1(
                schema_version=1,
                arm_id=item.value,
                kind=item,
                required=True,
            )
            for item in BENCHMARK_ARM_ORDER
        ),
        oracle_version=1,
        metric_version=1,
        gate_version=1,
        evidence_schema_version=1,
        summary_schema_version=1,
        profile_schema_version=1,
    )


def _payload(cases: tuple[BenchmarkCaseEvidenceV1, ...]) -> BenchmarkPassPayloadV1:
    aggregate = aggregate_benchmark_cases(cases)
    return BenchmarkPassPayloadV1(
        schema_version=1,
        resolved_manifest=_manifest(),
        cases=cases,
        comparison_complete=aggregate is not None,
        safety=BenchmarkSafetyEvidenceV1(
            schema_version=1,
            owner_leak_count=0,
            quarantine_leak_count=0,
            cross_arm_contamination_count=0,
            source_mutation_count=0,
            source_unchanged=True,
            clone_isolation_verified=True,
        ),
        aggregate=aggregate,
    )


def _gate_state(payload: BenchmarkPassPayloadV1) -> dict[str, bool]:
    return {
        item.gate_id: item.passed
        for item in evaluate_pilot_gates(payload, payload)
    }


def test_overall_one_micro_gate_boundary_uses_exact_integer_comparison() -> None:
    failed = _gate_state(_payload(_thirty_cases((50_000,) * 29 + (49_999,))))
    passed = _gate_state(_payload(_thirty_cases((50_000,) * 30)))

    assert failed["overall_effectiveness"] is False
    assert passed["overall_effectiveness"] is True


def test_stratum_one_micro_gate_boundary_uses_exact_integer_comparison() -> None:
    failed = _gate_state(_payload(_thirty_cases((-120_001,) + (0,) * 29)))
    passed = _gate_state(_payload(_thirty_cases((-20_000,) * 6 + (0,) * 24)))

    assert failed["stratum_effectiveness"] is False
    assert passed["stratum_effectiveness"] is True


def test_incomplete_payload_has_no_effectiveness_gate() -> None:
    complete = _thirty_cases((0,) * 30)
    incomplete = complete[:-1] + (
        _comparison(
            _case(case_id="case-29", stratum=BenchmarkStratum.IRRELEVANT_DISTRACTOR),
            baselines=(0, 0, 0),
            experience_hub=0,
            complete=False,
        ),
    )
    payload = BenchmarkPassPayloadV1.model_construct(
        schema_version=1,
        resolved_manifest=_manifest(),
        cases=incomplete,
        comparison_complete=False,
        safety=BenchmarkSafetyEvidenceV1(
            schema_version=1,
            owner_leak_count=0,
            quarantine_leak_count=0,
            cross_arm_contamination_count=0,
            source_mutation_count=0,
            source_unchanged=True,
            clone_isolation_verified=True,
        ),
        aggregate=None,
    )

    gates = {item.gate_id for item in evaluate_pilot_gates(payload, payload)}

    assert "overall_effectiveness" not in gates
    assert "stratum_effectiveness" not in gates


def test_declared_complete_case_with_a_failed_arm_cannot_open_effectiveness_gates(
) -> None:
    cases = _thirty_cases((0,) * 30)
    invalid_case = BenchmarkCaseEvidenceV1.model_construct(
        **{
            **cases[0].model_dump(),
            "arms": cases[0].arms[:-1]
            + (_arm("experience_hub", 0, complete=False),),
        }
    )
    payload = BenchmarkPassPayloadV1.model_construct(
        schema_version=1,
        resolved_manifest=_manifest(),
        cases=(invalid_case, *cases[1:]),
        comparison_complete=True,
        safety=BenchmarkSafetyEvidenceV1(
            schema_version=1,
            owner_leak_count=0,
            quarantine_leak_count=0,
            cross_arm_contamination_count=0,
            source_mutation_count=0,
            source_unchanged=True,
            clone_isolation_verified=True,
        ),
        aggregate=aggregate_benchmark_cases(cases),
    )

    gates = {
        item.gate_id: item.passed
        for item in evaluate_pilot_gates(payload, payload)
    }

    assert gates["complete_arms"] is False
    assert "overall_effectiveness" not in gates
