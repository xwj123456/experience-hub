from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.benchmark.experiencebench_factories import valid_case_document

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments import (
    REPLAY_WORKSPACE_POLICY,
    ExperimentOutputError,
    prepare_owned_workspace,
)
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    BENCHMARK_STRATUM_ORDER,
    BenchmarkArmDescriptorV1,
    BenchmarkArmEvidenceV1,
    BenchmarkArmObservationV1,
    BenchmarkCaseEvidenceV1,
    BenchmarkCaseV1,
    BenchmarkEvidenceDataV1,
    BenchmarkEvidenceReportV1,
    BenchmarkPassPayloadV1,
    BenchmarkProfileDataV1,
    BenchmarkProfileReportV1,
    BenchmarkSafetyEvidenceV1,
    BenchmarkStratum,
    ResolvedBenchmarkManifestV1,
)
from experience_hub.experiments.benchmarks.gates import evaluate_pilot_gates
from experience_hub.experiments.benchmarks.metrics import (
    aggregate_benchmark_cases,
    derive_case_comparison,
)
from experience_hub.experiments.benchmarks.oracles import score_benchmark_observation
from experience_hub.experiments.benchmarks.reports import (
    canonical_benchmark_evidence_bytes,
    canonical_benchmark_pass_bytes,
    canonical_benchmark_profile_bytes,
    canonical_benchmark_summary_bytes,
    derive_benchmark_summary,
    verify_benchmark_evidence_bytes,
    verify_benchmark_summary_bytes,
    write_benchmark_artifacts,
)


def _manifest(*, cases_sha256: str = "b" * 64) -> ResolvedBenchmarkManifestV1:
    return ResolvedBenchmarkManifestV1(
        schema_version=1,
        pack_id="experiencebench-s-pilot",
        maturity="pilot-30",
        manifest_sha256="a" * 64,
        cases_sha256=cases_sha256,
        source_fixture_sha256="c" * 64,
        snapshot_sha256="d" * 64,
        source_schema_revision=1,
        frozen_at=datetime(2026, 8, 20, tzinfo=UTC),
        seed=20260820,
        arms=tuple(
            BenchmarkArmDescriptorV1(
                schema_version=1, arm_id=kind.value, kind=kind, required=True
            )
            for kind in BENCHMARK_ARM_ORDER
        ),
        oracle_version=1,
        metric_version=1,
        gate_version=1,
        evidence_schema_version=1,
        summary_schema_version=1,
        profile_schema_version=1,
    )


def _arm(
    case: BenchmarkCaseV1,
    arm_id: str,
    *,
    returned_labels: tuple[str, ...] = (),
) -> BenchmarkArmEvidenceV1:
    observation = BenchmarkArmObservationV1(
        schema_version=1,
        returned_labels=returned_labels,
        selected_content_bytes=0,
    )
    return BenchmarkArmEvidenceV1(
        schema_version=1,
        arm_id=arm_id,
        status="complete",
        observation=observation,
        oracle=score_benchmark_observation(case, observation),
        error_code=None,
        error_stage=None,
    )


def _case(index: int, stratum: BenchmarkStratum) -> BenchmarkCaseEvidenceV1:
    document = valid_case_document()
    document["case_id"] = f"case-{index}"
    document["stratum"] = stratum.value
    case = BenchmarkCaseV1.model_validate_json(canonical_json_bytes(document))
    experience_labels = tuple(
        item.label for item in (*case.required, *case.optional)
    )
    arms = tuple(
        _arm(
            case,
            kind.value,
            returned_labels=(
                experience_labels if kind.value == "experience_hub" else ()
            ),
        )
        for kind in BENCHMARK_ARM_ORDER
    )
    return derive_case_comparison(case, arms)


def _payload() -> BenchmarkPassPayloadV1:
    cases = tuple(
        _case(index, stratum)
        for index, stratum in enumerate(
            item for item in BENCHMARK_STRATUM_ORDER for _ in range(6)
        )
    )
    return BenchmarkPassPayloadV1(
        schema_version=1,
        resolved_manifest=_manifest(
            cases_sha256=sha256_hex(
                b"".join(canonical_json_bytes(case.case) + b"\n" for case in cases)
            )
        ),
        cases=cases,
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


def _report() -> BenchmarkEvidenceReportV1:
    payload = _payload()
    gates = evaluate_pilot_gates(payload, payload)
    return BenchmarkEvidenceReportV1(
        data=BenchmarkEvidenceDataV1(
            schema_version=1,
            pass_payload=payload,
            deterministic_replay_match=True,
            gates=gates,
            expansion_gate_passed=True,
            valid=True,
        )
    )


def _profile() -> BenchmarkProfileReportV1:
    return BenchmarkProfileReportV1(
        data=BenchmarkProfileDataV1(
            schema_version=1,
            pack_id="experiencebench-s-pilot",
            profile_complete=True,
            wall_duration_ns=25,
            database_bytes=4096,
            clone_count=240,
            fts5_available=True,
        )
    )


def test_benchmark_reports_are_canonical_and_summary_binds_evidence_hash() -> None:
    payload = _payload()
    pass_body = canonical_benchmark_pass_bytes(payload)
    report = _report()

    evidence_body = canonical_benchmark_evidence_bytes(report)
    summary = derive_benchmark_summary(report, evidence_body=evidence_body)
    summary_body = canonical_json_bytes(summary)

    assert (
        sha256_hex(pass_body)
        == "8e68ca375695a8b8e571d8dec70253f9b9e43e9f1b2acb41ed1de3c6c28a0bac"
    )
    assert verify_benchmark_evidence_bytes(evidence_body) == report
    assert (
        verify_benchmark_summary_bytes(summary_body, evidence_body=evidence_body)
        == summary
    )
    assert summary.data.evidence_sha256 == sha256_hex(evidence_body)


def test_benchmark_report_recomputes_metric_gate_and_summary_state() -> None:
    report = _report()
    payload = report.data.pass_payload
    assert payload.aggregate is not None
    tampered_aggregate = payload.aggregate.model_copy(
        update={
            "overall": payload.aggregate.overall.model_copy(
                update={"sum_delta_micros": 1_500_030, "mean_delta_micros": 50_001}
            )
        }
    )
    tampered_payload = payload.model_copy(update={"aggregate": tampered_aggregate})
    with pytest.raises(ExperimentOutputError):
        canonical_benchmark_pass_bytes(tampered_payload)

    evidence = canonical_benchmark_evidence_bytes(report)
    document = json.loads(evidence)
    document["data"]["gates"][0]["passed"] = False
    document["data"]["expansion_gate_passed"] = False
    with pytest.raises(ExperimentOutputError):
        verify_benchmark_evidence_bytes(canonical_json_bytes(document))

    summary = derive_benchmark_summary(report, evidence_body=evidence)
    summary_document = summary.model_dump(mode="json")
    summary_document["data"]["claim_boundary"] = "Some other unsupported claim."
    with pytest.raises(ExperimentOutputError):
        verify_benchmark_summary_bytes(
            canonical_json_bytes(summary_document), evidence_body=evidence
        )


def test_benchmark_evidence_rejects_coherently_rederived_oracle_tampering() -> None:
    document = json.loads(canonical_benchmark_evidence_bytes(_report()))
    payload = document["data"]["pass_payload"]
    cases = payload["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        arms = case["arms"]
        assert isinstance(arms, list)
        experience_hub = arms[-1]
        assert isinstance(experience_hub, dict)
        oracle = experience_hub["oracle"]
        assert isinstance(oracle, dict)
        oracle.update(
            {
                "returned_labels": [],
                "required_labels": [],
                "optional_labels": [],
                "violation_labels": [],
                "satisfied_checkpoint_indexes": [],
                "scores": {
                    "schema_version": 1,
                    "required_coverage_micros": 0,
                    "avoidance_micros": 0,
                    "recovery_order_micros": 0,
                    "evidence_efficiency_micros": 0,
                    "utility_micros": 0,
                },
            }
        )
        case.update(
            {
                "comparator_arm_id": "no_memory",
                "comparator_utility_micros": 300000,
                "experience_hub_utility_micros": 0,
                "delta_utility_micros": -300000,
            }
        )
    aggregate = payload["aggregate"]
    assert isinstance(aggregate, dict)
    overall = aggregate["overall"]
    strata = aggregate["strata"]
    assert isinstance(overall, dict)
    assert isinstance(strata, list)
    overall.update({"sum_delta_micros": -9000000, "mean_delta_micros": -300000})
    for stratum in strata:
        assert isinstance(stratum, dict)
        stratum.update({"sum_delta_micros": -1800000, "mean_delta_micros": -300000})
    gates = document["data"]["gates"]
    assert isinstance(gates, list)
    for gate in gates:
        assert isinstance(gate, dict)
        if gate["gate_id"] in {"overall_effectiveness", "stratum_effectiveness"}:
            gate["passed"] = False
    document["data"]["expansion_gate_passed"] = False
    tampered = canonical_json_bytes(document)

    BenchmarkEvidenceReportV1.model_validate_json(tampered, strict=True)
    with pytest.raises(ExperimentOutputError, match="oracle evidence"):
        verify_benchmark_evidence_bytes(tampered)


def _rederived_rubric_tampered_report() -> BenchmarkEvidenceReportV1:
    original = _report()
    altered_cases: list[BenchmarkCaseEvidenceV1] = []
    for evidence_case in original.data.pass_payload.cases:
        document = evidence_case.case.model_dump(mode="json")
        document["required"][0]["label"] = "queue-forbidden"
        document["forbidden"][0]["label"] = "queue-required"
        document["checkpoints"][0]["labels"] = ["queue-forbidden"]
        rubric = BenchmarkCaseV1.model_validate_json(canonical_json_bytes(document))
        arms = tuple(
            arm.model_copy(
                update={
                    "oracle": score_benchmark_observation(rubric, arm.observation)
                }
            )
            for arm in evidence_case.arms
            if arm.observation is not None
        )
        altered_cases.append(derive_case_comparison(rubric, arms))
    cases = tuple(altered_cases)
    payload = original.data.pass_payload.model_copy(
        update={"cases": cases, "aggregate": aggregate_benchmark_cases(cases)}
    )
    return BenchmarkEvidenceReportV1(
        data=BenchmarkEvidenceDataV1(
            schema_version=1,
            pass_payload=payload,
            deterministic_replay_match=True,
            gates=evaluate_pilot_gates(payload, payload),
            expansion_gate_passed=all(
                gate.passed for gate in evaluate_pilot_gates(payload, payload)
            ),
            valid=True,
        )
    )


def test_benchmark_evidence_rejects_rederived_rubrics_without_manifest_hash_change(
) -> None:
    forged = _rederived_rubric_tampered_report()
    body = canonical_json_bytes(forged)

    with pytest.raises(ExperimentOutputError, match="cases"):
        canonical_benchmark_evidence_bytes(forged)
    with pytest.raises(ExperimentOutputError, match="cases"):
        verify_benchmark_evidence_bytes(body)


def test_benchmark_evidence_rejects_embedded_case_order_changes() -> None:
    original = _report()
    cases = tuple(reversed(original.data.pass_payload.cases))
    payload = original.data.pass_payload.model_copy(
        update={"cases": cases, "aggregate": aggregate_benchmark_cases(cases)}
    )
    reordered = BenchmarkEvidenceReportV1(
        data=BenchmarkEvidenceDataV1(
            schema_version=1,
            pass_payload=payload,
            deterministic_replay_match=True,
            gates=evaluate_pilot_gates(payload, payload),
            expansion_gate_passed=True,
            valid=True,
        )
    )
    body = canonical_json_bytes(reordered)

    with pytest.raises(ExperimentOutputError, match="cases"):
        canonical_benchmark_evidence_bytes(reordered)
    with pytest.raises(ExperimentOutputError, match="cases"):
        verify_benchmark_evidence_bytes(body)


def test_incomplete_evidence_is_validly_encoded_but_not_valid() -> None:
    payload = _payload()
    failed = BenchmarkArmEvidenceV1(
        schema_version=1,
        arm_id="experience_hub",
        status="failed",
        observation=None,
        oracle=None,
        error_code="benchmark_policy_failed",
        error_stage="policy",
    )
    original = payload.cases[0]
    incomplete = BenchmarkCaseEvidenceV1(
        schema_version=1,
        case=original.case,
        case_id=original.case_id,
        source_class=original.source_class,
        stratum=original.stratum,
        status="incomplete",
        arms=(*original.arms[:-1], failed),
        comparator_arm_id=None,
        comparator_utility_micros=None,
        experience_hub_utility_micros=None,
        delta_utility_micros=None,
    )
    cases = (incomplete, *payload.cases[1:])
    incomplete_payload = payload.model_copy(
        update={"cases": cases, "comparison_complete": False, "aggregate": None}
    )
    report = BenchmarkEvidenceReportV1(
        data=BenchmarkEvidenceDataV1(
            schema_version=1,
            pass_payload=incomplete_payload,
            deterministic_replay_match=True,
            gates=evaluate_pilot_gates(incomplete_payload, incomplete_payload),
            expansion_gate_passed=False,
            valid=False,
        )
    )

    assert (
        verify_benchmark_evidence_bytes(canonical_benchmark_evidence_bytes(report))
        == report
    )


@pytest.mark.parametrize(
    "unsafe",
    (
        "/private/workspace/report.json",
        "10000000-0000-4000-8000-000000000701",
        "2026-08-20T10:11:12Z",
    ),
)
def test_benchmark_evidence_rejects_private_or_unstable_values(unsafe: str) -> None:
    original = _payload()
    case = original.cases[0].case.model_copy(update={"query": unsafe})
    first = original.cases[0].model_copy(update={"case": case})
    payload = original.model_copy(update={"cases": (first, *original.cases[1:])})
    report = _report().model_copy(
        update={"data": _report().data.model_copy(update={"pass_payload": payload})}
    )

    with pytest.raises(ExperimentOutputError):
        canonical_benchmark_evidence_bytes(report)


def test_benchmark_evidence_rejects_credentials_and_output_over_cap() -> None:
    report = _report()
    document = report.model_dump(mode="python")
    document["data"]["credential"] = "private"

    with pytest.raises(ExperimentOutputError):
        verify_benchmark_evidence_bytes(canonical_json_bytes(document))
    with pytest.raises(ExperimentOutputError):
        verify_benchmark_evidence_bytes(b" " * (2 * 1024 * 1024 + 1))


def test_benchmark_summary_rejects_an_unsafe_schema_valid_claim_boundary() -> None:
    report = _report()
    evidence_body = canonical_benchmark_evidence_bytes(report)
    summary = derive_benchmark_summary(report, evidence_body=evidence_body).model_copy(
        update={
            "data": derive_benchmark_summary(report, evidence_body=evidence_body)
            .data.model_copy(update={"claim_boundary": "/private/benchmark-claim"})
        }
    )

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_benchmark_summary_bytes(summary)

    assert captured.value.code == "invalid_benchmark_summary"


def test_public_benchmark_encoders_reject_oversized_models_before_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = "x" * (2 * 1024 * 1024 + 1)
    payload = _payload()
    oversized_case = payload.cases[0].case.model_copy(update={"query": oversized})
    oversized_evidence_case = payload.cases[0].model_copy(
        update={"case": oversized_case}
    )
    oversized_payload = payload.model_copy(
        update={"cases": (oversized_evidence_case, *payload.cases[1:])}
    )
    oversized_report = _report().model_copy(
        update={
            "data": _report().data.model_copy(
                update={"pass_payload": oversized_payload}
            )
        }
    )
    summary = derive_benchmark_summary(
        _report(), evidence_body=canonical_benchmark_evidence_bytes(_report())
    ).model_copy(
        update={
            "data": derive_benchmark_summary(
                _report(), evidence_body=canonical_benchmark_evidence_bytes(_report())
            ).data.model_copy(update={"claim_boundary": oversized})
        }
    )
    profile = BenchmarkProfileReportV1.model_construct(
        data=BenchmarkProfileDataV1.model_construct(
            schema_version=1,
            pack_id=oversized,
            profile_complete=True,
            wall_duration_ns=25,
            database_bytes=4096,
            clone_count=240,
            fts5_available=True,
        )
    )

    def unexpected_dump(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized output reached model_dump")

    for encoder, value, model_type in (
        (canonical_benchmark_pass_bytes, oversized_payload, BenchmarkPassPayloadV1),
        (
            canonical_benchmark_evidence_bytes,
            oversized_report,
            BenchmarkEvidenceReportV1,
        ),
        (canonical_benchmark_summary_bytes, summary, type(summary)),
        (canonical_benchmark_profile_bytes, profile, BenchmarkProfileReportV1),
    ):
        monkeypatch.setattr(model_type, "model_dump", unexpected_dump)
        with pytest.raises(ExperimentOutputError) as captured:
            encoder(value)
        assert captured.value.code == "output_too_large"


def test_public_profile_encoder_rejects_a_cycle_before_unbounded_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingCycle(list[object]):
        visits = 0

        def __iter__(self) -> object:
            self.visits += 1
            if self.visits > 8:
                raise AssertionError("preflight revisited cyclic list")
            return super().__iter__()

    cycle = CountingCycle()
    cycle.append(cycle)
    profile = BenchmarkProfileReportV1.model_construct(
        data=BenchmarkProfileDataV1.model_construct(
            schema_version=1,
            pack_id=cycle,
            profile_complete=True,
            wall_duration_ns=25,
            database_bytes=4096,
            clone_count=240,
            fts5_available=True,
        )
    )

    def unexpected_dump(*args: object, **kwargs: object) -> object:
        raise AssertionError("cyclic output reached model_dump")

    monkeypatch.setattr(BenchmarkProfileReportV1, "model_dump", unexpected_dump)
    with pytest.raises(ExperimentOutputError) as captured:
        canonical_benchmark_profile_bytes(profile)

    assert captured.value.code == "invalid_benchmark_profile"
    assert cycle.visits == 1


def test_preflight_keeps_repeated_noncyclic_model_aliases_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = BenchmarkProfileDataV1.model_construct(
        schema_version=1,
        pack_id="experiencebench-s-pilot",
        profile_complete=True,
        wall_duration_ns=25,
        database_bytes=4096,
        clone_count=240,
        fts5_available=True,
    )
    profile = BenchmarkProfileReportV1.model_construct(
        data=BenchmarkProfileDataV1.model_construct(
            schema_version=1,
            pack_id=(shared, shared),
            profile_complete=True,
            wall_duration_ns=25,
            database_bytes=4096,
            clone_count=240,
            fts5_available=True,
        )
    )
    calls = 0
    real_dump = BenchmarkProfileReportV1.model_dump

    def record_dump(
        self: BenchmarkProfileReportV1, *args: object, **kwargs: object
    ) -> object:
        nonlocal calls
        calls += 1
        return real_dump(self, *args, **kwargs)

    monkeypatch.setattr(BenchmarkProfileReportV1, "model_dump", record_dump)
    with pytest.raises(ExperimentOutputError) as captured:
        canonical_benchmark_profile_bytes(profile)

    assert captured.value.code == "invalid_benchmark_profile"
    assert calls == 1


def test_benchmark_artifacts_keep_profile_separate_and_write_exact_paths(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    artifacts = write_benchmark_artifacts(
        workspace, evidence=_report(), profile=_profile()
    )

    assert (
        artifacts.evidence_path == workspace.root / "artifacts/benchmark-evidence.json"
    )
    assert artifacts.summary_path == workspace.root / "artifacts/benchmark-summary.json"
    assert artifacts.profile_path == workspace.root / "artifacts/profile.json"
    assert artifacts.evidence_path.read_bytes() == artifacts.evidence_body
    assert artifacts.summary_path.read_bytes() == artifacts.summary_body
    assert artifacts.profile_path.read_bytes() == artifacts.profile_body
    assert b"wall_duration_ns" not in artifacts.evidence_body
    assert b"wall_duration_ns" not in artifacts.summary_body


def test_omitted_or_failed_profile_cannot_change_evidence_or_expansion_gate(
    tmp_path: Path,
) -> None:
    first_workspace = prepare_owned_workspace(
        tmp_path / "first",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    second_workspace = prepare_owned_workspace(
        tmp_path / "second",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )

    with_profile = write_benchmark_artifacts(
        first_workspace, evidence=_report(), profile=_profile()
    )
    without_profile = write_benchmark_artifacts(
        second_workspace, evidence=_report(), profile=None
    )

    assert without_profile.profile_path is None
    assert without_profile.profile_body is None
    assert without_profile.evidence_body == with_profile.evidence_body
    assert without_profile.summary_body == with_profile.summary_body

    previous_entries = {
        path: (path.read_bytes(), path.stat().st_ino)
        for path in (
            with_profile.evidence_path,
            with_profile.summary_path,
            with_profile.profile_path,
        )
        if path is not None
    }
    with pytest.raises(ExperimentOutputError) as captured:
        write_benchmark_artifacts(first_workspace, evidence=_report(), profile=None)

    assert captured.value.code == "artifact_write_failed"
    assert {
        path: (path.read_bytes(), path.stat().st_ino) for path in previous_entries
    } == previous_entries


def test_profile_serialization_failure_retains_the_previous_artifact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    previous = write_benchmark_artifacts(
        workspace, evidence=_report(), profile=_profile()
    )
    previous_entries = {
        path: (path.read_bytes(), path.stat().st_ino)
        for path in (
            previous.evidence_path,
            previous.summary_path,
            previous.profile_path,
        )
        if path is not None
    }

    def fail_model_dump(*args: object, **kwargs: object) -> object:
        raise OSError("/private/profile-serialization")

    monkeypatch.setattr(BenchmarkProfileReportV1, "model_dump", fail_model_dump)
    with pytest.raises(ExperimentOutputError) as captured:
        write_benchmark_artifacts(workspace, evidence=_report(), profile=_profile())

    assert captured.value.code == "invalid_benchmark_profile"
    assert "/private/profile-serialization" not in str(captured.value)
    assert {
        path: (path.read_bytes(), path.stat().st_ino) for path in previous_entries
    } == previous_entries
