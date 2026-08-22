from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
    BenchmarkCategoryScoresV1,
    BenchmarkEvidenceDataV1,
    BenchmarkEvidenceReportV1,
    BenchmarkOracleEvidenceV1,
    BenchmarkPassPayloadV1,
    BenchmarkProfileDataV1,
    BenchmarkProfileReportV1,
    BenchmarkSafetyEvidenceV1,
    BenchmarkSourceClass,
    BenchmarkStratum,
    ResolvedBenchmarkManifestV1,
)
from experience_hub.experiments.benchmarks.gates import evaluate_pilot_gates
from experience_hub.experiments.benchmarks.metrics import (
    aggregate_benchmark_cases,
)
from experience_hub.experiments.benchmarks.reports import (
    canonical_benchmark_evidence_bytes,
    canonical_benchmark_pass_bytes,
    derive_benchmark_summary,
    verify_benchmark_evidence_bytes,
    verify_benchmark_summary_bytes,
    write_benchmark_artifacts,
)
from experience_hub.experiments.workspace import OwnedWorkspace


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


def _arm(arm_id: str, utility_micros: int) -> BenchmarkArmEvidenceV1:
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
            scores=BenchmarkCategoryScoresV1(
                schema_version=1,
                required_coverage_micros=utility_micros,
                avoidance_micros=0,
                recovery_order_micros=0,
                evidence_efficiency_micros=0,
                utility_micros=utility_micros,
            ),
        ),
        error_code=None,
        error_stage=None,
    )


def _case(index: int, stratum: BenchmarkStratum) -> BenchmarkCaseEvidenceV1:
    arms = tuple(
        _arm(kind.value, 500_000 + (50_000 if kind.value == "experience_hub" else 0))
        for kind in BENCHMARK_ARM_ORDER
    )
    return BenchmarkCaseEvidenceV1(
        schema_version=1,
        case_id=f"case-{index}",
        source_class=BenchmarkSourceClass.PUBLIC_AUTHORED,
        stratum=stratum,
        status="complete",
        arms=arms,
        comparator_arm_id="no_memory",
        comparator_utility_micros=500_000,
        experience_hub_utility_micros=550_000,
        delta_utility_micros=50_000,
    )


def _payload() -> BenchmarkPassPayloadV1:
    cases = tuple(
        _case(index, stratum)
        for index, stratum in enumerate(
            item for item in BENCHMARK_STRATUM_ORDER for _ in range(6)
        )
    )
    return BenchmarkPassPayloadV1(
        schema_version=1,
        resolved_manifest=_manifest(),
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

    assert pass_body == canonical_json_bytes(payload)
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
    payload = _payload().model_copy(
        update={"resolved_manifest": _manifest().model_copy(update={"pack_id": unsafe})}
    )
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


def test_benchmark_artifact_write_failures_are_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )

    def fail_write(*args: object, **kwargs: object) -> Path:
        raise OSError("/private/write-failure")

    monkeypatch.setattr(OwnedWorkspace, "atomic_write", fail_write)
    with pytest.raises(ExperimentOutputError) as captured:
        write_benchmark_artifacts(workspace, evidence=_report(), profile=None)

    assert captured.value.code == "artifact_write_failed"
    assert "/private/write-failure" not in str(captured.value)
