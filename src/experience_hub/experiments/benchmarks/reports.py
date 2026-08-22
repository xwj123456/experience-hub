"""Canonical ExperienceBench-S evidence, summaries, and artifact publication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    MAX_BENCHMARK_OUTPUT_BYTES,
    BenchmarkArmEvidenceV1,
    BenchmarkCaseEvidenceV1,
    BenchmarkEvidenceDataV1,
    BenchmarkEvidenceReportV1,
    BenchmarkGateResultV1,
    BenchmarkPassPayloadV1,
    BenchmarkProfileReportV1,
    BenchmarkSummaryDataV1,
    BenchmarkSummaryReportV1,
)
from experience_hub.experiments.benchmarks.gates import evaluate_pilot_gates
from experience_hub.experiments.benchmarks.metrics import aggregate_benchmark_cases
from experience_hub.experiments.errors import ExperimentIsolationError
from experience_hub.experiments.reports import (
    ExperimentOutputError,
    validate_safe_evidence_document,
)
from experience_hub.experiments.workspace import OwnedWorkspace

_FROZEN_AT_PATH = ("data", "pass_payload", "resolved_manifest", "frozen_at")
_PASS_FROZEN_AT_PATH = ("resolved_manifest", "frozen_at")
_EVIDENCE_PATH = PurePosixPath("artifacts/benchmark-evidence.json")
_SUMMARY_PATH = PurePosixPath("artifacts/benchmark-summary.json")
_PROFILE_PATH = PurePosixPath("artifacts/profile.json")
_CLAIM_BOUNDARY = (
    "Pilot evidence is limited to this frozen benchmark and does not establish "
    "general effectiveness, truth, or human-equivalent memory."
)


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactSet:
    """Exact, independently verifiable benchmark artifact bodies and paths."""

    evidence_path: Path
    summary_path: Path
    profile_path: Path | None
    evidence_body: bytes
    summary_body: bytes
    profile_body: bytes | None


def _reject(code: str, message: str) -> ExperimentOutputError:
    return ExperimentOutputError(code, message)


def _safe_document(
    document: object,
    *,
    pass_payload: bool = False,
    error_code: str = "invalid_benchmark_evidence",
    document_name: str = "Benchmark evidence",
) -> None:
    try:
        validate_safe_evidence_document(
            document,
            allowed_timestamp_paths=frozenset(
                {_PASS_FROZEN_AT_PATH if pass_payload else _FROZEN_AT_PATH}
            ),
        )
    except ExperimentOutputError:
        raise _reject(
            error_code,
            f"{document_name} contains unsafe values",
        ) from None


def _validate_arm(arm: BenchmarkArmEvidenceV1) -> bool:
    if arm.status == "complete":
        if arm.observation is None or arm.oracle is None:
            raise _reject("invalid_benchmark_evidence", "Benchmark arm is invalid")
        scores = arm.oracle.scores
        if scores.utility_micros != sum(
            (
                scores.required_coverage_micros,
                scores.avoidance_micros,
                scores.recovery_order_micros,
                scores.evidence_efficiency_micros,
            )
        ):
            raise _reject(
                "invalid_benchmark_evidence", "Benchmark category scores are invalid"
            )
        return True
    if arm.observation is not None or arm.oracle is not None:
        raise _reject("invalid_benchmark_evidence", "Benchmark arm is invalid")
    return False


def _validate_case(case: BenchmarkCaseEvidenceV1) -> BenchmarkCaseEvidenceV1:
    if tuple(arm.arm_id for arm in case.arms) != tuple(
        item.value for item in BENCHMARK_ARM_ORDER
    ):
        raise _reject("invalid_benchmark_evidence", "Benchmark case arms are invalid")
    complete = tuple(_validate_arm(arm) for arm in case.arms)
    if all(complete):
        utilities = tuple(
            arm.oracle.scores.utility_micros
            for arm in case.arms
            if arm.oracle is not None
        )
        if len(utilities) != len(BENCHMARK_ARM_ORDER):
            raise _reject("invalid_benchmark_evidence", "Benchmark arm is invalid")
        baseline = utilities[:3]
        comparator_utility = max(baseline)
        comparator_index = baseline.index(comparator_utility)
        expected = case.model_copy(
            update={
                "status": "complete",
                "comparator_arm_id": BENCHMARK_ARM_ORDER[comparator_index].value,
                "comparator_utility_micros": comparator_utility,
                "experience_hub_utility_micros": utilities[3],
                "delta_utility_micros": utilities[3] - comparator_utility,
            }
        )
    else:
        expected = case.model_copy(
            update={
                "status": "incomplete",
                "comparator_arm_id": None,
                "comparator_utility_micros": None,
                "experience_hub_utility_micros": None,
                "delta_utility_micros": None,
            }
        )
    if case != expected:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark case comparison is invalid"
        )
    return expected


def _validated_pass(payload: BenchmarkPassPayloadV1) -> BenchmarkPassPayloadV1:
    if not isinstance(payload, BenchmarkPassPayloadV1):
        raise _reject("invalid_benchmark_evidence", "Benchmark pass is invalid")
    try:
        document = payload.model_dump(mode="python", warnings=False)
        _safe_document(document, pass_payload=True)
        validated = BenchmarkPassPayloadV1.model_validate(document, strict=True)
        _safe_document(
            validated.model_dump(mode="python", warnings=False), pass_payload=True
        )
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark pass is invalid"
        ) from None
    cases = tuple(_validate_case(case) for case in validated.cases)
    try:
        aggregate = aggregate_benchmark_cases(cases)
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark aggregate is invalid"
        ) from None
    if (
        validated.comparison_complete != (aggregate is not None)
        or validated.aggregate != aggregate
    ):
        raise _reject("invalid_benchmark_evidence", "Benchmark aggregate is invalid")
    return validated


def canonical_benchmark_pass_bytes(payload: BenchmarkPassPayloadV1) -> bytes:
    """Encode one deterministic pass payload for equality comparison."""
    validated = _validated_pass(payload)
    try:
        body = canonical_json_bytes(validated)
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark pass is invalid"
        ) from None
    _require_output_cap(body, document="Benchmark pass")
    return body


def _expected_gates(
    payload: BenchmarkPassPayloadV1, *, deterministic_replay_match: bool
) -> tuple[BenchmarkGateResultV1, ...]:
    gates = evaluate_pilot_gates(payload, payload)
    return tuple(
        BenchmarkGateResultV1(
            schema_version=1,
            gate_id=gate.gate_id,
            passed=(
                deterministic_replay_match
                if gate.gate_id == "deterministic_replay"
                else gate.passed
            ),
        )
        for gate in gates
    )


def _validated_evidence(report: BenchmarkEvidenceReportV1) -> BenchmarkEvidenceReportV1:
    if not isinstance(report, BenchmarkEvidenceReportV1):
        raise _reject("invalid_benchmark_evidence", "Benchmark evidence is invalid")
    try:
        document = report.model_dump(mode="python", warnings=False)
        _safe_document(document)
        validated = BenchmarkEvidenceReportV1.model_validate(document, strict=True)
        _safe_document(validated.model_dump(mode="python", warnings=False))
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark evidence is invalid"
        ) from None
    payload = _validated_pass(validated.data.pass_payload)
    expected_gates = _expected_gates(
        payload,
        deterministic_replay_match=validated.data.deterministic_replay_match,
    )
    valid = (
        payload.comparison_complete
        and validated.data.deterministic_replay_match
        and payload.safety.is_safe
    )
    expansion = valid and all(gate.passed for gate in expected_gates)
    expected = BenchmarkEvidenceDataV1(
        schema_version=1,
        pass_payload=payload,
        deterministic_replay_match=validated.data.deterministic_replay_match,
        gates=expected_gates,
        expansion_gate_passed=expansion,
        valid=valid,
    )
    if validated.data != expected:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark derived evidence is invalid"
        )
    return BenchmarkEvidenceReportV1(data=expected)


def _require_output_cap(body: bytes, *, document: str) -> None:
    if len(body) > MAX_BENCHMARK_OUTPUT_BYTES:
        raise _reject("output_too_large", f"{document} exceeds the output byte limit")


def canonical_benchmark_evidence_bytes(report: BenchmarkEvidenceReportV1) -> bytes:
    """Validate and encode authoritative pilot evidence."""
    validated = _validated_evidence(report)
    try:
        body = canonical_json_bytes(validated)
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark evidence is invalid"
        ) from None
    _require_output_cap(body, document="Benchmark evidence")
    return body


def verify_benchmark_evidence_bytes(body: bytes) -> BenchmarkEvidenceReportV1:
    """Boundedly decode and recompute canonical benchmark evidence."""
    if not isinstance(body, bytes):
        raise _reject("invalid_benchmark_evidence", "Benchmark evidence must be bytes")
    _require_output_cap(body, document="Benchmark evidence")
    try:
        document = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _reject("invalid_json", "Benchmark evidence must be UTF-8 JSON") from None
    if not isinstance(document, dict):
        raise _reject("invalid_json", "Benchmark evidence must be a JSON object")
    try:
        if canonical_json_bytes(document) != body:
            raise _reject(
                "noncanonical_json", "Benchmark evidence must use canonical JSON"
            )
        _safe_document(document)
        report = BenchmarkEvidenceReportV1.model_validate_json(body, strict=True)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_benchmark_evidence", "Benchmark evidence is invalid"
        ) from None
    validated = _validated_evidence(report)
    if canonical_benchmark_evidence_bytes(validated) != body:
        raise _reject(
            "noncanonical_json", "Benchmark evidence does not match its derivation"
        )
    return validated


def derive_benchmark_summary(
    report: BenchmarkEvidenceReportV1, *, evidence_body: bytes
) -> BenchmarkSummaryReportV1:
    """Derive the public summary and bind it to evidence SHA-256."""
    validated = _validated_evidence(report)
    if verify_benchmark_evidence_bytes(evidence_body) != validated:
        raise _reject(
            "invalid_benchmark_summary", "Benchmark evidence does not match report"
        )
    payload = validated.data.pass_payload
    manifest = payload.resolved_manifest
    return BenchmarkSummaryReportV1(
        data=BenchmarkSummaryDataV1(
            schema_version=1,
            pack_id=manifest.pack_id,
            manifest_sha256=manifest.manifest_sha256,
            cases_sha256=manifest.cases_sha256,
            source_fixture_sha256=manifest.source_fixture_sha256,
            snapshot_sha256=manifest.snapshot_sha256,
            evidence_sha256=sha256_hex(evidence_body),
            case_count=30,
            arm_count=4,
            comparison_complete=payload.comparison_complete,
            deterministic_replay_match=validated.data.deterministic_replay_match,
            safety=payload.safety,
            aggregate=payload.aggregate,
            gates=validated.data.gates,
            expansion_gate_passed=validated.data.expansion_gate_passed,
            valid=validated.data.valid,
            claim_boundary=_CLAIM_BOUNDARY,
        )
    )


def canonical_benchmark_summary_bytes(summary: BenchmarkSummaryReportV1) -> bytes:
    """Encode an already-derived benchmark summary within the output cap."""
    if not isinstance(summary, BenchmarkSummaryReportV1):
        raise _reject("invalid_benchmark_summary", "Benchmark summary is invalid")
    try:
        document = summary.model_dump(mode="python", warnings=False)
        _safe_document(
            document,
            error_code="invalid_benchmark_summary",
            document_name="Benchmark summary",
        )
        validated = BenchmarkSummaryReportV1.model_validate(document, strict=True)
        body = canonical_json_bytes(validated)
    except Exception:
        raise _reject(
            "invalid_benchmark_summary", "Benchmark summary is invalid"
        ) from None
    _require_output_cap(body, document="Benchmark summary")
    return body


def verify_benchmark_summary_bytes(
    body: bytes, *, evidence_body: bytes
) -> BenchmarkSummaryReportV1:
    """Verify a summary by regenerating it from the authoritative evidence."""
    if not isinstance(body, bytes):
        raise _reject("invalid_benchmark_summary", "Benchmark summary must be bytes")
    _require_output_cap(body, document="Benchmark summary")
    try:
        document = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _reject("invalid_json", "Benchmark summary must be UTF-8 JSON") from None
    if not isinstance(document, dict):
        raise _reject("invalid_json", "Benchmark summary must be a JSON object")
    try:
        if canonical_json_bytes(document) != body:
            raise _reject(
                "noncanonical_json", "Benchmark summary must use canonical JSON"
            )
        _safe_document(
            document,
            error_code="invalid_benchmark_summary",
            document_name="Benchmark summary",
        )
        summary = BenchmarkSummaryReportV1.model_validate_json(body, strict=True)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_benchmark_summary", "Benchmark summary is invalid"
        ) from None
    evidence = verify_benchmark_evidence_bytes(evidence_body)
    expected = derive_benchmark_summary(evidence, evidence_body=evidence_body)
    if summary != expected or canonical_benchmark_summary_bytes(summary) != body:
        raise _reject(
            "invalid_benchmark_summary", "Benchmark summary does not match evidence"
        )
    return expected


def canonical_benchmark_profile_bytes(profile: BenchmarkProfileReportV1) -> bytes:
    """Encode profile-only runtime measurements independently from evidence."""
    if not isinstance(profile, BenchmarkProfileReportV1):
        raise _reject("invalid_benchmark_profile", "Benchmark profile is invalid")
    try:
        validated = BenchmarkProfileReportV1.model_validate(
            profile.model_dump(mode="python", warnings=False), strict=True
        )
        body = canonical_json_bytes(validated)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_benchmark_profile", "Benchmark profile is invalid"
        ) from None
    _require_output_cap(body, document="Benchmark profile")
    return body


def write_benchmark_artifacts(
    workspace: OwnedWorkspace,
    *,
    evidence: BenchmarkEvidenceReportV1,
    profile: BenchmarkProfileReportV1 | None,
) -> BenchmarkArtifactSet:
    """Publish separate profile, summary, and evidence through atomic writes."""
    if not isinstance(workspace, OwnedWorkspace):
        raise _reject(
            "artifact_write_failed", "Benchmark artifacts require an owned workspace"
        )
    evidence_body = canonical_benchmark_evidence_bytes(evidence)
    summary = derive_benchmark_summary(evidence, evidence_body=evidence_body)
    summary_body = canonical_benchmark_summary_bytes(summary)
    profile_body = (
        canonical_benchmark_profile_bytes(profile) if profile is not None else None
    )
    if (
        profile is not None
        and profile.data.pack_id != evidence.data.pass_payload.resolved_manifest.pack_id
    ):
        raise _reject(
            "invalid_benchmark_profile", "Benchmark profile does not match evidence"
        )
    try:
        profile_path = (
            workspace.atomic_write(_PROFILE_PATH, profile_body)
            if profile_body is not None
            else None
        )
        summary_path = workspace.atomic_write(_SUMMARY_PATH, summary_body)
        evidence_path = workspace.atomic_write(_EVIDENCE_PATH, evidence_body)
    except (ExperimentIsolationError, OSError):
        raise _reject(
            "artifact_write_failed", "Benchmark artifact bytes could not be published"
        ) from None
    return BenchmarkArtifactSet(
        evidence_path=evidence_path,
        summary_path=summary_path,
        profile_path=profile_path,
        evidence_body=evidence_body,
        summary_body=summary_body,
        profile_body=profile_body,
    )


__all__ = [
    "BenchmarkArtifactSet",
    "canonical_benchmark_evidence_bytes",
    "canonical_benchmark_pass_bytes",
    "canonical_benchmark_profile_bytes",
    "canonical_benchmark_summary_bytes",
    "derive_benchmark_summary",
    "verify_benchmark_evidence_bytes",
    "verify_benchmark_summary_bytes",
    "write_benchmark_artifacts",
]
