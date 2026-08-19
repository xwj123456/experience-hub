"""Contract tests for canonical release evidence documents."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experience_hub import release_evidence
from experience_hub.release_evidence import (
    REQUIRED_RELEASE_CHECKS,
    CheckName,
    ReleaseEvidenceReportV1,
)
from experience_hub.release_evidence.errors import ReleaseEvidenceError


def valid_release_evidence_document() -> dict[str, object]:
    """Return a complete valid release-evidence document fixture."""
    return {
        "data": {
            "schema_version": 1,
            "verified_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "verified_on": "2026-08-03",
            "python_version": "3.12",
            "checks": [
                {"name": check, "passed": True}
                for check in CheckName
            ],
            "test_count": 42,
            "demo": {"all_invariants_hold": True, "stage_count": 3},
            "benchmark": {
                "passed": True,
                "case_count": 8,
                "gate_count": 4,
                "passed_gate_count": 4,
                "byte_identical_replay": True,
                "pending_capsule_leakage_count": 0,
            },
        }
    }


def _data(document: dict[str, object]) -> dict[str, object]:
    data = document["data"]
    assert isinstance(data, dict)
    return data


def _checks(document: dict[str, object]) -> list[dict[str, object]]:
    checks = _data(document)["checks"]
    assert isinstance(checks, list)
    assert all(isinstance(check, dict) for check in checks)
    return checks


def _benchmark(document: dict[str, object]) -> dict[str, object]:
    benchmark = _data(document)["benchmark"]
    assert isinstance(benchmark, dict)
    return benchmark


def _demo(document: dict[str, object]) -> dict[str, object]:
    demo = _data(document)["demo"]
    assert isinstance(demo, dict)
    return demo


def test_release_evidence_accepts_the_complete_canonical_document() -> None:
    report = ReleaseEvidenceReportV1.model_validate(
        valid_release_evidence_document(), strict=True
    )

    assert report.data.schema_version == 1
    assert tuple(check.name for check in report.data.checks) == REQUIRED_RELEASE_CHECKS
    assert all(check.passed for check in report.data.checks)


def test_release_evidence_requires_exact_ordered_checks() -> None:
    document = valid_release_evidence_document()
    _checks(document).reverse()

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_duplicate_check_names() -> None:
    document = valid_release_evidence_document()
    checks = _checks(document)
    checks[1]["name"] = checks[0]["name"]

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_missing_required_check() -> None:
    document = valid_release_evidence_document()
    _checks(document).pop()

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_unknown_fields() -> None:
    document = valid_release_evidence_document()
    _data(document)["unrecognized"] = "value"

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_non_boolean_checks() -> None:
    document = valid_release_evidence_document()
    _checks(document)[0]["passed"] = "true"

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("verified_commit", "a" * 39),
        ("source_tree_sha256", "A" * 64),
        ("verified_on", "2026-8-3"),
        ("test_count", 0),
    ),
)
def test_release_evidence_rejects_invalid_stable_fields(
    field: str,
    value: object,
) -> None:
    document = valid_release_evidence_document()
    _data(document)[field] = value

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_failed_demo_invariant() -> None:
    document = valid_release_evidence_document()
    _demo(document)["all_invariants_hold"] = False

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_unequal_benchmark_gate_counts() -> None:
    document = valid_release_evidence_document()
    _benchmark(document)["passed_gate_count"] = 3

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_nonzero_pending_capsule_leakage() -> None:
    document = valid_release_evidence_document()
    _benchmark(document)["pending_capsule_leakage_count"] = 1

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_nonidentical_benchmark_replay() -> None:
    document = valid_release_evidence_document()
    _benchmark(document)["byte_identical_replay"] = False

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("demo", "all_invariants_hold"),
        ("benchmark", "passed"),
        ("benchmark", "byte_identical_replay"),
    ),
)
def test_release_evidence_rejects_integer_one_for_literal_true(
    section: str,
    field: str,
) -> None:
    document = valid_release_evidence_document()
    target = _data(document)[section]
    assert isinstance(target, dict)
    target[field] = 1

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_rejects_boolean_false_for_literal_integer_zero() -> None:
    document = valid_release_evidence_document()
    _benchmark(document)["pending_capsule_leakage_count"] = False

    with pytest.raises(ValidationError):
        ReleaseEvidenceReportV1.model_validate(document, strict=True)


def test_release_evidence_exports_the_locked_v1_contract() -> None:
    assert tuple(CheckName) == REQUIRED_RELEASE_CHECKS
    assert set(release_evidence.__all__) == {
        "BenchmarkEvidenceV1",
        "CheckEvidenceV1",
        "CheckName",
        "DemoEvidenceV1",
        "REQUIRED_RELEASE_CHECKS",
        "ReleaseEvidenceDataV1",
        "ReleaseEvidenceReportV1",
    }
    assert not hasattr(release_evidence, "ReleaseEvidenceError")


def test_release_evidence_error_retains_only_stable_public_fields() -> None:
    error = ReleaseEvidenceError(code="invalid_evidence", message="invalid report")

    assert error.code == "invalid_evidence"
    assert error.message == "invalid report"
