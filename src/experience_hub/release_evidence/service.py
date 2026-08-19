"""Release-evidence orchestration over collection, storage, and source closure."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from experience_hub.release_evidence.collection import (
    CheckRunner,
    collect_release_evidence,
)
from experience_hub.release_evidence.contracts import ReleaseEvidenceReportV1
from experience_hub.release_evidence.errors import ReleaseEvidenceError
from experience_hub.release_evidence.source_tree import (
    SourceTreeIdentity,
    inspect_source_tree,
    verify_source_tree,
)
from experience_hub.release_evidence.storage import (
    load_release_evidence,
    store_release_evidence,
)


def collect_and_store_release_evidence(
    repository: Path,
    evidence_path: Path,
    *,
    verified_on: str,
    runner: CheckRunner,
) -> ReleaseEvidenceReportV1:
    """Collect checks for one closed tree and atomically retain their evidence."""
    _validate_verified_on(verified_on)
    inspect_source_tree(repository)
    report = collect_release_evidence(
        repository,
        verified_on=verified_on,
        runner=runner,
    )
    final_identity = inspect_source_tree(repository)
    if _report_identity(report) != final_identity:
        raise _stale_evidence()
    store_release_evidence(repository, evidence_path, report)
    return report


def verify_release_evidence(
    repository: Path,
    evidence_path: Path,
) -> ReleaseEvidenceReportV1:
    """Verify stored evidence still closes over the current source tree."""
    report = load_release_evidence(repository, evidence_path)
    if not all(check.passed for check in report.data.checks):
        raise _invalid_evidence()
    try:
        verify_source_tree(repository, _report_identity(report))
    except ReleaseEvidenceError:
        raise _stale_evidence() from None
    return report


def _report_identity(report: ReleaseEvidenceReportV1) -> SourceTreeIdentity:
    return SourceTreeIdentity(
        verified_commit=report.data.verified_commit,
        source_tree_sha256=report.data.source_tree_sha256,
    )


def _validate_verified_on(value: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise _invalid_evidence() from None
    if parsed.isoformat() != value:
        raise _invalid_evidence()


def _invalid_evidence() -> ReleaseEvidenceError:
    return ReleaseEvidenceError(
        code="invalid_release_evidence",
        message="release evidence is invalid",
    )


def _stale_evidence() -> ReleaseEvidenceError:
    return ReleaseEvidenceError(
        code="stale_release_evidence",
        message="release evidence is stale",
    )
