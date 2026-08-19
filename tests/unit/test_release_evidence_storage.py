"""Tests for bounded canonical release-evidence storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from experience_hub.canonical import canonical_json_bytes
from experience_hub.release_evidence import storage
from experience_hub.release_evidence.contracts import CheckName, ReleaseEvidenceReportV1
from experience_hub.release_evidence.errors import ReleaseEvidenceError


def _valid_evidence() -> ReleaseEvidenceReportV1:
    return ReleaseEvidenceReportV1.model_validate(
        {
            "data": {
                "schema_version": 1,
                "verified_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "verified_on": "2026-08-03",
                "python_version": "3.12",
                "checks": [
                    {"name": name, "passed": True} for name in CheckName
                ],
                "test_count": 2670,
                "demo": {"all_invariants_hold": True, "stage_count": 11},
                "benchmark": {
                    "passed": True,
                    "case_count": 15,
                    "gate_count": 11,
                    "passed_gate_count": 11,
                    "byte_identical_replay": True,
                    "pending_capsule_leakage_count": 0,
                },
            }
        },
        strict=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    return repository


def _evidence_path(repository: Path) -> Path:
    return repository / "docs" / "evidence" / "release-evidence.json"


def _assert_invalid(operation: object) -> ReleaseEvidenceError:
    assert callable(operation)
    with pytest.raises(ReleaseEvidenceError) as raised:
        operation()
    assert raised.value.code == "invalid_release_evidence"
    return raised.value


def test_store_writes_and_load_reads_exact_canonical_document(
    repository: Path,
) -> None:
    evidence = _valid_evidence()
    evidence_path = _evidence_path(repository)

    storage.store_release_evidence(repository, evidence_path, evidence)

    assert evidence_path.read_bytes() == canonical_json_bytes(evidence)
    assert storage.load_release_evidence(repository, evidence_path) == evidence


def test_load_rejects_document_larger_than_128_kib(repository: Path) -> None:
    evidence_path = _evidence_path(repository)
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(b" " * (storage.MAX_RELEASE_EVIDENCE_BYTES + 1))

    error = _assert_invalid(
        lambda: storage.load_release_evidence(repository, evidence_path)
    )

    assert str(evidence_path) not in error.message


@pytest.mark.parametrize("kind", ("directory", "symlink"))
def test_load_rejects_non_regular_evidence_path(
    repository: Path,
    kind: str,
) -> None:
    evidence_path = _evidence_path(repository)
    evidence_path.parent.mkdir(parents=True)
    if kind == "directory":
        evidence_path.mkdir()
    else:
        target = repository / "private-evidence.json"
        target.write_bytes(canonical_json_bytes(_valid_evidence()))
        evidence_path.symlink_to(target)

    _assert_invalid(lambda: storage.load_release_evidence(repository, evidence_path))


@pytest.mark.parametrize("body", (b"{", canonical_json_bytes({})))
def test_load_rejects_malformed_or_wrong_schema_document(
    repository: Path,
    body: bytes,
) -> None:
    evidence_path = _evidence_path(repository)
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(body)

    _assert_invalid(lambda: storage.load_release_evidence(repository, evidence_path))


def test_load_rejects_noncanonical_document(repository: Path) -> None:
    evidence_path = _evidence_path(repository)
    evidence_path.parent.mkdir(parents=True)
    body = canonical_json_bytes(_valid_evidence()).replace(b":", b": ", 1)
    evidence_path.write_bytes(body)

    _assert_invalid(lambda: storage.load_release_evidence(repository, evidence_path))


def test_store_rejects_destination_outside_docs_evidence(repository: Path) -> None:
    outside = repository / "private-evidence.json"

    _assert_invalid(
        lambda: storage.store_release_evidence(repository, outside, _valid_evidence())
    )

    assert not outside.exists()


@pytest.mark.parametrize(
    "evidence_path",
    (
        "docs/evidence/release-evidence-copy.json",
        "docs/evidence/archive/release-evidence.json",
    ),
)
def test_store_rejects_noncanonical_evidence_paths(
    repository: Path,
    evidence_path: str,
) -> None:
    _assert_invalid(
        lambda: storage.store_release_evidence(
            repository,
            repository / evidence_path,
            _valid_evidence(),
        )
    )


@pytest.mark.parametrize(
    "evidence_path",
    (
        "docs/evidence/release-evidence-copy.json",
        "docs/evidence/archive/release-evidence.json",
    ),
)
def test_load_rejects_noncanonical_evidence_paths(
    repository: Path,
    evidence_path: str,
) -> None:
    candidate = repository / evidence_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(canonical_json_bytes(_valid_evidence()))

    _assert_invalid(lambda: storage.load_release_evidence(repository, candidate))


def test_store_rejects_a_symlink_destination(repository: Path) -> None:
    evidence_path = _evidence_path(repository)
    evidence_path.parent.mkdir(parents=True)
    target = repository / "private-evidence.json"
    evidence_path.symlink_to(target)

    _assert_invalid(
        lambda: storage.store_release_evidence(
            repository,
            evidence_path,
            _valid_evidence(),
        )
    )

    assert evidence_path.is_symlink()
    assert not target.exists()


def test_failed_atomic_replacement_keeps_existing_valid_document(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
) -> None:
    evidence_path = _evidence_path(repository)
    existing = _valid_evidence()
    storage.store_release_evidence(repository, evidence_path, existing)
    before = evidence_path.read_bytes()

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("private replacement failure")

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    _assert_invalid(
        lambda: storage.store_release_evidence(repository, evidence_path, existing)
    )

    assert evidence_path.read_bytes() == before
