"""Bounded canonical storage for generated release evidence."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import CanonicalizationError
from experience_hub.release_evidence.contracts import ReleaseEvidenceReportV1
from experience_hub.release_evidence.errors import ReleaseEvidenceError

MAX_RELEASE_EVIDENCE_BYTES = 128 * 1024


def load_release_evidence(
    repository: Path,
    evidence_path: Path,
) -> ReleaseEvidenceReportV1:
    """Load one bounded, exact canonical release evidence document."""
    path = _evidence_path(repository, evidence_path, create_directory=False)
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError
        if metadata.st_size > MAX_RELEASE_EVIDENCE_BYTES:
            raise OSError
        with path.open("rb") as evidence_file:
            body = evidence_file.read(MAX_RELEASE_EVIDENCE_BYTES + 1)
    except OSError:
        raise _invalid_evidence() from None
    if len(body) > MAX_RELEASE_EVIDENCE_BYTES:
        raise _invalid_evidence()
    try:
        report = ReleaseEvidenceReportV1.model_validate_json(body, strict=True)
        if canonical_json_bytes(report) != body:
            raise ValueError
    except (CanonicalizationError, ValidationError, ValueError, UnicodeDecodeError):
        raise _invalid_evidence() from None
    return report


def store_release_evidence(
    repository: Path,
    evidence_path: Path,
    report: ReleaseEvidenceReportV1,
) -> None:
    """Atomically replace one repository-owned canonical evidence document."""
    path = _evidence_path(repository, evidence_path, create_directory=True)
    try:
        body = canonical_json_bytes(report)
    except CanonicalizationError:
        raise _invalid_evidence() from None
    if len(body) > MAX_RELEASE_EVIDENCE_BYTES:
        raise _invalid_evidence()
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        raise _invalid_evidence() from None
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise _invalid_evidence()

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".release-evidence-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as evidence_file:
            evidence_file.write(body)
            evidence_file.flush()
            os.fsync(evidence_file.fileno())
        if not stat.S_ISREG(temporary_path.lstat().st_mode):
            raise OSError
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except OSError:
        raise _invalid_evidence() from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _evidence_path(
    repository: Path,
    evidence_path: Path,
    *,
    create_directory: bool,
) -> Path:
    try:
        root = repository.resolve(strict=True)
        if not root.is_dir():
            raise OSError
        evidence_directory = root / "docs" / "evidence"
        candidate = (
            evidence_path if evidence_path.is_absolute() else root / evidence_path
        )
        path = Path(os.path.abspath(candidate))
        if path != evidence_directory / "release-evidence.json":
            raise ValueError
        _require_directory(root / "docs", create=create_directory)
        _require_directory(evidence_directory, create=create_directory)
    except (OSError, ValueError):
        raise _invalid_evidence() from None
    return path


def _require_directory(path: Path, *, create: bool) -> None:
    if not path.exists():
        if not create:
            raise OSError
        path.mkdir(parents=True, exist_ok=False)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _invalid_evidence() -> ReleaseEvidenceError:
    return ReleaseEvidenceError(
        code="invalid_release_evidence",
        message="release evidence is invalid",
    )
