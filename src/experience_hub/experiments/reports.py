"""Canonical replay evidence, separate profiling, and artifact publication."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

from experience_hub.canonical import canonical_json_bytes
from experience_hub.experiments.contracts import (
    MAX_REPLAY_OUTPUT_BYTES,
    ArmEvidenceV1,
    CaseEvidenceV1,
    ReplayEvidenceReportV1,
    ReplayProfileReportV1,
)
from experience_hub.experiments.errors import ExperimentIsolationError
from experience_hub.experiments.workspace import OwnedWorkspace

_UUID_TEXT = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TIMESTAMP_TEXT = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:[Zz]|[+-]\d{2}:?\d{2})?"
)
_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?:/|[A-Za-z]:[\\/]|file://|sqlite(?:\+[A-Za-z0-9_]+)?://)"
)
_LOGICAL_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "database",
        "database_path",
        "path",
        "event_id",
        "receipt_id",
        "run_id",
        "created_at",
        "completed_at",
        "elapsed_ns",
        "wall_duration_ns",
        "exception",
        "traceback",
        "api_key",
        "token",
        "secret",
    }
)
_CREDENTIAL_EVIDENCE_KEYS = frozenset(
    {
        "access_token",
        "credential",
        "credentials",
        "password",
        "refresh_token",
    }
)
_FROZEN_AT_PATH = ("data", "resolved_manifest", "frozen_at")
_EVIDENCE_PATH = PurePosixPath("artifacts/evidence.json")
_PROFILE_PATH = PurePosixPath("artifacts/profile.json")


class ExperimentOutputError(ValueError):
    """A stable, private-detail-free replay output rejection."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class ReplayArtifactSet:
    """Exact artifact bodies and paths published for one replay."""

    evidence_path: Path
    profile_path: Path | None
    evidence_body: bytes
    profile_body: bytes | None


def _reject(code: str, message: str) -> ExperimentOutputError:
    return ExperimentOutputError(code, message)


def _is_credential_key(key: str) -> bool:
    return key in _CREDENTIAL_EVIDENCE_KEYS or key.endswith(
        ("_api_key", "_credential", "_credentials", "_password", "_secret", "_token")
    )


def _validate_evidence_value(
    value: object,
    *,
    path: tuple[str | int, ...],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _reject(
                    "unsafe_evidence",
                    "Replay evidence contains an unsupported object key",
                )
            if key in _FORBIDDEN_EVIDENCE_KEYS or _is_credential_key(key):
                raise _reject(
                    "unsafe_evidence",
                    "Replay evidence contains a forbidden field",
                )
            _validate_evidence_value(item, path=(*path, key))
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _validate_evidence_value(item, path=(*path, index))
        return
    if isinstance(value, BaseException):
        raise _reject(
            "unsafe_evidence",
            "Replay evidence contains an exception value",
        )
    if isinstance(value, UUID):
        raise _reject("unsafe_evidence", "Replay evidence contains a raw UUID")
    if isinstance(value, datetime):
        if path != _FROZEN_AT_PATH:
            raise _reject(
                "unsafe_evidence",
                "Replay evidence contains a runtime timestamp",
            )
        return
    if isinstance(value, str):
        if _UUID_TEXT.search(value):
            raise _reject("unsafe_evidence", "Replay evidence contains a raw UUID")
        if _TIMESTAMP_TEXT.search(value) and path != _FROZEN_AT_PATH:
            raise _reject(
                "unsafe_evidence",
                "Replay evidence contains a runtime timestamp",
            )
        if _ABSOLUTE_PATH_TEXT.match(value):
            raise _reject(
                "unsafe_evidence",
                "Replay evidence contains an absolute path",
            )
        return
    if value is None or isinstance(value, (bool, int)):
        return
    raise _reject(
        "unsafe_evidence",
        "Replay evidence contains an unsupported value",
    )


def _validate_arm(arm: ArmEvidenceV1) -> bool:
    if arm.status == "complete":
        if (
            arm.observation is None
            or arm.utility_micros is None
            or arm.error_code is not None
            or arm.error_stage is not None
        ):
            raise _reject(
                "invalid_evidence",
                "Complete replay arms require observation and utility only",
            )
        return True
    if (
        arm.observation is not None
        or arm.utility_micros is not None
        or arm.error_code is None
        or arm.error_stage is None
        or _LOGICAL_LABEL.fullmatch(arm.error_code) is None
        or _LOGICAL_LABEL.fullmatch(arm.error_stage) is None
    ):
        raise _reject(
            "invalid_evidence",
            "Failed replay arms require stable error labels only",
        )
    return False


def _validate_case(
    case: CaseEvidenceV1,
    *,
    expected_arm_ids: tuple[str, ...],
) -> bool:
    arm_ids = tuple(arm.arm_id for arm in case.arms)
    if arm_ids != expected_arm_ids:
        raise _reject(
            "invalid_evidence",
            "Replay case arms do not match the resolved manifest",
        )
    arms_complete = tuple(_validate_arm(arm) for arm in case.arms)
    complete = all(arms_complete)
    if (case.status == "complete") != complete:
        raise _reject(
            "invalid_evidence",
            "Replay case completeness does not match its arms",
        )
    if complete:
        no_memory, experience_hub = case.arms
        if (
            no_memory.utility_micros is None
            or experience_hub.utility_micros is None
        ):
            raise _reject(
                "invalid_evidence",
                "Complete replay arms require utility values",
            )
        expected_delta = (
            experience_hub.utility_micros - no_memory.utility_micros
        )
        if case.delta_utility_micros != expected_delta:
            raise _reject(
                "invalid_evidence",
                "Replay case delta does not match arm utilities",
            )
    elif case.delta_utility_micros is not None:
        raise _reject(
            "invalid_evidence",
            "Incomplete replay cases cannot contain a utility delta",
        )
    return complete


def _validate_evidence_assertions(report: ReplayEvidenceReportV1) -> None:
    data = report.data
    expected_arm_ids = tuple(
        arm.arm_id for arm in data.resolved_manifest.policy_arms if arm.required
    )
    if not data.cases:
        raise _reject(
            "invalid_evidence",
            "Replay evidence must contain at least one case",
        )
    case_completeness = tuple(
        _validate_case(case, expected_arm_ids=expected_arm_ids)
        for case in data.cases
    )
    cases_complete = all(case_completeness)
    if data.comparison_complete != cases_complete:
        raise _reject(
            "invalid_evidence",
            "Replay comparison completeness does not match its cases",
        )
    expected_valid = (
        data.comparison_complete
        and data.source_unchanged
        and data.clone_isolation_verified
        and data.deterministic_replay_match
    )
    if data.valid != expected_valid:
        raise _reject(
            "invalid_evidence",
            "Replay validity does not match its required assertions",
        )


def _validated_evidence(
    report: ReplayEvidenceReportV1,
) -> ReplayEvidenceReportV1:
    if not isinstance(report, ReplayEvidenceReportV1):
        raise _reject(
            "invalid_evidence",
            "Replay evidence does not match the versioned schema",
        )
    try:
        document = report.model_dump(mode="python", warnings=False)
        _validate_evidence_value(document, path=())
        validated = ReplayEvidenceReportV1.model_validate(document, strict=True)
        _validate_evidence_value(
            validated.model_dump(mode="python", warnings=False),
            path=(),
        )
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_evidence",
            "Replay evidence does not match the versioned schema",
        ) from None
    _validate_evidence_assertions(validated)
    return validated


def canonical_evidence_bytes(report: ReplayEvidenceReportV1) -> bytes:
    """Validate and encode evidence eligible for byte-identical comparison."""
    validated = _validated_evidence(report)
    try:
        body = canonical_json_bytes(validated)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_evidence",
            "Replay evidence cannot be encoded as canonical JSON",
        ) from None
    if len(body) > MAX_REPLAY_OUTPUT_BYTES:
        raise _reject(
            "output_too_large",
            "Replay evidence exceeds the output byte limit",
        )
    return body


def verify_evidence_bytes(body: bytes) -> ReplayEvidenceReportV1:
    """Verify bounded canonical evidence and all internally derived assertions.

    External manifest, case JSONL, and SQLite bytes are intentionally absent
    here. The replay runner must compare their hashes to the resolved bindings.
    """
    if not isinstance(body, bytes):
        raise _reject("invalid_evidence", "Replay evidence must be exact bytes")
    if len(body) > MAX_REPLAY_OUTPUT_BYTES:
        raise _reject(
            "output_too_large",
            "Replay evidence exceeds the output byte limit",
        )
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _reject(
            "invalid_json",
            "Replay evidence must be UTF-8 JSON",
        ) from None
    if not isinstance(decoded, dict):
        raise _reject("invalid_json", "Replay evidence must be a JSON object")
    try:
        if canonical_json_bytes(decoded) != body:
            raise _reject(
                "noncanonical_json",
                "Replay evidence must use canonical JSON",
            )
        _validate_evidence_value(decoded, path=())
        report = ReplayEvidenceReportV1.model_validate_json(body, strict=True)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_evidence",
            "Replay evidence does not match the versioned schema",
        ) from None
    validated = _validated_evidence(report)
    if canonical_evidence_bytes(validated) != body:
        raise _reject(
            "noncanonical_json",
            "Replay evidence must use its canonical model encoding",
        )
    return validated


def canonical_profile_bytes(report: ReplayProfileReportV1) -> bytes:
    """Strictly validate and deterministically encode non-evidence profiling."""
    if not isinstance(report, ReplayProfileReportV1):
        raise _reject(
            "invalid_profile",
            "Replay profile does not match the versioned schema",
        )
    try:
        validated = ReplayProfileReportV1.model_validate(
            report.model_dump(mode="python", warnings=False),
            strict=True,
        )
        body = canonical_json_bytes(validated)
    except ExperimentOutputError:
        raise
    except Exception:
        raise _reject(
            "invalid_profile",
            "Replay profile does not match the versioned schema",
        ) from None
    if len(body) > MAX_REPLAY_OUTPUT_BYTES:
        raise _reject(
            "output_too_large",
            "Replay profile exceeds the output byte limit",
        )
    return body


def write_replay_artifacts(
    workspace: OwnedWorkspace,
    *,
    evidence: ReplayEvidenceReportV1,
    profile: ReplayProfileReportV1 | None,
) -> ReplayArtifactSet:
    """Publish profile first and evidence last through workspace atomic writes.

    The two files are not a cross-file transaction. A profile failure preserves
    previous evidence; an evidence failure may leave a new profile beside it.
    """
    if not isinstance(workspace, OwnedWorkspace):
        raise _reject(
            "artifact_write_failed",
            "Replay artifacts require an owned workspace",
        )
    evidence_body = canonical_evidence_bytes(evidence)
    profile_body = (
        canonical_profile_bytes(profile) if profile is not None else None
    )
    if (
        profile is not None
        and profile.data.experiment_id
        != evidence.data.resolved_manifest.experiment_id
    ):
        raise _reject(
            "invalid_profile",
            "Replay profile experiment does not match its evidence",
        )
    try:
        profile_path = (
            workspace.atomic_write(_PROFILE_PATH, profile_body)
            if profile_body is not None
            else None
        )
        evidence_path = workspace.atomic_write(_EVIDENCE_PATH, evidence_body)
    except ExperimentIsolationError:
        raise _reject(
            "artifact_write_failed",
            "Replay artifact bytes could not be published",
        ) from None
    return ReplayArtifactSet(
        evidence_path=evidence_path,
        profile_path=profile_path,
        evidence_body=evidence_body,
        profile_body=profile_body,
    )


__all__ = [
    "ExperimentOutputError",
    "ReplayArtifactSet",
    "canonical_evidence_bytes",
    "canonical_profile_bytes",
    "verify_evidence_bytes",
    "write_replay_artifacts",
]
