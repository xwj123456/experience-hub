"""Shared reconstruction checks for persisted trajectory evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast
from uuid import UUID

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.models import (
    MAX_CAPTURE_EXCERPT_UTF8_BYTES,
    MAX_TRAJECTORY_STEPS,
    CapturedEvidenceV1,
    TrajectoryField,
)
from experience_hub.storage.tables import TrajectoryBundleRow, TrajectoryEvidenceRow

_MANIFEST_KEYS = {
    "adapter",
    "owner_agent_id",
    "sanitization",
    "schema_version",
    "source_completed_at",
    "source_started_at",
    "steps",
    "trajectory_id",
}
_STEP_KEYS = {
    "action_hash",
    "candidate_signal_hash",
    "observation_hash",
    "occurred_at",
    "ordinal",
    "outcome_hash",
    "status",
    "step_id",
}


@dataclass(frozen=True, slots=True)
class AuthenticatedTrajectoryStep:
    """One immutable, authenticated trajectory manifest step."""

    step_id: str
    ordinal: int
    action_hash: str
    observation_hash: str
    outcome_hash: str
    candidate_signal_hash: str | None
    occurred_at: str
    status: str

    def source_hash(self, field: TrajectoryField) -> str:
        if field is TrajectoryField.ACTION:
            return self.action_hash
        if field is TrajectoryField.OBSERVATION:
            return self.observation_hash
        return self.outcome_hash


@dataclass(frozen=True, slots=True)
class _TrajectoryManifestBinding:
    bundle_id: UUID
    owner_agent_id: UUID
    trajectory_id: str
    adapter_kind: str
    adapter_version: int
    sanitization_profile: str
    manifest: bytes
    manifest_hash: str
    source_started_at: datetime
    source_completed_at: datetime

    @classmethod
    def from_bundle(cls, bundle: TrajectoryBundleRow) -> _TrajectoryManifestBinding:
        return cls(
            bundle_id=bundle.bundle_id,
            owner_agent_id=bundle.owner_agent_id,
            trajectory_id=bundle.trajectory_id,
            adapter_kind=bundle.adapter_kind,
            adapter_version=bundle.adapter_version,
            sanitization_profile=bundle.sanitization_profile,
            manifest=bundle.manifest,
            manifest_hash=bundle.manifest_hash,
            source_started_at=bundle.source_started_at,
            source_completed_at=bundle.source_completed_at,
        )

    def matches(self, bundle: TrajectoryBundleRow) -> bool:
        return (
            bundle.bundle_id == self.bundle_id
            and bundle.owner_agent_id == self.owner_agent_id
            and bundle.trajectory_id == self.trajectory_id
            and bundle.adapter_kind == self.adapter_kind
            and bundle.adapter_version == self.adapter_version
            and bundle.sanitization_profile == self.sanitization_profile
            and bundle.manifest == self.manifest
            and bundle.manifest_hash == self.manifest_hash
            and bundle.source_started_at == self.source_started_at
            and bundle.source_completed_at == self.source_completed_at
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedTrajectoryManifest:
    """Immutable authenticated manifest with ordered and indexed steps."""

    steps: tuple[AuthenticatedTrajectoryStep, ...]
    _binding: _TrajectoryManifestBinding
    step_index: Mapping[
        tuple[str, int],
        AuthenticatedTrajectoryStep,
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            MappingProxyType(
                {(step.step_id, step.ordinal): step for step in self.steps}
            ),
        )

    def require_step(
        self,
        *,
        step_id: str,
        ordinal: int,
    ) -> AuthenticatedTrajectoryStep:
        try:
            return self.step_index[(step_id, ordinal)]
        except KeyError:
            raise ValueError(
                "trajectory evidence step is not in the manifest"
            ) from None

    def require_bundle(self, bundle: TrajectoryBundleRow) -> None:
        if not self._binding.matches(bundle):
            raise ValueError(
                "authenticated trajectory manifest does not belong to bundle"
            )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def authenticate_trajectory_manifest(
    row: TrajectoryBundleRow,
) -> AuthenticatedTrajectoryManifest:
    """Parse and authenticate a persisted manifest with exact JSON wire types."""

    raw_manifest = row.manifest
    if not isinstance(raw_manifest, bytes):
        raise ValueError("trajectory manifest is invalid")
    loaded: object = json.loads(raw_manifest)
    if not isinstance(loaded, dict) or any(
        not isinstance(key, str) for key in loaded
    ):
        raise ValueError("trajectory manifest is invalid")
    document = cast(dict[str, object], loaded)
    adapter = document.get("adapter")
    sanitization = document.get("sanitization")
    raw_steps = document.get("steps")
    if (
        set(document) != _MANIFEST_KEYS
        or canonical_json_bytes(document) != raw_manifest
        or not _is_sha256(row.manifest_hash)
        or sha256_hex(raw_manifest) != row.manifest_hash
        or type(row.adapter_version) is not int
        or not isinstance(adapter, dict)
        or set(adapter) != {"kind", "version"}
        or not isinstance(adapter["kind"], str)
        or adapter["kind"] != row.adapter_kind
        or type(adapter["version"]) is not int
        or adapter["version"] != row.adapter_version
        or not isinstance(document["owner_agent_id"], str)
        or document["owner_agent_id"] != str(row.owner_agent_id)
        or not isinstance(sanitization, dict)
        or set(sanitization) != {"input_sanitized", "profile_id"}
        or sanitization["input_sanitized"] is not True
        or not isinstance(sanitization["profile_id"], str)
        or sanitization["profile_id"] != row.sanitization_profile
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["source_started_at"], str)
        or document["source_started_at"] != _timestamp(row.source_started_at)
        or not isinstance(document["source_completed_at"], str)
        or document["source_completed_at"] != _timestamp(row.source_completed_at)
        or not isinstance(document["trajectory_id"], str)
        or document["trajectory_id"] != row.trajectory_id
        or not isinstance(raw_steps, list)
        or not 1 <= len(raw_steps) <= MAX_TRAJECTORY_STEPS
    ):
        raise ValueError("trajectory manifest is invalid")

    steps: list[AuthenticatedTrajectoryStep] = []
    step_ids: set[str] = set()
    previous_timestamp = row.source_started_at
    for expected_ordinal, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict) or any(
            not isinstance(key, str) for key in raw_step
        ):
            raise ValueError("trajectory step is invalid")
        step = cast(dict[str, object], raw_step)
        step_id = step.get("step_id")
        status = step.get("status")
        occurred_at = step.get("occurred_at")
        if (
            set(step) != _STEP_KEYS
            or type(step["ordinal"]) is not int
            or step["ordinal"] != expected_ordinal
            or not isinstance(step_id, str)
            or not step_id.strip()
            or step_id in step_ids
            or not isinstance(status, str)
            or status not in {"succeeded", "failed", "unknown"}
            or not isinstance(occurred_at, str)
        ):
            raise ValueError("trajectory step is invalid")
        for trajectory_field in TrajectoryField:
            if not _is_sha256(step[f"{trajectory_field.value}_hash"]):
                raise ValueError("trajectory step hash is invalid")
        signal_hash = step["candidate_signal_hash"]
        if signal_hash is not None and not _is_sha256(signal_hash):
            raise ValueError("candidate signal hash is invalid")
        try:
            parsed_timestamp = datetime.fromisoformat(
                occurred_at.removesuffix("Z") + "+00:00"
            )
        except ValueError:
            raise ValueError("trajectory step timestamp is invalid") from None
        if (
            parsed_timestamp.utcoffset() is None
            or _timestamp(parsed_timestamp) != occurred_at
            or not row.source_started_at
            <= parsed_timestamp
            <= row.source_completed_at
            or parsed_timestamp < previous_timestamp
        ):
            raise ValueError("trajectory step timestamp is invalid")
        previous_timestamp = parsed_timestamp
        step_ids.add(step_id)
        steps.append(
            AuthenticatedTrajectoryStep(
                step_id=step_id,
                ordinal=expected_ordinal,
                action_hash=cast(str, step["action_hash"]),
                observation_hash=cast(str, step["observation_hash"]),
                outcome_hash=cast(str, step["outcome_hash"]),
                candidate_signal_hash=cast(
                    str | None,
                    step["candidate_signal_hash"],
                ),
                occurred_at=occurred_at,
                status=status,
            )
        )
    ordered_steps = tuple(steps)
    return AuthenticatedTrajectoryManifest(
        steps=ordered_steps,
        _binding=_TrajectoryManifestBinding.from_bundle(row),
    )


def reconstruct_captured_evidence(
    *,
    bundle: TrajectoryBundleRow,
    rows: tuple[TrajectoryEvidenceRow, ...],
    evidence_ids: tuple[UUID, ...],
    owner_agent_id: UUID,
    manifest: AuthenticatedTrajectoryManifest | None = None,
) -> tuple[CapturedEvidenceV1, ...]:
    """Authenticate evidence rows and reconstruct them in reference order."""

    authenticated_manifest = (
        authenticate_trajectory_manifest(bundle) if manifest is None else manifest
    )
    authenticated_manifest.require_bundle(bundle)
    if bundle.owner_agent_id != owner_agent_id:
        raise ValueError("trajectory bundle owner is inconsistent")
    by_id = {row.evidence_id: row for row in rows}
    if (
        len(by_id) != len(rows)
        or len(evidence_ids) != len(set(evidence_ids))
        or set(by_id) != set(evidence_ids)
    ):
        raise ValueError("trajectory evidence rows are incomplete")

    result: list[CapturedEvidenceV1] = []
    for evidence_id in evidence_ids:
        row = by_id[evidence_id]
        if (
            row.owner_agent_id != owner_agent_id
            or row.owner_agent_id != bundle.owner_agent_id
            or row.bundle_id != bundle.bundle_id
            or not isinstance(row.step_id, str)
            or type(row.ordinal) is not int
            or row.ordinal <= 0
            or not isinstance(row.field, str)
            or not isinstance(row.excerpt, str)
            or not _is_sha256(row.source_hash)
            or not _is_sha256(row.excerpt_hash)
        ):
            raise ValueError("trajectory evidence identity is inconsistent")
        field = TrajectoryField(row.field)
        step = authenticated_manifest.require_step(
            step_id=row.step_id,
            ordinal=row.ordinal,
        )
        excerpt_bytes = row.excerpt.encode("utf-8")
        if (
            len(excerpt_bytes) > MAX_CAPTURE_EXCERPT_UTF8_BYTES
            or step.source_hash(field) != row.source_hash
            or sha256_hex(excerpt_bytes) != row.excerpt_hash
        ):
            raise ValueError("trajectory evidence authentication failed")
        result.append(
            CapturedEvidenceV1(
                step_id=row.step_id,
                field=field,
                excerpt=row.excerpt,
                source_hash=row.source_hash,
                excerpt_hash=row.excerpt_hash,
            )
        )
    return tuple(result)


__all__ = [
    "AuthenticatedTrajectoryManifest",
    "AuthenticatedTrajectoryStep",
    "authenticate_trajectory_manifest",
    "reconstruct_captured_evidence",
]
