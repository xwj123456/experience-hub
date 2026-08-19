"""Canonical, raw-text-free trajectory manifest hashing."""

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.models import (
    MAX_CAPTURE_EVIDENCE_ITEMS,
    MAX_CAPTURE_EXCERPT_UTF8_BYTES,
    TrajectoryBundleV1,
)


def extractor_configuration_document() -> dict[str, object]:
    return {
        "kind": "deterministic_signal_v1",
        "max_evidence_items": MAX_CAPTURE_EVIDENCE_ITEMS,
        "max_excerpt_utf8_bytes": MAX_CAPTURE_EXCERPT_UTF8_BYTES,
        "version": 1,
    }


def extractor_configuration_hash() -> str:
    return sha256_hex(canonical_json_bytes(extractor_configuration_document()))


def trajectory_manifest_document(
    bundle: TrajectoryBundleV1,
) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "adapter": bundle.adapter,
        "owner_agent_id": bundle.owner_agent_id,
        "trajectory_id": bundle.trajectory_id,
        "source_started_at": bundle.source_started_at,
        "source_completed_at": bundle.source_completed_at,
        "sanitization": bundle.sanitization,
        "steps": tuple(
            {
                "step_id": step.step_id,
                "ordinal": step.ordinal,
                "occurred_at": step.occurred_at,
                "status": step.status,
                "observation_hash": sha256_hex(step.observation.encode("utf-8")),
                "action_hash": sha256_hex(step.action.encode("utf-8")),
                "outcome_hash": sha256_hex(step.outcome.encode("utf-8")),
                "candidate_signal_hash": (
                    None
                    if step.candidate_signal is None
                    else sha256_hex(canonical_json_bytes(step.candidate_signal))
                ),
            }
            for step in bundle.steps
        ),
    }


def hash_trajectory_manifest(bundle: TrajectoryBundleV1) -> str:
    return sha256_hex(canonical_json_bytes(trajectory_manifest_document(bundle)))
