"""Deterministic candidate extraction from explicitly declared signals."""

from __future__ import annotations

from typing import Protocol

from experience_hub import sha256_hex
from experience_hub.capture.hashing import extractor_configuration_hash
from experience_hub.capture.models import (
    MAX_CAPTURE_EVIDENCE_ITEMS,
    MAX_CAPTURE_EXCERPT_UTF8_BYTES,
    CandidateDraftV1,
    CapturedEvidenceV1,
    TrajectoryBundleV1,
    TrajectoryField,
    TrajectoryStepV1,
)
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import VersionContent

_FIELD_ORDER = {
    TrajectoryField.OBSERVATION: 0,
    TrajectoryField.ACTION: 1,
    TrajectoryField.OUTCOME: 2,
}


class CandidateExtractor(Protocol):
    def extract(
        self,
        bundle: TrajectoryBundleV1,
    ) -> tuple[CandidateDraftV1, ...]: ...


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _source_field(step: TrajectoryStepV1, field: TrajectoryField) -> str:
    if field is TrajectoryField.OBSERVATION:
        return step.observation
    if field is TrajectoryField.ACTION:
        return step.action
    return step.outcome


class DeterministicSignalExtractor:
    def extract(
        self,
        bundle: TrajectoryBundleV1,
    ) -> tuple[CandidateDraftV1, ...]:
        steps_by_id = {step.step_id: step for step in bundle.steps}
        retained_hashes: set[str] = set()
        drafts: list[CandidateDraftV1] = []

        for signal_step in bundle.steps:
            signal = signal_step.candidate_signal
            if signal is None:
                continue

            resolved: dict[
                tuple[str, TrajectoryField],
                tuple[TrajectoryStepV1, TrajectoryField],
            ] = {}
            for pointer in signal.evidence:
                source_step = steps_by_id.get(pointer.step_id)
                if source_step is None:
                    raise ValueError("Evidence pointer names an unknown step")
                resolved[(pointer.step_id, pointer.field)] = (
                    source_step,
                    pointer.field,
                )

            canonical_pointers = sorted(
                resolved.values(),
                key=lambda item: (
                    item[0].ordinal,
                    _FIELD_ORDER[item[1]],
                    item[0].step_id,
                ),
            )[:MAX_CAPTURE_EVIDENCE_ITEMS]
            evidence = tuple(
                _captured_evidence(source_step, field)
                for source_step, field in canonical_pointers
            )
            content = VersionContent(
                body=signal.body,
                summary=signal.summary,
                mechanism=signal.mechanism,
                tags=signal.tags,
                applicability=signal.applicability,
                evidence=tuple(
                    TypedEvidence(
                        type="trajectory_field",
                        id=(
                            f"{bundle.manifest_hash}:{item.step_id}:"
                            f"{item.field.value}"
                        ),
                    )
                    for item in evidence
                ),
                falsifiers=signal.falsifiers,
            )
            content_hash = encode_version_content(
                kind=signal.kind,
                content=content,
            ).content_hash
            if content_hash in retained_hashes:
                continue
            retained_hashes.add(content_hash)
            drafts.append(
                CandidateDraftV1(
                    source_manifest_hash=bundle.manifest_hash,
                    kind=signal.kind,
                    content=content,
                    content_hash=content_hash,
                    evidence=evidence,
                    extractor_kind="deterministic_signal_v1",
                    extractor_configuration_hash=extractor_configuration_hash(),
                )
            )
        return tuple(drafts)


def _captured_evidence(
    step: TrajectoryStepV1,
    field: TrajectoryField,
) -> CapturedEvidenceV1:
    source = _source_field(step, field)
    excerpt = _truncate_utf8(source, MAX_CAPTURE_EXCERPT_UTF8_BYTES)
    return CapturedEvidenceV1(
        step_id=step.step_id,
        field=field,
        excerpt=excerpt,
        source_hash=sha256_hex(source.encode("utf-8")),
        excerpt_hash=sha256_hex(excerpt.encode("utf-8")),
    )
