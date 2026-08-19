"""Pure trajectory capture protocol and generic JSONL adapter."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from experience_hub.capture.extraction import (
    CandidateExtractor,
    DeterministicSignalExtractor,
)
from experience_hub.capture.hashing import (
    extractor_configuration_document,
    extractor_configuration_hash,
    hash_trajectory_manifest,
    trajectory_manifest_document,
)
from experience_hub.capture.jsonl import GenericJsonlAdapter
from experience_hub.capture.models import (
    AdapterDescriptorV1,
    CandidateDraftV1,
    CandidateSignalV1,
    CapturedEvidenceV1,
    EvidencePointerV1,
    OutcomeStatus,
    PreparedCaptureV1,
    SanitizationDeclarationV1,
    SensitiveField,
    SensitiveMatchV1,
    TrajectoryBundleV1,
    TrajectoryField,
    TrajectoryStepV1,
)
from experience_hub.capture.sanitization import DefaultSecretScanner, SecretScanner

if TYPE_CHECKING:
    from experience_hub.capture.service import CapturePreparer, CaptureService

_LAZY_EXPORT_MODULES = {
    "CapturePreparer": "experience_hub.capture.service",
    "CaptureService": "experience_hub.capture.service",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "AdapterDescriptorV1",
    "CandidateDraftV1",
    "CandidateExtractor",
    "CandidateSignalV1",
    "CapturePreparer",
    "CaptureService",
    "CapturedEvidenceV1",
    "DefaultSecretScanner",
    "DeterministicSignalExtractor",
    "EvidencePointerV1",
    "GenericJsonlAdapter",
    "OutcomeStatus",
    "PreparedCaptureV1",
    "SanitizationDeclarationV1",
    "SecretScanner",
    "SensitiveField",
    "SensitiveMatchV1",
    "TrajectoryBundleV1",
    "TrajectoryField",
    "TrajectoryStepV1",
    "extractor_configuration_document",
    "extractor_configuration_hash",
    "hash_trajectory_manifest",
    "trajectory_manifest_document",
]
