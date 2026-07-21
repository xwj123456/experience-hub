"""Public experience values and transaction-bound service contracts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from experience_hub.experiences.content import (
    decode_payload,
    decode_version_content,
    encode_version_content,
    preferred_payload_codec,
    reencode_payload,
)
from experience_hub.experiences.models import (
    EncodedVersionContent,
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    PayloadCodec,
    Temperature,
    VersionContent,
)

if TYPE_CHECKING:
    from experience_hub.experiences.contracts import (
        ConfirmExperience,
        CreateExperience,
        CreateExperienceVersion,
        ExperienceCreation,
        ExperienceDraft,
        ExperienceMutationReason,
        ExperienceRecord,
        PinExperience,
        RefuteExperience,
        RestoreExperience,
        ShareableExperienceVersion,
        UnpinExperience,
        VersionLinkInput,
        canonicalize_version_links,
    )
    from experience_hub.experiences.events import ExperienceStateSnapshotV1
    from experience_hub.experiences.queries import (
        ExperienceNotFoundError,
        ExperienceQuery,
    )
    from experience_hub.experiences.repository import (
        ExperienceRepository,
        ExperienceWriter,
    )
    from experience_hub.experiences.service import (
        ExperienceRetrievalAdapter,
        ExperienceService,
    )
    from experience_hub.experiences.transitions import ExperienceMutationWriter

_LAZY_CONTRACT_EXPORTS = frozenset(
    {
        "ConfirmExperience",
        "CreateExperience",
        "CreateExperienceVersion",
        "ExperienceCreation",
        "ExperienceDraft",
        "ExperienceMutationReason",
        "ExperienceRecord",
        "PinExperience",
        "RefuteExperience",
        "RestoreExperience",
        "ShareableExperienceVersion",
        "UnpinExperience",
        "VersionLinkInput",
        "canonicalize_version_links",
    }
)
_LAZY_EVENT_EXPORTS = frozenset({"ExperienceStateSnapshotV1"})
_LAZY_QUERY_EXPORTS = frozenset(
    {"ExperienceNotFoundError", "ExperienceQuery"}
)
_LAZY_REPOSITORY_EXPORTS = frozenset(
    {"ExperienceRepository", "ExperienceWriter"}
)
_LAZY_SERVICE_EXPORTS = frozenset(
    {"ExperienceRetrievalAdapter", "ExperienceService"}
)
_LAZY_TRANSITION_EXPORTS = frozenset({"ExperienceMutationWriter"})
_LAZY_EXPORT_MODULES = {
    **{
        name: "experience_hub.experiences.contracts"
        for name in _LAZY_CONTRACT_EXPORTS
    },
    **{
        name: "experience_hub.experiences.events"
        for name in _LAZY_EVENT_EXPORTS
    },
    **{
        name: "experience_hub.experiences.queries"
        for name in _LAZY_QUERY_EXPORTS
    },
    **{
        name: "experience_hub.experiences.repository"
        for name in _LAZY_REPOSITORY_EXPORTS
    },
    **{
        name: "experience_hub.experiences.service"
        for name in _LAZY_SERVICE_EXPORTS
    },
    **{
        name: "experience_hub.experiences.transitions"
        for name in _LAZY_TRANSITION_EXPORTS
    },
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ConfirmExperience",
    "CreateExperience",
    "CreateExperienceVersion",
    "EncodedVersionContent",
    "ExperienceCreation",
    "ExperienceDraft",
    "ExperienceKind",
    "ExperienceMutationReason",
    "ExperienceMutationWriter",
    "ExperienceNotFoundError",
    "ExperienceOrigin",
    "ExperienceQuery",
    "ExperienceRecord",
    "ExperienceRepository",
    "ExperienceRetrievalAdapter",
    "ExperienceService",
    "ExperienceStateSnapshotV1",
    "ExperienceWriter",
    "LinkRelation",
    "PayloadCodec",
    "PinExperience",
    "RefuteExperience",
    "RestoreExperience",
    "ShareableExperienceVersion",
    "Temperature",
    "UnpinExperience",
    "VersionContent",
    "VersionLinkInput",
    "canonicalize_version_links",
    "decode_payload",
    "decode_version_content",
    "encode_version_content",
    "preferred_payload_codec",
    "reencode_payload",
]
