"""Offline single-version Passport values and verification."""

from typing import TYPE_CHECKING

from experience_hub.capture.sanitization import TextSensitiveMatchV1
from experience_hub.passports.codec import (
    VerifiedPassportV1,
    build_passport_document,
    compute_passport_hash,
    encode_passport_document,
    verify_passport_bytes,
)
from experience_hub.passports.contracts import (
    AdoptPassport,
    EmbeddedExcerptSnapshotV1,
    EvidencePassportV1,
    EvidenceSnapshotV1,
    ImportPassport,
    PassportAdoptionResultV1,
    PassportAdoptionV1,
    PassportDeclarationV1,
    PassportHopV1,
    PassportImportViewV1,
    PassportInspectionV1,
    PassportPageV1,
    PassportProvenanceV1,
    PassportState,
    PassportSubjectV1,
    ReferenceOnlySnapshotV1,
    RejectPassport,
)

if TYPE_CHECKING:
    from experience_hub.passports.export import PassportExportService
    from experience_hub.passports.queries import PassportQuery
    from experience_hub.passports.service import PassportService

__all__ = [
    "AdoptPassport",
    "RejectPassport",
    "PassportAdoptionV1",
    "PassportAdoptionResultV1",
    "EmbeddedExcerptSnapshotV1",
    "EvidencePassportV1",
    "EvidenceSnapshotV1",
    "ImportPassport",
    "PassportDeclarationV1",
    "PassportHopV1",
    "PassportInspectionV1",
    "PassportImportViewV1",
    "PassportPageV1",
    "PassportQuery",
    "PassportService",
    "PassportExportService",
    "PassportProvenanceV1",
    "PassportState",
    "PassportSubjectV1",
    "ReferenceOnlySnapshotV1",
    "TextSensitiveMatchV1",
    "VerifiedPassportV1",
    "build_passport_document",
    "compute_passport_hash",
    "encode_passport_document",
    "verify_passport_bytes",
]


def __getattr__(name: str) -> object:
    # Tables import transport contracts; lazy application exports avoid a cycle.
    if name == "PassportService":
        from experience_hub.passports.service import PassportService

        return PassportService
    if name == "PassportQuery":
        from experience_hub.passports.queries import PassportQuery

        return PassportQuery
    if name == "PassportExportService":
        from experience_hub.passports.export import PassportExportService

        return PassportExportService
    raise AttributeError(name)
