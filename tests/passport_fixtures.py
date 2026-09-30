"""Synthetic Passport values shared by protocol and persistence tests."""

from uuid import UUID

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.models import TrajectoryField
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    VersionContent,
)
from experience_hub.passports import (
    EmbeddedExcerptSnapshotV1,
    EvidencePassportV1,
    PassportDeclarationV1,
    PassportHopV1,
    PassportProvenanceV1,
    PassportSubjectV1,
    build_passport_document,
)

SOURCE_AGENT = UUID("20000000-0000-4000-8000-000000000001")
SOURCE_EXPERIENCE = UUID("20000000-0000-4000-8000-000000000002")
SOURCE_VERSION = UUID("20000000-0000-4000-8000-000000000003")
MANIFEST_HASH = sha256_hex(b"synthetic manifest")


def passport_document(*, tag: str = "recovery") -> EvidencePassportV1:
    reference = TypedEvidence(
        type="trajectory_field", id=f"{MANIFEST_HASH}:step-1:outcome"
    )
    content = VersionContent(
        body="Retry only after verifying that the local operation rolled back.",
        summary="Verify rollback before retry.",
        mechanism="An atomic transaction prevents duplicate local effects.",
        tags=(tag,),
        applicability=("local SQLite commands",),
        evidence=(reference,),
        falsifiers=("A source survives a rolled-back command.",),
    )
    content_hash = encode_version_content(
        kind=ExperienceKind.PROCEDURAL, content=content
    ).content_hash
    subject = PassportSubjectV1(
        source_agent_id=SOURCE_AGENT,
        source_experience_id=SOURCE_EXPERIENCE,
        source_version_id=SOURCE_VERSION,
        source_origin=ExperienceOrigin.LOCAL,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        content_hash=content_hash,
    )
    excerpt = "The failed command left no committed source."
    return build_passport_document(
        subject=subject,
        evidence_snapshots=(
            EmbeddedExcerptSnapshotV1(
                mode="embedded_excerpt",
                reference=reference,
                excerpt=excerpt,
                excerpt_hash=sha256_hex(excerpt.encode("utf-8")),
                source_hash=sha256_hex(b"synthetic full outcome"),
                source_manifest_hash=MANIFEST_HASH,
                step_id="step-1",
                field=TrajectoryField.OUTCOME,
            ),
        ),
        provenance=PassportProvenanceV1(
            scope="passport_transfers_only",
            hops=(
                PassportHopV1(
                    source_agent_id=SOURCE_AGENT,
                    source_experience_id=SOURCE_EXPERIENCE,
                    source_version_id=SOURCE_VERSION,
                    source_origin=ExperienceOrigin.LOCAL,
                    content_hash=content_hash,
                    parent_passport_hash=None,
                ),
            ),
            origin_fingerprint=sha256_hex(
                canonical_json_bytes(
                    {
                        "source_agent_id": SOURCE_AGENT,
                        "content_hash": content_hash,
                    }
                )
            ),
        ),
        declaration=PassportDeclarationV1(
            input_sanitized=True,
            profile_id="synthetic-v1",
            sharing_authorized=True,
        ),
    )
