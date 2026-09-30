"""Deterministic export of one owned immutable version and retained evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.experiences.queries import ExperienceNotFoundError, ExperienceQuery
from experience_hub.passports.codec import (
    VerifiedPassportV1,
    build_passport_document,
    encode_passport_document,
    verify_passport_bytes,
)
from experience_hub.passports.contracts import (
    MAX_PASSPORT_HOPS,
    PassportDeclarationV1,
    PassportHopV1,
    PassportProvenanceV1,
    PassportSubjectV1,
)
from experience_hub.passports.errors import PassportError
from experience_hub.storage.validation import SourceIntegrityError

if TYPE_CHECKING:
    from experience_hub.experiences.evidence_snapshots import (
        ExperienceEvidenceSnapshotReader,
    )
    from experience_hub.passports.queries import PassportQuery


class PassportExportService:
    """Read only caller-authorized sources; never initialize or mutate a database."""

    def __init__(
        self,
        *,
        experience_query: ExperienceQuery | None = None,
        evidence_reader: ExperienceEvidenceSnapshotReader | None = None,
        passport_query: PassportQuery | None = None,
    ) -> None:
        from experience_hub.experiences.evidence_snapshots import (
            ExperienceEvidenceSnapshotReader,
        )
        from experience_hub.passports.queries import PassportQuery
        from experience_hub.passports.repository import PassportRepository

        self._experiences = experience_query or ExperienceQuery()
        self._evidence = evidence_reader or ExperienceEvidenceSnapshotReader()
        self._passports = passport_query or PassportQuery(
            repository=PassportRepository()
        )

    async def export(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
        version_id: UUID | None,
        declaration: PassportDeclarationV1,
        parent_adoption_id: UUID | None = None,
    ) -> VerifiedPassportV1:
        if (
            not isinstance(owner_agent_id, UUID)
            or not isinstance(experience_id, UUID)
            or (version_id is not None and not isinstance(version_id, UUID))
            or (
                parent_adoption_id is not None
                and not isinstance(parent_adoption_id, UUID)
            )
            or not isinstance(declaration, PassportDeclarationV1)
        ):
            raise PassportError("invalid")
        try:
            declaration = PassportDeclarationV1.model_validate(declaration)
            selected = await self._experiences.get_owned_shareable_version(
                session=session,
                owner_agent_id=owner_agent_id,
                experience_id=experience_id,
                version_id=version_id,
            )
        except ExperienceNotFoundError:
            raise PassportError("not_found") from None
        except ValidationError:
            raise PassportError("invalid") from None
        except SourceIntegrityError:
            raise SourceIntegrityError(
                "Owned Passport export source is invalid",
                mismatch_key="passport_export",
            ) from None
        if selected.temperature == Temperature.ARCHIVED:
            raise PassportError("restore_required")
        subject = PassportSubjectV1(
            source_agent_id=selected.owner_agent_id,
            source_experience_id=selected.experience_id,
            source_version_id=selected.version_id,
            source_origin=selected.origin,
            kind=selected.kind,
            content=selected.content,
            content_hash=selected.content_hash,
        )
        hop = PassportHopV1(
            source_agent_id=selected.owner_agent_id,
            source_experience_id=selected.experience_id,
            source_version_id=selected.version_id,
            source_origin=selected.origin,
            content_hash=selected.content_hash,
            parent_passport_hash=None,
        )
        if parent_adoption_id is None:
            if selected.origin == ExperienceOrigin.ADOPTED_PASSPORT:
                raise PassportError("derivation_unsupported")
            snapshots = await self._evidence.read(session=session, version=selected)
            provenance = PassportProvenanceV1(
                scope="passport_transfers_only",
                hops=(hop,),
                origin_fingerprint=sha256_hex(
                    canonical_json_bytes(
                        {
                            "source_agent_id": selected.owner_agent_id,
                            "content_hash": selected.content_hash,
                        }
                    )
                ),
            )
        else:
            adoption = await self._passports.get_adoption(
                session=session,
                owner_agent_id=owner_agent_id,
                adoption_id=parent_adoption_id,
            )
            parent = adoption.prepared.document
            if (
                adoption.resulting_experience_id != selected.experience_id
                or adoption.resulting_content_hash != selected.content_hash
                or parent.subject.content_hash != selected.content_hash
            ):
                raise PassportError("derivation_unsupported")
            previous = parent.provenance.hops
            identity = (
                selected.owner_agent_id,
                selected.experience_id,
                selected.version_id,
            )
            if len(previous) >= MAX_PASSPORT_HOPS or any(
                (
                    item.source_agent_id,
                    item.source_experience_id,
                    item.source_version_id,
                )
                == identity
                for item in previous
            ):
                raise PassportError("provenance_limit")
            snapshots = parent.evidence_snapshots
            provenance = PassportProvenanceV1(
                scope="passport_transfers_only",
                hops=(
                    *previous,
                    hop.model_copy(
                        update={"parent_passport_hash": parent.passport_hash}
                    ),
                ),
                origin_fingerprint=parent.provenance.origin_fingerprint,
            )
        document = build_passport_document(
            subject=subject,
            evidence_snapshots=snapshots,
            provenance=provenance,
            declaration=declaration,
        )
        return verify_passport_bytes(encode_passport_document(document))
