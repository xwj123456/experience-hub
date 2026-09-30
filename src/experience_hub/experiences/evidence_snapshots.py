"""Owner-scoped evidence snapshots for an immutable experience version."""

from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import canonical_json_bytes
from experience_hub.domain import ReplayableCommandError, TypedEvidence
from experience_hub.experiences.candidate_models import CandidateDecision
from experience_hub.experiences.candidate_repository import CandidateRepository
from experience_hub.experiences.candidate_service import CandidateService
from experience_hub.experiences.contracts import ShareableExperienceVersion
from experience_hub.passports.contracts import (
    EmbeddedExcerptSnapshotV1,
    EvidenceSnapshotV1,
    ReferenceOnlySnapshotV1,
)
from experience_hub.storage.tables import (
    CandidateAdoptionRow,
    ExperienceRow,
    ExperienceVersionRow,
)
from experience_hub.storage.validation import SourceIntegrityError


class ExperienceEvidenceSnapshotReader:
    """Resolve retained capture excerpts only through an owned adoption anchor."""

    async def read(
        self,
        *,
        session: AsyncSession,
        version: ShareableExperienceVersion,
    ) -> tuple[EvidenceSnapshotV1, ...]:
        try:
            # A read must not flush unrelated pending objects in its caller's
            # session. Export normally supplies a physically read-only session.
            with session.no_autoflush:
                return await self._read(session=session, version=version)
        except (
            LookupError,
            ReplayableCommandError,
            SourceIntegrityError,
            StatementError,
            TypeError,
            ValueError,
        ):
            raise SourceIntegrityError(
                "Owned capture evidence is invalid",
                mismatch_key="passport_evidence",
            ) from None

    async def _read(
        self,
        *,
        session: AsyncSession,
        version: ShareableExperienceVersion,
    ) -> tuple[EvidenceSnapshotV1, ...]:
        # Do not join source tables here: a damaged owned lineage must remain
        # visible and fail closed, rather than disappear into reference-only.
        lineages = tuple(
            (
                await session.scalars(
                    select(CandidateAdoptionRow)
                    .where(
                        CandidateAdoptionRow.owner_agent_id == version.owner_agent_id,
                        CandidateAdoptionRow.resulting_experience_id
                        == version.experience_id,
                        CandidateAdoptionRow.resulting_content_hash
                        == version.content_hash,
                    )
                    .order_by(CandidateAdoptionRow.adoption_id)
                )
            ).all()
        )
        repository = CandidateRepository()
        candidate_service = CandidateService(repository=repository)
        embedded: dict[bytes, EmbeddedExcerptSnapshotV1] = {}
        for lineage in lineages:
            resulting_version = await session.scalar(
                select(ExperienceVersionRow.version_id)
                .join(
                    ExperienceRow,
                    ExperienceRow.experience_id == ExperienceVersionRow.experience_id,
                )
                .where(
                    ExperienceRow.owner_agent_id == version.owner_agent_id,
                    ExperienceRow.experience_id == version.experience_id,
                    ExperienceVersionRow.version_id == lineage.resulting_version_id,
                    ExperienceVersionRow.experience_id == version.experience_id,
                    ExperienceVersionRow.content_hash == version.content_hash,
                )
            )
            if resulting_version is None:
                raise ValueError("Located result version is unavailable")
            record = await repository.find_owned(
                session=session,
                owner_agent_id=version.owner_agent_id,
                candidate_id=lineage.candidate_id,
            )
            if record is None:
                raise ValueError("Located candidate is unavailable")
            # The candidate's public read path authenticates the trajectory
            # manifest and reconstructs captured evidence using shared checks.
            candidate = await candidate_service.get_owned(
                session=session,
                owner_agent_id=version.owner_agent_id,
                candidate_id=lineage.candidate_id,
            )
            if (
                candidate.decision is not CandidateDecision.ADOPTED
                or record.state.adoption_id != lineage.adoption_id
                or candidate.resulting_experience_id != lineage.resulting_experience_id
                or candidate.resulting_version_id != lineage.resulting_version_id
                or candidate.decided_at != lineage.adopted_at
                or candidate.kind is not version.kind
                or candidate.content_hash != version.content_hash
                or candidate.content != version.content
            ):
                raise ValueError("Located adoption differs from selected content")
            for captured in candidate.evidence:
                reference = TypedEvidence(
                    type="trajectory_field",
                    id=(
                        f"{record.bundle.manifest_hash}:{captured.step_id}:"
                        f"{captured.field.value}"
                    ),
                )
                if reference not in version.content.evidence:
                    raise ValueError("Located evidence exceeds selected closure")
                snapshot = EmbeddedExcerptSnapshotV1(
                    mode="embedded_excerpt",
                    reference=reference,
                    excerpt=captured.excerpt,
                    excerpt_hash=captured.excerpt_hash,
                    source_hash=captured.source_hash,
                    source_manifest_hash=record.bundle.manifest_hash,
                    step_id=captured.step_id,
                    field=captured.field,
                )
                key = canonical_json_bytes(reference)
                previous = embedded.get(key)
                if previous is not None and canonical_json_bytes(
                    previous
                ) != canonical_json_bytes(snapshot):
                    raise ValueError("Owned excerpts conflict")
                embedded[key] = snapshot

        return tuple(
            embedded.get(canonical_json_bytes(reference))
            or ReferenceOnlySnapshotV1(mode="reference_only", reference=reference)
            for reference in version.content.evidence
        )
