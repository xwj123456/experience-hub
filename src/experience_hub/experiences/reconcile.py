"""Fail-closed physical payload codec reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.experiences.content import preferred_payload_codec
from experience_hub.experiences.models import (
    ExperienceKind,
    PayloadCodec,
    VersionContent,
)
from experience_hub.experiences.reconcile_contracts import (
    PayloadReconcileIssue,
    PayloadReconcileReport,
)
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.storage.payload_rewrite import (
    PayloadRewriteConflictError,
    rewrite_payload_codec,
)
from experience_hub.storage.tables import (
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


@dataclass(slots=True)
class _Candidate:
    identity: ExperienceRow
    version: ExperienceVersionRow
    payload: ExperiencePayloadRow
    preferred_codec: PayloadCodec
    content: VersionContent
    kind: ExperienceKind
    content_hash: str
    payload_hash: str


@dataclass(frozen=True, slots=True)
class _Inspection:
    candidates: tuple[_Candidate, ...]
    report: PayloadReconcileReport


class PayloadReconciler:
    """Restore codecs to the current temperature preference without hash drift."""

    async def run(self, *, uow: UnitOfWork) -> PayloadReconcileReport:
        if (
            not isinstance(uow, UnitOfWork)
            or not uow.immediate
            or not uow.session.in_transaction()
        ):
            raise RuntimeError(
                "Payload reconciliation requires an active caller-owned immediate UOW"
            )

        inspection = await self._inspect(uow.session)
        if inspection.report.error_count:
            return inspection.report
        candidates = inspection.candidates

        changed_count = 0
        skipped_count = 0
        active_version: tuple[UUID, int, UUID] | None = None
        phase = "rewrite"
        try:
            async with uow.session.begin_nested():
                for candidate in candidates:
                    if candidate.payload.codec is candidate.preferred_codec:
                        skipped_count += 1
                        continue

                    active_version = (
                        candidate.identity.experience_id,
                        candidate.version.version_number,
                        candidate.version.version_id,
                    )
                    phase = "rewrite"
                    await rewrite_payload_codec(
                        session=uow.session,
                        version_id=candidate.version.version_id,
                        codec=candidate.preferred_codec,
                    )
                    phase = "validation"
                    await uow.session.refresh(candidate.identity)
                    await uow.session.refresh(candidate.version)
                    await uow.session.refresh(candidate.payload)
                    content = decode_and_verify_version(
                        identity=candidate.identity,
                        version=candidate.version,
                        payload=candidate.payload,
                    )
                    if (
                        candidate.payload.codec is not candidate.preferred_codec
                        or candidate.identity.kind is not candidate.kind
                        or candidate.version.content_hash != candidate.content_hash
                        or candidate.payload.payload_hash != candidate.payload_hash
                        or content != candidate.content
                    ):
                        raise SourceIntegrityError(
                            "Payload reconciliation changed semantic version content"
                        )
                    changed_count += 1
        except (
            LookupError,
            ValueError,
            PayloadRewriteConflictError,
            SourceIntegrityError,
        ):
            if active_version is None:
                raise
            experience_id, version_number, version_id = active_version
            issue = PayloadReconcileIssue(
                experience_id=experience_id,
                version_number=version_number,
                version_id=version_id,
                code=(
                    "rewrite_validation_failed"
                    if phase == "validation"
                    else "rewrite_failed"
                ),
            )
            return PayloadReconcileReport(
                changed_count=0,
                skipped_count=len(candidates) - 1,
                error_count=1,
                errors=(issue,),
            )

        return PayloadReconcileReport(
            changed_count=changed_count,
            skipped_count=skipped_count,
            error_count=0,
        )

    async def diagnose(self, session: AsyncSession) -> PayloadReconcileReport:
        """Return the same all-version preflight report without writing."""
        return (await self._inspect(session)).report

    async def _inspect(self, session: AsyncSession) -> _Inspection:
        await self._validate_no_orphan_sources(session)
        rows = (
            await session.execute(
                select(
                    ExperienceRow,
                    ExperienceVersionRow,
                    ExperiencePayloadRow,
                    ExperienceStateRow,
                )
                .select_from(ExperienceVersionRow)
                .outerjoin(
                    ExperienceRow,
                    ExperienceRow.experience_id == ExperienceVersionRow.experience_id,
                )
                .outerjoin(
                    ExperiencePayloadRow,
                    ExperiencePayloadRow.version_id == ExperienceVersionRow.version_id,
                )
                .outerjoin(
                    ExperienceStateRow,
                    ExperienceStateRow.experience_id
                    == ExperienceVersionRow.experience_id,
                )
                .order_by(
                    ExperienceVersionRow.experience_id,
                    ExperienceVersionRow.version_number,
                    ExperienceVersionRow.version_id,
                )
            )
        ).all()

        candidates: list[_Candidate] = []
        errors: list[PayloadReconcileIssue] = []
        for identity, version, payload, state in rows:
            if identity is None:
                errors.append(
                    PayloadReconcileIssue(
                        experience_id=version.experience_id,
                        version_number=version.version_number,
                        version_id=version.version_id,
                        code="missing_identity",
                    )
                )
                continue
            if payload is None:
                errors.append(
                    PayloadReconcileIssue(
                        experience_id=identity.experience_id,
                        version_number=version.version_number,
                        version_id=version.version_id,
                        code="missing_payload",
                    )
                )
                continue
            if state is None:
                errors.append(
                    PayloadReconcileIssue(
                        experience_id=identity.experience_id,
                        version_number=version.version_number,
                        version_id=version.version_id,
                        code="missing_state",
                    )
                )
                continue
            try:
                content = decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=payload,
                )
            except SourceIntegrityError:
                errors.append(
                    PayloadReconcileIssue(
                        experience_id=identity.experience_id,
                        version_number=version.version_number,
                        version_id=version.version_id,
                        code="semantic_validation_failed",
                    )
                )
                continue
            candidates.append(
                _Candidate(
                    identity=identity,
                    version=version,
                    payload=payload,
                    preferred_codec=preferred_payload_codec(state.temperature),
                    content=content,
                    kind=identity.kind,
                    content_hash=version.content_hash,
                    payload_hash=payload.payload_hash,
                )
            )

        return _Inspection(
            candidates=tuple(candidates),
            report=PayloadReconcileReport(
                changed_count=0,
                skipped_count=len(candidates),
                error_count=len(errors),
                errors=tuple(errors),
            ),
        )

    @staticmethod
    async def _validate_no_orphan_sources(session: AsyncSession) -> None:
        orphan_identity_id = await session.scalar(
            select(ExperienceRow.experience_id)
            .outerjoin(
                ExperienceVersionRow,
                ExperienceVersionRow.experience_id == ExperienceRow.experience_id,
            )
            .where(ExperienceVersionRow.version_id.is_(None))
            .order_by(ExperienceRow.experience_id)
            .limit(1)
        )
        if orphan_identity_id is not None:
            raise SourceIntegrityError(
                "Experience identity requires at least one version",
                mismatch_key=f"experience_identity:{orphan_identity_id}",
            )

        orphan_payload_version_id = await session.scalar(
            select(ExperiencePayloadRow.version_id)
            .outerjoin(
                ExperienceVersionRow,
                ExperienceVersionRow.version_id == ExperiencePayloadRow.version_id,
            )
            .where(ExperienceVersionRow.version_id.is_(None))
            .order_by(ExperiencePayloadRow.version_id)
            .limit(1)
        )
        if orphan_payload_version_id is not None:
            raise SourceIntegrityError(
                "Experience payload has no matching version",
                mismatch_key=(
                    f"experience_version_payload:{orphan_payload_version_id}"
                ),
            )


__all__ = [
    "PayloadReconcileIssue",
    "PayloadReconcileReport",
    "PayloadReconciler",
]
