"""Owner-anchored persistence queries for quarantined candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.experiences.candidate_models import CandidateDecision
from experience_hub.storage.tables import (
    CandidateStateRow,
    ExperienceCandidateRow,
    TrajectoryBundleRow,
    TrajectoryEvidenceRow,
)


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One fully owner-joined candidate source and projection row."""

    candidate: ExperienceCandidateRow
    state: CandidateStateRow
    bundle: TrajectoryBundleRow


class CandidateRepository:
    """Issue only owner-first candidate statements."""

    @staticmethod
    def _owned_statement(
        owner_agent_id: UUID,
    ) -> Select[
        tuple[
            ExperienceCandidateRow,
            CandidateStateRow,
            TrajectoryBundleRow,
        ]
    ]:
        return (
            select(
                ExperienceCandidateRow,
                CandidateStateRow,
                TrajectoryBundleRow,
            )
            .select_from(ExperienceCandidateRow)
            .join(
                CandidateStateRow,
                and_(
                    CandidateStateRow.candidate_id
                    == ExperienceCandidateRow.candidate_id,
                    CandidateStateRow.owner_agent_id
                    == ExperienceCandidateRow.owner_agent_id,
                ),
            )
            .join(
                TrajectoryBundleRow,
                and_(
                    TrajectoryBundleRow.bundle_id
                    == ExperienceCandidateRow.bundle_id,
                    TrajectoryBundleRow.owner_agent_id
                    == ExperienceCandidateRow.owner_agent_id,
                ),
            )
            .where(
                ExperienceCandidateRow.owner_agent_id == owner_agent_id,
                CandidateStateRow.owner_agent_id == owner_agent_id,
                TrajectoryBundleRow.owner_agent_id == owner_agent_id,
            )
        )

    async def find_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        candidate_id: UUID,
    ) -> CandidateRecord | None:
        row = (
            await session.execute(
                self._owned_statement(owner_agent_id).where(
                    ExperienceCandidateRow.candidate_id == candidate_id
                )
            )
        ).one_or_none()
        if row is None:
            return None
        return CandidateRecord(
            candidate=row[0],
            state=row[1],
            bundle=row[2],
        )

    async def lock_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        candidate_id: UUID,
    ) -> CandidateRecord | None:
        row = (
            await session.execute(
                self._owned_statement(owner_agent_id)
                .where(ExperienceCandidateRow.candidate_id == candidate_id)
                .with_for_update()
            )
        ).one_or_none()
        if row is None:
            return None
        return CandidateRecord(
            candidate=row[0],
            state=row[1],
            bundle=row[2],
        )

    async def list_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        decision: CandidateDecision | None,
        limit: int,
        after: tuple[datetime, UUID] | None,
    ) -> tuple[CandidateRecord, ...]:
        statement = self._owned_statement(owner_agent_id)
        if decision is not None:
            statement = statement.where(
                CandidateStateRow.decision == decision.value
            )
        if after is not None:
            created_at, candidate_id = after
            statement = statement.where(
                or_(
                    ExperienceCandidateRow.created_at < created_at,
                    and_(
                        ExperienceCandidateRow.created_at == created_at,
                        ExperienceCandidateRow.candidate_id < candidate_id,
                    ),
                )
            )
        rows = (
            await session.execute(
                statement.order_by(
                    ExperienceCandidateRow.created_at.desc(),
                    ExperienceCandidateRow.candidate_id.desc(),
                ).limit(limit)
            )
        ).all()
        return tuple(
            CandidateRecord(candidate=row[0], state=row[1], bundle=row[2])
            for row in rows
        )

    async def evidence_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        bundle_id: UUID,
        evidence_ids: tuple[UUID, ...],
    ) -> tuple[TrajectoryEvidenceRow, ...]:
        if not evidence_ids:
            return ()
        rows = tuple(
            (
                await session.scalars(
                    select(TrajectoryEvidenceRow)
                    .select_from(TrajectoryEvidenceRow)
                    .join(
                        TrajectoryBundleRow,
                        and_(
                            TrajectoryBundleRow.bundle_id
                            == TrajectoryEvidenceRow.bundle_id,
                            TrajectoryBundleRow.owner_agent_id
                            == TrajectoryEvidenceRow.owner_agent_id,
                        ),
                    )
                    .where(
                        TrajectoryEvidenceRow.owner_agent_id == owner_agent_id,
                        TrajectoryBundleRow.owner_agent_id == owner_agent_id,
                        TrajectoryEvidenceRow.bundle_id == bundle_id,
                        TrajectoryEvidenceRow.evidence_id.in_(evidence_ids),
                    )
                )
            ).all()
        )
        return rows

    async def evidence_owned_batch(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        evidence_ids: tuple[UUID, ...],
    ) -> tuple[TrajectoryEvidenceRow, ...]:
        if not evidence_ids:
            return ()
        rows = tuple(
            (
                await session.scalars(
                    select(TrajectoryEvidenceRow)
                    .select_from(TrajectoryEvidenceRow)
                    .join(
                        TrajectoryBundleRow,
                        and_(
                            TrajectoryBundleRow.bundle_id
                            == TrajectoryEvidenceRow.bundle_id,
                            TrajectoryBundleRow.owner_agent_id
                            == TrajectoryEvidenceRow.owner_agent_id,
                        ),
                    )
                    .where(
                        TrajectoryEvidenceRow.owner_agent_id == owner_agent_id,
                        TrajectoryBundleRow.owner_agent_id == owner_agent_id,
                        TrajectoryEvidenceRow.evidence_id.in_(evidence_ids),
                    )
                )
            ).all()
        )
        return rows


__all__ = ["CandidateRepository"]
