"""Owner-first persistence access; ORM records never leave this package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.passports.contracts import PassportState
from experience_hub.passports.errors import PassportError
from experience_hub.storage.tables import (
    AgentRow,
    PassportAdoptionRow,
    PassportImportRow,
    PassportStateRow,
)


@dataclass(frozen=True, slots=True)
class PassportRecord:
    source: PassportImportRow
    state: PassportStateRow
    adoption: PassportAdoptionRow | None


class PassportRepository:
    async def find_adoption_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        adoption_id: UUID,
    ) -> PassportRecord | None:
        adoption = await session.scalar(
            select(PassportAdoptionRow).where(
                PassportAdoptionRow.adoption_id == adoption_id,
                PassportAdoptionRow.owner_agent_id == owner_agent_id,
            )
        )
        if adoption is None:
            return None
        record = await self.find_owned(
            session=session, owner_agent_id=owner_agent_id, import_id=adoption.import_id
        )
        if record is None or record.adoption is None:
            raise PassportError("invalid")
        return record

    async def require_owner(
        self, *, session: AsyncSession, owner_agent_id: UUID
    ) -> datetime:
        if not isinstance(owner_agent_id, UUID):
            raise PassportError("not_found")
        try:
            created_at = await session.scalar(
                select(AgentRow.created_at).where(AgentRow.agent_id == owner_agent_id)
            )
        except (StatementError, TypeError, ValueError, LookupError):
            raise PassportError("invalid") from None
        if created_at is None:
            raise PassportError("not_found")
        return created_at

    async def find_owned(
        self, *, session: AsyncSession, owner_agent_id: UUID, import_id: UUID
    ) -> PassportRecord | None:
        source = await self._source(
            session,
            select(PassportImportRow).where(
                PassportImportRow.owner_agent_id == owner_agent_id,
                PassportImportRow.import_id == import_id,
            ),
        )
        return None if source is None else await self._record(session, source)

    async def find_by_hash(
        self, *, session: AsyncSession, owner_agent_id: UUID, passport_hash: str
    ) -> PassportRecord | None:
        source = await self._source(
            session,
            select(PassportImportRow).where(
                PassportImportRow.owner_agent_id == owner_agent_id,
                PassportImportRow.passport_hash == passport_hash,
            ),
        )
        return None if source is None else await self._record(session, source)

    async def list_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        state: PassportState | None,
        limit: int,
        after: tuple[datetime, UUID] | None,
    ) -> tuple[PassportRecord, ...]:
        # A state filter must not quietly turn an owned orphan into an empty page.
        orphan = await session.scalar(
            select(PassportImportRow.import_id)
            .outerjoin(
                PassportStateRow,
                and_(
                    PassportStateRow.import_id == PassportImportRow.import_id,
                    PassportStateRow.owner_agent_id == PassportImportRow.owner_agent_id,
                ),
            )
            .where(
                PassportImportRow.owner_agent_id == owner_agent_id,
                or_(
                    PassportStateRow.import_id.is_(None),
                    PassportStateRow.state.not_in(tuple(PassportState)),
                ),
            )
            .limit(1)
        )
        if orphan is not None:
            raise PassportError("invalid")
        statement = select(PassportImportRow).where(
            PassportImportRow.owner_agent_id == owner_agent_id
        )
        if state is not None:
            statement = statement.join(
                PassportStateRow,
                and_(
                    PassportStateRow.import_id == PassportImportRow.import_id,
                    PassportStateRow.owner_agent_id == owner_agent_id,
                ),
            ).where(PassportStateRow.state == state)
        if after is not None:
            imported_at, import_id = after
            statement = statement.where(
                or_(
                    PassportImportRow.imported_at < imported_at,
                    and_(
                        PassportImportRow.imported_at == imported_at,
                        PassportImportRow.import_id < import_id,
                    ),
                )
            )
        try:
            sources = (
                await session.scalars(
                    statement.order_by(
                        PassportImportRow.imported_at.desc(),
                        PassportImportRow.import_id.desc(),
                    ).limit(limit)
                )
            ).all()
        except (StatementError, TypeError, ValueError, LookupError):
            raise PassportError("invalid") from None
        return tuple([await self._record(session, source) for source in sources])

    @staticmethod
    async def _source(
        session: AsyncSession, statement: Select[tuple[PassportImportRow]]
    ) -> PassportImportRow | None:
        try:
            return (await session.scalars(statement)).one_or_none()
        except (StatementError, TypeError, ValueError, LookupError):
            raise PassportError("invalid") from None

    @staticmethod
    async def _record(
        session: AsyncSession, source: PassportImportRow
    ) -> PassportRecord:
        try:
            return await PassportRepository._load_record(session, source)
        except (StatementError, TypeError, ValueError, LookupError):
            raise PassportError("invalid") from None

    @staticmethod
    async def _load_record(
        session: AsyncSession, source: PassportImportRow
    ) -> PassportRecord:
        state = await session.scalar(
            select(PassportStateRow).where(
                PassportStateRow.import_id == source.import_id,
                PassportStateRow.owner_agent_id == source.owner_agent_id,
            )
        )
        if state is None:
            raise PassportError("invalid")
        adoption = await session.scalar(
            select(PassportAdoptionRow).where(
                PassportAdoptionRow.import_id == source.import_id,
                PassportAdoptionRow.owner_agent_id == source.owner_agent_id,
            )
        )
        return PassportRecord(source, state, adoption)
