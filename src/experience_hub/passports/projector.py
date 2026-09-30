"""Fail-closed replay of Passport state from owned immutable sources."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

from experience_hub import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry, StoredEvent
from experience_hub.passports.codec import verify_passport_bytes
from experience_hub.passports.errors import PassportError
from experience_hub.passports.events import (
    PASSPORT_EVENT_TYPES,
    PassportAdoptedV1,
    PassportImportedV1,
    PassportRejectedV1,
)
from experience_hub.passports.scopes import (
    PASSPORT_ADOPT_SCOPE,
    PASSPORT_IMPORT_SCOPE,
    PASSPORT_REJECT_SCOPE,
)
from experience_hub.storage.tables import DomainEventRow, IdempotencyRecordRow
from experience_hub.storage.tables.passports import (
    PassportAdoptionRow,
    PassportImportRow,
)

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class PassportProjectionIntegrityError(RuntimeError):
    """A Passport event cannot be reconciled with its immutable sources."""

    code = "passport_projection_integrity_error"


def _fail(message: str) -> PassportProjectionIntegrityError:
    return PassportProjectionIntegrityError(message)


def _target(prefix: str | None) -> str:
    if prefix is None:
        return 'main."passport_state"'
    name = f"{prefix}passport_state"
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError("Unsafe Passport projection target")
    return f'temp."{name}"'


def _utc(value: datetime) -> str:
    return require_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


async def _create_rebuild_table(session: AsyncSession, target: str) -> None:
    await session.execute(
        text(
            f"CREATE TEMP TABLE {target} ("
            "import_id VARCHAR(36) NOT NULL PRIMARY KEY, "
            "owner_agent_id VARCHAR(36) NOT NULL, state VARCHAR(8) NOT NULL, "
            "adoption_id VARCHAR(36), resulting_experience_id VARCHAR(36), "
            "resulting_version_id VARCHAR(36), reason_code VARCHAR, "
            "reason_text VARCHAR, "
            "reason_text_hash VARCHAR(64), decided_at VARCHAR(27), "
            "projection_event_id INTEGER NOT NULL, "
            "CHECK(state IN ('pending','adopted','rejected')), "
            "CHECK(projection_event_id > 0), "
            "CHECK((state='pending' AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NULL) "
            "OR (state='adopted' AND adoption_id IS NOT NULL "
            "AND resulting_experience_id IS NOT NULL "
            "AND resulting_version_id IS NOT NULL "
            "AND reason_code IS NULL AND reason_text IS NULL "
            "AND reason_text_hash IS NULL AND decided_at IS NOT NULL) "
            "OR (state='rejected' AND adoption_id IS NULL "
            "AND resulting_experience_id IS NULL AND resulting_version_id IS NULL "
            "AND reason_code IS NOT NULL AND length(trim(reason_code)) > 0 "
            "AND reason_text IS NOT NULL AND length(trim(reason_text)) > 0 "
            "AND reason_text_hash IS NOT NULL AND decided_at IS NOT NULL)))"
        )
    )


async def _receipt(
    session: AsyncSession,
    event: StoredEvent,
    *,
    owner: UUID,
    scope: str,
    resource: str,
    result: UUID,
) -> None:
    receipt = await session.scalar(
        select(IdempotencyRecordRow).where(
            IdempotencyRecordRow.receipt_id == event.causation_id,
            IdempotencyRecordRow.caller_scope == f"agent:{owner}",
        )
    )
    if (
        receipt is None
        or receipt.scope != scope
        or receipt.result_resource_type != resource
        or receipt.result_resource_id != result
        or receipt.created_at != event.occurred_at
    ):
        raise _fail("Passport command receipt anchor is inconsistent")
    in_progress = (
        receipt.state == "in_progress"
        and receipt.completed_at is None
        and receipt.response_status_code is None
        and receipt.response_body is None
        and receipt.response_content_type is None
        and receipt.response_headers is None
    )
    completed = (
        receipt.state == "completed"
        and receipt.completed_at is not None
        and receipt.completed_at >= event.occurred_at
        and receipt.response_status_code is not None
        and receipt.response_body is not None
        and receipt.response_content_type is not None
        and receipt.response_headers is not None
    )
    if not (in_progress or completed):
        raise _fail("Passport command receipt state is inconsistent")


async def _owned_source(
    session: AsyncSession, import_id: UUID, owner: UUID
) -> PassportImportRow:
    source = await session.scalar(
        select(PassportImportRow).where(
            PassportImportRow.import_id == import_id,
            PassportImportRow.owner_agent_id == owner,
        )
    )
    if source is None:
        raise _fail("Passport owned source anchor is inconsistent")
    try:
        prepared = verify_passport_bytes(source.canonical_bytes)
        if prepared.document.passport_hash != source.passport_hash:
            raise ValueError("Hash mismatch")
        require_utc(source.imported_at)
    except (PassportError, TypeError, ValueError):
        raise _fail("Passport source bytes are invalid") from None
    return source


class PassportStateProjector:
    """Project only quarantine decisions; source and lineage are never rewritten."""

    name = "passport_state"
    version = 1
    event_types = PASSPORT_EVENT_TYPES

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    def stored_event_from_row(self, row: DomainEventRow) -> StoredEvent:
        try:
            payload = self._event_registry.decode(
                event_type=row.event_type, payload=row.payload
            )
        except (TypeError, ValueError):
            raise _fail("Passport event payload is invalid") from None
        return StoredEvent(
            event_id=row.event_id,
            aggregate_type=row.aggregate_type,
            aggregate_id=row.aggregate_id,
            sequence=row.sequence,
            event_type=row.event_type,
            payload=payload,
            actor_agent_id=row.actor_agent_id,
            causation_id=row.causation_id,
            occurred_at=row.occurred_at,
        )

    async def apply(self, session: AsyncSession, event: StoredEvent) -> None:
        await self._apply(session, event, target=_target(None))
        session.expire_all()

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        target = _target(target_prefix)
        await _create_rebuild_table(session, target)
        rows = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.event_type.in_(self.event_types),
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        for row in rows:
            await self._apply(session, self.stored_event_from_row(row), target=target)

    async def _apply(
        self, session: AsyncSession, event: StoredEvent, *, target: str
    ) -> None:
        try:
            require_utc(event.occurred_at)
            payload = self._event_registry.decode(
                event_type=event.event_type,
                payload=canonical_json_bytes(event.payload),
            )
        except (TypeError, ValueError):
            raise _fail("Passport event payload or timestamp is invalid") from None
        if (
            event.aggregate_type != "passport_import"
            or event.event_id < 1
            or not isinstance(
                payload, (PassportImportedV1, PassportAdoptedV1, PassportRejectedV1)
            )
            or event.aggregate_id != payload.import_id
            or event.actor_agent_id != payload.owner_agent_id
        ):
            raise _fail("Passport event aggregate anchor is inconsistent")
        source = await _owned_source(session, payload.import_id, payload.owner_agent_id)
        if isinstance(payload, PassportImportedV1):
            if (
                event.sequence != 1
                or source.passport_hash != payload.passport_hash
                or source.imported_at != event.occurred_at
            ):
                raise _fail("Passport import source anchor is inconsistent")
            await _receipt(
                session,
                event,
                owner=payload.owner_agent_id,
                scope=PASSPORT_IMPORT_SCOPE,
                resource="passport_import",
                result=payload.import_id,
            )
            await self._write(
                session,
                text(
                    f"INSERT INTO {target} "
                    "(import_id,owner_agent_id,state,projection_event_id) "
                    "VALUES (:id,:owner,'pending',:event)"
                ),
                {
                    "id": str(payload.import_id),
                    "owner": str(payload.owner_agent_id),
                    "event": event.event_id,
                },
            )
            return
        before = await self._pending(session, event, payload, target, source)
        values: dict[str, object] = {
            "id": str(payload.import_id),
            "owner": str(payload.owner_agent_id),
            "event": event.event_id,
            "before": before,
            "when": _utc(event.occurred_at),
        }
        if isinstance(payload, PassportAdoptedV1):
            adoption = await session.scalar(
                select(PassportAdoptionRow).where(
                    PassportAdoptionRow.adoption_id == payload.adoption_id,
                    PassportAdoptionRow.import_id == payload.import_id,
                    PassportAdoptionRow.owner_agent_id == payload.owner_agent_id,
                )
            )
            if (
                adoption is None
                or adoption.resulting_experience_id != payload.resulting_experience_id
                or adoption.resulting_version_id != payload.resulting_version_id
                or adoption.resulting_content_hash != payload.resulting_content_hash
                or adoption.created is not payload.created
                or adoption.importance != payload.importance
                or adoption.confidence != payload.confidence
                or adoption.adopted_at != event.occurred_at
            ):
                raise _fail("Passport adoption source anchor is inconsistent")
            await _receipt(
                session,
                event,
                owner=payload.owner_agent_id,
                scope=PASSPORT_ADOPT_SCOPE,
                resource="passport_adoption",
                result=payload.adoption_id,
            )
            values.update(
                {
                    "adoption": str(payload.adoption_id),
                    "experience": str(payload.resulting_experience_id),
                    "version": str(payload.resulting_version_id),
                }
            )
            changes = (
                "state='adopted',adoption_id=:adoption,"
                "resulting_experience_id=:experience,resulting_version_id=:version"
            )
        else:
            await _receipt(
                session,
                event,
                owner=payload.owner_agent_id,
                scope=PASSPORT_REJECT_SCOPE,
                resource="passport_import",
                result=payload.import_id,
            )
            values.update(
                {
                    "code": payload.reason.code,
                    "reason": payload.reason.text,
                    "hash": payload.reason.text_hash,
                }
            )
            changes = (
                "state='rejected',reason_code=:code,reason_text=:reason,"
                "reason_text_hash=:hash"
            )
        await self._write(
            session,
            text(
                f"UPDATE {target} SET {changes},decided_at=:when,"
                "projection_event_id=:event "
                "WHERE import_id=:id AND owner_agent_id=:owner "
                "AND state='pending' AND projection_event_id=:before"
            ),
            values,
        )

    async def _pending(
        self,
        session: AsyncSession,
        event: StoredEvent,
        payload: PassportAdoptedV1 | PassportRejectedV1,
        target: str,
        source: PassportImportRow,
    ) -> int:
        if event.sequence != 2:
            raise _fail("Passport decision sequence is inconsistent")
        rows = (
            await session.scalars(
                select(DomainEventRow).where(
                    DomainEventRow.aggregate_type == "passport_import",
                    DomainEventRow.aggregate_id == payload.import_id,
                    DomainEventRow.sequence == 1,
                    DomainEventRow.event_type == PassportImportedV1.event_type,
                )
            )
        ).all()
        if len(rows) != 1:
            raise _fail("Passport decision requires one import event")
        imported = self.stored_event_from_row(rows[0])
        if (
            not isinstance(imported.payload, PassportImportedV1)
            or imported.payload.import_id != payload.import_id
            or imported.payload.owner_agent_id != payload.owner_agent_id
            or imported.payload.passport_hash != source.passport_hash
            or imported.actor_agent_id != payload.owner_agent_id
            or imported.occurred_at != source.imported_at
            or imported.occurred_at > event.occurred_at
            or imported.event_id >= event.event_id
        ):
            raise _fail("Passport decision import event is inconsistent")
        await _receipt(
            session,
            imported,
            owner=payload.owner_agent_id,
            scope=PASSPORT_IMPORT_SCOPE,
            resource="passport_import",
            result=payload.import_id,
        )
        pending = (
            await session.execute(
                text(
                    f"SELECT projection_event_id FROM {target} WHERE import_id=:id "
                    "AND owner_agent_id=:owner AND state='pending'"
                ),
                {"id": str(payload.import_id), "owner": str(payload.owner_agent_id)},
            )
        ).scalar()
        if pending != imported.event_id:
            raise _fail("Passport decision does not match pending state")
        return imported.event_id

    async def _write(
        self, session: AsyncSession, statement: TextClause, values: dict[str, object]
    ) -> None:
        try:
            await session.execute(statement, values)
            changed = await session.scalar(text("SELECT changes()"))
        except SQLAlchemyError:
            raise _fail("Passport projection compare-and-set failed") from None
        if changed != 1:
            raise _fail("Passport projection must change exactly one state")


__all__ = ["PassportProjectionIntegrityError", "PassportStateProjector"]
