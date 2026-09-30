"""Verified owner-scoped read values and scope-bound deterministic pagination."""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import UTC, datetime
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import StructuredReason
from experience_hub.passports.codec import VerifiedPassportV1, verify_passport_bytes
from experience_hub.passports.contracts import (
    PassportAdoptionV1,
    PassportImportViewV1,
    PassportPageV1,
    PassportState,
)
from experience_hub.passports.errors import PassportError
from experience_hub.passports.events import (
    PassportAdoptedV1,
    PassportImportedV1,
    PassportRejectedV1,
)
from experience_hub.passports.repository import PassportRecord, PassportRepository
from experience_hub.passports.requests import passport_import_request
from experience_hub.passports.responses import passport_import_response
from experience_hub.storage.tables import DomainEventRow, IdempotencyRecordRow


class PassportQuery:
    def __init__(self, *, repository: PassportRepository) -> None:
        self._repository = repository

    async def get_adoption(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        adoption_id: UUID,
    ) -> PassportAdoptionV1:
        await self._repository.require_owner(
            session=session, owner_agent_id=owner_agent_id
        )
        record = await self._repository.find_adoption_owned(
            session=session, owner_agent_id=owner_agent_id, adoption_id=adoption_id
        )
        if record is None:
            raise PassportError("not_found")
        await self.view(session=session, record=record)
        row = record.adoption
        if row is None or record.state.state is not PassportState.ADOPTED:
            raise PassportError("invalid")
        # The selected historical version must still be owned and authenticated.
        from experience_hub.errors import DomainError
        from experience_hub.experiences.queries import ExperienceQuery
        from experience_hub.storage.validation import SourceIntegrityError

        try:
            version = await ExperienceQuery().get_owned_shareable_version(
                session=session,
                owner_agent_id=owner_agent_id,
                experience_id=row.resulting_experience_id,
                version_id=row.resulting_version_id,
            )
            if version.content_hash != row.resulting_content_hash:
                raise PassportError("invalid")
        except (DomainError, SourceIntegrityError, TypeError, ValueError):
            raise PassportError("invalid") from None
        return PassportAdoptionV1(
            adoption_id=row.adoption_id,
            import_id=row.import_id,
            owner_agent_id=row.owner_agent_id,
            resulting_experience_id=row.resulting_experience_id,
            resulting_version_id=row.resulting_version_id,
            resulting_content_hash=row.resulting_content_hash,
            created=row.created,
            importance=row.importance,
            confidence=row.confidence,
            adopted_at=row.adopted_at,
            prepared=self.prepared(record),
        )

    async def get_owned(
        self, *, session: AsyncSession, owner_agent_id: UUID, import_id: UUID
    ) -> PassportImportViewV1:
        await self._repository.require_owner(
            session=session, owner_agent_id=owner_agent_id
        )
        record = await self._repository.find_owned(
            session=session, owner_agent_id=owner_agent_id, import_id=import_id
        )
        if record is None:
            raise PassportError("not_found")
        return await self.view(session=session, record=record)

    async def list_owned(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        state: PassportState | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> PassportPageV1:
        await self._repository.require_owner(
            session=session, owner_agent_id=owner_agent_id
        )
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or (state is not None and not isinstance(state, PassportState))
        ):
            raise PassportError("invalid")
        after = (
            None if cursor is None else _decode_cursor(cursor, owner_agent_id, state)
        )
        records = await self._repository.list_owned(
            session=session,
            owner_agent_id=owner_agent_id,
            state=state,
            limit=limit + 1,
            after=after,
        )
        retained = records[:limit]
        items = tuple([await self.view(session=session, record=r) for r in retained])
        next_cursor = None
        if len(records) > limit:
            source = retained[-1].source
            next_cursor = _encode_cursor(
                owner_agent_id, state, source.imported_at, source.import_id
            )
        return PassportPageV1(items=items, next_cursor=next_cursor)

    @staticmethod
    def prepared(record: PassportRecord) -> VerifiedPassportV1:
        try:
            prepared = verify_passport_bytes(record.source.canonical_bytes)
            if prepared.document.passport_hash != record.source.passport_hash:
                raise PassportError("invalid")
            return prepared
        except (PassportError, TypeError, ValueError):
            raise PassportError("invalid") from None

    async def view(
        self, *, session: AsyncSession, record: PassportRecord
    ) -> PassportImportViewV1:
        source, state, adoption = record.source, record.state, record.adoption
        prepared = self.prepared(record)
        events = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "passport_import",
                    DomainEventRow.aggregate_id == source.import_id,
                )
                .order_by(DomainEventRow.sequence)
            )
        ).all()
        try:
            if len(events) != (1 if state.state is PassportState.PENDING else 2):
                raise ValueError
            anchor = events[0]
            imported = PassportImportedV1.model_validate_json(anchor.payload)
            if (
                anchor.event_type != PassportImportedV1.event_type
                or anchor.sequence != 1
                or anchor.actor_agent_id != source.owner_agent_id
                or anchor.occurred_at != source.imported_at
                or imported.import_id != source.import_id
                or imported.owner_agent_id != source.owner_agent_id
                or imported.passport_hash != source.passport_hash
                or state.projection_event_id != events[-1].event_id
            ):
                raise ValueError
            receipt = await session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.receipt_id == anchor.causation_id,
                    IdempotencyRecordRow.caller_scope
                    == f"agent:{source.owner_agent_id}",
                )
            )
            if (
                receipt is None
                or receipt.scope != "passport.import"
                or receipt.state != "completed"
                or receipt.response_status_code != 201
                or receipt.result_resource_type != "passport_import"
                or receipt.result_resource_id != source.import_id
                or receipt.created_at != source.imported_at
                or receipt.completed_at is None
                or receipt.completed_at < receipt.created_at
                or receipt.request_hash
                != passport_import_request(
                    owner_agent_id=source.owner_agent_id,
                    passport_hash=source.passport_hash,
                    idempotency_key=receipt.idempotency_key,
                ).request_hash
                or receipt.response_body
                != passport_import_response(
                    import_id=source.import_id,
                    owner_agent_id=source.owner_agent_id,
                    passport_hash=source.passport_hash,
                    state=PassportState.PENDING,
                    status_code=201,
                ).body
            ):
                raise ValueError
            reason = None
            if state.state is PassportState.PENDING:
                if adoption is not None or state.decided_at is not None:
                    raise ValueError
            else:
                decision = events[-1]
                if (
                    decision.sequence != 2
                    or decision.occurred_at != state.decided_at
                    or decision.occurred_at < source.imported_at
                    or decision.actor_agent_id != source.owner_agent_id
                ):
                    raise ValueError
                if state.state is PassportState.REJECTED:
                    payload: PassportRejectedV1 | PassportAdoptedV1
                    payload = PassportRejectedV1.model_validate_json(decision.payload)
                    if (
                        state.reason_code is None
                        or state.reason_text is None
                        or state.reason_text_hash is None
                    ):
                        raise ValueError
                    reason = StructuredReason(
                        code=state.reason_code,
                        text=state.reason_text,
                        text_hash=state.reason_text_hash,
                    )
                    if (
                        decision.event_type != PassportRejectedV1.event_type
                        or payload.reason != reason
                        or adoption is not None
                    ):
                        raise ValueError
                else:
                    adopted = PassportAdoptedV1.model_validate_json(decision.payload)
                    if (
                        decision.event_type != PassportAdoptedV1.event_type
                        or adoption is None
                        or adoption.adoption_id != state.adoption_id
                        or adoption.resulting_experience_id
                        != state.resulting_experience_id
                        or adoption.resulting_version_id != state.resulting_version_id
                        or adoption.adopted_at != state.decided_at
                        or adopted.adoption_id != adoption.adoption_id
                        or adopted.resulting_experience_id
                        != adoption.resulting_experience_id
                        or adopted.resulting_version_id != adoption.resulting_version_id
                        or adopted.resulting_content_hash
                        != adoption.resulting_content_hash
                        or adopted.created != adoption.created
                        or adopted.importance != adoption.importance
                        or adopted.confidence != adoption.confidence
                        or adoption.resulting_content_hash
                        != prepared.document.subject.content_hash
                    ):
                        raise ValueError
                    payload = adopted
                if payload.import_id != source.import_id or (
                    payload.owner_agent_id != source.owner_agent_id
                ):
                    raise ValueError
            return PassportImportViewV1(
                import_id=source.import_id,
                owner_agent_id=source.owner_agent_id,
                passport_hash=source.passport_hash,
                document=prepared.document,
                state=state.state,
                adoption_id=state.adoption_id,
                resulting_experience_id=state.resulting_experience_id,
                resulting_version_id=state.resulting_version_id,
                resulting_content_hash=None
                if adoption is None
                else adoption.resulting_content_hash,
                created=None if adoption is None else adoption.created,
                importance=None if adoption is None else adoption.importance,
                confidence=None if adoption is None else adoption.confidence,
                reason=reason,
                imported_at=source.imported_at,
                decided_at=state.decided_at,
            )
        except (TypeError, ValueError, ValidationError):
            raise PassportError("invalid") from None


def _encode_cursor(
    owner: UUID, state: PassportState | None, imported_at: datetime, import_id: UUID
) -> str:
    body = canonical_json_bytes(
        {
            "owner_agent_id": owner,
            "state": state,
            "import_id": import_id,
            "imported_at": require_utc(imported_at).isoformat(timespec="microseconds"),
        }
    )
    return base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")


def _decode_cursor(
    cursor: str, owner: UUID, state: PassportState | None
) -> tuple[datetime, UUID]:
    try:
        if (
            not isinstance(cursor, str)
            or not 1 <= len(cursor) <= 8192
            or re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None
        ):
            raise ValueError
        body = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True
        )
        value = json.loads(body)
        if not isinstance(value, dict) or set(value) != {
            "owner_agent_id",
            "state",
            "import_id",
            "imported_at",
        }:
            raise ValueError
        imported_at = datetime.fromisoformat(value["imported_at"])
        import_id = UUID(value["import_id"])
        if (
            imported_at.tzinfo != UTC
            or value["owner_agent_id"] != str(owner)
            or value["state"] != state
            or _encode_cursor(owner, state, imported_at, import_id) != cursor
        ):
            raise ValueError
        return imported_at, import_id
    except (ValueError, TypeError, KeyError, binascii.Error, UnicodeError):
        raise PassportError("invalid") from None
