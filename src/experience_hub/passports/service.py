"""Transactional import and explicit owner decisions for offline Passports."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.exc import MultipleResultsFound

from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandContext, CommandRequest, PendingEvent
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.errors import DomainError
from experience_hub.experiences.contracts import ExperienceDraft, ExperienceRecord
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.repository import ExperienceWriter
from experience_hub.ids import IdGenerator
from experience_hub.passports.contracts import (
    AdoptPassport,
    ImportPassport,
    PassportState,
    RejectPassport,
)
from experience_hub.passports.errors import PassportError
from experience_hub.passports.events import (
    PassportAdoptedV1,
    PassportImportedV1,
    PassportRejectedV1,
)
from experience_hub.passports.queries import PassportQuery
from experience_hub.passports.repository import PassportRecord, PassportRepository
from experience_hub.passports.requests import (
    passport_adopt_request,
    passport_import_request,
    passport_reject_request,
)
from experience_hub.passports.responses import (
    passport_adoption_response,
    passport_import_response,
)
from experience_hub.storage import ReceiptStore, StoredResponse, UnitOfWork
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import PassportAdoptionRow, PassportImportRow
from experience_hub.storage.validation import SourceIntegrityError


class PassportService:
    def __init__(
        self,
        *,
        repository: PassportRepository,
        experience_writer: ExperienceWriter,
        receipt_store: ReceiptStore,
        clock: Clock,
        id_generator: IdGenerator,
    ) -> None:
        self._repository = repository
        self._experience_writer = experience_writer
        self._receipt_store = receipt_store
        self._clock = clock
        self._id_generator = id_generator
        self._query = PassportQuery(repository=repository)

    async def adopt(
        self,
        *,
        uow: UnitOfWork,
        request: AdoptPassport,
        command: CommandContext,
    ) -> StoredResponse:
        self._caller(command, request.owner_agent_id, "passport.adopt")
        request = AdoptPassport(
            request.owner_agent_id,
            request.import_id,
            request.importance,
            request.confidence,
        )
        expected = passport_adopt_request(
            owner_agent_id=request.owner_agent_id,
            import_id=request.import_id,
            importance=request.importance,
            confidence=request.confidence,
            idempotency_key=command.idempotency_key,
        )
        decided_at = await self._operation_time(uow, command, expected)
        record = await self._pending(
            uow, request.owner_agent_id, request.import_id, decided_at
        )
        prepared = self._query.prepared(record)
        subject = prepared.document.subject
        try:
            equivalent = await self._experience_writer.find_current_equivalent(
                session=uow.session,
                owner_agent_id=request.owner_agent_id,
                content_hash=subject.content_hash,
            )
        except MultipleResultsFound:
            raise _replayable(PassportError("equivalent_ambiguous")) from None
        if equivalent is not None:
            if equivalent.temperature is Temperature.ARCHIVED:
                raise _replayable(PassportError("restore_required"))
            try:
                version = await ExperienceQuery().get_owned_shareable_version(
                    session=uow.session,
                    owner_agent_id=request.owner_agent_id,
                    experience_id=equivalent.experience_id,
                    version_id=equivalent.current_version_id,
                )
                if (
                    version.latest_causal_at > decided_at
                    or version.content_hash != subject.content_hash
                    or version.kind != subject.kind
                ):
                    raise PassportError("invalid")
            except (DomainError, SourceIntegrityError, TypeError, ValueError):
                raise PassportError("invalid") from None
        adoption_id = self._id_generator.new()
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_adoption",
            resource_id=adoption_id,
        )
        if equivalent is None:
            creation = await self._experience_writer.create_from_draft(
                uow=uow,
                draft=ExperienceDraft(
                    owner_agent_id=request.owner_agent_id,
                    actor_agent_id=request.owner_agent_id,
                    kind=subject.kind,
                    origin=ExperienceOrigin.ADOPTED_PASSPORT,
                    content=subject.content,
                    importance=request.importance,
                    confidence=request.confidence,
                    source_trust=0.25,
                    initial_temperature=Temperature.WARM,
                    links=(),
                    occurred_at=decided_at,
                ),
                command=command,
            )
            result = ExperienceRecord(
                creation.experience_id,
                request.owner_agent_id,
                creation.version_id,
                creation.content_hash,
                Temperature.WARM,
            )
        else:
            result = equivalent
        if result.current_content_hash != subject.content_hash:
            raise PassportError("invalid")
        created = equivalent is None
        uow.session.add(
            PassportAdoptionRow(
                adoption_id=adoption_id,
                import_id=request.import_id,
                owner_agent_id=request.owner_agent_id,
                resulting_experience_id=result.experience_id,
                resulting_version_id=result.current_version_id,
                resulting_content_hash=result.current_content_hash,
                created=created,
                importance=request.importance,
                confidence=request.confidence,
                adopted_at=decided_at,
            )
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="passport_import",
                    aggregate_id=request.import_id,
                    event_type=PassportAdoptedV1.event_type,
                    payload=PassportAdoptedV1(
                        schema_version=1,
                        import_id=request.import_id,
                        owner_agent_id=request.owner_agent_id,
                        state_before=PassportState.PENDING,
                        state_after=PassportState.ADOPTED,
                        adoption_id=adoption_id,
                        resulting_experience_id=result.experience_id,
                        resulting_version_id=result.current_version_id,
                        resulting_content_hash=result.current_content_hash,
                        created=created,
                        importance=request.importance,
                        confidence=request.confidence,
                    ),
                    actor_agent_id=request.owner_agent_id,
                    occurred_at=decided_at,
                ),
            ),
        )
        return passport_adoption_response(
            adoption_id=adoption_id, experience=result, created=created
        )

    async def reject(
        self,
        *,
        uow: UnitOfWork,
        request: RejectPassport,
        command: CommandContext,
    ) -> StoredResponse:
        self._caller(command, request.owner_agent_id, "passport.reject")
        request = RejectPassport(
            request.owner_agent_id, request.import_id, request.reason
        )
        expected = passport_reject_request(
            owner_agent_id=request.owner_agent_id,
            import_id=request.import_id,
            reason=request.reason,
            idempotency_key=command.idempotency_key,
        )
        decided_at = await self._operation_time(uow, command, expected)
        record = await self._pending(
            uow, request.owner_agent_id, request.import_id, decided_at
        )
        passport_hash = record.source.passport_hash
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_import",
            resource_id=request.import_id,
        )
        # Rejection has no separate source row; retain the pre-ledger fault boundary.
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="passport_import",
                    aggregate_id=request.import_id,
                    event_type=PassportRejectedV1.event_type,
                    payload=PassportRejectedV1(
                        schema_version=1,
                        import_id=request.import_id,
                        owner_agent_id=request.owner_agent_id,
                        state_before=PassportState.PENDING,
                        state_after=PassportState.REJECTED,
                        reason=request.reason,
                    ),
                    actor_agent_id=request.owner_agent_id,
                    occurred_at=decided_at,
                ),
            ),
        )
        return passport_import_response(
            import_id=request.import_id,
            owner_agent_id=request.owner_agent_id,
            passport_hash=passport_hash,
            state=PassportState.REJECTED,
        )

    async def _pending(
        self,
        uow: UnitOfWork,
        owner: UUID,
        import_id: UUID,
        decided_at: datetime,
    ) -> PassportRecord:
        await self._repository.require_owner(session=uow.session, owner_agent_id=owner)
        record = await self._repository.find_owned(
            session=uow.session, owner_agent_id=owner, import_id=import_id
        )
        if record is None:
            raise PassportError("not_found")
        view = await self._query.view(session=uow.session, record=record)
        if view.state is not PassportState.PENDING:
            raise _replayable(PassportError("decision_conflict"))
        if decided_at < view.imported_at:
            raise PassportError("invalid")
        return record

    async def import_passport(
        self,
        *,
        uow: UnitOfWork,
        request: ImportPassport,
        command: CommandContext,
    ) -> StoredResponse:
        self._caller(command, request.owner_agent_id, "passport.import")
        request = ImportPassport(request.owner_agent_id, request.prepared)
        expected = passport_import_request(
            owner_agent_id=request.owner_agent_id,
            passport_hash=request.prepared.document.passport_hash,
            idempotency_key=command.idempotency_key,
        )
        imported_at = await self._operation_time(uow, command, expected)
        owner_created_at = await self._repository.require_owner(
            session=uow.session, owner_agent_id=request.owner_agent_id
        )
        if imported_at < owner_created_at:
            raise PassportError("invalid")
        existing = await self._repository.find_by_hash(
            session=uow.session,
            owner_agent_id=request.owner_agent_id,
            passport_hash=request.prepared.document.passport_hash,
        )
        if existing is not None:
            view = await self._query.view(session=uow.session, record=existing)
            if (
                existing.source.canonical_bytes != request.prepared.canonical_bytes
                or imported_at
                < max(view.imported_at, view.decided_at or view.imported_at)
            ):
                raise PassportError("invalid")
            import_id, state, status = view.import_id, view.state, 200
        else:
            import_id = self._id_generator.new()
            state, status = PassportState.PENDING, 201
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_import",
            resource_id=import_id,
        )
        if existing is None:
            uow.session.add(
                PassportImportRow(
                    import_id=import_id,
                    owner_agent_id=request.owner_agent_id,
                    passport_hash=request.prepared.document.passport_hash,
                    canonical_bytes=request.prepared.canonical_bytes,
                    imported_at=imported_at,
                )
            )
            await uow.session.flush()
            uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
            await uow.append_events(
                command,
                (
                    PendingEvent(
                        aggregate_type="passport_import",
                        aggregate_id=import_id,
                        event_type=PassportImportedV1.event_type,
                        payload=PassportImportedV1(
                            schema_version=1,
                            import_id=import_id,
                            owner_agent_id=request.owner_agent_id,
                            passport_hash=request.prepared.document.passport_hash,
                            state_after=PassportState.PENDING,
                        ),
                        actor_agent_id=request.owner_agent_id,
                        occurred_at=imported_at,
                    ),
                ),
            )
        return passport_import_response(
            import_id=import_id,
            owner_agent_id=request.owner_agent_id,
            passport_hash=request.prepared.document.passport_hash,
            state=state,
            status_code=status,
        )

    @staticmethod
    def _caller(command: CommandContext, owner: UUID, scope: str) -> None:
        if command.caller_scope != f"agent:{owner}" or command.operation_scope != scope:
            raise PassportError("not_found")

    async def _operation_time(
        self,
        uow: UnitOfWork,
        command: CommandContext,
        expected: CommandRequest,
    ) -> datetime:
        if not uow.immediate:
            raise RuntimeError("Passport commands require an immediate transaction")
        if command.request_hash != expected.request_hash:
            raise PassportError("invalid")
        receipt = await self._receipt_store.get_by_id(
            session=uow.session, receipt_id=command.receipt_id
        )
        if (
            receipt is None
            or receipt.state != "in_progress"
            or receipt.caller_scope != command.caller_scope
            or receipt.operation_scope != command.operation_scope
            or receipt.idempotency_key != command.idempotency_key
            or receipt.request_hash != command.request_hash
            or receipt.result_resource_type is not None
            or receipt.result_resource_id is not None
        ):
            raise PassportError("invalid")
        occurred_at = require_utc(receipt.created_at)
        if occurred_at > require_utc(self._clock.now()):
            raise PassportError("invalid")
        return occurred_at


def _replayable(error: PassportError) -> ReplayableCommandError:
    return ReplayableCommandError(
        code=error.code, message=error.message, status_code=error.status_code
    )
