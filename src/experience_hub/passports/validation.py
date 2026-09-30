"""Reconcile Passport sources with their immutable event and receipt graph."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import canonical_json_bytes
from experience_hub.domain import CommandRequest, EventPayload, EventRegistry
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.experiences.repository import decode_and_verify_version
from experience_hub.passports import (
    PassportState,
    VerifiedPassportV1,
    verify_passport_bytes,
)
from experience_hub.passports.errors import PassportError
from experience_hub.passports.events import (
    PASSPORT_EVENT_TYPES,
    PassportAdoptedV1,
    PassportImportedV1,
    PassportRejectedV1,
)
from experience_hub.passports.requests import (
    passport_adopt_request,
    passport_import_request,
    passport_reject_request,
)
from experience_hub.passports.responses import (
    passport_adoption_response,
    passport_import_response,
)
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
    PassportAdoptionRow,
    PassportImportRow,
)
from experience_hub.storage.validation import SourceIntegrityError, SourceValidator

type PassportPayload = PassportImportedV1 | PassportAdoptedV1 | PassportRejectedV1
type PassportEvent = tuple[DomainEventRow, PassportPayload]


def _fail() -> SourceIntegrityError:
    # Graph-wide diagnostics must not expose another owner's identifiers.
    return SourceIntegrityError(
        "Passport source graph is inconsistent", mismatch_key="passport_graph"
    )


class PassportSourceValidator:
    name = "passport_graph"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        imports = tuple((await session.scalars(select(PassportImportRow))).all())
        adoptions = tuple((await session.scalars(select(PassportAdoptionRow))).all())
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        by_import: dict[UUID, list[PassportEvent]] = defaultdict(list)
        by_causation: dict[UUID, list[DomainEventRow]] = defaultdict(list)
        for row in rows:
            by_causation[row.causation_id].append(row)
            if (
                row.aggregate_type != "passport_import"
                and not row.event_type.startswith("passport.")
            ):
                continue
            try:
                payload = self._registry.decode(
                    event_type=row.event_type, payload=row.payload
                )
                if not isinstance(
                    payload, (PassportImportedV1, PassportAdoptedV1, PassportRejectedV1)
                ):
                    raise ValueError("unexpected Passport payload")
                if (
                    row.event_type not in PASSPORT_EVENT_TYPES
                    or canonical_json_bytes(payload) != row.payload
                ):
                    raise ValueError("invalid Passport event encoding")
                by_import[payload.import_id].append((row, payload))
            except (ValueError, TypeError):
                raise _fail() from None
        adoption_by_import = {row.import_id: row for row in adoptions}
        known_imports = {row.import_id for row in imports}
        if (
            len(adoption_by_import) != len(adoptions)
            or set(adoption_by_import) - known_imports
            or set(by_import) != known_imports
            or len({(row.owner_agent_id, row.passport_hash) for row in imports})
            != len(imports)
        ):
            raise _fail()
        created_targets: dict[UUID, int] = defaultdict(int)
        for source in imports:
            try:
                prepared = verify_passport_bytes(source.canonical_bytes)
                if prepared.document.passport_hash != source.passport_hash:
                    raise _fail()
                events = by_import[source.import_id]
                if not 1 <= len(events) <= 2:
                    raise _fail()
                for sequence, (row, payload) in enumerate(events, 1):
                    if (
                        row.aggregate_type != "passport_import"
                        or row.aggregate_id != source.import_id
                        or row.sequence != sequence
                        or row.actor_agent_id != source.owner_agent_id
                        or payload.owner_agent_id != source.owner_agent_id
                        or row.occurred_at < source.imported_at
                    ):
                        raise _fail()
                row, imported = events[0]
                if (
                    not isinstance(imported, PassportImportedV1)
                    or row.occurred_at != source.imported_at
                    or imported.passport_hash != source.passport_hash
                    or tuple(by_causation[row.causation_id]) != (row,)
                ):
                    raise _fail()
                receipt = await self._receipt(session, row, source.owner_agent_id)
                await self._require_result(
                    receipt=receipt,
                    request=passport_import_request(
                        owner_agent_id=source.owner_agent_id,
                        passport_hash=source.passport_hash,
                        idempotency_key=receipt.idempotency_key,
                    ),
                    resource_type="passport_import",
                    resource_id=source.import_id,
                    response=passport_import_response(
                        import_id=source.import_id,
                        owner_agent_id=source.owner_agent_id,
                        passport_hash=source.passport_hash,
                        state=PassportState.PENDING,
                        status_code=201,
                    ),
                )
                adoption = adoption_by_import.get(source.import_id)
                if len(events) == 1:
                    if adoption is not None:
                        raise _fail()
                    continue
                decision_row, decision = events[1]
                if isinstance(decision, PassportAdoptedV1):
                    await self._adoption(
                        session,
                        source,
                        prepared,
                        adoption,
                        decision_row,
                        decision,
                        rows,
                        tuple(by_causation[decision_row.causation_id]),
                    )
                    if adoption is not None and adoption.created:
                        created_targets[adoption.resulting_experience_id] += 1
                elif isinstance(decision, PassportRejectedV1):
                    if adoption is not None or tuple(
                        by_causation[decision_row.causation_id]
                    ) != (decision_row,):
                        raise _fail()
                    receipt = await self._receipt(
                        session, decision_row, source.owner_agent_id
                    )
                    await self._require_result(
                        receipt=receipt,
                        request=passport_reject_request(
                            owner_agent_id=source.owner_agent_id,
                            import_id=source.import_id,
                            reason=decision.reason,
                            idempotency_key=receipt.idempotency_key,
                        ),
                        resource_type="passport_import",
                        resource_id=source.import_id,
                        response=passport_import_response(
                            import_id=source.import_id,
                            owner_agent_id=source.owner_agent_id,
                            passport_hash=source.passport_hash,
                            state=PassportState.REJECTED,
                        ),
                    )
                else:
                    raise _fail()
            except (PassportError, ValueError, TypeError, AttributeError):
                raise _fail() from None
        await self._receipt_coverage(session, imports, adoptions, by_import)
        identities = tuple(
            (
                await session.scalars(
                    select(ExperienceRow).where(
                        ExperienceRow.origin == ExperienceOrigin.ADOPTED_PASSPORT
                    )
                )
            ).all()
        )
        if {identity.experience_id for identity in identities} != set(
            created_targets
        ) or any(count != 1 for count in created_targets.values()):
            raise _fail()

    async def _receipt_coverage(
        self,
        session: AsyncSession,
        imports: tuple[PassportImportRow, ...],
        adoptions: tuple[PassportAdoptionRow, ...],
        by_import: dict[UUID, list[PassportEvent]],
    ) -> None:
        sources = {row.import_id: row for row in imports}
        decisions = {row.adoption_id: row for row in adoptions}
        receipts = tuple(
            (
                await session.scalars(
                    select(IdempotencyRecordRow).where(
                        IdempotencyRecordRow.scope.in_(
                            ("passport.import", "passport.adopt", "passport.reject")
                        ),
                        IdempotencyRecordRow.state == "completed",
                    )
                )
            ).all()
        )
        for receipt in receipts:
            if receipt.result_resource_id is None:
                if (
                    receipt.result_resource_type is not None
                    or receipt.response_status_code is None
                    or receipt.response_status_code < 400
                ):
                    raise _fail()
                # Replayed domain refusals have no durable mutation resource.
                continue
            identifier = receipt.result_resource_id
            if receipt.scope == "passport.adopt":
                adoption = decisions.get(identifier)
                if adoption is None:
                    raise _fail()
                source = sources.get(adoption.import_id)
            else:
                source = sources.get(identifier)
            if (
                source is None
                or receipt.caller_scope != f"agent:{source.owner_agent_id}"
            ):
                raise _fail()
            events = by_import[source.import_id]
            anchored = any(row.causation_id == receipt.receipt_id for row, _ in events)
            if receipt.scope != "passport.import":
                if not anchored:
                    raise _fail()
                continue
            if anchored:
                continue
            if (
                receipt.created_at < source.imported_at
                or receipt.completed_at is None
                or receipt.completed_at < receipt.created_at
            ):
                raise _fail()
            # Tied injected clocks do not order eventless dedup receipts. Allow
            # either state at the tied instant, but require exact canonical bytes.
            states = [PassportState.PENDING]
            if len(events) == 2:
                event, payload = events[1]
                terminal = (
                    PassportState.ADOPTED
                    if isinstance(payload, PassportAdoptedV1)
                    else PassportState.REJECTED
                )
                if event.occurred_at <= receipt.created_at:
                    states.append(terminal)
                if event.occurred_at < receipt.created_at:
                    states.remove(PassportState.PENDING)
            matches = [
                passport_import_response(
                    import_id=source.import_id,
                    owner_agent_id=source.owner_agent_id,
                    passport_hash=source.passport_hash,
                    state=state,
                )
                for state in states
            ]
            response = next(
                (item for item in matches if item.body == receipt.response_body), None
            )
            if response is None:
                raise _fail()
            await self._require_result(
                receipt=receipt,
                request=passport_import_request(
                    owner_agent_id=source.owner_agent_id,
                    passport_hash=source.passport_hash,
                    idempotency_key=receipt.idempotency_key,
                ),
                resource_type="passport_import",
                resource_id=source.import_id,
                response=response,
            )

    async def _receipt(
        self,
        session: AsyncSession,
        event: DomainEventRow,
        owner: UUID,
    ) -> IdempotencyRecordRow:
        receipt = await session.get(IdempotencyRecordRow, event.causation_id)
        if (
            receipt is None
            or receipt.state != "completed"
            or receipt.caller_scope != f"agent:{owner}"
            or receipt.created_at != event.occurred_at
            or receipt.completed_at is None
            or receipt.completed_at < event.occurred_at
        ):
            raise _fail()
        return receipt

    async def _require_result(
        self,
        *,
        receipt: IdempotencyRecordRow,
        request: CommandRequest,
        resource_type: str,
        resource_id: UUID,
        response: StoredResponse,
    ) -> None:
        if (
            receipt.scope != request.operation_scope
            or receipt.request_hash != request.request_hash
            or receipt.result_resource_type != resource_type
            or receipt.result_resource_id != resource_id
            or receipt.response_status_code != response.status_code
            or receipt.response_body != response.body
            or receipt.response_content_type != response.content_type
            or receipt.response_headers
            != canonical_json_bytes(dict(response.headers or {}))
        ):
            raise _fail()

    async def _adoption(
        self,
        session: AsyncSession,
        source: PassportImportRow,
        prepared: VerifiedPassportV1,
        adoption: PassportAdoptionRow | None,
        event: DomainEventRow,
        decision: PassportAdoptedV1,
        rows: tuple[DomainEventRow, ...],
        causal: tuple[DomainEventRow, ...],
    ) -> None:
        if (
            adoption is None
            or adoption.owner_agent_id != source.owner_agent_id
            or adoption.adoption_id != decision.adoption_id
            or adoption.resulting_experience_id != decision.resulting_experience_id
            or adoption.resulting_version_id != decision.resulting_version_id
            or adoption.resulting_content_hash != decision.resulting_content_hash
            or adoption.resulting_content_hash != prepared.document.subject.content_hash
            or adoption.created is not decision.created
            or adoption.importance != decision.importance
            or adoption.confidence != decision.confidence
            or adoption.adopted_at != event.occurred_at
        ):
            raise _fail()
        identity = await session.get(ExperienceRow, adoption.resulting_experience_id)
        version = await session.get(ExperienceVersionRow, adoption.resulting_version_id)
        payload = await session.get(ExperiencePayloadRow, adoption.resulting_version_id)
        if (
            identity is None
            or version is None
            or payload is None
            or identity.owner_agent_id != source.owner_agent_id
            or identity.kind != prepared.document.subject.kind
            or version.experience_id != identity.experience_id
            or version.content_hash != adoption.resulting_content_hash
            or identity.created_at > event.occurred_at
            or version.created_at > event.occurred_at
            or decode_and_verify_version(
                identity=identity, version=version, payload=payload
            )
            != prepared.document.subject.content
        ):
            raise _fail()
        if adoption.created:
            expected = (
                ExperienceCreatedV1.event_type,
                ExperienceVersionCreatedV1.event_type,
                PassportAdoptedV1.event_type,
            )
            if (
                tuple(row.event_type for row in causal) != expected
                or identity.origin != ExperienceOrigin.ADOPTED_PASSPORT
                or identity.created_at != event.occurred_at
                or version.created_at != event.occurred_at
                or version.version_number != 1
                or version.supersedes_version_id is not None
            ):
                raise _fail()
            first, second, _ = causal
            created = self._registry.decode(
                event_type=first.event_type, payload=first.payload
            )
            version_created = self._registry.decode(
                event_type=second.event_type, payload=second.payload
            )
            if (
                not isinstance(created, ExperienceCreatedV1)
                or not isinstance(version_created, ExperienceVersionCreatedV1)
                or any(
                    row.aggregate_type != "experience"
                    or row.aggregate_id != identity.experience_id
                    or row.actor_agent_id != source.owner_agent_id
                    or row.occurred_at != event.occurred_at
                    for row in (first, second)
                )
                or first.sequence != 1
                or second.sequence != 2
                or created.experience_id != identity.experience_id
                or created.version_id != version.version_id
                or created.after.owner_agent_id != source.owner_agent_id
                or created.after.current_content_hash != adoption.resulting_content_hash
                or created.after.source_trust != 0.25
                or created.after.temperature != Temperature.WARM
                or created.after.importance != adoption.importance
                or created.after.confidence != adoption.confidence
                or version_created.experience_id != identity.experience_id
                or version_created.version_id != version.version_id
                or version_created.version_number != 1
                or version_created.supersedes_version_id is not None
                or version_created.links
                or version_created.before != created.after
                or version_created.after != created.after
            ):
                raise _fail()
            temperature = Temperature.WARM
        else:
            if causal != (event,):
                raise _fail()
            snapshot = self._prior_snapshot(
                rows, identity.experience_id, event.event_id
            )
            if (
                snapshot is None
                or snapshot.owner_agent_id != source.owner_agent_id
                or snapshot.current_version_id != adoption.resulting_version_id
                or snapshot.current_content_hash != adoption.resulting_content_hash
                or snapshot.temperature == Temperature.ARCHIVED
            ):
                raise _fail()
            temperature = snapshot.temperature
        receipt = await self._receipt(session, event, source.owner_agent_id)
        await self._require_result(
            receipt=receipt,
            request=passport_adopt_request(
                owner_agent_id=source.owner_agent_id,
                import_id=source.import_id,
                importance=adoption.importance,
                confidence=adoption.confidence,
                idempotency_key=receipt.idempotency_key,
            ),
            resource_type="passport_adoption",
            resource_id=adoption.adoption_id,
            response=passport_adoption_response(
                adoption_id=adoption.adoption_id,
                experience=ExperienceRecord(
                    experience_id=identity.experience_id,
                    owner_agent_id=source.owner_agent_id,
                    current_version_id=version.version_id,
                    current_content_hash=adoption.resulting_content_hash,
                    temperature=temperature,
                ),
                created=adoption.created,
            ),
        )

    def _prior_snapshot(
        self,
        rows: tuple[DomainEventRow, ...],
        experience_id: UUID,
        before_event_id: int,
    ) -> ExperienceStateSnapshotV1 | None:
        result: ExperienceStateSnapshotV1 | None = None
        for row in rows:
            if row.event_id >= before_event_id:
                break
            if row.aggregate_type != "experience" or row.aggregate_id != experience_id:
                continue
            payload: EventPayload = self._registry.decode(
                event_type=row.event_type, payload=row.payload
            )
            after = getattr(payload, "after", None)
            if isinstance(after, ExperienceStateSnapshotV1):
                result = after
        return result


def register_passport_source_validator(validator: SourceValidator) -> None:
    validator.register(PassportSourceValidator(validator.event_registry))
