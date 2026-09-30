from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import URL
from tests.passport_fixtures import passport_document

from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    StructuredReason,
)
from experience_hub.experiences.contracts import ExperienceDraft, ExperienceRecord
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.ids import SequenceIdGenerator
from experience_hub.passports import PassportState, encode_passport_document
from experience_hub.passports.events import (
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
from experience_hub.passports.validation import PassportSourceValidator
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import (
    AgentRow,
    PassportAdoptionRow,
    PassportImportRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

NOW = datetime(2026, 9, 30, 8, tzinfo=UTC)


@pytest.fixture
async def container(tmp_path: Path) -> AsyncIterator[ApplicationContainer]:
    settings = Settings(
        database_url=URL.create(
            "sqlite+aiosqlite", database=str(tmp_path / "passport.sqlite3")
        )
    )
    runtime = ApplicationRuntime(
        settings=settings,
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(tuple(UUID(int=value) for value in range(100, 300))),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as value:

        async def create(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
            return await value.agent_service.create(
                uow=uow,
                command=CreateAgent(name="Synthetic recipient"),
                command_context=command,
            )

        result = await value.command_executor.execute(
            CommandRequest(
                caller_scope="system:local",
                operation_scope="agent.create",
                idempotency_key="recipient",
                method="POST",
                route_template="/v1/agents",
                body={"name": "Synthetic recipient"},
            ),
            create,
        )
        assert result.status_code == 201
        yield value


async def _seed_import(container: ApplicationContainer) -> UUID:
    async with container.database.read_session() as session:
        owner = await session.scalar(select(AgentRow.agent_id))
    assert owner is not None
    document = passport_document()
    import_id = container.ids.new()

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        uow.session.add(
            PassportImportRow(
                import_id=import_id,
                owner_agent_id=owner,
                passport_hash=document.passport_hash,
                canonical_bytes=encode_passport_document(document),
                imported_at=NOW,
            )
        )
        await uow.session.flush()
        await container.receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_import",
            resource_id=import_id,
        )
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
                        owner_agent_id=owner,
                        passport_hash=document.passport_hash,
                        state_after=PassportState.PENDING,
                    ),
                    actor_agent_id=owner,
                    occurred_at=NOW,
                ),
            ),
        )
        return passport_import_response(
            import_id=import_id,
            owner_agent_id=owner,
            passport_hash=document.passport_hash,
            state=PassportState.PENDING,
            status_code=201,
        )

    result = await container.command_executor.execute(
        passport_import_request(
            owner_agent_id=owner,
            passport_hash=document.passport_hash,
            idempotency_key="import",
        ),
        handler,
    )
    assert result.status_code == 201
    return import_id


async def test_valid_pending_source_uses_foreign_ids_only_as_provenance(
    container: ApplicationContainer,
) -> None:
    await _seed_import(container)
    async with container.database.read_session() as session:
        await PassportSourceValidator(container.event_registry).validate(session)
        await container.source_validator.validate(session)


@pytest.mark.parametrize(
    "column",
    [
        "passport_hash",
        "canonical_bytes",
        "imported_at",
    ],
)
async def test_owned_source_damage_is_not_silently_repaired(
    container: ApplicationContainer,
    column: str,
) -> None:
    await _seed_import(container)
    changes = {
        "passport_hash": "'" + "f" * 64 + "'",
        "canonical_bytes": "CAST('{}' AS BLOB)",
        "imported_at": "'2026-09-30T09:00:00.000000Z'",
    }
    async with container.database._engine.begin() as connection:
        await connection.execute(text("DROP TRIGGER passport_imports_reject_update"))
        await connection.execute(
            text(f"UPDATE passport_imports SET {column}={changes[column]}")
        )
    async with container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="Passport source graph"):
            await PassportSourceValidator(container.event_registry).validate(session)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_event",
        "wrong_actor",
        "wrong_aggregate",
        "wrong_time",
        "wrong_causation_scope",
        "wrong_request_hash",
        "wrong_response",
    ],
)
async def test_source_and_receipt_graph_mismatches_fail_closed(
    container: ApplicationContainer,
    mutation: str,
) -> None:
    await _seed_import(container)
    async with container.database._engine.begin() as connection:
        if mutation == "missing_event":
            await connection.execute(text("DELETE FROM passport_state"))
            await connection.execute(text("DROP TRIGGER domain_events_reject_delete"))
            sql = "DELETE FROM domain_events WHERE event_type='passport.imported'"
        elif mutation.startswith("wrong_causation"):
            sql = (
                "UPDATE idempotency_records SET scope='passport.reject' "
                "WHERE scope='passport.import'"
            )
        elif mutation == "wrong_request_hash":
            sql = (
                "UPDATE idempotency_records SET request_hash='"
                + "f" * 64
                + "' WHERE scope='passport.import'"
            )
        elif mutation == "wrong_response":
            sql = (
                "UPDATE idempotency_records SET response_body=CAST('{}' AS BLOB) "
                "WHERE scope='passport.import'"
            )
        else:
            await connection.execute(text("DROP TRIGGER domain_events_reject_update"))
            changes = {
                "wrong_actor": "actor_agent_id=NULL",
                "wrong_aggregate": "aggregate_type='wrong'",
                "wrong_time": "occurred_at='2026-09-30T09:00:00.000000Z'",
            }
            sql = (
                "UPDATE domain_events SET "
                + changes[mutation]
                + " WHERE event_type='passport.imported'"
            )
        await connection.execute(text(sql))
    async with container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="Passport source graph"):
            await PassportSourceValidator(container.event_registry).validate(session)


async def _seed_decision(container: ApplicationContainer, terminal: str) -> UUID:
    import_id = await _seed_import(container)
    async with container.database.read_session() as session:
        source = await session.get(PassportImportRow, import_id)
        assert source is not None
        owner, passport_hash = source.owner_agent_id, source.passport_hash
    subject = passport_document().subject
    reason = StructuredReason.from_user_text("Synthetic rejection")

    async def draft(
        uow: UnitOfWork, command: CommandContext, origin: ExperienceOrigin
    ) -> ExperienceRecord:
        created = await container.experience_writer.create_from_draft(
            uow=uow,
            command=command,
            draft=ExperienceDraft(
                owner_agent_id=owner,
                actor_agent_id=owner,
                kind=subject.kind,
                origin=origin,
                content=subject.content,
                importance=0.7,
                confidence=0.6,
                source_trust=0.25,
                initial_temperature=Temperature.WARM,
                links=(),
                occurred_at=NOW,
            ),
        )
        return ExperienceRecord(
            experience_id=created.experience_id,
            owner_agent_id=owner,
            current_version_id=created.version_id,
            current_content_hash=created.content_hash,
            temperature=Temperature.WARM,
        )

    equivalent: ExperienceRecord | None = None
    if terminal == "reused":

        async def local_handler(
            uow: UnitOfWork, command: CommandContext
        ) -> StoredResponse:
            nonlocal equivalent
            equivalent = await draft(uow, command, ExperienceOrigin.LOCAL)
            return StoredResponse(status_code=201, body=b"{}")

        await container.command_executor.execute(
            CommandRequest(
                caller_scope=f"agent:{owner}",
                operation_scope="experience.create",
                idempotency_key="equivalent",
                method="POST",
                route_template="/synthetic-experience",
                body={},
            ),
            local_handler,
        )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        if terminal == "rejected":
            await container.receipt_store.attach_resource(
                uow=uow,
                receipt_id=command.receipt_id,
                resource_type="passport_import",
                resource_id=import_id,
            )
            await uow.append_events(
                command,
                (
                    PendingEvent(
                        aggregate_type="passport_import",
                        aggregate_id=import_id,
                        event_type=PassportRejectedV1.event_type,
                        actor_agent_id=owner,
                        occurred_at=NOW,
                        payload=PassportRejectedV1(
                            schema_version=1,
                            import_id=import_id,
                            owner_agent_id=owner,
                            state_before=PassportState.PENDING,
                            state_after=PassportState.REJECTED,
                            reason=reason,
                        ),
                    ),
                ),
            )
            return passport_import_response(
                import_id=import_id,
                owner_agent_id=owner,
                passport_hash=passport_hash,
                state=PassportState.REJECTED,
            )
        record = equivalent
        if record is None:
            record = await draft(uow, command, ExperienceOrigin.ADOPTED_PASSPORT)
        adoption_id = container.ids.new()
        await container.receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_adoption",
            resource_id=adoption_id,
        )
        uow.session.add(
            PassportAdoptionRow(
                adoption_id=adoption_id,
                import_id=import_id,
                owner_agent_id=owner,
                resulting_experience_id=record.experience_id,
                resulting_version_id=record.current_version_id,
                resulting_content_hash=record.current_content_hash,
                created=terminal == "created",
                importance=0.7,
                confidence=0.6,
                adopted_at=NOW,
            )
        )
        await uow.session.flush()
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="passport_import",
                    aggregate_id=import_id,
                    event_type=PassportAdoptedV1.event_type,
                    actor_agent_id=owner,
                    occurred_at=NOW,
                    payload=PassportAdoptedV1(
                        schema_version=1,
                        import_id=import_id,
                        owner_agent_id=owner,
                        state_before=PassportState.PENDING,
                        state_after=PassportState.ADOPTED,
                        adoption_id=adoption_id,
                        resulting_experience_id=record.experience_id,
                        resulting_version_id=record.current_version_id,
                        resulting_content_hash=record.current_content_hash,
                        created=terminal == "created",
                        importance=0.7,
                        confidence=0.6,
                    ),
                ),
            ),
        )
        return passport_adoption_response(
            adoption_id=adoption_id,
            experience=record,
            created=terminal == "created",
        )

    request = (
        passport_reject_request(
            owner_agent_id=owner,
            import_id=import_id,
            reason=reason,
            idempotency_key="decision",
        )
        if terminal == "rejected"
        else passport_adopt_request(
            owner_agent_id=owner,
            import_id=import_id,
            importance=0.7,
            confidence=0.6,
            idempotency_key="decision",
        )
    )
    result = await container.command_executor.execute(request, handler)
    assert result.status_code == 200
    return import_id


@pytest.mark.parametrize("terminal", ["created", "reused", "rejected"])
async def test_valid_complete_graph_is_rebuildable(
    container: ApplicationContainer, terminal: str
) -> None:
    await _seed_decision(container, terminal)
    async with container.database.read_session() as session:
        await PassportSourceValidator(container.event_registry).validate(session)
        await container.source_validator.validate(session)


@pytest.mark.parametrize("terminal", ["created", "reused", "rejected"])
async def test_terminal_graph_without_decision_event_is_invalid(
    container: ApplicationContainer, terminal: str
) -> None:
    await _seed_decision(container, terminal)
    async with container.database._engine.begin() as connection:
        await connection.execute(text("DELETE FROM passport_state"))
        await connection.execute(text("DROP TRIGGER domain_events_reject_delete"))
        await connection.execute(
            text(
                "DELETE FROM domain_events WHERE event_type IN "
                "('passport.adopted','passport.rejected')"
            )
        )
    async with container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="Passport source graph"):
            await PassportSourceValidator(container.event_registry).validate(session)


@pytest.mark.parametrize(
    "mutation",
    ["scores", "time", "origin", "receipt", "creation_payload", "orphan"],
)
async def test_created_adoption_graph_rejects_lineage_damage(
    container: ApplicationContainer, mutation: str
) -> None:
    await _seed_decision(container, "created")
    async with container.database._engine.begin() as connection:
        if mutation in {"scores", "time"}:
            await connection.execute(
                text("DROP TRIGGER passport_adoptions_reject_update")
            )
            change = (
                "importance=0.8"
                if mutation == "scores"
                else "adopted_at='2026-09-30T09:00:00.000000Z'"
            )
            await connection.execute(text("UPDATE passport_adoptions SET " + change))
        elif mutation == "origin":
            await connection.execute(text("DROP TRIGGER experiences_reject_update"))
            await connection.execute(text("UPDATE experiences SET origin='local'"))
        elif mutation == "receipt":
            await connection.execute(
                text(
                    "UPDATE idempotency_records SET response_body=CAST('{}' AS BLOB) "
                    "WHERE scope='passport.adopt'"
                )
            )
        elif mutation == "creation_payload":
            await connection.execute(text("DROP TRIGGER domain_events_reject_update"))
            await connection.execute(
                text(
                    "UPDATE domain_events SET payload=CAST(json_set(payload, "
                    "'$.after.source_trust',0.9) AS BLOB) "
                    "WHERE event_type='experience.created'"
                )
            )
        else:
            await connection.execute(text("DELETE FROM passport_state"))
            await connection.execute(
                text("DROP TRIGGER passport_adoptions_reject_delete")
            )
            await connection.execute(text("DROP TRIGGER domain_events_reject_delete"))
            await connection.execute(text("DELETE FROM passport_adoptions"))
            await connection.execute(
                text("DELETE FROM domain_events WHERE event_type='passport.adopted'")
            )
            # Remove the decision receipt too: only the unexplained origin remains.
            await connection.execute(
                text(
                    "UPDATE idempotency_records SET response_status_code=409, "
                    "result_resource_type=NULL,result_resource_id=NULL "
                    "WHERE scope='passport.adopt'"
                )
            )
    async with container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="Passport source graph"):
            await PassportSourceValidator(container.event_registry).validate(session)
