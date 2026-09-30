from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select, text, update
from tests.passport_service_fixtures import (
    OWNER,
    PassportStack,
    import_passport,
    prepared_passport,
    result_id,
)
from tests.passport_service_fixtures import (
    passport_stack as passport_stack,
)

from experience_hub.domain import CommandContext, PendingEvent, StructuredReason
from experience_hub.errors import DomainError
from experience_hub.passports.contracts import PassportState
from experience_hub.passports.errors import PassportError
from experience_hub.passports.events import PassportRejectedV1
from experience_hub.passports.requests import passport_reject_request
from experience_hub.passports.responses import passport_import_response
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceRow,
    ExperienceTermRow,
    IdempotencyRecordRow,
    PassportAdoptionRow,
    PassportImportRow,
    PassportStateRow,
)


async def test_import_is_quarantined_and_exact_request_replays(
    passport_stack: PassportStack,
) -> None:
    first = await import_passport(passport_stack)
    replay = await import_passport(passport_stack)
    assert first.status_code == 201
    assert first.replayed is False and replay.replayed is True
    assert first.body == replay.body
    async with passport_stack.container.database.read_session() as session:
        view = await passport_stack.query.get_owned(
            session=session, owner_agent_id=OWNER, import_id=result_id(first)
        )
        assert view.state is PassportState.PENDING
        assert view.document == prepared_passport().document
        for table, expected in (
            (PassportImportRow, 1),
            (PassportStateRow, 1),
            (PassportAdoptionRow, 0),
            (ExperienceRow, 0),
            (ExperienceTermRow, 0),
        ):
            assert (
                await session.scalar(select(func.count()).select_from(table))
                == expected
            )
        events = (
            await session.scalars(
                select(DomainEventRow).where(
                    DomainEventRow.aggregate_type == "passport_import"
                )
            )
        ).all()
        assert len(events) == 1 and events[0].sequence == 1
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "passport.import"
            )
        )
        assert receipt is not None and receipt.state == "completed"
        assert receipt.result_resource_id == result_id(first)
        await passport_stack.container.source_validator.validate(session)


async def test_new_key_dedup_retains_one_source_and_event(
    passport_stack: PassportStack,
) -> None:
    first = await import_passport(passport_stack)
    duplicate = await import_passport(passport_stack, key="different-key")
    assert duplicate.status_code == 200
    assert result_id(first) == result_id(duplicate)
    assert json.loads(duplicate.body)["data"]["state"] == "pending"
    async with passport_stack.container.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(PassportImportRow))
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.aggregate_type == "passport_import")
            )
            == 1
        )


async def test_changed_file_same_key_conflicts(passport_stack: PassportStack) -> None:
    await import_passport(passport_stack)
    with pytest.raises(DomainError) as error:
        await import_passport(passport_stack, prepared=prepared_passport("changed"))
    assert error.value.code == "idempotency_key_conflict"


@pytest.mark.parametrize("checkpoint", tuple(FaultCheckpoint))
async def test_import_all_checkpoints_rollback_and_same_key_retry(
    passport_stack: PassportStack, checkpoint: FaultCheckpoint
) -> None:
    passport_stack.fault.checkpoint = checkpoint
    with pytest.raises(RuntimeError, match="injected passport checkpoint"):
        await import_passport(passport_stack)
    async with passport_stack.container.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(PassportImportRow))
            == 0
        )
        assert (
            await session.scalar(select(func.count()).select_from(PassportStateRow))
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == "passport.import")
            )
            == 0
        )
    passport_stack.fault.checkpoint = None
    assert (await import_passport(passport_stack)).status_code == 201


async def test_import_database_busy_rolls_back_reservation_and_retries(
    passport_stack: PassportStack,
) -> None:
    database = passport_stack.container.database
    async with database.read_session() as session:
        await session.execute(text("PRAGMA busy_timeout=10"))
    path = database._engine.url.database
    assert path is not None
    lock = sqlite3.connect(path)
    lock.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(DomainError) as error:
            await import_passport(passport_stack)
        assert error.value.code == "database_busy"
    finally:
        lock.rollback()
        lock.close()
    async with database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == "passport.import")
            )
            == 0
        )
    assert (await import_passport(passport_stack)).status_code == 201


async def test_new_key_reimport_preserves_terminal_state(
    passport_stack: PassportStack,
) -> None:
    first = await import_passport(passport_stack)
    identifier = result_id(first)
    reason = StructuredReason.from_user_text("Not applicable to this owner.")
    request = passport_reject_request(
        owner_agent_id=OWNER,
        import_id=identifier,
        reason=reason,
        idempotency_key="seed-rejection",
    )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        await passport_stack.container.receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="passport_import",
            resource_id=identifier,
        )
        await uow.append_events(
            command,
            (
                PendingEvent(
                    aggregate_type="passport_import",
                    aggregate_id=identifier,
                    event_type=PassportRejectedV1.event_type,
                    payload=PassportRejectedV1(
                        schema_version=1,
                        import_id=identifier,
                        owner_agent_id=OWNER,
                        state_before=PassportState.PENDING,
                        state_after=PassportState.REJECTED,
                        reason=reason,
                    ),
                    actor_agent_id=OWNER,
                    occurred_at=passport_stack.clock.now(),
                ),
            ),
        )
        return passport_import_response(
            import_id=identifier,
            owner_agent_id=OWNER,
            passport_hash=prepared_passport().document.passport_hash,
            state=PassportState.REJECTED,
        )

    assert (
        await passport_stack.container.command_executor.execute(request, handler)
    ).status_code == 200
    duplicate = await import_passport(passport_stack, key="after-rejection")
    assert duplicate.status_code == 200 and result_id(duplicate) == identifier
    assert json.loads(duplicate.body)["data"]["state"] == "rejected"
    async with passport_stack.container.database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(DomainEventRow)
                .where(DomainEventRow.aggregate_type == "passport_import")
            )
            == 2
        )
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceRow)) == 0
        )
        await passport_stack.container.source_validator.validate(session)


async def test_forged_prepared_is_rejected_again_at_service_boundary(
    passport_stack: PassportStack,
) -> None:
    from experience_hub.passports.contracts import ImportPassport
    from experience_hub.passports.requests import passport_import_request

    prepared = prepared_passport()
    value = ImportPassport(OWNER, prepared)
    request = passport_import_request(
        owner_agent_id=OWNER,
        passport_hash=prepared.document.passport_hash,
        idempotency_key="forged",
    )
    object.__setattr__(value.prepared, "canonical_bytes", b"{}")

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await passport_stack.service.import_passport(
            uow=uow, request=value, command=command
        )

    with pytest.raises(PassportError) as error:
        await passport_stack.container.command_executor.execute(request, handler)
    assert error.value.code == "passport_invalid"


async def test_request_hash_must_bind_prepared_document(
    passport_stack: PassportStack,
) -> None:
    from experience_hub.passports.requests import passport_import_request

    incorrect = passport_import_request(
        owner_agent_id=OWNER, passport_hash="a" * 64, idempotency_key="incorrect"
    )
    with pytest.raises(PassportError) as error:
        await import_passport(passport_stack, command_request=incorrect)
    assert error.value.code == "passport_invalid"


async def test_owned_query_and_dedup_revalidate_import_receipt_anchor(
    passport_stack: PassportStack,
) -> None:
    first = await import_passport(passport_stack)
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.scope == "passport.import")
            .values(request_hash="b" * 64)
        )
    with pytest.raises(PassportError) as error:
        await import_passport(passport_stack, key="after-receipt-corruption")
    assert error.value.code == "passport_invalid"
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(PassportError) as read_error:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=result_id(first)
            )
        assert read_error.value.code == "passport_invalid"


@pytest.mark.parametrize(
    "change", ("receipt_id", "idempotency_key", "request_hash", "resource")
)
async def test_full_receipt_context_must_bind_import(
    passport_stack: PassportStack,
    change: str,
) -> None:
    from experience_hub.passports.contracts import ImportPassport
    from experience_hub.passports.requests import passport_import_request

    prepared = prepared_passport()
    value = ImportPassport(OWNER, prepared)
    request = passport_import_request(
        owner_agent_id=OWNER,
        passport_hash=prepared.document.passport_hash,
        idempotency_key="context",
    )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        if change == "resource":
            await passport_stack.container.receipt_store.attach_resource(
                uow=uow,
                receipt_id=command.receipt_id,
                resource_type="passport_import",
                resource_id=UUID(int=5999),
            )
        else:
            changes = {
                "receipt_id": UUID(int=5999),
                "idempotency_key": "forged-key",
                "request_hash": "c" * 64,
            }
            command = replace(command, **{change: changes[change]})
        return await passport_stack.service.import_passport(
            uow=uow, request=value, command=command
        )

    with pytest.raises(PassportError) as error:
        await passport_stack.container.command_executor.execute(request, handler)
    assert error.value.code == "passport_invalid"


async def test_source_timestamp_conversion_corruption_is_a_fixed_domain_error(
    passport_stack: PassportStack,
) -> None:
    first = await import_passport(passport_stack)
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(text("DROP TRIGGER passport_imports_reject_update"))
        await uow.session.execute(text("PRAGMA ignore_check_constraints=ON"))
        await uow.session.execute(text("UPDATE passport_imports SET imported_at='bad'"))
        await uow.session.execute(text("PRAGMA ignore_check_constraints=OFF"))
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(PassportError) as error:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=result_id(first)
            )
        assert error.value.code == "passport_invalid"


async def test_import_cannot_precede_owner_creation(
    passport_stack: PassportStack,
) -> None:
    passport_stack.clock.advance(timedelta(seconds=-1))
    with pytest.raises(PassportError) as error:
        await import_passport(passport_stack)
    assert error.value.code == "passport_invalid"
