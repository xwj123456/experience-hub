"""Read-only paths authenticate the same receipts as authority validation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.engine import make_url
from tests.passport_export_fixtures import create_export_experience
from tests.passport_service_fixtures import (
    MISSING,
    OTHER,
    OWNER,
    PassportStack,
    decide_passport,
    import_passport,
    result_id,
)
from tests.passport_service_fixtures import passport_stack as passport_stack

from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences import PinExperience
from experience_hub.passports import PassportDeclarationV1
from experience_hub.passports.errors import PassportError
from experience_hub.passports.export import PassportExportService
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.readonly import readonly_sqlite_session
from experience_hub.storage.validation import SourceIntegrityError

DAMAGED_FIELDS = (
    "request_hash",
    "caller_scope",
    "scope",
    "idempotency_key",
    "result_resource_type",
    "result_resource_id",
    "response_status_code",
    "response_body",
    "response_content_type",
    "response_headers",
    "created_at",
    "completed_at",
    "missing",
    "in_progress",
)


def _damage_receipt(database: Path, key: str, field: str) -> None:
    # Authority corruption is intentional; the owning writer has already closed.
    with closing(sqlite3.connect(database)) as connection, connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key=?",
            (key,),
        ).fetchone() == (1,)
        if field == "missing":
            connection.execute(
                "DELETE FROM idempotency_records WHERE idempotency_key=?", (key,)
            )
            return
        if field == "in_progress":
            connection.execute(
                "UPDATE idempotency_records SET state='in_progress',"
                "response_status_code=NULL,response_body=NULL,"
                "response_content_type=NULL,response_headers=NULL,completed_at=NULL "
                "WHERE idempotency_key=?",
                (key,),
            )
            return
        values = {
            "request_hash": "f" * 64,
            "caller_scope": f"agent:{OTHER}",
            "scope": "passport.inspect",
            "idempotency_key": " noncanonical-decision-key ",
            "result_resource_type": "synthetic_wrong_resource",
            "result_resource_id": str(MISSING),
            "response_status_code": 202,
            "response_body": b"{}",
            "response_content_type": "text/plain",
            "response_headers": b'{"synthetic":"unexpected"}',
            "created_at": "2026-09-29T08:00:00.000000Z",
            "completed_at": "2026-09-29T08:00:00.000000Z",
        }
        if field == "idempotency_key":
            # A valid nonce is not part of request semantics. Corrupt the stored
            # key's canonical form instead, bypassing only its SQLite CHECK.
            connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            f"UPDATE idempotency_records SET {field}=? WHERE idempotency_key=?",
            (values[field], key),
        )
        if field == "idempotency_key":
            connection.execute("PRAGMA ignore_check_constraints=OFF")


async def _damage(stack: PassportStack, *, key: str, field: str) -> Path:
    database = make_url(stack.container.settings.database_url).database
    assert database is not None
    path = Path(database)
    await stack.container.close()
    await asyncio.to_thread(_damage_receipt, path, key, field)
    return path


@pytest.mark.parametrize("action", ("adopt", "reject"))
@pytest.mark.parametrize("field", DAMAGED_FIELDS)
async def test_owner_terminal_reads_refuse_damaged_decision_receipts(
    passport_stack: PassportStack, action: str, field: str
) -> None:
    # Omitting terminal receipt authentication must fail both legal decisions.
    own = result_id(await import_passport(passport_stack))
    assert (
        await decide_passport(passport_stack, own, action=action)
    ).status_code == 200
    database = await _damage(passport_stack, key="decision", field=field)
    async with readonly_sqlite_session(database) as session:
        with pytest.raises(PassportError) as caught:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=own
            )
        assert caught.value.code == "passport_invalid"
        assert str(OWNER) not in str(caught.value) and str(own) not in str(caught.value)
        assert (
            await passport_stack.query.list_owned(session=session, owner_agent_id=OTHER)
        ).items == ()
        for identifier in (own, MISSING):
            with pytest.raises(PassportError) as absent:
                await passport_stack.query.get_owned(
                    session=session, owner_agent_id=OTHER, import_id=identifier
                )
            assert absent.value.code == "passport_not_found"


@pytest.mark.parametrize("field", ("request_hash", "missing", "response_headers"))
async def test_readonly_reexport_refuses_damaged_adoption_receipt(
    passport_stack: PassportStack, field: str
) -> None:
    own = result_id(await import_passport(passport_stack))
    result = await decide_passport(passport_stack, own)
    assert result.status_code == 200
    data = json.loads(result.body)["data"]
    database = await _damage(passport_stack, key="decision", field=field)
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError):
            await passport_stack.container.source_validator.validate(session)
    await passport_stack.container.database.dispose()
    before = database.read_bytes()
    async with readonly_sqlite_session(database) as session:
        with pytest.raises(PassportError) as caught:
            await PassportExportService().export(
                session=session,
                owner_agent_id=OWNER,
                experience_id=UUID(data["experience"]["experience_id"]),
                version_id=None,
                parent_adoption_id=UUID(data["adoption_id"]),
                declaration=PassportDeclarationV1(
                    input_sanitized=True,
                    profile_id="synthetic-receipt-v1",
                    sharing_authorized=True,
                ),
            )
        assert caught.value.code == "passport_invalid"
    assert database.read_bytes() == before


@pytest.mark.parametrize("field", ("response_content_type", "response_headers"))
async def test_pending_reads_authenticate_exact_import_response(
    passport_stack: PassportStack, field: str
) -> None:
    own = result_id(await import_passport(passport_stack))
    database = await _damage(passport_stack, key="import", field=field)
    async with readonly_sqlite_session(database) as session:
        with pytest.raises(PassportError) as caught:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=own
            )
        assert caught.value.code == "passport_invalid"


@pytest.mark.parametrize("reuse_local", (False, True))
async def test_valid_adoption_receipt_uses_historical_not_current_temperature(
    passport_stack: PassportStack, reuse_local: bool
) -> None:
    # Comparing the stored decision to today's temperature would falsely refuse.
    if reuse_local:
        await create_export_experience(passport_stack.container, OWNER)
    own = result_id(await import_passport(passport_stack))
    decision = await decide_passport(passport_stack, own)
    assert decision.status_code == 200
    data = json.loads(decision.body)["data"]
    assert data["created"] is not reuse_local
    assert data["experience"]["temperature"] == "warm"
    experience = UUID(data["experience"]["experience_id"])
    passport_stack.clock.advance(timedelta(minutes=1))

    async def pin(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await passport_stack.container.experience_service.pin(
            uow=uow,
            command=PinExperience(owner_agent_id=OWNER, experience_id=experience),
            command_context=context,
        )

    pinned = await passport_stack.container.command_executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER}",
            operation_scope="experience.pin",
            idempotency_key="later-pin",
            method="POST",
            route_template="/v1/agents/{agent_id}/experiences/{experience_id}:pin",
            path_parameters={"agent_id": OWNER, "experience_id": experience},
            body={"reason": None},
        ),
        pin,
    )
    assert pinned.status_code == 200
    assert json.loads(pinned.body)["data"]["temperature"] == "hot"
    async with passport_stack.container.database.read_session() as session:
        await passport_stack.container.source_validator.validate(session)
    database = make_url(passport_stack.container.settings.database_url).database
    assert database is not None
    await passport_stack.container.close()
    async with readonly_sqlite_session(Path(database)) as session:
        viewed = await passport_stack.query.get_owned(
            session=session, owner_agent_id=OWNER, import_id=own
        )
        assert str(viewed.state) == "adopted"
        exported = await PassportExportService().export(
            session=session,
            owner_agent_id=OWNER,
            experience_id=experience,
            version_id=None,
            parent_adoption_id=UUID(data["adoption_id"]),
            declaration=PassportDeclarationV1(
                input_sanitized=True,
                profile_id="synthetic-history-v1",
                sharing_authorized=True,
            ),
        )
        assert len(exported.document.provenance.hops) == 2
