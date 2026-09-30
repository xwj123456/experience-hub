from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select, text
from tests.passport_service_fixtures import (
    MISSING,
    OTHER,
    OWNER,
    PassportStack,
    import_passport,
    prepared_passport,
    result_id,
)
from tests.passport_service_fixtures import (
    passport_stack as passport_stack,
)

from experience_hub.domain import CommandRequest
from experience_hub.passports.contracts import PassportState
from experience_hub.passports.errors import PassportError
from experience_hub.storage.tables import PassportImportRow, PassportStateRow


async def test_same_bytes_and_literal_key_are_separate_for_actual_owners(
    passport_stack: PassportStack,
) -> None:
    own = await import_passport(passport_stack)
    other = await import_passport(passport_stack, owner=OTHER)
    assert result_id(own) != result_id(other)
    async with passport_stack.container.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(PassportImportRow))
            == 2
        )
        with pytest.raises(PassportError) as error:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OTHER, import_id=result_id(own)
            )
        assert error.value.code == "passport_not_found"
        for owner, import_id in ((OWNER, MISSING), (MISSING, result_id(own))):
            with pytest.raises(PassportError) as missing:
                await passport_stack.query.get_owned(
                    session=session, owner_agent_id=owner, import_id=import_id
                )
            assert str(missing.value) == str(error.value)


async def test_forged_caller_path_owner_is_not_found(
    passport_stack: PassportStack,
) -> None:
    prepared = prepared_passport()
    request = CommandRequest(
        caller_scope=f"agent:{OTHER}",
        operation_scope="passport.import",
        idempotency_key="mismatch",
        method="POST",
        route_template="/v1/agents/{agent_id}/passports",
        path_parameters={"agent_id": OWNER},
        body={"passport_hash": prepared.document.passport_hash},
    )
    with pytest.raises(PassportError) as error:
        await import_passport(passport_stack, command_request=request)
    assert error.value.code == "passport_not_found"
    with pytest.raises(PassportError) as missing:
        await import_passport(passport_stack, owner=MISSING)
    assert missing.value.code == "passport_not_found"


async def test_owned_pagination_ties_and_cursor_owner_state_binding(
    passport_stack: PassportStack,
) -> None:
    ids = [
        result_id(
            await import_passport(
                passport_stack, key=str(i), prepared=prepared_passport(f"tag-{i}")
            )
        )
        for i in range(3)
    ]
    async with passport_stack.container.database.read_session() as session:
        page = await passport_stack.query.list_owned(
            session=session, owner_agent_id=OWNER, state=None, limit=1, cursor=None
        )
        assert [item.import_id for item in page.items] == sorted(ids, reverse=True)[:1]
        assert page.next_cursor is not None and "=" not in page.next_cursor
        next_page = await passport_stack.query.list_owned(
            session=session,
            owner_agent_id=OWNER,
            state=None,
            limit=100,
            cursor=page.next_cursor,
        )
        assert [item.import_id for item in next_page.items] == sorted(
            ids, reverse=True
        )[1:]
        assert next_page.next_cursor is None
        for owner, state, cursor in (
            (OTHER, None, page.next_cursor),
            (OWNER, PassportState.PENDING, page.next_cursor),
            (OWNER, None, "?invalid"),
            (OWNER, None, "x" * 8193),
            (OWNER, None, page.next_cursor + "="),
        ):
            with pytest.raises(PassportError) as error:
                await passport_stack.query.list_owned(
                    session=session,
                    owner_agent_id=owner,
                    state=state,
                    limit=1,
                    cursor=cursor,
                )
            assert error.value.code == "passport_invalid"
        for limit in (0, 101, True):
            with pytest.raises(PassportError):
                await passport_stack.query.list_owned(
                    session=session,
                    owner_agent_id=OWNER,
                    state=None,
                    limit=limit,
                    cursor=None,
                )


async def test_state_filtered_page_cannot_hide_an_owned_orphan_source(
    passport_stack: PassportStack,
) -> None:
    own = result_id(await import_passport(passport_stack))
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            delete(PassportStateRow).where(PassportStateRow.import_id == own)
        )
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(PassportError) as error:
            await passport_stack.query.list_owned(
                session=session,
                owner_agent_id=OWNER,
                state=PassportState.PENDING,
                limit=1,
                cursor=None,
            )
        assert error.value.code == "passport_invalid"
        unaffected = await passport_stack.query.list_owned(
            session=session,
            owner_agent_id=OTHER,
            state=PassportState.PENDING,
            limit=1,
            cursor=None,
        )
        assert unaffected.items == ()


async def test_corrupt_owned_enum_is_a_fixed_domain_error(
    passport_stack: PassportStack,
) -> None:
    own = result_id(await import_passport(passport_stack))
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(text("PRAGMA ignore_check_constraints=ON"))
        await uow.session.execute(text("UPDATE passport_state SET state='broken'"))
        await uow.session.execute(text("PRAGMA ignore_check_constraints=OFF"))
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(PassportError) as error:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=own
            )
        assert error.value.code == "passport_invalid"
