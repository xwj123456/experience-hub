from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import timedelta

import pytest
from sqlalchemy import func, select, text, update
from tests.passport_export_fixtures import create_export_experience
from tests.passport_service_fixtures import (
    MISSING,
    OTHER,
    OWNER,
    PassportStack,
    decide_passport,
    import_passport,
    prepared_passport,
    result_id,
)
from tests.passport_service_fixtures import (
    passport_stack as passport_stack,
)

from experience_hub.domain import CommandContext, StructuredReason
from experience_hub.errors import DomainError
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.passports.contracts import PassportState
from experience_hub.passports.errors import PassportError
from experience_hub.retrieval import RetrievalMode
from experience_hub.retrieval.contracts import PeekExperiences
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperienceRow,
    ExperienceStateRow,
    IdempotencyRecordRow,
    PassportAdoptionRow,
    PassportImportRow,
    PassportStateRow,
)


async def test_adopt_creates_warm_owned_experience_and_replays(
    passport_stack: PassportStack,
) -> None:
    prepared = prepared_passport()
    import_id = result_id(await import_passport(passport_stack))
    passport_stack.clock.advance(timedelta(seconds=1))
    first = await decide_passport(passport_stack, import_id)
    replay = await decide_passport(passport_stack, import_id)
    assert first.status_code == 200 and not first.replayed
    assert replay.body == first.body and replay.replayed
    data = json.loads(first.body)["data"]
    assert data["created"] is True
    adoption_id = result_id(first, "adoption_id")
    async with passport_stack.container.database.read_session() as session:
        view = await passport_stack.query.get_owned(
            session=session, owner_agent_id=OWNER, import_id=import_id
        )
        assert view.state is PassportState.ADOPTED
        adoption = await passport_stack.query.get_adoption(
            session=session, owner_agent_id=OWNER, adoption_id=adoption_id
        )
        assert adoption.prepared == prepared
        identity = await session.get(ExperienceRow, adoption.resulting_experience_id)
        state = await session.get(ExperienceStateRow, adoption.resulting_experience_id)
        assert (
            identity is not None
            and identity.origin is ExperienceOrigin.ADOPTED_PASSPORT
        )
        assert state is not None and state.temperature is Temperature.WARM
        assert state.importance == 0.7 and state.confidence == 0.6
        assert state.source_trust == 0.25
        assert state.current_content_hash == prepared.document.subject.content_hash
        assert state.current_version_id != prepared.document.subject.source_version_id
        assert identity.experience_id != prepared.document.subject.source_experience_id
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceLinkRow))
            == 0
        )
        assert (
            await session.scalar(select(func.count()).select_from(PassportAdoptionRow))
            == 1
        )
        receipt = await session.scalar(
            select(IdempotencyRecordRow).where(
                IdempotencyRecordRow.scope == "passport.adopt"
            )
        )
        assert (
            receipt is not None and receipt.result_resource_type == "passport_adoption"
        )
        assert identity.created_at == adoption.adopted_at == receipt.created_at
        events = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.causation_id == receipt.receipt_id)
                .order_by(DomainEventRow.event_id)
            )
        )
        assert events == (
            "experience.created",
            "experience.version_created",
            "passport.adopted",
        )
        await passport_stack.container.source_validator.validate(session)
        hits = await passport_stack.container.experience_evidence_reader.peek(
            session=session,
            query=PeekExperiences(
                owner_agent_id=OWNER, query="rollback retry", mode=RetrievalMode.FOCUSED
            ),
        )
        assert adoption.resulting_experience_id in {
            hit.experience.experience_id for hit in hits.hits
        }
    duplicate = await import_passport(passport_stack, key="after-adoption")
    assert json.loads(duplicate.body)["data"]["state"] == "adopted"


async def test_rejection_is_terminal_without_ordinary_experience(
    passport_stack: PassportStack,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    rejected = await decide_passport(passport_stack, import_id, action="reject")
    replay = await decide_passport(passport_stack, import_id, action="reject")
    assert (
        rejected.status_code == 200 and replay.body == rejected.body and replay.replayed
    )
    async with passport_stack.container.database.read_session() as session:
        view = await passport_stack.query.get_owned(
            session=session, owner_agent_id=OWNER, import_id=import_id
        )
        assert view.state is PassportState.REJECTED
        assert view.reason == StructuredReason.from_user_text("Not reusable here.")
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceRow)) == 0
        )
        await passport_stack.container.source_validator.validate(session)
    terminal = await decide_passport(passport_stack, import_id, key="new-terminal-key")
    assert terminal.status_code == 409
    assert json.loads(terminal.body)["error"]["code"] == "passport_decision_conflict"
    replay_terminal = await decide_passport(
        passport_stack, import_id, key="new-terminal-key"
    )
    assert replay_terminal.body == terminal.body and replay_terminal.replayed


@pytest.mark.parametrize("action", ("adopt", "reject"))
@pytest.mark.parametrize("checkpoint", tuple(FaultCheckpoint))
async def test_decision_checkpoints_rollback_every_result_and_retry(
    passport_stack: PassportStack,
    action: str,
    checkpoint: FaultCheckpoint,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    passport_stack.fault.checkpoint = checkpoint
    with pytest.raises(RuntimeError, match="injected passport checkpoint"):
        await decide_passport(passport_stack, import_id, action=action)
    async with passport_stack.container.database.read_session() as session:
        state = await session.get(PassportStateRow, import_id)
        assert state is not None and state.state is PassportState.PENDING
        for table in (ExperienceRow, PassportAdoptionRow):
            assert await session.scalar(select(func.count()).select_from(table)) == 0
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == f"passport.{action}")
            )
            == 0
        )
    passport_stack.fault.checkpoint = None
    assert (
        await decide_passport(passport_stack, import_id, action=action)
    ).status_code == 200


async def test_reuse_equivalent_keeps_all_local_state_unchanged(
    passport_stack: PassportStack,
) -> None:
    seed = await create_export_experience(passport_stack.container, OWNER)
    import_id = result_id(await import_passport(passport_stack))
    async with passport_stack.container.database.read_session() as session:
        state = await session.get(ExperienceStateRow, seed.experience_id)
        assert state is not None
        snapshot = tuple(
            getattr(state, column.name)
            for column in ExperienceStateRow.__table__.columns
        )
    adopted = await decide_passport(
        passport_stack, import_id, importance=0.1, confidence=0.2
    )
    assert json.loads(adopted.body)["data"]["created"] is False
    async with passport_stack.container.database.read_session() as session:
        state = await session.get(ExperienceStateRow, seed.experience_id)
        assert state is not None
        assert (
            tuple(
                getattr(state, column.name)
                for column in ExperienceStateRow.__table__.columns
            )
            == snapshot
        )
        lineage = await session.get(
            PassportAdoptionRow, result_id(adopted, "adoption_id")
        )
        assert lineage is not None
        assert lineage.resulting_version_id == seed.version_id
        assert lineage.importance == 0.1 and lineage.confidence == 0.2
        await passport_stack.container.source_validator.validate(session)


async def test_foreign_equivalent_never_participates(
    passport_stack: PassportStack,
) -> None:
    await create_export_experience(passport_stack.container, OTHER)
    import_id = result_id(await import_passport(passport_stack))
    result = await decide_passport(passport_stack, import_id)
    assert json.loads(result.body)["data"]["created"] is True


async def test_archived_equivalent_requires_explicit_restore(
    passport_stack: PassportStack,
) -> None:
    seed = await create_export_experience(passport_stack.container, OWNER)
    import_id = result_id(await import_passport(passport_stack))
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == seed.experience_id)
            .values(temperature=Temperature.ARCHIVED)
        )
    result = await decide_passport(passport_stack, import_id)
    assert result.status_code == 409
    assert json.loads(result.body)["error"]["code"] == "passport_restore_required"


@pytest.mark.parametrize("action", ("adopt", "reject"))
async def test_foreign_missing_decisions_and_adoption_queries_are_not_found(
    passport_stack: PassportStack,
    action: str,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    for owner, identifier in (
        (OTHER, import_id),
        (OWNER, MISSING),
        (MISSING, import_id),
    ):
        with pytest.raises(PassportError) as error:
            await decide_passport(
                passport_stack, identifier, action=action, owner=owner
            )
        assert error.value.code == "passport_not_found"


async def test_decision_cannot_precede_import(passport_stack: PassportStack) -> None:
    import_id = result_id(await import_passport(passport_stack))
    passport_stack.clock.advance(timedelta(seconds=-1))
    with pytest.raises(PassportError) as error:
        await decide_passport(passport_stack, import_id)
    assert error.value.code == "passport_invalid"


async def test_changed_decision_same_key_conflicts(
    passport_stack: PassportStack,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    await decide_passport(passport_stack, import_id)
    with pytest.raises(DomainError) as error:
        await decide_passport(passport_stack, import_id, confidence=0.3)
    assert error.value.code == "idempotency_key_conflict"


@pytest.mark.parametrize(
    "score", (True, False, float("nan"), float("inf"), -0.1, 1.1, "0.7")
)
def test_adopt_command_rejects_invalid_numbers(score: object) -> None:
    contracts = importlib.import_module("experience_hub.passports.contracts")
    command_type = getattr(contracts, "AdoptPassport", None)
    assert command_type is not None, "AdoptPassport is missing"
    for field in ("importance", "confidence"):
        arguments = {
            "owner_agent_id": OWNER,
            "import_id": MISSING,
            "importance": 0.7,
            "confidence": 0.6,
            field: score,
        }
        with pytest.raises(PassportError) as error:
            command_type(**arguments)
        assert error.value.code == "passport_invalid"


async def test_multiple_owned_current_equivalents_fail_closed(
    passport_stack: PassportStack,
) -> None:
    first = await create_export_experience(passport_stack.container, OWNER)
    # Deliberately remove a catalog guard to model ambiguous legacy corruption.
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(text("DROP INDEX ux_experience_state_owner_content"))
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == first.experience_id)
            .values(current_content_hash="a" * 64)
        )
    await create_export_experience(
        passport_stack.container, OWNER, key="duplicate-legacy"
    )
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == first.experience_id)
            .values(current_content_hash=first.content_hash)
        )
    import_id = result_id(await import_passport(passport_stack))
    result = await decide_passport(passport_stack, import_id)
    assert result.status_code == 409
    assert json.loads(result.body)["error"]["code"] == "passport_equivalent_ambiguous"
    async with passport_stack.container.database.read_session() as session:
        assert (
            await session.scalar(select(func.count()).select_from(PassportAdoptionRow))
            == 0
        )


@pytest.mark.parametrize(
    "checkpoint",
    (
        FaultCheckpoint.AFTER_SOURCE_INSERT,
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
    ),
)
async def test_created_adoption_rolls_back_at_passport_stage(
    passport_stack: PassportStack,
    checkpoint: FaultCheckpoint,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    passport_stack.fault.checkpoint = checkpoint
    passport_stack.fault.occurrence = 2
    with pytest.raises(RuntimeError, match="injected passport checkpoint"):
        await decide_passport(passport_stack, import_id)
    assert passport_stack.fault.seen == 2
    async with passport_stack.container.database.read_session() as session:
        for table in (ExperienceRow, PassportAdoptionRow):
            assert await session.scalar(select(func.count()).select_from(table)) == 0
        state = await session.get(PassportStateRow, import_id)
        assert state is not None and state.state is PassportState.PENDING
    passport_stack.fault.checkpoint = None
    assert (await decide_passport(passport_stack, import_id)).status_code == 200


@pytest.mark.parametrize("action", ("adopt", "reject"))
async def test_decisions_database_busy_leave_no_receipt_and_retry(
    passport_stack: PassportStack,
    action: str,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    database = passport_stack.container.database
    async with database.read_session() as session:
        await session.execute(text("PRAGMA busy_timeout=10"))
    path = database._engine.url.database
    assert path is not None
    lock = sqlite3.connect(path)
    lock.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(DomainError) as error:
            await decide_passport(passport_stack, import_id, action=action)
        assert error.value.code == "database_busy"
    finally:
        lock.rollback()
        lock.close()
    async with database.read_session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.scope == f"passport.{action}")
            )
            == 0
        )
    assert (
        await decide_passport(passport_stack, import_id, action=action)
    ).status_code == 200


@pytest.mark.parametrize("action", ("adopt", "reject"))
async def test_corrupt_owned_source_cannot_be_decided_or_shown(
    passport_stack: PassportStack,
    action: str,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        await uow.session.execute(text("DROP TRIGGER passport_imports_reject_update"))
        await uow.session.execute(
            update(PassportImportRow)
            .where(PassportImportRow.import_id == import_id)
            .values(canonical_bytes=b"{}")
        )
    with pytest.raises(PassportError) as error:
        await decide_passport(passport_stack, import_id, action=action)
    assert error.value.code == "passport_invalid"
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(PassportError) as read_error:
            await passport_stack.query.get_owned(
                session=session, owner_agent_id=OWNER, import_id=import_id
            )
        assert read_error.value.code == "passport_invalid"
        assert (
            await session.scalar(select(func.count()).select_from(ExperienceRow)) == 0
        )


@pytest.mark.parametrize("action", ("adopt", "reject"))
async def test_decision_revalidates_forged_command_at_service_boundary(
    passport_stack: PassportStack,
    action: str,
) -> None:
    from experience_hub.passports.contracts import AdoptPassport, RejectPassport
    from experience_hub.passports.requests import (
        passport_adopt_request,
        passport_reject_request,
    )

    import_id = result_id(await import_passport(passport_stack))
    if action == "adopt":
        value = AdoptPassport(OWNER, import_id, 0.7, 0.6)
        request = passport_adopt_request(
            owner_agent_id=OWNER,
            import_id=import_id,
            importance=0.7,
            confidence=0.6,
            idempotency_key="forge",
        )
        object.__setattr__(value, "importance", True)
    else:
        reason = StructuredReason.from_user_text("No benefit.")
        value = RejectPassport(OWNER, import_id, reason)
        request = passport_reject_request(
            owner_agent_id=OWNER,
            import_id=import_id,
            reason=reason,
            idempotency_key="forge",
        )
        object.__setattr__(value.reason, "text_hash", "a" * 64)

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        if action == "adopt":
            return await passport_stack.service.adopt(
                uow=uow, request=value, command=command
            )
        return await passport_stack.service.reject(
            uow=uow, request=value, command=command
        )

    with pytest.raises(PassportError) as error:
        await passport_stack.container.command_executor.execute(request, handler)
    assert error.value.code == "passport_invalid"


async def test_adoption_query_is_owner_scoped_and_keeps_historical_version(
    passport_stack: PassportStack,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    adopted = await decide_passport(passport_stack, import_id)
    adoption_id = result_id(adopted, "adoption_id")
    async with passport_stack.container.database.read_session() as session:
        lineage = await passport_stack.query.get_adoption(
            session=session, owner_agent_id=OWNER, adoption_id=adoption_id
        )
        for owner, identifier in (
            (OTHER, adoption_id),
            (OWNER, MISSING),
            (MISSING, adoption_id),
        ):
            with pytest.raises(PassportError) as error:
                await passport_stack.query.get_adoption(
                    session=session, owner_agent_id=owner, adoption_id=identifier
                )
            assert error.value.code == "passport_not_found"
    await create_export_experience(
        passport_stack.container,
        OWNER,
        experience_id=lineage.resulting_experience_id,
        content=prepared_passport("later-version").document.subject.content,
        key="later-version",
    )
    async with passport_stack.container.database.read_session() as session:
        retained = await passport_stack.query.get_adoption(
            session=session, owner_agent_id=OWNER, adoption_id=adoption_id
        )
        assert retained == lineage


async def test_integer_scores_are_normalized_and_float_form_replays(
    passport_stack: PassportStack,
) -> None:
    import_id = result_id(await import_passport(passport_stack))
    first = await decide_passport(passport_stack, import_id, importance=1, confidence=0)
    replay = await decide_passport(
        passport_stack, import_id, importance=1.0, confidence=0.0
    )
    assert first.status_code == 200 and replay.body == first.body and replay.replayed
    async with passport_stack.container.database.read_session() as session:
        await passport_stack.container.source_validator.validate(session)


async def test_real_service_sources_and_all_projections_rebuild_identically(
    passport_stack: PassportStack,
) -> None:
    first = result_id(await import_passport(passport_stack))
    await import_passport(passport_stack, key="pending-dedup")
    await decide_passport(passport_stack, first)
    await import_passport(passport_stack, key="terminal-dedup")
    second = result_id(
        await import_passport(
            passport_stack,
            key="rejected-import",
            prepared=prepared_passport("rejected"),
        )
    )
    await decide_passport(passport_stack, second, action="reject", key="reject")
    terminal = await decide_passport(passport_stack, second, key="terminal-conflict")
    assert terminal.status_code == 409
    report = await passport_stack.container.projection_manager.verify(
        passport_stack.container.database
    )
    assert report.matches
