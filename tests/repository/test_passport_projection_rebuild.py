from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from tests.passport_fixtures import passport_document

from experience_hub import canonical_json_bytes
from experience_hub.domain import EventRegistry, StoredEvent, StructuredReason
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import ExperienceOrigin
from experience_hub.passports.codec import encode_passport_document
from experience_hub.passports.contracts import PassportState
from experience_hub.storage.tables import (
    AgentRow,
    Base,
    DomainEventRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)
OWNER = UUID(int=201)
IMPORT = UUID(int=202)
ADOPTION = UUID(int=203)
EXPERIENCE = UUID(int=204)
VERSION = UUID(int=205)
RECEIPT = UUID(int=206)
DECISION_RECEIPT = UUID(int=207)


def _components():
    try:
        from experience_hub.passports.events import (
            PassportAdoptedV1,
            PassportImportedV1,
            PassportRejectedV1,
            register_passport_events,
        )
        from experience_hub.passports.projector import PassportStateProjector
        from experience_hub.storage.tables.passports import (
            PassportAdoptionRow,
            PassportImportRow,
        )
    except ImportError:
        pytest.fail("Passport source reducer is not implemented")
    return (
        PassportAdoptedV1,
        PassportImportedV1,
        PassportRejectedV1,
        register_passport_events,
        PassportStateProjector,
        PassportAdoptionRow,
        PassportImportRow,
    )


def _receipt(
    identifier: UUID, scope: str, resource: str, result: UUID, when: datetime
) -> IdempotencyRecordRow:
    return IdempotencyRecordRow(
        receipt_id=identifier,
        caller_scope=f"agent:{OWNER}",
        scope=scope,
        idempotency_key=str(identifier),
        request_hash="a" * 64,
        state="in_progress",
        result_resource_type=resource,
        result_resource_id=result,
        created_at=when,
    )


async def _graph(session: AsyncSession, terminal: str):
    (
        adopted_type,
        imported_type,
        rejected_type,
        register,
        projector_type,
        adoption_type,
        import_type,
    ) = _components()
    registry = EventRegistry()
    register(registry)
    reducer = projector_type(registry)
    document = passport_document()
    session.add(AgentRow(agent_id=OWNER, name="Projection Owner", created_at=NOW))
    await session.flush()
    session.add(
        import_type(
            import_id=IMPORT,
            owner_agent_id=OWNER,
            passport_hash=document.passport_hash,
            canonical_bytes=encode_passport_document(document),
            imported_at=NOW,
        )
    )
    session.add(_receipt(RECEIPT, "passport.import", "passport_import", IMPORT, NOW))
    imported = imported_type(
        schema_version=1,
        import_id=IMPORT,
        owner_agent_id=OWNER,
        passport_hash=document.passport_hash,
        state_after=PassportState.PENDING,
    )
    events = [
        StoredEvent(
            1,
            "passport_import",
            IMPORT,
            1,
            "passport.imported",
            imported,
            OWNER,
            RECEIPT,
            NOW,
        )
    ]
    if terminal != "pending":
        when = NOW + timedelta(seconds=1)
        common = {
            "schema_version": 1,
            "import_id": IMPORT,
            "owner_agent_id": OWNER,
            "state_before": PassportState.PENDING,
        }
        if terminal == "rejected":
            payload = rejected_type(
                **common,
                state_after=PassportState.REJECTED,
                reason=StructuredReason.from_user_text("Not needed."),
            )
            resource, result = "passport_import", IMPORT
        else:
            subject = document.subject
            encoded = encode_version_content(kind=subject.kind, content=subject.content)
            session.add(
                ExperienceRow(
                    experience_id=EXPERIENCE,
                    owner_agent_id=OWNER,
                    kind=subject.kind,
                    origin=ExperienceOrigin.LOCAL,
                    created_at=NOW,
                )
            )
            await session.flush()
            session.add(
                ExperienceVersionRow(
                    version_id=VERSION,
                    experience_id=EXPERIENCE,
                    version_number=1,
                    summary=subject.content.summary,
                    mechanism=subject.content.mechanism,
                    tags=canonical_json_bytes(subject.content.tags),
                    applicability=canonical_json_bytes(subject.content.applicability),
                    evidence=canonical_json_bytes(subject.content.evidence),
                    falsifiers=canonical_json_bytes(subject.content.falsifiers),
                    content_hash=encoded.content_hash,
                    created_at=NOW,
                )
            )
            await session.flush()
            session.add(
                adoption_type(
                    adoption_id=ADOPTION,
                    import_id=IMPORT,
                    owner_agent_id=OWNER,
                    resulting_experience_id=EXPERIENCE,
                    resulting_version_id=VERSION,
                    resulting_content_hash=encoded.content_hash,
                    created=False,
                    importance=0.7,
                    confidence=0.6,
                    adopted_at=when,
                )
            )
            payload = adopted_type(
                **common,
                state_after=PassportState.ADOPTED,
                adoption_id=ADOPTION,
                resulting_experience_id=EXPERIENCE,
                resulting_version_id=VERSION,
                resulting_content_hash=encoded.content_hash,
                created=False,
                importance=0.7,
                confidence=0.6,
            )
            resource, result = "passport_adoption", ADOPTION
        session.add(
            _receipt(
                DECISION_RECEIPT,
                f"passport.{terminal[:-2]}"
                if terminal == "adopted"
                else "passport.reject",
                resource,
                result,
                when,
            )
        )
        events.append(
            StoredEvent(
                2,
                "passport_import",
                IMPORT,
                2,
                payload.event_type,
                payload,
                OWNER,
                DECISION_RECEIPT,
                when,
            )
        )
    await session.flush()
    for event in events:
        session.add(
            DomainEventRow(
                event_id=event.event_id,
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                sequence=event.sequence,
                event_type=event.event_type,
                payload=canonical_json_bytes(event.payload),
                actor_agent_id=event.actor_agent_id,
                causation_id=event.causation_id,
                occurred_at=event.occurred_at,
            )
        )
    await session.flush()
    return reducer, events


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ("pending", "adopted", "rejected"))
async def test_rebuild_matches_apply_and_retains_sources(terminal: str) -> None:
    _components()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, terminal)
            for event in events:
                await reducer.apply(session, event)
            before = tuple(await session.execute(text("SELECT * FROM passport_state")))
            await reducer.rebuild(session, "verify_")
            assert (
                tuple(
                    await session.execute(
                        text("SELECT * FROM temp.verify_passport_state")
                    )
                )
                == before
            )
            assert before[0][2] == terminal
            assert (
                await session.scalar(text("SELECT count(*) FROM passport_imports")) == 1
            )
            assert await session.scalar(
                text("SELECT count(*) FROM passport_adoptions")
            ) == (1 if terminal == "adopted" else 0)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    (
        {"aggregate_type": "candidate"},
        {"aggregate_id": UUID(int=211)},
        {"sequence": 2},
        {"actor_agent_id": UUID(int=212)},
        {"causation_id": UUID(int=213)},
        {"occurred_at": NOW + timedelta(seconds=2)},
    ),
)
async def test_import_reducer_rejects_inconsistent_ledger_anchors(update) -> None:
    _components()
    from experience_hub.passports.projector import PassportProjectionIntegrityError

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, "pending")
            with pytest.raises(PassportProjectionIntegrityError):
                await reducer.apply(session, replace(events[0], **update))
            assert (
                await session.scalar(text("SELECT count(*) FROM passport_state")) == 0
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ("adopted", "rejected"))
async def test_decision_reducer_requires_pending_and_refuses_second_terminal(
    terminal: str,
) -> None:
    _components()
    from experience_hub.passports.projector import PassportProjectionIntegrityError

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, terminal)
            with pytest.raises(PassportProjectionIntegrityError):
                await reducer.apply(session, events[1])
            await reducer.apply(session, events[0])
            await reducer.apply(session, events[1])
            with pytest.raises(PassportProjectionIntegrityError):
                await reducer.apply(session, events[1])
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_adoption_reducer_binds_request_scores_to_source() -> None:
    _components()
    from experience_hub.passports.projector import PassportProjectionIntegrityError

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, "adopted")
            await reducer.apply(session, events[0])
            payload = events[1].payload.model_copy(update={"confidence": 0.9})
            with pytest.raises(PassportProjectionIntegrityError):
                await reducer.apply(session, replace(events[1], payload=payload))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_rebuild_refuses_unsafe_target_prefix() -> None:
    _components()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with AsyncSession(engine) as session:
            register = _components()[3]
            registry = EventRegistry()
            register(registry)
            reducer = _components()[4](registry)
            with pytest.raises(ValueError):
                await reducer.rebuild(session, 'unsafe"; DROP TABLE agents;--')
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_corrupt_source_bytes_raise_fixed_projection_error() -> None:
    _components()
    from experience_hub.passports.projector import PassportProjectionIntegrityError

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, "pending")
            await session.execute(text("DROP TRIGGER passport_imports_reject_update"))
            await session.execute(
                text(
                    "UPDATE passport_imports SET canonical_bytes=:bytes "
                    "WHERE import_id=:id"
                ),
                {"bytes": b"{}", "id": str(IMPORT)},
            )
            session.expire_all()
            with pytest.raises(PassportProjectionIntegrityError) as failure:
                await reducer.apply(session, events[0])
            assert str(failure.value) == "Passport source bytes are invalid"
            assert failure.value.code == "passport_projection_integrity_error"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("import_id", "hash", "time", "causation"))
async def test_decision_revalidates_its_import_ledger_anchor(mutation: str) -> None:
    _components()
    from experience_hub.passports.projector import PassportProjectionIntegrityError

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with AsyncSession(engine) as session, session.begin():
            reducer, events = await _graph(session, "adopted")
            await reducer.apply(session, events[0])
            imported = await session.get(DomainEventRow, 1)
            assert imported is not None
            if mutation in {"import_id", "hash"}:
                updates = (
                    {"import_id": UUID(int=221)}
                    if mutation == "import_id"
                    else {"passport_hash": "f" * 64}
                )
                imported.payload = canonical_json_bytes(
                    events[0].payload.model_copy(update=updates)
                )
            elif mutation == "time":
                imported.occurred_at = NOW - timedelta(seconds=1)
            else:
                imported.causation_id = UUID(int=222)
            await session.flush()
            with pytest.raises(PassportProjectionIntegrityError):
                await reducer.apply(session, events[1])
    finally:
        await engine.dispose()
