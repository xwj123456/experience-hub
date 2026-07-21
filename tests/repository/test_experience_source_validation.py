from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select, text, update
from tests.integration.test_create_experience import (
    EXPERIENCE_IDS,
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    VERSION_IDS,
    Stack,
    build_stack,
    content,
    create,
)
from tests.integration.test_create_experience_version import correct

from experience_hub import storage
from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import TypedEvidence
from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
)
from experience_hub.experiences.contracts import VersionLinkInput
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceVersionCreatedV1,
    VersionLinkRefV1,
)
from experience_hub.storage import validation
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
)

SOURCE_EVIDENCE = TypedEvidence(type="observation", id="source-v1")


def test_experience_source_validator_has_public_registration_entrypoint() -> None:
    assert storage.ExperienceSourceValidator is validation.ExperienceSourceValidator
    assert (
        storage.register_experience_source_validator
        is validation.register_experience_source_validator
    )


@pytest.fixture
async def source_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-source-validation.sqlite3",
    )
    _, target = await create(stack, key="source-target", value=content("target"))
    target_id = UUID(target["data"]["experience_id"])
    _, source = await create(
        stack,
        key="source-linked",
        value=content("source-v1").model_copy(update={"evidence": (SOURCE_EVIDENCE,)}),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )
    source_id = UUID(source["data"]["experience_id"])
    await correct(
        stack,
        key="source-linked-v2",
        experience_id=source_id,
        value=content("source-v2"),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.TESTS,
            ),
        ),
    )
    try:
        yield stack
    finally:
        await stack.database.dispose()


def experience_validator(stack: Stack) -> validation.SourceValidator:
    validator = validation.SourceValidator(stack.registry)
    validation.register_experience_source_validator(validator)
    return validator


@pytest.mark.asyncio
async def test_experience_source_validator_accepts_complete_source_graph(
    source_stack: Stack,
) -> None:
    validator = experience_validator(source_stack)

    async with source_stack.database.read_session() as session:
        await validator.validate(session)


@pytest.mark.asyncio
async def test_experience_source_validator_reports_stable_payload_mismatch_key(
    source_stack: Stack,
) -> None:
    async with source_stack.database.read_session() as session:
        version_id = await session.scalar(
            select(ExperienceVersionRow.version_id)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
    assert version_id is not None
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_payloads_reject_semantic_update")
        )
        await uow.session.execute(
            update(ExperiencePayloadRow)
            .where(ExperiencePayloadRow.version_id == version_id)
            .values(payload_hash="f" * 64)
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    expected_key = f"experience_version_payload:{version_id}"
    assert caught.value.mismatch_key == expected_key
    assert str(caught.value).startswith(f"{expected_key}:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "payload_bytes",
        "content_hash",
        "metadata",
        "identity_kind",
        "missing_payload",
    ],
)
async def test_experience_source_validator_recomputes_semantic_hashes_by_kind(
    source_stack: Stack,
    corruption: str,
) -> None:
    async with source_stack.database.read_session() as session:
        version_id = await session.scalar(
            select(ExperienceVersionRow.version_id)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
    assert version_id is not None
    async with source_stack.database.transaction() as uow:
        if corruption == "payload_bytes":
            await uow.session.execute(
                text("DROP TRIGGER experience_payloads_reject_unguarded_rewrite")
            )
            await uow.session.execute(
                update(ExperiencePayloadRow)
                .where(ExperiencePayloadRow.version_id == version_id)
                .values(payload=b'{"body":"tampered source body"}')
            )
        elif corruption == "content_hash":
            await uow.session.execute(
                text("DROP TRIGGER experience_versions_reject_update")
            )
            await uow.session.execute(
                update(ExperienceVersionRow)
                .where(ExperienceVersionRow.version_id == version_id)
                .values(content_hash="e" * 64)
            )
        elif corruption == "metadata":
            await uow.session.execute(
                text("DROP TRIGGER experience_versions_reject_update")
            )
            await uow.session.execute(
                update(ExperienceVersionRow)
                .where(ExperienceVersionRow.version_id == version_id)
                .values(summary="Tampered semantic metadata")
            )
        elif corruption == "identity_kind":
            await uow.session.execute(text("DROP TRIGGER experiences_reject_update"))
            await uow.session.execute(
                update(ExperienceRow)
                .where(ExperienceRow.experience_id == EXPERIENCE_IDS[1])
                .values(kind=ExperienceKind.SEMANTIC)
            )
        else:
            await uow.session.execute(
                text("DROP TRIGGER experience_payloads_reject_delete")
            )
            await uow.session.execute(
                delete(ExperiencePayloadRow).where(
                    ExperiencePayloadRow.version_id == version_id
                )
            )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_version_payload:{version_id}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "corrupted"),
    [
        (
            "tags",
            canonical_json_bytes(("source-v1", "memory", "memory")),
        ),
        (
            "applicability",
            canonical_json_bytes(("single writer", "single writer")),
        ),
        (
            "falsifiers",
            canonical_json_bytes(("overlap observed", "overlap observed")),
        ),
        (
            "evidence",
            canonical_json_bytes((SOURCE_EVIDENCE, SOURCE_EVIDENCE)),
        ),
    ],
)
async def test_experience_source_validator_requires_exact_canonical_metadata_bytes(
    source_stack: Stack,
    field: str,
    corrupted: bytes,
) -> None:
    async with source_stack.database.read_session() as session:
        version_id = await session.scalar(
            select(ExperienceVersionRow.version_id)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
    assert version_id is not None
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        await uow.session.execute(
            update(ExperienceVersionRow)
            .where(ExperienceVersionRow.version_id == version_id)
            .values(**{field: corrupted})
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_version_payload:{version_id}"


@pytest.mark.asyncio
async def test_experience_source_validator_rejects_noncanonical_metadata_json(
    source_stack: Stack,
) -> None:
    async with source_stack.database.read_session() as session:
        version_id = await session.scalar(
            select(ExperienceVersionRow.version_id)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
    assert version_id is not None
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        await uow.session.execute(
            text(
                "UPDATE experience_versions SET tags = :tags "
                "WHERE version_id = :version_id"
            ),
            {
                "tags": b'["memory", "source-v1"]',
                "version_id": str(version_id),
            },
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_version_payload:{version_id}"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["gap", "wrong_supersession"])
async def test_experience_source_validator_requires_contiguous_version_numbers(
    source_stack: Stack,
    corruption: str,
) -> None:
    async with source_stack.database.read_session() as session:
        newest = await session.scalar(
            select(ExperienceVersionRow)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number.desc())
            .limit(1)
        )
    assert newest is not None
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        replacement = (
            3
            if corruption == "gap"
            else (
                await uow.session.scalar(
                    select(ExperienceVersionRow.version_id)
                    .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[0])
                    .limit(1)
                )
            )
        )
        assert replacement is not None
        await uow.session.execute(
            update(ExperienceVersionRow)
            .where(ExperienceVersionRow.version_id == newest.version_id)
            .values(
                **(
                    {"version_number": replacement}
                    if corruption == "gap"
                    else {"supersedes_version_id": replacement}
                )
            )
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert (
        caught.value.mismatch_key == f"experience_version_sequence:{EXPERIENCE_IDS[1]}"
    )


@pytest.mark.asyncio
async def test_experience_source_validator_requires_identity_creation_event(
    source_stack: Stack,
) -> None:
    orphan_id = UUID("00000000-0000-0000-0000-000000000999")
    async with source_stack.database.transaction() as uow:
        uow.session.add(
            ExperienceRow(
                experience_id=orphan_id,
                owner_agent_id=OWNER_ID,
                kind=ExperienceKind.PROCEDURAL,
                origin=ExperienceOrigin.LOCAL,
                created_at=NOW,
            )
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_identity:{orphan_id}"


@pytest.mark.asyncio
async def test_experience_source_validator_requires_one_event_per_version_row(
    source_stack: Stack,
) -> None:
    async with source_stack.database.read_session() as session:
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_id == EXPERIENCE_IDS[1],
                        DomainEventRow.event_type
                        == ExperienceVersionCreatedV1.event_type,
                    )
                    .order_by(DomainEventRow.sequence)
                )
            ).all()
        )
    assert len(events) == 2
    first, second = events
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == second.event_id)
            .values(payload=first.payload)
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key.startswith("experience_version_event:")


@pytest.mark.asyncio
async def test_experience_source_validator_anchors_creation_to_v1_event_time(
    source_stack: Stack,
) -> None:
    async with source_stack.database.read_session() as session:
        initial = await session.scalar(
            select(ExperienceVersionRow)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[0])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
        creation_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == EXPERIENCE_IDS[0],
                DomainEventRow.event_type == ExperienceCreatedV1.event_type,
            )
        )
        version_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == EXPERIENCE_IDS[0],
                DomainEventRow.event_type == ExperienceVersionCreatedV1.event_type,
            )
        )
    assert initial is not None
    assert creation_event is not None
    assert version_event is not None
    shifted = NOW + timedelta(hours=1)
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(ExperienceVersionRow)
            .where(ExperienceVersionRow.version_id == initial.version_id)
            .values(created_at=shifted)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == version_event.event_id)
            .values(occurred_at=shifted)
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_identity:{EXPERIENCE_IDS[0]}"


@pytest.mark.asyncio
async def test_experience_source_validator_requires_monotonic_version_event_order(
    source_stack: Stack,
) -> None:
    async with source_stack.database.read_session() as session:
        newest = await session.scalar(
            select(ExperienceVersionRow)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number.desc())
            .limit(1)
        )
    assert newest is not None
    async with source_stack.database.read_session() as session:
        newest_event = await session.scalar(
            select(DomainEventRow)
            .join(
                ExperienceLinkRow,
                ExperienceLinkRow.source_event_id == DomainEventRow.event_id,
            )
            .where(
                ExperienceLinkRow.source_version_id == newest.version_id,
            )
        )
    assert newest_event is not None
    shifted = NOW - timedelta(hours=1)
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_versions_reject_update")
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(ExperienceVersionRow)
            .where(ExperienceVersionRow.version_id == newest.version_id)
            .values(created_at=shifted)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == newest_event.event_id)
            .values(occurred_at=shifted)
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_version_event:{newest.version_id}"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["changed", "missing", "extra"])
async def test_experience_source_validator_matches_links_bidirectionally_to_event(
    source_stack: Stack,
    corruption: str,
) -> None:
    async with source_stack.database.read_session() as session:
        newest = await session.scalar(
            select(ExperienceVersionRow)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number.desc())
            .limit(1)
        )
    assert newest is not None
    async with source_stack.database.transaction() as uow:
        if corruption == "changed":
            await uow.session.execute(
                text("DROP TRIGGER experience_links_reject_update")
            )
            await uow.session.execute(
                update(ExperienceLinkRow)
                .where(ExperienceLinkRow.source_version_id == newest.version_id)
                .values(relation=LinkRelation.SUPPORTS)
            )
        elif corruption == "missing":
            await uow.session.execute(
                text("DROP TRIGGER experience_links_reject_delete")
            )
            await uow.session.execute(
                delete(ExperienceLinkRow).where(
                    ExperienceLinkRow.source_version_id == newest.version_id
                )
            )
        else:
            existing = await uow.session.scalar(
                select(ExperienceLinkRow).where(
                    ExperienceLinkRow.source_version_id == newest.version_id
                )
            )
            assert existing is not None
            uow.session.add(
                ExperienceLinkRow(
                    source_experience_id=existing.source_experience_id,
                    source_version_id=existing.source_version_id,
                    target_experience_id=existing.target_experience_id,
                    relation=LinkRelation.SUPPORTS,
                    source_event_id=existing.source_event_id,
                )
            )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_links:{newest.version_id}"


@pytest.mark.asyncio
async def test_experience_source_validator_rejects_cross_owner_link_identity(
    source_stack: Stack,
) -> None:
    status, foreign = await create(
        source_stack,
        key="foreign-target",
        owner_agent_id=OTHER_OWNER_ID,
        value=content("foreign-target"),
    )
    assert status == 201
    foreign_id = UUID(foreign["data"]["experience_id"])
    async with source_stack.database.read_session() as session:
        newest = await session.scalar(
            select(ExperienceVersionRow)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number.desc())
            .limit(1)
        )
    assert newest is not None
    async with source_stack.database.read_session() as session:
        link = await session.scalar(
            select(ExperienceLinkRow).where(
                ExperienceLinkRow.source_version_id == newest.version_id
            )
        )
    assert link is not None
    async with source_stack.database.read_session() as session:
        event = await session.get(DomainEventRow, link.source_event_id)
    assert event is not None
    payload = ExperienceVersionCreatedV1.model_validate_json(event.payload)
    tampered_payload = payload.model_copy(
        update={
            "links": (
                VersionLinkRefV1(
                    target_experience_id=foreign_id,
                    relation=link.relation,
                ),
            )
        }
    )
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER experience_links_reject_update"))
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            update(ExperienceLinkRow)
            .where(ExperienceLinkRow.source_version_id == newest.version_id)
            .values(target_experience_id=foreign_id)
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == event.event_id)
            .values(payload=canonical_json_bytes(tampered_payload))
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_links:{newest.version_id}"


@pytest.mark.asyncio
async def test_experience_source_validator_rejects_link_to_future_identity(
    source_stack: Stack,
) -> None:
    source_experience_id = EXPERIENCE_IDS[0]
    source_version_id = VERSION_IDS[0]
    future_target_id = EXPERIENCE_IDS[1]
    async with source_stack.database.read_session() as session:
        source_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == source_experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
        )
        target_creation_event = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_id == future_target_id,
                DomainEventRow.event_type == ExperienceCreatedV1.event_type,
            )
        )
    assert source_event is not None
    assert target_creation_event is not None
    assert target_creation_event.event_id > source_event.event_id

    payload = ExperienceVersionCreatedV1.model_validate_json(source_event.payload)
    tampered_payload = payload.model_copy(
        update={
            "links": (
                VersionLinkRefV1(
                    target_experience_id=future_target_id,
                    relation=LinkRelation.SUPPORTS,
                ),
            )
        }
    )
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER domain_events_reject_update")
        )
        uow.session.add(
            ExperienceLinkRow(
                source_experience_id=source_experience_id,
                source_version_id=source_version_id,
                target_experience_id=future_target_id,
                relation=LinkRelation.SUPPORTS,
                source_event_id=source_event.event_id,
            )
        )
        await uow.session.execute(
            update(DomainEventRow)
            .where(DomainEventRow.event_id == source_event.event_id)
            .values(payload=canonical_json_bytes(tampered_payload))
        )

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key == f"experience_links:{source_version_id}"


@pytest.mark.asyncio
async def test_experience_source_validator_rejects_duplicate_current_content(
    source_stack: Stack,
) -> None:
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            delete(ExperienceStateRow).where(
                ExperienceStateRow.experience_id == EXPERIENCE_IDS[0]
            )
        )
    status, duplicate = await create(
        source_stack,
        key="duplicate-current-source",
        value=content("target"),
    )
    assert status == 201
    duplicate_id = UUID(duplicate["data"]["experience_id"])

    validator = experience_validator(source_stack)
    with pytest.raises(validation.SourceIntegrityError) as caught:
        async with source_stack.database.read_session() as session:
            await validator.validate(session)

    assert caught.value.mismatch_key.startswith(
        f"experience_current_content:{OWNER_ID}:"
    )
    assert str(EXPERIENCE_IDS[0]) in str(caught.value)
    assert str(duplicate_id) in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["verify", "repair"])
async def test_projection_maintenance_stops_before_projection_on_source_damage(
    source_stack: Stack,
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with source_stack.database.read_session() as session:
        version_id = await session.scalar(
            select(ExperienceVersionRow.version_id)
            .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[1])
            .order_by(ExperienceVersionRow.version_number)
            .limit(1)
        )
    assert version_id is not None
    async with source_stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_payloads_reject_semantic_update")
        )
        await uow.session.execute(
            update(ExperiencePayloadRow)
            .where(ExperiencePayloadRow.version_id == version_id)
            .values(payload_hash="d" * 64)
        )
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == EXPERIENCE_IDS[1])
            .values(confidence=0.123456789)
        )

    rebuild_called = False

    async def unexpected_rebuild(*_args: object, **_kwargs: object) -> None:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("projection rebuild must not run before source validation")

    monkeypatch.setattr(source_stack.projector, "rebuild", unexpected_rebuild)
    with pytest.raises(validation.SourceIntegrityError):
        await getattr(source_stack.manager, operation)(source_stack.database)

    async with source_stack.database.read_session() as session:
        state = await session.get(ExperienceStateRow, EXPERIENCE_IDS[1])
        payload = await session.get(ExperiencePayloadRow, version_id)
    assert not rebuild_called
    assert state is not None and state.confidence == 0.123456789
    assert payload is not None and payload.payload_hash == "d" * 64
