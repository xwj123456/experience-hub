from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from tests.integration.test_create_experience import (
    Stack,
    build_stack,
    create,
)
from tests.integration.test_create_experience_version import correct

from experience_hub.agents import AgentCreated
from experience_hub.domain import StoredEvent
from experience_hub.experiences import VersionContent
from experience_hub.experiences.events import (
    TASK2_EXPERIENCE_EVENT_TYPES,
    ExperienceVersionCreatedV1,
)
from experience_hub.experiences.projector import (
    ExperienceProjectionIntegrityError,
    ExperienceTermsProjector,
)
from experience_hub.retrieval.tokenizer import TermCue, index_version_terms
from experience_hub.storage.database import payload_rewrite_guard
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceTermRow,
    ProjectionVersionRow,
)


def searchable_content(label: str) -> VersionContent:
    return VersionContent(
        body=f"Cache {label} 记忆",
        summary=f"Summary {label}",
        mechanism=f"Lease-Handoff {label}",
        tags=("Memory Ops", label),
        applicability=("single writer", label),
        evidence=(),
        falsifiers=(),
    )


@pytest.fixture
async def projected_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[tuple[Stack, ExperienceTermsProjector]]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-terms.sqlite3",
    )
    terms_projector = ExperienceTermsProjector(stack.registry)
    stack.manager.registry.register(terms_projector)
    try:
        yield stack, terms_projector
    finally:
        await stack.database.dispose()


async def _term_rows(
    stack: Stack,
    experience_id: UUID,
) -> tuple[tuple[str, str, float], ...]:
    async with stack.database.read_session() as session:
        rows = (
            await session.execute(
                select(
                    ExperienceTermRow.term,
                    ExperienceTermRow.term_kind,
                    ExperienceTermRow.weight,
                )
                .where(ExperienceTermRow.experience_id == experience_id)
                .order_by(
                    ExperienceTermRow.term,
                    ExperienceTermRow.term_kind,
                )
            )
        ).all()
    return tuple((str(term), str(kind), float(weight)) for term, kind, weight in rows)


def _expected_rows(cues: tuple[TermCue, ...]) -> tuple[tuple[str, str, float], ...]:
    return tuple((cue.term, cue.term_kind, cue.weight) for cue in cues)


async def _latest_experience_event(
    stack: Stack,
    terms_projector: ExperienceTermsProjector,
    experience_id: UUID,
) -> StoredEvent:
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "experience",
                DomainEventRow.aggregate_id == experience_id,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
    assert row is not None
    return terms_projector.stored_event_from_row(row)


@pytest.mark.asyncio
async def test_creation_projects_canonical_version_terms_in_sorted_order(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, terms_projector = projected_stack
    value = searchable_content("Alpha")

    _, created = await create(
        stack,
        key="term-create",
        value=value,
    )
    experience_id = UUID(created["data"]["experience_id"])

    assert terms_projector.event_types == TASK2_EXPERIENCE_EVENT_TYPES
    assert await _term_rows(stack, experience_id) == _expected_rows(
        index_version_terms(value)
    )
    async with stack.database.read_session() as session:
        insertion_order = (
            await session.execute(
                text(
                    "SELECT term, term_kind FROM experience_terms "
                    "WHERE experience_id = :experience_id ORDER BY rowid"
                ),
                {"experience_id": str(experience_id)},
            )
        ).all()
        checkpoint = await session.get(
            ProjectionVersionRow,
            "experience_terms",
        )
        version_event_id = await session.scalar(
            select(DomainEventRow.event_id)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )

    assert tuple(insertion_order) == tuple(
        sorted(insertion_order, key=lambda item: (item[0], item[1]))
    )
    assert checkpoint is not None
    assert checkpoint.reducer_version == 1
    assert checkpoint.last_applied_event_id == version_event_id
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_correction_replaces_all_old_terms_instead_of_accumulating(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, _ = projected_stack
    original = searchable_content("ObsoleteMarker")
    replacement = searchable_content("ReplacementMarker")
    _, created = await create(
        stack,
        key="term-original",
        value=original,
    )
    experience_id = UUID(created["data"]["experience_id"])

    status, _ = await correct(
        stack,
        key="term-correction",
        experience_id=experience_id,
        value=replacement,
    )

    assert status == 201
    rows = await _term_rows(stack, experience_id)
    assert rows == _expected_rows(index_version_terms(replacement))
    assert ("obsoletemarker", "word", 1.0) not in rows
    assert ("replacementmarker", "word", 1.0) in rows
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_same_content_correction_keeps_terms_and_advances_checkpoint(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, _ = projected_stack
    value = searchable_content("Stable")
    _, created = await create(
        stack,
        key="term-stable-create",
        value=value,
    )
    experience_id = UUID(created["data"]["experience_id"])
    rows_before = await _term_rows(stack, experience_id)
    async with stack.database.read_session() as session:
        checkpoint_before = await session.get(
            ProjectionVersionRow,
            "experience_terms",
        )
        assert checkpoint_before is not None
        event_id_before = checkpoint_before.last_applied_event_id

    status, _ = await correct(
        stack,
        key="term-stable-correction",
        experience_id=experience_id,
        value=value,
    )

    assert status == 201
    assert await _term_rows(stack, experience_id) == rows_before
    async with stack.database.read_session() as session:
        checkpoint_after = await session.get(
            ProjectionVersionRow,
            "experience_terms",
        )
        latest_version_event_id = await session.scalar(
            select(DomainEventRow.event_id)
            .where(
                DomainEventRow.aggregate_id == experience_id,
                DomainEventRow.event_type
                == ExperienceVersionCreatedV1.event_type,
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
    assert checkpoint_after is not None
    assert checkpoint_after.last_applied_event_id > event_id_before
    assert checkpoint_after.last_applied_event_id == latest_version_event_id
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_punctuation_only_version_replaces_projection_with_no_terms(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, _ = projected_stack
    _, created = await create(
        stack,
        key="term-nonempty",
        value=searchable_content("Before"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    punctuation_only = VersionContent(
        body="!!!",
        summary="？？？",
        mechanism="——",
        tags=("...",),
        applicability=("，。",),
        evidence=(),
        falsifiers=(),
    )

    status, _ = await correct(
        stack,
        key="term-empty",
        experience_id=experience_id,
        value=punctuation_only,
    )

    assert status == 201
    assert index_version_terms(punctuation_only) == ()
    assert await _term_rows(stack, experience_id) == ()
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_temp_rebuild_matches_online_terms_without_mutating_main(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, terms_projector = projected_stack
    _, created = await create(
        stack,
        key="term-temp",
        value=searchable_content("Replay"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    online_before = await _term_rows(stack, experience_id)

    async with stack.database.read_session() as session, session.begin():
        await terms_projector.rebuild(session, "term_shadow_")
        rebuilt = (
            await session.execute(
                text(
                    "SELECT term, term_kind, weight "
                    "FROM temp.term_shadow_experience_terms "
                    "WHERE experience_id = :experience_id "
                    "ORDER BY term, term_kind"
                ),
                {"experience_id": str(experience_id)},
            )
        ).all()
        online_during = (
            await session.execute(
                text(
                    "SELECT term, term_kind, weight "
                    "FROM main.experience_terms "
                    "WHERE experience_id = :experience_id "
                    "ORDER BY term, term_kind"
                ),
                {"experience_id": str(experience_id)},
            )
        ).all()
        await session.execute(
            text("DROP TABLE temp.term_shadow_experience_terms")
        )

    assert tuple(rebuilt) == online_before
    assert tuple(online_during) == online_before
    assert await _term_rows(stack, experience_id) == online_before


@pytest.mark.asyncio
async def test_wrong_aggregate_event_fails_without_replacing_terms(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, terms_projector = projected_stack
    _, created = await create(
        stack,
        key="term-wrong-aggregate",
        value=searchable_content("Untouched"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    online_before = await _term_rows(stack, experience_id)
    event = await _latest_experience_event(
        stack,
        terms_projector,
        experience_id,
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="wrong aggregate type",
        ):
            await terms_projector.apply(
                uow.session,
                replace(event, aggregate_type="agent"),
            )

    assert await _term_rows(stack, experience_id) == online_before


@pytest.mark.asyncio
async def test_unsupported_event_fails_without_replacing_terms(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, terms_projector = projected_stack
    _, created = await create(
        stack,
        key="term-unsupported",
        value=searchable_content("Untouched"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    online_before = await _term_rows(stack, experience_id)
    event = await _latest_experience_event(
        stack,
        terms_projector,
        experience_id,
    )
    unsupported = replace(
        event,
        event_type=AgentCreated.event_type,
        payload=AgentCreated(
            schema_version=1,
            agent_id=experience_id,
            name="Unsupported",
        ),
    )

    async with stack.database.transaction() as uow:
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="Unsupported experience term event",
        ):
            await terms_projector.apply(uow.session, unsupported)

    assert await _term_rows(stack, experience_id) == online_before


@pytest.mark.asyncio
async def test_corrupt_source_aborts_rebuild_without_touching_main_terms(
    projected_stack: tuple[Stack, ExperienceTermsProjector],
) -> None:
    stack, terms_projector = projected_stack
    _, created = await create(
        stack,
        key="term-corrupt",
        value=searchable_content("Integrity"),
    )
    experience_id = UUID(created["data"]["experience_id"])
    version_id = UUID(created["data"]["version_id"])
    online_before = await _term_rows(stack, experience_id)

    async with stack.database.transaction() as uow:
        connection = await uow.session.connection()
        with payload_rewrite_guard(connection):
            await uow.session.execute(
                update(ExperiencePayloadRow)
                .where(ExperiencePayloadRow.version_id == version_id)
                .values(payload=b'{"body":"corrupt"}')
            )

    async with stack.database.read_session() as session, session.begin():
        with pytest.raises(
            ExperienceProjectionIntegrityError,
            match="source content is corrupt",
        ):
            await terms_projector.rebuild(session, "corrupt_shadow_")

    assert await _term_rows(stack, experience_id) == online_before
