from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text
from tests.integration.test_create_experience import (
    Stack,
    build_stack,
    content,
    create,
)
from tests.integration.test_create_experience_version import correct

from experience_hub.experiences import LinkRelation
from experience_hub.experiences.contracts import VersionLinkInput
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.storage.projections import ProjectionMismatch

_SOURCE_TABLE_ORDER = {
    "experiences": "experience_id",
    "experience_versions": "experience_id, version_number, version_id",
    "experience_payloads": "version_id",
    "experience_links": ("source_version_id, target_experience_id, relation"),
    "domain_events": "event_id",
}
_PROJECTION_TABLE_ORDER = {
    "experience_state": "experience_id",
    "experience_terms": "experience_id, term, term_kind",
}


@pytest.fixture
async def projected_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "experience-projection-rebuild.sqlite3",
    )
    stack.manager.registry.register(ExperienceTermsProjector(stack.registry))
    try:
        yield stack
    finally:
        await stack.database.dispose()


async def _seed_complete_history(stack: Stack) -> UUID:
    target_status, target = await create(
        stack,
        key="projection-anchor",
        value=content("投影 anchor"),
    )
    target_id = UUID(target["data"]["experience_id"])
    linked_status, linked = await create(
        stack,
        key="projection-linked",
        value=content("projection linked 初版"),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.SUPPORTS,
            ),
        ),
    )
    linked_id = UUID(linked["data"]["experience_id"])
    corrected_status, _ = await correct(
        stack,
        key="projection-corrected",
        experience_id=linked_id,
        value=content("projection linked 修订版"),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.DERIVED_FROM,
            ),
        ),
    )
    assert (target_status, linked_status, corrected_status) == (201, 201, 201)
    return linked_id


async def _table_rows(
    stack: Stack,
    tables: dict[str, str],
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    async with stack.database.read_session() as session:
        snapshots: dict[str, tuple[tuple[Any, ...], ...]] = {}
        for table_name, ordering in tables.items():
            result = await session.execute(
                text(f"SELECT * FROM {table_name} ORDER BY {ordering}")
            )
            snapshots[table_name] = tuple(tuple(row) for row in result)
    return snapshots


async def _assert_exact_repair(
    stack: Stack,
    *,
    expected_projection: str,
    expected_key: str,
    golden_sources: dict[str, tuple[tuple[Any, ...], ...]],
    golden_projections: dict[str, tuple[tuple[Any, ...], ...]],
) -> None:
    mismatches: list[tuple[str, ...]] = []
    for _ in range(2):
        with pytest.raises(ProjectionMismatch) as caught:
            await stack.manager.verify(stack.database)
        assert tuple(
            difference.projection for difference in caught.value.report.differences
        ) == (expected_projection,)
        keys = caught.value.report.differences[0].differing_keys
        mismatches.append(keys)
    assert mismatches == [(expected_key,), (expected_key,)]

    report = await stack.manager.repair(stack.database)

    assert report.matches
    assert await _table_rows(stack, _PROJECTION_TABLE_ORDER) == golden_projections
    assert await _table_rows(stack, _SOURCE_TABLE_ORDER) == golden_sources
    async with stack.database.read_session() as session:
        result = await session.execute(
            text(
                "SELECT name, reducer_version "
                "FROM projection_versions "
                "WHERE name IN ('experience_state', 'experience_terms') "
                "ORDER BY name"
            )
        )
        versions = tuple((str(row.name), int(row.reducer_version)) for row in result)
    assert versions == (
        ("experience_state", 1),
        ("experience_terms", 1),
    )


@pytest.mark.asyncio
async def test_state_mismatch_has_stable_key_and_repair_is_byte_exact(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    linked_id = await _seed_complete_history(stack)
    assert stack.projector.version == 1
    golden_sources = await _table_rows(stack, _SOURCE_TABLE_ORDER)
    golden_projections = await _table_rows(stack, _PROJECTION_TABLE_ORDER)

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE experience_state SET confidence = 0.123 "
                "WHERE experience_id = :experience_id"
            ),
            {"experience_id": str(linked_id)},
        )

    await _assert_exact_repair(
        stack,
        expected_projection="experience_state",
        expected_key=str(linked_id),
        golden_sources=golden_sources,
        golden_projections=golden_projections,
    )


@pytest.mark.asyncio
async def test_terms_mismatch_has_stable_key_and_repair_is_byte_exact(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    linked_id = await _seed_complete_history(stack)
    terms_projector = next(
        reducer
        for reducer in stack.manager.registry.reducers
        if reducer.name == "experience_terms"
    )
    assert terms_projector.version == 1
    golden_sources = await _table_rows(stack, _SOURCE_TABLE_ORDER)
    golden_projections = await _table_rows(stack, _PROJECTION_TABLE_ORDER)

    async with stack.database.transaction() as uow:
        term_row = (
            await uow.session.execute(
                text(
                    "SELECT term, term_kind FROM experience_terms "
                    "WHERE experience_id = :experience_id "
                    "ORDER BY term, term_kind LIMIT 1"
                ),
                {"experience_id": str(linked_id)},
            )
        ).one()
        await uow.session.execute(
            text(
                "UPDATE experience_terms SET weight = weight / 2.0 "
                "WHERE experience_id = :experience_id "
                "AND term = :term AND term_kind = :term_kind"
            ),
            {
                "experience_id": str(linked_id),
                "term": str(term_row.term),
                "term_kind": str(term_row.term_kind),
            },
        )

    expected_key = json.dumps(
        [str(linked_id), str(term_row.term), str(term_row.term_kind)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    await _assert_exact_repair(
        stack,
        expected_projection="experience_terms",
        expected_key=expected_key,
        golden_sources=golden_sources,
        golden_projections=golden_projections,
    )


@pytest.mark.asyncio
async def test_reducers_rebuild_from_sources_when_online_tables_are_detached(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    await _seed_complete_history(stack)
    golden_sources = await _table_rows(stack, _SOURCE_TABLE_ORDER)
    golden_projections = await _table_rows(stack, _PROJECTION_TABLE_ORDER)
    terms_projector = next(
        reducer
        for reducer in stack.manager.registry.reducers
        if reducer.name == "experience_terms"
    )

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text("ALTER TABLE experience_state RENAME TO detached_experience_state")
        )
        await uow.session.execute(
            text("ALTER TABLE experience_terms RENAME TO detached_experience_terms")
        )
        await stack.projector.rebuild(uow.session, "source_only_")
        await terms_projector.rebuild(uow.session, "source_only_")
        state_ddl = await uow.session.scalar(
            text(
                "SELECT sql FROM temp.sqlite_master "
                "WHERE type = 'table' "
                "AND name = 'source_only_experience_state'"
            )
        )
        assert state_ddl is not None
        assert "length(current_content_hash) = 64" in state_ddl
        assert "current_content_hash NOT GLOB '*[^0-9a-f]*'" in state_ddl
        state_rows = tuple(
            tuple(row)
            for row in await uow.session.execute(
                text(
                    "SELECT * FROM temp.source_only_experience_state "
                    "ORDER BY experience_id"
                )
            )
        )
        term_rows = tuple(
            tuple(row)
            for row in await uow.session.execute(
                text(
                    "SELECT * FROM temp.source_only_experience_terms "
                    "ORDER BY experience_id, term, term_kind"
                )
            )
        )
        await uow.session.execute(text("DROP TABLE temp.source_only_experience_state"))
        await uow.session.execute(text("DROP TABLE temp.source_only_experience_terms"))
        await uow.session.execute(
            text("ALTER TABLE detached_experience_state RENAME TO experience_state")
        )
        await uow.session.execute(
            text("ALTER TABLE detached_experience_terms RENAME TO experience_terms")
        )

    assert state_rows == golden_projections["experience_state"]
    assert term_rows == golden_projections["experience_terms"]
    assert await _table_rows(stack, _SOURCE_TABLE_ORDER) == golden_sources
