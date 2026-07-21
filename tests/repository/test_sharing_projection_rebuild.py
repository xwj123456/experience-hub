from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import text
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    CAPSULE_ID,
    ITEM_ID,
    PUBLISHER_ID,
    AdoptionStack,
    arrange_pending_capsule,
    build_stack,
)
from tests.integration.test_capsule_feedback import record_feedback
from tests.integration.test_capsule_rejection import reject

from experience_hub.canonical import canonical_json_bytes
from experience_hub.sharing.models import FeedbackVerdict, ProvenanceHop
from experience_hub.sharing.projector import (
    AgentReputationProjector,
    CapsuleStateProjector,
    InboxItemProjector,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
)
from experience_hub.storage.validation import SourceIntegrityError

_SOURCE_TABLE_ORDER = {
    "topics": "topic_id",
    "subscriptions": "subscription_id",
    "experience_capsules": "capsule_id",
    "adoption_records": "adoption_id",
    "capsule_feedback": ("observer_agent_id, capsule_id, revision, feedback_id"),
    "domain_events": "event_id",
}
_PROJECTION_TABLE_ORDER = {
    "experience_state": "experience_id",
    "capsule_state": "capsule_id",
    "inbox_items": "item_id",
    "agent_reputation": "subject_agent_id, observer_agent_id",
}
_SHARING_PROJECTION_TABLE_ORDER = {
    name: ordering
    for name, ordering in _PROJECTION_TABLE_ORDER.items()
    if name != "experience_state"
}


def _manager(stack: AdoptionStack) -> ProjectionManager:
    manager = cast(Any, stack.database)._projection_applier
    assert isinstance(manager, ProjectionManager)
    return manager


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "sharing-projection-rebuild.sqlite3",
    )
    _manager(value).registry.register(AgentReputationProjector(value.registry))
    await arrange_pending_capsule(value)
    rejected = await reject(value, key="reject-before-sharing-rebuild")
    assert rejected.status_code == 200
    feedback = await record_feedback(
        value,
        key="feedback-before-sharing-rebuild",
        verdict=FeedbackVerdict.USEFUL,
    )
    assert feedback.status_code == 201
    assert (await _manager(value).verify(value.database)).matches
    try:
        yield value
    finally:
        await value.database.dispose()


async def _table_rows(
    stack: AdoptionStack,
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


async def _projection_versions(
    stack: AdoptionStack,
) -> tuple[tuple[Any, ...], ...]:
    async with stack.database.read_session() as session:
        rows = await session.execute(
            text(
                "SELECT name, reducer_version, last_applied_event_id, "
                "last_verified_hash, last_verified_at "
                "FROM projection_versions ORDER BY name"
            )
        )
        return tuple(tuple(row) for row in rows)


async def _temp_rebuild_table_count(stack: AdoptionStack) -> int:
    async with stack.database.read_session() as session:
        count = await session.scalar(
            text("SELECT count(*) FROM sqlite_temp_master WHERE name LIKE '_rebuild_%'")
        )
    assert count is not None
    return int(count)


@pytest.mark.parametrize(
    ("projection", "expected_key"),
    (
        ("capsule_state", str(CAPSULE_ID)),
        ("inbox_items", str(ITEM_ID)),
        (
            "agent_reputation",
            json.dumps(
                [str(PUBLISHER_ID), str(ADOPTER_ID)],
                separators=(",", ":"),
            ),
        ),
    ),
)
@pytest.mark.asyncio
async def test_sharing_projection_mismatch_key_is_stable_and_repair_is_exact(
    stack: AdoptionStack,
    projection: str,
    expected_key: str,
) -> None:
    golden_sources = await _table_rows(stack, _SOURCE_TABLE_ORDER)
    golden_projections = await _table_rows(
        stack,
        _PROJECTION_TABLE_ORDER,
    )
    async with stack.database.transaction() as uow:
        if projection == "capsule_state":
            await uow.session.execute(
                text(
                    "UPDATE capsule_state SET status = 'retracted' "
                    "WHERE capsule_id = :capsule_id"
                ),
                {"capsule_id": str(CAPSULE_ID)},
            )
        elif projection == "inbox_items":
            await uow.session.execute(
                text(
                    "UPDATE inbox_items SET state = 'pending' WHERE item_id = :item_id"
                ),
                {"item_id": str(ITEM_ID)},
            )
        else:
            await uow.session.execute(
                text(
                    "UPDATE agent_reputation "
                    "SET useful_count = 0, refuted_count = 0, "
                    "harmful_count = 1, alpha = 2, beta = 3 "
                    "WHERE subject_agent_id = :subject_agent_id "
                    "AND observer_agent_id = :observer_agent_id"
                ),
                {
                    "subject_agent_id": str(PUBLISHER_ID),
                    "observer_agent_id": str(ADOPTER_ID),
                },
            )

    observed_keys: list[tuple[str, ...]] = []
    for _ in range(2):
        with pytest.raises(ProjectionMismatch) as caught:
            await _manager(stack).verify(stack.database)
        differences = caught.value.report.differences
        assert tuple(item.projection for item in differences) == (projection,)
        observed_keys.append(differences[0].differing_keys)
    assert observed_keys == [(expected_key,), (expected_key,)]

    repaired = await _manager(stack).repair(stack.database)

    assert repaired.matches
    assert await _table_rows(stack, _PROJECTION_TABLE_ORDER) == golden_projections
    assert await _table_rows(stack, _SOURCE_TABLE_ORDER) == golden_sources
    assert await _temp_rebuild_table_count(stack) == 0


async def _corrupt_capsule_source(
    stack: AdoptionStack,
    *,
    corruption: str,
) -> None:
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_capsules_reject_update")
        )
        if corruption == "capsule_hash":
            await uow.session.execute(
                text(
                    "UPDATE experience_capsules "
                    "SET capsule_hash = "
                    "CASE WHEN capsule_hash = :first THEN :second ELSE :first END "
                    "WHERE capsule_id = :capsule_id"
                ),
                {
                    "first": "0" * 64,
                    "second": "1" * 64,
                    "capsule_id": str(CAPSULE_ID),
                },
            )
            return
        damaged_chain = (
            ProvenanceHop(
                capsule_id=CAPSULE_ID,
                publisher_agent_id=PUBLISHER_ID,
            ),
        )
        await uow.session.execute(
            text(
                "UPDATE experience_capsules "
                "SET provenance_chain = :provenance_chain, hop_count = 1 "
                "WHERE capsule_id = :capsule_id"
            ),
            {
                "provenance_chain": canonical_json_bytes(damaged_chain),
                "capsule_id": str(CAPSULE_ID),
            },
        )


@pytest.mark.parametrize("operation", ("verify", "repair"))
@pytest.mark.parametrize(
    "corruption",
    ("capsule_hash", "provenance_chain"),
)
@pytest.mark.asyncio
async def test_corrupt_capsule_source_aborts_atomically_before_projection_swap(
    stack: AdoptionStack,
    operation: str,
    corruption: str,
) -> None:
    await _corrupt_capsule_source(stack, corruption=corruption)
    damaged_sources = await _table_rows(stack, _SOURCE_TABLE_ORDER)
    before_projections = await _table_rows(
        stack,
        _PROJECTION_TABLE_ORDER,
    )
    before_versions = await _projection_versions(stack)

    with pytest.raises(SourceIntegrityError) as caught:
        await getattr(_manager(stack), operation)(stack.database)

    assert caught.value.mismatch_key == f"capsule:{CAPSULE_ID}"
    assert await _table_rows(stack, _PROJECTION_TABLE_ORDER) == before_projections
    assert await _projection_versions(stack) == before_versions
    assert await _table_rows(stack, _SOURCE_TABLE_ORDER) == damaged_sources
    assert await _temp_rebuild_table_count(stack) == 0


@pytest.mark.asyncio
async def test_each_registered_sharing_reducer_replays_without_online_tables(
    stack: AdoptionStack,
) -> None:
    manager = _manager(stack)
    reducers = {reducer.name: reducer for reducer in manager.registry.reducers}
    assert isinstance(reducers.get("capsule_state"), CapsuleStateProjector)
    assert isinstance(reducers.get("inbox_items"), InboxItemProjector)
    assert isinstance(
        reducers.get("agent_reputation"),
        AgentReputationProjector,
    )
    assert {
        name: reducers[name].version for name in _SHARING_PROJECTION_TABLE_ORDER
    } == {
        "capsule_state": 1,
        "inbox_items": 1,
        "agent_reputation": 1,
    }
    golden = await _table_rows(stack, _SHARING_PROJECTION_TABLE_ORDER)

    async with stack.database.transaction() as uow:
        for table_name in _SHARING_PROJECTION_TABLE_ORDER:
            await uow.session.execute(
                text(f"ALTER TABLE {table_name} RENAME TO detached_{table_name}")
            )
        for table_name in _SHARING_PROJECTION_TABLE_ORDER:
            await reducers[table_name].rebuild(
                uow.session,
                "source_only_",
            )
        replayed = {
            table_name: tuple(
                tuple(row)
                for row in await uow.session.execute(
                    text(
                        f"SELECT * FROM temp.source_only_{table_name} "
                        f"ORDER BY {_SHARING_PROJECTION_TABLE_ORDER[table_name]}"
                    )
                )
            )
            for table_name in _SHARING_PROJECTION_TABLE_ORDER
        }
        for table_name in _SHARING_PROJECTION_TABLE_ORDER:
            await uow.session.execute(text(f"DROP TABLE temp.source_only_{table_name}"))
            await uow.session.execute(
                text(f"ALTER TABLE detached_{table_name} RENAME TO {table_name}")
            )

    assert replayed == golden
