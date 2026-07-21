from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import BaseModel
from sqlalchemy import select, text
from tests.integration.test_inspiration_run import (
    FakeSnapshotBuilder,
    Stack,
    build_stack,
    command,
    request,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import EventPayload, EventRegistry, StoredEvent
from experience_hub.inspiration.events import (
    InspirationIdeaGeneratedV1,
    InspirationOperatorCompletedV1,
    register_inspiration_events,
)
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationProjectionIntegrityError,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
)
from experience_hub.storage.tables import DomainEventRow

_PROJECTION_ORDER = {
    "inspiration_run_state": "run_id",
    "mechanism_incubation": "cluster_id",
    "idea_state": "idea_id",
}
_SECOND_SNAPSHOT_ITEM_ID = UUID("00000000-0000-0000-0000-000000000402")
_MISSING_ID = UUID("00000000-0000-0000-0000-000000000999")


def _manager(stack: Stack) -> ProjectionManager:
    manager = cast(Any, stack.database)._projection_applier
    assert isinstance(manager, ProjectionManager)
    return manager


def _projector(
    stack: Stack,
    projector_type: type[
        InspirationRunProjector | MechanismIncubationProjector | IdeaStateProjector
    ],
) -> InspirationRunProjector | MechanismIncubationProjector | IdeaStateProjector:
    projector = next(
        reducer
        for reducer in _manager(stack).registry.reducers
        if isinstance(reducer, projector_type)
    )
    return projector


def _event_registry() -> EventRegistry:
    registry = EventRegistry()
    register_inspiration_events(registry)
    return registry


def _stored_event(
    registry: EventRegistry,
    row: DomainEventRow,
) -> StoredEvent:
    return StoredEvent(
        event_id=row.event_id,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        sequence=row.sequence,
        event_type=row.event_type,
        payload=registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        ),
        actor_agent_id=row.actor_agent_id,
        causation_id=row.causation_id,
        occurred_at=row.occurred_at,
    )


def _validated_copy(
    payload: EventPayload,
    **updates: Any,
) -> EventPayload:
    document = payload.model_dump(mode="python", warnings=False)
    document.update(updates)
    payload_type = type(payload)
    assert issubclass(payload_type, BaseModel)
    return payload_type.model_validate(document, strict=True)


async def _projection_rows(
    stack: Stack,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    async with stack.database.read_session() as session:
        return {
            table_name: tuple(
                tuple(row)
                for row in await session.execute(
                    text(f"SELECT * FROM {table_name} ORDER BY {ordering}")
                )
            )
            for table_name, ordering in _PROJECTION_ORDER.items()
        }


@pytest.fixture
async def projected_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    builder = FakeSnapshotBuilder(
        item_ids=[
            UUID("00000000-0000-0000-0000-000000000401"),
            _SECOND_SNAPSHOT_ITEM_ID,
        ],
        content_hashes=["a" * 64, "b" * 64],
    )
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-projection-protocol.sqlite3",
        snapshot_builder=builder,
    )
    try:
        for key in ("projection-first", "projection-second"):
            response = await stack.executor.execute(
                request=request(key=key),
                run=command(),
            )
            assert response.status_code == 201
        yield stack
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_incremental_run_mechanism_and_idea_projections_equal_rebuild(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    online = await _projection_rows(stack)
    manager = _manager(stack)
    assert {reducer.name: reducer.version for reducer in manager.registry.reducers} == {
        "inspiration_run_state": 1,
        "mechanism_incubation": 1,
        "idea_state": 1,
    }

    prefix = "source_only_"
    async with stack.database.transaction() as uow:
        for reducer in manager.registry.reducers:
            await reducer.rebuild(uow.session, prefix)
        rebuilt = {
            table_name: tuple(
                tuple(row)
                for row in await uow.session.execute(
                    text(f"SELECT * FROM temp.{prefix}{table_name} ORDER BY {ordering}")
                )
            )
            for table_name, ordering in _PROJECTION_ORDER.items()
        }
        for table_name in _PROJECTION_ORDER:
            await uow.session.execute(text(f"DROP TABLE temp.{prefix}{table_name}"))

    assert rebuilt == online
    assert (await manager.verify(stack.database)).matches


@pytest.mark.parametrize(
    ("projection", "corruption_sql"),
    (
        (
            "inspiration_run_state",
            "UPDATE inspiration_run_state SET status = 'failed' "
            "WHERE run_id = (SELECT min(run_id) FROM inspiration_run_state)",
        ),
        (
            "mechanism_incubation",
            "UPDATE mechanism_incubation SET occurrence_count = occurrence_count + 1",
        ),
        (
            "idea_state",
            "UPDATE idea_state SET owner_decision = 'archived' "
            "WHERE idea_id = (SELECT min(idea_id) FROM idea_state)",
        ),
    ),
)
@pytest.mark.asyncio
async def test_projection_mismatch_is_detected_and_repair_restores_exact_rows(
    projected_stack: Stack,
    projection: str,
    corruption_sql: str,
) -> None:
    stack = projected_stack
    manager = _manager(stack)
    golden = await _projection_rows(stack)
    assert (await manager.verify(stack.database)).matches

    async with stack.database.transaction() as uow:
        await uow.session.execute(text(corruption_sql))

    with pytest.raises(ProjectionMismatch) as caught:
        await manager.verify(stack.database)
    assert tuple(
        difference.projection for difference in caught.value.report.differences
    ) == (projection,)
    assert caught.value.report.differences[0].differing_keys

    repaired = await manager.repair(stack.database)

    assert repaired.matches
    assert await _projection_rows(stack) == golden
    assert (await manager.verify(stack.database)).matches


@pytest.mark.asyncio
async def test_snapshot_hash_mismatch_repairs_all_projections_from_one_rebuild(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    manager = _manager(stack)
    golden = await _projection_rows(stack)
    async with stack.database.transaction() as uow:
        row = (
            await uow.session.execute(
                text(
                    "SELECT run_id, snapshot_hash FROM inspiration_run_state "
                    "ORDER BY run_id LIMIT 1"
                )
            )
        ).one()
        run_id = str(row.run_id)
        damaged_hash = "0" * 64 if row.snapshot_hash != "0" * 64 else "1" * 64
        await uow.session.execute(
            text(
                "UPDATE inspiration_run_state SET snapshot_hash=:snapshot_hash "
                "WHERE run_id=:run_id"
            ),
            {
                "snapshot_hash": damaged_hash,
                "run_id": run_id,
            },
        )

    mismatch_keys: list[tuple[str, ...]] = []
    for _ in range(2):
        with pytest.raises(ProjectionMismatch) as caught:
            await manager.verify(stack.database)
        assert tuple(
            difference.projection for difference in caught.value.report.differences
        ) == ("inspiration_run_state",)
        mismatch_keys.append(caught.value.report.differences[0].differing_keys)
    assert mismatch_keys == [(run_id,), (run_id,)]

    repaired = await manager.repair(stack.database)

    assert repaired.matches
    assert await _projection_rows(stack) == golden
    assert (await manager.verify(stack.database)).matches


@pytest.mark.parametrize(
    ("aggregate_type", "aggregate_id"),
    (
        ("idea", None),
        ("inspiration_run", _MISSING_ID),
    ),
)
@pytest.mark.asyncio
async def test_run_reducer_rejects_wrong_aggregate_anchor(
    projected_stack: Stack,
    aggregate_type: str,
    aggregate_id: UUID | None,
) -> None:
    stack = projected_stack
    registry = _event_registry()
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(DomainEventRow.event_type == "inspiration.started")
            .order_by(DomainEventRow.event_id)
            .limit(1)
        )
        assert row is not None
        stored = _stored_event(registry, row)
        forged = replace(
            stored,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id or stored.aggregate_id,
        )

        with pytest.raises(
            InspirationProjectionIntegrityError,
            match="invalid aggregate anchor",
        ):
            await _projector(stack, InspirationRunProjector).apply(
                session,
                forged,
            )


@pytest.mark.asyncio
async def test_run_reducer_rejects_valid_event_with_forged_before_counters(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    registry = _event_registry()
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.event_type == InspirationOperatorCompletedV1.event_type
            )
            .order_by(DomainEventRow.event_id)
            .limit(1)
        )
        assert row is not None
        stored = _stored_event(registry, row)
        assert isinstance(stored.payload, InspirationOperatorCompletedV1)
        payload = stored.payload
        predecessor = await session.scalar(
            select(DomainEventRow).where(
                DomainEventRow.aggregate_type == stored.aggregate_type,
                DomainEventRow.aggregate_id == stored.aggregate_id,
                DomainEventRow.sequence == stored.sequence - 1,
            )
        )
        assert predecessor is not None
        forged_payload = _validated_copy(
            payload,
            output_tokens_reserved_before=(payload.output_tokens_reserved_before + 1),
            output_tokens_reserved_after=(payload.output_tokens_reserved_after + 1),
        )
        forged = replace(stored, payload=forged_payload)
        await session.execute(
            text(
                "UPDATE inspiration_run_state SET status='running',"
                "operator_outcomes=:outcomes,output_tokens_reserved=0,"
                "output_tokens_consumed=0,elapsed_milliseconds=0,"
                "completed_at=NULL,projection_event_id=:predecessor_id "
                "WHERE run_id=:run_id"
            ),
            {
                "outcomes": canonical_json_bytes(()),
                "predecessor_id": predecessor.event_id,
                "run_id": str(payload.run_id),
            },
        )

        with pytest.raises(
            InspirationProjectionIntegrityError,
            match="locked budget state",
        ):
            await _projector(stack, InspirationRunProjector).apply(
                session,
                forged,
            )
        await session.rollback()


@pytest.mark.asyncio
async def test_mechanism_reducer_rejects_forged_cluster_transition(
    projected_stack: Stack,
) -> None:
    stack = projected_stack
    registry = _event_registry()
    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type
                        == InspirationIdeaGeneratedV1.event_type
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        assert len(rows) == 2
        first = _stored_event(registry, rows[0])
        second = _stored_event(registry, rows[1])
        assert isinstance(first.payload, InspirationIdeaGeneratedV1)
        assert isinstance(second.payload, InspirationIdeaGeneratedV1)
        first_transition: Mapping[str, Any] = {
            name: getattr(first.payload, name)
            for name in (
                "cluster_id",
                "canonical_mechanism_hash",
                "member_hashes_before",
                "member_hashes_after",
                "occurrence_count_before",
                "occurrence_count_after",
                "distinct_snapshot_count_before",
                "distinct_snapshot_count_after",
                "distinct_adopter_count_before",
                "distinct_adopter_count_after",
                "supported_count_before",
                "supported_count_after",
                "refuted_count_before",
                "refuted_count_after",
                "maturity_before",
                "maturity_after",
                "candidate_since_before",
                "candidate_since_after",
                "last_signal_at_before",
                "last_signal_at_after",
            )
        }
        forged_payload = _validated_copy(
            second.payload,
            **first_transition,
        )
        forged = replace(second, payload=forged_payload)

        with pytest.raises(
            InspirationProjectionIntegrityError,
            match="before-state does not match projection",
        ):
            await _projector(stack, MechanismIncubationProjector).apply(
                session,
                forged,
            )


@pytest.mark.parametrize("missing_anchor", ("idea", "occurrence", "run"))
@pytest.mark.asyncio
async def test_generated_idea_reducers_reject_missing_source_anchor(
    projected_stack: Stack,
    missing_anchor: str,
) -> None:
    stack = projected_stack
    registry = _event_registry()
    async with stack.database.read_session() as session:
        row = await session.scalar(
            select(DomainEventRow)
            .where(DomainEventRow.event_type == InspirationIdeaGeneratedV1.event_type)
            .order_by(DomainEventRow.event_id)
            .limit(1)
        )
        assert row is not None
        stored = _stored_event(registry, row)
        assert isinstance(stored.payload, InspirationIdeaGeneratedV1)
        field = {
            "idea": "idea_id",
            "occurrence": "occurrence_id",
            "run": "run_id",
        }[missing_anchor]
        forged_payload = _validated_copy(
            stored.payload,
            **{field: _MISSING_ID},
        )
        forged = replace(
            stored,
            aggregate_id=(
                _MISSING_ID if missing_anchor == "idea" else stored.aggregate_id
            ),
            payload=forged_payload,
        )

        for projector_type in (
            MechanismIncubationProjector,
            IdeaStateProjector,
        ):
            with pytest.raises(
                InspirationProjectionIntegrityError,
                match="source anchor is missing",
            ):
                await _projector(stack, projector_type).apply(
                    session,
                    forged,
                )
