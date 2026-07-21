from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select, text
from tests.integration.test_idea_adoption import (
    OWNER_A,
    AdoptionStack,
    SeededIdea,
    build_adoption_stack,
    create_experience,
    experience_spec,
    generate_idea,
)

from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    VersionContent,
)
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionMismatch,
)
from experience_hub.storage.tables import (
    IdeaOccurrenceRow,
    IdeaStateRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationRunStateRow,
    InspirationSnapshotItemRow,
)
from experience_hub.storage.validation import (
    SourceIntegrityError,
    SourceValidator,
    register_inspiration_source_validator,
)

SOURCE_CONTENT = VersionContent(
    body="Acknowledging bounded work releases capacity without changing ownership.",
    summary="A stable bounded-queue observation",
    mechanism="Acknowledgement releases bounded capacity.",
    tags=("queue", "capacity"),
    applicability=("bounded queue",),
    evidence=(TypedEvidence(type="experiment", id="stable-source"),),
    falsifiers=("Capacity remains blocked after acknowledgement.",),
)

_AUTHORITATIVE_SOURCE_TABLES = (
    "domain_events",
    "idempotency_records",
    "inspiration_runs",
    "inspiration_snapshot_items",
    "inspiration_ideas",
    "idea_occurrences",
)


@dataclass(frozen=True, slots=True)
class EquivalentHistory:
    first: SeededIdea
    second: SeededIdea


@dataclass(frozen=True, slots=True)
class ProjectionSnapshot:
    tables: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]
    versions: tuple[tuple[object, ...], ...]


def _enable_inspiration_source_validation(stack: AdoptionStack) -> None:
    validator = stack.manager._source_validator
    assert isinstance(validator, SourceValidator)
    register_inspiration_source_validator(validator)


async def _seed_equivalent_history(stack: AdoptionStack) -> EquivalentHistory:
    source = await create_experience(
        stack,
        owner_agent_id=OWNER_A,
        content=SOURCE_CONTENT,
        key="replay-source",
        kind=ExperienceKind.SEMANTIC,
        origin=ExperienceOrigin.LOCAL,
    )
    first = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="equivalent-run-one",
        specs=(
            experience_spec(
                marker=81,
                experience_id=source.experience_id,
                version_id=source.version_id,
                content_hash=source.content_hash,
            ),
        ),
    )
    second = await generate_idea(
        stack,
        owner_agent_id=OWNER_A,
        key="equivalent-run-two",
        specs=(
            experience_spec(
                marker=82,
                experience_id=source.experience_id,
                version_id=source.version_id,
                content_hash=source.content_hash,
            ),
        ),
    )
    return EquivalentHistory(first=first, second=second)


@pytest.fixture
async def replay_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[tuple[AdoptionStack, EquivalentHistory]]:
    stack = await build_adoption_stack(
        repository_root=repository_root,
        database_path=tmp_path / "inspiration-e2e-replay.sqlite3",
    )
    _enable_inspiration_source_validation(stack)
    try:
        yield stack, await _seed_equivalent_history(stack)
    finally:
        await stack.database.dispose()


async def _online_and_rebuilt_hashes(
    stack: AdoptionStack,
) -> tuple[int, dict[str, str], dict[str, str]]:
    manager = stack.manager
    assert isinstance(manager, ProjectionManager)
    async with stack.database.read_session() as session:
        event_head = await manager._event_head(session)
        rebuild = await manager._rebuild(session, event_head)
        try:
            online = await manager._hashes(session, rebuild, rebuilt=False)
            replayed = await manager._hashes(session, rebuild, rebuilt=True)
        finally:
            await manager._drop_temp_tables(session, rebuild.tables.values())
    return event_head, online, replayed


async def _projection_snapshot(stack: AdoptionStack) -> ProjectionSnapshot:
    tables: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    async with stack.database.read_session() as session:
        for reducer in stack.manager.registry.reducers:
            rows = tuple(
                tuple(row)
                for row in (
                    await session.execute(
                        text(f'SELECT * FROM "{reducer.name}" ORDER BY rowid')
                    )
                )
            )
            tables.append((reducer.name, rows))
        versions = tuple(
            tuple(row)
            for row in (
                await session.execute(
                    text("SELECT * FROM projection_versions ORDER BY name")
                )
            )
        )
    return ProjectionSnapshot(tables=tuple(tables), versions=versions)


async def _authoritative_source_snapshot(
    stack: AdoptionStack,
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    retained: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    async with stack.database.read_session() as session:
        for table in _AUTHORITATIVE_SOURCE_TABLES:
            rows = tuple(
                tuple(row)
                for row in (
                    await session.execute(
                        text(f'SELECT * FROM "{table}" ORDER BY rowid')
                    )
                )
            )
            retained.append((table, rows))
    return tuple(retained)


async def _corrupt_authoritative_source(
    stack: AdoptionStack,
    history: EquivalentHistory,
    corruption: str,
) -> None:
    first = str(history.first.run_id)
    second = str(history.second.run_id)
    bad_hash = "0" * 64
    async with stack.database.transaction() as uow:
        session = uow.session
        if corruption == "snapshot_stable_key":
            await session.execute(
                text("DROP TRIGGER inspiration_snapshot_items_reject_update")
            )
            await session.execute(
                text(
                    "UPDATE inspiration_snapshot_items "
                    "SET stable_evidence_key = :bad_hash "
                    "WHERE run_id = :run_id"
                ),
                {"bad_hash": bad_hash, "run_id": first},
            )
        elif corruption == "snapshot_event_hash":
            await session.execute(text("DROP TRIGGER domain_events_reject_update"))
            await session.execute(
                text(
                    "UPDATE domain_events "
                    "SET payload = CAST("
                    "json_set(CAST(payload AS TEXT), '$.snapshot_hash', :bad_hash) "
                    "AS BLOB) "
                    "WHERE aggregate_type = 'inspiration_run' "
                    "AND aggregate_id = :run_id "
                    "AND event_type = 'inspiration.snapshot_frozen'"
                ),
                {"bad_hash": bad_hash, "run_id": first},
            )
        elif corruption == "idea_content_hash":
            await session.execute(text("DROP TRIGGER inspiration_ideas_reject_update"))
            await session.execute(
                text(
                    "UPDATE inspiration_ideas "
                    "SET idea_content_hash = :bad_hash "
                    "WHERE idea_id = :idea_id"
                ),
                {
                    "bad_hash": bad_hash,
                    "idea_id": str(history.first.idea_id),
                },
            )
        elif corruption == "occurrence_snapshot_hash":
            await session.execute(text("DROP TRIGGER idea_occurrences_reject_update"))
            await session.execute(
                text(
                    "UPDATE idea_occurrences "
                    "SET snapshot_hash = :bad_hash "
                    "WHERE idea_id = :idea_id"
                ),
                {
                    "bad_hash": bad_hash,
                    "idea_id": str(history.first.idea_id),
                },
            )
        elif corruption == "terminal_time":
            await session.execute(text("DROP TRIGGER domain_events_reject_update"))
            await session.execute(
                text(
                    "UPDATE domain_events "
                    "SET occurred_at = '2000-01-01T00:00:00.000000Z' "
                    "WHERE event_id = ("
                    "SELECT max(event_id) FROM domain_events "
                    "WHERE aggregate_type = 'inspiration_run' "
                    "AND aggregate_id = :run_id)"
                ),
                {"run_id": first},
            )
        elif corruption == "receipt_resource_link":
            await session.execute(
                text(
                    "UPDATE idempotency_records "
                    "SET result_resource_id = :other_run_id "
                    "WHERE receipt_id = ("
                    "SELECT causation_id FROM domain_events "
                    "WHERE aggregate_type = 'inspiration_run' "
                    "AND aggregate_id = :run_id "
                    "ORDER BY sequence LIMIT 1)"
                ),
                {"other_run_id": second, "run_id": first},
            )
        else:  # pragma: no cover - guarded by the parameters below
            raise AssertionError(corruption)


async def _temporary_rebuild_table_count(stack: AdoptionStack) -> int:
    async with stack.database.read_session() as session:
        return int(
            await session.scalar(
                text(
                    "SELECT count(*) FROM sqlite_temp_master "
                    "WHERE name LIKE '_rebuild_%'"
                )
            )
            or 0
        )


@pytest.mark.asyncio
async def test_equivalent_runs_preserve_all_cross_run_domain_hashes(
    replay_stack: tuple[AdoptionStack, EquivalentHistory],
) -> None:
    stack, history = replay_stack
    run_ids = (history.first.run_id, history.second.run_id)
    async with stack.database.read_session() as session:
        runs = tuple(
            (
                await session.scalars(
                    select(InspirationRunRow)
                    .where(InspirationRunRow.run_id.in_(run_ids))
                    .order_by(InspirationRunRow.run_id)
                )
            ).all()
        )
        snapshots = tuple(
            (
                await session.scalars(
                    select(InspirationSnapshotItemRow)
                    .where(InspirationSnapshotItemRow.run_id.in_(run_ids))
                    .order_by(InspirationSnapshotItemRow.run_id)
                )
            ).all()
        )
        run_states = tuple(
            (
                await session.scalars(
                    select(InspirationRunStateRow)
                    .where(InspirationRunStateRow.run_id.in_(run_ids))
                    .order_by(InspirationRunStateRow.run_id)
                )
            ).all()
        )
        ideas = tuple(
            (
                await session.scalars(
                    select(InspirationIdeaRow)
                    .where(InspirationIdeaRow.run_id.in_(run_ids))
                    .order_by(InspirationIdeaRow.run_id)
                )
            ).all()
        )
        occurrences = tuple(
            (
                await session.scalars(
                    select(IdeaOccurrenceRow)
                    .where(IdeaOccurrenceRow.run_id.in_(run_ids))
                    .order_by(IdeaOccurrenceRow.run_id)
                )
            ).all()
        )
        idea_states = tuple(
            (
                await session.scalars(
                    select(IdeaStateRow)
                    .where(
                        IdeaStateRow.idea_id.in_(
                            (history.first.idea_id, history.second.idea_id)
                        )
                    )
                    .order_by(IdeaStateRow.idea_id)
                )
            ).all()
        )

    assert tuple(map(len, (runs, snapshots, run_states, ideas, occurrences))) == (
        2,
        2,
        2,
        2,
        2,
    )
    assert len(idea_states) == 2
    persisted_configuration = tuple(
        (
            row.goal,
            row.context,
            row.mode,
            row.generator_kind,
            row.generator_configuration,
            row.operators,
            row.include_inbox,
            row.branches_per_operator,
            row.output_tokens_per_operator,
            row.total_output_tokens,
            row.operator_timeout_seconds,
            row.global_timeout_seconds,
            row.request_hash,
        )
        for row in runs
    )
    assert persisted_configuration[0] == persisted_configuration[1]

    assert snapshots[0].snapshot_item_id != snapshots[1].snapshot_item_id
    assert snapshots[0].stable_evidence_key == snapshots[1].stable_evidence_key
    assert run_states[0].snapshot_hash == run_states[1].snapshot_hash
    assert ideas[0].idea_content_hash == ideas[1].idea_content_hash
    assert ideas[0].mechanism_hash == ideas[1].mechanism_hash
    assert occurrences[0].snapshot_hash == occurrences[1].snapshot_hash
    assert occurrences[0].mechanism_hash == occurrences[1].mechanism_hash
    assert (
        idea_states[0].mechanism_cluster_id
        == idea_states[1].mechanism_cluster_id
        == ideas[0].mechanism_hash
    )


@pytest.mark.asyncio
async def test_incremental_replay_and_repair_have_one_hash_at_one_event_head(
    replay_stack: tuple[AdoptionStack, EquivalentHistory],
) -> None:
    stack, _ = replay_stack
    report = await stack.manager.verify(stack.database)
    event_head, incremental, replayed = await _online_and_rebuilt_hashes(stack)

    assert report.matches
    assert report.event_head == event_head
    assert incremental == replayed
    original_hashes = incremental

    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE inspiration_run_state "
                "SET elapsed_milliseconds = elapsed_milliseconds + 1"
            )
        )
        await uow.session.execute(
            text(
                "UPDATE mechanism_incubation "
                "SET occurrence_count = occurrence_count + 1"
            )
        )
        await uow.session.execute(
            text("UPDATE idea_state SET owner_decision = 'archived'")
        )

    with pytest.raises(ProjectionMismatch) as caught:
        await stack.manager.verify(stack.database)
    assert {
        difference.projection for difference in caught.value.report.differences
    } == {
        "idea_state",
        "inspiration_run_state",
        "mechanism_incubation",
    }

    repaired = await stack.manager.repair(stack.database)
    repaired_head, repaired_online, repaired_replay = await _online_and_rebuilt_hashes(
        stack
    )

    assert repaired.matches
    assert repaired.event_head == repaired_head == event_head
    assert repaired_online == repaired_replay == original_hashes
    assert (await stack.manager.verify(stack.database)).matches


@pytest.mark.parametrize(
    "corruption",
    [
        "snapshot_stable_key",
        "snapshot_event_hash",
        "idea_content_hash",
        "occurrence_snapshot_hash",
        "terminal_time",
        "receipt_resource_link",
    ],
)
@pytest.mark.asyncio
async def test_repair_rejects_corrupt_sources_without_writing_projections(
    replay_stack: tuple[AdoptionStack, EquivalentHistory],
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack, history = replay_stack
    before = await _projection_snapshot(stack)

    await _corrupt_authoritative_source(stack, history, corruption)
    corrupt_source = await _authoritative_source_snapshot(stack)
    rebuild_calls = 0
    original_rebuild = stack.manager._rebuild

    async def tracked_rebuild(*args: object, **kwargs: object) -> object:
        nonlocal rebuild_calls
        rebuild_calls += 1
        return await original_rebuild(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(stack.manager, "_rebuild", tracked_rebuild)

    with pytest.raises(SourceIntegrityError):
        await stack.manager.repair(stack.database)

    assert rebuild_calls == 0
    assert await _authoritative_source_snapshot(stack) == corrupt_source
    assert await _projection_snapshot(stack) == before
    assert await _temporary_rebuild_table_count(stack) == 0
