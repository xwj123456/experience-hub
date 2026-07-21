from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_inspiration_run import (
    FakeGenerator,
    FakeSnapshotBuilder,
    Stack,
    build_stack,
    command,
    request,
)

from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.models import (
    InspirationOperator,
    SnapshotItem,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    IdeaStateRow,
    MechanismIncubationRow,
)


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


@dataclass(slots=True)
class BarrierGenerator(FakeGenerator):
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    arrivals: int = 0

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        self.arrivals += 1
        if self.arrivals == 2:
            self.ready.set()
        await self.ready.wait()
        return await FakeGenerator.generate(
            self,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operator=operator,
            branch_limit=branch_limit,
            output_token_limit=output_token_limit,
        )


@pytest.fixture
async def concurrent_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    generator = BarrierGenerator()
    builder = FakeSnapshotBuilder(
        item_ids=[uid(401), uid(402)],
    )
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "cross-run.sqlite3",
        generator=generator,
        snapshot_builder=builder,
    )
    try:
        yield value
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_concurrent_final_transactions_declare_contiguous_cluster_counts(
    concurrent_stack: Stack,
) -> None:
    first, second = await asyncio.gather(
        concurrent_stack.executor.execute(
            request=request(key="first"),
            run=command(),
        ),
        concurrent_stack.executor.execute(
            request=request(key="second"),
            run=command(),
        ),
    )
    assert first.status_code == second.status_code == 201
    assert (
        json.loads(first.body)["data"]["status"],
        json.loads(second.body)["data"]["status"],
    ) == ("completed", "completed")

    async with concurrent_stack.database.read_session() as session:
        generated = (
            await session.scalars(
                select(DomainEventRow)
                .where(
                    DomainEventRow.event_type
                    == "inspiration.idea_generated"
                )
                .order_by(DomainEventRow.event_id)
            )
        ).all()
        cluster = await session.scalar(select(MechanismIncubationRow))
        ideas = (
            await session.scalars(
                select(IdeaStateRow).order_by(IdeaStateRow.projection_event_id)
            )
        ).all()
    payloads = [json.loads(event.payload) for event in generated]
    assert [
        (
            payload["occurrence_count_before"],
            payload["occurrence_count_after"],
        )
        for payload in payloads
    ] == [(0, 1), (1, 2)]
    assert [
        (
            payload["distinct_snapshot_count_before"],
            payload["distinct_snapshot_count_after"],
        )
        for payload in payloads
    ] == [(0, 1), (1, 1)]
    assert cluster is not None
    assert cluster.occurrence_count == 2
    assert cluster.distinct_snapshot_count == 1
    assert cluster.maturity == "speculative"
    assert len(ideas) == 2
    assert payloads[0]["duplicate_relation"] is None
    assert payloads[1]["duplicate_relation"] == str(ideas[0].idea_id)
