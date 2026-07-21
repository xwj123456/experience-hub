from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select, update
from tests.integration.test_create_experience import (
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    Stack,
    build_stack,
    content,
    create,
)
from tests.integration.test_create_experience_version import (
    append_unprojected_same_aggregate_event,
    correct,
)

from experience_hub.experiences.contracts import VersionLinkInput
from experience_hub.experiences.models import LinkRelation, Temperature
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.storage.tables import ExperienceStateRow, LifecycleLeaseRow
from experience_hub.storage.validation import SourceIntegrityError


@pytest.fixture
async def lifecycle_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "lifecycle-repository.sqlite3",
    )
    try:
        yield stack
    finally:
        await stack.database.dispose()


@pytest.mark.asyncio
async def test_lease_claim_respects_owner_expiry_and_owner_only_release(
    lifecycle_stack: Stack,
) -> None:
    repository = LifecycleRepository()
    acquired_at = datetime(2026, 7, 18, 9, tzinfo=UTC)
    ttl = timedelta(seconds=30)

    async with lifecycle_stack.database.transaction(immediate=True) as uow:
        await repository.ensure_lease(uow.session)
        assert await repository.claim_lease(
            uow.session,
            owner_id=OWNER_ID,
            at=acquired_at,
            ttl=ttl,
        )
        assert not await repository.claim_lease(
            uow.session,
            owner_id=OTHER_OWNER_ID,
            at=acquired_at + ttl - timedelta(microseconds=1),
            ttl=ttl,
        )
        assert not await repository.release_lease(
            uow.session,
            owner_id=OTHER_OWNER_ID,
        )
        assert await repository.claim_lease(
            uow.session,
            owner_id=OTHER_OWNER_ID,
            at=acquired_at + ttl,
            ttl=ttl,
        )
        assert not await repository.release_lease(
            uow.session,
            owner_id=OWNER_ID,
        )
        assert await repository.release_lease(
            uow.session,
            owner_id=OTHER_OWNER_ID,
        )

    async with lifecycle_stack.database.read_session() as session:
        rows = tuple((await session.scalars(select(LifecycleLeaseRow))).all())

    assert len(rows) == 1
    assert rows[0].lease_name == "lifecycle"
    assert (
        rows[0].owner_id,
        rows[0].acquired_at,
        rows[0].expires_at,
    ) == (None, None, None)


@pytest.mark.asyncio
async def test_same_owner_can_renew_lease_from_new_utc_anchor(
    lifecycle_stack: Stack,
) -> None:
    repository = LifecycleRepository()
    first_at = datetime(2026, 7, 18, 9, tzinfo=UTC)
    renewed_at = first_at + timedelta(seconds=10)

    async with lifecycle_stack.database.transaction(immediate=True) as uow:
        assert await repository.claim_lease(
            uow.session,
            owner_id=OWNER_ID,
            at=first_at,
            ttl=timedelta(seconds=30),
        )
        assert await repository.claim_lease(
            uow.session,
            owner_id=OWNER_ID,
            at=renewed_at,
            ttl=timedelta(seconds=45),
        )

    async with lifecycle_stack.database.read_session() as session:
        row = await session.get(LifecycleLeaseRow, "lifecycle")

    assert row is not None
    assert (row.owner_id, row.acquired_at, row.expires_at) == (
        OWNER_ID,
        renewed_at,
        renewed_at + timedelta(seconds=45),
    )


@pytest.mark.parametrize(
    ("at", "ttl", "message"),
    [
        (
            datetime(2026, 7, 18, 9),
            timedelta(seconds=1),
            "timezone-aware",
        ),
        (
            datetime(2026, 7, 18, 9, tzinfo=UTC),
            timedelta(0),
            "positive",
        ),
        (
            datetime(2026, 7, 18, 9, tzinfo=UTC),
            timedelta(microseconds=-1),
            "positive",
        ),
    ],
)
@pytest.mark.asyncio
async def test_lease_claim_requires_aware_time_and_positive_ttl(
    lifecycle_stack: Stack,
    at: datetime,
    ttl: timedelta,
    message: str,
) -> None:
    repository = LifecycleRepository()

    async with lifecycle_stack.database.transaction(immediate=True) as uow:
        with pytest.raises(ValueError, match=message):
            await repository.claim_lease(
                uow.session,
                owner_id=OWNER_ID,
                at=at,
                ttl=ttl,
            )


@pytest.mark.asyncio
async def test_list_current_is_row_free_complete_and_uuid_byte_ordered(
    lifecycle_stack: Stack,
) -> None:
    _, first = await create(
        lifecycle_stack,
        key="first-lifecycle-record",
        value=content("first-lifecycle-record"),
    )
    _, second = await create(
        lifecycle_stack,
        key="second-lifecycle-record",
        owner_agent_id=OTHER_OWNER_ID,
        value=content("second-lifecycle-record"),
    )
    first_id = UUID(first["data"]["experience_id"])
    second_id = UUID(second["data"]["experience_id"])
    _, corrected = await correct(
        lifecycle_stack,
        key="corrected-lifecycle-record",
        experience_id=first_id,
        value=content("corrected-lifecycle-record"),
    )
    repository = LifecycleRepository()

    async with lifecycle_stack.database.read_session() as session:
        records = await repository.list_current(session)

    assert [record.experience_id for record in records] == sorted(
        (first_id, second_id),
        key=lambda value: value.bytes,
    )
    by_id = {record.experience_id: record for record in records}
    first_record = by_id[first_id]
    second_record = by_id[second_id]
    assert (
        first_record.created_at,
        first_record.current_version_id,
        first_record.current_version_number,
        first_record.current_version_created_at,
    ) == (
        NOW,
        UUID(corrected["data"]["version_id"]),
        2,
        NOW,
    )
    assert first_record.state.current_version_id == first_record.current_version_id
    assert first_record.state.temperature is Temperature.WARM
    assert first_record.latest_causal_at == NOW
    assert second_record.owner_agent_id == OTHER_OWNER_ID
    assert second_record.current_version_number == 1
    assert all(
        not hasattr(record, "_sa_instance_state")
        and not hasattr(record.state, "_sa_instance_state")
        for record in records
    )


@pytest.mark.asyncio
async def test_active_dependents_use_only_current_source_version_and_relations(
    lifecycle_stack: Stack,
) -> None:
    _, first_target = await create(
        lifecycle_stack,
        key="first-dependency-target",
        value=content("first-dependency-target"),
    )
    _, second_target = await create(
        lifecycle_stack,
        key="second-dependency-target",
        value=content("second-dependency-target"),
    )
    first_target_id = UUID(first_target["data"]["experience_id"])
    second_target_id = UUID(second_target["data"]["experience_id"])
    _, source = await create(
        lifecycle_stack,
        key="dependency-source",
        value=content("dependency-source"),
        links=(
            VersionLinkInput(
                target_experience_id=first_target_id,
                relation=LinkRelation.DERIVED_FROM,
            ),
        ),
    )
    source_id = UUID(source["data"]["experience_id"])
    await correct(
        lifecycle_stack,
        key="current-dependency-source",
        experience_id=source_id,
        value=content("current-dependency-source"),
        links=(
            VersionLinkInput(
                target_experience_id=second_target_id,
                relation=LinkRelation.SUPPORTS,
            ),
            VersionLinkInput(
                target_experience_id=second_target_id,
                relation=LinkRelation.TESTS,
            ),
            VersionLinkInput(
                target_experience_id=first_target_id,
                relation=LinkRelation.CONTRADICTS,
            ),
        ),
    )
    repository = LifecycleRepository()

    async with lifecycle_stack.database.read_session() as session:
        dependent_ids = await repository.active_dependent_target_ids(session)

    assert dependent_ids == frozenset({second_target_id})


@pytest.mark.asyncio
async def test_archived_source_does_not_block_its_dependency_targets(
    lifecycle_stack: Stack,
) -> None:
    _, target = await create(
        lifecycle_stack,
        key="archived-dependency-target",
        value=content("archived-dependency-target"),
    )
    target_id = UUID(target["data"]["experience_id"])
    _, source = await create(
        lifecycle_stack,
        key="archived-dependency-source",
        value=content("archived-dependency-source"),
        links=(
            VersionLinkInput(
                target_experience_id=target_id,
                relation=LinkRelation.DERIVED_FROM,
            ),
        ),
    )
    source_id = UUID(source["data"]["experience_id"])
    async with lifecycle_stack.database.transaction() as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == source_id)
            .values(temperature=Temperature.ARCHIVED)
        )
    repository = LifecycleRepository()

    async with lifecycle_stack.database.read_session() as session:
        dependent_ids = await repository.active_dependent_target_ids(session)

    assert dependent_ids == frozenset()


@pytest.mark.asyncio
async def test_list_current_refuses_projection_behind_aggregate_head(
    lifecycle_stack: Stack,
) -> None:
    _, created = await create(
        lifecycle_stack,
        key="lifecycle-head-source",
    )
    experience_id = UUID(created["data"]["experience_id"])
    await append_unprojected_same_aggregate_event(
        lifecycle_stack,
        experience_id=experience_id,
    )

    async with lifecycle_stack.database.read_session() as session:
        with pytest.raises(SourceIntegrityError, match="checkpoint"):
            await LifecycleRepository.list_current(session)
