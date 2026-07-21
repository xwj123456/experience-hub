from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from tests.integration.test_create_experience import (
    EXPERIENCE_IDS,
    NOW,
    OWNER_ID,
    VERSION_IDS,
    Stack,
    build_stack,
    create,
)
from tests.integration.test_create_experience_version import correct

from experience_hub.experiences import (
    ExperienceKind,
    ExperienceOrigin,
    PayloadCodec,
    Temperature,
)
from experience_hub.experiences import reconcile as reconcile_module
from experience_hub.experiences.reconcile import (
    PayloadReconcileIssue,
    PayloadReconciler,
    PayloadReconcileReport,
)
from experience_hub.storage.database import payload_rewrite_guard
from experience_hub.storage.payload_rewrite import rewrite_payload_codec
from experience_hub.storage.tables import (
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
)
from experience_hub.storage.validation import SourceIntegrityError


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[Stack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "payload-reconcile.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


@pytest.mark.asyncio
async def test_reconciles_all_temperatures_in_deterministic_order_without_hash_drift(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temperatures = (
        Temperature.HOT,
        Temperature.WARM,
        Temperature.COLD,
        Temperature.ARCHIVED,
    )
    for temperature in temperatures:
        status, _ = await create(stack, key=f"reconcile-{temperature.value}")
        assert status == 201

    async with stack.database.transaction(immediate=True) as uow:
        for experience_id, version_id, temperature in zip(
            EXPERIENCE_IDS,
            VERSION_IDS,
            temperatures,
            strict=True,
        ):
            await uow.session.execute(
                update(ExperienceStateRow)
                .where(ExperienceStateRow.experience_id == experience_id)
                .values(temperature=temperature)
            )
            if temperature in {Temperature.HOT, Temperature.WARM}:
                await rewrite_payload_codec(
                    session=uow.session,
                    version_id=version_id,
                    codec=PayloadCodec.ZLIB,
                )
        hashes_before: dict[UUID, str] = {
            version_id: payload_hash
            for version_id, payload_hash in (
                await uow.session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.payload_hash,
                    )
                )
            ).all()
        }

    rewrite_order: list[UUID] = []

    async def recording_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        rewrite_order.append(version_id)
        await rewrite_payload_codec(
            session=session,
            version_id=version_id,
            codec=codec,
        )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        recording_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
        report.errors,
    ) == (4, 0, 0, ())
    assert rewrite_order == list(VERSION_IDS)

    async with stack.database.read_session() as session:
        rows = (
            await session.execute(
                select(
                    ExperiencePayloadRow.version_id,
                    ExperiencePayloadRow.codec,
                    ExperiencePayloadRow.payload_hash,
                ).order_by(ExperiencePayloadRow.version_id)
            )
        ).all()

    assert [(row.version_id, row.codec) for row in rows] == [
        (VERSION_IDS[0], PayloadCodec.PLAIN),
        (VERSION_IDS[1], PayloadCodec.PLAIN),
        (VERSION_IDS[2], PayloadCodec.ZLIB),
        (VERSION_IDS[3], PayloadCodec.ZLIB),
    ]
    assert {row.version_id: row.payload_hash for row in rows} == hashes_before

    async with stack.database.transaction(immediate=True) as uow:
        second_report = await PayloadReconciler().run(uow=uow)
    assert (
        second_report.changed_count,
        second_report.skipped_count,
        second_report.error_count,
    ) == (0, 4, 0)
    assert rewrite_order == list(VERSION_IDS)


@pytest.mark.asyncio
async def test_reconciles_every_historical_version_in_version_number_order(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = await create(stack, key="history-v1")
    assert status == 201
    status, _ = await correct(
        stack,
        key="history-v2",
        experience_id=EXPERIENCE_IDS[0],
    )
    assert status == 201

    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == EXPERIENCE_IDS[0])
            .values(temperature=Temperature.COLD)
        )
        version_ids = tuple(
            (
                await uow.session.scalars(
                    select(ExperiencePayloadRow.version_id)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperiencePayloadRow.version_id,
                    )
                    .where(ExperienceVersionRow.experience_id == EXPERIENCE_IDS[0])
                    .order_by(ExperienceVersionRow.version_number)
                )
            ).all()
        )
        hashes_before: dict[UUID, str] = {
            version_id: payload_hash
            for version_id, payload_hash in (
                await uow.session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.payload_hash,
                    ).where(ExperiencePayloadRow.version_id.in_(version_ids))
                )
            ).all()
        }

    rewrite_order: list[UUID] = []

    async def recording_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        rewrite_order.append(version_id)
        await rewrite_payload_codec(
            session=session,
            version_id=version_id,
            codec=codec,
        )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        recording_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
    ) == (2, 0, 0)
    assert rewrite_order == list(version_ids)

    async with stack.database.read_session() as session:
        rows = tuple(
            (
                await session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.codec,
                        ExperiencePayloadRow.payload_hash,
                    ).where(ExperiencePayloadRow.version_id.in_(version_ids))
                )
            ).all()
        )
    assert {row.version_id: row.codec for row in rows} == {
        version_id: PayloadCodec.ZLIB for version_id in version_ids
    }
    assert {row.version_id: row.payload_hash for row in rows} == hashes_before


@pytest.mark.asyncio
async def test_semantically_corrupt_version_aborts_all_rewrites_with_error_report(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("valid-cold", "corrupt-cold"):
        status, _ = await create(stack, key=key)
        assert status == 201

    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id.in_(EXPERIENCE_IDS[:2]))
            .values(temperature=Temperature.COLD)
        )
        connection = await uow.session.connection()
        with payload_rewrite_guard(connection):
            await uow.session.execute(
                update(ExperiencePayloadRow)
                .where(
                    ExperiencePayloadRow.version_id == VERSION_IDS[1],
                )
                .values(payload=b"corrupt canonical body")
            )
        rows_before = tuple(
            (
                await uow.session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.codec,
                        ExperiencePayloadRow.payload,
                        ExperiencePayloadRow.payload_hash,
                    ).order_by(ExperiencePayloadRow.version_id)
                )
            ).all()
        )

    async def unexpected_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        raise AssertionError(
            f"rewrite called during failed preflight: {session}, {version_id}, {codec}"
        )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        unexpected_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
    ) == (0, 1, 1)
    assert [
        (
            issue.experience_id,
            issue.version_number,
            issue.version_id,
            issue.code,
        )
        for issue in report.errors
    ] == [
        (
            EXPERIENCE_IDS[1],
            1,
            VERSION_IDS[1],
            "semantic_validation_failed",
        )
    ]

    async with stack.database.read_session() as session:
        rows_after = tuple(
            (
                await session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.codec,
                        ExperiencePayloadRow.payload,
                        ExperiencePayloadRow.payload_hash,
                    ).order_by(ExperiencePayloadRow.version_id)
                )
            ).all()
        )
    assert rows_after == rows_before


@pytest.mark.asyncio
async def test_missing_payload_and_state_are_accounted_without_silent_omission(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("valid-source", "missing-payload", "missing-state"):
        status, _ = await create(stack, key=key)
        assert status == 201

    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id.in_(EXPERIENCE_IDS[:3]))
            .values(temperature=Temperature.COLD)
        )
        await uow.session.execute(
            text("DROP TRIGGER experience_payloads_reject_delete")
        )
        await uow.session.execute(
            delete(ExperiencePayloadRow).where(
                ExperiencePayloadRow.version_id == VERSION_IDS[1]
            )
        )
        await uow.session.execute(
            delete(ExperienceStateRow).where(
                ExperienceStateRow.experience_id == EXPERIENCE_IDS[2]
            )
        )

    async def unexpected_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        raise AssertionError(
            f"rewrite called with missing source: {session}, {version_id}, {codec}"
        )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        unexpected_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
    ) == (0, 1, 2)
    assert [
        (
            issue.experience_id,
            issue.version_id,
            issue.code,
        )
        for issue in report.errors
    ] == [
        (
            EXPERIENCE_IDS[1],
            VERSION_IDS[1],
            "missing_payload",
        ),
        (
            EXPERIENCE_IDS[2],
            VERSION_IDS[2],
            "missing_state",
        ),
    ]


@pytest.mark.asyncio
async def test_post_rewrite_validation_failure_rolls_back_the_whole_pass(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("rollback-first", "rollback-second"):
        status, _ = await create(stack, key=key)
        assert status == 201

    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id.in_(EXPERIENCE_IDS[:2]))
            .values(temperature=Temperature.COLD)
        )
        rows_before = tuple(
            (
                await uow.session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.codec,
                        ExperiencePayloadRow.payload,
                        ExperiencePayloadRow.payload_hash,
                    ).order_by(ExperiencePayloadRow.version_id)
                )
            ).all()
        )

    async def corrupt_after_guarded_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        await rewrite_payload_codec(
            session=session,
            version_id=version_id,
            codec=codec,
        )
        if version_id == VERSION_IDS[1]:
            connection = await session.connection()
            with payload_rewrite_guard(connection):
                await session.execute(
                    update(ExperiencePayloadRow)
                    .where(ExperiencePayloadRow.version_id == version_id)
                    .values(payload=b"post-rewrite corruption")
                )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        corrupt_after_guarded_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
    ) == (0, 1, 1)
    assert [
        (
            issue.experience_id,
            issue.version_id,
            issue.code,
        )
        for issue in report.errors
    ] == [
        (
            EXPERIENCE_IDS[1],
            VERSION_IDS[1],
            "rewrite_validation_failed",
        )
    ]

    async with stack.database.read_session() as session:
        rows_after = tuple(
            (
                await session.execute(
                    select(
                        ExperiencePayloadRow.version_id,
                        ExperiencePayloadRow.codec,
                        ExperiencePayloadRow.payload,
                        ExperiencePayloadRow.payload_hash,
                    ).order_by(ExperiencePayloadRow.version_id)
                )
            ).all()
        )
    assert rows_after == rows_before


@pytest.mark.asyncio
async def test_noop_rewrite_is_reported_as_post_validation_failure(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = await create(stack, key="noop-rewrite")
    assert status == 201
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == EXPERIENCE_IDS[0])
            .values(temperature=Temperature.COLD)
        )

    async def noop_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        del session, version_id, codec

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        noop_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        report = await PayloadReconciler().run(uow=uow)

    assert (
        report.changed_count,
        report.skipped_count,
        report.error_count,
    ) == (0, 0, 1)
    assert report.errors[0].code == "rewrite_validation_failed"
    async with stack.database.read_session() as session:
        payload = await session.get(ExperiencePayloadRow, VERSION_IDS[0])
    assert payload is not None
    assert payload.codec is PayloadCodec.PLAIN


@pytest.mark.asyncio
async def test_reconciler_requires_caller_owned_immediate_transaction(
    stack: Stack,
) -> None:
    async with stack.database.transaction() as uow:
        with pytest.raises(
            RuntimeError,
            match="caller-owned immediate UOW",
        ):
            await PayloadReconciler().run(uow=uow)


@pytest.mark.asyncio
async def test_reconciler_rejects_a_closed_immediate_uow(stack: Stack) -> None:
    async with stack.database.transaction(immediate=True) as uow:
        stale_uow = uow

    assert not stale_uow.session.in_transaction()
    with pytest.raises(
        RuntimeError,
        match="active caller-owned immediate UOW",
    ):
        await PayloadReconciler().run(uow=stale_uow)


@pytest.mark.asyncio
async def test_orphan_identity_aborts_reconciliation_with_stable_source_key(
    stack: Stack,
) -> None:
    orphan_id = UUID("00000000-0000-0000-0000-000000000299")
    async with stack.database.transaction(immediate=True) as uow:
        uow.session.add(
            ExperienceRow(
                experience_id=orphan_id,
                owner_agent_id=OWNER_ID,
                kind=ExperienceKind.SEMANTIC,
                origin=ExperienceOrigin.LOCAL,
                created_at=NOW,
            )
        )

    async with stack.database.transaction(immediate=True) as uow:
        with pytest.raises(SourceIntegrityError) as captured:
            await PayloadReconciler().run(uow=uow)

    assert captured.value.mismatch_key == f"experience_identity:{orphan_id}"


@pytest.mark.asyncio
async def test_programming_error_during_rewrite_is_not_hidden_in_a_report(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = await create(stack, key="programming-error")
    assert status == 201
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == EXPERIENCE_IDS[0])
            .values(temperature=Temperature.COLD)
        )

    async def broken_rewrite(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        del session, version_id, codec
        raise AssertionError("reconciler implementation bug")

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        broken_rewrite,
    )

    async with stack.database.transaction(immediate=True) as uow:
        with pytest.raises(AssertionError, match="implementation bug"):
            await PayloadReconciler().run(uow=uow)


@pytest.mark.asyncio
async def test_database_error_during_rewrite_is_not_hidden_in_a_report(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = await create(stack, key="database-error")
    assert status == 201
    async with stack.database.transaction(immediate=True) as uow:
        await uow.session.execute(
            update(ExperienceStateRow)
            .where(ExperienceStateRow.experience_id == EXPERIENCE_IDS[0])
            .values(temperature=Temperature.COLD)
        )

    async def unavailable_database(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
    ) -> None:
        del session, version_id, codec
        raise OperationalError(
            "UPDATE experience_payloads",
            {},
            RuntimeError("connection unavailable"),
        )

    monkeypatch.setattr(
        reconcile_module,
        "rewrite_payload_codec",
        unavailable_database,
    )

    async with stack.database.transaction(immediate=True) as uow:
        with pytest.raises(OperationalError, match="connection unavailable"):
            await PayloadReconciler().run(uow=uow)


def test_report_rejects_inconsistent_counts_and_error_order() -> None:
    first = PayloadReconcileIssue(
        experience_id=EXPERIENCE_IDS[0],
        version_number=1,
        version_id=VERSION_IDS[0],
        code="missing_payload",
    )
    second = PayloadReconcileIssue(
        experience_id=EXPERIENCE_IDS[1],
        version_number=1,
        version_id=VERSION_IDS[1],
        code="missing_state",
    )

    with pytest.raises(
        ValueError,
        match="error_count must equal",
    ):
        PayloadReconcileReport(
            changed_count=0,
            skipped_count=0,
            error_count=1,
        )
    with pytest.raises(
        ValueError,
        match="deterministic version order",
    ):
        PayloadReconcileReport(
            changed_count=0,
            skipped_count=0,
            error_count=2,
            errors=(second, first),
        )
    with pytest.raises(
        ValueError,
        match="changed_count must be a non-negative integer",
    ):
        PayloadReconcileReport(
            changed_count=-1,
            skipped_count=0,
            error_count=0,
        )
