from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.exc import IntegrityError, OperationalError

from experience_hub import canonical_json_bytes
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.content import (
    decode_payload,
    encode_version_content,
    reencode_payload,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.storage.database import Database, payload_rewrite_guard
from experience_hub.storage.payload_rewrite import rewrite_payload_codec
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdempotencyRecordRow,
)

NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


@pytest.fixture
def migrated_database_path(
    repository_root: Path,
    tmp_path: Path,
) -> Iterator[Path]:
    database_path = tmp_path / "payloads.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "head")
    yield database_path


@pytest.fixture
async def database(migrated_database_path: Path) -> AsyncIterator[Database]:
    database = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    try:
        yield database
    finally:
        await database.dispose()


@pytest.fixture
def sync_engine(migrated_database_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{migrated_database_path}")
    try:
        yield engine
    finally:
        engine.dispose()


def _version_content() -> VersionContent:
    return VersionContent(
        body="repeat " * 100,
        summary="Safe handoff",
        mechanism="lease handoff",
        tags=("ops",),
        applicability=("single writer",),
        evidence=(TypedEvidence(type="log", id="case-1"),),
        falsifiers=("overlapping owners",),
    )


async def _seed_experience(database: Database) -> dict[str, UUID | int | str]:
    owner_id = uuid4()
    target_id = uuid4()
    version_id = uuid4()
    receipt_id = uuid4()
    encoded = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=_version_content(),
    )
    async with database.transaction() as uow:
        uow.session.add(AgentRow(agent_id=owner_id, name="Owner", created_at=NOW))
        uow.session.add(
            IdempotencyRecordRow(
                receipt_id=receipt_id,
                caller_scope="system:local",
                scope="experience.seed",
                idempotency_key="seed",
                request_hash="a" * 64,
                state="in_progress",
                result_resource_type=None,
                result_resource_id=None,
                response_status_code=None,
                response_body=None,
                response_content_type=None,
                response_headers=None,
                created_at=NOW,
                completed_at=None,
            )
        )
        await uow.session.flush()
        event = DomainEventRow(
            aggregate_type="experience",
            aggregate_id=version_id,
            sequence=1,
            event_type="experience.version_created",
            payload=canonical_json_bytes({"schema_version": 1}),
            actor_agent_id=owner_id,
            causation_id=receipt_id,
            occurred_at=NOW,
        )
        uow.session.add(event)
        await uow.session.flush()
        event_id = event.event_id
        uow.session.add_all(
            [
                ExperienceRow(
                    experience_id=target_id,
                    owner_agent_id=owner_id,
                    kind=ExperienceKind.SEMANTIC,
                    origin=ExperienceOrigin.LOCAL,
                    created_at=NOW,
                ),
                ExperienceRow(
                    experience_id=version_id,
                    owner_agent_id=owner_id,
                    kind=ExperienceKind.PROCEDURAL,
                    origin=ExperienceOrigin.LOCAL,
                    created_at=NOW,
                ),
            ]
        )
        await uow.session.flush()
        uow.session.add(
            ExperienceVersionRow(
                version_id=version_id,
                experience_id=version_id,
                version_number=1,
                summary=_version_content().summary,
                mechanism=_version_content().mechanism,
                tags=canonical_json_bytes(_version_content().tags),
                applicability=canonical_json_bytes(
                    _version_content().applicability
                ),
                evidence=canonical_json_bytes(_version_content().evidence),
                falsifiers=canonical_json_bytes(_version_content().falsifiers),
                content_hash=encoded.content_hash,
                supersedes_version_id=None,
                created_at=NOW,
            )
        )
        await uow.session.flush()
        uow.session.add_all(
            [
                ExperiencePayloadRow(
                    version_id=version_id,
                    codec=encoded.codec,
                    payload=encoded.payload,
                    payload_hash=encoded.payload_hash,
                ),
                ExperienceLinkRow(
                    source_experience_id=version_id,
                    source_version_id=version_id,
                    target_experience_id=target_id,
                    relation=LinkRelation.SUPPORTS,
                    source_event_id=event_id,
                ),
                ExperienceStateRow(
                    experience_id=version_id,
                    owner_agent_id=owner_id,
                    current_version_id=version_id,
                    current_content_hash=encoded.content_hash,
                    temperature=Temperature.WARM,
                    importance=0.35,
                    confidence=0.50,
                    activation_score=0.48,
                    source_trust=1.0,
                    access_count=0,
                    access_strength=0.0,
                    strength_updated_at=NOW,
                    last_accessed_at=None,
                    last_transition_at=NOW,
                    last_lifecycle_evaluated_at=None,
                    consecutive_below_threshold=0,
                    pinned=False,
                    projection_event_id=event_id,
                ),
            ]
        )

    return {
        "owner_id": owner_id,
        "target_id": target_id,
        "experience_id": version_id,
        "version_id": version_id,
        "event_id": event_id,
        "payload_hash": encoded.payload_hash,
        "content_hash": encoded.content_hash,
    }


@pytest.mark.asyncio
async def test_identity_versions_and_links_reject_update_delete_and_replace(
    database: Database,
) -> None:
    seeded = await _seed_experience(database)
    cases = (
        (
            "UPDATE experiences SET origin = 'adopted_idea' "
            "WHERE experience_id = :id",
            "experiences rows are immutable",
        ),
        (
            "DELETE FROM experiences WHERE experience_id = :id",
            "experiences rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO experiences "
            "(experience_id, owner_agent_id, kind, origin, created_at) "
            "SELECT experience_id, owner_agent_id, kind, 'adopted_idea', created_at "
            "FROM experiences WHERE experience_id = :id",
            "experiences identity already exists",
        ),
        (
            "UPDATE experience_versions SET summary = 'changed' "
            "WHERE version_id = :id",
            "experience_versions rows are immutable",
        ),
        (
            "DELETE FROM experience_versions WHERE version_id = :id",
            "experience_versions rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO experience_versions "
            "SELECT * FROM experience_versions WHERE version_id = :id",
            "experience_versions identity already exists",
        ),
        (
            "UPDATE experience_links SET relation = 'tests' "
            "WHERE source_version_id = :id",
            "experience_links rows are immutable",
        ),
        (
            "DELETE FROM experience_links WHERE source_version_id = :id",
            "experience_links rows are immutable",
        ),
        (
            "INSERT OR REPLACE INTO experience_links "
            "SELECT * FROM experience_links WHERE source_version_id = :id",
            "experience_links identity already exists",
        ),
    )

    for statement, message in cases:
        with pytest.raises(IntegrityError, match=message):
            async with database.transaction() as uow:
                await uow.session.execute(
                    text(statement),
                    {"id": str(seeded["experience_id"])},
                )


@pytest.mark.asyncio
async def test_payload_semantic_identity_and_arbitrary_rewrites_are_rejected(
    database: Database,
) -> None:
    seeded = await _seed_experience(database)
    version_id = str(seeded["version_id"])
    cases = (
        (
            "DELETE FROM experience_payloads WHERE version_id = :id",
            "experience_payloads rows are immutable",
        ),
        (
            "UPDATE experience_payloads SET version_id = :new_id "
            "WHERE version_id = :id",
            "payload semantic identity is immutable",
        ),
        (
            "UPDATE experience_payloads SET payload_hash = :hash "
            "WHERE version_id = :id",
            "payload semantic identity is immutable",
        ),
        (
            "UPDATE experience_payloads SET codec = 'zlib', payload = :payload "
            "WHERE version_id = :id",
            "payload rewrite is not allowed",
        ),
        (
            "INSERT OR REPLACE INTO experience_payloads "
            "(version_id, codec, payload, payload_hash) "
            "SELECT version_id, codec, payload, payload_hash "
            "FROM experience_payloads WHERE version_id = :id",
            "experience_payloads identity already exists",
        ),
    )
    parameters = {
        "id": version_id,
        "new_id": str(uuid4()),
        "hash": "b" * 64,
        "payload": b"not-the-same",
    }

    for statement, message in cases:
        with pytest.raises(IntegrityError, match=message):
            async with database.transaction() as uow:
                await uow.session.execute(text(statement), parameters)


@pytest.mark.asyncio
async def test_safe_codec_rewrite_preserves_semantics_and_resets_guard(
    database: Database,
) -> None:
    seeded = await _seed_experience(database)
    version_id = seeded["version_id"]
    assert isinstance(version_id, UUID)

    async with database.transaction() as uow:
        before_guard = await uow.session.scalar(
            text("SELECT experience_hub_payload_rewrite_allowed()")
        )
        before = await uow.session.get(ExperiencePayloadRow, version_id)
        before_content_hash = await uow.session.scalar(
            select(ExperienceVersionRow.content_hash).where(
                ExperienceVersionRow.version_id == version_id
            )
        )
        assert before is not None
        before_payload = before.payload
        decoded_before = decode_payload(before.codec, before.payload)

        await rewrite_payload_codec(
            session=uow.session,
            version_id=version_id,
            codec=PayloadCodec.ZLIB,
        )

        after = await uow.session.get(ExperiencePayloadRow, version_id)
        after_content_hash = await uow.session.scalar(
            select(ExperienceVersionRow.content_hash).where(
                ExperienceVersionRow.version_id == version_id
            )
        )
        after_guard = await uow.session.scalar(
            text("SELECT experience_hub_payload_rewrite_allowed()")
        )
        assert after is not None
        assert after is before
        assert before_guard == after_guard == 0
        assert after.codec is PayloadCodec.ZLIB
        assert after.payload != before_payload
        assert decode_payload(after.codec, after.payload) == decoded_before
        assert after.payload_hash == seeded["payload_hash"]
        assert before_content_hash == after_content_hash == seeded["content_hash"]

        with pytest.raises(IntegrityError, match="payload rewrite is not allowed"):
            await uow.session.execute(
                text(
                    "UPDATE experience_payloads SET codec = 'plain' "
                    "WHERE version_id = :version_id"
                ),
                {"version_id": str(version_id)},
            )


@pytest.mark.asyncio
async def test_codec_rewrite_stale_cas_fails_closed_on_session_connection(
    database: Database,
) -> None:
    seeded = await _seed_experience(database)
    version_id = seeded["version_id"]
    assert isinstance(version_id, UUID)
    observed_connections: list[object] = []
    injected_rowcounts: list[int] = []
    original_payload: bytes
    sync_connection: object

    with pytest.raises(
        RuntimeError,
        match="Payload changed concurrently during codec rewrite",
    ):
        async with database.transaction() as uow:
            loaded = await uow.session.get(ExperiencePayloadRow, version_id)
            assert loaded is not None
            original_payload = loaded.payload
            replacement = reencode_payload(
                decode_payload(loaded.codec, loaded.payload),
                PayloadCodec.ZLIB,
            )
            async_connection = await uow.session.connection()
            sync_connection = async_connection.sync_connection

            def inject_competing_rewrite(
                connection: Any,
                cursor: Any,
                statement: str,
                parameters: Any,
                context: Any,
                executemany: bool,
            ) -> None:
                if not statement.lstrip().startswith(
                    "UPDATE experience_payloads"
                ):
                    return
                observed_connections.append(connection)
                cursor.execute(
                    "UPDATE experience_payloads "
                    "SET codec = ?, payload = ? "
                    "WHERE version_id = ?",
                    (
                        PayloadCodec.ZLIB.value,
                        replacement,
                        str(version_id),
                    ),
                )
                injected_rowcounts.append(cursor.rowcount)

            event.listen(
                sync_connection,
                "before_cursor_execute",
                inject_competing_rewrite,
            )
            try:
                await rewrite_payload_codec(
                    session=uow.session,
                    version_id=version_id,
                    codec=PayloadCodec.ZLIB,
                )
            finally:
                event.remove(
                    sync_connection,
                    "before_cursor_execute",
                    inject_competing_rewrite,
                )

    assert observed_connections == [sync_connection]
    assert injected_rowcounts == [1]
    async with database.read_session() as session:
        restored = await session.get(ExperiencePayloadRow, version_id)
        assert restored is not None
        assert restored.codec is PayloadCodec.PLAIN
        assert restored.payload == original_payload


@pytest.mark.asyncio
async def test_sync_engine_without_rewrite_udf_fails_closed_and_keeps_bytes(
    database: Database,
    sync_engine: Engine,
) -> None:
    seeded = await _seed_experience(database)
    version_id = seeded["version_id"]
    assert isinstance(version_id, UUID)
    query = text(
        "SELECT codec, payload, payload_hash FROM experience_payloads "
        "WHERE version_id = :version_id"
    )
    parameters = {"version_id": str(version_id)}

    with sync_engine.connect() as connection:
        before = connection.execute(query, parameters).one()

    with (
        pytest.raises(
            OperationalError,
            match="experience_hub_payload_rewrite_allowed",
        ),
        sync_engine.begin() as connection,
    ):
        connection.execute(
            text(
                "UPDATE experience_payloads "
                "SET codec = 'zlib', payload = :payload "
                "WHERE version_id = :version_id"
            ),
            {
                **parameters,
                "payload": b"untrusted physical rewrite",
            },
        )

    with sync_engine.connect() as connection:
        after = connection.execute(query, parameters).one()
    assert after == before


@pytest.mark.asyncio
async def test_payload_rewrite_guard_is_connection_local_and_defaults_closed(
    migrated_database_path: Path,
) -> None:
    first = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    second = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    try:
        async with (
            first.read_session() as first_session,
            second.read_session() as second_session,
        ):
            assert (
                await first_session.scalar(
                    text("SELECT experience_hub_payload_rewrite_allowed()")
                )
                == 0
            )
            assert (
                await second_session.scalar(
                    text("SELECT experience_hub_payload_rewrite_allowed()")
                )
                == 0
            )
    finally:
        await first.dispose()
        await second.dispose()


@pytest.mark.asyncio
async def test_payload_rewrite_guard_resets_on_real_pool_recycle(
    database: Database,
) -> None:
    marker_name = "payload_guard_pool_marker"
    async with database.read_session() as original:
        await original.execute(
            text(f"CREATE TEMP TABLE {marker_name} (value INTEGER)")
        )
        await original.commit()
        connection = await original.connection()
        assert (
            await original.scalar(
                text(
                    "SELECT count(*) FROM sqlite_temp_master "
                    "WHERE type = 'table' AND name = :name"
                ),
                {"name": marker_name},
            )
            == 1
        )

        with payload_rewrite_guard(connection):
            assert (
                await original.scalar(
                    text("SELECT experience_hub_payload_rewrite_allowed()")
                )
                == 1
            )
            await original.close()

            async with database.read_session() as recycled:
                assert (
                    await recycled.scalar(
                        text(
                            "SELECT count(*) FROM sqlite_temp_master "
                            "WHERE type = 'table' AND name = :name"
                        ),
                        {"name": marker_name},
                    )
                    == 1
                )
                assert (
                    await recycled.scalar(
                        text(
                            "SELECT "
                            "experience_hub_payload_rewrite_allowed()"
                        )
                    )
                    == 0
                )


@pytest.mark.asyncio
async def test_payload_rewrite_guard_isolated_while_open_and_resets_on_error(
    migrated_database_path: Path,
) -> None:
    first = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    second = Database.create(f"sqlite+aiosqlite:///{migrated_database_path}")
    try:
        async with (
            first.read_session() as first_session,
            second.read_session() as second_session,
        ):
            first_connection = await first_session.connection()
            with (
                pytest.raises(RuntimeError, match="forced guard exit"),
                payload_rewrite_guard(first_connection),
            ):
                assert (
                    await first_session.scalar(
                        text(
                            "SELECT "
                            "experience_hub_payload_rewrite_allowed()"
                        )
                    )
                    == 1
                )
                assert (
                    await second_session.scalar(
                        text(
                            "SELECT "
                            "experience_hub_payload_rewrite_allowed()"
                        )
                    )
                    == 0
                )
                raise RuntimeError("forced guard exit")

            assert (
                await first_session.scalar(
                    text("SELECT experience_hub_payload_rewrite_allowed()")
                )
                == 0
            )
    finally:
        await first.dispose()
        await second.dispose()
