from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text
from tests.passport_export_fixtures import (
    create_export_experience,
    create_export_owner,
    export_runtime,
    seed_export_database,
)

from experience_hub.experiences.models import VersionContent
from experience_hub.passports import PassportDeclarationV1
from experience_hub.passports.errors import PassportError
from experience_hub.storage.readonly import readonly_sqlite_session

DECLARATION = PassportDeclarationV1(
    input_sanitized=True, profile_id="synthetic-v1", sharing_authorized=True
)


def _exporter():
    from experience_hub.passports.export import PassportExportService

    return PassportExportService()


async def test_closed_source_export_is_repeatable_and_non_mutating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source?精确.sqlite3"
    seed = await seed_export_database(path)
    before = path.read_bytes()
    sidecars = {
        item.name: item.read_bytes() for item in tmp_path.iterdir() if item != path
    }
    async with readonly_sqlite_session(path) as session:
        counts = (
            await session.execute(
                text(
                    "SELECT (SELECT count(*) FROM domain_events),"
                    "(SELECT count(*) FROM idempotency_records),"
                    "(SELECT count(*) FROM experience_state),"
                    "(SELECT count(*) FROM experience_terms)"
                )
            )
        ).one()
        first = await _exporter().export(
            session=session,
            owner_agent_id=seed.owner_agent_id,
            experience_id=seed.experience_id,
            version_id=None,
            declaration=DECLARATION,
        )
        second = await _exporter().export(
            session=session,
            owner_agent_id=seed.owner_agent_id,
            experience_id=seed.experience_id,
            version_id=seed.version_id,
            declaration=DECLARATION,
        )
        after = (
            await session.execute(
                text(
                    "SELECT (SELECT count(*) FROM domain_events),"
                    "(SELECT count(*) FROM idempotency_records),"
                    "(SELECT count(*) FROM experience_state),"
                    "(SELECT count(*) FROM experience_terms)"
                )
            )
        ).one()
    assert first.canonical_bytes == second.canonical_bytes
    assert first.document.subject.content_hash == seed.content_hash
    assert first.document.subject.source_version_id == seed.version_id
    assert first.report.reference_only_count == 1
    assert first.report.embedded_excerpt_count == 0
    assert counts == after
    assert path.read_bytes() == before
    assert {
        item.name: item.read_bytes() for item in tmp_path.iterdir() if item != path
    } == sidecars


async def test_selected_historical_version_not_current_content(tmp_path: Path) -> None:
    path = tmp_path / "historical.sqlite3"
    async with export_runtime(path).initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        owner = await create_export_owner(container, "publisher")
        first = await create_export_experience(container, owner)
        second = await create_export_experience(
            container,
            owner,
            experience_id=first.experience_id,
            key="next-version",
            content=VersionContent(
                body="New immutable content",
                summary="New summary",
                mechanism="New mechanism",
                tags=(),
                applicability=(),
                evidence=(),
                falsifiers=(),
            ),
        )
    async with readonly_sqlite_session(path) as session:
        historical = await _exporter().export(
            session=session,
            owner_agent_id=owner,
            experience_id=first.experience_id,
            version_id=first.version_id,
            declaration=DECLARATION,
        )
        current = await _exporter().export(
            session=session,
            owner_agent_id=owner,
            experience_id=first.experience_id,
            version_id=None,
            declaration=DECLARATION,
        )
    assert historical.document.subject.content_hash == first.content_hash
    assert current.document.subject.content_hash == second.content_hash
    assert historical.canonical_bytes != current.canonical_bytes


@pytest.mark.parametrize("foreign", [True, False])
async def test_foreign_and_missing_export_return_identical_error(
    tmp_path: Path, foreign: bool
) -> None:
    path = tmp_path / "isolation.sqlite3"
    seed = await seed_export_database(path)
    async with readonly_sqlite_session(path) as session:
        with pytest.raises(PassportError) as caught:
            await _exporter().export(
                session=session,
                owner_agent_id=UUID(int=9000) if foreign else seed.owner_agent_id,
                experience_id=seed.experience_id if foreign else UUID(int=9000),
                version_id=None,
                declaration=DECLARATION,
            )
    assert caught.value.code == "passport_not_found"
    assert caught.value.details == {}
