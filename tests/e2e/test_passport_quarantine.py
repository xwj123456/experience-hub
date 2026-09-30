from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.passport_export_fixtures import (
    NOW,
    create_export_owner,
    export_runtime,
    seed_export_database,
)
from tests.passport_fixtures import passport_document
from typer.testing import CliRunner

import experience_hub.runtime as runtime_module
from experience_hub import canonical_json_bytes
from experience_hub.cli.app import app
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.ids import SequenceIdGenerator
from experience_hub.passports import encode_passport_document
from experience_hub.retrieval import RetrievalMode, SearchExperiences
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.tables import PassportAdoptionRow, PassportImportRow


def test_two_databases_quarantine_adoption_rejection_and_exact_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source?中文.sqlite3"
    target = tmp_path / "target#中文.sqlite3"
    seed = asyncio.run(seed_export_database(source))

    async def recipient() -> UUID:
        async with export_runtime(target, offset=2000).initialize(
            start_lifecycle_worker=False, recover_interrupted=False
        ) as container:
            return await create_export_owner(container, "recipient")

    owner = asyncio.run(recipient())
    ids = SequenceIdGenerator(tuple(UUID(int=i) for i in range(10000, 10300)))
    clock = FrozenClock(NOW + timedelta(hours=1))

    def runtime(settings: Settings) -> ApplicationRuntime:
        return ApplicationRuntime(settings, clock=clock, ids=ids)

    monkeypatch.setattr(runtime_module, "ApplicationRuntime", runtime)
    runner = CliRunner()

    def cli(*args: str) -> tuple[bytes, dict[str, object]]:
        result = runner.invoke(app, ["passport", *args])
        assert result.exit_code == 0, result.output
        document = json.loads(result.stdout)
        body = canonical_json_bytes(document)
        assert result.stdout == body.decode() + "\n"
        assert result.stderr == ""
        return body, document["data"]

    async def search(key: str) -> list[object]:
        settings = export_runtime(target).settings
        async with ApplicationRuntime(settings, clock=clock, ids=ids).initialize(
            start_lifecycle_worker=False, recover_interrupted=False
        ) as container:
            result = await container.retrieval_adapter.search(
                query=SearchExperiences(
                    owner_agent_id=owner,
                    query="rollback retry local transaction",
                    mode=RetrievalMode.FOCUSED,
                ),
                idempotency_key=key,
            )
        assert result.status_code == 200
        return json.loads(result.body)["data"]["hits"]

    output = tmp_path / "source.passport.json"
    before = source.read_bytes()
    cli(
        "export",
        str(seed.owner_agent_id),
        str(seed.experience_id),
        "--database",
        str(source),
        "--output",
        str(output),
        "--input-sanitized",
        "--sharing-authorized",
        "--sanitization-profile",
        "synthetic-v1",
    )
    assert source.read_bytes() == before
    _, inspected = cli("inspect", str(output))
    assert inspected["publisher_identity"] == "unverified"

    import_args = (
        "import",
        str(owner),
        str(output),
        "--database",
        str(target),
        "--idempotency-key",
        "import-1",
    )
    first, imported = cli(*import_args)
    assert cli(*import_args)[0] == first
    import_id = str(imported["import_id"])
    _, shown = cli("show", str(owner), import_id, "--database", str(target))
    assert shown["state"] == "pending"
    _, listed = cli("list", str(owner), "--database", str(target), "--state", "pending")
    assert len(listed["items"]) == 1
    assert asyncio.run(search("search-before")) == []

    adopt_args = (
        "adopt",
        str(owner),
        import_id,
        "--database",
        str(target),
        "--importance",
        "0.7",
        "--confidence",
        "0.6",
        "--idempotency-key",
        "adopt-1",
    )
    adopted_bytes, adopted = cli(*adopt_args)
    assert cli(*adopt_args)[0] == adopted_bytes
    hits = asyncio.run(search("search-after"))
    assert len(hits) == 1
    assert (
        hits[0]["experience"]["experience_id"] == adopted["experience"]["experience_id"]
    )
    _, dedup = cli(
        "import",
        str(owner),
        str(output),
        "--database",
        str(target),
        "--idempotency-key",
        "import-new-key",
    )
    assert dedup["import_id"] == import_id
    assert dedup["state"] == "adopted"

    rejected_file = tmp_path / "reject.passport.json"
    rejected_file.write_bytes(
        encode_passport_document(passport_document(tag="different"))
    )
    _, reject_import = cli(
        "import",
        str(owner),
        str(rejected_file),
        "--database",
        str(target),
        "--idempotency-key",
        "reject-import",
    )
    rejected_id = str(reject_import["import_id"])
    reject_args = (
        "reject",
        str(owner),
        rejected_id,
        "--database",
        str(target),
        "--reason",
        "Not needed locally",
        "--idempotency-key",
        "reject-1",
    )
    rejected, decision = cli(*reject_args)
    assert decision["state"] == "rejected"
    assert cli(*reject_args)[0] == rejected
    assert len(asyncio.run(search("search-after-rejection"))) == 1
    _, preserved = cli(
        "import",
        str(owner),
        str(rejected_file),
        "--database",
        str(target),
        "--idempotency-key",
        "rejected-new-key",
    )
    assert preserved["import_id"] == rejected_id
    assert preserved["state"] == "rejected"

    async def count_lineage() -> None:
        settings = export_runtime(target).settings
        async with (
            ApplicationRuntime(settings, clock=clock, ids=ids).initialize(
                start_lifecycle_worker=False, recover_interrupted=False
            ) as container,
            container.database.read_session() as session,
        ):
            assert (
                await session.scalar(
                    select(func.count()).select_from(PassportImportRow)
                )
                == 2
            )
            assert (
                await session.scalar(
                    select(func.count()).select_from(PassportAdoptionRow)
                )
                == 1
            )
            await container.source_validator.validate(session)

    asyncio.run(count_lineage())
