from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from tests.cli.test_capture_commands import FIXTURE, NOW, OWNER_ID, _seed_owner

from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval import RetrievalMode, SearchExperiences
from experience_hub.runtime import ApplicationRuntime


def _run_cli(*arguments: str) -> bytes:
    executable = Path(sys.executable).with_name("experience-hub")
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("EXPERIENCE_HUB_OPENAI_COMPATIBLE_"):
            environment.pop(name)
    completed = subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    return completed.stdout


def _document(body: bytes) -> dict[str, Any]:
    value = json.loads(body)
    assert isinstance(value, dict)
    return value


async def _fresh_runtime_search(
    database: Path,
    *,
    key: str,
    receipt_id: UUID,
    at: datetime = NOW,
) -> list[dict[str, Any]]:
    runtime = ApplicationRuntime(
        Settings(database_url=f"sqlite+aiosqlite:///{database}"),
        clock=FrozenClock(at),
        ids=SequenceIdGenerator((receipt_id,)),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        result = await container.retrieval_adapter.search(
            query=SearchExperiences(
                owner_agent_id=OWNER_ID,
                query="confirmed stale cache",
                mode=RetrievalMode.FOCUSED,
            ),
            idempotency_key=key,
        )
    assert result.status_code == 200
    hits = json.loads(result.body)["data"]["hits"]
    assert isinstance(hits, list)
    return hits


def test_five_minute_offline_capture_quarantine_flow(
    tmp_path: Path,
) -> None:
    database = tmp_path / "capture-quarantine.sqlite3"
    asyncio.run(_seed_owner(database))

    inspected = _document(_run_cli("capture", "inspect", str(FIXTURE)))
    assert inspected["data"]["persisted"] is False
    assert inspected["data"]["candidate_count"] == 1

    import_arguments = (
        "capture",
        "import",
        str(FIXTURE),
        "--database",
        str(database),
        "--idempotency-key",
        "e2e-capture-import",
    )
    first_import = _run_cli(*import_arguments)
    replayed_import = _run_cli(*import_arguments)
    assert replayed_import == first_import
    imported = _document(first_import)
    assert imported["data"]["manifest_hash"] == inspected["data"]["manifest_hash"]
    candidate_id = UUID(imported["data"]["candidate_ids"][0])

    listed = _document(
        _run_cli(
            "candidates",
            "list",
            str(OWNER_ID),
            "--decision",
            "pending",
            "--database",
            str(database),
        )
    )
    shown = _document(
        _run_cli(
            "candidates",
            "show",
            str(OWNER_ID),
            str(candidate_id),
            "--database",
            str(database),
        )
    )
    assert [item["candidate_id"] for item in listed["data"]["items"]] == [
        str(candidate_id)
    ]
    assert shown["data"] == listed["data"]["items"][0]
    assert shown["data"]["decision"] == "pending"

    before = asyncio.run(
        _fresh_runtime_search(
            database,
            key="e2e-search-before-adoption",
            receipt_id=UUID("10000000-0000-4000-8000-000000000710"),
        )
    )
    assert before == []

    adopt_arguments = (
        "candidates",
        "adopt",
        str(OWNER_ID),
        str(candidate_id),
        "--importance",
        "0.7",
        "--confidence",
        "0.8",
        "--idempotency-key",
        "e2e-adopt-candidate",
        "--database",
        str(database),
    )
    first_decision = _run_cli(*adopt_arguments)
    replayed_decision = _run_cli(*adopt_arguments)
    assert replayed_decision == first_decision
    adopted = _document(first_decision)["data"]
    assert adopted["decision"] == "adopted"
    resulting_experience_id = UUID(adopted["resulting_experience_id"])
    resulting_version_id = UUID(adopted["resulting_version_id"])

    after = asyncio.run(
        _fresh_runtime_search(
            database,
            key="e2e-search-after-adoption",
            receipt_id=UUID("10000000-0000-4000-8000-000000000711"),
            at=datetime.fromisoformat(adopted["decided_at"]),
        )
    )
    assert [hit["experience"]["experience_id"] for hit in after] == [
        str(resulting_experience_id)
    ]
    assert after[0]["experience"]["version_id"] == str(resulting_version_id)
    assert after[0]["experience"]["body"] == shown["data"]["content"]["body"]

    verified = _document(
        _run_cli(
            "projections",
            "rebuild",
            "--verify",
            "--database",
            str(database),
        )
    )
    assert verified["data"]["matches"] is True
    assert verified["data"]["differences"] == []

    with sqlite3.connect(database) as connection:
        lineage = connection.execute(
            "SELECT candidate_id, resulting_experience_id, "
            "resulting_version_id, resulting_content_hash "
            "FROM candidate_adoptions WHERE candidate_id = ?",
            (str(candidate_id),),
        ).fetchall()
    assert len(lineage) == 1
    assert lineage[0][:3] == (
        str(candidate_id),
        str(resulting_experience_id),
        str(resulting_version_id),
    )
    assert lineage[0][3] == shown["data"]["content_hash"]
