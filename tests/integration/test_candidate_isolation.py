from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_candidate_decisions import (
    OTHER_OWNER_ID,
    OWNER_ID,
    DecisionStack,
    _adopt,
    _capture_pending,
    _reject,
)
from tests.integration.test_candidate_decisions import (
    stack as candidate_stack,  # noqa: F401 - pytest discovers imported fixture
)

from experience_hub import canonical_json_bytes
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.queries import ExperienceNotFoundError
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.retrieval import RetrievalMode, SearchExperiences
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.tables import (
    CandidateAdoptionRow,
    ExperienceRow,
    ExperienceTermRow,
)

DISCARDED_SENTINEL = b"discarded-step-sentinel-7f9b2d6e"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _assert_sqlite_bytes_absent(database: Path, forbidden: bytes) -> None:
    with sqlite3.connect(database) as connection:
        connection.text_factory = bytes
        raw_tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for (raw_table,) in raw_tables:
            table = bytes(raw_table).decode("utf-8")
            columns = connection.execute(
                f"PRAGMA table_info({_quote_identifier(table)})"
            ).fetchall()
            for column in columns:
                name = bytes(column[1]).decode("utf-8")
                declared = bytes(column[2]).decode("utf-8").upper()
                if not any(
                    marker in declared
                    for marker in ("TEXT", "CHAR", "CLOB", "BLOB")
                ):
                    continue
                rows = connection.execute(
                    f"SELECT {_quote_identifier(name)} "
                    f"FROM {_quote_identifier(table)}"
                ).fetchall()
                for (value,) in rows:
                    if value is None:
                        continue
                    retained = (
                        value.encode("utf-8")
                        if isinstance(value, str)
                        else bytes(value)
                    )
                    if forbidden in retained:
                        raise AssertionError(
                            "forbidden bytes retained at "
                            f"{table}.{name}"
                        )


def _fixture_with_discarded_sentinel(repository_root: Path) -> bytes:
    source = (
        repository_root
        / "examples"
        / "trajectories"
        / "coding-agent-recovery.jsonl"
    )
    documents = [json.loads(line) for line in source.read_bytes().splitlines()]
    documents[0]["owner_agent_id"] = str(OWNER_ID)
    documents[0]["trajectory_id"] = "candidate-isolation-discarded-fields"
    sentinel = DISCARDED_SENTINEL.decode("ascii")
    documents[1]["observation"] = sentinel
    documents[1]["action"] = sentinel
    documents[1]["outcome"] = sentinel
    signal = documents[3]["candidate_signal"]
    assert isinstance(signal, dict)
    signal["body"] = "Apply zzqqxx77 before retrying."
    signal["summary"] = "Apply zzqqxx77."
    signal["mechanism"] = "zzqqxx77 isolates this candidate."
    signal["tags"] = ["zzqqxx77"]
    return b"\n".join(canonical_json_bytes(document) for document in documents)


async def _capture_discarded_sentinel(
    stack: DecisionStack,
    *,
    repository_root: Path,
) -> UUID:
    prepared = stack.container.capture_preparer.prepare_jsonl(
        _fixture_with_discarded_sentinel(repository_root)
    )
    request = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope=TRAJECTORY_IMPORT_SCOPE,
        idempotency_key="capture-discarded-sentinel",
        method="POST",
        route_template="/v1/agents/{agent_id}/trajectory-bundles",
        path_parameters={"agent_id": OWNER_ID},
        body={
            "candidate_count": len(prepared.candidates),
            "manifest_hash": prepared.bundle.manifest_hash,
        },
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await stack.container.capture_service.capture(
            uow=uow,
            prepared=prepared,
            command=command,
        )

    result = await stack.container.command_executor.execute(request, handler)
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["candidate_ids"][0])


async def _search(
    stack: DecisionStack,
    *,
    query: str,
    key: str,
) -> list[dict[str, object]]:
    result = await stack.container.retrieval_adapter.search(
        query=SearchExperiences(
            owner_agent_id=OWNER_ID,
            query=query,
            mode=RetrievalMode.FOCUSED,
        ),
        idempotency_key=key,
    )
    assert result.status_code == 200
    document = json.loads(result.body)
    hits = document["data"]["hits"]
    assert isinstance(hits, list)
    return hits


async def _snapshot_source_ids(
    stack: DecisionStack,
    *,
    query: str,
    run_id: UUID,
) -> tuple[UUID, ...]:
    async with stack.container.database.transaction(immediate=True) as uow:
        snapshot = await stack.container.snapshot_builder.freeze(
            uow=uow,
            request=StartInspirationRun(
                owner_agent_id=OWNER_ID,
                goal=query,
                mode=RetrievalMode.FOCUSED,
            ),
            run_id=run_id,
            at=stack.clock.now(),
        )
    return tuple(item.source_id for item in snapshot.items)


@pytest.mark.asyncio
async def test_candidate_quarantine_isolates_every_normal_evidence_path(
    candidate_stack: DecisionStack,  # noqa: F811 - injected imported fixture
    repository_root: Path,
    tmp_path: Path,
) -> None:
    pending_id = await _capture_discarded_sentinel(
        candidate_stack,
        repository_root=repository_root,
    )
    rejected_id = await _capture_pending(
        candidate_stack,
        key="capture-rejected-isolation",
        label="zzrrww66",
    )
    adopted_id = await _capture_pending(
        candidate_stack,
        key="capture-adopted-isolation",
        label="adoptmz9",
    )
    foreign_id = await _capture_pending(
        candidate_stack,
        owner_agent_id=OTHER_OWNER_ID,
        key="capture-foreign-isolation",
        label="foreign-boundary",
    )
    queries = {
        pending_id: "zzqqxx77",
        rejected_id: "zzrrww66",
        adopted_id: "adoptmz9",
    }

    for index, (candidate_id, query) in enumerate(queries.items(), start=1):
        assert await _search(
            candidate_stack,
            query=query,
            key=f"search-pending-{index}",
        ) == []
        assert await _snapshot_source_ids(
            candidate_stack,
            query=query,
            run_id=UUID(int=8_100 + index),
        ) == ()
        async with candidate_stack.container.database.read_session() as session:
            with pytest.raises(ExperienceNotFoundError):
                query_service = candidate_stack.container.experience_query
                await query_service.get_owned_shareable_version(
                    session=session,
                    owner_agent_id=OWNER_ID,
                    experience_id=candidate_id,
                    version_id=None,
                )

    async with candidate_stack.container.database.read_session() as session:
        term_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceTermRow)
                .where(
                    ExperienceTermRow.experience_id.in_(tuple(queries))
                )
            )
            or 0
        )
        owner_page = await candidate_stack.container.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
        )
        foreign_page = await candidate_stack.container.candidate_service.list_owned(
            session=session,
            owner_agent_id=OTHER_OWNER_ID,
        )
    assert term_count == 0
    assert {item.candidate_id for item in owner_page.items} == set(queries)
    assert {item.candidate_id for item in foreign_page.items} == {foreign_id}

    first_rejection = await _reject(
        candidate_stack,
        candidate_id=rejected_id,
        key="reject-isolated-candidate",
    )
    replayed_rejection = await _reject(
        candidate_stack,
        candidate_id=rejected_id,
        key="reject-isolated-candidate",
    )
    assert replayed_rejection.replayed is True
    assert replayed_rejection.body == first_rejection.body
    assert await _search(
        candidate_stack,
        query=queries[rejected_id],
        key="search-rejected",
    ) == []

    first_adoption = await _adopt(
        candidate_stack,
        candidate_id=adopted_id,
        key="adopt-isolated-candidate",
    )
    replayed_adoption = await _adopt(
        candidate_stack,
        candidate_id=adopted_id,
        key="adopt-isolated-candidate",
    )
    assert replayed_adoption.replayed is True
    assert replayed_adoption.body == first_adoption.body
    async with candidate_stack.container.database.read_session() as session:
        adopted = await candidate_stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=adopted_id,
        )
        assert adopted.resulting_experience_id is not None
        assert adopted.resulting_version_id is not None
        adoption_count = int(
            await session.scalar(
                select(func.count())
                .select_from(CandidateAdoptionRow)
                .where(CandidateAdoptionRow.candidate_id == adopted_id)
            )
            or 0
        )
        adoption_row = await session.scalar(
            select(CandidateAdoptionRow).where(
                CandidateAdoptionRow.candidate_id == adopted_id
            )
        )
        assert adoption_row is not None
        lineage_before_repair = (
            adoption_row.candidate_id,
            adoption_row.resulting_experience_id,
            adoption_row.resulting_version_id,
            adoption_row.resulting_content_hash,
        )
        query_service = candidate_stack.container.experience_query
        selected = await query_service.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=adopted.resulting_experience_id,
            version_id=adopted.resulting_version_id,
        )
        result_term_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceTermRow)
                .where(
                    ExperienceTermRow.experience_id
                    == adopted.resulting_experience_id
                )
            )
            or 0
        )
        owner_experience_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.owner_agent_id == OWNER_ID)
            )
            or 0
        )
        foreign_experience_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.owner_agent_id == OTHER_OWNER_ID)
            )
            or 0
        )

    hits = await _search(
        candidate_stack,
        query=queries[adopted_id],
        key="search-adopted",
    )
    assert [hit["experience"]["experience_id"] for hit in hits] == [
        str(adopted.resulting_experience_id)
    ]
    assert selected.experience_id == adopted.resulting_experience_id
    assert selected.version_id == adopted.resulting_version_id
    assert adoption_count == 1
    assert result_term_count > 0
    assert (owner_experience_count, foreign_experience_count) == (1, 0)
    assert await _snapshot_source_ids(
        candidate_stack,
        query=queries[adopted_id],
        run_id=UUID(int=8_200),
    ) == (adopted.resulting_experience_id,)

    report = await candidate_stack.container.projection_manager.repair(
        candidate_stack.container.database
    )
    assert report.matches is True
    assert await _search(
        candidate_stack,
        query=queries[pending_id],
        key="search-pending-after-repair",
    ) == []
    assert await _search(
        candidate_stack,
        query=queries[rejected_id],
        key="search-rejected-after-repair",
    ) == []
    repaired_hits = await _search(
        candidate_stack,
        query=queries[adopted_id],
        key="search-adopted-after-repair",
    )
    assert [hit["experience"]["experience_id"] for hit in repaired_hits] == [
        str(adopted.resulting_experience_id)
    ]
    assert await _snapshot_source_ids(
        candidate_stack,
        query=queries[pending_id],
        run_id=UUID(int=8_201),
    ) == ()
    assert await _snapshot_source_ids(
        candidate_stack,
        query=queries[rejected_id],
        run_id=UUID(int=8_202),
    ) == ()
    assert await _snapshot_source_ids(
        candidate_stack,
        query=queries[adopted_id],
        run_id=UUID(int=8_203),
    ) == (adopted.resulting_experience_id,)

    async with candidate_stack.container.database.read_session() as session:
        query_service = candidate_stack.container.experience_query
        for candidate_id in (pending_id, rejected_id):
            with pytest.raises(ExperienceNotFoundError):
                await query_service.get_owned_shareable_version(
                    session=session,
                    owner_agent_id=OWNER_ID,
                    experience_id=candidate_id,
                    version_id=None,
                )
        repaired_term_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceTermRow)
                .where(
                    ExperienceTermRow.experience_id.in_(
                        (pending_id, rejected_id)
                    )
                )
            )
            or 0
        )
        repaired_selected = await query_service.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=adopted.resulting_experience_id,
            version_id=adopted.resulting_version_id,
        )
        repaired_result_term_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceTermRow)
                .where(
                    ExperienceTermRow.experience_id
                    == adopted.resulting_experience_id
                )
            )
            or 0
        )
        repaired_owner_experience_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.owner_agent_id == OWNER_ID)
            )
            or 0
        )
        repaired_foreign_experience_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperienceRow)
                .where(ExperienceRow.owner_agent_id == OTHER_OWNER_ID)
            )
            or 0
        )
        repaired_adoption_rows = tuple(
            await session.scalars(
                select(CandidateAdoptionRow).where(
                    CandidateAdoptionRow.candidate_id == adopted_id
                )
            )
        )
        repaired_owner_page = (
            await candidate_stack.container.candidate_service.list_owned(
                session=session,
                owner_agent_id=OWNER_ID,
            )
        )
        repaired_foreign_page = (
            await candidate_stack.container.candidate_service.list_owned(
                session=session,
                owner_agent_id=OTHER_OWNER_ID,
            )
        )
    repaired_decisions = {
        item.candidate_id: item.decision.value
        for item in repaired_owner_page.items
    }
    assert repaired_term_count == 0
    assert repaired_selected.experience_id == adopted.resulting_experience_id
    assert repaired_selected.version_id == adopted.resulting_version_id
    assert repaired_result_term_count > 0
    assert (
        repaired_owner_experience_count,
        repaired_foreign_experience_count,
    ) == (1, 0)
    assert len(repaired_adoption_rows) == 1
    repaired_adoption = repaired_adoption_rows[0]
    assert (
        repaired_adoption.candidate_id,
        repaired_adoption.resulting_experience_id,
        repaired_adoption.resulting_version_id,
        repaired_adoption.resulting_content_hash,
    ) == lineage_before_repair == (
        adopted_id,
        adopted.resulting_experience_id,
        adopted.resulting_version_id,
        adopted.content_hash,
    )
    assert repaired_decisions == {
        pending_id: "pending",
        rejected_id: "rejected",
        adopted_id: "adopted",
    }
    assert {item.candidate_id for item in repaired_foreign_page.items} == {
        foreign_id
    }

    _assert_sqlite_bytes_absent(
        tmp_path / "candidate-decisions.sqlite3",
        DISCARDED_SENTINEL,
    )


def test_sqlite_scanner_handles_non_utf8_and_redacts_matches(
    tmp_path: Path,
) -> None:
    database = tmp_path / "scanner.sqlite3"
    private = b"private-scan-value-0d8c"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE probe (value BLOB NOT NULL)")
        connection.execute("INSERT INTO probe VALUES (?)", (b"\xff" + private,))

    absent = b"absent-scan-value-2f91"
    _assert_sqlite_bytes_absent(database, absent)

    with pytest.raises(AssertionError) as captured:
        _assert_sqlite_bytes_absent(database, private)

    assert private not in str(captured.value).encode("utf-8")
    _assert_sqlite_bytes_absent(database, absent)
