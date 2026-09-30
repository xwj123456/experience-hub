from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import event, select
from tests.integration.test_candidate_decisions import (
    IDS,
    NOW,
    OTHER_OWNER_ID,
    OWNER_ID,
    DecisionStack,
    FailAt,
    _adopt,
    _capture_pending,
    _create_agent,
    _create_equivalent,
    _jsonl,
)
from tests.passport_export_fixtures import (
    create_export_experience,
    create_export_owner,
    export_runtime,
)

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.models import TrajectoryField
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, TypedEvidence
from experience_hub.experiences.evidence_snapshots import (
    ExperienceEvidenceSnapshotReader,
)
from experience_hub.experiences.models import VersionContent
from experience_hub.ids import SequenceIdGenerator
from experience_hub.passports.contracts import (
    EmbeddedExcerptSnapshotV1,
    EvidenceSnapshotV1,
    ReferenceOnlySnapshotV1,
)
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.tables import (
    AgentRow,
    CandidateAdoptionRow,
    TrajectoryEvidenceRow,
)
from experience_hub.storage.validation import SourceIntegrityError


def _reader() -> ExperienceEvidenceSnapshotReader:
    return ExperienceEvidenceSnapshotReader()


@pytest.fixture
async def capture_stack(tmp_path: Path) -> AsyncIterator[DecisionStack]:
    runtime = ApplicationRuntime(
        Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'evidence.sqlite3'}"),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(IDS),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        await _create_agent(container, key="owner", name="Owner")
        await _create_agent(container, key="other", name="Other")
        yield DecisionStack(container, FailAt())


async def _adopt_source(stack: DecisionStack) -> tuple[UUID, UUID, UUID]:
    candidate_id = await _capture_pending(stack)
    result = await _adopt(stack, candidate_id=candidate_id, key="adopt")
    assert result.status_code == 200
    data = json.loads(result.body)["data"]
    return (
        candidate_id,
        UUID(data["resulting_experience_id"]),
        UUID(data["resulting_version_id"]),
    )


async def _snapshots(
    stack: DecisionStack,
    experience_id: UUID,
    version_id: UUID,
) -> tuple[EvidenceSnapshotV1, ...]:
    async with stack.container.database.read_session() as session:
        version = await stack.container.experience_query.get_owned_shareable_version(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=experience_id,
            version_id=version_id,
        )
        return await _reader().read(session=session, version=version)


async def _tamper(tmp_path: Path, statements: tuple[str, ...]) -> None:
    def execute() -> None:
        with sqlite3.connect(tmp_path / "evidence.sqlite3") as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            for statement in statements:
                connection.execute(statement)

    await asyncio.to_thread(execute)


@pytest.mark.parametrize(
    "evidence",
    [
        (),
        (
            TypedEvidence(type="trajectory_field", id="unavailable:step:observation"),
            TypedEvidence(type="document", id="synthetic-reference"),
        ),
    ],
)
async def test_reference_closure_uses_selected_canonical_evidence(
    tmp_path: Path, evidence: tuple[TypedEvidence, ...]
) -> None:
    async with export_runtime(tmp_path / "references.sqlite3").initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        owner = await create_export_owner(container, "publisher")
        content = VersionContent(
            body="Synthetic procedure",
            summary="Synthetic summary",
            mechanism="Synthetic mechanism",
            tags=(),
            applicability=(),
            evidence=evidence,
            falsifiers=(),
        )
        seed = await create_export_experience(container, owner, content=content)
        async with container.database.read_session() as session:
            version = await container.experience_query.get_owned_shareable_version(
                session=session,
                owner_agent_id=owner,
                experience_id=seed.experience_id,
                version_id=seed.version_id,
            )
            snapshots = await _reader().read(session=session, version=version)
        assert snapshots == tuple(
            ReferenceOnlySnapshotV1(mode="reference_only", reference=reference)
            for reference in content.evidence
        )


async def test_owned_adoption_embeds_authenticated_capture_excerpt(
    capture_stack: DecisionStack,
) -> None:
    candidate_id, experience_id, version_id = await _adopt_source(capture_stack)
    async with capture_stack.container.database.read_session() as session:
        candidate = await capture_stack.container.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
    snapshots = await _snapshots(capture_stack, experience_id, version_id)
    captured = candidate.evidence[0]
    manifest_hash, step_id, field = candidate.content.evidence[0].id.split(":")
    assert snapshots == (
        EmbeddedExcerptSnapshotV1(
            mode="embedded_excerpt",
            reference=candidate.content.evidence[0],
            excerpt=captured.excerpt,
            excerpt_hash=captured.excerpt_hash,
            source_hash=captured.source_hash,
            source_manifest_hash=manifest_hash,
            step_id=step_id,
            field=TrajectoryField(field),
        ),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "excerpt",
        "manifest",
        "source_hash",
        "candidate_missing",
        "bundle_missing",
        "evidence_missing",
        "candidate_foreign",
        "bundle_foreign",
        "evidence_foreign",
        "candidate_content",
        "refs_missing",
        "state_missing",
        "state_anchor",
    ],
)
async def test_located_owned_lineage_damage_fails_closed_without_source_details(
    capture_stack: DecisionStack,
    tmp_path: Path,
    damage: str,
) -> None:
    candidate_id, experience_id, version_id = await _adopt_source(capture_stack)
    mutations = {
        "excerpt": ("trajectory_evidence", "UPDATE", "excerpt='changed'"),
        "manifest": (
            "trajectory_bundles",
            "UPDATE",
            "manifest_hash='" + "0" * 64 + "'",
        ),
        "source_hash": (
            "trajectory_evidence",
            "UPDATE",
            "source_hash='" + "0" * 64 + "'",
        ),
        "candidate_missing": ("experience_candidates", "DELETE", ""),
        "bundle_missing": ("trajectory_bundles", "DELETE", ""),
        "evidence_missing": ("trajectory_evidence", "DELETE", ""),
        "candidate_foreign": (
            "experience_candidates",
            "UPDATE",
            f"owner_agent_id='{OTHER_OWNER_ID}'",
        ),
        "bundle_foreign": (
            "trajectory_bundles",
            "UPDATE",
            f"owner_agent_id='{OTHER_OWNER_ID}'",
        ),
        "evidence_foreign": (
            "trajectory_evidence",
            "UPDATE",
            f"owner_agent_id='{OTHER_OWNER_ID}'",
        ),
        "candidate_content": ("experience_candidates", "UPDATE", "body='changed'"),
        "refs_missing": (
            "experience_candidates",
            "UPDATE",
            "evidence_refs=CAST('"
            + canonical_json_bytes((str(UUID(int=99999)),)).decode()
            + "' AS BLOB)",
        ),
        "state_missing": ("candidate_state", "DELETE", ""),
        "state_anchor": (
            "candidate_state",
            "UPDATE",
            f"adoption_id='{UUID(int=99999)}'",
        ),
    }
    table, operation, assignment = mutations[damage]
    statement = (
        f"{operation} {table}"
        if operation == "DELETE"
        else f"UPDATE {table} SET {assignment}"
    )
    if operation == "DELETE":
        statement = f"DELETE FROM {table}"
    await _tamper(
        tmp_path,
        (
            f"DROP TRIGGER IF EXISTS {table}_reject_{operation.lower()}",
            statement,
        ),
    )
    with pytest.raises(SourceIntegrityError) as caught:
        await _snapshots(capture_stack, experience_id, version_id)
    assert caught.value.mismatch_key == "passport_evidence"
    assert str(caught.value) == "passport_evidence: Owned capture evidence is invalid"
    assert str(candidate_id) not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("candidate_owner", [OWNER_ID, OTHER_OWNER_ID])
async def test_unadopted_reference_never_scans_candidates(
    capture_stack: DecisionStack,
    candidate_owner: UUID,
) -> None:
    foreign_id = await _capture_pending(
        capture_stack,
        owner_agent_id=candidate_owner,
    )
    experience_id, version_id = await _create_equivalent(
        capture_stack,
        candidate_id=foreign_id,
        candidate_owner_id=candidate_owner,
        experience_owner_id=OWNER_ID,
        key="copied-reference",
    )
    statements: list[str] = []

    def observe(_connection, _cursor, statement, parameters, _context, _many):
        statements.append(statement.lower() + repr(parameters))

    async with capture_stack.container.database.read_session() as session:
        version = (
            await capture_stack.container.experience_query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=version_id,
            )
        )
        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", observe)
        try:
            snapshots = await _reader().read(session=session, version=version)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
    assert snapshots == tuple(
        ReferenceOnlySnapshotV1(mode="reference_only", reference=reference)
        for reference in version.content.evidence
    )
    assert all("experience_candidates" not in sql for sql in statements)
    assert all("trajectory_evidence" not in sql for sql in statements)
    assert all(str(OTHER_OWNER_ID) not in sql for sql in statements)


async def test_owned_reader_queries_are_owner_scoped_and_do_not_autoflush(
    capture_stack: DecisionStack,
) -> None:
    _, experience_id, version_id = await _adopt_source(capture_stack)
    await _capture_pending(
        capture_stack,
        owner_agent_id=OTHER_OWNER_ID,
        key="foreign-capture",
    )
    statements: list[str] = []

    def observe(_connection, _cursor, statement, parameters, _context, _many):
        statements.append(statement.lower() + repr(parameters))

    async with capture_stack.container.database.read_session() as session:
        version = (
            await capture_stack.container.experience_query.get_owned_shareable_version(
                session=session,
                owner_agent_id=OWNER_ID,
                experience_id=experience_id,
                version_id=version_id,
            )
        )
        owner = await session.scalar(
            select(AgentRow).where(AgentRow.agent_id == OWNER_ID)
        )
        assert owner is not None
        owner.name = "Caller pending mutation must not flush"
        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", observe)
        try:
            snapshots = await _reader().read(session=session, version=version)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert owner in session.dirty
    assert snapshots[0].mode == "embedded_excerpt"
    assert any("candidate_adoptions" in sql for sql in statements)
    assert any("experience_candidates" in sql for sql in statements)
    assert any("trajectory_evidence" in sql for sql in statements)
    for sql in statements:
        assert sql.lstrip().startswith("select")
        assert "owner_agent_id" in sql.partition("where")[2]
        assert str(OWNER_ID) in sql
        assert str(OTHER_OWNER_ID) not in sql
    async with capture_stack.container.database.read_session() as session:
        assert (
            await session.scalar(
                select(AgentRow.name).where(AgentRow.agent_id == OWNER_ID)
            )
            == "Owner"
        )


async def test_multiple_fields_unicode_excerpt_and_colon_step_keep_exact_closure(
    capture_stack: DecisionStack,
) -> None:
    header, step = (json.loads(line) for line in _jsonl().splitlines())
    step["step_id"] = "step:证据"
    step["observation"] = "证" * 400
    step["candidate_signal"]["evidence"] = [
        {"step_id": step["step_id"], "field": field.value} for field in TrajectoryField
    ]
    prepared = capture_stack.container.capture_preparer.prepare_jsonl(
        b"\n".join((canonical_json_bytes(header), canonical_json_bytes(step))),
    )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await capture_stack.container.capture_service.capture(
            uow=uow,
            prepared=prepared,
            command=command,
        )

    result = await capture_stack.container.command_executor.execute(
        CommandRequest(
            caller_scope=f"agent:{OWNER_ID}",
            operation_scope=TRAJECTORY_IMPORT_SCOPE,
            idempotency_key="multi-fields",
            method="POST",
            route_template="/capture",
            body={"manifest_hash": prepared.bundle.manifest_hash},
        ),
        handler,
    )
    assert result.status_code == 201
    candidate_id = UUID(json.loads(result.body)["data"]["candidate_ids"][0])
    result = await _adopt(capture_stack, candidate_id=candidate_id, key="adopt")
    assert result.status_code == 200
    data = json.loads(result.body)["data"]
    snapshots = await _snapshots(
        capture_stack,
        UUID(data["resulting_experience_id"]),
        UUID(data["resulting_version_id"]),
    )
    assert (
        tuple(item.reference for item in snapshots)
        == prepared.candidates[0].content.evidence
    )
    assert len(snapshots) == 3
    for snapshot in snapshots:
        assert isinstance(snapshot, EmbeddedExcerptSnapshotV1)
        assert (
            snapshot.reference.id
            == f"{prepared.bundle.manifest_hash}:step:证据:{snapshot.field.value}"
        )
        assert snapshot.source_manifest_hash == prepared.bundle.manifest_hash
        assert snapshot.step_id == "step:证据"
        assert snapshot.excerpt_hash == sha256_hex(snapshot.excerpt.encode())
        if snapshot.field is TrajectoryField.OBSERVATION:
            assert len(snapshot.excerpt.encode()) == 510
            assert snapshot.source_hash == sha256_hex(("证" * 400).encode())
            assert snapshot.source_hash != snapshot.excerpt_hash


@pytest.mark.parametrize("foreign", [False, True])
async def test_agreeing_lineage_and_state_cannot_name_unowned_result_version(
    capture_stack: DecisionStack,
    tmp_path: Path,
    foreign: bool,
) -> None:
    candidate_id, experience_id, version_id = await _adopt_source(capture_stack)
    target_version_id = UUID(int=99997)
    if foreign:
        _, target_version_id = await _create_equivalent(
            capture_stack,
            candidate_id=candidate_id,
            candidate_owner_id=OWNER_ID,
            experience_owner_id=OTHER_OWNER_ID,
            key="foreign-equivalent",
        )
    await _tamper(
        tmp_path,
        (
            "DROP TRIGGER candidate_adoptions_reject_update",
            "UPDATE candidate_adoptions "
            f"SET resulting_version_id='{target_version_id}'",
            f"UPDATE candidate_state SET resulting_version_id='{target_version_id}'",
        ),
    )
    with pytest.raises(SourceIntegrityError) as caught:
        await _snapshots(capture_stack, experience_id, version_id)
    assert caught.value.mismatch_key == "passport_evidence"


async def test_selected_historical_content_does_not_use_current_lineage(
    capture_stack: DecisionStack,
) -> None:
    _, experience_id, version_id = await _adopt_source(capture_stack)
    before = await _snapshots(capture_stack, experience_id, version_id)
    next_version = await create_export_experience(
        capture_stack.container,
        OWNER_ID,
        experience_id=experience_id,
        key="new-version",
        content=VersionContent(
            body="Changed procedure",
            summary="Changed summary",
            mechanism="Changed mechanism",
            tags=(),
            applicability=(),
            evidence=(TypedEvidence(type="document", id="new-reference"),),
            falsifiers=(),
        ),
    )
    assert await _snapshots(capture_stack, experience_id, version_id) == before
    assert await _snapshots(capture_stack, experience_id, next_version.version_id) == (
        ReferenceOnlySnapshotV1(
            mode="reference_only",
            reference=TypedEvidence(type="document", id="new-reference"),
        ),
    )


async def _duplicate_adoptions(
    stack: DecisionStack,
    tmp_path: Path,
) -> tuple[UUID, UUID, UUID]:
    first_id, experience_id, version_id = await _adopt_source(stack)
    second_id = UUID(int=99000)
    second_adoption_id = UUID(int=99001)

    def duplicate() -> None:
        # Current extraction deduplicates equal drafts. Preserve a real captured
        # and adopted baseline, then synthesize the equivalent retained source
        # shape to exercise the reader's required multiple-lineage defense.
        with sqlite3.connect(tmp_path / "evidence.sqlite3") as connection:
            connection.row_factory = sqlite3.Row
            for table in (
                "experience_candidates",
                "candidate_state",
                "candidate_adoptions",
            ):
                row = dict(
                    connection.execute(
                        f"SELECT * FROM {table} WHERE candidate_id=?",
                        (str(first_id),),
                    ).fetchone()
                )
                row["candidate_id"] = str(second_id)
                if table == "experience_candidates":
                    row["candidate_ordinal"] = 2
                else:
                    row["adoption_id"] = str(second_adoption_id)
                if table == "candidate_adoptions":
                    row["created"] = 0
                connection.execute(
                    f"INSERT INTO {table} ({','.join(row)}) VALUES "
                    f"({','.join('?' for _ in row)})",
                    tuple(row.values()),
                )

    await asyncio.to_thread(duplicate)
    return second_id, experience_id, version_id


async def test_duplicate_owned_lineages_deduplicate_identical_snapshots(
    capture_stack: DecisionStack,
    tmp_path: Path,
) -> None:
    _, experience_id, version_id = await _duplicate_adoptions(capture_stack, tmp_path)
    async with capture_stack.container.database.read_session() as session:
        lineages = tuple((await session.scalars(select(CandidateAdoptionRow))).all())
    assert len(lineages) == 2
    assert len({item.resulting_content_hash for item in lineages}) == 1
    snapshots = await _snapshots(capture_stack, experience_id, version_id)
    assert len(snapshots) == 1
    assert snapshots[0].mode == "embedded_excerpt"


@pytest.mark.parametrize("conflict", [False, True])
async def test_every_matching_lineage_is_validated_and_conflicts_are_refused(
    capture_stack: DecisionStack,
    tmp_path: Path,
    conflict: bool,
) -> None:
    candidate_id, experience_id, version_id = await _duplicate_adoptions(
        capture_stack, tmp_path
    )
    async with capture_stack.container.database.read_session() as session:
        evidence = (await session.scalars(select(TrajectoryEvidenceRow))).one()
        bundle_id = evidence.bundle_id
        source_hash = evidence.source_hash
    if conflict:
        excerpt = "A different internally hashed retained excerpt."
        evidence_id = UUID(int=99998)
        await _tamper(
            tmp_path,
            (
                "DROP INDEX ux_trajectory_evidence_bundle_step_field",
                "DROP TRIGGER trajectory_evidence_reject_conflicting_insert",
                "DROP TRIGGER experience_candidates_reject_update",
                "INSERT INTO trajectory_evidence (evidence_id,bundle_id,owner_agent_id,"
                "step_id,field,ordinal,excerpt,excerpt_hash,source_hash) VALUES ("
                f"'{evidence_id}','{bundle_id}','{OWNER_ID}','step-1','observation',1,"
                f"'{excerpt}','{sha256_hex(excerpt.encode())}','{source_hash}')",
                "UPDATE experience_candidates SET evidence_refs=CAST('"
                + canonical_json_bytes((str(evidence_id),)).decode()
                + f"' AS BLOB) WHERE candidate_id='{candidate_id}'",
            ),
        )
    else:
        await _tamper(
            tmp_path,
            (
                "DROP TRIGGER experience_candidates_reject_delete",
                "DELETE FROM experience_candidates "
                f"WHERE candidate_id='{candidate_id}'",
            ),
        )
    with pytest.raises(SourceIntegrityError, match="Owned capture evidence is invalid"):
        await _snapshots(capture_stack, experience_id, version_id)
