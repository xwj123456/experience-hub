from __future__ import annotations

import json
import os
import shutil
import sqlite3
import traceback
from collections.abc import Awaitable, Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences import (
    CreateExperience,
    ExperienceKind,
    VersionContent,
)
from experience_hub.experiments import policies as policy_module
from experience_hub.experiments.contracts import (
    ExperienceLabelV1,
    PolicyArmDescriptorV1,
    PolicyArmKind,
    ReplayCaseV1,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.policies import (
    ExperienceHubPolicyArm,
    NoMemoryPolicyArm,
    PolicyExecutionContext,
    build_policy_arm,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval.contracts import PeekExperiences, SearchResult
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import ExperienceEvidenceReader
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.database import DatabaseBusy
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.unit_of_work import UnitOfWork

FROZEN_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)

CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]
_SQLITE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _ids() -> SequenceIdGenerator:
    return SequenceIdGenerator(
        tuple(
            UUID(f"00000000-0000-4000-8000-{value:012d}")
            for value in range(1, 200)
        )
    )


async def _execute(
    container: ApplicationContainer,
    request: CommandRequest,
    handler: CommandHandler,
) -> CommandResult:
    result = await container.command_executor.execute(request, handler)
    assert result.status_code == 201
    return result


async def _create_agent(
    container: ApplicationContainer,
    *,
    name: str,
    key: str,
) -> UUID:
    request = CommandRequest(
        caller_scope="system:replay-test",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        body={"name": name},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=name),
            command_context=context,
        )

    response = await _execute(container, request, handler)
    return UUID(cast(str, json.loads(response.body)["data"]["agent_id"]))


async def _create_experience(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    key: str,
    marker: str,
) -> UUID:
    content = VersionContent(
        body=f"Queue pressure replay evidence {marker}.",
        summary="Queue pressure replay evidence",
        mechanism="bounded backpressure",
        tags=("queue", "pressure"),
        applicability=("local replay",),
        evidence=(),
        falsifiers=(),
    )
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        importance=0.50,
        confidence=0.70,
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="experience.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": owner_agent_id},
        body={
            **content.model_dump(mode="python", warnings=False),
            "confidence": 0.70,
            "importance": 0.50,
            "kind": ExperienceKind.PROCEDURAL,
            "links": (),
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.experience_service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    response = await _execute(container, request, handler)
    return UUID(cast(str, json.loads(response.body)["data"]["experience_id"]))


async def _seed_source(
    source_path: Path,
) -> tuple[UUID, UUID, UUID, UUID]:
    runtime = ApplicationRuntime(
        Settings(database_url=f"sqlite+aiosqlite:///{source_path}"),
        clock=FrozenClock(FROZEN_AT),
        ids=_ids(),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        owner_id = await _create_agent(container, name="Replay Owner", key="owner")
        foreign_owner_id = await _create_agent(
            container,
            name="Foreign Owner",
            key="foreign-owner",
        )
        expected_id = await _create_experience(
            container,
            owner_agent_id=owner_id,
            key="expected",
            marker="expected",
        )
        unmapped_id = await _create_experience(
            container,
            owner_agent_id=owner_id,
            key="unmapped",
            marker="unmapped",
        )
        foreign_id = await _create_experience(
            container,
            owner_agent_id=foreign_owner_id,
            key="foreign",
            marker="foreign",
        )
    return owner_id, expected_id, unmapped_id, foreign_id


def _descriptor(kind: PolicyArmKind) -> PolicyArmDescriptorV1:
    return PolicyArmDescriptorV1(
        schema_version=1,
        arm_id=kind.value,
        kind=kind,
        required=True,
    )


def _case(
    *,
    owner_id: UUID,
    expected_id: UUID,
    foreign_id: UUID,
) -> ReplayCaseV1:
    return ReplayCaseV1(
        schema_version=1,
        case_id="owner-scope",
        owner_agent_id=owner_id,
        query="queue pressure",
        mode=RetrievalMode.FOCUSED,
        tags=("queue",),
        mechanism_cues=("bounded-backpressure",),
        limit=10,
        content_budget_bytes=8_192,
        expand_cold=False,
        expected=(
            ExperienceLabelV1(
                label="owned-expected",
                experience_id=expected_id,
            ),
        ),
        forbidden=(
            ExperienceLabelV1(
                label="foreign-forbidden",
                experience_id=foreign_id,
            ),
        ),
    )


def _retained_sqlite_bytes(path: Path) -> dict[str, bytes | None]:
    return {
        suffix: (
            Path(f"{path}{suffix}").read_bytes()
            if Path(f"{path}{suffix}").is_file()
            else None
        )
        for suffix in _SQLITE_SUFFIXES
    }


def _context(case: ReplayCaseV1, clone_path: Path) -> PolicyExecutionContext:
    return PolicyExecutionContext(
        case=case,
        clone_path=clone_path,
        frozen_at=FROZEN_AT,
        seed=17,
    )


@pytest.mark.asyncio
async def test_no_memory_returns_empty_without_opening_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone_path = tmp_path / "must-not-open.sqlite3"
    calls: list[object] = []
    original_connect = sqlite3.connect

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        calls.append(args[0] if args else None)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    case = _case(
        owner_id=UUID("00000000-0000-4000-8000-000000000001"),
        expected_id=UUID("00000000-0000-4000-8000-000000000002"),
        foreign_id=UUID("00000000-0000-4000-8000-000000000003"),
    )

    observation = await build_policy_arm(
        _descriptor(PolicyArmKind.NO_MEMORY)
    ).execute(
        PolicyExecutionContext(
            case=case,
            clone_path=clone_path,
            frozen_at=FROZEN_AT,
            seed=17,
        )
    )

    assert observation.returned_labels == ()
    assert observation.unmapped_count == 0
    assert calls == []
    assert not clone_path.exists()


@pytest.mark.asyncio
async def test_experience_hub_peeks_clone_with_owner_isolation_and_logical_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "authoritative.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    source_before = source_path.read_bytes()
    clone_path = tmp_path / "disposable-clone.sqlite3"
    shutil.copyfile(source_path, clone_path)

    opened: list[Path] = []
    original_connect = sqlite3.connect

    def guarded_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        if args:
            opened_path = Path(str(args[0])).resolve()
            if opened_path == source_path.resolve():
                raise AssertionError("policy arm opened the source database")
            opened.append(opened_path)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    peek_queries: list[PeekExperiences] = []
    original_peek = ExperienceEvidenceReader.peek

    async def recording_peek(
        self: ExperienceEvidenceReader,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        peek_queries.append(query)
        return await original_peek(self, session=session, query=query)

    monkeypatch.setattr(ExperienceEvidenceReader, "peek", recording_peek)
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    observation = await build_policy_arm(
        _descriptor(PolicyArmKind.EXPERIENCE_HUB)
    ).execute(
        PolicyExecutionContext(
            case=case,
            clone_path=clone_path,
            frozen_at=FROZEN_AT,
            seed=17,
        )
    )

    assert observation.returned_labels == ("owned-expected",)
    assert observation.unmapped_count == 1
    assert peek_queries == [
        PeekExperiences(
            owner_agent_id=owner_id,
            query=case.query,
            mode=case.mode,
            tags=case.tags,
            mechanism_cues=case.mechanism_cues,
            limit=case.limit,
            content_budget_bytes=case.content_budget_bytes,
            expand_cold=case.expand_cold,
        )
    ]
    assert clone_path.resolve() in opened
    assert source_path.resolve() not in opened

    with closing(sqlite3.connect(clone_path)) as connection:
        connection.execute("CREATE TABLE replay_clone_marker(value INTEGER)")
        connection.commit()
        marker = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'replay_clone_marker'"
        ).fetchone()
    assert marker == ("replay_clone_marker",)
    assert source_path.read_bytes() == source_before
    assert str(expected_id) not in repr(observation)
    assert str(foreign_id) not in repr(observation)


def test_policy_registry_is_closed_and_exhaustive() -> None:
    assert isinstance(
        build_policy_arm(_descriptor(PolicyArmKind.NO_MEMORY)),
        NoMemoryPolicyArm,
    )
    assert isinstance(
        build_policy_arm(_descriptor(PolicyArmKind.EXPERIENCE_HUB)),
        ExperienceHubPolicyArm,
    )
    forged = _descriptor(PolicyArmKind.NO_MEMORY).model_copy(
        update={"kind": "third_party.module"}
    )

    with pytest.raises(ExperimentInputError) as raised:
        build_policy_arm(forged)

    assert raised.value.code == "replay_policy_unsupported"
    assert "third_party.module" not in str(raised.value)


@pytest.mark.asyncio
async def test_experience_hub_maps_clone_schema_failure_without_private_detail(
    tmp_path: Path,
) -> None:
    clone_path = tmp_path / "private-clone-name.sqlite3"
    private_revision = "private-revision-detail"
    with closing(sqlite3.connect(clone_path)) as connection:
        connection.execute(
            "CREATE TABLE alembic_version "
            "(version_num VARCHAR(255) NOT NULL PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO alembic_version(version_num) VALUES (?)",
            (private_revision,),
        )
        connection.commit()
    case = _case(
        owner_id=UUID("00000000-0000-4000-8000-000000000001"),
        expected_id=UUID("00000000-0000-4000-8000-000000000002"),
        foreign_id=UUID("00000000-0000-4000-8000-000000000003"),
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_policy_arm(
            _descriptor(PolicyArmKind.EXPERIENCE_HUB)
        ).execute(
            PolicyExecutionContext(
                case=case,
                clone_path=clone_path,
                frozen_at=FROZEN_AT,
                seed=17,
            )
        )

    assert raised.value.code == "replay_policy_clone_invalid"
    assert private_revision not in str(raised.value)
    assert str(clone_path) not in str(raised.value)


@pytest.mark.parametrize(
    "alias_kind",
    ["leaf-symlink", "parent-symlink", "hard-link"],
)
@pytest.mark.asyncio
async def test_experience_hub_rejects_observable_clone_aliases_without_source_write(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    source_path = tmp_path / "authoritative-alias-source.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    source_before = _retained_sqlite_bytes(source_path)
    if alias_kind == "leaf-symlink":
        clone_path = tmp_path / "leaf-alias.sqlite3"
        clone_path.symlink_to(source_path)
    elif alias_kind == "parent-symlink":
        alias_parent = tmp_path / "parent-alias"
        alias_parent.symlink_to(source_path.parent, target_is_directory=True)
        clone_path = alias_parent / source_path.name
    else:
        clone_path = tmp_path / "hard-link-alias.sqlite3"
        os.link(source_path, clone_path)
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_policy_arm(
            _descriptor(PolicyArmKind.EXPERIENCE_HUB)
        ).execute(_context(case, clone_path))

    assert raised.value.code == "replay_policy_clone_invalid"
    assert raised.value.__cause__ is None
    assert str(source_path) not in str(raised.value)
    assert _retained_sqlite_bytes(source_path) == source_before


@pytest.mark.asyncio
async def test_experience_hub_opens_exact_clone_name_containing_question_mark(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "question-source.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    clone_path = tmp_path / "exact-clone?literal.sqlite3"
    shutil.copyfile(source_path, clone_path)
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    observation = await build_policy_arm(
        _descriptor(PolicyArmKind.EXPERIENCE_HUB)
    ).execute(_context(case, clone_path))

    assert observation.returned_labels == ("owned-expected",)
    assert observation.unmapped_count == 1
    assert clone_path.is_file()
    assert not (tmp_path / "exact-clone").exists()


@pytest.mark.asyncio
async def test_experience_hub_rechecks_declared_clone_identity_after_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "identity-source.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    source_before = source_path.read_bytes()
    clone_path = tmp_path / "identity-clone.sqlite3"
    shutil.copyfile(source_path, clone_path)
    replacement_path = tmp_path / "replacement.sqlite3"
    original_peek = ExperienceEvidenceReader.peek

    async def replacing_peek(
        self: ExperienceEvidenceReader,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        result = await original_peek(self, session=session, query=query)
        shutil.copyfile(source_path, replacement_path)
        os.replace(replacement_path, clone_path)
        return result

    monkeypatch.setattr(ExperienceEvidenceReader, "peek", replacing_peek)
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_policy_arm(
            _descriptor(PolicyArmKind.EXPERIENCE_HUB)
        ).execute(_context(case, clone_path))

    assert raised.value.code == "replay_policy_clone_invalid"
    assert raised.value.__cause__ is None
    assert source_path.read_bytes() == source_before


@pytest.mark.asyncio
async def test_experience_hub_maps_database_busy_without_private_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "busy-source.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    clone_path = tmp_path / "busy-clone.sqlite3"
    shutil.copyfile(source_path, clone_path)
    private_detail = f"SELECT secret FROM {clone_path}"

    async def busy_migrator(settings: Settings) -> str:
        del settings
        error = DatabaseBusy()
        error.args = (private_detail,)
        raise error

    monkeypatch.setattr(policy_module, "require_current_schema", busy_migrator)
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_policy_arm(
            _descriptor(PolicyArmKind.EXPERIENCE_HUB)
        ).execute(_context(case, clone_path))

    assert raised.value.code == "replay_policy_clone_invalid"
    assert raised.value.__cause__ is None
    assert private_detail not in str(raised.value)
    assert str(clone_path) not in str(raised.value)


@pytest.mark.asyncio
async def test_clone_identity_failure_suppresses_inflight_runtime_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "context-source.sqlite3"
    owner_id, expected_id, _, foreign_id = await _seed_source(source_path)
    clone_path = tmp_path / "context-clone.sqlite3"
    shutil.copyfile(source_path, clone_path)
    replacement_path = tmp_path / "context-replacement.sqlite3"
    private_detail = f"private SQL SELECT * FROM {clone_path}"

    async def replacing_failed_peek(
        self: ExperienceEvidenceReader,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        del self, session, query
        shutil.copyfile(source_path, replacement_path)
        os.replace(replacement_path, clone_path)
        raise SQLAlchemyError(private_detail)

    monkeypatch.setattr(
        ExperienceEvidenceReader,
        "peek",
        replacing_failed_peek,
    )
    case = _case(
        owner_id=owner_id,
        expected_id=expected_id,
        foreign_id=foreign_id,
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_policy_arm(
            _descriptor(PolicyArmKind.EXPERIENCE_HUB)
        ).execute(_context(case, clone_path))

    rendered = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert raised.value.code == "replay_policy_clone_invalid"
    assert str(raised.value) == (
        "replay_policy_clone_invalid: "
        "Replay policy clone is not a valid current-schema database"
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert private_detail not in rendered
    assert str(clone_path) not in rendered
