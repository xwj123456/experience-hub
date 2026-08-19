from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import UUID

import pytest

import experience_hub.experiments.runner as runner_module
from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences import (
    CreateExperience,
    ExperienceKind,
    VersionContent,
)
from experience_hub.experiments import (
    ArmObservationV1,
    ExperimentIsolationError,
    ExperimentOutputError,
    OracleEvidenceV1,
    ReplayExecution,
    WorkspacePolicy,
    checkpoint_owned_sqlite,
    inspect_replay,
    run_replay,
    verify_replay_report,
)
from experience_hub.experiments.policies import PolicyArm, PolicyExecutionContext
from experience_hub.experiments.workspace import OwnedWorkspace
from experience_hub.ids import SequenceIdGenerator
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import CommandResult, StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork

FROZEN_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]


def _ids() -> SequenceIdGenerator:
    return SequenceIdGenerator(
        tuple(
            UUID(f"00000000-0000-4000-8000-{value:012d}")
            for value in range(1, 300)
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
        caller_scope="system:replay-runner-test",
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
    body: str,
) -> UUID:
    content = VersionContent(
        body=body,
        summary=body,
        mechanism="bounded backpressure",
        tags=("queue", "pressure"),
        applicability=("local replay",),
        evidence=(),
        falsifiers=(),
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
            "confidence": 0.7,
            "importance": 0.5,
            "kind": ExperienceKind.PROCEDURAL,
            "links": (),
        },
    )
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        importance=0.5,
        confidence=0.7,
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


@pytest.fixture
async def replay_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, tuple[str, ...]]:
    source = tmp_path / "source.sqlite3"
    runtime = ApplicationRuntime(
        Settings(database_url=f"sqlite+aiosqlite:///{source}"),
        clock=FrozenClock(FROZEN_AT),
        ids=_ids(),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        owner_id = await _create_agent(container, name="Owner", key="owner")
        foreign_owner_id = await _create_agent(
            container,
            name="Foreign",
            key="foreign-owner",
        )
        expected_id = await _create_experience(
            container,
            owner_agent_id=owner_id,
            key="expected",
            body="Durable acknowledgement bounds queue pressure.",
        )
        await _create_experience(
            container,
            owner_agent_id=owner_id,
            key="distractor",
            body="Cold archive typography is unrelated.",
        )
        foreign_id = await _create_experience(
            container,
            owner_agent_id=foreign_owner_id,
            key="foreign",
            body="Durable acknowledgement bounds queue pressure.",
        )
    checkpoint_owned_sqlite(source)

    case_ids = ("queue-case", "pressure-case")
    case_documents = tuple(
        {
            "case_id": case_id,
            "content_budget_bytes": 8192,
            "expand_cold": False,
            "expected": [
                {
                    "experience_id": str(expected_id),
                    "label": "owned-expected",
                }
            ],
            "forbidden": [
                {
                    "experience_id": str(foreign_id),
                    "label": "foreign-forbidden",
                }
            ],
            "limit": 10,
            "mechanism_cues": ["bounded-backpressure"],
            "mode": "focused",
            "owner_agent_id": str(owner_id),
            "query": "durable acknowledgement queue pressure",
            "schema_version": 1,
            "tags": ["queue"],
        }
        for case_id in case_ids
    )
    cases_body = b"".join(
        canonical_json_bytes(document) + b"\n" for document in case_documents
    )
    (tmp_path / "cases.jsonl").write_bytes(cases_body)
    manifest = {
        "arms": [
            {
                "arm_id": "no_memory",
                "kind": "no_memory",
                "required": True,
                "schema_version": 1,
            },
            {
                "arm_id": "experience_hub",
                "kind": "experience_hub",
                "required": True,
                "schema_version": 1,
            },
        ],
        "dataset": {
            "cases_file": "cases.jsonl",
            "cases_sha256": sha256_hex(cases_body),
            "dataset_id": "runner-cases",
            "schema_version": 1,
        },
        "deterministic_replay_runs": 2,
        "evidence_schema_version": 1,
        "experiment_id": "runner-smoke",
        "frozen_at": "2026-07-26T12:00:00.000000Z",
        "oracle": {
            "kind": "retrieval_labels",
            "schema_version": 1,
            "version": 1,
        },
        "profile_schema_version": 1,
        "schema_version": 1,
        "seed": 17,
        "snapshot_binding": "validated_source",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path, source, case_ids


@pytest.mark.asyncio
async def test_runner_uses_ordered_independent_exact_clones_and_canonical_artifacts(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, case_ids = replay_fixture
    source_before = source.read_bytes()
    clones: list[tuple[Path, str]] = []
    prepared: list[Path] = []
    original_clone = runner_module.clone_frozen_sqlite
    original_prepare = runner_module.prepare_owned_workspace

    def recording_clone(
        snapshot: runner_module.FrozenSqliteSnapshot,
        destination: Path,
    ) -> Path:
        clone = original_clone(snapshot, destination)
        if "arms" in clone.parts:
            clones.append((clone, sha256_hex(clone.read_bytes())))
        return clone

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", recording_clone)

    def recording_prepare(
        path: Path,
        **kwargs: object,
    ) -> OwnedWorkspace:
        prepared.append(path)
        return original_prepare(
            path,
            policy=cast(WorkspacePolicy, kwargs["policy"]),
            replace_owned=cast(bool, kwargs["replace_owned"]),
            allow_unmarked_empty=cast(bool, kwargs["allow_unmarked_empty"]),
        )

    monkeypatch.setattr(
        runner_module,
        "prepare_owned_workspace",
        recording_prepare,
    )
    workspace = tmp_path / "replay"

    execution = await run_replay(
        manifest_path=manifest_path,
        database_path=source,
        workspace_path=workspace,
    )

    assert isinstance(execution, ReplayExecution)
    assert execution.valid is True
    assert execution.profile_complete is True
    assert execution.evidence.data.source_unchanged is True
    assert execution.evidence.data.clone_isolation_verified is True
    assert execution.evidence.data.deterministic_replay_match is True
    assert execution.evidence_body == (
        workspace / "artifacts" / "evidence.json"
    ).read_bytes()
    assert execution.profile_body == (
        workspace / "artifacts" / "profile.json"
    ).read_bytes()
    assert source.read_bytes() == source_before
    assert tuple(case.case_id for case in execution.evidence.data.cases) == case_ids
    assert tuple(
        tuple(arm.arm_id for arm in case.arms)
        for case in execution.evidence.data.cases
    ) == (("no_memory", "experience_hub"),) * len(case_ids)
    assert len(clones) == 2 * len(case_ids) * 2
    assert prepared == [workspace]
    assert {digest for _, digest in clones} == {
        execution.evidence.data.resolved_manifest.snapshot_sha256
    }
    assert [path.parts[-4:] for path, _ in clones] == [
        ("arms", replay, case_id, f"{arm}.sqlite3")
        for replay in ("replay-a", "replay-b")
        for case_id in case_ids
        for arm in ("no_memory", "experience_hub")
    ]
    evidence_text = execution.evidence_body.decode()
    assert str(source) not in evidence_text
    assert str(workspace) not in evidence_text
    assert str(replay_fixture) not in evidence_text
    assert b"wall_duration_ns" not in execution.evidence_body
    assert b"database_bytes" not in execution.evidence_body


@pytest.mark.asyncio
async def test_runner_runs_symlink_preflight_off_the_event_loop_thread(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    event_loop_thread = threading.get_ident()
    preflight_threads: list[int] = []
    original_preflight = runner_module._require_no_symlink_components

    def recording_preflight(*paths: Path) -> None:
        preflight_threads.append(threading.get_ident())
        original_preflight(*paths)

    monkeypatch.setattr(
        runner_module,
        "_require_no_symlink_components",
        recording_preflight,
    )

    execution = await run_replay(
        manifest_path,
        source,
        tmp_path / "replay",
    )

    assert execution.valid is True
    assert len(preflight_threads) == 2
    assert all(
        thread_id != event_loop_thread for thread_id in preflight_threads
    )


@pytest.mark.asyncio
async def test_arm_exception_is_a_stable_incomplete_required_arm(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    private_detail = "/private/owner/source.sqlite3"
    original_builder = runner_module.build_policy_arm
    original_score = runner_module.score_retrieval_observation
    scored: list[str] = []

    class FailingArm:
        def __init__(self, wrapped: PolicyArm) -> None:
            self.descriptor = wrapped.descriptor

        async def execute(self, context: PolicyExecutionContext) -> object:
            del context
            raise RuntimeError(private_detail)

    def failing_builder(
        descriptor: runner_module.PolicyArmDescriptorV1,
    ) -> PolicyArm:
        arm = original_builder(descriptor)
        if descriptor.arm_id == "experience_hub":
            return cast(PolicyArm, FailingArm(arm))
        return arm

    monkeypatch.setattr(runner_module, "build_policy_arm", failing_builder)

    def recording_score(
        case: runner_module.ReplayCaseV1,
        observation: ArmObservationV1,
    ) -> OracleEvidenceV1:
        scored.append(case.case_id)
        return original_score(case, observation)

    monkeypatch.setattr(
        runner_module,
        "score_retrieval_observation",
        recording_score,
    )

    execution = await run_replay(
        manifest_path=manifest_path,
        database_path=source,
        workspace_path=tmp_path / "replay",
    )

    case = execution.evidence.data.cases[0]
    failed = case.arms[1]
    assert case.status == "incomplete"
    assert failed.status == "failed"
    assert failed.error_code == "arm_infrastructure_failure"
    assert failed.error_stage == "execute"
    assert case.delta_utility_micros is None
    assert execution.evidence.data.comparison_complete is False
    assert execution.valid is False
    assert private_detail not in execution.evidence_body.decode()
    assert scored == ["queue-case", "pressure-case"] * 2


@pytest.mark.asyncio
async def test_inspect_returns_only_logical_counts_and_hashes(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
) -> None:
    manifest_path, source, case_ids = replay_fixture

    inspection = await inspect_replay(manifest_path, source)

    assert inspection.case_count == len(case_ids)
    assert inspection.arm_count == 2
    assert inspection.manifest_sha256 == sha256_hex(manifest_path.read_bytes())
    assert inspection.cases_sha256 == sha256_hex(
        manifest_path.with_name("cases.jsonl").read_bytes()
    )
    assert inspection.snapshot_sha256 == sha256_hex(source.read_bytes())
    assert "/" not in repr(inspection)
    assert str(source) not in repr(inspection)


def test_verify_replay_report_uses_the_canonical_bounded_verifier(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evidence.json"
    report.write_bytes(b"x" * (2 * 1024 * 1024 + 1))

    with pytest.raises(runner_module.ExperimentOutputError) as captured:
        verify_replay_report(report)

    assert captured.value.code == "output_too_large"


@pytest.mark.asyncio
async def test_invalid_oracle_result_fails_the_required_arm_closed(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_score = runner_module.score_retrieval_observation

    def invalid_score(
        case: runner_module.ReplayCaseV1,
        observation: ArmObservationV1,
    ) -> OracleEvidenceV1:
        valid = original_score(case, observation)
        return valid.model_copy(update={"utility_micros": -1})

    monkeypatch.setattr(
        runner_module,
        "score_retrieval_observation",
        invalid_score,
    )

    execution = await run_replay(
        manifest_path,
        source,
        tmp_path / "replay",
    )

    assert execution.valid is False
    assert execution.evidence.data.comparison_complete is False
    assert all(
        arm.error_code == "oracle_validation_failure"
        for case in execution.evidence.data.cases
        for arm in case.arms
    )
    report = verify_replay_report(
        tmp_path / "replay" / "artifacts" / "evidence.json"
    )
    assert report.data.valid is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("main", "wal"))
async def test_source_mutation_during_execution_aborts_without_final_evidence(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_builder = runner_module.build_policy_arm
    mutated = False

    class MutatingArm:
        def __init__(self, wrapped: PolicyArm) -> None:
            self.descriptor = wrapped.descriptor
            self._wrapped = wrapped

        async def execute(
            self,
            context: PolicyExecutionContext,
        ) -> ArmObservationV1:
            nonlocal mutated
            observation = await self._wrapped.execute(context)
            if not mutated:
                mutated = True
                if mutation == "main":
                    source.write_bytes(source.read_bytes() + b"private mutation")
                else:
                    Path(f"{source}-wal").write_bytes(b"private wal")
            return observation

    def mutating_builder(
        descriptor: runner_module.PolicyArmDescriptorV1,
    ) -> PolicyArm:
        arm = original_builder(descriptor)
        if descriptor.arm_id == "no_memory":
            return MutatingArm(arm)
        return arm

    monkeypatch.setattr(runner_module, "build_policy_arm", mutating_builder)

    with pytest.raises(ExperimentIsolationError) as captured:
        await run_replay(manifest_path, source, tmp_path / "replay")

    assert captured.value.code == "replay_snapshot_changed"
    assert not (tmp_path / "replay" / "artifacts" / "evidence.json").exists()


@pytest.mark.asyncio
async def test_clone_corruption_before_arm_startup_aborts_without_execution(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_clone = runner_module.clone_frozen_sqlite
    corrupted = False

    def corrupting_clone(
        snapshot: runner_module.FrozenSqliteSnapshot,
        destination: Path,
    ) -> Path:
        nonlocal corrupted
        clone = original_clone(snapshot, destination)
        if "arms" in clone.parts and not corrupted:
            corrupted = True
            clone.write_bytes(clone.read_bytes() + b"corrupt")
        return clone

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", corrupting_clone)

    with pytest.raises(ExperimentIsolationError) as captured:
        await run_replay(manifest_path, source, tmp_path / "replay")

    assert captured.value.code == "replay_clone_corrupted"
    assert not (tmp_path / "replay" / "artifacts" / "evidence.json").exists()


@pytest.mark.asyncio
async def test_second_pass_divergence_is_invalid_and_not_published(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_suite = runner_module._execute_suite

    async def divergent_suite(**kwargs: object) -> runner_module._SuiteResult:
        result = await original_suite(**kwargs)
        if kwargs["replay_name"] != "replay-b":
            return result
        case = result.cases[0]
        treatment = case.arms[1].model_copy(
            update={
                "observation": ArmObservationV1(
                    schema_version=1,
                    returned_labels=(),
                    unmapped_count=0,
                ),
                "utility_micros": 0,
            }
        )
        changed = case.model_copy(
            update={
                "arms": (case.arms[0], treatment),
                "delta_utility_micros": 0,
            }
        )
        return runner_module._SuiteResult(
            cases=(changed, *result.cases[1:]),
            clone_identities=result.clone_identities,
        )

    monkeypatch.setattr(runner_module, "_execute_suite", divergent_suite)

    execution = await run_replay(
        manifest_path,
        source,
        tmp_path / "replay",
    )

    assert execution.valid is False
    assert execution.evidence.data.deterministic_replay_match is False
    assert execution.evidence.data.valid is False
    assert not (tmp_path / "replay" / "artifacts" / "evidence.json").exists()


@pytest.mark.asyncio
async def test_evidence_write_failure_never_claims_success(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_write = OwnedWorkspace.atomic_write

    def fail_evidence(
        self: OwnedWorkspace,
        relative: PurePosixPath,
        body: bytes,
    ) -> Path:
        if str(relative) == "artifacts/evidence.json":
            raise ExperimentIsolationError("private-write", "/private/path")
        return original_write(self, relative, body)

    monkeypatch.setattr(OwnedWorkspace, "atomic_write", fail_evidence)

    with pytest.raises(ExperimentOutputError) as captured:
        await run_replay(manifest_path, source, tmp_path / "replay")

    assert captured.value.code == "artifact_write_failed"
    assert "/private/path" not in str(captured.value)
    assert not (tmp_path / "replay" / "artifacts" / "evidence.json").exists()


@pytest.mark.asyncio
async def test_profile_collection_failure_preserves_validated_evidence_and_scores(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    manifest_path, source, _ = replay_fixture
    baseline = await run_replay(
        manifest_path,
        source,
        tmp_path / "baseline",
    )

    def failed_profiler() -> int:
        raise RuntimeError("/private/runtime clock")

    execution = await run_replay(
        manifest_path,
        source,
        tmp_path / "replay",
        profiler=failed_profiler,
    )

    assert execution.valid is False
    assert execution.profile_complete is False
    assert execution.profile is not None
    assert execution.profile.data.profile_complete is False
    assert execution.evidence.data.valid is True
    assert execution.evidence_body == baseline.evidence_body
    assert verify_replay_report(
        tmp_path / "replay" / "artifacts" / "evidence.json"
    ) == execution.evidence


@pytest.mark.asyncio
async def test_profile_write_failure_preserves_validated_evidence_and_scores(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    baseline = await run_replay(
        manifest_path,
        source,
        tmp_path / "baseline",
    )
    original_write = OwnedWorkspace.atomic_write

    def fail_profile(
        self: OwnedWorkspace,
        relative: PurePosixPath,
        body: bytes,
    ) -> Path:
        if str(relative) == "artifacts/profile.json":
            raise ExperimentIsolationError("private-write", "/private/path")
        return original_write(self, relative, body)

    monkeypatch.setattr(OwnedWorkspace, "atomic_write", fail_profile)

    execution = await run_replay(
        manifest_path,
        source,
        tmp_path / "replay",
    )

    assert execution.valid is False
    assert execution.profile_complete is False
    assert execution.profile_body is None
    assert execution.evidence.data.valid is True
    assert execution.evidence_body == baseline.evidence_body
    assert verify_replay_report(
        tmp_path / "replay" / "artifacts" / "evidence.json"
    ) == execution.evidence
    assert not (tmp_path / "replay" / "artifacts" / "profile.json").exists()


@pytest.mark.asyncio
async def test_runner_refuses_owned_output_without_replace_and_unknown_entries(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    manifest_path, source, _ = replay_fixture
    workspace = tmp_path / "replay"
    await run_replay(manifest_path, source, workspace)

    with pytest.raises(ExperimentIsolationError) as exists:
        await run_replay(manifest_path, source, workspace)
    assert exists.value.code == "replay_workspace_exists"

    (workspace / "unknown.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExperimentIsolationError) as unknown:
        await run_replay(
            manifest_path,
            source,
            workspace,
            replace_owned=True,
        )
    assert unknown.value.code == "replay_workspace_unowned"
    assert (workspace / "unknown.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_role", "owned_entry", "parent_segment"),
    (
        ("database", "snapshot", False),
        ("manifest", "validation", False),
        ("cases", "arms", False),
        ("database", "artifacts", True),
    ),
)
async def test_runner_rejects_owned_input_overlap_before_deleting_any_output(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    input_role: str,
    owned_entry: str,
    parent_segment: bool,
) -> None:
    manifest_path, source, _ = replay_fixture
    workspace = tmp_path / "replay"
    runner_module.prepare_owned_workspace(
        workspace,
        policy=runner_module.REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    nested = workspace / owned_entry / "nested-input"
    nested.mkdir(parents=True)

    replay_manifest = manifest_path
    replay_database = source
    if input_role == "database":
        protected_input = nested / "source.sqlite3"
        shutil.copy2(source, protected_input)
        replay_database = protected_input
        if parent_segment:
            replay_database = (
                workspace
                / ".."
                / workspace.name
                / owned_entry
                / nested.name
                / protected_input.name
            )
    else:
        replay_manifest = nested / "manifest.json"
        replay_cases = nested / "cases.jsonl"
        shutil.copy2(manifest_path, replay_manifest)
        shutil.copy2(manifest_path.with_name("cases.jsonl"), replay_cases)
        protected_input = (
            replay_manifest if input_role == "manifest" else replay_cases
        )

    sentinels = []
    for entry in sorted(runner_module.REPLAY_WORKSPACE_POLICY.owned_entries):
        sentinel = workspace / entry / "preflight-sentinel.txt"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(f"retain-{entry}".encode())
        sentinels.append(sentinel)
    retained_paths = (*sentinels, protected_input)
    retained = {
        path: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.read_bytes(),
        )
        for path in retained_paths
    }

    with pytest.raises(ExperimentIsolationError) as captured:
        await run_replay(
            replay_manifest,
            replay_database,
            workspace,
            replace_owned=True,
        )

    assert captured.value.code == "replay_workspace_input_overlap"
    assert str(workspace) not in str(captured.value)
    assert str(protected_input) not in str(captured.value)
    for path, (device, inode, body) in retained.items():
        status = path.lstat()
        assert (status.st_dev, status.st_ino) == (device, inode)
        assert path.read_bytes() == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("alias_kind", "owned_entry"),
    (
        ("database-final", "snapshot"),
        ("database-parent", "validation"),
        ("cases-final", "arms"),
    ),
)
async def test_runner_rejects_symlinked_owned_input_before_deleting_any_output(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    alias_kind: str,
    owned_entry: str,
) -> None:
    manifest_path, source, _ = replay_fixture
    workspace = tmp_path / "replay"
    runner_module.prepare_owned_workspace(
        workspace,
        policy=runner_module.REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    nested = workspace / owned_entry / "aliased-input"
    nested.mkdir(parents=True)

    replay_manifest = manifest_path
    replay_database = source
    if alias_kind == "cases-final":
        protected_input = nested / "cases.jsonl"
        shutil.copy2(manifest_path.with_name("cases.jsonl"), protected_input)
        external_inputs = tmp_path / "external-inputs"
        external_inputs.mkdir()
        replay_manifest = external_inputs / "manifest.json"
        shutil.copy2(manifest_path, replay_manifest)
        alias = external_inputs / "cases.jsonl"
        alias.symlink_to(protected_input)
    else:
        protected_input = nested / "source.sqlite3"
        shutil.copy2(source, protected_input)
        if alias_kind == "database-final":
            alias = tmp_path / "database-link.sqlite3"
            alias.symlink_to(protected_input)
            replay_database = alias
        else:
            alias = tmp_path / "database-parent"
            alias.symlink_to(nested, target_is_directory=True)
            replay_database = alias / protected_input.name

    sentinels = []
    for entry in sorted(runner_module.REPLAY_WORKSPACE_POLICY.owned_entries):
        sentinel = workspace / entry / "alias-sentinel.txt"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(f"retain-{entry}".encode())
        sentinels.append(sentinel)
    retained_paths = (*sentinels, protected_input)
    retained = {
        path: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.read_bytes(),
        )
        for path in retained_paths
    }
    alias_status = alias.lstat()
    alias_target = alias.readlink()

    with pytest.raises(ExperimentIsolationError) as captured:
        await run_replay(
            replay_manifest,
            replay_database,
            workspace,
            replace_owned=True,
        )

    assert captured.value.code == "replay_workspace_input_overlap"
    assert str(workspace) not in str(captured.value)
    assert str(protected_input) not in str(captured.value)
    for path, (device, inode, body) in retained.items():
        status = path.lstat()
        assert (status.st_dev, status.st_ino) == (device, inode)
        assert path.read_bytes() == body
    final_alias_status = alias.lstat()
    assert (final_alias_status.st_dev, final_alias_status.st_ino) == (
        alias_status.st_dev,
        alias_status.st_ino,
    )
    assert alias.readlink() == alias_target


def test_verify_replay_report_rejects_symlink_and_nonfile(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    symlink = tmp_path / "report.json"
    symlink.symlink_to(target)

    for path in (symlink, tmp_path):
        with pytest.raises(ExperimentOutputError) as captured:
            verify_replay_report(path)
        assert captured.value.code == "invalid_report_path"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("main", "wal"))
async def test_final_source_check_runs_after_profile_collection(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest_path, source, _ = replay_fixture
    profiler_calls = 0

    def mutating_profiler() -> int:
        nonlocal profiler_calls
        profiler_calls += 1
        if profiler_calls == 2:
            if mutation == "main":
                source.write_bytes(source.read_bytes() + b"profile mutation")
            else:
                Path(f"{source}-wal").write_bytes(b"profile wal")
        return profiler_calls * 10

    with pytest.raises(ExperimentIsolationError) as captured:
        await run_replay(
            manifest_path,
            source,
            tmp_path / "replay",
            profiler=mutating_profiler,
        )

    assert captured.value.code == "replay_snapshot_changed"
    assert profiler_calls == 2
    assert not (tmp_path / "replay" / "artifacts" / "evidence.json").exists()
    assert not (tmp_path / "replay" / "artifacts" / "profile.json").exists()


@pytest.mark.asyncio
async def test_inspect_cleanup_refuses_replaced_temporary_root(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_validate = runner_module.validate_frozen_snapshot
    roots: list[tuple[Path, Path]] = []

    async def replace_after_validation(
        snapshot: runner_module.FrozenSqliteSnapshot,
        *,
        validation_path: Path,
        frozen_at: datetime,
        seed: int,
    ) -> str:
        revision = await original_validate(
            snapshot,
            validation_path=validation_path,
            frozen_at=frozen_at,
            seed=seed,
        )
        root = validation_path.parents[1]
        retained = root.with_name(f"{root.name}-retained")
        root.rename(retained)
        root.mkdir()
        (root / "keep.txt").write_text("replacement", encoding="utf-8")
        roots.append((root, retained))
        return revision

    monkeypatch.setattr(
        runner_module,
        "validate_frozen_snapshot",
        replace_after_validation,
    )

    with pytest.raises(ExperimentIsolationError) as captured:
        await inspect_replay(manifest_path, source)

    assert captured.value.code == "replay_inspect_cleanup_unsafe"
    assert captured.value.__cause__ is None
    root, retained = roots[0]
    assert (root / "keep.txt").read_text(encoding="utf-8") == "replacement"
    assert retained.is_dir()
    shutil.rmtree(root)
    shutil.rmtree(retained)


@pytest.mark.asyncio
async def test_inspect_cleanup_preserves_unknown_injected_data(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_validate = runner_module.validate_frozen_snapshot
    roots: list[Path] = []

    async def inject_unknown_after_validation(
        snapshot: runner_module.FrozenSqliteSnapshot,
        *,
        validation_path: Path,
        frozen_at: datetime,
        seed: int,
    ) -> str:
        revision = await original_validate(
            snapshot,
            validation_path=validation_path,
            frozen_at=frozen_at,
            seed=seed,
        )
        root = validation_path.parents[1]
        (root / "keep.txt").write_text("unknown", encoding="utf-8")
        roots.append(root)
        return revision

    monkeypatch.setattr(
        runner_module,
        "validate_frozen_snapshot",
        inject_unknown_after_validation,
    )

    with pytest.raises(ExperimentIsolationError) as captured:
        await inspect_replay(manifest_path, source)

    assert captured.value.code == "replay_inspect_cleanup_unsafe"
    assert captured.value.__cause__ is None
    assert str(roots[0]) not in str(captured.value)
    assert (roots[0] / "keep.txt").read_text(encoding="utf-8") == "unknown"
    shutil.rmtree(roots[0])


@pytest.mark.asyncio
async def test_inspect_cleanup_failure_does_not_override_validation_failure(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    original_validate = runner_module.validate_frozen_snapshot
    roots: list[Path] = []

    async def fail_after_unknown_injection(
        snapshot: runner_module.FrozenSqliteSnapshot,
        *,
        validation_path: Path,
        frozen_at: datetime,
        seed: int,
    ) -> str:
        await original_validate(
            snapshot,
            validation_path=validation_path,
            frozen_at=frozen_at,
            seed=seed,
        )
        root = validation_path.parents[1]
        (root / "keep.txt").write_text("recover", encoding="utf-8")
        roots.append(root)
        raise ExperimentIsolationError(
            "replay_projection_mismatch",
            "Replay source projections do not match authoritative replay",
        )

    monkeypatch.setattr(
        runner_module,
        "validate_frozen_snapshot",
        fail_after_unknown_injection,
    )

    with pytest.raises(ExperimentIsolationError) as captured:
        await inspect_replay(manifest_path, source)

    assert captured.value.code == "replay_projection_mismatch"
    assert (roots[0] / "keep.txt").read_text(encoding="utf-8") == "recover"
    shutil.rmtree(roots[0])


@pytest.mark.asyncio
async def test_inspect_cleanup_maps_oserror_without_private_detail(
    replay_fixture: tuple[Path, Path, tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source, _ = replay_fixture
    roots: list[Path] = []

    def fail_root_removal(workspace: OwnedWorkspace) -> None:
        roots.append(workspace.root)
        raise OSError("/private/inspect-cleanup")

    monkeypatch.setattr(
        runner_module,
        "_remove_empty_inspection_root",
        fail_root_removal,
    )

    with pytest.raises(ExperimentIsolationError) as captured:
        await inspect_replay(manifest_path, source)

    assert captured.value.code == "replay_inspect_cleanup_unsafe"
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True
    assert "/private/inspect-cleanup" not in str(captured.value)
    assert roots[0].is_dir()
    assert (
        roots[0] / runner_module.REPLAY_WORKSPACE_POLICY.marker_name
    ).is_file()
    shutil.rmtree(roots[0])


def test_clone_hashing_is_incremental_without_materializing_joined_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = tmp_path / "clone.sqlite3"
    body = b"x" * (2 * 1024 * 1024 + 137)
    clone.write_bytes(body)
    read_sizes: list[int] = []
    original_read = runner_module.os.read

    def bounded_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(descriptor, size)

    def forbidden_materialized_hash(body: bytes) -> str:
        del body
        raise AssertionError("clone hashing materialized the complete database")

    monkeypatch.setattr(runner_module.os, "read", bounded_read)
    monkeypatch.setattr(runner_module, "sha256_hex", forbidden_materialized_hash)

    _, digest = runner_module._clone_identity_and_hash(
        clone,
        expected_size=len(body),
    )

    assert digest == hashlib.sha256(body).hexdigest()
    assert read_sizes
    assert max(read_sizes) <= 1024 * 1024
