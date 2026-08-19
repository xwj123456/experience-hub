from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from experience_hub.benchmark import InspirationCase
from experience_hub.benchmark import runner as benchmark_runner
from experience_hub.benchmark.runner import (
    BenchmarkIsolationError,
    clone_closed_database,
    prepare_benchmark_snapshot,
    run_benchmark,
)
from experience_hub.experiments.snapshots import (
    FrozenSqliteSnapshot,
    checkpoint_owned_sqlite,
    clone_frozen_sqlite,
)
from experience_hub.experiments.workspace import (
    WorkspacePolicy,
    prepare_owned_workspace,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
SEED_PATH = REPOSITORY_ROOT / "benchmarks" / "seed.json"
CASES_PATH = REPOSITORY_ROOT / "benchmarks" / "cases.jsonl"


@pytest.mark.asyncio
async def test_seed_snapshot_is_closed_checkpointed_and_cloned_by_exact_bytes(
    tmp_path: Path,
) -> None:
    snapshot = await prepare_benchmark_snapshot(
        seed_path=SEED_PATH,
        cases_path=CASES_PATH,
        workspace=tmp_path / "benchmark",
    )

    source_bytes = snapshot.database_path.read_bytes()
    assert benchmark_runner._WORKSPACE_MARKER == (
        "experience-hub deterministic benchmark workspace\n"
    )
    assert isinstance(snapshot.frozen_database, FrozenSqliteSnapshot)
    assert snapshot.checkpoint_result == (0, 0, 0)
    assert snapshot.database_bytes == source_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == snapshot.database_sha256
    assert snapshot.database_path.is_file()
    assert not Path(f"{snapshot.database_path}-wal").exists()
    assert not Path(f"{snapshot.database_path}-shm").exists()

    first = tmp_path / "clones" / "case.sqlite3"
    baseline = tmp_path / "clones" / "baseline.sqlite3"
    clone_closed_database(snapshot, first)
    clone_closed_database(snapshot, baseline)

    assert first.read_bytes() == baseline.read_bytes() == source_bytes
    with closing(sqlite3.connect(first)) as connection:
        connection.execute("CREATE TABLE clone_only (value INTEGER)")
        connection.commit()

    with closing(sqlite3.connect(baseline)) as connection:
        clone_only = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'clone_only'"
        ).fetchone()
    assert clone_only is None
    assert snapshot.database_path.read_bytes() == source_bytes


@pytest.mark.asyncio
async def test_snapshot_preparation_delegates_owned_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpointed: list[Path] = []

    def recording_checkpoint(path: Path) -> tuple[int, int, int]:
        checkpointed.append(path)
        return checkpoint_owned_sqlite(path)

    monkeypatch.setattr(
        benchmark_runner,
        "checkpoint_owned_sqlite",
        recording_checkpoint,
        raising=False,
    )

    snapshot = await prepare_benchmark_snapshot(
        seed_path=SEED_PATH,
        cases_path=CASES_PATH,
        workspace=tmp_path / "benchmark",
    )

    assert checkpointed == [snapshot.database_path]
    assert snapshot.checkpoint_result == (0, 0, 0)


@pytest.mark.asyncio
async def test_clone_delegates_frozen_sqlite_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = await prepare_benchmark_snapshot(
        seed_path=SEED_PATH,
        cases_path=CASES_PATH,
        workspace=tmp_path / "benchmark",
    )
    cloned: list[tuple[FrozenSqliteSnapshot, Path]] = []

    def recording_clone(
        frozen: FrozenSqliteSnapshot,
        destination: Path,
    ) -> Path:
        cloned.append((frozen, destination))
        return clone_frozen_sqlite(frozen, destination)

    monkeypatch.setattr(
        benchmark_runner,
        "clone_frozen_sqlite",
        recording_clone,
        raising=False,
    )
    destination = tmp_path / "clone" / "case.sqlite3"

    result = clone_closed_database(snapshot, destination)

    assert cloned == [(snapshot.frozen_database, destination)]
    assert result == destination
    assert result.read_bytes() == snapshot.database_bytes


def test_workspace_reset_delegates_legacy_owned_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared: list[tuple[Path, WorkspacePolicy, bool, bool]] = []

    def recording_prepare(
        path: Path,
        *,
        policy: WorkspacePolicy,
        replace_owned: bool,
        allow_unmarked_empty: bool,
    ) -> object:
        prepared.append(
            (path, policy, replace_owned, allow_unmarked_empty)
        )
        return prepare_owned_workspace(
            path,
            policy=policy,
            replace_owned=replace_owned,
            allow_unmarked_empty=allow_unmarked_empty,
        )

    monkeypatch.setattr(
        benchmark_runner,
        "prepare_owned_workspace",
        recording_prepare,
        raising=False,
    )
    default_workspace = tmp_path / "default"
    custom_workspace = tmp_path / "custom"

    benchmark_runner._reset_owned_workspace(
        default_workspace,
        allow_unmarked=True,
    )
    benchmark_runner._reset_owned_workspace(
        custom_workspace,
        allow_unmarked=False,
    )

    expected_policy = WorkspacePolicy(
        marker_name=".experience-hub-benchmark-workspace",
        marker_body=b"experience-hub deterministic benchmark workspace\n",
        owned_entries=frozenset({"replay-a", "replay-b", "snapshot"}),
    )
    assert prepared == [
        (default_workspace, expected_policy, True, True),
        (custom_workspace, expected_policy, True, True),
    ]
    assert (
        default_workspace / ".experience-hub-benchmark-workspace"
    ).read_bytes() == expected_policy.marker_body
    assert (
        custom_workspace / ".experience-hub-benchmark-workspace"
    ).read_bytes() == expected_policy.marker_body


def test_existing_empty_custom_workspace_is_adopted_without_allow_unmarked(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "custom"
    workspace.mkdir()

    benchmark_runner._reset_owned_workspace(
        workspace,
        allow_unmarked=False,
    )

    assert tuple(entry.name for entry in workspace.iterdir()) == (
        ".experience-hub-benchmark-workspace",
    )
    assert (
        workspace / ".experience-hub-benchmark-workspace"
    ).read_bytes() == b"experience-hub deterministic benchmark workspace\n"


def test_default_workspace_safely_adopts_only_legacy_owned_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    retained = workspace / "replay-a" / "nested" / "retained-before-reset.txt"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"old benchmark output")
    (workspace / "replay-b").mkdir()
    (workspace / "snapshot").mkdir()
    prepare_count = 0

    def recording_prepare(
        path: Path,
        *,
        policy: WorkspacePolicy,
        replace_owned: bool,
        allow_unmarked_empty: bool,
    ) -> object:
        nonlocal prepare_count
        prepare_count += 1
        return prepare_owned_workspace(
            path,
            policy=policy,
            replace_owned=replace_owned,
            allow_unmarked_empty=allow_unmarked_empty,
        )

    monkeypatch.setattr(
        benchmark_runner,
        "prepare_owned_workspace",
        recording_prepare,
    )

    benchmark_runner._reset_owned_workspace(
        workspace,
        allow_unmarked=True,
    )

    assert prepare_count == 1
    assert tuple(entry.name for entry in workspace.iterdir()) == (
        ".experience-hub-benchmark-workspace",
    )
    assert (
        workspace / ".experience-hub-benchmark-workspace"
    ).read_bytes() == b"experience-hub deterministic benchmark workspace\n"


def test_unmarked_default_workspace_with_unknown_entry_is_preserved(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    workspace.mkdir()
    retained = workspace / "unknown.txt"
    retained.write_bytes(b"user data")

    with pytest.raises(BenchmarkIsolationError, match="not owned"):
        benchmark_runner._reset_owned_workspace(
            workspace,
            allow_unmarked=True,
        )

    assert retained.read_bytes() == b"user data"
    assert not (
        workspace / ".experience-hub-benchmark-workspace"
    ).exists()


def test_unmarked_custom_workspace_with_owned_entry_is_preserved(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "custom"
    retained = workspace / "replay-a" / "retained.txt"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"old benchmark output")

    with pytest.raises(BenchmarkIsolationError, match="not owned"):
        benchmark_runner._reset_owned_workspace(
            workspace,
            allow_unmarked=False,
        )

    assert retained.read_bytes() == b"old benchmark output"
    assert not (
        workspace / ".experience-hub-benchmark-workspace"
    ).exists()


@pytest.mark.asyncio
async def test_clone_refuses_mutated_snapshot_or_nonempty_wal(
    tmp_path: Path,
) -> None:
    snapshot = await prepare_benchmark_snapshot(
        seed_path=SEED_PATH,
        cases_path=CASES_PATH,
        workspace=tmp_path / "benchmark",
    )
    wal_path = Path(f"{snapshot.database_path}-wal")
    wal_path.write_bytes(b"uncheckpointed")

    with pytest.raises(BenchmarkIsolationError, match="WAL"):
        clone_closed_database(snapshot, tmp_path / "rejected.sqlite3")

    wal_path.unlink()
    snapshot.database_path.write_bytes(snapshot.database_path.read_bytes() + b"x")
    with pytest.raises(BenchmarkIsolationError, match="immutable"):
        clone_closed_database(snapshot, tmp_path / "rejected.sqlite3")


@pytest.mark.asyncio
async def test_runner_never_deletes_an_unowned_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "user-directory"
    workspace.mkdir()
    user_file = workspace / "keep-me.txt"
    user_file.write_text("user data", encoding="utf-8")

    with pytest.raises(BenchmarkIsolationError, match="not owned"):
        await run_benchmark(
            seed_path=SEED_PATH,
            cases_path=CASES_PATH,
            workspace=workspace,
        )

    assert user_file.read_text(encoding="utf-8") == "user data"


@pytest.mark.asyncio
async def test_missing_same_snapshot_recurrence_becomes_a_canonical_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = await prepare_benchmark_snapshot(
        seed_path=SEED_PATH,
        cases_path=CASES_PATH,
        workspace=tmp_path / "benchmark",
    )
    case = next(
        case for case in snapshot.cases if isinstance(case, InspirationCase)
    )
    original_execute = benchmark_runner._execute_inspiration_run
    retained_first_run: Any = None
    invocation_count = 0

    async def no_second_run_ideas(
        container: Any,
        *,
        command: Any,
        key: str,
    ) -> Any:
        nonlocal invocation_count, retained_first_run
        invocation_count += 1
        if invocation_count == 1:
            result = await original_execute(
                container,
                command=command,
                key=key,
            )
            retained_first_run = result[0]
            return result
        assert retained_first_run is not None
        return (
            retained_first_run.model_copy(
                update={"run_id": UUID(int=2_000_000)}
            ),
            (),
        )

    monkeypatch.setattr(
        benchmark_runner,
        "_execute_inspiration_run",
        no_second_run_ideas,
    )
    incubation, violation_count = (
        await benchmark_runner._run_same_snapshot_incubation(
            snapshot,
            case=case,
            destination=tmp_path / "same-snapshot.sqlite3",
        )
    )

    assert incubation["occurrence_advanced"] is False
    assert violation_count > 0

    suite = benchmark_runner._SuiteResult(
        cases=(),
        same_snapshot_incubation=incubation,
        focused_recall=Decimal("0.90"),
        cold_recall=Decimal("0.85"),
        cold_baseline_recall=Decimal("0.60"),
        distractor_false_reactivations=0,
        pending_leakage_count=0,
        adopted_provenance_complete=1,
        adopted_provenance_total=1,
        persisted_idea_count=12,
        valid_idea_count=12,
        distinct_mechanism_count=9,
        same_snapshot_incubation_promotions=violation_count,
        inspiration_evidence_coverage_failures=0,
    )
    report = benchmark_runner._report_document(
        suite,
        byte_identical_replay=True,
    )
    body = benchmark_runner.canonical_benchmark_bytes(report)
    data = cast(dict[str, Any], json.loads(body)["data"])

    assert data["passed"] is False
    assert data["failed_gates"] == ["same_snapshot_incubation_promotion"]
