from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments import benchmarks as benchmark_package
from experience_hub.experiments.benchmarks import (
    BENCHMARK_ARM_ORDER,
    MAX_BENCHMARK_OUTPUT_BYTES,
    BenchmarkArmEvidenceV1,
    BenchmarkArmObservationV1,
    BenchmarkCaseEvidenceV1,
    BenchmarkSafetyEvidenceV1,
    BenchmarkSourceCandidateV1,
)
from experience_hub.experiments.benchmarks import runner as runner_module
from experience_hub.experiments.benchmarks.loading import load_benchmark_pack
from experience_hub.experiments.benchmarks.metrics import derive_case_comparison
from experience_hub.experiments.benchmarks.oracles import score_benchmark_observation
from experience_hub.experiments.benchmarks.runner import (
    BenchmarkExecution,
    BenchmarkInspection,
    inspect_benchmark_pack,
    run_benchmark_pilot,
    verify_benchmark_report,
)
from experience_hub.experiments.benchmarks.source import build_benchmark_source
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.reports import ExperimentOutputError
from experience_hub.experiments.snapshots import (
    FrozenSqliteSnapshot,
    verify_source_unchanged,
)
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    OwnedWorkspace,
    prepare_owned_workspace,
)

FROZEN_AT = datetime(2026, 8, 20, tzinfo=UTC)


def _case_document(ordinal: int) -> dict[str, object]:
    source_class = "public_authored" if ordinal < 20 else "reviewed_abstraction"
    return {
        "schema_version": 1,
        "case_id": f"case-{ordinal + 1}",
        "source_class": source_class,
        "review_status": (
            "authored" if source_class == "public_authored" else "maintainer_reviewed"
        ),
        "stratum": (
            "recurring_workflow",
            "environment_gotcha",
            "state_change",
            "failure_recovery",
            "irrelevant_distractor",
        )[ordinal % 5],
        "language": ("zh", "en", "mixed")[ordinal // 10],
        "difficulty": "A",
        "owner_label": "alpha",
        "query": "bounded workflow",
        "mode": "focused",
        "tags": ["workflow"],
        "mechanism_cues": ["bounded"],
        "limit": 1,
        "content_budget_bytes": 1024,
        "source_labels": [
            "active-note",
            "forbidden-note",
            "stale-note",
            "misleading-note",
        ],
        "required": [{"label": "active-note", "weight_micros": 450_000}],
        "optional": [],
        "forbidden": [{"label": "forbidden-note", "weight_micros": 100_000}],
        "stale": [{"label": "stale-note", "weight_micros": 100_000}],
        "misleading": [{"label": "misleading-note", "weight_micros": 100_000}],
        "checkpoints": [
            {
                "predicate": "required_set",
                "labels": ["active-note"],
                "weight_micros": 150_000,
            }
        ],
        "oracle_version": 1,
    }


def _source_documents() -> tuple[dict[str, object], ...]:
    def experience(
        *,
        label: str,
        owner: str,
        created_at: datetime,
        summary: str,
        mechanism: str,
        tags: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_type": "experience",
            "label": label,
            "owner_label": owner,
            "created_at": created_at,
            "temperature": "warm",
            "kind": "procedural",
            "body": summary,
            "summary": summary,
            "mechanism": mechanism,
            "tags": list(tags),
            "applicability": ["active operations"],
            "evidence": [],
            "falsifiers": ["The workflow has no duplicate risk."],
            "importance_micros": 800_000,
            "confidence_micros": 900_000,
        }

    return (
        {"schema_version": 1, "record_type": "agent", "label": "alpha"},
        {"schema_version": 1, "record_type": "agent", "label": "beta"},
        experience(
            label="active-note",
            owner="alpha",
            created_at=FROZEN_AT - timedelta(days=2),
            summary="Keep the workflow bounded.",
            mechanism="A bounded workflow prevents duplicate work.",
            tags=("workflow",),
        ),
        experience(
            label="forbidden-note",
            owner="alpha",
            created_at=FROZEN_AT - timedelta(hours=20),
            summary="Forbidden workflow.",
            mechanism="Forbidden behavior is excluded.",
            tags=("forbidden",),
        ),
        experience(
            label="stale-note",
            owner="alpha",
            created_at=FROZEN_AT - timedelta(hours=16),
            summary="Stale workflow.",
            mechanism="Stale behavior is excluded.",
            tags=("stale",),
        ),
        experience(
            label="misleading-note",
            owner="alpha",
            created_at=FROZEN_AT - timedelta(hours=12),
            summary="Misleading workflow.",
            mechanism="Misleading behavior is excluded.",
            tags=("misleading",),
        ),
        experience(
            label="foreign-note",
            owner="beta",
            created_at=FROZEN_AT - timedelta(hours=10),
            summary="Foreign bounded workflow.",
            mechanism="Owner isolation excludes this record.",
            tags=("workflow",),
        ),
        {
            "schema_version": 1,
            "record_type": "candidate",
            "label": "pending-note",
            "owner_label": "alpha",
            "created_at": FROZEN_AT - timedelta(hours=8),
            "kind": "procedural",
            "body": "Keep this pending candidate quarantined.",
            "summary": "Pending candidate stays quarantined.",
            "mechanism": "Capture requires explicit adoption.",
            "tags": ["pending"],
            "applicability": ["candidate review"],
            "falsifiers": ["The candidate was adopted."],
        },
    )


def _write_pack(root: Path) -> Path:
    root.mkdir(parents=True)
    cases_body = b"".join(
        canonical_json_bytes(_case_document(ordinal)) + b"\n" for ordinal in range(30)
    )
    source_body = b"".join(
        canonical_json_bytes(document) + b"\n" for document in _source_documents()
    )
    (root / "cases.jsonl").write_bytes(cases_body)
    (root / "source.jsonl").write_bytes(source_body)
    manifest = {
        "schema_version": 1,
        "pack_id": "experiencebench-s-pilot",
        "maturity": "pilot-30",
        "cases": {"file": "cases.jsonl", "sha256": sha256_hex(cases_body)},
        "source": {"file": "source.jsonl", "sha256": sha256_hex(source_body)},
        "frozen_at": FROZEN_AT,
        "seed": 20260820,
        "arms": [
            {"schema_version": 1, "arm_id": arm, "kind": arm, "required": True}
            for arm in (
                "no_memory",
                "recent_notes",
                "sqlite_bm25",
                "experience_hub",
            )
        ],
        "oracle_version": 1,
        "metric_version": 1,
        "gate_version": 1,
        "evidence_schema_version": 1,
        "summary_schema_version": 1,
        "profile_schema_version": 1,
        "deterministic_replay_runs": 2,
        "composition": {
            "case_count": 30,
            "public_authored": 20,
            "reviewed_abstractions": 10,
            "cases_per_stratum": 6,
            "chinese": 10,
            "english": 10,
            "mixed": 10,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.asyncio
async def test_inspection_validates_pack_without_mutating_its_tree(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    manifest_path = _write_pack(pack_root)
    before = _tree_bytes(pack_root)

    inspection = await inspect_benchmark_pack(manifest_path)

    assert inspection == BenchmarkInspection(
        pack_id="experiencebench-s-pilot",
        case_count=30,
        arm_count=4,
        manifest_sha256=sha256_hex(before["manifest.json"]),
        cases_sha256=sha256_hex(before["cases.jsonl"]),
        source_fixture_sha256=sha256_hex(before["source.jsonl"]),
        fts5_available=True,
    )
    assert _tree_bytes(pack_root) == before


def test_benchmark_package_exports_runner_safety_contract() -> None:
    assert BenchmarkSafetyEvidenceV1.__name__ == "BenchmarkSafetyEvidenceV1"
    assert benchmark_package.BenchmarkExecution is BenchmarkExecution
    assert benchmark_package.BenchmarkInspection is BenchmarkInspection
    assert benchmark_package.inspect_benchmark_pack is inspect_benchmark_pack
    assert benchmark_package.run_benchmark_pilot is run_benchmark_pilot
    assert benchmark_package.verify_benchmark_report is verify_benchmark_report
    assert {
        "BenchmarkExecution",
        "BenchmarkInspection",
        "BenchmarkSafetyEvidenceV1",
        "inspect_benchmark_pack",
        "run_benchmark_pilot",
        "verify_benchmark_report",
    }.issubset(benchmark_package.__all__)


async def _built_source(
    manifest_path: Path,
    workspace_path: Path,
) -> tuple[object, OwnedWorkspace, object]:
    pack = load_benchmark_pack(manifest_path)
    workspace = prepare_owned_workspace(
        workspace_path,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    built = await build_benchmark_source(pack, workspace)
    return pack, workspace, built


@pytest.mark.asyncio
async def test_private_pass_seam_uses_two_cases_four_arms_and_fresh_clones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack_value, workspace, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    pack = cast(runner_module.LoadedBenchmarkPack, pack_value)
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    clone_starts: list[tuple[Path, str, tuple[int, int]]] = []
    clone_failures: list[tuple[str, str | None]] = []
    original_clone = runner_module.clone_frozen_sqlite

    def recording_clone(snapshot: object, destination: Path) -> Path:
        try:
            clone = original_clone(
                cast(FrozenSqliteSnapshot, snapshot), destination
            )
        except Exception as error:
            clone_failures.append((type(error).__name__, getattr(error, "code", None)))
            raise
        status = clone.stat()
        clone_starts.append(
            (clone, sha256_hex(clone.read_bytes()), (status.st_dev, status.st_ino))
        )
        return clone

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", recording_clone)
    identities: set[tuple[int, int]] = set()

    first = await runner_module._execute_benchmark_cases(
        pass_name="pass-a",
        cases=pack.cases[:2],
        descriptors=pack.manifest.arms,
        workspace=workspace,
        source=built,
        source_records=pack.source,
        seen_identities=identities,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )
    second = await runner_module._execute_benchmark_cases(
        pass_name="pass-b",
        cases=pack.cases[:2],
        descriptors=pack.manifest.arms,
        workspace=workspace,
        source=built,
        source_records=pack.source,
        seen_identities=identities,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    assert len(first.cases) == len(second.cases) == 2
    assert clone_failures == []
    assert len(clone_starts) == 16
    assert [path.parts[-4:] for path, _, _ in clone_starts] == [
        ("arms", pass_name, case_id, f"{arm.value}.sqlite3")
        for pass_name in ("pass-a", "pass-b")
        for case_id in ("case-1", "case-2")
        for arm in BENCHMARK_ARM_ORDER
    ]
    assert {digest for _, digest, _ in clone_starts} == {built.snapshot.database_sha256}
    clone_identities = [identity for _, _, identity in clone_starts]
    assert len(set(clone_identities)) == len(clone_identities)
    source_status = built.path.stat()
    assert (source_status.st_dev, source_status.st_ino) not in clone_identities
    verify_source_unchanged(built.snapshot)


@pytest.mark.asyncio
async def test_clone_normalization_never_writes_through_a_racing_public_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack_value, workspace, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    pack = cast(runner_module.LoadedBenchmarkPack, pack_value)
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    clone_path = (
        workspace.root / "arms" / "pass-a" / "case-1" / "no_memory.sqlite3"
    )
    parked_path = clone_path.with_name("parked.sqlite3")
    source_status = built.path.stat()
    source_before = (
        built.path.read_bytes(),
        built.path.read_bytes()[18:20],
        (source_status.st_dev, source_status.st_ino),
        tuple(
            (
                suffix,
                candidate.read_bytes() if candidate.is_file() else None,
            )
            for suffix in ("-wal", "-shm", "-journal")
            if (candidate := Path(f"{built.path}{suffix}")).exists()
        ),
    )
    original_connect = sqlite3.connect
    original_open = runner_module.os.open
    sqlite_raced = False
    descriptor_raced = False

    def install_racing_symlink() -> None:
        clone_path.replace(parked_path)
        clone_path.symlink_to(built.path)

    def restore_clone_path() -> None:
        clone_path.unlink()
        parked_path.replace(clone_path)

    def racing_connect(database: object, *args: object, **kwargs: object) -> object:
        nonlocal sqlite_raced
        if Path(cast(str | Path, database)) != clone_path:
            return original_connect(database, *args, **kwargs)
        sqlite_raced = True
        install_racing_symlink()
        try:
            connection = original_connect(database, *args, **kwargs)
        finally:
            restore_clone_path()
        return connection

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal descriptor_raced
        if (
            dir_fd is not None
            or Path(cast(str | Path, path)) != clone_path
            or flags & runner_module.os.O_ACCMODE != runner_module.os.O_RDWR
        ):
            return original_open(path, flags, mode, dir_fd=dir_fd)
        descriptor_raced = True
        install_racing_symlink()
        try:
            return original_open(path, flags, mode)
        finally:
            restore_clone_path()

    monkeypatch.setattr(sqlite3, "connect", racing_connect)
    monkeypatch.setattr(runner_module.os, "open", racing_open)

    result = await runner_module._execute_benchmark_cases(
        pass_name="pass-a",
        cases=pack.cases[:1],
        descriptors=pack.manifest.arms,
        workspace=workspace,
        source=built,
        source_records=pack.source,
        seen_identities=set(),
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    final_status = built.path.stat()
    assert (
        built.path.read_bytes(),
        built.path.read_bytes()[18:20],
        (final_status.st_dev, final_status.st_ino),
        tuple(
            (
                suffix,
                candidate.read_bytes() if candidate.is_file() else None,
            )
            for suffix in ("-wal", "-shm", "-journal")
            if (candidate := Path(f"{built.path}{suffix}")).exists()
        ),
    ) == source_before
    verify_source_unchanged(built.snapshot)
    assert sqlite_raced is False
    assert descriptor_raced is True
    arm = result.cases[0].arms[0]
    assert arm.status == "failed"
    assert arm.error_code == "benchmark_safety_failure"
    assert arm.error_stage == "clone"


@pytest.mark.asyncio
async def test_clone_normalization_rejects_same_inode_mutation_after_final_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    _, workspace, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    clone_path = workspace.root / "race" / "clone.sqlite3"
    clone = runner_module.clone_frozen_sqlite(built.snapshot, clone_path)
    clone_status = clone.stat()
    clone_identity = (clone_status.st_dev, clone_status.st_ino)
    original_read = runner_module.os.read
    original_pwrite = runner_module.os.pwrite
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        status = runner_module.os.fstat(descriptor)
        if (
            not mutated
            and chunk
            and (status.st_dev, status.st_ino) == clone_identity
            and runner_module.os.pread(descriptor, 2, 18) == b"\x01\x01"
            and runner_module.os.lseek(descriptor, 0, runner_module.os.SEEK_CUR)
            == clone_status.st_size
        ):
            mutated = True
            final_offset = clone_status.st_size - 1
            current = runner_module.os.pread(descriptor, 1, final_offset)
            assert len(current) == 1
            assert original_pwrite(
                descriptor,
                bytes((current[0] ^ 1,)),
                final_offset,
            ) == 1
            runner_module.os.fsync(descriptor)
            runner_module.os.utime(
                descriptor,
                ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000),
            )
        return chunk

    monkeypatch.setattr(runner_module.os, "read", mutating_read)

    with pytest.raises(ExperimentInputError) as raised:
        runner_module._prepare_clone_for_descriptor_policy(
            clone,
            expected_identity=clone_identity,
            expected_size=clone_status.st_size,
            expected_sha256=built.snapshot.database_sha256,
        )

    assert raised.value.code == "benchmark_safety_failure"
    assert mutated is True
    final_status = clone.stat()
    assert (final_status.st_dev, final_status.st_ino) == clone_identity
    assert final_status.st_size == clone_status.st_size
    normalized = bytearray(built.snapshot.database_bytes)
    normalized[18:20] = b"\x01\x01"
    assert clone.read_bytes() != bytes(normalized)


@pytest.mark.asyncio
async def test_clone_normalization_pwrite_retains_clone_authority_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    _, workspace, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    clone_path = workspace.root / "race" / "clone.sqlite3"
    clone = runner_module.clone_frozen_sqlite(built.snapshot, clone_path)
    clone_status = clone.stat()
    clone_identity = (clone_status.st_dev, clone_status.st_ino)
    parked_path = clone.with_name("parked.sqlite3")
    source_status = built.path.stat()
    source_before = (
        built.path.read_bytes(),
        built.path.read_bytes()[18:20],
        (source_status.st_dev, source_status.st_ino),
        tuple(
            (
                suffix,
                candidate.read_bytes() if candidate.is_file() else None,
            )
            for suffix in ("-wal", "-shm", "-journal")
            if (candidate := Path(f"{built.path}{suffix}")).exists()
        ),
    )
    original_pwrite = runner_module.os.pwrite
    pwrite_raced = False
    retained_header: bytes | None = None

    def racing_pwrite(descriptor: int, data: bytes, offset: int) -> int:
        nonlocal pwrite_raced, retained_header
        status = runner_module.os.fstat(descriptor)
        if (
            pwrite_raced
            or (status.st_dev, status.st_ino) != clone_identity
            or data != b"\x01\x01"
            or offset != 18
        ):
            return original_pwrite(descriptor, data, offset)
        pwrite_raced = True
        clone.replace(parked_path)
        clone.symlink_to(built.path)
        written = original_pwrite(descriptor, data, offset)
        retained_header = runner_module.os.pread(descriptor, 2, 18)
        return written

    monkeypatch.setattr(runner_module.os, "pwrite", racing_pwrite)

    try:
        with pytest.raises(ExperimentInputError) as raised:
            runner_module._prepare_clone_for_descriptor_policy(
                clone,
                expected_identity=clone_identity,
                expected_size=clone_status.st_size,
                expected_sha256=built.snapshot.database_sha256,
            )

        assert raised.value.code == "benchmark_safety_failure"
        assert pwrite_raced is True
        assert retained_header == b"\x01\x01"
        assert parked_path.read_bytes()[18:20] == b"\x01\x01"
        assert clone.is_symlink()
        final_source_status = built.path.stat()
        assert (
            built.path.read_bytes(),
            built.path.read_bytes()[18:20],
            (final_source_status.st_dev, final_source_status.st_ino),
            tuple(
                (
                    suffix,
                    candidate.read_bytes() if candidate.is_file() else None,
                )
                for suffix in ("-wal", "-shm", "-journal")
                if (candidate := Path(f"{built.path}{suffix}")).exists()
            ),
        ) == source_before
        verify_source_unchanged(built.snapshot)
    finally:
        if clone.is_symlink():
            clone.unlink()
        if parked_path.exists():
            parked_path.replace(clone)


class _ObservationPolicy:
    def __init__(self, returned_labels: tuple[str, ...]) -> None:
        self._returned_labels = returned_labels

    async def execute(self, context: object) -> BenchmarkArmObservationV1:
        del context
        return BenchmarkArmObservationV1(
            schema_version=1,
            returned_labels=self._returned_labels,
            selected_content_bytes=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned_labels", "error_code", "stage", "owner_leaks", "quarantine_leaks"),
    (
        (("foreign-note",), "benchmark_safety_failure", "safety", 1, 0),
        (("pending-note",), "benchmark_safety_failure", "safety", 0, 1),
        (("unknown-note",), "benchmark_oracle_invalid", "oracle", 0, 0),
        (("active-note",), "benchmark_oracle_invalid", "oracle", 0, 0),
    ),
)
async def test_arm_failures_map_owner_quarantine_and_unknown_labels_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returned_labels: tuple[str, ...],
    error_code: str,
    stage: str,
    owner_leaks: int,
    quarantine_leaks: int,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack_value, _, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    pack = cast(runner_module.LoadedBenchmarkPack, pack_value)
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    monkeypatch.setattr(
        runner_module,
        "build_benchmark_policy",
        lambda descriptor: _ObservationPolicy(returned_labels),
    )

    result = await runner_module._execute_benchmark_arm(
        descriptor=pack.manifest.arms[0],
        case=pack.cases[0],
        clone_path=built.path,
        source_index=built.index,
        source_records=pack.source,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    assert result.evidence.status == "failed"
    assert result.evidence.error_code == error_code
    assert result.evidence.error_stage == stage
    assert result.owner_leak_count == owner_leaks
    assert result.quarantine_leak_count == quarantine_leaks


@pytest.mark.asyncio
async def test_foreign_candidate_increments_both_named_safety_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack_value, _, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    pack = cast(runner_module.LoadedBenchmarkPack, pack_value)
    built = cast(runner_module.BuiltBenchmarkSource, built_value)
    source_records = tuple(
        record.model_copy(update={"owner_label": "beta"})
        if isinstance(record, BenchmarkSourceCandidateV1)
        else record
        for record in pack.source
    )
    monkeypatch.setattr(
        runner_module,
        "build_benchmark_policy",
        lambda descriptor: _ObservationPolicy(("pending-note",)),
    )

    result = await runner_module._execute_benchmark_arm(
        descriptor=pack.manifest.arms[0],
        case=pack.cases[0],
        clone_path=built.path,
        source_index=built.index,
        source_records=source_records,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    assert result.evidence.error_code == "benchmark_safety_failure"
    assert result.evidence.error_stage == "safety"
    assert result.owner_leak_count == 1
    assert result.quarantine_leak_count == 1


@pytest.mark.asyncio
async def test_required_arm_execution_failure_is_retained_without_private_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack_value, _, built_value = await _built_source(
        manifest_path, tmp_path / "workspace"
    )
    pack = cast(runner_module.LoadedBenchmarkPack, pack_value)
    built = cast(runner_module.BuiltBenchmarkSource, built_value)

    class FailedPolicy:
        async def execute(self, context: object) -> BenchmarkArmObservationV1:
            del context
            raise RuntimeError("sensitive-detail-should-never-be-persisted")

    monkeypatch.setattr(
        runner_module, "build_benchmark_policy", lambda descriptor: FailedPolicy()
    )

    result = await runner_module._execute_benchmark_arm(
        descriptor=pack.manifest.arms[0],
        case=pack.cases[0],
        clone_path=built.path,
        source_index=built.index,
        source_records=pack.source,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    assert result.evidence.error_code == "benchmark_arm_incomplete"
    assert result.evidence.error_stage == "execute"
    assert "sensitive-detail" not in repr(result.evidence)


def test_cross_arm_contamination_counts_each_reused_identity_once() -> None:
    first = runner_module._PassResult(
        cases=(),
        clone_identities=((7, 1),),
        owner_leak_count=0,
        quarantine_leak_count=0,
        cross_arm_contamination_count=0,
        clone_isolation_verified=True,
    )
    second = runner_module._PassResult(
        cases=(),
        clone_identities=((7, 1),),
        owner_leak_count=0,
        quarantine_leak_count=0,
        cross_arm_contamination_count=1,
        clone_isolation_verified=False,
    )

    safety = runner_module._combined_safety(
        first,
        second,
        expected_clone_count=2,
        source_unchanged=True,
    )

    assert safety.cross_arm_contamination_count == 1
    assert safety.clone_isolation_verified is False


def _case_evidence(
    case: runner_module.BenchmarkCaseV1,
    *,
    changed: bool = False,
    incomplete: bool = False,
) -> BenchmarkCaseEvidenceV1:
    arms: list[BenchmarkArmEvidenceV1] = []
    for kind in BENCHMARK_ARM_ORDER:
        if incomplete and kind.value == "recent_notes":
            arms.append(
                BenchmarkArmEvidenceV1(
                    schema_version=1,
                    arm_id=kind.value,
                    status="failed",
                    observation=None,
                    oracle=None,
                    error_code="benchmark_arm_incomplete",
                    error_stage="execute",
                )
            )
            continue
        returned = (
            ("active-note",) if changed and kind.value == "experience_hub" else ()
        )
        observation = BenchmarkArmObservationV1(
            schema_version=1,
            returned_labels=returned,
            selected_content_bytes=0,
        )
        arms.append(
            BenchmarkArmEvidenceV1(
                schema_version=1,
                arm_id=kind.value,
                status="complete",
                observation=observation,
                oracle=score_benchmark_observation(case, observation),
                error_code=None,
                error_stage=None,
            )
        )
    return derive_case_comparison(case, tuple(arms))


def _pass_result(
    pack: runner_module.LoadedBenchmarkPack,
    *,
    changed: bool = False,
    incomplete: bool = False,
    identity_offset: int = 0,
) -> runner_module._PassResult:
    return runner_module._PassResult(
        cases=tuple(
            _case_evidence(case, changed=changed, incomplete=incomplete)
            for case in pack.cases
        ),
        clone_identities=tuple(
            (7, identity_offset + ordinal) for ordinal in range(1, 121)
        ),
        owner_leak_count=0,
        quarantine_leak_count=0,
        cross_arm_contamination_count=0,
        clone_isolation_verified=True,
    )


@pytest.mark.asyncio
async def test_full_runner_publishes_two_real_thirty_case_four_arm_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack_root = tmp_path / "pack"
    manifest_path = _write_pack(pack_root)
    pack_before = _tree_bytes(pack_root)
    recorded_clones: list[tuple[Path, str, tuple[int, int]]] = []
    policy_calls: list[tuple[str, str, str, Path, tuple[int, int]]] = []
    original_clone = runner_module.clone_frozen_sqlite
    original_build_policy = runner_module.build_benchmark_policy

    def recording_clone(snapshot: object, destination: Path) -> Path:
        clone = original_clone(
            cast(FrozenSqliteSnapshot, snapshot), destination
        )
        status = clone.stat()
        if "arms" in clone.parts:
            recorded_clones.append(
                (clone, sha256_hex(clone.read_bytes()), (status.st_dev, status.st_ino))
            )
        return clone

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", recording_clone)

    def recording_build_policy(
        descriptor: runner_module.BenchmarkArmDescriptorV1,
    ) -> object:
        policy = original_build_policy(descriptor)

        class RecordingPolicy:
            async def execute(
                self,
                context: runner_module.BenchmarkPolicyContext,
            ) -> BenchmarkArmObservationV1:
                status = context.clone_path.stat()
                policy_calls.append(
                    (
                        context.clone_path.parts[-3],
                        context.case.case_id,
                        descriptor.arm_id,
                        context.clone_path,
                        (status.st_dev, status.st_ino),
                    )
                )
                return await policy.execute(context)

        return RecordingPolicy()

    monkeypatch.setattr(
        runner_module,
        "build_benchmark_policy",
        recording_build_policy,
    )
    clock = iter((100, 175))
    workspace = tmp_path / "pilot"

    execution = await run_benchmark_pilot(
        manifest_path,
        workspace,
        profiler=lambda: next(clock),
    )

    assert isinstance(execution, BenchmarkExecution)
    assert [
        (case.case_id, tuple((arm.arm_id, arm.error_code) for arm in case.arms))
        for case in execution.evidence.data.pass_payload.cases
        if case.status != "complete"
    ] == []
    assert execution.evidence_valid is True
    assert execution.comparison_complete is True
    assert execution.deterministic_replay_match is True
    assert execution.profile_complete is True
    assert execution.profile is not None
    assert execution.profile.data.wall_duration_ns == 75
    assert execution.profile.data.clone_count == 240
    assert (
        execution.evidence_body
        == (workspace / "artifacts" / "benchmark-evidence.json").read_bytes()
    )
    assert (
        execution.summary_body
        == (workspace / "artifacts" / "benchmark-summary.json").read_bytes()
    )
    assert (
        execution.profile_body
        == (workspace / "artifacts" / "profile.json").read_bytes()
    )
    artifacts = workspace / "artifacts"
    artifacts_before_verification = _tree_bytes(artifacts)
    assert (
        verify_benchmark_report(artifacts / "benchmark-evidence.json")
        == execution.evidence
    )
    assert _tree_bytes(artifacts) == artifacts_before_verification

    public_evidence = tmp_path / "experiencebench-s-pilot"
    public_evidence.mkdir()
    (public_evidence / "benchmark-evidence.json").write_bytes(
        execution.evidence_body
    )
    (public_evidence / "benchmark-summary.json").write_bytes(
        execution.summary_body
    )
    assert (
        verify_benchmark_report(public_evidence / "benchmark-evidence.json")
        == execution.evidence
    )

    unexpected = artifacts / "unexpected.txt"
    unexpected.write_text("retain", encoding="utf-8")
    with pytest.raises(ExperimentOutputError) as unknown_entry:
        verify_benchmark_report(artifacts / "benchmark-evidence.json")
    assert unknown_entry.value.code == "invalid_report_path"
    assert unexpected.read_text(encoding="utf-8") == "retain"
    assert len(recorded_clones) == 240
    assert [path.parts[-4:] for path, _, _ in recorded_clones] == [
        ("arms", pass_name, f"case-{case}", f"{arm.value}.sqlite3")
        for pass_name in ("pass-a", "pass-b")
        for case in range(1, 31)
        for arm in BENCHMARK_ARM_ORDER
    ]
    assert {digest for _, digest, _ in recorded_clones} == {
        execution.evidence.data.pass_payload.resolved_manifest.snapshot_sha256
    }
    assert len({identity for _, _, identity in recorded_clones}) == 240
    clone_identities = {path: identity for path, _, identity in recorded_clones}
    manifest_arm_ids = tuple(
        descriptor.arm_id
        for descriptor in load_benchmark_pack(manifest_path).manifest.arms
    )
    assert policy_calls == [
        (
            pass_name,
            f"case-{case}",
            arm_id,
            workspace
            / "arms"
            / pass_name
            / f"case-{case}"
            / f"{arm_id}.sqlite3",
            clone_identities[
                workspace
                / "arms"
                / pass_name
                / f"case-{case}"
                / f"{arm_id}.sqlite3"
            ],
        )
        for pass_name in ("pass-a", "pass-b")
        for case in range(1, 31)
        for arm_id in manifest_arm_ids
    ]
    assert len({path for _, _, _, path, _ in policy_calls}) == 240
    assert len({identity for _, _, _, _, identity in policy_calls}) == 240
    source_path = workspace / "snapshot" / "source.sqlite3"
    assert source_path.read_bytes()[18:20] == b"\x02\x02"
    source_status = source_path.stat()
    assert (source_status.st_dev, source_status.st_ino) not in {
        identity for _, _, identity in recorded_clones
    }
    assert not any(
        Path(f"{source_path}{suffix}").exists()
        for suffix in ("-wal", "-shm", "-journal")
    )
    assert _tree_bytes(pack_root) == pack_before


@pytest.mark.asyncio
async def test_incomplete_required_arm_is_retained_and_blocks_all_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    first = _pass_result(pack, incomplete=True, identity_offset=0)
    second = _pass_result(pack, incomplete=True, identity_offset=120)
    results = iter((first, second))

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)

    execution = await run_benchmark_pilot(manifest_path, tmp_path / "pilot")

    payload = execution.evidence.data.pass_payload
    assert len(payload.cases) == 30
    assert payload.cases[0].status == "incomplete"
    assert payload.cases[0].arms[1].error_code == "benchmark_arm_incomplete"
    assert payload.comparison_complete is False
    assert payload.aggregate is None
    assert execution.comparison_complete is False
    assert execution.evidence_valid is False


@pytest.mark.asyncio
async def test_pass_mismatch_and_profiler_failure_remain_separate_truthful_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    results = iter(
        (
            _pass_result(pack, identity_offset=0),
            _pass_result(pack, changed=True, identity_offset=120),
        )
    )

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    def failed_profiler() -> int:
        raise RuntimeError("machine-local failure")

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)

    execution = await run_benchmark_pilot(
        manifest_path,
        tmp_path / "pilot",
        profiler=failed_profiler,
    )

    assert execution.comparison_complete is True
    assert execution.deterministic_replay_match is False
    assert execution.evidence_valid is False
    assert execution.profile_complete is False
    assert execution.profile is None
    assert execution.profile_body is None
    assert not (tmp_path / "pilot" / "artifacts" / "profile.json").exists()


@pytest.mark.asyncio
async def test_profiler_failure_does_not_invalidate_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    results = iter(
        (
            _pass_result(pack, identity_offset=0),
            _pass_result(pack, identity_offset=120),
        )
    )

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    def failed_profiler() -> int:
        raise RuntimeError("machine-local failure")

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)

    execution = await run_benchmark_pilot(
        manifest_path,
        tmp_path / "pilot",
        profiler=failed_profiler,
    )

    assert execution.evidence_valid is True
    assert execution.comparison_complete is True
    assert execution.deterministic_replay_match is True
    assert execution.profile_complete is False
    assert execution.profile is None
    assert execution.profile_body is None
    assert not (tmp_path / "pilot" / "artifacts" / "profile.json").exists()


class _ProfilerArithmeticBomb(int):
    def __sub__(self, other: object) -> int:
        del other
        raise AssertionError("profiler arithmetic must not run")

    def __str__(self) -> str:
        raise AssertionError("profiler string conversion must not run")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profiler_values",
    (
        (False,),
        ("not-an-int",),
        (-1,),
        (1, 0),
        (1 << 1_000_000,),
        (0, 1 << 1_000_000),
        (_ProfilerArithmeticBomb(0), _ProfilerArithmeticBomb(1)),
    ),
    ids=(
        "bool",
        "string",
        "negative",
        "backward",
        "huge-start",
        "huge-end",
        "int-subclass",
    ),
)
async def test_invalid_profiler_values_omit_profile_without_affecting_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiler_values: tuple[object, ...],
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    results = iter(
        (
            _pass_result(pack, identity_offset=0),
            _pass_result(pack, identity_offset=120),
        )
    )

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    profiler_results = iter(profiler_values)
    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)

    execution = await run_benchmark_pilot(
        manifest_path,
        tmp_path / "pilot",
        profiler=profiler_results.__next__,
    )

    assert execution.evidence_valid is True
    assert execution.comparison_complete is True
    assert execution.deterministic_replay_match is True
    assert execution.profile_complete is False
    assert execution.profile is None
    assert execution.profile_body is None
    artifacts = tmp_path / "pilot" / "artifacts"
    assert (artifacts / "benchmark-evidence.json").is_file()
    assert (artifacts / "benchmark-summary.json").is_file()
    assert not (artifacts / "profile.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ("construct", "encode"))
async def test_profile_construction_and_encoding_failures_remain_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    results = iter(
        (
            _pass_result(pack, identity_offset=0),
            _pass_result(pack, identity_offset=120),
        )
    )

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    def fail_profile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("profile-local failure")

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)
    target = (
        "_profile"
        if failure_stage == "construct"
        else "canonical_benchmark_profile_bytes"
    )
    monkeypatch.setattr(
        runner_module,
        target,
        fail_profile,
    )
    clock = iter((100, 175))

    execution = await run_benchmark_pilot(
        manifest_path,
        tmp_path / "pilot",
        profiler=clock.__next__,
    )

    assert execution.evidence_valid is True
    assert execution.deterministic_replay_match is True
    assert execution.profile_complete is False
    assert execution.profile is None
    assert execution.profile_body is None
    artifacts = tmp_path / "pilot" / "artifacts"
    assert (artifacts / "benchmark-evidence.json").is_file()
    assert (artifacts / "benchmark-summary.json").is_file()
    assert not (artifacts / "profile.json").exists()


@pytest.mark.asyncio
async def test_final_source_mutation_is_published_as_unsafe_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)
    results = iter(
        (
            _pass_result(pack, identity_offset=0),
            _pass_result(pack, identity_offset=120),
        )
    )

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return next(results)

    def changed_source(snapshot: object) -> None:
        del snapshot
        raise ExperimentIsolationError("replay_snapshot_changed", "private")

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)
    monkeypatch.setattr(runner_module, "verify_source_unchanged", changed_source)

    execution = await run_benchmark_pilot(manifest_path, tmp_path / "pilot")

    safety = execution.evidence.data.pass_payload.safety
    assert safety.source_mutation_count == 1
    assert safety.source_unchanged is False
    assert execution.evidence_valid is False
    assert execution.deterministic_replay_match is True


@pytest.mark.asyncio
async def test_runner_rejects_pack_overlap_and_unknown_workspace_entries_before_delete(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "pilot"
    owned = prepare_owned_workspace(
        workspace,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    pack_root = workspace / "snapshot" / "pack"
    manifest_path = _write_pack(pack_root)
    sentinel = pack_root / "sentinel.txt"
    sentinel.write_text("retain", encoding="utf-8")

    with pytest.raises(ExperimentIsolationError) as overlap:
        await run_benchmark_pilot(
            manifest_path,
            workspace,
            replace_owned=True,
        )
    assert overlap.value.code == "benchmark_workspace_input_overlap"
    assert sentinel.read_text(encoding="utf-8") == "retain"

    clean_pack = _write_pack(tmp_path / "clean-pack")
    unknown = owned.root / "unknown.txt"
    unknown.write_text("retain", encoding="utf-8")
    with pytest.raises(ExperimentIsolationError):
        await run_benchmark_pilot(clean_pack, workspace, replace_owned=True)
    assert unknown.read_text(encoding="utf-8") == "retain"


@pytest.mark.asyncio
async def test_replace_owned_requires_marker_and_replaces_only_known_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_pack(tmp_path / "pack")
    pack = load_benchmark_pack(manifest_path)

    def result_sequence() -> object:
        yield _pass_result(pack, identity_offset=0)
        yield _pass_result(pack, identity_offset=120)

    calls = iter(result_sequence())

    async def execute_cases(**kwargs: object) -> runner_module._PassResult:
        del kwargs
        return cast(runner_module._PassResult, next(calls))

    monkeypatch.setattr(runner_module, "_execute_benchmark_cases", execute_cases)
    workspace = tmp_path / "pilot"
    first = await run_benchmark_pilot(manifest_path, workspace)

    with pytest.raises(ExperimentIsolationError) as exists:
        await run_benchmark_pilot(manifest_path, workspace)
    assert exists.value.code == "replay_workspace_exists"
    assert (
        workspace / "artifacts" / "benchmark-evidence.json"
    ).read_bytes() == first.evidence_body

    calls = iter(result_sequence())
    replaced = await run_benchmark_pilot(
        manifest_path,
        workspace,
        replace_owned=True,
    )
    assert replaced.evidence_body == first.evidence_body


def test_report_verification_is_bounded_symlink_safe_and_read_only(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = artifacts / "benchmark-evidence.json"
    report.write_bytes(b"{}")
    summary = artifacts / "benchmark-summary.json"
    summary.write_bytes(b"{}")
    before = _tree_bytes(tmp_path)

    with pytest.raises(ExperimentOutputError):
        verify_benchmark_report(report)
    assert _tree_bytes(tmp_path) == before

    misplaced = tmp_path / "benchmark-evidence.json"
    misplaced.write_bytes(b"{}")
    with pytest.raises(ExperimentOutputError) as placement:
        verify_benchmark_report(misplaced)
    assert placement.value.code == "invalid_report_path"

    report.unlink()
    report.symlink_to(summary)
    with pytest.raises(ExperimentOutputError) as linked:
        verify_benchmark_report(report)
    assert linked.value.code == "invalid_report_path"


def test_report_verification_rejects_oversized_evidence_before_reading(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = artifacts / "benchmark-evidence.json"
    report.write_bytes(b"x" * (MAX_BENCHMARK_OUTPUT_BYTES + 1))
    (artifacts / "benchmark-summary.json").write_bytes(b"{}")

    with pytest.raises(ExperimentOutputError) as oversized:
        verify_benchmark_report(report)

    assert oversized.value.code == "output_too_large"
