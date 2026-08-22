from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from threading import Event

import pytest

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.experiences.models import ExperienceKind, Temperature
from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmKind,
    BenchmarkSourceAgentV1,
    BenchmarkSourceCandidateV1,
    BenchmarkSourceExperienceV1,
)
from experience_hub.experiments.benchmarks.loading import (
    LoadedBenchmarkPack,
    load_benchmark_pack,
)
from experience_hub.experiments.benchmarks.source import (
    build_benchmark_source,
    search_document_bytes,
)
from experience_hub.experiments.errors import ExperimentInputError
from experience_hub.experiments.snapshots import verify_source_unchanged
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    prepare_owned_workspace,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.runtime import ApplicationRuntime, require_current_schema

FROZEN_AT = datetime(2026, 8, 20, tzinfo=UTC)


def _pack(tmp_path: Path) -> LoadedBenchmarkPack:
    manifest_document = {
        "schema_version": 1,
        "pack_id": "experiencebench-s-pilot",
        "maturity": "pilot-30",
        "cases": {"file": "cases.jsonl", "sha256": "0" * 64},
        "source": {"file": "source.jsonl", "sha256": "1" * 64},
        "frozen_at": FROZEN_AT,
        "seed": 20260820,
        "arms": tuple(
            {
                "schema_version": 1,
                "arm_id": value,
                "kind": BenchmarkArmKind(value),
                "required": True,
            }
            for value in (
                "no_memory",
                "recent_notes",
                "sqlite_bm25",
                "experience_hub",
            )
        ),
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
    source = (
        BenchmarkSourceAgentV1(schema_version=1, record_type="agent", label="alpha"),
        BenchmarkSourceAgentV1(schema_version=1, record_type="agent", label="beta"),
        BenchmarkSourceExperienceV1(
            schema_version=1,
            record_type="experience",
            label="archived-note",
            owner_label="beta",
            created_at=FROZEN_AT - timedelta(days=100),
            temperature=Temperature.ARCHIVED,
            kind=ExperienceKind.SEMANTIC,
            body="The obsolete setting is not retained.",
            summary="Retire the obsolete setting.",
            mechanism="Weak, old evidence is archived.",
            tags=("obsolete",),
            applicability=("retired settings",),
            evidence=(),
            falsifiers=("The setting remains supported.",),
            importance_micros=100_000,
            confidence_micros=100_000,
        ),
        BenchmarkSourceExperienceV1(
            schema_version=1,
            record_type="experience",
            label="active-note",
            owner_label="alpha",
            created_at=FROZEN_AT - timedelta(days=2),
            temperature=Temperature.WARM,
            kind=ExperienceKind.PROCEDURAL,
            body="Keep the active workflow bounded.",
            summary="Keep the workflow bounded.",
            mechanism="A bounded workflow prevents duplicate work.",
            tags=("workflow",),
            applicability=("active operations",),
            evidence=(),
            falsifiers=("The workflow has no duplicate risk.",),
            importance_micros=800_000,
            confidence_micros=900_000,
        ),
        BenchmarkSourceCandidateV1(
            schema_version=1,
            record_type="candidate",
            label="pending-note",
            owner_label="alpha",
            created_at=FROZEN_AT - timedelta(days=1),
            kind=ExperienceKind.PROCEDURAL,
            body="Keep this pending candidate quarantined.",
            summary="Pending candidate stays quarantined.",
            mechanism="Capture requires explicit adoption.",
            tags=("pending",),
            applicability=("candidate review",),
            falsifiers=("The candidate was adopted.",),
        ),
    )
    active = source[3]
    assert isinstance(active, BenchmarkSourceExperienceV1)
    candidate = source[4]
    source = (
        *source[:4],
        active.model_copy(
            update={
                "label": "forbidden-note",
                "created_at": FROZEN_AT - timedelta(hours=20),
                "body": "Do not apply the forbidden workflow.",
                "summary": "Forbidden workflow.",
                "mechanism": "Forbidden behavior is excluded.",
                "tags": ("forbidden",),
            }
        ),
        active.model_copy(
            update={
                "label": "stale-note",
                "created_at": FROZEN_AT - timedelta(hours=16),
                "body": "Do not apply the stale workflow.",
                "summary": "Stale workflow.",
                "mechanism": "Stale behavior is excluded.",
                "tags": ("stale",),
            }
        ),
        active.model_copy(
            update={
                "label": "misleading-note",
                "created_at": FROZEN_AT - timedelta(hours=12),
                "body": "Do not apply the misleading workflow.",
                "summary": "Misleading workflow.",
                "mechanism": "Misleading behavior is excluded.",
                "tags": ("misleading",),
            }
        ),
        candidate.model_copy(update={"created_at": FROZEN_AT - timedelta(hours=8)}),
    )
    cases: list[dict[str, object]] = []
    for ordinal in range(30):
        source_class = "public_authored" if ordinal < 20 else "reviewed_abstraction"
        case = {
            "schema_version": 1,
            "case_id": f"case-{ordinal + 1}",
            "source_class": source_class,
            "review_status": (
                "authored"
                if source_class == "public_authored"
                else "maintainer_reviewed"
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
        cases.append(case)
    source_body = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        for record in source
    )
    cases_body = b"".join(canonical_json_bytes(case) + b"\n" for case in cases)
    manifest_document["cases"] = {
        "file": "cases.jsonl",
        "sha256": sha256_hex(cases_body),
    }
    manifest_document["source"] = {
        "file": "source.jsonl",
        "sha256": sha256_hex(source_body),
    }
    root = tmp_path / "pack"
    root.mkdir()
    (root / "cases.jsonl").write_bytes(cases_body)
    (root / "source.jsonl").write_bytes(source_body)
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest_document))
    return load_benchmark_pack(manifest_path)


def _workspace(path: Path):
    return prepare_owned_workspace(
        path,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )


async def _verify_frozen_projection(path: Path) -> None:
    runtime = ApplicationRuntime(
        Settings(database_url=f"sqlite+aiosqlite:///{path}"),
        clock=FrozenClock(FROZEN_AT),
        ids=SequenceIdGenerator(()),
        migrator=require_current_schema,
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        assert (await container.projection_manager.verify(container.database)).matches


def test_build_source_is_deterministic_and_quarantines_pending_candidates(
    tmp_path: Path,
) -> None:
    pack = _pack(tmp_path)
    first = asyncio.run(build_benchmark_source(pack, _workspace(tmp_path / "first")))
    second = asyncio.run(build_benchmark_source(pack, _workspace(tmp_path / "second")))

    assert first.snapshot.database_sha256 == second.snapshot.database_sha256
    assert first.snapshot.database_bytes == second.snapshot.database_bytes
    assert first.index.agent_ids == second.index.agent_ids
    assert first.index.experience_ids == second.index.experience_ids
    assert first.index.candidate_ids == second.index.candidate_ids
    active = next(
        record
        for record in pack.source
        if isinstance(record, BenchmarkSourceExperienceV1)
        and record.label == "active-note"
    )
    assert search_document_bytes(active) == (
        b"Keep the workflow bounded.\n"
        b"A bounded workflow prevents duplicate work.\n"
        b"workflow\nactive operations"
    )
    assert set(first.index.experience_ids) == {
        "active-note",
        "archived-note",
        "forbidden-note",
        "stale-note",
        "misleading-note",
    }
    assert set(first.index.candidate_ids) == {"pending-note"}
    assert set(first.index.agent_ids) == {"alpha", "beta"}
    assert first.index.labels_by_experience_id == {
        identifier: label
        for label, identifier in first.index.experience_ids.items()
    }
    assert first.index.content_bytes_by_label == {
        "active-note": 97,
        "archived-note": 86,
        "forbidden-note": 79,
        "stale-note": 67,
        "misleading-note": 82,
    }
    assert not any(
        "-" in label and len(label) == 36
        for label in (*first.index.experience_ids, *first.index.candidate_ids)
    )
    for built in (first, second):
        assert not Path(f"{built.path}-wal").exists()
        assert not Path(f"{built.path}-shm").exists()
        assert not Path(f"{built.path}-journal").exists()
        assert not Path(f"{built.path}.canonical").exists()
        asyncio.run(_verify_frozen_projection(built.path))
        verify_source_unchanged(built.snapshot)
        assert not any(
            candidate.exists()
            for candidate in (
                Path(f"{built.path}-wal"),
                Path(f"{built.path}-shm"),
                Path(f"{built.path}-journal"),
                Path(f"{built.path}.canonical"),
            )
        )

    with sqlite3.connect(first.path) as connection:
        experience_count = connection.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()
        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM candidate_state WHERE decision = 'pending'"
        ).fetchone()
        persisted_agents = dict(
            connection.execute("SELECT name, agent_id FROM agents").fetchall()
        )
        persisted_experiences = dict(
            connection.execute(
                "SELECT experience_id, owner_agent_id FROM experiences"
            ).fetchall()
        )
        persisted_candidates = dict(
            connection.execute(
                "SELECT candidate_id, owner_agent_id FROM candidate_state"
            ).fetchall()
        )
    assert experience_count == (5,)
    assert candidate_count == (1,)
    assert persisted_agents == {
        label: str(identifier) for label, identifier in first.index.agent_ids.items()
    }
    assert set(persisted_experiences) == {
        str(identifier) for identifier in first.index.experience_ids.values()
    }
    assert set(persisted_candidates) == {
        str(identifier) for identifier in first.index.candidate_ids.values()
    }
    for record in pack.source:
        if isinstance(record, BenchmarkSourceExperienceV1):
            experience_id = str(first.index.experience_ids[record.label])
            assert persisted_experiences[experience_id] == str(
                first.index.agent_ids[record.owner_label]
            )
        if isinstance(record, BenchmarkSourceCandidateV1):
            candidate_id = str(first.index.candidate_ids[record.label])
            assert persisted_candidates[candidate_id] == str(
                first.index.agent_ids[record.owner_label]
            )


@pytest.mark.parametrize(
    "failure_point", ("runtime", "command", "checkpoint", "freeze")
)
def test_source_failure_removes_only_its_owned_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str
) -> None:
    from experience_hub.experiments.benchmarks import source as source_module

    if failure_point == "runtime":
        def fail_runtime_initialize(*_: object, **__: object) -> object:
            raise RuntimeError("private runtime path")

        monkeypatch.setattr(ApplicationRuntime, "initialize", fail_runtime_initialize)
    elif failure_point == "command":
        async def fail_command(*_: object, **__: object) -> object:
            raise RuntimeError("private command path")

        monkeypatch.setattr(source_module, "_create_agent", fail_command)
    elif failure_point == "checkpoint":
        def fail_checkpoint(*_: object, **__: object) -> object:
            raise RuntimeError("private checkpoint path")

        monkeypatch.setattr(source_module, "checkpoint_owned_sqlite", fail_checkpoint)
    else:
        def fail_freeze(*_: object, **__: object) -> object:
            raise RuntimeError("private freeze path")

        monkeypatch.setattr(source_module, "freeze_closed_sqlite", fail_freeze)

    workspace = _workspace(tmp_path / "workspace")
    with pytest.raises(ExperimentInputError) as captured:
        asyncio.run(build_benchmark_source(_pack(tmp_path), workspace))

    assert captured.value.code == "benchmark_source_invalid"
    assert "private" not in str(captured.value)
    assert not (workspace.root / "snapshot" / "source.sqlite3").exists()
    assert not (workspace.root / "snapshot").exists()
    assert not list(workspace.root.glob("snapshot/*.sqlite3*"))
    assert not list(workspace.root.glob("snapshot/*.tmp"))


def test_source_cancellation_waits_for_checkpoint_worker_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from experience_hub.experiments.benchmarks import source as source_module

    started = Event()
    release = Event()

    def blocking_checkpoint(*_: object, **__: object) -> tuple[int, int, int]:
        started.set()
        assert release.wait(timeout=1)
        return (0, 0, 0)

    monkeypatch.setattr(source_module, "checkpoint_owned_sqlite", blocking_checkpoint)
    workspace = _workspace(tmp_path / "workspace")

    async def scenario() -> None:
        task = asyncio.create_task(build_benchmark_source(_pack(tmp_path), workspace))
        await asyncio.to_thread(started.wait)
        contender = asyncio.create_task(
            workspace.reserve_new_file_scoped_async(
                PurePosixPath("snapshot/source.sqlite3")
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not contender.done()
        assert (workspace.root / "snapshot" / "source.sqlite3").exists()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        reservation = await contender
        reservation.rollback()
        reservation.close()

    asyncio.run(scenario())
    assert not (workspace.root / "snapshot").exists()
