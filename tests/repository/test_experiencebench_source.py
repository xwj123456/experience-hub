from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from experience_hub.experiences.models import ExperienceKind, Temperature
from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmKind,
    BenchmarkPackManifestV1,
    BenchmarkSourceAgentV1,
    BenchmarkSourceCandidateV1,
    BenchmarkSourceExperienceV1,
)
from experience_hub.experiments.benchmarks.loading import LoadedBenchmarkPack
from experience_hub.experiments.benchmarks.source import build_benchmark_source
from experience_hub.experiments.workspace import (
    WorkspacePolicy,
    prepare_owned_workspace,
)

FROZEN_AT = datetime(2026, 8, 20, tzinfo=UTC)
_POLICY = WorkspacePolicy(
    marker_name=".experiencebench-source-workspace",
    marker_body=b"experiencebench source workspace v1\n",
    owned_entries=frozenset({"snapshot"}),
)


def _pack() -> LoadedBenchmarkPack:
    manifest = BenchmarkPackManifestV1.model_validate(
        {
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
    )
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
            created_at=FROZEN_AT - timedelta(days=1),
            temperature=Temperature.HOT,
            kind=ExperienceKind.PROCEDURAL,
            body="Keep the active workflow bounded.",
            summary="Keep the workflow bounded.",
            mechanism="A bounded workflow prevents duplicate work.",
            tags=("workflow",),
            applicability=("active operations",),
            evidence=(),
            falsifiers=("The workflow has no duplicate risk.",),
            importance_micros=900_000,
            confidence_micros=900_000,
        ),
        BenchmarkSourceCandidateV1(
            schema_version=1,
            record_type="candidate",
            label="pending-note",
            owner_label="alpha",
            created_at=FROZEN_AT - timedelta(hours=1),
            kind=ExperienceKind.PROCEDURAL,
            body="Keep this pending candidate quarantined.",
            summary="Pending candidate stays quarantined.",
            mechanism="Capture requires explicit adoption.",
            tags=("pending",),
            applicability=("candidate review",),
            falsifiers=("The candidate was adopted.",),
        ),
    )
    return LoadedBenchmarkPack(
        manifest=manifest,
        cases=(),
        source=source,
        manifest_body=b"{}",
        cases_body=b"",
        source_body=b"",
        source_labels=frozenset(record.label for record in source),
        total_input_bytes=2,
        _parent=Path("."),
    )


def _workspace(path: Path):
    return prepare_owned_workspace(
        path,
        policy=_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )


def test_build_source_is_deterministic_and_quarantines_pending_candidates(
    tmp_path: Path,
) -> None:
    pack = _pack()
    first = asyncio.run(build_benchmark_source(pack, _workspace(tmp_path / "first")))
    second = asyncio.run(build_benchmark_source(pack, _workspace(tmp_path / "second")))

    assert first.snapshot.database_sha256 == second.snapshot.database_sha256
    assert first.snapshot.database_bytes == second.snapshot.database_bytes
    assert set(first.index.experience_ids) == {"active-note", "archived-note"}
    assert set(first.index.candidate_ids) == {"pending-note"}
    assert first.index.content_bytes_by_label == {
        "active-note": 97,
        "archived-note": 86,
    }
    assert not any(
        "-" in label and len(label) == 36
        for label in (*first.index.experience_ids, *first.index.candidate_ids)
    )
    assert not Path(f"{first.path}-wal").exists()
    assert not Path(f"{first.path}-shm").exists()
    assert not Path(f"{first.path}-journal").exists()

    with sqlite3.connect(first.path) as connection:
        experience_count = connection.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()
        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM candidate_state WHERE decision = 'pending'"
        ).fetchone()
    assert experience_count == (2,)
    assert candidate_count == (1,)
