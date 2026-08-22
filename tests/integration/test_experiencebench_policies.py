"""Integration coverage for the closed ExperienceBench policy registry."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import pytest

from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmDescriptorV1,
    BenchmarkArmKind,
    BenchmarkCaseV1,
)
from experience_hub.experiments.benchmarks.source import BenchmarkSourceIndex
from experience_hub.experiments.errors import ExperimentInputError

FROZEN_AT = datetime(2026, 8, 20, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-4000-8000-000000000001")


def _case() -> BenchmarkCaseV1:
    return BenchmarkCaseV1.model_validate_json(
        json.dumps(
            {
            "schema_version": 1,
            "case_id": "queue-recovery",
            "source_class": "public_authored",
            "review_status": "authored",
            "stratum": "failure_recovery",
            "language": "en",
            "difficulty": "B",
            "owner_label": "queue-owner",
            "query": 'queue "recovery"',
            "mode": "focused",
            "tags": ["queue"],
            "mechanism_cues": ["bounded-recovery"],
            "limit": 2,
            "content_budget_bytes": 20,
            "source_labels": [
                "queue-required",
                "queue-optional",
                "queue-forbidden",
                "queue-stale",
                "queue-misleading",
            ],
            "required": [{"label": "queue-required", "weight_micros": 450000}],
            "optional": [{"label": "queue-optional", "weight_micros": 0}],
            "forbidden": [{"label": "queue-forbidden", "weight_micros": 100000}],
            "stale": [{"label": "queue-stale", "weight_micros": 100000}],
            "misleading": [
                {"label": "queue-misleading", "weight_micros": 100000}
            ],
            "checkpoints": [
                {
                    "predicate": "ordered_subsequence",
                    "labels": ["queue-required"],
                    "weight_micros": 150000,
                }
            ],
            "oracle_version": 1,
            }
        ),
    )


def _source_index() -> BenchmarkSourceIndex:
    first = UUID("00000000-0000-4000-8000-000000000010")
    second = UUID("00000000-0000-4000-8000-000000000011")
    return BenchmarkSourceIndex(
        agent_ids=MappingProxyType({"queue-owner": OWNER_ID}),
        experience_ids=MappingProxyType(
            {"queue-required": first, "queue-optional": second}
        ),
        candidate_ids=MappingProxyType({}),
        labels_by_experience_id=MappingProxyType(
            {first: "queue-required", second: "queue-optional"}
        ),
        content_bytes_by_label=MappingProxyType(
            {"queue-required": 12, "queue-optional": 12}
        ),
    )


def _descriptor(kind: BenchmarkArmKind) -> BenchmarkArmDescriptorV1:
    return BenchmarkArmDescriptorV1(
        schema_version=1,
        arm_id=kind.value,
        kind=kind,
        required=True,
    )


def _clone_with_owned_records(path: Path) -> None:
    first = UUID("00000000-0000-4000-8000-000000000010")
    second = UUID("00000000-0000-4000-8000-000000000011")
    archived = UUID("00000000-0000-4000-8000-000000000012")
    foreign = UUID("00000000-0000-4000-8000-000000000013")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE experiences(experience_id TEXT, owner_agent_id TEXT,
                                     created_at TEXT);
            CREATE TABLE experience_versions(version_id TEXT, summary TEXT,
                                             mechanism TEXT, tags BLOB,
                                             applicability BLOB);
            CREATE TABLE experience_state(experience_id TEXT, owner_agent_id TEXT,
                                          current_version_id TEXT, temperature TEXT);
            CREATE TABLE experience_candidates(candidate_id TEXT, owner_agent_id TEXT);
            """
        )
        rows = (
            (
                first,
                OWNER_ID,
                "2026-08-19T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
            (
                second,
                OWNER_ID,
                "2026-08-19T00:00:00Z",
                "ordinary note",
                "other mechanism",
            ),
            (
                archived,
                OWNER_ID,
                "2026-08-20T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
            (
                foreign,
                UUID("00000000-0000-4000-8000-000000000002"),
                "2026-08-21T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
        )
        for ordinal, (
            experience_id,
            owner_id,
            created_at,
            summary,
            mechanism,
        ) in enumerate(rows):
            version_id = f"version-{ordinal}"
            connection.execute(
                "INSERT INTO experiences VALUES (?, ?, ?)",
                (str(experience_id), str(owner_id), created_at),
            )
            connection.execute(
                "INSERT INTO experience_versions VALUES (?, ?, ?, ?, ?)",
                (
                    version_id,
                    summary,
                    mechanism,
                    json.dumps(["queue"]),
                    json.dumps(["local queue"]),
                ),
            )
            connection.execute(
                "INSERT INTO experience_state VALUES (?, ?, ?, ?)",
                (
                    str(experience_id),
                    str(owner_id),
                    version_id,
                    "archived" if experience_id == archived else "warm",
                ),
            )
        connection.execute(
            "INSERT INTO experience_candidates VALUES (?, ?)",
            ("pending-candidate", str(OWNER_ID)),
        )


def _context(clone_path: Path):
    from experience_hub.experiments.benchmarks.policies import BenchmarkPolicyContext

    return BenchmarkPolicyContext(
        case=_case(),
        clone_path=clone_path,
        owner_agent_id=OWNER_ID,
        source_index=_source_index(),
        frozen_at=FROZEN_AT,
        seed=20260820,
    )


def test_registry_accepts_exactly_the_four_closed_baseline_arms() -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    for kind in BenchmarkArmKind:
        assert build_benchmark_policy(_descriptor(kind)).descriptor.kind is kind

    forged = _descriptor(BenchmarkArmKind.NO_MEMORY).model_copy(
        update={"kind": "external-policy"}
    )
    with pytest.raises(ValueError) as raised:
        build_benchmark_policy(forged)
    assert "external-policy" not in str(raised.value)


@pytest.mark.asyncio
async def test_no_memory_never_opens_the_clone_and_has_no_selected_bytes(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import (
        BenchmarkPolicyContext,
        build_benchmark_policy,
    )

    observation = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.NO_MEMORY)
    ).execute(
        BenchmarkPolicyContext(
            case=_case(),
            clone_path=tmp_path / "must-not-open.sqlite3",
            owner_agent_id=OWNER_ID,
            source_index=_source_index(),
            frozen_at=FROZEN_AT,
            seed=20260820,
        )
    )

    assert observation.returned_labels == ()
    assert observation.selected_content_bytes == 0


@pytest.mark.asyncio
async def test_recent_and_bm25_use_only_owned_nonarchived_records_and_prefix_budget(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "disposable.sqlite3"
    _clone_with_owned_records(clone_path)
    context = _context(clone_path)

    recent = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.RECENT_NOTES)
    ).execute(context)
    bm25 = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(context)

    assert recent.returned_labels == ("queue-optional",)
    assert recent.selected_content_bytes == 12
    assert bm25.returned_labels == ("queue-required",)
    assert bm25.selected_content_bytes == 12
    assert "pending-candidate" not in (*recent.returned_labels, *bm25.returned_labels)


def test_fts_phrase_escapes_quotes_and_missing_capability_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module

    assert policy_module._fts_phrase('queue "recovery"') == '"queue ""recovery"""'
    original_connect = sqlite3.connect

    def unavailable_connect(path: object, *args: object, **kwargs: object):
        if path == ":memory:":
            raise sqlite3.OperationalError("disabled")
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(policy_module.sqlite3, "connect", unavailable_connect)
    with pytest.raises(ExperimentInputError) as raised:
        policy_module.preflight_benchmark_capabilities()
    assert raised.value.code == "benchmark_capability_unavailable"
