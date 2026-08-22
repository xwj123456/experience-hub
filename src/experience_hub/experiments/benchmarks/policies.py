"""Closed deterministic baseline policies for ExperienceBench-S."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmDescriptorV1,
    BenchmarkArmKind,
    BenchmarkArmObservationV1,
    BenchmarkCaseV1,
)
from experience_hub.experiments.benchmarks.source import BenchmarkSourceIndex
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
    PolicyExecutionContext,
)
from experience_hub.experiments.policy_clones import (
    PolicyCloneIdentity,
    require_safe_policy_clone,
    require_same_policy_clone,
)

_FTS_SQL = """
CREATE VIRTUAL TABLE pilot_fts USING fts5(
  label UNINDEXED,
  summary,
  mechanism,
  tags,
  applicability,
  tokenize='unicode61 remove_diacritics 2'
)
"""
_OWNER_VISIBLE_ROWS_SQL = """
SELECT experiences.experience_id, experiences.created_at,
       experience_versions.summary, experience_versions.mechanism,
       experience_versions.tags, experience_versions.applicability
FROM experiences
JOIN experience_state
  ON experience_state.experience_id = experiences.experience_id
JOIN experience_versions
  ON experience_versions.version_id = experience_state.current_version_id
WHERE experiences.owner_agent_id = ?
  AND experience_state.owner_agent_id = ?
  AND experience_state.temperature != 'archived'
"""


@dataclass(frozen=True, slots=True)
class BenchmarkPolicyContext:
    case: BenchmarkCaseV1
    clone_path: Path
    owner_agent_id: UUID
    source_index: BenchmarkSourceIndex
    frozen_at: datetime
    seed: int


class BenchmarkPolicyArm(Protocol):
    @property
    def descriptor(self) -> BenchmarkArmDescriptorV1: ...

    async def execute(
        self, context: BenchmarkPolicyContext
    ) -> BenchmarkArmObservationV1: ...


@dataclass(frozen=True, slots=True)
class _OwnedRecord:
    label: str
    created_at: str
    summary: str
    mechanism: str
    tags: str
    applicability: str


def _oracle_invalid() -> ExperimentInputError:
    return ExperimentInputError(
        "benchmark_oracle_invalid",
        "Benchmark policy encountered an unmapped source experience",
    )


def _capability_unavailable() -> ExperimentInputError:
    return ExperimentInputError(
        "benchmark_capability_unavailable",
        "SQLite FTS5 capability is unavailable",
    )


def _execution_failed() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "benchmark_policy_execution_failed",
        "Benchmark policy execution failed",
    )


def _select_prefix(
    context: BenchmarkPolicyContext,
    ranked_labels: Iterable[str],
) -> BenchmarkArmObservationV1:
    returned: list[str] = []
    selected_bytes = 0
    for label in ranked_labels:
        try:
            content_bytes = context.source_index.content_bytes_by_label[label]
        except KeyError:
            raise _oracle_invalid() from None
        if len(returned) >= context.case.limit:
            break
        if selected_bytes + content_bytes > context.case.content_budget_bytes:
            break
        returned.append(label)
        selected_bytes += content_bytes
    return BenchmarkArmObservationV1(
        schema_version=1,
        returned_labels=tuple(returned),
        selected_content_bytes=selected_bytes,
    )


def _same_clone_or_raise(identity: PolicyCloneIdentity) -> None:
    try:
        require_same_policy_clone(identity)
    except ExperimentIsolationError:
        raise _execution_failed() from None


def _owner_visible_records(
    connection: sqlite3.Connection,
    context: BenchmarkPolicyContext,
) -> tuple[_OwnedRecord, ...]:
    rows = connection.execute(
        _OWNER_VISIBLE_ROWS_SQL,
        (str(context.owner_agent_id), str(context.owner_agent_id)),
    ).fetchall()
    records: list[_OwnedRecord] = []
    for experience_id, created_at, summary, mechanism, tags, applicability in rows:
        try:
            label = context.source_index.labels_by_experience_id[UUID(experience_id)]
        except (KeyError, TypeError, ValueError):
            raise _oracle_invalid() from None
        if not all(
            isinstance(value, str) for value in (created_at, summary, mechanism)
        ):
            raise _execution_failed()
        records.append(
            _OwnedRecord(
                label=label,
                created_at=created_at,
                summary=summary,
                mechanism=mechanism,
                tags=_json_terms(tags),
                applicability=_json_terms(applicability),
            )
        )
    return tuple(records)


def _json_terms(value: object) -> str:
    if not isinstance(value, (str, bytes, bytearray)):
        raise _execution_failed()
    try:
        terms = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _execution_failed() from None
    if not isinstance(terms, list) or not all(isinstance(item, str) for item in terms):
        raise _execution_failed()
    return " ".join(terms)


def _recent_notes(
    identity: PolicyCloneIdentity,
    context: BenchmarkPolicyContext,
) -> BenchmarkArmObservationV1:
    with sqlite3.connect(identity.path) as connection:
        records = _owner_visible_records(connection, context)
    ranked = sorted(records, key=lambda record: record.label)
    ranked.sort(key=lambda record: record.created_at, reverse=True)
    return _select_prefix(context, (record.label for record in ranked))


def _fts_phrase(value: str) -> str:
    return f'"{value.replace("\"", "\"\"")}"'


def _fts_query(case: BenchmarkCaseV1) -> str:
    return " OR ".join(
        _fts_phrase(value)
        for value in (case.query, *case.tags, *case.mechanism_cues)
    )


def preflight_benchmark_capabilities() -> None:
    """Prove the exact FTS5 feature and tokenizer needed by the pilot arm."""
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute(_FTS_SQL)
            connection.execute(
                "INSERT INTO pilot_fts(label, summary, mechanism, tags, applicability) "
                "VALUES (?, ?, ?, ?, ?)",
                ("probe", "probe", "probe", "probe", "probe"),
            )
            connection.execute(
                "SELECT label FROM pilot_fts WHERE pilot_fts MATCH ?", ('"probe"',)
            ).fetchall()
    except sqlite3.DatabaseError:
        raise _capability_unavailable() from None


def _sqlite_bm25(
    identity: PolicyCloneIdentity,
    context: BenchmarkPolicyContext,
) -> BenchmarkArmObservationV1:
    with sqlite3.connect(identity.path) as connection:
        connection.execute(_FTS_SQL)
        records = _owner_visible_records(connection, context)
        connection.executemany(
            "INSERT INTO pilot_fts(label, summary, mechanism, tags, applicability) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (
                    record.label,
                    record.summary,
                    record.mechanism,
                    record.tags,
                    record.applicability,
                )
                for record in records
            ),
        )
        rows = connection.execute(
            "SELECT label FROM pilot_fts WHERE pilot_fts MATCH ? "
            "ORDER BY bm25(pilot_fts, 0.0, 3.0, 2.0, 1.5, 1.0), label ASC",
            (_fts_query(context.case),),
        ).fetchall()
    return _select_prefix(context, (row[0] for row in rows))


@dataclass(frozen=True, slots=True)
class NoMemoryBenchmarkPolicyArm:
    descriptor: BenchmarkArmDescriptorV1

    async def execute(
        self, context: BenchmarkPolicyContext
    ) -> BenchmarkArmObservationV1:
        del context
        return BenchmarkArmObservationV1(
            schema_version=1,
            returned_labels=(),
            selected_content_bytes=0,
        )


@dataclass(frozen=True, slots=True)
class RecentNotesPolicyArm:
    descriptor: BenchmarkArmDescriptorV1

    async def execute(
        self, context: BenchmarkPolicyContext
    ) -> BenchmarkArmObservationV1:
        identity = require_safe_policy_clone(context.clone_path)
        try:
            return await asyncio.to_thread(_recent_notes, identity, context)
        except ExperimentInputError:
            raise
        except (ExperimentIsolationError, OSError, sqlite3.DatabaseError):
            raise _execution_failed() from None
        finally:
            _same_clone_or_raise(identity)


@dataclass(frozen=True, slots=True)
class SqliteBm25PolicyArm:
    descriptor: BenchmarkArmDescriptorV1

    async def execute(
        self, context: BenchmarkPolicyContext
    ) -> BenchmarkArmObservationV1:
        preflight_benchmark_capabilities()
        identity = require_safe_policy_clone(context.clone_path)
        try:
            return await asyncio.to_thread(_sqlite_bm25, identity, context)
        except ExperimentInputError:
            raise
        except (ExperimentIsolationError, OSError, sqlite3.DatabaseError):
            raise _execution_failed() from None
        finally:
            _same_clone_or_raise(identity)


@dataclass(frozen=True, slots=True)
class ExperienceHubBenchmarkPolicyArm:
    descriptor: BenchmarkArmDescriptorV1

    async def execute(
        self, context: BenchmarkPolicyContext
    ) -> BenchmarkArmObservationV1:
        replay_case = ReplayCaseV1(
            schema_version=1,
            case_id=context.case.case_id,
            owner_agent_id=context.owner_agent_id,
            query=context.case.query,
            mode=context.case.mode,
            tags=context.case.tags,
            mechanism_cues=context.case.mechanism_cues,
            limit=context.case.limit,
            content_budget_bytes=context.case.content_budget_bytes,
            expand_cold=False,
            expected=tuple(
                ExperienceLabelV1(label=label, experience_id=experience_id)
                for label, experience_id in sorted(
                    context.source_index.experience_ids.items()
                )
            ),
            forbidden=(),
        )
        replay_arm = ExperienceHubPolicyArm(
            PolicyArmDescriptorV1(
                schema_version=1,
                arm_id=PolicyArmKind.EXPERIENCE_HUB.value,
                kind=PolicyArmKind.EXPERIENCE_HUB,
                required=True,
            )
        )
        observation = await replay_arm.execute(
            PolicyExecutionContext(
                case=replay_case,
                clone_path=context.clone_path,
                frozen_at=context.frozen_at,
                seed=context.seed,
            )
        )
        if observation.unmapped_count:
            raise _oracle_invalid()
        return _select_prefix(context, observation.returned_labels)


def build_benchmark_policy(
    descriptor: BenchmarkArmDescriptorV1,
) -> BenchmarkPolicyArm:
    """Build only one of the four fixed first-party benchmark baselines."""
    if not isinstance(descriptor, BenchmarkArmDescriptorV1):
        raise TypeError("descriptor must be BenchmarkArmDescriptorV1")
    match descriptor.kind:
        case BenchmarkArmKind.NO_MEMORY:
            return NoMemoryBenchmarkPolicyArm(descriptor)
        case BenchmarkArmKind.RECENT_NOTES:
            return RecentNotesPolicyArm(descriptor)
        case BenchmarkArmKind.SQLITE_BM25:
            return SqliteBm25PolicyArm(descriptor)
        case BenchmarkArmKind.EXPERIENCE_HUB:
            return ExperienceHubBenchmarkPolicyArm(descriptor)
        case _:
            raise ExperimentInputError(
                "benchmark_policy_unsupported",
                "Benchmark policy kind is not supported",
            )


__all__ = [
    "BenchmarkPolicyArm",
    "BenchmarkPolicyContext",
    "ExperienceHubBenchmarkPolicyArm",
    "NoMemoryBenchmarkPolicyArm",
    "RecentNotesPolicyArm",
    "SqliteBm25PolicyArm",
    "build_benchmark_policy",
    "preflight_benchmark_capabilities",
]
