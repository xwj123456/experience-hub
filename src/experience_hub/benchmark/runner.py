"""Isolated, deterministic effectiveness benchmark execution."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession

import experience_hub.config as config
from experience_hub.agents.models import CreateAgent
from experience_hub.benchmark.cases import (
    BenchmarkCase,
    BenchmarkSeed,
    ColdCueCase,
    InspirationCase,
    IrrelevantDistractorCase,
    PropagationCase,
    RetrievalCase,
    SeedExperience,
    load_cases,
    load_seed,
)
from experience_hub.benchmark.metrics import (
    macro_recall_at_five,
    recall_at_five,
    unique_mechanism_ratio,
)
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.errors import CanonicalizationError
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.events import ExperienceReactivatedV1
from experience_hub.experiences.models import Temperature, VersionContent
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.service import ExperienceRetrievalAdapter
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.hashing import (
    hash_idea_content,
    hash_mechanism,
    stable_evidence_key,
)
from experience_hub.inspiration.models import (
    EvidenceSourceType,
    Idea,
    InspirationRun,
    MechanismMaturity,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    RetrievalCandidate,
    RetrievalRecord,
    SearchExperiences,
    SearchHit,
    SearchResult,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import RetrievalService, retrieval_query_hash
from experience_hub.runtime import ApplicationRuntime
from experience_hub.sharing.hashing import compute_original_root_fingerprint
from experience_hub.sharing.models import (
    AdoptCapsule,
    CreateSubscription,
    CreateTopic,
    InboxState,
    PublishCapsule,
)
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    InspirationSnapshotItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

type CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]

_SCORE_QUANTUM = Decimal("0.000000000001")
_UUID_TEXT = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TIMESTAMP_TEXT = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:[Zz]|[+-]\d{2}:?\d{2})?"
)
_DECIMAL_TEXT = re.compile(
    r"[+-]?(?:\d+\.\d+|\d+(?:\.\d*)?[Ee][+-]?\d+)\Z"
)
_QUANTIZED_DECIMAL_TEXT = re.compile(r"-?\d+\.\d{12}\Z")
_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?:/|[A-Za-z]:[\\/]|file://|sqlite(?:\+[A-Za-z0-9_]+)?://)"
)
_FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "database",
        "database_path",
        "receipt_id",
        "event_id",
        "run_id",
        "idea_id",
        "occurrence_id",
        "snapshot_item_id",
        "created_at",
        "completed_at",
        "occurred_at",
        "started_at",
        "evaluated_at",
        "last_signal_at",
        "elapsed_milliseconds",
        "wall_duration",
    }
)
_WORKSPACE_ENTRIES = frozenset(
    {
        ".experience-hub-benchmark-workspace",
        "replay-a",
        "replay-b",
        "snapshot",
    }
)
_WORKSPACE_MARKER = "experience-hub deterministic benchmark workspace\n"


class BenchmarkIsolationError(RuntimeError):
    """The pre-run SQLite snapshot is not safe to clone."""


class BenchmarkOutputError(ValueError):
    """The benchmark report contains unstable runtime identity."""


@dataclass(frozen=True, slots=True)
class SeedIndex:
    """Stable fixture labels and internal resource identities."""

    agent_ids: Mapping[str, UUID]
    experience_ids: Mapping[str, UUID]
    version_ids: Mapping[str, UUID]
    content_hashes: Mapping[str, str]
    experience_ordinals: Mapping[str, str]

    @property
    def labels_by_experience_id(self) -> Mapping[UUID, str]:
        return MappingProxyType(
            {
                experience_id: label
                for label, experience_id in self.experience_ids.items()
            }
        )


@dataclass(frozen=True, slots=True)
class ClosedBenchmarkSnapshot:
    """One checkpointed immutable main-database byte snapshot."""

    seed: BenchmarkSeed
    cases: tuple[BenchmarkCase, ...]
    database_path: Path
    database_bytes: bytes
    database_sha256: str
    index: SeedIndex
    checkpoint_result: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EffectivenessMetrics:
    """Exact aggregate inputs for the locked effectiveness gates."""

    focused_macro_recall_at_five: Decimal
    cold_macro_recall_at_five: Decimal
    cold_baseline_macro_recall_at_five: Decimal
    distractor_false_reactivations: int
    pending_leakage_count: int
    adopted_provenance_complete: int
    adopted_provenance_total: int
    valid_idea_count: int
    idea_schema_evidence_valid_count: int
    distinct_mechanism_count: int
    same_snapshot_incubation_promotions: int
    byte_identical_replay: bool
    inspiration_evidence_coverage_failures: int = 0

    def __post_init__(self) -> None:
        for name in (
            "focused_macro_recall_at_five",
            "cold_macro_recall_at_five",
            "cold_baseline_macro_recall_at_five",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not Decimal(0) <= value <= Decimal(1):
                raise ValueError(f"{name} must be a Decimal between zero and one")
        for name in (
            "distractor_false_reactivations",
            "pending_leakage_count",
            "adopted_provenance_complete",
            "adopted_provenance_total",
            "valid_idea_count",
            "idea_schema_evidence_valid_count",
            "distinct_mechanism_count",
            "same_snapshot_incubation_promotions",
            "inspiration_evidence_coverage_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.adopted_provenance_complete > self.adopted_provenance_total:
            raise ValueError("complete provenance cannot exceed total adoptions")
        if self.idea_schema_evidence_valid_count > self.valid_idea_count:
            raise ValueError("valid schema/evidence count cannot exceed valid ideas")
        if self.distinct_mechanism_count > self.idea_schema_evidence_valid_count:
            raise ValueError("distinct mechanisms cannot exceed valid ideas")
        if not isinstance(self.byte_identical_replay, bool):
            raise ValueError("byte_identical_replay must be a bool")

    @property
    def cold_recall_gain(self) -> Decimal:
        return self.cold_macro_recall_at_five - (
            self.cold_baseline_macro_recall_at_five
        )

    @property
    def provenance_ratio(self) -> Decimal:
        if self.adopted_provenance_total == 0:
            return Decimal(0)
        return Decimal(self.adopted_provenance_complete) / Decimal(
            self.adopted_provenance_total
        )

    @property
    def idea_validity_ratio(self) -> Decimal:
        if self.valid_idea_count == 0:
            return Decimal(0)
        return Decimal(self.idea_schema_evidence_valid_count) / Decimal(
            self.valid_idea_count
        )

    @property
    def mechanism_ratio(self) -> Decimal:
        return unique_mechanism_ratio(
            self.idea_schema_evidence_valid_count,
            self.distinct_mechanism_count,
        )

    @property
    def effective_idea_validity_ratio(self) -> Decimal:
        if self.inspiration_evidence_coverage_failures:
            return Decimal(0)
        return self.idea_validity_ratio


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    actual: str | int | bool
    comparison: str
    required: str | int | bool
    passed: bool

    def document(self) -> dict[str, Any]:
        return {
            "actual": self.actual,
            "comparison": self.comparison,
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkExecution:
    """Canonical report bytes and process-facing gate outcome."""

    report: Mapping[str, Any]
    body: bytes
    passed: bool
    failed_gates: tuple[str, ...]


def _quantized(value: Decimal | float) -> str:
    retained = value if isinstance(value, Decimal) else Decimal(str(value))
    return format(
        retained.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN),
        "f",
    )


def evaluate_effectiveness_gates(
    metrics: EffectivenessMetrics,
) -> tuple[GateResult, ...]:
    """Evaluate every gate independently at its exact inclusive boundary."""
    if not isinstance(metrics, EffectivenessMetrics):
        raise TypeError("metrics must be EffectivenessMetrics")
    return (
        GateResult(
            name="focused_macro_recall_at_5",
            actual=_quantized(metrics.focused_macro_recall_at_five),
            comparison="greater_than_or_equal",
            required=_quantized(Decimal("0.90")),
            passed=metrics.focused_macro_recall_at_five >= Decimal("0.90"),
        ),
        GateResult(
            name="cold_macro_recall_at_5",
            actual=_quantized(metrics.cold_macro_recall_at_five),
            comparison="greater_than_or_equal",
            required=_quantized(Decimal("0.85")),
            passed=metrics.cold_macro_recall_at_five >= Decimal("0.85"),
        ),
        GateResult(
            name="cold_recall_gain_over_hot_warm_baseline",
            actual=_quantized(metrics.cold_recall_gain),
            comparison="greater_than_or_equal",
            required=_quantized(Decimal("0.25")),
            passed=metrics.cold_recall_gain >= Decimal("0.25"),
        ),
        GateResult(
            name="distractor_false_reactivations",
            actual=metrics.distractor_false_reactivations,
            comparison="equal",
            required=0,
            passed=metrics.distractor_false_reactivations == 0,
        ),
        GateResult(
            name="pending_capsule_leakage",
            actual=metrics.pending_leakage_count,
            comparison="equal",
            required=0,
            passed=metrics.pending_leakage_count == 0,
        ),
        GateResult(
            name="adopted_provenance_completeness",
            actual=_quantized(metrics.provenance_ratio),
            comparison="equal",
            required=_quantized(Decimal(1)),
            passed=metrics.provenance_ratio == Decimal(1),
        ),
        GateResult(
            name="valid_idea_count",
            actual=metrics.idea_schema_evidence_valid_count,
            comparison="greater_than_or_equal",
            required=12,
            passed=metrics.idea_schema_evidence_valid_count >= 12,
        ),
        GateResult(
            name="idea_schema_and_evidence_validity",
            actual=_quantized(metrics.effective_idea_validity_ratio),
            comparison="equal",
            required=_quantized(Decimal(1)),
            passed=metrics.effective_idea_validity_ratio == Decimal(1),
        ),
        GateResult(
            name="unique_mechanism_ratio",
            actual=_quantized(metrics.mechanism_ratio),
            comparison="greater_than_or_equal",
            required=_quantized(Decimal("0.70")),
            passed=metrics.mechanism_ratio >= Decimal("0.70"),
        ),
        GateResult(
            name="same_snapshot_incubation_promotion",
            actual=metrics.same_snapshot_incubation_promotions,
            comparison="equal",
            required=0,
            passed=metrics.same_snapshot_incubation_promotions == 0,
        ),
        GateResult(
            name="byte_identical_replay",
            actual=metrics.byte_identical_replay,
            comparison="equal",
            required=True,
            passed=metrics.byte_identical_replay,
        ),
    )


def _validate_output_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise BenchmarkOutputError(f"{path} contains a non-string key")
            if (
                key in _FORBIDDEN_OUTPUT_FIELDS
                or key == "workspace"
                or key.endswith("_path")
                or key.startswith("database_")
            ):
                raise BenchmarkOutputError(f"{path}.{key} is an unstable field")
            _validate_output_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_output_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, UUID):
        raise BenchmarkOutputError(f"{path} contains a raw UUID")
    if isinstance(value, datetime):
        raise BenchmarkOutputError(f"{path} contains a timestamp")
    if isinstance(value, (float, Decimal)):
        raise BenchmarkOutputError(
            f"{path} contains an unquantized numeric value"
        )
    if isinstance(value, str):
        if _UUID_TEXT.search(value):
            raise BenchmarkOutputError(f"{path} contains a raw UUID")
        if _TIMESTAMP_TEXT.search(value):
            raise BenchmarkOutputError(f"{path} contains a timestamp")
        if _ABSOLUTE_PATH_TEXT.match(value):
            raise BenchmarkOutputError(f"{path} contains an absolute path")
        if _DECIMAL_TEXT.fullmatch(value) and not (
            _QUANTIZED_DECIMAL_TEXT.fullmatch(value)
        ):
            raise BenchmarkOutputError(
                f"{path} contains a non-quantized decimal"
            )


def canonical_benchmark_bytes(report: Mapping[str, Any]) -> bytes:
    """Validate and encode the stable, identity-free benchmark document."""
    if not isinstance(report, Mapping):
        raise BenchmarkOutputError("benchmark report must be a mapping")
    _validate_output_value(report, path="$")
    try:
        body = canonical_json_bytes(report)
    except (CanonicalizationError, TypeError, ValueError) as error:
        raise BenchmarkOutputError("benchmark report is not canonical JSON") from error
    decoded = json.loads(body)
    _validate_output_value(decoded, path="$")
    if canonical_json_bytes(decoded) != body:
        raise BenchmarkOutputError("benchmark report is not canonical JSON")
    return body


def _settings(path: Path) -> Settings:
    url = URL.create("sqlite+aiosqlite", database=str(path))
    return Settings(database_url=url.render_as_string(hide_password=False))


def _container_factory(
    seed: BenchmarkSeed,
) -> Callable[..., ApplicationContainer]:
    return partial(
        ApplicationContainer.build,
        lifecycle_config=seed.config.lifecycle.to_domain(),
    )


def _id_sequence(*, start: int, stop: int) -> SequenceIdGenerator:
    return SequenceIdGenerator(tuple(UUID(int=value) for value in range(start, stop)))


def _response_document(
    result: CommandResult | StoredResponse,
    *,
    expected_status: int,
) -> dict[str, Any]:
    try:
        decoded = json.loads(result.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Benchmark service returned invalid JSON") from error
    if (
        result.status_code != expected_status
        or not isinstance(decoded, dict)
        or canonical_json_bytes(decoded) != result.body
    ):
        raise RuntimeError(
            f"Benchmark command failed with status {result.status_code}: {decoded}"
        )
    data = decoded.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Benchmark command response has no data object")
    return cast(dict[str, Any], data)


async def _execute(
    container: ApplicationContainer,
    request: CommandRequest,
    handler: CommandHandler,
) -> CommandResult:
    return await container.command_executor.execute(request, handler)


async def _create_agent(
    container: ApplicationContainer,
    *,
    label: str,
    key: str,
) -> UUID:
    request = CommandRequest(
        caller_scope="system:benchmark",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        body={"name": label},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=label),
            command_context=context,
        )

    data = _response_document(
        await _execute(container, request, handler),
        expected_status=201,
    )
    return UUID(cast(str, data["agent_id"]))


async def _create_experience(
    container: ApplicationContainer,
    *,
    fixture: SeedExperience,
    owner_agent_id: UUID,
    key: str,
) -> tuple[UUID, UUID, str]:
    content = VersionContent(
        body=fixture.body,
        summary=fixture.summary,
        mechanism=fixture.mechanism,
        tags=fixture.tags,
        applicability=fixture.applicability,
        evidence=fixture.evidence,
        falsifiers=fixture.falsifiers,
    )
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=fixture.kind,
        content=content,
        importance=fixture.importance,
        confidence=fixture.confidence,
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
            "confidence": fixture.confidence,
            "importance": fixture.importance,
            "kind": fixture.kind,
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

    data = _response_document(
        await _execute(container, request, handler),
        expected_status=201,
    )
    return (
        UUID(cast(str, data["experience_id"])),
        UUID(cast(str, data["version_id"])),
        cast(str, data["content_hash"]),
    )


def _clone_ids() -> SequenceIdGenerator:
    return _id_sequence(start=1_000_000, stop=1_004_096)


async def _run_lifecycle(
    container: ApplicationContainer,
    *,
    key: str,
) -> None:
    evaluated_at = container.clock.now()
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=key,
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": evaluated_at, "mode": "manual"},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.lifecycle_service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode="manual",
        )

    _response_document(
        await _execute(container, request, handler),
        expected_status=200,
    )


def _checkpoint_truncate(path: Path) -> tuple[int, int, int]:
    with closing(sqlite3.connect(path, isolation_level=None, timeout=5)) as connection:
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if (
        row is None
        or len(row) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in row)
    ):
        raise BenchmarkIsolationError("SQLite returned an invalid WAL checkpoint")
    result = cast(tuple[int, int, int], tuple(row))
    if result != (0, 0, 0):
        raise BenchmarkIsolationError(
            f"SQLite WAL checkpoint did not fully truncate: {result}"
        )
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            if sidecar.stat().st_size:
                raise BenchmarkIsolationError(
                    f"SQLite {suffix} sidecar remains nonempty after checkpoint"
                )
            sidecar.unlink()
    return result


async def prepare_benchmark_snapshot(
    *,
    seed_path: Path,
    cases_path: Path,
    workspace: Path,
) -> ClosedBenchmarkSnapshot:
    """Seed, validate, close, checkpoint, and freeze one benchmark database."""
    seed = load_seed(seed_path)
    cases = load_cases(cases_path, seed=seed)
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "pre-run.sqlite3"
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)

    clock = FrozenClock(seed.clock.started_at)
    runtime = ApplicationRuntime(
        _settings(database_path),
        clock=clock,
        ids=_id_sequence(start=1, stop=4_096),
        container_factory=_container_factory(seed),
    )
    agent_ids: dict[str, UUID] = {}
    experience_ids: dict[str, UUID] = {}
    version_ids: dict[str, UUID] = {}
    content_hashes: dict[str, str] = {}
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        if container.lifecycle_config != seed.config.lifecycle.to_domain():
            raise BenchmarkIsolationError(
                "Pinned benchmark lifecycle configuration differs from runtime"
            )
        for ordinal, agent in enumerate(seed.agents, start=1):
            agent_ids[agent.label] = await _create_agent(
                container,
                label=agent.label,
                key=f"benchmark-seed-agent-{ordinal}",
            )
        for ordinal, experience in enumerate(seed.experiences, start=1):
            experience_id, version_id, content_hash = await _create_experience(
                container,
                fixture=experience,
                owner_agent_id=agent_ids[experience.owner_label],
                key=f"benchmark-seed-experience-{ordinal}",
            )
            experience_ids[experience.label] = experience_id
            version_ids[experience.label] = version_id
            content_hashes[experience.label] = content_hash

        interval = container.lifecycle_config.minimum_cycle_interval
        first_cycle_at = seed.clock.cold_at - interval - timedelta(seconds=1)
        clock.advance(first_cycle_at - clock.now())
        await _run_lifecycle(container, key="benchmark-seed-lifecycle-1")
        clock.advance(seed.clock.cold_at - clock.now())
        await _run_lifecycle(container, key="benchmark-seed-lifecycle-2")

        async with container.database.read_session() as session:
            for experience in seed.experiences:
                record = await container.experience_query.get_owned_retrieval_record(
                    session=session,
                    owner_agent_id=agent_ids[experience.owner_label],
                    experience_id=experience_ids[experience.label],
                )
                if (
                    record is None
                    or record.state.temperature is not experience.target_temperature
                ):
                    raise BenchmarkIsolationError(
                        f"Seed experience {experience.label} did not reach "
                        f"{experience.target_temperature.value}"
                    )
        verification = await container.projection_manager.verify(container.database)
        if not verification.matches:
            raise BenchmarkIsolationError(
                "Seed projections do not match authoritative replay"
            )

    checkpoint_result = _checkpoint_truncate(database_path)
    database_bytes = database_path.read_bytes()
    digest = hashlib.sha256(database_bytes).hexdigest()
    return ClosedBenchmarkSnapshot(
        seed=seed,
        cases=cases,
        database_path=database_path,
        database_bytes=database_bytes,
        database_sha256=digest,
        index=SeedIndex(
            agent_ids=MappingProxyType(dict(agent_ids)),
            experience_ids=MappingProxyType(dict(experience_ids)),
            version_ids=MappingProxyType(dict(version_ids)),
            content_hashes=MappingProxyType(dict(content_hashes)),
            experience_ordinals=MappingProxyType(
                {
                    experience.label: f"experience#{ordinal}"
                    for ordinal, experience in enumerate(seed.experiences, start=1)
                }
            ),
        ),
        checkpoint_result=checkpoint_result,
    )


def clone_closed_database(
    snapshot: ClosedBenchmarkSnapshot,
    destination: Path,
) -> Path:
    """Clone only exact immutable main-database bytes into a fresh path."""
    if not isinstance(snapshot, ClosedBenchmarkSnapshot):
        raise TypeError("snapshot must be ClosedBenchmarkSnapshot")
    source_bytes = snapshot.database_path.read_bytes()
    if (
        source_bytes != snapshot.database_bytes
        or hashlib.sha256(source_bytes).hexdigest() != snapshot.database_sha256
    ):
        raise BenchmarkIsolationError("Benchmark snapshot is no longer immutable")
    for suffix in ("-wal", "-shm", "-journal"):
        source_sidecar = Path(f"{snapshot.database_path}{suffix}")
        if source_sidecar.exists() and source_sidecar.stat().st_size:
            label = "WAL" if suffix == "-wal" else suffix
            raise BenchmarkIsolationError(
                f"Benchmark snapshot has a nonempty {label} sidecar"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{destination}{suffix}").unlink(missing_ok=True)
    destination.write_bytes(snapshot.database_bytes)
    if destination.read_bytes() != snapshot.database_bytes:
        raise BenchmarkIsolationError("Benchmark clone bytes changed during copy")
    return destination


class _HotWarmOnlyQuery:
    """Clone-local baseline that changes only cold candidate admission."""

    def __init__(self, delegate: ExperienceQuery) -> None:
        self._delegate = delegate

    async def get_owned_retrieval_record(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> RetrievalRecord | None:
        return await self._delegate.get_owned_retrieval_record(
            session=session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )

    async def select_retrieval_candidates(
        self,
        *,
        session: AsyncSession,
        selection: CandidateSelection,
    ) -> tuple[RetrievalCandidate, ...]:
        candidates = await self._delegate.select_retrieval_candidates(
            session=session,
            selection=selection,
        )
        return tuple(
            candidate
            for candidate in candidates
            if candidate.record.state.temperature is not Temperature.COLD
        )

    async def load_decoded_payloads(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        version_ids: Sequence[UUID],
    ) -> dict[UUID, bytes]:
        return await self._delegate.load_decoded_payloads(
            session=session,
            owner_agent_id=owner_agent_id,
            version_ids=version_ids,
        )


def _hot_warm_only_adapter(
    container: ApplicationContainer,
) -> ExperienceRetrievalAdapter:
    service = RetrievalService(
        clock=container.clock,
        query=_HotWarmOnlyQuery(container.experience_query),
        mutation_writer=container.experience_mutation_writer,
        lifecycle_config=container.lifecycle_config,
    )
    return ExperienceRetrievalAdapter(
        executor=container.command_executor,
        retrieval_service=service,
        id_generator=container.ids,
    )


def _search_request(query: SearchExperiences, *, key: str) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{query.owner_agent_id}",
        operation_scope="experience.search",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences:search",
        path_parameters={"agent_id": query.owner_agent_id},
        body={
            "query": query.query,
            "mode": query.mode,
            "tags": query.tags,
            "mechanism_cues": query.mechanism_cues,
            "limit": query.limit,
            "content_budget_bytes": query.content_budget_bytes,
            "expand_cold": query.expand_cold,
        },
    )


@dataclass(frozen=True, slots=True)
class _SearchObservation:
    result: SearchResult
    reactivation_ids: frozenset[UUID]


async def _search(
    container: ApplicationContainer,
    *,
    query: SearchExperiences,
    key: str,
    adapter: ExperienceRetrievalAdapter | None = None,
) -> _SearchObservation:
    retained_adapter = adapter or container.retrieval_adapter
    result = await retained_adapter.search(query=query, idempotency_key=key)
    data = _response_document(result, expected_status=200)
    parsed = SearchResult.model_validate_json(canonical_json_bytes(data))
    request = _search_request(query, key=key)
    async with container.database.read_session() as session:
        receipt = await container.receipt_store.find_for_request(
            session=session,
            request=request,
        )
        if receipt is None:
            raise RuntimeError("Benchmark search receipt is missing")
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.causation_id == receipt.receipt_id,
                        DomainEventRow.event_type == ExperienceReactivatedV1.event_type,
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
    reactivation_ids: set[UUID] = set()
    expected_query_hash = retrieval_query_hash(query)
    for row in rows:
        payload = container.event_registry.decode(
            event_type=row.event_type,
            payload=row.payload,
        )
        if (
            not isinstance(payload, ExperienceReactivatedV1)
            or payload.experience_id != row.aggregate_id
            or payload.query_hash != expected_query_hash
            or payload.mode != query.mode.value
        ):
            raise RuntimeError("Benchmark reactivation event does not match its search")
        reactivation_ids.add(payload.experience_id)
    return _SearchObservation(
        result=parsed,
        reactivation_ids=frozenset(reactivation_ids),
    )


def _search_query(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    owner_label: str,
    query: str,
    mode: Any,
) -> SearchExperiences:
    return SearchExperiences(
        owner_agent_id=snapshot.index.agent_ids[owner_label],
        query=query,
        mode=mode,
        limit=snapshot.seed.config.retrieval_limit,
        content_budget_bytes=snapshot.seed.config.content_budget_bytes,
        expand_cold=snapshot.seed.config.expand_cold,
    )


def _label_for_hit(
    snapshot: ClosedBenchmarkSnapshot,
    hit: SearchHit,
    *,
    additional_labels: Mapping[UUID, str] | None = None,
) -> str:
    label = snapshot.index.labels_by_experience_id.get(hit.experience.experience_id)
    if label is None and additional_labels is not None:
        label = additional_labels.get(hit.experience.experience_id)
    if label is None:
        raise RuntimeError("Search returned an experience without a stable label")
    return label


def _hit_document(
    snapshot: ClosedBenchmarkSnapshot,
    hit: SearchHit,
    *,
    rank: int,
    additional_labels: Mapping[UUID, str] | None = None,
) -> dict[str, Any]:
    return {
        "activation": _quantized(hit.activation),
        "expanded": hit.expanded,
        "label": _label_for_hit(
            snapshot,
            hit,
            additional_labels=additional_labels,
        ),
        "mechanism_relevance": _quantized(hit.mechanism_relevance),
        "rank": rank,
        "ranking_relevance": _quantized(hit.ranking_relevance),
        "reactivated": hit.reactivated,
        "score": _quantized(hit.score),
    }


def _returned_labels(
    snapshot: ClosedBenchmarkSnapshot,
    observation: _SearchObservation,
    *,
    additional_labels: Mapping[UUID, str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        _label_for_hit(snapshot, hit, additional_labels=additional_labels)
        for hit in observation.result.hits
    )


def _credited_cold_labels(
    snapshot: ClosedBenchmarkSnapshot,
    observation: _SearchObservation,
    *,
    expected: frozenset[str],
) -> tuple[str, ...]:
    credited: list[str] = []
    for hit in observation.result.hits[:5]:
        label = _label_for_hit(snapshot, hit)
        if (
            label in expected
            and hit.expanded
            and hit.reactivated
            and hit.experience.experience_id in observation.reactivation_ids
        ):
            credited.append(label)
    return tuple(credited)


async def _run_retrieval_case(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: RetrievalCase,
    destination: Path,
) -> tuple[dict[str, Any], tuple[frozenset[str], tuple[str, ...]]]:
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        observation = await _search(
            container,
            query=_search_query(
                snapshot,
                owner_label=case.owner_label,
                query=case.query,
                mode=case.mode,
            ),
            key=f"benchmark-{case.id}-search",
        )
    returned = _returned_labels(snapshot, observation)
    recall = recall_at_five(case.relevant_labels, returned)
    document = {
        "case_id": case.id,
        "credited_labels": sorted(case.relevant_labels.intersection(returned[:5])),
        "expected_labels": sorted(case.relevant_labels),
        "hits": [
            _hit_document(snapshot, hit, rank=rank)
            for rank, hit in enumerate(observation.result.hits, start=1)
        ],
        "recall_at_5": _quantized(recall),
        "type": "retrieval",
    }
    return document, (case.relevant_labels, returned)


async def _run_cold_case(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: ColdCueCase,
    destination: Path,
    baseline_destination: Path,
) -> tuple[
    dict[str, Any],
    tuple[frozenset[str], tuple[str, ...]],
    tuple[frozenset[str], tuple[str, ...]],
]:
    query = _search_query(
        snapshot,
        owner_label=case.owner_label,
        query=case.query,
        mode=case.mode,
    )
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        system = await _search(
            container,
            query=query,
            key=f"benchmark-{case.id}-system",
        )

    clone_closed_database(snapshot, baseline_destination)
    baseline_runtime = ApplicationRuntime(
        _settings(baseline_destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with baseline_runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        baseline = await _search(
            container,
            query=query,
            key=f"benchmark-{case.id}-baseline",
            adapter=_hot_warm_only_adapter(container),
        )

    system_credited = _credited_cold_labels(
        snapshot,
        system,
        expected=case.expected_reactivated_labels,
    )
    baseline_returned = _returned_labels(snapshot, baseline)
    baseline_credited = tuple(
        label for label in baseline_returned[:5] if label in case.relevant_labels
    )
    document = {
        "baseline_recall_at_5": _quantized(
            recall_at_five(case.relevant_labels, baseline_credited)
        ),
        "case_id": case.id,
        "expected_labels": sorted(case.relevant_labels),
        "hot_warm_baseline": {
            "credited_labels": list(baseline_credited),
            "hits": [
                _hit_document(snapshot, hit, rank=rank)
                for rank, hit in enumerate(baseline.result.hits, start=1)
            ],
        },
        "recall_at_5": _quantized(
            recall_at_five(case.relevant_labels, system_credited)
        ),
        "system": {
            "credited_labels": list(system_credited),
            "hits": [
                _hit_document(snapshot, hit, rank=rank)
                for rank, hit in enumerate(system.result.hits, start=1)
            ],
            "reactivation_transitions": [
                {
                    "label": label,
                    "state": ["cold", "reactivated"],
                }
                for label in system_credited
            ],
        },
        "type": "cold_cue",
    }
    return (
        document,
        (case.relevant_labels, system_credited),
        (case.relevant_labels, baseline_credited),
    )


async def _run_distractor_case(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: IrrelevantDistractorCase,
    destination: Path,
) -> tuple[dict[str, Any], int]:
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        observation = await _search(
            container,
            query=_search_query(
                snapshot,
                owner_label=case.owner_label,
                query=case.query,
                mode=case.mode,
            ),
            key=f"benchmark-{case.id}-search",
        )
    cold_ids = frozenset(
        snapshot.index.experience_ids[experience.label]
        for experience in snapshot.seed.experiences
        if experience.target_temperature is Temperature.COLD
    )
    response_false_ids = frozenset(
        hit.experience.experience_id
        for hit in observation.result.hits
        if hit.experience.experience_id in cold_ids
        and (hit.expanded or hit.reactivated)
    )
    false_ids = response_false_ids | (
        observation.reactivation_ids.intersection(cold_ids)
    )
    labels_by_id = snapshot.index.labels_by_experience_id
    false_labels = sorted(labels_by_id[value] for value in false_ids)
    document = {
        "case_id": case.id,
        "false_reactivated_labels": false_labels,
        "false_reactivation_count": len(false_ids),
        "hits": [
            _hit_document(snapshot, hit, rank=rank)
            for rank, hit in enumerate(observation.result.hits, start=1)
        ],
        "type": "irrelevant_distractor",
    }
    return document, len(false_ids)


async def _create_topic(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    key: str,
) -> UUID:
    command = CreateTopic(
        owner_agent_id=owner_agent_id,
        name="Benchmark propagation",
        description="Deterministic quarantined propagation case.",
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="topic.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/topics",
        body={
            "description": command.description,
            "name": command.name,
            "owner_agent_id": command.owner_agent_id,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_topic(
            uow=uow,
            command=command,
            command_context=context,
        )

    data = _response_document(
        await _execute(container, request, handler),
        expected_status=201,
    )
    return UUID(cast(str, data["topic_id"]))


async def _create_subscription(
    container: ApplicationContainer,
    *,
    subscriber_agent_id: UUID,
    topic_id: UUID,
    key: str,
) -> None:
    command = CreateSubscription(
        subscriber_agent_id=subscriber_agent_id,
        topic_id=topic_id,
    )
    request = CommandRequest(
        caller_scope=f"agent:{subscriber_agent_id}",
        operation_scope="subscription.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/subscriptions",
        path_parameters={"agent_id": subscriber_agent_id},
        body={"topic_id": topic_id},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_subscription(
            uow=uow,
            command=command,
            command_context=context,
        )

    _response_document(
        await _execute(container, request, handler),
        expected_status=201,
    )


async def _publish_capsule(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    topic_id: UUID,
    experience_id: UUID,
    version_id: UUID,
    key: str,
) -> UUID:
    command = PublishCapsule(
        owner_agent_id=owner_agent_id,
        topic_id=topic_id,
        experience_id=experience_id,
        version_id=version_id,
        expires_at=container.clock.now() + timedelta(days=30),
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="capsule.publish",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/capsules",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "experience_id": command.experience_id,
            "expires_at": command.expires_at,
            "parent_adoption_id": command.parent_adoption_id,
            "topic_id": command.topic_id,
            "version_id": command.version_id,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    data = _response_document(
        await _execute(container, request, handler),
        expected_status=201,
    )
    return UUID(cast(str, data["capsule_id"]))


async def _adopt_capsule(
    container: ApplicationContainer,
    *,
    adopter_agent_id: UUID,
    item_id: UUID,
    key: str,
) -> tuple[UUID, UUID, UUID, str]:
    command = AdoptCapsule(
        adopter_agent_id=adopter_agent_id,
        item_id=item_id,
        importance=1.0,
    )
    request = CommandRequest(
        caller_scope=f"agent:{adopter_agent_id}",
        operation_scope="capsule.adopt",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inbox/{item_id}:adopt",
        path_parameters={
            "agent_id": adopter_agent_id,
            "item_id": item_id,
        },
        body={"importance": command.importance},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.adopt_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await _execute(container, request, handler)
    data = _response_document(result, expected_status=200)
    location = result.headers.get("location")
    experience = data.get("experience")
    if location is None or not isinstance(experience, dict):
        raise RuntimeError("Benchmark adoption response is incomplete")
    return (
        UUID(location.rsplit("/", 1)[-1]),
        UUID(cast(str, experience["experience_id"])),
        UUID(cast(str, experience["current_version_id"])),
        cast(str, experience["current_content_hash"]),
    )


async def _run_propagation_case(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: PropagationCase,
    destination: Path,
) -> tuple[dict[str, Any], int, int, int]:
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    sender_id = snapshot.index.agent_ids[case.sender_label]
    recipient_id = snapshot.index.agent_ids[case.recipient_label]
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        topic_id = await _create_topic(
            container,
            owner_agent_id=sender_id,
            key=f"benchmark-{case.id}-topic",
        )
        await _create_subscription(
            container,
            subscriber_agent_id=recipient_id,
            topic_id=topic_id,
            key=f"benchmark-{case.id}-subscription",
        )
        capsule_labels: dict[UUID, str] = {}
        for ordinal, label in enumerate(sorted(case.source_labels), start=1):
            capsule_id = await _publish_capsule(
                container,
                owner_agent_id=sender_id,
                topic_id=topic_id,
                experience_id=snapshot.index.experience_ids[label],
                version_id=snapshot.index.version_ids[label],
                key=f"benchmark-{case.id}-publish-{ordinal}",
            )
            capsule_labels[capsule_id] = label

        async with container.database.read_session() as session:
            pending_page = await container.sharing_query.list_inbox(
                session=session,
                owner_agent_id=recipient_id,
                state=InboxState.PENDING,
                limit=100,
                at=container.clock.now(),
            )
        if len(pending_page.items) != len(case.source_labels):
            raise RuntimeError("Propagation case did not deliver every capsule")

        query = _search_query(
            snapshot,
            owner_label=case.recipient_label,
            query=case.query,
            mode=RetrievalMode.FOCUSED,
        )
        pending_search = await _search(
            container,
            query=query,
            key=f"benchmark-{case.id}-pending-search",
        )
        pending_returned = _returned_labels(snapshot, pending_search)
        source_hashes = frozenset(
            snapshot.index.content_hashes[label] for label in case.source_labels
        )
        pending_leaks = tuple(
            hit
            for hit in pending_search.result.hits
            if hit.experience.content_hash in source_hashes
        )

        adopted_labels_by_id: dict[UUID, str] = {}
        provenance_results: list[dict[str, Any]] = []
        complete_count = 0
        for ordinal, item in enumerate(pending_page.items, start=1):
            source_label = capsule_labels[item.capsule_id]
            (
                adoption_id,
                resulting_experience_id,
                _,
                resulting_content_hash,
            ) = await _adopt_capsule(
                container,
                adopter_agent_id=recipient_id,
                item_id=item.item_id,
                key=f"benchmark-{case.id}-adopt-{ordinal}",
            )
            adopted_labels_by_id[resulting_experience_id] = source_label
            async with container.database.read_session() as session:
                capsule = await container.sharing_repository.get_owned_capsule(
                    session=session,
                    publisher_agent_id=sender_id,
                    capsule_id=item.capsule_id,
                )
                adoption = await container.sharing_repository.get_owned_adoption(
                    session=session,
                    adopter_agent_id=recipient_id,
                    adoption_id=adoption_id,
                )
                parent = await container.sharing_repository.get_owned_parent_adoption(
                    session=session,
                    adopter_agent_id=recipient_id,
                    adoption_id=adoption_id,
                )
                source = await container.experience_query.get_owned_shareable_version(
                    session=session,
                    owner_agent_id=sender_id,
                    experience_id=snapshot.index.experience_ids[source_label],
                    version_id=snapshot.index.version_ids[source_label],
                )
                resulting = await container.experience_query.get_owned_retrieval_record(
                    session=session,
                    owner_agent_id=recipient_id,
                    experience_id=resulting_experience_id,
                )
            expected_root = compute_original_root_fingerprint(
                root_publisher_id=sender_id,
                source_content_hash=snapshot.index.content_hashes[source_label],
            )
            complete = (
                capsule is not None
                and adoption is not None
                and parent is not None
                and resulting is not None
                and capsule.source_experience_id
                == snapshot.index.experience_ids[source_label]
                and capsule.source_version_id
                == snapshot.index.version_ids[source_label]
                and capsule.source_content_hash == source.content_hash
                and source.content_hash == snapshot.index.content_hashes[source_label]
                and resulting.current_content_hash == source.content_hash
                and resulting_content_hash == source.content_hash
                and adoption.resulting_experience_id == resulting_experience_id
                and adoption.provenance_chain == parent.provenance_chain
                and len(adoption.provenance_chain) == 1
                and adoption.provenance_chain[0].capsule_id == capsule.capsule_id
                and adoption.provenance_chain[0].publisher_agent_id == sender_id
                and adoption.root_fingerprint
                == parent.root_fingerprint
                == capsule.root_fingerprint
                == expected_root
            )
            complete_count += int(complete)
            provenance_results.append(
                {
                    "capsule": f"capsule#{ordinal}",
                    "complete": complete,
                    "publisher": case.sender_label,
                    "source_label": source_label,
                }
            )

        adopted_search = await _search(
            container,
            query=query,
            key=f"benchmark-{case.id}-adopted-search",
        )

    adopted_returned = _returned_labels(
        snapshot,
        adopted_search,
        additional_labels=adopted_labels_by_id,
    )
    document = {
        "adopted": {
            "credited_labels": sorted(
                case.adopted_relevant_labels.intersection(adopted_returned[:5])
            ),
            "provenance": provenance_results,
            "retrieved_labels": list(adopted_returned),
            "state_transition": ["pending", "adopted"],
        },
        "case_id": case.id,
        "pending": {
            "leakage_count": len(pending_leaks),
            "retrieved_labels": list(pending_returned),
            "state_transition": ["absent", "pending"],
        },
        "type": "propagation",
    }
    return document, len(pending_leaks), complete_count, len(case.source_labels)


def _inspiration_request(
    run: StartInspirationRun,
    *,
    key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{run.owner_agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": run.owner_agent_id},
        body={
            "branches_per_operator": run.branches_per_operator,
            "context": run.context,
            "generator": run.generator.value,
            "global_timeout_seconds": run.global_timeout_seconds,
            "goal": run.goal,
            "include_inbox": run.include_inbox,
            "mode": run.mode.value,
            "operator_timeout_seconds": run.operator_timeout_seconds,
            "operators": tuple(operator.value for operator in run.operators),
            "output_tokens_per_operator": run.output_tokens_per_operator,
            "total_output_tokens": run.total_output_tokens,
        },
    )


def _inspiration_command(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: InspirationCase,
) -> StartInspirationRun:
    settings = snapshot.seed.config
    return StartInspirationRun(
        owner_agent_id=snapshot.index.agent_ids[case.owner_label],
        goal=case.goal,
        context=case.context,
        mode=case.mode,
        generator=settings.generator,
        operators=settings.operators,
        include_inbox=settings.include_inbox,
        branches_per_operator=settings.branches_per_operator,
        output_tokens_per_operator=settings.output_tokens_per_operator,
        total_output_tokens=settings.total_output_tokens,
        operator_timeout_seconds=settings.operator_timeout_seconds,
        global_timeout_seconds=settings.global_timeout_seconds,
    )


async def _execute_inspiration_run(
    container: ApplicationContainer,
    *,
    command: StartInspirationRun,
    key: str,
) -> tuple[InspirationRun, tuple[Idea, ...]]:
    response = await container.inspiration_run_executor.execute(
        request=_inspiration_request(command, key=key),
        run=command,
    )
    data = _response_document(response, expected_status=201)
    run_id = UUID(cast(str, data["run_id"]))
    async with container.database.read_session() as session:
        run = await container.inspiration_repository.get_run(
            session=session,
            run_id=run_id,
        )
        ideas = await container.inspiration_repository.list_owned_ideas(
            session=session,
            owner_agent_id=command.owner_agent_id,
            run_id=run_id,
            after=None,
            limit=101,
        )
    if run is None or run.snapshot_hash is None:
        raise RuntimeError("Benchmark inspiration run has no frozen snapshot")
    return run, ideas


async def _validate_idea(
    container: ApplicationContainer,
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: InspirationCase,
    idea: Idea,
) -> tuple[bool, tuple[str, ...]]:
    try:
        Idea.model_validate(
            idea.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except ValueError:
        return False, ()
    if (
        hash_idea_content(idea.draft) != idea.idea_content_hash
        or hash_mechanism(idea.draft.mechanism) != idea.mechanism_hash
    ):
        return False, ()
    references = idea.draft.evidence
    if len({reference.id for reference in references}) != len(references):
        return False, ()
    if len({reference.stable_evidence_key for reference in references}) != len(
        references
    ):
        return False, ()

    labels_by_id = snapshot.index.labels_by_experience_id
    evidence_labels: list[str] = []
    async with container.database.read_session() as session:
        for reference in references:
            row = await session.get(InspirationSnapshotItemRow, reference.id)
            if (
                row is None
                or row.run_id != idea.run_id
                or row.stable_evidence_key != reference.stable_evidence_key
            ):
                return False, ()
            try:
                source_type = EvidenceSourceType(row.source_type)
            except ValueError:
                return False, ()
            if (
                stable_evidence_key(
                    source_type=source_type,
                    source_id=row.source_id,
                    source_version_id=row.source_version_id,
                    content_hash=row.content_hash,
                )
                != row.stable_evidence_key
                or source_type is not EvidenceSourceType.EXPERIENCE
            ):
                return False, ()
            label = labels_by_id.get(row.source_id)
            if (
                label is None
                or snapshot.index.version_ids[label] != row.source_version_id
                or snapshot.index.content_hashes[label] != row.content_hash
            ):
                return False, ()
            source = await container.experience_query.get_owned_shareable_version(
                session=session,
                owner_agent_id=snapshot.index.agent_ids[case.owner_label],
                experience_id=row.source_id,
                version_id=row.source_version_id,
            )
            if source.content_hash != row.content_hash:
                return False, ()
            evidence_labels.append(label)
    return True, tuple(dict.fromkeys(evidence_labels))


async def _run_inspiration_case(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: InspirationCase,
    destination: Path,
) -> tuple[dict[str, Any], int, int, frozenset[str], int]:
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        run, ideas = await _execute_inspiration_run(
            container,
            command=_inspiration_command(snapshot, case=case),
            key=f"benchmark-{case.id}-run",
        )
        async with container.database.read_session() as session:
            snapshot_rows = tuple(
                (
                    await session.scalars(
                        select(InspirationSnapshotItemRow)
                        .where(InspirationSnapshotItemRow.run_id == run.run_id)
                        .order_by(InspirationSnapshotItemRow.rank)
                    )
                ).all()
            )
        labels_by_id = snapshot.index.labels_by_experience_id
        snapshot_labels: list[str] = []
        for row in snapshot_rows:
            label = labels_by_id.get(row.source_id)
            if label is None:
                raise RuntimeError(
                    "Owned-only benchmark snapshot has an unknown source"
                )
            snapshot_labels.append(label)
        idea_documents: list[dict[str, Any]] = []
        valid_count = 0
        valid_clusters: set[str] = set()
        cluster_ordinals: dict[str, str] = {}
        for ordinal, idea in enumerate(ideas, start=1):
            valid, evidence_labels = await _validate_idea(
                container,
                snapshot,
                case=case,
                idea=idea,
            )
            if valid:
                valid_count += 1
                valid_clusters.add(idea.mechanism_cluster_id)
            cluster_label = cluster_ordinals.setdefault(
                idea.mechanism_cluster_id,
                f"cluster#{len(cluster_ordinals) + 1}",
            )
            idea_documents.append(
                {
                    "cluster": cluster_label,
                    "content_hash": idea.idea_content_hash,
                    "evidence_labels": list(evidence_labels),
                    "idea": f"idea#{ordinal}",
                    "mechanism_hash": idea.mechanism_hash,
                    "operator": idea.operator.value,
                    "ordinal": idea.ordinal,
                    "valid": valid,
                }
            )

    evidence_coverage_complete = case.evidence_labels.issubset(snapshot_labels)
    fixture_expectations_met = (
        evidence_coverage_complete
        and valid_count >= case.expected_min_valid_ideas
        and len(valid_clusters) >= case.expected_min_distinct_mechanisms
    )
    document = {
        "case_id": case.id,
        "evidence_coverage_complete": evidence_coverage_complete,
        "expected_evidence_labels": sorted(case.evidence_labels),
        "fixture_expectations_met": fixture_expectations_met,
        "ideas": idea_documents,
        "persisted_idea_count": len(ideas),
        "snapshot_evidence_labels": list(snapshot_labels),
        "type": "inspiration",
        "valid_idea_count": valid_count,
    }
    return (
        document,
        len(ideas),
        valid_count,
        frozenset(valid_clusters),
        int(not evidence_coverage_complete),
    )


_MATURITY_ORDER = {
    MechanismMaturity.SPECULATIVE: 0,
    MechanismMaturity.INCUBATING: 1,
    MechanismMaturity.CANDIDATE: 2,
}


async def _run_same_snapshot_incubation(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    case: InspirationCase,
    destination: Path,
) -> tuple[dict[str, Any], int]:
    clone_closed_database(snapshot, destination)
    runtime = ApplicationRuntime(
        _settings(destination),
        clock=FrozenClock(snapshot.seed.clock.cold_at),
        ids=_clone_ids(),
        container_factory=_container_factory(snapshot.seed),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        command = _inspiration_command(snapshot, case=case)
        first_run, first_ideas = await _execute_inspiration_run(
            container,
            command=command,
            key="benchmark-same-snapshot-run-1",
        )
        async with container.database.read_session() as session:
            first_clusters = {
                cluster.state.cluster_id: cluster
                for cluster in (
                    await container.inspiration_repository.load_clusters(
                        session=session
                    )
                )
            }
        second_run, second_ideas = await _execute_inspiration_run(
            container,
            command=command,
            key="benchmark-same-snapshot-run-2",
        )
        async with container.database.read_session() as session:
            second_clusters = {
                cluster.state.cluster_id: cluster
                for cluster in (
                    await container.inspiration_repository.load_clusters(
                        session=session
                    )
                )
            }

    snapshot_equivalent = (
        first_run.run_id != second_run.run_id
        and first_run.snapshot_hash == second_run.snapshot_hash
    )
    first_cluster_order = tuple(
        dict.fromkeys(idea.mechanism_cluster_id for idea in first_ideas)
    )
    second_cluster_ids = frozenset(idea.mechanism_cluster_id for idea in second_ideas)
    transitions: list[dict[str, Any]] = []
    promotion_count = 0
    occurrence_advanced = False
    for ordinal, cluster_id in enumerate(first_cluster_order, start=1):
        before = first_clusters.get(cluster_id)
        after = second_clusters.get(cluster_id)
        if before is None or after is None:
            promotion_count += 1
            continue
        occurrence_advanced = occurrence_advanced or (
            after.state.occurrence_count > before.state.occurrence_count
        )
        promoted = (
            after.state.distinct_snapshot_count != before.state.distinct_snapshot_count
            or _MATURITY_ORDER[after.state.maturity]
            > _MATURITY_ORDER[before.state.maturity]
        )
        promotion_count += int(promoted)
        transitions.append(
            {
                "cluster": f"cluster#{ordinal}",
                "distinct_snapshot_count": [
                    before.state.distinct_snapshot_count,
                    after.state.distinct_snapshot_count,
                ],
                "maturity": [
                    before.state.maturity.value,
                    after.state.maturity.value,
                ],
                "occurrence_count": [
                    before.state.occurrence_count,
                    after.state.occurrence_count,
                ],
            }
        )
    if second_cluster_ids != frozenset(first_cluster_order):
        promotion_count += len(
            second_cluster_ids.symmetric_difference(frozenset(first_cluster_order))
        )
    # A missing recurrence signal is an effectiveness regression, not an
    # infrastructure failure. Keep it in the canonical report so callers get
    # the named gate failure instead of an opaque maintenance exception.
    promotion_count += int(not occurrence_advanced)
    if not snapshot_equivalent:
        promotion_count += 1
    document = {
        "case_id": "same_snapshot_incubation",
        "clusters": transitions,
        "occurrence_advanced": occurrence_advanced,
        "promotion_count": promotion_count,
        "snapshot_equivalent": snapshot_equivalent,
    }
    return document, promotion_count


@dataclass(frozen=True, slots=True)
class _SuiteResult:
    cases: tuple[dict[str, Any], ...]
    same_snapshot_incubation: dict[str, Any]
    focused_recall: Decimal
    cold_recall: Decimal
    cold_baseline_recall: Decimal
    distractor_false_reactivations: int
    pending_leakage_count: int
    adopted_provenance_complete: int
    adopted_provenance_total: int
    persisted_idea_count: int
    valid_idea_count: int
    distinct_mechanism_count: int
    same_snapshot_incubation_promotions: int
    inspiration_evidence_coverage_failures: int

    def effectiveness(
        self,
        *,
        byte_identical_replay: bool,
    ) -> EffectivenessMetrics:
        return EffectivenessMetrics(
            focused_macro_recall_at_five=self.focused_recall,
            cold_macro_recall_at_five=self.cold_recall,
            cold_baseline_macro_recall_at_five=self.cold_baseline_recall,
            distractor_false_reactivations=(self.distractor_false_reactivations),
            pending_leakage_count=self.pending_leakage_count,
            adopted_provenance_complete=self.adopted_provenance_complete,
            adopted_provenance_total=self.adopted_provenance_total,
            valid_idea_count=self.persisted_idea_count,
            idea_schema_evidence_valid_count=self.valid_idea_count,
            distinct_mechanism_count=self.distinct_mechanism_count,
            same_snapshot_incubation_promotions=(
                self.same_snapshot_incubation_promotions
            ),
            byte_identical_replay=byte_identical_replay,
            inspiration_evidence_coverage_failures=(
                self.inspiration_evidence_coverage_failures
            ),
        )


async def _execute_suite(
    snapshot: ClosedBenchmarkSnapshot,
    *,
    workspace: Path,
) -> _SuiteResult:
    workspace.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    focused_cases: list[tuple[frozenset[str], tuple[str, ...]]] = []
    cold_cases: list[tuple[frozenset[str], tuple[str, ...]]] = []
    baseline_cases: list[tuple[frozenset[str], tuple[str, ...]]] = []
    distractor_false_reactivations = 0
    pending_leakage_count = 0
    adopted_provenance_complete = 0
    adopted_provenance_total = 0
    persisted_idea_count = 0
    valid_idea_count = 0
    distinct_mechanisms: set[str] = set()
    inspiration_evidence_coverage_failures = 0
    inspiration_cases: list[InspirationCase] = []

    for case in snapshot.cases:
        destination = workspace / f"{case.id}.sqlite3"
        if isinstance(case, RetrievalCase):
            document, metric_case = await _run_retrieval_case(
                snapshot,
                case=case,
                destination=destination,
            )
            documents.append(document)
            focused_cases.append(metric_case)
        elif isinstance(case, ColdCueCase):
            document, metric_case, baseline_case = await _run_cold_case(
                snapshot,
                case=case,
                destination=destination,
                baseline_destination=workspace / f"{case.id}-hot-warm-baseline.sqlite3",
            )
            documents.append(document)
            cold_cases.append(metric_case)
            baseline_cases.append(baseline_case)
        elif isinstance(case, IrrelevantDistractorCase):
            document, false_count = await _run_distractor_case(
                snapshot,
                case=case,
                destination=destination,
            )
            documents.append(document)
            distractor_false_reactivations += false_count
        elif isinstance(case, PropagationCase):
            (
                document,
                leakage_count,
                complete_count,
                total_count,
            ) = await _run_propagation_case(
                snapshot,
                case=case,
                destination=destination,
            )
            documents.append(document)
            pending_leakage_count += leakage_count
            adopted_provenance_complete += complete_count
            adopted_provenance_total += total_count
        elif isinstance(case, InspirationCase):
            (
                document,
                persisted,
                valid,
                clusters,
                evidence_coverage_failures,
            ) = await _run_inspiration_case(
                snapshot,
                case=case,
                destination=destination,
            )
            documents.append(document)
            inspiration_cases.append(case)
            persisted_idea_count += persisted
            valid_idea_count += valid
            distinct_mechanisms.update(clusters)
            inspiration_evidence_coverage_failures += (
                evidence_coverage_failures
            )
        else:
            raise TypeError(f"Unsupported benchmark case: {type(case).__name__}")

    if not inspiration_cases:
        raise RuntimeError("Benchmark requires at least one inspiration case")
    incubation, incubation_promotions = await _run_same_snapshot_incubation(
        snapshot,
        case=inspiration_cases[0],
        destination=workspace / "same-snapshot-incubation.sqlite3",
    )
    return _SuiteResult(
        cases=tuple(documents),
        same_snapshot_incubation=incubation,
        focused_recall=macro_recall_at_five(tuple(focused_cases)),
        cold_recall=macro_recall_at_five(tuple(cold_cases)),
        cold_baseline_recall=macro_recall_at_five(tuple(baseline_cases)),
        distractor_false_reactivations=distractor_false_reactivations,
        pending_leakage_count=pending_leakage_count,
        adopted_provenance_complete=adopted_provenance_complete,
        adopted_provenance_total=adopted_provenance_total,
        persisted_idea_count=persisted_idea_count,
        valid_idea_count=valid_idea_count,
        distinct_mechanism_count=len(distinct_mechanisms),
        same_snapshot_incubation_promotions=incubation_promotions,
        inspiration_evidence_coverage_failures=(
            inspiration_evidence_coverage_failures
        ),
    )


def _metrics_document(metrics: EffectivenessMetrics) -> dict[str, Any]:
    return {
        "adopted_provenance_complete_ratio": _quantized(metrics.provenance_ratio),
        "byte_identical_replay": metrics.byte_identical_replay,
        "cold_baseline_macro_recall_at_5": _quantized(
            metrics.cold_baseline_macro_recall_at_five
        ),
        "cold_macro_recall_at_5": _quantized(metrics.cold_macro_recall_at_five),
        "cold_recall_gain": _quantized(metrics.cold_recall_gain),
        "distractor_false_reactivation_count": (metrics.distractor_false_reactivations),
        "distinct_mechanism_count": metrics.distinct_mechanism_count,
        "focused_macro_recall_at_5": _quantized(metrics.focused_macro_recall_at_five),
        "idea_schema_evidence_valid_ratio": _quantized(
            metrics.effective_idea_validity_ratio
        ),
        "inspiration_evidence_coverage_failure_count": (
            metrics.inspiration_evidence_coverage_failures
        ),
        "pending_capsule_leakage_count": metrics.pending_leakage_count,
        "persisted_idea_count": metrics.valid_idea_count,
        "same_snapshot_incubation_promotion_count": (
            metrics.same_snapshot_incubation_promotions
        ),
        "unique_mechanism_ratio": _quantized(metrics.mechanism_ratio),
        "valid_idea_count": metrics.idea_schema_evidence_valid_count,
    }


def _core_document(suite: _SuiteResult) -> dict[str, Any]:
    metrics = suite.effectiveness(byte_identical_replay=False)
    document = _metrics_document(metrics)
    document.pop("byte_identical_replay")
    return {
        "cases": list(suite.cases),
        "metrics": document,
        "same_snapshot_incubation": suite.same_snapshot_incubation,
        "schema_version": 1,
    }


def _report_document(
    suite: _SuiteResult,
    *,
    byte_identical_replay: bool,
) -> dict[str, Any]:
    metrics = suite.effectiveness(byte_identical_replay=byte_identical_replay)
    gates = evaluate_effectiveness_gates(metrics)
    failed = [gate.name for gate in gates if not gate.passed]
    return {
        "data": {
            "cases": list(suite.cases),
            "failed_gates": failed,
            "gates": [gate.document() for gate in gates],
            "metrics": _metrics_document(metrics),
            "passed": not failed,
            "same_snapshot_incubation": suite.same_snapshot_incubation,
            "schema_version": 1,
        }
    }


def _reset_owned_workspace(path: Path, *, allow_unmarked: bool) -> None:
    if path.is_symlink():
        raise BenchmarkIsolationError("Benchmark workspace cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise BenchmarkIsolationError("Benchmark workspace must be a directory")
    if path.exists():
        entries = tuple(path.iterdir())
        unexpected = sorted(
            entry.name for entry in entries if entry.name not in _WORKSPACE_ENTRIES
        )
        if unexpected:
            raise BenchmarkIsolationError(
                "Benchmark workspace contains files not owned by the benchmark"
            )
        marker = path / ".experience-hub-benchmark-workspace"
        if entries and not marker.exists() and not allow_unmarked:
            raise BenchmarkIsolationError(
                "Existing custom benchmark workspace has no ownership marker"
            )
        if marker.exists():
            try:
                marker_body = marker.read_text(encoding="utf-8")
            except OSError as error:
                raise BenchmarkIsolationError(
                    "Benchmark workspace ownership marker is unreadable"
                ) from error
            if marker_body != _WORKSPACE_MARKER:
                raise BenchmarkIsolationError(
                    "Benchmark workspace ownership marker is invalid"
                )
        for name in sorted(_WORKSPACE_ENTRIES):
            entry = path / name
            if entry.is_symlink() or entry.is_file():
                entry.unlink(missing_ok=True)
            elif entry.is_dir():
                shutil.rmtree(entry)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".experience-hub-benchmark-workspace").write_text(
        _WORKSPACE_MARKER,
        encoding="utf-8",
    )


async def run_benchmark(
    *,
    seed_path: Path | None = None,
    cases_path: Path | None = None,
    workspace: Path | None = None,
) -> BenchmarkExecution:
    """Run two complete fresh-clone suites and return one canonical report."""
    root = config.repository_root()
    retained_seed_path = seed_path or root / "benchmarks" / "seed.json"
    retained_cases_path = cases_path or root / "benchmarks" / "cases.jsonl"
    default_workspace = root / ".data" / "benchmark"
    retained_workspace = workspace or default_workspace
    _reset_owned_workspace(
        retained_workspace,
        allow_unmarked=(
            retained_workspace.resolve() == default_workspace.resolve()
        ),
    )

    snapshot = await prepare_benchmark_snapshot(
        seed_path=retained_seed_path,
        cases_path=retained_cases_path,
        workspace=retained_workspace / "snapshot",
    )
    first = await _execute_suite(
        snapshot,
        workspace=retained_workspace / "replay-a",
    )
    second = await _execute_suite(
        snapshot,
        workspace=retained_workspace / "replay-b",
    )
    first_core = canonical_benchmark_bytes(_core_document(first))
    second_core = canonical_benchmark_bytes(_core_document(second))
    replay_equal = first_core == second_core

    first_report = _report_document(
        first,
        byte_identical_replay=replay_equal,
    )
    second_report = _report_document(
        second,
        byte_identical_replay=replay_equal,
    )
    first_body = canonical_benchmark_bytes(first_report)
    second_body = canonical_benchmark_bytes(second_report)
    replay_equal = replay_equal and first_body == second_body
    if not replay_equal:
        first_report = _report_document(
            first,
            byte_identical_replay=False,
        )
        first_body = canonical_benchmark_bytes(first_report)

    data = cast(Mapping[str, Any], first_report["data"])
    failed_gates = tuple(cast(list[str], data["failed_gates"]))
    return BenchmarkExecution(
        report=MappingProxyType(first_report),
        body=first_body,
        passed=not failed_gates,
        failed_gates=failed_gates,
    )


__all__ = [
    "BenchmarkExecution",
    "BenchmarkIsolationError",
    "BenchmarkOutputError",
    "ClosedBenchmarkSnapshot",
    "EffectivenessMetrics",
    "GateResult",
    "SeedIndex",
    "canonical_benchmark_bytes",
    "clone_closed_database",
    "evaluate_effectiveness_gates",
    "prepare_benchmark_snapshot",
    "run_benchmark",
]
