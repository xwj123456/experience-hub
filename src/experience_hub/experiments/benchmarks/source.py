"""Deterministically build an owned SQLite source for ExperienceBench-S."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID

from sqlalchemy.engine import URL

from experience_hub.agents.models import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, TypedEvidence
from experience_hub.experiences.candidate_models import CandidateDecision
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.models import VersionContent
from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkSourceCandidateV1,
    BenchmarkSourceExperienceV1,
)
from experience_hub.experiments.benchmarks.loading import LoadedBenchmarkPack
from experience_hub.experiments.errors import ExperimentInputError
from experience_hub.experiments.snapshots import (
    FrozenSqliteSnapshot,
    checkpoint_owned_sqlite,
    freeze_closed_sqlite,
    verify_source_unchanged,
)
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    OwnedWorkspace,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.unit_of_work import UnitOfWork

PILOT_LIFECYCLE_CONFIG = LifecycleConfig(
    recency_half_life_hours=168.0,
    frequency_half_life_hours=336.0,
    warm_to_hot_threshold=0.75,
    hot_to_warm_threshold=0.62,
    warm_to_cold_threshold=0.3,
    demotion_cycles=2,
    archive_after_days=90.0,
    archive_importance_threshold=0.75,
    archive_confidence_threshold=0.25,
    archive_strength_threshold=0.1,
    minimum_cycle_interval_seconds=900.0,
    worker_interval_seconds=900.0,
    lease_duration_seconds=300.0,
)


@dataclass(frozen=True, slots=True)
class BenchmarkSourceIndex:
    agent_ids: Mapping[str, UUID]
    experience_ids: Mapping[str, UUID]
    candidate_ids: Mapping[str, UUID]
    labels_by_experience_id: Mapping[UUID, str]
    content_bytes_by_label: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BuiltBenchmarkSource:
    path: Path
    snapshot: FrozenSqliteSnapshot
    index: BenchmarkSourceIndex
    schema_revision: int


type CommandHandler = Callable[[UnitOfWork, CommandContext], Awaitable[StoredResponse]]


def search_document_bytes(record: BenchmarkSourceExperienceV1) -> bytes:
    """Encode exactly the source document evaluated by retrieval baselines."""
    return (
        record.summary
        + "\n"
        + record.mechanism
        + "\n"
        + " ".join(record.tags)
        + "\n"
        + " ".join(record.applicability)
    ).encode("utf-8")


def _invalid() -> ExperimentInputError:
    return ExperimentInputError(
        "benchmark_source_invalid", "benchmark source could not be constructed"
    )


def _ids_for(
    pack_id: str,
    label: str,
    record_type: str,
    *,
    count: int = 32,
) -> tuple[UUID, ...]:
    return tuple(
        UUID(
            bytes=hashlib.sha256(
                f"{pack_id}\x1f{label}\x1f{record_type}\x1f{ordinal}".encode()
            ).digest()[:16]
        )
        for ordinal in range(1, count + 1)
    )


def _source_ids(pack: LoadedBenchmarkPack) -> SequenceIdGenerator:
    values: list[UUID] = []
    for record in pack.source:
        values.extend(_ids_for(pack.manifest.pack_id, record.label, record.record_type))
    return SequenceIdGenerator(tuple(values))


def _settings(path: Path) -> Settings:
    url = URL.create("sqlite+aiosqlite", database=str(path))
    return Settings(database_url=url.render_as_string(hide_password=False))


def _advance_to(clock: FrozenClock, target: datetime) -> None:
    delta = target - clock.now()
    if delta <= timedelta(0):
        raise _invalid()
    clock.advance(delta)


def _response_data(result: CommandResult, expected_status: int) -> dict[str, Any]:
    if result.status_code != expected_status:
        raise _invalid()
    try:
        parsed = json.loads(result.body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _invalid() from None
    if canonical_json_bytes(parsed) != result.body:
        raise _invalid()
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), dict):
        raise _invalid()
    return cast(dict[str, Any], parsed["data"])


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
    ordinal: int,
) -> UUID:
    request = CommandRequest(
        caller_scope="system:benchmark",
        operation_scope="agent.create",
        idempotency_key=f"source-agent-{ordinal}",
        method="POST",
        route_template="/v1/agents",
        body={"name": label},
    )

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow, command=CreateAgent(name=label), command_context=context
        )

    data = _response_data(await _execute(container, request, handler), 201)
    value = data.get("agent_id")
    if not isinstance(value, str):
        raise _invalid()
    return UUID(value)


async def _create_experience(
    container: ApplicationContainer,
    *,
    record: BenchmarkSourceExperienceV1,
    owner_agent_id: UUID,
    ordinal: int,
) -> UUID:
    content = VersionContent(
        body=record.body,
        summary=record.summary,
        mechanism=record.mechanism,
        tags=record.tags,
        applicability=record.applicability,
        evidence=tuple(
            TypedEvidence(type=item.type, id=item.label) for item in record.evidence
        ),
        falsifiers=record.falsifiers,
    )
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=record.kind,
        content=content,
        importance=record.importance_micros / 1_000_000,
        confidence=record.confidence_micros / 1_000_000,
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="experience.create",
        idempotency_key=f"source-experience-{ordinal}",
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": owner_agent_id},
        body={
            **content.model_dump(mode="python", warnings=False),
            "confidence": command.confidence,
            "importance": command.importance,
            "kind": command.kind,
            "links": (),
        },
    )

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await container.experience_service.create(
            uow=uow, command=command, command_context=context
        )

    data = _response_data(await _execute(container, request, handler), 201)
    value = data.get("experience_id")
    if not isinstance(value, str):
        raise _invalid()
    return UUID(value)


def _candidate_jsonl(
    record: BenchmarkSourceCandidateV1,
    *,
    owner_agent_id: UUID,
) -> bytes:
    timestamp = record.created_at.isoformat().replace("+00:00", "Z")
    return b"\n".join(
        (
            canonical_json_bytes(
                {
                    "adapter": {"kind": "generic_jsonl", "version": 1},
                    "owner_agent_id": owner_agent_id,
                    "record_type": "header",
                    "sanitization": {
                        "input_sanitized": True,
                        "profile_id": "experiencebench-s-v1",
                    },
                    "schema_version": 1,
                    "source_completed_at": timestamp,
                    "source_started_at": timestamp,
                    "trajectory_id": f"experiencebench-{record.label}",
                }
            ),
            canonical_json_bytes(
                {
                    "action": "Recorded a successful deterministic source step.",
                    "candidate_signal": {
                        "applicability": record.applicability,
                        "body": record.body,
                        "evidence": [{"field": "observation", "step_id": "step-1"}],
                        "falsifiers": record.falsifiers,
                        "kind": record.kind,
                        "mechanism": record.mechanism,
                        "summary": record.summary,
                        "tags": record.tags,
                    },
                    "observation": record.body,
                    "occurred_at": timestamp,
                    "ordinal": 1,
                    "outcome": "The deterministic source step succeeded.",
                    "record_type": "step",
                    "status": "succeeded",
                    "step_id": "step-1",
                }
            ),
        )
    )


async def _capture_candidate(
    container: ApplicationContainer,
    *,
    record: BenchmarkSourceCandidateV1,
    owner_agent_id: UUID,
    ordinal: int,
) -> UUID:
    prepared = container.capture_preparer.prepare_jsonl(
        _candidate_jsonl(record, owner_agent_id=owner_agent_id)
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=TRAJECTORY_IMPORT_SCOPE,
        idempotency_key=f"source-candidate-{ordinal}",
        method="POST",
        route_template="/v1/agents/{agent_id}/trajectory-bundles",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "adapter": "generic_jsonl",
            "candidate_count": len(prepared.candidates),
            "manifest_hash": prepared.bundle.manifest_hash,
        },
    )

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await container.capture_service.capture(
            uow=uow, prepared=prepared, command=context
        )

    data = _response_data(await _execute(container, request, handler), 201)
    candidate_ids = data.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != 1
        or not isinstance(candidate_ids[0], str)
    ):
        raise _invalid()
    return UUID(candidate_ids[0])


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

    async def handler(uow: UnitOfWork, context: CommandContext) -> StoredResponse:
        return await container.lifecycle_service.run(
            uow=uow, evaluated_at=evaluated_at, command=context, mode="manual"
        )

    _response_data(await _execute(container, request, handler), 200)


async def _complete_threaded[T](operation: Callable[..., T], *arguments: object) -> T:
    """Keep cancellation from outliving a worker that owns the source path."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *arguments))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
            continue
        except Exception:
            if cancellation is None:
                raise
            break
        if cancellation is not None:
            raise cancellation
        return result
    assert cancellation is not None
    raise cancellation


async def build_benchmark_source(
    pack: LoadedBenchmarkPack,
    workspace: OwnedWorkspace,
) -> BuiltBenchmarkSource:
    """Build, verify, close, and freeze a source in the owned snapshot slot."""
    if not isinstance(pack, LoadedBenchmarkPack) or not isinstance(
        workspace, OwnedWorkspace
    ):
        raise _invalid()
    try:
        workspace.require_policy(REPLAY_WORKSPACE_POLICY)
        source_relative = PurePosixPath("snapshot/source.sqlite3")
        reservation = await workspace.reserve_new_file_scoped_async(source_relative)
        with reservation:
            path = reservation.path
            content_records = tuple(
                record for record in pack.source if record.record_type != "agent"
            )
            if not content_records:
                raise _invalid()
            clock = FrozenClock(
                min(record.created_at for record in content_records)
                - timedelta(
                    seconds=1
                    + sum(1 for record in pack.source if record.record_type == "agent")
                )
            )
            runtime = ApplicationRuntime(
                _settings(path),
                clock=clock,
                ids=_source_ids(pack),
                container_factory=partial(
                    ApplicationContainer.build, lifecycle_config=PILOT_LIFECYCLE_CONFIG
                ),
            )
            agent_ids: dict[str, UUID] = {}
            experience_ids: dict[str, UUID] = {}
            candidate_ids: dict[str, UUID] = {}
            content_bytes: dict[str, int] = {}
            async with runtime.initialize(
                start_lifecycle_worker=False, recover_interrupted=False
            ) as container:
                for ordinal, record in enumerate(pack.source, start=1):
                    if record.record_type != "agent":
                        continue
                    clock.advance(timedelta(seconds=1))
                    agent_ids[record.label] = await _create_agent(
                        container, label=record.label, ordinal=ordinal
                    )
                for content_index, record in enumerate(content_records):
                    owner_agent_id = agent_ids.get(record.owner_label)
                    if owner_agent_id is None:
                        raise _invalid()
                    _advance_to(clock, record.created_at)
                    ordinal = pack.source.index(record) + 1
                    if isinstance(record, BenchmarkSourceExperienceV1):
                        experience_ids[record.label] = await _create_experience(
                            container,
                            record=record,
                            owner_agent_id=owner_agent_id,
                            ordinal=ordinal,
                        )
                        content_bytes[record.label] = len(search_document_bytes(record))
                        if record.temperature.value == "archived":
                            first_cycle = record.created_at + timedelta(days=8)
                            second_cycle = first_cycle + timedelta(seconds=901)
                            if (
                                content_index + 1 < len(content_records)
                                and content_records[content_index + 1].created_at
                                <= second_cycle
                            ):
                                raise _invalid()
                            _advance_to(clock, first_cycle)
                            await _run_lifecycle(
                                container,
                                key=f"source-archive-prepare-{content_index}-first",
                            )
                            _advance_to(clock, second_cycle)
                            await _run_lifecycle(
                                container,
                                key=f"source-archive-prepare-{content_index}-second",
                            )
                    elif isinstance(record, BenchmarkSourceCandidateV1):
                        candidate_ids[record.label] = await _capture_candidate(
                            container,
                            record=record,
                            owner_agent_id=owner_agent_id,
                            ordinal=ordinal,
                        )
                    else:
                        raise _invalid()

                _advance_to(clock, pack.manifest.frozen_at - timedelta(seconds=901))
                await _run_lifecycle(container, key="source-lifecycle-first")
                _advance_to(clock, pack.manifest.frozen_at)
                await _run_lifecycle(container, key="source-lifecycle-final")
                async with container.database.read_session() as session:
                    for record in pack.source:
                        if not isinstance(record, BenchmarkSourceExperienceV1):
                            continue
                        retrieval = (
                            await container.experience_query.get_owned_retrieval_record(
                                session=session,
                                owner_agent_id=agent_ids[record.owner_label],
                                experience_id=experience_ids[record.label],
                            )
                        )
                        if (
                            retrieval is None
                            or retrieval.state.temperature != record.temperature
                        ):
                            raise _invalid()
                    for record in pack.source:
                        if not isinstance(record, BenchmarkSourceCandidateV1):
                            continue
                        candidate_id = candidate_ids[record.label]
                        owner_agent_id = agent_ids[record.owner_label]
                        candidate = await container.candidate_service.get_owned(
                            session=session,
                            owner_agent_id=owner_agent_id,
                            candidate_id=candidate_id,
                        )
                        candidate_as_experience = (
                            await container.experience_query.get_owned_retrieval_record(
                                session=session,
                                owner_agent_id=owner_agent_id,
                                experience_id=candidate_id,
                            )
                        )
                        if (
                            candidate.candidate_id != candidate_id
                            or candidate.owner_agent_id != owner_agent_id
                            or candidate.decision is not CandidateDecision.PENDING
                            or candidate_as_experience is not None
                        ):
                            raise _invalid()
                verification = await container.projection_manager.verify(
                    container.database
                )
                if not verification.matches:
                    raise _invalid()
            reservation.verify()
            await _complete_threaded(checkpoint_owned_sqlite, path)
            snapshot = await _complete_threaded(freeze_closed_sqlite, path)
            reservation.verify()
            await _complete_threaded(verify_source_unchanged, snapshot)
            source_index = BenchmarkSourceIndex(
                agent_ids=MappingProxyType(dict(agent_ids)),
                experience_ids=MappingProxyType(dict(experience_ids)),
                candidate_ids=MappingProxyType(dict(candidate_ids)),
                labels_by_experience_id=MappingProxyType(
                    {value: label for label, value in experience_ids.items()}
                ),
                content_bytes_by_label=MappingProxyType(dict(content_bytes)),
            )
            built = BuiltBenchmarkSource(
                path=path,
                snapshot=snapshot,
                index=source_index,
                schema_revision=pack.manifest.schema_version,
            )
            reservation.commit()
            return built
    except Exception:
        raise _invalid() from None
    raise _invalid()


__all__ = [
    "BenchmarkSourceIndex",
    "BuiltBenchmarkSource",
    "PILOT_LIFECYCLE_CONFIG",
    "build_benchmark_source",
    "search_document_bytes",
]
