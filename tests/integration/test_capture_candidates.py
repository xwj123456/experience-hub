from __future__ import annotations

import json
from base64 import urlsafe_b64encode
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

import experience_hub.capture.source_integrity as capture_source_integrity
from experience_hub import canonical_json_bytes
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.capture.extraction import DeterministicSignalExtractor
from experience_hub.capture.hashing import extractor_configuration_hash
from experience_hub.capture.jsonl import GenericJsonlAdapter
from experience_hub.capture.sanitization import DefaultSecretScanner
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.capture.service import (
    CapturePreparer,
    CaptureService,
    SensitiveInputError,
)
from experience_hub.capture.validation import register_capture_source_validator
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, EventRegistry
from experience_hub.errors import DomainError
from experience_hub.experiences.candidate_events import (
    CandidateCreatedV1,
    register_candidate_events,
)
from experience_hub.experiences.candidate_models import CandidateDecision
from experience_hub.experiences.candidate_projector import CandidateStateProjector
from experience_hub.experiences.candidate_repository import CandidateRepository
from experience_hub.experiences.candidate_service import (
    CandidatePageV1,
    CandidateService,
    CandidateViewV1,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandResult,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projections import ProjectionManager, ProjectionRegistry
from experience_hub.storage.tables import (
    AgentRow,
    CandidateStateRow,
    DomainEventRow,
    ExperienceCandidateRow,
    IdempotencyRecordRow,
    ProjectionVersionRow,
    TrajectoryBundleRow,
    TrajectoryEvidenceRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError, SourceValidator

NOW = datetime(2026, 7, 22, 9, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000501")
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000502")
BUNDLE_ID = UUID("00000000-0000-0000-0000-000000000503")
EVIDENCE_ID = UUID("00000000-0000-0000-0000-000000000504")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000505")
OTHER_OWNER_ID = UUID("00000000-0000-0000-0000-000000000506")
MISSING_OWNER_ID = UUID("00000000-0000-0000-0000-000000000507")
EXTRA_IDS = tuple(
    UUID(f"00000000-0000-0000-0000-{value:012d}")
    for value in range(508, 810)
)

ZERO_COUNTS = {
    "candidate_state": 0,
    "domain_events": 0,
    "experience_candidates": 0,
    "idempotency_records": 0,
    "projection_versions": 0,
    "trajectory_bundles": 0,
    "trajectory_evidence": 0,
}


def _synthetic_private_key_marker() -> str:
    return "-----BEGIN " + "PRIVATE KEY-----"


def _synthetic_openai_key() -> str:
    return "sk-" + "proj-abcdefghijklmnopqrstuvwxyz"


class FailAt:
    def __init__(self, checkpoint: FaultCheckpoint | None = None) -> None:
        self.checkpoint = checkpoint
        self.source_number: int | None = None
        self._source_count = 0

    def __call__(self, checkpoint: FaultCheckpoint) -> None:
        if checkpoint is FaultCheckpoint.AFTER_SOURCE_INSERT:
            self._source_count += 1
            if self._source_count == self.source_number:
                raise RuntimeError(f"injected:source:{self._source_count}")
        if checkpoint == self.checkpoint:
            raise RuntimeError(f"injected:{checkpoint.value}")


class FailingCandidateProjection:
    name = "failing_candidate_projection"
    version = 1
    event_types = frozenset({"candidate.created"})

    async def apply(self, session: AsyncSession, event: object) -> None:
        _ = (session, event)
        raise RuntimeError("injected:projection")

    async def rebuild(self, session: AsyncSession, target_prefix: str) -> None:
        _ = (session, target_prefix)
        raise AssertionError("rebuild is not part of live capture")


@dataclass(slots=True)
class CaptureStack:
    database: Database
    clock: FrozenClock
    executor: CommandExecutor
    preparer: CapturePreparer
    service: CaptureService
    candidate_service: CandidateService
    manager: ProjectionManager
    source_validator: SourceValidator
    fault: FailAt


def _header(*, owner_agent_id: UUID = OWNER_ID) -> dict[str, object]:
    timestamp = NOW.isoformat().replace("+00:00", "Z")
    return {
        "adapter": {"kind": "generic_jsonl", "version": 1},
        "owner_agent_id": str(owner_agent_id),
        "record_type": "header",
        "sanitization": {
            "input_sanitized": True,
            "profile_id": "trusted-v1",
        },
        "schema_version": 1,
        "source_completed_at": timestamp,
        "source_started_at": timestamp,
        "trajectory_id": "trajectory-1",
    }


def _step(
    *,
    observation: str = "A stale cache caused the command to fail.",
    candidate: bool = True,
) -> dict[str, object]:
    signal: dict[str, object] | None = None
    if candidate:
        signal = {
            "applicability": ["local caches"],
            "body": "Invalidate stale caches before retrying the command.",
            "evidence": [{"field": "observation", "step_id": "step-1"}],
            "falsifiers": ["A fresh cache still fails"],
            "kind": "procedural",
            "mechanism": "Invalidation removes stale state before the retry.",
            "summary": "Invalidate stale caches before retrying.",
            "tags": ["cache", "retry"],
        }
    return {
        "action": "Invalidated the cache and retried.",
        "candidate_signal": signal,
        "observation": observation,
        "occurred_at": NOW.isoformat().replace("+00:00", "Z"),
        "ordinal": 1,
        "outcome": "The retry succeeded.",
        "record_type": "step",
        "status": "succeeded",
        "step_id": "step-1",
    }


def _jsonl(
    *,
    owner_agent_id: UUID = OWNER_ID,
    observation: str = "A stale cache caused the command to fail.",
    candidate: bool = True,
) -> bytes:
    return b"\n".join(
        (
            canonical_json_bytes(_header(owner_agent_id=owner_agent_id)),
            canonical_json_bytes(
                _step(observation=observation, candidate=candidate)
            ),
        )
    )


def _jsonl_with_candidates(count: int) -> bytes:
    records: list[bytes] = [canonical_json_bytes(_header())]
    for ordinal in range(1, count + 1):
        step_id = f"step-{ordinal}"
        step = _step(observation=f"Observation {ordinal}.")
        step["ordinal"] = ordinal
        step["step_id"] = step_id
        signal = step["candidate_signal"]
        assert isinstance(signal, dict)
        signal["body"] = f"Use deterministic response {ordinal}."
        signal["summary"] = f"Deterministic response {ordinal}."
        signal["mechanism"] = f"Mechanism {ordinal}."
        signal["evidence"] = [
            {"field": "observation", "step_id": step_id}
        ]
        records.append(canonical_json_bytes(step))
    return b"\n".join(records)


def jsonl_with_private_key() -> bytes:
    return _jsonl(
        observation=_synthetic_private_key_marker(),
    )


async def all_capture_row_counts(database: Database) -> dict[str, int]:
    tables = {
        "candidate_state": CandidateStateRow,
        "domain_events": DomainEventRow,
        "experience_candidates": ExperienceCandidateRow,
        "idempotency_records": IdempotencyRecordRow,
        "projection_versions": ProjectionVersionRow,
        "trajectory_bundles": TrajectoryBundleRow,
        "trajectory_evidence": TrajectoryEvidenceRow,
    }
    async with database.read_session() as session:
        return {
            name: int(await session.scalar(select(func.count()).select_from(row)) or 0)
            for name, row in tables.items()
        }


async def _build_stack(
    *,
    repository_root: Path,
    database_path: Path,
) -> CaptureStack:
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    alembic_command.upgrade(config, "head")

    registry = EventRegistry()
    register_candidate_events(registry)
    source_validator = SourceValidator(registry)
    register_capture_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry((CandidateStateProjector(registry),)),
        source_validator=source_validator,
    )
    fault = FailAt()
    database = Database.create(
        f"sqlite+aiosqlite:///{database_path}",
        event_registry=registry,
        projection_applier=manager,
        fault_injector=fault,
    )
    async with database.transaction() as uow:
        uow.session.add_all(
            (
                AgentRow(agent_id=OWNER_ID, name="Owner", created_at=NOW),
                AgentRow(
                    agent_id=OTHER_OWNER_ID,
                    name="Other Owner",
                    created_at=NOW,
                ),
            )
        )

    ids = SequenceIdGenerator(
        (
            RECEIPT_ID,
            BUNDLE_ID,
            EVIDENCE_ID,
            CANDIDATE_ID,
            *EXTRA_IDS,
        )
    )
    clock = FrozenClock(NOW)
    receipts = ReceiptStore(clock=clock, id_generator=ids)
    candidate_repository = CandidateRepository()
    return CaptureStack(
        database=database,
        clock=clock,
        executor=CommandExecutor(
            database=database,
            receipt_store=receipts,
            clock=clock,
        ),
        preparer=CapturePreparer(
            adapter=GenericJsonlAdapter(),
            scanner=DefaultSecretScanner(),
            extractor=DeterministicSignalExtractor(),
        ),
        service=CaptureService(
            clock=clock,
            id_generator=ids,
            receipt_store=receipts,
        ),
        candidate_service=CandidateService(repository=candidate_repository),
        manager=manager,
        source_validator=source_validator,
        fault=fault,
    )


@pytest.fixture
async def stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[CaptureStack]:
    value = await _build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capture-candidates.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


def _request(
    *,
    key: str,
    manifest_hash: str,
    candidate_count: int,
    owner_agent_id: UUID = OWNER_ID,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=TRAJECTORY_IMPORT_SCOPE,
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/trajectory-bundles",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "adapter": "generic_jsonl",
            "candidate_count": candidate_count,
            "manifest_hash": manifest_hash,
        },
    )


async def _capture(
    stack: CaptureStack,
    *,
    key: str = "capture-1",
    data: bytes | None = None,
    caller_owner_id: UUID | None = None,
) -> CommandResult:
    prepared = stack.preparer.prepare_jsonl(data or _jsonl())

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.service.capture(
            uow=uow,
            prepared=prepared,
            command=context,
        )

    return await stack.executor.execute(
        _request(
            key=key,
            manifest_hash=prepared.bundle.manifest_hash,
            candidate_count=len(prepared.candidates),
            owner_agent_id=(
                prepared.bundle.owner_agent_id
                if caller_owner_id is None
                else caller_owner_id
            ),
        ),
        handler,
    )


@pytest.mark.asyncio
async def test_sensitive_match_creates_no_receipt_source_or_event(
    stack: CaptureStack,
) -> None:
    with pytest.raises(SensitiveInputError) as captured:
        stack.preparer.prepare_jsonl(jsonl_with_private_key())

    assert captured.value.code == "sensitive_input_detected"
    assert captured.value.details == {
        "matches": [
            {
                "field": "observation",
                "rule_id": "private_key",
                "step_id": "step-1",
            }
        ]
    }
    assert await all_capture_row_counts(stack.database) == ZERO_COUNTS


@pytest.mark.parametrize(
    ("header_field", "reported_field"),
    (
        ("trajectory_id", "trajectory_id"),
        ("sanitization_profile", "sanitization_profile_id"),
    ),
)
@pytest.mark.asyncio
async def test_sensitive_header_creates_no_receipt_source_or_event(
    stack: CaptureStack,
    header_field: str,
    reported_field: str,
) -> None:
    probe = _synthetic_openai_key()
    header = _header()
    if header_field == "trajectory_id":
        header["trajectory_id"] = probe
    else:
        sanitization = header["sanitization"]
        assert isinstance(sanitization, dict)
        sanitization["profile_id"] = probe
    data = b"\n".join(
        (canonical_json_bytes(header), canonical_json_bytes(_step()))
    )

    with pytest.raises(SensitiveInputError) as captured:
        stack.preparer.prepare_jsonl(data)

    assert captured.value.details == {
        "matches": [
            {
                "field": reported_field,
                "rule_id": "openai_key",
                "step_id": "header",
            }
        ]
    }
    assert probe not in repr(captured.value.details)
    assert await all_capture_row_counts(stack.database) == ZERO_COUNTS


@pytest.mark.asyncio
async def test_successful_capture_persists_sources_events_projection_and_receipt(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    body: dict[str, Any] = json.loads(result.body)

    assert (result.status_code, result.replayed) == (201, False)
    assert body == {
        "data": {
            "bundle_id": str(BUNDLE_ID),
            "candidate_count": 1,
            "candidate_ids": [str(CANDIDATE_ID)],
            "captured_at": "2026-07-22T09:00:00.000000Z",
            "manifest_hash": body["data"]["manifest_hash"],
            "owner_agent_id": str(OWNER_ID),
        }
    }
    async with stack.database.read_session() as session:
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        receipt = await session.get(IdempotencyRecordRow, RECEIPT_ID)
        state = await session.get(CandidateStateRow, CANDIDATE_ID)

    assert [event.event_type for event in events] == [
        "trajectory.captured",
        "candidate.created",
    ]
    assert [(event.aggregate_type, event.sequence) for event in events] == [
        ("trajectory_bundle", 1),
        ("experience_candidate", 1),
    ]
    assert state is not None and state.decision == "pending"
    assert receipt is not None
    assert TRAJECTORY_IMPORT_SCOPE == "capture.trajectory.import"
    assert receipt.scope == TRAJECTORY_IMPORT_SCOPE
    assert (
        receipt.state,
        receipt.result_resource_type,
        receipt.result_resource_id,
    ) == ("completed", "trajectory_bundle", BUNDLE_ID)
    assert await all_capture_row_counts(stack.database) == {
        "candidate_state": 1,
        "domain_events": 2,
        "experience_candidates": 1,
        "idempotency_records": 1,
        "projection_versions": 1,
        "trajectory_bundles": 1,
        "trajectory_evidence": 1,
    }


@pytest.mark.asyncio
async def test_zero_candidate_capture_still_persists_bundle_and_ledger(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack, data=_jsonl(candidate=False))

    assert result.status_code == 201
    body = json.loads(result.body)
    assert body["data"]["candidate_count"] == 0
    assert body["data"]["candidate_ids"] == []
    assert await all_capture_row_counts(stack.database) == {
        "candidate_state": 0,
        "domain_events": 1,
        "experience_candidates": 0,
        "idempotency_records": 1,
        "projection_versions": 0,
        "trajectory_bundles": 1,
        "trajectory_evidence": 0,
    }


@pytest.mark.asyncio
async def test_forged_owner_completes_safe_error_without_capture_rows(
    stack: CaptureStack,
) -> None:
    result = await _capture(
        stack,
        data=_jsonl(owner_agent_id=OTHER_OWNER_ID),
        caller_owner_id=OWNER_ID,
    )

    assert result.status_code == 403
    assert json.loads(result.body)["error"]["code"] == "capture_owner_invalid"
    counts = await all_capture_row_counts(stack.database)
    assert counts == {**ZERO_COUNTS, "idempotency_records": 1}


@pytest.mark.asyncio
async def test_missing_owner_agent_completes_safe_error_without_capture_rows(
    stack: CaptureStack,
) -> None:
    result = await _capture(
        stack,
        data=_jsonl(owner_agent_id=MISSING_OWNER_ID),
    )

    assert result.status_code == 404
    assert json.loads(result.body)["error"]["code"] == "agent_not_found"
    counts = await all_capture_row_counts(stack.database)
    assert counts == {**ZERO_COUNTS, "idempotency_records": 1}


@pytest.mark.asyncio
async def test_same_key_replays_exact_response_without_new_rows(
    stack: CaptureStack,
) -> None:
    first = await _capture(stack, key="same-key")
    before = await all_capture_row_counts(stack.database)

    second = await _capture(stack, key="same-key")

    assert (first.status_code, second.status_code) == (201, 201)
    assert second.replayed is True
    assert second.body == first.body
    assert second.headers == first.headers
    assert await all_capture_row_counts(stack.database) == before


@pytest.mark.asyncio
async def test_duplicate_manifest_under_new_key_is_a_replayable_conflict(
    stack: CaptureStack,
) -> None:
    first = await _capture(stack, key="first-key")
    before = await all_capture_row_counts(stack.database)

    duplicate = await _capture(stack, key="different-key")

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert json.loads(duplicate.body)["error"]["code"] == (
        "trajectory_already_captured"
    )
    assert await all_capture_row_counts(stack.database) == {
        **before,
        "idempotency_records": before["idempotency_records"] + 1,
    }


@pytest.mark.parametrize("source_number", (1, 2, 3))
@pytest.mark.asyncio
async def test_fault_after_each_source_insert_rolls_back_everything(
    stack: CaptureStack,
    source_number: int,
) -> None:
    stack.fault.source_number = source_number

    with pytest.raises(RuntimeError, match=f"injected:source:{source_number}"):
        await _capture(stack)

    assert await all_capture_row_counts(stack.database) == ZERO_COUNTS


@pytest.mark.parametrize(
    "checkpoint",
    (
        FaultCheckpoint.AFTER_EVENT_APPEND,
        FaultCheckpoint.AFTER_PROJECTION_APPLY,
        FaultCheckpoint.AFTER_RECEIPT_COMPLETION,
    ),
)
@pytest.mark.asyncio
async def test_command_faults_roll_back_sources_events_projection_and_receipt(
    stack: CaptureStack,
    checkpoint: FaultCheckpoint,
) -> None:
    stack.fault.checkpoint = checkpoint

    with pytest.raises(RuntimeError, match=f"injected:{checkpoint.value}"):
        await _capture(stack)

    assert await all_capture_row_counts(stack.database) == ZERO_COUNTS


@pytest.mark.asyncio
async def test_projection_failure_rolls_back_sources_events_and_receipt(
    stack: CaptureStack,
) -> None:
    stack.manager.registry.register(FailingCandidateProjection())

    with pytest.raises(RuntimeError, match="injected:projection"):
        await _capture(stack)

    assert await all_capture_row_counts(stack.database) == ZERO_COUNTS


@pytest.mark.asyncio
async def test_capture_sources_validate_after_receipt_completion(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    assert result.status_code == 201

    async with stack.database.read_session() as session:
        await stack.source_validator.validate(session)


@pytest.mark.asyncio
async def test_multi_field_evidence_capture_passes_startup_source_validation(
    stack: CaptureStack,
) -> None:
    step = _step()
    signal = step["candidate_signal"]
    assert isinstance(signal, dict)
    signal["evidence"] = [
        {"field": "observation", "step_id": "step-1"},
        {"field": "action", "step_id": "step-1"},
        {"field": "outcome", "step_id": "step-1"},
    ]
    data = b"\n".join(
        (canonical_json_bytes(_header()), canonical_json_bytes(step))
    )

    result = await _capture(stack, data=data)

    assert result.status_code == 201
    async with stack.database.read_session() as session:
        await stack.source_validator.validate(session)
        candidate_id = UUID(json.loads(result.body)["data"]["candidate_ids"][0])
        evidence_rows = tuple(
            (
                await session.scalars(
                    select(TrajectoryEvidenceRow).order_by(
                        TrajectoryEvidenceRow.evidence_id
                    )
                )
            ).all()
        )
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        candidate = await stack.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=candidate_id,
        )
    assert tuple(
        (row.evidence_id, row.field) for row in evidence_rows
    ) == (
        (EVIDENCE_ID, "observation"),
        (CANDIDATE_ID, "action"),
        (EXTRA_IDS[0], "outcome"),
    )
    assert json.loads(events[0].payload)["evidence_ids"] == [
        str(EVIDENCE_ID),
        str(CANDIDATE_ID),
        str(EXTRA_IDS[0]),
    ]
    assert json.loads(events[1].payload)["evidence_ids"] == [
        str(EVIDENCE_ID),
        str(CANDIDATE_ID),
        str(EXTRA_IDS[0]),
    ]
    assert tuple(item.field.value for item in candidate.evidence) == (
        "observation",
        "action",
        "outcome",
    )


@pytest.mark.asyncio
async def test_shared_evidence_location_is_stored_once_for_ordered_candidates(
    stack: CaptureStack,
) -> None:
    first = _step()
    first_signal = first["candidate_signal"]
    assert isinstance(first_signal, dict)
    first_signal["evidence"] = [
        {"field": "observation", "step_id": "step-2"}
    ]
    second = _step(observation="A second signal was observed.")
    second["ordinal"] = 2
    second["step_id"] = "step-2"
    second_signal = second["candidate_signal"]
    assert isinstance(second_signal, dict)
    second_signal.update(
        {
            "body": "Check cache generation before a retry.",
            "evidence": [
                {"field": "action", "step_id": "step-1"},
                {"field": "observation", "step_id": "step-2"},
                {"field": "outcome", "step_id": "step-2"},
            ],
            "mechanism": "Generation checks detect stale state.",
            "summary": "Check cache generation before retrying.",
        }
    )
    data = b"\n".join(
        (
            canonical_json_bytes(_header()),
            canonical_json_bytes(first),
            canonical_json_bytes(second),
        )
    )

    result = await _capture(stack, data=data)

    assert result.status_code == 201
    candidate_ids = tuple(
        UUID(value) for value in json.loads(result.body)["data"]["candidate_ids"]
    )
    assert candidate_ids == (EXTRA_IDS[1], EXTRA_IDS[2])
    async with stack.database.read_session() as session:
        evidence_rows = tuple(
            (
                await session.scalars(
                    select(TrajectoryEvidenceRow).order_by(
                        TrajectoryEvidenceRow.evidence_id
                    )
                )
            ).all()
        )
        evidence_refs = tuple(
            (
                await session.scalars(
                    select(ExperienceCandidateRow.evidence_refs).order_by(
                        ExperienceCandidateRow.candidate_ordinal
                    )
                )
            ).all()
        )
        events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        await stack.source_validator.validate(session)

    assert tuple(
        (row.evidence_id, row.step_id, row.field) for row in evidence_rows
    ) == (
        (EVIDENCE_ID, "step-2", "observation"),
        (CANDIDATE_ID, "step-1", "action"),
        (EXTRA_IDS[0], "step-2", "outcome"),
    )
    assert evidence_refs == (
        canonical_json_bytes((str(EVIDENCE_ID),)),
        canonical_json_bytes(
            (str(CANDIDATE_ID), str(EVIDENCE_ID), str(EXTRA_IDS[0]))
        ),
    )
    assert tuple(event.event_type for event in events) == (
        "trajectory.captured",
        "candidate.created",
        "candidate.created",
    )
    assert json.loads(events[0].payload)["evidence_ids"] == [
        str(EVIDENCE_ID),
        str(CANDIDATE_ID),
        str(EXTRA_IDS[0]),
    ]
    assert json.loads(events[1].payload)["evidence_ids"] == [str(EVIDENCE_ID)]
    assert json.loads(events[2].payload)["evidence_ids"] == [
        str(CANDIDATE_ID),
        str(EVIDENCE_ID),
        str(EXTRA_IDS[0]),
    ]
    counts = await all_capture_row_counts(stack.database)
    assert counts["trajectory_evidence"] == 3
    assert counts["experience_candidates"] == 2
    assert counts["candidate_state"] == 2


@pytest.mark.asyncio
async def test_ordinary_experience_sources_remain_empty_after_capture(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    assert result.status_code == 201

    async with stack.database.read_session() as db_session:
        experience_count = await db_session.scalar(
            text("SELECT count(*) FROM experiences")
        )
        version_count = await db_session.scalar(
            text("SELECT count(*) FROM experience_versions")
        )
        state_count = await db_session.scalar(
            text("SELECT count(*) FROM experience_state")
        )

    assert (experience_count, version_count, state_count) == (0, 0, 0)


@pytest.mark.asyncio
async def test_get_owned_reconstructs_immutable_pending_candidate(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    assert result.status_code == 201

    async with stack.database.read_session() as session:
        candidate = await stack.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=CANDIDATE_ID,
        )

    assert candidate.candidate_id == CANDIDATE_ID
    assert candidate.bundle_id == BUNDLE_ID
    assert candidate.owner_agent_id == OWNER_ID
    assert candidate.decision is CandidateDecision.PENDING
    assert candidate.kind.value == "procedural"
    assert candidate.content.body == (
        "Invalidate stale caches before retrying the command."
    )
    assert candidate.content.summary == (
        "Invalidate stale caches before retrying."
    )
    assert candidate.content_hash
    assert tuple(
        (item.step_id, item.field.value, item.excerpt)
        for item in candidate.evidence
    ) == (
        (
            "step-1",
            "observation",
            "A stale cache caused the command to fail.",
        ),
    )
    assert candidate.extractor_kind == "deterministic_signal_v1"
    assert candidate.resulting_experience_id is None
    assert candidate.resulting_version_id is None
    assert candidate.reason is None
    assert candidate.created_at == NOW
    assert candidate.decided_at is None


@pytest.mark.asyncio
async def test_candidate_view_rejects_partial_result_lineage(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    assert result.status_code == 201
    async with stack.database.read_session() as session:
        candidate = await stack.candidate_service.get_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            candidate_id=CANDIDATE_ID,
        )

    with pytest.raises(ValidationError):
        CandidateViewV1.model_validate(
            {
                **candidate.model_dump(),
                "resulting_experience_id": UUID(int=1),
            }
        )


@pytest.mark.asyncio
async def test_get_owned_fails_closed_on_corrupt_extractor_anchor(
    stack: CaptureStack,
) -> None:
    result = await _capture(stack)
    assert result.status_code == 201
    assert extractor_configuration_hash() != "f" * 64
    async with stack.database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_candidates_reject_update")
        )
        await uow.session.execute(
            text(
                "UPDATE experience_candidates "
                "SET extractor_configuration_hash = :corrupted "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(CANDIDATE_ID), "corrupted": "f" * 64},
        )

    with pytest.raises(SourceIntegrityError):
        async with stack.database.read_session() as session:
            await stack.candidate_service.get_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                candidate_id=CANDIDATE_ID,
            )


@pytest.mark.asyncio
async def test_foreign_and_missing_candidate_are_indistinguishable(
    stack: CaptureStack,
) -> None:
    captured = await _capture(
        stack,
        data=_jsonl(owner_agent_id=OTHER_OWNER_ID),
    )
    assert captured.status_code == 201
    foreign_id = UUID(json.loads(captured.body)["data"]["candidate_ids"][0])

    observed: list[tuple[str, str, int, dict[str, Any]]] = []
    for candidate_id in (foreign_id, UUID(int=999_999)):
        with pytest.raises(DomainError) as caught:
            async with stack.database.read_session() as session:
                await stack.candidate_service.get_owned(
                    session=session,
                    owner_agent_id=OWNER_ID,
                    candidate_id=candidate_id,
                )
        observed.append(
            (
                caught.value.code,
                caught.value.message,
                caught.value.status_code,
                caught.value.details,
            )
        )

    assert observed == [
        (
            "candidate_not_found",
            "Candidate was not found",
            404,
            {},
        ),
        (
            "candidate_not_found",
            "Candidate was not found",
            404,
            {},
        ),
    ]


async def _captured_candidate_id(
    stack: CaptureStack,
    *,
    key: str,
    observation: str,
    owner_agent_id: UUID = OWNER_ID,
) -> UUID:
    result = await _capture(
        stack,
        key=key,
        data=_jsonl(
            owner_agent_id=owner_agent_id,
            observation=observation,
        ),
    )
    assert result.status_code == 201
    return UUID(json.loads(result.body)["data"]["candidate_ids"][0])


@pytest.mark.asyncio
async def test_list_owned_orders_descending_and_excludes_foreign_candidates(
    stack: CaptureStack,
) -> None:
    oldest = await _captured_candidate_id(
        stack,
        key="oldest",
        observation="Old owner observation.",
    )
    stack.clock.advance(timedelta(minutes=1))
    newer_low_id = await _captured_candidate_id(
        stack,
        key="newer-low",
        observation="Newer owner observation A.",
    )
    newer_high_id = await _captured_candidate_id(
        stack,
        key="newer-high",
        observation="Newer owner observation B.",
    )
    await _captured_candidate_id(
        stack,
        key="foreign",
        observation="Foreign observation.",
        owner_agent_id=OTHER_OWNER_ID,
    )

    async with stack.database.read_session() as session:
        page = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
        )

    assert tuple(item.candidate_id for item in page.items) == (
        newer_high_id,
        newer_low_id,
        oldest,
    )
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_list_owned_loads_all_page_evidence_in_one_query(
    stack: CaptureStack,
) -> None:
    for index in range(3):
        await _captured_candidate_id(
            stack,
            key=f"batch-evidence-{index}",
            observation=f"Batched evidence {index}.",
        )

    evidence_selects: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        normalized = statement.lower()
        if "select" in normalized and "trajectory_evidence" in normalized:
            evidence_selects.append(normalized)

    event.listen(
        stack.database._engine.sync_engine,
        "before_cursor_execute",
        record_statement,
    )
    try:
        async with stack.database.read_session() as session:
            page = await stack.candidate_service.list_owned(
                session=session,
                owner_agent_id=OWNER_ID,
            )
    finally:
        event.remove(
            stack.database._engine.sync_engine,
            "before_cursor_execute",
            record_statement,
        )

    assert len(page.items) == 3
    assert len(evidence_selects) == 1


@pytest.mark.asyncio
async def test_capture_response_reconstruction_is_constant_per_ten_candidate_bundle(
    stack: CaptureStack,
) -> None:
    full_bundle_selects: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        normalized = " ".join(statement.lower().split())
        if (
            "from experience_candidates" in normalized
            and "where experience_candidates.bundle_id" in normalized
            and "order by experience_candidates.candidate_ordinal" in normalized
        ):
            full_bundle_selects.append(normalized)

    event.listen(
        stack.database._engine.sync_engine,
        "before_cursor_execute",
        record_statement,
    )
    try:
        result = await _capture(
            stack,
            data=_jsonl_with_candidates(10),
        )
    finally:
        event.remove(
            stack.database._engine.sync_engine,
            "before_cursor_execute",
            record_statement,
        )
    assert result.status_code == 201
    candidate_ids = tuple(
        UUID(value) for value in json.loads(result.body)["data"]["candidate_ids"]
    )
    assert len(candidate_ids) == 10
    assert full_bundle_selects == []

    registry = EventRegistry()
    register_candidate_events(registry)
    projector = CandidateStateProjector(registry)
    async with stack.database.transaction() as uow:
        rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type == CandidateCreatedV1.event_type,
                        DomainEventRow.aggregate_id.in_(candidate_ids),
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        completed_events = tuple(
            projector.stored_event_from_row(row) for row in rows
        )
        assert len(completed_events) == 10
        await uow.session.execute(text("DELETE FROM candidate_state"))
        event.listen(
            stack.database._engine.sync_engine,
            "before_cursor_execute",
            record_statement,
        )
        try:
            for stored_event in completed_events:
                await projector.apply(uow.session, stored_event)
        finally:
            event.remove(
                stack.database._engine.sync_engine,
                "before_cursor_execute",
                record_statement,
            )
    assert len(full_bundle_selects) == 1

    full_bundle_selects.clear()
    event.listen(
        stack.database._engine.sync_engine,
        "before_cursor_execute",
        record_statement,
    )
    try:
        assert (await stack.manager.verify(stack.database)).matches
    finally:
        event.remove(
            stack.database._engine.sync_engine,
            "before_cursor_execute",
            record_statement,
        )
    assert len(full_bundle_selects) == 1


@pytest.mark.parametrize("candidate_count", (10, 100))
@pytest.mark.asyncio
async def test_list_authenticates_a_shared_bundle_manifest_once(
    stack: CaptureStack,
    monkeypatch: pytest.MonkeyPatch,
    candidate_count: int,
) -> None:
    result = await _capture(
        stack,
        data=_jsonl_with_candidates(candidate_count),
    )
    assert result.status_code == 201
    calls: list[UUID] = []
    original = capture_source_integrity.authenticate_trajectory_manifest

    def counting_authenticate(
        bundle: TrajectoryBundleRow,
    ) -> capture_source_integrity.AuthenticatedTrajectoryManifest:
        calls.append(bundle.bundle_id)
        return original(bundle)

    monkeypatch.setattr(
        capture_source_integrity,
        "authenticate_trajectory_manifest",
        counting_authenticate,
    )

    async with stack.database.read_session() as session:
        page = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            limit=100,
        )

    assert len(page.items) == candidate_count
    assert calls == [BUNDLE_ID]


@pytest.mark.asyncio
async def test_list_and_startup_authenticate_each_distinct_bundle_once(
    stack: CaptureStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_bundle_ids: set[UUID] = set()
    for ordinal in range(3):
        result = await _capture(
            stack,
            key=f"manifest-group-{ordinal}",
            data=_jsonl(observation=f"Bundle observation {ordinal}."),
        )
        assert result.status_code == 201
        expected_bundle_ids.add(UUID(json.loads(result.body)["data"]["bundle_id"]))
    calls: list[UUID] = []
    original = capture_source_integrity.authenticate_trajectory_manifest

    def counting_authenticate(
        bundle: TrajectoryBundleRow,
    ) -> capture_source_integrity.AuthenticatedTrajectoryManifest:
        calls.append(bundle.bundle_id)
        return original(bundle)

    monkeypatch.setattr(
        capture_source_integrity,
        "authenticate_trajectory_manifest",
        counting_authenticate,
    )

    async with stack.database.read_session() as session:
        page = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
        )
    assert len(page.items) == 3
    expected_counts = Counter(
        {bundle_id: 1 for bundle_id in expected_bundle_ids}
    )
    assert Counter(calls) == expected_counts

    calls.clear()
    await stack.manager.validate_startup(stack.database)
    assert Counter(calls) == expected_counts


@pytest.mark.asyncio
async def test_list_owned_filters_decision_after_owner_predicate(
    stack: CaptureStack,
) -> None:
    pending_id = await _captured_candidate_id(
        stack,
        key="pending",
        observation="Pending observation.",
    )
    rejected_id = await _captured_candidate_id(
        stack,
        key="rejected",
        observation="Rejected observation.",
    )
    foreign_id = await _captured_candidate_id(
        stack,
        key="foreign-rejected",
        observation="Foreign rejected observation.",
        owner_agent_id=OTHER_OWNER_ID,
    )
    reason_text = "Not reusable."
    from hashlib import sha256

    async with stack.database.transaction() as uow:
        for candidate_id in (rejected_id, foreign_id):
            await uow.session.execute(
                text(
                    "UPDATE candidate_state SET decision = 'rejected', "
                    "reason_code = 'user_provided', reason_text = :reason_text, "
                    "reason_text_hash = :reason_hash, decided_at = :decided_at "
                    "WHERE candidate_id = :candidate_id"
                ),
                {
                    "candidate_id": str(candidate_id),
                    "decided_at": NOW.isoformat(),
                    "reason_hash": sha256(reason_text.encode()).hexdigest(),
                    "reason_text": reason_text,
                },
            )

    async with stack.database.read_session() as session:
        rejected = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            decision=CandidateDecision.REJECTED,
        )
        pending = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            decision=CandidateDecision.PENDING,
        )

    assert tuple(item.candidate_id for item in rejected.items) == (rejected_id,)
    assert rejected.items[0].reason is not None
    assert rejected.items[0].reason.text == reason_text
    assert tuple(item.candidate_id for item in pending.items) == (pending_id,)


@pytest.mark.asyncio
async def test_list_owned_cursor_is_canonical_and_pages_without_overlap(
    stack: CaptureStack,
) -> None:
    first_id = await _captured_candidate_id(
        stack,
        key="page-first",
        observation="First page observation.",
    )
    second_id = await _captured_candidate_id(
        stack,
        key="page-second",
        observation="Second page observation.",
    )

    async with stack.database.read_session() as session:
        first = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            limit=1,
        )
        assert first.next_cursor is not None
        second = await stack.candidate_service.list_owned(
            session=session,
            owner_agent_id=OWNER_ID,
            limit=1,
            cursor=first.next_cursor,
        )

    assert tuple(item.candidate_id for item in first.items) == (second_id,)
    assert tuple(item.candidate_id for item in second.items) == (first_id,)
    assert second.next_cursor is None
    cursor = first.next_cursor
    assert cursor is not None and "=" not in cursor
    padded = cursor + "=" * (-len(cursor) % 4)
    document = json.loads(__import__("base64").urlsafe_b64decode(padded))
    assert document == {
        "candidate_id": str(second_id),
        "created_at": "2026-07-22T09:00:00.000000Z",
    }
    assert cursor == urlsafe_b64encode(canonical_json_bytes(document)).decode().rstrip(
        "="
    )


def _cursor(document: object) -> str:
    return urlsafe_b64encode(canonical_json_bytes(document)).decode().rstrip("=")


@pytest.mark.parametrize("limit", (True, 0, 101))
@pytest.mark.asyncio
async def test_list_owned_rejects_invalid_limits(
    stack: CaptureStack,
    limit: int,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        async with stack.database.read_session() as session:
            await stack.candidate_service.list_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                limit=limit,
            )


@pytest.mark.parametrize(
    "cursor",
    (
        "not-base64!",
        _cursor({"candidate_id": str(CANDIDATE_ID)}),
        _cursor(
            {
                "candidate_id": str(CANDIDATE_ID),
                "created_at": 1,
            }
        ),
        _cursor(
            {
                "candidate_id": "not-a-uuid",
                "created_at": "2026-07-22T09:00:00.000000Z",
            }
        ),
        _cursor(
            {
                "candidate_id": str(CANDIDATE_ID),
                "created_at": "2026-07-22T09:00:00Z",
                "extra": True,
            }
        ),
    ),
)
@pytest.mark.asyncio
async def test_list_owned_rejects_malformed_cursor_before_query(
    stack: CaptureStack,
    cursor: str,
) -> None:
    with pytest.raises(DomainError) as caught:
        async with stack.database.read_session() as session:
            await stack.candidate_service.list_owned(
                session=session,
                owner_agent_id=OWNER_ID,
                cursor=cursor,
            )

    assert (
        caught.value.code,
        caught.value.message,
        caught.value.status_code,
    ) == ("invalid_cursor", "The cursor is invalid.", 400)


def test_candidate_page_is_frozen_and_strict() -> None:
    with pytest.raises(ValidationError):
        CandidatePageV1(items=(), next_cursor=None, extra=True)


@pytest.mark.asyncio
async def test_application_container_wires_capture_and_candidate_services(
    tmp_path: Path,
) -> None:
    container = ApplicationContainer.build(
        Settings(
            database_url=(
                f"sqlite+aiosqlite:///{tmp_path / 'bootstrap-capture.sqlite3'}"
            )
        ),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(EXTRA_IDS),
    )
    try:
        assert isinstance(container.capture_preparer, CapturePreparer)
        assert isinstance(container.capture_service, CaptureService)
        assert isinstance(container.candidate_repository, CandidateRepository)
        assert isinstance(container.candidate_service, CandidateService)
    finally:
        await container.close()
