from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from inspect import signature
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from tests.repository.test_candidate_projection_rebuild import (
    ADOPTED_ID,
    ADOPTION_RECEIPT_ID,
    BUNDLE_ID,
    CAPTURE_RECEIPT_ID,
    EVIDENCE_IDS,
    HASH_A,
    OWNER_ID,
    PENDING_ID,
    REJECTED_ID,
    REJECTION_RECEIPT_ID,
    REUSED_ADOPTION_RECEIPT_ID,
    REUSED_ID,
    REUSED_TARGET_RECEIPT_ID,
    VERSION_ID,
    seed_candidate_graph,
)

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture.models import MAX_TRAJECTORY_STEPS
from experience_hub.capture.source_integrity import (
    authenticate_trajectory_manifest,
    reconstruct_captured_evidence,
)
from experience_hub.capture.validation import (
    CaptureSourceValidator,
    _manifest_document,
    register_capture_source_validator,
)
from experience_hub.config import Settings
from experience_hub.domain import EventRegistry
from experience_hub.experiences.candidate_events import register_candidate_events
from experience_hub.experiences.candidate_projector import CandidateStateProjector
from experience_hub.experiences.events import register_experience_events
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.database import Database
from experience_hub.storage.projections import ProjectionManager, ProjectionRegistry
from experience_hub.storage.tables import TrajectoryBundleRow
from experience_hub.storage.validation import SourceIntegrityError, SourceValidator

NOW = datetime(2026, 7, 22, 9, tzinfo=UTC)


def _manifest_row(step_times: tuple[str, ...]) -> TrajectoryBundleRow:
    document = {
        "adapter": {"kind": "generic_jsonl", "version": 1},
        "owner_agent_id": str(OWNER_ID),
        "sanitization": {"input_sanitized": True, "profile_id": "trusted-v1"},
        "schema_version": 1,
        "source_completed_at": NOW + timedelta(seconds=2),
        "source_started_at": NOW,
        "steps": [
            {
                "action_hash": HASH_A,
                "candidate_signal_hash": None,
                "observation_hash": HASH_A,
                "occurred_at": occurred_at,
                "ordinal": ordinal,
                "outcome_hash": HASH_A,
                "status": "succeeded",
                "step_id": f"step-{ordinal}",
            }
            for ordinal, occurred_at in enumerate(step_times, start=1)
        ],
        "trajectory_id": "manifest-validation",
    }
    manifest = canonical_json_bytes(document)
    return TrajectoryBundleRow(
        bundle_id=BUNDLE_ID,
        owner_agent_id=OWNER_ID,
        trajectory_id="manifest-validation",
        adapter_kind="generic_jsonl",
        adapter_version=1,
        sanitization_profile="trusted-v1",
        manifest=manifest,
        manifest_hash=sha256_hex(manifest),
        source_started_at=NOW,
        source_completed_at=NOW + timedelta(seconds=2),
        captured_at=NOW + timedelta(seconds=3),
    )


@pytest.mark.parametrize(
    "step_times",
    (
        ("2026-07-22T09:00:00+00:00",),
        ("2026-07-22T08:59:59.999999Z",),
        ("2026-07-22T09:00:02.000001Z",),
        (
            "2026-07-22T09:00:01.000000Z",
            "2026-07-22T09:00:00.000000Z",
        ),
    ),
    ids=("noncanonical", "before_bounds", "after_bounds", "decreasing"),
)
def test_stored_manifest_rejects_invalid_step_timestamps(
    step_times: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="trajectory step"):
        _manifest_document(_manifest_row(step_times))


def test_stored_manifest_rejects_more_than_the_input_step_limit() -> None:
    step_times = tuple(
        f"2026-07-22T09:00:00.{ordinal:06d}Z"
        for ordinal in range(MAX_TRAJECTORY_STEPS + 1)
    )

    with pytest.raises(ValueError, match="trajectory manifest"):
        _manifest_document(_manifest_row(step_times))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", True),
        ("adapter_version", True),
        ("input_sanitized", 1),
        ("step_ordinal", True),
    ),
)
def test_stored_manifest_rejects_bool_integer_wire_type_aliases(
    field: str,
    invalid_value: object,
) -> None:
    row = _manifest_row(("2026-07-22T09:00:00.000000Z",))
    document = json.loads(row.manifest)
    if field == "schema_version":
        document["schema_version"] = invalid_value
    elif field == "adapter_version":
        document["adapter"]["version"] = invalid_value
    elif field == "input_sanitized":
        document["sanitization"]["input_sanitized"] = invalid_value
    else:
        document["steps"][0]["ordinal"] = invalid_value
    manifest = canonical_json_bytes(document)
    row.manifest = manifest
    row.manifest_hash = sha256_hex(manifest)

    with pytest.raises(ValueError, match="trajectory (manifest|step)"):
        _manifest_document(row)


def test_authenticated_manifest_exposes_only_immutable_typed_steps_and_index(
) -> None:
    row = _manifest_row(
        (
            "2026-07-22T09:00:00.000000Z",
            "2026-07-22T09:00:00.000001Z",
        )
    )

    manifest = authenticate_trajectory_manifest(row)

    assert manifest.steps[0].step_id == "step-1"
    assert manifest.require_step(step_id="step-2", ordinal=2) is manifest.steps[1]
    with pytest.raises(FrozenInstanceError):
        manifest.steps[0].ordinal = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.step_index[("step-1", 1)] = manifest.steps[1]  # type: ignore[index]


def test_authenticated_manifest_step_lookup_does_not_scan_ordered_steps() -> None:
    step_times = tuple(
        f"2026-07-22T09:00:00.{ordinal:06d}Z"
        for ordinal in range(MAX_TRAJECTORY_STEPS)
    )
    manifest = authenticate_trajectory_manifest(_manifest_row(step_times))
    expected = manifest.steps[-1]

    class _IterationForbidden(tuple):
        def __iter__(self) -> object:
            raise AssertionError("require_step scanned ordered steps")

    object.__setattr__(manifest, "steps", _IterationForbidden(manifest.steps))

    assert manifest.require_step(
        step_id=expected.step_id,
        ordinal=expected.ordinal,
    ) is expected


@pytest.mark.parametrize("binding_field", ("bundle_id", "owner", "manifest_hash"))
def test_reusable_manifest_rejects_a_different_bundle_identity(
    binding_field: str,
) -> None:
    row = _manifest_row(("2026-07-22T09:00:00.000000Z",))
    manifest = authenticate_trajectory_manifest(row)
    if binding_field == "bundle_id":
        row.bundle_id = UUID(int=9_001)
    elif binding_field == "owner":
        row.owner_agent_id = UUID(int=9_002)
    else:
        row.manifest_hash = "f" * 64

    with pytest.raises(ValueError, match="manifest.*bundle"):
        reconstruct_captured_evidence(
            bundle=row,
            rows=(),
            evidence_ids=(),
            owner_agent_id=row.owner_agent_id,
            manifest=manifest,
        )


def test_per_source_validation_receives_an_event_index_not_the_event_ledger() -> None:
    assert "decoded_events" not in signature(
        CaptureSourceValidator._validate_bundle
    ).parameters
    assert "decoded_events" not in signature(
        CaptureSourceValidator._validate_candidate
    ).parameters


@pytest.fixture
async def capture_stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[tuple[Database, ProjectionManager]]:
    path = tmp_path / "capture-validation.sqlite3"
    config = Config(repository_root / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "head")
    registry = EventRegistry()
    register_experience_events(registry)
    register_candidate_events(registry)
    source_validator = SourceValidator(registry)
    register_capture_source_validator(source_validator)
    manager = ProjectionManager(
        ProjectionRegistry([CandidateStateProjector(registry)]),
        source_validator=source_validator,
    )
    database = Database.create(
        f"sqlite+aiosqlite:///{path}",
        event_registry=registry,
        projection_applier=manager,
    )
    await seed_candidate_graph(database, manager, registry)
    await manager.validate_startup(database)
    try:
        yield database, manager
    finally:
        await database.dispose()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", True),
        ("adapter_version", True),
        ("input_sanitized", 1),
        ("step_ordinal", True),
    ),
)
@pytest.mark.asyncio
async def test_startup_rejects_manifest_bool_integer_wire_type_aliases(
    capture_stack: tuple[Database, ProjectionManager],
    field: str,
    invalid_value: object,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        bundle = await uow.session.scalar(
            text("SELECT manifest FROM trajectory_bundles")
        )
        assert isinstance(bundle, bytes)
        document = json.loads(bundle)
        if field == "schema_version":
            document["schema_version"] = invalid_value
        elif field == "adapter_version":
            document["adapter"]["version"] = invalid_value
        elif field == "input_sanitized":
            document["sanitization"]["input_sanitized"] = invalid_value
        else:
            document["steps"][0]["ordinal"] = invalid_value
        manifest = canonical_json_bytes(document)
        await uow.session.execute(
            text("DROP TRIGGER trajectory_bundles_reject_update")
        )
        await uow.session.execute(
            text(
                "UPDATE trajectory_bundles SET manifest = :manifest, "
                "manifest_hash = :manifest_hash"
            ),
            {"manifest": manifest, "manifest_hash": sha256_hex(manifest)},
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


class _CaptureRuntimeContainer:
    def __init__(self, database: Database, manager: ProjectionManager) -> None:
        self.database = database
        self.projection_manager = manager
        self.schema_revision: str | None = None

    async def close(self) -> None:
        """The owning fixture disposes the shared database after the assertion."""


@pytest.mark.parametrize(
    ("statements", "parameters"),
    (
        (
            (
                "DROP TRIGGER trajectory_bundles_reject_update",
                "UPDATE trajectory_bundles SET manifest_hash = :hash",
            ),
            {"hash": HASH_A},
        ),
        (
            (
                "DROP TRIGGER trajectory_evidence_reject_update",
                "UPDATE trajectory_evidence SET ordinal = ordinal + 1 "
                "WHERE evidence_id = :evidence_id",
            ),
            {"evidence_id": str(EVIDENCE_IDS[0])},
        ),
        (
            (
                "DROP TRIGGER trajectory_evidence_reject_update",
                "UPDATE trajectory_evidence SET source_hash = :hash "
                "WHERE evidence_id = :evidence_id",
            ),
            {"hash": HASH_A, "evidence_id": str(EVIDENCE_IDS[0])},
        ),
        (
            (
                "DROP TRIGGER trajectory_evidence_reject_update",
                "UPDATE trajectory_evidence SET excerpt_hash = :hash "
                "WHERE evidence_id = :evidence_id",
            ),
            {"hash": HASH_A, "evidence_id": str(EVIDENCE_IDS[0])},
        ),
        (
            (
                "DROP TRIGGER trajectory_evidence_reject_update",
                "UPDATE trajectory_evidence SET excerpt = 'valid tampering' "
                "WHERE evidence_id = :evidence_id",
            ),
            {"evidence_id": str(EVIDENCE_IDS[0])},
        ),
        (
            (
                "DROP TRIGGER trajectory_evidence_reject_update",
                "UPDATE trajectory_evidence SET excerpt = "
                "CAST(x'534f555243455f534543524554ff' AS BLOB) "
                "WHERE evidence_id = :evidence_id",
            ),
            {"evidence_id": str(EVIDENCE_IDS[0])},
        ),
        (
            (
                "DROP TRIGGER experience_candidates_reject_update",
                "UPDATE experience_candidates SET content_hash = :hash "
                "WHERE candidate_id = :candidate_id",
            ),
            {"hash": HASH_A, "candidate_id": str(PENDING_ID)},
        ),
        (
            (
                "DROP TRIGGER experience_candidates_reject_update",
                "UPDATE experience_candidates SET evidence_refs = :refs "
                "WHERE candidate_id = :candidate_id",
            ),
            {
                "refs": canonical_json_bytes([EVIDENCE_IDS[1]]),
                "candidate_id": str(PENDING_ID),
            },
        ),
        (
            (
                "DROP TRIGGER experience_versions_reject_update",
                "UPDATE experience_versions SET summary = :secret "
                "WHERE version_id = :version_id",
            ),
            {"secret": "SOURCE_SECRET", "version_id": str(VERSION_ID)},
        ),
        (
            (
                "DROP TRIGGER domain_events_reject_update",
                "UPDATE domain_events SET aggregate_type = 'damaged' "
                "WHERE aggregate_id = :candidate_id AND sequence = 1",
            ),
            {"candidate_id": str(PENDING_ID)},
        ),
        (
            (
                "DROP TRIGGER domain_events_reject_update",
                "DELETE FROM candidate_state WHERE candidate_id = :candidate_id",
                "CREATE TEMP TABLE capture_event_swap "
                "(capture_id INTEGER NOT NULL, created_id INTEGER NOT NULL)",
                "INSERT INTO capture_event_swap SELECT "
                "(SELECT event_id FROM domain_events "
                "WHERE aggregate_id = :bundle_id AND sequence = 1), "
                "(SELECT event_id FROM domain_events "
                "WHERE aggregate_id = :candidate_id AND sequence = 1)",
                "UPDATE domain_events SET event_id = 1000 "
                "WHERE aggregate_id = :bundle_id AND sequence = 1",
                "UPDATE domain_events SET event_id = "
                "(SELECT capture_id FROM capture_event_swap) "
                "WHERE aggregate_id = :candidate_id AND sequence = 1",
                "UPDATE domain_events SET event_id = "
                "(SELECT created_id FROM capture_event_swap) "
                "WHERE event_id = 1000",
                "DROP TABLE capture_event_swap",
            ),
            {
                "bundle_id": str(BUNDLE_ID),
                "candidate_id": str(PENDING_ID),
            },
        ),
        (
            (
                "DROP TRIGGER domain_events_reject_update",
                "DELETE FROM candidate_state WHERE candidate_id IN "
                "(:first_candidate_id, :second_candidate_id)",
                "CREATE TEMP TABLE candidate_event_swap "
                "(first_id INTEGER NOT NULL, second_id INTEGER NOT NULL)",
                "INSERT INTO candidate_event_swap SELECT "
                "(SELECT event_id FROM domain_events "
                "WHERE aggregate_id = :first_candidate_id AND sequence = 1), "
                "(SELECT event_id FROM domain_events "
                "WHERE aggregate_id = :second_candidate_id AND sequence = 1)",
                "UPDATE domain_events SET event_id = 1000 "
                "WHERE aggregate_id = :first_candidate_id AND sequence = 1",
                "UPDATE domain_events SET event_id = "
                "(SELECT first_id FROM candidate_event_swap) "
                "WHERE aggregate_id = :second_candidate_id AND sequence = 1",
                "UPDATE domain_events SET event_id = "
                "(SELECT second_id FROM candidate_event_swap) "
                "WHERE event_id = 1000",
                "DROP TABLE candidate_event_swap",
            ),
            {
                "first_candidate_id": str(PENDING_ID),
                "second_candidate_id": str(ADOPTED_ID),
            },
        ),
        (
            (
                "UPDATE idempotency_records SET result_resource_type = 'damaged' "
                "WHERE receipt_id = :receipt_id",
            ),
            {"receipt_id": str(CAPTURE_RECEIPT_ID)},
        ),
    ),
    ids=(
        "manifest",
        "evidence_ordinal",
        "source_hash",
        "excerpt_hash",
        "evidence_valid_utf8_tamper",
        "evidence_blob",
        "candidate_content_hash",
        "candidate_evidence_order",
        "adoption_target",
        "event_aggregate",
        "capture_after_candidate_created",
        "candidate_created_global_order",
        "receipt_anchor",
    ),
)
@pytest.mark.asyncio
async def test_capture_source_corruption_fails_startup_without_leaking_content(
    capture_stack: tuple[Database, ProjectionManager],
    statements: tuple[str, ...],
    parameters: dict[str, Any],
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        for statement in statements:
            await uow.session.execute(text(statement), parameters)

    container = _CaptureRuntimeContainer(database, manager)

    def container_factory(**_: Any) -> _CaptureRuntimeContainer:
        return container

    async def migrator(_: Settings) -> str:
        return "0007_capture_evidence_hashes"

    runtime = ApplicationRuntime(
        Settings(database_url="sqlite+aiosqlite:///:memory:"),
        container_factory=container_factory,  # type: ignore[arg-type]
        migrator=migrator,
    )
    served = False
    with pytest.raises(SourceIntegrityError) as caught:
        async with runtime.initialize(
            start_lifecycle_worker=False,
            recover_interrupted=False,
        ):
            served = True

    assert served is False
    assert "SOURCE_SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_startup_accepts_utf8_truncated_evidence_with_full_field_hash(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack
    async with database.read_session() as session:
        row = (
            await session.execute(
                text(
                    "SELECT excerpt, source_hash, excerpt_hash "
                    "FROM trajectory_evidence "
                    "WHERE evidence_id = :evidence_id"
                ),
                {"evidence_id": str(EVIDENCE_IDS[0])},
            )
        ).mappings().one()

    assert row["excerpt"] == "界" * 170
    assert len(row["excerpt"].encode("utf-8")) == 510
    assert row["source_hash"] == sha256_hex(("界" * 171 + "tail").encode())
    assert row["excerpt_hash"] == sha256_hex(("界" * 170).encode())
    await manager.validate_startup(database)


@pytest.mark.asyncio
async def test_startup_rejects_noncanonical_manifest_step_timestamp(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        raw = await uow.session.scalar(
            text(
                "SELECT manifest FROM trajectory_bundles "
                "WHERE bundle_id = :bundle_id"
            ),
            {"bundle_id": str(BUNDLE_ID)},
        )
        assert isinstance(raw, bytes)
        document = json.loads(raw)
        document["steps"][0]["occurred_at"] = "2026-07-22T09:00:00+00:00"
        manifest = canonical_json_bytes(document)
        await uow.session.execute(text("DROP TRIGGER trajectory_bundles_reject_update"))
        await uow.session.execute(
            text(
                "UPDATE trajectory_bundles SET manifest = :manifest, "
                "manifest_hash = :manifest_hash WHERE bundle_id = :bundle_id"
            ),
            {
                "bundle_id": str(BUNDLE_ID),
                "manifest": manifest,
                "manifest_hash": sha256_hex(manifest),
            },
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.asyncio
async def test_reused_adoption_is_event_only_under_its_command_causation(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack
    async with database.read_session() as session:
        created = await session.scalar(
            text(
                "SELECT created FROM candidate_adoptions "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(REUSED_ID)},
        )
        causal_types = tuple(
            await session.scalars(
                text(
                    "SELECT event_type FROM domain_events "
                    "WHERE causation_id = :receipt_id ORDER BY event_id"
                ),
                {"receipt_id": str(REUSED_ADOPTION_RECEIPT_ID)},
            )
        )

    assert created == 0
    assert causal_types == ("candidate.adopted",)
    await manager.validate_startup(database)


@pytest.mark.asyncio
async def test_startup_requires_capture_receipt_to_be_completed(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE idempotency_records SET state = 'in_progress', "
                "response_status_code = NULL, response_body = NULL, "
                "response_content_type = NULL, response_headers = NULL, "
                "completed_at = NULL WHERE receipt_id = :receipt_id"
            ),
            {"receipt_id": str(CAPTURE_RECEIPT_ID)},
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.parametrize(
    ("receipt_id", "event_time"),
    (
        (CAPTURE_RECEIPT_ID, NOW),
        (ADOPTION_RECEIPT_ID, NOW + timedelta(seconds=1)),
        (REJECTION_RECEIPT_ID, NOW + timedelta(seconds=3)),
    ),
    ids=("capture", "adopt", "reject"),
)
@pytest.mark.asyncio
async def test_startup_rejects_receipt_creation_time_drift_inside_window(
    capture_stack: tuple[Database, ProjectionManager],
    receipt_id: UUID,
    event_time: datetime,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                "UPDATE idempotency_records SET created_at = :created_at "
                "WHERE receipt_id = :receipt_id"
            ),
            {
                "receipt_id": str(receipt_id),
                "created_at": event_time - timedelta(microseconds=1),
            },
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.asyncio
async def test_startup_rejects_candidate_created_after_bundle_capture(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack
    drifted = NOW + timedelta(microseconds=1)
    async with database.transaction() as uow:
        await uow.session.execute(
            text("DROP TRIGGER experience_candidates_reject_update")
        )
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text(
                "UPDATE experience_candidates SET created_at = :created_at "
                "WHERE candidate_id = :candidate_id"
            ),
            {"candidate_id": str(PENDING_ID), "created_at": drifted},
        )
        await uow.session.execute(
            text(
                "UPDATE domain_events SET occurred_at = :occurred_at "
                "WHERE aggregate_id = :candidate_id AND sequence = 1"
            ),
            {"candidate_id": str(PENDING_ID), "occurred_at": drifted},
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.parametrize(
    ("target_receipt_id", "source_receipt_id"),
    (
        (CAPTURE_RECEIPT_ID, REUSED_TARGET_RECEIPT_ID),
        (REJECTION_RECEIPT_ID, REUSED_TARGET_RECEIPT_ID),
    ),
    ids=("capture", "reject"),
)
@pytest.mark.asyncio
async def test_capture_and_reject_causations_reject_extra_events(
    capture_stack: tuple[Database, ProjectionManager],
    target_receipt_id: UUID,
    source_receipt_id: UUID,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text(
                "UPDATE domain_events SET causation_id = :target_receipt "
                "WHERE causation_id = :source_receipt AND sequence = 1"
            ),
            {
                "target_receipt": str(target_receipt_id),
                "source_receipt": str(source_receipt_id),
            },
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.parametrize(
    ("event_aggregate_id", "event_sequence", "target_receipt_id"),
    (
        (PENDING_ID, 1, REJECTION_RECEIPT_ID),
        (REJECTED_ID, 2, CAPTURE_RECEIPT_ID),
    ),
    ids=("capture", "reject"),
)
@pytest.mark.asyncio
async def test_capture_and_reject_causations_reject_missing_events(
    capture_stack: tuple[Database, ProjectionManager],
    event_aggregate_id: UUID,
    event_sequence: int,
    target_receipt_id: UUID,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        await uow.session.execute(
            text(
                "UPDATE domain_events SET causation_id = :target_receipt "
                "WHERE aggregate_id = :aggregate_id AND sequence = :sequence"
            ),
            {
                "target_receipt": str(target_receipt_id),
                "aggregate_id": str(event_aggregate_id),
                "sequence": event_sequence,
            },
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.parametrize(
    "receipt_id",
    (CAPTURE_RECEIPT_ID, ADOPTION_RECEIPT_ID, REJECTION_RECEIPT_ID),
    ids=("capture", "adopt", "reject"),
)
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("response_status_code", 202),
        ("response_body", canonical_json_bytes({"data": {"tampered": True}})),
        ("response_content_type", "text/plain"),
        ("response_headers", canonical_json_bytes({"etag": "tampered"})),
    ),
    ids=("status", "body", "content_type", "headers"),
)
@pytest.mark.asyncio
async def test_startup_rejects_tampered_completed_response(
    capture_stack: tuple[Database, ProjectionManager],
    receipt_id: UUID,
    field: str,
    value: object,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(
            text(
                f"UPDATE idempotency_records SET {field} = :value "
                "WHERE receipt_id = :receipt_id"
            ),
            {"receipt_id": str(receipt_id), "value": value},
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)


@pytest.mark.asyncio
async def test_canonical_capture_and_decision_responses_survive_replay(
    capture_stack: tuple[Database, ProjectionManager],
) -> None:
    database, manager = capture_stack

    assert (await manager.verify(database)).matches


@pytest.mark.parametrize(
    ("receipt_id", "event_type"),
    (
        (ADOPTION_RECEIPT_ID, "experience.created"),
        (REUSED_ADOPTION_RECEIPT_ID, None),
    ),
    ids=("created_unrelated_event", "reused_carries_creation_events"),
)
@pytest.mark.asyncio
async def test_adoption_causation_rejects_extra_experience_creation_events(
    capture_stack: tuple[Database, ProjectionManager],
    receipt_id: UUID,
    event_type: str | None,
) -> None:
    database, manager = capture_stack
    async with database.transaction() as uow:
        await uow.session.execute(text("DROP TRIGGER domain_events_reject_update"))
        conditions = "causation_id = :source_receipt"
        if event_type is not None:
            conditions += " AND event_type = :event_type"
        await uow.session.execute(
            text(
                "UPDATE domain_events SET causation_id = :target_receipt "
                f"WHERE {conditions}"
            ),
            {
                "source_receipt": str(REUSED_TARGET_RECEIPT_ID),
                "target_receipt": str(receipt_id),
                "event_type": event_type,
            },
        )

    with pytest.raises(SourceIntegrityError):
        await manager.validate_startup(database)
