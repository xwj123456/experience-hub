from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import UUID

import pytest
from click import unstyle
from typer.testing import CliRunner

import experience_hub.runtime as runtime_module
from experience_hub import canonical_json_bytes
from experience_hub.agents import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.capture.extraction import DeterministicSignalExtractor
from experience_hub.capture.jsonl import MAX_JSONL_INPUT_BYTES, GenericJsonlAdapter
from experience_hub.capture.sanitization import DefaultSecretScanner
from experience_hub.capture.scopes import TRAJECTORY_IMPORT_SCOPE
from experience_hub.capture.service import CapturePreparer
from experience_hub.cli.app import app
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest, StructuredReason
from experience_hub.experiences.candidate_models import (
    CANDIDATE_ADOPT_SCOPE,
    CANDIDATE_REJECT_SCOPE,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork

RUNNER = CliRunner()
FIXTURE = (
    Path(__file__).parents[2]
    / "examples"
    / "trajectories"
    / "coding-agent-recovery.jsonl"
)
OWNER_ID = UUID("10000000-0000-4000-8000-000000000701")
AGENT_RECEIPT_ID = UUID("10000000-0000-4000-8000-000000000700")
MISSING_CANDIDATE_ID = UUID("10000000-0000-4000-8000-000000000799")
OTHER_OWNER_ID = UUID("10000000-0000-4000-8000-000000000798")
NOW = datetime(2026, 7, 22, 2, tzinfo=UTC)


def _synthetic_private_key_marker() -> str:
    return "-----BEGIN " + "PRIVATE KEY-----"


def _synthetic_openai_key() -> str:
    return "sk-" + "proj-abcdefghijklmnopqrstuvwxyz"


def _canonical_success(result: Any) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert result.stdout == canonical_json_bytes(document).decode("utf-8") + "\n"
    assert isinstance(document, dict)
    return document


def _canonical_error(result: Any) -> dict[str, Any]:
    assert result.exit_code == 1, result.output
    document = json.loads(result.stdout)
    assert result.stdout == canonical_json_bytes(document).decode("utf-8") + "\n"
    assert isinstance(document, dict)
    return document


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            item
            for nested in value.values()
            for item in _strings(nested)
        )
    if isinstance(value, list):
        return tuple(item for nested in value for item in _strings(nested))
    return ()


async def _seed_owner(database_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{database_path}"
    )
    runtime = runtime_module.ApplicationRuntime(
        settings,
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator((AGENT_RECEIPT_ID, OWNER_ID)),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        assert isinstance(container, ApplicationContainer)
        request = CommandRequest(
            caller_scope="system:local",
            operation_scope="agent.create",
            idempotency_key="seed-cli-owner",
            method="POST",
            route_template="/v1/agents",
            body={"name": "Synthetic CLI owner"},
        )

        async def handler(
            uow: UnitOfWork,
            context: CommandContext,
        ) -> StoredResponse:
            return await container.agent_service.create(
                uow=uow,
                command=CreateAgent(name="Synthetic CLI owner"),
                command_context=context,
            )

        result = await container.command_executor.execute(request, handler)
        assert result.status_code == 201


def _variant_fixture(tmp_path: Path, label: str) -> Path:
    records = [json.loads(line) for line in FIXTURE.read_bytes().split(b"\n")]
    records[0]["trajectory_id"] = f"coding-agent-recovery-{label}"
    records[1]["observation"] = (
        "The synthetic workspace/project cache generation was stale "
        f"for run {label}."
    )
    path = tmp_path / f"trajectory-{label}.jsonl"
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records))
    return path


def _database_counts(database_path: Path) -> tuple[int, int]:
    with sqlite3.connect(database_path) as connection:
        receipt_count = connection.execute(
            "SELECT count(*) FROM idempotency_records"
        ).fetchone()
        event_count = connection.execute(
            "SELECT count(*) FROM domain_events"
        ).fetchone()
    assert receipt_count is not None
    assert event_count is not None
    return int(receipt_count[0]), int(event_count[0])


def _receipt(database_path: Path, key: str) -> sqlite3.Row:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT scope, request_hash, response_body FROM idempotency_records "
            "WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
    assert row is not None
    return row


def test_capture_and_candidate_help_expose_the_workflow() -> None:
    capture = RUNNER.invoke(app, ["capture", "--help"])
    candidates = RUNNER.invoke(app, ["candidates", "--help"])

    assert capture.exit_code == 0, capture.output
    assert candidates.exit_code == 0, candidates.output
    assert {"inspect", "import"} <= set(unstyle(capture.output).split())
    assert {"list", "show", "adopt", "reject"} <= set(
        unstyle(candidates.output).split()
    )


def test_capture_inspect_is_offline_non_persistent_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_runtime(*_: object, **__: object) -> object:
        raise AssertionError("capture inspect must not initialize runtime")

    monkeypatch.setattr(runtime_module, "ApplicationRuntime", forbidden_runtime)

    result = RUNNER.invoke(app, ["capture", "inspect", str(FIXTURE)])

    document = _canonical_success(result)
    assert document == {
        "data": {
            "adapter": "generic_jsonl",
            "candidate_count": 1,
            "manifest_hash": document["data"]["manifest_hash"],
            "owner_agent_id": "10000000-0000-4000-8000-000000000701",
            "persisted": False,
            "step_count": 3,
            "trajectory_id": "coding-agent-recovery-001",
        }
    }
    assert "stale cache" not in result.stdout.lower()
    assert "workspace/project" not in result.stdout


@pytest.mark.parametrize("kind", ("missing", "directory"))
def test_capture_inspect_rejects_non_files_with_stable_private_error(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "private-local-name"
    if kind == "directory":
        path.mkdir()

    result = RUNNER.invoke(app, ["capture", "inspect", str(path)])

    assert _canonical_error(result) == {
        "error": {
            "code": "capture_input_not_file",
            "details": {},
            "message": "Capture input must be a regular file",
        }
    }
    assert str(path) not in result.output


def test_capture_inspect_maps_read_failure_without_leaking_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private-unreadable-input.jsonl"
    path.write_bytes(b"retained only in the test temp directory")
    original_open = Path.open

    def fail_target_open(self: Path, *args: object, **kwargs: object) -> object:
        if self == path:
            raise PermissionError("private operating-system detail")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target_open)

    result = RUNNER.invoke(app, ["capture", "inspect", str(path)])

    assert _canonical_error(result) == {
        "error": {
            "code": "capture_input_read_failed",
            "details": {},
            "message": "Capture input could not be read",
        }
    }
    assert str(path) not in result.output
    assert "private operating-system detail" not in result.output


def test_capture_inspect_rejects_oversize_before_opening_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversize.jsonl"
    path.write_bytes(b"x" * (MAX_JSONL_INPUT_BYTES + 1))

    def forbidden_open(*_: object, **__: object) -> object:
        raise AssertionError("oversize input must be rejected before reading")

    monkeypatch.setattr(Path, "open", forbidden_open)

    result = RUNNER.invoke(app, ["capture", "inspect", str(path)])

    assert _canonical_error(result) == {
        "error": {
            "code": "capture_input_too_large",
            "details": {"max_bytes": MAX_JSONL_INPUT_BYTES},
            "message": "Capture input exceeds the size limit",
        }
    }


def test_capture_inspect_maps_invalid_input_without_echoing_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.jsonl"
    private_input = b"private invalid trajectory bytes"
    path.write_bytes(private_input)

    result = RUNNER.invoke(app, ["capture", "inspect", str(path)])

    assert _canonical_error(result) == {
        "error": {
            "code": "capture_input_invalid",
            "details": {},
            "message": "Capture input is invalid",
        }
    }
    assert private_input.decode("utf-8") not in result.output


def test_capture_inspect_reports_sensitive_location_without_echoing_secret(
    tmp_path: Path,
) -> None:
    records = [json.loads(line) for line in FIXTURE.read_bytes().split(b"\n")]
    probe = _synthetic_private_key_marker()
    records[1]["observation"] = probe
    path = tmp_path / "sensitive.jsonl"
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records))

    result = RUNNER.invoke(app, ["capture", "inspect", str(path)])

    assert _canonical_error(result) == {
        "error": {
            "code": "sensitive_input_detected",
            "details": {
                "matches": [
                    {
                        "field": "observation",
                        "rule_id": "private_key",
                        "step_id": "step-1",
                    }
                ]
            },
            "message": "Sensitive input was detected",
        }
    }
    assert probe not in result.output


def test_capture_inspect_has_no_database_option(tmp_path: Path) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(
        app,
        ["capture", "inspect", str(FIXTURE), "--database", str(database)],
    )

    assert result.exit_code == 2
    assert not database.exists()


def test_committed_fixture_is_canonical_synthetic_and_extracts_one_candidate() -> None:
    data = FIXTURE.read_bytes()
    lines = data.split(b"\n")
    assert len(lines) == 4
    for line in lines:
        assert canonical_json_bytes(json.loads(line)) == line

    preparer = CapturePreparer(
        adapter=GenericJsonlAdapter(),
        scanner=DefaultSecretScanner(),
        extractor=DeterministicSignalExtractor(),
    )
    prepared = preparer.prepare_jsonl(data)

    assert len(prepared.bundle.steps) == 3
    assert len(prepared.candidates) == 1
    assert tuple(step.status.value for step in prepared.bundle.steps) == (
        "unknown",
        "failed",
        "succeeded",
    )
    assert tuple(
        (item.step_id, item.field.value)
        for item in prepared.candidates[0].evidence
    ) == (("step-2", "outcome"), ("step-3", "action"))
    assert DefaultSecretScanner().scan(prepared.bundle) == ()
    assert "workspace/project" in data.decode("utf-8")
    for value in _strings([json.loads(line) for line in lines]):
        assert not Path(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()
    for forbidden in ("Bearer ", "PRIVATE KEY", "AKIA", "ghp_", "sk-"):
        assert forbidden.encode("utf-8") not in data


def test_real_cli_import_list_show_and_adopt_replay_exactly(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate-adopt.sqlite3"
    asyncio.run(_seed_owner(database))
    import_arguments = [
        "capture",
        "import",
        str(FIXTURE),
        "--database",
        str(database),
        "--idempotency-key",
        "capture-adopt-path",
    ]

    imported = RUNNER.invoke(app, import_arguments)

    imported_document = _canonical_success(imported)
    candidate_id = UUID(imported_document["data"]["candidate_ids"][0])
    assert imported_document["data"]["candidate_count"] == 1
    prepared = CapturePreparer(
        adapter=GenericJsonlAdapter(),
        scanner=DefaultSecretScanner(),
        extractor=DeterministicSignalExtractor(),
    ).prepare_jsonl(FIXTURE.read_bytes())
    import_request = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope=TRAJECTORY_IMPORT_SCOPE,
        idempotency_key="capture-adopt-path",
        method="POST",
        route_template="/v1/agents/{agent_id}/trajectory-bundles",
        path_parameters={"agent_id": OWNER_ID},
        body={
            "adapter": "generic_jsonl",
            "candidate_count": 1,
            "manifest_hash": prepared.bundle.manifest_hash,
        },
    )
    capture_receipt = _receipt(database, "capture-adopt-path")
    assert capture_receipt["scope"] == TRAJECTORY_IMPORT_SCOPE
    assert capture_receipt["request_hash"] == import_request.request_hash
    assert bytes(capture_receipt["response_body"]) == imported.stdout.rstrip(
        "\n"
    ).encode("utf-8")
    counts_after_import = _database_counts(database)

    replayed_import = RUNNER.invoke(app, import_arguments)

    assert replayed_import.exit_code == 0
    assert replayed_import.stdout == imported.stdout
    assert _database_counts(database) == counts_after_import

    counts_before_reads = _database_counts(database)
    listed = RUNNER.invoke(
        app,
        [
            "candidates",
            "list",
            str(OWNER_ID),
            "--database",
            str(database),
        ],
    )
    shown = RUNNER.invoke(
        app,
        [
            "candidates",
            "show",
            str(OWNER_ID),
            str(candidate_id),
            "--database",
            str(database),
        ],
    )

    listed_document = _canonical_success(listed)
    shown_document = _canonical_success(shown)
    assert [item["candidate_id"] for item in listed_document["data"]["items"]] == [
        str(candidate_id)
    ]
    assert listed_document["data"]["next_cursor"] is None
    assert listed_document["data"]["items"][0]["decision"] == "pending"
    assert shown_document["data"] == listed_document["data"]["items"][0]
    assert _database_counts(database) == counts_before_reads

    adopt_arguments = [
        "candidates",
        "adopt",
        str(OWNER_ID),
        str(candidate_id),
        "--importance",
        "0.7",
        "--confidence",
        "0.8",
        "--idempotency-key",
        "adopt-cli-candidate",
        "--database",
        str(database),
    ]
    adopted = RUNNER.invoke(app, adopt_arguments)

    adopted_document = _canonical_success(adopted)
    assert adopted_document["data"]["candidate_id"] == str(candidate_id)
    assert adopted_document["data"]["decision"] == "adopted"
    assert adopted_document["data"]["resulting_experience_id"] is not None
    assert adopted_document["data"]["resulting_version_id"] is not None
    adopt_request = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope=CANDIDATE_ADOPT_SCOPE,
        idempotency_key="adopt-cli-candidate",
        method="POST",
        route_template=(
            "/v1/agents/{agent_id}/candidates/{candidate_id}:adopt"
        ),
        path_parameters={
            "agent_id": OWNER_ID,
            "candidate_id": candidate_id,
        },
        body={"confidence": 0.8, "importance": 0.7},
    )
    adoption_receipt = _receipt(database, "adopt-cli-candidate")
    assert adoption_receipt["scope"] == CANDIDATE_ADOPT_SCOPE
    assert adoption_receipt["request_hash"] == adopt_request.request_hash
    counts_after_adoption = _database_counts(database)

    replayed_adoption = RUNNER.invoke(app, adopt_arguments)

    assert replayed_adoption.exit_code == 0
    assert replayed_adoption.stdout == adopted.stdout
    assert _database_counts(database) == counts_after_adoption


def test_real_cli_rejects_with_structured_reason_and_paginates_pending(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate-reject.sqlite3"
    asyncio.run(_seed_owner(database))
    candidate_ids: list[UUID] = []
    for label in ("first", "second", "third"):
        path = _variant_fixture(tmp_path, label)
        result = RUNNER.invoke(
            app,
            [
                "capture",
                "import",
                str(path),
                "--database",
                str(database),
                "--idempotency-key",
                f"capture-{label}",
            ],
        )
        document = _canonical_success(result)
        candidate_ids.append(UUID(document["data"]["candidate_ids"][0]))

    reason_text = "The recovery applies only to the synthetic fixture."
    reason = StructuredReason.from_user_text(reason_text)
    reject_arguments = [
        "candidates",
        "reject",
        str(OWNER_ID),
        str(candidate_ids[0]),
        "--reason",
        reason_text,
        "--idempotency-key",
        "reject-cli-candidate",
        "--database",
        str(database),
    ]

    rejected = RUNNER.invoke(app, reject_arguments)

    rejected_document = _canonical_success(rejected)
    assert rejected_document["data"]["decision"] == "rejected"
    assert rejected_document["data"]["reason"] == reason.model_dump(mode="json")
    reject_request = CommandRequest(
        caller_scope=f"agent:{OWNER_ID}",
        operation_scope=CANDIDATE_REJECT_SCOPE,
        idempotency_key="reject-cli-candidate",
        method="POST",
        route_template=(
            "/v1/agents/{agent_id}/candidates/{candidate_id}:reject"
        ),
        path_parameters={
            "agent_id": OWNER_ID,
            "candidate_id": candidate_ids[0],
        },
        body={"reason": reason.model_dump(mode="json")},
    )
    rejection_receipt = _receipt(database, "reject-cli-candidate")
    assert rejection_receipt["scope"] == CANDIDATE_REJECT_SCOPE
    assert rejection_receipt["request_hash"] == reject_request.request_hash
    counts_after_rejection = _database_counts(database)

    replayed_rejection = RUNNER.invoke(app, reject_arguments)

    assert replayed_rejection.exit_code == 0
    assert replayed_rejection.stdout == rejected.stdout
    assert _database_counts(database) == counts_after_rejection

    rejected_page = RUNNER.invoke(
        app,
        [
            "candidates",
            "list",
            str(OWNER_ID),
            "--decision",
            "rejected",
            "--database",
            str(database),
        ],
    )
    first_pending_page = RUNNER.invoke(
        app,
        [
            "candidates",
            "list",
            str(OWNER_ID),
            "--decision",
            "pending",
            "--limit",
            "1",
            "--database",
            str(database),
        ],
    )
    rejected_data = _canonical_success(rejected_page)["data"]
    first_pending_data = _canonical_success(first_pending_page)["data"]
    assert [item["candidate_id"] for item in rejected_data["items"]] == [
        str(candidate_ids[0])
    ]
    assert first_pending_data["next_cursor"] is not None

    second_pending_page = RUNNER.invoke(
        app,
        [
            "candidates",
            "list",
            str(OWNER_ID),
            "--decision",
            "pending",
            "--limit",
            "1",
            "--cursor",
            first_pending_data["next_cursor"],
            "--database",
            str(database),
        ],
    )
    second_pending_data = _canonical_success(second_pending_page)["data"]
    paged_ids = {
        UUID(first_pending_data["items"][0]["candidate_id"]),
        UUID(second_pending_data["items"][0]["candidate_id"]),
    }
    assert paged_ids == set(candidate_ids[1:])
    assert second_pending_data["next_cursor"] is None

    foreign = RUNNER.invoke(
        app,
        [
            "candidates",
            "show",
            str(OTHER_OWNER_ID),
            str(candidate_ids[0]),
            "--database",
            str(database),
        ],
    )
    missing = RUNNER.invoke(
        app,
        [
            "candidates",
            "show",
            str(OWNER_ID),
            str(MISSING_CANDIDATE_ID),
            "--database",
            str(database),
        ],
    )
    expected_not_found = {
        "error": {
            "code": "candidate_not_found",
            "details": {},
            "message": "Candidate was not found",
        }
    }
    assert _canonical_error(foreign) == expected_not_found
    assert _canonical_error(missing) == expected_not_found
    assert _database_counts(database) == counts_after_rejection


def test_command_help_exposes_required_keys_and_query_controls() -> None:
    expected = {
        ("capture", "inspect"): (),
        ("capture", "import"): ("--database", "--idempotency-key"),
        ("candidates", "list"): (
            "--cursor",
            "--database",
            "--decision",
            "--limit",
        ),
        ("candidates", "show"): ("--database",),
        ("candidates", "adopt"): (
            "--confidence",
            "--database",
            "--idempotency-key",
            "--importance",
        ),
        ("candidates", "reject"): (
            "--database",
            "--idempotency-key",
            "--reason",
        ),
    }
    for command, options in expected.items():
        result = RUNNER.invoke(app, [*command, "--help"])
        plain = unstyle(result.output)
        assert result.exit_code == 0, result.output
        for option in options:
            assert option in plain


@pytest.mark.parametrize(
    "arguments",
    (
        ("capture", "import", str(FIXTURE)),
        (
            "candidates",
            "adopt",
            str(OWNER_ID),
            str(MISSING_CANDIDATE_ID),
            "--importance",
            "0.7",
            "--confidence",
            "0.8",
        ),
        (
            "candidates",
            "reject",
            str(OWNER_ID),
            str(MISSING_CANDIDATE_ID),
            "--reason",
            "Not applicable.",
        ),
    ),
)
def test_mutations_require_explicit_idempotency_key(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(app, [*arguments, "--database", str(database)])

    assert result.exit_code == 2
    assert not database.exists()


@pytest.mark.parametrize("key", ("   ", "x" * 129))
def test_mutations_reject_invalid_explicit_key_before_runtime(
    tmp_path: Path,
    key: str,
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "capture",
            "import",
            str(FIXTURE),
            "--idempotency-key",
            key,
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 2
    assert not database.exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ("candidates", "list", "not-a-uuid"),
        ("candidates", "show", str(OWNER_ID), "not-a-uuid"),
        (
            "candidates",
            "adopt",
            str(OWNER_ID),
            "not-a-uuid",
            "--importance",
            "0.7",
            "--confidence",
            "0.8",
            "--idempotency-key",
            "invalid-adopt",
        ),
        (
            "candidates",
            "reject",
            str(OWNER_ID),
            "not-a-uuid",
            "--reason",
            "Not applicable.",
            "--idempotency-key",
            "invalid-reject",
        ),
        (
            "candidates",
            "list",
            str(OWNER_ID),
            "--decision",
            "PENDING",
        ),
        ("candidates", "list", str(OWNER_ID), "--limit", "0"),
        ("candidates", "list", str(OWNER_ID), "--limit", "101"),
        ("candidates", "list", str(OWNER_ID), "--limit", "1.5"),
    ),
)
def test_candidate_usage_validation_precedes_database_creation(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(app, [*arguments, "--database", str(database)])

    assert result.exit_code == 2
    assert not database.exists()


@pytest.mark.parametrize("score", ("-0.1", "1.1", "nan", "inf"))
def test_adopt_rejects_non_finite_or_out_of_range_scores_before_runtime(
    tmp_path: Path,
    score: str,
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "candidates",
            "adopt",
            str(OWNER_ID),
            str(MISSING_CANDIDATE_ID),
            "--importance",
            score,
            "--confidence",
            "0.8",
            "--idempotency-key",
            "invalid-score",
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 2
    assert not database.exists()


@pytest.mark.parametrize("reason", ("   ", "x" * 2_001))
def test_reject_validates_reason_before_runtime(
    tmp_path: Path,
    reason: str,
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    result = RUNNER.invoke(
        app,
        [
            "candidates",
            "reject",
            str(OWNER_ID),
            str(MISSING_CANDIDATE_ID),
            "--reason",
            reason,
            "--idempotency-key",
            "invalid-reason",
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 2
    assert not database.exists()


def test_invalid_cursor_is_stable_and_read_only(tmp_path: Path) -> None:
    database = tmp_path / "invalid-cursor.sqlite3"
    asyncio.run(_seed_owner(database))
    counts_before = _database_counts(database)

    result = RUNNER.invoke(
        app,
        [
            "candidates",
            "list",
            str(OWNER_ID),
            "--cursor",
            "not-a-canonical-cursor!",
            "--database",
            str(database),
        ],
    )

    assert _canonical_error(result) == {
        "error": {
            "code": "invalid_cursor",
            "details": {},
            "message": "The cursor is invalid.",
        }
    }
    assert _database_counts(database) == counts_before


def test_capture_import_rejects_sensitive_input_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [json.loads(line) for line in FIXTURE.read_bytes().split(b"\n")]
    probe = _synthetic_private_key_marker()
    records[1]["observation"] = probe
    path = tmp_path / "sensitive-import.jsonl"
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records))
    database = tmp_path / "must-not-exist.sqlite3"

    def forbidden_runtime(*_: object, **__: object) -> object:
        raise AssertionError("invalid input must not initialize runtime")

    monkeypatch.setattr(runtime_module, "ApplicationRuntime", forbidden_runtime)

    result = RUNNER.invoke(
        app,
        [
            "capture",
            "import",
            str(path),
            "--idempotency-key",
            "sensitive-import",
            "--database",
            str(database),
        ],
    )

    assert _canonical_error(result)["error"]["code"] == "sensitive_input_detected"
    assert probe not in result.output
    assert not database.exists()


@pytest.mark.parametrize(
    ("command", "header_field", "reported_field"),
    (
        ("inspect", "trajectory_id", "trajectory_id"),
        ("inspect", "sanitization_profile", "sanitization_profile_id"),
        ("import", "trajectory_id", "trajectory_id"),
        ("import", "sanitization_profile", "sanitization_profile_id"),
    ),
)
def test_cli_rejects_sensitive_header_before_runtime_without_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    header_field: str,
    reported_field: str,
) -> None:
    records = [json.loads(line) for line in FIXTURE.read_bytes().split(b"\n")]
    probe = _synthetic_openai_key()
    if header_field == "trajectory_id":
        records[0]["trajectory_id"] = probe
    else:
        records[0]["sanitization"]["profile_id"] = probe
    path = tmp_path / f"sensitive-{command}-{header_field}.jsonl"
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records))
    database = tmp_path / "must-not-exist.sqlite3"

    def forbidden_runtime(*_: object, **__: object) -> object:
        raise AssertionError("sensitive input must not initialize runtime")

    monkeypatch.setattr(runtime_module, "ApplicationRuntime", forbidden_runtime)
    arguments = ["capture", command, str(path)]
    if command == "import":
        arguments.extend(
            (
                "--idempotency-key",
                "sensitive-header-import",
                "--database",
                str(database),
            )
        )

    result = RUNNER.invoke(app, arguments)

    assert _canonical_error(result) == {
        "error": {
            "code": "sensitive_input_detected",
            "details": {
                "matches": [
                    {
                        "field": reported_field,
                        "rule_id": "openai_key",
                        "step_id": "header",
                    }
                ]
            },
            "message": "Sensitive input was detected",
        }
    }
    assert probe not in result.output
    assert not database.exists()


def test_runtime_failure_is_canonical_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "must-not-exist.sqlite3"

    def fail_runtime(*_: object, **__: object) -> object:
        raise RuntimeError("private runtime diagnostic")

    monkeypatch.setattr(runtime_module, "ApplicationRuntime", fail_runtime)

    result = RUNNER.invoke(
        app,
        [
            "capture",
            "import",
            str(FIXTURE),
            "--idempotency-key",
            "runtime-failure",
            "--database",
            str(database),
        ],
    )

    assert _canonical_error(result) == {
        "error": {
            "code": "internal_error",
            "details": {},
            "message": "The operation failed unexpectedly",
        }
    }
    assert "private runtime diagnostic" not in result.output
