from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest
from click import unstyle
from typer.testing import CliRunner

import experience_hub.config as config
import experience_hub.experiments.runner as runner_module
from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.cli.app import app
from experience_hub.experiments import (
    PolicyArm,
    PolicyExecutionContext,
)

RUNNER = CliRunner()
ROOT = Path(__file__).parents[2]
SMOKE_MANIFEST = ROOT / "examples" / "replay" / "smoke-manifest.json"
SMOKE_CASES = ROOT / "examples" / "replay" / "smoke-cases.jsonl"
CASES_SHA256 = "7d74fd4618af3adc800ccfaac039d3eb749a4b54de51dd6913f82fd5321a058e"
PRIVATE_UUID = "10000000-0000-4000-8000-000000000799"


def _canonical_document(result: Any, *, exit_code: int) -> dict[str, Any]:
    assert result.exit_code == exit_code, f"{result.output}\n{result.exception!r}"
    assert result.stdout.count("\n") == 1
    body = result.stdout.removesuffix("\n").encode("utf-8")
    document = cast(dict[str, Any], json.loads(body))
    assert canonical_json_bytes(document) == body
    return document


def _demo_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    monkeypatch.setattr(config, "repository_root", lambda: tmp_path)
    result = RUNNER.invoke(app, ["demo", "--reset"])
    assert result.exit_code == 0, result.output
    database = tmp_path / ".data" / "demo.db"
    assert database.is_file()
    return database


def _run(
    *,
    manifest: Path,
    database: Path,
    workspace: Path,
    replace_owned: bool = False,
) -> Any:
    arguments = [
        "replay",
        "run",
        "--manifest",
        str(manifest),
        "--database",
        str(database),
        "--workspace",
        str(workspace),
    ]
    if replace_owned:
        arguments.append("--replace-owned")
    return RUNNER.invoke(app, arguments)


def test_replay_help_exposes_inspect_run_and_verify() -> None:
    result = RUNNER.invoke(app, ["replay", "--help"])

    assert result.exit_code == 0, result.output
    assert {"inspect", "run", "verify"} <= set(unstyle(result.output).split())


def test_committed_smoke_fixture_is_exact_and_canonical() -> None:
    cases = SMOKE_CASES.read_bytes()
    manifest = SMOKE_MANIFEST.read_bytes()

    assert cases.endswith(b"\n")
    assert cases.count(b"\n") == 2
    assert sha256_hex(cases) == CASES_SHA256
    assert canonical_json_bytes(json.loads(manifest)) == manifest
    assert json.loads(manifest) == {
        "arms": [
            {
                "arm_id": "no_memory",
                "kind": "no_memory",
                "required": True,
                "schema_version": 1,
            },
            {
                "arm_id": "experience_hub",
                "kind": "experience_hub",
                "required": True,
                "schema_version": 1,
            },
        ],
        "dataset": {
            "cases_file": "smoke-cases.jsonl",
            "cases_sha256": CASES_SHA256,
            "dataset_id": "smoke-cases",
            "schema_version": 1,
        },
        "deterministic_replay_runs": 2,
        "evidence_schema_version": 1,
        "experiment_id": "smoke-replay",
        "frozen_at": "2026-01-03T09:15:01.000000Z",
        "oracle": {
            "kind": "retrieval_labels",
            "schema_version": 1,
            "version": 1,
        },
        "profile_schema_version": 1,
        "schema_version": 1,
        "seed": 20260726,
        "snapshot_binding": "validated_source",
    }


def test_demo_replay_smoke_is_canonical_private_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    workspace = tmp_path / "replay"
    original_clone = runner_module.clone_frozen_sqlite

    def forbidden_clone(*_: object, **__: object) -> Path:
        raise AssertionError("inspect must not create arm clones")

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", forbidden_clone)
    inspected = RUNNER.invoke(
        app,
        [
            "replay",
            "inspect",
            "--manifest",
            str(SMOKE_MANIFEST),
            "--database",
            str(database),
        ],
    )
    inspect_document = _canonical_document(inspected, exit_code=0)
    inspect_data = cast(dict[str, Any], inspect_document["data"])
    assert inspect_data == {
        "arm_count": 2,
        "case_count": 2,
        "cases_sha256": CASES_SHA256,
        "dataset_id": "smoke-cases",
        "experiment_id": "smoke-replay",
        "manifest_sha256": sha256_hex(SMOKE_MANIFEST.read_bytes()),
        "snapshot_sha256": inspect_data["snapshot_sha256"],
        "source_schema_revision": inspect_data["source_schema_revision"],
    }
    assert not (SMOKE_MANIFEST.parent / "arms").exists()

    monkeypatch.setattr(runner_module, "clone_frozen_sqlite", original_clone)
    run_result = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
    )
    run_document = _canonical_document(run_result, exit_code=0)
    run_data = cast(dict[str, Any], run_document["data"])
    assert run_data == {
        "arm_count": 2,
        "case_count": 2,
        "cases_sha256": CASES_SHA256,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "manifest_sha256": inspect_data["manifest_sha256"],
        "profile_complete": True,
        "snapshot_sha256": inspect_data["snapshot_sha256"],
        "valid": True,
    }
    assert "artifacts" not in run_result.stdout
    assert str(workspace) not in run_result.stdout
    assert PRIVATE_UUID not in run_result.stdout

    report = workspace / "artifacts" / "evidence.json"
    verified = RUNNER.invoke(
        app,
        ["replay", "verify", "--report", str(report)],
    )
    verify_document = _canonical_document(verified, exit_code=0)
    verify_data = cast(dict[str, Any], verify_document["data"])
    assert verify_data == {
        "arm_count": 2,
        "case_count": 2,
        "cases_sha256": CASES_SHA256,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "manifest_sha256": inspect_data["manifest_sha256"],
        "snapshot_sha256": inspect_data["snapshot_sha256"],
        "valid": True,
    }
    assert str(report) not in verified.stdout


@pytest.mark.parametrize("kind", ("tampered", "missing", "symlink"))
def test_verify_rejects_untrusted_report_without_leaking_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    workspace = tmp_path / "replay"
    generated = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
    )
    assert generated.exit_code == 0, generated.output
    report = workspace / "artifacts" / "evidence.json"
    expected_code = "invalid_evidence"
    if kind == "tampered":
        document = json.loads(report.read_bytes())
        document["data"]["valid"] = False
        report.write_bytes(canonical_json_bytes(document))
    elif kind == "missing":
        report = tmp_path / "private-missing-evidence.json"
        expected_code = "invalid_report_path"
    else:
        target = report
        report = tmp_path / "private-symlink-evidence.json"
        report.symlink_to(target)
        expected_code = "invalid_report_path"

    result = RUNNER.invoke(
        app,
        ["replay", "verify", "--report", str(report)],
    )

    assert _canonical_document(result, exit_code=1) == {
        "error": {
            "code": expected_code,
            "details": {},
            "message": "Replay evidence is invalid",
        }
    }
    assert str(report) not in result.stdout


def test_run_rejects_nonempty_unmarked_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    workspace = tmp_path / "private-replay"
    workspace.mkdir()
    retained = workspace / "keep.txt"
    retained.write_text("private retained content", encoding="utf-8")

    result = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
    )

    assert _canonical_document(result, exit_code=1) == {
        "error": {
            "code": "replay_workspace_unowned",
            "details": {},
            "message": "Replay isolation requirements were not met",
        }
    }
    assert retained.read_text(encoding="utf-8") == "private retained content"
    assert str(workspace) not in result.stdout


def test_run_replaces_only_an_owned_workspace_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    workspace = tmp_path / "replay"
    first = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
    )
    assert first.exit_code == 0, first.output

    refused = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
    )
    assert _canonical_document(refused, exit_code=1) == {
        "error": {
            "code": "replay_workspace_exists",
            "details": {},
            "message": "Replay isolation requirements were not met",
        }
    }

    replaced = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=workspace,
        replace_owned=True,
    )
    document = _canonical_document(replaced, exit_code=0)
    assert document["data"]["valid"] is True


def test_inspect_rejects_nonempty_wal_without_leaking_private_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    wal = Path(f"{database}-wal")
    wal.write_text(f"{PRIVATE_UUID} private WAL detail", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "replay",
            "inspect",
            "--manifest",
            str(SMOKE_MANIFEST),
            "--database",
            str(database),
        ],
    )

    assert _canonical_document(result, exit_code=1) == {
        "error": {
            "code": "replay_snapshot_invalid",
            "details": {},
            "message": "Replay isolation requirements were not met",
        }
    }
    assert str(database) not in result.stdout
    assert PRIVATE_UUID not in result.stdout


def test_run_exits_one_when_a_required_arm_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    original_builder = runner_module.build_policy_arm

    class FailingArm:
        def __init__(self, wrapped: PolicyArm) -> None:
            self.descriptor = wrapped.descriptor

        async def execute(self, context: PolicyExecutionContext) -> object:
            del context
            raise RuntimeError(f"{tmp_path}/{PRIVATE_UUID}/private failure")

    def failing_builder(descriptor: Any) -> PolicyArm:
        arm = original_builder(descriptor)
        if descriptor.arm_id == "experience_hub":
            return cast(PolicyArm, FailingArm(arm))
        return arm

    monkeypatch.setattr(runner_module, "build_policy_arm", failing_builder)
    result = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=tmp_path / "replay",
    )

    document = _canonical_document(result, exit_code=1)
    data = cast(dict[str, Any], document["data"])
    assert data["comparison_complete"] is False
    assert data["deterministic_replay_match"] is True
    assert data["profile_complete"] is True
    assert data["valid"] is False
    assert str(tmp_path) not in result.stdout
    assert PRIVATE_UUID not in result.stdout
    assert "private failure" not in result.stdout


def test_run_exits_one_when_profile_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _demo_database(monkeypatch, tmp_path)
    replay_commands = import_module("experience_hub.cli.replay_commands")
    real_run = runner_module.run_replay

    async def run_with_failed_profiler(
        manifest_path: Path,
        database_path: Path,
        workspace_path: Path,
        *,
        replace_owned: bool = False,
    ) -> Any:
        def failed_profiler() -> int:
            raise RuntimeError(f"{tmp_path}/{PRIVATE_UUID}/private profile failure")

        return await real_run(
            manifest_path,
            database_path,
            workspace_path,
            replace_owned=replace_owned,
            profiler=failed_profiler,
        )

    monkeypatch.setattr(replay_commands, "run_replay", run_with_failed_profiler)
    result = _run(
        manifest=SMOKE_MANIFEST,
        database=database,
        workspace=tmp_path / "replay",
    )

    document = _canonical_document(result, exit_code=1)
    data = cast(dict[str, Any], document["data"])
    assert data["comparison_complete"] is True
    assert data["deterministic_replay_match"] is True
    assert data["profile_complete"] is False
    assert data["valid"] is False
    assert str(tmp_path) not in result.stdout
    assert PRIVATE_UUID not in result.stdout
    assert "private profile failure" not in result.stdout


def test_private_unexpected_exception_is_mapped_without_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay_commands = import_module("experience_hub.cli.replay_commands")

    async def fail_inspect(*_: object, **__: object) -> object:
        raise RuntimeError(f"{tmp_path}/{PRIVATE_UUID}/private exception")

    monkeypatch.setattr(replay_commands, "inspect_replay", fail_inspect)
    result = RUNNER.invoke(
        app,
        [
            "replay",
            "inspect",
            "--manifest",
            str(SMOKE_MANIFEST),
            "--database",
            str(tmp_path / "private.db"),
        ],
    )

    assert _canonical_document(result, exit_code=1) == {
        "error": {
            "code": "internal_error",
            "details": {},
            "message": "The replay operation failed unexpectedly",
        }
    }
    assert str(tmp_path) not in result.stdout
    assert PRIVATE_UUID not in result.stdout
    assert "private exception" not in result.stdout
