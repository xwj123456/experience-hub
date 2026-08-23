from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest
from click import unstyle
from tests.benchmark.test_experiencebench_reports import _report as _passing_report
from tests.integration.test_experiencebench_runner import _tree_bytes, _write_pack
from typer.testing import CliRunner

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.cli.app import app
from experience_hub.experiments.benchmarks import (
    BenchmarkEvidenceReportV1,
    BenchmarkExecution,
    run_benchmark_pilot,
)
from experience_hub.experiments.benchmarks.reports import (
    canonical_benchmark_evidence_bytes,
    derive_benchmark_summary,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.reports import ExperimentOutputError

RUNNER = CliRunner()
PRIVATE_VALUE = "private-benchmark-value-8157"


@pytest.fixture(scope="module")
def pilot_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, BenchmarkExecution]:
    root = tmp_path_factory.mktemp("experiencebench-cli", numbered=True).resolve()
    pack = _write_pack(root / "pack")
    workspace = root / "workspace"
    execution = asyncio.run(run_benchmark_pilot(pack, workspace))
    return pack, workspace, execution


def _canonical_document(result: Any, *, exit_code: int) -> dict[str, Any]:
    assert result.exit_code == exit_code, f"{result.output}\n{result.exception!r}"
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    body = result.stdout.removesuffix("\n").encode("utf-8")
    document = cast(dict[str, Any], json.loads(body))
    assert canonical_json_bytes(document) == body
    return document


def _benchmark_commands() -> Any:
    return import_module("experience_hub.cli.benchmark_commands")


def _run(pack: Path, workspace: Path, *, replace_owned: bool = False) -> Any:
    arguments = [
        "replay",
        "benchmark",
        "run",
        "--pack",
        str(pack),
        "--workspace",
        str(workspace),
    ]
    if replace_owned:
        arguments.append("--replace-owned")
    return RUNNER.invoke(app, arguments)


def _positive_execution(execution: BenchmarkExecution) -> BenchmarkExecution:
    evidence = _passing_report()
    evidence_body = canonical_benchmark_evidence_bytes(evidence)
    summary = derive_benchmark_summary(evidence, evidence_body=evidence_body)
    return replace(
        execution,
        evidence=evidence,
        evidence_body=evidence_body,
        summary=summary,
        evidence_valid=True,
        comparison_complete=True,
        deterministic_replay_match=True,
        expansion_gate_passed=True,
        profile_complete=True,
    )


def test_benchmark_help_and_no_args_expose_exact_nested_commands() -> None:
    help_result = RUNNER.invoke(app, ["replay", "benchmark", "--help"])
    no_args = RUNNER.invoke(app, ["replay", "benchmark"])

    assert help_result.exit_code == 0, help_result.output
    assert no_args.exit_code == 2, no_args.output
    for result in (help_result, no_args):
        output = unstyle(result.output)
        assert set(
            re.findall(r"^│ ([a-z][a-z-]*)\s{2,}", output, flags=re.MULTILINE)
        ) == {"inspect", "run", "verify"}
        assert "Inspect and run the ExperienceBench-S pilot." in output


@pytest.mark.parametrize(
    ("command", "expected_options"),
    (
        ("inspect", {"--help", "--pack"}),
        ("run", {"--help", "--pack", "--replace-owned", "--workspace"}),
        ("verify", {"--help", "--report"}),
    ),
)
def test_benchmark_subcommand_options_are_exact(
    command: str,
    expected_options: set[str],
) -> None:
    result = RUNNER.invoke(app, ["replay", "benchmark", command, "--help"])

    assert result.exit_code == 0, result.output
    assert set(re.findall(r"--[a-z][a-z-]*", unstyle(result.output))) == (
        expected_options
    )


def test_inspect_emits_canonical_summary_without_writing_pack(
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
) -> None:
    pack, _, execution = pilot_bundle
    before = _tree_bytes(pack.parent)

    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "inspect", "--pack", str(pack)],
    )

    document = _canonical_document(result, exit_code=0)
    assert document == {
        "data": {
            "arm_count": 4,
            "case_count": 30,
            "cases_sha256": execution.summary.data.cases_sha256,
            "fts5_available": True,
            "manifest_sha256": execution.summary.data.manifest_sha256,
            "pack_id": "experiencebench-s-pilot",
            "source_fixture_sha256": execution.summary.data.source_fixture_sha256,
        }
    }
    assert _tree_bytes(pack.parent) == before
    assert str(pack) not in result.stdout


def test_run_success_emits_exact_canonical_summary_fields(
    monkeypatch: pytest.MonkeyPatch,
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
    tmp_path: Path,
) -> None:
    pack, _, execution = pilot_bundle
    positive = _positive_execution(execution)
    expected_workspace = tmp_path / "unused-workspace"

    async def successful_run(
        pack_path: Path,
        workspace_path: Path,
        *,
        replace_owned: bool = False,
    ) -> BenchmarkExecution:
        if (
            pack_path != pack
            or workspace_path != expected_workspace
            or not replace_owned
        ):
            raise AssertionError("benchmark run options were not forwarded")
        return positive

    monkeypatch.setattr(_benchmark_commands(), "run_benchmark_pilot", successful_run)
    result = _run(pack, expected_workspace, replace_owned=True)

    document = _canonical_document(result, exit_code=0)
    assert set(cast(dict[str, Any], document["data"])) == {
        "arm_count",
        "case_count",
        "comparison_complete",
        "deterministic_replay_match",
        "evidence_sha256",
        "evidence_valid",
        "expansion_gate_passed",
        "manifest_sha256",
        "profile_complete",
        "snapshot_sha256",
        "source_fixture_sha256",
    }
    assert document["data"] == {
        "arm_count": 4,
        "case_count": 30,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "evidence_sha256": positive.summary.data.evidence_sha256,
        "evidence_valid": True,
        "expansion_gate_passed": True,
        "manifest_sha256": positive.summary.data.manifest_sha256,
        "profile_complete": True,
        "snapshot_sha256": positive.summary.data.snapshot_sha256,
        "source_fixture_sha256": positive.summary.data.source_fixture_sha256,
    }
    combined = result.stdout + result.stderr
    assert str(pack) not in combined
    assert str(expected_workspace) not in combined


def test_run_valid_negative_gate_exits_one_and_retains_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
    tmp_path: Path,
) -> None:
    pack, _, execution = pilot_bundle
    retained_workspace = tmp_path / "negative-workspace"

    async def negative_run(
        _: Path,
        workspace_path: Path,
        *,
        replace_owned: bool = False,
    ) -> BenchmarkExecution:
        assert replace_owned is False
        artifacts = workspace_path / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "benchmark-evidence.json").write_bytes(execution.evidence_body)
        (artifacts / "benchmark-summary.json").write_bytes(execution.summary_body)
        return execution

    monkeypatch.setattr(_benchmark_commands(), "run_benchmark_pilot", negative_run)
    result = _run(pack, retained_workspace)

    document = _canonical_document(result, exit_code=1)
    assert document["data"]["evidence_valid"] is True
    assert document["data"]["expansion_gate_passed"] is False
    assert (retained_workspace / "artifacts" / "benchmark-evidence.json").is_file()
    assert (retained_workspace / "artifacts" / "benchmark-summary.json").is_file()


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("comparison_complete", False),
        ("deterministic_replay_match", False),
        ("evidence_valid", False),
        ("profile_complete", False),
    ),
)
def test_run_exits_one_for_each_incomplete_status(
    monkeypatch: pytest.MonkeyPatch,
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
    tmp_path: Path,
    field: str,
    expected: bool,
) -> None:
    pack, _, execution = pilot_bundle
    changed = replace(_positive_execution(execution), **{field: expected})

    async def incomplete_run(*_: object, **__: object) -> BenchmarkExecution:
        return changed

    monkeypatch.setattr(_benchmark_commands(), "run_benchmark_pilot", incomplete_run)
    result = _run(pack, tmp_path / f"unused-{field}")

    document = _canonical_document(result, exit_code=1)
    assert document["data"][field] is expected


def test_verify_success_emits_canonical_authoritative_summary(
    monkeypatch: pytest.MonkeyPatch,
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
) -> None:
    _, workspace, _ = pilot_bundle
    evidence = _passing_report()
    report = workspace / "artifacts" / "benchmark-evidence.json"

    def successful_verify(path: Path) -> BenchmarkEvidenceReportV1:
        if path != report:
            raise AssertionError("benchmark report option was not forwarded")
        return evidence

    monkeypatch.setattr(
        _benchmark_commands(), "verify_benchmark_report", successful_verify
    )
    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "verify", "--report", str(report)],
    )

    document = _canonical_document(result, exit_code=0)
    data = cast(dict[str, Any], document["data"])
    resolved = evidence.data.pass_payload.resolved_manifest
    assert data == {
        "arm_count": 4,
        "case_count": 30,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "evidence_sha256": sha256_hex(canonical_json_bytes(evidence)),
        "evidence_valid": True,
        "expansion_gate_passed": True,
        "manifest_sha256": resolved.manifest_sha256,
        "snapshot_sha256": resolved.snapshot_sha256,
        "source_fixture_sha256": resolved.source_fixture_sha256,
    }
    assert str(report) not in result.stdout + result.stderr


def test_verify_valid_negative_report_emits_truthful_nonzero_summary(
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
) -> None:
    _, workspace, execution = pilot_bundle
    report = workspace / "artifacts" / "benchmark-evidence.json"

    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "verify", "--report", str(report)],
    )

    document = _canonical_document(result, exit_code=1)
    assert document["data"] == {
        "arm_count": 4,
        "case_count": 30,
        "comparison_complete": True,
        "deterministic_replay_match": True,
        "evidence_sha256": execution.summary.data.evidence_sha256,
        "evidence_valid": True,
        "expansion_gate_passed": False,
        "manifest_sha256": execution.summary.data.manifest_sha256,
        "snapshot_sha256": execution.summary.data.snapshot_sha256,
        "source_fixture_sha256": execution.summary.data.source_fixture_sha256,
    }
    assert str(report) not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "field",
    (
        "comparison_complete",
        "deterministic_replay_match",
        "evidence_valid",
        "expansion_gate_passed",
    ),
)
def test_verify_exits_one_for_each_invalid_evidence_status(
    monkeypatch: pytest.MonkeyPatch,
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
    field: str,
) -> None:
    _, workspace, _ = pilot_bundle
    evidence = _passing_report()
    data = evidence.data
    if field == "comparison_complete":
        payload = data.pass_payload.model_copy(
            update={"comparison_complete": False}
        )
        data = data.model_copy(update={"pass_payload": payload})
    elif field == "evidence_valid":
        data = data.model_copy(update={"valid": False})
    else:
        data = data.model_copy(update={field: False})
    changed = BenchmarkEvidenceReportV1.model_construct(data=data)

    def failed_gate(_: Path) -> BenchmarkEvidenceReportV1:
        return changed

    monkeypatch.setattr(_benchmark_commands(), "verify_benchmark_report", failed_gate)
    report = workspace / "artifacts" / "benchmark-evidence.json"
    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "verify", "--report", str(report)],
    )

    document = _canonical_document(result, exit_code=1)
    assert document["data"][field] is False


def test_verify_rejects_tampered_summary_without_leaking_path(
    pilot_bundle: tuple[Path, Path, BenchmarkExecution],
    tmp_path: Path,
) -> None:
    _, workspace, _ = pilot_bundle
    artifacts = tmp_path / "artifacts"
    shutil.copytree(workspace / "artifacts", artifacts)
    summary = artifacts / "benchmark-summary.json"
    document = json.loads(summary.read_bytes())
    document["data"]["evidence_sha256"] = "0" * 64
    summary.write_bytes(canonical_json_bytes(document))
    report = artifacts / "benchmark-evidence.json"

    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "verify", "--report", str(report)],
    )

    assert _canonical_document(result, exit_code=1) == {
        "error": {
            "code": "invalid_benchmark_summary",
            "details": {},
            "message": "Benchmark evidence is invalid",
        }
    }
    assert str(report) not in result.stdout
    assert str(report) not in result.stderr


@pytest.mark.parametrize(
    ("error", "message"),
    (
        (
            ExperimentInputError("benchmark_invalid_pack", PRIVATE_VALUE),
            "Benchmark input is invalid",
        ),
        (
            ExperimentIsolationError("benchmark_workspace_invalid", PRIVATE_VALUE),
            "Benchmark isolation requirements were not met",
        ),
        (
            ExperimentOutputError("invalid_benchmark_evidence", PRIVATE_VALUE),
            "Benchmark evidence is invalid",
        ),
        (RuntimeError(PRIVATE_VALUE), "The benchmark operation failed unexpectedly"),
    ),
)
def test_errors_use_stable_codes_and_hide_private_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    message: str,
) -> None:
    private_pack = tmp_path / PRIVATE_VALUE / "manifest.json"

    async def failed_inspect(_: Path) -> object:
        raise error

    monkeypatch.setattr(_benchmark_commands(), "inspect_benchmark_pack", failed_inspect)
    result = RUNNER.invoke(
        app,
        ["replay", "benchmark", "inspect", "--pack", str(private_pack)],
    )

    code = getattr(error, "code", "internal_error")
    assert _canonical_document(result, exit_code=1) == {
        "error": {"code": code, "details": {}, "message": message}
    }
    combined = result.stdout + result.stderr
    assert str(private_pack) not in combined
    assert PRIVATE_VALUE not in combined
