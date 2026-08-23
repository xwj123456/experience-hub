"""CLI adapter for the isolated ExperienceBench-S pilot."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments.benchmarks import (
    BenchmarkEvidenceReportV1,
    BenchmarkExecution,
    BenchmarkInspection,
    inspect_benchmark_pack,
    run_benchmark_pilot,
    verify_benchmark_report,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.reports import ExperimentOutputError

benchmark_app = typer.Typer(
    help="Inspect and run the ExperienceBench-S pilot.",
    no_args_is_help=True,
    add_completion=False,
)


def _error_document(error: BaseException) -> dict[str, Any]:
    if isinstance(error, ExperimentInputError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Benchmark input is invalid",
            }
        }
    if isinstance(error, ExperimentIsolationError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Benchmark isolation requirements were not met",
            }
        }
    if isinstance(error, ExperimentOutputError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Benchmark evidence is invalid",
            }
        }
    return {
        "error": {
            "code": "internal_error",
            "details": {},
            "message": "The benchmark operation failed unexpectedly",
        }
    }


def _emit_document(document: Mapping[str, Any]) -> None:
    from experience_hub.cli.app import _emit_document as emit_document

    emit_document(document)


def _exit_with_error(error: BaseException) -> NoReturn:
    _emit_document(_error_document(error))
    raise typer.Exit(1) from None


def _inspection_document(inspection: BenchmarkInspection) -> dict[str, Any]:
    return {
        "data": {
            "arm_count": inspection.arm_count,
            "case_count": inspection.case_count,
            "cases_sha256": inspection.cases_sha256,
            "fts5_available": inspection.fts5_available,
            "manifest_sha256": inspection.manifest_sha256,
            "pack_id": inspection.pack_id,
            "source_fixture_sha256": inspection.source_fixture_sha256,
        }
    }


def _execution_document(execution: BenchmarkExecution) -> dict[str, Any]:
    summary = execution.summary.data
    return {
        "data": {
            "arm_count": summary.arm_count,
            "case_count": summary.case_count,
            "comparison_complete": execution.comparison_complete,
            "deterministic_replay_match": execution.deterministic_replay_match,
            "evidence_sha256": summary.evidence_sha256,
            "evidence_valid": execution.evidence_valid,
            "expansion_gate_passed": execution.expansion_gate_passed,
            "manifest_sha256": summary.manifest_sha256,
            "profile_complete": execution.profile_complete,
            "snapshot_sha256": summary.snapshot_sha256,
            "source_fixture_sha256": summary.source_fixture_sha256,
        }
    }


def _verification_document(
    evidence: BenchmarkEvidenceReportV1,
) -> dict[str, Any]:
    data = evidence.data
    payload = data.pass_payload
    resolved = payload.resolved_manifest
    return {
        "data": {
            "arm_count": len(resolved.arms),
            "case_count": len(payload.cases),
            "comparison_complete": payload.comparison_complete,
            "deterministic_replay_match": data.deterministic_replay_match,
            "evidence_sha256": sha256_hex(canonical_json_bytes(evidence)),
            "evidence_valid": data.valid,
            "expansion_gate_passed": data.expansion_gate_passed,
            "manifest_sha256": resolved.manifest_sha256,
            "snapshot_sha256": resolved.snapshot_sha256,
            "source_fixture_sha256": resolved.source_fixture_sha256,
        }
    }


@benchmark_app.command("inspect")
def benchmark_inspect(
    pack: Annotated[Path, typer.Option("--pack")],
) -> None:
    """Validate one pilot pack without creating retained output."""
    try:
        inspection = asyncio.run(inspect_benchmark_pack(pack))
        document = _inspection_document(inspection)
    except Exception as error:
        _exit_with_error(error)
    _emit_document(document)


@benchmark_app.command("run")
def benchmark_run(
    pack: Annotated[Path, typer.Option("--pack")],
    workspace: Annotated[Path, typer.Option("--workspace")],
    replace_owned: Annotated[
        bool,
        typer.Option(
            "--replace-owned",
            help="Replace only entries owned by an existing replay workspace.",
        ),
    ] = False,
) -> None:
    """Run the two-pass pilot and publish verified evidence."""
    try:
        execution = asyncio.run(
            run_benchmark_pilot(
                pack,
                workspace,
                replace_owned=replace_owned,
            )
        )
        document = _execution_document(execution)
    except Exception as error:
        _exit_with_error(error)
    _emit_document(document)
    if (
        not execution.evidence_valid
        or not execution.comparison_complete
        or not execution.deterministic_replay_match
        or not execution.expansion_gate_passed
        or not execution.profile_complete
    ):
        raise typer.Exit(1)


@benchmark_app.command("verify")
def benchmark_verify(
    report: Annotated[Path, typer.Option("--report")],
) -> None:
    """Verify canonical benchmark evidence and its colocated summary."""
    try:
        evidence = verify_benchmark_report(report)
        document = _verification_document(evidence)
    except Exception as error:
        _exit_with_error(error)
    _emit_document(document)
    data = evidence.data
    if (
        not data.valid
        or not data.pass_payload.comparison_complete
        or not data.deterministic_replay_match
        or not data.expansion_gate_passed
    ):
        raise typer.Exit(1)


__all__ = ["benchmark_app"]
