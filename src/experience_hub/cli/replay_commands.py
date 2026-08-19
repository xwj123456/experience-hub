"""CLI adapter for isolated deterministic replay experiments."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from experience_hub.experiments import (
    ExperimentInputError,
    ExperimentIsolationError,
    ExperimentOutputError,
    ReplayEvidenceReportV1,
    ReplayExecution,
    ReplayInspection,
    inspect_replay,
    run_replay,
    verify_replay_report,
)

replay_app = typer.Typer(
    help="Inspect and run isolated deterministic replays.",
    no_args_is_help=True,
    add_completion=False,
)


def _error_document(error: BaseException) -> dict[str, Any]:
    if isinstance(error, ExperimentInputError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Replay input is invalid",
            }
        }
    if isinstance(error, ExperimentIsolationError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Replay isolation requirements were not met",
            }
        }
    if isinstance(error, ExperimentOutputError):
        return {
            "error": {
                "code": error.code,
                "details": {},
                "message": "Replay evidence is invalid",
            }
        }
    return {
        "error": {
            "code": "internal_error",
            "details": {},
            "message": "The replay operation failed unexpectedly",
        }
    }


def _emit_document(document: Mapping[str, Any]) -> None:
    from experience_hub.cli.app import _emit_document as emit_document

    emit_document(document)


def _exit_with_error(error: BaseException) -> NoReturn:
    _emit_document(_error_document(error))
    raise typer.Exit(1) from None


def _inspection_document(inspection: ReplayInspection) -> dict[str, Any]:
    return {
        "data": {
            "arm_count": inspection.arm_count,
            "case_count": inspection.case_count,
            "cases_sha256": inspection.cases_sha256,
            "dataset_id": inspection.dataset_id,
            "experiment_id": inspection.experiment_id,
            "manifest_sha256": inspection.manifest_sha256,
            "snapshot_sha256": inspection.snapshot_sha256,
            "source_schema_revision": inspection.source_schema_revision,
        }
    }


def _execution_document(execution: ReplayExecution) -> dict[str, Any]:
    data = execution.evidence.data
    resolved = data.resolved_manifest
    return {
        "data": {
            "arm_count": len(resolved.policy_arms),
            "case_count": len(data.cases),
            "cases_sha256": resolved.cases_sha256,
            "comparison_complete": data.comparison_complete,
            "deterministic_replay_match": data.deterministic_replay_match,
            "manifest_sha256": resolved.manifest_sha256,
            "profile_complete": execution.profile_complete,
            "snapshot_sha256": resolved.snapshot_sha256,
            "valid": execution.valid,
        }
    }


def _verification_document(report: ReplayEvidenceReportV1) -> dict[str, Any]:
    data = report.data
    resolved = data.resolved_manifest
    return {
        "data": {
            "arm_count": len(resolved.policy_arms),
            "case_count": len(data.cases),
            "cases_sha256": resolved.cases_sha256,
            "comparison_complete": data.comparison_complete,
            "deterministic_replay_match": data.deterministic_replay_match,
            "manifest_sha256": resolved.manifest_sha256,
            "snapshot_sha256": resolved.snapshot_sha256,
            "valid": data.valid,
        }
    }


@replay_app.command("inspect")
def replay_inspect(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical replay manifest."),
    ],
    database: Annotated[
        Path,
        typer.Option("--database", help="Closed SQLite source database."),
    ],
) -> None:
    """Validate replay inputs without retaining arm clones."""
    try:
        inspection = asyncio.run(inspect_replay(manifest, database))
    except Exception as error:
        _exit_with_error(error)
    _emit_document(_inspection_document(inspection))


@replay_app.command("run")
def replay_run(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical replay manifest."),
    ],
    database: Annotated[
        Path,
        typer.Option("--database", help="Closed SQLite source database."),
    ],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", help="Owned replay output workspace."),
    ],
    replace_owned: Annotated[
        bool,
        typer.Option(
            "--replace-owned",
            help="Replace only entries owned by an existing replay workspace.",
        ),
    ] = False,
) -> None:
    """Run the required replay arms and publish verified evidence."""
    try:
        execution = asyncio.run(
            run_replay(
                manifest,
                database,
                workspace,
                replace_owned=replace_owned,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_document(_execution_document(execution))
    if (
        not execution.valid
        or not execution.evidence.data.comparison_complete
        or not execution.profile_complete
    ):
        raise typer.Exit(1)


@replay_app.command("verify")
def replay_verify(
    report: Annotated[
        Path,
        typer.Option("--report", help="Canonical replay evidence report."),
    ],
) -> None:
    """Verify one canonical replay evidence report."""
    try:
        evidence = verify_replay_report(report)
    except Exception as error:
        _exit_with_error(error)
    _emit_document(_verification_document(evidence))
    if not evidence.data.valid or not evidence.data.comparison_complete:
        raise typer.Exit(1)


__all__ = ["replay_app"]
