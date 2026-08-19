"""Public CLI adapter for bounded release evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import typer

from experience_hub import config
from experience_hub.release_evidence.collection import SubprocessCheckRunner
from experience_hub.release_evidence.contracts import ReleaseEvidenceReportV1
from experience_hub.release_evidence.errors import ReleaseEvidenceError
from experience_hub.release_evidence.service import (
    collect_and_store_release_evidence,
    verify_release_evidence,
)

release_app = typer.Typer(
    help="Collect and verify public release evidence.",
    no_args_is_help=True,
    add_completion=False,
)


@release_app.command("collect")
def release_collect(
    verified_on: str = typer.Option(..., "--verified-on", help="ISO calendar date."),
) -> None:
    """Collect and atomically store release evidence for this repository."""
    try:
        repository = config.repository_root()
        evidence = collect_and_store_release_evidence(
            repository,
            repository / "docs" / "evidence" / "release-evidence.json",
            verified_on=verified_on,
            runner=SubprocessCheckRunner(repository),
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_document(_summary(evidence))


@release_app.command("verify")
def release_verify() -> None:
    """Verify the repository's generated release evidence without mutation."""
    try:
        repository = config.repository_root()
        evidence = verify_release_evidence(
            repository,
            repository / "docs" / "evidence" / "release-evidence.json",
        )
    except Exception as error:
        _exit_with_error(error)
    _emit_document(_summary(evidence))


def _summary(evidence: ReleaseEvidenceReportV1) -> dict[str, object]:
    data = evidence.data
    return {
        "data": {
            "benchmark_cases": data.benchmark.case_count,
            "benchmark_gates": data.benchmark.gate_count,
            "byte_identical_replay": data.benchmark.byte_identical_replay,
            "source_tree_sha256": data.source_tree_sha256,
            "test_count": data.test_count,
            "verified": True,
            "verified_on": data.verified_on,
        }
    }


def _emit_document(document: Mapping[str, Any]) -> None:
    from experience_hub.cli.app import _emit_document as emit_document

    emit_document(document)


def _exit_with_error(error: BaseException) -> NoReturn:
    _emit_document(_error_document(error))
    raise typer.Exit(1) from None


def _error_document(error: BaseException) -> dict[str, object]:
    if isinstance(error, ReleaseEvidenceError):
        if error.code == "release_check_failed":
            return _stable_error(
                code="release_check_failed",
                message="release verification check failed",
            )
        if error.code == "stale_release_evidence":
            return _stable_error(
                code="stale_release_evidence",
                message="release evidence is stale",
            )
    return _stable_error(
        code="invalid_release_evidence",
        message="release evidence is invalid",
    )


def _stable_error(*, code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "details": {}, "message": message}}


__all__ = ["release_app"]
