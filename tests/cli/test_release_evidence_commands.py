"""Public command tests for release evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from experience_hub.cli import release_commands
from experience_hub.cli.app import app
from experience_hub.release_evidence.contracts import CheckName, ReleaseEvidenceReportV1
from experience_hub.release_evidence.errors import ReleaseEvidenceError

RUNNER = CliRunner()


@pytest.fixture
def valid_evidence() -> ReleaseEvidenceReportV1:
    return ReleaseEvidenceReportV1.model_validate(
        {
            "data": {
                "schema_version": 1,
                "verified_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "verified_on": "2026-08-03",
                "python_version": "3.12",
                "checks": [
                    {"name": name, "passed": True} for name in CheckName
                ],
                "test_count": 2670,
                "demo": {"all_invariants_hold": True, "stage_count": 11},
                "benchmark": {
                    "passed": True,
                    "case_count": 15,
                    "gate_count": 11,
                    "passed_gate_count": 11,
                    "byte_identical_replay": True,
                    "pending_capsule_leakage_count": 0,
                },
            }
        },
        strict=True,
    )


def test_release_collect_emits_public_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_evidence: ReleaseEvidenceReportV1,
) -> None:
    monkeypatch.setattr(release_commands.config, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        release_commands,
        "collect_and_store_release_evidence",
        lambda *_args, **_kwargs: valid_evidence,
    )

    result = RUNNER.invoke(
        app,
        ["release", "collect", "--verified-on", "2026-08-03"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["data"] == _summary_data()


def test_release_verify_emits_public_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_evidence: ReleaseEvidenceReportV1,
) -> None:
    monkeypatch.setattr(release_commands.config, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        release_commands,
        "verify_release_evidence",
        lambda *_: valid_evidence,
    )

    result = RUNNER.invoke(app, ["release", "verify"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["data"] == _summary_data()


@pytest.mark.parametrize(
    ("command", "error", "expected_code"),
    (
        (
            ("release", "collect", "--verified-on", "2026-08-03"),
            ReleaseEvidenceError(
                code="release_check_failed",
                message="private command output /private/check.log",
            ),
            "release_check_failed",
        ),
        (
            ("release", "verify"),
            ReleaseEvidenceError(
                code="invalid_release_evidence",
                message="private document /private/evidence.json",
            ),
            "invalid_release_evidence",
        ),
        (
            ("release", "verify"),
            ReleaseEvidenceError(
                code="stale_release_evidence",
                message="private tree /private/repository",
            ),
            "stale_release_evidence",
        ),
    ),
)
def test_release_commands_emit_stable_errors_without_private_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: tuple[str, ...],
    error: ReleaseEvidenceError,
    expected_code: str,
) -> None:
    monkeypatch.setattr(release_commands.config, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        release_commands,
        "collect_and_store_release_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        release_commands,
        "verify_release_evidence",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    result = RUNNER.invoke(app, list(command))

    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert document["error"]["code"] == expected_code
    assert "/private/" not in result.stdout
    assert "private command output" not in result.stdout


@pytest.mark.parametrize(
    "command",
    (
        ("release", "collect", "--verified-on", "2026-08-03"),
        ("release", "verify"),
    ),
)
def test_release_commands_keep_repository_discovery_failures_private(
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
) -> None:
    def fail_discovery() -> Path:
        raise RuntimeError("private repository /private/release-owner")

    monkeypatch.setattr(release_commands.config, "repository_root", fail_discovery)

    result = RUNNER.invoke(app, list(command))

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "invalid_release_evidence"
    assert "/private/" not in result.stdout
    assert "private repository" not in result.stdout


def _summary_data() -> dict[str, object]:
    return {
        "benchmark_cases": 15,
        "benchmark_gates": 11,
        "byte_identical_replay": True,
        "source_tree_sha256": "b" * 64,
        "test_count": 2670,
        "verified": True,
        "verified_on": "2026-08-03",
    }
