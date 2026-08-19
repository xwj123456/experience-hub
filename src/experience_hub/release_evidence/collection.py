"""Collect stable release evidence without retaining command output."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeGuard

from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import CanonicalizationError
from experience_hub.release_evidence.contracts import (
    BenchmarkEvidenceV1,
    CheckEvidenceV1,
    CheckName,
    DemoEvidenceV1,
    ReleaseEvidenceDataV1,
    ReleaseEvidenceReportV1,
)
from experience_hub.release_evidence.errors import ReleaseEvidenceError
from experience_hub.release_evidence.source_tree import inspect_source_tree

COMMANDS: tuple[tuple[CheckName, tuple[str, ...]], ...] = (
    (CheckName.LOCK, ("uv", "lock", "--check")),
    (CheckName.RUFF, ("uv", "run", "ruff", "check", ".")),
    (CheckName.MYPY, ("uv", "run", "mypy", "src")),
    (CheckName.PYTEST, ("uv", "run", "pytest", "--no-cov", "-q")),
    (CheckName.DEMO, ("uv", "run", "experience-hub", "demo", "--reset")),
    (CheckName.BENCHMARK, ("uv", "run", "experience-hub", "benchmark")),
    (CheckName.BUILD, ("uv", "build")),
)

_PYTEST_SUMMARY = re.compile(
    rb"^(?P<count>[1-9][0-9]*) passed in "
    rb"[0-9]+(?:\.[0-9]+)?s(?: \([0-9]+:[0-9]{2}(?::[0-9]{2})?\))?$",
    re.MULTILINE,
)
_PYTEST_PROGRESS = re.compile(
    rb"(?:[.]+[ \t]+\[[ \t]*(?:100|[1-9][0-9]?)%\]\n)*"
)
_DEFAULT_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class CompletedCheck:
    """The byte-level result of a required release command."""

    name: CheckName
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


class CheckRunner(Protocol):
    """Run one fixed release check."""

    def run(
        self,
        *,
        name: CheckName,
        argv: tuple[str, ...],
    ) -> CompletedCheck:
        raise RuntimeError("CheckRunner is an interface")


@dataclass(frozen=True, slots=True)
class SubprocessCheckRunner:
    """Local, bounded subprocess implementation for fixed release checks."""

    repository: Path
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def run(
        self,
        *,
        name: CheckName,
        argv: tuple[str, ...],
    ) -> CompletedCheck:
        """Run one command without inheriting credential-bearing variables."""
        try:
            completed = subprocess.run(
                argv,
                check=False,
                cwd=self.repository,
                env=_public_environment(),
                capture_output=True,
                shell=False,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            return CompletedCheck(
                name=name,
                argv=argv,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        raise ReleaseEvidenceError(
            code="release_check_unavailable",
            message="release check cannot be run",
        )


def collect_release_evidence(
    repository: Path,
    *,
    verified_on: str,
    runner: CheckRunner,
) -> ReleaseEvidenceReportV1:
    """Run every required check and retain only their stable summaries."""
    source_tree = inspect_source_tree(repository)
    checks: list[CheckEvidenceV1] = []
    test_count: int | None = None
    demo: DemoEvidenceV1 | None = None
    benchmark: BenchmarkEvidenceV1 | None = None

    for name, argv in COMMANDS:
        completed = runner.run(name=name, argv=argv)
        _require_expected_completion(completed, name=name, argv=argv)
        if completed.returncode != 0:
            raise ReleaseEvidenceError(
                code="release_check_failed",
                message="release verification check failed",
            )
        checks.append(CheckEvidenceV1(name=name, passed=True))
        if name is CheckName.PYTEST:
            test_count = _parse_pytest_count(completed.stdout)
        elif name is CheckName.DEMO:
            demo = _parse_demo(completed.stdout)
        elif name is CheckName.BENCHMARK:
            benchmark = _parse_benchmark(completed.stdout)

    if test_count is None or demo is None or benchmark is None:
        raise ReleaseEvidenceError(
            code="invalid_check_output",
            message="release check output is invalid",
        )
    return ReleaseEvidenceReportV1(
        data=ReleaseEvidenceDataV1(
            schema_version=1,
            verified_commit=source_tree.verified_commit,
            source_tree_sha256=source_tree.source_tree_sha256,
            verified_on=verified_on,
            python_version="3.12",
            checks=tuple(checks),
            test_count=test_count,
            demo=demo,
            benchmark=benchmark,
        )
    )


def _public_environment() -> dict[str, str]:
    """Return only execution variables that cannot carry provider credentials."""
    return {
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONUTF8": "1",
        "UV_OFFLINE": "1",
    }


def _require_expected_completion(
    completed: CompletedCheck,
    *,
    name: CheckName,
    argv: tuple[str, ...],
) -> None:
    if completed.name != name or completed.argv != argv:
        raise ReleaseEvidenceError(
            code="invalid_check_output",
            message="release check output is invalid",
        )


def _parse_pytest_count(output: bytes) -> int:
    matches = tuple(_PYTEST_SUMMARY.finditer(output))
    if len(matches) != 1:
        raise ReleaseEvidenceError(
            code="invalid_check_output",
            message="release check output is invalid",
        )
    match = matches[0]
    if (
        _PYTEST_PROGRESS.fullmatch(output[: match.start()]) is None
        or output[match.end() :] != b"\n"
    ):
        raise ReleaseEvidenceError(
            code="invalid_check_output",
            message="release check output is invalid",
        )
    return int(match.group("count"))


def _parse_demo(output: bytes) -> DemoEvidenceV1:
    document = _parse_canonical_json_document(output)
    try:
        data = _required_mapping(document, "data")
        all_invariants_hold = data["all_invariants_hold"]
        stages = data["stages"]
        if all_invariants_hold is not True or not isinstance(stages, list):
            raise ValueError
        return DemoEvidenceV1(
            all_invariants_hold=all_invariants_hold,
            stage_count=len(stages),
        )
    except (KeyError, ValueError):
        pass
    raise ReleaseEvidenceError(
        code="invalid_check_output",
        message="release check output is invalid",
    )


def _parse_benchmark(output: bytes) -> BenchmarkEvidenceV1:
    document = _parse_canonical_json_document(output)
    try:
        data = _required_mapping(document, "data")
        passed = data["passed"]
        cases = data["cases"]
        gates = data["gates"]
        metrics = _required_mapping(data, "metrics")
        byte_identical_replay = metrics["byte_identical_replay"]
        pending_capsule_leakage_count = metrics["pending_capsule_leakage_count"]
        if (
            passed is not True
            or not isinstance(cases, list)
            or not isinstance(gates, list)
            or byte_identical_replay is not True
            or not _is_literal_integer_zero(pending_capsule_leakage_count)
        ):
            raise ValueError
        passed_gate_count = 0
        for gate in gates:
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise ValueError
            passed_gate_count += 1
        return BenchmarkEvidenceV1(
            passed=passed,
            case_count=len(cases),
            gate_count=len(gates),
            passed_gate_count=passed_gate_count,
            byte_identical_replay=byte_identical_replay,
            pending_capsule_leakage_count=pending_capsule_leakage_count,
        )
    except (KeyError, ValueError):
        pass
    raise ReleaseEvidenceError(
        code="invalid_check_output",
        message="release check output is invalid",
    )


def _parse_canonical_json_document(output: bytes) -> dict[str, object]:
    try:
        if not output.endswith(b"\n") or output.count(b"\n") != 1:
            raise ValueError
        body = output[:-1]
        decoded: object = json.loads(body)
        if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != body:
            raise ValueError
        return decoded
    except (
        CanonicalizationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        pass
    raise ReleaseEvidenceError(
        code="invalid_check_output",
        message="release check output is invalid",
    )


def _required_mapping(
    parent: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError
    return value


def _is_literal_integer_zero(value: object) -> TypeGuard[Literal[0]]:
    return type(value) is int and value == 0
