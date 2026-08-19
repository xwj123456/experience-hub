"""Integration tests for stable release-evidence collection."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from experience_hub.canonical import canonical_json_bytes
from experience_hub.release_evidence.collection import (
    COMMANDS,
    CompletedCheck,
    SubprocessCheckRunner,
    collect_release_evidence,
)
from experience_hub.release_evidence.contracts import CheckName
from experience_hub.release_evidence.errors import ReleaseEvidenceError


@dataclass
class FakeRunner:
    """A deterministic injected check runner for collector behavior tests."""

    responses: dict[CheckName, CompletedCheck]
    calls: list[tuple[CheckName, tuple[str, ...]]] = field(default_factory=list)

    def run(
        self,
        *,
        name: CheckName,
        argv: tuple[str, ...],
    ) -> CompletedCheck:
        self.calls.append((name, argv))
        return self.responses[name]


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create the smallest committed source tree accepted by Task 2."""
    repository = tmp_path / "repository"
    repository.mkdir()
    environment = {"LC_ALL": "C", "PATH": os.environ.get("PATH", os.defpath)}
    for arguments in (
        ("init", "--quiet"),
        ("config", "user.name", "Release Evidence Test"),
        ("config", "user.email", "release-evidence@example.test"),
    ):
        subprocess.run(
            ("git", *arguments),
            check=True,
            cwd=repository,
            capture_output=True,
            env=environment,
        )
    (repository / "source.py").write_text("answer = 42\n", encoding="utf-8")
    for arguments in (("add", "."), ("commit", "--quiet", "-m", "test repository")):
        subprocess.run(
            ("git", *arguments),
            check=True,
            cwd=repository,
            capture_output=True,
            env=environment,
        )
    return repository


def _demo_stdout() -> bytes:
    return canonical_json_bytes(
        {
            "data": {
                "all_invariants_hold": True,
                "database_path": "/private/demo.sqlite3",
                "stages": [
                    {
                        "result": {
                            "case_body": "private-demo-stage-body",
                            "id": "00000000-0000-0000-0000-000000000001",
                        },
                        "step": number,
                    }
                    for number in range(1, 12)
                ],
            }
        }
    ) + b"\n"


def _benchmark_document() -> dict[str, object]:
    return {
        "data": {
            "cases": [
                {
                    "case_body": "private-benchmark-case-body",
                    "case_id": f"case-{number}",
                }
                for number in range(1, 16)
            ],
            "gates": [
                {"name": f"gate-{number}", "passed": True}
                for number in range(1, 12)
            ],
            "metrics": {
                "byte_identical_replay": True,
                "pending_capsule_leakage_count": 0,
            },
            "passed": True,
        }
    }


def _benchmark_stdout(document: dict[str, object] | None = None) -> bytes:
    return canonical_json_bytes(document or _benchmark_document()) + b"\n"


def _successful_responses() -> dict[CheckName, CompletedCheck]:
    responses: dict[CheckName, CompletedCheck] = {}
    for name, argv in COMMANDS:
        stdout = b""
        if name is CheckName.PYTEST:
            stdout = b"...................... [100%]\n2670 passed in 1.23s\n"
        elif name is CheckName.DEMO:
            stdout = _demo_stdout()
        elif name is CheckName.BENCHMARK:
            stdout = _benchmark_stdout()
        responses[name] = CompletedCheck(
            name=name,
            argv=argv,
            returncode=0,
            stdout=stdout,
            stderr=b"private command stderr /private/check.log",
        )
    return responses


@pytest.fixture
def successful_runner() -> FakeRunner:
    return FakeRunner(_successful_responses())


def test_collection_retains_stable_release_summary(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    evidence = collect_release_evidence(
        repository,
        verified_on="2026-08-03",
        runner=successful_runner,
    )

    encoded = canonical_json_bytes(evidence)
    assert evidence.data.test_count == 2670
    assert evidence.data.demo.stage_count == 11
    assert evidence.data.benchmark.case_count == 15
    assert evidence.data.benchmark.gate_count == 11
    assert evidence.data.benchmark.passed_gate_count == 11
    assert successful_runner.calls == list(COMMANDS)
    for forbidden in (
        b"1.23s",
        b"private command stderr",
        b"/private/",
        b"00000000-0000-0000-0000-000000000001",
        b"private-benchmark-case-body",
        b"private-demo-stage-body",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize("name", tuple(CheckName))
def test_collection_rejects_every_nonzero_command(
    repository: Path,
    successful_runner: FakeRunner,
    name: CheckName,
) -> None:
    completed = successful_runner.responses[name]
    successful_runner.responses[name] = CompletedCheck(
        name=completed.name,
        argv=completed.argv,
        returncode=1,
        stdout=completed.stdout,
        stderr=b"private command stderr /private/check.log",
    )

    with pytest.raises(ReleaseEvidenceError) as raised:
        collect_release_evidence(
            repository,
            verified_on="2026-08-03",
            runner=successful_runner,
        )

    assert raised.value.code == "release_check_failed"
    assert "private command stderr" not in raised.value.message
    assert "/private/check.log" not in raised.value.message


def test_collection_rejects_malformed_utf8_demo_output(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    _replace_stdout(successful_runner, CheckName.DEMO, b"\xff\n")

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_multiple_json_lines(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    _replace_stdout(successful_runner, CheckName.DEMO, _demo_stdout() + b"{}\n")

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_noncanonical_json(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    noncanonical = _demo_stdout().replace(b":", b": ", 1)
    _replace_stdout(successful_runner, CheckName.DEMO, noncanonical)

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_nonfinite_json_values(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    _replace_stdout(
        successful_runner,
        CheckName.DEMO,
        b'{"data":{"all_invariants_hold":true,"stages":[NaN]}}\n',
    )

    _assert_invalid_output(repository, successful_runner)


@pytest.mark.parametrize(
    "missing_path",
    (
        ("data", "cases"),
        ("data", "gates"),
        ("data", "metrics"),
        ("data", "passed"),
        ("data", "metrics", "byte_identical_replay"),
        ("data", "metrics", "pending_capsule_leakage_count"),
    ),
)
def test_collection_rejects_missing_benchmark_keys(
    repository: Path,
    successful_runner: FakeRunner,
    missing_path: tuple[str, ...],
) -> None:
    document = _benchmark_document()
    parent: dict[str, object] = document
    for key in missing_path[:-1]:
        nested = parent[key]
        assert isinstance(nested, dict)
        parent = nested
    del parent[missing_path[-1]]
    _replace_stdout(successful_runner, CheckName.BENCHMARK, _benchmark_stdout(document))

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_failed_benchmark_gate(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    document = _benchmark_document()
    data = document["data"]
    assert isinstance(data, dict)
    gates = data["gates"]
    assert isinstance(gates, list)
    gate = gates[0]
    assert isinstance(gate, dict)
    gate["passed"] = False
    _replace_stdout(successful_runner, CheckName.BENCHMARK, _benchmark_stdout(document))

    _assert_invalid_output(repository, successful_runner)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("data", "passed"), 1),
        (("data", "gates", 0, "passed"), 1),
        (("data", "metrics", "byte_identical_replay"), 1),
        (("data", "metrics", "pending_capsule_leakage_count"), False),
    ),
)
def test_collection_rejects_boolean_integer_confusion_in_benchmark_metrics(
    repository: Path,
    successful_runner: FakeRunner,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    document = _benchmark_document()
    target: object = document
    for component in path[:-1]:
        assert isinstance(target, (dict, list))
        target = target[component]  # type: ignore[index]
    assert isinstance(target, (dict, list))
    target[path[-1]] = value  # type: ignore[index]
    _replace_stdout(successful_runner, CheckName.BENCHMARK, _benchmark_stdout(document))

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_integer_one_for_demo_literal_true(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    document = {
        "data": {
            "all_invariants_hold": 1,
            "stages": [{"step": 1}],
        }
    }
    _replace_stdout(
        successful_runner,
        CheckName.DEMO,
        canonical_json_bytes(document) + b"\n",
    )

    _assert_invalid_output(repository, successful_runner)


@pytest.mark.parametrize(
    "stdout",
    (
        b"2670.5 passed in 1.23s\n",
        b"2670 passed, 1 skipped in 1.23s\n",
        b"2670 passed in 1.23s\n2670 passed in 1.23s\n",
        b"...................... [\n99%]\n2670 passed in 1.23s\n",
    ),
)
def test_collection_rejects_unstable_pytest_summary(
    repository: Path,
    successful_runner: FakeRunner,
    stdout: bytes,
) -> None:
    _replace_stdout(successful_runner, CheckName.PYTEST, stdout)

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_the_standard_pytest_warning_summary(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    _replace_stdout(
        successful_runner,
        CheckName.PYTEST,
        b"......................                  [100%]\n"
        b"=============================== warnings summary "
        b"===============================\n"
        b"tests/example_test.py::test_example\n\n"
        b"  DeprecationWarning: standard pytest warning\n\n"
        b"-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
        b"2670 passed, 1 warning in 1.23s (0:01:01)\n",
    )

    _assert_invalid_output(repository, successful_runner)


def test_collection_rejects_unrecognized_pytest_warning_preamble(
    repository: Path,
    successful_runner: FakeRunner,
) -> None:
    _replace_stdout(
        successful_runner,
        CheckName.PYTEST,
        b"...................... [100%]\n"
        b"unrecognized output before the test summary\n"
        b"2670 passed in 1.23s\n",
    )

    _assert_invalid_output(repository, successful_runner)


def test_subprocess_runner_keeps_the_required_suite_timeout_bounded() -> None:
    runner = SubprocessCheckRunner(Path("/repository"))

    assert runner.timeout_seconds == 900.0


def test_subprocess_runner_maps_timeout_to_a_stable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SubprocessCheckRunner(Path("/repository"))

    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(("uv", "lock", "--check"), 900.0)

    monkeypatch.setattr(subprocess, "run", timed_out)

    with pytest.raises(ReleaseEvidenceError) as raised:
        runner.run(
            name=CheckName.LOCK,
            argv=("uv", "lock", "--check"),
        )

    assert raised.value.code == "release_check_unavailable"
    assert raised.value.message == "release check cannot be run"


def _replace_stdout(runner: FakeRunner, name: CheckName, stdout: bytes) -> None:
    completed = runner.responses[name]
    runner.responses[name] = CompletedCheck(
        name=completed.name,
        argv=completed.argv,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=completed.stderr,
    )


def _assert_invalid_output(repository: Path, runner: FakeRunner) -> None:
    with pytest.raises(ReleaseEvidenceError) as raised:
        collect_release_evidence(
            repository,
            verified_on="2026-08-03",
            runner=runner,
        )

    assert raised.value.code == "invalid_check_output"
    assert "/private/" not in raised.value.message
