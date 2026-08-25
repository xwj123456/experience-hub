from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_PACK = _ROOT / "examples" / "experience-bench-s" / "pilot-manifest.json"
_CASES = _PACK.with_name("pilot-cases.jsonl")
_SOURCE_FIXTURE = _PACK.with_name("pilot-source.jsonl")
_MANIFEST_SHA256 = "4f403ad87b649c722c28ac2fefe537526efd385d00c16e3293da65e1c771b606"
_CASES_SHA256 = "827ae76eef7c9a17925c5fd9a358f7a8117ab36126a23a05f2fdb6cffb9acda2"
_SOURCE_FIXTURE_SHA256 = (
    "85c02a96972623132988e9112e9743911c89e0e169417dccb07675df9d5a78b8"
)
_ARMS = ("no_memory", "recent_notes", "sqlite_bm25", "experience_hub")
_NEGATIVE_TRANSFER_CASES = {
    "distractor-zh-foreign-owner",
    "distractor-en-pending-candidate",
    "distractor-mix-archived-note",
}
_GATES = (
    "comparison_complete",
    "complete_arms",
    "deterministic_replay",
    "safety",
    "overall_effectiveness",
    "stratum_effectiveness",
)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canonical_json(document: object) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_document(document: object) -> str:
    return _canonical_json(document) + "\n"


def _tree_inventory(root: Path) -> tuple[tuple[str, str, int, int, str], ...]:
    entries: list[tuple[str, str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        status = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(status.st_mode):
            entries.append((relative, "directory", status.st_mode & 0o777, 0, ""))
        elif stat.S_ISREG(status.st_mode):
            body = path.read_bytes()
            entries.append(
                (
                    relative,
                    "file",
                    status.st_mode & 0o777,
                    len(body),
                    _sha256(body),
                )
            )
        else:
            entries.append((relative, "other", status.st_mode & 0o777, 0, ""))
    return tuple(entries)


def _invoke(
    *arguments: str,
    record_root: Path,
    record_name: str,
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("experience-hub")
    assert executable.is_file()
    result = subprocess.run(
        (str(executable), *arguments),
        cwd=_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / f"{record_name}.stdout").write_text(
        result.stdout, encoding="utf-8"
    )
    (record_root / f"{record_name}.stderr").write_text(
        result.stderr, encoding="utf-8"
    )
    (record_root / f"{record_name}.exit-code").write_text(
        f"{result.returncode}\n", encoding="ascii"
    )
    return result


def _parse_canonical_output(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    document = json.loads(result.stdout)
    assert isinstance(document, dict)
    assert result.stdout == _canonical_document(document)
    return document


def _canonical_json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    body = path.read_bytes()
    document = json.loads(body)
    assert isinstance(document, dict)
    assert body == _canonical_json(document).encode()
    return body, document


def _assert_workspace(
    workspace: Path,
    *,
    case_ids: tuple[str, ...],
) -> tuple[
    bytes,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    set[tuple[int, int]],
]:
    artifacts = workspace / "artifacts"
    assert {path.name for path in artifacts.iterdir()} == {
        "benchmark-evidence.json",
        "benchmark-summary.json",
        "profile.json",
    }
    evidence_body, evidence = _canonical_json_file(
        artifacts / "benchmark-evidence.json"
    )
    summary_body, summary = _canonical_json_file(
        artifacts / "benchmark-summary.json"
    )
    profile_body, profile = _canonical_json_file(artifacts / "profile.json")

    expected_clones = {
        Path("arms") / pass_name / case_id / f"{arm}.sqlite3"
        for pass_name in ("pass-a", "pass-b")
        for case_id in case_ids
        for arm in _ARMS
    }
    clones = sorted(workspace.glob("arms/*/*/*.sqlite3"))
    assert {path.relative_to(workspace) for path in clones} == expected_clones
    assert len(clones) == 240
    assert not list(workspace.rglob("*.sqlite3-wal"))
    assert not list(workspace.rglob("*.sqlite3-shm"))
    assert not list(workspace.rglob("*.sqlite3-journal"))

    source = workspace / "snapshot" / "source.sqlite3"
    validation = workspace / "validation" / "source.sqlite3"
    assert source.is_file() and not source.is_symlink()
    assert validation.is_file() and not validation.is_symlink()
    source_status = source.stat()
    source_identity = (source_status.st_dev, source_status.st_ino)
    clone_identities: set[tuple[int, int]] = set()
    for clone in clones:
        assert clone.is_file() and not clone.is_symlink()
        clone_status = clone.stat()
        assert stat.S_ISREG(clone_status.st_mode)
        assert clone_status.st_nlink == 1
        clone_identities.add((clone_status.st_dev, clone_status.st_ino))
    assert len(clone_identities) == 240
    assert source_identity not in clone_identities

    evidence_data = evidence["data"]
    payload = evidence_data["pass_payload"]
    resolved = payload["resolved_manifest"]
    summary_data = summary["data"]
    profile_data = profile["data"]
    assert _sha256(source.read_bytes()) == resolved["snapshot_sha256"]
    assert _sha256(source.read_bytes()) == summary_data["snapshot_sha256"]
    assert _sha256(evidence_body) == summary_data["evidence_sha256"]
    assert b'"profile"' not in evidence_body
    assert b'"wall_duration_ns"' not in evidence_body
    assert b'"wall_duration_ns"' not in summary_body
    assert set(profile_data) == {
        "clone_count",
        "database_bytes",
        "fts5_available",
        "pack_id",
        "profile_complete",
        "schema_version",
        "wall_duration_ns",
    }
    assert profile_data["profile_complete"] is True
    assert profile_data["clone_count"] == 240
    assert profile_data["database_bytes"] > 0
    assert profile_data["fts5_available"] is True
    assert profile_data["wall_duration_ns"] >= 0
    assert profile_body != evidence_body

    return evidence_body, evidence, summary, profile, clone_identities


def test_frozen_pilot_passes_through_the_external_cli_twice(tmp_path: Path) -> None:
    record_root = tmp_path / "cli-output"
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    pack_root = _PACK.parent
    pack_before = _tree_inventory(pack_root)
    cases = tuple(
        json.loads(line)["case_id"] for line in _CASES.read_bytes().splitlines()
    )
    assert len(cases) == 30
    assert len(set(cases)) == 30
    assert _sha256(_PACK.read_bytes()) == _MANIFEST_SHA256
    assert _sha256(_CASES.read_bytes()) == _CASES_SHA256
    assert _sha256(_SOURCE_FIXTURE.read_bytes()) == _SOURCE_FIXTURE_SHA256

    inspect_result = _invoke(
        "replay",
        "benchmark",
        "inspect",
        "--pack",
        str(_PACK),
        record_root=record_root,
        record_name="inspect",
    )
    inspect_document = _parse_canonical_output(inspect_result)
    assert inspect_result.returncode == 0
    assert inspect_document == {
        "data": {
            "arm_count": 4,
            "case_count": 30,
            "cases_sha256": _CASES_SHA256,
            "fts5_available": True,
            "manifest_sha256": _MANIFEST_SHA256,
            "pack_id": "experiencebench-s-pilot",
            "source_fixture_sha256": _SOURCE_FIXTURE_SHA256,
        }
    }
    assert not first_workspace.exists()
    assert not second_workspace.exists()
    assert _tree_inventory(pack_root) == pack_before

    first_result = _invoke(
        "replay",
        "benchmark",
        "run",
        "--pack",
        str(_PACK),
        "--workspace",
        str(first_workspace),
        record_root=record_root,
        record_name="first-run",
    )
    second_result = _invoke(
        "replay",
        "benchmark",
        "run",
        "--pack",
        str(_PACK),
        "--workspace",
        str(second_workspace),
        record_root=record_root,
        record_name="second-run",
    )
    first_document = _parse_canonical_output(first_result)
    second_document = _parse_canonical_output(second_result)

    first_report = first_workspace / "artifacts" / "benchmark-evidence.json"
    second_report = second_workspace / "artifacts" / "benchmark-evidence.json"
    first_verify_result = _invoke(
        "replay",
        "benchmark",
        "verify",
        "--report",
        str(first_report),
        record_root=record_root,
        record_name="first-verify",
    )
    second_verify_result = _invoke(
        "replay",
        "benchmark",
        "verify",
        "--report",
        str(second_report),
        record_root=record_root,
        record_name="second-verify",
    )
    first_verify = _parse_canonical_output(first_verify_result)
    second_verify = _parse_canonical_output(second_verify_result)

    first_evidence_body, first_evidence, first_summary, _, first_identities = (
        _assert_workspace(first_workspace, case_ids=cases)
    )
    second_evidence_body, second_evidence, second_summary, _, second_identities = (
        _assert_workspace(second_workspace, case_ids=cases)
    )
    assert first_identities.isdisjoint(second_identities)
    assert first_evidence_body == second_evidence_body
    assert first_evidence == second_evidence
    assert first_summary == second_summary
    assert _tree_inventory(pack_root) == pack_before
    assert _sha256(_PACK.read_bytes()) == _MANIFEST_SHA256
    assert _sha256(_CASES.read_bytes()) == _CASES_SHA256
    assert _sha256(_SOURCE_FIXTURE.read_bytes()) == _SOURCE_FIXTURE_SHA256

    first_data = first_document["data"]
    second_data = second_document["data"]
    assert first_data == second_data
    assert set(first_data) == {
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
    assert first_data["case_count"] == 30
    assert first_data["arm_count"] == 4
    assert first_data["comparison_complete"] is True
    assert first_data["deterministic_replay_match"] is True
    assert first_data["evidence_valid"] is True
    assert first_data["profile_complete"] is True
    assert first_data["manifest_sha256"] == _MANIFEST_SHA256
    assert first_data["source_fixture_sha256"] == _SOURCE_FIXTURE_SHA256
    assert first_data["evidence_sha256"] == _sha256(first_evidence_body)

    evidence_data = first_evidence["data"]
    payload = evidence_data["pass_payload"]
    resolved = payload["resolved_manifest"]
    assert resolved["manifest_sha256"] == _MANIFEST_SHA256
    assert resolved["cases_sha256"] == _CASES_SHA256
    assert resolved["source_fixture_sha256"] == _SOURCE_FIXTURE_SHA256
    assert [arm["arm_id"] for arm in resolved["arms"]] == list(_ARMS)
    assert payload["comparison_complete"] is True
    assert evidence_data["deterministic_replay_match"] is True
    assert evidence_data["valid"] is True

    case_evidence = payload["cases"]
    assert [item["case_id"] for item in case_evidence] == list(cases)
    assert all(item["status"] == "complete" for item in case_evidence)
    assert all(
        [arm["arm_id"] for arm in item["arms"]] == list(_ARMS)
        for item in case_evidence
    )
    assert all(
        arm["status"] == "complete"
        and arm["observation"] is not None
        and arm["oracle"] is not None
        for item in case_evidence
        for arm in item["arms"]
    )
    assert all(item["delta_utility_micros"] is not None for item in case_evidence)
    negative_deltas = {
        item["case_id"]: item["delta_utility_micros"]
        for item in case_evidence
        if item["delta_utility_micros"] < 0
    }
    assert negative_deltas
    assert {item["case_id"] for item in case_evidence} >= _NEGATIVE_TRANSFER_CASES

    safety = payload["safety"]
    assert safety == {
        "clone_isolation_verified": True,
        "cross_arm_contamination_count": 0,
        "owner_leak_count": 0,
        "quarantine_leak_count": 0,
        "schema_version": 1,
        "source_mutation_count": 0,
        "source_unchanged": True,
    }
    aggregate = payload["aggregate"]
    assert aggregate is not None
    assert aggregate["overall"]["case_count"] == 30
    assert len(aggregate["strata"]) == 5
    assert all(item["case_count"] == 6 for item in aggregate["strata"])

    gates = evidence_data["gates"]
    assert tuple(item["gate_id"] for item in gates) == _GATES
    assert all(item["passed"] for item in gates[:4])
    assert evidence_data["expansion_gate_passed"] == all(
        item["passed"] for item in gates
    )
    assert first_data["expansion_gate_passed"] == evidence_data[
        "expansion_gate_passed"
    ]

    expected_verification = {
        "data": {
            key: first_data[key]
            for key in (
                "arm_count",
                "case_count",
                "comparison_complete",
                "deterministic_replay_match",
                "evidence_sha256",
                "evidence_valid",
                "expansion_gate_passed",
                "manifest_sha256",
                "snapshot_sha256",
                "source_fixture_sha256",
            )
        }
    }
    assert first_verify == expected_verification
    assert second_verify == expected_verification
    expected_exit = 0 if evidence_data["expansion_gate_passed"] else 1
    assert first_result.returncode == expected_exit
    assert second_result.returncode == expected_exit
    assert first_verify_result.returncode == expected_exit
    assert second_verify_result.returncode == expected_exit

    assert evidence_data["expansion_gate_passed"] is True
