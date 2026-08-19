from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner, Result

import experience_hub.config as config
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli.app import app

RUNNER = CliRunner()
ROOT = Path(__file__).parents[2]
SMOKE_MANIFEST = ROOT / "examples" / "replay" / "smoke-manifest.json"
RAW_UUID_SENTINEL = "00000000-0000-0000-0000-000000000006"
RAW_UUID = re.compile(
    rb"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    rb"[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT = re.compile(
    rb"(?:api[_-]?key|password|token|secret)\s*[:=]",
    re.IGNORECASE,
)


def _invoke(*arguments: str) -> tuple[Result, bytes, dict[str, Any]]:
    result = RUNNER.invoke(app, list(arguments))
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert result.stdout.count("\n") == 1
    body = result.stdout.removesuffix("\n").encode("utf-8")
    document = cast(dict[str, Any], json.loads(body))
    assert canonical_json_bytes(document) == body
    return result, body, document


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projection_hashes(database: Path) -> dict[str, str]:
    database_uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        names = tuple(
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM projection_versions ORDER BY name"
            )
        )
        hashes: dict[str, str] = {}
        for name in names:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name)
            columns = tuple(
                str(row["name"])
                for row in connection.execute(f'PRAGMA table_info("{name}")')
            )
            rows = [
                {
                    column: (
                        value.hex() if isinstance(value, bytes) else value
                    )
                    for column, value in zip(columns, row, strict=True)
                }
                for row in connection.execute(
                    f'SELECT * FROM "{name}" ORDER BY rowid'
                )
            ]
            body = canonical_json_bytes({"projection": name, "rows": rows})
            hashes[name] = hashlib.sha256(body).hexdigest()
    assert hashes
    return hashes


def _assert_private_values_absent(
    bodies: tuple[bytes, ...],
    *,
    private_paths: tuple[Path, ...],
    secret_sentinel: str,
) -> None:
    for body in bodies:
        for path in private_paths:
            assert str(path).encode() not in body
        assert secret_sentinel.encode() not in body
        assert SECRET_ASSIGNMENT.search(body) is None
        assert RAW_UUID.search(body) is None


def test_privacy_scanner_rejects_raw_project_uuid_sentinel() -> None:
    with pytest.raises(AssertionError):
        _assert_private_values_absent(
            (f'{{"source_id":"{RAW_UUID_SENTINEL}"}}'.encode(),),
            private_paths=(),
            secret_sentinel="absent-provider-credential",
        )


def test_five_minute_replay_flow_is_isolated_private_and_repeatable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_sentinel = "task9-private-provider-secret"
    monkeypatch.setenv(
        "EXPERIENCE_HUB_OPENAI_COMPATIBLE_API_KEY",
        secret_sentinel,
    )
    monkeypatch.setattr(config, "repository_root", lambda: tmp_path)

    _, _, demo = _invoke("demo", "--reset")
    assert cast(dict[str, Any], demo["data"])["all_invariants_hold"] is True
    database = tmp_path / ".data" / "demo.db"
    workspace = tmp_path / ".data" / "replay-lab"
    assert database.is_file()

    source_sha256_before = _file_sha256(database)
    projections_before = _projection_hashes(database)

    _, inspect_body, inspected = _invoke(
        "replay",
        "inspect",
        "--manifest",
        str(SMOKE_MANIFEST),
        "--database",
        str(database),
    )
    inspect_data = cast(dict[str, Any], inspected["data"])
    assert inspect_data["case_count"] == 2
    assert inspect_data["arm_count"] == 2
    assert inspect_data["snapshot_sha256"] == source_sha256_before

    _, run_body, ran = _invoke(
        "replay",
        "run",
        "--manifest",
        str(SMOKE_MANIFEST),
        "--database",
        str(database),
        "--workspace",
        str(workspace),
    )
    run_data = cast(dict[str, Any], ran["data"])
    assert run_data["snapshot_sha256"] == source_sha256_before
    assert run_data["comparison_complete"] is True
    assert run_data["deterministic_replay_match"] is True
    assert run_data["profile_complete"] is True
    assert run_data["valid"] is True

    evidence_path = workspace / "artifacts" / "evidence.json"
    profile_path = workspace / "artifacts" / "profile.json"
    first_evidence = evidence_path.read_bytes()
    first_profile = profile_path.read_bytes()
    evidence = cast(dict[str, Any], json.loads(first_evidence))
    evidence_data = cast(dict[str, Any], evidence["data"])
    assert evidence_data["clone_isolation_verified"] is True
    assert evidence_data["source_unchanged"] is True
    clone_paths = tuple(
        sorted((workspace / "arms").glob("*/*/*.sqlite3"))
    )
    assert len(clone_paths) == 8
    assert {_file_sha256(path) for path in clone_paths} == {
        source_sha256_before
    }

    _, verify_body, verified = _invoke(
        "replay",
        "verify",
        "--report",
        str(evidence_path),
    )
    verify_data = cast(dict[str, Any], verified["data"])
    assert verify_data["snapshot_sha256"] == source_sha256_before
    assert verify_data["deterministic_replay_match"] is True
    assert verify_data["valid"] is True

    _, repeated_body, repeated = _invoke(
        "replay",
        "run",
        "--manifest",
        str(SMOKE_MANIFEST),
        "--database",
        str(database),
        "--workspace",
        str(workspace),
        "--replace-owned",
    )
    repeated_data = cast(dict[str, Any], repeated["data"])
    assert repeated_data == run_data
    assert evidence_path.read_bytes() == first_evidence

    assert _file_sha256(database) == source_sha256_before
    assert _projection_hashes(database) == projections_before
    _assert_private_values_absent(
        (
            inspect_body,
            run_body,
            verify_body,
            repeated_body,
            first_evidence,
            first_profile,
        ),
        private_paths=(ROOT, tmp_path, database, workspace),
        secret_sentinel=secret_sentinel,
    )
