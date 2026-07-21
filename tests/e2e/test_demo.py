from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

import experience_hub.config as config
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli import app

RUNNER = CliRunner()
GOLDEN_PATH = Path(__file__).parents[1] / "golden" / "demo_output.json"
NORMALIZED_DATABASE_PATH = "<repository>/.data/demo.db"
EXPECTED_STAGE_NAMES = (
    "create Alice and Bob",
    "Alice records operational experiences",
    "Bob subscribes to Alice's topic",
    "Alice publishes a capsule",
    "Bob receives and explicitly adopts it",
    "advance time until an unrelated experience is cold",
    "show an ordinary query returning only its blurred projection",
    "issue a strong contextual cue that expands and reactivates it",
    "generate deterministic inspiration from frozen evidence",
    "prove generated ideas are absent from experience recall",
    "explicitly adopt one idea and retrieve the resulting hypothesis",
)


def _normalized_report(result: Any, *, expected_database: Path) -> bytes:
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    encoded = result.stdout[:-1].encode("utf-8")
    decoded = cast(dict[str, Any], json.loads(encoded))
    assert canonical_json_bytes(decoded) == encoded

    data = cast(dict[str, Any], decoded["data"])
    assert data["database_path"] == str(expected_database)
    stages = cast(list[dict[str, Any]], data["stages"])
    assert [stage["step"] for stage in stages] == list(range(1, 12))
    assert tuple(stage["name"] for stage in stages) == EXPECTED_STAGE_NAMES
    invariants = cast(dict[str, Any], data["invariants"])
    assert invariants
    assert all(value is True for value in invariants.values())
    assert data["all_invariants_hold"] is True
    data["database_path"] = NORMALIZED_DATABASE_PATH
    return canonical_json_bytes(decoded)


def test_demo_resets_cleanly_and_matches_the_golden_report(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "repository_root", lambda: tmp_path)
    expected_database = tmp_path / ".data" / "demo.db"

    first = RUNNER.invoke(app, ["demo", "--reset"])
    assert expected_database.is_file()
    first_report = _normalized_report(
        first,
        expected_database=expected_database,
    )
    verification = RUNNER.invoke(
        app,
        [
            "projections",
            "rebuild",
            "--verify",
            "--database",
            str(expected_database),
        ],
    )
    assert verification.exit_code == 0, verification.output
    verification_data = cast(
        dict[str, Any],
        cast(dict[str, Any], json.loads(verification.stdout))["data"],
    )
    assert verification_data["matches"] is True
    assert verification_data["differences"] == []
    with closing(sqlite3.connect(expected_database)) as connection:
        connection.execute("CREATE TABLE demo_reset_sentinel (value INTEGER)")
        connection.commit()
    sidecars = tuple(
        Path(f"{expected_database}{suffix}") for suffix in ("-wal", "-shm", "-journal")
    )
    for sidecar in sidecars:
        sidecar.write_bytes(b"stale-demo-sidecar")

    second = RUNNER.invoke(app, ["demo", "--reset"])
    second_report = _normalized_report(
        second,
        expected_database=expected_database,
    )
    with closing(sqlite3.connect(expected_database)) as connection:
        sentinel = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'demo_reset_sentinel'"
        ).fetchone()

    golden = GOLDEN_PATH.read_bytes().rstrip(b"\n")
    assert canonical_json_bytes(json.loads(golden)) == golden
    assert sentinel is None
    assert all(not sidecar.exists() for sidecar in sidecars)
    assert first_report == second_report == golden

    retained = RUNNER.invoke(app, ["demo"])
    assert retained.exit_code == 1
    assert retained.stdout.endswith("\n")
    assert retained.stdout.count("\n") == 1
    retained_encoded = retained.stdout[:-1].encode("utf-8")
    retained_error = cast(dict[str, Any], json.loads(retained_encoded))["error"]
    assert canonical_json_bytes(json.loads(retained_encoded)) == retained_encoded
    assert retained_error == {
        "code": "demo_database_exists",
        "details": {
            "database_path": str(expected_database),
            "retry_with_reset": True,
        },
        "message": "Demo database already exists; rerun with --reset",
    }
