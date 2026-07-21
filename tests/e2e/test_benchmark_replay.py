from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

import experience_hub.config as config
from experience_hub.canonical import canonical_json_bytes
from experience_hub.cli import app

RUNNER = CliRunner()
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)


def test_benchmark_replays_from_exact_snapshot_and_passes_every_gate(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "repository_root", lambda: tmp_path)
    source_root = Path(__file__).parents[2]
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    seed_document = json.loads(
        (source_root / "benchmarks" / "seed.json").read_bytes()
    )
    seed_document["config"]["lifecycle"]["worker_interval_seconds"] = 901.0
    (benchmark_dir / "seed.json").write_bytes(
        canonical_json_bytes(seed_document)
    )
    (benchmark_dir / "cases.jsonl").write_bytes(
        (source_root / "benchmarks" / "cases.jsonl").read_bytes()
    )

    result = RUNNER.invoke(app, ["benchmark"])

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    body = result.stdout[:-1].encode("utf-8")
    report = cast(dict[str, Any], json.loads(body))
    assert canonical_json_bytes(report) == body
    data = cast(dict[str, Any], report["data"])
    assert data["passed"] is True
    assert data["failed_gates"] == []
    assert all(gate["passed"] is True for gate in data["gates"])
    assert data["metrics"]["byte_identical_replay"] is True
    assert data["metrics"]["inspiration_evidence_coverage_failure_count"] == 0
    assert len(data["cases"]) == 15
    inspiration_cases = [
        case for case in data["cases"] if case["type"] == "inspiration"
    ]
    assert all(case["evidence_coverage_complete"] for case in inspiration_cases)
    assert all(case["fixture_expectations_met"] for case in inspiration_cases)
    assert _UUID.search(result.stdout) is None
    assert "database_path" not in result.stdout

    workspace = tmp_path / ".data" / "benchmark"
    snapshot_database = workspace / "snapshot" / "pre-run.sqlite3"
    retrieval_clone = (
        workspace / "replay-a" / "retrieval_english_queue.sqlite3"
    )
    inspiration_clone = (
        workspace / "replay-a" / "inspiration_orion_feedback.sqlite3"
    )

    def counts(path: Path) -> tuple[int, int]:
        with closing(sqlite3.connect(path)) as connection:
            events = cast(
                int,
                connection.execute(
                    "SELECT count(*) FROM domain_events"
                ).fetchone()[0],
            )
            ideas = cast(
                int,
                connection.execute(
                    "SELECT count(*) FROM inspiration_ideas"
                ).fetchone()[0],
            )
        return events, ideas

    snapshot_counts = counts(snapshot_database)
    retrieval_counts = counts(retrieval_clone)
    inspiration_counts = counts(inspiration_clone)
    assert snapshot_counts[1] == retrieval_counts[1] == 0
    assert retrieval_counts[0] > snapshot_counts[0]
    assert inspiration_counts[1] > 0
    assert counts(snapshot_database) == snapshot_counts
