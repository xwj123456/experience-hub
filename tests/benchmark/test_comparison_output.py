from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from importlib import import_module
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from experience_hub.benchmark.runner import (
    BenchmarkExecution,
    BenchmarkOutputError,
    canonical_benchmark_bytes,
)
from experience_hub.canonical import canonical_json_bytes

_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)


def _representative_report() -> dict[str, Any]:
    return {
        "data": {
            "cases": [
                {
                    "case_id": "retrieval_english_queue",
                    "returned": [
                        {
                            "label": "english_bounded_queue",
                            "rank": 1,
                            "score": "0.982100000000",
                        }
                    ],
                    "type": "retrieval",
                },
                {
                    "case_id": "inspiration_orion_feedback",
                    "ideas": [
                        {
                            "content_hash": "a" * 64,
                            "idea": "idea#1",
                            "mechanism_hash": "b" * 64,
                            "operator": "causal_gap",
                            "ordinal": 1,
                        }
                    ],
                    "type": "inspiration",
                },
            ],
            "failed_gates": [],
            "gates": [],
            "metrics": {},
            "passed": True,
            "schema_version": 1,
        }
    }


def test_canonical_output_is_single_stable_document_without_runtime_identity() -> None:
    report = _representative_report()

    first = canonical_benchmark_bytes(report)
    second = canonical_benchmark_bytes(json.loads(first))

    assert first == second == canonical_json_bytes(report)
    assert first.count(b"\n") == 0
    assert _UUID.search(first.decode("utf-8")) is None
    decoded = cast(dict[str, Any], json.loads(first))
    assert decoded["data"]["cases"][1]["ideas"][0]["idea"] == "idea#1"


@pytest.mark.parametrize(
    "forbidden",
    (
        {"receipt_id": "receipt#1"},
        {"event_id": 42},
        {"run_id": "run#1"},
        {"idea_id": "idea#1"},
        {"occurrence_id": "occurrence#1"},
        {"snapshot_item_id": "snapshot-item#1"},
        {"created_at": "2026-02-01T00:00:00Z"},
        {"database_path": "/tmp/benchmark.sqlite3"},
        {"elapsed_milliseconds": 1},
        {"observed": datetime(2026, 2, 1, tzinfo=UTC)},
        {"observed": "2026-02-01T00:00:00"},
        {"score": 0.9821234567891234},
        {"score": float("nan")},
        {"score": Decimal("0.9821")},
        {"score": "0.9821"},
        {"workspace": "/tmp/private.sqlite3"},
    ),
)
def test_canonical_output_rejects_forbidden_runtime_fields(
    forbidden: dict[str, object],
) -> None:
    report = _representative_report()
    report["data"]["cases"][0].update(forbidden)

    with pytest.raises(BenchmarkOutputError):
        canonical_benchmark_bytes(report)


def test_cli_exits_nonzero_with_failed_gate_names(monkeypatch: Any) -> None:
    cli_module = import_module("experience_hub.cli.app")
    report = _representative_report()
    report["data"]["failed_gates"] = ["cold_macro_recall_at_5"]
    report["data"]["passed"] = False
    body = canonical_benchmark_bytes(report)

    async def failed_benchmark() -> BenchmarkExecution:
        return BenchmarkExecution(
            report=report,
            body=body,
            passed=False,
            failed_gates=("cold_macro_recall_at_5",),
        )

    monkeypatch.setattr(cli_module, "run_benchmark", failed_benchmark)
    result = CliRunner().invoke(cli_module.app, ["benchmark"])

    assert result.exit_code == 1
    assert result.stdout == body.decode("utf-8") + "\n"
    assert "cold_macro_recall_at_5" in result.stdout
