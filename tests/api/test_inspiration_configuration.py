from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from experience_hub.api.app import create_app
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest
from experience_hub.inspiration import (
    INSPIRATION_OPERATOR_ORDER,
    GeneratorKind,
    InspirationOperator,
    StartInspirationRun,
)
from experience_hub.retrieval import RetrievalMode
from experience_hub.storage.idempotency import StoredResponse

NOW = datetime(2026, 7, 20, 9, 15, tzinfo=UTC)
_SUCCESS_BYTES = b'{"data":{"sentinel":"configuration accepted"}}'


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


@dataclass(frozen=True, slots=True)
class _ExecutorCall:
    request: CommandRequest
    run: StartInspirationRun


class _SpyRunExecutor:
    def __init__(self) -> None:
        self.calls: list[_ExecutorCall] = []

    async def execute(
        self,
        *,
        request: CommandRequest,
        run: StartInspirationRun,
    ) -> StoredResponse:
        self.calls.append(_ExecutorCall(request=request, run=run))
        return StoredResponse(
            status_code=201,
            body=_SUCCESS_BYTES,
            headers={"location": "/v1/sentinel-inspiration-run"},
        )


class _ForbiddenCommandExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, *_: Any, **__: Any) -> None:
        self.calls += 1
        raise AssertionError(
            "inspiration configuration must not use the ordinary CommandExecutor"
        )


@dataclass(frozen=True, slots=True)
class _Harness:
    client: TestClient
    database_path: Path
    owner_id: UUID
    run_executor: _SpyRunExecutor
    command_executor: _ForbiddenCommandExecutor
    baseline_counts: dict[str, int]

    @property
    def route(self) -> str:
        return f"/v1/agents/{self.owner_id}/inspiration-runs"


def _database_counts(path: Path) -> dict[str, int]:
    tables = {
        "events": "domain_events",
        "receipts": "idempotency_records",
        "runs": "inspiration_runs",
    }
    with sqlite3.connect(path) as connection:
        retained: dict[str, int] = {}
        for label, table in tables.items():
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            retained[label] = int(row[0])
    return retained


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    database_path = tmp_path / "inspiration-configuration.sqlite3"
    app = create_app(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/agents",
            headers={"Idempotency-Key": "configuration-owner"},
            json={"name": "Inspiration configuration owner"},
        )
        assert created.status_code == 201, created.text
        owner_id = UUID(created.json()["data"]["agent_id"])

        run_executor = _SpyRunExecutor()
        command_executor = _ForbiddenCommandExecutor()
        app.state.container.inspiration_run_executor = run_executor
        app.state.container.command_executor = command_executor
        yield _Harness(
            client=client,
            database_path=database_path,
            owner_id=owner_id,
            run_executor=run_executor,
            command_executor=command_executor,
            baseline_counts=_database_counts(database_path),
        )


def _post(
    harness: _Harness,
    *,
    key: str,
    body: dict[str, object],
    query: str = "",
) -> httpx.Response:
    return cast(
        httpx.Response,
        harness.client.post(
            f"{harness.route}{query}",
            headers={"Idempotency-Key": key},
            json=body,
        ),
    )


def _assert_no_mutation(
    harness: _Harness,
    *,
    executor_calls_before: int,
) -> None:
    assert len(harness.run_executor.calls) == executor_calls_before
    assert harness.command_executor.calls == 0
    assert _database_counts(harness.database_path) == harness.baseline_counts


def test_omitted_and_explicit_defaults_produce_the_same_canonical_command(
    harness: _Harness,
) -> None:
    goal = "Explore a bounded inspiration configuration."
    omitted = _post(
        harness,
        key="configuration-defaults-omitted",
        body={"goal": goal},
    )
    explicit = _post(
        harness,
        key="configuration-defaults-explicit",
        body={
            "branches_per_operator": 3,
            "context": "",
            "generator": "deterministic",
            "global_timeout_seconds": 90,
            "goal": goal,
            "include_inbox": False,
            "mode": "associative",
            "operator_timeout_seconds": 30,
            "operators": [
                "causal_gap",
                "counterfactual",
                "distant_analogy",
            ],
            "output_tokens_per_operator": 1_200,
            "total_output_tokens": 3_600,
        },
    )

    assert omitted.status_code == explicit.status_code == 201
    assert omitted.content == explicit.content == _SUCCESS_BYTES
    omitted_call, explicit_call = harness.run_executor.calls
    expected = StartInspirationRun(
        owner_agent_id=harness.owner_id,
        goal=goal,
    )
    assert omitted_call.run == explicit_call.run == expected
    assert omitted_call.request.request_hash == explicit_call.request.request_hash
    assert expected.branches_per_operator * len(expected.operators) == 9
    assert dict(omitted_call.request.body) == {
        "branches_per_operator": 3,
        "context": "",
        "generator": "deterministic",
        "global_timeout_seconds": 90,
        "goal": goal,
        "include_inbox": False,
        "mode": "associative",
        "operator_timeout_seconds": 30,
        "operators": (
            "causal_gap",
            "counterfactual",
            "distant_analogy",
        ),
        "output_tokens_per_operator": 1_200,
        "total_output_tokens": 3_600,
    }
    assert harness.command_executor.calls == 0


def test_every_accepted_boundary_reaches_the_split_transaction_executor(
    harness: _Harness,
) -> None:
    cases: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
        (
            "branches-minimum",
            {"branches_per_operator": 1},
            {"branches_per_operator": 1},
        ),
        (
            "branches-maximum",
            {"branches_per_operator": 3},
            {"branches_per_operator": 3},
        ),
        (
            "operator-tokens-minimum",
            {"output_tokens_per_operator": 1},
            {"output_tokens_per_operator": 1},
        ),
        (
            "operator-tokens-maximum",
            {"output_tokens_per_operator": 1_200},
            {"output_tokens_per_operator": 1_200},
        ),
        (
            "total-tokens-minimum",
            {"total_output_tokens": 1},
            {"total_output_tokens": 1},
        ),
        (
            "total-tokens-maximum",
            {"total_output_tokens": 3_600},
            {"total_output_tokens": 3_600},
        ),
        (
            "operator-timeout-minimum",
            {
                "global_timeout_seconds": 1,
                "operator_timeout_seconds": 1,
            },
            {
                "global_timeout_seconds": 1,
                "operator_timeout_seconds": 1,
            },
        ),
        (
            "operator-timeout-maximum",
            {"operator_timeout_seconds": 30},
            {"operator_timeout_seconds": 30},
        ),
        (
            "global-timeout-minimum",
            {
                "global_timeout_seconds": 1,
                "operator_timeout_seconds": 1,
            },
            {
                "global_timeout_seconds": 1,
                "operator_timeout_seconds": 1,
            },
        ),
        (
            "global-timeout-maximum",
            {"global_timeout_seconds": 90},
            {"global_timeout_seconds": 90},
        ),
        (
            "equal-timeouts",
            {
                "global_timeout_seconds": 17,
                "operator_timeout_seconds": 17,
            },
            {
                "global_timeout_seconds": 17,
                "operator_timeout_seconds": 17,
            },
        ),
        (
            "goal-and-context-maximum",
            {
                "context": "c" * 4_000,
                "goal": "g" * 2_000,
            },
            {
                "context": "c" * 4_000,
                "goal": "g" * 2_000,
            },
        ),
        (
            "strict-boolean-true",
            {"include_inbox": True},
            {"include_inbox": True},
        ),
        (
            "focused-mode",
            {"mode": "focused"},
            {"mode": RetrievalMode.FOCUSED},
        ),
        (
            "optional-provider-enum",
            {"generator": "openai_compatible"},
            {"generator": GeneratorKind.OPENAI_COMPATIBLE},
        ),
    )

    for name, overrides, expected_fields in cases:
        body: dict[str, object] = {"goal": "Exercise one accepted public boundary."}
        body.update(overrides)
        response = _post(
            harness,
            key=f"configuration-valid-{name}",
            body=body,
        )

        assert response.status_code == 201, (name, response.text)
        call = harness.run_executor.calls[-1]
        assert call.run.owner_agent_id == harness.owner_id
        for field, expected in expected_fields.items():
            assert getattr(call.run, field) == expected, name

    assert len(harness.run_executor.calls) == len(cases)
    assert harness.command_executor.calls == 0


def test_invalid_budget_types_ranges_and_cross_field_values_are_pre_mutation(
    harness: _Harness,
) -> None:
    invalid: list[tuple[str, dict[str, object]]] = [
        ("branches-below", {"branches_per_operator": 0}),
        ("branches-above", {"branches_per_operator": 4}),
        ("operator-tokens-below", {"output_tokens_per_operator": 0}),
        ("operator-tokens-above", {"output_tokens_per_operator": 1_201}),
        ("total-tokens-below", {"total_output_tokens": 0}),
        ("total-tokens-above", {"total_output_tokens": 3_601}),
        ("operator-timeout-below", {"operator_timeout_seconds": 0}),
        ("operator-timeout-above", {"operator_timeout_seconds": 31}),
        ("global-timeout-below", {"global_timeout_seconds": 0}),
        ("global-timeout-above", {"global_timeout_seconds": 91}),
        (
            "global-less-than-operator",
            {
                "global_timeout_seconds": 29,
                "operator_timeout_seconds": 30,
            },
        ),
    ]
    integer_fields = (
        "branches_per_operator",
        "output_tokens_per_operator",
        "total_output_tokens",
        "operator_timeout_seconds",
        "global_timeout_seconds",
    )
    nonintegers: tuple[tuple[str, object], ...] = (
        ("boolean-true", True),
        ("boolean-false", False),
        ("float", 1.0),
        ("string", "1"),
    )
    invalid.extend(
        (
            f"{field}-{label}",
            {field: value},
        )
        for field in integer_fields
        for label, value in nonintegers
    )
    invalid.extend(
        (
            f"include-inbox-{label}",
            {"include_inbox": value},
        )
        for label, value in (
            ("integer-zero", 0),
            ("integer-one", 1),
            ("string", "true"),
            ("null", None),
        )
    )

    for index, (name, overrides) in enumerate(invalid):
        calls_before = len(harness.run_executor.calls)
        body: dict[str, object] = {"goal": "Reject an invalid bounded configuration."}
        body.update(overrides)
        response = _post(
            harness,
            key=f"configuration-invalid-budget-{index}",
            body=body,
        )

        assert response.status_code == 422, (name, response.text)
        assert response.json()["error"]["code"] == "validation_error"
        _assert_no_mutation(
            harness,
            executor_calls_before=calls_before,
        )


def test_malformed_text_enums_and_extra_inputs_are_pre_mutation(
    harness: _Harness,
) -> None:
    invalid: tuple[tuple[str, dict[str, object]], ...] = (
        ("blank-goal", {"goal": " \t "}),
        ("goal-too-long", {"goal": "g" * 2_001}),
        (
            "context-too-long",
            {
                "context": "c" * 4_001,
                "goal": "Reject an oversized context.",
            },
        ),
        ("goal-not-text", {"goal": 1}),
        (
            "context-not-text",
            {
                "context": 1,
                "goal": "Reject a non-text context.",
            },
        ),
        (
            "unknown-mode",
            {
                "goal": "Reject an unknown mode.",
                "mode": "creative",
            },
        ),
        (
            "unknown-generator",
            {
                "generator": "ambient_provider",
                "goal": "Reject an unknown generator.",
            },
        ),
        (
            "extra-field",
            {
                "goal": "Reject an extra input.",
                "temperature": 0.7,
            },
        ),
    )

    for index, (name, body) in enumerate(invalid):
        calls_before = len(harness.run_executor.calls)
        response = _post(
            harness,
            key=f"configuration-invalid-shape-{index}",
            body=body,
        )

        assert response.status_code == 422, (name, response.text)
        assert response.json()["error"]["code"] == "validation_error"
        _assert_no_mutation(
            harness,
            executor_calls_before=calls_before,
        )

    calls_before = len(harness.run_executor.calls)
    unknown_query = _post(
        harness,
        key="configuration-unknown-query",
        body={"goal": "Reject an unknown query parameter."},
        query="?unexpected=true",
    )
    assert unknown_query.status_code == 422, unknown_query.text
    assert unknown_query.json()["error"]["code"] == "validation_error"
    _assert_no_mutation(
        harness,
        executor_calls_before=calls_before,
    )


def test_every_nonempty_operator_subset_accepts_all_permutations_canonically(
    harness: _Harness,
) -> None:
    expected_subsets: set[tuple[InspirationOperator, ...]] = set()
    request_hashes: dict[
        tuple[InspirationOperator, ...],
        set[str],
    ] = {}
    case_number = 0

    for size in range(1, len(INSPIRATION_OPERATOR_ORDER) + 1):
        for subset in combinations(INSPIRATION_OPERATOR_ORDER, size):
            canonical = tuple(
                operator
                for operator in INSPIRATION_OPERATOR_ORDER
                if operator in subset
            )
            expected_subsets.add(canonical)
            request_hashes[canonical] = set()
            for permutation in permutations(subset):
                response = _post(
                    harness,
                    key=f"configuration-operator-permutation-{case_number}",
                    body={
                        "goal": "Canonicalize the selected operators.",
                        "operators": [operator.value for operator in permutation],
                    },
                )
                case_number += 1

                assert response.status_code == 201, (
                    permutation,
                    response.text,
                )
                call = harness.run_executor.calls[-1]
                assert call.run.operators == canonical
                assert tuple(call.request.body["operators"]) == tuple(
                    operator.value for operator in canonical
                )
                request_hashes[canonical].add(call.request.request_hash)
                assert len(call.run.operators) * call.run.branches_per_operator <= 9

    assert len(expected_subsets) == 7
    assert case_number == 15
    assert all(len(hashes) == 1 for hashes in request_hashes.values())
    assert len(harness.run_executor.calls) == case_number
    assert harness.command_executor.calls == 0


def test_invalid_operator_selections_are_rejected_before_any_mutation(
    harness: _Harness,
) -> None:
    invalid: tuple[tuple[str, object], ...] = (
        ("empty", []),
        ("duplicate", ["causal_gap", "causal_gap"]),
        ("unknown", ["causal_gap", "free_association"]),
        (
            "too-many",
            [
                "causal_gap",
                "counterfactual",
                "distant_analogy",
                "causal_gap",
            ],
        ),
        ("not-a-list", "causal_gap"),
        ("null", None),
    )

    for index, (name, operators) in enumerate(invalid):
        calls_before = len(harness.run_executor.calls)
        response = _post(
            harness,
            key=f"configuration-invalid-operators-{index}",
            body={
                "goal": "Reject an invalid operator selection.",
                "operators": operators,
            },
        )

        assert response.status_code == 422, (name, response.text)
        assert response.json()["error"]["code"] == "validation_error"
        _assert_no_mutation(
            harness,
            executor_calls_before=calls_before,
        )
