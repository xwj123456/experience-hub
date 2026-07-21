"""Canonical response protocol for durable inspiration runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.models import (
    GeneratorKind,
    InspirationOperator,
    InspirationRun,
    InspirationRunStatus,
    OperatorOutcome,
)
from experience_hub.inspiration.response_codec import (
    InspirationErrorResponseV1,
    InspirationResponseCodec,
    InspirationRunResponseV1,
)
from experience_hub.retrieval.ranking import RetrievalMode

RUN_ID = UUID("00000000-0000-0000-0000-000000000601")
RECEIPT_ID = UUID("00000000-0000-0000-0000-000000000602")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000603")
OCCURRED_AT = datetime(2026, 7, 19, 1, 2, 3, 4, tzinfo=UTC)
REQUEST_HASH = "a" * 64
SNAPSHOT_HASH = "b" * 64


def _terminal_run(
    *,
    status: InspirationRunStatus = InspirationRunStatus.COMPLETED,
    completed_at: datetime | None = OCCURRED_AT,
) -> InspirationRun:
    success = OperatorOutcome(
        operator=InspirationOperator.CAUSAL_GAP,
        succeeded=True,
        persisted_ideas=1,
        duplicate_count=0,
        output_tokens_consumed=0,
    )
    failure_code = (
        OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
        if status is InspirationRunStatus.TIMED_OUT
        else OperatorFailureCode.NO_VALID_BRANCHES
    )
    failure = OperatorOutcome(
        operator=(
            InspirationOperator.COUNTERFACTUAL
            if status is InspirationRunStatus.COMPLETED_WITH_ERRORS
            else InspirationOperator.CAUSAL_GAP
        ),
        succeeded=False,
        persisted_ideas=0,
        error_code=failure_code,
        output_tokens_consumed=0,
    )
    operators = (
        (
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        )
        if status is InspirationRunStatus.COMPLETED_WITH_ERRORS
        else (InspirationOperator.CAUSAL_GAP,)
    )
    outcomes = (
        (success, failure)
        if status is InspirationRunStatus.COMPLETED_WITH_ERRORS
        else (
            (success,)
            if status is InspirationRunStatus.COMPLETED
            else (failure,)
        )
    )
    return InspirationRun(
        run_id=RUN_ID,
        owner_agent_id=OWNER_ID,
        goal="Find a falsifiable bridge",
        context="Canonical context",
        mode=RetrievalMode.ASSOCIATIVE,
        generator=GeneratorKind.DETERMINISTIC,
        operators=operators,
        include_inbox=False,
        branches_per_operator=1,
        output_tokens_per_operator=1200,
        total_output_tokens=3600,
        operator_timeout_seconds=30,
        global_timeout_seconds=90,
        request_hash=REQUEST_HASH,
        snapshot_hash=SNAPSHOT_HASH,
        status=status,
        operator_outcomes=outcomes,
        output_tokens_reserved=0,
        output_tokens_consumed=0,
        elapsed_milliseconds=17,
        created_at=OCCURRED_AT,
        completed_at=completed_at,
    )


def test_success_dto_is_strict_frozen_and_has_exact_data_shape() -> None:
    run = _terminal_run()

    response = InspirationRunResponseV1(data=run)

    assert response.model_dump(mode="python") == {"data": run.model_dump(mode="python")}
    with pytest.raises(ValidationError):
        InspirationRunResponseV1.model_validate({"data": run, "extra": True})
    with pytest.raises(ValidationError):
        response.data = run


def test_error_dto_is_strict_frozen_and_has_exact_shared_shape() -> None:
    response = InspirationErrorResponseV1.model_validate(
        {
            "error": {
                "code": "operation_in_progress",
                "message": "The operation is still in progress",
                "details": {"run_id": str(RUN_ID)},
            }
        }
    )

    assert response.model_dump(mode="python") == {
        "error": {
            "code": "operation_in_progress",
            "message": "The operation is still in progress",
            "details": {"run_id": str(RUN_ID)},
        }
    }
    with pytest.raises(ValidationError):
        InspirationErrorResponseV1.model_validate(
            {
                "error": {
                    "code": "operation_in_progress",
                    "message": "The operation is still in progress",
                    "details": {},
                    "raw_exception": "must never escape",
                }
            }
        )


def test_error_dto_reports_noncanonical_details_as_validation_failure() -> None:
    with pytest.raises(ValidationError, match="canonical JSON"):
        InspirationErrorResponseV1.model_validate(
            {
                "error": {
                    "code": "operation_in_progress",
                    "message": "The operation is still in progress",
                    "details": {"unsafe": object()},
                }
            }
        )
    with pytest.raises(ValidationError):
        InspirationErrorResponseV1.model_validate(
            {
                "error": {
                    "code": "operation in progress",
                    "message": "The operation is still in progress",
                    "details": {},
                }
            }
        )


@pytest.mark.parametrize(
    "status",
    [
        InspirationRunStatus.COMPLETED,
        InspirationRunStatus.COMPLETED_WITH_ERRORS,
        InspirationRunStatus.FAILED,
        InspirationRunStatus.TIMED_OUT,
    ],
)
def test_terminal_response_is_canonical_201_for_every_terminal_status(
    status: InspirationRunStatus,
) -> None:
    run = _terminal_run(status=status)

    response = InspirationResponseCodec.terminal(run)

    assert response.status_code == 201
    assert response.content_type == "application/json"
    assert dict(response.headers or {}) == {
        "location": (
            f"/v1/agents/{OWNER_ID}/inspiration-runs/{RUN_ID}"
        )
    }
    assert response.body == canonical_json_bytes({"data": run})
    assert canonical_json_bytes(json.loads(response.body)) == response.body
    assert set(json.loads(response.body)) == {"data"}


def test_terminal_response_rejects_running_run() -> None:
    running = _terminal_run(
        status=InspirationRunStatus.RUNNING,
        completed_at=None,
    )

    with pytest.raises(ValueError, match="terminal"):
        InspirationResponseCodec.terminal(running)


def test_terminal_response_rejects_missing_completion_time() -> None:
    incomplete = _terminal_run(completed_at=None)

    with pytest.raises(ValueError, match="completed_at"):
        InspirationResponseCodec.terminal(incomplete)


def test_terminal_response_rejects_forged_status_and_accounting() -> None:
    completed = _terminal_run()

    with pytest.raises(ValueError, match="status"):
        InspirationResponseCodec.terminal(
            completed.model_copy(
                update={"status": InspirationRunStatus.FAILED}
            )
        )
    with pytest.raises(ValueError, match="accounting"):
        InspirationResponseCodec.terminal(
            completed.model_copy(update={"output_tokens_consumed": 1})
        )


def test_terminal_allows_released_reservations_to_exceed_total_budget() -> None:
    operators = tuple(InspirationOperator)
    outcomes = tuple(
        OperatorOutcome(
            operator=operator,
            succeeded=True,
            persisted_ideas=1,
            output_tokens_consumed=0,
        )
        for operator in operators
    )
    completed = _terminal_run().model_copy(
        update={
            "operators": operators,
            "operator_outcomes": outcomes,
            "total_output_tokens": 1_200,
            "output_tokens_reserved": 3_600,
        }
    )

    response = InspirationResponseCodec.terminal(completed)

    assert response.status_code == 201
    assert json.loads(response.body)["data"]["output_tokens_reserved"] == 3_600


def test_in_progress_response_has_exact_body_and_only_retry_header() -> None:
    response = InspirationResponseCodec.in_progress(
        receipt_id=RECEIPT_ID,
        run_id=RUN_ID,
    )

    assert response.status_code == 409
    assert response.content_type == "application/json"
    assert dict(response.headers or {}) == {"retry-after": "1"}
    assert response.body == (
        b'{"error":{"code":"operation_in_progress","details":'
        b'{"receipt_id":"00000000-0000-0000-0000-000000000602",'
        b'"resource":{"id":"00000000-0000-0000-0000-000000000601",'
        b'"type":"inspiration_run"}},'
        b'"message":"The operation is still in progress"}}'
    )


@pytest.mark.parametrize(
    ("receipt_id", "run_id"),
    [(str(RECEIPT_ID), RUN_ID), (RECEIPT_ID, str(RUN_ID))],
)
def test_in_progress_response_rejects_non_uuid_identifiers(
    receipt_id: object,
    run_id: object,
) -> None:
    with pytest.raises(ValueError, match="UUID"):
        InspirationResponseCodec.in_progress(  # type: ignore[arg-type]
            receipt_id=receipt_id,
            run_id=run_id,
        )


def test_generator_not_configured_response_is_exact_replayable_422() -> None:
    response = InspirationResponseCodec.generator_not_configured()

    assert response.status_code == 422
    assert response.content_type == "application/json"
    assert dict(response.headers or {}) == {}
    assert response.body == (
        b'{"error":{"code":"generator_not_configured","details":{},'
        b'"message":"The selected inspiration generator is not configured."}}'
    )


def test_generator_not_configured_accepts_only_the_fixed_sanitized_error() -> None:
    matching = ReplayableCommandError(
        code="generator_not_configured",
        message="The selected inspiration generator is not configured.",
        status_code=422,
    )
    response = InspirationResponseCodec.generator_not_configured(matching)
    assert response == InspirationResponseCodec.generator_not_configured()

    for error in (
        ReplayableCommandError(
            code="other_error",
            message="The selected inspiration generator is not configured.",
            status_code=422,
        ),
        ReplayableCommandError(
            code="generator_not_configured",
            message="raw provider detail",
            status_code=422,
        ),
        ReplayableCommandError(
            code="generator_not_configured",
            message="The selected inspiration generator is not configured.",
            details={"raw": "must never escape"},
            status_code=422,
        ),
    ):
        with pytest.raises(ValueError, match="generator_not_configured"):
            InspirationResponseCodec.generator_not_configured(error)


def test_codec_rejects_wrong_input_types_instead_of_coercing() -> None:
    with pytest.raises(ValueError, match="InspirationRun"):
        InspirationResponseCodec.terminal({"run_id": str(RUN_ID)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ReplayableCommandError"):
        InspirationResponseCodec.generator_not_configured(Exception("unsafe"))  # type: ignore[arg-type]


def test_terminal_can_represent_sanitized_failed_operator_outcome() -> None:
    failed_run = _terminal_run(status=InspirationRunStatus.FAILED).model_copy(
        update={
            "operator_outcomes": (
                OperatorOutcome(
                    operator=InspirationOperator.CAUSAL_GAP,
                    succeeded=False,
                    persisted_ideas=0,
                    error_code=OperatorFailureCode.NO_VALID_BRANCHES,
                    output_tokens_consumed=0,
                ),
            )
        }
    )

    response = InspirationResponseCodec.terminal(failed_run)

    assert b"no_valid_branches" in response.body
    assert b"raw" not in response.body
