"""Canonical stored responses for the durable inspiration-run protocol."""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from pydantic import field_validator

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.errors import CanonicalizationError
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.models import (
    InspirationModel,
    InspirationRun,
    InspirationRunStatus,
)
from experience_hub.storage.idempotency import StoredResponse

_ERROR_CODE = re.compile(r"(?=.{1,64}\Z)[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*\Z")
_CONTENT_TYPE = "application/json"
_GENERATOR_NOT_CONFIGURED_CODE = "generator_not_configured"
_GENERATOR_NOT_CONFIGURED_MESSAGE = (
    "The selected inspiration generator is not configured."
)
_IN_PROGRESS_MESSAGE = "The operation is still in progress"
_TERMINAL_STATUSES = frozenset(
    {
        InspirationRunStatus.COMPLETED,
        InspirationRunStatus.COMPLETED_WITH_ERRORS,
        InspirationRunStatus.FAILED,
        InspirationRunStatus.TIMED_OUT,
    }
)


def _require_terminal_semantics(run: InspirationRun) -> None:
    outcomes = run.operator_outcomes
    succeeded = sum(outcome.succeeded for outcome in outcomes)
    if run.completed_at is None:
        raise ValueError("a terminal run requires completed_at")
    if (
        run.completed_at < run.created_at
        or isinstance(run.output_tokens_reserved, bool)
        or isinstance(run.output_tokens_consumed, bool)
        or isinstance(run.elapsed_milliseconds, bool)
        or not 0
        <= run.output_tokens_consumed
        <= run.output_tokens_reserved
        <= 3_600
        or not 0
        <= run.output_tokens_consumed
        <= run.total_output_tokens
        <= 3_600
        or run.elapsed_milliseconds < 0
        or run.output_tokens_consumed
        != sum(outcome.output_tokens_consumed for outcome in outcomes)
        or (
            outcomes
            and tuple(outcome.operator for outcome in outcomes)
            != run.operators
        )
        or (outcomes and run.snapshot_hash is None)
    ):
        raise ValueError("run terminal accounting is inconsistent")
    if run.status is InspirationRunStatus.COMPLETED:
        valid = bool(outcomes) and succeeded == len(outcomes)
    elif run.status is InspirationRunStatus.COMPLETED_WITH_ERRORS:
        valid = 0 < succeeded < len(outcomes)
    elif run.status is InspirationRunStatus.FAILED:
        valid = succeeded == 0
    else:
        valid = any(
            outcome.error_code
            is OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
            for outcome in outcomes
        )
    if not valid:
        raise ValueError("run terminal status is inconsistent")


class InspirationRunResponseV1(InspirationModel):
    """The exact public success envelope retained by a run receipt."""

    data: InspirationRun


class _InspirationErrorV1(InspirationModel):
    code: str
    message: str
    details: dict[str, Any]

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value != value.strip() or not _ERROR_CODE.fullmatch(value):
            raise ValueError("error code must be lowercase snake case")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("error message must be nonblank and trimmed")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("error message must contain valid Unicode") from error
        return value

    @field_validator("details", mode="before")
    @classmethod
    def validate_details(cls, value: Any) -> Any:
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise ValueError("error details must be an object with string keys")
        try:
            canonical_json_bytes(value)
        except CanonicalizationError as error:
            raise ValueError("error details must use canonical JSON values") from error
        return value


class InspirationErrorResponseV1(InspirationModel):
    """The byte-compatible shared public error envelope."""

    error: _InspirationErrorV1


class InspirationResponseCodec:
    """Create only the approved stored responses for inspiration execution."""

    @classmethod
    def terminal(cls, run: InspirationRun) -> StoredResponse:
        """Encode any bounded terminal run as one retained HTTP-equivalent 201."""
        if not isinstance(run, InspirationRun):
            raise ValueError("run must be an InspirationRun")
        try:
            retained = InspirationRun.model_validate(
                run.model_dump(mode="python", warnings=False),
                strict=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("run must be a valid InspirationRun") from error
        if retained.status not in _TERMINAL_STATUSES:
            raise ValueError("run status must be terminal")
        _require_terminal_semantics(retained)
        envelope = InspirationRunResponseV1(data=retained)
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(envelope),
            content_type=_CONTENT_TYPE,
            headers={
                "location": (
                    f"/v1/agents/{retained.owner_agent_id}/"
                    f"inspiration-runs/{retained.run_id}"
                ),
            },
        )

    @classmethod
    def in_progress(
        cls,
        *,
        receipt_id: UUID,
        run_id: UUID,
    ) -> StoredResponse:
        """Encode a visible retained run that has not reached transaction three."""
        if not isinstance(receipt_id, UUID) or not isinstance(run_id, UUID):
            raise ValueError("receipt_id and run_id must be UUID values")
        return cls._error_response(
            status_code=409,
            code="operation_in_progress",
            message=_IN_PROGRESS_MESSAGE,
            details={
                "receipt_id": str(receipt_id),
                "resource": {
                    "id": str(run_id),
                    "type": "inspiration_run",
                },
            },
            headers={"retry-after": "1"},
        )

    @classmethod
    def generator_not_configured(
        cls,
        error: ReplayableCommandError | None = None,
    ) -> StoredResponse:
        """Encode only the fixed sanitized selected-generator failure."""
        if error is not None:
            if not isinstance(error, ReplayableCommandError):
                raise ValueError("error must be a ReplayableCommandError")
            if (
                error.code != _GENERATOR_NOT_CONFIGURED_CODE
                or error.message != _GENERATOR_NOT_CONFIGURED_MESSAGE
                or error.status_code != 422
                or error.details != {}
            ):
                raise ValueError(
                    "error must be the fixed generator_not_configured error"
                )
        return cls._error_response(
            status_code=422,
            code=_GENERATOR_NOT_CONFIGURED_CODE,
            message=_GENERATOR_NOT_CONFIGURED_MESSAGE,
            details={},
        )

    @staticmethod
    def _error_response(
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> StoredResponse:
        envelope = InspirationErrorResponseV1(
            error=_InspirationErrorV1(
                code=code,
                message=message,
                details=details,
            )
        )
        return StoredResponse(
            status_code=status_code,
            body=canonical_json_bytes(envelope),
            content_type=_CONTENT_TYPE,
            headers=headers,
        )


__all__ = [
    "InspirationErrorResponseV1",
    "InspirationResponseCodec",
    "InspirationRunResponseV1",
]
