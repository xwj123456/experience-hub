"""Closed, sanitized failure codes for inspiration operators."""

from enum import StrEnum


class OperatorFailureCode(StrEnum):
    """Sanitized failure categories shared by every generator boundary."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_VALID_BRANCHES = "no_valid_branches"
    GENERATOR_ERROR = "generator_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_HTTP_ERROR = "provider_http_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_EVIDENCE_REFERENCE = "invalid_evidence_reference"
    PROVIDER_BUDGET_VIOLATION = "provider_budget_violation"
    INSUFFICIENT_TOKEN_RESERVATION = "insufficient_token_reservation"
    GLOBAL_DEADLINE_EXHAUSTED = "global_deadline_exhausted"


__all__ = ["OperatorFailureCode"]
