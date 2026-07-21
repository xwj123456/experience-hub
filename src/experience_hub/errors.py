"""Stable, user-safe domain error types."""

from collections.abc import Mapping
from typing import Any


class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.status_code = status_code


class CanonicalizationError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__("canonicalization_error", message, status_code=400)


class CallerScope(str):
    """A validated caller scope, such as ``system:local`` or ``agent:<uuid>``."""

    def __new__(cls, value: str) -> "CallerScope":
        normalized = value.strip()
        if not normalized:
            raise ValueError("Caller scope must not be blank")
        return super().__new__(cls, normalized)
