"""Strict generic response and error envelopes."""

from __future__ import annotations

import re

from pydantic import JsonValue, field_validator

from experience_hub.domain.values import StrictModel

_ERROR_CODE = re.compile(r"(?=.{1,64}\Z)[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*\Z")


class ErrorBody(StrictModel):
    code: str
    message: str
    details: dict[str, JsonValue]

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value != value.strip() or _ERROR_CODE.fullmatch(value) is None:
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


class ErrorEnvelope(StrictModel):
    error: ErrorBody


class DataEnvelope[T](StrictModel):
    data: T


class Page(StrictModel):
    next_cursor: str | None


class PageEnvelope[T](StrictModel):
    data: tuple[T, ...]
    page: Page


__all__ = [
    "DataEnvelope",
    "ErrorBody",
    "ErrorEnvelope",
    "Page",
    "PageEnvelope",
]
