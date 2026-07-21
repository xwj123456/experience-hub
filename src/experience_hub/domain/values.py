"""Strict shared domain value objects."""

import re
from hashlib import sha256
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TypedEvidence(StrictModel):
    type: str
    id: str

    @field_validator("type", "id")
    @classmethod
    def require_nonempty_trimmed(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence fields must not be blank")
        return normalized


_REASON_CODE = re.compile(r"(?=.{1,64}\Z)[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class StructuredReason(StrictModel):
    code: str
    text: str
    text_hash: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value != value.strip() or not _REASON_CODE.fullmatch(value):
            raise ValueError(
                "Reason code must be lowercase snake case (1-64 characters)"
            )
        return value

    @field_validator("text")
    @classmethod
    def retain_trimmed_text(cls, value: str) -> str:
        normalized = value.strip()
        if not 1 <= len(normalized) <= 2_000:
            raise ValueError(
                "Reason text must contain 1-2,000 characters after trimming"
            )
        return normalized

    @field_validator("text_hash")
    @classmethod
    def validate_hash_shape(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Reason text hash must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_retained_text_hash(self) -> Self:
        expected = sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_hash != expected:
            raise ValueError("Reason text hash does not match retained text")
        return self

    @classmethod
    def from_user_text(cls, text: str) -> Self:
        retained = text.strip()
        return cls(
            code="user_provided",
            text=retained,
            text_hash=sha256(retained.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def policy_due(cls) -> Self:
        text = "Archived because the retention policy is due."
        return cls(
            code="policy_due",
            text=text,
            text_hash=sha256(text.encode("utf-8")).hexdigest(),
        )
