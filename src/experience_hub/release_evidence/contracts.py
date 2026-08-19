"""Strict, versioned contracts for release verification evidence."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from experience_hub.domain import StrictModel

_STRICT = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
    revalidate_instances="always",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _ReleaseEvidenceModel(StrictModel):
    model_config = _STRICT


class CheckName(StrEnum):
    LOCK = "lock"
    RUFF = "ruff"
    MYPY = "mypy"
    PYTEST = "pytest"
    DEMO = "demo"
    BENCHMARK = "benchmark"
    BUILD = "build"


REQUIRED_RELEASE_CHECKS = tuple(CheckName)


class CheckEvidenceV1(_ReleaseEvidenceModel):
    name: CheckName
    passed: StrictBool


class DemoEvidenceV1(_ReleaseEvidenceModel):
    all_invariants_hold: Literal[True]
    stage_count: Annotated[StrictInt, Field(ge=1)]

    @field_validator("all_invariants_hold", mode="before")
    @classmethod
    def require_literal_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("all_invariants_hold must be the boolean true")
        return value


class BenchmarkEvidenceV1(_ReleaseEvidenceModel):
    passed: Literal[True]
    case_count: Annotated[StrictInt, Field(ge=1)]
    gate_count: Annotated[StrictInt, Field(ge=1)]
    passed_gate_count: Annotated[StrictInt, Field(ge=1)]
    byte_identical_replay: Literal[True]
    pending_capsule_leakage_count: Literal[0]

    @field_validator("passed", "byte_identical_replay", mode="before")
    @classmethod
    def require_literal_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("benchmark truth values must be the boolean true")
        return value

    @field_validator("pending_capsule_leakage_count", mode="before")
    @classmethod
    def require_literal_integer_zero(cls, value: object) -> object:
        if type(value) is not int or value != 0:
            raise ValueError("pending capsule leakage must be the integer zero")
        return value


class ReleaseEvidenceDataV1(_ReleaseEvidenceModel):
    schema_version: Literal[1]
    verified_commit: str
    source_tree_sha256: str
    verified_on: str
    python_version: Literal["3.12"]
    checks: tuple[CheckEvidenceV1, ...]
    test_count: Annotated[StrictInt, Field(ge=1)]
    demo: DemoEvidenceV1
    benchmark: BenchmarkEvidenceV1

    @field_validator("checks", mode="before")
    @classmethod
    def preserve_json_check_order(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("verified_commit")
    @classmethod
    def validate_verified_commit(cls, value: str) -> str:
        if not _COMMIT.fullmatch(value):
            raise ValueError("verified_commit must be a lowercase 40-character commit")
        return value

    @field_validator("source_tree_sha256")
    @classmethod
    def validate_source_tree_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError(
                "source_tree_sha256 must be a lowercase SHA-256 hex digest"
            )
        return value

    @field_validator("verified_on")
    @classmethod
    def validate_verified_on(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("verified_on must be an ISO calendar date") from error
        if parsed.isoformat() != value:
            raise ValueError("verified_on must be an ISO calendar date")
        return value

    @model_validator(mode="after")
    def validate_check_closure(self) -> Self:
        names = tuple(check.name for check in self.checks)
        if names != REQUIRED_RELEASE_CHECKS or not all(
            check.passed for check in self.checks
        ):
            raise ValueError("checks must contain every ordered passing release check")
        if self.benchmark.passed_gate_count != self.benchmark.gate_count:
            raise ValueError("every benchmark gate must pass")
        return self


class ReleaseEvidenceReportV1(_ReleaseEvidenceModel):
    data: ReleaseEvidenceDataV1
