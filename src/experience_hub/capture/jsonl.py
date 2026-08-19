"""Bounded canonical JSONL adapter for sanitized trajectory input."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import ValidationError, field_validator

from experience_hub import canonical_json_bytes, require_utc
from experience_hub.capture.hashing import hash_trajectory_manifest
from experience_hub.capture.models import (
    MAX_TRAJECTORY_STEPS,
    AdapterDescriptorV1,
    CandidateSignalV1,
    OutcomeStatus,
    SanitizationDeclarationV1,
    TrajectoryBundleV1,
    TrajectoryStepV1,
)
from experience_hub.domain import StrictModel
from experience_hub.errors import CanonicalizationError

MAX_JSONL_INPUT_BYTES = 1_048_576

_RFC3339_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})\Z"
)

__all__ = ["GenericJsonlAdapter"]


def _require_rfc3339_timestamp(value: object) -> object:
    if not isinstance(value, str) or not _RFC3339_TIMESTAMP.fullmatch(value):
        raise ValueError("Wire timestamps must be RFC 3339 strings")
    return value


class GenericJsonlHeaderV1(StrictModel):
    record_type: Literal["header"]
    schema_version: Literal[1]
    adapter: AdapterDescriptorV1
    owner_agent_id: UUID
    trajectory_id: str
    source_started_at: datetime
    source_completed_at: datetime
    sanitization: SanitizationDeclarationV1

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_integer_schema_version(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise ValueError("Trajectory schema version must be integer one")
        return value

    @field_validator("source_started_at", "source_completed_at", mode="before")
    @classmethod
    def require_wire_timestamp_strings(cls, value: object) -> object:
        return _require_rfc3339_timestamp(value)

    @field_validator("source_started_at", "source_completed_at", mode="after")
    @classmethod
    def normalize_source_timestamp(cls, value: datetime) -> datetime:
        return require_utc(value)


class GenericJsonlStepRecordV1(StrictModel):
    record_type: Literal["step"]
    step_id: str
    ordinal: int
    occurred_at: datetime
    observation: str
    action: str
    outcome: str
    status: OutcomeStatus
    candidate_signal: CandidateSignalV1 | None = None

    @field_validator("occurred_at", mode="before")
    @classmethod
    def require_wire_timestamp_string(cls, value: object) -> object:
        return _require_rfc3339_timestamp(value)

    @field_validator("ordinal", mode="before")
    @classmethod
    def reject_coerced_ordinal(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Trajectory step ordinal must be an integer")
        return value


def _canonical_json_object(line: str) -> dict[str, object]:
    try:
        parsed = cast(object, json.loads(line))
        encoded = canonical_json_bytes(parsed)
    except (CanonicalizationError, json.JSONDecodeError) as error:
        raise ValueError("Each JSONL line must contain canonical JSON") from error
    if encoded != line.encode("utf-8"):
        raise ValueError("Each JSONL line must use canonical JSON bytes")
    if not isinstance(parsed, dict):
        raise ValueError("Each JSONL line must contain one JSON object")
    if not all(isinstance(key, str) for key in parsed):
        raise ValueError("JSONL object keys must be strings")
    return cast(dict[str, object], parsed)


class GenericJsonlAdapter:
    def parse(self, data: bytes) -> TrajectoryBundleV1:
        if not isinstance(data, bytes):
            raise ValueError("Generic JSONL input must be bytes")
        if not data or len(data) > MAX_JSONL_INPUT_BYTES:
            raise ValueError(
                f"Generic JSONL input must contain 1-{MAX_JSONL_INPUT_BYTES} bytes"
            )
        if b"\r" in data:
            raise ValueError("Generic JSONL input must not contain carriage returns")
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("Generic JSONL input must be valid UTF-8") from error

        lines = text.split("\n")
        if any(not line for line in lines):
            raise ValueError("Generic JSONL input must not contain blank lines")
        if len(lines) < 2:
            raise ValueError("Generic JSONL input must contain a header and steps")
        if len(lines) - 1 > MAX_TRAJECTORY_STEPS:
            raise ValueError(
                f"Generic JSONL input may contain at most {MAX_TRAJECTORY_STEPS} steps"
            )

        records = tuple(_canonical_json_object(line) for line in lines)
        try:
            header = GenericJsonlHeaderV1.model_validate(records[0])
            input_steps = tuple(
                GenericJsonlStepRecordV1.model_validate(record)
                for record in records[1:]
            )
            steps = tuple(
                TrajectoryStepV1(
                    step_id=record.step_id,
                    ordinal=record.ordinal,
                    occurred_at=record.occurred_at,
                    observation=record.observation,
                    action=record.action,
                    outcome=record.outcome,
                    status=record.status,
                    candidate_signal=record.candidate_signal,
                )
                for record in input_steps
            )
        except ValidationError as error:
            raise ValueError(
                "Generic JSONL input does not match the trajectory schema"
            ) from error

        payload: dict[str, object] = {
            "schema_version": header.schema_version,
            "adapter": header.adapter,
            "owner_agent_id": header.owner_agent_id,
            "trajectory_id": header.trajectory_id,
            "source_started_at": header.source_started_at,
            "source_completed_at": header.source_completed_at,
            "sanitization": header.sanitization,
            "steps": steps,
        }
        unchecked = TrajectoryBundleV1.model_construct(
            schema_version=header.schema_version,
            adapter=header.adapter,
            owner_agent_id=header.owner_agent_id,
            trajectory_id=header.trajectory_id,
            source_started_at=header.source_started_at,
            source_completed_at=header.source_completed_at,
            sanitization=header.sanitization,
            steps=steps,
            manifest_hash="",
        )
        payload["manifest_hash"] = hash_trajectory_manifest(unchecked)
        return TrajectoryBundleV1.model_validate(payload)
