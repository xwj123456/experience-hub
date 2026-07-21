"""Strict values for immutable experience content and storage."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import field_validator

from experience_hub import canonical_json_bytes
from experience_hub.domain import StrictModel, TypedEvidence

MAX_BODY_UTF8_BYTES = 64 * 1_024
MAX_SUMMARY_CHARACTERS = 1_000
MAX_MECHANISM_CHARACTERS = 2_000
MAX_VERSION_LIST_ITEMS = 32


class ExperienceKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    HYPOTHESIS = "hypothesis"


class ExperienceOrigin(StrEnum):
    LOCAL = "local"
    ADOPTED_CAPSULE = "adopted_capsule"
    ADOPTED_IDEA = "adopted_idea"


class Temperature(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"


class PayloadCodec(StrEnum):
    PLAIN = "plain"
    ZLIB = "zlib"


class LinkRelation(StrEnum):
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    TESTS = "tests"


def _canonical_tuple(values: tuple[Any, ...]) -> tuple[Any, ...]:
    unique = {canonical_json_bytes(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


class VersionContent(StrictModel):
    """Semantic version content, excluding identity-owned experience kind."""

    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[TypedEvidence, ...]
    falsifiers: tuple[str, ...]

    @field_validator("tags", "applicability", "evidence", "falsifiers", mode="before")
    @classmethod
    def enforce_input_array_limit(cls, values: Any) -> Any:
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            return values
        if len(values) > MAX_VERSION_LIST_ITEMS:
            raise ValueError(
                f"Version content arrays may contain at most "
                f"{MAX_VERSION_LIST_ITEMS} input items"
            )
        return values

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Body must contain valid Unicode") from error
        if not value.strip():
            raise ValueError("Body must not be blank")
        if len(encoded) > MAX_BODY_UTF8_BYTES:
            raise ValueError("Body must be at most 64 KiB when UTF-8 encoded")
        return value

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Summary must contain valid Unicode") from error
        if not value.strip():
            raise ValueError("Summary must not be blank")
        if len(value) > MAX_SUMMARY_CHARACTERS:
            raise ValueError("Summary must contain at most 1,000 characters")
        return value

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Mechanism must contain valid Unicode") from error
        if not value.strip():
            raise ValueError("Mechanism must not be blank")
        if len(value) > MAX_MECHANISM_CHARACTERS:
            raise ValueError("Mechanism must contain at most 2,000 characters")
        return value

    @field_validator("tags", "applicability", "falsifiers", mode="after")
    @classmethod
    def canonicalize_string_array(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    "Version list values must contain valid Unicode"
                ) from error
            if not value.strip():
                raise ValueError("Version list values must not be blank")
        return _canonical_tuple(values)

    @field_validator("evidence", mode="after")
    @classmethod
    def canonicalize_evidence_array(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        for evidence in values:
            for value in (evidence.type, evidence.id):
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ValueError(
                        "Evidence values must contain valid Unicode"
                    ) from error
        return _canonical_tuple(values)


_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class EncodedVersionContent(StrictModel):
    """Physical body bytes plus the two distinct semantic digests."""

    codec: PayloadCodec
    payload: bytes
    payload_hash: str
    content_hash: str

    @field_validator("payload_hash", "content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("Hash must be a lowercase SHA-256 hex digest")
        return value
