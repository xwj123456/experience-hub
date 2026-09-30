"""Bounded canonical encoding, integrity checks and metadata-only inspection."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from experience_hub import sha256_hex
from experience_hub.capture.sanitization import scan_text_fields
from experience_hub.experiences.content import encode_version_content
from experience_hub.passports.contracts import (
    MAX_PASSPORT_BYTES,
    MAX_PASSPORT_DEPTH,
    EmbeddedExcerptSnapshotV1,
    EvidencePassportV1,
    EvidenceSnapshotV1,
    PassportDeclarationV1,
    PassportInspectionV1,
    PassportProvenanceV1,
    PassportSubjectV1,
)
from experience_hub.passports.errors import PassportError


class _BoundedWriter:
    def __init__(self) -> None:
        self.body = bytearray()

    def append(self, value: bytes) -> None:
        if len(self.body) + len(value) > MAX_PASSPORT_BYTES:
            raise PassportError("size_limit")
        self.body.extend(value)

    def string(self, value: str) -> None:
        remaining = MAX_PASSPORT_BYTES - len(self.body)
        size = 2
        for character in value:
            point = ord(character)
            if 0xD800 <= point <= 0xDFFF:
                raise PassportError("invalid")
            if character in '"\\' or character in "\b\f\n\r\t":
                size += 2
            elif point < 32:
                size += 6
            else:
                size += (
                    1
                    if point < 128
                    else 2
                    if point < 2048
                    else 3
                    if point < 65536
                    else 4
                )
            if size > remaining:
                raise PassportError("size_limit")
        # Quoting is now bounded by the exact remaining byte budget.
        self.append(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def write(self, value: object, depth: int = 0) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python", warnings="error")
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, UUID):
            value = str(value)
        if isinstance(value, str):
            self.string(value)
        elif value is None:
            self.append(b"null")
        elif isinstance(value, bool):
            self.append(b"true" if value else b"false")
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise PassportError("invalid")
                value = 0.0 if value == 0 else value
            self.append(json.dumps(value, allow_nan=False).encode("ascii"))
        elif isinstance(value, (Mapping, tuple, list)):
            if depth >= MAX_PASSPORT_DEPTH:
                raise PassportError("invalid")
            if isinstance(value, Mapping):
                keys: list[str] = []
                for key in value:
                    if not isinstance(key, str):
                        raise PassportError("invalid")
                    keys.append(key)
                self.append(b"{")
                for index, key in enumerate(sorted(keys)):
                    if index:
                        self.append(b",")
                    self.string(key)
                    self.append(b":")
                    self.write(value[key], depth + 1)
                self.append(b"}")
            else:
                self.append(b"[")
                for index, item in enumerate(value):
                    if index:
                        self.append(b",")
                    self.write(item, depth + 1)
                self.append(b"]")
        else:
            raise PassportError("invalid")


def bounded_canonical_bytes(value: object) -> bytes:
    """Encode the Passport value subset, never accumulating over 512 KiB."""
    writer = _BoundedWriter()
    try:
        writer.write(value)
    except (PydanticSerializationError, UnicodeError, ValueError, RecursionError):
        raise PassportError("invalid") from None
    return bytes(writer.body)


def compute_passport_hash(document: EvidencePassportV1) -> str:
    bounded_canonical_bytes(document)
    values = document.model_dump(
        mode="python", exclude={"passport_hash"}, warnings="error"
    )
    return sha256_hex(bounded_canonical_bytes(values))


def _text_fields(value: object, position: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield position, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _text_fields(item, f"{position}.{key}".lstrip("."))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _text_fields(item, f"{position}[{index}]")


def validate_document(document: EvidencePassportV1) -> None:
    """Check size before any legacy encoder can quote unbounded metadata."""
    bounded_canonical_bytes(document)
    subject = document.subject
    if (
        subject.content_hash
        != encode_version_content(
            kind=subject.kind, content=subject.content
        ).content_hash
    ):
        raise PassportError("invalid")
    references = tuple(snapshot.reference for snapshot in document.evidence_snapshots)
    if references != subject.content.evidence:
        raise PassportError("invalid")
    for snapshot in document.evidence_snapshots:
        if isinstance(snapshot, EmbeddedExcerptSnapshotV1) and (
            snapshot.excerpt_hash != sha256_hex(snapshot.excerpt.encode("utf-8"))
            or snapshot.reference.type != "trajectory_field"
            or snapshot.reference.id
            != (
                f"{snapshot.source_manifest_hash}:{snapshot.step_id}:"
                f"{snapshot.field.value}"
            )
        ):
            raise PassportError("invalid")
    hops = document.provenance.hops
    identities = set()
    for index, hop in enumerate(hops):
        identity = (
            hop.source_agent_id,
            hop.source_experience_id,
            hop.source_version_id,
        )
        if (
            identity in identities
            or hop.content_hash != subject.content_hash
            or (hop.parent_passport_hash is None) != (index == 0)
        ):
            raise PassportError("invalid")
        identities.add(identity)
    last = hops[-1]
    if (
        last.source_agent_id != subject.source_agent_id
        or last.source_experience_id != subject.source_experience_id
        or last.source_version_id != subject.source_version_id
        or last.source_origin != subject.source_origin
    ):
        raise PassportError("invalid")
    root = hops[0]
    fingerprint = sha256_hex(
        bounded_canonical_bytes(
            {
                "source_agent_id": root.source_agent_id,
                "content_hash": root.content_hash,
            }
        )
    )
    if document.provenance.origin_fingerprint != fingerprint:
        raise PassportError("invalid")
    if document.passport_hash != compute_passport_hash(document):
        raise PassportError("invalid")
    matches = scan_text_fields(
        _text_fields(document.model_dump(mode="python", warnings="error"))
    )
    if matches:
        raise PassportError("sensitive_content", matches=matches)


def encode_passport_document(document: EvidencePassportV1) -> bytes:
    try:
        bounded_canonical_bytes(document)
        retained = EvidencePassportV1.model_validate(
            document.model_dump(mode="python", warnings="error"), strict=True
        )
        return bounded_canonical_bytes(retained)
    except (ValidationError, ValueError, TypeError, UnicodeError):
        raise PassportError("invalid") from None


def build_passport_document(
    *,
    subject: PassportSubjectV1,
    evidence_snapshots: tuple[EvidenceSnapshotV1, ...],
    provenance: PassportProvenanceV1,
    declaration: PassportDeclarationV1,
) -> EvidencePassportV1:
    try:
        draft = EvidencePassportV1.model_construct(
            schema_version=1,
            format="experience_passport",
            subject=subject,
            evidence_snapshots=evidence_snapshots,
            provenance=provenance,
            declaration=declaration,
            passport_hash="0" * 64,
        )
        bounded_canonical_bytes(draft)
        values = draft.model_dump(mode="python", warnings="error")
        values["passport_hash"] = compute_passport_hash(draft)
        return EvidencePassportV1.model_validate(values, strict=True)
    except (ValidationError, PydanticSerializationError, ValueError, UnicodeError):
        raise PassportError("invalid") from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PassportError("invalid")
        result[key] = value
    return result


def _check_input_depth(data: bytes) -> None:
    depth = 0
    quoted = False
    escaped = False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_PASSPORT_DEPTH:
                raise PassportError("invalid")
        elif byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise PassportError("invalid")


def _decode_document(data: bytes) -> EvidencePassportV1:
    if type(data) is not bytes:
        raise PassportError("invalid")
    if len(data) > MAX_PASSPORT_BYTES:
        raise PassportError("size_limit")
    _check_input_depth(data)
    try:
        value: object = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_object
        )
        if bounded_canonical_bytes(value) != data:
            raise PassportError("invalid")
        document = EvidencePassportV1.model_validate_json(data, strict=True)
        if bounded_canonical_bytes(document) != data:
            raise PassportError("invalid")
        return document
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise PassportError("invalid") from None


def _report(document: EvidencePassportV1) -> PassportInspectionV1:
    embedded = sum(
        isinstance(snapshot, EmbeddedExcerptSnapshotV1)
        for snapshot in document.evidence_snapshots
    )
    total = len(document.evidence_snapshots)
    return PassportInspectionV1(
        passport_hash=document.passport_hash,
        source_agent_id=document.subject.source_agent_id,
        kind=document.subject.kind,
        embedded_excerpt_count=embedded,
        reference_only_count=total - embedded,
        unavailable_preimage_count=total,
    )


@dataclass(frozen=True, slots=True)
class VerifiedPassportV1:
    document: EvidencePassportV1
    canonical_bytes: bytes
    report: PassportInspectionV1

    def __post_init__(self) -> None:
        document = _decode_document(self.canonical_bytes)
        if encode_passport_document(self.document) != self.canonical_bytes:
            raise PassportError("invalid")
        try:
            report = PassportInspectionV1.model_validate(
                self.report.model_dump(mode="python", warnings="error"), strict=True
            )
        except (ValidationError, PydanticSerializationError):
            raise PassportError("invalid") from None
        if report != _report(document):
            raise PassportError("invalid")
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "report", report)


def verify_passport_bytes(data: bytes) -> VerifiedPassportV1:
    document = _decode_document(data)
    return VerifiedPassportV1(document, data, _report(document))
