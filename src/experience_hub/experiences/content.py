"""Canonical experience payload encoding and semantic hashing."""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable
from typing import Any

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import (
    MAX_BODY_UTF8_BYTES,
    EncodedVersionContent,
    ExperienceKind,
    PayloadCodec,
    Temperature,
    VersionContent,
)

# JSON escaping can expand each one-byte control character to six bytes.
_MAX_DECODED_PAYLOAD_BYTES = 6 * MAX_BODY_UTF8_BYTES + len(b'{"body":""}')


def preferred_payload_codec(temperature: Temperature) -> PayloadCodec:
    """Return the physical codec preferred by an effective temperature."""
    if temperature in {Temperature.HOT, Temperature.WARM}:
        return PayloadCodec.PLAIN
    return PayloadCodec.ZLIB


def _validated_decoded_payload(decoded: bytes) -> bytes:
    if len(decoded) > _MAX_DECODED_PAYLOAD_BYTES:
        raise ValueError("Decoded payload body exceeds the 64 KiB UTF-8 limit")
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Decoded payload must be valid UTF-8") from error

    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Decoded payload must be valid JSON") from error

    if not isinstance(value, dict):
        raise ValueError("Decoded payload must be a JSON object")
    if set(value) != {"body"}:
        raise ValueError("Decoded payload may contain only the body field")
    if not isinstance(value["body"], str):
        raise ValueError("Decoded payload body must be a string")
    try:
        body_bytes = value["body"].encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Decoded payload body must contain valid Unicode") from error
    if not value["body"].strip():
        raise ValueError("Decoded payload body must not be blank")
    if len(body_bytes) > MAX_BODY_UTF8_BYTES:
        raise ValueError("Decoded payload body exceeds the 64 KiB UTF-8 limit")
    if canonical_json_bytes(value) != decoded:
        raise ValueError("Decoded payload must use canonical JSON encoding")
    return decoded


def decode_payload(codec: PayloadCodec, payload: bytes) -> bytes:
    """Decode and validate a stored payload, returning canonical body bytes."""
    encoded = bytes(payload)
    if codec is PayloadCodec.PLAIN:
        decoded = encoded
    elif codec is PayloadCodec.ZLIB:
        decompressor = zlib.decompressobj()
        try:
            decoded = decompressor.decompress(
                encoded,
                _MAX_DECODED_PAYLOAD_BYTES + 1,
            )
        except zlib.error as error:
            raise ValueError("Invalid zlib payload") from error
        if decompressor.unconsumed_tail or len(decoded) > _MAX_DECODED_PAYLOAD_BYTES:
            raise ValueError("Decoded payload body exceeds the 64 KiB UTF-8 limit")
        try:
            decoded += decompressor.flush()
        except zlib.error as error:
            raise ValueError("Invalid zlib payload") from error
        if not decompressor.eof:
            raise ValueError("Invalid zlib payload: compressed stream is incomplete")
        if decompressor.unused_data:
            raise ValueError("Invalid zlib payload: trailing data is not allowed")
    else:  # pragma: no cover - exhaustive for the declared enum
        raise ValueError(f"Unsupported payload codec: {codec}")
    return _validated_decoded_payload(decoded)


def reencode_payload(decoded_payload: bytes, codec: PayloadCodec) -> bytes:
    """Encode already-decoded canonical bytes without changing their meaning."""
    decoded = _validated_decoded_payload(bytes(decoded_payload))
    if codec is PayloadCodec.PLAIN:
        return decoded
    if codec is PayloadCodec.ZLIB:
        return zlib.compress(decoded, level=9)
    raise ValueError(f"Unsupported payload codec: {codec}")


def _content_hash_document(
    *,
    kind: ExperienceKind,
    content: VersionContent,
    payload_hash: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "summary": content.summary,
        "mechanism": content.mechanism,
        "tags": content.tags,
        "applicability": content.applicability,
        "evidence": content.evidence,
        "falsifiers": content.falsifiers,
        "payload_hash": payload_hash,
    }


def encode_version_content(
    *,
    kind: ExperienceKind,
    content: VersionContent,
    codec: PayloadCodec = PayloadCodec.PLAIN,
) -> EncodedVersionContent:
    """Encode a body and compute the approved payload/content hash formulas."""
    decoded_payload = canonical_json_bytes({"body": content.body})
    payload_hash = sha256_hex(decoded_payload)
    content_hash = sha256_hex(
        canonical_json_bytes(
            _content_hash_document(
                kind=kind,
                content=content,
                payload_hash=payload_hash,
            )
        )
    )
    return EncodedVersionContent(
        codec=codec,
        payload=reencode_payload(decoded_payload, codec),
        payload_hash=payload_hash,
        content_hash=content_hash,
    )


def decode_version_content(
    *,
    body_payload: bytes,
    summary: str,
    mechanism: str,
    tags: Iterable[str],
    applicability: Iterable[str],
    evidence: Iterable[TypedEvidence],
    falsifiers: Iterable[str],
) -> VersionContent:
    """Combine validated decoded body bytes with immutable version metadata."""
    decoded = _validated_decoded_payload(bytes(body_payload))
    value = json.loads(decoded)
    return VersionContent(
        body=value["body"],
        summary=summary,
        mechanism=mechanism,
        tags=tuple(tags),
        applicability=tuple(applicability),
        evidence=tuple(evidence),
        falsifiers=tuple(falsifiers),
    )
