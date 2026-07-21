from __future__ import annotations

import json
import zlib
from typing import Any

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.content import (
    decode_payload,
    decode_version_content,
    encode_version_content,
    preferred_payload_codec,
    reencode_payload,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    PayloadCodec,
    Temperature,
    VersionContent,
)


def _content(**overrides: object) -> VersionContent:
    values: dict[str, Any] = {
        "body": "保留  原始空格\n和换行",
        "summary": "Restart safely",
        "mechanism": "lease handoff",
        "tags": ("ops",),
        "applicability": ("single writer",),
        "evidence": (TypedEvidence(type="log", id="case-1"),),
        "falsifiers": ("two owners overlap",),
    }
    values.update(overrides)
    return VersionContent(**values)


def test_version_content_canonicalizes_array_values_by_canonical_json() -> None:
    content = _content(
        tags=("zeta", "alpha", "zeta"),
        applicability=("single writer", "linux", "single writer"),
        evidence=(
            TypedEvidence(type="trace", id="2"),
            TypedEvidence(type="log", id="1"),
            TypedEvidence(type="trace", id="2"),
        ),
        falsifiers=("z", "a", "z"),
    )

    assert content.tags == ("alpha", "zeta")
    assert content.applicability == ("linux", "single writer")
    assert content.evidence == (
        TypedEvidence(type="log", id="1"),
        TypedEvidence(type="trace", id="2"),
    )
    assert content.falsifiers == ("a", "z")


def test_content_hash_excludes_identity_and_includes_payload_hash() -> None:
    content = VersionContent(
        body="保留 原始空格",
        summary="Restart safely",
        mechanism="lease handoff",
        tags=("ops", "ops"),
        applicability=("single writer",),
        evidence=(TypedEvidence(type="log", id="case-1"),),
        falsifiers=("two owners overlap",),
    )

    encoded = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content,
    )

    assert encoded.payload_hash != encoded.content_hash
    assert decode_payload(encoded.codec, encoded.payload) == (
        b'{"body":"\xe4\xbf\x9d\xe7\x95\x99 '
        b'\xe5\x8e\x9f\xe5\xa7\x8b\xe7\xa9\xba\xe6\xa0\xbc"}'
    )
    expected_content = canonical_json_bytes(
        {
            "applicability": ["single writer"],
            "evidence": [{"id": "case-1", "type": "log"}],
            "falsifiers": ["two owners overlap"],
            "kind": "procedural",
            "mechanism": "lease handoff",
            "payload_hash": encoded.payload_hash,
            "summary": "Restart safely",
            "tags": ["ops"],
        }
    )
    assert encoded.content_hash == sha256_hex(expected_content)

    changed_kind = encode_version_content(
        kind=ExperienceKind.SEMANTIC,
        content=content,
    )
    assert changed_kind.payload_hash == encoded.payload_hash
    assert changed_kind.content_hash != encoded.content_hash


def test_body_is_preserved_while_metadata_round_trips_canonically() -> None:
    content = _content(tags=("ops", "ops"), body="  body\twith\nspacing  ")

    encoded = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content,
    )
    decoded = decode_version_content(
        body_payload=decode_payload(encoded.codec, encoded.payload),
        summary=content.summary,
        mechanism=content.mechanism,
        tags=content.tags,
        applicability=content.applicability,
        evidence=content.evidence,
        falsifiers=content.falsifiers,
    )

    assert decoded == content
    assert decoded.body == "  body\twith\nspacing  "
    assert decoded.tags == ("ops",)


@pytest.mark.parametrize("codec", list(PayloadCodec))
def test_plain_and_zlib_payloads_round_trip(codec: PayloadCodec) -> None:
    content = _content()

    encoded = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        codec=codec,
    )
    decoded = decode_payload(encoded.codec, encoded.payload)

    assert decoded == canonical_json_bytes({"body": content.body})
    assert encoded.payload_hash == sha256_hex(decoded)


def test_codec_rewrite_changes_physical_bytes_but_preserves_decoded_hash() -> None:
    decoded = canonical_json_bytes({"body": "repeat " * 100})

    plain = reencode_payload(decoded, PayloadCodec.PLAIN)
    compressed = reencode_payload(decoded, PayloadCodec.ZLIB)

    assert plain != compressed
    assert decode_payload(PayloadCodec.PLAIN, plain) == decoded
    assert decode_payload(PayloadCodec.ZLIB, compressed) == decoded
    assert sha256_hex(decode_payload(PayloadCodec.PLAIN, plain)) == sha256_hex(
        decode_payload(PayloadCodec.ZLIB, compressed)
    )


@pytest.mark.parametrize(
    ("codec", "payload", "message"),
    [
        (PayloadCodec.PLAIN, b'{"body":"\xff"}', "UTF-8"),
        (PayloadCodec.PLAIN, b"[]", "object"),
        (PayloadCodec.PLAIN, b'{"body":1}', "string"),
        (PayloadCodec.PLAIN, b'{"body":"ok","extra":1}', "only"),
        (PayloadCodec.PLAIN, b'{ "body": "ok" }', "canonical"),
        (PayloadCodec.ZLIB, b"not-zlib", "zlib"),
    ],
)
def test_decode_payload_rejects_invalid_decoded_content(
    codec: PayloadCodec,
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_payload(codec, payload)


def test_version_content_has_no_kind_field_and_is_strict() -> None:
    values = json.loads(_content().model_dump_json())
    values["kind"] = "procedural"

    with pytest.raises(ValidationError, match="kind"):
        VersionContent.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("body", "x" * 65_537, "64 KiB"),
        ("summary", "x" * 1_001, "1,000"),
        ("mechanism", "x" * 2_001, "2,000"),
        ("summary", "   ", "blank"),
        ("mechanism", "", "blank"),
        ("summary", "\ud800", "Unicode"),
        ("mechanism", "\udfff", "Unicode"),
    ],
)
def test_version_content_enforces_core_text_limits(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _content(**{field: value})


def test_array_limit_is_checked_before_deduplication() -> None:
    with pytest.raises(ValidationError, match="32"):
        _content(tags=("duplicate",) * 33)


def test_evidence_rejects_invalid_unicode_before_hashing() -> None:
    evidence = (TypedEvidence(type="\ud800", id="case-1"),)

    with pytest.raises(ValidationError, match="Unicode"):
        _content(evidence=evidence)


def test_zlib_decode_rejects_trailing_or_oversized_data() -> None:
    decoded = canonical_json_bytes({"body": "valid"})
    with pytest.raises(ValueError, match="trailing"):
        decode_payload(PayloadCodec.ZLIB, zlib.compress(decoded) + b"trailing")

    oversized = canonical_json_bytes({"body": "x" * 65_537})
    with pytest.raises(ValueError, match="64 KiB"):
        decode_payload(PayloadCodec.ZLIB, zlib.compress(oversized))


@pytest.mark.parametrize(
    ("temperature", "codec"),
    [
        (Temperature.HOT, PayloadCodec.PLAIN),
        (Temperature.WARM, PayloadCodec.PLAIN),
        (Temperature.COLD, PayloadCodec.ZLIB),
        (Temperature.ARCHIVED, PayloadCodec.ZLIB),
    ],
)
def test_preferred_codec_depends_only_on_effective_temperature(
    temperature: Temperature,
    codec: PayloadCodec,
) -> None:
    assert preferred_payload_codec(temperature) is codec
