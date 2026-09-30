from __future__ import annotations

import json
import warnings
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.passport_fixtures import passport_document

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.passports import (
    EvidencePassportV1,
    VerifiedPassportV1,
    build_passport_document,
    compute_passport_hash,
    encode_passport_document,
    verify_passport_bytes,
)
from experience_hub.passports.codec import bounded_canonical_bytes
from experience_hub.passports.errors import PassportError


def _payload() -> dict[str, object]:
    return json.loads(encode_passport_document(passport_document()))


def _rehashed(payload: dict[str, object]) -> bytes:
    payload.pop("passport_hash", None)
    payload["passport_hash"] = sha256_hex(canonical_json_bytes(payload))
    return canonical_json_bytes(payload)


def test_round_trip_reports_integrity_not_truth_or_identity() -> None:
    document = passport_document()
    encoded = encode_passport_document(document)
    prepared = verify_passport_bytes(encoded)

    assert prepared.document == document
    assert prepared.canonical_bytes == encoded == canonical_json_bytes(document)
    assert compute_passport_hash(document) == document.passport_hash
    assert prepared.report.publisher_identity == "unverified"
    assert prepared.report.semantic_assessment == "not_assessed"
    assert prepared.report.embedded_excerpt_count == 1
    assert prepared.report.reference_only_count == 0
    assert prepared.report.unavailable_preimage_count == 1
    assert prepared.report.persisted is False


@pytest.mark.parametrize("value", [True, 1.0, "1", 2])
def test_schema_requires_exact_integer_one(value: object) -> None:
    payload = _payload()
    payload["schema_version"] = value
    with pytest.raises(PassportError, match="Invalid evidence passport"):
        verify_passport_bytes(_rehashed(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "missing",
        "duplicate_evidence",
        "extra_evidence",
        "excerpt_hash",
        "pointer",
        "semantic_hash",
        "root_hash",
        "last_subject",
        "first_parent",
        "mixed_hash",
        "repeat_identity",
        "missing_parent",
        "fifth_hop",
        "noncanonical_arrays",
        "wrong_scalar",
    ],
)
def test_rehashed_invalid_structure_is_rejected(mutation: str) -> None:
    payload = _payload()
    subject = payload["subject"]
    snapshots = payload["evidence_snapshots"]
    provenance = payload["provenance"]
    assert isinstance(subject, dict)
    assert isinstance(snapshots, list)
    assert isinstance(provenance, dict)
    content = subject["content"]
    hops = provenance["hops"]
    if mutation == "unknown":
        payload["signed"] = True
    elif mutation == "missing":
        snapshots.clear()
    elif mutation == "duplicate_evidence":
        snapshots.append(snapshots[0])
    elif mutation == "extra_evidence":
        snapshots.append(
            {"mode": "reference_only", "reference": {"type": "test", "id": "unknown"}}
        )
    elif mutation == "excerpt_hash":
        snapshots[0]["excerpt_hash"] = "a" * 64
    elif mutation == "pointer":
        snapshots[0]["step_id"] = "step-2"
    elif mutation == "semantic_hash":
        content["body"] = "A different claim."
    elif mutation == "root_hash":
        provenance["origin_fingerprint"] = "a" * 64
    elif mutation == "last_subject":
        hops[0]["source_version_id"] = str(UUID(int=9))
    elif mutation == "first_parent":
        hops[0]["parent_passport_hash"] = "a" * 64
    elif mutation == "mixed_hash":
        hops[0]["content_hash"] = "a" * 64
    elif mutation == "repeat_identity":
        hops.append(dict(hops[0], parent_passport_hash="a" * 64))
    elif mutation in {"missing_parent", "fifth_hop"}:
        for number in range(1, 5 if mutation == "fifth_hop" else 2):
            hops.insert(0, dict(hops[-1], source_version_id=str(UUID(int=number))))
        for index, hop in enumerate(hops):
            hop["parent_passport_hash"] = None if index == 0 else "a" * 64
        if mutation == "missing_parent":
            hops[-1]["parent_passport_hash"] = None
    elif mutation == "noncanonical_arrays":
        content["tags"] = ["z", "a", "a"]
    elif mutation == "wrong_scalar":
        snapshots[0]["excerpt"] = 5
    with pytest.raises(PassportError):
        verify_passport_bytes(_rehashed(payload))


@pytest.mark.parametrize("suffix", [b"\n", b" ", b"\x00"])
def test_noncanonical_suffix_is_rejected(suffix: bytes) -> None:
    with pytest.raises(PassportError):
        verify_passport_bytes(encode_passport_document(passport_document()) + suffix)


@pytest.mark.parametrize(
    "data",
    [
        b"\xef\xbb\xbf{}",
        b'{"x":1,"x":1}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":"\xff"}',
        b'{"x":"\\ud800"}',
    ],
)
def test_invalid_json_and_unicode_are_rejected(data: bytes) -> None:
    with pytest.raises(PassportError):
        verify_passport_bytes(data)


def test_valid_hash_tampering_is_rejected() -> None:
    payload = _payload()
    payload["passport_hash"] = "f" * 64
    with pytest.raises(PassportError):
        verify_passport_bytes(canonical_json_bytes(payload))


def test_bounded_encoder_exact_limits_and_depth() -> None:
    assert len(bounded_canonical_bytes("a" * (524288 - 2))) == 524288
    with pytest.raises(PassportError) as error:
        bounded_canonical_bytes("a" * (524288 - 1))
    assert error.value.code == "passport_size_limit"
    value: object = "literal { [ ] }"
    for _ in range(16):
        value = [value]
    assert bounded_canonical_bytes(value)
    with pytest.raises(PassportError):
        bounded_canonical_bytes([value])


def test_encoder_aborts_before_quoting_oversized_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = json.dumps
    called_sizes: list[int] = []

    def guarded(value: object, **kwargs: object) -> str:
        if isinstance(value, str):
            called_sizes.append(len(value))
            assert len(value) <= 524288
        return original(value, **kwargs)

    monkeypatch.setattr(json, "dumps", guarded)
    with pytest.raises(PassportError):
        bounded_canonical_bytes({"tag": "a" * 2_000_000})
    assert max(called_sizes, default=0) < 2_000_000


def test_decoder_rejects_oversized_bytes_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> object:
        pytest.fail("Oversized Passport must not enter JSON parsing")

    monkeypatch.setattr(json, "loads", unexpected)
    with pytest.raises(PassportError) as error:
        verify_passport_bytes(b" " * 524289)
    assert error.value.code == "passport_size_limit"


def test_verified_constructor_rejects_forged_values() -> None:
    prepared = verify_passport_bytes(encode_passport_document(passport_document()))
    with pytest.raises((PassportError, ValidationError, ValueError)):
        VerifiedPassportV1(
            document=prepared.document,
            canonical_bytes=b"{}",
            report=prepared.report,
        )
    tampered = EvidencePassportV1.model_construct(
        **dict(prepared.document.model_dump(mode="python"), passport_hash="a" * 64)
    )
    with pytest.raises(PassportError):
        encode_passport_document(tampered)


def test_verified_constructor_does_not_trust_model_copy_scalar_equality() -> None:
    prepared = verify_passport_bytes(encode_passport_document(passport_document()))
    with pytest.raises(PassportError):
        VerifiedPassportV1(
            document=prepared.document.model_copy(update={"schema_version": True}),
            canonical_bytes=prepared.canonical_bytes,
            report=prepared.report,
        )


def test_unchecked_model_errors_never_log_retained_values() -> None:
    prepared = verify_passport_bytes(encode_passport_document(passport_document()))
    unchecked = EvidencePassportV1.model_construct(
        **dict(prepared.document.model_dump(mode="python"), passport_hash="a" * 64)
    )
    with (
        warnings.catch_warnings(record=True) as captured,
        pytest.raises(PassportError),
    ):
        encode_passport_document(unchecked)
    assert captured == []


def test_secret_scanning_covers_retained_content_and_positions() -> None:
    payload = _payload()
    declaration = payload["declaration"]
    assert isinstance(declaration, dict)
    probe = "ghp_" + "a" * 36
    declaration["profile_id"] = probe
    with pytest.raises(PassportError) as error:
        verify_passport_bytes(_rehashed(payload))
    assert error.value.code == "passport_sensitive_content"
    assert probe not in str(error.value)
    assert probe not in repr(error.value.details)
    assert error.value.details == {
        "matches": [{"rule_id": "github_token", "position": "declaration.profile_id"}]
    }


def test_builder_maps_unchecked_values_to_safe_domain_error() -> None:
    document = passport_document()
    probe = "ghp_" + "Z" * 36
    with pytest.raises(PassportError) as error:
        build_passport_document(
            subject=document.subject.model_copy(update={"content_hash": probe}),
            evidence_snapshots=document.evidence_snapshots,
            provenance=document.provenance,
            declaration=document.declaration,
        )
    assert error.value.code == "passport_invalid"
    assert probe not in str(error.value)
