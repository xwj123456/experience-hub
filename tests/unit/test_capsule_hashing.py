from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import (
    ExperienceKind,
    Temperature,
    VersionContent,
)
from experience_hub.sharing.hashing import (
    compute_capsule_hash,
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import ProvenanceHop

CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000101")
TOPIC_ID = UUID("00000000-0000-0000-0000-000000000102")
SOURCE_EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000103")
SOURCE_VERSION_ID = UUID("00000000-0000-0000-0000-000000000104")
PUBLISHER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
ROOT_FINGERPRINT = "b" * 64
SOURCE_CONTENT_HASH = "c" * 64
CREATED_AT = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 7, 25, 8, 30, tzinfo=UTC)

FIRST_HOP = ProvenanceHop(
    capsule_id=UUID("00000000-0000-0000-0000-000000000201"),
    publisher_agent_id=UUID("00000000-0000-0000-0000-000000000301"),
)
SECOND_HOP = ProvenanceHop(
    capsule_id=UUID("00000000-0000-0000-0000-000000000202"),
    publisher_agent_id=UUID("00000000-0000-0000-0000-000000000302"),
)


def _version_content() -> VersionContent:
    return VersionContent(
        body="  保留正文空格\n与换行  ",
        summary="Restart safely",
        mechanism="lease handoff",
        tags=("ops", "recovery"),
        applicability=("linux", "single writer"),
        evidence=(
            TypedEvidence(type="log", id="case-1"),
            TypedEvidence(type="trace", id="case-2"),
        ),
        falsifiers=("two owners overlap", "lease is stale"),
    )


def _transport(**overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "transport_schema_version": 1,
        "capsule_id": CAPSULE_ID,
        "topic_id": TOPIC_ID,
        "source_experience_id": SOURCE_EXPERIENCE_ID,
        "source_version_id": SOURCE_VERSION_ID,
        "publisher_agent_id": PUBLISHER_ID,
        "kind": ExperienceKind.PROCEDURAL,
        "body": "  保留正文空格\n与换行  ",
        "summary": "Restart safely",
        "mechanism": "lease handoff",
        "tags": ("ops", "recovery"),
        "applicability": ("linux", "single writer"),
        "evidence": (
            TypedEvidence(type="log", id="case-1"),
            TypedEvidence(type="trace", id="case-2"),
        ),
        "falsifiers": ("two owners overlap", "lease is stale"),
        "publisher_confidence": 0.75,
        "provenance_chain": (FIRST_HOP, SECOND_HOP),
        "root_fingerprint": ROOT_FINGERPRINT,
        "source_content_hash": SOURCE_CONTENT_HASH,
        "created_at": CREATED_AT,
        "expires_at": EXPIRES_AT,
        "hop_count": 2,
    }
    values.update(overrides)
    return values


def test_source_content_hash_excludes_owner_and_lifecycle_fields() -> None:
    content = _version_content()
    first_source = {
        "owner_agent_id": UUID("00000000-0000-0000-0000-000000000401"),
        "temperature": Temperature.HOT,
        "archived": False,
        "content": content,
    }
    adopted_source = {
        "owner_agent_id": UUID("00000000-0000-0000-0000-000000000402"),
        "temperature": Temperature.ARCHIVED,
        "archived": True,
        "content": content,
    }

    first_hash = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content,
    ).content_hash
    adopted_hash = encode_version_content(
        kind=ExperienceKind.PROCEDURAL,
        content=content,
    ).content_hash

    assert first_source["owner_agent_id"] != adopted_source["owner_agent_id"]
    assert first_source["temperature"] != adopted_source["temperature"]
    assert first_source["archived"] != adopted_source["archived"]
    assert first_hash == adopted_hash


def test_original_root_fingerprint_has_locked_canonical_formula() -> None:
    expected_document = canonical_json_bytes(
        {
            "root_publisher_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "source_content_hash": SOURCE_CONTENT_HASH,
        }
    )

    assert compute_original_root_fingerprint(
        root_publisher_id=PUBLISHER_ID,
        source_content_hash=SOURCE_CONTENT_HASH,
    ) == sha256_hex(expected_document)


def test_capsule_hash_has_locked_canonical_transport_formula() -> None:
    expected_document = canonical_json_bytes(
        {
            "transport_schema_version": 1,
            "capsule_id": CAPSULE_ID,
            "topic_id": TOPIC_ID,
            "source_experience_id": SOURCE_EXPERIENCE_ID,
            "source_version_id": SOURCE_VERSION_ID,
            "publisher_agent_id": PUBLISHER_ID,
            "kind": ExperienceKind.PROCEDURAL,
            "body": "  保留正文空格\n与换行  ",
            "summary": "Restart safely",
            "mechanism": "lease handoff",
            "tags": ("ops", "recovery"),
            "applicability": ("linux", "single writer"),
            "evidence": (
                TypedEvidence(type="log", id="case-1"),
                TypedEvidence(type="trace", id="case-2"),
            ),
            "falsifiers": ("lease is stale", "two owners overlap"),
            "publisher_confidence": 0.75,
            "provenance_chain": (FIRST_HOP, SECOND_HOP),
            "root_fingerprint": ROOT_FINGERPRINT,
            "source_content_hash": SOURCE_CONTENT_HASH,
            "created_at": CREATED_AT,
            "expires_at": EXPIRES_AT,
            "hop_count": 2,
        }
    )

    assert compute_capsule_hash(**_transport()) == sha256_hex(expected_document)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("transport_schema_version", 2),
        ("capsule_id", UUID("00000000-0000-0000-0000-000000000111")),
        ("topic_id", UUID("00000000-0000-0000-0000-000000000112")),
        (
            "source_experience_id",
            UUID("00000000-0000-0000-0000-000000000113"),
        ),
        ("source_version_id", UUID("00000000-0000-0000-0000-000000000114")),
        ("publisher_agent_id", UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")),
        ("kind", ExperienceKind.SEMANTIC),
        ("body", "  保留正文空格\n与换行 changed  "),
        ("summary", "Restart deterministically"),
        ("mechanism", "fenced lease handoff"),
        ("tags", ("ops", "recovery", "sqlite")),
        ("applicability", ("linux", "single writer", "sqlite")),
        (
            "evidence",
            (
                TypedEvidence(type="log", id="case-1"),
                TypedEvidence(type="trace", id="case-3"),
            ),
        ),
        ("falsifiers", ("two owners overlap", "lease is not fenced")),
        ("publisher_confidence", 0.76),
        ("provenance_chain", (SECOND_HOP, FIRST_HOP)),
        ("root_fingerprint", "d" * 64),
        ("source_content_hash", "e" * 64),
        ("created_at", CREATED_AT + timedelta(microseconds=1)),
        ("expires_at", EXPIRES_AT + timedelta(microseconds=1)),
        ("hop_count", 3),
    ],
)
def test_capsule_hash_covers_each_transport_field(
    field: str,
    changed: object,
) -> None:
    assert compute_capsule_hash(
        **_transport(**{field: changed})
    ) != compute_capsule_hash(**_transport())


def test_capsule_hash_canonicalizes_metadata_arrays_and_evidence() -> None:
    canonical = compute_capsule_hash(**_transport())
    reordered_with_duplicates = compute_capsule_hash(
        **_transport(
            tags=("recovery", "ops", "ops"),
            applicability=("single writer", "linux", "linux"),
            evidence=(
                TypedEvidence(type="trace", id="case-2"),
                TypedEvidence(type="log", id="case-1"),
                TypedEvidence(type="trace", id="case-2"),
            ),
            falsifiers=(
                "lease is stale",
                "two owners overlap",
                "lease is stale",
            ),
        )
    )

    assert reordered_with_duplicates == canonical


def test_capsule_hash_normalizes_equivalent_aware_timestamps_to_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))

    assert compute_capsule_hash(
        **_transport(
            created_at=CREATED_AT.astimezone(plus_eight),
            expires_at=EXPIRES_AT.astimezone(plus_eight),
        )
    ) == compute_capsule_hash(**_transport())


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            compute_original_root_fingerprint,
            {
                "root_publisher_id": str(PUBLISHER_ID),
                "source_content_hash": SOURCE_CONTENT_HASH,
            },
        ),
        (
            compute_original_root_fingerprint,
            {
                "root_publisher_id": PUBLISHER_ID,
                "source_content_hash": SOURCE_CONTENT_HASH.upper(),
            },
        ),
        (
            compute_capsule_hash,
            _transport(capsule_id=str(CAPSULE_ID)),
        ),
        (
            compute_capsule_hash,
            _transport(created_at=CREATED_AT.replace(tzinfo=None)),
        ),
        (
            compute_capsule_hash,
            _transport(root_fingerprint=ROOT_FINGERPRINT.upper()),
        ),
        (
            compute_capsule_hash,
            _transport(source_content_hash="c" * 63),
        ),
    ],
)
def test_hash_inputs_require_uuid_utc_and_lowercase_sha256(
    function: Any,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(**kwargs)
