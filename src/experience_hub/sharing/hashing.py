"""Canonical semantic and transport hashes for experience capsules."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any
from uuid import UUID

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.clock import require_utc
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import ExperienceKind, VersionContent
from experience_hub.sharing.models import ProvenanceHop

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _uuid(name: str, value: Any) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{name} must be a UUID")
    return value


def _hash(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _timestamp(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    try:
        return require_utc(value)
    except ValueError as error:
        raise ValueError(f"{name} must be timezone-aware") from error


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("publisher_confidence must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("publisher_confidence must be between zero and one")
    return converted


def _positive_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("transport_schema_version must be a positive integer")
    return int(value)


def _hop_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("hop_count must be a non-negative integer")
    return int(value)


def _provenance(values: Any) -> tuple[ProvenanceHop, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(value, ProvenanceHop) for value in values
    ):
        raise TypeError("provenance_chain must be a tuple of ProvenanceHop")
    return values


def compute_original_root_fingerprint(
    *,
    root_publisher_id: UUID,
    source_content_hash: str,
) -> str:
    """Hash the independent root publisher and semantic source content."""
    document = {
        "root_publisher_id": _uuid(
            "root_publisher_id",
            root_publisher_id,
        ),
        "source_content_hash": _hash(
            "source_content_hash",
            source_content_hash,
        ),
    }
    return sha256_hex(canonical_json_bytes(document))


def compute_capsule_hash(
    *,
    transport_schema_version: int,
    capsule_id: UUID,
    topic_id: UUID,
    source_experience_id: UUID,
    source_version_id: UUID,
    publisher_agent_id: UUID,
    kind: ExperienceKind,
    body: str,
    summary: str,
    mechanism: str,
    tags: tuple[str, ...],
    applicability: tuple[str, ...],
    evidence: tuple[TypedEvidence, ...],
    falsifiers: tuple[str, ...],
    publisher_confidence: float,
    provenance_chain: tuple[ProvenanceHop, ...],
    root_fingerprint: str,
    source_content_hash: str,
    created_at: datetime,
    expires_at: datetime,
    hop_count: int,
) -> str:
    """Hash every immutable transport field in its canonical representation."""
    if not isinstance(kind, ExperienceKind):
        raise TypeError("kind must be an ExperienceKind")
    content = VersionContent(
        body=body,
        summary=summary,
        mechanism=mechanism,
        tags=tags,
        applicability=applicability,
        evidence=evidence,
        falsifiers=falsifiers,
    )
    document = {
        "transport_schema_version": _positive_schema_version(transport_schema_version),
        "capsule_id": _uuid("capsule_id", capsule_id),
        "topic_id": _uuid("topic_id", topic_id),
        "source_experience_id": _uuid(
            "source_experience_id",
            source_experience_id,
        ),
        "source_version_id": _uuid("source_version_id", source_version_id),
        "publisher_agent_id": _uuid(
            "publisher_agent_id",
            publisher_agent_id,
        ),
        "kind": kind,
        "body": content.body,
        "summary": content.summary,
        "mechanism": content.mechanism,
        "tags": content.tags,
        "applicability": content.applicability,
        "evidence": content.evidence,
        "falsifiers": content.falsifiers,
        "publisher_confidence": _confidence(publisher_confidence),
        "provenance_chain": _provenance(provenance_chain),
        "root_fingerprint": _hash("root_fingerprint", root_fingerprint),
        "source_content_hash": _hash(
            "source_content_hash",
            source_content_hash,
        ),
        "created_at": _timestamp("created_at", created_at),
        "expires_at": _timestamp("expires_at", expires_at),
        "hop_count": _hop_count(hop_count),
    }
    return sha256_hex(canonical_json_bytes(document))


__all__ = [
    "compute_capsule_hash",
    "compute_original_root_fingerprint",
]
