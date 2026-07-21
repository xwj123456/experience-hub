"""Stable canonical hashes for frozen evidence and inspiration ideas."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.inspiration.models import (
    EvidenceSourceType,
    IdeaDraft,
    SnapshotItem,
)
from experience_hub.retrieval.tokenizer import normalize_text

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
NEAR_DUPLICATE_THRESHOLD = 0.82


def truncate_utf8(value: str, limit: int) -> str:
    """Retain at most ``limit`` UTF-8 bytes without splitting a code point."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("value must contain valid Unicode") from error
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _uuid(name: str, value: Any) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{name} must be a UUID")
    return value


def _hash(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def stable_evidence_key(
    *,
    source_type: EvidenceSourceType,
    source_id: UUID,
    source_version_id: UUID,
    content_hash: str,
) -> str:
    """Hash source identity and semantic content, independent of a run."""
    if not isinstance(source_type, EvidenceSourceType):
        raise TypeError("source_type must be an EvidenceSourceType")
    document = {
        "source_type": source_type,
        "source_id": _uuid("source_id", source_id),
        "source_version_id": _uuid("source_version_id", source_version_id),
        "content_hash": _hash("content_hash", content_hash),
    }
    return sha256_hex(canonical_json_bytes(document))


def _snapshot_document(item: SnapshotItem) -> dict[str, Any]:
    if not isinstance(item, SnapshotItem):
        raise TypeError("snapshot items must be SnapshotItem values")
    return {
        "stable_evidence_key": item.stable_evidence_key,
        "source_type": item.source_type,
        "source_id": item.source_id,
        "source_version_id": item.source_version_id,
        "content_hash": item.content_hash,
        "rank": item.rank,
        "source_state": item.source_state,
        "summary": item.summary,
        "mechanism": item.mechanism,
        "applicability": item.applicability,
        "tags": item.tags,
        "falsifiers": item.falsifiers,
        "excerpt": item.excerpt,
    }


def snapshot_canonical_bytes(items: tuple[SnapshotItem, ...]) -> bytes:
    """Encode the exact bounded document shared by size checks and hashing."""
    if not isinstance(items, tuple):
        raise TypeError("items must be an immutable tuple")
    return canonical_json_bytes(
        {"items": [_snapshot_document(item) for item in items]}
    )


def hash_snapshot(items: tuple[SnapshotItem, ...]) -> str:
    """Hash ordered frozen semantics, excluding run-local identity and clocks."""
    return sha256_hex(snapshot_canonical_bytes(items))


def normalize_mechanism(value: str) -> str:
    """Normalize mechanism text for boundary-aware clustering."""
    if not isinstance(value, str):
        raise TypeError("mechanism must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("mechanism must contain valid Unicode") from error
    return normalize_text(value)


def hash_mechanism(value: str) -> str:
    """Hash the normalized mechanism used as an exact cluster identity."""
    normalized = normalize_mechanism(value)
    if not normalized:
        raise ValueError("mechanism must contain normalized semantic text")
    return sha256_hex(normalized.encode("utf-8"))


def _padded_character_trigrams(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    padded = f"  {value}  "
    return frozenset(
        padded[index : index + 3] for index in range(len(padded) - 2)
    )


def mechanism_similarity(left: str, right: str) -> float:
    """Return padded-character-trigram Jaccard similarity."""
    left_terms = _padded_character_trigrams(normalize_mechanism(left))
    right_terms = _padded_character_trigrams(normalize_mechanism(right))
    if not left_terms and not right_terms:
        return 1.0
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _sorted_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    unique = {canonical_json_bytes(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def hash_idea_content(idea: IdeaDraft) -> str:
    """Hash semantic idea content while excluding run-local identities."""
    if not isinstance(idea, IdeaDraft):
        raise TypeError("idea must be an IdeaDraft")
    document = {
        "schema_version": 1,
        "title": idea.title,
        "hypothesis": idea.hypothesis,
        "mechanism": idea.mechanism,
        "predictions": _sorted_unique_strings(idea.predictions),
        "falsifiers": _sorted_unique_strings(idea.falsifiers),
        "assumptions": _sorted_unique_strings(idea.assumptions),
        "proposed_test": idea.proposed_test,
        "evidence_stable_keys": _sorted_unique_strings(
            reference.stable_evidence_key for reference in idea.evidence
        ),
    }
    return sha256_hex(canonical_json_bytes(document))


__all__ = [
    "NEAR_DUPLICATE_THRESHOLD",
    "hash_idea_content",
    "hash_mechanism",
    "hash_snapshot",
    "mechanism_similarity",
    "normalize_mechanism",
    "snapshot_canonical_bytes",
    "stable_evidence_key",
    "truncate_utf8",
]
