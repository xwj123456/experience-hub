"""Pure, replayable mapping from an idea source to a hypothesis."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import VersionContent
from experience_hub.inspiration.models import SnapshotEvidenceReference


class IdeaContentSource(Protocol):
    """Authoritative idea fields required by the adoption mapping."""

    title: str
    hypothesis: str
    mechanism: str
    operator: str
    predictions: bytes
    falsifiers: bytes
    assumptions: bytes
    proposed_test: str


def canonical_string_tuple(raw: bytes, *, label: str) -> tuple[str, ...]:
    """Decode one canonical JSON string array."""
    encoded = bytes(raw)
    try:
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if (
        not isinstance(decoded, list)
        or any(not isinstance(value, str) for value in decoded)
        or canonical_json_bytes(decoded) != encoded
    ):
        raise ValueError(f"{label} is not a canonical string array")
    return tuple(decoded)


def decode_idea_evidence(
    raw: bytes,
) -> tuple[SnapshotEvidenceReference, ...]:
    """Decode canonical, unique frozen-snapshot evidence references."""
    encoded = bytes(raw)
    try:
        decoded = json.loads(encoded)
        if (
            not isinstance(decoded, list)
            or not decoded
            or canonical_json_bytes(decoded) != encoded
        ):
            raise ValueError("idea evidence is not canonical")
        retained = tuple(
            SnapshotEvidenceReference.model_validate_json(canonical_json_bytes(value))
            for value in decoded
        )
    except (
        TypeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("idea evidence source is invalid") from error
    identities = tuple((value.id, value.stable_evidence_key) for value in retained)
    if (
        len(identities) != len(set(identities))
        or len({value.id for value in retained}) != len(retained)
        or len({value.stable_evidence_key for value in retained}) != len(retained)
    ):
        raise ValueError("idea evidence repeats a source")
    return retained


def adopted_hypothesis_content(
    *,
    idea: IdeaContentSource,
    evidence: tuple[SnapshotEvidenceReference, ...],
) -> VersionContent:
    """Apply the one approved idea-to-hypothesis content mapping."""
    assumptions = canonical_string_tuple(
        idea.assumptions,
        label="assumptions",
    )
    return VersionContent(
        body=canonical_json_bytes(
            {
                "assumptions": assumptions,
                "hypothesis": idea.hypothesis,
                "predictions": canonical_string_tuple(
                    idea.predictions,
                    label="predictions",
                ),
                "proposed_test": idea.proposed_test,
            }
        ).decode("utf-8"),
        summary=idea.title,
        mechanism=idea.mechanism,
        tags=("inspiration", f"operator:{idea.operator}"),
        applicability=assumptions,
        evidence=tuple(
            TypedEvidence(
                type="inspiration_evidence",
                id=reference.stable_evidence_key,
            )
            for reference in evidence
        ),
        falsifiers=canonical_string_tuple(
            idea.falsifiers,
            label="falsifiers",
        ),
    )


__all__ = [
    "IdeaContentSource",
    "adopted_hypothesis_content",
    "canonical_string_tuple",
    "decode_idea_evidence",
]
