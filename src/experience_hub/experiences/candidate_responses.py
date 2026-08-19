"""Canonical reconstruction and durable responses for candidate commands."""

import json

from experience_hub import canonical_json_bytes
from experience_hub.domain import TypedEvidence
from experience_hub.experiences.candidate_models import CandidateViewV1
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.models import VersionContent
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import ExperienceCandidateRow


def _canonical_array(raw: bytes) -> list[object]:
    value: object = json.loads(raw)
    if not isinstance(value, list) or canonical_json_bytes(value) != bytes(raw):
        raise ValueError("not a canonical array")
    return value


def reconstruct_candidate_content(row: ExperienceCandidateRow) -> VersionContent:
    """Rebuild and authenticate the immutable candidate content document."""
    tags = _canonical_array(row.tags)
    applicability = _canonical_array(row.applicability)
    evidence_values = _canonical_array(row.evidence)
    falsifiers = _canonical_array(row.falsifiers)
    string_values = (*tags, *applicability, *falsifiers)
    if not all(isinstance(value, str) for value in string_values):
        raise ValueError("candidate string array is invalid")
    evidence = tuple(TypedEvidence.model_validate(value) for value in evidence_values)
    content = VersionContent(
        body=row.body,
        summary=row.summary,
        mechanism=row.mechanism,
        tags=tuple(value for value in tags if isinstance(value, str)),
        applicability=tuple(
            value for value in applicability if isinstance(value, str)
        ),
        evidence=evidence,
        falsifiers=tuple(value for value in falsifiers if isinstance(value, str)),
    )
    if (
        canonical_json_bytes(content.tags) != row.tags
        or canonical_json_bytes(content.applicability) != row.applicability
        or canonical_json_bytes(content.evidence) != row.evidence
        or canonical_json_bytes(content.falsifiers) != row.falsifiers
        or encode_version_content(kind=row.kind, content=content).content_hash
        != row.content_hash
    ):
        raise ValueError("candidate content is not canonical")
    return content


def candidate_stored_response(view: CandidateViewV1) -> StoredResponse:
    """Build the one replayable response for a completed candidate decision."""
    return StoredResponse(
        status_code=200,
        body=canonical_json_bytes({"data": view.model_dump(mode="json")}),
    )


__all__ = ["candidate_stored_response", "reconstruct_candidate_content"]
