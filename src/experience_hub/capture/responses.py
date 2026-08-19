"""Canonical durable responses for trajectory capture commands."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from experience_hub import canonical_json_bytes
from experience_hub.storage.idempotency import StoredResponse


def capture_stored_response(
    *,
    bundle_id: UUID,
    owner_agent_id: UUID,
    manifest_hash: str,
    candidate_ids: Sequence[UUID],
    captured_at: datetime,
) -> StoredResponse:
    """Build the one replayable response for a completed capture command."""
    retained_candidate_ids = tuple(candidate_ids)
    return StoredResponse(
        status_code=201,
        body=canonical_json_bytes(
            {
                "data": {
                    "bundle_id": bundle_id,
                    "owner_agent_id": owner_agent_id,
                    "manifest_hash": manifest_hash,
                    "candidate_ids": retained_candidate_ids,
                    "candidate_count": len(retained_candidate_ids),
                    "captured_at": captured_at,
                }
            }
        ),
        headers={
            "location": (
                f"/v1/agents/{owner_agent_id}/trajectory-bundles/{bundle_id}"
            )
        },
    )


__all__ = ["capture_stored_response"]
