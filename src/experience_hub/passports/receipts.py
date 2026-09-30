"""Shared authentication of retained Passport command results."""

from collections.abc import Sequence
from uuid import UUID

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandRequest, EventRegistry
from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.tables import DomainEventRow, IdempotencyRecordRow


def require_receipt_anchor(
    *, receipt: IdempotencyRecordRow | None, event: DomainEventRow, owner: UUID
) -> IdempotencyRecordRow:
    if (
        receipt is None
        or receipt.receipt_id != event.causation_id
        or receipt.state != "completed"
        or receipt.caller_scope != f"agent:{owner}"
        or receipt.created_at != event.occurred_at
        or receipt.completed_at is None
        or receipt.completed_at < event.occurred_at
    ):
        raise ValueError("Invalid Passport receipt")
    return receipt


def require_receipt_result(
    *,
    receipt: IdempotencyRecordRow,
    request: CommandRequest,
    resource_type: str,
    resource_id: UUID,
    response: StoredResponse,
) -> None:
    if (
        receipt.state != "completed"
        or receipt.caller_scope != request.caller_scope
        or receipt.idempotency_key != request.idempotency_key
        or receipt.scope != request.operation_scope
        or receipt.request_hash != request.request_hash
        or receipt.result_resource_type != resource_type
        or receipt.result_resource_id != resource_id
        or receipt.response_status_code != response.status_code
        or receipt.response_body != response.body
        or receipt.response_content_type != response.content_type
        or receipt.response_headers
        != canonical_json_bytes(dict(response.headers or {}))
    ):
        raise ValueError("Invalid Passport receipt")


def prior_experience_snapshot(
    *,
    registry: EventRegistry,
    rows: Sequence[DomainEventRow],
    experience_id: UUID,
    before_event_id: int,
) -> ExperienceStateSnapshotV1 | None:
    """Reconstruct historical response state, never substitute current state."""
    result: ExperienceStateSnapshotV1 | None = None
    for row in rows:
        if row.event_id >= before_event_id:
            break
        if row.aggregate_type != "experience" or row.aggregate_id != experience_id:
            continue
        payload = registry.decode(event_type=row.event_type, payload=row.payload)
        after = getattr(payload, "after", None)
        if isinstance(after, ExperienceStateSnapshotV1):
            result = after
    return result
