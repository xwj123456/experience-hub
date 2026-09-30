"""Canonical virtual command semantics shared by CLI and source validation."""

from uuid import UUID

from experience_hub.domain import CommandRequest, StructuredReason
from experience_hub.passports.scopes import (
    PASSPORT_ADOPT_SCOPE,
    PASSPORT_IMPORT_SCOPE,
    PASSPORT_REJECT_SCOPE,
)


def passport_import_request(
    *, owner_agent_id: UUID, passport_hash: str, idempotency_key: str
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=PASSPORT_IMPORT_SCOPE,
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/passports",
        path_parameters={"agent_id": owner_agent_id},
        body={"passport_hash": passport_hash},
    )


def passport_adopt_request(
    *,
    owner_agent_id: UUID,
    import_id: UUID,
    importance: float,
    confidence: float,
    idempotency_key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=PASSPORT_ADOPT_SCOPE,
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/passports/{import_id}/adopt",
        path_parameters={"agent_id": owner_agent_id, "import_id": import_id},
        body={"importance": importance, "confidence": confidence},
    )


def passport_reject_request(
    *,
    owner_agent_id: UUID,
    import_id: UUID,
    reason: StructuredReason,
    idempotency_key: str,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope=PASSPORT_REJECT_SCOPE,
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/passports/{import_id}/reject",
        path_parameters={"agent_id": owner_agent_id, "import_id": import_id},
        body={"reason": reason},
    )
