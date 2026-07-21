"""Canonical HTTP-equivalent request identities for idea commands."""

from __future__ import annotations

from experience_hub.domain import CommandRequest, StructuredReason
from experience_hub.inspiration.commands import (
    AdoptIdea,
    ArchiveIdea,
    RejectIdea,
)
from experience_hub.inspiration.models import IdeaEvaluation


def evaluation_command_request(
    evaluation: IdeaEvaluation,
    *,
    idempotency_key: str,
) -> CommandRequest:
    """Reconstruct the one canonical evaluation request."""
    return CommandRequest(
        caller_scope=f"agent:{evaluation.evaluator_agent_id}",
        operation_scope="inspiration.idea.evaluate",
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/ideas/{idea_id}:evaluate",
        path_parameters={
            "agent_id": evaluation.evaluator_agent_id,
            "idea_id": evaluation.idea_id,
        },
        body={
            "evaluated_at": evaluation.evaluated_at,
            "evidence": tuple(
                reference.model_dump(mode="json") for reference in evaluation.evidence
            ),
            "reason": (
                None
                if evaluation.reason is None
                else evaluation.reason.model_dump(mode="json")
            ),
            "verdict": evaluation.verdict.value,
        },
    )


def decision_command_request(
    command: RejectIdea | ArchiveIdea,
    *,
    idempotency_key: str,
) -> CommandRequest:
    """Reconstruct the one canonical reject or archive request."""
    action = "reject" if isinstance(command, RejectIdea) else "archive"
    normalized = (
        StructuredReason.from_user_text(command.reason)
        if isinstance(command.reason, str)
        else command.reason
    )
    return CommandRequest(
        caller_scope=f"agent:{command.owner_agent_id}",
        operation_scope=f"inspiration.idea.{action}",
        idempotency_key=idempotency_key,
        method="POST",
        route_template=(f"/v1/agents/{{agent_id}}/ideas/{{idea_id}}:{action}"),
        path_parameters={
            "agent_id": command.owner_agent_id,
            "idea_id": command.idea_id,
        },
        body={"reason": normalized.model_dump(mode="json")},
    )


def adoption_command_request(
    command: AdoptIdea,
    *,
    idempotency_key: str,
) -> CommandRequest:
    """Reconstruct the one canonical idea-adoption request."""
    return CommandRequest(
        caller_scope=f"agent:{command.owner_agent_id}",
        operation_scope="inspiration.idea.adopt",
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/agents/{agent_id}/ideas/{idea_id}:adopt",
        path_parameters={
            "agent_id": command.owner_agent_id,
            "idea_id": command.idea_id,
        },
        body={
            "confidence": command.confidence,
            "importance": command.importance,
        },
    )


__all__ = [
    "adoption_command_request",
    "decision_command_request",
    "evaluation_command_request",
]
