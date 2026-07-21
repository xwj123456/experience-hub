from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from tests.integration.test_capsule_adoption import (
    ADOPTER_ID,
    CAPSULE_ID,
    ITEM_ID,
    OTHER_AGENT_ID,
    UNKNOWN_ITEM_ID,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
    error_code,
    request,
    stored_event,
)

from experience_hub.domain import CommandContext, StructuredReason
from experience_hub.sharing.events import CapsuleRejectedV1
from experience_hub.sharing.models import InboxState, RejectInboxItem
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.tables import (
    AdoptionRecordRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

REJECTION_REASON = StructuredReason.from_user_text(
    "  当前证据不足，保留胶囊但不纳入本地经验。  "
)
SECOND_REJECTION_REASON = StructuredReason.from_user_text(
    "A later command must not replace the first retained reason."
)


@pytest.fixture(name="stack")
async def _stack(
    repository_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AdoptionStack]:
    value = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "capsule-rejection.sqlite3",
    )
    try:
        yield value
    finally:
        await value.database.dispose()


async def reject(
    stack: AdoptionStack,
    *,
    key: str,
    recipient_agent_id: UUID = ADOPTER_ID,
    item_id: UUID = ITEM_ID,
    reason: StructuredReason = REJECTION_REASON,
) -> CommandResult:
    command = RejectInboxItem(
        recipient_agent_id=recipient_agent_id,
        item_id=item_id,
        reason=reason,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await stack.sharing_service.reject_inbox_item(
            uow=uow,
            command=command,
            command_context=context,
        )

    return await stack.executor.execute(
        request(
            key=key,
            operation_scope="capsule.reject",
            route_template="/v1/agents/{agent_id}/inbox/{item_id}:reject",
            agent_id=recipient_agent_id,
            path_parameters={
                "agent_id": recipient_agent_id,
                "item_id": item_id,
            },
            body={"reason": reason.model_dump(mode="json")},
        ),
        handler,
    )


def _rejection_payload(
    stack: AdoptionStack,
    row: DomainEventRow,
) -> CapsuleRejectedV1:
    payload = stored_event(stack, row).payload
    assert isinstance(payload, CapsuleRejectedV1)
    return payload


def test_reject_command_requires_a_structured_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        RejectInboxItem(
            recipient_agent_id=ADOPTER_ID,
            item_id=ITEM_ID,
            reason=None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_pending_item_rejection_retains_reason_and_emits_one_transition_event(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)

    result = await reject(stack, key="reject-pending")

    assert result.status_code == 200
    assert not result.replayed
    response = json.loads(result.body)
    assert set(response) == {"data"}
    assert response["data"]["item_id"] == str(ITEM_ID)
    assert response["data"]["recipient_agent_id"] == str(ADOPTER_ID)
    assert response["data"]["capsule_id"] == str(CAPSULE_ID)
    assert response["data"]["state"] == InboxState.REJECTED
    assert set(response["data"]) == {
        "item_id",
        "recipient_agent_id",
        "capsule_id",
        "capsule",
        "state",
        "effective_availability",
    }
    assert response["data"]["capsule"]["capsule_id"] == str(CAPSULE_ID)
    assert response["data"]["capsule"]["status"] == "active"
    assert response["data"]["effective_availability"] == "active"

    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        capsule = await session.get(ExperienceCapsuleRow, CAPSULE_ID)
        rejection_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_type == "inbox_item",
                        DomainEventRow.aggregate_id == ITEM_ID,
                        DomainEventRow.event_type == CapsuleRejectedV1.event_type,
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        adoption_count = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
        adopter_experience_count = await session.scalar(
            select(func.count())
            .select_from(ExperienceRow)
            .where(ExperienceRow.owner_agent_id == ADOPTER_ID)
        )

    assert item is not None
    assert capsule is not None
    assert item.state is InboxState.REJECTED
    assert item.capsule_id == capsule.capsule_id == CAPSULE_ID
    assert len(rejection_rows) == 1
    rejection_row = rejection_rows[0]
    assert (
        rejection_row.sequence,
        rejection_row.actor_agent_id,
        rejection_row.occurred_at,
    ) == (2, ADOPTER_ID, stack.clock.now())
    payload = _rejection_payload(stack, rejection_row)
    assert (
        payload.item_id,
        payload.capsule_id,
        payload.recipient_agent_id,
        payload.reason,
        payload.state_before,
        payload.state_after,
    ) == (
        ITEM_ID,
        CAPSULE_ID,
        ADOPTER_ID,
        REJECTION_REASON,
        InboxState.PENDING,
        InboxState.REJECTED,
    )
    assert payload.reason.text == "当前证据不足，保留胶囊但不纳入本地经验。"
    assert adoption_count == 0
    assert adopter_experience_count == 0


@pytest.mark.asyncio
async def test_rejection_reports_expired_capsule_availability(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(days=7))

    result = await reject(stack, key="reject-expired")

    assert result.status_code == 200
    response = json.loads(result.body)["data"]
    assert response["state"] == InboxState.REJECTED
    assert response["capsule"]["status"] == "active"
    assert response["effective_availability"] == "expired"


@pytest.mark.asyncio
async def test_rejection_hides_foreign_and_missing_items_with_identical_404(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)

    foreign = await reject(
        stack,
        key="reject-foreign",
        recipient_agent_id=OTHER_AGENT_ID,
    )
    missing = await reject(
        stack,
        key="reject-missing",
        item_id=UNKNOWN_ITEM_ID,
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.body == missing.body
    assert error_code(foreign) == "resource_not_found"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        rejection_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == CapsuleRejectedV1.event_type)
        )
    assert item is not None and item.state is InboxState.PENDING
    assert rejection_count == 0


@pytest.mark.asyncio
async def test_rejection_rejects_clock_regression_without_mutation(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    stack.clock.advance(timedelta(microseconds=-1))

    result = await reject(stack, key="reject-clock-regression")

    assert result.status_code == 409
    assert error_code(result) == "clock_regression"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        rejection_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == CapsuleRejectedV1.event_type)
        )
    assert item is not None and item.state is InboxState.PENDING
    assert rejection_count == 0


@pytest.mark.asyncio
async def test_rejection_replays_same_receipt_but_different_receipt_cannot_rewrite_it(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    first = await reject(stack, key="reject-once")
    same_receipt = await reject(stack, key="reject-once")
    different_receipt = await reject(
        stack,
        key="reject-again",
        reason=SECOND_REJECTION_REASON,
    )

    assert first.status_code == same_receipt.status_code == 200
    assert not first.replayed
    assert same_receipt.replayed
    assert (
        same_receipt.status_code,
        same_receipt.body,
        same_receipt.headers,
    ) == (first.status_code, first.body, first.headers)
    assert different_receipt.status_code == 409
    assert error_code(different_receipt) == "inbox_item_not_pending"

    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow).where(
                        DomainEventRow.event_type == CapsuleRejectedV1.event_type
                    )
                )
            ).all()
        )
    assert item is not None and item.state is InboxState.REJECTED
    assert len(rows) == 1
    assert _rejection_payload(stack, rows[0]).reason == REJECTION_REASON


@pytest.mark.asyncio
async def test_rejected_item_is_a_stable_terminal_state_for_adoption(
    stack: AdoptionStack,
) -> None:
    await arrange_pending_capsule(stack)
    assert (await reject(stack, key="reject-before-adopt")).status_code == 200

    result = await adopt(stack, key="adopt-after-reject")

    assert result.status_code == 409
    assert error_code(result) == "inbox_item_not_pending"
    async with stack.database.read_session() as session:
        item = await session.get(InboxItemRow, ITEM_ID)
        adoption_count = await session.scalar(
            select(func.count()).select_from(AdoptionRecordRow)
        )
        adopted_event_count = await session.scalar(
            select(func.count())
            .select_from(DomainEventRow)
            .where(DomainEventRow.event_type == "capsule.adopted")
        )
    assert item is not None and item.state is InboxState.REJECTED
    assert adoption_count == adopted_event_count == 0
