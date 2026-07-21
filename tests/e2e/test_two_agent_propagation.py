from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select
from tests.integration.test_capsule_adoption import (
    ADOPTED_EXPERIENCE_ID,
    ADOPTER_ID,
    CAPSULE_ID,
    ITEM_ID,
    PUBLISHER_ID,
    SOURCE_CONTENT,
    AdoptionStack,
    adopt,
    arrange_pending_capsule,
    build_stack,
)
from tests.integration.test_capsule_feedback import record_feedback
from tests.integration.test_experience_search import search_request

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import CommandContext
from experience_hub.experiences.projector import ExperienceTermsProjector
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import RetrievalService
from experience_hub.retrieval.tokenizer import query_cues
from experience_hub.sharing.models import FeedbackVerdict, InboxState
from experience_hub.sharing.projector import AgentReputationProjector
from experience_hub.sharing.queries import (
    InboxEvidenceReader,
    QuarantinedCapsuleEvidence,
)
from experience_hub.sharing.repository import SharingRepository
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.tables import (
    AgentReputationRow,
    DomainEventRow,
    InboxItemRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork


async def _search(
    stack: AdoptionStack,
    service: RetrievalService,
    *,
    key: str,
) -> dict[str, Any]:
    query = SearchExperiences(
        owner_agent_id=ADOPTER_ID,
        query=SOURCE_CONTENT.summary,
        mode=RetrievalMode.FOCUSED,
    )

    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        result = await service.search(
            uow=uow,
            query=query,
            command=command,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": result}),
        )

    result = await stack.executor.execute(
        search_request(query, key=key),
        handler,
    )
    assert result.status_code == 200
    response = cast(dict[str, Any], json.loads(result.body))
    return cast(dict[str, Any], response["data"])


async def _pending_evidence(
    stack: AdoptionStack,
) -> tuple[QuarantinedCapsuleEvidence, ...]:
    reader = InboxEvidenceReader(
        repository=SharingRepository(event_registry=stack.registry)
    )
    async with stack.database.read_session() as session:
        return await reader.list_available_pending(
            session=session,
            recipient_agent_id=ADOPTER_ID,
            as_of=stack.clock.now(),
            query_cues=query_cues(SOURCE_CONTENT.summary),
            mode=RetrievalMode.FOCUSED,
            limit=12,
        )


@pytest.mark.asyncio
async def test_subscription_quarantines_until_explicit_adoption_then_learns_trust(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    stack = await build_stack(
        repository_root=repository_root,
        database_path=tmp_path / "two-agent-propagation.sqlite3",
    )
    manager = cast(
        ProjectionManager,
        stack.database._projection_applier,  # noqa: SLF001
    )
    manager.registry.register(ExperienceTermsProjector(stack.registry))
    manager.registry.register(AgentReputationProjector(stack.registry))
    retrieval = RetrievalService(
        clock=stack.clock,
        query=ExperienceQuery(event_registry=stack.registry),
        mutation_writer=ExperienceMutationWriter(
            repository=stack.experience_repository
        ),
    )

    try:
        await arrange_pending_capsule(stack)

        # Subscription delivery is quarantined: normal memory cannot see it,
        # while the explicit inspiration surface can expose bounded evidence.
        before_adoption = await _search(
            stack,
            retrieval,
            key="search-before-explicit-adoption",
        )
        assert before_adoption["hits"] == []
        pending = await _pending_evidence(stack)
        assert len(pending) == 1
        assert (
            pending[0].item_id,
            pending[0].capsule_id,
            pending[0].source_state,
            pending[0].excerpt,
        ) == (
            ITEM_ID,
            CAPSULE_ID,
            "quarantined",
            SOURCE_CONTENT.body,
        )
        assert pending[0].source_trust == pytest.approx(0.25)
        async with stack.database.read_session() as session:
            item = await session.get(InboxItemRow, ITEM_ID)
        assert item is not None and item.state is InboxState.PENDING

        adoption = await adopt(stack, key="explicit-two-agent-adoption")
        assert adoption.status_code == 200
        adoption_data = cast(
            dict[str, Any],
            cast(dict[str, Any], json.loads(adoption.body))["data"],
        )
        adopted_experience = cast(dict[str, Any], adoption_data["experience"])
        assert adoption_data["created"] is True
        assert adoption_data["corroboration_applied"] is False
        assert UUID(adopted_experience["experience_id"]) == ADOPTED_EXPERIENCE_ID
        assert await _pending_evidence(stack) == ()

        recalled = await _search(
            stack,
            retrieval,
            key="search-after-explicit-adoption",
        )
        hits = cast(list[dict[str, Any]], recalled["hits"])
        assert len(hits) == 1
        recalled_experience = cast(dict[str, Any], hits[0]["experience"])
        assert UUID(recalled_experience["experience_id"]) == ADOPTED_EXPERIENCE_ID
        assert recalled_experience["owner_agent_id"] == str(ADOPTER_ID)
        assert recalled_experience["body"] == SOURCE_CONTENT.body

        feedback = await record_feedback(
            stack,
            key="useful-two-agent-feedback",
            verdict=FeedbackVerdict.USEFUL,
        )
        assert feedback.status_code == 201
        repository = SharingRepository(event_registry=stack.registry)
        async with stack.database.read_session() as session:
            reputation = await session.get(
                AgentReputationRow,
                (PUBLISHER_ID, ADOPTER_ID),
            )
            learned_trust = await repository.strict_observer_trust(
                session=session,
                subject_agent_id=PUBLISHER_ID,
                observer_agent_id=ADOPTER_ID,
            )
            propagation_events = tuple(
                (
                    await session.scalars(
                        select(DomainEventRow.event_type)
                        .where(
                            DomainEventRow.event_type.in_(
                                (
                                    "subscription.created",
                                    "capsule.published",
                                    "capsule.received",
                                    "capsule.adopted",
                                    "capsule.feedback_recorded",
                                )
                            )
                        )
                        .order_by(DomainEventRow.event_id)
                    )
                ).all()
            )

        assert reputation is not None
        assert (
            reputation.useful_count,
            reputation.refuted_count,
            reputation.harmful_count,
            reputation.alpha,
            reputation.beta,
        ) == (1, 0, 0, 3, 2)
        assert learned_trust == pytest.approx(0.60)
        assert propagation_events == (
            "subscription.created",
            "capsule.published",
            "capsule.received",
            "capsule.adopted",
            "capsule.feedback_recorded",
        )
    finally:
        await stack.database.dispose()
