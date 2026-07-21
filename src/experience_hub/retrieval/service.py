"""Deterministic owner retrieval, blurred recall, and access planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandContext, PendingEvent
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import decode_version_content
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.events import (
    ExperienceAccessedV1,
    ExperienceReactivatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
)
from experience_hub.experiences.models import Temperature
from experience_hub.experiences.queries import ExperienceNotFoundError
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
    record_access,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    ExperienceView,
    PeekExperiences,
    RetrievalCandidate,
    RetrievalRecord,
    SearchExperiences,
    SearchHit,
    SearchResult,
)
from experience_hub.retrieval.ranking import (
    RankedCandidate,
    RankingCandidate,
    RetrievalMode,
    rank_candidates,
)
from experience_hub.retrieval.tokenizer import query_cues
from experience_hub.storage.unit_of_work import UnitOfWork

FOCUSED_COLD_EXPANSION_THRESHOLD = 0.72
ASSOCIATIVE_COLD_EXPANSION_THRESHOLD = 0.65


class _ExperienceQuery(Protocol):
    async def get_owned_retrieval_record(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> RetrievalRecord | None: ...

    async def select_retrieval_candidates(
        self,
        *,
        session: AsyncSession,
        selection: CandidateSelection,
    ) -> tuple[RetrievalCandidate, ...]: ...

    async def load_decoded_payloads(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        version_ids: Sequence[UUID],
    ) -> dict[UUID, bytes]: ...


class _MutationWriter(Protocol):
    async def apply_ordered_events(
        self,
        *,
        uow: UnitOfWork,
        experience_id: UUID,
        resulting_state: ExperienceStateSnapshotV1,
        events: Sequence[PendingEvent],
        command: CommandContext,
    ) -> ExperienceRecord: ...


@dataclass(frozen=True, slots=True)
class _AccessIntent:
    experience_id: UUID
    resulting_state: ExperienceStateSnapshotV1
    events: tuple[PendingEvent, ...]


@dataclass(frozen=True, slots=True)
class _RetrievalPlan:
    """Private fixed response and its ordered mutation intents."""

    result: SearchResult
    intents: tuple[_AccessIntent, ...]


SearchQuery = SearchExperiences | PeekExperiences


def _empty_query() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="empty_query",
        message="Search query produced no retrievable cues",
        status_code=422,
    )


def _clock_regression() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="clock_regression",
        message="Command time precedes experience state",
        status_code=409,
    )


def _require_caller_owner(
    command: CommandContext,
    owner_agent_id: UUID,
) -> None:
    if not isinstance(command, CommandContext):
        raise ValueError("command must be a CommandContext")
    if command.caller_scope != f"agent:{owner_agent_id}":
        raise ExperienceNotFoundError


def retrieval_query_hash(query: SearchQuery) -> str:
    """Hash only canonical cue identity, never transport or raw event text."""
    if not isinstance(query, (SearchExperiences, PeekExperiences)):
        raise ValueError("query must be a retrieval query")
    return sha256_hex(
        canonical_json_bytes(
            {
                "query": query.query,
                "tags": query.tags,
                "mechanism_cues": query.mechanism_cues,
            }
        )
    )


def _activation_inputs(record: RetrievalRecord) -> ActivationInputs:
    state = record.state
    return ActivationInputs(
        importance=state.importance,
        confidence=state.confidence,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        created_at=record.created_at,
    )


def _after_access(
    record: RetrievalRecord,
    *,
    at: datetime,
    lifecycle_config: LifecycleConfig,
) -> ExperienceStateSnapshotV1:
    occurred_at = require_utc(at)
    if occurred_at < record.latest_causal_at:
        raise _clock_regression()
    try:
        update = record_access(
            _activation_inputs(record),
            occurred_at,
            lifecycle_config,
        )
        updated_inputs = ActivationInputs(
            importance=record.state.importance,
            confidence=record.state.confidence,
            access_count=update.access_count,
            access_strength=update.access_strength,
            strength_updated_at=update.strength_updated_at,
            last_accessed_at=update.last_accessed_at,
            created_at=record.created_at,
        )
        activation = activation_at(
            updated_inputs,
            occurred_at,
            lifecycle_config,
        )
    except ValueError as error:
        raise _clock_regression() from error
    return record.state.model_copy(
        update={
            "access_count": update.access_count,
            "access_strength": update.access_strength,
            "strength_updated_at": update.strength_updated_at,
            "last_accessed_at": update.last_accessed_at,
            "activation_score": activation.score,
        }
    )


def _access_intent(
    *,
    record: RetrievalRecord,
    at: datetime,
    lifecycle_config: LifecycleConfig,
    query_hash: str | None = None,
    mode: RetrievalMode | None = None,
    signal: float | None = None,
) -> _AccessIntent:
    occurred_at = require_utc(at)
    after_access = _after_access(
        record,
        at=occurred_at,
        lifecycle_config=lifecycle_config,
    )
    events = [
        PendingEvent(
            aggregate_type="experience",
            aggregate_id=record.experience_id,
            event_type=ExperienceAccessedV1.event_type,
            payload=ExperienceAccessedV1(
                schema_version=1,
                experience_id=record.experience_id,
                version_id=record.current_version_id,
                before=record.state,
                after=after_access,
            ),
            actor_agent_id=record.owner_agent_id,
            occurred_at=occurred_at,
        )
    ]
    resulting_state = after_access
    if record.state.temperature is Temperature.COLD:
        if query_hash is None or mode is None or signal is None:
            raise ValueError("Cold access requires complete reactivation evidence")
        reactivated = ExperienceReactivatedV1(
            schema_version=1,
            experience_id=record.experience_id,
            query_hash=query_hash,
            mode=mode.value,
            signal=signal,
            before=after_access,
            after=after_access,
        )
        events.append(
            PendingEvent(
                aggregate_type="experience",
                aggregate_id=record.experience_id,
                event_type=ExperienceReactivatedV1.event_type,
                payload=reactivated,
                actor_agent_id=record.owner_agent_id,
                occurred_at=occurred_at,
            )
        )
        resulting_state = after_access.model_copy(
            update={
                "temperature": Temperature.WARM,
                "last_transition_at": occurred_at,
                "consecutive_below_threshold": 0,
            }
        )
        events.append(
            PendingEvent(
                aggregate_type="experience",
                aggregate_id=record.experience_id,
                event_type=ExperienceTemperatureChangedV1.event_type,
                payload=ExperienceTemperatureChangedV1(
                    schema_version=1,
                    experience_id=record.experience_id,
                    cause="cold_reactivation",
                    cycle_id=None,
                    before=after_access,
                    after=resulting_state,
                ),
                actor_agent_id=record.owner_agent_id,
                occurred_at=occurred_at,
            )
        )
    return _AccessIntent(
        experience_id=record.experience_id,
        resulting_state=resulting_state,
        events=tuple(events),
    )


def _cold_signal(
    ranked: RankedCandidate,
    *,
    mode: RetrievalMode,
) -> float | None:
    if mode is RetrievalMode.FOCUSED:
        signal = ranked.lexical_or_trigram_relevance
        return (
            signal
            if signal >= FOCUSED_COLD_EXPANSION_THRESHOLD
            else None
        )
    signal = ranked.mechanism_relevance
    return (
        signal
        if signal >= ASSOCIATIVE_COLD_EXPANSION_THRESHOLD
        else None
    )


def _utf8_prefix(value: str, maximum_bytes: int) -> str:
    if maximum_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def _view(
    record: RetrievalRecord,
    *,
    state: ExperienceStateSnapshotV1,
    body: str | None,
    body_is_excerpt: bool,
) -> ExperienceView:
    return ExperienceView(
        experience_id=record.experience_id,
        owner_agent_id=record.owner_agent_id,
        kind=record.kind,
        origin=record.origin,
        created_at=record.created_at,
        version_id=record.current_version_id,
        version_number=record.current_version_number,
        version_created_at=record.current_version_created_at,
        content_hash=record.current_content_hash,
        temperature=state.temperature,
        importance=state.importance,
        confidence=state.confidence,
        activation_score=state.activation_score,
        source_trust=state.source_trust,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        last_transition_at=state.last_transition_at,
        last_lifecycle_evaluated_at=state.last_lifecycle_evaluated_at,
        consecutive_below_threshold=state.consecutive_below_threshold,
        pinned=state.pinned,
        summary=record.summary,
        mechanism=record.mechanism,
        tags=record.tags,
        applicability=record.applicability,
        evidence=record.evidence,
        falsifiers=record.falsifiers,
        blurred=body is None,
        body=body,
        body_is_excerpt=body_is_excerpt,
    )


class _RetrievalPlanner:
    def __init__(
        self,
        *,
        query: _ExperienceQuery,
        lifecycle_config: LifecycleConfig,
    ) -> None:
        self._query = query
        self._lifecycle_config = lifecycle_config

    async def search(
        self,
        *,
        session: AsyncSession,
        query: SearchQuery,
        at: datetime,
        peek: bool,
    ) -> _RetrievalPlan:
        at = require_utc(at)
        cues = query_cues(
            query.query,
            tags=query.tags,
            mechanisms=query.mechanism_cues,
        )
        if not cues:
            raise _empty_query()
        selected = await self._query.select_retrieval_candidates(
            session=session,
            selection=CandidateSelection(
                owner_agent_id=query.owner_agent_id,
                query_cues=cues,
                mode=query.mode,
                requested_limit=query.limit,
            ),
        )
        by_id = {
            candidate.record.experience_id: candidate
            for candidate in selected
        }
        ranked = rank_candidates(
            (
                RankingCandidate(
                    experience_id=candidate.record.experience_id,
                    temperature=candidate.record.state.temperature,
                    current_version_created_at=(
                        candidate.record.current_version_created_at
                    ),
                    terms=candidate.terms,
                    activation_inputs=_activation_inputs(candidate.record),
                    source_trust=candidate.record.state.source_trust,
                )
                for candidate in selected
            ),
            query_cues=cues,
            mode=query.mode,
            at=at,
            lifecycle_config=self._lifecycle_config,
        )[: query.limit]

        potential: list[UUID] = []
        signals: dict[UUID, float] = {}
        for value in ranked:
            record = by_id[value.experience_id].record
            if record.state.temperature in {
                Temperature.HOT,
                Temperature.WARM,
            }:
                potential.append(record.current_version_id)
            elif (
                record.state.temperature is Temperature.COLD
                and query.expand_cold
                and (signal := _cold_signal(value, mode=query.mode))
                is not None
            ):
                potential.append(record.current_version_id)
                signals[record.experience_id] = signal
        decoded = (
            {}
            if not potential or query.content_budget_bytes == 0
            else await self._query.load_decoded_payloads(
                session=session,
                owner_agent_id=query.owner_agent_id,
                version_ids=tuple(potential),
            )
        )

        remaining = query.content_budget_bytes
        digest = retrieval_query_hash(query)
        hits: list[SearchHit] = []
        intents: list[_AccessIntent] = []
        for value in ranked:
            record = by_id[value.experience_id].record
            body: str | None = None
            body_is_excerpt = False
            decoded_payload = decoded.get(record.current_version_id)
            if decoded_payload is not None and remaining > 0:
                content = decode_version_content(
                    body_payload=decoded_payload,
                    summary=record.summary,
                    mechanism=record.mechanism,
                    tags=record.tags,
                    applicability=record.applicability,
                    evidence=record.evidence,
                    falsifiers=record.falsifiers,
                )
                if peek:
                    peek_query = query
                    if not isinstance(peek_query, PeekExperiences):
                        raise TypeError("Peek planning requires PeekExperiences")
                    allowed = min(
                        remaining,
                        peek_query.per_hit_excerpt_bytes,
                    )
                    excerpt = _utf8_prefix(content.body, allowed)
                    if excerpt:
                        body = excerpt
                        body_is_excerpt = True
                else:
                    body_bytes = len(content.body.encode("utf-8"))
                    if body_bytes <= remaining:
                        body = content.body
            state = record.state
            reactivated = False
            if body is not None:
                consumed = len(body.encode("utf-8"))
                remaining -= consumed
                if not peek:
                    intent = _access_intent(
                        record=record,
                        at=at,
                        lifecycle_config=self._lifecycle_config,
                        query_hash=(
                            digest
                            if record.state.temperature is Temperature.COLD
                            else None
                        ),
                        mode=(
                            query.mode
                            if record.state.temperature is Temperature.COLD
                            else None
                        ),
                        signal=signals.get(record.experience_id),
                    )
                    intents.append(intent)
                    state = intent.resulting_state
                    reactivated = (
                        record.state.temperature is Temperature.COLD
                    )
            hits.append(
                SearchHit(
                    experience=_view(
                        record,
                        state=state,
                        body=body,
                        body_is_excerpt=body_is_excerpt,
                    ),
                    score=value.score,
                    ranking_relevance=value.ranking_relevance,
                    lexical_or_trigram_relevance=(
                        value.lexical_or_trigram_relevance
                    ),
                    mechanism_relevance=value.mechanism_relevance,
                    activation=value.activation,
                    expanded=body is not None,
                    reactivated=reactivated,
                )
            )
        return _RetrievalPlan(
            result=SearchResult(
                hits=tuple(hits),
                remaining_content_budget_bytes=remaining,
            ),
            intents=tuple(intents),
        )


class RetrievalService:
    """Mutating search and direct GET inside one caller-owned unit of work."""

    def __init__(
        self,
        *,
        clock: Clock,
        query: _ExperienceQuery,
        mutation_writer: _MutationWriter,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._clock = clock
        self._query = query
        self._mutation_writer = mutation_writer
        self._lifecycle_config = lifecycle_config or LifecycleConfig()
        self._planner = _RetrievalPlanner(
            query=query,
            lifecycle_config=self._lifecycle_config,
        )

    async def search(
        self,
        *,
        uow: UnitOfWork,
        query: SearchExperiences,
        command: CommandContext,
    ) -> SearchResult:
        if not isinstance(query, SearchExperiences):
            raise ValueError("query must be SearchExperiences")
        _require_caller_owner(command, query.owner_agent_id)
        if not uow.immediate:
            raise RuntimeError("Retrieval search requires an immediate unit of work")
        plan = await self._planner.search(
            session=uow.session,
            query=query,
            at=self._clock.now(),
            peek=False,
        )
        for intent in plan.intents:
            await self._mutation_writer.apply_ordered_events(
                uow=uow,
                experience_id=intent.experience_id,
                resulting_state=intent.resulting_state,
                events=intent.events,
                command=command,
            )
        return plan.result

    async def get(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        experience_id: UUID,
        command: CommandContext,
    ) -> ExperienceView:
        _require_caller_owner(command, owner_agent_id)
        if not uow.immediate:
            raise RuntimeError("Retrieval GET requires an immediate unit of work")
        record = await self._query.get_owned_retrieval_record(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )
        if record is None:
            raise ExperienceNotFoundError
        if record.state.temperature in {
            Temperature.COLD,
            Temperature.ARCHIVED,
        }:
            return _view(
                record,
                state=record.state,
                body=None,
                body_is_excerpt=False,
            )
        decoded = await self._query.load_decoded_payloads(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            version_ids=(record.current_version_id,),
        )
        content = decode_version_content(
            body_payload=decoded[record.current_version_id],
            summary=record.summary,
            mechanism=record.mechanism,
            tags=record.tags,
            applicability=record.applicability,
            evidence=record.evidence,
            falsifiers=record.falsifiers,
        )
        intent = _access_intent(
            record=record,
            at=self._clock.now(),
            lifecycle_config=self._lifecycle_config,
        )
        response = _view(
            record,
            state=intent.resulting_state,
            body=content.body,
            body_is_excerpt=False,
        )
        await self._mutation_writer.apply_ordered_events(
            uow=uow,
            experience_id=intent.experience_id,
            resulting_state=intent.resulting_state,
            events=intent.events,
            command=command,
        )
        return response


class ExperienceEvidenceReader:
    """Session-bound read-only retrieval for frozen inspiration evidence."""

    def __init__(
        self,
        *,
        clock: Clock,
        query: _ExperienceQuery,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._clock = clock
        self._planner = _RetrievalPlanner(
            query=query,
            lifecycle_config=lifecycle_config or LifecycleConfig(),
        )

    async def peek(
        self,
        *,
        session: AsyncSession,
        query: PeekExperiences,
    ) -> SearchResult:
        if not isinstance(query, PeekExperiences):
            raise ValueError("query must be PeekExperiences")
        plan = await self._planner.search(
            session=session,
            query=query,
            at=self._clock.now(),
            peek=True,
        )
        if plan.intents:
            raise RuntimeError("Read-only retrieval planned a mutation")
        return plan.result


__all__ = [
    "ASSOCIATIVE_COLD_EXPANSION_THRESHOLD",
    "FOCUSED_COLD_EXPANSION_THRESHOLD",
    "ExperienceEvidenceReader",
    "RetrievalService",
    "retrieval_query_hash",
]
