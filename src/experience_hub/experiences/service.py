"""Experience creation, retrieval adapters, and explicit state mutations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.contracts import (
    ConfirmExperience,
    CreateExperience,
    CreateExperienceVersion,
    ExperienceCreation,
    ExperienceDraft,
    ExperienceMutationReason,
    ExperienceRecord,
    PinExperience,
    RefuteExperience,
    RestoreExperience,
    UnpinExperience,
)
from experience_hub.experiences.events import (
    ExperienceConfirmedV1,
    ExperienceCorroboratedV1,
    ExperiencePinnedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
)
from experience_hub.experiences.models import ExperienceOrigin, Temperature
from experience_hub.experiences.queries import (
    ExperienceNotFoundError,
    ExperienceQuery,
)
from experience_hub.experiences.repository import ExperienceWriter
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import IdGenerator
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.retrieval.contracts import (
    ExperienceView,
    RetrievalRecord,
    SearchExperiences,
)
from experience_hub.retrieval.service import RetrievalService
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandResult,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import AgentRow
from experience_hub.storage.unit_of_work import UnitOfWork


class ExperienceRetrievalAdapter:
    """Turn optional-key retrieval calls into ordinary durable commands."""

    def __init__(
        self,
        *,
        executor: CommandExecutor,
        retrieval_service: RetrievalService,
        id_generator: IdGenerator,
    ) -> None:
        self._executor = executor
        self._retrieval_service = retrieval_service
        self._id_generator = id_generator

    async def search(
        self,
        *,
        query: SearchExperiences,
        idempotency_key: str | None = None,
    ) -> CommandResult:
        if not isinstance(query, SearchExperiences):
            raise ValueError("query must be SearchExperiences")
        request = CommandRequest(
            caller_scope=f"agent:{query.owner_agent_id}",
            operation_scope="experience.search",
            idempotency_key=self._key(idempotency_key),
            method="POST",
            route_template="/v1/agents/{agent_id}/experiences:search",
            path_parameters={"agent_id": query.owner_agent_id},
            body={
                "query": query.query,
                "mode": query.mode,
                "tags": query.tags,
                "mechanism_cues": query.mechanism_cues,
                "limit": query.limit,
                "content_budget_bytes": query.content_budget_bytes,
                "expand_cold": query.expand_cold,
            },
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            if await uow.session.get(AgentRow, query.owner_agent_id) is None:
                raise ReplayableCommandError(
                    code="agent_not_found",
                    message="Agent was not found",
                    status_code=404,
                )
            result = await self._retrieval_service.search(
                uow=uow,
                query=query,
                command=command,
            )
            return StoredResponse(
                status_code=200,
                body=canonical_json_bytes({"data": result}),
            )

        return await self._executor.execute(request, handler)

    async def get(
        self,
        *,
        owner_agent_id: UUID,
        experience_id: UUID,
        idempotency_key: str | None = None,
    ) -> CommandResult:
        if not isinstance(owner_agent_id, UUID):
            raise ValueError("owner_agent_id must be a UUID")
        if not isinstance(experience_id, UUID):
            raise ValueError("experience_id must be a UUID")
        request = CommandRequest(
            caller_scope=f"agent:{owner_agent_id}",
            operation_scope="experience.get",
            idempotency_key=self._key(idempotency_key),
            method="GET",
            route_template=(
                "/v1/agents/{agent_id}/experiences/{experience_id}"
            ),
            path_parameters={
                "agent_id": owner_agent_id,
                "experience_id": experience_id,
            },
            body=None,
        )

        async def handler(
            uow: UnitOfWork,
            command: CommandContext,
        ) -> StoredResponse:
            result = await self._retrieval_service.get(
                uow=uow,
                owner_agent_id=owner_agent_id,
                experience_id=experience_id,
                command=command,
            )
            return StoredResponse(
                status_code=200,
                body=canonical_json_bytes({"data": result}),
            )

        return await self._executor.execute(request, handler)

    def _key(self, idempotency_key: str | None) -> str:
        if idempotency_key is None:
            return str(self._id_generator.new())
        if not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be a string or None")
        return idempotency_key


class ExperienceService:
    def __init__(
        self,
        *,
        clock: Clock,
        receipt_store: ReceiptStore,
        writer: ExperienceWriter,
        mutation_writer: ExperienceMutationWriter | None = None,
        query: ExperienceQuery | None = None,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._clock = clock
        self._receipt_store = receipt_store
        self._writer = writer
        self._lifecycle_config = lifecycle_config or LifecycleConfig()
        self._mutation_writer = mutation_writer or ExperienceMutationWriter(
            lifecycle_config=self._lifecycle_config
        )
        self._query = query or ExperienceQuery()

    async def create(
        self,
        *,
        uow: UnitOfWork,
        command: CreateExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        temperature = (
            Temperature.HOT
            if isinstance(command.importance, (int, float))
            and not isinstance(command.importance, bool)
            and command.importance >= 0.85
            else Temperature.WARM
        )
        creation = await self._writer.create_from_draft(
            uow=uow,
            draft=ExperienceDraft(
                owner_agent_id=command.owner_agent_id,
                actor_agent_id=command.owner_agent_id,
                kind=command.kind,
                origin=ExperienceOrigin.LOCAL,
                content=command.content,
                importance=command.importance,
                confidence=command.confidence,
                source_trust=1.0,
                initial_temperature=temperature,
                links=command.links,
                occurred_at=self._clock.now(),
            ),
            command=command_context,
        )
        return await self._local_response(
            uow=uow,
            command_context=command_context,
            creation=creation,
        )

    async def create_version(
        self,
        *,
        uow: UnitOfWork,
        command: CreateExperienceVersion,
        command_context: CommandContext,
    ) -> StoredResponse:
        creation = await self._writer.create_version(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            actor_agent_id=command.owner_agent_id,
            content=command.content,
            links=command.links,
            occurred_at=self._clock.now(),
            command=command_context,
        )
        return await self._local_response(
            uow=uow,
            command_context=command_context,
            creation=creation,
        )

    async def confirm(
        self,
        *,
        uow: UnitOfWork,
        command: ConfirmExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        record, occurred_at = await self._mutation_record(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            command_context=command_context,
            require_archived=False,
        )
        _require_mutation_clock(record, occurred_at)
        confidence = record.state.confidence
        after = self._materialized_state(
            record,
            at=occurred_at,
            confidence=min(1.0, confidence + (1.0 - confidence) * 0.20),
        )
        events = [
            self._pending(
                record=record,
                occurred_at=occurred_at,
                payload=ExperienceConfirmedV1(
                    schema_version=1,
                    experience_id=record.experience_id,
                    reason=_normalize_reason(command.reason),
                    evidence=_canonical_evidence(command.evidence),
                    before=record.state,
                    after=after,
                ),
            )
        ]
        resulting_state = after
        if record.state.temperature in {Temperature.WARM, Temperature.COLD}:
            resulting_state = _temperature_after(
                after,
                temperature=Temperature.HOT,
                at=occurred_at,
            )
            events.append(
                self._pending(
                    record=record,
                    occurred_at=occurred_at,
                    payload=ExperienceTemperatureChangedV1(
                        schema_version=1,
                        experience_id=record.experience_id,
                        cause="confirmation",
                        cycle_id=None,
                        before=after,
                        after=resulting_state,
                    ),
                )
            )
        return await self._apply_mutation(
            uow=uow,
            record=record,
            resulting_state=resulting_state,
            events=events,
            command_context=command_context,
        )

    async def refute(
        self,
        *,
        uow: UnitOfWork,
        command: RefuteExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        record, occurred_at = await self._mutation_record(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            command_context=command_context,
            require_archived=False,
        )
        _require_mutation_clock(record, occurred_at)
        resulting_state = self._materialized_state(
            record,
            at=occurred_at,
            confidence=max(0.0, record.state.confidence * 0.65),
        )
        event = self._pending(
            record=record,
            occurred_at=occurred_at,
            payload=ExperienceRefutedV1(
                schema_version=1,
                experience_id=record.experience_id,
                reason=_normalize_reason(command.reason),
                evidence=_canonical_evidence(command.evidence),
                before=record.state,
                after=resulting_state,
            ),
        )
        return await self._apply_mutation(
            uow=uow,
            record=record,
            resulting_state=resulting_state,
            events=(event,),
            command_context=command_context,
        )

    async def pin(
        self,
        *,
        uow: UnitOfWork,
        command: PinExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        record, occurred_at = await self._mutation_record(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            command_context=command_context,
            require_archived=False,
        )
        reason = _normalize_reason(command.reason)
        if record.state.pinned:
            return _metadata_response(record)
        _require_mutation_clock(record, occurred_at)
        after = self._materialized_state(
            record,
            at=occurred_at,
            pinned=True,
        )
        events = [
            self._pending(
                record=record,
                occurred_at=occurred_at,
                payload=ExperiencePinnedV1(
                    schema_version=1,
                    experience_id=record.experience_id,
                    reason=reason,
                    before=record.state,
                    after=after,
                ),
            )
        ]
        resulting_state = after
        if record.state.temperature in {Temperature.WARM, Temperature.COLD}:
            resulting_state = _temperature_after(
                after,
                temperature=Temperature.HOT,
                at=occurred_at,
            )
            events.append(
                self._pending(
                    record=record,
                    occurred_at=occurred_at,
                    payload=ExperienceTemperatureChangedV1(
                        schema_version=1,
                        experience_id=record.experience_id,
                        cause="pin",
                        cycle_id=None,
                        before=after,
                        after=resulting_state,
                    ),
                )
            )
        return await self._apply_mutation(
            uow=uow,
            record=record,
            resulting_state=resulting_state,
            events=events,
            command_context=command_context,
        )

    async def unpin(
        self,
        *,
        uow: UnitOfWork,
        command: UnpinExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        record, occurred_at = await self._mutation_record(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            command_context=command_context,
            require_archived=False,
        )
        reason = _normalize_reason(command.reason)
        if not record.state.pinned:
            return _metadata_response(record)
        _require_mutation_clock(record, occurred_at)
        resulting_state = self._materialized_state(
            record,
            at=occurred_at,
            pinned=False,
        )
        event = self._pending(
            record=record,
            occurred_at=occurred_at,
            payload=ExperienceUnpinnedV1(
                schema_version=1,
                experience_id=record.experience_id,
                reason=reason,
                before=record.state,
                after=resulting_state,
            ),
        )
        return await self._apply_mutation(
            uow=uow,
            record=record,
            resulting_state=resulting_state,
            events=(event,),
            command_context=command_context,
        )

    async def restore(
        self,
        *,
        uow: UnitOfWork,
        command: RestoreExperience,
        command_context: CommandContext,
    ) -> StoredResponse:
        record, occurred_at = await self._mutation_record(
            uow=uow,
            owner_agent_id=command.owner_agent_id,
            experience_id=command.experience_id,
            command_context=command_context,
            require_archived=True,
        )
        _require_mutation_clock(record, occurred_at)
        materialized = self._materialized_state(record, at=occurred_at)
        restored = self._pending(
            record=record,
            occurred_at=occurred_at,
            payload=ExperienceRestoredV1(
                schema_version=1,
                experience_id=record.experience_id,
                reason=_normalize_reason(command.reason),
                before=record.state,
                after=materialized,
            ),
        )
        resulting_state = _temperature_after(
            materialized,
            temperature=Temperature.WARM,
            at=occurred_at,
        )
        transitioned = self._pending(
            record=record,
            occurred_at=occurred_at,
            payload=ExperienceTemperatureChangedV1(
                schema_version=1,
                experience_id=record.experience_id,
                cause="restore",
                cycle_id=None,
                before=materialized,
                after=resulting_state,
            ),
        )
        return await self._apply_mutation(
            uow=uow,
            record=record,
            resulting_state=resulting_state,
            events=(restored, transitioned),
            command_context=command_context,
        )

    async def corroborate_from_capsule(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        experience_id: UUID,
        adoption_id: UUID,
        capsule_id: UUID,
        root_fingerprint: str,
        captured_trust: float,
        occurred_at: datetime,
        command_context: CommandContext,
    ) -> ExperienceRecord:
        """Apply one already-claimed independent capsule root."""
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError(
                "Capsule corroboration requires a caller-owned immediate UOW"
            )
        if (
            command_context.caller_scope != f"agent:{owner_agent_id}"
            or command_context.operation_scope != "capsule.adopt"
        ):
            raise ExperienceNotFoundError
        at = require_utc(occurred_at)
        record = await self._query.get_owned_retrieval_record(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )
        if record is None:
            raise ExperienceNotFoundError
        if record.state.temperature is Temperature.ARCHIVED:
            raise ReplayableCommandError(
                code="restore_required",
                message="Archived experiences must be restored before mutation",
                status_code=409,
            )
        _require_mutation_clock(record, at)
        trust = _unit_float("captured_trust", captured_trust)
        confidence = record.state.confidence + (
            1.0 - record.state.confidence
        ) * 0.20 * trust
        after = self._materialized_state(
            record,
            at=at,
            confidence=confidence,
        )
        events = [
            self._pending(
                record=record,
                occurred_at=at,
                payload=ExperienceCorroboratedV1(
                    schema_version=1,
                    experience_id=record.experience_id,
                    adoption_id=adoption_id,
                    capsule_id=capsule_id,
                    root_fingerprint=root_fingerprint,
                    captured_trust=trust,
                    before=record.state,
                    after=after,
                ),
            )
        ]
        resulting_state = after
        if record.state.temperature is Temperature.COLD:
            resulting_state = _temperature_after(
                after,
                temperature=Temperature.HOT,
                at=at,
            )
            events.append(
                self._pending(
                    record=record,
                    occurred_at=at,
                    payload=ExperienceTemperatureChangedV1(
                        schema_version=1,
                        experience_id=record.experience_id,
                        cause="capsule_corroboration",
                        cycle_id=None,
                        before=after,
                        after=resulting_state,
                    ),
                )
            )
        return await self._mutation_writer.apply_ordered_events(
            uow=uow,
            experience_id=record.experience_id,
            resulting_state=resulting_state,
            events=events,
            command=command_context,
        )

    async def _mutation_record(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        experience_id: UUID,
        command_context: CommandContext,
        require_archived: bool,
    ) -> tuple[RetrievalRecord, datetime]:
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError(
                "Experience mutation requires a caller-owned immediate UOW"
            )
        if (
            not isinstance(command_context, CommandContext)
            or command_context.caller_scope != f"agent:{owner_agent_id}"
        ):
            raise ExperienceNotFoundError
        record = await self._query.get_owned_retrieval_record(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )
        if record is None:
            raise ExperienceNotFoundError
        is_archived = record.state.temperature is Temperature.ARCHIVED
        if require_archived and not is_archived:
            raise ReplayableCommandError(
                code="experience_not_archived",
                message="Only an archived experience can be restored",
                status_code=409,
            )
        if not require_archived and is_archived:
            raise ReplayableCommandError(
                code="restore_required",
                message="Archived experiences must be restored before mutation",
                status_code=409,
            )
        occurred_at = require_utc(self._clock.now())
        return record, occurred_at

    def _materialized_state(
        self,
        record: RetrievalRecord,
        *,
        at: datetime,
        confidence: float | None = None,
        pinned: bool | None = None,
    ) -> ExperienceStateSnapshotV1:
        state = record.state
        resulting_confidence = (
            state.confidence if confidence is None else confidence
        )
        result = activation_at(
            ActivationInputs(
                importance=state.importance,
                confidence=resulting_confidence,
                access_count=state.access_count,
                access_strength=state.access_strength,
                strength_updated_at=state.strength_updated_at,
                last_accessed_at=state.last_accessed_at,
                created_at=record.created_at,
            ),
            at,
            self._lifecycle_config,
        )
        updates: dict[str, object] = {
            "confidence": resulting_confidence,
            "access_strength": result.decayed_strength,
            "strength_updated_at": at,
            "activation_score": result.score,
        }
        if pinned is not None:
            updates["pinned"] = pinned
        return state.model_copy(update=updates)

    @staticmethod
    def _pending(
        *,
        record: RetrievalRecord,
        occurred_at: datetime,
        payload: (
            ExperienceConfirmedV1
            | ExperienceCorroboratedV1
            | ExperienceRefutedV1
            | ExperiencePinnedV1
            | ExperienceUnpinnedV1
            | ExperienceRestoredV1
            | ExperienceTemperatureChangedV1
        ),
    ) -> PendingEvent:
        return PendingEvent(
            aggregate_type="experience",
            aggregate_id=record.experience_id,
            event_type=type(payload).event_type,
            payload=payload,
            actor_agent_id=record.owner_agent_id,
            occurred_at=occurred_at,
        )

    async def _apply_mutation(
        self,
        *,
        uow: UnitOfWork,
        record: RetrievalRecord,
        resulting_state: ExperienceStateSnapshotV1,
        events: Sequence[PendingEvent],
        command_context: CommandContext,
    ) -> StoredResponse:
        await self._mutation_writer.apply_ordered_events(
            uow=uow,
            experience_id=record.experience_id,
            resulting_state=resulting_state,
            events=events,
            command=command_context,
        )
        updated = await self._query.get_owned_retrieval_record(
            session=uow.session,
            owner_agent_id=record.owner_agent_id,
            experience_id=record.experience_id,
        )
        if updated is None:
            raise RuntimeError("Experience disappeared after its mutation")
        return _metadata_response(updated)

    async def _local_response(
        self,
        *,
        uow: UnitOfWork,
        command_context: CommandContext,
        creation: ExperienceCreation,
    ) -> StoredResponse:
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="experience",
            resource_id=creation.experience_id,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "content_hash": creation.content_hash,
                        "experience_id": str(creation.experience_id),
                        "version_id": str(creation.version_id),
                    }
                }
            ),
            headers={
                "location": f"/v1/experiences/{creation.experience_id}",
            },
        )


def _normalize_reason(
    reason: ExperienceMutationReason,
) -> StructuredReason | None:
    if reason is None or isinstance(reason, StructuredReason):
        return reason
    if not isinstance(reason, str):
        raise ValueError("reason must be a string, StructuredReason, or None")
    try:
        return StructuredReason.from_user_text(reason)
    except ValueError as error:
        raise ReplayableCommandError(
            code="invalid_reason",
            message="Reason must contain 1-2,000 nonblank characters",
            status_code=422,
        ) from error


def _canonical_evidence(
    evidence: tuple[TypedEvidence, ...],
) -> tuple[TypedEvidence, ...]:
    if not isinstance(evidence, tuple) or any(
        not isinstance(item, TypedEvidence) for item in evidence
    ):
        raise ValueError("evidence must be a tuple of TypedEvidence values")
    by_bytes = {canonical_json_bytes(item): item for item in evidence}
    return tuple(by_bytes[key] for key in sorted(by_bytes))


def _unit_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return converted


def _require_mutation_clock(
    record: RetrievalRecord,
    occurred_at: datetime,
) -> None:
    if occurred_at < record.latest_causal_at:
        raise ReplayableCommandError(
            code="clock_regression",
            message="Command time precedes existing experience state",
            status_code=409,
        )


def _temperature_after(
    before: ExperienceStateSnapshotV1,
    *,
    temperature: Temperature,
    at: datetime,
) -> ExperienceStateSnapshotV1:
    return before.model_copy(
        update={
            "temperature": temperature,
            "last_transition_at": at,
            "consecutive_below_threshold": 0,
        }
    )


def _metadata_view(record: RetrievalRecord) -> ExperienceView:
    state = record.state
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
        blurred=True,
        body=None,
        body_is_excerpt=False,
    )


def _metadata_response(record: RetrievalRecord) -> StoredResponse:
    return StoredResponse(
        status_code=200,
        body=canonical_json_bytes({"data": _metadata_view(record)}),
    )


__all__ = ["ExperienceRetrievalAdapter", "ExperienceService"]
