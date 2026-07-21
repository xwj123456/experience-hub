"""Pure lifecycle transition planning and deterministic cycle identity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid5

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    PendingEvent,
    StructuredReason,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.events import (
    ExperienceArchivedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    TemperatureChangeCause,
)
from experience_hub.experiences.models import Temperature
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.inspiration.events import InspirationIdeaArchivedV1
from experience_hub.inspiration.models import IdeaOwnerDecision
from experience_hub.lifecycle.contracts import (
    IdeaArchivePlanner,
    LifecycleResult,
    NullIdeaArchivePlanner,
    decode_lifecycle_result,
    encode_lifecycle_result,
)
from experience_hub.lifecycle.repository import (
    LifecycleRecord,
    LifecycleRepository,
)
from experience_hub.lifecycle.scoring import (
    MAX_ACCESS_STRENGTH,
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.storage.idempotency import (
    ReceiptRecord,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.unit_of_work import UnitOfWork

_LIFECYCLE_CYCLE_NAMESPACE = UUID("9f3109e0-8a3d-5f95-b15e-fd6d7b6ea1ea")


class LifecycleThresholdTarget(StrEnum):
    NONE = "none"
    PROMOTE_HOT = "promote_hot"
    DEMOTE_WARM = "demote_warm"
    DEMOTE_COLD = "demote_cold"
    ARCHIVE = "archive"


@dataclass(frozen=True, slots=True)
class LifecycleEvaluation:
    """One pure evaluation result before any event is constructed."""

    eligible: bool
    materialized_strength: float
    activation: float
    threshold_target: LifecycleThresholdTarget
    counter_before: int
    counter_after: int
    transition: Temperature | None

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be a boolean")
        for name, value, maximum in (
            (
                "materialized_strength",
                self.materialized_strength,
                MAX_ACCESS_STRENGTH,
            ),
            ("activation", self.activation, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= maximum
            ):
                raise ValueError(f"{name} is outside its valid range")
            object.__setattr__(self, name, float(value))
        if not isinstance(
            self.threshold_target,
            LifecycleThresholdTarget,
        ):
            raise ValueError("threshold_target must be a LifecycleThresholdTarget")
        for name, value in (
            ("counter_before", self.counter_before),
            ("counter_after", self.counter_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.transition is not None and not isinstance(
            self.transition,
            Temperature,
        ):
            raise ValueError("transition must be a Temperature or None")
        target_temperatures = {
            LifecycleThresholdTarget.PROMOTE_HOT: Temperature.HOT,
            LifecycleThresholdTarget.DEMOTE_WARM: Temperature.WARM,
            LifecycleThresholdTarget.DEMOTE_COLD: Temperature.COLD,
            LifecycleThresholdTarget.ARCHIVE: Temperature.ARCHIVED,
        }
        if self.transition is not None and (
            target_temperatures.get(self.threshold_target) is not self.transition
        ):
            raise ValueError("transition does not match the lifecycle threshold target")
        if not self.eligible and (
            self.threshold_target is not LifecycleThresholdTarget.NONE
            or self.transition is not None
            or self.counter_after != self.counter_before
        ):
            raise ValueError("An ineligible evaluation must be a complete no-op")


def lifecycle_config_hash(config: LifecycleConfig) -> str:
    """Hash the complete normalized lifecycle configuration."""
    if not isinstance(config, LifecycleConfig):
        raise ValueError("config must be a LifecycleConfig")
    return sha256_hex(
        canonical_json_bytes(
            {field.name: getattr(config, field.name) for field in fields(config)}
        )
    )


def lifecycle_cycle_id(
    *,
    evaluated_at: datetime,
    config: LifecycleConfig,
) -> UUID:
    """Derive UUIDv5 from canonical UTC evaluation time and config hash."""
    if not isinstance(evaluated_at, datetime):
        raise ValueError("evaluated_at must be a timezone-aware datetime")
    normalized = require_utc(evaluated_at)
    name = canonical_json_bytes(
        {
            "config_hash": lifecycle_config_hash(config),
            "evaluated_at": normalized,
        }
    ).decode("utf-8")
    return uuid5(_LIFECYCLE_CYCLE_NAMESPACE, name)


def _no_op(
    state: ExperienceStateSnapshotV1,
) -> LifecycleEvaluation:
    return LifecycleEvaluation(
        eligible=False,
        materialized_strength=state.access_strength,
        activation=state.activation_score,
        threshold_target=LifecycleThresholdTarget.NONE,
        counter_before=state.consecutive_below_threshold,
        counter_after=state.consecutive_below_threshold,
        transition=None,
    )


def _activation_inputs(
    state: ExperienceStateSnapshotV1,
    *,
    created_at: datetime,
) -> ActivationInputs:
    return ActivationInputs(
        importance=state.importance,
        confidence=state.confidence,
        access_count=state.access_count,
        access_strength=state.access_strength,
        strength_updated_at=state.strength_updated_at,
        last_accessed_at=state.last_accessed_at,
        created_at=created_at,
    )


def evaluate_transition(
    *,
    state: ExperienceStateSnapshotV1,
    created_at: datetime,
    at: datetime,
    config: LifecycleConfig,
    has_active_dependents: bool,
) -> LifecycleEvaluation:
    """Materialize one eligible lifecycle evaluation without side effects."""
    if not isinstance(state, ExperienceStateSnapshotV1):
        raise ValueError("state must be an ExperienceStateSnapshotV1")
    if not isinstance(config, LifecycleConfig):
        raise ValueError("config must be a LifecycleConfig")
    if not isinstance(has_active_dependents, bool):
        raise ValueError("has_active_dependents must be a boolean")
    if not isinstance(created_at, datetime) or not isinstance(at, datetime):
        raise ValueError("Lifecycle times must be timezone-aware datetimes")
    created_at = require_utc(created_at)
    at = require_utc(at)
    anchors = [
        created_at,
        state.strength_updated_at,
        state.last_transition_at,
    ]
    anchors.extend(
        value
        for value in (
            state.last_accessed_at,
            state.last_lifecycle_evaluated_at,
        )
        if value is not None
    )
    if at < max(anchors):
        raise ValueError("Lifecycle evaluation time would regress state")
    if state.temperature is Temperature.ARCHIVED:
        return _no_op(state)
    if (
        state.last_lifecycle_evaluated_at is not None
        and at - state.last_lifecycle_evaluated_at < config.minimum_cycle_interval
    ):
        return _no_op(state)

    materialized = activation_at(
        _activation_inputs(state, created_at=created_at),
        at,
        config,
    )
    counter_before = state.consecutive_below_threshold
    target = LifecycleThresholdTarget.NONE
    counter_after = 0
    transition: Temperature | None = None

    if state.temperature is Temperature.HOT:
        if not state.pinned and materialized.score < config.hot_to_warm_threshold:
            target = LifecycleThresholdTarget.DEMOTE_WARM
    elif state.temperature is Temperature.WARM:
        if state.pinned or (materialized.score >= config.warm_to_hot_threshold):
            target = LifecycleThresholdTarget.PROMOTE_HOT
            transition = Temperature.HOT
        elif materialized.score < config.warm_to_cold_threshold:
            target = LifecycleThresholdTarget.DEMOTE_COLD
    elif state.temperature is Temperature.COLD:
        cold_age = at - state.last_transition_at
        if (
            cold_age >= timedelta(days=config.archive_after_days)
            and state.importance < config.archive_importance_threshold
            and state.confidence < config.archive_confidence_threshold
            and materialized.decayed_strength < config.archive_strength_threshold
            and not state.pinned
            and not has_active_dependents
        ):
            target = LifecycleThresholdTarget.ARCHIVE
            transition = Temperature.ARCHIVED

    if target in {
        LifecycleThresholdTarget.DEMOTE_WARM,
        LifecycleThresholdTarget.DEMOTE_COLD,
    }:
        counter_after = min(
            config.demotion_cycles,
            counter_before + 1,
        )
        if counter_after >= config.demotion_cycles:
            transition = (
                Temperature.WARM
                if target is LifecycleThresholdTarget.DEMOTE_WARM
                else Temperature.COLD
            )

    return LifecycleEvaluation(
        eligible=True,
        materialized_strength=materialized.decayed_strength,
        activation=materialized.score,
        threshold_target=target,
        counter_before=counter_before,
        counter_after=counter_after,
        transition=transition,
    )


type LifecycleRunMode = Literal["manual", "background"]


@dataclass(frozen=True, slots=True)
class _LifecycleMutation:
    record: LifecycleRecord
    resulting_state: ExperienceStateSnapshotV1
    events: tuple[PendingEvent, ...]
    transitioned: bool
    archived: bool


class LifecycleService:
    """Run one deterministic lifecycle cycle in the caller's transaction."""

    def __init__(
        self,
        *,
        clock: Clock,
        receipt_store: ReceiptStore,
        repository: LifecycleRepository | None = None,
        mutation_writer: ExperienceMutationWriter | None = None,
        config: LifecycleConfig | None = None,
        idea_archive_planner: IdeaArchivePlanner | None = None,
    ) -> None:
        self._clock = clock
        self._receipt_store = receipt_store
        self._repository = repository or LifecycleRepository()
        self._config = config or LifecycleConfig()
        self._mutation_writer = mutation_writer or ExperienceMutationWriter(
            lifecycle_config=self._config
        )
        self._idea_archive_planner = idea_archive_planner or NullIdeaArchivePlanner()

    @property
    def config(self) -> LifecycleConfig:
        return self._config

    def cycle_id(self, evaluated_at: datetime) -> UUID:
        return lifecycle_cycle_id(
            evaluated_at=evaluated_at,
            config=self._config,
        )

    async def run(
        self,
        *,
        uow: UnitOfWork,
        evaluated_at: datetime,
        command: CommandContext,
        mode: LifecycleRunMode,
        evaluated_at_was_omitted: bool = False,
    ) -> StoredResponse:
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError("Lifecycle run requires a caller-owned immediate UOW")
        if not isinstance(command, CommandContext):
            raise ValueError("command must be a CommandContext")
        if (
            command.caller_scope != "system:local"
            or command.operation_scope != "lifecycle.run"
        ):
            raise _resource_not_found()
        if mode not in {"manual", "background"}:
            raise ValueError("mode must be manual or background")
        if not isinstance(evaluated_at_was_omitted, bool):
            raise ValueError("evaluated_at_was_omitted must be a boolean")
        if evaluated_at_was_omitted and mode != "manual":
            raise ValueError(
                "Only a manual lifecycle request may omit evaluated_at"
            )
        if evaluated_at_was_omitted:
            receipt = await self._receipt_store.get_by_id(
                session=uow.session,
                receipt_id=command.receipt_id,
            )
            if receipt is None or (
                receipt.caller_scope,
                receipt.operation_scope,
                receipt.idempotency_key,
                receipt.request_hash,
            ) != (
                command.caller_scope,
                command.operation_scope,
                command.idempotency_key,
                command.request_hash,
            ):
                raise _resource_not_found()
            evaluated_at = receipt.created_at
        if not isinstance(evaluated_at, datetime):
            raise _invalid_evaluated_at()
        try:
            normalized_at = require_utc(evaluated_at)
            current = self._clock.now()
            if not isinstance(current, datetime):
                raise ValueError("Clock returned a non-datetime value")
            clock_now = require_utc(current)
        except (TypeError, ValueError) as error:
            raise _invalid_evaluated_at() from error
        if normalized_at > clock_now:
            raise _invalid_evaluated_at()
        expected_request = CommandRequest(
            caller_scope="system:local",
            operation_scope="lifecycle.run",
            idempotency_key=command.idempotency_key,
            method="POST",
            route_template="/v1/lifecycle:run",
            body={
                "evaluated_at": (
                    None if evaluated_at_was_omitted else normalized_at
                ),
                "mode": mode,
            },
        )
        if command.request_hash != expected_request.request_hash:
            raise _resource_not_found()

        cycle_id = lifecycle_cycle_id(
            evaluated_at=normalized_at,
            config=self._config,
        )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command.receipt_id,
            resource_type="lifecycle_cycle",
            resource_id=cycle_id,
        )
        prior = await self._receipt_store.find_completed_resource(
            session=uow.session,
            resource_type="lifecycle_cycle",
            resource_id=cycle_id,
        )
        if prior is not None:
            return _completed_cycle_response(prior)

        acquired = await self._repository.claim_lease(
            uow.session,
            owner_id=command.receipt_id,
            at=clock_now,
            ttl=self._config.lease_duration,
        )
        if not acquired:
            raise ReplayableCommandError(
                code="lifecycle_in_progress",
                message="Another lifecycle cycle is already in progress",
                details={"mode": mode},
                status_code=409,
            )
        try:
            prior = await self._receipt_store.find_completed_resource(
                session=uow.session,
                resource_type="lifecycle_cycle",
                resource_id=cycle_id,
            )
            if prior is not None:
                return _completed_cycle_response(prior)

            records = await self._repository.list_current(uow.session)
            dependent_ids = await self._repository.active_dependent_target_ids(
                uow.session
            )
            plans = tuple(
                self._plan_mutation(
                    record,
                    at=normalized_at,
                    cycle_id=cycle_id,
                    has_active_dependents=(record.experience_id in dependent_ids),
                )
                for record in records
            )
            eligible_plans = tuple(plan for plan in plans if plan is not None)
            for plan in eligible_plans:
                await self._mutation_writer.apply_ordered_events(
                    uow=uow,
                    experience_id=plan.record.experience_id,
                    resulting_state=plan.resulting_state,
                    events=plan.events,
                    command=command,
                    lifecycle_config=self._config,
                )

            idea_events = await self._idea_archive_planner.due_archive_events(
                session=uow.session,
                evaluated_at=normalized_at,
                cycle_id=cycle_id,
            )
            ordered_idea_events = _ordered_idea_events(
                idea_events,
                evaluated_at=normalized_at,
                cycle_id=cycle_id,
            )
            if ordered_idea_events:
                await uow.append_events(command, ordered_idea_events)

            result = LifecycleResult(
                cycle_id=cycle_id,
                evaluated_at=normalized_at,
                evaluated_count=len(eligible_plans),
                transition_count=sum(plan.transitioned for plan in eligible_plans),
                archive_count=sum(plan.archived for plan in eligible_plans),
                idea_archive_count=len(ordered_idea_events),
            )
            return StoredResponse(
                status_code=200,
                body=encode_lifecycle_result(result),
            )
        finally:
            released = await self._repository.release_lease(
                uow.session,
                owner_id=command.receipt_id,
            )
            if not released:
                raise RuntimeError("Lifecycle lease ownership was lost")

    def _plan_mutation(
        self,
        record: LifecycleRecord,
        *,
        at: datetime,
        cycle_id: UUID,
        has_active_dependents: bool,
    ) -> _LifecycleMutation | None:
        if at < record.latest_causal_at:
            raise ReplayableCommandError(
                code="clock_regression",
                message="Lifecycle time precedes existing experience state",
                status_code=409,
            )
        try:
            evaluation = evaluate_transition(
                state=record.state,
                created_at=record.created_at,
                at=at,
                config=self._config,
                has_active_dependents=has_active_dependents,
            )
        except ValueError as error:
            raise ReplayableCommandError(
                code="clock_regression",
                message="Lifecycle time precedes existing experience state",
                status_code=409,
            ) from error
        if not evaluation.eligible:
            return None

        evaluated_state = record.state.model_copy(
            update={
                "access_strength": evaluation.materialized_strength,
                "strength_updated_at": at,
                "activation_score": evaluation.activation,
                "last_lifecycle_evaluated_at": at,
                "consecutive_below_threshold": evaluation.counter_after,
            }
        )
        events = [
            _experience_event(
                record=record,
                occurred_at=at,
                payload=ExperienceLifecycleEvaluatedV1(
                    schema_version=1,
                    experience_id=record.experience_id,
                    cycle_id=cycle_id,
                    evaluated_at=at,
                    threshold_target=evaluation.threshold_target.value,
                    before=record.state,
                    after=evaluated_state,
                ),
            )
        ]
        resulting_state = evaluated_state
        archived = evaluation.transition is Temperature.ARCHIVED
        if archived:
            events.append(
                _experience_event(
                    record=record,
                    occurred_at=at,
                    payload=ExperienceArchivedV1(
                        schema_version=1,
                        experience_id=record.experience_id,
                        cycle_id=cycle_id,
                        reason=StructuredReason.policy_due(),
                        before=evaluated_state,
                        after=evaluated_state,
                    ),
                )
            )
        if evaluation.transition is not None:
            resulting_state = evaluated_state.model_copy(
                update={
                    "temperature": evaluation.transition,
                    "last_transition_at": at,
                    "consecutive_below_threshold": 0,
                }
            )
            cause: TemperatureChangeCause = (
                "policy_archive"
                if archived
                else (
                    "lifecycle_activation"
                    if evaluation.transition is Temperature.HOT
                    else "lifecycle_demotion"
                )
            )
            events.append(
                _experience_event(
                    record=record,
                    occurred_at=at,
                    payload=ExperienceTemperatureChangedV1(
                        schema_version=1,
                        experience_id=record.experience_id,
                        cause=cause,
                        cycle_id=cycle_id,
                        before=evaluated_state,
                        after=resulting_state,
                    ),
                )
            )
        return _LifecycleMutation(
            record=record,
            resulting_state=resulting_state,
            events=tuple(events),
            transitioned=evaluation.transition is not None,
            archived=archived,
        )


def _invalid_evaluated_at() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="invalid_evaluated_at",
        message="Lifecycle evaluation time must be UTC-aware and not in the future",
        status_code=422,
    )


def _resource_not_found() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="resource_not_found",
        message="The command resource was not found",
        status_code=404,
    )


def _completed_cycle_response(receipt: ReceiptRecord) -> StoredResponse:
    if (
        receipt.response is None
        or receipt.response.status_code != 200
        or receipt.result_resource_type != "lifecycle_cycle"
        or receipt.result_resource_id is None
    ):
        raise RuntimeError("Completed lifecycle resource has an invalid response")
    try:
        result = decode_lifecycle_result(receipt.response.body)
    except ValueError as error:
        raise RuntimeError(
            "Completed lifecycle resource has a corrupt response"
        ) from error
    if result.cycle_id != receipt.result_resource_id:
        raise RuntimeError(
            "Completed lifecycle resource response has a mismatched cycle ID"
        )
    return StoredResponse(
        status_code=200,
        body=encode_lifecycle_result(result),
    )


def _experience_event(
    *,
    record: LifecycleRecord,
    occurred_at: datetime,
    payload: (
        ExperienceLifecycleEvaluatedV1
        | ExperienceArchivedV1
        | ExperienceTemperatureChangedV1
    ),
) -> PendingEvent:
    return PendingEvent(
        aggregate_type="experience",
        aggregate_id=record.experience_id,
        event_type=type(payload).event_type,
        payload=payload,
        actor_agent_id=None,
        occurred_at=occurred_at,
    )


def _ordered_idea_events(
    events: Sequence[PendingEvent],
    *,
    evaluated_at: datetime,
    cycle_id: UUID,
) -> tuple[PendingEvent, ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise ValueError("Idea archive planner must return an event sequence")
    supplied = tuple(events)
    if any(not isinstance(event, PendingEvent) for event in supplied):
        raise ValueError("Idea archive planner must return only PendingEvent values")
    ordered = tuple(sorted(supplied, key=lambda event: event.aggregate_id.bytes))
    if len({event.aggregate_id for event in ordered}) != len(ordered):
        raise ValueError("Idea archive planner returned duplicate idea events")
    for event in ordered:
        if (
            event.aggregate_type != "idea"
            or event.event_type != InspirationIdeaArchivedV1.event_type
            or not isinstance(
                event.payload,
                InspirationIdeaArchivedV1,
            )
            or event.payload.idea_id != event.aggregate_id
            or event.payload.owner_decision_before is not IdeaOwnerDecision.ACTIVE
            or event.payload.owner_decision_after is not IdeaOwnerDecision.ARCHIVED
            or event.payload.reason != StructuredReason.policy_due()
            or event.actor_agent_id is not None
            or require_utc(event.occurred_at) != evaluated_at
            or event.payload.cycle_id != cycle_id
        ):
            raise ValueError("Idea archive planner returned an invalid lifecycle event")
    return ordered


__all__ = [
    "LifecycleEvaluation",
    "LifecycleRunMode",
    "LifecycleService",
    "LifecycleThresholdTarget",
    "evaluate_transition",
    "lifecycle_config_hash",
    "lifecycle_cycle_id",
]
