"""Shared transaction-bound writer for ordered experience-state mutations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import select

from experience_hub.clock import require_utc
from experience_hub.domain import CommandContext, PendingEvent
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import preferred_payload_codec
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.experiences.events import (
    ExperienceAccessedV1,
    ExperienceArchivedV1,
    ExperienceConfirmedV1,
    ExperienceCorroboratedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperiencePinnedV1,
    ExperienceReactivatedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
)
from experience_hub.experiences.models import Temperature
from experience_hub.experiences.repository import (
    ExperienceRepository,
    decode_and_verify_version,
    snapshot_from_state_row,
)
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.storage.payload_rewrite import rewrite_payload_codec
from experience_hub.storage.tables import (
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


def _experience_not_found() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="experience_not_found",
        message="Experience was not found",
        status_code=404,
    )


def _clock_regression() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="clock_regression",
        message="Command time precedes existing experience state",
        status_code=409,
    )


def _event_snapshot(
    event: PendingEvent,
    field_name: str,
) -> ExperienceStateSnapshotV1:
    value = getattr(event.payload, field_name, None)
    if not isinstance(value, ExperienceStateSnapshotV1):
        raise ValueError(
            f"Every mutation event must contain a complete {field_name} state"
        )
    return value


def _event_time(event: PendingEvent) -> datetime:
    if not isinstance(event.occurred_at, datetime):
        raise ValueError("Mutation event time must be timezone-aware")
    try:
        return require_utc(event.occurred_at)
    except ValueError as error:
        raise ValueError("Mutation event time must be timezone-aware") from error


class ExperienceMutationWriter:
    """Apply an exact event sequence and physical codec transition atomically."""

    def __init__(
        self,
        *,
        repository: ExperienceRepository | None = None,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._repository = repository or ExperienceRepository()
        self._lifecycle_config = lifecycle_config or LifecycleConfig()

    async def apply_ordered_events(
        self,
        *,
        uow: UnitOfWork,
        experience_id: UUID,
        resulting_state: ExperienceStateSnapshotV1,
        events: Sequence[PendingEvent],
        command: CommandContext,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> ExperienceRecord:
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError(
                "Experience mutation requires a caller-owned immediate UOW"
            )
        if not isinstance(experience_id, UUID):
            raise ValueError("experience_id must be a UUID")
        if not isinstance(resulting_state, ExperienceStateSnapshotV1):
            raise ValueError(
                "resulting_state must be an ExperienceStateSnapshotV1"
            )
        if not isinstance(command, CommandContext):
            raise ValueError("command must be a CommandContext")
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise ValueError("events must be an ordered event sequence")
        ordered = tuple(events)
        if not ordered:
            raise ValueError("events must not be empty")
        if any(not isinstance(event, PendingEvent) for event in ordered):
            raise ValueError("events must contain only PendingEvent values")

        first_before = _event_snapshot(ordered[0], "before")
        owner_agent_id = first_before.owner_agent_id
        actor_agent_id = ordered[0].actor_agent_id
        occurred_at = _event_time(ordered[0])
        previous_after: ExperienceStateSnapshotV1 | None = None
        for event in ordered:
            before = _event_snapshot(event, "before")
            after = _event_snapshot(event, "after")
            try:
                payload_event_type = type(event.payload).event_type
            except AttributeError as error:
                raise ValueError(
                    "Mutation payload must declare an event type"
                ) from error
            if (
                event.aggregate_type != "experience"
                or event.aggregate_id != experience_id
                or event.event_type != payload_event_type
                or before.experience_id != experience_id
                or after.experience_id != experience_id
            ):
                raise ValueError(
                    "Mutation events must target the requested experience"
                )
            if (
                before.owner_agent_id != owner_agent_id
                or after.owner_agent_id != owner_agent_id
            ):
                raise ValueError("Mutation events must retain one owner")
            if (
                event.actor_agent_id != actor_agent_id
                or event.actor_agent_id not in {owner_agent_id, None}
            ):
                raise ValueError(
                    "Mutation events must retain the owner or system actor"
                )
            if previous_after is not None and before != previous_after:
                raise ValueError(
                    "Mutation event before state must equal the prior after state"
                )
            if _event_time(event) != occurred_at:
                raise ValueError("Mutation events must share one command time")
            previous_after = after
        assert previous_after is not None
        if (
            resulting_state.experience_id != experience_id
            or resulting_state.owner_agent_id != owner_agent_id
            or previous_after != resulting_state
        ):
            raise ValueError(
                "resulting_state must equal the final mutation after state"
            )
        self._require_exact_lifecycle_sequence(
            ordered=ordered,
            first_before=first_before,
            actor_agent_id=actor_agent_id,
            owner_agent_id=owner_agent_id,
            lifecycle_config=lifecycle_config or self._lifecycle_config,
        )
        has_access = any(
            isinstance(event.payload, ExperienceAccessedV1)
            for event in ordered
        )
        has_reactivation = any(
            isinstance(event.payload, ExperienceReactivatedV1)
            for event in ordered
        )
        has_cold_reactivation_transition = any(
            isinstance(
                event.payload,
                ExperienceTemperatureChangedV1,
            )
            and event.payload.cause == "cold_reactivation"
            for event in ordered
        )
        if has_access and first_before.temperature is Temperature.ARCHIVED:
            raise ValueError("Archived experience cannot be accessed")
        if (
            has_access or has_reactivation
        ) and actor_agent_id != owner_agent_id:
            raise ValueError("Retrieval mutations require the owner actor")
        needs_cold_reactivation_chain = (
            has_reactivation
            or has_cold_reactivation_transition
            or (
                has_access
                and first_before.temperature is Temperature.COLD
            )
        )
        if needs_cold_reactivation_chain and (
            len(ordered) != 3
            or not isinstance(ordered[0].payload, ExperienceAccessedV1)
            or not isinstance(ordered[1].payload, ExperienceReactivatedV1)
            or not isinstance(
                ordered[2].payload,
                ExperienceTemperatureChangedV1,
            )
            or ordered[2].payload.cause != "cold_reactivation"
        ):
            raise ValueError(
                "Cold access requires the exact three-event reactivation sequence"
            )
        if has_access and not needs_cold_reactivation_chain and (
            len(ordered) != 1
            or not isinstance(ordered[0].payload, ExperienceAccessedV1)
        ):
            raise ValueError("Ordinary access requires one exact access event")

        current = await self._repository.get_owned_current(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )
        if current is None:
            raise _experience_not_found()
        identity, current_version, state, projection_event = current
        locked = snapshot_from_state_row(state)
        if first_before != locked:
            raise ValueError(
                "First mutation event before state does not match locked state"
            )
        causal_times = [
            identity.created_at,
            current_version.created_at,
            projection_event.occurred_at,
            locked.strength_updated_at,
            locked.last_transition_at,
        ]
        causal_times.extend(
            value
            for value in (
                locked.last_accessed_at,
                locked.last_lifecycle_evaluated_at,
            )
            if value is not None
        )
        if occurred_at < max(causal_times):
            raise _clock_regression()

        current_version_id = current_version.version_id
        version_ids: tuple[UUID, ...] | None = None
        if locked.temperature is not resulting_state.temperature:
            version_ids = await self._load_validated_owned_version_ids(
                uow=uow,
                identity=identity,
                owner_agent_id=owner_agent_id,
                experience_id=experience_id,
                current_version_id=current_version_id,
            )
        await uow.append_events(command, ordered)
        if version_ids is not None:
            await self._rewrite_owned_versions(
                uow=uow,
                resulting_state=resulting_state,
                version_ids=version_ids,
            )
        return ExperienceRecord(
            experience_id=experience_id,
            owner_agent_id=owner_agent_id,
            current_version_id=resulting_state.current_version_id,
            current_content_hash=resulting_state.current_content_hash,
            temperature=resulting_state.temperature,
        )

    def _require_exact_lifecycle_sequence(
        self,
        *,
        ordered: tuple[PendingEvent, ...],
        first_before: ExperienceStateSnapshotV1,
        actor_agent_id: UUID | None,
        owner_agent_id: UUID,
        lifecycle_config: LifecycleConfig,
    ) -> None:
        payloads = tuple(event.payload for event in ordered)
        has_capsule_corroboration = any(
            isinstance(payload, ExperienceCorroboratedV1)
            or (
                isinstance(payload, ExperienceTemperatureChangedV1)
                and payload.cause == "capsule_corroboration"
            )
            for payload in payloads
        )
        if has_capsule_corroboration:
            if (
                actor_agent_id != owner_agent_id
                or first_before.temperature is Temperature.ARCHIVED
                or not isinstance(payloads[0], ExperienceCorroboratedV1)
            ):
                raise ValueError(
                    "Capsule corroboration requires the active owner actor"
                )
            if first_before.temperature is Temperature.COLD:
                if _matches_temperature_sequence(
                    payloads,
                    head_type=ExperienceCorroboratedV1,
                    cause="capsule_corroboration",
                ):
                    return
            elif len(payloads) == 1:
                return
            raise ValueError(
                "Capsule corroboration requires its exact event sequence"
            )

        task6_payload_types = (
            ExperienceArchivedV1,
            ExperienceConfirmedV1,
            ExperienceLifecycleEvaluatedV1,
            ExperiencePinnedV1,
            ExperienceRefutedV1,
            ExperienceRestoredV1,
            ExperienceUnpinnedV1,
        )
        task6_temperature_causes = {
            "confirmation",
            "pin",
            "lifecycle_activation",
            "lifecycle_demotion",
            "policy_archive",
            "restore",
        }
        has_task6_payload = any(
            isinstance(payload, task6_payload_types)
            or (
                isinstance(payload, ExperienceTemperatureChangedV1)
                and payload.cause in task6_temperature_causes
            )
            for payload in payloads
        )
        if not has_task6_payload:
            return

        first = payloads[0]
        if isinstance(first, ExperienceConfirmedV1):
            if actor_agent_id != owner_agent_id:
                raise ValueError("Confirmed events require the owner actor")
            if first_before.temperature is Temperature.ARCHIVED:
                raise ValueError("Archived experience must be restored first")
            if first_before.temperature is Temperature.HOT:
                if len(payloads) == 1:
                    return
            elif _matches_temperature_sequence(
                payloads,
                head_type=ExperienceConfirmedV1,
                cause="confirmation",
            ):
                return
            raise ValueError(
                "Confirmation requires the exact confirmed event sequence"
            )

        if isinstance(first, ExperienceRefutedV1):
            if (
                actor_agent_id == owner_agent_id
                and first_before.temperature is not Temperature.ARCHIVED
                and len(payloads) == 1
            ):
                return
            raise ValueError("Refutation requires one exact refuted event")

        if isinstance(first, ExperiencePinnedV1):
            if actor_agent_id != owner_agent_id:
                raise ValueError("Pinned events require the owner actor")
            if first_before.temperature is Temperature.ARCHIVED:
                raise ValueError("Archived experience must be restored first")
            if first_before.temperature is Temperature.HOT:
                if len(payloads) == 1:
                    return
            elif _matches_temperature_sequence(
                payloads,
                head_type=ExperiencePinnedV1,
                cause="pin",
            ):
                return
            raise ValueError("Pin requires the exact pinned event sequence")

        if isinstance(first, ExperienceUnpinnedV1):
            if (
                actor_agent_id == owner_agent_id
                and first_before.temperature is not Temperature.ARCHIVED
                and len(payloads) == 1
            ):
                return
            raise ValueError("Unpin requires one exact unpinned event")

        if isinstance(first, ExperienceRestoredV1):
            if (
                actor_agent_id == owner_agent_id
                and first_before.temperature is Temperature.ARCHIVED
                and _matches_temperature_sequence(
                    payloads,
                    head_type=ExperienceRestoredV1,
                    cause="restore",
                )
            ):
                return
            raise ValueError(
                "Restore requires the exact restored event sequence"
            )

        if isinstance(first, ExperienceLifecycleEvaluatedV1):
            if actor_agent_id is not None:
                raise ValueError("Lifecycle events require the system actor")
            target = first.threshold_target
            if target == "none":
                if len(payloads) == 1:
                    return
            elif target == "promote_hot":
                if _matches_lifecycle_temperature_sequence(
                    payloads,
                    first=first,
                    cause="lifecycle_activation",
                ):
                    return
            elif target in {"demote_warm", "demote_cold"}:
                transition_due = (
                    first.after.consecutive_below_threshold
                    >= lifecycle_config.demotion_cycles
                )
                if transition_due and _matches_lifecycle_temperature_sequence(
                    payloads,
                    first=first,
                    cause="lifecycle_demotion",
                ):
                    return
                if not transition_due and len(payloads) == 1:
                    return
            elif (
                target == "archive"
                and len(payloads) == 3
                and isinstance(payloads[1], ExperienceArchivedV1)
                and isinstance(
                    payloads[2],
                    ExperienceTemperatureChangedV1,
                )
                and payloads[1].cycle_id == first.cycle_id
                and payloads[2].cause == "policy_archive"
                and payloads[2].cycle_id == first.cycle_id
            ):
                return
            raise ValueError(
                "Lifecycle evaluation requires its exact transition sequence"
            )

        raise ValueError(
            "Lifecycle mutation events must start with their protocol event"
        )

    async def _load_validated_owned_version_ids(
        self,
        *,
        uow: UnitOfWork,
        identity: ExperienceRow,
        owner_agent_id: UUID,
        experience_id: UUID,
        current_version_id: UUID,
    ) -> tuple[UUID, ...]:
        rows = (
            await uow.session.execute(
                select(ExperienceVersionRow, ExperiencePayloadRow)
                .join(
                    ExperienceRow,
                    ExperienceRow.experience_id
                    == ExperienceVersionRow.experience_id,
                )
                .outerjoin(
                    ExperiencePayloadRow,
                    ExperiencePayloadRow.version_id
                    == ExperienceVersionRow.version_id,
                )
                .where(
                    ExperienceRow.owner_agent_id == owner_agent_id,
                    ExperienceVersionRow.experience_id == experience_id,
                )
                .order_by(
                    ExperienceVersionRow.version_number,
                    ExperienceVersionRow.version_id,
                )
            )
        ).all()
        version_ids = tuple(version.version_id for version, _ in rows)
        if (
            not rows
            or current_version_id not in version_ids
            or any(payload is None for _, payload in rows)
        ):
            raise SourceIntegrityError(
                f"Owned experience {experience_id} has incomplete versions"
            )
        for version, payload in rows:
            assert payload is not None
            decode_and_verify_version(
                identity=identity,
                version=version,
                payload=payload,
            )
        return version_ids

    async def _rewrite_owned_versions(
        self,
        *,
        uow: UnitOfWork,
        resulting_state: ExperienceStateSnapshotV1,
        version_ids: tuple[UUID, ...],
    ) -> None:
        codec = preferred_payload_codec(resulting_state.temperature)
        for version_id in version_ids:
            await rewrite_payload_codec(
                session=uow.session,
                version_id=version_id,
                codec=codec,
            )


def _matches_temperature_sequence(
    payloads: tuple[object, ...],
    *,
    head_type: type[object],
    cause: str,
) -> bool:
    return (
        len(payloads) == 2
        and isinstance(payloads[0], head_type)
        and isinstance(payloads[1], ExperienceTemperatureChangedV1)
        and payloads[1].cause == cause
        and payloads[1].cycle_id is None
    )


def _matches_lifecycle_temperature_sequence(
    payloads: tuple[object, ...],
    *,
    first: ExperienceLifecycleEvaluatedV1,
    cause: str,
) -> bool:
    return (
        len(payloads) == 2
        and isinstance(payloads[1], ExperienceTemperatureChangedV1)
        and payloads[1].cause == cause
        and payloads[1].cycle_id == first.cycle_id
    )


__all__ = ["ExperienceMutationWriter"]
