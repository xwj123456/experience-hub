"""Transaction-bound source persistence for immutable experiences."""

from __future__ import annotations

import json
import math
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.clock import require_utc
from experience_hub.domain import (
    CommandContext,
    EventRegistry,
    PendingEvent,
    TypedEvidence,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import (
    decode_payload,
    decode_version_content,
    encode_version_content,
    preferred_payload_codec,
)
from experience_hub.experiences.contracts import (
    ExperienceCreation,
    ExperienceDraft,
    ExperienceRecord,
    VersionLinkInput,
    canonicalize_version_links,
)
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
    VersionLinkRefV1,
    register_experience_events,
)
from experience_hub.experiences.models import (
    PayloadCodec,
    Temperature,
    VersionContent,
)
from experience_hub.ids import IdGenerator
from experience_hub.lifecycle.scoring import (
    ActivationInputs,
    LifecycleConfig,
    activation_at,
)
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError


def _invalid_experience(message: str) -> ReplayableCommandError:
    return ReplayableCommandError(
        code="invalid_experience",
        message=message,
        status_code=422,
    )


def _invalid_link(message: str) -> ReplayableCommandError:
    return ReplayableCommandError(
        code="invalid_experience_link",
        message=message,
        status_code=422,
    )


def _duplicate_experience() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="duplicate_experience",
        message="Another current experience has the same semantic content",
        status_code=409,
    )


def _experience_not_found() -> ReplayableCommandError:
    return ReplayableCommandError(
        code="experience_not_found",
        message="Experience was not found",
        status_code=404,
    )


def _unit_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_experience(f"{name} must be between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise _invalid_experience(f"{name} must be between zero and one")
    return converted


def snapshot_from_state_row(row: ExperienceStateRow) -> ExperienceStateSnapshotV1:
    """Copy an ORM projection row into its immutable event representation."""
    return ExperienceStateSnapshotV1(
        experience_id=row.experience_id,
        owner_agent_id=row.owner_agent_id,
        current_version_id=row.current_version_id,
        current_content_hash=row.current_content_hash,
        temperature=row.temperature,
        importance=row.importance,
        confidence=row.confidence,
        activation_score=row.activation_score,
        source_trust=row.source_trust,
        access_count=row.access_count,
        access_strength=row.access_strength,
        strength_updated_at=row.strength_updated_at,
        last_accessed_at=row.last_accessed_at,
        last_transition_at=row.last_transition_at,
        last_lifecycle_evaluated_at=row.last_lifecycle_evaluated_at,
        consecutive_below_threshold=row.consecutive_below_threshold,
        pinned=row.pinned,
    )


def decode_and_verify_version(
    *,
    identity: ExperienceRow,
    version: ExperienceVersionRow,
    payload: ExperiencePayloadRow,
) -> VersionContent:
    """Reconstruct a source version and prove both semantic hashes."""
    try:
        if (
            version.experience_id != identity.experience_id
            or payload.version_id != version.version_id
        ):
            raise ValueError("Version source IDs are inconsistent")
        decoded = decode_payload(payload.codec, payload.payload)
        if sha256_hex(decoded) != payload.payload_hash:
            raise ValueError("Decoded payload hash does not match source")
        tags = json.loads(version.tags)
        applicability = json.loads(version.applicability)
        evidence_values = json.loads(version.evidence)
        falsifiers = json.loads(version.falsifiers)
        if not all(
            isinstance(value, list)
            for value in (
                tags,
                applicability,
                evidence_values,
                falsifiers,
            )
        ):
            raise ValueError("Stored experience metadata must use arrays")
        evidence = tuple(
            TypedEvidence.model_validate(item) for item in evidence_values
        )
        content = decode_version_content(
            body_payload=decoded,
            summary=version.summary,
            mechanism=version.mechanism,
            tags=tags,
            applicability=applicability,
            evidence=evidence,
            falsifiers=falsifiers,
        )
        encoded = encode_version_content(
            kind=identity.kind,
            content=content,
            codec=payload.codec,
        )
        if encoded.payload_hash != payload.payload_hash:
            raise ValueError("Recomputed payload hash does not match source")
        if encoded.content_hash != version.content_hash:
            raise ValueError("Recomputed content hash does not match source")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SourceIntegrityError(
            f"Experience version {version.version_id} failed semantic validation"
        ) from error
    return content


async def require_current_aggregate_head(
    *,
    session: AsyncSession,
    experience_id: UUID,
    projection_event: DomainEventRow,
    event_registry: EventRegistry,
    handled_event_types: frozenset[str],
) -> None:
    """Prove that a persisted projection checkpoint is the aggregate head."""
    head_sequence = await session.scalar(
        select(func.max(DomainEventRow.sequence)).where(
            DomainEventRow.aggregate_type == "experience",
            DomainEventRow.aggregate_id == experience_id,
        )
    )
    if (
        head_sequence is None
        or projection_event.sequence != head_sequence
        or projection_event.event_type not in handled_event_types
        or projection_event.event_type not in event_registry.event_types
    ):
        raise SourceIntegrityError(
            f"Experience {experience_id} projection checkpoint is invalid"
        )
    try:
        payload = event_registry.decode(
            event_type=projection_event.event_type,
            payload=projection_event.payload,
        )
    except (TypeError, ValueError) as error:
        raise SourceIntegrityError(
            f"Experience {experience_id} projection checkpoint is invalid"
        ) from error
    if getattr(payload, "experience_id", None) != experience_id:
        raise SourceIntegrityError(
            f"Experience {experience_id} projection checkpoint is invalid"
        )


class ExperienceRepository:
    """Low-level owner-scoped source operations used by transaction writers."""

    def __init__(
        self,
        *,
        event_registry: EventRegistry | None = None,
        handled_event_types: frozenset[str] = STATE_EXPERIENCE_EVENT_TYPES,
    ) -> None:
        if event_registry is None:
            event_registry = EventRegistry()
            register_experience_events(event_registry)
        self._event_registry = event_registry
        self._handled_event_types = handled_event_types

    async def find_current_equivalent(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        content_hash: str,
    ) -> ExperienceRecord | None:
        row = await session.execute(
            select(ExperienceStateRow).where(
                ExperienceStateRow.owner_agent_id == owner_agent_id,
                ExperienceStateRow.current_content_hash == content_hash,
            )
        )
        state = row.scalar_one_or_none()
        if state is None:
            return None
        return ExperienceRecord(
            experience_id=state.experience_id,
            owner_agent_id=state.owner_agent_id,
            current_version_id=state.current_version_id,
            current_content_hash=state.current_content_hash,
            temperature=state.temperature,
        )

    async def get_owned_current(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> tuple[
        ExperienceRow,
        ExperienceVersionRow,
        ExperienceStateRow,
        DomainEventRow,
    ] | None:
        identity = await session.scalar(
            select(ExperienceRow).where(
                ExperienceRow.owner_agent_id == owner_agent_id,
                ExperienceRow.experience_id == experience_id,
            )
        )
        if identity is None:
            return None
        result = await session.execute(
            select(
                ExperienceVersionRow,
                ExperienceStateRow,
                DomainEventRow,
            )
            .select_from(ExperienceStateRow)
            .join(
                ExperienceVersionRow,
                ExperienceVersionRow.version_id
                == ExperienceStateRow.current_version_id,
            )
            .join(
                DomainEventRow,
                DomainEventRow.event_id
                == ExperienceStateRow.projection_event_id,
            )
            .where(
                ExperienceStateRow.experience_id == experience_id,
            )
        )
        value = result.one_or_none()
        if value is None:
            raise SourceIntegrityError(
                f"Owned experience {experience_id} has no complete current state"
            )
        version, state, projection_event = value
        if (
            state.owner_agent_id != owner_agent_id
            or version.experience_id != experience_id
            or state.current_content_hash != version.content_hash
            or projection_event.aggregate_type != "experience"
            or projection_event.aggregate_id != experience_id
        ):
            raise SourceIntegrityError(
                f"Owned experience {experience_id} has inconsistent current state"
            )
        await require_current_aggregate_head(
            session=session,
            experience_id=experience_id,
            projection_event=projection_event,
            event_registry=self._event_registry,
            handled_event_types=self._handled_event_types,
        )
        return identity, version, state, projection_event

    async def validate_link_targets(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        links: tuple[VersionLinkRefV1, ...],
    ) -> None:
        targets = {link.target_experience_id for link in links}
        if not targets:
            return
        found = set(
            (
                await session.scalars(
                    select(ExperienceRow.experience_id).where(
                        ExperienceRow.owner_agent_id == owner_agent_id,
                        ExperienceRow.experience_id.in_(targets),
                    )
                )
            ).all()
        )
        if found != targets:
            raise _invalid_link(
                "Every link target must be an experience owned by the same agent"
            )

    @staticmethod
    def add_identity(
        *,
        session: AsyncSession,
        experience_id: UUID,
        draft: ExperienceDraft,
    ) -> None:
        session.add(
            ExperienceRow(
                experience_id=experience_id,
                owner_agent_id=draft.owner_agent_id,
                kind=draft.kind,
                origin=draft.origin,
                created_at=draft.occurred_at,
            )
        )

    @staticmethod
    def add_version(
        *,
        session: AsyncSession,
        experience_id: UUID,
        version_id: UUID,
        version_number: int,
        supersedes_version_id: UUID | None,
        content: VersionContent,
        content_hash: str,
        created_at: datetime,
    ) -> None:
        session.add(
            ExperienceVersionRow(
                version_id=version_id,
                experience_id=experience_id,
                version_number=version_number,
                summary=content.summary,
                mechanism=content.mechanism,
                tags=canonical_json_bytes(content.tags),
                applicability=canonical_json_bytes(content.applicability),
                evidence=canonical_json_bytes(content.evidence),
                falsifiers=canonical_json_bytes(content.falsifiers),
                content_hash=content_hash,
                supersedes_version_id=supersedes_version_id,
                created_at=created_at,
            )
        )

    @staticmethod
    def add_payload(
        *,
        session: AsyncSession,
        version_id: UUID,
        codec: PayloadCodec,
        payload: bytes,
        payload_hash: str,
    ) -> None:
        session.add(
            ExperiencePayloadRow(
                version_id=version_id,
                codec=codec,
                payload=payload,
                payload_hash=payload_hash,
            )
        )

    @staticmethod
    def add_links(
        *,
        session: AsyncSession,
        source_experience_id: UUID,
        source_version_id: UUID,
        source_event_id: int,
        links: tuple[VersionLinkRefV1, ...],
    ) -> None:
        session.add_all(
            [
                ExperienceLinkRow(
                    source_experience_id=source_experience_id,
                    source_version_id=source_version_id,
                    target_experience_id=link.target_experience_id,
                    relation=link.relation,
                    source_event_id=source_event_id,
                )
                for link in links
            ]
        )


class ExperienceWriter:
    """Append experience source rows and events inside a caller-owned UoW."""

    def __init__(
        self,
        *,
        id_generator: IdGenerator,
        repository: ExperienceRepository | None = None,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self._id_generator = id_generator
        self._repository = repository or ExperienceRepository()
        self._lifecycle_config = lifecycle_config or LifecycleConfig()

    async def find_current_equivalent(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        content_hash: str,
    ) -> ExperienceRecord | None:
        return await self._repository.find_current_equivalent(
            session=session,
            owner_agent_id=owner_agent_id,
            content_hash=content_hash,
        )

    async def create_from_draft(
        self,
        *,
        uow: UnitOfWork,
        draft: ExperienceDraft,
        command: CommandContext,
    ) -> ExperienceCreation:
        occurred_at = require_utc(draft.occurred_at)
        importance = _unit_float("importance", draft.importance)
        confidence = _unit_float("confidence", draft.confidence)
        source_trust = _unit_float("source_trust", draft.source_trust)
        codec = preferred_payload_codec(draft.initial_temperature)
        encoded = encode_version_content(
            kind=draft.kind,
            content=draft.content,
            codec=codec,
        )
        equivalent = await self.find_current_equivalent(
            session=uow.session,
            owner_agent_id=draft.owner_agent_id,
            content_hash=encoded.content_hash,
        )
        if equivalent is not None:
            raise _duplicate_experience()

        experience_id = self._id_generator.new()
        version_id = self._id_generator.new()
        try:
            links = canonicalize_version_links(
                source_experience_id=experience_id,
                links=draft.links,
            )
        except ValueError as error:
            raise _invalid_link(str(error)) from error
        await self._repository.validate_link_targets(
            session=uow.session,
            owner_agent_id=draft.owner_agent_id,
            links=links,
        )

        activation = activation_at(
            ActivationInputs(
                importance=importance,
                confidence=confidence,
                access_count=0,
                access_strength=0.0,
                strength_updated_at=occurred_at,
                last_accessed_at=None,
                created_at=occurred_at,
            ),
            occurred_at,
            self._lifecycle_config,
        )
        after = ExperienceStateSnapshotV1(
            experience_id=experience_id,
            owner_agent_id=draft.owner_agent_id,
            current_version_id=version_id,
            current_content_hash=encoded.content_hash,
            temperature=draft.initial_temperature,
            importance=importance,
            confidence=confidence,
            activation_score=activation.score,
            source_trust=source_trust,
            access_count=0,
            access_strength=0.0,
            strength_updated_at=occurred_at,
            last_accessed_at=None,
            last_transition_at=occurred_at,
            last_lifecycle_evaluated_at=None,
            consecutive_below_threshold=0,
            pinned=False,
        )
        self._repository.add_identity(
            session=uow.session,
            experience_id=experience_id,
            draft=draft,
        )
        await uow.session.flush()
        self._repository.add_version(
            session=uow.session,
            experience_id=experience_id,
            version_id=version_id,
            version_number=1,
            supersedes_version_id=None,
            content=draft.content,
            content_hash=encoded.content_hash,
            created_at=occurred_at,
        )
        await uow.session.flush()
        self._repository.add_payload(
            session=uow.session,
            version_id=version_id,
            codec=encoded.codec,
            payload=encoded.payload,
            payload_hash=encoded.payload_hash,
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)

        stored = await uow.append_events(
            command,
            [
                PendingEvent(
                    aggregate_type="experience",
                    aggregate_id=experience_id,
                    event_type=ExperienceCreatedV1.event_type,
                    payload=ExperienceCreatedV1(
                        schema_version=1,
                        experience_id=experience_id,
                        version_id=version_id,
                        after=after,
                    ),
                    actor_agent_id=draft.actor_agent_id,
                    occurred_at=occurred_at,
                ),
                PendingEvent(
                    aggregate_type="experience",
                    aggregate_id=experience_id,
                    event_type=ExperienceVersionCreatedV1.event_type,
                    payload=ExperienceVersionCreatedV1(
                        schema_version=1,
                        experience_id=experience_id,
                        version_id=version_id,
                        version_number=1,
                        supersedes_version_id=None,
                        links=links,
                        before=after,
                        after=after,
                    ),
                    actor_agent_id=draft.actor_agent_id,
                    occurred_at=occurred_at,
                ),
            ],
        )
        version_event = stored[1]
        self._repository.add_links(
            session=uow.session,
            source_experience_id=experience_id,
            source_version_id=version_id,
            source_event_id=version_event.event_id,
            links=links,
        )
        await uow.session.flush()
        return ExperienceCreation(
            experience_id=experience_id,
            version_id=version_id,
            content_hash=encoded.content_hash,
        )

    async def create_version(
        self,
        *,
        uow: UnitOfWork,
        owner_agent_id: UUID,
        experience_id: UUID,
        actor_agent_id: UUID,
        content: VersionContent,
        links: tuple[VersionLinkInput, ...],
        occurred_at: datetime,
        command: CommandContext,
    ) -> ExperienceCreation:
        occurred_at = require_utc(occurred_at)
        current = await self._repository.get_owned_current(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
        )
        if current is None:
            raise _experience_not_found()
        identity, previous_version, state, projection_event = current
        if state.temperature is Temperature.ARCHIVED:
            raise ReplayableCommandError(
                code="restore_required",
                message="Archived experiences must be restored before mutation",
                status_code=409,
            )
        causal_times = [
            identity.created_at,
            previous_version.created_at,
            projection_event.occurred_at,
            state.strength_updated_at,
            state.last_transition_at,
        ]
        causal_times.extend(
            value
            for value in (
                state.last_accessed_at,
                state.last_lifecycle_evaluated_at,
            )
            if value is not None
        )
        if occurred_at < max(causal_times):
            raise ReplayableCommandError(
                code="clock_regression",
                message="Command time precedes existing experience state",
                status_code=409,
            )

        try:
            canonical_links = canonicalize_version_links(
                source_experience_id=experience_id,
                links=links,
            )
        except ValueError as error:
            raise _invalid_link(str(error)) from error
        await self._repository.validate_link_targets(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            links=canonical_links,
        )
        encoded = encode_version_content(
            kind=identity.kind,
            content=content,
            codec=preferred_payload_codec(state.temperature),
        )
        equivalent = await self.find_current_equivalent(
            session=uow.session,
            owner_agent_id=owner_agent_id,
            content_hash=encoded.content_hash,
        )
        if equivalent is not None and equivalent.experience_id != experience_id:
            raise _duplicate_experience()

        version_id = self._id_generator.new()
        before = snapshot_from_state_row(state)
        materialized = activation_at(
            ActivationInputs(
                importance=state.importance,
                confidence=state.confidence,
                access_count=state.access_count,
                access_strength=state.access_strength,
                strength_updated_at=state.strength_updated_at,
                last_accessed_at=state.last_accessed_at,
                created_at=identity.created_at,
            ),
            occurred_at,
            self._lifecycle_config,
        )
        after = ExperienceStateSnapshotV1.model_validate(
            {
                **before.model_dump(mode="python"),
                "current_version_id": version_id,
                "current_content_hash": encoded.content_hash,
                "access_strength": materialized.decayed_strength,
                "strength_updated_at": occurred_at,
                "activation_score": materialized.score,
            }
        )
        version_number = previous_version.version_number + 1
        self._repository.add_version(
            session=uow.session,
            experience_id=experience_id,
            version_id=version_id,
            version_number=version_number,
            supersedes_version_id=previous_version.version_id,
            content=content,
            content_hash=encoded.content_hash,
            created_at=occurred_at,
        )
        await uow.session.flush()
        self._repository.add_payload(
            session=uow.session,
            version_id=version_id,
            codec=encoded.codec,
            payload=encoded.payload,
            payload_hash=encoded.payload_hash,
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        stored = await uow.append_events(
            command,
            [
                PendingEvent(
                    aggregate_type="experience",
                    aggregate_id=experience_id,
                    event_type=ExperienceVersionCreatedV1.event_type,
                    payload=ExperienceVersionCreatedV1(
                        schema_version=1,
                        experience_id=experience_id,
                        version_id=version_id,
                        version_number=version_number,
                        supersedes_version_id=previous_version.version_id,
                        links=canonical_links,
                        before=before,
                        after=after,
                    ),
                    actor_agent_id=actor_agent_id,
                    occurred_at=occurred_at,
                )
            ],
        )
        self._repository.add_links(
            session=uow.session,
            source_experience_id=experience_id,
            source_version_id=version_id,
            source_event_id=stored[0].event_id,
            links=canonical_links,
        )
        await uow.session.flush()
        return ExperienceCreation(
            experience_id=experience_id,
            version_id=version_id,
            content_hash=encoded.content_hash,
        )


__all__ = [
    "ExperienceRepository",
    "ExperienceWriter",
    "decode_and_verify_version",
    "require_current_aggregate_head",
    "snapshot_from_state_row",
]
