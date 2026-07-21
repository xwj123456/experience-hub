"""Validation of authoritative source rows before projection replay."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.agents.events import AgentCreated
from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain.commands import CommandRequest
from experience_hub.domain.events import EventRegistry
from experience_hub.domain.values import TypedEvidence
from experience_hub.experiences.events import (
    ExperienceCreatedV1,
    ExperienceStateSnapshotV1,
    ExperienceVersionCreatedV1,
    is_valid_version_event_sequence,
)
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.reconcile_contracts import (
    PayloadReconcileReport,
)
from experience_hub.storage.tables import (
    AgentRow,
    DomainEventRow,
    ExperienceCapsuleRow,
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdempotencyRecordRow,
    InboxItemRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationSnapshotItemRow,
)

if TYPE_CHECKING:
    from experience_hub.inspiration.commands import (
        AdoptIdea,
        ArchiveIdea,
        RejectIdea,
    )
    from experience_hub.inspiration.events import (
        INSPIRATION_EVENT_AGGREGATE_TYPES,
        INSPIRATION_EVENT_TYPES,
        InspirationCompletedV1,
        InspirationFailedV1,
        InspirationIdeaAdoptedV1,
        InspirationIdeaAdoptedV2,
        InspirationIdeaArchivedV1,
        InspirationIdeaEvaluatedV1,
        InspirationIdeaGeneratedV1,
        InspirationIdeaRejectedV1,
        InspirationOperatorCompletedV1,
        InspirationOperatorFailedV1,
        InspirationRunFailureCode,
        InspirationSnapshotFrozenV1,
        InspirationStartedV1,
        InspirationTimedOutV1,
    )
    from experience_hub.inspiration.generators.openai_compatible import (
        validate_persisted_generator_configuration,
    )
    from experience_hub.inspiration.hashing import (
        NEAR_DUPLICATE_THRESHOLD,
        hash_idea_content,
        hash_mechanism,
        hash_snapshot,
        mechanism_similarity,
        snapshot_canonical_bytes,
        stable_evidence_key,
    )
    from experience_hub.inspiration.incubation import (
        AdoptionTransition,
        ClusterTransition,
        EvaluationTransition,
        IncubationCluster,
        IncubationMember,
        plan_adoption_transition,
        plan_evaluation_transition,
        plan_occurrence,
    )
    from experience_hub.inspiration.models import (
        MAX_SNAPSHOT_UTF8_BYTES,
        EvidenceSourceState,
        EvidenceSourceType,
        ExperienceVersionEvidenceReference,
        GeneratorKind,
        IdeaDraft,
        IdeaEvaluation,
        IdeaOwnerDecision,
        InspirationOperator,
        MechanismIncubation,
        MechanismMaturity,
        SnapshotEvidenceReference,
        SnapshotItem,
    )
    from experience_hub.inspiration.request_hashing import (
        adoption_command_request,
        decision_command_request,
        evaluation_command_request,
    )
    from experience_hub.inspiration.response_codec import (
        InspirationRunResponseV1,
    )
    from experience_hub.lifecycle.contracts import decode_lifecycle_result
    from experience_hub.retrieval.ranking import RetrievalMode


class SourceIntegrityError(RuntimeError):
    """Authoritative source state cannot safely be replayed."""

    code = "source_integrity_error"

    def __init__(
        self,
        message: str = "Authoritative source integrity validation failed",
        *,
        mismatch_key: str = "source_integrity",
    ) -> None:
        self.mismatch_key = mismatch_key
        super().__init__(f"{mismatch_key}: {message}")


class PayloadReconcileValidationError(SourceIntegrityError):
    """Reportable payload damage that still prevents runtime readiness."""

    def __init__(self, report: PayloadReconcileReport) -> None:
        if report.changed_count != 0 or report.error_count == 0:
            raise ValueError(
                "payload validation error requires a zero-change error report"
            )
        self.report = report
        first = report.errors[0]
        super().__init__(
            "Experience payload reconciliation preflight failed",
            mismatch_key=f"experience_version_payload:{first.version_id}",
        )


class SourceValidationHook(Protocol):
    name: str

    async def validate(self, session: AsyncSession) -> None: ...


class SourceValidator:
    """Validate the common event ledger and registered feature invariants."""

    def __init__(
        self,
        event_registry: EventRegistry,
        hooks: Iterable[SourceValidationHook] = (),
    ) -> None:
        self.event_registry = event_registry
        self._hooks: dict[str, SourceValidationHook] = {}
        for hook in hooks:
            self.register(hook)

    def register(self, hook: SourceValidationHook) -> None:
        if not hook.name or hook.name != hook.name.strip():
            raise ValueError("Source validator name must be a non-empty trimmed string")
        if hook.name in self._hooks:
            raise ValueError(f"Source validator {hook.name!r} is already registered")
        self._hooks[hook.name] = hook

    async def validate(self, session: AsyncSession) -> None:
        await self._validate_causation(session)
        await self._validate_foreign_keys(session)
        await self._validate_sequences(session)
        await self._validate_registered_events(session)
        for hook in self._hooks.values():
            await hook.validate(session)

    async def _validate_causation(self, session: AsyncSession) -> None:
        missing = await session.execute(
            text(
                "SELECT e.event_id FROM domain_events e "
                "LEFT JOIN idempotency_records r "
                "ON r.receipt_id = e.causation_id "
                "WHERE e.causation_id IS NULL OR r.receipt_id IS NULL "
                "ORDER BY e.event_id LIMIT 50"
            )
        )
        event_ids = tuple(int(value) for value in missing.scalars())
        if event_ids:
            raise SourceIntegrityError(
                f"Event causation receipt is missing for events {event_ids}"
            )

    async def _validate_foreign_keys(self, session: AsyncSession) -> None:
        result = await session.execute(text("PRAGMA foreign_key_check"))
        violations = list(result.all())
        if violations:
            raise SourceIntegrityError(
                f"SQLite foreign key violations exist: {violations[:50]}"
            )

    async def _validate_sequences(self, session: AsyncSession) -> None:
        mismatch = await session.execute(
            text(
                "SELECT aggregate_type, aggregate_id, sequence, expected "
                "FROM ("
                "SELECT aggregate_type, aggregate_id, sequence, "
                "row_number() OVER ("
                "PARTITION BY aggregate_type, aggregate_id ORDER BY event_id"
                ") AS expected "
                "FROM domain_events"
                ") WHERE sequence != expected "
                "ORDER BY aggregate_type, aggregate_id, expected LIMIT 50"
            )
        )
        rows = list(mismatch.mappings())
        if rows:
            raise SourceIntegrityError(
                f"Aggregate event sequence is not contiguous from one: {rows}"
            )

    async def _validate_registered_events(self, session: AsyncSession) -> None:
        rows = (
            await session.execute(
                select(
                    DomainEventRow.event_id,
                    DomainEventRow.event_type,
                    DomainEventRow.payload,
                ).order_by(DomainEventRow.event_id)
            )
        ).all()
        for event_id, event_type, payload in rows:
            try:
                self.event_registry.decode(event_type=event_type, payload=payload)
            except (TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    f"Could not decode registered event {event_id} ({event_type!r})"
                ) from error


_INSPIRATION_DEPENDENCIES_LOADED = False


def _load_inspiration_validation_dependencies() -> None:
    """Load inspiration-only code after storage initialization is complete."""
    global _INSPIRATION_DEPENDENCIES_LOADED
    global _INSPIRATION_OPERATOR_TYPES
    global _INSPIRATION_TERMINAL_TYPES

    if _INSPIRATION_DEPENDENCIES_LOADED:
        return
    modules = {
        "events": import_module("experience_hub.inspiration.events"),
        "commands": import_module("experience_hub.inspiration.commands"),
        "hashing": import_module("experience_hub.inspiration.hashing"),
        "generator_configuration": import_module(
            "experience_hub.inspiration.generators.openai_compatible"
        ),
        "incubation": import_module("experience_hub.inspiration.incubation"),
        "models": import_module("experience_hub.inspiration.models"),
        "response_codec": import_module("experience_hub.inspiration.response_codec"),
        "request_hashing": import_module("experience_hub.inspiration.request_hashing"),
        "lifecycle_contracts": import_module("experience_hub.lifecycle.contracts"),
        "retrieval": import_module("experience_hub.retrieval.ranking"),
    }
    exports = {
        "INSPIRATION_EVENT_AGGREGATE_TYPES": (
            modules["events"].INSPIRATION_EVENT_AGGREGATE_TYPES
        ),
        "INSPIRATION_EVENT_TYPES": modules["events"].INSPIRATION_EVENT_TYPES,
        "InspirationCompletedV1": modules["events"].InspirationCompletedV1,
        "InspirationFailedV1": modules["events"].InspirationFailedV1,
        "InspirationIdeaAdoptedV1": (modules["events"].InspirationIdeaAdoptedV1),
        "InspirationIdeaAdoptedV2": (modules["events"].InspirationIdeaAdoptedV2),
        "InspirationIdeaArchivedV1": (modules["events"].InspirationIdeaArchivedV1),
        "InspirationIdeaEvaluatedV1": (modules["events"].InspirationIdeaEvaluatedV1),
        "InspirationIdeaGeneratedV1": (modules["events"].InspirationIdeaGeneratedV1),
        "InspirationIdeaRejectedV1": (modules["events"].InspirationIdeaRejectedV1),
        "InspirationOperatorCompletedV1": (
            modules["events"].InspirationOperatorCompletedV1
        ),
        "InspirationOperatorFailedV1": (modules["events"].InspirationOperatorFailedV1),
        "InspirationRunFailureCode": (modules["events"].InspirationRunFailureCode),
        "InspirationSnapshotFrozenV1": (modules["events"].InspirationSnapshotFrozenV1),
        "InspirationStartedV1": modules["events"].InspirationStartedV1,
        "InspirationTimedOutV1": modules["events"].InspirationTimedOutV1,
        "AdoptIdea": modules["commands"].AdoptIdea,
        "ArchiveIdea": modules["commands"].ArchiveIdea,
        "RejectIdea": modules["commands"].RejectIdea,
        "hash_idea_content": modules["hashing"].hash_idea_content,
        "hash_mechanism": modules["hashing"].hash_mechanism,
        "hash_snapshot": modules["hashing"].hash_snapshot,
        "mechanism_similarity": modules["hashing"].mechanism_similarity,
        "NEAR_DUPLICATE_THRESHOLD": (modules["hashing"].NEAR_DUPLICATE_THRESHOLD),
        "snapshot_canonical_bytes": modules["hashing"].snapshot_canonical_bytes,
        "stable_evidence_key": modules["hashing"].stable_evidence_key,
        "validate_persisted_generator_configuration": (
            modules[
                "generator_configuration"
            ].validate_persisted_generator_configuration
        ),
        "AdoptionTransition": modules["incubation"].AdoptionTransition,
        "ClusterTransition": modules["incubation"].ClusterTransition,
        "EvaluationTransition": modules["incubation"].EvaluationTransition,
        "IncubationCluster": modules["incubation"].IncubationCluster,
        "IncubationMember": modules["incubation"].IncubationMember,
        "plan_adoption_transition": (modules["incubation"].plan_adoption_transition),
        "plan_evaluation_transition": (
            modules["incubation"].plan_evaluation_transition
        ),
        "plan_occurrence": modules["incubation"].plan_occurrence,
        "EvidenceSourceState": modules["models"].EvidenceSourceState,
        "EvidenceSourceType": modules["models"].EvidenceSourceType,
        "ExperienceVersionEvidenceReference": (
            modules["models"].ExperienceVersionEvidenceReference
        ),
        "GeneratorKind": modules["models"].GeneratorKind,
        "IdeaDraft": modules["models"].IdeaDraft,
        "IdeaEvaluation": modules["models"].IdeaEvaluation,
        "IdeaOwnerDecision": modules["models"].IdeaOwnerDecision,
        "InspirationOperator": modules["models"].InspirationOperator,
        "MAX_SNAPSHOT_UTF8_BYTES": modules["models"].MAX_SNAPSHOT_UTF8_BYTES,
        "MechanismIncubation": modules["models"].MechanismIncubation,
        "MechanismMaturity": modules["models"].MechanismMaturity,
        "SnapshotEvidenceReference": (modules["models"].SnapshotEvidenceReference),
        "SnapshotItem": modules["models"].SnapshotItem,
        "InspirationRunResponseV1": (
            modules["response_codec"].InspirationRunResponseV1
        ),
        "adoption_command_request": (
            modules["request_hashing"].adoption_command_request
        ),
        "decision_command_request": (
            modules["request_hashing"].decision_command_request
        ),
        "evaluation_command_request": (
            modules["request_hashing"].evaluation_command_request
        ),
        "decode_lifecycle_result": (
            modules["lifecycle_contracts"].decode_lifecycle_result
        ),
        "RetrievalMode": modules["retrieval"].RetrievalMode,
    }
    globals().update(exports)
    _INSPIRATION_TERMINAL_TYPES = (
        exports["InspirationCompletedV1"],
        exports["InspirationFailedV1"],
        exports["InspirationTimedOutV1"],
    )
    _INSPIRATION_OPERATOR_TYPES = (
        exports["InspirationOperatorCompletedV1"],
        exports["InspirationOperatorFailedV1"],
    )
    _INSPIRATION_DEPENDENCIES_LOADED = True


class AgentSourceValidator:
    """Prove a bijection between agent source rows and creation events."""

    name = "agents"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        agents = {
            str(row.agent_id): row
            for row in (
                await session.execute(select(AgentRow).order_by(AgentRow.agent_id))
            ).scalars()
        }
        events = (
            await session.execute(
                select(DomainEventRow)
                .where(DomainEventRow.event_type == AgentCreated.event_type)
                .order_by(DomainEventRow.event_id)
            )
        ).scalars()
        created_by_agent: dict[str, list[DomainEventRow]] = {}
        for event in events:
            try:
                payload = cast(
                    AgentCreated,
                    self._event_registry.decode(
                        event_type=event.event_type,
                        payload=event.payload,
                    ),
                )
            except (TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    f"Could not decode agent.created event {event.event_id}"
                ) from error
            agent_id = str(payload.agent_id)
            created_by_agent.setdefault(agent_id, []).append(event)
            row = agents.get(agent_id)
            if (
                row is None
                or event.aggregate_type != "agent"
                or str(event.aggregate_id) != agent_id
                or event.sequence != 1
                or row.name != payload.name
                or row.created_at != event.occurred_at
            ):
                raise SourceIntegrityError(
                    f"agent.created event {event.event_id} has no matching agent row"
                )

        unmatched = [
            agent_id
            for agent_id in sorted(set(agents) | set(created_by_agent))
            if len(created_by_agent.get(agent_id, ())) != 1 or agent_id not in agents
        ]
        if unmatched:
            raise SourceIntegrityError(
                "Every agent row requires exactly one matching agent.created event: "
                f"{unmatched[:50]}"
            )


def register_agent_source_validator(validator: SourceValidator) -> None:
    validator.register(AgentSourceValidator(validator.event_registry))


class ExperienceSourceValidator:
    """Prove correspondence between immutable experience source rows and events."""

    name = "experiences"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        identities = tuple(
            (
                await session.scalars(
                    select(ExperienceRow).order_by(ExperienceRow.experience_id)
                )
            ).all()
        )
        versions = tuple(
            (
                await session.scalars(
                    select(ExperienceVersionRow).order_by(
                        ExperienceVersionRow.experience_id,
                        ExperienceVersionRow.version_number,
                        ExperienceVersionRow.version_id,
                    )
                )
            ).all()
        )
        payloads = tuple(
            (
                await session.scalars(
                    select(ExperiencePayloadRow).order_by(
                        ExperiencePayloadRow.version_id
                    )
                )
            ).all()
        )
        links = tuple(
            (
                await session.scalars(
                    select(ExperienceLinkRow).order_by(
                        ExperienceLinkRow.source_version_id,
                        ExperienceLinkRow.target_experience_id,
                        ExperienceLinkRow.relation,
                    )
                )
            ).all()
        )
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.event_type.in_(
                            (
                                ExperienceCreatedV1.event_type,
                                ExperienceVersionCreatedV1.event_type,
                            )
                        )
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )

        identity_by_id = {row.experience_id: row for row in identities}
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]] = {}
        for version in versions:
            versions_by_experience.setdefault(version.experience_id, []).append(version)
        payload_by_version = {row.version_id: row for row in payloads}
        created_events: dict[
            UUID, list[tuple[DomainEventRow, ExperienceCreatedV1]]
        ] = {}
        version_events: dict[
            UUID, list[tuple[DomainEventRow, ExperienceVersionCreatedV1]]
        ] = {}
        for event in event_rows:
            try:
                decoded = self._event_registry.decode(
                    event_type=event.event_type,
                    payload=event.payload,
                )
            except (TypeError, ValueError) as error:
                raise SourceIntegrityError(
                    "Experience source event cannot be decoded",
                    mismatch_key=f"experience_event:{event.event_id}",
                ) from error
            if isinstance(decoded, ExperienceCreatedV1):
                created_events.setdefault(decoded.experience_id, []).append(
                    (event, decoded)
                )
            elif isinstance(decoded, ExperienceVersionCreatedV1):
                version_events.setdefault(decoded.version_id, []).append(
                    (event, decoded)
                )
            else:  # pragma: no cover - selected event names pin these types
                raise SourceIntegrityError(
                    "Experience source event decoded to the wrong schema",
                    mismatch_key=f"experience_event:{event.event_id}",
                )

        await self._validate_version_payloads(
            session=session,
            identity_by_id=identity_by_id,
            versions=versions,
            payload_by_version=payload_by_version,
        )
        self._validate_identities(
            identity_by_id=identity_by_id,
            versions_by_experience=versions_by_experience,
            created_events=created_events,
        )
        self._validate_version_sequences(
            identity_by_id=identity_by_id,
            versions_by_experience=versions_by_experience,
        )
        one_event_by_version = self._validate_version_events(
            identity_by_id=identity_by_id,
            versions=versions,
            version_events=version_events,
        )
        self._validate_creation_version_anchor(
            identity_by_id=identity_by_id,
            versions_by_experience=versions_by_experience,
            created_events=created_events,
            event_by_version=one_event_by_version,
        )
        self._validate_version_event_order(
            versions_by_experience=versions_by_experience,
            event_by_version=one_event_by_version,
        )
        self._validate_links(
            identity_by_id=identity_by_id,
            created_events=created_events,
            versions=versions,
            links=links,
            event_by_version=one_event_by_version,
        )
        self._validate_current_content_uniqueness(
            identity_by_id=identity_by_id,
            versions_by_experience=versions_by_experience,
        )

    @staticmethod
    def _validate_identities(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]],
        created_events: dict[UUID, list[tuple[DomainEventRow, ExperienceCreatedV1]]],
    ) -> None:
        all_ids = sorted(
            set(identity_by_id) | set(created_events),
            key=lambda value: value.bytes,
        )
        for experience_id in all_ids:
            key = f"experience_identity:{experience_id}"
            identity = identity_by_id.get(experience_id)
            events = created_events.get(experience_id, ())
            if identity is None or len(events) != 1:
                raise SourceIntegrityError(
                    "Experience identity requires exactly one creation event",
                    mismatch_key=key,
                )
            versions = versions_by_experience.get(experience_id, ())
            if not versions:
                raise SourceIntegrityError(
                    "Experience identity requires at least one version",
                    mismatch_key=key,
                )
            initial = min(
                versions,
                key=lambda row: (row.version_number, row.version_id.bytes),
            )
            event, payload = events[0]
            if (
                event.aggregate_type != "experience"
                or event.aggregate_id != experience_id
                or event.sequence != 1
                or event.occurred_at != identity.created_at
                or payload.experience_id != experience_id
                or payload.version_id != initial.version_id
                or payload.after.experience_id != experience_id
                or payload.after.owner_agent_id != identity.owner_agent_id
                or payload.after.current_version_id != initial.version_id
                or payload.after.current_content_hash != initial.content_hash
            ):
                raise SourceIntegrityError(
                    "Experience identity does not match its creation event",
                    mismatch_key=key,
                )

    @staticmethod
    def _validate_version_sequences(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]],
    ) -> None:
        all_ids = sorted(
            set(identity_by_id) | set(versions_by_experience),
            key=lambda value: value.bytes,
        )
        for experience_id in all_ids:
            ordered = sorted(
                versions_by_experience.get(experience_id, ()),
                key=lambda row: (row.version_number, row.version_id.bytes),
            )
            valid = bool(ordered)
            previous: ExperienceVersionRow | None = None
            for expected, version in enumerate(ordered, start=1):
                if (
                    version.version_number != expected
                    or (expected == 1 and version.supersedes_version_id is not None)
                    or (
                        expected > 1
                        and (
                            previous is None
                            or version.supersedes_version_id != previous.version_id
                        )
                    )
                ):
                    valid = False
                    break
                previous = version
            if not valid:
                raise SourceIntegrityError(
                    "Experience versions must be contiguous from one "
                    "with an adjacent supersession chain",
                    mismatch_key=(f"experience_version_sequence:{experience_id}"),
                )

    @staticmethod
    async def _validate_version_payloads(
        *,
        session: AsyncSession,
        identity_by_id: dict[UUID, ExperienceRow],
        versions: tuple[ExperienceVersionRow, ...],
        payload_by_version: dict[UUID, ExperiencePayloadRow],
    ) -> None:
        # Local import avoids validation <-> repository import initialization cycle.
        from experience_hub.experiences.repository import (
            decode_and_verify_version,
        )

        version_ids = {version.version_id for version in versions}
        all_version_ids = tuple(version.version_id for version in versions) + tuple(
            sorted(
                set(payload_by_version) - version_ids,
                key=lambda value: value.bytes,
            )
        )
        version_by_id = {version.version_id: version for version in versions}
        reportable_payload_issue = False
        for version_id in all_version_ids:
            key = f"experience_version_payload:{version_id}"
            version = version_by_id.get(version_id)
            payload = payload_by_version.get(version_id)
            identity = (
                None if version is None else identity_by_id.get(version.experience_id)
            )
            if version is None or identity is None:
                raise SourceIntegrityError(
                    "Experience version, identity, and payload require "
                    "one-to-one correspondence",
                    mismatch_key=key,
                )
            if payload is None:
                reportable_payload_issue = True
                continue
            try:
                content = decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=payload,
                )
            except SourceIntegrityError:
                reportable_payload_issue = True
                continue
            authoritative_metadata = (
                ("tags", content.tags),
                ("applicability", content.applicability),
                ("evidence", content.evidence),
                ("falsifiers", content.falsifiers),
            )
            if any(
                getattr(version, field) != canonical_json_bytes(value)
                for field, value in authoritative_metadata
            ):
                raise SourceIntegrityError(
                    "Experience metadata arrays are not the exact canonical content",
                    mismatch_key=key,
                )
        if reportable_payload_issue:
            # Runtime import avoids validation <-> reconciliation initialization
            # cycles while keeping one diagnostic implementation for exact counts.
            from experience_hub.experiences.reconcile import PayloadReconciler

            report = await PayloadReconciler().diagnose(session)
            raise PayloadReconcileValidationError(report)

    @staticmethod
    def _validate_version_events(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        versions: tuple[ExperienceVersionRow, ...],
        version_events: dict[
            UUID, list[tuple[DomainEventRow, ExperienceVersionCreatedV1]]
        ],
    ) -> dict[UUID, tuple[DomainEventRow, ExperienceVersionCreatedV1]]:
        version_by_id = {row.version_id: row for row in versions}
        one_event_by_version: dict[
            UUID, tuple[DomainEventRow, ExperienceVersionCreatedV1]
        ] = {}
        all_version_ids = sorted(
            set(version_by_id) | set(version_events),
            key=lambda value: value.bytes,
        )
        for version_id in all_version_ids:
            key = f"experience_version_event:{version_id}"
            version = version_by_id.get(version_id)
            events = version_events.get(version_id, ())
            if version is None or len(events) != 1:
                raise SourceIntegrityError(
                    "Experience version requires exactly one version-created event",
                    mismatch_key=key,
                )
            identity = identity_by_id.get(version.experience_id)
            event, payload = events[0]
            if (
                identity is None
                or event.aggregate_type != "experience"
                or event.aggregate_id != version.experience_id
                or event.occurred_at != version.created_at
                or not is_valid_version_event_sequence(
                    version_number=version.version_number,
                    aggregate_sequence=event.sequence,
                )
                or payload.experience_id != version.experience_id
                or payload.version_id != version.version_id
                or payload.version_number != version.version_number
                or payload.supersedes_version_id != version.supersedes_version_id
                or payload.after.owner_agent_id != identity.owner_agent_id
                or payload.after.current_version_id != version.version_id
                or payload.after.current_content_hash != version.content_hash
            ):
                raise SourceIntegrityError(
                    "Experience version row does not match its explaining event",
                    mismatch_key=key,
                )
            one_event_by_version[version_id] = (event, payload)
        return one_event_by_version

    @staticmethod
    def _validate_creation_version_anchor(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]],
        created_events: dict[UUID, list[tuple[DomainEventRow, ExperienceCreatedV1]]],
        event_by_version: dict[UUID, tuple[DomainEventRow, ExperienceVersionCreatedV1]],
    ) -> None:
        for experience_id in sorted(identity_by_id, key=lambda value: value.bytes):
            identity = identity_by_id[experience_id]
            initial = min(
                versions_by_experience[experience_id],
                key=lambda row: (row.version_number, row.version_id.bytes),
            )
            creation_event = created_events[experience_id][0][0]
            version_event = event_by_version[initial.version_id][0]
            if (
                identity.created_at != initial.created_at
                or identity.created_at != creation_event.occurred_at
                or identity.created_at != version_event.occurred_at
                or creation_event.event_id >= version_event.event_id
                or creation_event.sequence != 1
                or version_event.sequence != 2
            ):
                raise SourceIntegrityError(
                    "Experience identity, v1 row, and creation events "
                    "do not share one creation anchor",
                    mismatch_key=f"experience_identity:{experience_id}",
                )

    @staticmethod
    def _validate_version_event_order(
        *,
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]],
        event_by_version: dict[UUID, tuple[DomainEventRow, ExperienceVersionCreatedV1]],
    ) -> None:
        for experience_id in sorted(
            versions_by_experience,
            key=lambda value: value.bytes,
        ):
            ordered = sorted(
                versions_by_experience[experience_id],
                key=lambda row: (row.version_number, row.version_id.bytes),
            )
            previous_event: DomainEventRow | None = None
            for version in ordered:
                event = event_by_version[version.version_id][0]
                if previous_event is not None and (
                    event.event_id <= previous_event.event_id
                    or event.sequence <= previous_event.sequence
                    or event.occurred_at < previous_event.occurred_at
                ):
                    raise SourceIntegrityError(
                        "Experience version events are not monotonic by version number",
                        mismatch_key=(f"experience_version_event:{version.version_id}"),
                    )
                previous_event = event

    @staticmethod
    def _validate_links(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        created_events: dict[UUID, list[tuple[DomainEventRow, ExperienceCreatedV1]]],
        versions: tuple[ExperienceVersionRow, ...],
        links: tuple[ExperienceLinkRow, ...],
        event_by_version: dict[UUID, tuple[DomainEventRow, ExperienceVersionCreatedV1]],
    ) -> None:
        links_by_version: dict[UUID, list[ExperienceLinkRow]] = {}
        for link in links:
            links_by_version.setdefault(link.source_version_id, []).append(link)
        version_by_id = {version.version_id: version for version in versions}
        all_version_ids = sorted(
            set(version_by_id) | set(links_by_version),
            key=lambda value: value.bytes,
        )
        for version_id in all_version_ids:
            key = f"experience_links:{version_id}"
            version = version_by_id.get(version_id)
            event_pair = event_by_version.get(version_id)
            if version is None or event_pair is None:
                raise SourceIntegrityError(
                    "Experience link requires a matching version event",
                    mismatch_key=key,
                )
            event, payload = event_pair
            actual = tuple(
                sorted(
                    (
                        (link.target_experience_id, link.relation)
                        for link in links_by_version.get(version_id, ())
                    ),
                    key=lambda item: (item[0].bytes, item[1].value),
                )
            )
            expected = tuple(
                (link.target_experience_id, link.relation) for link in payload.links
            )
            rows_match_event = all(
                link.source_experience_id == version.experience_id
                and link.source_event_id == event.event_id
                for link in links_by_version.get(version_id, ())
            )
            source_identity = identity_by_id.get(version.experience_id)
            links_have_valid_targets = source_identity is not None
            for link in links_by_version.get(version_id, ()):
                target_identity = identity_by_id.get(link.target_experience_id)
                target_created = created_events.get(
                    link.target_experience_id,
                    (),
                )
                if (
                    target_identity is None
                    or source_identity is None
                    or target_identity.owner_agent_id != source_identity.owner_agent_id
                    or len(target_created) != 1
                    or target_created[0][0].event_id >= event.event_id
                ):
                    links_have_valid_targets = False
                    break
            if (
                actual != expected
                or not rows_match_event
                or not links_have_valid_targets
            ):
                raise SourceIntegrityError(
                    "Experience links do not exactly match their version event",
                    mismatch_key=key,
                )

    @staticmethod
    def _validate_current_content_uniqueness(
        *,
        identity_by_id: dict[UUID, ExperienceRow],
        versions_by_experience: dict[UUID, list[ExperienceVersionRow]],
    ) -> None:
        current: dict[tuple[UUID, str], list[UUID]] = {}
        for experience_id in sorted(identity_by_id, key=lambda value: value.bytes):
            identity = identity_by_id[experience_id]
            versions = versions_by_experience.get(experience_id, ())
            if not versions:
                continue
            latest = max(
                versions,
                key=lambda row: (row.version_number, row.version_id.bytes),
            )
            current.setdefault(
                (identity.owner_agent_id, latest.content_hash),
                [],
            ).append(experience_id)
        for (owner_agent_id, content_hash), experience_ids in sorted(
            current.items(),
            key=lambda item: (
                item[0][0].bytes,
                item[0][1],
            ),
        ):
            if len(experience_ids) > 1:
                ordered = tuple(sorted(experience_ids, key=lambda value: value.bytes))
                raise SourceIntegrityError(
                    f"Current semantic content is duplicated by {ordered}",
                    mismatch_key=(
                        f"experience_current_content:{owner_agent_id}:{content_hash}"
                    ),
                )


def register_experience_source_validator(validator: SourceValidator) -> None:
    validator.register(ExperienceSourceValidator(validator.event_registry))


_INSPIRATION_TERMINAL_TYPES: tuple[type[Any], ...] = ()
_INSPIRATION_OPERATOR_TYPES: tuple[type[Any], ...] = ()
_NONCANDIDATE_RETENTION = timedelta(days=180)
_CANDIDATE_RETENTION = timedelta(days=365)


@dataclass(frozen=True, slots=True)
class _InspirationEvent:
    row: DomainEventRow
    payload: Any


@dataclass(slots=True)
class _IdeaSourceState:
    idea: InspirationIdeaRow
    run: InspirationRunRow
    generated: InspirationIdeaGeneratedV1
    decision: IdeaOwnerDecision
    last_signal_at: datetime
    last_event_at: datetime
    latest_evaluations: dict[UUID, InspirationIdeaEvaluatedV1]


def _inspiration_error(
    message: str,
    *,
    mismatch_key: str,
) -> SourceIntegrityError:
    return SourceIntegrityError(message, mismatch_key=mismatch_key)


def _canonical_value(
    raw: bytes,
    *,
    label: str,
    mismatch_key: str,
) -> Any:
    try:
        value = json.loads(raw)
        if canonical_json_bytes(value) != raw:
            raise ValueError
    except (TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise _inspiration_error(
            f"{label} is not exact canonical JSON",
            mismatch_key=mismatch_key,
        ) from error
    return value


def _canonical_string_tuple(
    raw: bytes,
    *,
    label: str,
    mismatch_key: str,
    nonempty: bool,
) -> tuple[str, ...]:
    value = _canonical_value(
        raw,
        label=label,
        mismatch_key=mismatch_key,
    )
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not isinstance(item, str) for item in value)
    ):
        raise _inspiration_error(
            f"{label} must be a canonical string array",
            mismatch_key=mismatch_key,
        )
    return tuple(value)


def _canonical_semantic_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    by_encoding = {canonical_json_bytes(value): value for value in values}
    return tuple(by_encoding[key] for key in sorted(by_encoding))


async def _experience_state_before_inspiration_event(
    session: AsyncSession,
    *,
    experience_id: UUID,
    event_id: int,
    mismatch_key: str,
) -> tuple[ExperienceStateSnapshotV1, DomainEventRow]:
    row = await session.scalar(
        select(DomainEventRow)
        .where(
            DomainEventRow.aggregate_type == "experience",
            DomainEventRow.aggregate_id == experience_id,
            DomainEventRow.event_id < event_id,
        )
        .order_by(DomainEventRow.event_id.desc())
        .limit(1)
    )
    if row is None:
        raise _inspiration_error(
            "Experience has no historical state before the inspiration event",
            mismatch_key=mismatch_key,
        )
    try:
        document = json.loads(row.payload)
        if (
            not isinstance(document, dict)
            or canonical_json_bytes(document) != row.payload
            or not isinstance(document.get("after"), dict)
        ):
            raise ValueError("experience event has no canonical after state")
        state = ExperienceStateSnapshotV1.model_validate_json(
            canonical_json_bytes(document["after"])
        )
    except (TypeError, ValueError) as error:
        raise _inspiration_error(
            "Historical experience state is invalid",
            mismatch_key=mismatch_key,
        ) from error
    return state, row


def _cluster_transition(payload: InspirationIdeaGeneratedV1) -> ClusterTransition:
    return ClusterTransition(
        cluster_id=payload.cluster_id,
        canonical_mechanism_hash=payload.canonical_mechanism_hash,
        member_hashes_before=payload.member_hashes_before,
        member_hashes_after=payload.member_hashes_after,
        occurrence_count_before=payload.occurrence_count_before,
        occurrence_count_after=payload.occurrence_count_after,
        distinct_snapshot_count_before=payload.distinct_snapshot_count_before,
        distinct_snapshot_count_after=payload.distinct_snapshot_count_after,
        distinct_adopter_count_before=payload.distinct_adopter_count_before,
        distinct_adopter_count_after=payload.distinct_adopter_count_after,
        supported_count_before=payload.supported_count_before,
        supported_count_after=payload.supported_count_after,
        refuted_count_before=payload.refuted_count_before,
        refuted_count_after=payload.refuted_count_after,
        maturity_before=payload.maturity_before,
        maturity_after=payload.maturity_after,
        candidate_since_before=payload.candidate_since_before,
        candidate_since_after=payload.candidate_since_after,
        last_signal_at_before=payload.last_signal_at_before,
        last_signal_at_after=payload.last_signal_at_after,
    )


def _state_after_occurrence(
    transition: ClusterTransition,
) -> MechanismIncubation:
    return MechanismIncubation(
        cluster_id=transition.cluster_id,
        canonical_mechanism_hash=transition.canonical_mechanism_hash,
        member_hashes=transition.member_hashes_after,
        occurrence_count=transition.occurrence_count_after,
        distinct_snapshot_count=transition.distinct_snapshot_count_after,
        distinct_adopter_count=transition.distinct_adopter_count_after,
        supported_count=transition.supported_count_after,
        refuted_count=transition.refuted_count_after,
        maturity=transition.maturity_after,
        candidate_since=transition.candidate_since_after,
        last_signal_at=transition.last_signal_at_after,
    )


def _state_after_evaluation(
    state: MechanismIncubation,
    payload: InspirationIdeaEvaluatedV1,
) -> MechanismIncubation:
    return MechanismIncubation(
        cluster_id=state.cluster_id,
        canonical_mechanism_hash=state.canonical_mechanism_hash,
        member_hashes=state.member_hashes,
        occurrence_count=state.occurrence_count,
        distinct_snapshot_count=state.distinct_snapshot_count,
        distinct_adopter_count=state.distinct_adopter_count,
        supported_count=payload.supported_count_after,
        refuted_count=payload.refuted_count_after,
        maturity=payload.maturity_after,
        candidate_since=payload.candidate_since_after,
        last_signal_at=payload.last_signal_at_after,
    )


def _state_after_adoption(
    state: MechanismIncubation,
    payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2,
) -> MechanismIncubation:
    return MechanismIncubation(
        cluster_id=state.cluster_id,
        canonical_mechanism_hash=state.canonical_mechanism_hash,
        member_hashes=state.member_hashes,
        occurrence_count=state.occurrence_count,
        distinct_snapshot_count=state.distinct_snapshot_count,
        distinct_adopter_count=payload.distinct_adopter_count_after,
        supported_count=state.supported_count,
        refuted_count=state.refuted_count,
        maturity=payload.maturity_after,
        candidate_since=payload.candidate_since_after,
        last_signal_at=payload.last_signal_at_after,
    )


class InspirationSourceValidator:
    """Prove inspiration immutable sources, receipts, and event histories."""

    name = "inspiration"

    def __init__(self, event_registry: EventRegistry) -> None:
        self._event_registry = event_registry

    async def validate(self, session: AsyncSession) -> None:
        _load_inspiration_validation_dependencies()
        runs = tuple(
            (
                await session.scalars(
                    select(InspirationRunRow).order_by(InspirationRunRow.run_id)
                )
            ).all()
        )
        snapshots = tuple(
            (
                await session.scalars(
                    select(InspirationSnapshotItemRow).order_by(
                        InspirationSnapshotItemRow.run_id,
                        InspirationSnapshotItemRow.rank,
                        InspirationSnapshotItemRow.snapshot_item_id,
                    )
                )
            ).all()
        )
        ideas = tuple(
            (
                await session.scalars(
                    select(InspirationIdeaRow).order_by(InspirationIdeaRow.idea_id)
                )
            ).all()
        )
        occurrences = tuple(
            (
                await session.scalars(
                    select(IdeaOccurrenceRow).order_by(IdeaOccurrenceRow.occurrence_id)
                )
            ).all()
        )
        adoptions = tuple(
            (
                await session.scalars(
                    select(IdeaAdoptionRecordRow).order_by(
                        IdeaAdoptionRecordRow.adoption_id
                    )
                )
            ).all()
        )
        receipts = {
            row.receipt_id: row
            for row in (
                await session.scalars(
                    select(IdempotencyRecordRow).order_by(
                        IdempotencyRecordRow.receipt_id
                    )
                )
            ).all()
        }
        event_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type.in_(INSPIRATION_EVENT_TYPES))
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        event_ids_by_causation: dict[UUID, set[int]] = {}
        for event_id, causation_id in (
            await session.execute(
                select(
                    DomainEventRow.event_id,
                    DomainEventRow.causation_id,
                ).order_by(DomainEventRow.event_id)
            )
        ).all():
            event_ids_by_causation.setdefault(causation_id, set()).add(event_id)
        events = self._decode_events(event_rows)
        run_by_id = {row.run_id: row for row in runs}
        snapshot_by_run: dict[UUID, list[InspirationSnapshotItemRow]] = {}
        for row in snapshots:
            snapshot_by_run.setdefault(row.run_id, []).append(row)
        idea_by_id = {row.idea_id: row for row in ideas}
        occurrence_by_id = {row.occurrence_id: row for row in occurrences}
        adoption_by_id = {row.adoption_id: row for row in adoptions}

        frozen_by_run = await self._validate_snapshots(
            session=session,
            run_by_id=run_by_id,
            snapshot_by_run=snapshot_by_run,
            events=events,
        )
        generated_by_idea = self._validate_idea_sources(
            run_by_id=run_by_id,
            idea_by_id=idea_by_id,
            occurrence_by_id=occurrence_by_id,
            frozen_by_run=frozen_by_run,
            events=events,
        )
        self._validate_runs(
            run_by_id=run_by_id,
            idea_by_id=idea_by_id,
            generated_by_idea=generated_by_idea,
            frozen_by_run=frozen_by_run,
            events=events,
            receipts=receipts,
            event_ids_by_causation=event_ids_by_causation,
        )
        await self._validate_idea_histories(
            session=session,
            run_by_id=run_by_id,
            idea_by_id=idea_by_id,
            occurrence_by_id=occurrence_by_id,
            adoption_by_id=adoption_by_id,
            frozen_by_run=frozen_by_run,
            events=events,
            receipts=receipts,
        )

    def _decode_events(
        self,
        rows: tuple[DomainEventRow, ...],
    ) -> tuple[_InspirationEvent, ...]:
        retained: list[_InspirationEvent] = []
        for row in rows:
            key = f"inspiration_event:{row.event_id}"
            try:
                payload = self._event_registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                )
            except (TypeError, ValueError) as error:
                raise _inspiration_error(
                    "Inspiration event cannot be decoded",
                    mismatch_key=key,
                ) from error
            expected_aggregate = INSPIRATION_EVENT_AGGREGATE_TYPES.get(row.event_type)
            payload_id = (
                getattr(payload, "run_id", None)
                if expected_aggregate == "inspiration_run"
                else getattr(payload, "idea_id", None)
            )
            if (
                expected_aggregate is None
                or row.aggregate_type != expected_aggregate
                or row.aggregate_id != payload_id
            ):
                raise _inspiration_error(
                    "Inspiration event aggregate does not match its payload",
                    mismatch_key=key,
                )
            retained.append(_InspirationEvent(row=row, payload=payload))
        return tuple(retained)

    @staticmethod
    async def _validate_capsule_snapshot_source(
        *,
        session: AsyncSession,
        row: InspirationSnapshotItemRow,
        run: InspirationRunRow,
        snapshot_event: _InspirationEvent,
        applicability: tuple[str, ...],
        tags: tuple[str, ...],
        falsifiers: tuple[str, ...],
        mismatch_key: str,
    ) -> None:
        capsule = await session.get(ExperienceCapsuleRow, row.source_id)
        inbox = await session.scalar(
            select(InboxItemRow).where(
                InboxItemRow.recipient_agent_id == run.owner_agent_id,
                InboxItemRow.capsule_id == row.source_id,
            )
        )
        capsule_event = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "capsule",
                DomainEventRow.aggregate_id == row.source_id,
                DomainEventRow.event_id < snapshot_event.row.event_id,
                DomainEventRow.event_type.in_(
                    ("capsule.published", "capsule.retracted")
                ),
            )
            .order_by(DomainEventRow.event_id.desc())
            .limit(1)
        )
        inbox_event = (
            None
            if inbox is None
            else await session.scalar(
                select(DomainEventRow)
                .where(
                    DomainEventRow.aggregate_type == "inbox_item",
                    DomainEventRow.aggregate_id == inbox.item_id,
                    DomainEventRow.event_id < snapshot_event.row.event_id,
                    DomainEventRow.event_type.in_(
                        (
                            "capsule.received",
                            "capsule.adopted",
                            "capsule.rejected",
                        )
                    ),
                )
                .order_by(DomainEventRow.event_id.desc())
                .limit(1)
            )
        )
        if (
            capsule is None
            or inbox is None
            or capsule_event is None
            or inbox_event is None
        ):
            raise _inspiration_error(
                "Snapshot capsule has no owner-visible historical source",
                mismatch_key=mismatch_key,
            )
        published = _canonical_value(
            capsule_event.payload,
            label="capsule publication event",
            mismatch_key=mismatch_key,
        )
        received = _canonical_value(
            inbox_event.payload,
            label="capsule receipt event",
            mismatch_key=mismatch_key,
        )
        if (
            not run.include_inbox
            or row.source_version_id != capsule.source_version_id
            or row.content_hash != capsule.source_content_hash
            or row.source_trust != 0.25
            or capsule.created_at > run.created_at
            or capsule.expires_at <= run.created_at
            or row.summary != capsule.summary
            or row.mechanism != capsule.mechanism
            or capsule.tags != canonical_json_bytes(tags)
            or capsule.applicability != canonical_json_bytes(applicability)
            or capsule.falsifiers != canonical_json_bytes(falsifiers)
            or not capsule.body.startswith(row.excerpt)
            or inbox.recipient_agent_id != run.owner_agent_id
            or inbox.capsule_id != capsule.capsule_id
            or capsule_event.event_type != "capsule.published"
            or capsule_event.sequence != 1
            or capsule_event.occurred_at != capsule.created_at
            or capsule_event.actor_agent_id != capsule.publisher_agent_id
            or inbox_event.event_type != "capsule.received"
            or inbox_event.sequence != 1
            or inbox_event.occurred_at != capsule.created_at
            or inbox_event.actor_agent_id != capsule.publisher_agent_id
            or inbox_event.causation_id != capsule_event.causation_id
            or inbox_event.event_id <= capsule_event.event_id
            or not isinstance(published, dict)
            or published.get("capsule_id") != str(capsule.capsule_id)
            or published.get("topic_id") != str(capsule.topic_id)
            or published.get("source_experience_id")
            != str(capsule.source_experience_id)
            or published.get("publisher_agent_id") != str(capsule.publisher_agent_id)
            or published.get("source_version_id") != str(capsule.source_version_id)
            or published.get("capsule_hash") != capsule.capsule_hash
            or published.get("root_fingerprint") != capsule.root_fingerprint
            or published.get("status_after") != "active"
            or not isinstance(received, dict)
            or received.get("item_id") != str(inbox.item_id)
            or received.get("capsule_id") != str(capsule.capsule_id)
            or received.get("recipient_agent_id") != str(run.owner_agent_id)
            or received.get("state_after") != "pending"
        ):
            raise _inspiration_error(
                "Snapshot capsule was not available to the run owner at freeze time",
                mismatch_key=mismatch_key,
            )

    async def _validate_snapshots(
        self,
        *,
        session: AsyncSession,
        run_by_id: dict[UUID, InspirationRunRow],
        snapshot_by_run: dict[UUID, list[InspirationSnapshotItemRow]],
        events: tuple[_InspirationEvent, ...],
    ) -> dict[UUID, tuple[SnapshotItem, ...]]:
        event_by_run: dict[UUID, list[_InspirationEvent]] = {}
        for event in events:
            if isinstance(event.payload, InspirationSnapshotFrozenV1):
                event_by_run.setdefault(event.payload.run_id, []).append(event)
        all_run_ids = sorted(
            set(run_by_id) | set(snapshot_by_run) | set(event_by_run),
            key=lambda value: value.bytes,
        )
        retained: dict[UUID, tuple[SnapshotItem, ...]] = {}
        for run_id in all_run_ids:
            key = f"inspiration_snapshot:{run_id}"
            run = run_by_id.get(run_id)
            rows = tuple(
                sorted(
                    snapshot_by_run.get(run_id, ()),
                    key=lambda row: (row.rank, row.snapshot_item_id.bytes),
                )
            )
            snapshot_events = event_by_run.get(run_id, [])
            if run is None:
                raise _inspiration_error(
                    "Snapshot source or event has no inspiration run",
                    mismatch_key=key,
                )
            if rows and len(snapshot_events) != 1:
                raise _inspiration_error(
                    "Frozen snapshot rows require exactly one snapshot event",
                    mismatch_key=key,
                )
            if not rows and len(snapshot_events) > 1:
                raise _inspiration_error(
                    "A run may have at most one snapshot event",
                    mismatch_key=key,
                )
            expected_ranks = tuple(range(1, len(rows) + 1))
            if tuple(row.rank for row in rows) != expected_ranks:
                raise _inspiration_error(
                    "Snapshot ranks must be contiguous in canonical order",
                    mismatch_key=key,
                )
            items: list[SnapshotItem] = []
            for row in rows:
                item_key = f"inspiration_snapshot_item:{row.snapshot_item_id}"
                applicability = _canonical_string_tuple(
                    row.applicability,
                    label="snapshot applicability",
                    mismatch_key=item_key,
                    nonempty=False,
                )
                tags = _canonical_string_tuple(
                    row.tags,
                    label="snapshot tags",
                    mismatch_key=item_key,
                    nonempty=False,
                )
                falsifiers = _canonical_string_tuple(
                    row.falsifiers,
                    label="snapshot falsifiers",
                    mismatch_key=item_key,
                    nonempty=False,
                )
                try:
                    source_type = EvidenceSourceType(row.source_type)
                    source_state = EvidenceSourceState(row.source_state)
                    item = SnapshotItem(
                        snapshot_item_id=row.snapshot_item_id,
                        stable_evidence_key=row.stable_evidence_key,
                        run_id=row.run_id,
                        source_type=source_type,
                        source_id=row.source_id,
                        source_version_id=row.source_version_id,
                        source_state=source_state,
                        source_trust=row.source_trust,
                        rank=row.rank,
                        summary=row.summary,
                        mechanism=row.mechanism,
                        applicability=applicability,
                        tags=tags,
                        falsifiers=falsifiers,
                        excerpt=row.excerpt,
                        content_hash=row.content_hash,
                        captured_at=run.created_at,
                    )
                    expected_key = stable_evidence_key(
                        source_type=source_type,
                        source_id=row.source_id,
                        source_version_id=row.source_version_id,
                        content_hash=row.content_hash,
                    )
                except (TypeError, ValueError) as error:
                    raise _inspiration_error(
                        "Snapshot item cannot be reconstructed",
                        mismatch_key=item_key,
                    ) from error
                if row.stable_evidence_key != expected_key:
                    raise _inspiration_error(
                        "Snapshot stable evidence key does not match its source",
                        mismatch_key=item_key,
                    )
                if source_type is EvidenceSourceType.EXPERIENCE:
                    identity = await session.get(ExperienceRow, row.source_id)
                    version = await session.get(
                        ExperienceVersionRow,
                        row.source_version_id,
                    )
                    payload_row = await session.get(
                        ExperiencePayloadRow,
                        row.source_version_id,
                    )
                    if (
                        identity is None
                        or version is None
                        or payload_row is None
                        or identity.owner_agent_id != run.owner_agent_id
                        or version.experience_id != identity.experience_id
                        or version.content_hash != row.content_hash
                        or identity.created_at > run.created_at
                        or version.created_at > run.created_at
                    ):
                        raise _inspiration_error(
                            "Snapshot experience source is not an owned "
                            "immutable version available at freeze time",
                            mismatch_key=item_key,
                        )
                    from experience_hub.experiences.repository import (
                        decode_and_verify_version,
                    )

                    try:
                        content = decode_and_verify_version(
                            identity=identity,
                            version=version,
                            payload=payload_row,
                        )
                    except (TypeError, ValueError, RuntimeError) as error:
                        raise _inspiration_error(
                            "Snapshot experience source cannot be decoded",
                            mismatch_key=item_key,
                        ) from error
                    if (
                        row.summary != content.summary
                        or row.mechanism != content.mechanism
                        or tags != content.tags
                        or applicability != content.applicability
                        or falsifiers != content.falsifiers
                        or not content.body.startswith(row.excerpt)
                    ):
                        raise _inspiration_error(
                            "Snapshot experience content does not match its "
                            "immutable version",
                            mismatch_key=item_key,
                        )
                    (
                        historical_state,
                        historical_event,
                    ) = await _experience_state_before_inspiration_event(
                        session,
                        experience_id=identity.experience_id,
                        event_id=snapshot_events[0].row.event_id,
                        mismatch_key=item_key,
                    )
                    if (
                        historical_state.experience_id != identity.experience_id
                        or historical_state.owner_agent_id != run.owner_agent_id
                        or historical_state.current_version_id != version.version_id
                        or historical_state.current_content_hash != version.content_hash
                        or historical_state.temperature is Temperature.ARCHIVED
                        or source_state.value != historical_state.temperature.value
                        or row.source_trust != historical_state.source_trust
                        or historical_event.occurred_at > run.created_at
                    ):
                        raise _inspiration_error(
                            "Snapshot experience state does not match the "
                            "owner-visible current source at freeze time",
                            mismatch_key=item_key,
                        )
                else:
                    assert len(snapshot_events) == 1
                    await self._validate_capsule_snapshot_source(
                        session=session,
                        row=row,
                        run=run,
                        snapshot_event=snapshot_events[0],
                        applicability=applicability,
                        tags=tags,
                        falsifiers=falsifiers,
                        mismatch_key=item_key,
                    )
                items.append(item)
            frozen = tuple(items)
            if len(snapshot_canonical_bytes(frozen)) > MAX_SNAPSHOT_UTF8_BYTES:
                raise _inspiration_error(
                    "Frozen snapshot exceeds its canonical byte budget",
                    mismatch_key=key,
                )
            retained[run_id] = frozen
            if not snapshot_events:
                continue
            event = snapshot_events[0]
            payload = cast(InspirationSnapshotFrozenV1, event.payload)
            expected_hash = hash_snapshot(frozen)
            if (
                event.row.aggregate_type != "inspiration_run"
                or event.row.aggregate_id != run_id
                or payload.run_id != run_id
                or payload.snapshot_item_ids
                != tuple(item.snapshot_item_id for item in frozen)
                or payload.snapshot_hash != expected_hash
                or event.row.actor_agent_id != run.owner_agent_id
            ):
                raise _inspiration_error(
                    "Snapshot event does not exactly match frozen rows",
                    mismatch_key=key,
                )
        return retained

    def _idea_draft(
        self,
        row: InspirationIdeaRow,
    ) -> IdeaDraft:
        key = f"inspiration_idea:{row.idea_id}"
        predictions = _canonical_string_tuple(
            row.predictions,
            label="idea predictions",
            mismatch_key=key,
            nonempty=True,
        )
        falsifiers = _canonical_string_tuple(
            row.falsifiers,
            label="idea falsifiers",
            mismatch_key=key,
            nonempty=True,
        )
        assumptions = _canonical_string_tuple(
            row.assumptions,
            label="idea assumptions",
            mismatch_key=key,
            nonempty=True,
        )
        raw_evidence = _canonical_value(
            row.evidence_references,
            label="idea evidence",
            mismatch_key=key,
        )
        if not isinstance(raw_evidence, list):
            raise _inspiration_error(
                "Idea evidence must be a canonical array",
                mismatch_key=key,
            )
        try:
            evidence = tuple(
                SnapshotEvidenceReference.model_validate_json(
                    canonical_json_bytes(value)
                )
                for value in raw_evidence
            )
            draft = IdeaDraft(
                title=row.title,
                hypothesis=row.hypothesis,
                mechanism=row.mechanism,
                predictions=predictions,
                falsifiers=falsifiers,
                assumptions=assumptions,
                proposed_test=row.proposed_test,
                evidence=evidence,
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Idea source cannot be reconstructed",
                mismatch_key=key,
            ) from error
        canonical_fields = (
            (row.predictions, draft.predictions),
            (row.falsifiers, draft.falsifiers),
            (row.assumptions, draft.assumptions),
            (row.evidence_references, draft.evidence),
        )
        if any(raw != canonical_json_bytes(value) for raw, value in canonical_fields):
            raise _inspiration_error(
                "Idea arrays are not their exact canonical semantic values",
                mismatch_key=key,
            )
        if any(
            values != _canonical_semantic_strings(values)
            for values in (predictions, falsifiers, assumptions)
        ):
            raise _inspiration_error(
                "Idea semantic string arrays are not sorted and unique",
                mismatch_key=key,
            )
        return draft

    def _validate_idea_sources(
        self,
        *,
        run_by_id: dict[UUID, InspirationRunRow],
        idea_by_id: dict[UUID, InspirationIdeaRow],
        occurrence_by_id: dict[UUID, IdeaOccurrenceRow],
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        events: tuple[_InspirationEvent, ...],
    ) -> dict[UUID, _InspirationEvent]:
        generated_by_idea: dict[UUID, list[_InspirationEvent]] = {}
        for event in events:
            if isinstance(event.payload, InspirationIdeaGeneratedV1):
                generated_by_idea.setdefault(
                    event.payload.idea_id,
                    [],
                ).append(event)
        occurrences_by_idea: dict[UUID, list[IdeaOccurrenceRow]] = {}
        for occurrence in occurrence_by_id.values():
            occurrences_by_idea.setdefault(occurrence.idea_id, []).append(occurrence)
        all_idea_ids = sorted(
            set(idea_by_id) | set(generated_by_idea) | set(occurrences_by_idea),
            key=lambda value: value.bytes,
        )
        retained: dict[UUID, _InspirationEvent] = {}
        for idea_id in all_idea_ids:
            key = f"inspiration_idea:{idea_id}"
            idea = idea_by_id.get(idea_id)
            generated_events = generated_by_idea.get(idea_id, ())
            occurrences = occurrences_by_idea.get(idea_id, ())
            if idea is None or len(generated_events) != 1 or len(occurrences) != 1:
                raise _inspiration_error(
                    "Idea, generated event, and occurrence require a bijection",
                    mismatch_key=key,
                )
            event = generated_events[0]
            payload = cast(InspirationIdeaGeneratedV1, event.payload)
            occurrence = occurrences[0]
            run = run_by_id.get(idea.run_id)
            frozen = frozen_by_run.get(idea.run_id)
            if run is None or frozen is None:
                raise _inspiration_error(
                    "Idea run or frozen evidence source is missing",
                    mismatch_key=key,
                )
            snapshot_events = tuple(
                item
                for item in events
                if isinstance(item.payload, InspirationSnapshotFrozenV1)
                and item.payload.run_id == idea.run_id
            )
            if len(snapshot_events) != 1:
                raise _inspiration_error(
                    "Generated idea requires one frozen snapshot event",
                    mismatch_key=key,
                )
            snapshot_payload = cast(
                InspirationSnapshotFrozenV1,
                snapshot_events[0].payload,
            )
            draft = self._idea_draft(idea)
            try:
                operator = InspirationOperator(idea.operator)
                expected_idea_hash = hash_idea_content(draft)
                expected_mechanism_hash = hash_mechanism(idea.mechanism)
            except (TypeError, ValueError) as error:
                raise _inspiration_error(
                    "Idea hashes cannot be reconstructed",
                    mismatch_key=key,
                ) from error
            frozen_by_id = {item.snapshot_item_id: item for item in frozen}
            evidence_resolves = all(
                reference.id in frozen_by_id
                and frozen_by_id[reference.id].stable_evidence_key
                == reference.stable_evidence_key
                for reference in draft.evidence
            )
            canonical_evidence = (
                tuple(
                    sorted(
                        draft.evidence,
                        key=lambda reference: (
                            frozen_by_id[reference.id].rank,
                            reference.stable_evidence_key,
                            reference.id.bytes,
                        ),
                    )
                )
                if evidence_resolves
                else ()
            )
            evidence_is_canonical = (
                evidence_resolves
                and len({reference.id for reference in draft.evidence})
                == len(draft.evidence)
                and len({reference.stable_evidence_key for reference in draft.evidence})
                == len(draft.evidence)
                and draft.evidence == canonical_evidence
            )
            if (
                idea.idea_content_hash != expected_idea_hash
                or idea.mechanism_hash != expected_mechanism_hash
                or not evidence_is_canonical
                or payload.idea_id != idea.idea_id
                or payload.run_id != idea.run_id
                or payload.owner_agent_id != run.owner_agent_id
                or payload.operator is not operator
                or payload.ordinal != idea.ordinal
                or payload.evidence != draft.evidence
                or payload.idea_content_hash != idea.idea_content_hash
                or payload.mechanism_hash != idea.mechanism_hash
                or payload.duplicate_relation != idea.duplicate_relation
                or payload.snapshot_hash != snapshot_payload.snapshot_hash
                or payload.occurrence_id != occurrence.occurrence_id
                or occurrence.idea_id != idea.idea_id
                or occurrence.run_id != idea.run_id
                or occurrence.owner_agent_id != run.owner_agent_id
                or occurrence.mechanism_hash != idea.mechanism_hash
                or occurrence.snapshot_hash != payload.snapshot_hash
                or occurrence.occurred_at != event.row.occurred_at
                or event.row.actor_agent_id != run.owner_agent_id
                or event.row.occurred_at != run.created_at
                or payload.last_signal_at_after != event.row.occurred_at
            ):
                raise _inspiration_error(
                    "Idea hashes, evidence, occurrence, or event anchor disagree",
                    mismatch_key=key,
                )
            retained[idea_id] = event
        if set(occurrence_by_id) != {
            cast(InspirationIdeaGeneratedV1, event.payload).occurrence_id
            for event in retained.values()
        }:
            raise _inspiration_error(
                "Occurrence identities do not match generated events",
                mismatch_key="inspiration_occurrences",
            )
        return retained

    @staticmethod
    def _run_operators(run: InspirationRunRow) -> tuple[InspirationOperator, ...]:
        key = f"inspiration_run:{run.run_id}"
        raw = _canonical_value(
            run.operators,
            label="run operators",
            mismatch_key=key,
        )
        configuration = _canonical_value(
            run.generator_configuration,
            label="generator configuration",
            mismatch_key=key,
        )
        if not isinstance(raw, list) or not isinstance(configuration, dict):
            raise _inspiration_error(
                "Run operators or generator configuration are invalid",
                mismatch_key=key,
            )
        try:
            operators = tuple(InspirationOperator(value) for value in raw)
            generator = GeneratorKind(run.generator_kind)
            RetrievalMode(run.mode)
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Run enum configuration is invalid",
                mismatch_key=key,
            ) from error
        canonical_order = tuple(
            operator
            for operator in (
                InspirationOperator.CAUSAL_GAP,
                InspirationOperator.COUNTERFACTUAL,
                InspirationOperator.DISTANT_ANALOGY,
            )
            if operator in operators
        )
        try:
            validated_configuration = validate_persisted_generator_configuration(
                kind=generator,
                configuration=configuration,
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Run generator configuration is unsafe or noncanonical",
                mismatch_key=key,
            ) from error
        if (
            not operators
            or operators != canonical_order
            or len(operators) != len(set(operators))
            or configuration != validated_configuration
        ):
            raise _inspiration_error(
                "Run operators or generator configuration are noncanonical",
                mismatch_key=key,
            )
        return operators

    @staticmethod
    def _start_request_hash(
        *,
        run: InspirationRunRow,
        idempotency_key: str,
    ) -> str:
        operators = InspirationSourceValidator._run_operators(run)
        try:
            return CommandRequest(
                caller_scope=f"agent:{run.owner_agent_id}",
                operation_scope="inspiration.run.start",
                idempotency_key=idempotency_key,
                method="POST",
                route_template="/v1/agents/{agent_id}/inspiration-runs",
                path_parameters={"agent_id": run.owner_agent_id},
                body={
                    "goal": run.goal,
                    "context": run.context or "",
                    "mode": run.mode,
                    "generator": run.generator_kind,
                    "operators": tuple(operator.value for operator in operators),
                    "include_inbox": run.include_inbox,
                    "branches_per_operator": run.branches_per_operator,
                    "output_tokens_per_operator": (run.output_tokens_per_operator),
                    "total_output_tokens": run.total_output_tokens,
                    "operator_timeout_seconds": (run.operator_timeout_seconds),
                    "global_timeout_seconds": run.global_timeout_seconds,
                },
            ).request_hash
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Run source cannot reconstruct its canonical start request",
                mismatch_key=f"inspiration_run:{run.run_id}",
            ) from error

    @staticmethod
    def _recovery_request_hash(
        *,
        run_id: UUID,
        idempotency_key: str,
    ) -> str:
        try:
            return CommandRequest(
                caller_scope="system:local",
                operation_scope="inspiration.run.recover",
                idempotency_key=idempotency_key,
                method="POST",
                route_template="/internal/inspiration-runs/{run_id}:recover",
                path_parameters={"run_id": run_id},
                body={"failure_code": "process_interrupted"},
            ).request_hash
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Run recovery request cannot be reconstructed",
                mismatch_key=f"inspiration_run:{run_id}",
            ) from error

    @staticmethod
    def _require_run_receipt(
        *,
        receipt: IdempotencyRecordRow | None,
        run: InspirationRunRow,
        completed: bool,
    ) -> IdempotencyRecordRow:
        key = f"inspiration_run:{run.run_id}"
        expected_state = "completed" if completed else "in_progress"
        expected_request_hash = (
            None
            if receipt is None
            else InspirationSourceValidator._start_request_hash(
                run=run,
                idempotency_key=receipt.idempotency_key,
            )
        )
        if (
            receipt is None
            or receipt.caller_scope != f"agent:{run.owner_agent_id}"
            or receipt.scope != "inspiration.run.start"
            or receipt.request_hash != expected_request_hash
            or run.request_hash != expected_request_hash
            or receipt.state != expected_state
            or receipt.result_resource_type != "inspiration_run"
            or receipt.result_resource_id != run.run_id
            or (receipt.created_at > run.created_at)
        ):
            raise _inspiration_error(
                "Run trace lacks its exact attached start receipt",
                mismatch_key=key,
            )
        if not completed and any(
            value is not None
            for value in (
                receipt.response_status_code,
                receipt.response_body,
                receipt.response_content_type,
                receipt.response_headers,
                receipt.completed_at,
            )
        ):
            raise _inspiration_error(
                "In-progress run receipt contains completion data",
                mismatch_key=key,
            )
        return receipt

    @staticmethod
    def _validate_run_response(
        *,
        receipt: IdempotencyRecordRow,
        run: InspirationRunRow,
        terminal: _InspirationEvent,
        snapshot_hash: str | None,
    ) -> None:
        key = f"inspiration_run:{run.run_id}"
        payload = terminal.payload
        if not isinstance(payload, _INSPIRATION_TERMINAL_TYPES):
            raise _inspiration_error(
                "Run terminal payload is invalid",
                mismatch_key=key,
            )
        if (
            receipt.response_status_code != 201
            or receipt.response_body is None
            or receipt.response_content_type != "application/json"
            or receipt.response_headers is None
            or receipt.completed_at is None
            or receipt.created_at > terminal.row.occurred_at
            or receipt.completed_at != terminal.row.occurred_at
        ):
            raise _inspiration_error(
                "Completed run receipt has no canonical terminal response",
                mismatch_key=key,
            )
        body = _canonical_value(
            receipt.response_body,
            label="run response",
            mismatch_key=key,
        )
        headers = _canonical_value(
            receipt.response_headers,
            label="run response headers",
            mismatch_key=key,
        )
        try:
            response = InspirationRunResponseV1.model_validate_json(
                canonical_json_bytes(body)
            )
            operators = InspirationSourceValidator._run_operators(run)
            mode = RetrievalMode(run.mode)
            generator = GeneratorKind(run.generator_kind)
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Completed run response schema is invalid",
                mismatch_key=key,
            ) from error
        data = response.data
        expected_location = (
            f"/v1/agents/{run.owner_agent_id}/inspiration-runs/{run.run_id}"
        )
        if (
            canonical_json_bytes(response) != receipt.response_body
            or headers != {"location": expected_location}
            or data.run_id != run.run_id
            or data.owner_agent_id != run.owner_agent_id
            or data.goal != run.goal
            or data.context != (run.context or "")
            or data.mode is not mode
            or data.generator is not generator
            or data.operators != operators
            or data.include_inbox is not run.include_inbox
            or data.branches_per_operator != run.branches_per_operator
            or data.output_tokens_per_operator != run.output_tokens_per_operator
            or data.total_output_tokens != run.total_output_tokens
            or data.operator_timeout_seconds != run.operator_timeout_seconds
            or data.global_timeout_seconds != run.global_timeout_seconds
            or data.request_hash != run.request_hash
            or data.snapshot_hash != snapshot_hash
            or data.status is not payload.status_after
            or data.operator_outcomes != payload.operator_outcomes
            or data.output_tokens_reserved != payload.output_tokens_reserved_after
            or data.output_tokens_consumed != payload.output_tokens_consumed_after
            or data.elapsed_milliseconds != payload.elapsed_milliseconds_after
            or data.created_at != run.created_at
            or data.completed_at != terminal.row.occurred_at
        ):
            raise _inspiration_error(
                "Run response is not an exact rendering of its terminal ledger",
                mismatch_key=key,
            )

    def _validate_runs(
        self,
        *,
        run_by_id: dict[UUID, InspirationRunRow],
        idea_by_id: dict[UUID, InspirationIdeaRow],
        generated_by_idea: dict[UUID, _InspirationEvent],
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        events: tuple[_InspirationEvent, ...],
        receipts: dict[UUID, IdempotencyRecordRow],
        event_ids_by_causation: dict[UUID, set[int]],
    ) -> None:
        run_events: dict[UUID, list[_InspirationEvent]] = {}
        started_by_run: dict[UUID, list[_InspirationEvent]] = {}
        snapshot_by_run: dict[UUID, list[_InspirationEvent]] = {}
        for event in events:
            if event.row.aggregate_type == "inspiration_run":
                run_events.setdefault(event.row.aggregate_id, []).append(event)
            if isinstance(event.payload, InspirationStartedV1):
                started_by_run.setdefault(event.payload.run_id, []).append(event)
            elif isinstance(event.payload, InspirationSnapshotFrozenV1):
                snapshot_by_run.setdefault(event.payload.run_id, []).append(event)
        generated_by_run: dict[UUID, list[_InspirationEvent]] = {}
        for event in generated_by_idea.values():
            payload = cast(InspirationIdeaGeneratedV1, event.payload)
            generated_by_run.setdefault(payload.run_id, []).append(event)
        start_receipts_by_run: dict[UUID, list[IdempotencyRecordRow]] = {}
        recovery_receipts_by_run: dict[UUID, list[IdempotencyRecordRow]] = {}
        for receipt in receipts.values():
            if receipt.scope == "inspiration.run.start":
                if receipt.result_resource_type == "inspiration_run":
                    if (
                        receipt.result_resource_id is None
                        or receipt.result_resource_id not in run_by_id
                    ):
                        raise _inspiration_error(
                            "Start receipt is attached to an unknown inspiration run",
                            mismatch_key=f"inspiration_receipt:{receipt.receipt_id}",
                        )
                    start_receipts_by_run.setdefault(
                        receipt.result_resource_id,
                        [],
                    ).append(receipt)
                elif (
                    receipt.result_resource_type is None
                    and receipt.result_resource_id is None
                ):
                    self._validate_unconfigured_start_receipt(
                        receipt=receipt,
                        event_ids_by_causation=event_ids_by_causation,
                    )
                else:
                    raise _inspiration_error(
                        "Start receipt has an unsupported resource attachment",
                        mismatch_key=f"inspiration_receipt:{receipt.receipt_id}",
                    )
            elif receipt.scope == "inspiration.run.recover":
                if (
                    receipt.result_resource_type != "inspiration_run"
                    or receipt.result_resource_id is None
                    or receipt.result_resource_id not in run_by_id
                ):
                    raise _inspiration_error(
                        "Recovery receipt is not attached to a known inspiration run",
                        mismatch_key=f"inspiration_receipt:{receipt.receipt_id}",
                    )
                recovery_receipts_by_run.setdefault(
                    receipt.result_resource_id,
                    [],
                ).append(receipt)
        all_run_ids = sorted(
            set(run_by_id)
            | set(run_events)
            | set(started_by_run)
            | set(generated_by_run),
            key=lambda value: value.bytes,
        )
        for run_id in all_run_ids:
            key = f"inspiration_run:{run_id}"
            run = run_by_id.get(run_id)
            traces = tuple(
                sorted(
                    run_events.get(run_id, ()),
                    key=lambda item: item.row.event_id,
                )
            )
            starts = started_by_run.get(run_id, ())
            generated = tuple(
                sorted(
                    generated_by_run.get(run_id, ()),
                    key=lambda item: item.row.event_id,
                )
            )
            if run is None or len(starts) != 1 or not traces:
                raise _inspiration_error(
                    "Every run requires one source row and one started event",
                    mismatch_key=key,
                )
            started = starts[0]
            started_payload = cast(InspirationStartedV1, started.payload)
            if (
                traces[0].row.event_id != started.row.event_id
                or started.row.sequence != 1
                or started_payload.owner_agent_id != run.owner_agent_id
                or started.row.actor_agent_id != run.owner_agent_id
                or started.row.occurred_at != run.created_at
            ):
                raise _inspiration_error(
                    "Run start does not match its immutable identity",
                    mismatch_key=key,
                )
            operators = self._run_operators(run)
            snapshots = snapshot_by_run.get(run_id, ())
            if len(snapshots) > 1:
                raise _inspiration_error(
                    "Run has more than one frozen snapshot event",
                    mismatch_key=key,
                )
            snapshot_hash = (
                None
                if not snapshots
                else cast(
                    InspirationSnapshotFrozenV1,
                    snapshots[0].payload,
                ).snapshot_hash
            )
            terminals = tuple(
                event
                for event in traces
                if isinstance(event.payload, _INSPIRATION_TERMINAL_TYPES)
            )
            original = receipts.get(started.row.causation_id)
            attached_starts = start_receipts_by_run.get(run_id, [])
            if (
                len(attached_starts) != 1
                or original is None
                or attached_starts[0].receipt_id != original.receipt_id
            ):
                raise _inspiration_error(
                    "Run does not have exactly one causal start receipt",
                    mismatch_key=key,
                )
            attached_recoveries = recovery_receipts_by_run.get(run_id, [])
            if not terminals:
                self._require_run_receipt(
                    receipt=original,
                    run=run,
                    completed=False,
                )
                if attached_recoveries:
                    raise _inspiration_error(
                        "Nonterminal run already has a recovery receipt",
                        mismatch_key=key,
                    )
                legal = (
                    (InspirationStartedV1,)
                    if not snapshots
                    else (InspirationStartedV1, InspirationSnapshotFrozenV1)
                )
                if (
                    tuple(type(event.payload) for event in traces) != legal
                    or generated
                    or any(
                        event.row.causation_id != started.row.causation_id
                        or event.row.occurred_at != run.created_at
                        or event.row.actor_agent_id != run.owner_agent_id
                        for event in traces
                    )
                ):
                    raise _inspiration_error(
                        "Nonterminal run is not a legal retained phase",
                        mismatch_key=key,
                    )
                self._require_causation_closure(
                    receipt_id=started.row.causation_id,
                    expected_events=traces,
                    event_ids_by_causation=event_ids_by_causation,
                    mismatch_key=key,
                )
                continue
            if len(terminals) != 1 or traces[-1] is not terminals[0]:
                raise _inspiration_error(
                    "A completed run requires exactly one final terminal event",
                    mismatch_key=key,
                )
            terminal = terminals[0]
            terminal_payload = cast(
                InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1,
                terminal.payload,
            )
            recovered = (
                isinstance(terminal_payload, InspirationFailedV1)
                and terminal_payload.failure_code
                is InspirationRunFailureCode.PROCESS_INTERRUPTED
            )
            if (recovered and len(attached_recoveries) != 1) or (
                not recovered and attached_recoveries
            ):
                raise _inspiration_error(
                    "Recovery receipts do not match the run terminal",
                    mismatch_key=key,
                )
            original = self._require_run_receipt(
                receipt=original,
                run=run,
                completed=True,
            )
            if recovered:
                legal_prefix = (
                    (InspirationStartedV1,)
                    if not snapshots
                    else (InspirationStartedV1, InspirationSnapshotFrozenV1)
                )
                recovery_receipt = receipts.get(terminal.row.causation_id)
                if (
                    tuple(type(event.payload) for event in traces[:-1]) != legal_prefix
                    or generated
                    or terminal.row.actor_agent_id is not None
                    or terminal.row.causation_id == started.row.causation_id
                    or terminal.row.occurred_at < run.created_at
                    or recovery_receipt is None
                    or attached_recoveries[0].receipt_id != recovery_receipt.receipt_id
                    or recovery_receipt.caller_scope != "system:local"
                    or recovery_receipt.scope != "inspiration.run.recover"
                    or recovery_receipt.idempotency_key != f"recovery:{run.run_id}"
                    or recovery_receipt.request_hash
                    != self._recovery_request_hash(
                        run_id=run.run_id,
                        idempotency_key=recovery_receipt.idempotency_key,
                    )
                    or recovery_receipt.state != "completed"
                    or recovery_receipt.result_resource_type != "inspiration_run"
                    or recovery_receipt.result_resource_id != run.run_id
                    or recovery_receipt.response_body != original.response_body
                    or recovery_receipt.response_status_code
                    != original.response_status_code
                    or recovery_receipt.response_content_type
                    != original.response_content_type
                    or recovery_receipt.response_headers != original.response_headers
                ):
                    raise _inspiration_error(
                        "Recovered run does not have a legal recovery trace",
                        mismatch_key=key,
                    )
                assert recovery_receipt is not None
                recovered_at = max(
                    recovery_receipt.created_at,
                    *(event.row.occurred_at for event in traces[:-1]),
                )
                if (
                    terminal.row.occurred_at != recovered_at
                    or recovery_receipt.completed_at != recovered_at
                    or original.completed_at != recovered_at
                ):
                    raise _inspiration_error(
                        "Recovered run terminal does not match its recovery clock",
                        mismatch_key=key,
                    )
                for event in traces[:-1]:
                    if (
                        event.row.causation_id != started.row.causation_id
                        or event.row.occurred_at != run.created_at
                        or event.row.actor_agent_id != run.owner_agent_id
                    ):
                        raise _inspiration_error(
                            "Recovered run prefix changed its logical command anchor",
                            mismatch_key=key,
                        )
                self._require_terminal_accounting(
                    terminal_payload,
                    reserved=0,
                    consumed=0,
                    elapsed=0,
                    mismatch_key=key,
                )
                self._validate_run_response(
                    receipt=recovery_receipt,
                    run=run,
                    terminal=terminal,
                    snapshot_hash=snapshot_hash,
                )
                self._require_causation_closure(
                    receipt_id=terminal.row.causation_id,
                    expected_events=(terminal,),
                    event_ids_by_causation=event_ids_by_causation,
                    mismatch_key=key,
                )
            else:
                all_normal_events = (*traces, *generated)
                if any(
                    event.row.causation_id != started.row.causation_id
                    or event.row.occurred_at != run.created_at
                    or event.row.actor_agent_id != run.owner_agent_id
                    for event in all_normal_events
                ):
                    raise _inspiration_error(
                        "Normal run phases do not share one logical command anchor",
                        mismatch_key=key,
                    )
                self._validate_normal_run_history(
                    run=run,
                    idea_by_id=idea_by_id,
                    operators=operators,
                    traces=traces,
                    generated=generated,
                    snapshots=tuple(snapshots),
                    terminal=terminal,
                    frozen_items=frozen_by_run.get(run_id, ()),
                )
            self._require_causation_closure(
                receipt_id=started.row.causation_id,
                expected_events=(
                    (*traces[:-1], *generated) if recovered else (*traces, *generated)
                ),
                event_ids_by_causation=event_ids_by_causation,
                mismatch_key=key,
            )
            self._validate_run_response(
                receipt=original,
                run=run,
                terminal=terminal,
                snapshot_hash=snapshot_hash,
            )
            if frozen_by_run.get(run_id) and not snapshots:
                raise _inspiration_error(
                    "Run has snapshot rows without a snapshot event",
                    mismatch_key=key,
                )

    @staticmethod
    def _validate_unconfigured_start_receipt(
        *,
        receipt: IdempotencyRecordRow,
        event_ids_by_causation: dict[UUID, set[int]],
    ) -> None:
        key = f"inspiration_receipt:{receipt.receipt_id}"
        try:
            caller_prefix, raw_owner_id = receipt.caller_scope.split(":", 1)
            UUID(raw_owner_id)
        except (ValueError, AttributeError) as error:
            raise _inspiration_error(
                "Unconfigured generator receipt has an invalid caller",
                mismatch_key=key,
            ) from error
        expected_body = canonical_json_bytes(
            {
                "error": {
                    "code": "generator_not_configured",
                    "details": {},
                    "message": (
                        "The selected inspiration generator is not configured."
                    ),
                }
            }
        )
        if (
            caller_prefix != "agent"
            or receipt.state != "completed"
            or receipt.response_status_code != 422
            or receipt.response_body != expected_body
            or receipt.response_content_type != "application/json"
            or receipt.response_headers != canonical_json_bytes({})
            or receipt.completed_at is None
            or receipt.completed_at < receipt.created_at
            or event_ids_by_causation.get(receipt.receipt_id)
        ):
            raise _inspiration_error(
                "Unattached start receipt is not a canonical configuration failure",
                mismatch_key=key,
            )

    @staticmethod
    def _require_causation_closure(
        *,
        receipt_id: UUID,
        expected_events: tuple[_InspirationEvent, ...],
        event_ids_by_causation: dict[UUID, set[int]],
        mismatch_key: str,
    ) -> None:
        if event_ids_by_causation.get(receipt_id, set()) != {
            event.row.event_id for event in expected_events
        }:
            raise _inspiration_error(
                "Run receipt causation contains missing or foreign side effects",
                mismatch_key=mismatch_key,
            )

    @staticmethod
    def _validate_normal_run_history(
        *,
        run: InspirationRunRow,
        idea_by_id: dict[UUID, InspirationIdeaRow],
        operators: tuple[InspirationOperator, ...],
        traces: tuple[_InspirationEvent, ...],
        generated: tuple[_InspirationEvent, ...],
        snapshots: tuple[_InspirationEvent, ...],
        terminal: _InspirationEvent,
        frozen_items: tuple[SnapshotItem, ...],
    ) -> None:
        key = f"inspiration_run:{run.run_id}"
        terminal_payload = cast(
            InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1,
            terminal.payload,
        )
        preparation_failed = (
            isinstance(terminal_payload, InspirationFailedV1)
            and terminal_payload.failure_code
            is InspirationRunFailureCode.PREPARATION_FAILED
        )
        operator_events = tuple(
            event
            for event in traces
            if isinstance(event.payload, _INSPIRATION_OPERATOR_TYPES)
        )
        if preparation_failed:
            if (
                snapshots
                or generated
                or operator_events
                or tuple(type(event.payload) for event in traces)
                != (InspirationStartedV1, InspirationFailedV1)
            ):
                raise _inspiration_error(
                    "Preparation failure has an illegal event trace",
                    mismatch_key=key,
                )
            InspirationSourceValidator._require_terminal_accounting(
                terminal_payload,
                reserved=0,
                consumed=0,
                elapsed=0,
                mismatch_key=key,
            )
            return
        retained_mechanisms: list[tuple[str, str]] = []
        for event in generated:
            payload = cast(InspirationIdeaGeneratedV1, event.payload)
            idea = idea_by_id.get(payload.idea_id)
            if idea is None:
                raise _inspiration_error(
                    "Generated run idea has no immutable source row",
                    mismatch_key=key,
                )
            if any(
                payload.mechanism_hash == earlier_hash
                or mechanism_similarity(idea.mechanism, earlier_mechanism)
                >= NEAR_DUPLICATE_THRESHOLD
                for earlier_hash, earlier_mechanism in retained_mechanisms
            ):
                raise _inspiration_error(
                    "Run retained an exact or near-duplicate mechanism",
                    mismatch_key=key,
                )
            retained_mechanisms.append((payload.mechanism_hash, idea.mechanism))
        if (
            len(snapshots) != 1
            or len(operator_events) != len(operators)
            or tuple(
                cast(
                    InspirationOperatorCompletedV1 | InspirationOperatorFailedV1,
                    event.payload,
                ).operator
                for event in operator_events
            )
            != operators
            or terminal_payload.operator_outcomes
            != tuple(
                cast(
                    InspirationOperatorCompletedV1 | InspirationOperatorFailedV1,
                    event.payload,
                ).outcome
                for event in operator_events
            )
        ):
            raise _inspiration_error(
                "Run operator history does not match its configuration",
                mismatch_key=key,
            )
        snapshot_event_id = snapshots[0].row.event_id
        ideas_by_operator: dict[InspirationOperator, list[_InspirationEvent]] = {}
        for event in generated:
            generated_payload = cast(
                InspirationIdeaGeneratedV1,
                event.payload,
            )
            ideas_by_operator.setdefault(
                generated_payload.operator,
                [],
            ).append(event)
        previous_operator_event_id = snapshot_event_id
        expected_reserved = 0
        expected_consumed = 0
        expected_elapsed = 0
        global_deadline_exhausted = False
        for operator_event in operator_events:
            operator_payload = cast(
                InspirationOperatorCompletedV1 | InspirationOperatorFailedV1,
                operator_event.payload,
            )
            idea_events = ideas_by_operator.get(operator_payload.operator, ())
            ordinals = tuple(
                cast(InspirationIdeaGeneratedV1, event.payload).ordinal
                for event in idea_events
            )
            reservation = (
                operator_payload.output_tokens_reserved_after
                - operator_payload.output_tokens_reserved_before
            )
            error_code = operator_payload.outcome.error_code
            duplicate_count = operator_payload.outcome.duplicate_count
            duplicate_count_valid = (
                (
                    operator_payload.outcome.persisted_ideas + duplicate_count
                    <= run.branches_per_operator
                )
                if operator_payload.outcome.succeeded
                else (
                    duplicate_count <= run.branches_per_operator
                    if error_code is not None
                    and error_code.value == "no_valid_branches"
                    else duplicate_count == 0
                )
            )
            unattempted_failure = error_code is not None and error_code.value in {
                "insufficient_evidence",
                "insufficient_token_reservation",
            }
            token_reservation_available = (
                operator_payload.output_tokens_consumed_before
                + run.output_tokens_per_operator
                <= run.total_output_tokens
            )
            token_budget_decision_valid = (
                (
                    error_code is None
                    or error_code.value != "insufficient_token_reservation"
                )
                if run.generator_kind == GeneratorKind.DETERMINISTIC.value
                else (
                    (reservation == 0 or token_reservation_available)
                    and (
                        error_code is None
                        or error_code.value != "insufficient_token_reservation"
                        or not token_reservation_available
                    )
                )
            )
            deadline_chain_valid = not global_deadline_exhausted or (
                error_code is not None
                and error_code.value == "global_deadline_exhausted"
                and reservation == 0
                and operator_payload.outcome.output_tokens_consumed == 0
                and operator_payload.elapsed_milliseconds_after
                == operator_payload.elapsed_milliseconds_before
            )
            deadline_onset_valid = not (
                not global_deadline_exhausted
                and error_code is not None
                and error_code.value == "global_deadline_exhausted"
            ) or (
                operator_payload.elapsed_milliseconds_after
                >= run.global_timeout_seconds * 1_000
            )
            evidence_decision_valid = (
                error_code is None or error_code.value != "insufficient_evidence"
                if frozen_items
                else (
                    error_code is not None
                    and error_code.value == "insufficient_evidence"
                    and operator_payload.output_tokens_reserved_before == 0
                    and operator_payload.output_tokens_reserved_after == 0
                    and operator_payload.output_tokens_consumed_before == 0
                    and operator_payload.output_tokens_consumed_after == 0
                    and operator_payload.elapsed_milliseconds_before == 0
                    and operator_payload.elapsed_milliseconds_after == 0
                )
            )
            operator_timeout_valid = not (
                error_code is not None and error_code.value == "provider_timeout"
            ) or (
                operator_payload.elapsed_milliseconds_after
                - operator_payload.elapsed_milliseconds_before
                >= run.operator_timeout_seconds * 1_000
            )
            reservation_valid = (
                reservation == 0
                if run.generator_kind == GeneratorKind.DETERMINISTIC.value
                or unattempted_failure
                else (
                    reservation in {0, run.output_tokens_per_operator}
                    if error_code is not None
                    and error_code.value == "global_deadline_exhausted"
                    else reservation == run.output_tokens_per_operator
                )
            )
            if (
                operator_payload.outcome.persisted_ideas != len(idea_events)
                or len(idea_events) > run.branches_per_operator
                or ordinals != tuple(range(1, len(idea_events) + 1))
                or not duplicate_count_valid
                or operator_payload.output_tokens_reserved_before != expected_reserved
                or operator_payload.output_tokens_consumed_before != expected_consumed
                or operator_payload.elapsed_milliseconds_before != expected_elapsed
                or not reservation_valid
                or not token_budget_decision_valid
                or not deadline_chain_valid
                or not deadline_onset_valid
                or not evidence_decision_valid
                or not operator_timeout_valid
                or (
                    run.generator_kind != GeneratorKind.DETERMINISTIC.value
                    and reservation == 0
                    and (
                        operator_payload.outcome.output_tokens_consumed != 0
                        or operator_payload.elapsed_milliseconds_after
                        != operator_payload.elapsed_milliseconds_before
                    )
                )
                or operator_event.row.event_id <= previous_operator_event_id
                or any(
                    event.row.event_id <= previous_operator_event_id
                    or event.row.event_id >= operator_event.row.event_id
                    for event in idea_events
                )
            ):
                raise _inspiration_error(
                    "Operator outcome does not match persisted idea events",
                    mismatch_key=key,
                )
            previous_operator_event_id = operator_event.row.event_id
            expected_reserved = operator_payload.output_tokens_reserved_after
            expected_consumed = operator_payload.output_tokens_consumed_after
            expected_elapsed = operator_payload.elapsed_milliseconds_after
            global_deadline_exhausted = global_deadline_exhausted or (
                error_code is not None
                and error_code.value == "global_deadline_exhausted"
            )
        if global_deadline_exhausted != isinstance(
            terminal_payload,
            InspirationTimedOutV1,
        ):
            raise _inspiration_error(
                "Global deadline exhaustion does not match the terminal event",
                mismatch_key=key,
            )
        if (
            terminal_payload.output_tokens_reserved_before != expected_reserved
            or terminal_payload.output_tokens_reserved_after != expected_reserved
            or terminal_payload.output_tokens_consumed_before != expected_consumed
            or terminal_payload.output_tokens_consumed_after != expected_consumed
            or terminal_payload.elapsed_milliseconds_before != expected_elapsed
            or terminal_payload.elapsed_milliseconds_after != expected_elapsed
        ):
            raise _inspiration_error(
                "Terminal accounting does not continue operator counters",
                mismatch_key=key,
            )
        if any(operator not in operators for operator in ideas_by_operator):
            raise _inspiration_error(
                "Generated idea names an operator outside the run",
                mismatch_key=key,
            )

    @staticmethod
    def _require_terminal_accounting(
        payload: InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1,
        *,
        reserved: int,
        consumed: int,
        elapsed: int,
        mismatch_key: str,
    ) -> None:
        if (
            payload.output_tokens_reserved_before != reserved
            or payload.output_tokens_reserved_after != reserved
            or payload.output_tokens_consumed_before != consumed
            or payload.output_tokens_consumed_after != consumed
            or payload.elapsed_milliseconds_before != elapsed
            or payload.elapsed_milliseconds_after != elapsed
        ):
            raise _inspiration_error(
                "Terminal accounting does not match its legal prefix",
                mismatch_key=mismatch_key,
            )

    @staticmethod
    def _require_idea_receipt(
        *,
        event: _InspirationEvent,
        run: InspirationRunRow,
        receipts: dict[UUID, IdempotencyRecordRow],
        scope: str,
        resource_type: str,
        resource_id: UUID,
        request_hash: str,
        mismatch_key: str,
        allow_event_before_receipt: bool = False,
    ) -> IdempotencyRecordRow:
        receipt = receipts.get(event.row.causation_id)
        if (
            receipt is None
            or receipt.caller_scope != f"agent:{run.owner_agent_id}"
            or receipt.scope != scope
            or receipt.request_hash != request_hash
            or receipt.state != "completed"
            or receipt.result_resource_type != resource_type
            or receipt.result_resource_id != resource_id
            or receipt.response_status_code != 200
            or receipt.response_body is None
            or receipt.response_content_type != "application/json"
            or receipt.response_headers is None
            or receipt.completed_at is None
            or receipt.completed_at < receipt.created_at
            or (
                not allow_event_before_receipt
                and receipt.created_at > event.row.occurred_at
            )
            or receipt.completed_at < event.row.occurred_at
        ):
            raise _inspiration_error(
                "Idea event lacks its exact completed command receipt",
                mismatch_key=mismatch_key,
            )
        _canonical_value(
            receipt.response_body,
            label="idea command response",
            mismatch_key=mismatch_key,
        )
        headers = _canonical_value(
            receipt.response_headers,
            label="idea command response headers",
            mismatch_key=mismatch_key,
        )
        if headers != {}:
            raise _inspiration_error(
                "Idea command response has unexpected headers",
                mismatch_key=mismatch_key,
            )
        return receipt

    @staticmethod
    async def _validate_evaluation_evidence(
        *,
        session: AsyncSession,
        payload: InspirationIdeaEvaluatedV1,
        state: _IdeaSourceState,
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        evaluated_at: datetime,
    ) -> None:
        key = f"inspiration_evaluation:{payload.idea_id}:{payload.revision}"
        frozen = {
            item.snapshot_item_id: item
            for item in frozen_by_run.get(state.run.run_id, ())
        }
        for reference in payload.evidence:
            if isinstance(reference, SnapshotEvidenceReference):
                item = frozen.get(reference.id)
                if (
                    item is None
                    or item.stable_evidence_key != reference.stable_evidence_key
                ):
                    raise _inspiration_error(
                        "Evaluation snapshot evidence does not resolve",
                        mismatch_key=key,
                    )
                continue
            if isinstance(reference, ExperienceVersionEvidenceReference):
                version = await session.get(ExperienceVersionRow, reference.id)
                identity = (
                    None
                    if version is None
                    else await session.get(
                        ExperienceRow,
                        version.experience_id,
                    )
                )
                if (
                    version is None
                    or identity is None
                    or identity.owner_agent_id != payload.evaluator_agent_id
                    or identity.created_at > evaluated_at
                    or version.created_at > evaluated_at
                ):
                    raise _inspiration_error(
                        "Evaluation experience evidence was not an owned "
                        "version available at evaluation time",
                        mismatch_key=key,
                    )
                continue
            raise _inspiration_error(
                "Evaluation evidence has an unsupported type",
                mismatch_key=key,
            )

    @staticmethod
    def _evaluation_matches_plan(
        payload: InspirationIdeaEvaluatedV1,
        transition: EvaluationTransition,
    ) -> bool:
        return bool(
            payload.previous_verdict is transition.previous_verdict
            and payload.current_verdict is transition.current_verdict
            and payload.supported_count_before == transition.supported_count_before
            and payload.supported_count_after == transition.supported_count_after
            and payload.refuted_count_before == transition.refuted_count_before
            and payload.refuted_count_after == transition.refuted_count_after
            and payload.maturity_before is transition.maturity_before
            and payload.maturity_after is transition.maturity_after
            and payload.candidate_since_before == transition.candidate_since_before
            and payload.candidate_since_after == transition.candidate_since_after
            and payload.last_signal_at_before == transition.last_signal_at_before
            and payload.last_signal_at_after == transition.last_signal_at_after
        )

    @staticmethod
    def _adoption_matches_plan(
        payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2,
        transition: AdoptionTransition,
    ) -> bool:
        return bool(
            payload.distinct_adopter_count_before
            == transition.distinct_adopter_count_before
            and payload.distinct_adopter_count_after
            == transition.distinct_adopter_count_after
            and payload.maturity_before is transition.maturity_before
            and payload.maturity_after is transition.maturity_after
            and payload.candidate_since_before == transition.candidate_since_before
            and payload.candidate_since_after == transition.candidate_since_after
            and payload.last_signal_at_before == transition.last_signal_at_before
            and payload.last_signal_at_after == transition.last_signal_at_after
        )

    @staticmethod
    def _validate_lifecycle_archive_receipt(
        *,
        event: _InspirationEvent,
        payload: InspirationIdeaArchivedV1,
        receipts: dict[UUID, IdempotencyRecordRow],
    ) -> IdempotencyRecordRow:
        key = f"inspiration_archive:{payload.idea_id}"
        receipt = receipts.get(event.row.causation_id)
        try:
            expected_request_hashes = set()
            omitted_request_hash = None
            if receipt is not None:
                expected_request_hashes = {
                    CommandRequest(
                        caller_scope="system:local",
                        operation_scope="lifecycle.run",
                        idempotency_key=receipt.idempotency_key,
                        method="POST",
                        route_template="/v1/lifecycle:run",
                        body={
                            "evaluated_at": event.row.occurred_at,
                            "mode": mode,
                        },
                    ).request_hash
                    for mode in ("manual", "background")
                }
                omitted_request_hash = CommandRequest(
                    caller_scope="system:local",
                    operation_scope="lifecycle.run",
                    idempotency_key=receipt.idempotency_key,
                    method="POST",
                    route_template="/v1/lifecycle:run",
                    body={
                        "evaluated_at": None,
                        "mode": "manual",
                    },
                ).request_hash
                expected_request_hashes.add(omitted_request_hash)
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Automatic archive lifecycle request is invalid",
                mismatch_key=key,
            ) from error
        if (
            payload.cycle_id is None
            or receipt is None
            or receipt.caller_scope != "system:local"
            or receipt.scope != "lifecycle.run"
            or receipt.request_hash not in expected_request_hashes
            or receipt.state != "completed"
            or receipt.result_resource_type != "lifecycle_cycle"
            or receipt.result_resource_id != payload.cycle_id
            or receipt.response_status_code != 200
            or receipt.response_body is None
            or receipt.response_content_type != "application/json"
            or receipt.response_headers is None
            or receipt.completed_at is None
            or receipt.completed_at < receipt.created_at
            or receipt.completed_at < event.row.occurred_at
            or (
                receipt.request_hash == omitted_request_hash
                and receipt.created_at != event.row.occurred_at
            )
        ):
            raise _inspiration_error(
                "Automatic archive lacks its exact lifecycle receipt",
                mismatch_key=key,
            )
        try:
            result = decode_lifecycle_result(receipt.response_body)
        except ValueError as error:
            raise _inspiration_error(
                "Automatic archive lifecycle response is invalid",
                mismatch_key=key,
            ) from error
        headers = _canonical_value(
            receipt.response_headers,
            label="lifecycle response headers",
            mismatch_key=key,
        )
        if (
            headers != {}
            or result.cycle_id != payload.cycle_id
            or result.evaluated_at != event.row.occurred_at
        ):
            raise _inspiration_error(
                "Automatic archive is not bound to its lifecycle result",
                mismatch_key=key,
            )
        return receipt

    async def _validate_idea_histories(
        self,
        *,
        session: AsyncSession,
        run_by_id: dict[UUID, InspirationRunRow],
        idea_by_id: dict[UUID, InspirationIdeaRow],
        occurrence_by_id: dict[UUID, IdeaOccurrenceRow],
        adoption_by_id: dict[UUID, IdeaAdoptionRecordRow],
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        events: tuple[_InspirationEvent, ...],
        receipts: dict[UUID, IdempotencyRecordRow],
    ) -> None:
        clusters: dict[str, IncubationCluster] = {}
        idea_states: dict[UUID, _IdeaSourceState] = {}
        idea_event_counts: dict[UUID, int] = {}
        adopter_owners: dict[str, set[UUID]] = {}
        seen_adoptions: set[UUID] = set()
        lifecycle_archives: dict[UUID, list[InspirationIdeaArchivedV1]] = {}
        terminal_event_id_by_run = {
            cast(
                InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1,
                event.payload,
            ).run_id: event.row.event_id
            for event in events
            if isinstance(event.payload, _INSPIRATION_TERMINAL_TYPES)
        }
        idea_events = tuple(
            event for event in events if event.row.aggregate_type == "idea"
        )
        for event in idea_events:
            payload = event.payload
            idea_id = event.row.aggregate_id
            key = f"inspiration_idea_history:{idea_id}"
            expected_sequence = idea_event_counts.get(idea_id, 0) + 1
            if event.row.sequence != expected_sequence:
                raise _inspiration_error(
                    "Idea history sequence is not contiguous from generation",
                    mismatch_key=key,
                )
            idea_event_counts[idea_id] = expected_sequence
            if isinstance(payload, InspirationIdeaGeneratedV1):
                if expected_sequence != 1 or idea_id in idea_states:
                    raise _inspiration_error(
                        "Idea history must begin with one generated event",
                        mismatch_key=key,
                    )
                idea = idea_by_id.get(idea_id)
                run = run_by_id.get(payload.run_id)
                occurrence = occurrence_by_id.get(payload.occurrence_id)
                if idea is None or run is None or occurrence is None:
                    raise _inspiration_error(
                        "Generated idea source anchors are missing",
                        mismatch_key=key,
                    )
                historical_clusters = tuple(
                    clusters[cluster_id] for cluster_id in sorted(clusters)
                )
                try:
                    plan = plan_occurrence(
                        owner_agent_id=run.owner_agent_id,
                        mechanism=idea.mechanism,
                        snapshot_hash=occurrence.snapshot_hash,
                        run_occurred_at=event.row.occurred_at,
                        clusters=historical_clusters,
                    )
                except (TypeError, ValueError) as error:
                    raise _inspiration_error(
                        "Generated idea cluster plan cannot be reconstructed",
                        mismatch_key=key,
                    ) from error
                actual_transition = _cluster_transition(payload)
                if (
                    payload.mechanism_hash != plan.mechanism_hash
                    or payload.duplicate_relation != plan.duplicate_relation
                    or idea.duplicate_relation != plan.duplicate_relation
                    or actual_transition != plan.transition
                ):
                    raise _inspiration_error(
                        "Generated idea cluster transition is not authoritative",
                        mismatch_key=key,
                    )
                prior = clusters.get(plan.transition.cluster_id)
                members = () if prior is None else prior.members
                snapshot_hashes = (
                    frozenset() if prior is None else prior.snapshot_hashes
                )
                clusters[plan.transition.cluster_id] = IncubationCluster(
                    state=_state_after_occurrence(plan.transition),
                    members=(
                        *members,
                        IncubationMember(
                            idea_id=idea.idea_id,
                            owner_agent_id=run.owner_agent_id,
                            mechanism=idea.mechanism,
                            mechanism_hash=idea.mechanism_hash,
                        ),
                    ),
                    snapshot_hashes=(snapshot_hashes | {occurrence.snapshot_hash}),
                )
                idea_states[idea_id] = _IdeaSourceState(
                    idea=idea,
                    run=run,
                    generated=payload,
                    decision=IdeaOwnerDecision.ACTIVE,
                    last_signal_at=payload.last_signal_at_after,
                    last_event_at=event.row.occurred_at,
                    latest_evaluations={},
                )
                continue

            state = idea_states.get(idea_id)
            if state is None:
                raise _inspiration_error(
                    "Idea history event has no generated predecessor",
                    mismatch_key=key,
                )
            terminal_event_id = terminal_event_id_by_run.get(state.run.run_id)
            if terminal_event_id is None or event.row.event_id <= terminal_event_id:
                raise _inspiration_error(
                    "Idea decisions cannot precede their run terminal",
                    mismatch_key=key,
                )
            if (
                getattr(payload, "idea_id", None) != idea_id
                or getattr(
                    payload,
                    "owner_agent_id",
                    state.run.owner_agent_id,
                )
                != state.run.owner_agent_id
                or event.row.occurred_at < state.last_event_at
                or getattr(payload, "owner_decision_before", None) is not state.decision
            ):
                raise _inspiration_error(
                    "Idea history identity, time, or decision is discontinuous",
                    mismatch_key=key,
                )
            cluster = clusters.get(state.generated.cluster_id)
            if cluster is None:
                raise _inspiration_error(
                    "Idea history mechanism cluster is missing",
                    mismatch_key=key,
                )
            if isinstance(payload, InspirationIdeaEvaluatedV1):
                await self._apply_evaluation_source(
                    session=session,
                    event=event,
                    payload=payload,
                    state=state,
                    cluster=cluster,
                    clusters=clusters,
                    frozen_by_run=frozen_by_run,
                    receipts=receipts,
                )
            elif isinstance(payload, InspirationIdeaArchivedV1):
                if payload.cycle_id is None:
                    causal = receipts.get(event.row.causation_id)
                    try:
                        request_hash = (
                            ""
                            if causal is None
                            else decision_command_request(
                                ArchiveIdea(
                                    owner_agent_id=payload.owner_agent_id,
                                    idea_id=payload.idea_id,
                                    reason=payload.reason,
                                ),
                                idempotency_key=causal.idempotency_key,
                            ).request_hash
                        )
                    except (TypeError, ValueError) as error:
                        raise _inspiration_error(
                            "Explicit archive request cannot be reconstructed",
                            mismatch_key=f"inspiration_archive:{idea_id}",
                        ) from error
                    receipt = self._require_idea_receipt(
                        event=event,
                        run=state.run,
                        receipts=receipts,
                        scope="inspiration.idea.archive",
                        resource_type="idea",
                        resource_id=idea_id,
                        request_hash=request_hash,
                        mismatch_key=f"inspiration_archive:{idea_id}",
                    )
                    expected_body = canonical_json_bytes(
                        {
                            "data": {
                                "idea_id": idea_id,
                                "owner_decision": IdeaOwnerDecision.ARCHIVED,
                            }
                        }
                    )
                    if (
                        event.row.actor_agent_id != state.run.owner_agent_id
                        or receipt.response_body != expected_body
                    ):
                        raise _inspiration_error(
                            "Explicit archive ownership or response is invalid",
                            mismatch_key=f"inspiration_archive:{idea_id}",
                        )
                else:
                    receipt = self._validate_lifecycle_archive_receipt(
                        event=event,
                        payload=payload,
                        receipts=receipts,
                    )
                    lifecycle_archives.setdefault(
                        receipt.receipt_id,
                        [],
                    ).append(payload)
                    due_at = (
                        max(
                            state.last_signal_at,
                            cast(datetime, cluster.state.candidate_since),
                        )
                        + _CANDIDATE_RETENTION
                        if cluster.state.maturity is MechanismMaturity.CANDIDATE
                        else state.last_signal_at + _NONCANDIDATE_RETENTION
                    )
                    if event.row.occurred_at < due_at:
                        raise _inspiration_error(
                            "Automatic archive was not due in historical state",
                            mismatch_key=f"inspiration_archive:{idea_id}",
                        )
                    if event.row.actor_agent_id is not None:
                        raise _inspiration_error(
                            "Automatic archive must be system-authored",
                            mismatch_key=f"inspiration_archive:{idea_id}",
                        )
                state.decision = IdeaOwnerDecision.ARCHIVED
            elif isinstance(payload, InspirationIdeaRejectedV1):
                causal = receipts.get(event.row.causation_id)
                try:
                    request_hash = (
                        ""
                        if causal is None
                        else decision_command_request(
                            RejectIdea(
                                owner_agent_id=payload.owner_agent_id,
                                idea_id=payload.idea_id,
                                reason=payload.reason,
                            ),
                            idempotency_key=causal.idempotency_key,
                        ).request_hash
                    )
                except (TypeError, ValueError) as error:
                    raise _inspiration_error(
                        "Rejection request cannot be reconstructed",
                        mismatch_key=f"inspiration_rejection:{idea_id}",
                    ) from error
                receipt = self._require_idea_receipt(
                    event=event,
                    run=state.run,
                    receipts=receipts,
                    scope="inspiration.idea.reject",
                    resource_type="idea",
                    resource_id=idea_id,
                    request_hash=request_hash,
                    mismatch_key=f"inspiration_rejection:{idea_id}",
                )
                expected_body = canonical_json_bytes(
                    {
                        "data": {
                            "idea_id": idea_id,
                            "owner_decision": IdeaOwnerDecision.REJECTED,
                        }
                    }
                )
                if (
                    event.row.actor_agent_id != state.run.owner_agent_id
                    or receipt.response_body != expected_body
                ):
                    raise _inspiration_error(
                        "Idea rejection ownership or response is invalid",
                        mismatch_key=f"inspiration_rejection:{idea_id}",
                    )
                state.decision = IdeaOwnerDecision.REJECTED
            elif isinstance(
                payload,
                (InspirationIdeaAdoptedV1, InspirationIdeaAdoptedV2),
            ):
                await self._apply_adoption_source(
                    session=session,
                    event=event,
                    payload=payload,
                    state=state,
                    cluster=cluster,
                    clusters=clusters,
                    adoption_by_id=adoption_by_id,
                    seen_adoptions=seen_adoptions,
                    adopter_owners=adopter_owners,
                    frozen_by_run=frozen_by_run,
                    receipts=receipts,
                )
            else:
                raise _inspiration_error(
                    "Idea history has an unsupported event type",
                    mismatch_key=key,
                )
            state.last_event_at = event.row.occurred_at

        if set(idea_states) != set(idea_by_id):
            raise _inspiration_error(
                "Every immutable idea requires a complete generated history",
                mismatch_key="inspiration_ideas",
            )
        if seen_adoptions != set(adoption_by_id):
            raise _inspiration_error(
                "Adoption records and adoption events require a bijection",
                mismatch_key="inspiration_adoptions",
            )
        for receipt_id, archives in lifecycle_archives.items():
            receipt = receipts[receipt_id]
            assert receipt.response_body is not None
            try:
                result = decode_lifecycle_result(receipt.response_body)
            except ValueError as error:  # pragma: no cover - checked above
                raise _inspiration_error(
                    "Lifecycle result became invalid",
                    mismatch_key=f"inspiration_archive:{archives[0].idea_id}",
                ) from error
            if result.idea_archive_count != len(archives):
                raise _inspiration_error(
                    "Lifecycle result archive count does not match its events",
                    mismatch_key=f"inspiration_archive:{archives[0].idea_id}",
                )

    async def _apply_evaluation_source(
        self,
        *,
        session: AsyncSession,
        event: _InspirationEvent,
        payload: InspirationIdeaEvaluatedV1,
        state: _IdeaSourceState,
        cluster: IncubationCluster,
        clusters: dict[str, IncubationCluster],
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        receipts: dict[UUID, IdempotencyRecordRow],
    ) -> None:
        key = f"inspiration_evaluation:{payload.idea_id}:{payload.revision}"
        previous = state.latest_evaluations.get(payload.evaluator_agent_id)
        expected_revision = 1 if previous is None else previous.revision + 1
        expected_previous = None if previous is None else previous.current_verdict
        if (
            payload.evaluator_agent_id != state.run.owner_agent_id
            or payload.mechanism_cluster_id != cluster.state.cluster_id
            or payload.revision != expected_revision
            or payload.previous_verdict is not expected_previous
            or payload.owner_decision_after is not state.decision
            or event.row.actor_agent_id != payload.evaluator_agent_id
            or event.row.occurred_at != payload.last_signal_at_after
        ):
            raise _inspiration_error(
                "Evaluation identity or revision chain is invalid",
                mismatch_key=key,
            )
        await self._validate_evaluation_evidence(
            session=session,
            payload=payload,
            state=state,
            frozen_by_run=frozen_by_run,
            evaluated_at=event.row.occurred_at,
        )
        try:
            transition = plan_evaluation_transition(
                cluster=cluster.state,
                previous_verdict=expected_previous,
                current_verdict=payload.current_verdict,
                evaluated_at=event.row.occurred_at,
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Evaluation transition cannot be reconstructed",
                mismatch_key=key,
            ) from error
        if not self._evaluation_matches_plan(payload, transition):
            raise _inspiration_error(
                "Evaluation event does not match historical mechanism state",
                mismatch_key=key,
            )
        causal = receipts.get(event.row.causation_id)
        try:
            request_hash = (
                ""
                if causal is None
                else evaluation_command_request(
                    IdeaEvaluation(
                        evaluator_agent_id=payload.evaluator_agent_id,
                        idea_id=payload.idea_id,
                        verdict=payload.current_verdict,
                        reason=payload.reason,
                        evidence=payload.evidence,
                        evaluated_at=event.row.occurred_at,
                    ),
                    idempotency_key=causal.idempotency_key,
                ).request_hash
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Evaluation request cannot be reconstructed",
                mismatch_key=key,
            ) from error
        receipt = self._require_idea_receipt(
            event=event,
            run=state.run,
            receipts=receipts,
            scope="inspiration.idea.evaluate",
            resource_type="idea",
            resource_id=payload.idea_id,
            request_hash=request_hash,
            mismatch_key=key,
            allow_event_before_receipt=True,
        )
        expected_body = canonical_json_bytes(
            {
                "data": {
                    "idea_id": payload.idea_id,
                    "maturity": transition.maturity_after,
                    "owner_decision": state.decision,
                    "revision": payload.revision,
                }
            }
        )
        if receipt.response_body != expected_body:
            raise _inspiration_error(
                "Evaluation response does not match its event",
                mismatch_key=key,
            )
        clusters[cluster.state.cluster_id] = IncubationCluster(
            state=_state_after_evaluation(cluster.state, payload),
            members=cluster.members,
            snapshot_hashes=cluster.snapshot_hashes,
        )
        state.latest_evaluations[payload.evaluator_agent_id] = payload
        state.last_signal_at = payload.last_signal_at_after

    @staticmethod
    def _adoption_expected_content(
        idea: InspirationIdeaRow,
        evidence: tuple[SnapshotEvidenceReference, ...],
    ) -> VersionContent:
        key = f"inspiration_adoption:{idea.idea_id}"
        assumptions = _canonical_string_tuple(
            idea.assumptions,
            label="idea assumptions",
            mismatch_key=key,
            nonempty=True,
        )
        predictions = _canonical_string_tuple(
            idea.predictions,
            label="idea predictions",
            mismatch_key=key,
            nonempty=True,
        )
        falsifiers = _canonical_string_tuple(
            idea.falsifiers,
            label="idea falsifiers",
            mismatch_key=key,
            nonempty=True,
        )
        try:
            return VersionContent(
                body=canonical_json_bytes(
                    {
                        "assumptions": assumptions,
                        "hypothesis": idea.hypothesis,
                        "predictions": predictions,
                        "proposed_test": idea.proposed_test,
                    }
                ).decode("utf-8"),
                summary=idea.title,
                mechanism=idea.mechanism,
                tags=("inspiration", f"operator:{idea.operator}"),
                applicability=assumptions,
                evidence=tuple(
                    TypedEvidence(
                        type="inspiration_evidence",
                        id=reference.stable_evidence_key,
                    )
                    for reference in evidence
                ),
                falsifiers=falsifiers,
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Idea cannot map to canonical hypothesis content",
                mismatch_key=key,
            ) from error

    async def _validate_adoption_result(
        self,
        *,
        session: AsyncSession,
        event: _InspirationEvent,
        payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2,
        state: _IdeaSourceState,
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
    ) -> tuple[ExperienceVersionRow, ExperienceStateSnapshotV1]:
        key = f"inspiration_adoption:{payload.adoption_id}"
        identity = await session.get(
            ExperienceRow,
            payload.resulting_experience_id,
        )
        version = await session.get(
            ExperienceVersionRow,
            payload.resulting_version_id,
        )
        version_payload = await session.get(
            ExperiencePayloadRow,
            payload.resulting_version_id,
        )
        if (
            identity is None
            or version is None
            or version_payload is None
            or identity.owner_agent_id != payload.owner_agent_id
            or identity.kind is not ExperienceKind.HYPOTHESIS
            or version.experience_id != identity.experience_id
        ):
            raise _inspiration_error(
                "Adoption resulting experience/version identity is invalid",
                mismatch_key=key,
            )
        # Local import avoids the validation <-> repository initialization cycle.
        from experience_hub.experiences.repository import (
            decode_and_verify_version,
        )

        try:
            actual_content = decode_and_verify_version(
                identity=identity,
                version=version,
                payload=version_payload,
            )
        except SourceIntegrityError as error:
            raise _inspiration_error(
                "Adoption resulting version cannot be decoded",
                mismatch_key=key,
            ) from error
        expected_content = self._adoption_expected_content(
            state.idea,
            payload.evidence,
        )
        if actual_content != expected_content:
            raise _inspiration_error(
                "Adoption result is not the exact mapped hypothesis",
                mismatch_key=key,
            )
        (
            historical_state,
            historical_event,
        ) = await _experience_state_before_inspiration_event(
            session,
            experience_id=identity.experience_id,
            event_id=event.row.event_id,
            mismatch_key=key,
        )
        requested_importance = (
            payload.requested_importance
            if isinstance(payload, InspirationIdeaAdoptedV2)
            else historical_state.importance
            if payload.created
            else None
        )
        requested_confidence = (
            payload.requested_confidence
            if isinstance(payload, InspirationIdeaAdoptedV2)
            else historical_state.confidence
            if payload.created
            else None
        )
        expected_created_temperature = (
            None
            if requested_importance is None
            else (Temperature.HOT if requested_importance >= 0.85 else Temperature.WARM)
        )
        if (
            historical_state.experience_id != identity.experience_id
            or historical_state.owner_agent_id != identity.owner_agent_id
            or historical_state.current_version_id != version.version_id
            or historical_state.current_content_hash != version.content_hash
            or historical_state.temperature is Temperature.ARCHIVED
            or historical_event.occurred_at > event.row.occurred_at
            or version.created_at > event.row.occurred_at
            or (
                payload.created
                and (
                    historical_state.importance != requested_importance
                    or historical_state.confidence != requested_confidence
                    or historical_state.source_trust != 1.0
                    or historical_state.temperature is not expected_created_temperature
                )
            )
        ):
            raise _inspiration_error(
                "Adoption result was unavailable, noncurrent, or archived "
                "at adoption time",
                mismatch_key=key,
            )
        frozen = {
            item.snapshot_item_id: item
            for item in frozen_by_run.get(payload.run_id, ())
        }
        expected_targets: set[UUID] = set()
        for reference in payload.evidence:
            item = frozen.get(reference.id)
            if (
                item is None
                or item.stable_evidence_key != reference.stable_evidence_key
            ):
                raise _inspiration_error(
                    "Adoption evidence does not resolve in its frozen snapshot",
                    mismatch_key=key,
                )
            if item.source_type is not EvidenceSourceType.EXPERIENCE:
                continue
            source_version = await session.get(
                ExperienceVersionRow,
                item.source_version_id,
            )
            source_identity = await session.get(
                ExperienceRow,
                item.source_id,
            )
            if (
                source_version is None
                or source_identity is None
                or source_version.experience_id != source_identity.experience_id
                or source_identity.owner_agent_id != payload.owner_agent_id
                or source_version.content_hash != item.content_hash
            ):
                raise _inspiration_error(
                    "Adoption experience evidence is not an owned immutable version",
                    mismatch_key=key,
                )
            expected_targets.add(source_identity.experience_id)
        links = tuple(
            (
                await session.scalars(
                    select(ExperienceLinkRow)
                    .where(ExperienceLinkRow.source_version_id == version.version_id)
                    .order_by(
                        ExperienceLinkRow.target_experience_id,
                        ExperienceLinkRow.relation,
                    )
                )
            ).all()
        )
        actual_links = {
            link.target_experience_id
            for link in links
            if link.relation is LinkRelation.DERIVED_FROM
        }
        if payload.created and (
            len(links) != len(expected_targets)
            or actual_links != expected_targets
            or any(
                link.source_experience_id != identity.experience_id
                or link.relation is not LinkRelation.DERIVED_FROM
                for link in links
            )
        ):
            raise _inspiration_error(
                "Adoption result links do not match owned frozen evidence",
                mismatch_key=key,
            )
        result_events = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.causation_id == event.row.causation_id,
                        DomainEventRow.aggregate_type == "experience",
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        if payload.created:
            if (
                len(result_events) != 2
                or tuple(row.event_type for row in result_events)
                != (
                    ExperienceCreatedV1.event_type,
                    ExperienceVersionCreatedV1.event_type,
                )
                or identity.origin is not ExperienceOrigin.ADOPTED_IDEA
                or version.version_number != 1
                or any(
                    row.aggregate_id != identity.experience_id
                    or row.actor_agent_id != payload.owner_agent_id
                    or row.occurred_at != event.row.occurred_at
                    or row.event_id >= event.row.event_id
                    for row in result_events
                )
            ):
                raise _inspiration_error(
                    "Created adoption lacks its exact experience event anchors",
                    mismatch_key=key,
                )
        elif result_events:
            raise _inspiration_error(
                "Reused adoption unexpectedly created experience events",
                mismatch_key=key,
            )
        return version, historical_state

    @staticmethod
    def _validate_adoption_receipts(
        *,
        event: _InspirationEvent,
        payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2,
        version: ExperienceVersionRow,
        historical_state: ExperienceStateSnapshotV1,
        receipts: dict[UUID, IdempotencyRecordRow],
    ) -> None:
        key = f"inspiration_adoption:{payload.adoption_id}"
        attached = tuple(
            receipt
            for receipt in receipts.values()
            if receipt.result_resource_type == "idea_adoption"
            and receipt.result_resource_id == payload.adoption_id
        )
        causal = receipts.get(event.row.causation_id)
        if (
            causal is None
            or causal not in attached
            or causal.completed_at is None
            or causal.created_at > event.row.occurred_at
            or causal.completed_at < event.row.occurred_at
        ):
            raise _inspiration_error(
                "Adoption event lacks its creating receipt",
                mismatch_key=key,
            )
        canonical_body: bytes | None = None
        request_parameters = (
            (payload.requested_importance, payload.requested_confidence)
            if isinstance(payload, InspirationIdeaAdoptedV2)
            else (
                (historical_state.importance, historical_state.confidence)
                if payload.created
                else None
            )
        )
        for receipt in attached:
            expected_request_hash: str | None = None
            if request_parameters is not None:
                importance, confidence = request_parameters
                try:
                    expected_request_hash = adoption_command_request(
                        AdoptIdea(
                            owner_agent_id=payload.owner_agent_id,
                            idea_id=payload.idea_id,
                            importance=importance,
                            confidence=confidence,
                        ),
                        idempotency_key=receipt.idempotency_key,
                    ).request_hash
                except (TypeError, ValueError) as error:
                    raise _inspiration_error(
                        "Adoption request cannot be reconstructed",
                        mismatch_key=key,
                    ) from error
            request_hash_valid = (
                receipt.request_hash == expected_request_hash
                if expected_request_hash is not None
                else len(receipt.request_hash) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in receipt.request_hash
                )
            )
            if (
                receipt.caller_scope != f"agent:{payload.owner_agent_id}"
                or receipt.scope != "inspiration.idea.adopt"
                or not request_hash_valid
                or receipt.state != "completed"
                or receipt.response_status_code != 200
                or receipt.response_body is None
                or receipt.response_content_type != "application/json"
                or receipt.response_headers is None
                or receipt.completed_at is None
                or receipt.completed_at < receipt.created_at
                or (
                    receipt is not causal and receipt.created_at < event.row.occurred_at
                )
            ):
                raise _inspiration_error(
                    "Adoption receipt identity or completion is invalid",
                    mismatch_key=key,
                )
            body = _canonical_value(
                receipt.response_body,
                label="adoption response",
                mismatch_key=key,
            )
            headers = _canonical_value(
                receipt.response_headers,
                label="adoption response headers",
                mismatch_key=key,
            )
            if (
                not isinstance(body, dict)
                or set(body) != {"data"}
                or not isinstance(body["data"], dict)
                or set(body["data"]) != {"created", "experience"}
                or not isinstance(body["data"]["experience"], dict)
                or set(body["data"]["experience"])
                != {
                    "current_content_hash",
                    "current_version_id",
                    "experience_id",
                    "owner_agent_id",
                    "temperature",
                }
                or body["data"]["created"] is not payload.created
                or body["data"]["experience"]["current_content_hash"]
                != version.content_hash
                or body["data"]["experience"]["current_version_id"]
                != str(payload.resulting_version_id)
                or body["data"]["experience"]["experience_id"]
                != str(payload.resulting_experience_id)
                or body["data"]["experience"]["owner_agent_id"]
                != str(payload.owner_agent_id)
                or body["data"]["experience"]["temperature"]
                != historical_state.temperature.value
                or headers != {}
            ):
                raise _inspiration_error(
                    "Adoption response is not bound to its result event",
                    mismatch_key=key,
                )
            if canonical_body is None:
                canonical_body = receipt.response_body
            elif receipt.response_body != canonical_body:
                raise _inspiration_error(
                    "Receipts for one adoption retained different responses",
                    mismatch_key=key,
                )

    async def _apply_adoption_source(
        self,
        *,
        session: AsyncSession,
        event: _InspirationEvent,
        payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2,
        state: _IdeaSourceState,
        cluster: IncubationCluster,
        clusters: dict[str, IncubationCluster],
        adoption_by_id: dict[UUID, IdeaAdoptionRecordRow],
        seen_adoptions: set[UUID],
        adopter_owners: dict[str, set[UUID]],
        frozen_by_run: dict[UUID, tuple[SnapshotItem, ...]],
        receipts: dict[UUID, IdempotencyRecordRow],
    ) -> None:
        key = f"inspiration_adoption:{payload.adoption_id}"
        record = adoption_by_id.get(payload.adoption_id)
        occurrence = next(
            (
                row
                for row in (
                    await session.scalars(
                        select(IdeaOccurrenceRow).where(
                            IdeaOccurrenceRow.idea_id == payload.idea_id
                        )
                    )
                ).all()
            ),
            None,
        )
        expected_item_ids = canonical_json_bytes(
            tuple(reference.id for reference in payload.evidence)
        )
        expected_stable_keys = canonical_json_bytes(
            tuple(reference.stable_evidence_key for reference in payload.evidence)
        )
        if (
            record is None
            or payload.adoption_id in seen_adoptions
            or occurrence is None
            or payload.run_id != state.run.run_id
            or payload.owner_agent_id != state.run.owner_agent_id
            or payload.mechanism_cluster_id != cluster.state.cluster_id
            or payload.snapshot_hash != occurrence.snapshot_hash
            or payload.evidence != state.generated.evidence
            or payload.owner_decision_after is not IdeaOwnerDecision.ADOPTED
            or event.row.actor_agent_id != payload.owner_agent_id
            or event.row.occurred_at != payload.last_signal_at_after
            or record.owner_agent_id != payload.owner_agent_id
            or record.idea_id != payload.idea_id
            or record.run_id != payload.run_id
            or record.snapshot_hash != payload.snapshot_hash
            or record.evidence_snapshot_item_ids != expected_item_ids
            or record.evidence_stable_keys != expected_stable_keys
            or record.resulting_experience_id != payload.resulting_experience_id
            or record.resulting_version_id != payload.resulting_version_id
            or record.adopted_at != event.row.occurred_at
        ):
            raise _inspiration_error(
                "Adoption record, idea, run, evidence, and event disagree",
                mismatch_key=key,
            )
        owners = adopter_owners.setdefault(cluster.state.cluster_id, set())
        if len(owners) != cluster.state.distinct_adopter_count:
            raise _inspiration_error(
                "Historical distinct-adopter count is inconsistent",
                mismatch_key=key,
            )
        try:
            transition = plan_adoption_transition(
                cluster=cluster.state,
                owner_already_adopted=payload.owner_agent_id in owners,
                adopted_at=event.row.occurred_at,
            )
        except (TypeError, ValueError) as error:
            raise _inspiration_error(
                "Adoption transition cannot be reconstructed",
                mismatch_key=key,
            ) from error
        if not self._adoption_matches_plan(payload, transition):
            raise _inspiration_error(
                "Adoption event does not match historical mechanism state",
                mismatch_key=key,
            )
        version, historical_state = await self._validate_adoption_result(
            session=session,
            event=event,
            payload=payload,
            state=state,
            frozen_by_run=frozen_by_run,
        )
        self._validate_adoption_receipts(
            event=event,
            payload=payload,
            version=version,
            historical_state=historical_state,
            receipts=receipts,
        )
        owners.add(payload.owner_agent_id)
        seen_adoptions.add(payload.adoption_id)
        clusters[cluster.state.cluster_id] = IncubationCluster(
            state=_state_after_adoption(cluster.state, payload),
            members=cluster.members,
            snapshot_hashes=cluster.snapshot_hashes,
        )
        state.decision = IdeaOwnerDecision.ADOPTED
        state.last_signal_at = payload.last_signal_at_after


def register_inspiration_source_validator(validator: SourceValidator) -> None:
    """Register complete inspiration source validation for replay safety."""
    validator.register(InspirationSourceValidator(validator.event_registry))
