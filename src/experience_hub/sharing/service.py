"""Commands for immutable sharing namespaces and subscriptions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import (
    CommandContext,
    PendingEvent,
    StructuredReason,
    TypedEvidence,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences import (
    ExperienceDraft,
    ExperienceKind,
    ExperienceNotFoundError,
    ExperienceOrigin,
    ExperienceQuery,
    ExperienceRecord,
    ExperienceRepository,
    ExperienceService,
    ExperienceWriter,
    Temperature,
    VersionContent,
    encode_version_content,
)
from experience_hub.ids import IdGenerator
from experience_hub.sharing.confidence import initial_adoption_confidence
from experience_hub.sharing.events import (
    CapsuleAdoptedV1,
    CapsuleFeedbackRecordedV1,
    CapsulePublishedV1,
    CapsuleReceivedV1,
    CapsuleRejectedV1,
    CapsuleRetractedV1,
    SubscriptionCreatedV1,
    TopicCreatedV1,
)
from experience_hub.sharing.hashing import (
    compute_capsule_hash,
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import (
    MAX_PROVENANCE_HOPS,
    AdoptCapsule,
    AdoptionResult,
    Capsule,
    CapsuleStatus,
    CreateSubscription,
    CreateTopic,
    EffectiveAvailability,
    FeedbackRevision,
    FeedbackVerdict,
    InboxItem,
    InboxState,
    ProvenanceHop,
    PublishCapsule,
    RecordCapsuleFeedback,
    RejectInboxItem,
    RetractCapsule,
    SharingMutationReason,
    Subscription,
    Topic,
)
from experience_hub.sharing.repository import (
    AdoptionInboxSource,
    SharingRepository,
    StoredAdoption,
)
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import ReceiptStore, StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

_VALIDATION_TOPIC_ID = UUID(int=0)


def _command_error(
    *,
    code: str,
    message: str,
    status_code: int,
    details: Mapping[str, Any] | None = None,
) -> ReplayableCommandError:
    return ReplayableCommandError(
        code=code,
        message=message,
        details=details,
        status_code=status_code,
    )


def _require_agent_command_scope(
    *,
    command_context: CommandContext,
    agent_id: UUID,
    operation_scope: str,
) -> None:
    if (
        command_context.caller_scope != f"agent:{agent_id}"
        or command_context.operation_scope != operation_scope
    ):
        raise _command_error(
            code="resource_not_found",
            message="The command resource was not found",
            status_code=404,
        )


class SharingService:
    """Write sharing sources and events inside a caller-owned unit of work."""

    def __init__(
        self,
        *,
        clock: Clock,
        id_generator: IdGenerator,
        receipt_store: ReceiptStore,
        repository: SharingRepository | None = None,
        experience_query: ExperienceQuery | None = None,
        experience_writer: ExperienceWriter | None = None,
        experience_repository: ExperienceRepository | None = None,
        experience_service: ExperienceService | None = None,
    ) -> None:
        self._clock = clock
        self._id_generator = id_generator
        self._receipt_store = receipt_store
        self._repository = repository or SharingRepository()
        self._experience_query = experience_query or ExperienceQuery()
        self._experience_writer = experience_writer
        self._experience_repository = experience_repository
        self._experience_service = experience_service

    async def create_topic(
        self,
        *,
        uow: UnitOfWork,
        command: CreateTopic,
        command_context: CommandContext,
    ) -> StoredResponse:
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.owner_agent_id,
            operation_scope="topic.create",
        )
        occurred_at = require_utc(self._clock.now())
        canonical = _validated_topic_input(command, occurred_at=occurred_at)
        if not await self._repository.agent_exists(
            session=uow.session,
            agent_id=canonical.owner_agent_id,
        ):
            raise _command_error(
                code="agent_not_found",
                message="Topic owner was not found",
                status_code=404,
            )
        if (
            await self._repository.find_topic_by_name(
                session=uow.session,
                name=canonical.name,
            )
            is not None
        ):
            raise _command_error(
                code="topic_name_conflict",
                message="A topic with this name already exists",
                details={"name": canonical.name},
                status_code=409,
            )

        topic = canonical.model_copy(update={"topic_id": self._id_generator.new()})
        self._repository.add_topic(session=uow.session, topic=topic)
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="topic",
            resource_id=topic.topic_id,
        )
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="topic",
                    aggregate_id=topic.topic_id,
                    event_type=TopicCreatedV1.event_type,
                    payload=TopicCreatedV1(
                        schema_version=1,
                        topic_id=topic.topic_id,
                        owner_agent_id=topic.owner_agent_id,
                        name=topic.name,
                        description=topic.description,
                    ),
                    actor_agent_id=topic.owner_agent_id,
                    occurred_at=topic.created_at,
                ),
            ),
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "topic_id": topic.topic_id,
                        "owner_agent_id": topic.owner_agent_id,
                        "name": topic.name,
                        "description": topic.description,
                        "created_at": topic.created_at,
                    }
                }
            ),
            headers={"location": f"/v1/topics/{topic.topic_id}"},
        )

    async def create_subscription(
        self,
        *,
        uow: UnitOfWork,
        command: CreateSubscription,
        command_context: CommandContext,
    ) -> StoredResponse:
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.subscriber_agent_id,
            operation_scope="subscription.create",
        )
        occurred_at = require_utc(self._clock.now())
        if not await self._repository.agent_exists(
            session=uow.session,
            agent_id=command.subscriber_agent_id,
        ):
            raise _command_error(
                code="agent_not_found",
                message="Subscriber agent was not found",
                status_code=404,
            )
        topic = await self._repository.get_topic(
            session=uow.session,
            topic_id=command.topic_id,
        )
        if topic is None:
            raise _command_error(
                code="topic_not_found",
                message="Topic was not found",
                status_code=404,
            )
        if occurred_at < topic.created_at:
            raise _command_error(
                code="clock_regression",
                message="Command time precedes the topic creation",
                status_code=409,
            )
        if (
            await self._repository.find_subscription(
                session=uow.session,
                subscriber_agent_id=command.subscriber_agent_id,
                topic_id=command.topic_id,
            )
            is not None
        ):
            raise _command_error(
                code="already_subscribed",
                message="The agent is already subscribed to this topic",
                details={
                    "subscriber_agent_id": command.subscriber_agent_id,
                    "topic_id": command.topic_id,
                },
                status_code=409,
            )

        subscription_id = self._id_generator.new()
        stored = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="subscription",
                    aggregate_id=subscription_id,
                    event_type=SubscriptionCreatedV1.event_type,
                    payload=SubscriptionCreatedV1(
                        schema_version=1,
                        subscription_id=subscription_id,
                        subscriber_agent_id=command.subscriber_agent_id,
                        topic_id=command.topic_id,
                    ),
                    actor_agent_id=command.subscriber_agent_id,
                    occurred_at=occurred_at,
                ),
            ),
        )
        if len(stored) != 1:
            raise RuntimeError("Subscription creation must append exactly one event")
        subscription = Subscription(
            subscription_id=subscription_id,
            subscriber_agent_id=command.subscriber_agent_id,
            topic_id=command.topic_id,
            creation_event_id=stored[0].event_id,
            created_at=occurred_at,
        )
        self._repository.add_subscription(
            session=uow.session,
            subscription=subscription,
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="subscription",
            resource_id=subscription.subscription_id,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes(
                {
                    "data": {
                        "subscription_id": subscription.subscription_id,
                        "subscriber_agent_id": subscription.subscriber_agent_id,
                        "topic_id": subscription.topic_id,
                        "creation_event_id": subscription.creation_event_id,
                        "created_at": subscription.created_at,
                    }
                }
            ),
            headers={"location": f"/v1/subscriptions/{subscription.subscription_id}"},
        )

    async def publish_capsule(
        self,
        *,
        uow: UnitOfWork,
        command: PublishCapsule,
        command_context: CommandContext,
    ) -> StoredResponse:
        """Publish one immutable version and deliver it to prior subscribers."""
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.owner_agent_id,
            operation_scope="capsule.publish",
        )
        receipt = await self._receipt_store.get_by_id(
            session=uow.session,
            receipt_id=command_context.receipt_id,
        )
        if (
            receipt is None
            or receipt.state != "in_progress"
            or (
                receipt.caller_scope,
                receipt.operation_scope,
                receipt.idempotency_key,
                receipt.request_hash,
            )
            != (
                command_context.caller_scope,
                command_context.operation_scope,
                command_context.idempotency_key,
                command_context.request_hash,
            )
        ):
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )
        occurred_at = require_utc(receipt.created_at)
        try:
            expires_at = require_utc(command.expires_at)
        except (AttributeError, TypeError, ValueError) as error:
            raise _command_error(
                code="invalid_expiry",
                message="Capsule expiry must be a timezone-aware datetime",
                status_code=422,
            ) from error
        if expires_at <= occurred_at:
            raise _command_error(
                code="invalid_expiry",
                message="Capsule expiry must be strictly in the future",
                status_code=422,
            )

        topic = await self._repository.get_topic(
            session=uow.session,
            topic_id=command.topic_id,
        )
        if topic is None:
            raise _command_error(
                code="topic_not_found",
                message="Topic was not found",
                status_code=404,
            )
        if occurred_at < topic.created_at:
            raise _command_error(
                code="clock_regression",
                message="Command time precedes the topic creation",
                status_code=409,
            )
        try:
            selected = await self._experience_query.get_owned_shareable_version(
                session=uow.session,
                owner_agent_id=command.owner_agent_id,
                experience_id=command.experience_id,
                version_id=command.version_id,
            )
        except ExperienceNotFoundError:
            raise
        if selected.temperature is Temperature.ARCHIVED:
            raise _command_error(
                code="restore_required",
                message="Archived experience must be restored before publication",
                status_code=409,
            )
        if occurred_at < selected.latest_causal_at:
            raise _command_error(
                code="clock_regression",
                message="Command time precedes the experience causal head",
                status_code=409,
            )

        content = _validated_shareable_content(
            kind=selected.kind,
            content=selected.content,
            expected_content_hash=selected.content_hash,
        )
        provenance_chain: tuple[ProvenanceHop, ...] = ()
        if command.parent_adoption_id is not None:
            parent = await self._repository.get_owned_parent_adoption(
                session=uow.session,
                adopter_agent_id=command.owner_agent_id,
                adoption_id=command.parent_adoption_id,
            )
            if (
                parent is None
                or parent.resulting_experience_id != selected.experience_id
            ):
                raise _command_error(
                    code="parent_adoption_not_found",
                    message="Parent adoption was not found",
                    status_code=404,
                )
            provenance_chain = parent.provenance_chain
            root_fingerprint = parent.root_fingerprint
            if occurred_at < parent.adopted_at:
                raise _command_error(
                    code="clock_regression",
                    message="Command time precedes the parent adoption",
                    status_code=409,
                )
        else:
            if selected.origin is ExperienceOrigin.ADOPTED_CAPSULE:
                raise _command_error(
                    code="parent_adoption_required",
                    message=(
                        "An adopted-capsule experience requires its owned "
                        "parent adoption"
                    ),
                    status_code=422,
                )
            root_fingerprint = compute_original_root_fingerprint(
                root_publisher_id=command.owner_agent_id,
                source_content_hash=selected.content_hash,
            )
        hop_count = len(provenance_chain)
        if hop_count > MAX_PROVENANCE_HOPS:
            raise _command_error(
                code="max_provenance_hops",
                message="Capsule provenance may contain at most four hops",
                status_code=409,
            )

        capsule_id = self._id_generator.new()
        capsule_hash = compute_capsule_hash(
            transport_schema_version=1,
            capsule_id=capsule_id,
            topic_id=command.topic_id,
            source_experience_id=selected.experience_id,
            source_version_id=selected.version_id,
            publisher_agent_id=command.owner_agent_id,
            kind=selected.kind,
            body=content.body,
            summary=content.summary,
            mechanism=content.mechanism,
            tags=content.tags,
            applicability=content.applicability,
            evidence=content.evidence,
            falsifiers=content.falsifiers,
            publisher_confidence=selected.confidence,
            provenance_chain=provenance_chain,
            root_fingerprint=root_fingerprint,
            source_content_hash=selected.content_hash,
            created_at=occurred_at,
            expires_at=expires_at,
            hop_count=hop_count,
        )
        capsule = Capsule(
            capsule_id=capsule_id,
            transport_schema_version=1,
            topic_id=command.topic_id,
            source_experience_id=selected.experience_id,
            source_version_id=selected.version_id,
            publisher_agent_id=command.owner_agent_id,
            kind=selected.kind,
            body=content.body,
            summary=content.summary,
            mechanism=content.mechanism,
            tags=content.tags,
            applicability=content.applicability,
            evidence=content.evidence,
            falsifiers=content.falsifiers,
            publisher_confidence=selected.confidence,
            provenance_chain=provenance_chain,
            root_fingerprint=root_fingerprint,
            source_content_hash=selected.content_hash,
            created_at=occurred_at,
            expires_at=expires_at,
            hop_count=hop_count,
            capsule_hash=capsule_hash,
            status=CapsuleStatus.ACTIVE,
            last_transition_at=occurred_at,
        )
        self._repository.add_capsule(session=uow.session, capsule=capsule)
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="capsule",
            resource_id=capsule.capsule_id,
        )

        published = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="capsule",
                    aggregate_id=capsule.capsule_id,
                    event_type=CapsulePublishedV1.event_type,
                    payload=CapsulePublishedV1(
                        schema_version=1,
                        capsule_id=capsule.capsule_id,
                        topic_id=capsule.topic_id,
                        source_experience_id=capsule.source_experience_id,
                        source_version_id=capsule.source_version_id,
                        publisher_agent_id=capsule.publisher_agent_id,
                        capsule_hash=capsule.capsule_hash,
                        root_fingerprint=capsule.root_fingerprint,
                        status_after=CapsuleStatus.ACTIVE,
                    ),
                    actor_agent_id=command.owner_agent_id,
                    occurred_at=occurred_at,
                ),
            ),
        )
        if len(published) != 1:
            raise RuntimeError("Capsule publication must append exactly one event")
        publication_event_id = published[0].event_id
        await self._deliver_to_subscribers(
            uow=uow,
            command_context=command_context,
            capsule=capsule,
            topic_id=capsule.topic_id,
            publication_event_id=publication_event_id,
            publisher_agent_id=command.owner_agent_id,
        )
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes({"data": capsule}),
            headers={"location": f"/v1/capsules/{capsule.capsule_id}"},
        )

    async def adopt_capsule(
        self,
        *,
        uow: UnitOfWork,
        command: AdoptCapsule,
        command_context: CommandContext,
    ) -> StoredResponse:
        """Adopt quarantined content or corroborate one owned equivalent."""
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.adopter_agent_id,
            operation_scope="capsule.adopt",
        )
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError("Capsule adoption requires a caller-owned immediate UOW")
        if (
            self._experience_writer is None
            or self._experience_repository is None
            or self._experience_service is None
        ):
            raise RuntimeError("Capsule adoption dependencies were not configured")
        occurred_at = require_utc(self._clock.now())
        source = await self._repository.get_owned_adoption_inbox(
            session=uow.session,
            recipient_agent_id=command.adopter_agent_id,
            item_id=command.item_id,
        )
        if source is None:
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )

        prior = await self._repository.get_adoption_for_capsule(
            session=uow.session,
            adopter_agent_id=command.adopter_agent_id,
            capsule=source.capsule,
        )
        if prior is not None:
            return await self._prior_adoption_response(
                uow=uow,
                command_context=command_context,
                source=source,
                adoption=prior,
            )

        self._require_available_pending_adoption(
            source=source,
            occurred_at=occurred_at,
        )
        capsule = source.capsule
        if capsule.publisher_agent_id == command.adopter_agent_id:
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )
        captured_trust = await self._repository.strict_observer_trust(
            session=uow.session,
            subject_agent_id=capsule.publisher_agent_id,
            observer_agent_id=command.adopter_agent_id,
        )
        content = _validated_shareable_content(
            kind=capsule.kind,
            content=VersionContent(
                body=capsule.body,
                summary=capsule.summary,
                mechanism=capsule.mechanism,
                tags=capsule.tags,
                applicability=capsule.applicability,
                evidence=capsule.evidence,
                falsifiers=capsule.falsifiers,
            ),
            expected_content_hash=capsule.source_content_hash,
        )
        equivalent = await self._experience_repository.find_current_equivalent(
            session=uow.session,
            owner_agent_id=command.adopter_agent_id,
            content_hash=capsule.source_content_hash,
        )
        adoption_id = self._id_generator.new()
        provenance_chain = (
            *capsule.provenance_chain,
            ProvenanceHop(
                capsule_id=capsule.capsule_id,
                publisher_agent_id=capsule.publisher_agent_id,
            ),
        )
        created = equivalent is None
        corroboration_applied = False

        if equivalent is None:
            creation = await self._experience_writer.create_from_draft(
                uow=uow,
                draft=ExperienceDraft(
                    owner_agent_id=command.adopter_agent_id,
                    actor_agent_id=command.adopter_agent_id,
                    kind=capsule.kind,
                    origin=ExperienceOrigin.ADOPTED_CAPSULE,
                    content=content,
                    importance=command.importance,
                    confidence=initial_adoption_confidence(
                        capsule.publisher_confidence,
                        captured_trust,
                    ),
                    source_trust=captured_trust,
                    initial_temperature=Temperature.HOT,
                    links=(),
                    occurred_at=occurred_at,
                ),
                command=command_context,
            )
            if creation.content_hash != capsule.source_content_hash:
                raise SourceIntegrityError(
                    "Adopted experience semantic hash changed during creation"
                )
            experience = ExperienceRecord(
                experience_id=creation.experience_id,
                owner_agent_id=command.adopter_agent_id,
                current_version_id=creation.version_id,
                current_content_hash=creation.content_hash,
                temperature=Temperature.HOT,
            )
        else:
            current = await self._experience_repository.get_owned_current(
                session=uow.session,
                owner_agent_id=command.adopter_agent_id,
                experience_id=equivalent.experience_id,
            )
            if current is None:
                raise SourceIntegrityError(
                    "Equivalent experience disappeared during adoption"
                )
            identity, version, state, projection_event = current
            if state.temperature is Temperature.ARCHIVED:
                raise _command_error(
                    code="restore_required",
                    message=("Archived experience must be restored before adoption"),
                    status_code=409,
                )
            _require_equivalent_clock(
                occurred_at=occurred_at,
                identity_created_at=identity.created_at,
                version_created_at=version.created_at,
                projection_occurred_at=projection_event.occurred_at,
                state=state,
            )
            if state.current_content_hash != capsule.source_content_hash:
                raise SourceIntegrityError(
                    "Equivalent experience content hash changed during adoption"
                )
            corroboration_applied = not (
                await self._repository.root_is_represented(
                    session=uow.session,
                    resulting_experience_id=equivalent.experience_id,
                    root_fingerprint=capsule.root_fingerprint,
                )
            )
            experience = ExperienceRecord(
                experience_id=equivalent.experience_id,
                owner_agent_id=equivalent.owner_agent_id,
                current_version_id=state.current_version_id,
                current_content_hash=state.current_content_hash,
                temperature=state.temperature,
            )

        adoption = StoredAdoption(
            adoption_id=adoption_id,
            adopter_agent_id=command.adopter_agent_id,
            capsule_id=capsule.capsule_id,
            resulting_experience_id=experience.experience_id,
            captured_trust=captured_trust,
            provenance_chain=provenance_chain,
            root_fingerprint=capsule.root_fingerprint,
            corroboration_applied=corroboration_applied,
            adopted_at=occurred_at,
        )
        self._repository.add_adoption(
            session=uow.session,
            adoption=adoption,
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)

        if corroboration_applied:
            experience = await self._experience_service.corroborate_from_capsule(
                uow=uow,
                owner_agent_id=command.adopter_agent_id,
                experience_id=experience.experience_id,
                adoption_id=adoption_id,
                capsule_id=capsule.capsule_id,
                root_fingerprint=capsule.root_fingerprint,
                captured_trust=captured_trust,
                occurred_at=occurred_at,
                command_context=command_context,
            )

        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="adoption",
            resource_id=adoption_id,
        )
        stored = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="inbox_item",
                    aggregate_id=source.item_id,
                    event_type=CapsuleAdoptedV1.event_type,
                    payload=CapsuleAdoptedV1(
                        schema_version=1,
                        item_id=source.item_id,
                        capsule_id=capsule.capsule_id,
                        adopter_agent_id=command.adopter_agent_id,
                        adoption_id=adoption_id,
                        resulting_experience_id=experience.experience_id,
                        root_fingerprint=capsule.root_fingerprint,
                        created=created,
                        corroboration_applied=corroboration_applied,
                        state_before=InboxState.PENDING,
                        state_after=InboxState.ADOPTED,
                    ),
                    actor_agent_id=command.adopter_agent_id,
                    occurred_at=occurred_at,
                ),
            ),
        )
        if len(stored) != 1:
            raise RuntimeError("Capsule adoption must append exactly one event")
        return _adoption_response(
            adoption_id=adoption_id,
            experience=experience,
            created=created,
            corroboration_applied=corroboration_applied,
        )

    async def retract_capsule(
        self,
        *,
        uow: UnitOfWork,
        command: RetractCapsule,
        command_context: CommandContext,
    ) -> StoredResponse:
        """Withdraw one publisher-owned active transport without deleting it."""
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.publisher_agent_id,
            operation_scope="capsule.retract",
        )
        occurred_at = require_utc(self._clock.now())
        reason = _normalize_required_reason(command.reason)
        capsule = await self._repository.get_owned_capsule(
            session=uow.session,
            publisher_agent_id=command.publisher_agent_id,
            capsule_id=command.capsule_id,
        )
        if capsule is None:
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )
        if capsule.status is not CapsuleStatus.ACTIVE:
            raise _command_error(
                code="capsule_not_active",
                message="Only an active capsule can be retracted",
                status_code=409,
            )
        latest_causal_at = await self._repository.latest_capsule_causal_at(
            session=uow.session,
            capsule_id=capsule.capsule_id,
        )
        if occurred_at < max(capsule.last_transition_at, latest_causal_at):
            raise _command_error(
                code="clock_regression",
                message="Command time precedes capsule state",
                status_code=409,
            )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="capsule",
            resource_id=capsule.capsule_id,
        )
        stored = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="capsule",
                    aggregate_id=capsule.capsule_id,
                    event_type=CapsuleRetractedV1.event_type,
                    payload=CapsuleRetractedV1(
                        schema_version=1,
                        capsule_id=capsule.capsule_id,
                        publisher_agent_id=command.publisher_agent_id,
                        reason=reason,
                        status_before=CapsuleStatus.ACTIVE,
                        status_after=CapsuleStatus.RETRACTED,
                    ),
                    actor_agent_id=command.publisher_agent_id,
                    occurred_at=occurred_at,
                ),
            ),
        )
        if len(stored) != 1:
            raise RuntimeError("Capsule retraction must append exactly one event")
        retracted = capsule.model_copy(
            update={
                "status": CapsuleStatus.RETRACTED,
                "last_transition_at": occurred_at,
            }
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": retracted}),
            headers={"location": f"/v1/capsules/{capsule.capsule_id}"},
        )

    async def reject_inbox_item(
        self,
        *,
        uow: UnitOfWork,
        command: RejectInboxItem,
        command_context: CommandContext,
    ) -> StoredResponse:
        """Reject one recipient-owned pending route without deleting evidence."""
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.recipient_agent_id,
            operation_scope="capsule.reject",
        )
        occurred_at = require_utc(self._clock.now())
        reason = _normalize_required_reason(command.reason)
        source = await self._repository.get_owned_inbox_item(
            session=uow.session,
            recipient_agent_id=command.recipient_agent_id,
            item_id=command.item_id,
        )
        if source is None:
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )
        if source.state is not InboxState.PENDING:
            raise _command_error(
                code="inbox_item_not_pending",
                message="Only a pending inbox item can be rejected",
                status_code=409,
            )
        if occurred_at < source.latest_causal_at:
            raise _command_error(
                code="clock_regression",
                message="Command time precedes capsule or inbox state",
                status_code=409,
            )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="inbox_item",
            resource_id=source.item_id,
        )
        stored = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="inbox_item",
                    aggregate_id=source.item_id,
                    event_type=CapsuleRejectedV1.event_type,
                    payload=CapsuleRejectedV1(
                        schema_version=1,
                        item_id=source.item_id,
                        capsule_id=source.capsule.capsule_id,
                        recipient_agent_id=command.recipient_agent_id,
                        reason=reason,
                        state_before=InboxState.PENDING,
                        state_after=InboxState.REJECTED,
                    ),
                    actor_agent_id=command.recipient_agent_id,
                    occurred_at=occurred_at,
                ),
            ),
        )
        if len(stored) != 1:
            raise RuntimeError("Capsule rejection must append exactly one event")
        capsule = source.capsule
        if capsule.status is CapsuleStatus.RETRACTED:
            availability = EffectiveAvailability.RETRACTED
        elif occurred_at >= capsule.expires_at:
            availability = EffectiveAvailability.EXPIRED
        else:
            availability = EffectiveAvailability.ACTIVE
        rejected = InboxItem(
            item_id=source.item_id,
            recipient_agent_id=source.recipient_agent_id,
            capsule_id=capsule.capsule_id,
            capsule=capsule,
            state=InboxState.REJECTED,
            effective_availability=availability,
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes({"data": rejected}),
            headers={"location": f"/v1/inbox/{source.item_id}"},
        )

    async def record_capsule_feedback(
        self,
        *,
        uow: UnitOfWork,
        command: RecordCapsuleFeedback,
        command_context: CommandContext,
    ) -> StoredResponse:
        """Append one feedback revision and revise future observer trust."""
        _require_agent_command_scope(
            command_context=command_context,
            agent_id=command.observer_agent_id,
            operation_scope="capsule.feedback",
        )
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError("Capsule feedback requires a caller-owned immediate UOW")
        occurred_at = require_utc(self._clock.now())
        reason = _normalize_required_reason(command.reason)
        evidence = _validated_feedback_evidence(command.evidence)
        authorization = await self._repository.get_feedback_authorized_capsule(
            session=uow.session,
            observer_agent_id=command.observer_agent_id,
            capsule_id=command.capsule_id,
        )
        if authorization is None:
            raise _command_error(
                code="resource_not_found",
                message="The command resource was not found",
                status_code=404,
            )
        capsule = authorization.capsule
        previous = await self._repository.get_latest_feedback(
            session=uow.session,
            observer_agent_id=command.observer_agent_id,
            capsule_id=capsule.capsule_id,
        )
        reputation = await self._repository.get_reputation(
            session=uow.session,
            subject_agent_id=capsule.publisher_agent_id,
            observer_agent_id=command.observer_agent_id,
        )
        if previous is not None and reputation is None:
            raise SourceIntegrityError(
                "Feedback revisions exist without a reputation projection",
                mismatch_key=(
                    f"reputation:{capsule.publisher_agent_id}:"
                    f"{command.observer_agent_id}"
                ),
            )
        latest_capsule_at = await self._repository.latest_capsule_causal_at(
            session=uow.session,
            capsule_id=capsule.capsule_id,
        )
        causal_times = [
            capsule.created_at,
            capsule.last_transition_at,
            authorization.inbox_state_at,
            latest_capsule_at,
        ]
        if previous is not None:
            causal_times.append(previous.created_at)
        if reputation is not None:
            causal_times.append(reputation.last_feedback_at)
        if occurred_at < max(causal_times):
            raise _command_error(
                code="clock_regression",
                message=(
                    "Command time precedes capsule, feedback, or reputation state"
                ),
                status_code=409,
            )

        revision = 1 if previous is None else previous.revision + 1
        previous_verdict = None if previous is None else previous.verdict
        alpha_before = 2 if reputation is None else reputation.alpha
        beta_before = 2 if reputation is None else reputation.beta
        alpha_after, beta_after = _revised_effective_counts(
            alpha=alpha_before,
            beta=beta_before,
            previous_verdict=previous_verdict,
            current_verdict=command.verdict,
        )
        try:
            feedback = FeedbackRevision(
                feedback_id=self._id_generator.new(),
                observer_agent_id=command.observer_agent_id,
                capsule_id=capsule.capsule_id,
                revision=revision,
                verdict=command.verdict,
                reason=reason,
                evidence=evidence,
                created_at=occurred_at,
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise _command_error(
                code="invalid_feedback",
                message="Feedback verdict or evidence is invalid",
                status_code=422,
            ) from error
        self._repository.add_feedback(
            session=uow.session,
            feedback=feedback,
        )
        await uow.session.flush()
        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="feedback",
            resource_id=feedback.feedback_id,
        )
        stored = await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="capsule",
                    aggregate_id=capsule.capsule_id,
                    event_type=CapsuleFeedbackRecordedV1.event_type,
                    payload=CapsuleFeedbackRecordedV1(
                        schema_version=1,
                        feedback_id=feedback.feedback_id,
                        observer_agent_id=feedback.observer_agent_id,
                        capsule_id=feedback.capsule_id,
                        publisher_agent_id=capsule.publisher_agent_id,
                        revision=feedback.revision,
                        previous_verdict=previous_verdict,
                        current_verdict=feedback.verdict,
                        alpha_before=alpha_before,
                        beta_before=beta_before,
                        alpha_after=alpha_after,
                        beta_after=beta_after,
                    ),
                    actor_agent_id=feedback.observer_agent_id,
                    occurred_at=feedback.created_at,
                ),
            ),
        )
        if len(stored) != 1:
            raise RuntimeError("Capsule feedback must append exactly one event")
        return StoredResponse(
            status_code=201,
            body=canonical_json_bytes({"data": feedback}),
            headers={"location": f"/v1/feedback/{feedback.feedback_id}"},
        )

    async def _prior_adoption_response(
        self,
        *,
        uow: UnitOfWork,
        command_context: CommandContext,
        source: AdoptionInboxSource,
        adoption: StoredAdoption,
    ) -> StoredResponse:
        adopted_event = source.adopted_event
        if (
            adopted_event is None
            or adopted_event.adoption_id != adoption.adoption_id
            or adopted_event.resulting_experience_id != adoption.resulting_experience_id
            or adopted_event.root_fingerprint != adoption.root_fingerprint
            or adopted_event.corroboration_applied is not adoption.corroboration_applied
        ):
            raise SourceIntegrityError(
                "Prior adoption does not match the inbox checkpoint"
            )
        assert self._experience_repository is not None
        current = await self._experience_repository.get_owned_current(
            session=uow.session,
            owner_agent_id=adoption.adopter_agent_id,
            experience_id=adoption.resulting_experience_id,
        )
        if current is None:
            raise SourceIntegrityError("Prior adoption resulting experience is missing")
        _, _, state, _ = current
        experience = ExperienceRecord(
            experience_id=adoption.resulting_experience_id,
            owner_agent_id=adoption.adopter_agent_id,
            current_version_id=state.current_version_id,
            current_content_hash=state.current_content_hash,
            temperature=state.temperature,
        )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="adoption",
            resource_id=adoption.adoption_id,
        )
        return _adoption_response(
            adoption_id=adoption.adoption_id,
            experience=experience,
            created=adopted_event.created,
            corroboration_applied=adoption.corroboration_applied,
        )

    @staticmethod
    def _require_available_pending_adoption(
        *,
        source: AdoptionInboxSource,
        occurred_at: datetime,
    ) -> None:
        if source.state is not InboxState.PENDING:
            raise _command_error(
                code="inbox_item_not_pending",
                message="Only a pending inbox item can be adopted",
                status_code=409,
            )
        if occurred_at < source.latest_causal_at:
            raise _command_error(
                code="clock_regression",
                message="Command time precedes capsule or inbox state",
                status_code=409,
            )
        if source.capsule.status is CapsuleStatus.RETRACTED:
            raise _command_error(
                code="capsule_retracted",
                message="A retracted capsule cannot be adopted",
                status_code=409,
            )
        if occurred_at >= source.capsule.expires_at:
            raise _command_error(
                code="capsule_expired",
                message="An expired capsule cannot be adopted",
                status_code=409,
            )

    async def _deliver_to_subscribers(
        self,
        *,
        uow: UnitOfWork,
        command_context: CommandContext,
        capsule: Capsule,
        topic_id: UUID,
        publication_event_id: int,
        publisher_agent_id: UUID,
    ) -> None:
        """Append deterministic deliveries one bounded keyset page at a time."""
        after: tuple[UUID, UUID] | None = None
        while True:
            page = await self._repository.list_eligible_subscriptions(
                session=uow.session,
                topic_id=topic_id,
                publication_event_id=publication_event_id,
                after=after,
                limit=100,
                exclude_subscriber_agent_id=publisher_agent_id,
            )
            received_events = tuple(
                PendingEvent(
                    aggregate_type="inbox_item",
                    aggregate_id=item_id,
                    event_type=CapsuleReceivedV1.event_type,
                    payload=CapsuleReceivedV1(
                        schema_version=1,
                        item_id=item_id,
                        capsule_id=capsule.capsule_id,
                        recipient_agent_id=subscription.subscriber_agent_id,
                        state_after=InboxState.PENDING,
                    ),
                    actor_agent_id=publisher_agent_id,
                    occurred_at=capsule.created_at,
                )
                for subscription, item_id in (
                    (subscription, self._id_generator.new()) for subscription in page
                )
            )
            if received_events:
                delivered = await uow.append_events(
                    command_context,
                    received_events,
                )
                if len(delivered) != len(received_events):
                    raise RuntimeError(
                        "Capsule delivery must append one event per recipient"
                    )
            if len(page) < 100:
                return
            tail = page[-1]
            after = (tail.subscriber_agent_id, tail.subscription_id)


def _validated_topic_input(
    command: CreateTopic,
    *,
    occurred_at: datetime,
) -> Topic:
    try:
        return Topic(
            topic_id=_VALIDATION_TOPIC_ID,
            owner_agent_id=command.owner_agent_id,
            name=command.name,
            description=command.description,
            created_at=occurred_at,
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise _command_error(
            code="invalid_topic",
            message="Topic name or description is invalid",
            status_code=422,
        ) from error


def _validated_shareable_content(
    *,
    kind: ExperienceKind,
    content: VersionContent,
    expected_content_hash: str,
) -> VersionContent:
    """Revalidate query output and reproduce its semantic source digest."""
    try:
        evidence = tuple(
            TypedEvidence(
                type=item.type,
                id=item.id,
            )
            if isinstance(item, TypedEvidence)
            else TypedEvidence.model_validate(item)
            for item in content.evidence
        )
        validated = VersionContent(
            body=content.body,
            summary=content.summary,
            mechanism=content.mechanism,
            tags=content.tags,
            applicability=content.applicability,
            evidence=evidence,
            falsifiers=content.falsifiers,
        )
        encoded = encode_version_content(kind=kind, content=validated)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise SourceIntegrityError("Shareable experience content is invalid") from error
    if encoded.content_hash != expected_content_hash:
        raise SourceIntegrityError(
            "Shareable experience content hash does not match its source"
        )
    return validated


def _require_equivalent_clock(
    *,
    occurred_at: datetime,
    identity_created_at: datetime,
    version_created_at: datetime,
    projection_occurred_at: datetime,
    state: Any,
) -> None:
    causal_times = [
        require_utc(identity_created_at),
        require_utc(version_created_at),
        require_utc(projection_occurred_at),
        require_utc(state.strength_updated_at),
        require_utc(state.last_transition_at),
    ]
    causal_times.extend(
        require_utc(value)
        for value in (
            state.last_accessed_at,
            state.last_lifecycle_evaluated_at,
        )
        if value is not None
    )
    if occurred_at < max(causal_times):
        raise _command_error(
            code="clock_regression",
            message="Command time precedes existing experience state",
            status_code=409,
        )


def _adoption_response(
    *,
    adoption_id: UUID,
    experience: ExperienceRecord,
    created: bool,
    corroboration_applied: bool,
) -> StoredResponse:
    result = AdoptionResult(
        experience=experience,
        created=created,
        corroboration_applied=corroboration_applied,
    )
    return StoredResponse(
        status_code=200,
        body=canonical_json_bytes({"data": result}),
        headers={"location": f"/v1/adoptions/{adoption_id}"},
    )


def _normalize_required_reason(
    reason: SharingMutationReason,
) -> StructuredReason:
    try:
        if isinstance(reason, StructuredReason):
            return reason
        if not isinstance(reason, str):
            raise TypeError("reason must be a string or StructuredReason")
        return StructuredReason.from_user_text(reason)
    except (TypeError, ValidationError, ValueError) as error:
        raise _command_error(
            code="invalid_reason",
            message="A nonempty structured reason is required",
            status_code=422,
        ) from error


def _revised_effective_counts(
    *,
    alpha: int,
    beta: int,
    previous_verdict: FeedbackVerdict | None,
    current_verdict: FeedbackVerdict,
) -> tuple[int, int]:
    revised_alpha = alpha
    revised_beta = beta
    if previous_verdict is FeedbackVerdict.USEFUL:
        revised_alpha -= 1
    elif previous_verdict in (
        FeedbackVerdict.REFUTED,
        FeedbackVerdict.HARMFUL,
    ):
        revised_beta -= 1
    if current_verdict is FeedbackVerdict.USEFUL:
        revised_alpha += 1
    else:
        revised_beta += 1
    if revised_alpha < 2 or revised_beta < 2:
        raise SourceIntegrityError(
            "Feedback prior contribution is absent from reputation"
        )
    return revised_alpha, revised_beta


def _validated_feedback_evidence(
    evidence: tuple[TypedEvidence, ...],
) -> tuple[TypedEvidence, ...]:
    try:
        return tuple(TypedEvidence(type=item.type, id=item.id) for item in evidence)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise _command_error(
            code="invalid_feedback",
            message="Feedback verdict or evidence is invalid",
            status_code=422,
        ) from error


__all__ = ["SharingService"]
