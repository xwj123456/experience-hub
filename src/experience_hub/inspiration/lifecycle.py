"""Transaction-bound idea evaluation and deterministic decay planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import Clock, require_utc
from experience_hub.domain import (
    CommandContext,
    PendingEvent,
    StructuredReason,
)
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import encode_version_content
from experience_hub.experiences.contracts import (
    ExperienceDraft,
    ExperienceRecord,
    VersionLinkInput,
)
from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    LinkRelation,
    Temperature,
    VersionContent,
)
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.ids import IdGenerator
from experience_hub.inspiration.adoption import (
    adopted_hypothesis_content,
    decode_idea_evidence,
)
from experience_hub.inspiration.commands import (
    AdoptIdea,
    ArchiveIdea,
    RejectIdea,
)
from experience_hub.inspiration.events import (
    InspirationIdeaAdoptedV1,
    InspirationIdeaAdoptedV2,
    InspirationIdeaArchivedV1,
    InspirationIdeaEvaluatedV1,
    InspirationIdeaRejectedV1,
)
from experience_hub.inspiration.incubation import (
    plan_adoption_transition,
    plan_evaluation_transition,
)
from experience_hub.inspiration.models import (
    EvidenceSourceState,
    EvidenceSourceType,
    ExperienceVersionEvidenceReference,
    IdeaEvaluation,
    IdeaOwnerDecision,
    MechanismIncubation,
    MechanismMaturity,
    SnapshotEvidenceReference,
)
from experience_hub.inspiration.repository import (
    InspirationRepository,
    InspirationSourceIntegrityError,
)
from experience_hub.inspiration.request_hashing import (
    adoption_command_request,
    decision_command_request,
    evaluation_command_request,
)
from experience_hub.storage.idempotency import (
    ReceiptRecord,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceVersionRow,
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)
from experience_hub.storage.unit_of_work import UnitOfWork

_NONCANDIDATE_RETENTION = timedelta(days=180)
_CANDIDATE_RETENTION = timedelta(days=365)


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _command_error(
    *,
    code: str,
    message: str,
    status_code: int,
) -> ReplayableCommandError:
    return ReplayableCommandError(
        code=code,
        message=message,
        status_code=status_code,
    )


def _not_found() -> ReplayableCommandError:
    return _command_error(
        code="resource_not_found",
        message="The command resource was not found",
        status_code=404,
    )


def _clock_regression() -> ReplayableCommandError:
    return _command_error(
        code="clock_regression",
        message="Command time precedes existing idea state",
        status_code=409,
    )


def _invalid_evidence() -> ReplayableCommandError:
    return _command_error(
        code="invalid_evidence",
        message="Evaluation evidence does not resolve to an allowed source",
        status_code=422,
    )


def _idea_not_evaluable() -> ReplayableCommandError:
    return _command_error(
        code="idea_not_evaluable",
        message="The idea is already in a terminal owner decision",
        status_code=409,
    )


def _idea_not_decidable() -> ReplayableCommandError:
    return _command_error(
        code="idea_not_decidable",
        message="The idea is not eligible for this owner decision",
        status_code=409,
    )


def _invalid_reason() -> ReplayableCommandError:
    return _command_error(
        code="invalid_reason",
        message="The owner decision reason is invalid",
        status_code=422,
    )


def _restore_required() -> ReplayableCommandError:
    return _command_error(
        code="restore_required",
        message="Archived experiences must be restored before mutation",
        status_code=409,
    )


def _adoption_request_mismatch() -> ReplayableCommandError:
    return _command_error(
        code="adoption_request_mismatch",
        message="The idea was adopted with different parameters",
        status_code=409,
    )


def _strict_reason(value: StructuredReason | str) -> StructuredReason:
    try:
        if isinstance(value, str):
            return StructuredReason.from_user_text(value)
        if not isinstance(value, StructuredReason):
            raise TypeError("reason must be structured or textual")
        return StructuredReason.model_validate(
            value.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _invalid_reason() from error


def _strict_evaluation(value: IdeaEvaluation) -> IdeaEvaluation:
    if not isinstance(value, IdeaEvaluation):
        raise ValueError("evaluation must be an IdeaEvaluation")
    try:
        return IdeaEvaluation.model_validate(
            value.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _command_error(
            code="invalid_evaluation",
            message="The idea evaluation is invalid",
            status_code=422,
        ) from error


def _cluster_value(row: MechanismIncubationRow) -> MechanismIncubation:
    try:
        decoded = json.loads(row.member_hashes)
        if (
            not isinstance(decoded, list)
            or any(not isinstance(value, str) for value in decoded)
            or canonical_json_bytes(decoded) != row.member_hashes
        ):
            raise ValueError("member hashes are not canonical")
        return MechanismIncubation(
            cluster_id=row.cluster_id,
            canonical_mechanism_hash=row.canonical_mechanism_hash,
            member_hashes=tuple(decoded),
            occurrence_count=row.occurrence_count,
            distinct_snapshot_count=row.distinct_snapshot_count,
            distinct_adopter_count=row.distinct_adopter_count,
            supported_count=row.supported_count,
            refuted_count=row.refuted_count,
            maturity=MechanismMaturity(row.maturity),
            candidate_since=row.candidate_since,
            last_signal_at=row.last_signal_at,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise InspirationSourceIntegrityError(
            "mechanism incubation projection is invalid"
        ) from error


def _idea_evidence(
    raw: bytes,
) -> tuple[SnapshotEvidenceReference, ...]:
    try:
        return decode_idea_evidence(raw)
    except ValueError as error:
        raise InspirationSourceIntegrityError(
            "idea evidence source is invalid"
        ) from error


def _adopted_content(
    *,
    idea: InspirationIdeaRow,
    evidence: tuple[SnapshotEvidenceReference, ...],
) -> VersionContent:
    try:
        return adopted_hypothesis_content(
            idea=idea,
            evidence=evidence,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise InspirationSourceIntegrityError(
            "idea cannot be mapped to a hypothesis experience"
        ) from error


def _experience_latest_causal_at(
    *,
    identity: ExperienceRow,
    version: ExperienceVersionRow,
    state: ExperienceStateRow,
    projection_event: DomainEventRow,
) -> datetime:
    values = [
        identity.created_at,
        version.created_at,
        projection_event.occurred_at,
        state.strength_updated_at,
        state.last_transition_at,
    ]
    values.extend(
        value
        for value in (
            state.last_accessed_at,
            state.last_lifecycle_evaluated_at,
        )
        if value is not None
    )
    return max(require_utc(value) for value in values)


async def _experience_state_before_event(
    *,
    session: AsyncSession,
    experience_id: UUID,
    event_id: int,
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
        raise InspirationSourceIntegrityError(
            "adopted experience has no state before the adoption event"
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
    except (
        TypeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        raise InspirationSourceIntegrityError(
            "adopted experience historical state is invalid"
        ) from error
    return state, row


@dataclass(frozen=True, slots=True)
class _EvaluationContext:
    idea: InspirationIdeaRow
    state: IdeaStateRow
    cluster: MechanismIncubation
    run: InspirationRunRow
    cluster_row: MechanismIncubationRow


@dataclass(frozen=True, slots=True)
class _AdoptionMaterial:
    snapshot_hash: str
    evidence: tuple[SnapshotEvidenceReference, ...]
    content: VersionContent
    links: tuple[VersionLinkInput, ...]


class IdeaLifecycleService:
    """Evaluate one private idea and update its effective cluster signal."""

    def __init__(
        self,
        *,
        clock: Clock,
        receipt_store: ReceiptStore,
        repository: InspirationRepository,
        id_generator: IdGenerator | None = None,
        experience_writer: ExperienceWriter | None = None,
        experience_repository: ExperienceRepository | None = None,
    ) -> None:
        self._clock = clock
        self._receipt_store = receipt_store
        self._repository = repository
        self._id_generator = id_generator
        self._experience_writer = experience_writer
        self._experience_repository = experience_repository

    async def _current_command_receipt(
        self,
        *,
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> ReceiptRecord:
        receipt = await self._receipt_store.get_by_id(
            session=uow.session,
            receipt_id=command_context.receipt_id,
        )
        if (
            receipt is None
            or receipt.state != "in_progress"
            or receipt.caller_scope != command_context.caller_scope
            or receipt.operation_scope != command_context.operation_scope
            or receipt.idempotency_key != command_context.idempotency_key
            or receipt.request_hash != command_context.request_hash
        ):
            raise InspirationSourceIntegrityError(
                "idea command receipt and command context disagree"
            )
        return receipt

    async def _server_command_time(
        self,
        *,
        uow: UnitOfWork,
        command_context: CommandContext,
    ) -> datetime:
        receipt = await self._current_command_receipt(
            uow=uow,
            command_context=command_context,
        )
        return max(
            require_utc(receipt.created_at),
            require_utc(self._clock.now()),
        )

    async def evaluate(
        self,
        *,
        uow: UnitOfWork,
        evaluation: IdeaEvaluation,
        command_context: CommandContext,
    ) -> StoredResponse:
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError("Idea evaluation requires a caller-owned immediate UOW")
        if not isinstance(command_context, CommandContext):
            raise ValueError("command_context must be a CommandContext")
        retained = _strict_evaluation(evaluation)
        if (
            command_context.caller_scope != f"agent:{retained.evaluator_agent_id}"
            or command_context.operation_scope != "inspiration.idea.evaluate"
        ):
            raise _not_found()
        try:
            expected_request = evaluation_command_request(
                retained,
                idempotency_key=command_context.idempotency_key,
            )
        except (TypeError, ValueError) as error:
            raise _not_found() from error
        if command_context.request_hash != expected_request.request_hash:
            raise _not_found()

        try:
            evaluated_at = require_utc(retained.evaluated_at)
            now = require_utc(self._clock.now())
        except (TypeError, ValueError) as error:
            raise _command_error(
                code="invalid_evaluated_at",
                message="Evaluation time must be UTC-aware",
                status_code=422,
            ) from error
        if evaluated_at > now:
            raise _command_error(
                code="invalid_evaluated_at",
                message="Evaluation time must not be in the future",
                status_code=422,
            )

        context = await self._load_context(
            session=uow.session,
            idea_id=retained.idea_id,
            evaluator_agent_id=retained.evaluator_agent_id,
        )
        try:
            owner_decision = IdeaOwnerDecision(context.state.owner_decision)
        except ValueError as error:
            raise InspirationSourceIntegrityError(
                "idea owner-decision projection is invalid"
            ) from error
        if owner_decision in {
            IdeaOwnerDecision.ADOPTED,
            IdeaOwnerDecision.REJECTED,
        }:
            raise _idea_not_evaluable()

        await self._validate_evidence(
            session=uow.session,
            evaluation=retained,
            idea=context.idea,
        )
        previous = await self._repository.latest_evaluation(
            session=uow.session,
            idea_id=retained.idea_id,
            evaluator_agent_id=retained.evaluator_agent_id,
        )
        latest_causal_at = await self._latest_causal_at(
            session=uow.session,
            state=context.state,
            cluster_last_signal_at=context.cluster.last_signal_at,
            cluster_projection_event_id=(
                await self._cluster_projection_event_id(
                    session=uow.session,
                    cluster_id=context.cluster.cluster_id,
                )
            ),
        )
        if evaluated_at < latest_causal_at:
            raise _clock_regression()
        try:
            transition = plan_evaluation_transition(
                cluster=context.cluster,
                previous_verdict=(
                    None if previous is None else previous.current_verdict
                ),
                current_verdict=retained.verdict,
                evaluated_at=evaluated_at,
            )
        except (TypeError, ValueError) as error:
            raise InspirationSourceIntegrityError(
                "idea evaluation transition is inconsistent with its projections"
            ) from error

        revision = 1 if previous is None else previous.revision + 1
        payload = InspirationIdeaEvaluatedV1(
            schema_version=1,
            idea_id=retained.idea_id,
            evaluator_agent_id=retained.evaluator_agent_id,
            mechanism_cluster_id=context.cluster.cluster_id,
            revision=revision,
            previous_verdict=transition.previous_verdict,
            current_verdict=transition.current_verdict,
            evidence=retained.evidence,
            reason=retained.reason,
            owner_decision_before=owner_decision,
            owner_decision_after=owner_decision,
            supported_count_before=transition.supported_count_before,
            supported_count_after=transition.supported_count_after,
            refuted_count_before=transition.refuted_count_before,
            refuted_count_after=transition.refuted_count_after,
            maturity_before=transition.maturity_before,
            maturity_after=transition.maturity_after,
            candidate_since_before=transition.candidate_since_before,
            candidate_since_after=transition.candidate_since_after,
            last_signal_at_before=transition.last_signal_at_before,
            last_signal_at_after=transition.last_signal_at_after,
        )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="idea",
            resource_id=retained.idea_id,
        )
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=retained.idea_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=retained.evaluator_agent_id,
                    occurred_at=evaluated_at,
                ),
            ),
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes(
                {
                    "data": {
                        "idea_id": retained.idea_id,
                        "maturity": transition.maturity_after,
                        "owner_decision": owner_decision,
                        "revision": revision,
                    }
                }
            ),
        )

    async def reject(
        self,
        *,
        uow: UnitOfWork,
        command: RejectIdea,
        command_context: CommandContext,
    ) -> StoredResponse:
        if not isinstance(command, RejectIdea):
            raise ValueError("command must be a RejectIdea")
        return await self._decide(
            uow=uow,
            command=command,
            command_context=command_context,
            decision=IdeaOwnerDecision.REJECTED,
        )

    async def archive(
        self,
        *,
        uow: UnitOfWork,
        command: ArchiveIdea,
        command_context: CommandContext,
    ) -> StoredResponse:
        if not isinstance(command, ArchiveIdea):
            raise ValueError("command must be an ArchiveIdea")
        return await self._decide(
            uow=uow,
            command=command,
            command_context=command_context,
            decision=IdeaOwnerDecision.ARCHIVED,
        )

    async def adopt(
        self,
        *,
        uow: UnitOfWork,
        command: AdoptIdea,
        command_context: CommandContext,
    ) -> StoredResponse:
        if not isinstance(command, AdoptIdea):
            raise ValueError("command must be an AdoptIdea")
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError("Idea adoption requires a caller-owned immediate UOW")
        if not isinstance(command_context, CommandContext):
            raise ValueError("command_context must be a CommandContext")
        if (
            command_context.caller_scope != f"agent:{command.owner_agent_id}"
            or command_context.operation_scope != "inspiration.idea.adopt"
        ):
            raise _not_found()
        try:
            expected_request = adoption_command_request(
                command,
                idempotency_key=command_context.idempotency_key,
            )
        except (TypeError, ValueError) as error:
            raise _not_found() from error
        if command_context.request_hash != expected_request.request_hash:
            raise _not_found()
        context = await self._load_context(
            session=uow.session,
            idea_id=command.idea_id,
            evaluator_agent_id=command.owner_agent_id,
        )
        try:
            before = IdeaOwnerDecision(context.state.owner_decision)
        except ValueError as error:
            raise InspirationSourceIntegrityError(
                "idea owner-decision projection is invalid"
            ) from error
        if before is IdeaOwnerDecision.ADOPTED:
            return await self._existing_adoption_response(
                uow=uow,
                context=context,
                command=command,
                command_context=command_context,
            )
        if before is IdeaOwnerDecision.REJECTED:
            raise _idea_not_decidable()
        if (
            self._id_generator is None
            or self._experience_writer is None
            or self._experience_repository is None
        ):
            raise RuntimeError("Idea adoption dependencies are not configured")
        run_id = context.idea.run_id

        existing_record = await uow.session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == command.idea_id
            )
        )
        if existing_record is not None:
            raise InspirationSourceIntegrityError(
                "a non-adopted idea already has adoption provenance"
            )
        adopted_at = await self._server_command_time(
            uow=uow,
            command_context=command_context,
        )
        latest_causal_at = await self._latest_causal_at(
            session=uow.session,
            state=context.state,
            cluster_last_signal_at=context.cluster.last_signal_at,
            cluster_projection_event_id=(context.cluster_row.projection_event_id),
        )
        if adopted_at < latest_causal_at:
            raise _clock_regression()

        material = await self._adoption_material(
            session=uow.session,
            context=context,
        )
        encoded = encode_version_content(
            kind=ExperienceKind.HYPOTHESIS,
            content=material.content,
        )
        equivalent = await self._experience_writer.find_current_equivalent(
            session=uow.session,
            owner_agent_id=command.owner_agent_id,
            content_hash=encoded.content_hash,
        )
        created = equivalent is None
        if equivalent is not None:
            current = await self._experience_repository.get_owned_current(
                session=uow.session,
                owner_agent_id=command.owner_agent_id,
                experience_id=equivalent.experience_id,
            )
            if current is None:
                raise InspirationSourceIntegrityError(
                    "equivalent experience is not owner-visible"
                )
            identity, version, state, projection_event = current
            if (
                equivalent.current_version_id != version.version_id
                or equivalent.current_content_hash != version.content_hash
                or equivalent.temperature is not state.temperature
            ):
                raise InspirationSourceIntegrityError(
                    "equivalent experience projection is inconsistent"
                )
            if state.temperature is Temperature.ARCHIVED:
                raise _restore_required()
            if adopted_at < _experience_latest_causal_at(
                identity=identity,
                version=version,
                state=state,
                projection_event=projection_event,
            ):
                raise _clock_regression()

        adopter_owners = await self._cluster_adopter_owners(
            session=uow.session,
            cluster_id=context.cluster.cluster_id,
        )
        if len(adopter_owners) != context.cluster.distinct_adopter_count:
            raise InspirationSourceIntegrityError(
                "cluster distinct-adopter projection is inconsistent"
            )
        try:
            transition = plan_adoption_transition(
                cluster=context.cluster,
                owner_already_adopted=(command.owner_agent_id in adopter_owners),
                adopted_at=adopted_at,
            )
        except (TypeError, ValueError) as error:
            raise InspirationSourceIntegrityError(
                "idea adoption transition is inconsistent with its projections"
            ) from error

        if equivalent is None:
            temperature = (
                Temperature.HOT if command.importance >= 0.85 else Temperature.WARM
            )
            creation = await self._experience_writer.create_from_draft(
                uow=uow,
                draft=ExperienceDraft(
                    owner_agent_id=command.owner_agent_id,
                    actor_agent_id=command.owner_agent_id,
                    kind=ExperienceKind.HYPOTHESIS,
                    origin=ExperienceOrigin.ADOPTED_IDEA,
                    content=material.content,
                    importance=command.importance,
                    confidence=command.confidence,
                    source_trust=1.0,
                    initial_temperature=temperature,
                    links=material.links,
                    occurred_at=adopted_at,
                ),
                command=command_context,
            )
            if creation.content_hash != encoded.content_hash:
                raise InspirationSourceIntegrityError(
                    "created experience content hash is inconsistent"
                )
            experience = ExperienceRecord(
                experience_id=creation.experience_id,
                owner_agent_id=command.owner_agent_id,
                current_version_id=creation.version_id,
                current_content_hash=creation.content_hash,
                temperature=temperature,
            )
        else:
            experience = equivalent

        adoption_id = self._id_generator.new()
        uow.session.add(
            IdeaAdoptionRecordRow(
                adoption_id=adoption_id,
                owner_agent_id=command.owner_agent_id,
                idea_id=command.idea_id,
                run_id=run_id,
                snapshot_hash=material.snapshot_hash,
                evidence_snapshot_item_ids=canonical_json_bytes(
                    tuple(reference.id for reference in material.evidence)
                ),
                evidence_stable_keys=canonical_json_bytes(
                    tuple(
                        reference.stable_evidence_key for reference in material.evidence
                    )
                ),
                resulting_experience_id=experience.experience_id,
                resulting_version_id=experience.current_version_id,
                adopted_at=adopted_at,
            )
        )
        await uow.session.flush()
        payload = InspirationIdeaAdoptedV2(
            schema_version=2,
            adoption_id=adoption_id,
            idea_id=command.idea_id,
            run_id=run_id,
            owner_agent_id=command.owner_agent_id,
            snapshot_hash=material.snapshot_hash,
            evidence=material.evidence,
            resulting_experience_id=experience.experience_id,
            resulting_version_id=experience.current_version_id,
            created=created,
            requested_importance=command.importance,
            requested_confidence=command.confidence,
            mechanism_cluster_id=context.cluster.cluster_id,
            owner_decision_before=before,
            owner_decision_after=IdeaOwnerDecision.ADOPTED,
            distinct_adopter_count_before=(transition.distinct_adopter_count_before),
            distinct_adopter_count_after=(transition.distinct_adopter_count_after),
            maturity_before=transition.maturity_before,
            maturity_after=transition.maturity_after,
            candidate_since_before=transition.candidate_since_before,
            candidate_since_after=transition.candidate_since_after,
            last_signal_at_before=transition.last_signal_at_before,
            last_signal_at_after=transition.last_signal_at_after,
        )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="idea_adoption",
            resource_id=adoption_id,
        )
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=command.idea_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=command.owner_agent_id,
                    occurred_at=adopted_at,
                ),
            ),
        )
        return self._adoption_response(
            experience=experience,
            created=created,
        )

    async def _existing_adoption_response(
        self,
        *,
        uow: UnitOfWork,
        context: _EvaluationContext,
        command: AdoptIdea,
        command_context: CommandContext,
    ) -> StoredResponse:
        record = await uow.session.scalar(
            select(IdeaAdoptionRecordRow).where(
                IdeaAdoptionRecordRow.idea_id == context.idea.idea_id,
                IdeaAdoptionRecordRow.owner_agent_id == context.state.owner_agent_id,
            )
        )
        if (
            record is None
            or context.state.resulting_experience_id != record.resulting_experience_id
            or context.state.resulting_version_id != record.resulting_version_id
        ):
            raise InspirationSourceIntegrityError(
                "adopted idea provenance is missing or inconsistent"
            )
        event_rows = tuple(
            (
                await uow.session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_type == "idea",
                        DomainEventRow.aggregate_id == context.idea.idea_id,
                        DomainEventRow.event_type.in_(
                            (
                                InspirationIdeaAdoptedV1.event_type,
                                InspirationIdeaAdoptedV2.event_type,
                            )
                        ),
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        if len(event_rows) != 1:
            raise InspirationSourceIntegrityError(
                "adopted idea has no unique adoption event"
            )
        event_row = event_rows[0]
        try:
            payload: InspirationIdeaAdoptedV1 | InspirationIdeaAdoptedV2 = (
                InspirationIdeaAdoptedV2.model_validate_json(event_row.payload)
                if event_row.event_type == InspirationIdeaAdoptedV2.event_type
                else InspirationIdeaAdoptedV1.model_validate_json(event_row.payload)
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise InspirationSourceIntegrityError(
                "adopted idea event payload is invalid"
            ) from error
        if (
            payload.adoption_id != record.adoption_id
            or payload.idea_id != record.idea_id
            or payload.run_id != record.run_id
            or payload.owner_agent_id != record.owner_agent_id
            or payload.resulting_experience_id != record.resulting_experience_id
            or payload.resulting_version_id != record.resulting_version_id
            or event_row.actor_agent_id != record.owner_agent_id
            or event_row.occurred_at != record.adopted_at
        ):
            raise InspirationSourceIntegrityError(
                "adopted idea event and provenance disagree"
            )
        retry_receipt = await self._current_command_receipt(
            uow=uow,
            command_context=command_context,
        )
        if retry_receipt.created_at < event_row.occurred_at:
            raise _clock_regression()
        completed = await self._receipt_store.get_by_id(
            session=uow.session,
            receipt_id=event_row.causation_id,
        )
        identity = await uow.session.get(
            ExperienceRow,
            record.resulting_experience_id,
        )
        version = await uow.session.get(
            ExperienceVersionRow,
            record.resulting_version_id,
        )
        if identity is None or version is None:
            raise InspirationSourceIntegrityError(
                "adopted idea result source is missing"
            )
        historical, historical_event = await _experience_state_before_event(
            session=uow.session,
            experience_id=record.resulting_experience_id,
            event_id=event_row.event_id,
        )
        if (
            version.experience_id != identity.experience_id
            or identity.owner_agent_id != record.owner_agent_id
            or identity.created_at > event_row.occurred_at
            or version.created_at > event_row.occurred_at
            or historical.experience_id != identity.experience_id
            or historical.owner_agent_id != identity.owner_agent_id
            or historical.current_version_id != version.version_id
            or historical.current_content_hash != version.content_hash
            or historical.temperature is Temperature.ARCHIVED
            or historical_event.occurred_at > event_row.occurred_at
        ):
            raise InspirationSourceIntegrityError(
                "adopted idea result was not current at adoption"
            )
        expected_importance = (
            payload.requested_importance
            if isinstance(payload, InspirationIdeaAdoptedV2)
            else historical.importance
            if payload.created
            else None
        )
        expected_confidence = (
            payload.requested_confidence
            if isinstance(payload, InspirationIdeaAdoptedV2)
            else historical.confidence
            if payload.created
            else None
        )
        if (expected_importance is None) is not (expected_confidence is None):
            raise InspirationSourceIntegrityError(
                "adopted idea request parameters are incomplete"
            )
        if expected_importance is not None and (
            command.importance != expected_importance
            or command.confidence != expected_confidence
        ):
            raise _adoption_request_mismatch()
        expected_response = self._adoption_response(
            experience=ExperienceRecord(
                experience_id=identity.experience_id,
                owner_agent_id=identity.owner_agent_id,
                current_version_id=version.version_id,
                current_content_hash=version.content_hash,
                temperature=historical.temperature,
            ),
            created=payload.created,
        )
        expected_request_hash: str | None = None
        if completed is not None and expected_importance is not None:
            assert expected_confidence is not None
            expected_request_hash = adoption_command_request(
                AdoptIdea(
                    owner_agent_id=payload.owner_agent_id,
                    idea_id=payload.idea_id,
                    importance=expected_importance,
                    confidence=expected_confidence,
                ),
                idempotency_key=completed.idempotency_key,
            ).request_hash
        request_hash_matches = completed is not None and (
            completed.request_hash == expected_request_hash
            if expected_request_hash is not None
            else _is_lower_sha256(completed.request_hash)
        )
        if (
            completed is None
            or completed.state != "completed"
            or completed.caller_scope != f"agent:{record.owner_agent_id}"
            or completed.operation_scope != "inspiration.idea.adopt"
            or not request_hash_matches
            or completed.result_resource_type != "idea_adoption"
            or completed.result_resource_id != record.adoption_id
            or completed.completed_at is None
            or completed.created_at > event_row.occurred_at
            or completed.completed_at < event_row.occurred_at
            or completed.response != expected_response
        ):
            raise InspirationSourceIntegrityError(
                "adopted idea has no completed canonical response"
            )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="idea_adoption",
            resource_id=record.adoption_id,
        )
        assert completed.response is not None
        return completed.response

    async def _adoption_material(
        self,
        *,
        session: AsyncSession,
        context: _EvaluationContext,
    ) -> _AdoptionMaterial:
        occurrences = tuple(
            (
                await session.scalars(
                    select(IdeaOccurrenceRow).where(
                        IdeaOccurrenceRow.idea_id == context.idea.idea_id
                    )
                )
            ).all()
        )
        if len(occurrences) != 1:
            raise InspirationSourceIntegrityError(
                "idea occurrence provenance is missing or ambiguous"
            )
        occurrence = occurrences[0]
        if (
            occurrence.run_id != context.idea.run_id
            or occurrence.owner_agent_id != context.run.owner_agent_id
            or occurrence.mechanism_hash != context.idea.mechanism_hash
        ):
            raise InspirationSourceIntegrityError(
                "idea occurrence provenance is inconsistent"
            )
        evidence = _idea_evidence(context.idea.evidence_references)
        link_targets: set[UUID] = set()
        for reference in evidence:
            item = await session.get(
                InspirationSnapshotItemRow,
                reference.id,
            )
            if (
                item is None
                or item.run_id != context.idea.run_id
                or item.stable_evidence_key != reference.stable_evidence_key
            ):
                raise InspirationSourceIntegrityError(
                    "idea evidence does not resolve in its frozen snapshot"
                )
            try:
                source_type = EvidenceSourceType(item.source_type)
                source_state = EvidenceSourceState(item.source_state)
            except ValueError as error:
                raise InspirationSourceIntegrityError(
                    "idea snapshot evidence source is invalid"
                ) from error
            if source_type is EvidenceSourceType.CAPSULE:
                if source_state is not EvidenceSourceState.QUARANTINED:
                    raise InspirationSourceIntegrityError(
                        "capsule evidence is not quarantined"
                    )
                continue
            if source_state not in {
                EvidenceSourceState.HOT,
                EvidenceSourceState.WARM,
                EvidenceSourceState.COLD,
            }:
                raise InspirationSourceIntegrityError(
                    "experience evidence has an invalid state"
                )
            version = await session.get(
                ExperienceVersionRow,
                item.source_version_id,
            )
            identity = await session.get(ExperienceRow, item.source_id)
            if (
                version is None
                or identity is None
                or version.experience_id != identity.experience_id
                or identity.owner_agent_id != context.run.owner_agent_id
            ):
                raise InspirationSourceIntegrityError(
                    "experience evidence is not an owned immutable version"
                )
            link_targets.add(identity.experience_id)
        links = tuple(
            VersionLinkInput(
                target_experience_id=experience_id,
                relation=LinkRelation.DERIVED_FROM,
            )
            for experience_id in sorted(
                link_targets,
                key=lambda value: value.bytes,
            )
        )
        return _AdoptionMaterial(
            snapshot_hash=occurrence.snapshot_hash,
            evidence=evidence,
            content=_adopted_content(
                idea=context.idea,
                evidence=evidence,
            ),
            links=links,
        )

    @staticmethod
    async def _cluster_adopter_owners(
        *,
        session: AsyncSession,
        cluster_id: str,
    ) -> frozenset[UUID]:
        rows = tuple(
            (
                await session.execute(
                    select(IdeaAdoptionRecordRow, IdeaStateRow)
                    .join(
                        IdeaStateRow,
                        IdeaStateRow.idea_id == IdeaAdoptionRecordRow.idea_id,
                    )
                    .where(IdeaStateRow.mechanism_cluster_id == cluster_id)
                )
            ).all()
        )
        owners: set[UUID] = set()
        for record, state in rows:
            if (
                state.owner_decision != IdeaOwnerDecision.ADOPTED.value
                or state.owner_agent_id != record.owner_agent_id
                or state.resulting_experience_id != record.resulting_experience_id
                or state.resulting_version_id != record.resulting_version_id
            ):
                raise InspirationSourceIntegrityError(
                    "cluster adoption provenance is inconsistent"
                )
            owners.add(record.owner_agent_id)
        return frozenset(owners)

    @staticmethod
    def _adoption_response(
        *,
        experience: ExperienceRecord,
        created: bool,
    ) -> StoredResponse:
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes(
                {
                    "data": {
                        "created": created,
                        "experience": {
                            "current_content_hash": (experience.current_content_hash),
                            "current_version_id": (experience.current_version_id),
                            "experience_id": experience.experience_id,
                            "owner_agent_id": experience.owner_agent_id,
                            "temperature": experience.temperature,
                        },
                    }
                }
            ),
        )

    async def _decide(
        self,
        *,
        uow: UnitOfWork,
        command: RejectIdea | ArchiveIdea,
        command_context: CommandContext,
        decision: IdeaOwnerDecision,
    ) -> StoredResponse:
        if not isinstance(uow, UnitOfWork) or not uow.immediate:
            raise RuntimeError(
                "Idea owner decisions require a caller-owned immediate UOW"
            )
        if not isinstance(command_context, CommandContext):
            raise ValueError("command_context must be a CommandContext")
        action = "reject" if decision is IdeaOwnerDecision.REJECTED else "archive"
        reason = _strict_reason(command.reason)
        if (
            command_context.caller_scope != f"agent:{command.owner_agent_id}"
            or command_context.operation_scope != f"inspiration.idea.{action}"
        ):
            raise _not_found()
        try:
            expected_request = decision_command_request(
                (
                    RejectIdea(
                        owner_agent_id=command.owner_agent_id,
                        idea_id=command.idea_id,
                        reason=reason,
                    )
                    if decision is IdeaOwnerDecision.REJECTED
                    else ArchiveIdea(
                        owner_agent_id=command.owner_agent_id,
                        idea_id=command.idea_id,
                        reason=reason,
                    )
                ),
                idempotency_key=command_context.idempotency_key,
            )
        except (TypeError, ValueError) as error:
            raise _not_found() from error
        if command_context.request_hash != expected_request.request_hash:
            raise _not_found()

        context = await self._load_context(
            session=uow.session,
            idea_id=command.idea_id,
            evaluator_agent_id=command.owner_agent_id,
        )
        try:
            before = IdeaOwnerDecision(context.state.owner_decision)
        except ValueError as error:
            raise InspirationSourceIntegrityError(
                "idea owner-decision projection is invalid"
            ) from error
        allowed = (
            {IdeaOwnerDecision.ACTIVE, IdeaOwnerDecision.ARCHIVED}
            if decision is IdeaOwnerDecision.REJECTED
            else {IdeaOwnerDecision.ACTIVE}
        )
        if before not in allowed:
            raise _idea_not_decidable()
        decided_at = await self._server_command_time(
            uow=uow,
            command_context=command_context,
        )
        latest_causal_at = await self._latest_idea_causal_at(
            session=uow.session,
            state=context.state,
        )
        if decided_at < latest_causal_at:
            raise _clock_regression()

        if decision is IdeaOwnerDecision.REJECTED:
            payload: InspirationIdeaRejectedV1 | InspirationIdeaArchivedV1 = (
                InspirationIdeaRejectedV1(
                    schema_version=1,
                    idea_id=command.idea_id,
                    owner_agent_id=command.owner_agent_id,
                    reason=reason,
                    owner_decision_before=before,
                    owner_decision_after=decision,
                )
            )
        else:
            payload = InspirationIdeaArchivedV1(
                schema_version=1,
                idea_id=command.idea_id,
                owner_agent_id=command.owner_agent_id,
                reason=reason,
                owner_decision_before=before,
                owner_decision_after=decision,
                cycle_id=None,
            )
        await self._receipt_store.attach_resource(
            uow=uow,
            receipt_id=command_context.receipt_id,
            resource_type="idea",
            resource_id=command.idea_id,
        )
        await uow.append_events(
            command_context,
            (
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=command.idea_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=command.owner_agent_id,
                    occurred_at=decided_at,
                ),
            ),
        )
        return StoredResponse(
            status_code=200,
            body=canonical_json_bytes(
                {
                    "data": {
                        "idea_id": command.idea_id,
                        "owner_decision": decision,
                    }
                }
            ),
        )

    async def _load_context(
        self,
        *,
        session: AsyncSession,
        idea_id: UUID,
        evaluator_agent_id: UUID,
    ) -> _EvaluationContext:
        idea = await session.get(InspirationIdeaRow, idea_id)
        if idea is None:
            raise _not_found()
        run = await session.get(InspirationRunRow, idea.run_id)
        if run is None or run.owner_agent_id != evaluator_agent_id:
            raise _not_found()
        state = await session.get(IdeaStateRow, idea_id)
        if state is None:
            raise InspirationSourceIntegrityError(
                "idea source or projection is missing"
            )
        if state.owner_agent_id != run.owner_agent_id:
            raise InspirationSourceIntegrityError("idea source and projection disagree")
        cluster_row = await session.get(
            MechanismIncubationRow,
            state.mechanism_cluster_id,
        )
        if cluster_row is None:
            raise InspirationSourceIntegrityError(
                "idea mechanism incubation projection is missing"
            )
        cluster = _cluster_value(cluster_row)
        if idea.mechanism_hash not in cluster.member_hashes:
            raise InspirationSourceIntegrityError(
                "idea mechanism is absent from its projected cluster"
            )
        return _EvaluationContext(
            idea=idea,
            state=state,
            cluster=cluster,
            run=run,
            cluster_row=cluster_row,
        )

    @staticmethod
    async def _validate_evidence(
        *,
        session: AsyncSession,
        evaluation: IdeaEvaluation,
        idea: InspirationIdeaRow,
    ) -> None:
        for reference in evaluation.evidence:
            if isinstance(reference, SnapshotEvidenceReference):
                item = await session.get(
                    InspirationSnapshotItemRow,
                    reference.id,
                )
                if (
                    item is None
                    or item.run_id != idea.run_id
                    or item.stable_evidence_key != reference.stable_evidence_key
                ):
                    raise _invalid_evidence()
                continue
            if isinstance(
                reference,
                ExperienceVersionEvidenceReference,
            ):
                version = await session.get(
                    ExperienceVersionRow,
                    reference.id,
                )
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
                    or identity.owner_agent_id != evaluation.evaluator_agent_id
                    or identity.created_at > evaluation.evaluated_at
                    or version.created_at > evaluation.evaluated_at
                ):
                    raise _invalid_evidence()
                continue
            raise _invalid_evidence()

    @staticmethod
    async def _cluster_projection_event_id(
        *,
        session: AsyncSession,
        cluster_id: str,
    ) -> int:
        value = await session.scalar(
            select(MechanismIncubationRow.projection_event_id).where(
                MechanismIncubationRow.cluster_id == cluster_id
            )
        )
        if value is None:
            raise InspirationSourceIntegrityError(
                "mechanism incubation projection is missing"
            )
        return value

    @staticmethod
    async def _latest_causal_at(
        *,
        session: AsyncSession,
        state: IdeaStateRow,
        cluster_last_signal_at: datetime,
        cluster_projection_event_id: int,
    ) -> datetime:
        idea_causal_at = await IdeaLifecycleService._latest_idea_causal_at(
            session=session,
            state=state,
        )
        cluster_event_at = await session.scalar(
            select(DomainEventRow.occurred_at).where(
                DomainEventRow.event_id == cluster_projection_event_id
            )
        )
        if cluster_event_at is None:
            raise InspirationSourceIntegrityError(
                "cluster projection checkpoint is not backed by a source event"
            )
        return max(
            idea_causal_at,
            require_utc(cluster_last_signal_at),
            require_utc(cluster_event_at),
        )

    @staticmethod
    async def _latest_idea_causal_at(
        *,
        session: AsyncSession,
        state: IdeaStateRow,
    ) -> datetime:
        checkpoint = await session.get(
            DomainEventRow,
            state.projection_event_id,
        )
        head = await session.scalar(
            select(DomainEventRow)
            .where(
                DomainEventRow.aggregate_type == "idea",
                DomainEventRow.aggregate_id == state.idea_id,
            )
            .order_by(
                DomainEventRow.sequence.desc(),
                DomainEventRow.event_id.desc(),
            )
            .limit(1)
        )
        if (
            checkpoint is None
            or head is None
            or checkpoint.aggregate_type != "idea"
            or checkpoint.aggregate_id != state.idea_id
            or checkpoint.event_id != head.event_id
            or checkpoint.sequence != head.sequence
        ):
            raise InspirationSourceIntegrityError(
                "idea projection checkpoint is not its aggregate source head"
            )
        return max(
            require_utc(state.last_signal_at),
            require_utc(checkpoint.occurred_at),
            require_utc(head.occurred_at),
        )


class InspirationIdeaArchivePlanner:
    """Plan policy archives after all experience lifecycle mutations."""

    async def due_archive_events(
        self,
        *,
        session: AsyncSession,
        evaluated_at: datetime,
        cycle_id: UUID,
    ) -> tuple[PendingEvent, ...]:
        if not isinstance(evaluated_at, datetime):
            raise ValueError("evaluated_at must be a timezone-aware datetime")
        retained_at = require_utc(evaluated_at)
        if not isinstance(cycle_id, UUID):
            raise ValueError("cycle_id must be a UUID")
        rows = tuple(
            (
                await session.execute(
                    select(IdeaStateRow, MechanismIncubationRow)
                    .join(
                        MechanismIncubationRow,
                        MechanismIncubationRow.cluster_id
                        == IdeaStateRow.mechanism_cluster_id,
                    )
                    .where(
                        IdeaStateRow.owner_decision == IdeaOwnerDecision.ACTIVE.value
                    )
                )
            ).all()
        )
        active_count = int(
            await session.scalar(
                select(func.count())
                .select_from(IdeaStateRow)
                .where(IdeaStateRow.owner_decision == IdeaOwnerDecision.ACTIVE.value)
            )
            or 0
        )
        if len(rows) != active_count:
            raise InspirationSourceIntegrityError(
                "an active idea is missing its mechanism projection"
            )
        events: list[PendingEvent] = []
        for state, cluster_row in sorted(
            rows,
            key=lambda pair: pair[0].idea_id.bytes,
        ):
            cluster = _cluster_value(cluster_row)
            signal_at = require_utc(state.last_signal_at)
            if cluster.maturity is MechanismMaturity.CANDIDATE:
                if cluster.candidate_since is None:
                    raise InspirationSourceIntegrityError(
                        "candidate cluster has no candidate_since"
                    )
                anchor = max(signal_at, cluster.candidate_since)
                due_at = anchor + _CANDIDATE_RETENTION
            else:
                due_at = signal_at + _NONCANDIDATE_RETENTION
            if retained_at < due_at:
                continue
            latest_causal_at = await IdeaLifecycleService._latest_idea_causal_at(
                session=session,
                state=state,
            )
            if retained_at < latest_causal_at:
                raise _clock_regression()
            payload = InspirationIdeaArchivedV1(
                schema_version=1,
                idea_id=state.idea_id,
                owner_agent_id=state.owner_agent_id,
                reason=StructuredReason.policy_due(),
                owner_decision_before=IdeaOwnerDecision.ACTIVE,
                owner_decision_after=IdeaOwnerDecision.ARCHIVED,
                cycle_id=cycle_id,
            )
            events.append(
                PendingEvent(
                    aggregate_type="idea",
                    aggregate_id=state.idea_id,
                    event_type=payload.event_type,
                    payload=payload,
                    actor_agent_id=None,
                    occurred_at=retained_at,
                )
            )
        return tuple(events)


__all__ = [
    "IdeaLifecycleService",
    "InspirationIdeaArchivePlanner",
]
