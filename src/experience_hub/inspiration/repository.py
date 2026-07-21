"""Transaction-bound persistence for inspiration source and projection rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import require_utc
from experience_hub.domain import EventRegistry
from experience_hub.inspiration.adoption import (
    canonical_string_tuple,
    decode_idea_evidence,
)
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.dedup import RetainedIdea
from experience_hub.inspiration.events import (
    InspirationIdeaEvaluatedV1,
    InspirationIdeaGeneratedV1,
)
from experience_hub.inspiration.hashing import (
    hash_idea_content,
    hash_mechanism,
)
from experience_hub.inspiration.incubation import (
    IncubationCluster,
    IncubationMember,
    OccurrencePlan,
)
from experience_hub.inspiration.models import (
    FrozenSnapshot,
    GeneratorKind,
    Idea,
    IdeaDraft,
    IdeaOwnerDecision,
    InspirationOperator,
    InspirationRun,
    InspirationRunStatus,
    MechanismIncubation,
    MechanismMaturity,
    OperatorOutcome,
)
from experience_hub.storage.tables import (
    DomainEventRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationRunStateRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)


class InspirationSourceIntegrityError(RuntimeError):
    """Inspiration authoritative rows cannot be reconciled safely."""

    code = "inspiration_source_integrity_error"


@dataclass(frozen=True, slots=True)
class RunningInspirationTrace:
    """One nonterminal run trace and its original retained receipt."""

    run_id: UUID
    owner_agent_id: UUID
    receipt_id: UUID
    request_hash: str
    last_event_at: datetime


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        decoded = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspirationSourceIntegrityError(
            f"{label} is not canonical JSON"
        ) from error
    if canonical_json_bytes(decoded) != raw:
        raise InspirationSourceIntegrityError(f"{label} is not canonical JSON")
    return decoded


def _outcomes(raw: bytes) -> tuple[OperatorOutcome, ...]:
    decoded = _decode_json(raw, label="operator outcomes")
    if not isinstance(decoded, list):
        raise InspirationSourceIntegrityError("operator outcomes must be an array")
    try:
        retained: list[OperatorOutcome] = []
        for value in decoded:
            parsed = OperatorOutcome.model_validate_json(canonical_json_bytes(value))
            retained.append(
                OperatorOutcome.model_validate(
                    parsed.model_dump(mode="python", warnings=False),
                    strict=True,
                )
            )
        return tuple(retained)
    except (TypeError, ValueError, ValidationError) as error:
        raise InspirationSourceIntegrityError(
            "operator outcomes are invalid"
        ) from error


def _member_hashes(raw: bytes) -> tuple[str, ...]:
    decoded = _decode_json(raw, label="mechanism member hashes")
    if not isinstance(decoded, list) or any(
        not isinstance(value, str) for value in decoded
    ):
        raise InspirationSourceIntegrityError("mechanism member hashes are invalid")
    return tuple(decoded)


def _idea_value(
    *,
    source: InspirationIdeaRow,
    state: IdeaStateRow,
    cluster: MechanismIncubationRow,
    owner_agent_id: UUID,
    visible_duplicate_relations: frozenset[UUID],
) -> Idea:
    try:
        draft = IdeaDraft(
            title=source.title,
            hypothesis=source.hypothesis,
            mechanism=source.mechanism,
            predictions=canonical_string_tuple(
                source.predictions,
                label="idea predictions",
            ),
            falsifiers=canonical_string_tuple(
                source.falsifiers,
                label="idea falsifiers",
            ),
            assumptions=canonical_string_tuple(
                source.assumptions,
                label="idea assumptions",
            ),
            proposed_test=source.proposed_test,
            evidence=decode_idea_evidence(source.evidence_references),
        )
        decision = IdeaOwnerDecision(state.owner_decision)
        maturity = MechanismMaturity(cluster.maturity)
        members = _member_hashes(cluster.member_hashes)
        if (
            state.idea_id != source.idea_id
            or state.owner_agent_id != owner_agent_id
            or state.mechanism_cluster_id != cluster.cluster_id
            or source.mechanism_hash not in members
            or hash_idea_content(draft) != source.idea_content_hash
            or hash_mechanism(draft.mechanism) != source.mechanism_hash
            or (
                decision is IdeaOwnerDecision.ADOPTED
                and (
                    state.resulting_experience_id is None
                    or state.resulting_version_id is None
                )
            )
            or (
                decision is not IdeaOwnerDecision.ADOPTED
                and (
                    state.resulting_experience_id is not None
                    or state.resulting_version_id is not None
                )
            )
        ):
            raise ValueError("idea source and projections disagree")
        return Idea(
            idea_id=source.idea_id,
            run_id=source.run_id,
            owner_agent_id=owner_agent_id,
            operator=InspirationOperator(source.operator),
            ordinal=source.ordinal,
            draft=draft,
            idea_content_hash=source.idea_content_hash,
            mechanism_hash=source.mechanism_hash,
            duplicate_relation=(
                source.duplicate_relation
                if source.duplicate_relation in visible_duplicate_relations
                else None
            ),
            owner_decision=decision,
            mechanism_cluster_id=state.mechanism_cluster_id,
            maturity=maturity,
            last_signal_at=state.last_signal_at,
            resulting_experience_id=state.resulting_experience_id,
            resulting_version_id=state.resulting_version_id,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise InspirationSourceIntegrityError(
            "idea source or projection is invalid"
        ) from error


class InspirationRepository:
    """Persist immutable run facts and reload locked clustering state."""

    def __init__(self, event_registry: EventRegistry | None = None) -> None:
        self._event_registry = event_registry

    @staticmethod
    async def agent_exists(
        *,
        session: AsyncSession,
        agent_id: UUID,
    ) -> bool:
        from experience_hub.storage.tables import AgentRow

        return (
            await session.scalar(
                select(AgentRow.agent_id).where(AgentRow.agent_id == agent_id)
            )
            is not None
        )

    @staticmethod
    def add_run(
        *,
        session: AsyncSession,
        run_id: UUID,
        request: StartInspirationRun,
        generator_configuration: dict[str, str],
        request_hash: str,
        occurred_at: datetime,
    ) -> None:
        if request.generator is GeneratorKind.DETERMINISTIC:
            if generator_configuration != {}:
                raise ValueError("deterministic generator configuration must be empty")
        elif set(generator_configuration) != {"base_url", "model"} or any(
            not isinstance(value, str) or not value
            for value in generator_configuration.values()
        ):
            raise ValueError(
                "OpenAI-compatible persisted configuration must contain "
                "only base_url and model"
            )
        session.add(
            InspirationRunRow(
                run_id=run_id,
                owner_agent_id=request.owner_agent_id,
                goal=request.goal,
                context=request.context or None,
                mode=request.mode.value,
                generator_kind=request.generator.value,
                generator_configuration=canonical_json_bytes(generator_configuration),
                operators=canonical_json_bytes(
                    tuple(operator.value for operator in request.operators)
                ),
                include_inbox=request.include_inbox,
                branches_per_operator=request.branches_per_operator,
                output_tokens_per_operator=request.output_tokens_per_operator,
                total_output_tokens=request.total_output_tokens,
                operator_timeout_seconds=request.operator_timeout_seconds,
                global_timeout_seconds=request.global_timeout_seconds,
                request_hash=request_hash,
                created_at=require_utc(occurred_at),
            )
        )

    @staticmethod
    def add_snapshot(
        *,
        session: AsyncSession,
        snapshot: FrozenSnapshot,
    ) -> None:
        for item in snapshot.items:
            session.add(
                InspirationSnapshotItemRow(
                    snapshot_item_id=item.snapshot_item_id,
                    run_id=item.run_id,
                    stable_evidence_key=item.stable_evidence_key,
                    source_type=item.source_type.value,
                    source_id=item.source_id,
                    source_version_id=item.source_version_id,
                    source_state=item.source_state.value,
                    rank=item.rank,
                    summary=item.summary,
                    mechanism=item.mechanism,
                    applicability=canonical_json_bytes(item.applicability),
                    tags=canonical_json_bytes(item.tags),
                    falsifiers=canonical_json_bytes(item.falsifiers),
                    excerpt=item.excerpt,
                    source_trust=item.source_trust,
                    content_hash=item.content_hash,
                )
            )

    @staticmethod
    async def add_idea_occurrence(
        *,
        session: AsyncSession,
        idea_id: UUID,
        occurrence_id: UUID,
        run_id: UUID,
        owner_agent_id: UUID,
        snapshot_hash: str,
        retained: RetainedIdea,
        plan: OccurrencePlan,
        occurred_at: datetime,
    ) -> None:
        idea = InspirationIdeaRow(
            idea_id=idea_id,
            run_id=run_id,
            operator=retained.operator.value,
            ordinal=retained.ordinal,
            title=retained.draft.title,
            hypothesis=retained.draft.hypothesis,
            mechanism=retained.draft.mechanism,
            predictions=canonical_json_bytes(retained.draft.predictions),
            falsifiers=canonical_json_bytes(retained.draft.falsifiers),
            assumptions=canonical_json_bytes(retained.draft.assumptions),
            proposed_test=retained.draft.proposed_test,
            evidence_references=canonical_json_bytes(retained.draft.evidence),
            idea_content_hash=retained.idea_content_hash,
            mechanism_hash=retained.mechanism_hash,
            duplicate_relation=plan.duplicate_relation,
        )
        session.add(idea)
        await session.flush((idea,))
        session.add(
            IdeaOccurrenceRow(
                occurrence_id=occurrence_id,
                idea_id=idea_id,
                mechanism_hash=retained.mechanism_hash,
                run_id=run_id,
                snapshot_hash=snapshot_hash,
                owner_agent_id=owner_agent_id,
                occurred_at=require_utc(occurred_at),
            )
        )

    @staticmethod
    async def load_clusters(
        *,
        session: AsyncSession,
    ) -> tuple[IncubationCluster, ...]:
        cluster_rows = tuple(
            (
                await session.scalars(
                    select(MechanismIncubationRow).order_by(
                        MechanismIncubationRow.cluster_id
                    )
                )
            ).all()
        )
        retained: list[IncubationCluster] = []
        for row in cluster_rows:
            idea_state_rows = tuple(
                (
                    await session.scalars(
                        select(IdeaStateRow).where(
                            IdeaStateRow.mechanism_cluster_id == row.cluster_id
                        )
                    )
                ).all()
            )
            member_sources: list[tuple[int, IncubationMember, str]] = []
            for state_row in idea_state_rows:
                generated_rows = tuple(
                    (
                        await session.scalars(
                            select(DomainEventRow)
                            .where(
                                DomainEventRow.aggregate_type == "idea",
                                DomainEventRow.aggregate_id == state_row.idea_id,
                                DomainEventRow.event_type
                                == InspirationIdeaGeneratedV1.event_type,
                            )
                            .order_by(DomainEventRow.event_id)
                        )
                    ).all()
                )
                if len(generated_rows) != 1:
                    raise InspirationSourceIntegrityError(
                        "cluster member generated source is missing or ambiguous"
                    )
                generated_row = generated_rows[0]
                try:
                    _decode_json(
                        generated_row.payload,
                        label="cluster member generated source",
                    )
                    generated = InspirationIdeaGeneratedV1.model_validate_json(
                        generated_row.payload
                    )
                except (TypeError, ValueError, ValidationError) as error:
                    raise InspirationSourceIntegrityError(
                        "cluster member generated source is invalid"
                    ) from error
                idea = await session.get(
                    InspirationIdeaRow,
                    state_row.idea_id,
                )
                occurrence = await session.scalar(
                    select(IdeaOccurrenceRow).where(
                        IdeaOccurrenceRow.idea_id == state_row.idea_id
                    )
                )
                if idea is None or occurrence is None:
                    raise InspirationSourceIntegrityError(
                        "cluster member source is missing"
                    )
                run = await session.get(InspirationRunRow, idea.run_id)
                if run is None:
                    raise InspirationSourceIntegrityError(
                        "cluster member run is missing"
                    )
                if (
                    generated_row.sequence != 1
                    or generated_row.actor_agent_id != generated.owner_agent_id
                    or generated_row.occurred_at != generated.last_signal_at_after
                    or generated.idea_id != state_row.idea_id
                    or generated.owner_agent_id != state_row.owner_agent_id
                    or generated.cluster_id != state_row.mechanism_cluster_id
                    or generated.cluster_id != row.cluster_id
                    or generated.run_id != idea.run_id
                    or generated.run_id != occurrence.run_id
                    or generated.owner_agent_id != run.owner_agent_id
                    or generated.owner_agent_id != occurrence.owner_agent_id
                    or generated.occurrence_id != occurrence.occurrence_id
                    or generated.mechanism_hash != idea.mechanism_hash
                    or generated.mechanism_hash != occurrence.mechanism_hash
                    or generated.snapshot_hash != occurrence.snapshot_hash
                    or generated.idea_content_hash != idea.idea_content_hash
                    or generated.operator.value != idea.operator
                    or generated.ordinal != idea.ordinal
                    or generated.duplicate_relation != idea.duplicate_relation
                    or generated_row.occurred_at != occurrence.occurred_at
                ):
                    raise InspirationSourceIntegrityError(
                        "cluster member generated source does not match "
                        "its authoritative rows"
                    )
                member_sources.append(
                    (
                        generated_row.event_id,
                        IncubationMember(
                            idea_id=idea.idea_id,
                            owner_agent_id=run.owner_agent_id,
                            mechanism=idea.mechanism,
                            mechanism_hash=idea.mechanism_hash,
                        ),
                        occurrence.snapshot_hash,
                    )
                )
            member_sources.sort(key=lambda source: source[0])
            members = tuple(source[1] for source in member_sources)
            snapshot_hashes = frozenset(source[2] for source in member_sources)
            try:
                state = MechanismIncubation(
                    cluster_id=row.cluster_id,
                    canonical_mechanism_hash=row.canonical_mechanism_hash,
                    member_hashes=_member_hashes(row.member_hashes),
                    occurrence_count=row.occurrence_count,
                    distinct_snapshot_count=row.distinct_snapshot_count,
                    distinct_adopter_count=row.distinct_adopter_count,
                    supported_count=row.supported_count,
                    refuted_count=row.refuted_count,
                    maturity=MechanismMaturity(row.maturity),
                    candidate_since=row.candidate_since,
                    last_signal_at=row.last_signal_at,
                )
                retained.append(
                    IncubationCluster(
                        state=state,
                        members=members,
                        snapshot_hashes=snapshot_hashes,
                    )
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise InspirationSourceIntegrityError(
                    "cluster projection is invalid"
                ) from error
        return tuple(retained)

    @staticmethod
    async def owns_run(
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        run_id: UUID,
    ) -> bool:
        """Check authoritative run ownership without touching projections."""
        if not isinstance(owner_agent_id, UUID) or not isinstance(run_id, UUID):
            raise TypeError("run ownership identifiers must be UUID values")
        return (
            await session.scalar(
                select(InspirationRunRow.run_id).where(
                    InspirationRunRow.run_id == run_id,
                    InspirationRunRow.owner_agent_id == owner_agent_id,
                )
            )
            is not None
        )

    @staticmethod
    async def get_run(
        *,
        session: AsyncSession,
        run_id: UUID,
    ) -> InspirationRun | None:
        source = await session.get(InspirationRunRow, run_id)
        if source is None:
            return None
        state = await session.get(InspirationRunStateRow, run_id)
        if state is None:
            raise InspirationSourceIntegrityError("run projection is missing")
        operators_raw = _decode_json(source.operators, label="run operators")
        if not isinstance(operators_raw, list):
            raise InspirationSourceIntegrityError("run operators are invalid")
        from experience_hub.retrieval.ranking import RetrievalMode

        try:
            return InspirationRun(
                run_id=source.run_id,
                owner_agent_id=source.owner_agent_id,
                goal=source.goal,
                context=source.context or "",
                mode=RetrievalMode(source.mode),
                generator=GeneratorKind(source.generator_kind),
                operators=tuple(InspirationOperator(value) for value in operators_raw),
                include_inbox=source.include_inbox,
                branches_per_operator=source.branches_per_operator,
                output_tokens_per_operator=source.output_tokens_per_operator,
                total_output_tokens=source.total_output_tokens,
                operator_timeout_seconds=source.operator_timeout_seconds,
                global_timeout_seconds=source.global_timeout_seconds,
                request_hash=source.request_hash,
                snapshot_hash=state.snapshot_hash,
                status=InspirationRunStatus(state.status),
                operator_outcomes=_outcomes(state.operator_outcomes),
                output_tokens_reserved=state.output_tokens_reserved,
                output_tokens_consumed=state.output_tokens_consumed,
                elapsed_milliseconds=state.elapsed_milliseconds,
                created_at=source.created_at,
                completed_at=state.completed_at,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise InspirationSourceIntegrityError(
                "run source or projection is invalid"
            ) from error

    @staticmethod
    async def list_owned_ideas(
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        run_id: UUID,
        after: tuple[InspirationOperator, int, UUID] | None,
        limit: int,
    ) -> tuple[Idea, ...]:
        """List one owner's run ideas in stable operator/ordinal order."""
        if (
            not isinstance(owner_agent_id, UUID)
            or not isinstance(run_id, UUID)
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 101
        ):
            raise ValueError("owned idea query arguments are invalid")
        statement = (
            select(
                InspirationIdeaRow,
                IdeaStateRow,
                MechanismIncubationRow,
            )
            .join(
                InspirationRunRow,
                InspirationRunRow.run_id == InspirationIdeaRow.run_id,
            )
            .outerjoin(
                IdeaStateRow,
                IdeaStateRow.idea_id == InspirationIdeaRow.idea_id,
            )
            .outerjoin(
                MechanismIncubationRow,
                MechanismIncubationRow.cluster_id == IdeaStateRow.mechanism_cluster_id,
            )
            .where(
                InspirationIdeaRow.run_id == run_id,
                InspirationRunRow.owner_agent_id == owner_agent_id,
            )
        )
        if after is not None:
            operator, ordinal, idea_id = after
            if (
                not isinstance(operator, InspirationOperator)
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or not 1 <= ordinal <= 3
                or not isinstance(idea_id, UUID)
            ):
                raise ValueError("owned idea cursor sort is invalid")
            statement = statement.where(
                or_(
                    InspirationIdeaRow.operator > operator.value,
                    and_(
                        InspirationIdeaRow.operator == operator.value,
                        InspirationIdeaRow.ordinal > ordinal,
                    ),
                    and_(
                        InspirationIdeaRow.operator == operator.value,
                        InspirationIdeaRow.ordinal == ordinal,
                        InspirationIdeaRow.idea_id > idea_id,
                    ),
                )
            )
        rows = tuple(
            (
                await session.execute(
                    statement.order_by(
                        InspirationIdeaRow.operator,
                        InspirationIdeaRow.ordinal,
                        InspirationIdeaRow.idea_id,
                    ).limit(limit)
                )
            ).all()
        )
        if any(state is None or cluster is None for _, state, cluster in rows):
            raise InspirationSourceIntegrityError(
                "idea source or projection is invalid"
            )
        duplicate_relations = frozenset(
            source.duplicate_relation
            for source, _, _ in rows
            if source.duplicate_relation is not None
        )
        visible_duplicate_relations: frozenset[UUID] = frozenset()
        if duplicate_relations:
            visible_duplicate_relations = frozenset(
                (
                    await session.scalars(
                        select(InspirationIdeaRow.idea_id)
                        .join(
                            InspirationRunRow,
                            InspirationRunRow.run_id == InspirationIdeaRow.run_id,
                        )
                        .where(
                            InspirationIdeaRow.idea_id.in_(duplicate_relations),
                            InspirationRunRow.owner_agent_id == owner_agent_id,
                        )
                    )
                ).all()
            )
        return tuple(
            _idea_value(
                source=source,
                state=state,
                cluster=cluster,
                owner_agent_id=owner_agent_id,
                visible_duplicate_relations=visible_duplicate_relations,
            )
            for source, state, cluster in rows
        )

    async def latest_evaluation(
        self,
        *,
        session: AsyncSession,
        idea_id: UUID,
        evaluator_agent_id: UUID,
    ) -> InspirationIdeaEvaluatedV1 | None:
        """Return the latest source-ledger revision after validating its chain."""
        if self._event_registry is None:
            raise RuntimeError("idea evaluation lookup requires an event registry")
        rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(
                        DomainEventRow.aggregate_type == "idea",
                        DomainEventRow.aggregate_id == idea_id,
                        DomainEventRow.event_type
                        == InspirationIdeaEvaluatedV1.event_type,
                        DomainEventRow.actor_agent_id == evaluator_agent_id,
                    )
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        previous: InspirationIdeaEvaluatedV1 | None = None
        for expected_revision, row in enumerate(rows, start=1):
            try:
                payload = self._event_registry.decode(
                    event_type=row.event_type,
                    payload=row.payload,
                )
            except (TypeError, ValueError) as error:
                raise InspirationSourceIntegrityError(
                    "idea evaluation payload is invalid"
                ) from error
            if (
                not isinstance(payload, InspirationIdeaEvaluatedV1)
                or payload.idea_id != idea_id
                or payload.evaluator_agent_id != evaluator_agent_id
                or payload.revision != expected_revision
                or payload.previous_verdict
                != (None if previous is None else previous.current_verdict)
                or row.occurred_at != payload.last_signal_at_after
                or (
                    previous is not None
                    and payload.last_signal_at_before < previous.last_signal_at_after
                )
            ):
                raise InspirationSourceIntegrityError(
                    "idea evaluation revision chain is invalid"
                )
            previous = payload
        return previous

    async def running_traces(
        self,
        *,
        session: AsyncSession,
    ) -> tuple[RunningInspirationTrace, ...]:
        from experience_hub.inspiration.events import (
            InspirationCompletedV1,
            InspirationFailedV1,
            InspirationStartedV1,
            InspirationTimedOutV1,
        )

        if self._event_registry is None:
            raise RuntimeError("running trace recovery requires an event registry")
        started_rows = tuple(
            (
                await session.scalars(
                    select(DomainEventRow)
                    .where(DomainEventRow.event_type == InspirationStartedV1.event_type)
                    .order_by(DomainEventRow.event_id)
                )
            ).all()
        )
        terminal_types = {
            InspirationCompletedV1.event_type,
            InspirationFailedV1.event_type,
            InspirationTimedOutV1.event_type,
        }
        traces: list[RunningInspirationTrace] = []
        for started_row in started_rows:
            try:
                payload = self._event_registry.decode(
                    event_type=started_row.event_type,
                    payload=started_row.payload,
                )
            except (TypeError, ValueError) as error:
                raise InspirationSourceIntegrityError(
                    "started trace payload is invalid"
                ) from error
            if not isinstance(payload, InspirationStartedV1):
                raise InspirationSourceIntegrityError(
                    "started trace payload type is invalid"
                )
            terminal = await session.scalar(
                select(DomainEventRow.event_id)
                .where(
                    DomainEventRow.aggregate_type == "inspiration_run",
                    DomainEventRow.aggregate_id == payload.run_id,
                    DomainEventRow.event_type.in_(terminal_types),
                )
                .limit(1)
            )
            if terminal is not None:
                continue
            event_rows = tuple(
                (
                    await session.scalars(
                        select(DomainEventRow)
                        .where(
                            DomainEventRow.aggregate_type == "inspiration_run",
                            DomainEventRow.aggregate_id == payload.run_id,
                        )
                        .order_by(DomainEventRow.event_id)
                    )
                ).all()
            )
            causal_rows = tuple(
                (
                    await session.scalars(
                        select(DomainEventRow)
                        .where(DomainEventRow.causation_id == started_row.causation_id)
                        .order_by(DomainEventRow.event_id)
                    )
                ).all()
            )
            source = await session.get(InspirationRunRow, payload.run_id)
            legal_event_types = (
                (InspirationStartedV1.event_type,)
                if len(event_rows) == 1
                else (
                    InspirationStartedV1.event_type,
                    "inspiration.snapshot_frozen",
                )
            )
            if (
                started_row.aggregate_type != "inspiration_run"
                or started_row.aggregate_id != payload.run_id
                or started_row.sequence != 1
                or not event_rows
                or tuple(row.event_type for row in event_rows) != legal_event_types
                or tuple(row.sequence for row in event_rows)
                != tuple(range(1, len(event_rows) + 1))
                or tuple(row.event_id for row in causal_rows)
                != tuple(row.event_id for row in event_rows)
                or any(row.occurred_at != started_row.occurred_at for row in event_rows)
                or source is None
                or source.owner_agent_id != payload.owner_agent_id
                or source.created_at != started_row.occurred_at
            ):
                raise InspirationSourceIntegrityError(
                    "running trace is not a legal retained phase"
                )
            traces.append(
                RunningInspirationTrace(
                    run_id=payload.run_id,
                    owner_agent_id=payload.owner_agent_id,
                    receipt_id=started_row.causation_id,
                    request_hash=source.request_hash,
                    last_event_at=max(
                        require_utc(row.occurred_at) for row in event_rows
                    ),
                )
            )
        return tuple(traces)


__all__ = [
    "InspirationRepository",
    "InspirationSourceIntegrityError",
    "RunningInspirationTrace",
]
