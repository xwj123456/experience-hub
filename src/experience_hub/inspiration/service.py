"""Three-transaction execution of durable, bounded inspiration runs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from typing import Protocol
from uuid import UUID

from experience_hub.clock import Clock, require_utc
from experience_hub.domain import CommandContext, CommandRequest, PendingEvent
from experience_hub.ids import IdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.deadlines import (
    BoundedGenerationRunner,
    OperatorGeneration,
    OperatorGenerationRun,
)
from experience_hub.inspiration.dedup import (
    RunDeduplicationResult,
    deduplicate_run_batches,
)
from experience_hub.inspiration.events import (
    InspirationCompletedV1,
    InspirationFailedV1,
    InspirationIdeaGeneratedV1,
    InspirationOperatorCompletedV1,
    InspirationOperatorFailedV1,
    InspirationRunFailureCode,
    InspirationSnapshotFrozenV1,
    InspirationStartedV1,
    InspirationTimedOutV1,
)
from experience_hub.inspiration.generators.base import ManagedIdeaGenerator
from experience_hub.inspiration.generators.openai_compatible import (
    GeneratorNotConfiguredError,
)
from experience_hub.inspiration.incubation import ClusterTransition, plan_occurrence
from experience_hub.inspiration.models import (
    FrozenSnapshot,
    GeneratorKind,
    IdeaOwnerDecision,
    InspirationOperator,
    InspirationRunStatus,
    OperatorOutcome,
    SnapshotItem,
)
from experience_hub.inspiration.repository import InspirationRepository
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.inspiration.validation import (
    ValidatedOperatorBatch,
    validate_operator_batch,
)
from experience_hub.storage.database import Database
from experience_hub.storage.faults import FaultCheckpoint
from experience_hub.storage.idempotency import (
    CompletedReceipt,
    InProgressReceipt,
    NewReceipt,
    ReceiptReservation,
    ReceiptStore,
    StoredResponse,
)


class GeneratorFactory(Protocol):
    """Build only the explicitly selected managed generator."""

    def __call__(self, kind: GeneratorKind) -> ManagedIdeaGenerator: ...


class GenerationRunner(Protocol):
    """Provider-independent bounded generation used outside transactions."""

    async def run(
        self,
        *,
        generator: ManagedIdeaGenerator,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operators: tuple[InspirationOperator, ...],
        branch_limit: int,
        output_tokens_per_operator: int,
        total_output_tokens: int,
        operator_timeout_seconds: int,
        global_timeout_seconds: int,
    ) -> OperatorGenerationRun: ...


class _PreparationFailed(Exception):
    """Sanitized boundary for evidence preparation failures only."""


def _validated_run(value: object) -> StartInspirationRun:
    if not isinstance(value, StartInspirationRun):
        raise ValueError("run must be StartInspirationRun")
    try:
        rebuilt = StartInspirationRun(
            owner_agent_id=value.owner_agent_id,
            goal=value.goal,
            context=value.context,
            mode=value.mode,
            generator=value.generator,
            operators=value.operators,
            include_inbox=value.include_inbox,
            branches_per_operator=value.branches_per_operator,
            output_tokens_per_operator=value.output_tokens_per_operator,
            total_output_tokens=value.total_output_tokens,
            operator_timeout_seconds=value.operator_timeout_seconds,
            global_timeout_seconds=value.global_timeout_seconds,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("run must be a valid StartInspirationRun") from error
    if rebuilt != value:
        raise ValueError("run must be a canonical StartInspirationRun")
    return rebuilt


def _validate_scope(
    *,
    request: CommandRequest,
    run: StartInspirationRun,
) -> None:
    expected_body = {
        "goal": run.goal,
        "context": run.context,
        "mode": run.mode.value,
        "generator": run.generator.value,
        "operators": tuple(operator.value for operator in run.operators),
        "include_inbox": run.include_inbox,
        "branches_per_operator": run.branches_per_operator,
        "output_tokens_per_operator": run.output_tokens_per_operator,
        "total_output_tokens": run.total_output_tokens,
        "operator_timeout_seconds": run.operator_timeout_seconds,
        "global_timeout_seconds": run.global_timeout_seconds,
    }
    if not isinstance(request, CommandRequest):
        raise ValueError("request must be CommandRequest")
    if (
        request.caller_scope != f"agent:{run.owner_agent_id}"
        or request.operation_scope != "inspiration.run.start"
        or request.method != "POST"
        or request.route_template != "/v1/agents/{agent_id}/inspiration-runs"
        or dict(request.path_parameters) != {"agent_id": run.owner_agent_id}
        or request.query_parameters
        or request.semantic_headers
        or not isinstance(request.body, Mapping)
        or dict(request.body) != expected_body
    ):
        raise ValueError("request semantics do not match the inspiration run")


def _failed_batch(item: OperatorGeneration) -> ValidatedOperatorBatch:
    if item.result.error_code is None:
        raise ValueError("failed generator result requires an error code")
    return ValidatedOperatorBatch(
        operator=item.operator,
        ideas=(),
        error_code=item.result.error_code,
        output_tokens_consumed=item.result.output_tokens_consumed,
    )


def _validate_and_deduplicate(
    *,
    run_id: UUID,
    snapshot: FrozenSnapshot,
    generated: OperatorGenerationRun,
) -> RunDeduplicationResult:
    batches: list[ValidatedOperatorBatch] = []
    for item in generated.results:
        if item.result.error_code is not None:
            batches.append(_failed_batch(item))
            continue
        batches.append(
            validate_operator_batch(
                run_id=run_id,
                operator=item.operator,
                branches=item.result.ideas,
                snapshot_items=snapshot.items,
                output_tokens_consumed=item.result.output_tokens_consumed,
            )
        )
    return deduplicate_run_batches(tuple(batches))


def _status(
    *,
    outcomes: tuple[OperatorOutcome, ...],
    timed_out: bool,
) -> InspirationRunStatus:
    if timed_out:
        return InspirationRunStatus.TIMED_OUT
    succeeded = sum(outcome.succeeded for outcome in outcomes)
    if outcomes and succeeded == len(outcomes):
        return InspirationRunStatus.COMPLETED
    if succeeded:
        return InspirationRunStatus.COMPLETED_WITH_ERRORS
    return InspirationRunStatus.FAILED


def _generated_payload(
    *,
    idea_id: UUID,
    occurrence_id: UUID,
    run_id: UUID,
    owner_agent_id: UUID,
    snapshot_hash: str,
    retained: object,
    transition: ClusterTransition,
    duplicate_relation: UUID | None,
) -> InspirationIdeaGeneratedV1:
    from experience_hub.inspiration.dedup import RetainedIdea

    if not isinstance(retained, RetainedIdea):
        raise TypeError("retained must be RetainedIdea")
    return InspirationIdeaGeneratedV1(
        schema_version=1,
        idea_id=idea_id,
        occurrence_id=occurrence_id,
        run_id=run_id,
        owner_agent_id=owner_agent_id,
        operator=retained.operator,
        ordinal=retained.ordinal,
        snapshot_hash=snapshot_hash,
        evidence=retained.draft.evidence,
        idea_content_hash=retained.idea_content_hash,
        mechanism_hash=retained.mechanism_hash,
        duplicate_relation=duplicate_relation,
        owner_decision_after=IdeaOwnerDecision.ACTIVE,
        cluster_id=transition.cluster_id,
        canonical_mechanism_hash=transition.canonical_mechanism_hash,
        member_hashes_before=transition.member_hashes_before,
        member_hashes_after=transition.member_hashes_after,
        occurrence_count_before=transition.occurrence_count_before,
        occurrence_count_after=transition.occurrence_count_after,
        distinct_snapshot_count_before=transition.distinct_snapshot_count_before,
        distinct_snapshot_count_after=transition.distinct_snapshot_count_after,
        distinct_adopter_count_before=transition.distinct_adopter_count_before,
        distinct_adopter_count_after=transition.distinct_adopter_count_after,
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


class InspirationRunExecutor:
    """Own the split receipt, snapshot, generation, and finalization protocol."""

    def __init__(
        self,
        *,
        database: Database,
        receipt_store: ReceiptStore,
        repository: InspirationRepository,
        snapshot_builder: SnapshotBuilder,
        generator_factory: GeneratorFactory,
        response_codec: InspirationResponseCodec,
        clock: Clock,
        id_generator: IdGenerator,
        generation_runner: GenerationRunner | None = None,
    ) -> None:
        self.database = database
        self._receipt_store = receipt_store
        self._repository = repository
        self._snapshot_builder = snapshot_builder
        self._generator_factory = generator_factory
        self._runner = generation_runner or BoundedGenerationRunner()
        self._codec = response_codec
        self._clock = clock
        self._ids = id_generator

    async def execute(
        self,
        *,
        request: CommandRequest,
        run: StartInspirationRun,
    ) -> StoredResponse:
        canonical = _validated_run(run)
        _validate_scope(request=request, run=canonical)
        generator: ManagedIdeaGenerator | None = None
        try:
            phase_one = await self._start(
                request=request,
                run=canonical,
            )
            if isinstance(phase_one, StoredResponse):
                return phase_one
            reservation, command, run_id, occurred_at, generator = phase_one

            snapshot_value: FrozenSnapshot | None = None
            preparation_failed = False
            try:
                async with self.database.transaction(immediate=True) as uow:
                    try:
                        snapshot_value = await self._snapshot_builder.freeze(
                            uow=uow,
                            request=canonical,
                            run_id=run_id,
                            at=occurred_at,
                        )
                    except Exception as error:
                        raise _PreparationFailed from error
                    self._repository.add_snapshot(
                        session=uow.session,
                        snapshot=snapshot_value,
                    )
                    await uow.session.flush()
                    uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
                    await uow.append_events(
                        command,
                        (
                            PendingEvent(
                                aggregate_type="inspiration_run",
                                aggregate_id=run_id,
                                event_type=InspirationSnapshotFrozenV1.event_type,
                                payload=InspirationSnapshotFrozenV1(
                                    schema_version=1,
                                    run_id=run_id,
                                    snapshot_hash=snapshot_value.snapshot_hash,
                                    snapshot_item_ids=tuple(
                                        item.snapshot_item_id
                                        for item in snapshot_value.items
                                    ),
                                    status_before=InspirationRunStatus.RUNNING,
                                    status_after=InspirationRunStatus.RUNNING,
                                ),
                                actor_agent_id=canonical.owner_agent_id,
                                occurred_at=occurred_at,
                            ),
                        ),
                    )
            except _PreparationFailed:
                preparation_failed = True

            generated: OperatorGenerationRun | None = None
            deduplicated: RunDeduplicationResult | None = None
            if not preparation_failed:
                assert snapshot_value is not None
                generated = await self._runner.run(
                    generator=generator,
                    goal=canonical.goal,
                    context=canonical.context,
                    frozen_items=snapshot_value.items,
                    operators=canonical.operators,
                    branch_limit=canonical.branches_per_operator,
                    output_tokens_per_operator=canonical.output_tokens_per_operator,
                    total_output_tokens=canonical.total_output_tokens,
                    operator_timeout_seconds=canonical.operator_timeout_seconds,
                    global_timeout_seconds=canonical.global_timeout_seconds,
                )
                deduplicated = _validate_and_deduplicate(
                    run_id=run_id,
                    snapshot=snapshot_value,
                    generated=generated,
                )
            return await self._finalize(
                reservation=reservation,
                command=command,
                run_id=run_id,
                run=canonical,
                occurred_at=occurred_at,
                snapshot=snapshot_value,
                generated=generated,
                deduplicated=deduplicated,
                preparation_failed=preparation_failed,
            )
        finally:
            if generator is not None:
                with suppress(Exception):
                    await generator.aclose()

    async def _start(
        self,
        *,
        request: CommandRequest,
        run: StartInspirationRun,
    ) -> (
        StoredResponse
        | tuple[
            ReceiptReservation,
            CommandContext,
            UUID,
            datetime,
            ManagedIdeaGenerator,
        ]
    ):
        generator: ManagedIdeaGenerator | None = None
        try:
            async with self.database.transaction(immediate=True) as uow:
                decision = await self._receipt_store.reserve(
                    uow=uow,
                    request=request,
                )
                if isinstance(decision, CompletedReceipt):
                    assert decision.record.response is not None
                    result: (
                        StoredResponse
                        | tuple[
                            ReceiptReservation,
                            CommandContext,
                            UUID,
                            datetime,
                            ManagedIdeaGenerator,
                        ]
                    ) = decision.record.response
                elif isinstance(decision, InProgressReceipt):
                    record = decision.record
                    if (
                        record.result_resource_type != "inspiration_run"
                        or record.result_resource_id is None
                    ):
                        raise RuntimeError(
                            "in-progress inspiration receipt has no run attachment"
                        )
                    result = self._codec.in_progress(
                        receipt_id=record.receipt_id,
                        run_id=record.result_resource_id,
                    )
                elif isinstance(decision, NewReceipt):
                    reservation = decision.reservation
                    try:
                        generator = self._generator_factory(run.generator)
                    except GeneratorNotConfiguredError as error:
                        response = self._codec.generator_not_configured(error)
                        await self._receipt_store.complete(
                            uow=uow,
                            reservation=reservation,
                            response=response,
                            completed_at=max(
                                require_utc(reservation.created_at),
                                require_utc(self._clock.now()),
                            ),
                        )
                        result = response
                    else:
                        if not await self._repository.agent_exists(
                            session=uow.session,
                            agent_id=run.owner_agent_id,
                        ):
                            raise ValueError("inspiration owner does not exist")
                        run_id = self._ids.new()
                        occurred_at = max(
                            require_utc(reservation.created_at),
                            require_utc(self._clock.now()),
                        )
                        self._repository.add_run(
                            session=uow.session,
                            run_id=run_id,
                            request=run,
                            generator_configuration=(generator.persisted_configuration),
                            request_hash=request.request_hash,
                            occurred_at=occurred_at,
                        )
                        await uow.session.flush()
                        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
                        await self._receipt_store.attach_resource(
                            uow=uow,
                            receipt_id=reservation.receipt_id,
                            resource_type="inspiration_run",
                            resource_id=run_id,
                        )
                        command = reservation.command_context()
                        await uow.append_events(
                            command,
                            (
                                PendingEvent(
                                    aggregate_type="inspiration_run",
                                    aggregate_id=run_id,
                                    event_type=InspirationStartedV1.event_type,
                                    payload=InspirationStartedV1(
                                        schema_version=1,
                                        run_id=run_id,
                                        owner_agent_id=run.owner_agent_id,
                                        status_after=(InspirationRunStatus.RUNNING),
                                    ),
                                    actor_agent_id=run.owner_agent_id,
                                    occurred_at=occurred_at,
                                ),
                            ),
                        )
                        result = (
                            reservation,
                            command,
                            run_id,
                            occurred_at,
                            generator,
                        )
                else:
                    raise RuntimeError("unknown receipt decision")
        except BaseException:
            if generator is not None:
                with suppress(Exception):
                    await generator.aclose()
            raise
        return result

    async def _finalize(
        self,
        *,
        reservation: ReceiptReservation,
        command: CommandContext,
        run_id: UUID,
        run: StartInspirationRun,
        occurred_at: datetime,
        snapshot: FrozenSnapshot | None,
        generated: OperatorGenerationRun | None,
        deduplicated: RunDeduplicationResult | None,
        preparation_failed: bool,
    ) -> StoredResponse:
        retained_at = require_utc(occurred_at)
        async with self.database.transaction(immediate=True) as uow:
            terminal_payload: (
                InspirationCompletedV1 | InspirationFailedV1 | InspirationTimedOutV1
            )
            if preparation_failed:
                outcomes: tuple[OperatorOutcome, ...] = ()
                reserved = consumed = elapsed = 0
                terminal_payload = InspirationFailedV1(
                    schema_version=1,
                    run_id=run_id,
                    status_before=InspirationRunStatus.RUNNING,
                    status_after=InspirationRunStatus.FAILED,
                    operator_outcomes=outcomes,
                    output_tokens_reserved_before=reserved,
                    output_tokens_reserved_after=reserved,
                    output_tokens_consumed_before=consumed,
                    output_tokens_consumed_after=consumed,
                    elapsed_milliseconds_before=elapsed,
                    elapsed_milliseconds_after=elapsed,
                    failure_code=InspirationRunFailureCode.PREPARATION_FAILED,
                )
            else:
                assert snapshot is not None
                assert generated is not None
                assert deduplicated is not None
                outcomes = deduplicated.outcomes
                generated_by_operator = {
                    item.operator: item for item in generated.results
                }
                retained_by_operator = {
                    outcome.operator: tuple(
                        idea
                        for idea in deduplicated.ideas
                        if idea.operator is outcome.operator
                    )
                    for outcome in outcomes
                }
                reserved = consumed = elapsed = 0
                for outcome in outcomes:
                    item = generated_by_operator[outcome.operator]
                    for retained in retained_by_operator[outcome.operator]:
                        clusters = await self._repository.load_clusters(
                            session=uow.session
                        )
                        plan = plan_occurrence(
                            owner_agent_id=run.owner_agent_id,
                            mechanism=retained.draft.mechanism,
                            snapshot_hash=snapshot.snapshot_hash,
                            run_occurred_at=retained_at,
                            clusters=clusters,
                        )
                        idea_id = self._ids.new()
                        occurrence_id = self._ids.new()
                        await self._repository.add_idea_occurrence(
                            session=uow.session,
                            idea_id=idea_id,
                            occurrence_id=occurrence_id,
                            run_id=run_id,
                            owner_agent_id=run.owner_agent_id,
                            snapshot_hash=snapshot.snapshot_hash,
                            retained=retained,
                            plan=plan,
                            occurred_at=retained_at,
                        )
                        await uow.session.flush()
                        uow.inject_fault(FaultCheckpoint.AFTER_SOURCE_INSERT)
                        payload = _generated_payload(
                            idea_id=idea_id,
                            occurrence_id=occurrence_id,
                            run_id=run_id,
                            owner_agent_id=run.owner_agent_id,
                            snapshot_hash=snapshot.snapshot_hash,
                            retained=retained,
                            transition=plan.transition,
                            duplicate_relation=plan.duplicate_relation,
                        )
                        await uow.append_events(
                            command,
                            (
                                PendingEvent(
                                    aggregate_type="idea",
                                    aggregate_id=idea_id,
                                    event_type=payload.event_type,
                                    payload=payload,
                                    actor_agent_id=run.owner_agent_id,
                                    occurred_at=retained_at,
                                ),
                            ),
                        )

                    reserved_after = reserved + item.output_tokens_reserved
                    consumed_after = consumed + outcome.output_tokens_consumed
                    elapsed_after = item.elapsed_milliseconds_after
                    payload_type = (
                        InspirationOperatorCompletedV1
                        if outcome.succeeded
                        else InspirationOperatorFailedV1
                    )
                    operator_payload = payload_type(
                        schema_version=1,
                        run_id=run_id,
                        operator=outcome.operator,
                        outcome=outcome,
                        status_before=InspirationRunStatus.RUNNING,
                        status_after=InspirationRunStatus.RUNNING,
                        output_tokens_reserved_before=reserved,
                        output_tokens_reserved_after=reserved_after,
                        output_tokens_consumed_before=consumed,
                        output_tokens_consumed_after=consumed_after,
                        elapsed_milliseconds_before=elapsed,
                        elapsed_milliseconds_after=elapsed_after,
                    )
                    await uow.append_events(
                        command,
                        (
                            PendingEvent(
                                aggregate_type="inspiration_run",
                                aggregate_id=run_id,
                                event_type=operator_payload.event_type,
                                payload=operator_payload,
                                actor_agent_id=run.owner_agent_id,
                                occurred_at=retained_at,
                            ),
                        ),
                    )
                    reserved = reserved_after
                    consumed = consumed_after
                    elapsed = elapsed_after

                status = _status(
                    outcomes=outcomes,
                    timed_out=generated.timed_out,
                )
                if status is InspirationRunStatus.TIMED_OUT:
                    from experience_hub.inspiration.failures import (
                        OperatorFailureCode,
                    )

                    terminal_payload = InspirationTimedOutV1(
                        schema_version=1,
                        run_id=run_id,
                        status_before=InspirationRunStatus.RUNNING,
                        status_after=status,
                        operator_outcomes=outcomes,
                        output_tokens_reserved_before=reserved,
                        output_tokens_reserved_after=reserved,
                        output_tokens_consumed_before=consumed,
                        output_tokens_consumed_after=consumed,
                        elapsed_milliseconds_before=elapsed,
                        elapsed_milliseconds_after=elapsed,
                        failure_code=(OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED),
                    )
                elif status is InspirationRunStatus.FAILED:
                    terminal_payload = InspirationFailedV1(
                        schema_version=1,
                        run_id=run_id,
                        status_before=InspirationRunStatus.RUNNING,
                        status_after=status,
                        operator_outcomes=outcomes,
                        output_tokens_reserved_before=reserved,
                        output_tokens_reserved_after=reserved,
                        output_tokens_consumed_before=consumed,
                        output_tokens_consumed_after=consumed,
                        elapsed_milliseconds_before=elapsed,
                        elapsed_milliseconds_after=elapsed,
                        failure_code=(InspirationRunFailureCode.ALL_OPERATORS_FAILED),
                    )
                else:
                    terminal_payload = InspirationCompletedV1(
                        schema_version=1,
                        run_id=run_id,
                        status_before=InspirationRunStatus.RUNNING,
                        status_after=status,
                        operator_outcomes=outcomes,
                        output_tokens_reserved_before=reserved,
                        output_tokens_reserved_after=reserved,
                        output_tokens_consumed_before=consumed,
                        output_tokens_consumed_after=consumed,
                        elapsed_milliseconds_before=elapsed,
                        elapsed_milliseconds_after=elapsed,
                    )

            await uow.append_events(
                command,
                (
                    PendingEvent(
                        aggregate_type="inspiration_run",
                        aggregate_id=run_id,
                        event_type=terminal_payload.event_type,
                        payload=terminal_payload,
                        actor_agent_id=run.owner_agent_id,
                        occurred_at=retained_at,
                    ),
                ),
            )
            stored_run = await self._repository.get_run(
                session=uow.session,
                run_id=run_id,
            )
            if stored_run is None:
                raise RuntimeError("terminal inspiration run is missing")
            response = self._codec.terminal(stored_run)
            await self._receipt_store.complete_existing(
                uow=uow,
                receipt_id=reservation.receipt_id,
                response=response,
                completed_at=retained_at,
            )
            uow.inject_fault(FaultCheckpoint.AFTER_RECEIPT_COMPLETION)
            return response


__all__ = [
    "GenerationRunner",
    "GeneratorFactory",
    "InspirationRunExecutor",
]
