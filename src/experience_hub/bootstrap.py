"""Composition root for the complete Experience Hub application."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from experience_hub.agents.events import register_agent_events
from experience_hub.agents.service import AgentService
from experience_hub.clock import Clock, SystemClock
from experience_hub.config import Settings
from experience_hub.domain.events import EventRegistry
from experience_hub.experiences.projector import (
    ExperienceProjector,
    ExperienceTermsProjector,
)
from experience_hub.experiences.queries import ExperienceQuery
from experience_hub.experiences.reconcile import PayloadReconciler
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import (
    ExperienceRetrievalAdapter,
    ExperienceService,
)
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.ids import IdGenerator, Uuid4Generator
from experience_hub.inspiration.generators.base import ManagedIdeaGenerator
from experience_hub.inspiration.generators.openai_compatible import (
    build_idea_generator,
)
from experience_hub.inspiration.lifecycle import (
    IdeaLifecycleService,
    InspirationIdeaArchivePlanner,
)
from experience_hub.inspiration.models import GeneratorKind
from experience_hub.inspiration.projector import (
    IdeaStateProjector,
    InspirationRunProjector,
    MechanismIncubationProjector,
)
from experience_hub.inspiration.recovery import InspirationRunRecovery
from experience_hub.inspiration.repository import InspirationRepository
from experience_hub.inspiration.response_codec import InspirationResponseCodec
from experience_hub.inspiration.service import InspirationRunExecutor
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.lifecycle.repository import LifecycleRepository
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import LifecycleService
from experience_hub.lifecycle.worker import (
    LifecycleWorker,
    ProductionLifecycleTicker,
)
from experience_hub.retrieval.service import (
    ExperienceEvidenceReader,
    RetrievalService,
)
from experience_hub.sharing.events import register_sharing_events
from experience_hub.sharing.projector import (
    AgentReputationProjector,
    CapsuleStateProjector,
    InboxItemProjector,
)
from experience_hub.sharing.queries import InboxEvidenceReader, SharingQuery
from experience_hub.sharing.repository import SharingRepository
from experience_hub.sharing.service import SharingService
from experience_hub.sharing.validation import register_sharing_source_validator
from experience_hub.storage.database import Database
from experience_hub.storage.idempotency import CommandExecutor, ReceiptStore
from experience_hub.storage.projections import (
    ProjectionManager,
    ProjectionRegistry,
)
from experience_hub.storage.validation import (
    SourceValidator,
    register_agent_source_validator,
    register_experience_source_validator,
    register_inspiration_source_validator,
)


class GeneratorRegistry:
    """Create only the generator explicitly selected for one run."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self, kind: GeneratorKind) -> ManagedIdeaGenerator:
        return build_idea_generator(kind=kind, settings=self._settings)


type ShutdownHook = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class ApplicationContainer:
    """Own one coherent dependency graph and its resource lifecycle."""

    settings: Settings
    clock: Clock
    ids: IdGenerator
    lifecycle_config: LifecycleConfig
    event_registry: EventRegistry
    source_validator: SourceValidator
    projection_registry: ProjectionRegistry
    projection_manager: ProjectionManager
    database: Database
    receipt_store: ReceiptStore
    command_executor: CommandExecutor
    experience_repository: ExperienceRepository
    experience_query: ExperienceQuery
    experience_writer: ExperienceWriter
    experience_mutation_writer: ExperienceMutationWriter
    retrieval_service: RetrievalService
    retrieval_adapter: ExperienceRetrievalAdapter
    experience_evidence_reader: ExperienceEvidenceReader
    payload_reconciler: PayloadReconciler
    sharing_repository: SharingRepository
    sharing_query: SharingQuery
    inbox_evidence_reader: InboxEvidenceReader
    agent_service: AgentService
    experience_service: ExperienceService
    sharing_service: SharingService
    inspiration_repository: InspirationRepository
    inspiration_response_codec: InspirationResponseCodec
    snapshot_builder: SnapshotBuilder
    generator_registry: GeneratorRegistry
    inspiration_run_executor: InspirationRunExecutor
    idea_lifecycle_service: IdeaLifecycleService
    inspiration_recovery: InspirationRunRecovery
    idea_archive_planner: InspirationIdeaArchivePlanner
    lifecycle_repository: LifecycleRepository
    lifecycle_service: LifecycleService
    lifecycle_ticker: ProductionLifecycleTicker
    lifecycle_worker: LifecycleWorker
    schema_revision: str | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _shutdown_hooks: list[ShutdownHook] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def reducer_versions(self) -> dict[str, int]:
        return {
            reducer.name: reducer.version
            for reducer in self.projection_registry.reducers
        }

    def register_shutdown_hook(self, hook: ShutdownHook) -> None:
        """Register task/adaptor cleanup that must run before engine disposal."""
        if self._closed:
            raise RuntimeError("Application container is already closed")
        if not callable(hook):
            raise TypeError("shutdown hook must be callable")
        self._shutdown_hooks.append(hook)

    @classmethod
    def build(
        cls,
        settings: Settings,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> ApplicationContainer:
        """Build the graph synchronously; adapters open resources lazily."""
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        if lifecycle_config is not None and not isinstance(
            lifecycle_config,
            LifecycleConfig,
        ):
            raise TypeError("lifecycle_config must be LifecycleConfig or None")
        retained_clock = clock if clock is not None else SystemClock()
        retained_ids = ids if ids is not None else Uuid4Generator()
        retained_lifecycle_config = lifecycle_config or LifecycleConfig()

        event_registry = EventRegistry()
        register_agent_events(event_registry)
        from experience_hub.experiences.events import register_experience_events
        from experience_hub.inspiration.events import register_inspiration_events

        register_experience_events(event_registry)
        register_sharing_events(event_registry)
        register_inspiration_events(event_registry)

        source_validator = SourceValidator(event_registry)
        register_agent_source_validator(source_validator)
        register_experience_source_validator(source_validator)
        register_sharing_source_validator(source_validator)
        register_inspiration_source_validator(source_validator)

        projection_registry = ProjectionRegistry(
            (
                ExperienceProjector(event_registry, retained_lifecycle_config),
                ExperienceTermsProjector(event_registry),
                CapsuleStateProjector(event_registry),
                AgentReputationProjector(event_registry),
                InboxItemProjector(event_registry),
                InspirationRunProjector(event_registry),
                MechanismIncubationProjector(event_registry),
                IdeaStateProjector(event_registry),
            )
        )
        projection_manager = ProjectionManager(
            projection_registry,
            source_validator=source_validator,
        )
        database_url = settings.database_url
        assert database_url is not None
        database = Database.create(
            database_url,
            event_registry=event_registry,
            projection_applier=projection_manager,
        )

        receipt_store = ReceiptStore(
            clock=retained_clock,
            id_generator=retained_ids,
        )
        command_executor = CommandExecutor(
            database=database,
            receipt_store=receipt_store,
            clock=retained_clock,
        )

        experience_repository = ExperienceRepository(event_registry=event_registry)
        experience_query = ExperienceQuery(event_registry=event_registry)
        experience_writer = ExperienceWriter(
            id_generator=retained_ids,
            repository=experience_repository,
            lifecycle_config=retained_lifecycle_config,
        )
        experience_mutation_writer = ExperienceMutationWriter(
            repository=experience_repository,
            lifecycle_config=retained_lifecycle_config,
        )
        retrieval_service = RetrievalService(
            clock=retained_clock,
            query=experience_query,
            mutation_writer=experience_mutation_writer,
            lifecycle_config=retained_lifecycle_config,
        )
        retrieval_adapter = ExperienceRetrievalAdapter(
            executor=command_executor,
            retrieval_service=retrieval_service,
            id_generator=retained_ids,
        )
        experience_evidence_reader = ExperienceEvidenceReader(
            clock=retained_clock,
            query=experience_query,
            lifecycle_config=retained_lifecycle_config,
        )
        payload_reconciler = PayloadReconciler()

        sharing_repository = SharingRepository(event_registry=event_registry)
        sharing_query = SharingQuery(repository=sharing_repository)
        inbox_evidence_reader = InboxEvidenceReader(repository=sharing_repository)

        agent_service = AgentService(
            clock=retained_clock,
            id_generator=retained_ids,
            receipt_store=receipt_store,
        )
        experience_service = ExperienceService(
            clock=retained_clock,
            receipt_store=receipt_store,
            writer=experience_writer,
            mutation_writer=experience_mutation_writer,
            query=experience_query,
            lifecycle_config=retained_lifecycle_config,
        )
        sharing_service = SharingService(
            clock=retained_clock,
            id_generator=retained_ids,
            receipt_store=receipt_store,
            repository=sharing_repository,
            experience_query=experience_query,
            experience_writer=experience_writer,
            experience_repository=experience_repository,
            experience_service=experience_service,
        )

        inspiration_repository = InspirationRepository(event_registry)
        inspiration_response_codec = InspirationResponseCodec()
        snapshot_builder = SnapshotBuilder(
            experience_reader=experience_evidence_reader,
            inbox_reader=inbox_evidence_reader,
            id_generator=retained_ids,
        )
        generator_registry = GeneratorRegistry(settings)
        inspiration_run_executor = InspirationRunExecutor(
            database=database,
            receipt_store=receipt_store,
            repository=inspiration_repository,
            snapshot_builder=snapshot_builder,
            generator_factory=generator_registry,
            response_codec=inspiration_response_codec,
            clock=retained_clock,
            id_generator=retained_ids,
        )
        idea_lifecycle_service = IdeaLifecycleService(
            clock=retained_clock,
            receipt_store=receipt_store,
            repository=inspiration_repository,
            id_generator=retained_ids,
            experience_writer=experience_writer,
            experience_repository=experience_repository,
        )
        inspiration_recovery = InspirationRunRecovery(
            database=database,
            receipt_store=receipt_store,
            repository=inspiration_repository,
            response_codec=inspiration_response_codec,
            clock=retained_clock,
        )
        idea_archive_planner = InspirationIdeaArchivePlanner()

        lifecycle_repository = LifecycleRepository()
        lifecycle_service = LifecycleService(
            clock=retained_clock,
            receipt_store=receipt_store,
            repository=lifecycle_repository,
            mutation_writer=experience_mutation_writer,
            config=retained_lifecycle_config,
            idea_archive_planner=idea_archive_planner,
        )
        lifecycle_ticker = ProductionLifecycleTicker(
            retained_lifecycle_config.worker_interval
        )
        lifecycle_worker = LifecycleWorker(
            clock=retained_clock,
            ticker=lifecycle_ticker,
            executor=command_executor,
            service=lifecycle_service,
        )

        return cls(
            settings=settings,
            clock=retained_clock,
            ids=retained_ids,
            lifecycle_config=retained_lifecycle_config,
            event_registry=event_registry,
            source_validator=source_validator,
            projection_registry=projection_registry,
            projection_manager=projection_manager,
            database=database,
            receipt_store=receipt_store,
            command_executor=command_executor,
            experience_repository=experience_repository,
            experience_query=experience_query,
            experience_writer=experience_writer,
            experience_mutation_writer=experience_mutation_writer,
            retrieval_service=retrieval_service,
            retrieval_adapter=retrieval_adapter,
            experience_evidence_reader=experience_evidence_reader,
            payload_reconciler=payload_reconciler,
            sharing_repository=sharing_repository,
            sharing_query=sharing_query,
            inbox_evidence_reader=inbox_evidence_reader,
            agent_service=agent_service,
            experience_service=experience_service,
            sharing_service=sharing_service,
            inspiration_repository=inspiration_repository,
            inspiration_response_codec=inspiration_response_codec,
            snapshot_builder=snapshot_builder,
            generator_registry=generator_registry,
            inspiration_run_executor=inspiration_run_executor,
            idea_lifecycle_service=idea_lifecycle_service,
            inspiration_recovery=inspiration_recovery,
            idea_archive_planner=idea_archive_planner,
            lifecycle_repository=lifecycle_repository,
            lifecycle_service=lifecycle_service,
            lifecycle_ticker=lifecycle_ticker,
            lifecycle_worker=lifecycle_worker,
        )

    async def close(self) -> None:
        """Stop active work before releasing its database engine."""
        if self._closed:
            return
        self._closed = True
        pending_error: BaseException | None = None
        try:
            try:
                await self.lifecycle_worker.stop()
            except BaseException as error:
                pending_error = error
            for hook in self._shutdown_hooks:
                try:
                    await hook()
                except BaseException as error:
                    if pending_error is None:
                        pending_error = error
        finally:
            try:
                await self.database.dispose()
            except BaseException as error:
                if pending_error is None:
                    pending_error = error
        if pending_error is not None:
            raise pending_error


__all__ = ["ApplicationContainer", "GeneratorRegistry", "ShutdownHook"]
