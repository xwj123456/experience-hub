"""Deterministic, inspectable end-to-end demonstration."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy.engine import URL

import experience_hub.config as config
from experience_hub.agents.models import CreateAgent
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.canonical import canonical_json_bytes
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import (
    CommandContext,
    CommandRequest,
    TypedEvidence,
)
from experience_hub.errors import DomainError
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.models import (
    ExperienceKind,
    VersionContent,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.inspiration.commands import (
    AdoptIdea,
    StartInspirationRun,
)
from experience_hub.inspiration.hashing import stable_evidence_key
from experience_hub.inspiration.models import (
    EvidenceSourceType,
    GeneratorKind,
    Idea,
    IdeaOwnerDecision,
    InspirationOperator,
)
from experience_hub.inspiration.request_hashing import (
    adoption_command_request,
)
from experience_hub.lifecycle import decode_lifecycle_result
from experience_hub.retrieval.contracts import SearchExperiences
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.runtime import ApplicationRuntime
from experience_hub.sharing.hashing import (
    compute_original_root_fingerprint,
)
from experience_hub.sharing.models import (
    AdoptCapsule,
    CreateSubscription,
    CreateTopic,
    InboxState,
    ProvenanceHop,
    PublishCapsule,
)
from experience_hub.storage.idempotency import (
    CommandResult,
    StoredResponse,
)
from experience_hub.storage.unit_of_work import UnitOfWork

type CommandHandler = Callable[
    [UnitOfWork, CommandContext],
    Awaitable[StoredResponse],
]
type JsonResponse = CommandResult | StoredResponse

_STARTED_AT = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
_STAGE_NAMES = (
    "create Alice and Bob",
    "Alice records operational experiences",
    "Bob subscribes to Alice's topic",
    "Alice publishes a capsule",
    "Bob receives and explicitly adopts it",
    "advance time until an unrelated experience is cold",
    "show an ordinary query returning only its blurred projection",
    "issue a strong contextual cue that expands and reactivates it",
    "generate deterministic inspiration from frozen evidence",
    "prove generated ideas are absent from experience recall",
    "explicitly adopt one idea and retrieve the resulting hypothesis",
)


class DemoDatabaseExists(DomainError):
    """The retained demo must not be overwritten without explicit consent."""

    def __init__(self, database_path: Path) -> None:
        super().__init__(
            code="demo_database_exists",
            message="Demo database already exists; rerun with --reset",
            details={
                "database_path": str(database_path),
                "retry_with_reset": True,
            },
            status_code=409,
        )


def _database_path() -> Path:
    return config.repository_root() / ".data" / "demo.db"


def _reset_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _settings(path: Path) -> Settings:
    url = URL.create("sqlite+aiosqlite", database=str(path))
    return Settings(database_url=url.render_as_string(hide_password=False))


def _ids() -> SequenceIdGenerator:
    return SequenceIdGenerator(tuple(UUID(int=value) for value in range(1, 1_025)))


def _document(result: JsonResponse) -> dict[str, Any]:
    try:
        decoded = json.loads(result.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Demo command returned invalid JSON") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != result.body:
        raise RuntimeError("Demo command returned non-canonical JSON")
    return cast(dict[str, Any], decoded)


def _success_data(
    result: JsonResponse,
    *,
    expected_status: int,
) -> dict[str, Any]:
    decoded = _document(result)
    if result.status_code != expected_status:
        raise RuntimeError(
            f"Demo command failed with status {result.status_code}: {decoded}"
        )
    data = decoded.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Demo command response has no data object")
    return cast(dict[str, Any], data)


def _experience_hit(
    hits: list[dict[str, Any]],
    *,
    experience_id: UUID,
) -> dict[str, Any]:
    expected = str(experience_id)
    for hit in hits:
        experience = hit.get("experience")
        if isinstance(experience, dict) and experience.get("experience_id") == expected:
            return hit
    raise RuntimeError(f"Experience {experience_id} is absent from search results")


async def _execute(
    container: ApplicationContainer,
    request: CommandRequest,
    handler: CommandHandler,
) -> CommandResult:
    return await container.command_executor.execute(request, handler)


async def _create_agent(
    container: ApplicationContainer,
    *,
    name: str,
    key: str,
) -> UUID:
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        body={"name": name},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=name),
            command_context=context,
        )

    result = await _execute(container, request, handler)
    return UUID(_success_data(result, expected_status=201)["agent_id"])


async def _create_experience(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    key: str,
    kind: ExperienceKind,
    content: VersionContent,
    importance: float,
    confidence: float,
) -> dict[str, Any]:
    command = CreateExperience(
        owner_agent_id=owner_agent_id,
        kind=kind,
        content=content,
        importance=importance,
        confidence=confidence,
    )
    body = {
        **content.model_dump(mode="python", warnings=False),
        "confidence": confidence,
        "importance": importance,
        "kind": kind,
        "links": (),
    }
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="experience.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": owner_agent_id},
        body=body,
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.experience_service.create(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await _execute(container, request, handler)
    return _success_data(result, expected_status=201)


async def _create_topic(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
) -> dict[str, Any]:
    command = CreateTopic(
        owner_agent_id=owner_agent_id,
        name="Reliable Operations",
        description="Operational knowledge shared through explicit adoption.",
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="topic.create",
        idempotency_key="demo-topic-create",
        method="POST",
        route_template="/v1/topics",
        body={
            "description": command.description,
            "name": command.name,
            "owner_agent_id": command.owner_agent_id,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_topic(
            uow=uow,
            command=command,
            command_context=context,
        )

    return _success_data(
        await _execute(container, request, handler),
        expected_status=201,
    )


async def _create_subscription(
    container: ApplicationContainer,
    *,
    subscriber_agent_id: UUID,
    topic_id: UUID,
) -> dict[str, Any]:
    command = CreateSubscription(
        subscriber_agent_id=subscriber_agent_id,
        topic_id=topic_id,
    )
    request = CommandRequest(
        caller_scope=f"agent:{subscriber_agent_id}",
        operation_scope="subscription.create",
        idempotency_key="demo-subscription-create",
        method="POST",
        route_template="/v1/agents/{agent_id}/subscriptions",
        path_parameters={"agent_id": subscriber_agent_id},
        body={"topic_id": topic_id},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.create_subscription(
            uow=uow,
            command=command,
            command_context=context,
        )

    return _success_data(
        await _execute(container, request, handler),
        expected_status=201,
    )


async def _publish_capsule(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    topic_id: UUID,
    experience_id: UUID,
    version_id: UUID,
) -> dict[str, Any]:
    command = PublishCapsule(
        owner_agent_id=owner_agent_id,
        topic_id=topic_id,
        experience_id=experience_id,
        version_id=version_id,
        expires_at=container.clock.now() + timedelta(days=30),
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="capsule.publish",
        idempotency_key="demo-capsule-publish",
        method="POST",
        route_template="/v1/agents/{agent_id}/capsules",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "experience_id": command.experience_id,
            "expires_at": command.expires_at,
            "parent_adoption_id": command.parent_adoption_id,
            "topic_id": command.topic_id,
            "version_id": command.version_id,
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.publish_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    return _success_data(
        await _execute(container, request, handler),
        expected_status=201,
    )


async def _adopt_capsule(
    container: ApplicationContainer,
    *,
    adopter_agent_id: UUID,
    item_id: UUID,
) -> tuple[dict[str, Any], UUID]:
    command = AdoptCapsule(
        adopter_agent_id=adopter_agent_id,
        item_id=item_id,
        importance=1.0,
    )
    request = CommandRequest(
        caller_scope=f"agent:{adopter_agent_id}",
        operation_scope="capsule.adopt",
        idempotency_key="demo-capsule-adopt",
        method="POST",
        route_template="/v1/agents/{agent_id}/inbox/{item_id}:adopt",
        path_parameters={
            "agent_id": adopter_agent_id,
            "item_id": item_id,
        },
        body={"importance": command.importance},
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.sharing_service.adopt_capsule(
            uow=uow,
            command=command,
            command_context=context,
        )

    result = await _execute(container, request, handler)
    data = _success_data(result, expected_status=200)
    location = result.headers.get("location")
    if location is None:
        raise RuntimeError("Capsule adoption response has no location")
    return data, UUID(location.rsplit("/", 1)[-1])


async def _run_lifecycle(
    container: ApplicationContainer,
    *,
    key: str,
) -> dict[str, Any]:
    evaluated_at = container.clock.now()
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=key,
        method="POST",
        route_template="/v1/lifecycle:run",
        body={
            "evaluated_at": evaluated_at,
            "mode": "manual",
        },
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.lifecycle_service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=context,
            mode="manual",
        )

    result = await _execute(container, request, handler)
    if result.status_code != 200:
        raise RuntimeError(f"Lifecycle cycle failed: {_document(result)}")
    decoded = decode_lifecycle_result(result.body)
    return {
        "archive_count": decoded.archive_count,
        "cycle_id": str(decoded.cycle_id),
        "evaluated_at": decoded.evaluated_at,
        "evaluated_count": decoded.evaluated_count,
        "idea_archive_count": decoded.idea_archive_count,
        "transition_count": decoded.transition_count,
    }


async def _search(
    container: ApplicationContainer,
    *,
    query: SearchExperiences,
    key: str,
) -> tuple[CommandResult, dict[str, Any]]:
    result = await container.retrieval_adapter.search(
        query=query,
        idempotency_key=key,
    )
    return result, _success_data(result, expected_status=200)


async def _start_inspiration(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
) -> tuple[dict[str, Any], Idea]:
    run = StartInspirationRun(
        owner_agent_id=owner_agent_id,
        goal="queue acknowledgement bounded capacity",
        context="Derive one testable operational counterfactual.",
        mode=RetrievalMode.FOCUSED,
        generator=GeneratorKind.DETERMINISTIC,
        operators=(InspirationOperator.COUNTERFACTUAL,),
        include_inbox=False,
        branches_per_operator=1,
        output_tokens_per_operator=1_200,
        total_output_tokens=1_200,
        operator_timeout_seconds=30,
        global_timeout_seconds=30,
    )
    request = CommandRequest(
        caller_scope=f"agent:{owner_agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key="demo-inspiration-run",
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": owner_agent_id},
        body={
            "branches_per_operator": run.branches_per_operator,
            "context": run.context,
            "generator": run.generator.value,
            "global_timeout_seconds": run.global_timeout_seconds,
            "goal": run.goal,
            "include_inbox": run.include_inbox,
            "mode": run.mode.value,
            "operator_timeout_seconds": run.operator_timeout_seconds,
            "operators": tuple(operator.value for operator in run.operators),
            "output_tokens_per_operator": run.output_tokens_per_operator,
            "total_output_tokens": run.total_output_tokens,
        },
    )
    result = await container.inspiration_run_executor.execute(
        request=request,
        run=run,
    )
    run_data = _success_data(
        result,
        expected_status=201,
    )
    run_id = UUID(run_data["run_id"])
    async with container.database.read_session() as session:
        ideas = await container.inspiration_repository.list_owned_ideas(
            session=session,
            owner_agent_id=owner_agent_id,
            run_id=run_id,
            after=None,
            limit=10,
        )
    if len(ideas) != 1:
        raise RuntimeError("Demo inspiration must produce exactly one idea")
    return run_data, ideas[0]


async def _adopt_idea(
    container: ApplicationContainer,
    *,
    owner_agent_id: UUID,
    idea_id: UUID,
) -> dict[str, Any]:
    command = AdoptIdea(
        owner_agent_id=owner_agent_id,
        idea_id=idea_id,
        importance=0.80,
        confidence=0.60,
    )
    request = adoption_command_request(
        command,
        idempotency_key="demo-idea-adopt",
    )

    async def handler(
        uow: UnitOfWork,
        context: CommandContext,
    ) -> StoredResponse:
        return await container.idea_lifecycle_service.adopt(
            uow=uow,
            command=command,
            command_context=context,
        )

    return _success_data(
        await _execute(container, request, handler),
        expected_status=200,
    )


async def build_demo_report(*, reset: bool) -> dict[str, Any]:
    """Run the complete demo and return its canonical report document."""
    database_path = _database_path()
    if reset:
        _reset_database(database_path)
    elif database_path.exists():
        raise DemoDatabaseExists(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    clock = FrozenClock(_STARTED_AT)
    runtime = ApplicationRuntime(
        _settings(database_path),
        clock=clock,
        ids=_ids(),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=True,
    ) as container:
        stages: list[dict[str, Any]] = []

        alice_id = await _create_agent(
            container,
            name="Alice",
            key="demo-agent-alice",
        )
        bob_id = await _create_agent(
            container,
            name="Bob",
            key="demo-agent-bob",
        )
        stages.append(
            {
                "name": _STAGE_NAMES[0],
                "result": {
                    "alice_id": str(alice_id),
                    "bob_id": str(bob_id),
                },
                "step": 1,
            }
        )

        shared_content = VersionContent(
            body=(
                "Queue acknowledgement frees bounded capacity only after "
                "durable completion."
            ),
            summary="Queue acknowledgement frees bounded capacity",
            mechanism=("A durable acknowledgement frees one bounded-capacity slot."),
            tags=("acknowledgement", "capacity", "queue"),
            applicability=("bounded worker queue",),
            evidence=(TypedEvidence(type="observation", id="demo-queue-trace"),),
            falsifiers=("Capacity is freed before durable acknowledgement.",),
        )
        dormant_content = VersionContent(
            body="租约交接 lease handoff",
            summary="zzz",
            mechanism="zzz",
            tags=("zzz",),
            applicability=("zzz",),
            evidence=(TypedEvidence(type="observation", id="demo-dormant-trace"),),
            falsifiers=("zzz",),
        )
        shared = await _create_experience(
            container,
            owner_agent_id=alice_id,
            key="demo-experience-shared",
            kind=ExperienceKind.PROCEDURAL,
            content=shared_content,
            importance=1.0,
            confidence=1.0,
        )
        dormant = await _create_experience(
            container,
            owner_agent_id=alice_id,
            key="demo-experience-dormant",
            kind=ExperienceKind.EPISODIC,
            content=dormant_content,
            importance=0.1,
            confidence=0.1,
        )
        shared_experience_id = UUID(shared["experience_id"])
        shared_version_id = UUID(shared["version_id"])
        dormant_experience_id = UUID(dormant["experience_id"])
        stages.append(
            {
                "name": _STAGE_NAMES[1],
                "result": {
                    "experiences": [
                        {
                            "content_hash": shared["content_hash"],
                            "experience_id": shared["experience_id"],
                            "initial_temperature": "hot",
                            "label": "shared_operation",
                            "version_id": shared["version_id"],
                        },
                        {
                            "content_hash": dormant["content_hash"],
                            "experience_id": dormant["experience_id"],
                            "initial_temperature": "warm",
                            "label": "dormant_unrelated",
                            "version_id": dormant["version_id"],
                        },
                    ]
                },
                "step": 2,
            }
        )

        topic = await _create_topic(
            container,
            owner_agent_id=alice_id,
        )
        topic_id = UUID(topic["topic_id"])
        subscription = await _create_subscription(
            container,
            subscriber_agent_id=bob_id,
            topic_id=topic_id,
        )
        stages.append(
            {
                "name": _STAGE_NAMES[2],
                "result": {
                    "subscription_id": subscription["subscription_id"],
                    "topic_id": topic["topic_id"],
                },
                "step": 3,
            }
        )

        capsule = await _publish_capsule(
            container,
            owner_agent_id=alice_id,
            topic_id=topic_id,
            experience_id=shared_experience_id,
            version_id=shared_version_id,
        )
        capsule_id = UUID(capsule["capsule_id"])
        expected_root = compute_original_root_fingerprint(
            root_publisher_id=alice_id,
            source_content_hash=shared["content_hash"],
        )
        stages.append(
            {
                "name": _STAGE_NAMES[3],
                "result": {
                    "capsule_hash": capsule["capsule_hash"],
                    "capsule_id": capsule["capsule_id"],
                    "provenance_chain": capsule["provenance_chain"],
                    "root_fingerprint": capsule["root_fingerprint"],
                    "source_content_hash": capsule["source_content_hash"],
                    "source_experience_id": capsule["source_experience_id"],
                    "source_version_id": capsule["source_version_id"],
                },
                "step": 4,
            }
        )

        async with container.database.read_session() as session:
            pending_page = await container.sharing_query.list_inbox(
                session=session,
                owner_agent_id=bob_id,
                state=InboxState.PENDING,
                limit=10,
                at=clock.now(),
            )
        if len(pending_page.items) != 1:
            raise RuntimeError("Bob must receive exactly one pending capsule")
        pending_item = pending_page.items[0]
        adoption, capsule_adoption_id = await _adopt_capsule(
            container,
            adopter_agent_id=bob_id,
            item_id=pending_item.item_id,
        )
        adopted_experience = cast(dict[str, Any], adoption["experience"])
        async with container.database.read_session() as session:
            parent_adoption = (
                await container.sharing_repository.get_owned_parent_adoption(
                    session=session,
                    adopter_agent_id=bob_id,
                    adoption_id=capsule_adoption_id,
                )
            )
            adopted_page = await container.sharing_query.list_inbox(
                session=session,
                owner_agent_id=bob_id,
                state=InboxState.ADOPTED,
                limit=10,
                at=clock.now(),
            )
        if parent_adoption is None:
            raise RuntimeError("Capsule adoption provenance is missing")
        if len(adopted_page.items) != 1:
            raise RuntimeError("Bob must have exactly one adopted inbox item")
        adopted_item = adopted_page.items[0]
        stages.append(
            {
                "name": _STAGE_NAMES[4],
                "result": {
                    "adoption_id": str(capsule_adoption_id),
                    "captured_provenance": [
                        {
                            "capsule_id": str(hop.capsule_id),
                            "publisher_agent_id": str(hop.publisher_agent_id),
                        }
                        for hop in parent_adoption.provenance_chain
                    ],
                    "created": adoption["created"],
                    "inbox_item_id": str(pending_item.item_id),
                    "inbox_state_after": adopted_item.state.value,
                    "inbox_state_before": pending_item.state.value,
                    "resulting_content_hash": adopted_experience[
                        "current_content_hash"
                    ],
                    "resulting_experience_id": adopted_experience["experience_id"],
                    "root_fingerprint": parent_adoption.root_fingerprint,
                },
                "step": 5,
            }
        )

        clock.advance(timedelta(days=2))
        first_cycle = await _run_lifecycle(
            container,
            key="demo-lifecycle-one",
        )
        clock.advance(
            container.lifecycle_config.minimum_cycle_interval + timedelta(seconds=1)
        )
        second_cycle = await _run_lifecycle(
            container,
            key="demo-lifecycle-two",
        )
        async with container.database.read_session() as session:
            dormant_after_lifecycle = (
                await container.experience_query.get_owned_retrieval_record(
                    session=session,
                    owner_agent_id=alice_id,
                    experience_id=dormant_experience_id,
                )
            )
        if dormant_after_lifecycle is None:
            raise RuntimeError("Dormant experience disappeared after lifecycle")
        stages.append(
            {
                "name": _STAGE_NAMES[5],
                "result": {
                    "cycles": [first_cycle, second_cycle],
                    "experience_id": str(dormant_experience_id),
                    "temperature_transition": [
                        "warm",
                        dormant_after_lifecycle.state.temperature.value,
                    ],
                },
                "step": 6,
            }
        )

        _, blurred_search = await _search(
            container,
            query=SearchExperiences(
                owner_agent_id=alice_id,
                query="租约断开 lease nebula",
                mode=RetrievalMode.FOCUSED,
            ),
            key="demo-blurred-search",
        )
        blurred_hits = cast(list[dict[str, Any]], blurred_search["hits"])
        if len(blurred_hits) != 1:
            raise RuntimeError("Weak cue must return exactly one cold memory")
        blurred_hit = _experience_hit(
            blurred_hits,
            experience_id=dormant_experience_id,
        )
        blurred_view = cast(dict[str, Any], blurred_hit["experience"])
        stages.append(
            {
                "name": _STAGE_NAMES[6],
                "result": {
                    "body_present": blurred_view["body"] is not None,
                    "blurred": blurred_view["blurred"],
                    "expanded": blurred_hit["expanded"],
                    "experience_id": blurred_view["experience_id"],
                    "reactivated": blurred_hit["reactivated"],
                    "returned_hit_count": len(blurred_hits),
                    "temperature": blurred_view["temperature"],
                },
                "step": 7,
            }
        )

        _, cue_search = await _search(
            container,
            query=SearchExperiences(
                owner_agent_id=alice_id,
                query="租约交接 lease handoff",
                mode=RetrievalMode.FOCUSED,
            ),
            key="demo-strong-cue-search",
        )
        cue_hits = cast(list[dict[str, Any]], cue_search["hits"])
        cue_hit = _experience_hit(
            cue_hits,
            experience_id=dormant_experience_id,
        )
        cue_view = cast(dict[str, Any], cue_hit["experience"])
        stages.append(
            {
                "name": _STAGE_NAMES[7],
                "result": {
                    "body": cue_view["body"],
                    "blurred": cue_view["blurred"],
                    "content_hash": cue_view["content_hash"],
                    "expanded": cue_hit["expanded"],
                    "experience_id": cue_view["experience_id"],
                    "reactivated": cue_hit["reactivated"],
                    "returned_hit_count": len(cue_hits),
                    "temperature": cue_view["temperature"],
                },
                "step": 8,
            }
        )

        run, idea = await _start_inspiration(
            container,
            owner_agent_id=bob_id,
        )
        idea_evidence = [
            {
                "source_key": reference.stable_evidence_key,
            }
            for reference in idea.draft.evidence
        ]
        stages.append(
            {
                "name": _STAGE_NAMES[8],
                "result": {
                    "evidence": idea_evidence,
                    "idea_content_hash": idea.idea_content_hash,
                    "idea_id": str(idea.idea_id),
                    "mechanism_hash": idea.mechanism_hash,
                    "operator": idea.operator.value,
                    "run_id": run["run_id"],
                    "run_status": run["status"],
                    "snapshot_hash": run["snapshot_hash"],
                },
                "step": 9,
            }
        )

        idea_as_experience = await container.retrieval_adapter.get(
            owner_agent_id=bob_id,
            experience_id=idea.idea_id,
            idempotency_key="demo-idea-as-experience-before-adoption",
        )
        idea_as_experience_error = _document(idea_as_experience)
        _, before_adoption_search = await _search(
            container,
            query=SearchExperiences(
                owner_agent_id=bob_id,
                query="counterfactual inspiration",
                mode=RetrievalMode.FOCUSED,
                tags=("inspiration",),
            ),
            key="demo-idea-search-before-adoption",
        )
        before_hits = cast(
            list[dict[str, Any]],
            before_adoption_search["hits"],
        )
        before_hypothesis_hits = [
            hit
            for hit in before_hits
            if (
                cast(dict[str, Any], hit["experience"])["origin"] == "adopted_idea"
                or cast(dict[str, Any], hit["experience"])["content_hash"]
                == idea.idea_content_hash
            )
        ]
        stages.append(
            {
                "name": _STAGE_NAMES[9],
                "result": {
                    "experience_get_error": cast(
                        dict[str, Any],
                        idea_as_experience_error["error"],
                    )["code"],
                    "experience_get_status": idea_as_experience.status_code,
                    "hypothesis_hits": len(before_hypothesis_hits),
                    "idea_id": str(idea.idea_id),
                    "source_evidence_hits": len(before_hits),
                },
                "step": 10,
            }
        )

        idea_adoption = await _adopt_idea(
            container,
            owner_agent_id=bob_id,
            idea_id=idea.idea_id,
        )
        hypothesis = cast(dict[str, Any], idea_adoption["experience"])
        hypothesis_id = UUID(hypothesis["experience_id"])
        retrieved_result = await container.retrieval_adapter.get(
            owner_agent_id=bob_id,
            experience_id=hypothesis_id,
            idempotency_key="demo-hypothesis-get",
        )
        retrieved = _success_data(retrieved_result, expected_status=200)
        _, after_adoption_search = await _search(
            container,
            query=SearchExperiences(
                owner_agent_id=bob_id,
                query="counterfactual inspiration",
                mode=RetrievalMode.FOCUSED,
                tags=("inspiration",),
            ),
            key="demo-idea-search-after-adoption",
        )
        after_hits = cast(
            list[dict[str, Any]],
            after_adoption_search["hits"],
        )
        async with container.database.read_session() as session:
            adopted_ideas = await container.inspiration_repository.list_owned_ideas(
                session=session,
                owner_agent_id=bob_id,
                run_id=idea.run_id,
                after=None,
                limit=10,
            )
        if len(adopted_ideas) != 1:
            raise RuntimeError("Demo idea set changed after explicit adoption")
        adopted_idea = adopted_ideas[0]
        stages.append(
            {
                "name": _STAGE_NAMES[10],
                "result": {
                    "content_hash": hypothesis["current_content_hash"],
                    "idea_id": str(idea.idea_id),
                    "idea_state_transition": [
                        IdeaOwnerDecision.ACTIVE.value,
                        adopted_idea.owner_decision.value,
                    ],
                    "kind": retrieved["kind"],
                    "origin": retrieved["origin"],
                    "resulting_experience_id": hypothesis["experience_id"],
                    "resulting_version_id": hypothesis["current_version_id"],
                    "retrieved_body_present": retrieved["body"] is not None,
                    "retrieved_summary": retrieved["summary"],
                    "search_hit_ids": [
                        cast(dict[str, Any], hit["experience"])["experience_id"]
                        for hit in after_hits
                    ],
                },
                "step": 11,
            }
        )

        projection_report = await container.projection_manager.verify(
            container.database
        )
        expected_evidence_key = stable_evidence_key(
            source_type=EvidenceSourceType.EXPERIENCE,
            source_id=UUID(adopted_experience["experience_id"]),
            source_version_id=UUID(adopted_experience["current_version_id"]),
            content_hash=adopted_experience["current_content_hash"],
        )
        expected_hypothesis_content = VersionContent(
            body=canonical_json_bytes(
                {
                    "assumptions": idea.draft.assumptions,
                    "hypothesis": idea.draft.hypothesis,
                    "predictions": idea.draft.predictions,
                    "proposed_test": idea.draft.proposed_test,
                }
            ).decode("utf-8"),
            summary=idea.draft.title,
            mechanism=idea.draft.mechanism,
            tags=(
                "inspiration",
                f"operator:{idea.operator.value}",
            ),
            applicability=idea.draft.assumptions,
            evidence=tuple(
                TypedEvidence(
                    type="inspiration_evidence",
                    id=reference.stable_evidence_key,
                )
                for reference in idea.draft.evidence
            ),
            falsifiers=idea.draft.falsifiers,
        )
        invariants = {
            "adopted_idea_became_hypothesis": (
                retrieved["kind"] == ExperienceKind.HYPOTHESIS.value
                and retrieved["origin"] == "adopted_idea"
                and adopted_idea.owner_decision is IdeaOwnerDecision.ADOPTED
            ),
            "capsule_content_hash_preserved": (
                capsule["source_content_hash"]
                == shared["content_hash"]
                == adopted_experience["current_content_hash"]
            ),
            "capsule_required_explicit_adoption": (
                pending_item.state is InboxState.PENDING
                and adopted_item.state is InboxState.ADOPTED
                and adoption["created"] is True
            ),
            "cold_memory_content_hash_preserved": (
                cue_view["content_hash"] == dormant["content_hash"]
            ),
            "generated_idea_absent_before_adoption": (
                idea_as_experience.status_code == 404
                and cast(
                    dict[str, Any],
                    idea_as_experience_error["error"],
                )["code"]
                == "experience_not_found"
                and before_hypothesis_hits == []
            ),
            "hypothesis_retrieved_after_adoption": (
                retrieved["experience_id"] == hypothesis["experience_id"]
                and retrieved["summary"] == idea.draft.title
                and hypothesis["experience_id"]
                in {
                    cast(dict[str, Any], hit["experience"])["experience_id"]
                    for hit in after_hits
                }
            ),
            "hypothesis_mapping_preserved": (
                retrieved["body"] == expected_hypothesis_content.body
                and retrieved["summary"] == expected_hypothesis_content.summary
                and retrieved["mechanism"] == expected_hypothesis_content.mechanism
                and retrieved["tags"] == list(expected_hypothesis_content.tags)
                and retrieved["applicability"]
                == list(expected_hypothesis_content.applicability)
                and retrieved["evidence"]
                == [
                    value.model_dump(mode="json")
                    for value in expected_hypothesis_content.evidence
                ]
                and retrieved["falsifiers"]
                == list(expected_hypothesis_content.falsifiers)
            ),
            "inspiration_used_adopted_shared_experience": (
                len(idea.draft.evidence) == 1
                and idea.draft.evidence[0].stable_evidence_key == expected_evidence_key
            ),
            "ordinary_cold_query_was_blurred": (
                blurred_view["experience_id"] == dormant["experience_id"]
                and blurred_view["temperature"] == "cold"
                and blurred_view["blurred"] is True
                and blurred_view["body"] is None
                and blurred_hit["expanded"] is False
                and blurred_hit["reactivated"] is False
            ),
            "provenance_chain_preserved": (
                parent_adoption.provenance_chain
                == (
                    pending_item.capsule.provenance_chain
                    + (
                        ProvenanceHop(
                            capsule_id=capsule_id,
                            publisher_agent_id=alice_id,
                        ),
                    )
                )
            ),
            "projections_match": (
                projection_report.matches and projection_report.differences == ()
            ),
            "root_fingerprint_preserved": (
                expected_root
                == capsule["root_fingerprint"]
                == parent_adoption.root_fingerprint
            ),
            "strong_cue_reactivated_memory": (
                cue_view["experience_id"] == dormant["experience_id"]
                and cue_view["temperature"] == "warm"
                and cue_view["blurred"] is False
                and cue_view["body"] == dormant_content.body
                and cue_hit["expanded"] is True
                and cue_hit["reactivated"] is True
            ),
        }
        if not all(invariants.values()):
            failed = sorted(
                key for key, value in invariants.items() if value is not True
            )
            raise RuntimeError(f"Demo invariants failed: {failed}")

        return {
            "data": {
                "all_invariants_hold": True,
                "database_path": str(database_path),
                "invariants": invariants,
                "stages": stages,
            }
        }


__all__ = ["build_demo_report"]
