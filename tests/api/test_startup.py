from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from experience_hub.agents import CreateAgent
from experience_hub.api.app import create_app
from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.models import ExperienceKind, VersionContent
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.models import (
    GeneratorKind,
    InspirationOperator,
    SnapshotItem,
)
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.idempotency import StoredResponse
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import SourceIntegrityError

NOW = datetime(2026, 7, 19, 8, 30, tzinfo=UTC)
ORPHANED_CALLER_ID = UUID("00000000-0000-0000-0000-000000009999")


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


def _agent_request(*, key: str, name: str) -> CommandRequest:
    return CommandRequest(
        caller_scope="system:local",
        operation_scope="agent.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents",
        body={"name": name},
    )


def _run_request(
    *,
    agent_id: UUID,
    key: str,
    run: StartInspirationRun,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope="inspiration.run.start",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/inspiration-runs",
        path_parameters={"agent_id": agent_id},
        body={
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
        },
    )


def _experience_request(
    *,
    agent_id: UUID,
    key: str,
    content: VersionContent,
) -> CommandRequest:
    return CommandRequest(
        caller_scope=f"agent:{agent_id}",
        operation_scope="experience.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": agent_id},
        body={
            "kind": ExperienceKind.SEMANTIC.value,
            "content": content.model_dump(mode="json"),
            "importance": 0.6,
            "confidence": 0.7,
            "links": (),
        },
    )


@dataclass(slots=True)
class _GeneratorCallLedger:
    calls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _CancellingGenerator:
    ledger: _GeneratorCallLedger
    label: str

    @property
    def reserves_output_tokens(self) -> bool:
        return False

    @property
    def persisted_configuration(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        return None

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult:
        _ = (
            goal,
            context,
            frozen_items,
            operator,
            branch_limit,
            output_token_limit,
        )
        self.ledger.calls.append(self.label)
        raise asyncio.CancelledError


class _CancelledSnapshotBuilder:
    async def freeze(self, **_: Any) -> None:
        raise asyncio.CancelledError


async def _create_agent(
    container: ApplicationContainer,
    *,
    key: str,
    name: str,
) -> UUID:
    async def handler(
        uow: UnitOfWork,
        command: CommandContext,
    ) -> StoredResponse:
        return await container.agent_service.create(
            uow=uow,
            command=CreateAgent(name=name),
            command_context=command,
        )

    response = await container.command_executor.execute(
        _agent_request(key=key, name=name),
        handler,
    )
    assert response.status_code == 201
    return UUID(json.loads(response.body)["data"]["agent_id"])


async def _create_recovery_evidence(
    container: ApplicationContainer,
    *,
    agent_id: UUID,
) -> None:
    content = VersionContent(
        body="An interrupted inspiration process must resume from its ledger.",
        summary="Recover interrupted inspiration from durable state.",
        mechanism="The startup recovery ledger closes interrupted runs.",
        tags=("inspiration", "recovery"),
        applicability=("process interruption",),
        evidence=(),
        falsifiers=("Recovery starts the generator again.",),
    )
    command = CreateExperience(
        owner_agent_id=agent_id,
        kind=ExperienceKind.SEMANTIC,
        content=content,
        importance=0.6,
        confidence=0.7,
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

    response = await container.command_executor.execute(
        _experience_request(
            agent_id=agent_id,
            key="startup-evidence",
            content=content,
        ),
        handler,
    )
    assert response.status_code == 201


async def _seed_interrupted_runs(
    database_path: Path,
    *,
    include_snapshot_frozen: bool,
) -> tuple[_GeneratorCallLedger, tuple[str, ...]]:
    """Create legal retained traces, then close every first-process resource."""
    settings = _settings(database_path)
    ledger = _GeneratorCallLedger()
    runtime = ApplicationRuntime(
        settings=settings,
        clock=FrozenClock(NOW),
    )
    keys = ["started-only"]
    if include_snapshot_frozen:
        keys.append("snapshot-frozen")

    async with runtime.initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ) as container:
        agent_id = await _create_agent(
            container,
            key="startup-agent",
            name=f"Startup owner for {database_path.stem}",
        )
        await _create_recovery_evidence(container, agent_id=agent_id)
        run = StartInspirationRun(
            owner_agent_id=agent_id,
            goal="Recover an interrupted inspiration run",
            generator=GeneratorKind.DETERMINISTIC,
            operators=(InspirationOperator.CAUSAL_GAP,),
        )

        container.inspiration_run_executor._snapshot_builder = (  # noqa: SLF001
            _CancelledSnapshotBuilder()
        )
        container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
            lambda _: _CancellingGenerator(ledger, "started-only")
        )
        with pytest.raises(asyncio.CancelledError):
            await container.inspiration_run_executor.execute(
                request=_run_request(
                    agent_id=agent_id,
                    key="started-only",
                    run=run,
                ),
                run=run,
            )

        if include_snapshot_frozen:
            container.inspiration_run_executor._snapshot_builder = (  # noqa: SLF001
                container.snapshot_builder
            )
            container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
                lambda _: _CancellingGenerator(ledger, "snapshot-frozen")
            )
            with pytest.raises(asyncio.CancelledError):
                await container.inspiration_run_executor.execute(
                    request=_run_request(
                        agent_id=agent_id,
                        key="snapshot-frozen",
                        run=run,
                    ),
                    run=run,
                )

    return ledger, tuple(keys)


def _full_container_factory_with_generator_spy(
    startup_generator_calls: list[GeneratorKind],
) -> Callable[..., ApplicationContainer]:
    def factory(**kwargs: Any) -> ApplicationContainer:
        container = ApplicationContainer.build(**kwargs)

        def generator_factory(kind: GeneratorKind) -> _CancellingGenerator:
            startup_generator_calls.append(kind)
            return _CancellingGenerator(_GeneratorCallLedger(), "startup")

        container.inspiration_run_executor._generator_factory = (  # noqa: SLF001
            generator_factory
        )
        return container

    return factory


def _inspiration_counts(database_path: Path) -> tuple[int, int]:
    with sqlite3.connect(database_path) as connection:
        failed_events = int(
            connection.execute(
                "SELECT count(*) FROM domain_events "
                "WHERE event_type = 'inspiration.failed'"
            ).fetchone()[0]
        )
        recovery_receipts = int(
            connection.execute(
                "SELECT count(*) FROM idempotency_records "
                "WHERE scope = 'inspiration.run.recover'"
            ).fetchone()[0]
        )
    return failed_events, recovery_receipts


def _assert_recovered_ledger(
    database_path: Path,
    *,
    original_keys: tuple[str, ...],
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        originals = connection.execute(
            "SELECT receipt_id, idempotency_key, result_resource_id, state, "
            "response_status_code, response_body "
            "FROM idempotency_records "
            "WHERE scope = 'inspiration.run.start' "
            "ORDER BY idempotency_key"
        ).fetchall()
        recoveries = connection.execute(
            "SELECT result_resource_id, state, response_status_code, response_body "
            "FROM idempotency_records "
            "WHERE scope = 'inspiration.run.recover' "
            "ORDER BY result_resource_id"
        ).fetchall()
        failures = connection.execute(
            "SELECT aggregate_id, causation_id, payload "
            "FROM domain_events "
            "WHERE event_type = 'inspiration.failed' "
            "ORDER BY aggregate_id"
        ).fetchall()

    assert {row["idempotency_key"] for row in originals} == set(original_keys)
    assert len(originals) == len(recoveries) == len(failures) == len(original_keys)
    recovery_by_run = {row["result_resource_id"]: row for row in recoveries}
    failure_by_run = {row["aggregate_id"]: row for row in failures}
    for original in originals:
        run_id = original["result_resource_id"]
        recovery = recovery_by_run[run_id]
        failure = failure_by_run[run_id]
        assert original["state"] == recovery["state"] == "completed"
        assert original["response_status_code"] == 201
        assert recovery["response_status_code"] == 201
        assert original["response_body"] == recovery["response_body"]
        assert json.loads(original["response_body"])["data"]["status"] == "failed"
        assert json.loads(failure["payload"])["failure_code"] == ("process_interrupted")
        assert failure["causation_id"] != original["receipt_id"]


def test_lifespan_recovers_both_legal_interrupted_phases_before_ready(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-recovery.sqlite3"
    seed_ledger, original_keys = asyncio.run(
        _seed_interrupted_runs(
            database_path,
            include_snapshot_frozen=True,
        )
    )
    assert seed_ledger.calls == ["snapshot-frozen"]

    startup_generator_calls: list[GeneratorKind] = []
    runtime = ApplicationRuntime(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
        container_factory=_full_container_factory_with_generator_spy(
            startup_generator_calls
        ),
    )
    app = create_app(runtime=runtime)

    assert app.state.ready is False
    with TestClient(app) as client:
        assert app.state.ready is True
        assert client.get("/health").status_code == 200
        _assert_recovered_ledger(
            database_path,
            original_keys=original_keys,
        )
        assert startup_generator_calls == []
    assert app.state.ready is False

    recovered_counts = _inspiration_counts(database_path)
    assert recovered_counts == (2, 2)
    with TestClient(app) as client:
        assert app.state.ready is True
        assert client.get("/health").status_code == 200
        assert _inspiration_counts(database_path) == recovered_counts
        assert startup_generator_calls == []


def _startup_observations(database_path: Path) -> tuple[Any, ...]:
    with sqlite3.connect(database_path) as connection:
        events = connection.execute(
            "SELECT * FROM domain_events ORDER BY event_id"
        ).fetchall()
        run_projection = connection.execute(
            "SELECT * FROM inspiration_run_state ORDER BY run_id"
        ).fetchall()
        projection_versions = connection.execute(
            "SELECT * FROM projection_versions ORDER BY name"
        ).fetchall()
        receipt_count = connection.execute(
            "SELECT count(*) FROM idempotency_records"
        ).fetchone()[0]
    return events, run_projection, projection_versions, receipt_count


def test_lifespan_refuses_running_trace_with_forged_original_receipt_atomically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-orphan.sqlite3"
    asyncio.run(
        _seed_interrupted_runs(
            database_path,
            include_snapshot_frozen=False,
        )
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE idempotency_records SET caller_scope = ? "
            "WHERE scope = 'inspiration.run.start'",
            (f"agent:{ORPHANED_CALLER_ID}",),
        )

    before = _startup_observations(database_path)
    startup_generator_calls: list[GeneratorKind] = []
    runtime = ApplicationRuntime(
        settings=_settings(database_path),
        clock=FrozenClock(NOW),
        container_factory=_full_container_factory_with_generator_spy(
            startup_generator_calls
        ),
    )
    app = create_app(runtime=runtime)

    with pytest.raises(SourceIntegrityError), TestClient(app):
        pytest.fail("an orphaned running trace must not reach readiness")

    assert app.state.ready is False
    assert app.state.container is None
    assert _startup_observations(database_path) == before
    assert _inspiration_counts(database_path) == (0, 0)
    assert startup_generator_calls == []
