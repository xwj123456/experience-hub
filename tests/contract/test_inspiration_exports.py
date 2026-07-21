from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime
from importlib import import_module
from typing import Any, get_type_hints
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain import CommandRequest, PendingEvent
from experience_hub.inspiration.commands import (
    AdoptIdea,
    ArchiveIdea,
    RejectIdea,
    StartInspirationRun,
)
from experience_hub.inspiration.contracts import (
    ExperienceEvidenceReader,
    InboxEvidenceReader,
)
from experience_hub.inspiration.deadlines import (
    AsyncioDeadlineRunner,
    BoundedGenerationRunner,
    DeadlineExpired,
    DeadlineLimit,
    DeadlineRunner,
    MonotonicClock,
    OperatorGeneration,
    OperatorGenerationRun,
    SystemMonotonicClock,
)
from experience_hub.inspiration.generators import (
    DeterministicIdeaGenerator,
    GeneratorNotConfiguredError,
    GeneratorResult,
    IdeaGenerator,
    ManagedIdeaGenerator,
    OpenAICompatibleIdeaGenerator,
    build_idea_generator,
)
from experience_hub.inspiration.lifecycle import (
    IdeaLifecycleService,
    InspirationIdeaArchivePlanner,
)
from experience_hub.inspiration.models import (
    FrozenSnapshot,
    Idea,
    IdeaEvaluation,
    InspirationOperator,
    InspirationRun,
    OperatorOutcome,
    SnapshotItem,
)
from experience_hub.inspiration.response_codec import (
    InspirationErrorResponseV1,
    InspirationResponseCodec,
    InspirationRunResponseV1,
)
from experience_hub.inspiration.service import (
    GenerationRunner,
    GeneratorFactory,
    InspirationRunExecutor,
)
from experience_hub.inspiration.snapshot import SnapshotBuilder
from experience_hub.retrieval.contracts import (
    PeekExperiences,
    SearchResult,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue
from experience_hub.sharing.queries import QuarantinedCapsuleEvidence
from experience_hub.storage import StoredResponse, UnitOfWork

EXPECTED_EXPORTS: dict[str, object] = {
    "AdoptIdea": AdoptIdea,
    "ArchiveIdea": ArchiveIdea,
    "AsyncioDeadlineRunner": AsyncioDeadlineRunner,
    "BoundedGenerationRunner": BoundedGenerationRunner,
    "DeadlineExpired": DeadlineExpired,
    "DeadlineLimit": DeadlineLimit,
    "DeadlineRunner": DeadlineRunner,
    "DeterministicIdeaGenerator": DeterministicIdeaGenerator,
    "ExperienceEvidenceReader": ExperienceEvidenceReader,
    "FrozenSnapshot": FrozenSnapshot,
    "GenerationRunner": GenerationRunner,
    "GeneratorFactory": GeneratorFactory,
    "GeneratorNotConfiguredError": GeneratorNotConfiguredError,
    "GeneratorResult": GeneratorResult,
    "Idea": Idea,
    "IdeaEvaluation": IdeaEvaluation,
    "IdeaGenerator": IdeaGenerator,
    "IdeaLifecycleService": IdeaLifecycleService,
    "InboxEvidenceReader": InboxEvidenceReader,
    "InspirationErrorResponseV1": InspirationErrorResponseV1,
    "InspirationIdeaArchivePlanner": InspirationIdeaArchivePlanner,
    "InspirationResponseCodec": InspirationResponseCodec,
    "InspirationRun": InspirationRun,
    "InspirationRunExecutor": InspirationRunExecutor,
    "InspirationRunResponseV1": InspirationRunResponseV1,
    "ManagedIdeaGenerator": ManagedIdeaGenerator,
    "MonotonicClock": MonotonicClock,
    "OpenAICompatibleIdeaGenerator": OpenAICompatibleIdeaGenerator,
    "OperatorGeneration": OperatorGeneration,
    "OperatorGenerationRun": OperatorGenerationRun,
    "OperatorOutcome": OperatorOutcome,
    "RejectIdea": RejectIdea,
    "SnapshotBuilder": SnapshotBuilder,
    "StartInspirationRun": StartInspirationRun,
    "SystemMonotonicClock": SystemMonotonicClock,
    "build_idea_generator": build_idea_generator,
}

COMMAND_FIELDS = {
    StartInspirationRun: (
        "owner_agent_id",
        "goal",
        "context",
        "mode",
        "generator",
        "operators",
        "include_inbox",
        "branches_per_operator",
        "output_tokens_per_operator",
        "total_output_tokens",
        "operator_timeout_seconds",
        "global_timeout_seconds",
    ),
    RejectIdea: ("owner_agent_id", "idea_id", "reason"),
    ArchiveIdea: ("owner_agent_id", "idea_id", "reason"),
    AdoptIdea: ("owner_agent_id", "idea_id", "importance", "confidence"),
}

QUERY_MODEL_FIELDS = {
    InspirationRun: (
        "run_id",
        "owner_agent_id",
        "goal",
        "context",
        "mode",
        "generator",
        "operators",
        "include_inbox",
        "branches_per_operator",
        "output_tokens_per_operator",
        "total_output_tokens",
        "operator_timeout_seconds",
        "global_timeout_seconds",
        "request_hash",
        "snapshot_hash",
        "status",
        "operator_outcomes",
        "output_tokens_reserved",
        "output_tokens_consumed",
        "elapsed_milliseconds",
        "created_at",
        "completed_at",
    ),
    Idea: (
        "idea_id",
        "run_id",
        "owner_agent_id",
        "operator",
        "ordinal",
        "draft",
        "idea_content_hash",
        "mechanism_hash",
        "duplicate_relation",
        "owner_decision",
        "mechanism_cluster_id",
        "maturity",
        "last_signal_at",
        "resulting_experience_id",
        "resulting_version_id",
    ),
    IdeaEvaluation: (
        "evaluator_agent_id",
        "idea_id",
        "verdict",
        "reason",
        "evidence",
        "evaluated_at",
    ),
}


def _assert_keyword_only_after_self(method: Any) -> None:
    signature = inspect.signature(method)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )


def test_inspiration_package_reexports_the_stable_public_contract() -> None:
    package = import_module("experience_hub.inspiration")
    declared = vars(package).get("__all__")

    assert isinstance(declared, list)
    assert all(isinstance(name, str) for name in declared)
    assert len(declared) == len(set(declared))
    assert EXPECTED_EXPORTS.keys() <= set(declared)

    namespace: dict[str, object] = {}
    exec("from experience_hub.inspiration import *", namespace)
    for name in declared:
        assert namespace[name] is getattr(package, name)
    for name, symbol in EXPECTED_EXPORTS.items():
        assert getattr(package, name) is symbol
        assert namespace[name] is symbol


@pytest.mark.parametrize(
    "module_order",
    (
        (
            "experience_hub.inspiration",
            "experience_hub.inspiration.generators",
            "experience_hub.inspiration.service",
        ),
        (
            "experience_hub.inspiration.generators.openai_compatible",
            "experience_hub.inspiration.response_codec",
            "experience_hub.inspiration",
        ),
        (
            "experience_hub.inspiration.contracts",
            "experience_hub.inspiration.lifecycle",
            "experience_hub.inspiration.deadlines",
            "experience_hub.inspiration",
        ),
        (
            "experience_hub.storage",
            "experience_hub.inspiration",
        ),
        (
            "experience_hub.inspiration",
            "experience_hub.storage",
        ),
        (
            "experience_hub.storage.validation",
            "experience_hub.inspiration",
        ),
        (
            "experience_hub.inspiration",
            "experience_hub.storage.validation",
        ),
    ),
)
def test_inspiration_contract_imports_in_any_order_in_a_fresh_interpreter(
    module_order: tuple[str, ...],
) -> None:
    script = "\n".join(
        (
            "from importlib import import_module",
            *(f"import_module({module_name!r})" for module_name in module_order),
            "from experience_hub.inspiration import ("
            "ExperienceEvidenceReader, IdeaGenerator, "
            "InspirationResponseCodec, InspirationRunExecutor, "
            "SnapshotBuilder)",
            "from experience_hub.inspiration import *",
            "package = import_module('experience_hub.inspiration')",
            "assert package.ExperienceEvidenceReader is "
            "import_module('experience_hub.inspiration.contracts')."
            "ExperienceEvidenceReader",
            "assert package.IdeaGenerator is "
            "import_module('experience_hub.inspiration.generators.base')."
            "IdeaGenerator",
            "assert package.InspirationResponseCodec is "
            "import_module('experience_hub.inspiration.response_codec')."
            "InspirationResponseCodec",
            "assert package.InspirationRunExecutor is "
            "import_module('experience_hub.inspiration.service')."
            "InspirationRunExecutor",
            "assert package.SnapshotBuilder is "
            "import_module('experience_hub.inspiration.snapshot')."
            "SnapshotBuilder",
            "assert package.OperatorOutcome is "
            "import_module('experience_hub.inspiration.models')."
            "OperatorOutcome",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("command_type", "expected_fields"),
    COMMAND_FIELDS.items(),
)
def test_inspiration_run_and_decision_command_fields_are_frozen(
    command_type: Any,
    expected_fields: tuple[str, ...],
) -> None:
    assert tuple(field.name for field in fields(command_type)) == expected_fields
    assert tuple(inspect.signature(command_type).parameters) == expected_fields


@pytest.mark.parametrize(
    ("model_type", "expected_fields"),
    QUERY_MODEL_FIELDS.items(),
)
def test_inspiration_query_model_fields_are_frozen(
    model_type: Any,
    expected_fields: tuple[str, ...],
) -> None:
    assert tuple(model_type.model_fields) == expected_fields


def test_generator_and_snapshot_keep_their_bounded_contracts() -> None:
    generator = IdeaGenerator.generate
    assert inspect.iscoroutinefunction(generator)
    _assert_keyword_only_after_self(generator)
    assert tuple(inspect.signature(generator).parameters) == (
        "self",
        "goal",
        "context",
        "frozen_items",
        "operator",
        "branch_limit",
        "output_token_limit",
    )
    assert get_type_hints(generator) == {
        "goal": str,
        "context": str,
        "frozen_items": tuple[SnapshotItem, ...],
        "operator": InspirationOperator,
        "branch_limit": int,
        "output_token_limit": int,
        "return": GeneratorResult,
    }

    freeze = SnapshotBuilder.freeze
    assert inspect.iscoroutinefunction(freeze)
    _assert_keyword_only_after_self(freeze)
    assert tuple(inspect.signature(freeze).parameters) == (
        "self",
        "uow",
        "request",
        "run_id",
        "at",
    )
    assert get_type_hints(freeze) == {
        "uow": UnitOfWork,
        "request": StartInspirationRun,
        "run_id": UUID,
        "at": datetime,
        "return": FrozenSnapshot,
    }


def test_run_executor_and_response_codec_keep_verbatim_response_boundary() -> None:
    execute = InspirationRunExecutor.execute
    assert inspect.iscoroutinefunction(execute)
    _assert_keyword_only_after_self(execute)
    assert tuple(inspect.signature(execute).parameters) == (
        "self",
        "request",
        "run",
    )
    assert get_type_hints(execute) == {
        "request": CommandRequest,
        "run": StartInspirationRun,
        "return": StoredResponse,
    }

    terminal = InspirationResponseCodec.terminal
    assert tuple(inspect.signature(terminal).parameters) == ("run",)
    assert get_type_hints(terminal) == {
        "run": InspirationRun,
        "return": StoredResponse,
    }
    in_progress = InspirationResponseCodec.in_progress
    assert tuple(inspect.signature(in_progress).parameters) == (
        "receipt_id",
        "run_id",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(in_progress).parameters.values()
    )
    assert get_type_hints(in_progress) == {
        "receipt_id": UUID,
        "run_id": UUID,
        "return": StoredResponse,
    }


def test_monotonic_generation_runner_keeps_elapsed_time_separate() -> None:
    method = BoundedGenerationRunner.run
    assert inspect.iscoroutinefunction(method)
    _assert_keyword_only_after_self(method)
    assert tuple(inspect.signature(method).parameters) == (
        "self",
        "generator",
        "goal",
        "context",
        "frozen_items",
        "operators",
        "branch_limit",
        "output_tokens_per_operator",
        "total_output_tokens",
        "operator_timeout_seconds",
        "global_timeout_seconds",
    )
    assert get_type_hints(method) == {
        "generator": IdeaGenerator,
        "goal": str,
        "context": str,
        "frozen_items": tuple[SnapshotItem, ...],
        "operators": tuple[InspirationOperator, ...],
        "branch_limit": int,
        "output_tokens_per_operator": int,
        "total_output_tokens": int,
        "operator_timeout_seconds": int,
        "global_timeout_seconds": int,
        "return": OperatorGenerationRun,
    }
    assert get_type_hints(MonotonicClock.now) == {"return": float}


def test_archive_planner_keeps_its_session_bound_extension_contract() -> None:
    method = InspirationIdeaArchivePlanner.due_archive_events
    assert inspect.iscoroutinefunction(method)
    _assert_keyword_only_after_self(method)
    assert tuple(inspect.signature(method).parameters) == (
        "self",
        "session",
        "evaluated_at",
        "cycle_id",
    )
    assert get_type_hints(method) == {
        "session": AsyncSession,
        "evaluated_at": datetime,
        "cycle_id": UUID,
        "return": tuple[PendingEvent, ...],
    }


def test_owned_evidence_reader_is_read_only_and_session_bound() -> None:
    method = ExperienceEvidenceReader.peek
    assert inspect.iscoroutinefunction(method)
    _assert_keyword_only_after_self(method)
    assert tuple(inspect.signature(method).parameters) == (
        "self",
        "session",
        "query",
    )
    assert get_type_hints(method) == {
        "session": AsyncSession,
        "query": PeekExperiences,
        "return": SearchResult,
    }


def test_pending_evidence_reader_is_read_only_and_session_bound() -> None:
    method = InboxEvidenceReader.list_available_pending
    assert inspect.iscoroutinefunction(method)
    _assert_keyword_only_after_self(method)
    assert tuple(inspect.signature(method).parameters) == (
        "self",
        "session",
        "recipient_agent_id",
        "as_of",
        "query_cues",
        "mode",
        "limit",
    )
    assert get_type_hints(method) == {
        "session": AsyncSession,
        "recipient_agent_id": UUID,
        "as_of": datetime,
        "query_cues": Iterable[TermCue],
        "mode": RetrievalMode,
        "limit": int,
        "return": tuple[QuarantinedCapsuleEvidence, ...],
    }
