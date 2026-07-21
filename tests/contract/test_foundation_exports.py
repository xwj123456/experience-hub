import subprocess
import sys
from importlib import import_module

import pytest

from experience_hub.agents.events import AgentCreated, register_agent_events
from experience_hub.agents.models import CreateAgent
from experience_hub.agents.service import AgentService
from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.clock import Clock, FrozenClock, SystemClock, require_utc
from experience_hub.domain.commands import (
    CommandContext,
    CommandRequest,
    ReplayableCommandError,
)
from experience_hub.domain.events import (
    EventPayload,
    EventRegistry,
    PendingEvent,
    StoredEvent,
)
from experience_hub.domain.values import StrictModel, StructuredReason, TypedEvidence
from experience_hub.errors import CallerScope, CanonicalizationError, DomainError
from experience_hub.ids import IdGenerator, SequenceIdGenerator, Uuid4Generator
from experience_hub.storage.database import Database, DatabaseBusy
from experience_hub.storage.event_store import EventStore
from experience_hub.storage.idempotency import (
    CommandExecutor,
    CommandHandler,
    CommandResult,
    CompletedReceipt,
    IdempotencyIntegrityError,
    IdempotencyKeyConflict,
    InProgressReceipt,
    NewReceipt,
    ReceiptDecision,
    ReceiptRecord,
    ReceiptReservation,
    ReceiptStore,
    StoredResponse,
)
from experience_hub.storage.projection_contracts import (
    ProjectionApplier,
    ProjectionReducer,
)
from experience_hub.storage.projections import (
    EventHeadChanged,
    MaintenanceBlockedByInflight,
    ProjectionDiff,
    ProjectionManager,
    ProjectionMismatch,
    ProjectionRegistry,
    ReducerVersionMismatch,
    SourceValidatorRequired,
    VerificationReport,
)
from experience_hub.storage.unit_of_work import UnitOfWork
from experience_hub.storage.validation import (
    AgentSourceValidator,
    SourceIntegrityError,
    SourceValidationHook,
    SourceValidator,
    register_agent_source_validator,
)

EXPECTED_EXPORTS: dict[str, dict[str, object]] = {
    "experience_hub": {
        "canonical_json_bytes": canonical_json_bytes,
        "sha256_hex": sha256_hex,
        "Clock": Clock,
        "SystemClock": SystemClock,
        "FrozenClock": FrozenClock,
        "require_utc": require_utc,
        "IdGenerator": IdGenerator,
        "Uuid4Generator": Uuid4Generator,
        "SequenceIdGenerator": SequenceIdGenerator,
        "DomainError": DomainError,
        "CanonicalizationError": CanonicalizationError,
        "CallerScope": CallerScope,
    },
    "experience_hub.domain": {
        "StrictModel": StrictModel,
        "TypedEvidence": TypedEvidence,
        "StructuredReason": StructuredReason,
        "CommandRequest": CommandRequest,
        "CommandContext": CommandContext,
        "ReplayableCommandError": ReplayableCommandError,
        "EventPayload": EventPayload,
        "EventRegistry": EventRegistry,
        "PendingEvent": PendingEvent,
        "StoredEvent": StoredEvent,
    },
    "experience_hub.storage": {
        "Database": Database,
        "DatabaseBusy": DatabaseBusy,
        "UnitOfWork": UnitOfWork,
        "EventStore": EventStore,
        "ProjectionApplier": ProjectionApplier,
        "ProjectionReducer": ProjectionReducer,
        "CommandExecutor": CommandExecutor,
        "CommandResult": CommandResult,
        "CommandHandler": CommandHandler,
        "ReceiptStore": ReceiptStore,
        "ReceiptReservation": ReceiptReservation,
        "ReceiptRecord": ReceiptRecord,
        "StoredResponse": StoredResponse,
        "NewReceipt": NewReceipt,
        "CompletedReceipt": CompletedReceipt,
        "InProgressReceipt": InProgressReceipt,
        "ReceiptDecision": ReceiptDecision,
        "IdempotencyKeyConflict": IdempotencyKeyConflict,
        "IdempotencyIntegrityError": IdempotencyIntegrityError,
        "ProjectionRegistry": ProjectionRegistry,
        "ProjectionManager": ProjectionManager,
        "ProjectionDiff": ProjectionDiff,
        "VerificationReport": VerificationReport,
        "ProjectionMismatch": ProjectionMismatch,
        "ReducerVersionMismatch": ReducerVersionMismatch,
        "EventHeadChanged": EventHeadChanged,
        "MaintenanceBlockedByInflight": MaintenanceBlockedByInflight,
        "SourceValidatorRequired": SourceValidatorRequired,
        "SourceValidator": SourceValidator,
        "SourceValidationHook": SourceValidationHook,
        "SourceIntegrityError": SourceIntegrityError,
        "AgentSourceValidator": AgentSourceValidator,
        "register_agent_source_validator": register_agent_source_validator,
    },
    "experience_hub.agents": {
        "AgentCreated": AgentCreated,
        "AgentService": AgentService,
        "CreateAgent": CreateAgent,
        "register_agent_events": register_agent_events,
    },
}


@pytest.mark.parametrize(
    ("module_name", "expected"),
    EXPECTED_EXPORTS.items(),
)
def test_foundation_packages_reexport_stable_public_names(
    module_name: str,
    expected: dict[str, object],
) -> None:
    package = import_module(module_name)
    declared = vars(package).get("__all__")

    assert isinstance(declared, list)
    assert all(isinstance(name, str) for name in declared)
    assert len(declared) == len(set(declared))
    assert expected.keys() <= set(declared)

    namespace: dict[str, object] = {}
    exec(f"from {module_name} import *", namespace)
    for name, symbol in expected.items():
        assert getattr(package, name) is symbol
        assert namespace[name] is symbol


def test_foundation_packages_import_cleanly_in_a_fresh_interpreter() -> None:
    script = "\n".join(
        (
            "import experience_hub.storage",
            "import experience_hub.domain",
            "import experience_hub.agents",
            "import experience_hub",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
