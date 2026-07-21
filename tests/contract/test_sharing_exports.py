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

from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.tokenizer import TermCue
from experience_hub.sharing.models import (
    MAX_PROVENANCE_HOPS,
    MAX_TOPIC_DESCRIPTION_CHARACTERS,
    MAX_TOPIC_NAME_CHARACTERS,
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
    Reputation,
    RetractCapsule,
    SharingMutationReason,
    Subscription,
    Topic,
)
from experience_hub.sharing.queries import (
    InboxEvidenceReader,
    InboxPage,
    QuarantinedCapsuleEvidence,
    SharingQuery,
)

EXPECTED_EXPORTS: dict[str, object] = {
    "MAX_PROVENANCE_HOPS": MAX_PROVENANCE_HOPS,
    "MAX_TOPIC_DESCRIPTION_CHARACTERS": (MAX_TOPIC_DESCRIPTION_CHARACTERS),
    "MAX_TOPIC_NAME_CHARACTERS": MAX_TOPIC_NAME_CHARACTERS,
    "AdoptCapsule": AdoptCapsule,
    "AdoptionResult": AdoptionResult,
    "Capsule": Capsule,
    "CapsuleStatus": CapsuleStatus,
    "CreateSubscription": CreateSubscription,
    "CreateTopic": CreateTopic,
    "EffectiveAvailability": EffectiveAvailability,
    "FeedbackRevision": FeedbackRevision,
    "FeedbackVerdict": FeedbackVerdict,
    "InboxEvidenceReader": InboxEvidenceReader,
    "InboxItem": InboxItem,
    "InboxPage": InboxPage,
    "InboxState": InboxState,
    "ProvenanceHop": ProvenanceHop,
    "PublishCapsule": PublishCapsule,
    "QuarantinedCapsuleEvidence": QuarantinedCapsuleEvidence,
    "RecordCapsuleFeedback": RecordCapsuleFeedback,
    "RejectInboxItem": RejectInboxItem,
    "Reputation": Reputation,
    "RetractCapsule": RetractCapsule,
    "SharingMutationReason": SharingMutationReason,
    "SharingQuery": SharingQuery,
    "Subscription": Subscription,
    "Topic": Topic,
}

COMMAND_FIELDS = {
    CreateTopic: ("owner_agent_id", "name", "description"),
    CreateSubscription: ("subscriber_agent_id", "topic_id"),
    PublishCapsule: (
        "owner_agent_id",
        "topic_id",
        "experience_id",
        "version_id",
        "expires_at",
        "parent_adoption_id",
    ),
    AdoptCapsule: ("adopter_agent_id", "item_id", "importance"),
    RetractCapsule: ("publisher_agent_id", "capsule_id", "reason"),
    RejectInboxItem: ("recipient_agent_id", "item_id", "reason"),
    RecordCapsuleFeedback: (
        "observer_agent_id",
        "capsule_id",
        "verdict",
        "reason",
        "evidence",
    ),
}


def test_sharing_package_reexports_the_frozen_public_contract() -> None:
    package = import_module("experience_hub.sharing")
    declared = vars(package).get("__all__")

    assert isinstance(declared, list)
    assert all(isinstance(name, str) for name in declared)
    assert len(declared) == len(set(declared))
    assert set(declared) == set(EXPECTED_EXPORTS)

    namespace: dict[str, object] = {}
    exec("from experience_hub.sharing import *", namespace)
    for name, symbol in EXPECTED_EXPORTS.items():
        assert getattr(package, name) is symbol
        assert namespace[name] is symbol


@pytest.mark.parametrize(
    ("command_type", "expected_fields"),
    COMMAND_FIELDS.items(),
)
def test_sharing_command_field_contract_is_frozen(
    command_type: Any,
    expected_fields: tuple[str, ...],
) -> None:
    assert tuple(field.name for field in fields(command_type)) == (expected_fields)
    parameters = inspect.signature(command_type).parameters
    assert tuple(parameters) == expected_fields


@pytest.mark.parametrize(
    "module_order",
    (
        (
            "experience_hub.sharing",
            "experience_hub.sharing.models",
            "experience_hub.sharing.queries",
        ),
        (
            "experience_hub.sharing.queries",
            "experience_hub.sharing.models",
            "experience_hub.sharing",
        ),
        (
            "experience_hub.sharing.models",
            "experience_hub.sharing",
            "experience_hub.sharing.queries",
        ),
    ),
)
def test_sharing_contract_imports_in_any_order_in_a_fresh_interpreter(
    module_order: tuple[str, ...],
) -> None:
    script = "\n".join(
        (
            "from importlib import import_module",
            *(f"import_module({module_name!r})" for module_name in module_order),
            "from experience_hub.sharing import ("
            "AdoptCapsule, Capsule, InboxEvidenceReader, "
            "RecordCapsuleFeedback, SharingQuery)",
            "from experience_hub.sharing import *",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_owner_scoped_inbox_query_keeps_its_session_bound_contract() -> None:
    method = SharingQuery.list_inbox
    signature = inspect.signature(method)
    hints = get_type_hints(method)

    assert inspect.iscoroutinefunction(method)
    assert tuple(signature.parameters) == (
        "self",
        "session",
        "owner_agent_id",
        "state",
        "cursor",
        "limit",
        "at",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )
    assert hints == {
        "session": AsyncSession,
        "owner_agent_id": UUID,
        "state": InboxState | None,
        "cursor": str | None,
        "limit": int,
        "at": datetime,
        "return": InboxPage,
    }


def test_pending_evidence_reader_keeps_its_read_only_session_contract() -> None:
    method = InboxEvidenceReader.list_available_pending
    signature = inspect.signature(method)
    hints = get_type_hints(method)

    assert inspect.iscoroutinefunction(method)
    assert tuple(signature.parameters) == (
        "self",
        "session",
        "recipient_agent_id",
        "as_of",
        "query_cues",
        "mode",
        "limit",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )
    assert hints == {
        "session": AsyncSession,
        "recipient_agent_id": UUID,
        "as_of": datetime,
        "query_cues": Iterable[TermCue],
        "mode": RetrievalMode,
        "limit": int,
        "return": tuple[QuarantinedCapsuleEvidence, ...],
    }
