from __future__ import annotations

import inspect
import subprocess
import sys
from importlib import import_module
from typing import get_type_hints
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.experiences.contracts import (
    ConfirmExperience,
    CreateExperience,
    CreateExperienceVersion,
    ExperienceCreation,
    ExperienceDraft,
    ExperienceMutationReason,
    ExperienceRecord,
    PinExperience,
    RefuteExperience,
    RestoreExperience,
    ShareableExperienceVersion,
    UnpinExperience,
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
from experience_hub.experiences.queries import (
    ExperienceNotFoundError,
    ExperienceQuery,
)
from experience_hub.experiences.repository import (
    ExperienceRepository,
    ExperienceWriter,
)
from experience_hub.experiences.service import (
    ExperienceRetrievalAdapter,
    ExperienceService,
)
from experience_hub.experiences.transitions import ExperienceMutationWriter
from experience_hub.lifecycle.contracts import (
    IdeaArchivePlanner,
    LifecycleResult,
    NullIdeaArchivePlanner,
)
from experience_hub.lifecycle.repository import (
    LifecycleRecord,
    LifecycleRepository,
)
from experience_hub.lifecycle.scoring import (
    AccessUpdate,
    ActivationInputs,
    ActivationResult,
    LifecycleConfig,
    activation_at,
    decay_strength,
    record_access,
)
from experience_hub.lifecycle.service import (
    LifecycleEvaluation,
    LifecycleRunMode,
    LifecycleService,
    LifecycleThresholdTarget,
    evaluate_transition,
    lifecycle_config_hash,
    lifecycle_cycle_id,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    ExperienceView,
    PeekExperiences,
    RetrievalCandidate,
    RetrievalRecord,
    SearchExperiences,
    SearchHit,
    SearchResult,
)
from experience_hub.retrieval.ranking import RetrievalMode
from experience_hub.retrieval.service import (
    ASSOCIATIVE_COLD_EXPANSION_THRESHOLD,
    FOCUSED_COLD_EXPANSION_THRESHOLD,
    ExperienceEvidenceReader,
    RetrievalService,
    retrieval_query_hash,
)

EXPECTED_EXPORTS: dict[str, dict[str, object]] = {
    "experience_hub.experiences": {
        "ConfirmExperience": ConfirmExperience,
        "CreateExperience": CreateExperience,
        "CreateExperienceVersion": CreateExperienceVersion,
        "ExperienceCreation": ExperienceCreation,
        "ExperienceDraft": ExperienceDraft,
        "ExperienceKind": ExperienceKind,
        "ExperienceMutationReason": ExperienceMutationReason,
        "ExperienceMutationWriter": ExperienceMutationWriter,
        "ExperienceNotFoundError": ExperienceNotFoundError,
        "ExperienceOrigin": ExperienceOrigin,
        "ExperienceQuery": ExperienceQuery,
        "ExperienceRecord": ExperienceRecord,
        "ExperienceRepository": ExperienceRepository,
        "ExperienceRetrievalAdapter": ExperienceRetrievalAdapter,
        "ExperienceService": ExperienceService,
        "ExperienceStateSnapshotV1": ExperienceStateSnapshotV1,
        "ExperienceWriter": ExperienceWriter,
        "LinkRelation": LinkRelation,
        "PinExperience": PinExperience,
        "RefuteExperience": RefuteExperience,
        "RestoreExperience": RestoreExperience,
        "ShareableExperienceVersion": ShareableExperienceVersion,
        "Temperature": Temperature,
        "UnpinExperience": UnpinExperience,
        "VersionContent": VersionContent,
        "VersionLinkInput": VersionLinkInput,
    },
    "experience_hub.retrieval": {
        "ASSOCIATIVE_COLD_EXPANSION_THRESHOLD": (
            ASSOCIATIVE_COLD_EXPANSION_THRESHOLD
        ),
        "FOCUSED_COLD_EXPANSION_THRESHOLD": FOCUSED_COLD_EXPANSION_THRESHOLD,
        "CandidateSelection": CandidateSelection,
        "ExperienceEvidenceReader": ExperienceEvidenceReader,
        "ExperienceView": ExperienceView,
        "PeekExperiences": PeekExperiences,
        "RetrievalCandidate": RetrievalCandidate,
        "RetrievalMode": RetrievalMode,
        "RetrievalRecord": RetrievalRecord,
        "RetrievalService": RetrievalService,
        "SearchExperiences": SearchExperiences,
        "SearchHit": SearchHit,
        "SearchResult": SearchResult,
        "retrieval_query_hash": retrieval_query_hash,
    },
    "experience_hub.lifecycle": {
        "AccessUpdate": AccessUpdate,
        "ActivationInputs": ActivationInputs,
        "ActivationResult": ActivationResult,
        "IdeaArchivePlanner": IdeaArchivePlanner,
        "LifecycleConfig": LifecycleConfig,
        "LifecycleEvaluation": LifecycleEvaluation,
        "LifecycleRecord": LifecycleRecord,
        "LifecycleRepository": LifecycleRepository,
        "LifecycleResult": LifecycleResult,
        "LifecycleRunMode": LifecycleRunMode,
        "LifecycleService": LifecycleService,
        "LifecycleThresholdTarget": LifecycleThresholdTarget,
        "NullIdeaArchivePlanner": NullIdeaArchivePlanner,
        "activation_at": activation_at,
        "decay_strength": decay_strength,
        "evaluate_transition": evaluate_transition,
        "lifecycle_config_hash": lifecycle_config_hash,
        "lifecycle_cycle_id": lifecycle_cycle_id,
        "record_access": record_access,
    },
}


@pytest.mark.parametrize(("module_name", "expected"), EXPECTED_EXPORTS.items())
def test_feature_packages_reexport_stable_public_names(
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
    for name in declared:
        assert name in namespace
        assert namespace[name] is getattr(package, name)
    for name, symbol in expected.items():
        assert getattr(package, name) is symbol
        assert namespace[name] is symbol


@pytest.mark.parametrize(
    "module_order",
    (
        (
            "experience_hub.experiences",
            "experience_hub.retrieval",
            "experience_hub.lifecycle",
        ),
        (
            "experience_hub.lifecycle",
            "experience_hub.retrieval",
            "experience_hub.experiences",
        ),
        (
            "experience_hub.retrieval.service",
            "experience_hub.experiences.queries",
            "experience_hub.lifecycle.service",
        ),
        (
            "experience_hub.lifecycle.repository",
            "experience_hub.experiences.repository",
            "experience_hub.retrieval.contracts",
        ),
    ),
)
def test_feature_packages_import_in_any_order_in_a_fresh_interpreter(
    module_order: tuple[str, ...],
) -> None:
    script = "\n".join(
        (
            "from importlib import import_module",
            *(f"import_module({module_name!r})" for module_name in module_order),
            "from experience_hub.experiences import ("
            "ExperienceQuery, ExperienceStateSnapshotV1, "
            "ShareableExperienceVersion)",
            "from experience_hub.retrieval import ("
            "ExperienceEvidenceReader, PeekExperiences, SearchResult)",
            "from experience_hub.lifecycle import ("
            "IdeaArchivePlanner, LifecycleService, activation_at)",
            "from experience_hub.experiences import *",
            "from experience_hub.retrieval import *",
            "from experience_hub.lifecycle import *",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_shareable_experience_query_keeps_its_session_bound_contract() -> None:
    method = ExperienceQuery.get_owned_shareable_version
    signature = inspect.signature(method)
    hints = get_type_hints(method)

    assert inspect.iscoroutinefunction(method)
    assert tuple(signature.parameters) == (
        "self",
        "session",
        "owner_agent_id",
        "experience_id",
        "version_id",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )
    assert hints == {
        "session": AsyncSession,
        "owner_agent_id": UUID,
        "experience_id": UUID,
        "version_id": UUID | None,
        "return": ShareableExperienceVersion,
    }


def test_evidence_reader_keeps_its_read_only_session_contract() -> None:
    method = ExperienceEvidenceReader.peek
    signature = inspect.signature(method)
    hints = get_type_hints(method)

    assert inspect.iscoroutinefunction(method)
    assert tuple(signature.parameters) == ("self", "session", "query")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )
    assert hints == {
        "session": AsyncSession,
        "query": PeekExperiences,
        "return": SearchResult,
    }
