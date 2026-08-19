from __future__ import annotations

import inspect
import subprocess
import sys
from importlib import import_module
from typing import get_type_hints

from experience_hub.capture.models import PreparedCaptureV1
from experience_hub.capture.service import CapturePreparer, CaptureService
from experience_hub.domain import CommandContext
from experience_hub.storage import StoredResponse, UnitOfWork

CAPTURE_EXPORTS = {
    "CapturePreparer": CapturePreparer,
    "CaptureService": CaptureService,
    "GenericJsonlAdapter": import_module(
        "experience_hub.capture.jsonl"
    ).GenericJsonlAdapter,
    "PreparedCaptureV1": PreparedCaptureV1,
    "TrajectoryBundleV1": import_module(
        "experience_hub.capture.models"
    ).TrajectoryBundleV1,
}


def test_capture_package_reexports_stable_public_names() -> None:
    package = import_module("experience_hub.capture")
    declared = vars(package).get("__all__")

    assert isinstance(declared, list)
    assert all(isinstance(name, str) for name in declared)
    assert len(declared) == len(set(declared))
    assert CAPTURE_EXPORTS.keys() <= set(declared)

    namespace: dict[str, object] = {}
    exec("from experience_hub.capture import *", namespace)
    for name in declared:
        assert name in namespace
        assert namespace[name] is getattr(package, name)
    for name, symbol in CAPTURE_EXPORTS.items():
        assert getattr(package, name) is symbol
        assert namespace[name] is symbol


def test_capture_and_candidate_packages_import_in_any_order_fresh() -> None:
    for module_order in (
        ("experience_hub.capture", "experience_hub.experiences"),
        ("experience_hub.experiences", "experience_hub.capture"),
        ("experience_hub.capture.service", "experience_hub.experiences"),
    ):
        script = "\n".join(
            (
                "from importlib import import_module",
                *(f"import_module({name!r})" for name in module_order),
                "from experience_hub.capture import *",
                "from experience_hub.experiences import ("
                "AdoptCandidate, CandidateService, RejectCandidate)",
                "from experience_hub.experiences import *",
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_capture_service_keeps_transaction_bound_contract() -> None:
    method = CaptureService.capture
    signature = inspect.signature(method)

    assert inspect.iscoroutinefunction(method)
    assert tuple(signature.parameters) == ("self", "uow", "prepared", "command")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )
    assert get_type_hints(method) == {
        "uow": UnitOfWork,
        "prepared": PreparedCaptureV1,
        "command": CommandContext,
        "return": StoredResponse,
    }
