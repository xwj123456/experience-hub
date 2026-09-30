from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError

import pytest
from tests.passport_service_fixtures import OWNER, prepared_passport


def test_public_import_command_is_frozen_and_prepared_verified() -> None:
    passports = importlib.import_module("experience_hub.passports")
    command_type = getattr(passports, "ImportPassport", None)
    assert command_type is not None, "public ImportPassport is missing"
    prepared = prepared_passport()
    command = command_type(owner_agent_id=OWNER, prepared=prepared)
    assert command.prepared == prepared
    with pytest.raises(FrozenInstanceError):
        command.owner_agent_id = OWNER
    forged = prepared_passport()
    object.__setattr__(forged, "canonical_bytes", b"{}")
    with pytest.raises(Exception) as error:
        command_type(owner_agent_id=OWNER, prepared=forged)
    assert getattr(error.value, "code", None) == "passport_invalid"


def test_query_and_service_are_public() -> None:
    passports = importlib.import_module("experience_hub.passports")
    for name in (
        "PassportService",
        "PassportQuery",
        "PassportImportViewV1",
        "PassportPageV1",
    ):
        assert getattr(passports, name, None) is not None, f"public {name} is missing"


def test_export_service_has_public_lazy_export() -> None:
    from experience_hub.passports.export import PassportExportService

    passports = importlib.import_module("experience_hub.passports")
    assert getattr(passports, "PassportExportService", None) is PassportExportService
