from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest
from tests.passport_fixtures import passport_document
from typer.testing import CliRunner

from experience_hub import canonical_json_bytes
from experience_hub.cli.app import app
from experience_hub.passports import encode_passport_document, verify_passport_bytes
from experience_hub.runtime import ApplicationRuntime

runner = CliRunner()
OWNER = str(UUID(int=8000))
ITEM = str(UUID(int=8001))


def test_inspect_is_pure_and_returns_only_canonical_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = encode_passport_document(passport_document())
    path = tmp_path / "synthetic.passport.json"
    path.write_bytes(body)
    monkeypatch.chdir(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Inspect accessed runtime or SQLite")

    monkeypatch.setattr(ApplicationRuntime, "initialize", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    result = runner.invoke(app, ["passport", "inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert (
        result.stdout
        == canonical_json_bytes(
            {"data": verify_passport_bytes(body).report.model_dump(mode="json")}
        ).decode()
        + "\n"
    )
    assert result.stderr == ""
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize(
    "args",
    [
        ["export", OWNER, ITEM, "--database", "source.sqlite3", "--output", "out.json"],
        [
            "export",
            OWNER,
            ITEM,
            "--input-sanitized",
            "--sharing-authorized",
            "--sanitization-profile",
            "synthetic-v1",
        ],
        ["import", OWNER, "input.json", "--database", "recipient.sqlite3"],
        [
            "adopt",
            OWNER,
            ITEM,
            "--database",
            "recipient.sqlite3",
            "--idempotency-key",
            "decision",
        ],
        [
            "adopt",
            OWNER,
            ITEM,
            "--importance",
            "0.7",
            "--confidence",
            "0.6",
            "--idempotency-key",
            "decision",
        ],
        [
            "adopt",
            OWNER,
            ITEM,
            "--database",
            "recipient.sqlite3",
            "--importance",
            "nan",
            "--confidence",
            "0.6",
            "--idempotency-key",
            "decision",
        ],
        [
            "reject",
            OWNER,
            ITEM,
            "--database",
            "recipient.sqlite3",
            "--reason",
            "Synthetic",
        ],
        ["show", "private-invalid-owner", ITEM, "--database", "source.sqlite3"],
        ["list", OWNER],
        ["inspect", "input.json", "--private-input", "private-test-value"],
        ["inspect", "input.json", "--database"],
    ],
)
def test_missing_or_invalid_options_are_fixed_json_before_runtime(
    args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid command initialized runtime")

    monkeypatch.setattr(ApplicationRuntime, "initialize", forbidden)
    result = runner.invoke(app, ["passport", *args])
    assert result.exit_code != 0
    document = json.loads(result.stdout)
    assert document["error"]["code"] == "passport_invalid"
    assert document["error"]["details"] == {}
    assert result.stdout == canonical_json_bytes(document).decode() + "\n"
    assert "private-test-value" not in result.output
    assert "private-invalid-owner" not in result.output
    assert str(tmp_path) not in result.output
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("malformed", [b"{}", b"not-json"])
def test_inspect_corrupt_file_does_not_echo_input_or_path(
    malformed: bytes,
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-local-input.passport.json"
    path.write_bytes(malformed)
    result = runner.invoke(app, ["passport", "inspect", str(path)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "passport_invalid"
    assert str(path) not in result.output


def test_unknown_passport_command_is_canonical_json() -> None:
    result = runner.invoke(app, ["passport", "private-unknown-command"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "passport_invalid"
    assert "private-unknown-command" not in result.output
