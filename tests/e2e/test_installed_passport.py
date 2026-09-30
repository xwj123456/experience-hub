"""Actual installed-console acceptance, independent of source test helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_HOOK = dedent(
    """
    import os
    import sys

    audit_log = os.environ.get("PASSPORT_AUDIT_LOG")
    no_sqlite = os.environ.get("PASSPORT_NO_SQLITE") == "1"

    def refuse(kind):
        if audit_log:
            with open(audit_log, "a", encoding="utf-8") as stream:
                stream.write(kind + "\\n")
        raise AssertionError("Forbidden installed Passport access: " + kind)

    def audit(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
            refuse("network")
        if no_sqlite and event == "sqlite3.connect":
            refuse("sqlite")
        if event == "open" and arguments and isinstance(arguments[0], (str, bytes)):
            path = os.fsdecode(arguments[0])
            name = path.rsplit("/", 1)[-1]
            if (name == ".env" or name.startswith(".env.")
                    or "/.aws/" in path or name in {"auth.json", "credentials.json"}):
                refuse("credential-file")

    original_getitem = type(os.environ).__getitem__
    def guarded_getitem(self, key):
        if any(part in key.upper() for part in (
                "API_KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI")):
            refuse("credential-environment")
        return original_getitem(self, key)

    type(os.environ).__getitem__ = guarded_getitem
    sys.addaudithook(audit)
    """
)

SEED_PROGRAM = dedent(
    """
    import asyncio
    import json
    import sys
    from datetime import UTC, datetime
    from pathlib import Path
    from uuid import UUID
    import experience_hub
    from sqlalchemy import text
    from sqlalchemy.engine import URL
    from experience_hub.agents import CreateAgent
    from experience_hub.clock import FrozenClock
    from experience_hub.config import Settings
    from experience_hub.domain import CommandRequest, TypedEvidence
    from experience_hub.experiences.contracts import CreateExperience
    from experience_hub.experiences.models import ExperienceKind, VersionContent
    from experience_hub.ids import SequenceIdGenerator
    from experience_hub.runtime import ApplicationRuntime, installed_schema_head

    async def main():
        database, role = sys.argv[1:]
        assert "site-packages" in Path(experience_hub.__file__).parts
        offset = 1000 if role == "source" else 2000
        runtime = ApplicationRuntime(
            Settings(database_url=URL.create("sqlite+aiosqlite", database=database)),
            clock=FrozenClock(datetime(2026, 9, 30, 0, 0, tzinfo=UTC)),
            ids=SequenceIdGenerator(tuple(
                UUID(int=i) for i in range(offset, offset+200)))
        )
        async with runtime.initialize(
                start_lifecycle_worker=False, recover_interrupted=False) as container:
            owners = []
            for index in range(1 if role == "source" else 2):
                name = f"Synthetic installed {role} owner {index}"
                async def handler(uow, context):
                    return await container.agent_service.create(
                        uow=uow, command=CreateAgent(name=name),
                        command_context=context)
                result = await container.command_executor.execute(CommandRequest(
                    caller_scope="system:local", operation_scope="agent.create",
                    idempotency_key=f"seed-owner-{index}", method="POST",
                    route_template="/v1/agents", body={"name": name}), handler)
                assert result.status_code == 201
                owners.append(json.loads(result.body)["data"]["agent_id"])
            experiences = []
            if role == "source":
                owner = UUID(owners[0])
                for marker in ("rollback-alpha", "discard-zebra"):
                    content = VersionContent(
                        body=f"Synthetic {marker} retry transaction after rollback.",
                        summary=f"Synthetic {marker} independent transfer.",
                        mechanism="An atomic SQLite transaction rolls back effects.",
                        tags=(marker,), applicability=("local SQLite commands",),
                        evidence=(TypedEvidence(type="web_document",
                            id="https://example.invalid/never-fetch/" + marker),),
                        falsifiers=("A source survived the rollback.",))
                    command = CreateExperience(owner_agent_id=owner,
                        kind=ExperienceKind.PROCEDURAL, content=content,
                        importance=0.93, confidence=0.97)
                    async def handler(uow, context):
                        return await container.experience_service.create(
                            uow=uow, command=command, command_context=context)
                    result = await container.command_executor.execute(CommandRequest(
                        caller_scope=f"agent:{owner}",
                        operation_scope="experience.create",
                        idempotency_key="seed-" + marker, method="POST",
                        route_template="/v1/agents/{agent_id}/experiences",
                        path_parameters={"agent_id": str(owner)},
                        body={"content": content.model_dump(mode="json"),
                              "importance": 0.93, "confidence": 0.97}), handler)
                    assert result.status_code == 201
                    experiences.append(json.loads(result.body)["data"])
            async with container.database.read_session() as session:
                paths = list((await session.execute(
                    text("PRAGMA database_list"))).all())
                assert next(row[2] for row in paths if row[1] == "main") == database
                await container.source_validator.validate(session)
            print(json.dumps({"owners": owners, "experiences": experiences,
                "head": installed_schema_head(), "exact_database": True}))

    asyncio.run(main())
    """
)

OBSERVE_PROGRAM = dedent(
    """
    import asyncio
    import json
    import sys
    from uuid import UUID
    from sqlalchemy import func, select
    from sqlalchemy.engine import URL
    from experience_hub.config import Settings
    from experience_hub.retrieval import RetrievalMode, SearchExperiences
    from experience_hub.runtime import ApplicationRuntime
    from experience_hub.storage.tables import (
        ExperienceRow, ExperienceStateRow, ExperienceTermRow, ExperienceVersionRow,
        IdempotencyRecordRow, PassportAdoptionRow, PassportImportRow)

    async def main():
        database, owner, key = sys.argv[1:]
        owner = UUID(owner)
        runtime = ApplicationRuntime(Settings(database_url=URL.create(
            "sqlite+aiosqlite", database=database)))
        async with runtime.initialize(
                start_lifecycle_worker=False, recover_interrupted=False) as container:
            async with container.database.read_session() as session:
                await container.source_validator.validate(session)
                rows = list((await session.execute(select(ExperienceRow,
                    ExperienceStateRow, ExperienceVersionRow)
                    .join(ExperienceStateRow,
                        ExperienceStateRow.experience_id == ExperienceRow.experience_id)
                    .join(ExperienceVersionRow, ExperienceVersionRow.version_id ==
                        ExperienceStateRow.current_version_id)
                    .where(ExperienceRow.owner_agent_id == owner))).all())
                states = [{"experience_id": str(identity.experience_id),
                    "version_id": str(version.version_id), "origin": identity.origin,
                    "hash": version.content_hash, "confidence": state.confidence,
                    "importance": state.importance, "trust": state.source_trust,
                    "temperature": state.temperature}
                    for identity, state, version in rows]
                terms = await session.scalar(select(func.count())
                    .select_from(ExperienceTermRow).join(ExperienceRow,
                        ExperienceRow.experience_id == ExperienceTermRow.experience_id)
                    .where(ExperienceRow.owner_agent_id == owner))
                imports = await session.scalar(select(func.count())
                    .select_from(PassportImportRow)
                    .where(PassportImportRow.owner_agent_id == owner))
                adoptions = await session.scalar(select(func.count())
                    .select_from(PassportAdoptionRow)
                    .where(PassportAdoptionRow.owner_agent_id == owner))
                inflight = await session.scalar(select(func.count())
                    .select_from(IdempotencyRecordRow)
                    .where(IdempotencyRecordRow.state != "completed"))
            result = await container.retrieval_adapter.search(
                query=SearchExperiences(owner_agent_id=owner,
                    query="rollback transaction retry", mode=RetrievalMode.FOCUSED),
                idempotency_key=key)
            assert result.status_code == 200
            print(json.dumps({"states": states, "terms": terms, "imports": imports,
                "adoptions": adoptions, "inflight": inflight,
                "hits": json.loads(result.body)["data"]["hits"],
                "source_validation": True}))

    asyncio.run(main())
    """
)


@dataclass(frozen=True)
class InstalledPassport:
    python: Path
    console: Path
    audit_log: Path


def _environment(installed: InstalledPassport | None = None) -> dict[str, str]:
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(Path.home()),
        "UV_OFFLINE": "true",
        "UV_NO_CONFIG": "true",
        "PYTHONNOUSERSITE": "1",
    }
    if installed:
        environment["PASSPORT_AUDIT_LOG"] = str(installed.audit_log)
    return environment


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    installed: InstalledPassport | None = None,
    no_sqlite: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = _environment(installed)
    if no_sqlite:
        environment["PASSPORT_NO_SQLITE"] = "1"
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.fixture(scope="module")
def installed_passport(tmp_path_factory: pytest.TempPathFactory) -> InstalledPassport:
    root = tmp_path_factory.mktemp("installed-passport")
    wheels, requirements = root / "wheels", root / "requirements.txt"
    environment = root / "venv?井#percent%🧪"
    python = environment / "bin/python"
    for arguments in (
        ["uv", "build", "--offline", "--wheel", "--out-dir", str(wheels)],
        [
            "uv",
            "export",
            "--offline",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements),
        ],
        ["uv", "venv", "--offline", "--python", sys.executable, str(environment)],
        ["uv", "pip", "sync", "--offline", "--python", str(python), str(requirements)],
    ):
        result = _run(arguments, cwd=PROJECT_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = wheels.glob("*.whl")
    result = _run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            "--no-deps",
            str(wheel),
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run(
        [
            str(python),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    Path(result.stdout.strip(), "sitecustomize.py").write_text(AUDIT_HOOK)
    for kind, program, no_sqlite in (
        ("network", "import socket; socket.getaddrinfo('example.invalid', 80)", False),
        (
            "credential-environment",
            "import os; os.getenv('OPENAI_API_KEY')",
            False,
        ),
        ("sqlite", "import sqlite3; sqlite3.connect(':memory:')", True),
    ):
        probe_log = root / f"guard-proof-{kind}.log"
        probe = InstalledPassport(python, environment / "bin/experience-hub", probe_log)
        result = _run(
            [str(python), "-I", "-c", program],
            cwd=root,
            installed=probe,
            no_sqlite=no_sqlite,
        )
        assert result.returncode != 0
        assert probe_log.read_text() == kind + "\n"
    return InstalledPassport(
        python, environment / "bin/experience-hub", root / "audit.log"
    )


def _json_result(result: subprocess.CompletedProcess[str], *, success: bool) -> dict:
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert result.stderr == "", result.stderr
    document = json.loads(result.stdout)
    expected = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    assert result.stdout == expected
    assert ("data" in document) is success
    assert ("error" in document) is not success
    return document


def _cli(
    installed: InstalledPassport,
    cwd: Path,
    *arguments: str,
    success: bool = True,
    no_sqlite: bool = False,
) -> tuple[str, dict]:
    result = _run(
        [str(installed.console), "passport", *arguments],
        cwd=cwd,
        installed=installed,
        no_sqlite=no_sqlite,
    )
    return result.stdout, _json_result(result, success=success)


def _seed(installed: InstalledPassport, cwd: Path, database: Path, role: str) -> dict:
    result = _run(
        [str(installed.python), "-I", "-c", SEED_PROGRAM, str(database), role],
        cwd=cwd,
        installed=installed,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == "", result.stderr
    return json.loads(result.stdout)


def _observe(
    installed: InstalledPassport, cwd: Path, database: Path, owner: str, key: str
) -> dict:
    result = _run(
        [str(installed.python), "-I", "-c", OBSERVE_PROGRAM, str(database), owner, key],
        cwd=cwd,
        installed=installed,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == "", result.stderr
    return json.loads(result.stdout)


def _source_fingerprint(database: Path) -> dict[str, str | None]:
    return {
        suffix: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for suffix in ("", "-wal", "-shm", "-journal")
        for path in (Path(str(database) + suffix),)
    }


def _cwd(tmp_path: Path) -> Path:
    cwd = tmp_path / "unrelated cwd?井#percent%🧪"
    cwd.mkdir()
    (cwd / "pyproject.toml").write_text(
        '[project]\nname="unrelated-passport-test"\nversion="99.0.0"\n'
    )
    return cwd


def test_installed_console_transfers_quarantines_and_decides_offline(
    installed_passport: InstalledPassport, tmp_path: Path
) -> None:
    installed, cwd = installed_passport, _cwd(tmp_path)
    cold_probe = cwd / "cold?井#%🧪.passport.json"
    cold_probe.write_text('{"unknown":"synthetic-cold-inspect"}')
    _, cold_result = _cli(
        installed, cwd, "inspect", str(cold_probe), success=False, no_sqlite=True
    )
    assert cold_result["error"]["code"] == "passport_invalid"
    source = cwd / "source?精确#percent%🧪.sqlite3"
    target = cwd / "target?精确#percent%🧪.sqlite3"
    sender, recipient = (
        _seed(installed, cwd, source, "source"),
        _seed(installed, cwd, target, "target"),
    )
    assert sender["exact_database"] and recipient["exact_database"]
    assert sender["head"] == recipient["head"] == "0008_evidence_passports"
    owner, other = recipient["owners"]
    source_before = _source_fingerprint(source)
    files = [cwd / f"transfer-{index}?井#%🧪.passport.json" for index in range(2)]
    for experience, output in zip(sender["experiences"], files, strict=True):
        _cli(
            installed,
            cwd,
            "export",
            sender["owners"][0],
            experience["experience_id"],
            "--database",
            str(source),
            "--output",
            str(output),
            "--input-sanitized",
            "--sharing-authorized",
            "--sanitization-profile",
            "installed-合成-v1",
        )
    assert _source_fingerprint(source) == source_before
    _, inspected = _cli(installed, cwd, "inspect", str(files[0]), no_sqlite=True)
    report = inspected["data"]
    assert report["publisher_identity"] == "unverified"
    assert report["semantic_assessment"] == "not_assessed"
    assert report["reference_only_count"] == report["unavailable_preimage_count"] == 1
    assert report["embedded_excerpt_count"] == 0
    assert "body" not in report and "excerpt" not in report
    wire = json.loads(files[0].read_bytes())
    assert wire["subject"]["content_hash"] == sender["experiences"][0]["content_hash"]
    original_output = files[0].read_bytes()
    _cli(
        installed,
        cwd,
        "export",
        sender["owners"][0],
        sender["experiences"][0]["experience_id"],
        "--database",
        str(source),
        "--output",
        str(files[0]),
        "--input-sanitized",
        "--sharing-authorized",
        "--sanitization-profile",
        "installed-合成-v1",
    )
    assert files[0].read_bytes() == original_output
    probe = "ghp_" + "Z" * 36
    sensitive_wire = json.loads(original_output)
    sensitive_wire["declaration"]["profile_id"] = probe
    sensitive_wire.pop("passport_hash")
    encoded = json.dumps(
        sensitive_wire, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    sensitive_wire["passport_hash"] = hashlib.sha256(encoded).hexdigest()
    sensitive = cwd / "sensitive-canonical.passport.json"
    sensitive.write_text(
        json.dumps(
            sensitive_wire, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    )
    sensitive_bytes, sensitive_error = _cli(
        installed, cwd, "inspect", str(sensitive), success=False, no_sqlite=True
    )
    assert sensitive_error["error"]["code"] == "passport_sensitive_content"
    assert probe not in sensitive_bytes and str(cwd) not in sensitive_bytes
    import_args = (
        "import",
        owner,
        str(files[0]),
        "--database",
        str(target),
        "--idempotency-key",
        "transfer-import",
    )
    first, imported = _cli(installed, cwd, *import_args)
    assert _cli(installed, cwd, *import_args)[0] == first
    import_id = imported["data"]["import_id"]
    _, shown = _cli(installed, cwd, "show", owner, import_id, "--database", str(target))
    assert shown["data"]["state"] == "pending"
    _, listed = _cli(
        installed,
        cwd,
        "list",
        owner,
        "--database",
        str(target),
        "--state",
        "pending",
        "--limit",
        "1",
    )
    assert [item["import_id"] for item in listed["data"]["items"]] == [import_id]
    pending = _observe(installed, cwd, target, owner, "search-pending")
    assert pending["hits"] == pending["states"] == []
    assert pending["terms"] == pending["adoptions"] == pending["inflight"] == 0
    foreign, _ = _cli(
        installed,
        cwd,
        "show",
        other,
        import_id,
        "--database",
        str(target),
        success=False,
    )
    missing, _ = _cli(
        installed,
        cwd,
        "show",
        other,
        "90000000-0000-4000-8000-000000000099",
        "--database",
        str(target),
        success=False,
    )
    assert foreign == missing
    _, other_import = _cli(
        installed,
        cwd,
        "import",
        other,
        str(files[0]),
        "--database",
        str(target),
        "--idempotency-key",
        "transfer-import",
    )
    assert other_import["data"]["import_id"] != import_id
    adopt_args = (
        "adopt",
        owner,
        import_id,
        "--database",
        str(target),
        "--importance",
        "0.7",
        "--confidence",
        "0.6",
        "--idempotency-key",
        "transfer-adopt",
    )
    adopted_bytes, adopted = _cli(installed, cwd, *adopt_args)
    assert _cli(installed, cwd, *adopt_args)[0] == adopted_bytes
    local_id = adopted["data"]["experience"]["experience_id"]
    assert local_id != sender["experiences"][0]["experience_id"]
    adopted_state = _observe(installed, cwd, target, owner, "search-adopted")
    assert len(adopted_state["hits"]) == len(adopted_state["states"]) == 1
    state = adopted_state["states"][0]
    assert state["experience_id"] == local_id
    assert state["hash"] == wire["subject"]["content_hash"]
    assert state["origin"] == "adopted_passport" and state["temperature"] == "warm"
    assert (state["importance"], state["confidence"], state["trust"]) == (
        0.7,
        0.6,
        0.25,
    )
    assert adopted_state["terms"] > 0 and adopted_state["adoptions"] == 1
    assert _observe(installed, cwd, target, other, "search-other-pending")["hits"] == []
    _, terminal = _cli(
        installed,
        cwd,
        "import",
        owner,
        str(files[0]),
        "--database",
        str(target),
        "--idempotency-key",
        "terminal-new-key",
    )
    assert (terminal["data"]["import_id"], terminal["data"]["state"]) == (
        import_id,
        "adopted",
    )
    _, refused = _cli(installed, cwd, *adopt_args[:-1], "adopt-new-key", success=False)
    assert refused["error"]["code"] == "passport_decision_conflict"
    _, rejected_import = _cli(
        installed,
        cwd,
        "import",
        owner,
        str(files[1]),
        "--database",
        str(target),
        "--idempotency-key",
        "rejection-import",
    )
    rejected_id = rejected_import["data"]["import_id"]
    reject_args = (
        "reject",
        owner,
        rejected_id,
        "--database",
        str(target),
        "--reason",
        "Synthetic local rejection",
        "--idempotency-key",
        "reject-1",
    )
    rejection, decision = _cli(installed, cwd, *reject_args)
    assert decision["data"]["state"] == "rejected"
    assert _cli(installed, cwd, *reject_args)[0] == rejection
    _, preserved = _cli(
        installed,
        cwd,
        "import",
        owner,
        str(files[1]),
        "--database",
        str(target),
        "--idempotency-key",
        "rejected-new-key",
    )
    assert (preserved["data"]["import_id"], preserved["data"]["state"]) == (
        rejected_id,
        "rejected",
    )
    final = _observe(installed, cwd, target, owner, "search-after-rejection")
    assert len(final["states"]) == len(final["hits"]) == final["adoptions"] == 1
    assert final["imports"] == 2 and final["inflight"] == 0
    assert final["source_validation"] is True
    assert _source_fingerprint(source) == source_before
    assert not (cwd / ".data").exists()
    assert not (cwd / "source").exists() and not (cwd / "target").exists()
    assert not installed.audit_log.exists(), installed.audit_log.read_text()


def test_installed_inspect_rejects_bad_files_without_database_or_input_echo(
    installed_passport: InstalledPassport, tmp_path: Path
) -> None:
    installed, cwd = installed_passport, _cwd(tmp_path)
    probe = "ghp_" + "Z" * 36
    bad = cwd / "sensitive?私密#%🧪.passport.json"
    bad.write_text('{"synthetic_private_probe":"' + probe + '"}')
    before = set(cwd.iterdir())
    _, result = _cli(installed, cwd, "inspect", str(bad), success=False, no_sqlite=True)
    assert result["error"]["code"] == "passport_invalid"
    assert probe not in json.dumps(result) and str(cwd) not in json.dumps(result)
    assert set(cwd.iterdir()) == before
    linked = cwd / "linked.passport.json"
    linked.symlink_to(bad)
    _, result = _cli(
        installed, cwd, "inspect", str(linked), success=False, no_sqlite=True
    )
    assert result["error"]["code"] == "passport_file_invalid"
    assert bad.read_text().find(probe) >= 0
    assert not installed.audit_log.exists(), installed.audit_log.read_text()
