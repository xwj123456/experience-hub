"""Integration coverage for the closed ExperienceBench policy registry."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import pytest

from experience_hub.experiments.benchmarks.contracts import (
    BenchmarkArmDescriptorV1,
    BenchmarkArmKind,
    BenchmarkCaseV1,
)
from experience_hub.experiments.benchmarks.source import BenchmarkSourceIndex
from experience_hub.experiments.contracts import ArmObservationV1
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)

FROZEN_AT = datetime(2026, 8, 20, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-4000-8000-000000000001")


def _case() -> BenchmarkCaseV1:
    return BenchmarkCaseV1.model_validate_json(
        json.dumps(
            {
            "schema_version": 1,
            "case_id": "queue-recovery",
            "source_class": "public_authored",
            "review_status": "authored",
            "stratum": "failure_recovery",
            "language": "en",
            "difficulty": "B",
            "owner_label": "queue-owner",
            "query": 'queue "recovery"',
            "mode": "focused",
            "tags": ["queue"],
            "mechanism_cues": ["bounded-recovery"],
            "limit": 2,
            "content_budget_bytes": 20,
            "source_labels": [
                "queue-required",
                "queue-optional",
                "queue-forbidden",
                "queue-stale",
                "queue-misleading",
            ],
            "required": [{"label": "queue-required", "weight_micros": 450000}],
            "optional": [{"label": "queue-optional", "weight_micros": 0}],
            "forbidden": [{"label": "queue-forbidden", "weight_micros": 100000}],
            "stale": [{"label": "queue-stale", "weight_micros": 100000}],
            "misleading": [
                {"label": "queue-misleading", "weight_micros": 100000}
            ],
            "checkpoints": [
                {
                    "predicate": "ordered_subsequence",
                    "labels": ["queue-required"],
                    "weight_micros": 150000,
                }
            ],
            "oracle_version": 1,
            }
        ),
    )


def _source_index() -> BenchmarkSourceIndex:
    first = UUID("00000000-0000-4000-8000-000000000010")
    second = UUID("00000000-0000-4000-8000-000000000011")
    return BenchmarkSourceIndex(
        agent_ids=MappingProxyType({"queue-owner": OWNER_ID}),
        experience_ids=MappingProxyType(
            {"queue-required": first, "queue-optional": second}
        ),
        candidate_ids=MappingProxyType({}),
        labels_by_experience_id=MappingProxyType(
            {first: "queue-required", second: "queue-optional"}
        ),
        content_bytes_by_label=MappingProxyType(
            {"queue-required": 12, "queue-optional": 12}
        ),
    )


def _descriptor(kind: BenchmarkArmKind) -> BenchmarkArmDescriptorV1:
    return BenchmarkArmDescriptorV1(
        schema_version=1,
        arm_id=kind.value,
        kind=kind,
        required=True,
    )


def _clone_with_owned_records(path: Path) -> None:
    first = UUID("00000000-0000-4000-8000-000000000010")
    second = UUID("00000000-0000-4000-8000-000000000011")
    archived = UUID("00000000-0000-4000-8000-000000000012")
    foreign = UUID("00000000-0000-4000-8000-000000000013")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE experiences(experience_id TEXT, owner_agent_id TEXT,
                                     created_at TEXT);
            CREATE TABLE experience_versions(version_id TEXT, experience_id TEXT,
                                             summary TEXT, mechanism TEXT,
                                             tags BLOB, applicability BLOB);
            CREATE TABLE experience_state(experience_id TEXT, owner_agent_id TEXT,
                                          current_version_id TEXT, temperature TEXT);
            CREATE TABLE experience_candidates(candidate_id TEXT, owner_agent_id TEXT);
            """
        )
        rows = (
            (
                first,
                OWNER_ID,
                "2026-08-19T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
            (
                second,
                OWNER_ID,
                "2026-08-19T00:00:00Z",
                "ordinary note",
                "other mechanism",
            ),
            (
                archived,
                OWNER_ID,
                "2026-08-20T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
            (
                foreign,
                UUID("00000000-0000-4000-8000-000000000002"),
                "2026-08-21T00:00:00Z",
                "queue recovery",
                "bounded recovery",
            ),
        )
        for ordinal, (
            experience_id,
            owner_id,
            created_at,
            summary,
            mechanism,
        ) in enumerate(rows):
            version_id = f"version-{ordinal}"
            connection.execute(
                "INSERT INTO experiences VALUES (?, ?, ?)",
                (str(experience_id), str(owner_id), created_at),
            )
            connection.execute(
                "INSERT INTO experience_versions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    version_id,
                    str(experience_id),
                    summary,
                    mechanism,
                    json.dumps(["queue"]),
                    json.dumps(["local queue"]),
                ),
            )
            connection.execute(
                "INSERT INTO experience_state VALUES (?, ?, ?, ?)",
                (
                    str(experience_id),
                    str(owner_id),
                    version_id,
                    "archived" if experience_id == archived else "warm",
                ),
            )
        connection.execute(
            "INSERT INTO experience_candidates VALUES (?, ?)",
            ("pending-candidate", str(OWNER_ID)),
        )


def _context(clone_path: Path):
    from experience_hub.experiments.benchmarks.policies import BenchmarkPolicyContext

    return BenchmarkPolicyContext(
        case=_case(),
        clone_path=clone_path,
        owner_agent_id=OWNER_ID,
        source_index=_source_index(),
        frozen_at=FROZEN_AT,
        seed=20260820,
    )


def test_registry_accepts_exactly_the_four_closed_baseline_arms() -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    for kind in BenchmarkArmKind:
        assert build_benchmark_policy(_descriptor(kind)).descriptor.kind is kind

    forged = _descriptor(BenchmarkArmKind.NO_MEMORY).model_copy(
        update={"kind": "external-policy"}
    )
    with pytest.raises(ValueError) as raised:
        build_benchmark_policy(forged)
    assert "external-policy" not in str(raised.value)


@pytest.mark.asyncio
async def test_no_memory_never_opens_the_clone_and_has_no_selected_bytes(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import (
        BenchmarkPolicyContext,
        build_benchmark_policy,
    )

    observation = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.NO_MEMORY)
    ).execute(
        BenchmarkPolicyContext(
            case=_case(),
            clone_path=tmp_path / "must-not-open.sqlite3",
            owner_agent_id=OWNER_ID,
            source_index=_source_index(),
            frozen_at=FROZEN_AT,
            seed=20260820,
        )
    )

    assert observation.returned_labels == ()
    assert observation.selected_content_bytes == 0


@pytest.mark.asyncio
async def test_recent_and_bm25_use_only_owned_nonarchived_records_and_prefix_budget(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "disposable.sqlite3"
    _clone_with_owned_records(clone_path)
    context = _context(clone_path)

    recent = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.RECENT_NOTES)
    ).execute(context)
    bm25 = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(context)

    assert recent.returned_labels == ("queue-optional",)
    assert recent.selected_content_bytes == 12
    assert bm25.returned_labels == ("queue-required",)
    assert bm25.selected_content_bytes == 12
    assert "pending-candidate" not in (*recent.returned_labels, *bm25.returned_labels)


def test_fts_phrase_escapes_quotes_and_missing_capability_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module

    assert policy_module._fts_phrase('queue "recovery"') == '"queue ""recovery"""'
    original_connect = sqlite3.connect

    def unavailable_connect(path: object, *args: object, **kwargs: object):
        if path == ":memory:":
            raise sqlite3.OperationalError("disabled")
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(policy_module.sqlite3, "connect", unavailable_connect)
    with pytest.raises(ExperimentInputError) as raised:
        policy_module.preflight_benchmark_capabilities()
    assert raised.value.code == "benchmark_capability_unavailable"


def test_registry_rejects_a_forged_same_valued_plain_string() -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    forged = _descriptor(BenchmarkArmKind.NO_MEMORY).model_copy(
        update={"kind": "no_memory"}
    )

    with pytest.raises(ExperimentInputError) as raised:
        build_benchmark_policy(forged)

    assert raised.value.code == "benchmark_policy_unsupported"


@pytest.mark.asyncio
async def test_current_version_must_belong_to_the_selected_owned_experience(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "mismatched-version.sqlite3"
    _clone_with_owned_records(clone_path)
    with sqlite3.connect(clone_path) as connection:
        connection.execute(
            "UPDATE experience_state SET current_version_id = ? "
            "WHERE experience_id = ?",
            ("version-3", "00000000-0000-4000-8000-000000000010"),
        )

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_benchmark_policy(
            _descriptor(BenchmarkArmKind.RECENT_NOTES)
        ).execute(_context(clone_path))

    assert raised.value.code == "benchmark_policy_execution_failed"


@pytest.mark.asyncio
async def test_bm25_keeps_writes_on_the_retained_clone_during_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    source_path = tmp_path / "authoritative.sqlite3"
    clone_path = tmp_path / "clone.sqlite3"
    _clone_with_owned_records(source_path)
    _clone_with_owned_records(clone_path)
    parked_clone = tmp_path / "parked-clone.sqlite3"
    original_connect = sqlite3.connect
    replaced = False

    def replace_before_open(
        database: object, *args: object, **kwargs: object
    ) -> sqlite3.Connection:
        nonlocal replaced
        if database != ":memory:" and not replaced:
            replaced = True
            os.replace(clone_path, parked_clone)
            clone_path.symlink_to(source_path)
            try:
                return original_connect(database, *args, **kwargs)
            finally:
                clone_path.unlink()
                os.replace(parked_clone, clone_path)
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(policy_module.sqlite3, "connect", replace_before_open)
    await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(_context(clone_path))

    with sqlite3.connect(source_path) as source:
        assert source.execute(
            "SELECT name FROM sqlite_master WHERE name = 'pilot_fts'"
        ).fetchone() is None
    assert replaced is True


@pytest.mark.asyncio
async def test_bm25_preflight_runs_off_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "preflight.sqlite3"
    _clone_with_owned_records(clone_path)
    loop_thread = threading.get_ident()
    called_from: list[int] = []

    def recording_preflight() -> None:
        called_from.append(threading.get_ident())
        time.sleep(0.01)

    monkeypatch.setattr(
        policy_module, "preflight_benchmark_capabilities", recording_preflight
    )
    await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(_context(clone_path))

    assert called_from and called_from != [loop_thread]


@pytest.mark.asyncio
async def test_bm25_drains_worker_before_double_cancellation_is_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "cancellation.sqlite3"
    _clone_with_owned_records(clone_path)
    started = threading.Event()
    release = threading.Event()
    original_bm25 = policy_module._sqlite_bm25

    def blocking_bm25(*args: object) -> object:
        started.set()
        release.wait()
        return original_bm25(*args)

    monkeypatch.setattr(policy_module, "_sqlite_bm25", blocking_bm25)
    task = asyncio.create_task(
        build_benchmark_policy(_descriptor(BenchmarkArmKind.SQLITE_BM25)).execute(
            _context(clone_path)
        )
    )
    try:
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "kind",
    (BenchmarkArmKind.RECENT_NOTES, BenchmarkArmKind.SQLITE_BM25),
)
@pytest.mark.asyncio
async def test_policy_connection_closes_before_lease_cleanup_on_ordinary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: BenchmarkArmKind,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / f"{kind.value}-ordinary-error.sqlite3"
    _clone_with_owned_records(clone_path)
    events: list[str] = []
    original_connect = sqlite3.connect
    original_verify = policy_module.require_same_policy_clone
    original_lease_close = policy_module.PolicyCloneLease.close

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *args: object) -> None:
            events.append("transaction-exit")
            return None

        def close(self) -> None:
            events.append("connection-close")

        def execute(
            self,
            statement: str,
            values: tuple[object, ...] = (),
        ) -> Connection:
            del statement, values
            raise sqlite3.DatabaseError("ordinary-error")

    connection = Connection()

    def connect(database: object, *args: object, **kwargs: object) -> object:
        if database == ":memory:":
            return original_connect(database, *args, **kwargs)
        return connection

    def verify(identity: object) -> None:
        events.append("lease-verify")
        original_verify(identity)

    def close(lease: object) -> None:
        events.append("lease-close")
        original_lease_close(lease)

    monkeypatch.setattr(policy_module.sqlite3, "connect", connect)
    monkeypatch.setattr(policy_module, "require_same_policy_clone", verify)
    monkeypatch.setattr(policy_module.PolicyCloneLease, "close", close)
    task = asyncio.create_task(
        build_benchmark_policy(_descriptor(kind)).execute(_context(clone_path))
    )
    task.add_done_callback(lambda _: events.append("coroutine-complete"))

    with pytest.raises(ExperimentIsolationError) as raised:
        await task
    await asyncio.sleep(0)

    assert raised.value.code == "benchmark_policy_execution_failed"
    assert events == [
        "transaction-exit",
        "connection-close",
        "lease-verify",
        "lease-close",
        "coroutine-complete",
    ]


@pytest.mark.parametrize(
    "kind",
    (BenchmarkArmKind.RECENT_NOTES, BenchmarkArmKind.SQLITE_BM25),
)
@pytest.mark.asyncio
async def test_cancelled_policy_connection_closes_before_lease_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: BenchmarkArmKind,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / f"{kind.value}-cancelled-error.sqlite3"
    _clone_with_owned_records(clone_path)
    events: list[str] = []
    started = threading.Event()
    release = threading.Event()
    original_connect = sqlite3.connect
    original_verify = policy_module.require_same_policy_clone
    original_lease_close = policy_module.PolicyCloneLease.close

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *args: object) -> None:
            events.append("transaction-exit")
            return None

        def close(self) -> None:
            events.append("connection-close")

        def execute(
            self,
            statement: str,
            values: tuple[object, ...] = (),
        ) -> Connection:
            del statement, values
            started.set()
            release.wait()
            raise sqlite3.DatabaseError("cancelled-error")

    connection = Connection()

    def connect(database: object, *args: object, **kwargs: object) -> object:
        if database == ":memory:":
            return original_connect(database, *args, **kwargs)
        return connection

    def verify(identity: object) -> None:
        events.append("lease-verify")
        original_verify(identity)

    def close(lease: object) -> None:
        events.append("lease-close")
        original_lease_close(lease)

    monkeypatch.setattr(policy_module.sqlite3, "connect", connect)
    monkeypatch.setattr(policy_module, "require_same_policy_clone", verify)
    monkeypatch.setattr(policy_module.PolicyCloneLease, "close", close)
    task = asyncio.create_task(
        build_benchmark_policy(_descriptor(kind)).execute(_context(clone_path))
    )
    task.add_done_callback(lambda _: events.append("coroutine-complete"))
    try:
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert events == [
        "transaction-exit",
        "connection-close",
        "lease-verify",
        "lease-close",
        "coroutine-complete",
    ]


@pytest.mark.asyncio
async def test_recent_notes_orders_created_desc_then_label_and_never_skips_budget(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import (
        BenchmarkPolicyContext,
        build_benchmark_policy,
    )

    clone_path = tmp_path / "prefix.sqlite3"
    _clone_with_owned_records(clone_path)
    with sqlite3.connect(clone_path) as connection:
        connection.execute(
            "UPDATE experiences SET created_at = ? WHERE experience_id = ?",
            ("2026-08-20T00:00:00Z", "00000000-0000-4000-8000-000000000010"),
        )
    index = BenchmarkSourceIndex(
        agent_ids=_source_index().agent_ids,
        experience_ids=_source_index().experience_ids,
        candidate_ids=_source_index().candidate_ids,
        labels_by_experience_id=_source_index().labels_by_experience_id,
        content_bytes_by_label=MappingProxyType(
            {"queue-required": 21, "queue-optional": 8}
        ),
    )
    context = BenchmarkPolicyContext(
        case=_case(),
        clone_path=clone_path,
        owner_agent_id=OWNER_ID,
        source_index=index,
        frozen_at=FROZEN_AT,
        seed=20260820,
    )
    observation = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.RECENT_NOTES)
    ).execute(context)

    assert observation.returned_labels == ()
    assert observation.selected_content_bytes == 0

    with sqlite3.connect(clone_path) as connection:
        connection.execute(
            "UPDATE experiences SET created_at = ? WHERE experience_id = ?",
            ("2026-08-19T00:00:00Z", "00000000-0000-4000-8000-000000000010"),
        )
    tie_index = BenchmarkSourceIndex(
        agent_ids=index.agent_ids,
        experience_ids=index.experience_ids,
        candidate_ids=index.candidate_ids,
        labels_by_experience_id=index.labels_by_experience_id,
        content_bytes_by_label=MappingProxyType(
            {"queue-required": 8, "queue-optional": 8}
        ),
    )
    tie_observation = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.RECENT_NOTES)
    ).execute(
        BenchmarkPolicyContext(
            case=_case(),
            clone_path=clone_path,
            owner_agent_id=OWNER_ID,
            source_index=tie_index,
            frozen_at=FROZEN_AT,
            seed=20260820,
        )
    )
    assert tie_observation.returned_labels == ("queue-optional", "queue-required")


@pytest.mark.asyncio
async def test_bm25_uses_field_weights_and_logical_label_ties(tmp_path: Path) -> None:
    from experience_hub.experiments.benchmarks.policies import (
        BenchmarkPolicyContext,
        build_benchmark_policy,
    )

    clone_path = tmp_path / "ranking.sqlite3"
    _clone_with_owned_records(clone_path)
    with sqlite3.connect(clone_path) as connection:
        connection.execute(
            "UPDATE experience_versions SET summary = ?, mechanism = ? "
            "WHERE version_id = ?",
            ("ordinary", "bounded recovery", "version-0"),
        )
        connection.execute(
            "UPDATE experience_versions SET summary = ?, mechanism = ? "
            "WHERE version_id = ?",
            ("bounded recovery", "ordinary", "version-1"),
        )
    weighted_case = _case().model_copy(
        update={"query": "bounded recovery", "tags": ("absent",), "mechanism_cues": ()}
    )
    weighted = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(
        BenchmarkPolicyContext(
            case=weighted_case,
            clone_path=clone_path,
            owner_agent_id=OWNER_ID,
            source_index=_source_index(),
            frozen_at=FROZEN_AT,
            seed=20260820,
        )
    )
    assert weighted.returned_labels == ("queue-optional",)

    with sqlite3.connect(clone_path) as connection:
        connection.execute("DROP TABLE pilot_fts")
        connection.execute(
            "UPDATE experience_versions SET summary = ?, mechanism = ? "
            "WHERE version_id IN (?, ?)",
            ("same", "ordinary", "version-0", "version-1"),
        )
    tie_case = weighted_case.model_copy(update={"query": "same"})
    tied = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.SQLITE_BM25)
    ).execute(
        BenchmarkPolicyContext(
            case=tie_case,
            clone_path=clone_path,
            owner_agent_id=OWNER_ID,
            source_index=_source_index(),
            frozen_at=FROZEN_AT,
            seed=20260820,
        )
    )
    assert tied.returned_labels == ("queue-optional",)


def test_preflight_uses_exact_fts_table_and_parameterized_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module

    statements: list[tuple[str, tuple[object, ...]]] = []

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(
            self,
            statement: str,
            values: tuple[object, ...] = (),
        ) -> Connection:
            statements.append((statement, values))
            return self

        def fetchall(self) -> list[tuple[str]]:
            return [("probe",)]

    monkeypatch.setattr(policy_module.sqlite3, "connect", lambda _: Connection())
    policy_module.preflight_benchmark_capabilities()

    assert statements[0][0] == policy_module._FTS_SQL
    assert "tokenize='unicode61 remove_diacritics 2'" in statements[0][0]
    assert statements[-1] == (
        "SELECT label FROM pilot_fts WHERE pilot_fts MATCH ?",
        ('"probe"',),
    )


@pytest.mark.asyncio
async def test_experience_hub_maps_all_labels_rejects_unmapped_and_uses_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experience_hub.experiments.benchmarks import policies as policy_module
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    observed_labels: list[tuple[str, ...]] = []

    async def mapped_execute(
        self: object, context: object
    ) -> ArmObservationV1:
        replay_case = context.case
        observed_labels.append(tuple(item.label for item in replay_case.expected))
        return ArmObservationV1(
            schema_version=1,
            returned_labels=("queue-required", "queue-optional"),
            unmapped_count=0,
        )

    monkeypatch.setattr(policy_module.ExperienceHubPolicyArm, "execute", mapped_execute)
    observation = await build_benchmark_policy(
        _descriptor(BenchmarkArmKind.EXPERIENCE_HUB)
    ).execute(_context(tmp_path / "must-not-open.sqlite3"))
    assert observed_labels == [("queue-optional", "queue-required")]
    assert observation.returned_labels == ("queue-required",)
    assert observation.selected_content_bytes == 12

    async def unmapped_execute(
        self: object, context: object
    ) -> ArmObservationV1:
        del self, context
        return ArmObservationV1(
            schema_version=1,
            returned_labels=(),
            unmapped_count=1,
        )

    monkeypatch.setattr(
        policy_module.ExperienceHubPolicyArm,
        "execute",
        unmapped_execute,
    )
    with pytest.raises(ExperimentInputError) as raised:
        await build_benchmark_policy(
            _descriptor(BenchmarkArmKind.EXPERIENCE_HUB)
        ).execute(_context(tmp_path / "must-not-open.sqlite3"))
    assert raised.value.code == "benchmark_oracle_invalid"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
@pytest.mark.asyncio
async def test_bm25_rejects_clone_aliases_without_opening_or_mutating_source(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    source_path = tmp_path / "source.sqlite3"
    _clone_with_owned_records(source_path)
    source_before = source_path.read_bytes()
    alias_path = tmp_path / "alias.sqlite3"
    if alias_kind == "symlink":
        alias_path.symlink_to(source_path)
    else:
        os.link(source_path, alias_path)

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_benchmark_policy(
            _descriptor(BenchmarkArmKind.SQLITE_BM25)
        ).execute(_context(alias_path))

    assert raised.value.code == "benchmark_policy_execution_failed"
    assert source_path.read_bytes() == source_before


@pytest.mark.asyncio
async def test_bm25_maps_runtime_index_failure_to_its_stable_local_error(
    tmp_path: Path,
) -> None:
    from experience_hub.experiments.benchmarks.policies import build_benchmark_policy

    clone_path = tmp_path / "broken-index.sqlite3"
    with sqlite3.connect(clone_path) as connection:
        connection.execute("CREATE TABLE experiences(experience_id TEXT)")

    with pytest.raises(ExperimentIsolationError) as raised:
        await build_benchmark_policy(
            _descriptor(BenchmarkArmKind.SQLITE_BM25)
        ).execute(_context(clone_path))

    assert raised.value.code == "benchmark_policy_execution_failed"
