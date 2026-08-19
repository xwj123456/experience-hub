from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from alembic.util import CommandError
from sqlalchemy.engine import make_url

import experience_hub.experiments.snapshots as snapshots
import experience_hub.runtime as runtime_module
from experience_hub.config import Settings
from experience_hub.experiments.errors import ExperimentIsolationError
from experience_hub.experiments.snapshots import (
    checkpoint_owned_sqlite,
    clone_frozen_sqlite,
    freeze_closed_sqlite,
    validate_frozen_snapshot,
    verify_source_unchanged,
)
from experience_hub.runtime import ApplicationRuntime, SchemaRevisionError

NOW = datetime(2026, 7, 26, 10, 11, 12, tzinfo=UTC)


def _settings(path: Path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}")


def _remove_empty_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            assert sidecar.stat().st_size == 0
            sidecar.unlink()


def _checkpoint_for_test(path: Path) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        assert connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone() == (0, 0, 0)
    _remove_empty_sidecars(path)


@pytest.fixture
async def seeded_database(tmp_path: Path) -> Path:
    path = tmp_path / "source.sqlite3"
    async with ApplicationRuntime(_settings(path)).initialize(
        start_lifecycle_worker=False,
        recover_interrupted=False,
    ):
        pass
    _checkpoint_for_test(path)
    return path


@pytest.mark.asyncio
async def test_freeze_and_validate_clone_exact_closed_database_bytes(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    source_bytes = seeded_database.read_bytes()

    snapshot = freeze_closed_sqlite(seeded_database)
    clone = clone_frozen_sqlite(snapshot, tmp_path / "clone.sqlite3")
    revision = await validate_frozen_snapshot(
        snapshot,
        validation_path=tmp_path / "validation.sqlite3",
        frozen_at=NOW,
        seed=7,
    )

    assert snapshot.source_path == seeded_database
    assert snapshot.database_bytes == source_bytes
    assert len(snapshot.database_sha256) == 64
    assert clone.read_bytes() == source_bytes
    assert revision == "0007_capture_evidence_hashes"
    assert seeded_database.read_bytes() == source_bytes


def test_freeze_never_opens_the_authoritative_source_with_sqlite(
    seeded_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def forbidden_connect(*args: object, **kwargs: object) -> object:
        calls.append(args)
        raise AssertionError("freeze must not invoke SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden_connect)

    snapshot = freeze_closed_sqlite(seeded_database)

    assert snapshot.database_bytes == seeded_database.read_bytes()
    assert calls == []


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_freeze_rejects_non_regular_sources(
    tmp_path: Path,
    kind: str,
) -> None:
    source = tmp_path / "source.sqlite3"
    if kind == "symlink":
        target = tmp_path / "target.sqlite3"
        target.write_bytes(b"sqlite")
        source.symlink_to(target)
    else:
        source.mkdir()

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(source)

    assert raised.value.code == "replay_snapshot_invalid"
    assert str(source) not in str(raised.value)


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
def test_freeze_rejects_nonempty_sqlite_sidecars(
    seeded_database: Path,
    suffix: str,
) -> None:
    Path(f"{seeded_database}{suffix}").write_bytes(b"not checkpointed")

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(seeded_database)

    assert raised.value.code == "replay_snapshot_invalid"


def test_freeze_rejects_sidecar_symlinks(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside-wal"
    target.write_bytes(b"")
    Path(f"{seeded_database}-wal").symlink_to(target)

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(seeded_database)

    assert raised.value.code == "replay_snapshot_invalid"


def test_verify_rejects_source_mutation_after_freeze(
    seeded_database: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    seeded_database.write_bytes(snapshot.database_bytes + b"mutated")

    with pytest.raises(ExperimentIsolationError) as raised:
        verify_source_unchanged(snapshot)

    assert raised.value.code == "replay_snapshot_changed"
    assert "mutated" not in str(raised.value)


SidecarMutation = Literal["create", "delete", "replace"]


def _mutate_empty_sidecar(
    sidecar: Path,
    *,
    mutation: SidecarMutation,
    replacement: Path,
) -> None:
    if mutation == "create":
        sidecar.write_bytes(b"")
        return
    if mutation == "delete":
        sidecar.unlink()
        return
    original_status = sidecar.lstat()
    replacement.write_bytes(b"")
    replacement.replace(sidecar)
    assert sidecar.lstat().st_ino != original_status.st_ino


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.parametrize("mutation", ("create", "delete", "replace"))
def test_verify_rejects_empty_sidecar_state_change_after_freeze(
    seeded_database: Path,
    tmp_path: Path,
    suffix: str,
    mutation: SidecarMutation,
) -> None:
    sidecar = Path(f"{seeded_database}{suffix}")
    if mutation != "create":
        sidecar.write_bytes(b"")
    snapshot = freeze_closed_sqlite(seeded_database)

    _mutate_empty_sidecar(
        sidecar,
        mutation=mutation,
        replacement=tmp_path / f"replacement{suffix}",
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        verify_source_unchanged(snapshot)

    assert raised.value.code == "replay_snapshot_changed"
    assert str(sidecar) not in str(raised.value)


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.parametrize("mutation", ("create", "delete", "replace"))
def test_freeze_rejects_empty_sidecar_race_between_descriptor_checks(
    seeded_database: Path,
    tmp_path: Path,
    suffix: str,
    mutation: SidecarMutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = Path(f"{seeded_database}{suffix}")
    if mutation != "create":
        sidecar.write_bytes(b"")
    original_lstat = Path.lstat
    sidecar_checks = 0

    def racing_lstat(path: Path) -> os.stat_result:
        nonlocal sidecar_checks
        if path == sidecar:
            sidecar_checks += 1
            if sidecar_checks == 2:
                _mutate_empty_sidecar(
                    sidecar,
                    mutation=mutation,
                    replacement=tmp_path / f"replacement{suffix}",
                )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", racing_lstat)

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(seeded_database)

    assert raised.value.code == "replay_snapshot_changed"
    assert str(sidecar) not in str(raised.value)


def test_clone_rejects_destination_symlink_without_touching_target(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    target = tmp_path / "outside.sqlite3"
    target.write_bytes(b"outside")
    destination = tmp_path / "clone.sqlite3"
    destination.symlink_to(target)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_invalid"
    assert target.read_bytes() == b"outside"


def test_clone_replaces_only_destination_and_its_sidecars(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"stale")
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{destination}{suffix}").write_bytes(b"")
    unrelated = destination.parent / "unrelated.sqlite3"
    unrelated.write_bytes(b"keep")

    clone_frozen_sqlite(snapshot, destination)

    assert destination.read_bytes() == snapshot.database_bytes
    assert unrelated.read_bytes() == b"keep"
    assert all(
        not Path(f"{destination}{suffix}").exists()
        for suffix in ("-wal", "-shm", "-journal")
    )


def test_clone_rejects_a_symlinked_parent_without_touching_external_files(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    external = tmp_path / "external"
    external.mkdir()
    external_target = external / "clone.sqlite3"
    external_target.write_bytes(b"outside")
    external_sidecar = Path(f"{external_target}-wal")
    external_sidecar.write_bytes(b"outside-sidecar")
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "redirect").symlink_to(external, target_is_directory=True)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(
            snapshot,
            owned / "redirect" / "clone.sqlite3",
        )

    assert raised.value.code == "replay_clone_invalid"
    assert external_target.read_bytes() == b"outside"
    assert external_sidecar.read_bytes() == b"outside-sidecar"


@pytest.mark.parametrize("kind", ("symlink", "fifo", "socket", "nonempty"))
def test_clone_rejects_unsafe_sidecars_without_deleting_them(
    seeded_database: Path,
    tmp_path: Path,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    sidecar = Path(f"{destination}-wal")
    outside = tmp_path / "outside-wal"
    if kind == "symlink":
        outside.write_bytes(b"outside")
        sidecar.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(sidecar)
    elif kind == "socket":
        sidecar.write_bytes(b"socket-placeholder")
        regular_status = sidecar.lstat()
        values = list(regular_status)
        values[0] = stat.S_IFSOCK | 0o600
        socket_status = os.stat_result(values)
        real_stat = os.stat

        def report_socket(
            path: os.PathLike[str] | str | int,
            *args: Any,
            **kwargs: Any,
        ) -> os.stat_result:
            if path == sidecar.name and kwargs.get("dir_fd") is not None:
                return socket_status
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(snapshots.os, "stat", report_socket)
    else:
        sidecar.write_bytes(b"polluted")
    before = sidecar.lstat()

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    after = sidecar.lstat()
    assert raised.value.code == "replay_clone_invalid"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    if kind == "symlink":
        assert outside.read_bytes() == b"outside"
    elif kind in {"socket", "nonempty"}:
        expected = b"socket-placeholder" if kind == "socket" else b"polluted"
        assert sidecar.read_bytes() == expected


def test_clone_partial_write_preserves_the_previous_destination(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    real_write = os.write
    calls = 0

    def partial_then_fail(fd: int, body: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, body[:17])
        raise OSError("private partial-write failure")

    monkeypatch.setattr(snapshots.os, "write", partial_then_fail)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_failed"
    assert destination.read_bytes() == b"previous"
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_clone_final_source_change_preserves_the_previous_destination(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    real_verify = snapshots.verify_source_unchanged
    calls = 0

    def mutate_before_final_verify(value: snapshots.FrozenSqliteSnapshot) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            seeded_database.write_bytes(value.database_bytes + b"changed")
        real_verify(value)

    monkeypatch.setattr(
        snapshots,
        "verify_source_unchanged",
        mutate_before_final_verify,
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_snapshot_changed"
    assert destination.read_bytes() == b"previous"
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_clone_rejects_a_sidecar_created_during_final_source_reverification(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    injected = Path(f"{destination}-wal")
    real_verify = snapshots.verify_source_unchanged
    calls = 0

    def inject_sidecar(value: snapshots.FrozenSqliteSnapshot) -> None:
        nonlocal calls
        calls += 1
        real_verify(value)
        if calls == 2:
            injected.write_bytes(b"injected")

    monkeypatch.setattr(snapshots, "verify_source_unchanged", inject_sidecar)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_invalid"
    assert destination.read_bytes() == b"previous"
    assert injected.read_bytes() == b"injected"


def test_clone_has_no_fallible_validation_after_atomic_publication(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    real_replace = os.replace
    real_stat = os.stat
    published = False

    def recording_replace(
        source: str,
        target: str,
        **kwargs: Any,
    ) -> None:
        nonlocal published
        real_replace(source, target, **kwargs)
        if target == destination.name:
            published = True

    def fail_post_publication_stat(
        path: os.PathLike[str] | str | int,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        if (
            published
            and path == destination.name
            and kwargs.get("dir_fd") is not None
        ):
            raise OSError("private post-publication stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(snapshots.os, "replace", recording_replace)
    monkeypatch.setattr(snapshots.os, "stat", fail_post_publication_stat)

    assert clone_frozen_sqlite(snapshot, destination) == destination
    assert destination.read_bytes() == snapshot.database_bytes


def _previous_clone_with_sidecars(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    destination = tmp_path / "owned" / "clone.sqlite3"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")
    sidecars = tuple(
        Path(f"{destination}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
    )
    for sidecar in sidecars:
        sidecar.write_bytes(b"")
    return destination, sidecars


def _assert_previous_clone_state(
    destination: Path,
    sidecars: tuple[Path, ...],
) -> None:
    assert destination.read_bytes() == b"previous"
    assert all(
        sidecar.is_file() and sidecar.read_bytes() == b""
        for sidecar in sidecars
    )
    assert {entry.name for entry in destination.parent.iterdir()} == {
        destination.name,
        *(sidecar.name for sidecar in sidecars),
    }


def test_clone_restores_target_and_sidecars_when_sidecar_staging_fails(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination, sidecars = _previous_clone_with_sidecars(tmp_path)
    real_replace = os.replace

    def fail_midway(
        source: str,
        target: str,
        **kwargs: Any,
    ) -> None:
        if source == sidecars[1].name:
            raise OSError("private sidecar staging failure")
        real_replace(source, target, **kwargs)

    monkeypatch.setattr(snapshots.os, "replace", fail_midway)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_failed"
    _assert_previous_clone_state(destination, sidecars)


def test_clone_restores_after_first_staged_backup_stat_fails(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination, sidecars = _previous_clone_with_sidecars(tmp_path)
    real_stat = os.stat
    failed = False

    def fail_first_backup_slot_stat(
        path: os.PathLike[str] | str | int,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        nonlocal failed
        if (
            not failed
            and path == "target"
            and kwargs.get("dir_fd") is not None
        ):
            failed = True
            raise FileNotFoundError("private staged backup stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(snapshots.os, "stat", fail_first_backup_slot_stat)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_failed"
    _assert_previous_clone_state(destination, sidecars)

    assert clone_frozen_sqlite(snapshot, destination) == destination
    assert destination.read_bytes() == snapshot.database_bytes


def test_clone_restores_target_and_sidecars_when_publication_replace_fails(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    destination, sidecars = _previous_clone_with_sidecars(tmp_path)
    real_replace = os.replace
    failed = False

    def fail_publication(
        source: str,
        target: str,
        **kwargs: Any,
    ) -> None:
        nonlocal failed
        if (
            not failed
            and source.endswith(".experience-hub-clone.tmp")
            and target == destination.name
        ):
            failed = True
            raise OSError("private publication failure")
        real_replace(source, target, **kwargs)

    monkeypatch.setattr(snapshots.os, "replace", fail_publication)

    with pytest.raises(ExperimentIsolationError) as raised:
        clone_frozen_sqlite(snapshot, destination)

    assert raised.value.code == "replay_clone_failed"
    _assert_previous_clone_state(destination, sidecars)


@pytest.mark.asyncio
async def test_snapshot_validation_rejects_old_schema_without_migrating(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    with closing(sqlite3.connect(seeded_database)) as connection:
        connection.execute(
            "UPDATE alembic_version SET version_num = ?",
            ("0006_capture_candidates",),
        )
        connection.commit()
    _checkpoint_for_test(seeded_database)
    before = seeded_database.read_bytes()
    snapshot = freeze_closed_sqlite(seeded_database)

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=7,
        )

    assert raised.value.code == "replay_schema_unsupported"
    assert SchemaRevisionError.code not in str(raised.value)
    assert seeded_database.read_bytes() == before


@pytest.mark.asyncio
async def test_validation_maps_clone_source_change_to_stable_source_error(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    seeded_database.write_bytes(snapshot.database_bytes + b"private mutation")

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=19,
        )

    assert raised.value.code == "replay_source_invalid"
    assert "mutation" not in str(raised.value)
    assert "snapshot_changed" not in str(raised.value)


@pytest.mark.asyncio
async def test_require_current_schema_rejects_old_schema_without_upgrade(
    seeded_database: Path,
) -> None:
    with closing(sqlite3.connect(seeded_database)) as connection:
        connection.execute(
            "UPDATE alembic_version SET version_num = ?",
            ("0006_capture_candidates",),
        )
        connection.commit()
    before = seeded_database.read_bytes()

    with pytest.raises(SchemaRevisionError):
        await snapshots.require_current_schema(_settings(seeded_database))

    assert seeded_database.read_bytes() == before


class _MultipleHeads:
    def get_current_head(self) -> str:
        raise CommandError("private multiple-head migration paths")


def _add_real_database_head(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "INSERT INTO alembic_version(version_num) VALUES (?)",
            ("0006_capture_candidates",),
        )
        connection.commit()
    _checkpoint_for_test(path)


@pytest.mark.asyncio
async def test_require_current_schema_maps_real_database_multiple_heads(
    seeded_database: Path,
) -> None:
    _add_real_database_head(seeded_database)
    before = seeded_database.read_bytes()

    with pytest.raises(SchemaRevisionError) as raised:
        await runtime_module.require_current_schema(_settings(seeded_database))

    assert "more than one head" not in str(raised.value)
    assert seeded_database.read_bytes() == before


@pytest.mark.asyncio
async def test_validation_maps_real_database_multiple_heads_to_schema_error(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    _add_real_database_head(seeded_database)
    snapshot = freeze_closed_sqlite(seeded_database)

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=31,
        )

    assert raised.value.code == "replay_schema_unsupported"
    assert "more than one head" not in str(raised.value)


@pytest.mark.asyncio
async def test_require_current_schema_maps_multiple_heads_to_schema_error(
    seeded_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.ScriptDirectory,
        "from_config",
        staticmethod(lambda _: _MultipleHeads()),
    )

    with pytest.raises(SchemaRevisionError) as raised:
        await runtime_module.require_current_schema(_settings(seeded_database))

    assert "multiple-head" not in str(raised.value)


@pytest.mark.asyncio
async def test_validation_maps_multiple_heads_to_stable_schema_error(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    monkeypatch.setattr(
        runtime_module.ScriptDirectory,
        "from_config",
        staticmethod(lambda _: _MultipleHeads()),
    )

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=23,
        )

    assert raised.value.code == "replay_schema_unsupported"
    assert "multiple-head" not in str(raised.value)


@pytest.mark.asyncio
async def test_validation_opens_runtime_only_for_disposable_clone(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    validation_path = tmp_path / "validation.sqlite3"
    real_runtime = snapshots.ApplicationRuntime
    runtime_paths: list[Path] = []

    def recording_runtime(*args: Any, **kwargs: Any) -> ApplicationRuntime:
        settings = args[0] if args else kwargs["settings"]
        assert isinstance(settings, Settings)
        assert settings.database_url is not None
        database = make_url(settings.database_url).database
        assert database is not None
        runtime_paths.append(Path(database))
        return real_runtime(*args, **kwargs)

    monkeypatch.setattr(snapshots, "ApplicationRuntime", recording_runtime)

    await validate_frozen_snapshot(
        snapshot,
        validation_path=validation_path,
        frozen_at=NOW,
        seed=11,
    )

    assert runtime_paths == [validation_path]
    assert snapshot.source_path not in runtime_paths


@pytest.mark.asyncio
async def test_validation_opens_exact_clone_when_path_contains_question_mark(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    snapshot = freeze_closed_sqlite(seeded_database)
    validation_path = tmp_path / "validation?exact.sqlite3"

    revision = await validate_frozen_snapshot(
        snapshot,
        validation_path=validation_path,
        frozen_at=NOW,
        seed=37,
    )

    assert revision == "0007_capture_evidence_hashes"
    assert validation_path.read_bytes() == snapshot.database_bytes
    assert not (tmp_path / "validation").exists()


@pytest.mark.asyncio
async def test_validation_maps_invalid_authoritative_source_to_stable_error(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    missing_receipt = "00000000-0000-0000-0000-000000000099"
    with closing(sqlite3.connect(seeded_database)) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "INSERT INTO domain_events("
            "aggregate_type, aggregate_id, sequence, event_type, payload, "
            "actor_agent_id, causation_id, occurred_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "agent",
                "00000000-0000-0000-0000-000000000001",
                1,
                "agent.created",
                b'{"private":"must not escape"}',
                None,
                missing_receipt,
                "2026-07-26T10:11:12.000000Z",
            ),
        )
        connection.commit()
    _checkpoint_for_test(seeded_database)
    snapshot = freeze_closed_sqlite(seeded_database)

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=13,
        )

    assert raised.value.code == "replay_source_invalid"
    assert missing_receipt not in str(raised.value)
    assert "private" not in str(raised.value)


@pytest.mark.asyncio
async def test_validation_maps_projection_mismatch_to_stable_error(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    with closing(sqlite3.connect(seeded_database)) as connection:
        connection.execute(
            "INSERT INTO projection_versions("
            "name, reducer_version, last_applied_event_id"
            ") VALUES (?, ?, ?)",
            ("experience_state", 1, 1),
        )
        connection.commit()
    _checkpoint_for_test(seeded_database)
    snapshot = freeze_closed_sqlite(seeded_database)

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=17,
        )

    assert raised.value.code == "replay_projection_mismatch"
    assert "experience_state" not in str(raised.value)


@pytest.mark.asyncio
async def test_validation_maps_reducer_version_to_projection_error(
    seeded_database: Path,
    tmp_path: Path,
) -> None:
    with closing(sqlite3.connect(seeded_database)) as connection:
        connection.execute(
            "INSERT INTO projection_versions("
            "name, reducer_version, last_applied_event_id"
            ") VALUES (?, ?, ?)",
            ("experience_state", 99, 0),
        )
        connection.commit()
    _checkpoint_for_test(seeded_database)
    snapshot = freeze_closed_sqlite(seeded_database)

    with pytest.raises(ExperimentIsolationError) as raised:
        await validate_frozen_snapshot(
            snapshot,
            validation_path=tmp_path / "validation.sqlite3",
            frozen_at=NOW,
            seed=29,
        )

    assert raised.value.code == "replay_projection_mismatch"
    assert "running version" not in str(raised.value)


def test_checkpoint_owned_sqlite_truncates_and_removes_empty_sidecars(
    seeded_database: Path,
) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{seeded_database}{suffix}").write_bytes(b"")

    assert checkpoint_owned_sqlite(seeded_database) == (0, 0, 0)
    assert all(
        not Path(f"{seeded_database}{suffix}").exists()
        for suffix in ("-wal", "-shm", "-journal")
    )


def test_freeze_detects_identity_replacement_after_fd_open(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(seeded_database.read_bytes())
    real_fstat = os.fstat
    calls = 0

    def replacing_fstat(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        status = real_fstat(fd)
        if calls == 1:
            seeded_database.replace(tmp_path / "original.sqlite3")
            replacement.replace(seeded_database)
        return status

    monkeypatch.setattr(snapshots.os, "fstat", replacing_fstat)

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(seeded_database)

    assert raised.value.code == "replay_snapshot_changed"


def test_freeze_rechecks_source_after_the_final_sidecar_scan(
    seeded_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(seeded_database.read_bytes())
    original_lstat = Path.lstat
    journal = Path(f"{seeded_database}-journal")
    journal_checks = 0

    def replacing_lstat(path: Path) -> os.stat_result:
        nonlocal journal_checks
        if path == journal:
            journal_checks += 1
            if journal_checks == 2:
                seeded_database.replace(tmp_path / "original.sqlite3")
                replacement.replace(seeded_database)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", replacing_lstat)

    with pytest.raises(ExperimentIsolationError) as raised:
        freeze_closed_sqlite(seeded_database)

    assert raised.value.code == "replay_snapshot_changed"
