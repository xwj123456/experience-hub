from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path, PurePosixPath
from threading import Event, Thread

import pytest

from experience_hub.experiments import workspace as workspace_module
from experience_hub.experiments.errors import ExperimentIsolationError
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    prepare_owned_workspace,
)


def test_isolation_error_exposes_a_stable_path_free_contract() -> None:
    error = ExperimentIsolationError("replay_workspace_unowned", "workspace unsafe")

    assert error.code == "replay_workspace_unowned"
    assert error.message == "workspace unsafe"
    assert str(error) == "replay_workspace_unowned: workspace unsafe"


def test_workspace_rejects_a_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "workspace"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert target.is_dir()


def test_workspace_rejects_a_file_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert root.read_text(encoding="utf-8") == "not a directory"


def test_workspace_rejects_a_nonempty_unmarked_directory(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    retained = root / "keep.txt"
    retained.write_text("keep", encoding="utf-8")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert retained.read_text(encoding="utf-8") == "keep"


def test_workspace_rejects_an_invalid_marker(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    marker = root / REPLAY_WORKSPACE_POLICY.marker_name
    marker.write_bytes(b"some other workspace\n")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert marker.read_bytes() == b"some other workspace\n"


def test_workspace_rejects_a_marked_directory_with_an_unknown_entry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / REPLAY_WORKSPACE_POLICY.marker_name).write_bytes(
        REPLAY_WORKSPACE_POLICY.marker_body
    )
    retained = root / "keep.txt"
    retained.write_text("keep", encoding="utf-8")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert retained.read_text(encoding="utf-8") == "keep"


def test_workspace_rejects_default_overwrite_of_owned_entries(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    owned = workspace.root / "artifacts"
    owned.mkdir()
    (owned / "previous.json").write_bytes(b"old")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_exists"
    assert (owned / "previous.json").read_bytes() == b"old"


def test_workspace_explicitly_replaces_exact_owned_entries(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    owned = workspace.root / "artifacts"
    owned.mkdir()
    (owned / "previous.json").write_bytes(b"old")

    prepared = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=True,
        allow_unmarked_empty=False,
    )

    assert prepared.root == root
    assert not owned.exists()
    assert (
        root / REPLAY_WORKSPACE_POLICY.marker_name
    ).read_bytes() == REPLAY_WORKSPACE_POLICY.marker_body


def test_workspace_explicitly_replaces_an_owned_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    owned = workspace.root / "artifacts"
    owned.write_bytes(b"previous artifact")

    prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=True,
        allow_unmarked_empty=False,
    )

    assert not owned.exists()


def test_workspace_cleanup_cannot_follow_a_replaced_root_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    (workspace.root / "artifacts").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "artifacts"
    retained.mkdir()
    keep = retained / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    marker = root / REPLAY_WORKSPACE_POLICY.marker_name
    original_read_bytes = Path.read_bytes
    original_open = os.open
    replaced = False

    def replace_root_after_marker_read(path: Path) -> bytes:
        nonlocal replaced
        body = original_read_bytes(path)
        if path == marker and not replaced:
            replaced = True
            root.rename(tmp_path / "displaced-workspace")
            root.symlink_to(outside, target_is_directory=True)
        return body

    monkeypatch.setattr(Path, "read_bytes", replace_root_after_marker_read)

    def replace_root_after_marker_open(
        path: str, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal replaced
        if path == marker.name and kwargs.get("dir_fd") is not None and not replaced:
            replaced = True
            root.rename(tmp_path / "displaced-workspace")
            root.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_root_after_marker_open)

    with pytest.raises(ExperimentIsolationError):
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert (root / "artifacts" / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_workspace_cleanup_rejects_a_replaced_owned_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    artifacts = workspace.root / "artifacts"
    artifacts.mkdir()
    (artifacts / "generated.json").write_bytes(b"generated")
    outside = tmp_path / "outside"
    outside.mkdir()
    user_artifacts = outside / "user-artifacts"
    user_artifacts.mkdir()
    keep = user_artifacts / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    original_remove = workspace_module._remove_owned_entry
    replaced = False

    def replace_owned_directory(*args: object) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            user_artifacts.rename(root / "artifacts")
        original_remove(*args)

    monkeypatch.setattr(
        workspace_module, "_remove_owned_entry", replace_owned_directory
    )

    with pytest.raises(ExperimentIsolationError):
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert (root / "artifacts" / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_workspace_cleanup_rejects_an_entry_replaced_before_fd_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "workspace"
    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    artifacts = workspace.root / "artifacts"
    artifacts.mkdir()
    (artifacts / "generated.json").write_bytes(b"generated")
    user_artifacts = tmp_path / "user-artifacts"
    user_artifacts.mkdir()
    (user_artifacts / "keep.txt").write_text("keep", encoding="utf-8")
    original_open_directory = workspace_module._open_directory
    replaced = False
    artifact_opens = 0

    def replace_before_open(name: str, parent_fd: int) -> int:
        nonlocal artifact_opens, replaced
        if name == "artifacts":
            artifact_opens += 1
        if name == "artifacts" and artifact_opens == 2 and not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            user_artifacts.rename(root / "artifacts")
        return original_open_directory(name, parent_fd)

    monkeypatch.setattr(workspace_module, "_open_directory", replace_before_open)

    with pytest.raises(ExperimentIsolationError):
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert (root / "artifacts" / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("existing", (False, True))
def test_workspace_adoption_rejects_data_injected_before_marker_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bool
) -> None:
    root = tmp_path / "workspace"
    if existing:
        root.mkdir()
    marker = root / REPLAY_WORKSPACE_POLICY.marker_name
    original_open = os.open
    injected = False

    def inject_before_marker_openat(
        path: str, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal injected
        if (
            flags & os.O_CREAT
            and kwargs.get("dir_fd") is not None
            and not injected
        ):
            injected = True
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "keep.txt").write_text("keep", encoding="utf-8")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", inject_before_marker_openat)

    with pytest.raises(ExperimentIsolationError):
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=True,
        )

    assert (root / "artifacts" / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not marker.exists()


def test_workspace_never_deletes_unknown_user_data(tmp_path: Path) -> None:
    root = tmp_path / "user-data"
    root.mkdir()
    retained = root / "keep.txt"
    retained.write_text("keep", encoding="utf-8")

    with pytest.raises(ExperimentIsolationError) as captured:
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )

    assert captured.value.code == "replay_workspace_unowned"
    assert retained.read_text(encoding="utf-8") == "keep"


def test_workspace_allows_an_explicitly_adopted_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    workspace = prepare_owned_workspace(
        root,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )

    assert workspace.root == root
    assert (
        root / REPLAY_WORKSPACE_POLICY.marker_name
    ).read_bytes() == REPLAY_WORKSPACE_POLICY.marker_body


def test_atomic_write_persists_exact_bytes_after_readback(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )

    written = workspace.atomic_write(
        PurePosixPath("artifacts/evidence.json"),
        b'{"data":{"schema_version":1}}',
    )

    assert written == workspace.root / "artifacts" / "evidence.json"
    assert (
        workspace.root / "artifacts" / "evidence.json"
    ).read_bytes() == b'{"data":{"schema_version":1}}'
    assert not list((workspace.root / "artifacts").glob("*.tmp"))


def test_atomic_write_replaces_an_existing_artifact(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("artifacts/evidence.json")

    workspace.atomic_write(relative, b"old")
    workspace.atomic_write(relative, b"new")

    assert (workspace.root / relative).read_bytes() == b"new"


def _benchmark_artifact_bodies(
    *, include_profile: bool = True
) -> dict[PurePosixPath, bytes | None]:
    return {
        PurePosixPath("artifacts/benchmark-evidence.json"): b"evidence",
        PurePosixPath("artifacts/benchmark-summary.json"): b"summary",
        PurePosixPath("artifacts/profile.json"): (
            b"profile" if include_profile else None
        ),
    }


@pytest.mark.parametrize("failure_position", (1, 2, 3))
def test_group_publication_cleans_every_staged_member_failure(
    failure_position: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    calls = 0
    real_write = workspace_module._write_and_read_back

    def fail_staged_write(file_fd: int, body: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise OSError("injected staged-member failure")
        real_write(file_fd, body)

    monkeypatch.setattr(workspace_module, "_write_and_read_back", fail_staged_write)
    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_group_publication_rejects_existing_artifacts_without_replacing_them(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    first = workspace.atomic_write_group(_benchmark_artifact_bodies())
    previous = {
        path: (path.read_bytes(), path.stat().st_ino)
        for path in first.values()
        if path is not None
    }

    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies(include_profile=False))

    assert {
        path: (path.read_bytes(), path.stat().st_ino) for path in previous
    } == previous


def test_group_publication_requires_the_single_artifacts_parent(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )

    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(
            {
                PurePosixPath("artifacts/benchmark-evidence.json"): b"evidence",
                PurePosixPath("validation/benchmark-summary.json"): b"summary",
            }
        )

    assert not (workspace.root / "artifacts").exists()
    assert not (workspace.root / "validation").exists()


def test_group_publication_cleans_postwrite_fsync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    calls = 0
    real_fsync = os.fsync

    def fail_stage_fsync(file_fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected stage fsync failure")
        real_fsync(file_fd)

    monkeypatch.setattr(os, "fsync", fail_stage_fsync)
    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert not (workspace.root / "artifacts").exists()


def test_group_publication_cleans_postwrite_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise OSError("injected stage validation failure")

    monkeypatch.setattr(
        workspace_module, "_validate_benchmark_stage", fail_validation
    )
    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_group_publication_cleans_staging_fstat_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    calls = 0
    real_fstat = os.fstat

    def fail_member_fstat(file_fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected member fstat failure")
        return real_fstat(file_fd)

    monkeypatch.setattr(os, "fstat", fail_member_fstat)
    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert not (workspace.root / "artifacts").exists()


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, asyncio.CancelledError))
def test_group_publication_cleans_final_rename_interruption(
    interruption: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    real_rename = workspace_module._rename_directory_noreplace

    def interrupt_stage_rename(
        source: str,
        source_directory_fd: int,
        destination: str,
        destination_directory_fd: int,
    ) -> None:
        if source.startswith(".experience-hub-benchmark-stage-"):
            raise interruption("injected stage interruption")
        real_rename(source, source_directory_fd, destination, destination_directory_fd)

    monkeypatch.setattr(
        workspace_module, "_rename_directory_noreplace", interrupt_stage_rename
    )
    with pytest.raises(interruption):
        workspace.atomic_write_group(_benchmark_artifact_bodies(include_profile=False))

    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_group_publication_cleans_final_rename_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    real_rename = workspace_module._rename_directory_noreplace

    def fail_stage_rename(
        source: str,
        source_directory_fd: int,
        destination: str,
        destination_directory_fd: int,
    ) -> None:
        if source.startswith(".experience-hub-benchmark-stage-"):
            raise OSError("injected stage rename failure")
        real_rename(source, source_directory_fd, destination, destination_directory_fd)

    monkeypatch.setattr(
        workspace_module, "_rename_directory_noreplace", fail_stage_rename
    )
    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write_group(_benchmark_artifact_bodies(include_profile=False))

    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_group_publication_never_replaces_a_racing_empty_artifacts_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    existing = getattr(workspace_module, "_rename_directory_noreplace", None)
    injected = workspace.root / "artifacts"

    def create_racing_destination(
        source: str,
        source_directory_fd: int,
        destination: str,
        destination_directory_fd: int,
    ) -> None:
        injected.mkdir()
        (injected / "racing-sentinel").write_bytes(b"leave-this-directory")
        if existing is None:
            raise AssertionError("atomic no-replace wrapper was not implemented")
        existing(
            source,
            source_directory_fd,
            destination,
            destination_directory_fd,
        )

    monkeypatch.setattr(
        workspace_module,
        "_rename_directory_noreplace",
        create_racing_destination,
        raising=False,
    )
    with pytest.raises(ExperimentIsolationError) as captured:
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert captured.value.code == "replay_workspace_write_failed"
    assert (injected / "racing-sentinel").read_bytes() == b"leave-this-directory"
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, asyncio.CancelledError))
def test_group_publication_cleans_stage_created_before_mkdir_interruption(
    interruption: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    real_mkdir = os.mkdir

    def create_then_interrupt(
        name: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name.startswith(".experience-hub-benchmark-stage-"):
            raise interruption("injected mkdir interruption")

    monkeypatch.setattr(os, "mkdir", create_then_interrupt)
    with pytest.raises(interruption):
        workspace.atomic_write_group(_benchmark_artifact_bodies(include_profile=False))

    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_group_cleanup_refuses_a_stage_replaced_after_identity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    validation = workspace.root / "validation"
    validation.mkdir()
    stage_name = ".experience-hub-benchmark-stage-test"
    stage = validation / stage_name
    stage.mkdir()
    (stage / "benchmark-evidence.json").write_bytes(b"original-stage")
    stage_status = os.stat(stage, follow_symlinks=False)
    recovered = validation / "recoverable-stage"
    replacement = validation / stage_name
    real_open_directory = workspace_module._open_directory

    def replace_before_open(name: str, parent_fd: int) -> int:
        if name == stage_name:
            stage.rename(recovered)
            replacement.mkdir()
            (replacement / "benchmark-evidence.json").write_bytes(
                b"replacement-stage"
            )
        return real_open_directory(name, parent_fd)

    monkeypatch.setattr(workspace_module, "_open_directory", replace_before_open)
    validation_fd = os.open(validation, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ExperimentIsolationError):
            workspace_module._cleanup_benchmark_stage(
                stage_name,
                validation_fd,
                stage_status,
                frozenset({"benchmark-evidence.json"}),
            )
    finally:
        os.close(validation_fd)

    assert (
        replacement / "benchmark-evidence.json"
    ).read_bytes() == b"replacement-stage"
    assert (recovered / "benchmark-evidence.json").read_bytes() == b"original-stage"


def test_group_publication_reports_recoverable_stage_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=True,
    )
    fsync_calls = 0
    real_fsync = os.fsync
    real_unlink = os.unlink

    def fail_stage_fsync(file_fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 4:
            raise OSError("injected stage fsync failure")
        real_fsync(file_fd)

    def retain_staging_member(
        name: str, *args: object, **kwargs: object
    ) -> None:
        if name == "benchmark-evidence.json":
            raise OSError("injected cleanup failure")
        real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_stage_fsync)
    monkeypatch.setattr(os, "unlink", retain_staging_member)
    with pytest.raises(ExperimentIsolationError) as captured:
        workspace.atomic_write_group(_benchmark_artifact_bodies())

    assert captured.value.code == "replay_workspace_stage_cleanup_failed"
    assert not (workspace.root / "artifacts").exists()
    staging = workspace.root / "validation"
    assert staging.exists() and tuple(staging.iterdir())


def test_atomic_write_waits_for_a_cooperating_workspace_lock(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    root_fd = os.open(workspace.root, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(root_fd, fcntl.LOCK_EX)
    entered = Event()
    completed = Event()

    def write_in_another_thread() -> None:
        entered.set()
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"data")
        completed.set()

    worker = Thread(target=write_in_another_thread)
    worker.start()
    assert entered.wait(timeout=1)
    assert not completed.wait(timeout=0.05)
    fcntl.flock(root_fd, fcntl.LOCK_UN)
    worker.join(timeout=1)
    assert completed.is_set()
    os.close(root_fd)


def test_atomic_write_preserves_no_target_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    temporary_name = "evidence.json.experience-hub.tmp"
    original_replace = os.replace

    def reject_replace(
        source: str, destination: str, *args: object, **kwargs: object
    ) -> None:
        if source == temporary_name:
            raise OSError("injected unlink failure")
        original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", reject_replace)

    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"new")

    assert not (workspace.root / "artifacts" / "evidence.json").exists()


def test_atomic_write_rechecks_the_workspace_marker(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    (workspace.root / REPLAY_WORKSPACE_POLICY.marker_name).unlink()

    with pytest.raises(ExperimentIsolationError) as captured:
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"data")

    assert captured.value.code == "replay_workspace_unowned"
    assert not (workspace.root / "artifacts").exists()


def test_scoped_reservation_holds_workspace_lock_and_commits_reserved_file(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("snapshot/source.sqlite3")

    entered = Event()
    completed = Event()

    def replace_in_another_thread() -> None:
        entered.set()
        prepare_owned_workspace(
            workspace.root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )
        completed.set()

    with workspace.reserve_new_file_scoped(relative) as reservation:
        assert reservation.path == workspace.root / "snapshot" / "source.sqlite3"
        worker = Thread(target=replace_in_another_thread)
        worker.start()
        assert entered.wait(timeout=1)
        assert not completed.wait(timeout=0.05)
        reservation.commit()
    worker.join(timeout=1)

    assert completed.is_set()
    assert not (workspace.root / relative).exists()


def test_scoped_reservation_rejects_replacement_and_preserves_replacement_file(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("snapshot/source.sqlite3")

    with (
        pytest.raises(ExperimentIsolationError),
        workspace.reserve_new_file_scoped(relative) as reservation,
    ):
        reservation.path.unlink()
        reservation.path.write_bytes(b"replacement")
        reservation.verify()

    assert (workspace.root / relative).read_bytes() == b"replacement"


@pytest.mark.parametrize("replacement", ("root", "parent"))
def test_scoped_reservation_detects_replaced_authority_without_removing_new_file(
    tmp_path: Path, replacement: str
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("snapshot/source.sqlite3")

    with (
        pytest.raises(ExperimentIsolationError),
        workspace.reserve_new_file_scoped(relative) as reservation,
    ):
        reservation.path.with_name("source.sqlite3-wal").write_bytes(b"sidecar")
        if replacement == "root":
            workspace.root.rename(tmp_path / "displaced-workspace")
            workspace.root.mkdir()
            (workspace.root / REPLAY_WORKSPACE_POLICY.marker_name).write_bytes(
                REPLAY_WORKSPACE_POLICY.marker_body
            )
            (workspace.root / "snapshot").mkdir()
        else:
            (workspace.root / "snapshot").rename(tmp_path / "displaced-snapshot")
            (workspace.root / "snapshot").mkdir()
        replacement_file = workspace.root / relative
        replacement_file.write_bytes(b"replacement")
        reservation.verify()

    assert (workspace.root / relative).read_bytes() == b"replacement"
    displaced = tmp_path / (
        "displaced-workspace" if replacement == "root" else "displaced-snapshot"
    )
    original_parent = displaced / "snapshot" if replacement == "root" else displaced
    assert not (original_parent / "source.sqlite3").exists()
    assert not (original_parent / "source.sqlite3-wal").exists()


def test_async_scoped_reservation_does_not_block_holder_progress(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("snapshot/source.sqlite3")

    async def scenario() -> None:
        holder_ready = asyncio.Event()
        waiter_started = asyncio.Event()
        holder_progressed = asyncio.Event()

        async def holder() -> None:
            with await workspace.reserve_new_file_scoped_async(relative):
                holder_ready.set()
                await waiter_started.wait()
                await asyncio.sleep(0)
                holder_progressed.set()

        async def waiter() -> None:
            await holder_ready.wait()
            waiter_started.set()
            with await workspace.reserve_new_file_scoped_async(relative) as reservation:
                reservation.commit()

        holder_task = asyncio.create_task(holder())
        waiter_task = asyncio.create_task(waiter())
        await asyncio.wait_for(holder_progressed.wait(), timeout=1)
        await asyncio.wait_for(holder_task, timeout=1)
        await asyncio.wait_for(waiter_task, timeout=1)

    asyncio.run(scenario())
    assert (workspace.root / relative).exists()


def test_async_reservation_repeated_cancellation_waits_for_acquisition_cleanup(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    relative = PurePosixPath("snapshot/source.sqlite3")

    async def scenario() -> None:
        holder = await workspace.reserve_new_file_scoped_async(relative)
        waiter = asyncio.create_task(
            workspace.reserve_new_file_scoped_async(relative)
        )
        try:
            await asyncio.sleep(0)
            waiter.cancel("first cancellation")
            await asyncio.sleep(0)
            assert not waiter.done()
            waiter.cancel("second cancellation")
            await asyncio.sleep(0)
            assert not waiter.done()
        finally:
            holder.rollback()
            holder.close()
            with pytest.raises(asyncio.CancelledError) as captured:
                await waiter
            assert captured.value.args == ("first cancellation",)

    asyncio.run(scenario())
    assert not (workspace.root / "snapshot").exists()


def test_workspace_policy_requirement_rejects_a_different_marker_policy(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )

    with pytest.raises(ExperimentIsolationError):
        workspace.require_policy(
            workspace_module.WorkspacePolicy(
                marker_name=".other-marker",
                marker_body=b"other\n",
                owned_entries=frozenset({"snapshot"}),
            )
        )


def test_atomic_write_cannot_follow_a_replaced_parent_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    artifacts = workspace.root / "artifacts"
    artifacts.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "evidence.json"
    retained.write_bytes(b"keep")
    temporary = artifacts / "evidence.json.experience-hub.tmp"
    original_exists = Path.exists
    original_open = os.open
    replaced = False

    def replace_parent_before_temporary_open(path: Path) -> bool:
        nonlocal replaced
        exists = original_exists(path)
        if path == temporary and not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            artifacts.symlink_to(outside, target_is_directory=True)
        return exists

    monkeypatch.setattr(Path, "exists", replace_parent_before_temporary_open)

    def replace_parent_before_temporary_openat(
        path: str, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal replaced
        if path == temporary.name and kwargs.get("dir_fd") is not None and not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            artifacts.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_parent_before_temporary_openat)

    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"new")

    assert retained.read_bytes() == b"keep"
    assert not (tmp_path / "displaced-artifacts" / "evidence.json").exists()


def test_atomic_write_removes_its_final_artifact_after_a_post_commit_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    artifacts = workspace.root / "artifacts"
    artifacts.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    temporary_name = "evidence.json.experience-hub.tmp"
    original_replace = os.replace
    original_link = os.link
    replaced = False

    def replace_parent_after_commit(
        source: str, destination: str, *args: object, **kwargs: object
    ) -> None:
        nonlocal replaced
        original_replace(source, destination, *args, **kwargs)
        if source == temporary_name and not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            artifacts.symlink_to(outside, target_is_directory=True)

    def link_parent_after_commit(
        source: str, destination: str, *args: object, **kwargs: object
    ) -> None:
        nonlocal replaced
        original_link(source, destination, *args, **kwargs)
        if source == temporary_name and not replaced:
            replaced = True
            artifacts.rename(tmp_path / "displaced-artifacts")
            artifacts.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(os, "replace", replace_parent_after_commit)
    monkeypatch.setattr(os, "link", link_parent_after_commit)

    with pytest.raises(ExperimentIsolationError):
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"new")

    assert not (tmp_path / "displaced-artifacts" / "evidence.json").exists()
    assert not list(outside.iterdir())


def test_workspace_marker_creation_cannot_follow_a_replaced_marker_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside-marker"
    outside.write_bytes(b"keep")
    marker = root / REPLAY_WORKSPACE_POLICY.marker_name
    original_write_bytes = Path.write_bytes
    original_open = os.open
    replaced = False

    def replace_marker_before_write(path: Path, body: bytes) -> int:
        if path == marker:
            marker.symlink_to(outside)
        return original_write_bytes(path, body)

    monkeypatch.setattr(Path, "write_bytes", replace_marker_before_write)

    def replace_marker_before_openat(
        path: str, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal replaced
        if (
                flags & os.O_CREAT
                and kwargs.get("dir_fd") is not None
            and not replaced
        ):
            replaced = True
            marker.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_marker_before_openat)

    with pytest.raises(ExperimentIsolationError):
        prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=True,
        )

    assert outside.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "relative",
    (PurePosixPath("../evidence.json"), PurePosixPath("/evidence.json")),
)
def test_atomic_write_rejects_traversal_and_absolute_paths(
    tmp_path: Path, relative: PurePosixPath
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )

    with pytest.raises(ExperimentIsolationError) as captured:
        workspace.atomic_write(relative, b"data")

    assert captured.value.code == "replay_workspace_path_invalid"
    assert {entry.name for entry in workspace.root.iterdir()} == {
        REPLAY_WORKSPACE_POLICY.marker_name
    }


def test_atomic_write_rejects_a_symlink_parent_without_temp_artifacts(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "workspace",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    target = tmp_path / "target"
    target.mkdir()
    (workspace.root / "artifacts").symlink_to(target, target_is_directory=True)

    with pytest.raises(ExperimentIsolationError) as captured:
        workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), b"data")

    assert captured.value.code == "replay_workspace_path_invalid"
    assert not list(target.iterdir())
    assert not list(workspace.root.glob("*.tmp"))
