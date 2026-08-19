from __future__ import annotations

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
