from __future__ import annotations

import fcntl
import importlib
import os
import stat
from pathlib import Path
from threading import Event, Thread
from types import ModuleType
from uuid import UUID

import pytest
from tests.passport_fixtures import passport_document

from experience_hub.passports import encode_passport_document
from experience_hub.passports.errors import PassportError


def _files() -> ModuleType:
    try:
        return importlib.import_module("experience_hub.passports.files")
    except ModuleNotFoundError:
        pytest.fail("The safe Passport filesystem adapter is not implemented")


def _body() -> bytes:
    return encode_passport_document(passport_document())


def test_read_verifies_a_real_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.passport.json"
    path.write_bytes(_body())

    verified = _files().read_passport_file(path)

    assert verified.canonical_bytes == path.read_bytes()
    assert verified.report.embedded_excerpt_count == 1
    assert verified.report.publisher_identity == "unverified"


@pytest.mark.parametrize("node", ["symlink", "directory", "fifo", "missing"])
def test_read_refuses_nonregular_or_missing_nodes(tmp_path: Path, node: str) -> None:
    path = tmp_path / "input.passport.json"
    if node == "symlink":
        source = tmp_path / "source"
        source.write_bytes(_body())
        path.symlink_to(source)
    elif node == "directory":
        path.mkdir()
    elif node == "fifo":
        os.mkfifo(path)

    with pytest.raises(PassportError) as captured:
        _files().read_passport_file(path)

    assert captured.value.code == "passport_file_invalid"
    assert str(tmp_path) not in str(captured.value)


def test_read_refuses_a_symlink_directory_chain(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "input").write_bytes(_body())
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(PassportError) as captured:
        _files().read_passport_file(linked / "input")

    assert captured.value.code == "passport_file_invalid"


def test_read_budget_is_bounded_and_oversize_is_not_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "huge"
    path.write_bytes(b"x" * 600_000)
    requested: list[int] = []
    real_read = os.read

    def read(descriptor: int, count: int) -> bytes:
        requested.append(count)
        return real_read(descriptor, count)

    monkeypatch.setattr(_files().os, "read", read)
    with pytest.raises(PassportError) as captured:
        _files().read_passport_file(path)

    assert captured.value.code == "passport_size_limit"
    assert sum(requested) <= 524_289
    assert path.stat().st_size == 600_000


def test_publish_is_private_exact_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "output.passport.json"
    body = _body()

    _files().publish_passport_file(path, body)
    identity = path.stat().st_ino
    _files().publish_passport_file(path, body)

    assert path.read_bytes() == body
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_ino == identity
    assert tuple(tmp_path.iterdir()) == (path,)


def test_publish_preserves_existing_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    path.write_bytes(b"previous user file")
    identity = path.stat().st_ino

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_output_conflict"
    assert path.read_bytes() == b"previous user file"
    assert path.stat().st_ino == identity
    assert tuple(tmp_path.iterdir()) == (path,)


def test_relative_paths_do_not_require_symlink_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("relative.passport.json")

    _files().publish_passport_file(path, _body())

    assert _files().read_passport_file(path).canonical_bytes == _body()


def test_read_delegates_invalid_document_to_the_codec(tmp_path: Path) -> None:
    path = tmp_path / "invalid"
    path.write_bytes(b"{}")

    with pytest.raises(PassportError) as captured:
        _files().read_passport_file(path)

    assert captured.value.code == "passport_invalid"


@pytest.mark.parametrize("mutation", ["identity", "size", "same_size_content"])
def test_read_refuses_a_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / "input"
    path.write_bytes(_body())
    real_read = os.read
    changed = False

    def read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        retained = real_read(descriptor, count)
        if not changed:
            changed = True
            if mutation == "identity":
                path.rename(tmp_path / "original")
                path.write_bytes(_body())
            elif mutation == "size":
                with path.open("ab") as stream:
                    stream.write(b"x")
            else:
                path.write_bytes(b"x" * len(_body()))
        return retained

    monkeypatch.setattr(_files().os, "read", read)

    with pytest.raises(PassportError) as captured:
        _files().read_passport_file(path)

    assert captured.value.code == "passport_file_invalid"


@pytest.mark.parametrize("operation", ["read", "publish"])
@pytest.mark.parametrize("ancestor", [False, True])
def test_parent_directory_identity_changes_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    ancestor: bool,
) -> None:
    outer = tmp_path / "outer"
    parent = outer / "inner"
    parent.mkdir(parents=True)
    path = parent / "passport"
    if operation == "read":
        path.write_bytes(_body())
    replaced = outer if ancestor else parent
    real_read = os.read
    real_write = os.write
    changed = False

    def replace_parent() -> None:
        nonlocal changed
        if not changed:
            changed = True
            replaced.rename(tmp_path / "retained-directory")
            parent.mkdir(parents=True)
            path.write_bytes(b"competing parent file")

    def read(descriptor: int, count: int) -> bytes:
        result = real_read(descriptor, count)
        replace_parent()
        return result

    def write(descriptor: int, body: bytes | memoryview) -> int:
        result = real_write(descriptor, body)
        replace_parent()
        return result

    module = _files()
    monkeypatch.setattr(
        module.os,
        "read" if operation == "read" else "write",
        read if operation == "read" else write,
    )

    with pytest.raises(PassportError) as captured:
        if operation == "read":
            module.read_passport_file(path)
        else:
            module.publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert path.read_bytes() == b"competing parent file"


@pytest.mark.parametrize(
    "node",
    ["symlink", "directory", "fifo", "parent_link", "parent_missing", "traversal"],
)
def test_publish_refuses_unsafe_nodes_without_touching_user_files(
    tmp_path: Path, node: str
) -> None:
    retained = tmp_path / "retained"
    retained.write_bytes(b"user data")
    path = tmp_path / "output"
    if node == "symlink":
        path.symlink_to(retained)
    elif node == "directory":
        path.mkdir()
    elif node == "fifo":
        os.mkfifo(path)
    elif node == "parent_link":
        linked = tmp_path / "linked"
        linked.symlink_to(tmp_path, target_is_directory=True)
        path = linked / "output"
    elif node == "parent_missing":
        path = tmp_path / "missing" / "output"
    else:
        path = tmp_path / ".." / tmp_path.name / "output"

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert retained.read_bytes() == b"user data"
    assert not tuple(tmp_path.glob(".passport-*.tmp"))


def test_publication_race_preserves_competing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    real_link = os.link

    def link(source: str, target: str, **kwargs: object) -> None:
        path.write_bytes(b"competing publisher")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(_files().os, "link", link)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_output_conflict"
    assert path.read_bytes() == b"competing publisher"
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize(
    "failure", ["write", "zero_write", "fsync", "readback", "publication"]
)
def test_publication_failures_clean_only_the_owned_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "output"
    retained = tmp_path / "retained"
    retained.write_bytes(b"user data")
    module = _files()
    real_read = os.read

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected private path and body must not leak")

    def zero(descriptor: int, body: bytes | memoryview) -> int:
        return 0

    def corrupt_read(descriptor: int, count: int) -> bytes:
        body = real_read(descriptor, count)
        return b"x" + body[1:] if body else body

    if failure in {"write", "fsync", "publication"}:
        monkeypatch.setattr(
            module.os, {"publication": "link"}.get(failure, failure), fail
        )
    elif failure == "zero_write":
        monkeypatch.setattr(module.os, "write", zero)
    else:
        monkeypatch.setattr(module.os, "read", corrupt_read)

    with pytest.raises(PassportError) as captured:
        module.publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert "injected" not in str(captured.value)
    assert not path.exists()
    assert tuple(tmp_path.iterdir()) == (retained,)
    assert retained.read_bytes() == b"user data"


def test_short_reads_and_writes_preserve_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    real_read = os.read
    real_write = os.write

    def read(descriptor: int, count: int) -> bytes:
        return real_read(descriptor, min(count, 17))

    def write(descriptor: int, body: bytes | memoryview) -> int:
        return real_write(descriptor, body[:17])

    module = _files()
    monkeypatch.setattr(module.os, "read", read)
    monkeypatch.setattr(module.os, "write", write)
    module.publish_passport_file(path, _body())

    assert module.read_passport_file(path).canonical_bytes == _body()
    assert tuple(tmp_path.iterdir()) == (path,)


def test_cleanup_preserves_a_replaced_unowned_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    retained = tmp_path / "owned-retained"
    replaced: list[Path] = []

    def link(source: str, target: str, **kwargs: object) -> None:
        temporary = tmp_path / source
        temporary.rename(retained)
        temporary.write_bytes(b"unowned replacement")
        replaced.append(temporary)
        raise OSError("injected publication failure")

    monkeypatch.setattr(_files().os, "link", link)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert not path.exists()
    assert len(replaced) == 1
    assert replaced[0].read_bytes() == b"unowned replacement"
    assert retained.read_bytes() == _body()


def test_cleanup_failure_is_path_free_and_preserves_user_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    user = tmp_path / "user"
    user.write_bytes(b"user data")

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected filesystem path")

    module = _files()
    monkeypatch.setattr(module.os, "link", fail)
    monkeypatch.setattr(module.os, "unlink", fail)

    with pytest.raises(PassportError) as captured:
        module.publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert "injected" not in str(captured.value)
    assert not path.exists()
    assert user.read_bytes() == b"user data"
    assert len(tuple(tmp_path.glob(".passport-*.tmp"))) == 1


def test_interruption_cleans_the_owned_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"

    def interrupt(descriptor: int, body: bytes | memoryview) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(_files().os, "write", interrupt)

    with pytest.raises(KeyboardInterrupt):
        _files().publish_passport_file(path, _body())

    assert not tuple(tmp_path.iterdir())


def test_publish_waits_for_a_cooperating_directory_lock(tmp_path: Path) -> None:
    path = tmp_path / "output"
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    entered = Event()
    completed = Event()
    errors: list[BaseException] = []

    def publish() -> None:
        entered.set()
        try:
            _files().publish_passport_file(path, _body())
            completed.set()
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=publish)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        assert not completed.wait(timeout=0.05)
        assert not path.exists()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert not errors
    assert completed.is_set()
    assert path.read_bytes() == _body()


def test_oversized_publication_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "output"

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, b"x" * 524_289)

    assert captured.value.code == "passport_size_limit"
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["read", "publish"])
def test_nonsticky_writable_parent_is_refused(tmp_path: Path, operation: str) -> None:
    parent = tmp_path / "unsafe"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    path = parent / "passport"
    if operation == "read":
        path.write_bytes(_body())

    with pytest.raises(PassportError) as captured:
        if operation == "read":
            _files().read_passport_file(path)
        else:
            _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert not tuple(parent.glob(".passport-*.tmp"))


def test_sticky_writable_parent_allows_safe_read_and_publish(tmp_path: Path) -> None:
    parent = tmp_path / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)
    path = parent / "passport"

    _files().publish_passport_file(path, _body())

    assert _files().read_passport_file(path).canonical_bytes == _body()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_same_bytes_replacement_between_checks_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    path.write_bytes(_body())
    real_stat = os.stat
    changed = False

    def status(
        name: str | bytes | os.PathLike[str] | os.PathLike[bytes] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal changed
        result = real_stat(name, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if name == path.name and dir_fd is not None and not changed:
            changed = True
            path.rename(tmp_path / "original")
            path.write_bytes(_body())
        return result

    monkeypatch.setattr(_files().os, "stat", status)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert path.read_bytes() == _body()
    assert (tmp_path / "original").read_bytes() == _body()


def test_postlink_same_bytes_competitor_is_preserved_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    real_stat = os.stat
    changed = False
    competing_identity: list[int] = []

    def status(
        name: str | bytes | os.PathLike[str] | os.PathLike[bytes] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal changed
        result = real_stat(name, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if name == path.name and dir_fd is not None and not changed:
            changed = True
            path.rename(tmp_path / "original-output")
            path.write_bytes(_body())
            competing_identity.append(path.stat().st_ino)
        return result

    monkeypatch.setattr(_files().os, "stat", status)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert path.stat().st_ino == competing_identity[0]
    assert path.read_bytes() == _body()
    assert not tuple(tmp_path.glob(".passport-*.tmp"))


def test_cleanup_rechecks_entry_after_descriptor_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    retained = tmp_path / "retained-owned"
    real_fstat = os.fstat
    cleaning = False
    changed = False
    replacement: list[Path] = []

    def fail_publication(source: str, target: str, **kwargs: object) -> None:
        nonlocal cleaning
        cleaning = True
        raise OSError("injected publication failure")

    def fstat(descriptor: int) -> os.stat_result:
        nonlocal changed
        result = real_fstat(descriptor)
        if cleaning and not changed and stat.S_ISREG(result.st_mode):
            changed = True
            temporary = next(tmp_path.glob(".passport-*.tmp"))
            temporary.rename(retained)
            temporary.write_bytes(b"competitor during cleanup")
            replacement.append(temporary)
        return result

    module = _files()
    monkeypatch.setattr(module.os, "link", fail_publication)
    monkeypatch.setattr(module.os, "fstat", fstat)

    with pytest.raises(PassportError) as captured:
        module.publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert replacement[0].read_bytes() == b"competitor during cleanup"
    assert retained.read_bytes() == _body()
    assert not path.exists()


@pytest.mark.parametrize("failure", ["directory_fsync", "cleanup"])
def test_postpublication_failure_retains_complete_output_for_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "output"
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory sync failure")
        real_fsync(descriptor)

    def unlink(name: str, *, dir_fd: int | None = None) -> None:
        raise OSError("injected cleanup failure")

    module = _files()
    with monkeypatch.context() as injected:
        if failure == "directory_fsync":
            injected.setattr(module.os, "fsync", fsync)
        else:
            injected.setattr(module.os, "unlink", unlink)
        with pytest.raises(PassportError) as captured:
            module.publish_passport_file(path, _body())

    identity = path.stat().st_ino
    assert captured.value.code == "passport_file_invalid"
    assert path.read_bytes() == _body()
    module.publish_passport_file(path, _body())
    assert path.stat().st_ino == identity
    assert path.read_bytes() == _body()


def test_temporary_name_collision_preserves_unowned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifier = UUID("20000000-0000-4000-8000-000000000004")
    temporary = tmp_path / f".passport-{identifier.hex}.tmp"
    temporary.write_bytes(b"unowned file")
    path = tmp_path / "output"
    monkeypatch.setattr(_files(), "uuid4", lambda: identifier)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert temporary.read_bytes() == b"unowned file"
    assert not path.exists()


def test_partial_write_failure_removes_only_owned_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "output"
    real_write = os.write
    written = False

    def write(descriptor: int, body: bytes | memoryview) -> int:
        nonlocal written
        if written:
            raise OSError("injected failure after partial write")
        written = True
        return real_write(descriptor, body[:17])

    monkeypatch.setattr(_files().os, "write", write)

    with pytest.raises(PassportError) as captured:
        _files().publish_passport_file(path, _body())

    assert captured.value.code == "passport_file_invalid"
    assert not tuple(tmp_path.iterdir())
