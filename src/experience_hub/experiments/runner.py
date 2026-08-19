"""Two-pass orchestration for isolated deterministic replay experiments."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from time import monotonic_ns

from pydantic import ValidationError

from experience_hub.canonical import sha256_hex
from experience_hub.experiments.contracts import (
    MAX_REPLAY_OUTPUT_BYTES,
    ArmEvidenceV1,
    ArmObservationV1,
    CaseEvidenceV1,
    OracleEvidenceV1,
    PolicyArmDescriptorV1,
    ReplayCaseV1,
    ReplayEvidenceDataV1,
    ReplayEvidenceReportV1,
    ReplayProfileDataV1,
    ReplayProfileReportV1,
    ResolvedReplayManifestV1,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.loading import (
    LoadedReplayDataset,
    LoadedReplayManifest,
    load_replay_cases,
    load_replay_manifest,
)
from experience_hub.experiments.oracles import score_retrieval_observation
from experience_hub.experiments.policies import (
    PolicyExecutionContext,
    build_policy_arm,
)
from experience_hub.experiments.reports import (
    ExperimentOutputError,
    canonical_evidence_bytes,
    canonical_profile_bytes,
    verify_evidence_bytes,
)
from experience_hub.experiments.snapshots import (
    FrozenSqliteSnapshot,
    clone_frozen_sqlite,
    freeze_closed_sqlite,
    validate_frozen_snapshot,
    verify_source_unchanged,
)
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    OwnedWorkspace,
    prepare_owned_workspace,
)

_SCHEMA_REVISION = re.compile(r"0*(\d+)(?:_[a-z0-9_]+)?\Z")
_EVIDENCE_PATH = PurePosixPath("artifacts/evidence.json")
_PROFILE_PATH = PurePosixPath("artifacts/profile.json")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True, slots=True)
class ReplayExecution:
    """Canonical replay results and process-facing publication status."""

    evidence: ReplayEvidenceReportV1
    evidence_body: bytes
    profile: ReplayProfileReportV1 | None
    profile_body: bytes | None
    valid: bool
    profile_complete: bool


@dataclass(frozen=True, slots=True)
class ReplayInspection:
    """Location-independent logical summary of validated replay inputs."""

    experiment_id: str
    dataset_id: str
    case_count: int
    arm_count: int
    manifest_sha256: str
    cases_sha256: str
    snapshot_sha256: str
    source_schema_revision: int


@dataclass(frozen=True, slots=True)
class _SuiteResult:
    cases: tuple[CaseEvidenceV1, ...]
    clone_identities: tuple[tuple[int, int], ...]


def _schema_revision_number(revision: str) -> int:
    match = _SCHEMA_REVISION.fullmatch(revision)
    if match is None:
        raise ExperimentIsolationError(
            "replay_schema_unsupported",
            "Replay source schema is not supported",
        )
    return int(match.group(1))


def _resolved_manifest(
    loaded: LoadedReplayManifest,
    dataset: LoadedReplayDataset,
    snapshot: FrozenSqliteSnapshot,
    *,
    source_schema_revision: int,
) -> ResolvedReplayManifestV1:
    manifest = loaded.manifest
    return ResolvedReplayManifestV1(
        schema_version=1,
        experiment_id=manifest.experiment_id,
        manifest_sha256=sha256_hex(loaded.body),
        dataset_id=manifest.dataset.dataset_id,
        cases_sha256=sha256_hex(dataset.body),
        snapshot_sha256=snapshot.database_sha256,
        source_schema_revision=source_schema_revision,
        frozen_at=manifest.frozen_at,
        seed=manifest.seed,
        policy_arms=manifest.arms,
        oracle=manifest.oracle,
        evidence_schema_version=manifest.evidence_schema_version,
        profile_schema_version=manifest.profile_schema_version,
    )


def _failed_arm(
    descriptor: PolicyArmDescriptorV1,
    *,
    code: str,
    stage: str,
) -> ArmEvidenceV1:
    return ArmEvidenceV1(
        schema_version=1,
        arm_id=descriptor.arm_id,
        status="failed",
        observation=None,
        utility_micros=None,
        error_code=code,
        error_stage=stage,
    )


def _complete_arm(
    descriptor: PolicyArmDescriptorV1,
    *,
    observation: ArmObservationV1,
    oracle: OracleEvidenceV1,
) -> ArmEvidenceV1:
    return ArmEvidenceV1(
        schema_version=1,
        arm_id=descriptor.arm_id,
        status="complete",
        observation=observation,
        utility_micros=oracle.utility_micros,
        error_code=None,
        error_stage=None,
    )


def _clone_identity_and_hash(
    path: Path,
    *,
    expected_size: int,
) -> tuple[tuple[int, int], str]:
    descriptor = -1
    try:
        retained = path.lstat()
        if (
            not stat.S_ISREG(retained.st_mode)
            or retained.st_nlink != 1
            or retained.st_size != expected_size
        ):
            raise ExperimentIsolationError(
                "replay_clone_corrupted",
                "Replay clone does not retain the frozen source bytes",
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or opened.st_dev != retained.st_dev
            or opened.st_ino != retained.st_ino
        ):
            raise ExperimentIsolationError(
                "replay_clone_corrupted",
                "Replay clone does not retain the frozen source bytes",
            )
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ExperimentIsolationError(
                    "replay_clone_corrupted",
                    "Replay clone does not retain the frozen source bytes",
                )
            digest.update(chunk)
            remaining -= len(chunk)
        final_descriptor = os.fstat(descriptor)
        final = path.lstat()
        if (
            retained.st_dev != final_descriptor.st_dev
            or retained.st_ino != final_descriptor.st_ino
            or retained.st_size != final_descriptor.st_size
            or retained.st_mtime_ns != final_descriptor.st_mtime_ns
            or retained.st_ctime_ns != final_descriptor.st_ctime_ns
            or final_descriptor.st_dev != final.st_dev
            or final_descriptor.st_ino != final.st_ino
            or final_descriptor.st_size != final.st_size
            or final_descriptor.st_mtime_ns != final.st_mtime_ns
            or final_descriptor.st_ctime_ns != final.st_ctime_ns
        ):
            raise ExperimentIsolationError(
                "replay_clone_corrupted",
                "Replay clone does not retain the frozen source bytes",
            )
        return (retained.st_dev, retained.st_ino), digest.hexdigest()
    except ExperimentIsolationError:
        raise
    except OSError:
        raise ExperimentIsolationError(
            "replay_clone_corrupted",
            "Replay clone does not retain the frozen source bytes",
        ) from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


async def _execute_arm(
    *,
    descriptor: PolicyArmDescriptorV1,
    case: ReplayCaseV1,
    clone_path: Path,
    frozen_at: datetime,
    seed: int,
) -> ArmEvidenceV1:
    arm = build_policy_arm(descriptor)
    try:
        observation = await arm.execute(
            PolicyExecutionContext(
                case=case,
                clone_path=clone_path,
                frozen_at=frozen_at,
                seed=seed,
            )
        )
        observation = ArmObservationV1.model_validate(observation, strict=True)
    except Exception:
        return _failed_arm(
            descriptor,
            code="arm_infrastructure_failure",
            stage="execute",
        )

    try:
        oracle = score_retrieval_observation(case, observation)
        oracle = OracleEvidenceV1.model_validate(oracle, strict=True)
    except (ExperimentInputError, ValidationError):
        return _failed_arm(
            descriptor,
            code="oracle_validation_failure",
            stage="oracle",
        )
    return _complete_arm(
        descriptor,
        observation=observation,
        oracle=oracle,
    )


async def _execute_suite(
    *,
    replay_name: str,
    workspace: OwnedWorkspace,
    snapshot: FrozenSqliteSnapshot,
    loaded: LoadedReplayManifest,
    dataset: LoadedReplayDataset,
) -> _SuiteResult:
    cases: list[CaseEvidenceV1] = []
    clone_identities: list[tuple[int, int]] = []
    for case in dataset.cases:
        arm_evidence: list[ArmEvidenceV1] = []
        for descriptor in loaded.manifest.arms:
            destination = (
                workspace.root
                / "arms"
                / replay_name
                / case.case_id
                / f"{descriptor.arm_id}.sqlite3"
            )
            clone = await asyncio.to_thread(
                clone_frozen_sqlite,
                snapshot,
                destination,
            )
            identity, clone_sha256 = await asyncio.to_thread(
                _clone_identity_and_hash,
                clone,
                expected_size=len(snapshot.database_bytes),
            )
            if (
                clone_sha256 != snapshot.database_sha256
                or identity in clone_identities
            ):
                raise ExperimentIsolationError(
                    "replay_clone_corrupted",
                    "Replay clone does not retain the frozen source bytes",
                )
            clone_identities.append(identity)
            arm_evidence.append(
                await _execute_arm(
                    descriptor=descriptor,
                    case=case,
                    clone_path=clone,
                    frozen_at=loaded.manifest.frozen_at,
                    seed=loaded.manifest.seed,
                )
            )
        complete = all(arm.status == "complete" for arm in arm_evidence)
        delta: int | None = None
        if complete:
            baseline, treatment = arm_evidence
            assert baseline.utility_micros is not None
            assert treatment.utility_micros is not None
            delta = treatment.utility_micros - baseline.utility_micros
        cases.append(
            CaseEvidenceV1(
                schema_version=1,
                case_id=case.case_id,
                status="complete" if complete else "incomplete",
                arms=tuple(arm_evidence),
                delta_utility_micros=delta,
            )
        )
    return _SuiteResult(
        cases=tuple(cases),
        clone_identities=tuple(clone_identities),
    )


def _evidence_report(
    *,
    resolved: ResolvedReplayManifestV1,
    cases: tuple[CaseEvidenceV1, ...],
    source_unchanged: bool,
    clone_isolation_verified: bool,
    deterministic_replay_match: bool,
) -> ReplayEvidenceReportV1:
    comparison_complete = all(case.status == "complete" for case in cases)
    valid = (
        comparison_complete
        and source_unchanged
        and clone_isolation_verified
        and deterministic_replay_match
    )
    return ReplayEvidenceReportV1(
        data=ReplayEvidenceDataV1(
            schema_version=1,
            resolved_manifest=resolved,
            cases=cases,
            comparison_complete=comparison_complete,
            source_unchanged=source_unchanged,
            clone_isolation_verified=clone_isolation_verified,
            deterministic_replay_match=deterministic_replay_match,
            valid=valid,
        )
    )


def _core_evidence_body(
    resolved: ResolvedReplayManifestV1,
    suite: _SuiteResult,
) -> bytes:
    report = _evidence_report(
        resolved=resolved,
        cases=suite.cases,
        source_unchanged=True,
        clone_isolation_verified=True,
        deterministic_replay_match=True,
    )
    return canonical_evidence_bytes(report)


def _profile(
    *,
    experiment_id: str,
    complete: bool,
    duration_ns: int | None,
    database_bytes: int,
) -> ReplayProfileReportV1:
    return ReplayProfileReportV1(
        data=ReplayProfileDataV1(
            schema_version=1,
            experiment_id=experiment_id,
            profile_complete=complete,
            wall_duration_ns=duration_ns,
            database_bytes=database_bytes,
        )
    )


def _artifact_error() -> ExperimentOutputError:
    return ExperimentOutputError(
        "artifact_write_failed",
        "Replay artifact bytes could not be published",
    )


def _inspect_cleanup_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_inspect_cleanup_unsafe",
        "Replay inspection workspace could not be safely cleaned",
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _write_all(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError
        written += count


def _restore_inspection_marker(root_fd: int) -> None:
    marker_fd = -1
    try:
        marker_fd = os.open(
            REPLAY_WORKSPACE_POLICY.marker_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        _write_all(marker_fd, REPLAY_WORKSPACE_POLICY.marker_body)
        os.fsync(marker_fd)
    except Exception:
        return
    finally:
        if marker_fd >= 0:
            with suppress(OSError):
                os.close(marker_fd)


def _remove_empty_inspection_root(
    workspace: OwnedWorkspace,
) -> None:
    root = workspace.root
    parent_fd = -1
    root_fd = -1
    marker_fd = -1
    locked = False
    marker_removed = False
    try:
        parent_fd = os.open(root.parent, _DIRECTORY_FLAGS)
        retained = os.stat(
            root.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(retained.st_mode)
            or retained.st_dev != workspace._device
            or retained.st_ino != workspace._inode
        ):
            raise _inspect_cleanup_error()
        root_fd = os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(retained, opened):
            raise _inspect_cleanup_error()
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        locked = True
        current = os.stat(
            root.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not _same_identity(opened, current):
            raise _inspect_cleanup_error()
        if os.listdir(root_fd) != [REPLAY_WORKSPACE_POLICY.marker_name]:
            raise _inspect_cleanup_error()

        marker_fd = os.open(
            REPLAY_WORKSPACE_POLICY.marker_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        marker_status = os.fstat(marker_fd)
        if not stat.S_ISREG(marker_status.st_mode):
            raise _inspect_cleanup_error()
        marker_body = os.read(
            marker_fd,
            len(REPLAY_WORKSPACE_POLICY.marker_body) + 1,
        )
        current_marker = os.stat(
            REPLAY_WORKSPACE_POLICY.marker_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            marker_body != REPLAY_WORKSPACE_POLICY.marker_body
            or not _same_identity(marker_status, current_marker)
        ):
            raise _inspect_cleanup_error()
        os.close(marker_fd)
        marker_fd = -1

        os.unlink(REPLAY_WORKSPACE_POLICY.marker_name, dir_fd=root_fd)
        marker_removed = True
        if os.listdir(root_fd):
            raise _inspect_cleanup_error()
        current = os.stat(
            root.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not _same_identity(opened, current):
            raise _inspect_cleanup_error()
        os.rmdir(root.name, dir_fd=parent_fd)
        marker_removed = False
    except ExperimentIsolationError:
        if marker_removed and root_fd >= 0:
            _restore_inspection_marker(root_fd)
        raise
    except Exception:
        if marker_removed and root_fd >= 0:
            _restore_inspection_marker(root_fd)
        raise _inspect_cleanup_error() from None
    finally:
        if marker_fd >= 0:
            with suppress(OSError):
                os.close(marker_fd)
        if locked:
            with suppress(OSError):
                fcntl.flock(root_fd, fcntl.LOCK_UN)
        if root_fd >= 0:
            with suppress(OSError):
                os.close(root_fd)
        if parent_fd >= 0:
            with suppress(OSError):
                os.close(parent_fd)


def _cleanup_inspection_workspace(workspace: OwnedWorkspace) -> None:
    root = workspace.root
    try:
        retained = root.lstat()
        if (
            not stat.S_ISDIR(retained.st_mode)
            or retained.st_dev != workspace._device
            or retained.st_ino != workspace._inode
        ):
            raise _inspect_cleanup_error()
        cleaned = prepare_owned_workspace(
            root,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=True,
            allow_unmarked_empty=False,
        )
        if (
            cleaned._device != workspace._device
            or cleaned._inode != workspace._inode
        ):
            raise _inspect_cleanup_error()
        _remove_empty_inspection_root(cleaned)
    except ExperimentIsolationError:
        raise _inspect_cleanup_error() from None
    except Exception:
        raise _inspect_cleanup_error() from None


def _write_body(
    workspace: OwnedWorkspace,
    relative: PurePosixPath,
    body: bytes,
) -> Path:
    try:
        return workspace.atomic_write(relative, body)
    except ExperimentIsolationError:
        raise _artifact_error() from None


async def _prepare_inputs(
    manifest_path: Path,
    database_path: Path,
) -> tuple[
    LoadedReplayManifest,
    LoadedReplayDataset,
    FrozenSqliteSnapshot,
]:
    loaded = await asyncio.to_thread(load_replay_manifest, manifest_path)
    dataset = await asyncio.to_thread(load_replay_cases, loaded)
    snapshot = await asyncio.to_thread(freeze_closed_sqlite, database_path)
    return loaded, dataset, snapshot


def _require_inputs_outside_owned_workspace(
    manifest_path: Path,
    database_path: Path,
    workspace_path: Path,
    *,
    cases_path: Path | None = None,
) -> None:
    if not all(
        isinstance(path, Path)
        for path in (manifest_path, database_path, workspace_path)
    ):
        return
    # Normalize dot segments without resolving untrusted symbolic links.
    workspace_root = Path(os.path.abspath(workspace_path))
    input_paths: tuple[Path, ...] = (manifest_path, database_path)
    if cases_path is not None:
        input_paths = (*input_paths, cases_path)
    for input_path in input_paths:
        try:
            relative = Path(os.path.abspath(input_path)).relative_to(
                workspace_root
            )
        except ValueError:
            continue
        if (
            relative.parts
            and relative.parts[0] in REPLAY_WORKSPACE_POLICY.owned_entries
        ):
            raise _workspace_input_overlap()


def _workspace_input_overlap() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_workspace_input_overlap",
        "Replay inputs must remain outside workspace-owned output",
    )


def _require_no_symlink_components(*paths: Path) -> None:
    if not all(isinstance(path, Path) for path in paths):
        return
    for path in paths:
        absolute = Path(os.path.abspath(path))
        descriptor = -1
        try:
            descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
            parts = absolute.parts[1:]
            for index, part in enumerate(parts):
                try:
                    status = os.stat(
                        part,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(status.st_mode):
                    raise _workspace_input_overlap()
                if index == len(parts) - 1 or not stat.S_ISDIR(
                    status.st_mode
                ):
                    break
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _workspace_input_overlap() from None
        finally:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


async def run_replay(
    manifest_path: Path,
    database_path: Path,
    workspace_path: Path,
    *,
    replace_owned: bool = False,
    profiler: Callable[[], int] = monotonic_ns,
) -> ReplayExecution:
    """Run two ordered replay passes and publish canonical artifacts."""
    await asyncio.to_thread(
        _require_inputs_outside_owned_workspace,
        manifest_path,
        database_path,
        workspace_path,
    )
    await asyncio.to_thread(
        _require_no_symlink_components,
        manifest_path,
        database_path,
    )
    loaded = await asyncio.to_thread(load_replay_manifest, manifest_path)
    cases_path = (
        loaded._parent / loaded.manifest.dataset.cases_file
    )
    await asyncio.to_thread(
        _require_inputs_outside_owned_workspace,
        manifest_path,
        database_path,
        workspace_path,
        cases_path=cases_path,
    )
    await asyncio.to_thread(_require_no_symlink_components, cases_path)
    dataset = await asyncio.to_thread(load_replay_cases, loaded)
    snapshot = await asyncio.to_thread(freeze_closed_sqlite, database_path)
    workspace = await asyncio.to_thread(
        prepare_owned_workspace,
        workspace_path,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=replace_owned,
        allow_unmarked_empty=True,
    )
    revision = await validate_frozen_snapshot(
        snapshot,
        validation_path=workspace.root / "validation" / "source.sqlite3",
        frozen_at=loaded.manifest.frozen_at,
        seed=loaded.manifest.seed,
    )
    resolved = _resolved_manifest(
        loaded,
        dataset,
        snapshot,
        source_schema_revision=_schema_revision_number(revision),
    )

    profile_complete = True
    started_ns: int | None
    try:
        started_ns = profiler()
        if isinstance(started_ns, bool) or not isinstance(started_ns, int):
            raise TypeError
    except Exception:
        started_ns = None
        profile_complete = False

    first = await _execute_suite(
        replay_name="replay-a",
        workspace=workspace,
        snapshot=snapshot,
        loaded=loaded,
        dataset=dataset,
    )
    second = await _execute_suite(
        replay_name="replay-b",
        workspace=workspace,
        snapshot=snapshot,
        loaded=loaded,
        dataset=dataset,
    )
    first_body = _core_evidence_body(resolved, first)
    second_body = _core_evidence_body(resolved, second)
    deterministic_match = first_body == second_body

    all_identities = (*first.clone_identities, *second.clone_identities)
    clone_isolation_verified = len(all_identities) == len(set(all_identities))
    evidence = _evidence_report(
        resolved=resolved,
        cases=first.cases,
        source_unchanged=True,
        clone_isolation_verified=clone_isolation_verified,
        deterministic_replay_match=deterministic_match,
    )
    evidence_body = canonical_evidence_bytes(evidence)

    duration_ns: int | None = None
    if started_ns is not None:
        try:
            finished_ns = profiler()
            if isinstance(finished_ns, bool) or not isinstance(finished_ns, int):
                raise TypeError
            duration_ns = finished_ns - started_ns
            if duration_ns < 0:
                raise ValueError
        except Exception:
            duration_ns = None
            profile_complete = False
    profile = _profile(
        experiment_id=loaded.manifest.experiment_id,
        complete=profile_complete,
        duration_ns=duration_ns,
        database_bytes=len(snapshot.database_bytes),
    )
    profile_body: bytes | None = canonical_profile_bytes(profile)
    await asyncio.to_thread(verify_source_unchanged, snapshot)

    if not evidence.data.source_unchanged or not deterministic_match:
        return ReplayExecution(
            evidence=evidence,
            evidence_body=evidence_body,
            profile=profile,
            profile_body=profile_body,
            valid=False,
            profile_complete=profile_complete,
        )

    if profile_body is not None:
        try:
            await asyncio.to_thread(
                _write_body,
                workspace,
                _PROFILE_PATH,
                profile_body,
            )
        except ExperimentOutputError:
            profile_complete = False
            profile = _profile(
                experiment_id=loaded.manifest.experiment_id,
                complete=False,
                duration_ns=None,
                database_bytes=len(snapshot.database_bytes),
            )
            profile_body = None
    await asyncio.to_thread(
        _write_body,
        workspace,
        _EVIDENCE_PATH,
        evidence_body,
    )
    return ReplayExecution(
        evidence=evidence,
        evidence_body=evidence_body,
        profile=profile,
        profile_body=profile_body,
        valid=evidence.data.valid and profile_complete,
        profile_complete=profile_complete,
    )


async def inspect_replay(
    manifest_path: Path,
    database_path: Path,
) -> ReplayInspection:
    """Validate replay inputs without creating persistent arm clones."""
    loaded, dataset, snapshot = await _prepare_inputs(
        manifest_path,
        database_path,
    )
    temporary = Path(
        await asyncio.to_thread(
            tempfile.mkdtemp,
            prefix="experience-hub-replay-inspect-",
            dir=manifest_path.parent,
        )
    )
    workspace: OwnedWorkspace | None = None
    primary_error: BaseException | None = None
    try:
        workspace = await asyncio.to_thread(
            prepare_owned_workspace,
            temporary,
            policy=REPLAY_WORKSPACE_POLICY,
            replace_owned=False,
            allow_unmarked_empty=True,
        )
        revision = await validate_frozen_snapshot(
            snapshot,
            validation_path=workspace.root / "validation" / "source.sqlite3",
            frozen_at=loaded.manifest.frozen_at,
            seed=loaded.manifest.seed,
        )
        await asyncio.to_thread(verify_source_unchanged, snapshot)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if workspace is not None:
            try:
                await asyncio.to_thread(
                    _cleanup_inspection_workspace,
                    workspace,
                )
            except ExperimentIsolationError as cleanup_error:
                if primary_error is None:
                    raise cleanup_error from None
    return ReplayInspection(
        experiment_id=loaded.manifest.experiment_id,
        dataset_id=loaded.manifest.dataset.dataset_id,
        case_count=len(dataset.cases),
        arm_count=len(loaded.manifest.arms),
        manifest_sha256=sha256_hex(loaded.body),
        cases_sha256=sha256_hex(dataset.body),
        snapshot_sha256=snapshot.database_sha256,
        source_schema_revision=_schema_revision_number(revision),
    )


def verify_replay_report(path: Path) -> ReplayEvidenceReportV1:
    """Verify one bounded canonical evidence file without following symlinks."""
    if not isinstance(path, Path):
        raise ExperimentOutputError(
            "invalid_report_path",
            "Replay evidence must be a regular non-symlink file",
        )
    descriptor = -1
    try:
        retained = path.lstat()
        if not stat.S_ISREG(retained.st_mode):
            raise ExperimentOutputError(
                "invalid_report_path",
                "Replay evidence must be a regular non-symlink file",
            )
        if retained.st_size > MAX_REPLAY_OUTPUT_BYTES:
            raise ExperimentOutputError(
                "output_too_large",
                "Replay evidence exceeds the output byte limit",
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != retained.st_dev
            or opened.st_ino != retained.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_size != retained.st_size
        ):
            raise ExperimentOutputError(
                "invalid_report_path",
                "Replay evidence must be a regular non-symlink file",
            )
        chunks: list[bytes] = []
        remaining = retained.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise ExperimentOutputError(
                    "invalid_report_path",
                    "Replay evidence must be a stable regular file",
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        final = os.fstat(descriptor)
        final_path = path.lstat()
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
            or final_path.st_dev != final.st_dev
            or final_path.st_ino != final.st_ino
            or final_path.st_size != final.st_size
            or final_path.st_mtime_ns != final.st_mtime_ns
            or final_path.st_ctime_ns != final.st_ctime_ns
        ):
            raise ExperimentOutputError(
                "invalid_report_path",
                "Replay evidence must be a stable regular file",
            )
    except ExperimentOutputError:
        raise
    except OSError:
        raise ExperimentOutputError(
            "invalid_report_path",
            "Replay evidence must be a regular non-symlink file",
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return verify_evidence_bytes(body)


__all__ = [
    "ReplayExecution",
    "ReplayInspection",
    "inspect_replay",
    "run_replay",
    "verify_replay_report",
]
