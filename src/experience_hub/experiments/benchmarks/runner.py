"""Two-pass orchestration for the isolated ExperienceBench-S pilot."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic_ns

from pydantic import ValidationError

from experience_hub.canonical import sha256_hex
from experience_hub.experiments.benchmarks.contracts import (
    MAX_BENCHMARK_OUTPUT_BYTES,
    BenchmarkAggregateV1,
    BenchmarkArmDescriptorV1,
    BenchmarkArmEvidenceV1,
    BenchmarkArmObservationV1,
    BenchmarkCaseEvidenceV1,
    BenchmarkCaseV1,
    BenchmarkEvidenceDataV1,
    BenchmarkEvidenceReportV1,
    BenchmarkGateResultV1,
    BenchmarkPassPayloadV1,
    BenchmarkProfileDataV1,
    BenchmarkProfileReportV1,
    BenchmarkSafetyEvidenceV1,
    BenchmarkSourceCandidateV1,
    BenchmarkSourceExperienceV1,
    BenchmarkSourceRecordV1,
    BenchmarkSummaryReportV1,
    ResolvedBenchmarkManifestV1,
)
from experience_hub.experiments.benchmarks.gates import evaluate_pilot_gates
from experience_hub.experiments.benchmarks.loading import (
    LoadedBenchmarkPack,
    load_benchmark_pack,
)
from experience_hub.experiments.benchmarks.metrics import (
    aggregate_benchmark_cases,
    derive_case_comparison,
)
from experience_hub.experiments.benchmarks.oracles import (
    score_benchmark_observation,
)
from experience_hub.experiments.benchmarks.policies import (
    BenchmarkPolicyContext,
    build_benchmark_policy,
    preflight_benchmark_capabilities,
)
from experience_hub.experiments.benchmarks.reports import (
    canonical_benchmark_pass_bytes,
    canonical_benchmark_profile_bytes,
    verify_benchmark_evidence_bytes,
    verify_benchmark_summary_bytes,
    write_benchmark_artifacts,
)
from experience_hub.experiments.benchmarks.source import (
    BenchmarkSourceIndex,
    BuiltBenchmarkSource,
    build_benchmark_source,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.reports import ExperimentOutputError
from experience_hub.experiments.snapshots import (
    clone_frozen_sqlite,
    validate_frozen_snapshot,
    verify_source_unchanged,
)
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    OwnedWorkspace,
    prepare_owned_workspace,
)

_SCHEMA_REVISION = re.compile(r"0*(\d+)(?:_[a-z0-9_]+)?\Z")
_REPORT_NAME = "benchmark-evidence.json"
_SUMMARY_NAME = "benchmark-summary.json"
_PROFILE_NAME = "profile.json"
_MAX_PROFILE_CLOCK_NS = (1 << 63) - 1
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True, slots=True)
class BenchmarkInspection:
    """Location-independent logical summary of one validated pilot pack."""

    pack_id: str
    case_count: int
    arm_count: int
    manifest_sha256: str
    cases_sha256: str
    source_fixture_sha256: str
    fts5_available: bool


@dataclass(frozen=True, slots=True)
class BenchmarkExecution:
    """Canonical pilot results and process-facing status dimensions."""

    evidence: BenchmarkEvidenceReportV1
    evidence_body: bytes
    summary: BenchmarkSummaryReportV1
    summary_body: bytes
    profile: BenchmarkProfileReportV1 | None
    profile_body: bytes | None
    evidence_valid: bool
    comparison_complete: bool
    deterministic_replay_match: bool
    expansion_gate_passed: bool
    profile_complete: bool


@dataclass(frozen=True, slots=True)
class _ArmResult:
    evidence: BenchmarkArmEvidenceV1
    owner_leak_count: int
    quarantine_leak_count: int


@dataclass(frozen=True, slots=True)
class _PassResult:
    cases: tuple[BenchmarkCaseEvidenceV1, ...]
    clone_identities: tuple[tuple[int, int], ...]
    owner_leak_count: int
    quarantine_leak_count: int
    cross_arm_contamination_count: int
    clone_isolation_verified: bool


def _failed_arm(
    descriptor: BenchmarkArmDescriptorV1,
    *,
    code: str,
    stage: str,
) -> BenchmarkArmEvidenceV1:
    return BenchmarkArmEvidenceV1(
        schema_version=1,
        arm_id=descriptor.arm_id,
        status="failed",
        observation=None,
        oracle=None,
        error_code=code,
        error_stage=stage,
    )


def _source_owners(
    source_records: Sequence[BenchmarkSourceRecordV1],
) -> tuple[dict[str, str], dict[str, str]]:
    ordinary: dict[str, str] = {}
    candidates: dict[str, str] = {}
    for record in source_records:
        if isinstance(record, BenchmarkSourceExperienceV1):
            ordinary[record.label] = record.owner_label
        elif isinstance(record, BenchmarkSourceCandidateV1):
            candidates[record.label] = record.owner_label
    return ordinary, candidates


async def _execute_benchmark_arm(
    *,
    descriptor: BenchmarkArmDescriptorV1,
    case: BenchmarkCaseV1,
    clone_path: Path,
    source_index: BenchmarkSourceIndex,
    source_records: Sequence[BenchmarkSourceRecordV1],
    frozen_at: datetime,
    seed: int,
) -> _ArmResult:
    """Execute and independently validate one arm without leaking exceptions."""
    try:
        policy = build_benchmark_policy(descriptor)
        observation = await policy.execute(
            BenchmarkPolicyContext(
                case=case,
                clone_path=clone_path,
                owner_agent_id=source_index.agent_ids[case.owner_label],
                source_index=source_index,
                frozen_at=frozen_at,
                seed=seed,
            )
        )
        observation = BenchmarkArmObservationV1.model_validate(observation, strict=True)
    except asyncio.CancelledError:
        raise
    except Exception:
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_arm_incomplete", stage="execute"
            ),
            owner_leak_count=0,
            quarantine_leak_count=0,
        )

    ordinary_owners, candidate_owners = _source_owners(source_records)
    owners_by_label = ordinary_owners | candidate_owners
    owner_leaks = sum(
        owners_by_label.get(label) not in {None, case.owner_label}
        for label in observation.returned_labels
    )
    quarantine_leaks = sum(
        label in candidate_owners for label in observation.returned_labels
    )
    if owner_leaks or quarantine_leaks:
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_safety_failure", stage="safety"
            ),
            owner_leak_count=owner_leaks,
            quarantine_leak_count=quarantine_leaks,
        )
    known_labels = ordinary_owners.keys() | candidate_owners.keys()
    if any(label not in known_labels for label in observation.returned_labels):
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_oracle_invalid", stage="oracle"
            ),
            owner_leak_count=0,
            quarantine_leak_count=0,
        )
    try:
        expected_content_bytes = sum(
            source_index.content_bytes_by_label[label]
            for label in observation.returned_labels
        )
    except KeyError:
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_oracle_invalid", stage="oracle"
            ),
            owner_leak_count=0,
            quarantine_leak_count=0,
        )
    if observation.selected_content_bytes != expected_content_bytes:
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_oracle_invalid", stage="oracle"
            ),
            owner_leak_count=0,
            quarantine_leak_count=0,
        )
    try:
        oracle = score_benchmark_observation(case, observation)
    except (ExperimentInputError, ValidationError):
        return _ArmResult(
            evidence=_failed_arm(
                descriptor, code="benchmark_oracle_invalid", stage="oracle"
            ),
            owner_leak_count=0,
            quarantine_leak_count=0,
        )
    return _ArmResult(
        evidence=BenchmarkArmEvidenceV1(
            schema_version=1,
            arm_id=descriptor.arm_id,
            status="complete",
            observation=observation,
            oracle=oracle,
            error_code=None,
            error_stage=None,
        ),
        owner_leak_count=0,
        quarantine_leak_count=0,
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
            raise OSError
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != retained.st_dev
            or opened.st_ino != retained.st_ino
            or opened.st_size != retained.st_size
        ):
            raise OSError
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError
            digest.update(chunk)
            remaining -= len(chunk)
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        if (
            final_descriptor.st_dev != opened.st_dev
            or final_descriptor.st_ino != opened.st_ino
            or final_descriptor.st_size != opened.st_size
            or final_descriptor.st_mtime_ns != opened.st_mtime_ns
            or final_descriptor.st_ctime_ns != opened.st_ctime_ns
            or final_path.st_dev != final_descriptor.st_dev
            or final_path.st_ino != final_descriptor.st_ino
            or final_path.st_size != final_descriptor.st_size
            or final_path.st_mtime_ns != final_descriptor.st_mtime_ns
            or final_path.st_ctime_ns != final_descriptor.st_ctime_ns
        ):
            raise OSError
        return (opened.st_dev, opened.st_ino), digest.hexdigest()
    except OSError:
        raise ExperimentInputError(
            "benchmark_safety_failure", "Benchmark clone could not be verified"
        ) from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


def _prepare_clone_for_descriptor_policy(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Normalize only the identity-bound clone through one retained descriptor."""
    descriptor = -1
    try:
        retained = path.lstat()
        if (
            not stat.S_ISREG(retained.st_mode)
            or retained.st_nlink != 1
            or retained.st_size != expected_size
            or (retained.st_dev, retained.st_ino) != expected_identity
        ):
            raise OSError
        descriptor = os.open(
            path,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or (opened.st_dev, opened.st_ino) != expected_identity
        ):
            raise OSError

        original_digest = hashlib.sha256()
        normalized_digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < expected_size:
            chunk = os.read(descriptor, min(expected_size - offset, 1024 * 1024))
            if not chunk:
                raise OSError
            original_digest.update(chunk)
            normalized_chunk = bytearray(chunk)
            for header_offset in (18, 19):
                relative = header_offset - offset
                if 0 <= relative < len(normalized_chunk):
                    normalized_chunk[relative] = 1
            normalized_digest.update(normalized_chunk)
            offset += len(chunk)
        if (
            original_digest.hexdigest() != expected_sha256
            or os.pread(descriptor, 2, 18) not in {b"\x01\x01", b"\x02\x02"}
        ):
            raise OSError

        if os.pwrite(descriptor, b"\x01\x01", 18) != 2:
            raise OSError
        os.fsync(descriptor)
        verification_started = os.fstat(descriptor)
        if (
            not stat.S_ISREG(verification_started.st_mode)
            or verification_started.st_nlink != 1
            or verification_started.st_size != expected_size
            or (verification_started.st_dev, verification_started.st_ino)
            != expected_identity
        ):
            raise OSError
        final_digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError
            final_digest.update(chunk)
            remaining -= len(chunk)
        verification_finished = os.fstat(descriptor)
        final = path.lstat()
        sidecar_present = False
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                Path(f"{path}{suffix}").lstat()
            except FileNotFoundError:
                continue
            sidecar_present = True
        if (
            final_digest.hexdigest() != normalized_digest.hexdigest()
            or verification_finished.st_dev != verification_started.st_dev
            or verification_finished.st_ino != verification_started.st_ino
            or verification_finished.st_size != verification_started.st_size
            or verification_finished.st_nlink != verification_started.st_nlink
            or verification_finished.st_mtime_ns
            != verification_started.st_mtime_ns
            or verification_finished.st_ctime_ns
            != verification_started.st_ctime_ns
            or not stat.S_ISREG(final.st_mode)
            or final.st_dev != verification_finished.st_dev
            or final.st_ino != verification_finished.st_ino
            or final.st_size != verification_finished.st_size
            or final.st_nlink != verification_finished.st_nlink
            or final.st_mtime_ns != verification_finished.st_mtime_ns
            or final.st_ctime_ns != verification_finished.st_ctime_ns
            or sidecar_present
        ):
            raise OSError
    except OSError:
        raise ExperimentInputError(
            "benchmark_safety_failure",
            "Benchmark clone could not be prepared",
        ) from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


async def _execute_benchmark_cases(
    *,
    pass_name: str,
    cases: Sequence[BenchmarkCaseV1],
    descriptors: Sequence[BenchmarkArmDescriptorV1],
    workspace: OwnedWorkspace,
    source: BuiltBenchmarkSource,
    source_records: Sequence[BenchmarkSourceRecordV1],
    seen_identities: set[tuple[int, int]],
    frozen_at: datetime,
    seed: int,
) -> _PassResult:
    """Execute an ordered case subset; public callers always supply all 30."""
    case_evidence: list[BenchmarkCaseEvidenceV1] = []
    identities: list[tuple[int, int]] = []
    owner_leaks = 0
    quarantine_leaks = 0
    cross_arm_count = 0
    clone_isolation = True
    source_identity = (source.snapshot._device, source.snapshot._inode)
    for case in cases:
        arms: list[BenchmarkArmEvidenceV1] = []
        for descriptor in descriptors:
            clone_path = (
                workspace.root
                / "arms"
                / pass_name
                / case.case_id
                / f"{descriptor.arm_id}.sqlite3"
            )
            try:
                clone = await _complete_threaded(
                    clone_frozen_sqlite, source.snapshot, clone_path
                )
                identity, digest = await _complete_threaded(
                    _clone_identity_and_hash,
                    clone,
                    expected_size=len(source.snapshot.database_bytes),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                clone_isolation = False
                arms.append(
                    _failed_arm(
                        descriptor,
                        code="benchmark_arm_incomplete",
                        stage="clone",
                    )
                )
                continue
            identities.append(identity)
            if (
                digest != source.snapshot.database_sha256
                or identity == source_identity
                or identity in seen_identities
            ):
                clone_isolation = False
                if identity == source_identity or identity in seen_identities:
                    cross_arm_count += 1
                arms.append(
                    _failed_arm(
                        descriptor,
                        code="benchmark_safety_failure",
                        stage="clone",
                    )
                )
                seen_identities.add(identity)
                continue
            seen_identities.add(identity)
            try:
                await _complete_threaded(
                    _prepare_clone_for_descriptor_policy,
                    clone,
                    expected_identity=identity,
                    expected_size=len(source.snapshot.database_bytes),
                    expected_sha256=source.snapshot.database_sha256,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                clone_isolation = False
                arms.append(
                    _failed_arm(
                        descriptor,
                        code="benchmark_safety_failure",
                        stage="clone",
                    )
                )
                continue
            result = await _execute_benchmark_arm(
                descriptor=descriptor,
                case=case,
                clone_path=clone,
                source_index=source.index,
                source_records=source_records,
                frozen_at=frozen_at,
                seed=seed,
            )
            arms.append(result.evidence)
            owner_leaks += result.owner_leak_count
            quarantine_leaks += result.quarantine_leak_count
        case_evidence.append(derive_case_comparison(case, tuple(arms)))
    return _PassResult(
        cases=tuple(case_evidence),
        clone_identities=tuple(identities),
        owner_leak_count=owner_leaks,
        quarantine_leak_count=quarantine_leaks,
        cross_arm_contamination_count=cross_arm_count,
        clone_isolation_verified=clone_isolation,
    )


async def _complete_threaded[T](
    operation: Callable[..., T],
    *arguments: object,
    **keywords: object,
) -> T:
    """Drain thread-backed filesystem work before propagating cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *arguments, **keywords))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
            continue
        except Exception:
            if cancellation is None:
                raise
            break
        if cancellation is not None:
            raise cancellation
        return result
    assert cancellation is not None
    raise cancellation


def _schema_revision_number(revision: str) -> int:
    match = _SCHEMA_REVISION.fullmatch(revision)
    if match is None:
        raise ExperimentIsolationError(
            "benchmark_workspace_invalid",
            "Benchmark source schema is unsupported",
        )
    return int(match.group(1))


def _resolved_manifest(
    pack: LoadedBenchmarkPack,
    source: BuiltBenchmarkSource,
    *,
    source_schema_revision: int,
) -> ResolvedBenchmarkManifestV1:
    manifest = pack.manifest
    return ResolvedBenchmarkManifestV1(
        schema_version=1,
        pack_id=manifest.pack_id,
        maturity=manifest.maturity,
        manifest_sha256=sha256_hex(pack.manifest_body),
        cases_sha256=sha256_hex(pack.cases_body),
        source_fixture_sha256=sha256_hex(pack.source_body),
        snapshot_sha256=source.snapshot.database_sha256,
        source_schema_revision=source_schema_revision,
        frozen_at=manifest.frozen_at,
        seed=manifest.seed,
        arms=manifest.arms,
        oracle_version=manifest.oracle_version,
        metric_version=manifest.metric_version,
        gate_version=manifest.gate_version,
        evidence_schema_version=manifest.evidence_schema_version,
        summary_schema_version=manifest.summary_schema_version,
        profile_schema_version=manifest.profile_schema_version,
    )


def _combined_safety(
    first: _PassResult,
    second: _PassResult,
    *,
    expected_clone_count: int,
    source_unchanged: bool,
) -> BenchmarkSafetyEvidenceV1:
    identities = (*first.clone_identities, *second.clone_identities)
    clone_isolation = (
        first.clone_isolation_verified
        and second.clone_isolation_verified
        and len(identities) == expected_clone_count
        and len(set(identities)) == expected_clone_count
    )
    duplicate_count = len(identities) - len(set(identities))
    return BenchmarkSafetyEvidenceV1(
        schema_version=1,
        owner_leak_count=first.owner_leak_count + second.owner_leak_count,
        quarantine_leak_count=(
            first.quarantine_leak_count + second.quarantine_leak_count
        ),
        cross_arm_contamination_count=max(
            first.cross_arm_contamination_count
            + second.cross_arm_contamination_count,
            duplicate_count,
        ),
        source_mutation_count=0 if source_unchanged else 1,
        source_unchanged=source_unchanged,
        clone_isolation_verified=clone_isolation,
    )


def _pass_payload(
    resolved: ResolvedBenchmarkManifestV1,
    result: _PassResult,
    safety: BenchmarkSafetyEvidenceV1,
) -> BenchmarkPassPayloadV1:
    aggregate: BenchmarkAggregateV1 | None = aggregate_benchmark_cases(result.cases)
    return BenchmarkPassPayloadV1(
        schema_version=1,
        resolved_manifest=resolved,
        cases=result.cases,
        comparison_complete=aggregate is not None,
        safety=safety,
        aggregate=aggregate,
    )


def _gates_for_report(
    payload: BenchmarkPassPayloadV1,
    *,
    deterministic_replay_match: bool,
) -> tuple[BenchmarkGateResultV1, ...]:
    return tuple(
        BenchmarkGateResultV1(
            schema_version=1,
            gate_id=gate.gate_id,
            passed=(
                deterministic_replay_match
                if gate.gate_id == "deterministic_replay"
                else gate.passed
            ),
        )
        for gate in evaluate_pilot_gates(payload, payload)
    )


def _evidence_report(
    payload: BenchmarkPassPayloadV1,
    *,
    deterministic_replay_match: bool,
) -> BenchmarkEvidenceReportV1:
    gates = _gates_for_report(
        payload,
        deterministic_replay_match=deterministic_replay_match,
    )
    valid = (
        payload.comparison_complete
        and deterministic_replay_match
        and payload.safety.is_safe
    )
    return BenchmarkEvidenceReportV1(
        data=BenchmarkEvidenceDataV1(
            schema_version=1,
            pass_payload=payload,
            deterministic_replay_match=deterministic_replay_match,
            gates=gates,
            expansion_gate_passed=valid and all(gate.passed for gate in gates),
            valid=valid,
        )
    )


def _profile(
    *,
    pack_id: str,
    complete: bool,
    duration_ns: int | None,
    database_bytes: int,
    clone_count: int,
) -> BenchmarkProfileReportV1:
    return BenchmarkProfileReportV1(
        data=BenchmarkProfileDataV1(
            schema_version=1,
            pack_id=pack_id,
            profile_complete=complete,
            wall_duration_ns=duration_ns,
            database_bytes=database_bytes,
            clone_count=clone_count,
            fts5_available=True,
        )
    )


def _bounded_profiler_value(profiler: Callable[[], int]) -> int | None:
    try:
        value = profiler()
    except Exception:
        return None
    if type(value) is not int or not 0 <= value <= _MAX_PROFILE_CLOCK_NS:
        return None
    return value


def _workspace_overlap() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "benchmark_workspace_input_overlap",
        "Benchmark inputs must remain outside workspace-owned output",
    )


def _require_inputs_outside_owned_workspace(
    workspace_path: Path,
    *input_paths: Path,
) -> None:
    if not isinstance(workspace_path, Path) or not all(
        isinstance(path, Path) for path in input_paths
    ):
        return
    workspace_root = Path(os.path.abspath(workspace_path))
    for input_path in input_paths:
        try:
            relative = Path(os.path.abspath(input_path)).relative_to(workspace_root)
        except ValueError:
            continue
        if (
            relative.parts
            and relative.parts[0] in REPLAY_WORKSPACE_POLICY.owned_entries
        ):
            raise _workspace_overlap()


def _require_no_symlink_components(*paths: Path) -> None:
    if not all(isinstance(path, Path) for path in paths):
        return
    for path in paths:
        absolute = Path(os.path.abspath(path))
        descriptor = -1
        try:
            descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
            for index, part in enumerate(absolute.parts[1:]):
                try:
                    status = os.stat(
                        part,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(status.st_mode):
                    raise _workspace_overlap()
                if index == len(absolute.parts[1:]) - 1 or not stat.S_ISDIR(
                    status.st_mode
                ):
                    break
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except ExperimentIsolationError:
            raise
        except OSError:
            raise _workspace_overlap() from None
        finally:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


async def inspect_benchmark_pack(path: Path) -> BenchmarkInspection:
    """Validate a benchmark pack and capabilities without creating output."""
    loaded = await _complete_threaded(load_benchmark_pack, path)
    await _complete_threaded(preflight_benchmark_capabilities)
    return BenchmarkInspection(
        pack_id=loaded.manifest.pack_id,
        case_count=len(loaded.cases),
        arm_count=len(loaded.manifest.arms),
        manifest_sha256=sha256_hex(loaded.manifest_body),
        cases_sha256=sha256_hex(loaded.cases_body),
        source_fixture_sha256=sha256_hex(loaded.source_body),
        fts5_available=True,
    )


async def run_benchmark_pilot(
    pack_path: Path,
    workspace_path: Path,
    *,
    replace_owned: bool = False,
    profiler: Callable[[], int] = monotonic_ns,
) -> BenchmarkExecution:
    """Run two complete isolated passes and publish truthful artifacts."""
    await _complete_threaded(
        _require_inputs_outside_owned_workspace,
        workspace_path,
        pack_path,
    )
    await _complete_threaded(_require_no_symlink_components, pack_path)
    pack = await _complete_threaded(load_benchmark_pack, pack_path)
    cases_path = pack._parent / pack.manifest.cases.file
    source_fixture_path = pack._parent / pack.manifest.source.file
    await _complete_threaded(
        _require_inputs_outside_owned_workspace,
        workspace_path,
        pack_path,
        cases_path,
        source_fixture_path,
    )
    await _complete_threaded(
        _require_no_symlink_components,
        pack_path,
        cases_path,
        source_fixture_path,
    )
    await _complete_threaded(preflight_benchmark_capabilities)
    workspace = await _complete_threaded(
        prepare_owned_workspace,
        workspace_path,
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=replace_owned,
        allow_unmarked_empty=True,
    )
    source = await build_benchmark_source(pack, workspace)
    revision = await validate_frozen_snapshot(
        source.snapshot,
        validation_path=workspace.root / "validation" / "source.sqlite3",
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )
    resolved = _resolved_manifest(
        pack,
        source,
        source_schema_revision=_schema_revision_number(revision),
    )

    started_ns = _bounded_profiler_value(profiler)
    profile_complete = started_ns is not None

    seen_identities: set[tuple[int, int]] = set()
    first = await _execute_benchmark_cases(
        pass_name="pass-a",
        cases=pack.cases,
        descriptors=pack.manifest.arms,
        workspace=workspace,
        source=source,
        source_records=pack.source,
        seen_identities=seen_identities,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )
    second = await _execute_benchmark_cases(
        pass_name="pass-b",
        cases=pack.cases,
        descriptors=pack.manifest.arms,
        workspace=workspace,
        source=source,
        source_records=pack.source,
        seen_identities=seen_identities,
        frozen_at=pack.manifest.frozen_at,
        seed=pack.manifest.seed,
    )

    duration_ns: int | None = None
    if started_ns is not None:
        finished_ns = _bounded_profiler_value(profiler)
        if finished_ns is not None and finished_ns >= started_ns:
            duration_ns = finished_ns - started_ns
        else:
            profile_complete = False

    source_unchanged = True
    try:
        await _complete_threaded(verify_source_unchanged, source.snapshot)
    except ExperimentIsolationError:
        source_unchanged = False
    expected_clone_count = (
        len(pack.cases)
        * len(pack.manifest.arms)
        * pack.manifest.deterministic_replay_runs
    )
    safety = _combined_safety(
        first,
        second,
        expected_clone_count=expected_clone_count,
        source_unchanged=source_unchanged,
    )
    first_payload = _pass_payload(resolved, first, safety)
    second_payload = _pass_payload(resolved, second, safety)
    first_body = canonical_benchmark_pass_bytes(first_payload)
    second_body = canonical_benchmark_pass_bytes(second_payload)
    deterministic_match = first_body == second_body
    evidence = _evidence_report(
        first_payload,
        deterministic_replay_match=deterministic_match,
    )
    profile: BenchmarkProfileReportV1 | None = None
    if profile_complete and duration_ns is not None:
        try:
            candidate_profile = _profile(
                pack_id=pack.manifest.pack_id,
                complete=True,
                duration_ns=duration_ns,
                database_bytes=len(source.snapshot.database_bytes),
                clone_count=len((*first.clone_identities, *second.clone_identities)),
            )
            canonical_benchmark_profile_bytes(candidate_profile)
        except Exception:
            profile_complete = False
        else:
            profile = candidate_profile

    try:
        artifacts = await _complete_threaded(
            write_benchmark_artifacts,
            workspace,
            evidence=evidence,
            profile=profile,
        )
    except ExperimentOutputError as error:
        if profile is None or error.code not in {
            "invalid_benchmark_profile",
            "output_too_large",
        }:
            raise
        profile_complete = False
        profile = None
        artifacts = await _complete_threaded(
            write_benchmark_artifacts,
            workspace,
            evidence=evidence,
            profile=None,
        )

    verified_evidence = await _complete_threaded(
        verify_benchmark_report,
        artifacts.evidence_path,
    )
    if verified_evidence != evidence:
        raise ExperimentOutputError(
            "invalid_benchmark_evidence",
            "Published benchmark evidence does not match execution",
        )
    summary = verify_benchmark_summary_bytes(
        artifacts.summary_body,
        evidence_body=artifacts.evidence_body,
    )
    profile_body = artifacts.profile_body
    if (
        profile is not None
        and profile_body is not None
        and canonical_benchmark_profile_bytes(profile) != profile_body
    ):
        raise ExperimentOutputError(
            "invalid_benchmark_profile",
            "Published benchmark profile does not match execution",
        )
    return BenchmarkExecution(
        evidence=verified_evidence,
        evidence_body=artifacts.evidence_body,
        summary=summary,
        summary_body=artifacts.summary_body,
        profile=profile,
        profile_body=profile_body,
        evidence_valid=verified_evidence.data.valid,
        comparison_complete=(
            first_payload.comparison_complete and second_payload.comparison_complete
        ),
        deterministic_replay_match=deterministic_match,
        expansion_gate_passed=verified_evidence.data.expansion_gate_passed,
        profile_complete=profile_complete,
    )


def _report_path_error() -> ExperimentOutputError:
    return ExperimentOutputError(
        "invalid_report_path",
        "Benchmark report must be in one safe artifacts directory",
    )


def _open_directory_nofollow(path: Path) -> int:
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise _report_path_error() from None


def _read_report_member(parent_fd: int, name: str) -> bytes:
    descriptor = -1
    try:
        retained = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(retained.st_mode):
            raise _report_path_error()
        if retained.st_size > MAX_BENCHMARK_OUTPUT_BYTES:
            raise ExperimentOutputError(
                "output_too_large",
                "Benchmark report exceeds the output byte limit",
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != retained.st_dev
            or opened.st_ino != retained.st_ino
            or opened.st_size != retained.st_size
        ):
            raise _report_path_error()
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise _report_path_error()
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
            or current.st_dev != final.st_dev
            or current.st_ino != final.st_ino
            or current.st_size != final.st_size
            or current.st_mtime_ns != final.st_mtime_ns
            or current.st_ctime_ns != final.st_ctime_ns
        ):
            raise _report_path_error()
        return b"".join(chunks)
    except ExperimentOutputError:
        raise
    except OSError:
        raise _report_path_error() from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


def verify_benchmark_report(path: Path) -> BenchmarkEvidenceReportV1:
    """Verify evidence and its colocated summary without executing policies."""
    if not isinstance(path, Path):
        raise _report_path_error()
    absolute = Path(os.path.abspath(path))
    if absolute.name != _REPORT_NAME or absolute.parent.name != "artifacts":
        raise _report_path_error()
    parent_fd = -1
    try:
        parent_fd = _open_directory_nofollow(absolute.parent)
        entries = frozenset(os.listdir(parent_fd))
        required = frozenset({_REPORT_NAME, _SUMMARY_NAME})
        allowed = required | frozenset({_PROFILE_NAME})
        if not required.issubset(entries) or not entries.issubset(allowed):
            raise _report_path_error()
        evidence_body = _read_report_member(parent_fd, _REPORT_NAME)
        summary_body = _read_report_member(parent_fd, _SUMMARY_NAME)
        if _PROFILE_NAME in entries:
            _read_report_member(parent_fd, _PROFILE_NAME)
    except ExperimentOutputError:
        raise
    except OSError:
        raise _report_path_error() from None
    finally:
        if parent_fd >= 0:
            with suppress(OSError):
                os.close(parent_fd)
    evidence = verify_benchmark_evidence_bytes(evidence_body)
    verify_benchmark_summary_bytes(summary_body, evidence_body=evidence_body)
    return evidence


__all__ = [
    "BenchmarkExecution",
    "BenchmarkInspection",
    "inspect_benchmark_pack",
    "run_benchmark_pilot",
    "verify_benchmark_report",
]
