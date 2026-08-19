from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from experience_hub.canonical import canonical_json_bytes
from experience_hub.experiments import (
    REPLAY_WORKSPACE_POLICY,
    ArmEvidenceV1,
    ArmObservationV1,
    CaseEvidenceV1,
    ExperimentIsolationError,
    ExperimentOutputError,
    OracleDescriptorV1,
    PolicyArmDescriptorV1,
    PolicyArmKind,
    ReplayEvidenceReportV1,
    ReplayProfileReportV1,
    ResolvedReplayManifestV1,
    canonical_evidence_bytes,
    canonical_profile_bytes,
    prepare_owned_workspace,
    verify_evidence_bytes,
    write_replay_artifacts,
)
from experience_hub.experiments.contracts import (
    ReplayEvidenceDataV1,
    ReplayProfileDataV1,
)
from experience_hub.experiments.workspace import OwnedWorkspace

FROZEN_AT = datetime(2026, 7, 26, tzinfo=UTC)
PRIVATE_SERIALIZER_DETAIL = "/private/owner/source.sqlite3"


class LeakyEvidenceReport(ReplayEvidenceReportV1):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(PRIVATE_SERIALIZER_DETAIL)


class LeakyProfileReport(ReplayProfileReportV1):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(PRIVATE_SERIALIZER_DETAIL)


def _policy_arms() -> tuple[PolicyArmDescriptorV1, ...]:
    return (
        PolicyArmDescriptorV1(
            schema_version=1,
            arm_id="no_memory",
            kind=PolicyArmKind.NO_MEMORY,
            required=True,
        ),
        PolicyArmDescriptorV1(
            schema_version=1,
            arm_id="experience_hub",
            kind=PolicyArmKind.EXPERIENCE_HUB,
            required=True,
        ),
    )


def _arm(arm_id: str, utility_micros: int) -> ArmEvidenceV1:
    return ArmEvidenceV1(
        schema_version=1,
        arm_id=arm_id,
        status="complete",
        observation=ArmObservationV1(
            schema_version=1,
            returned_labels=(),
            unmapped_count=0,
        ),
        utility_micros=utility_micros,
        error_code=None,
        error_stage=None,
    )


def _valid_report() -> ReplayEvidenceReportV1:
    case = CaseEvidenceV1(
        schema_version=1,
        case_id="queue-case",
        status="complete",
        arms=(
            _arm("no_memory", 0),
            _arm("experience_hub", 750_000),
        ),
        delta_utility_micros=750_000,
    )
    return ReplayEvidenceReportV1(
        data=ReplayEvidenceDataV1(
            schema_version=1,
            resolved_manifest=ResolvedReplayManifestV1(
                schema_version=1,
                experiment_id="contract-smoke",
                manifest_sha256="a" * 64,
                dataset_id="contract-cases",
                cases_sha256="b" * 64,
                snapshot_sha256="c" * 64,
                source_schema_revision=1,
                frozen_at=FROZEN_AT,
                seed=7,
                policy_arms=_policy_arms(),
                oracle=OracleDescriptorV1(
                    schema_version=1,
                    kind="retrieval_labels",
                    version=1,
                ),
                evidence_schema_version=1,
                profile_schema_version=1,
            ),
            cases=(case,),
            comparison_complete=True,
            source_unchanged=True,
            clone_isolation_verified=True,
            deterministic_replay_match=True,
            valid=True,
        )
    )


def _valid_profile(
    *,
    wall_duration_ns: int = 25,
    database_bytes: int = 4_096,
) -> ReplayProfileReportV1:
    return ReplayProfileReportV1(
        data=ReplayProfileDataV1(
            schema_version=1,
            experiment_id="contract-smoke",
            profile_complete=True,
            wall_duration_ns=wall_duration_ns,
            database_bytes=database_bytes,
        )
    )


def _replace_data(
    report: ReplayEvidenceReportV1,
    **update: object,
) -> ReplayEvidenceReportV1:
    return report.model_copy(update={"data": report.data.model_copy(update=update)})


def _replace_manifest(
    report: ReplayEvidenceReportV1,
    **update: object,
) -> ReplayEvidenceReportV1:
    manifest = report.data.resolved_manifest.model_copy(update=update)
    return _replace_data(report, resolved_manifest=manifest)


def _replace_case(
    report: ReplayEvidenceReportV1,
    case: CaseEvidenceV1,
) -> ReplayEvidenceReportV1:
    return _replace_data(report, cases=(case,))


def test_evidence_round_trips_as_exact_canonical_bytes() -> None:
    report = _valid_report()

    body = canonical_evidence_bytes(report)

    assert verify_evidence_bytes(body) == report
    assert body == canonical_json_bytes(report)
    assert body.count(b"\n") == 0


@pytest.mark.parametrize(
    "forbidden",
    (
        "/private/tmp/source.sqlite3",
        "C:\\Users\\name\\source.sqlite3",
        "file:///tmp/report.json",
        "sqlite:///tmp/source.sqlite3",
        "10000000-0000-4000-8000-000000000701",
        "2026-07-26T10:11:12Z",
    ),
)
def test_evidence_rejects_unstable_or_private_strings(forbidden: str) -> None:
    report = _replace_manifest(_valid_report(), experiment_id=forbidden)

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(report)


@pytest.mark.parametrize(
    "key",
    (
        "database",
        "database_path",
        "path",
        "event_id",
        "receipt_id",
        "run_id",
        "created_at",
        "completed_at",
        "elapsed_ns",
        "wall_duration_ns",
        "exception",
        "traceback",
        "api_key",
        "token",
        "secret",
    ),
)
def test_evidence_recursively_rejects_every_forbidden_key(key: str) -> None:
    report = _valid_report()
    manifest = report.data.resolved_manifest.model_dump(mode="python")
    manifest[key] = "logical-label"
    tampered = _replace_data(report, resolved_manifest=manifest)

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(tampered)


def test_evidence_rejects_raw_exception_without_leaking_its_text() -> None:
    private_text = "/private/owner/database.sqlite3"
    report = _replace_manifest(
        _valid_report(),
        experiment_id=RuntimeError(private_text),
    )

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_evidence_bytes(report)

    assert private_text not in str(captured.value)
    assert captured.value.__cause__ is None


def test_evidence_serializer_failure_is_stable_and_private_detail_free() -> None:
    report = LeakyEvidenceReport(data=_valid_report().data)

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_evidence_bytes(report)

    assert captured.value.code == "invalid_evidence"
    assert (
        captured.value.message
        == "Replay evidence does not match the versioned schema"
    )
    assert PRIVATE_SERIALIZER_DETAIL not in str(captured.value)
    assert captured.value.__cause__ is None


def test_evidence_preserves_explicit_output_error_from_serializer() -> None:
    expected = ExperimentOutputError("explicit_rejection", "Explicit rejection")

    class RejectingEvidenceReport(ReplayEvidenceReportV1):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise expected

    report = RejectingEvidenceReport(data=_valid_report().data)

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_evidence_bytes(report)

    assert captured.value is expected


def test_evidence_does_not_swallow_keyboard_interrupt_from_serializer() -> None:
    class InterruptingEvidenceReport(ReplayEvidenceReportV1):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

    report = InterruptingEvidenceReport(data=_valid_report().data)

    with pytest.raises(KeyboardInterrupt):
        canonical_evidence_bytes(report)


def test_evidence_allows_only_resolved_manifest_frozen_timestamp() -> None:
    assert verify_evidence_bytes(canonical_evidence_bytes(_valid_report()))

    report = _replace_manifest(
        _valid_report(),
        dataset_id="2026-07-26T10:11:12Z",
    )
    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(report)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("manifest_sha256", "a" * 63),
        ("dataset_id", "Contract-Cases"),
        ("cases_sha256", "B" * 64),
        ("snapshot_sha256", "not-a-digest"),
    ),
)
def test_evidence_revalidates_the_resolved_hash_closure(
    field: str,
    tampered: str,
) -> None:
    report = _replace_manifest(_valid_report(), **{field: tampered})

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(report)


def test_verify_rejects_unknown_json_field() -> None:
    document = json.loads(canonical_evidence_bytes(_valid_report()))
    document["data"]["unknown"] = True

    with pytest.raises(ExperimentOutputError):
        verify_evidence_bytes(canonical_json_bytes(document))


def test_verify_rejects_noncanonical_json_bytes() -> None:
    document = json.loads(canonical_evidence_bytes(_valid_report()))
    noncanonical = json.dumps(document, indent=2).encode("utf-8")

    with pytest.raises(ExperimentOutputError):
        verify_evidence_bytes(noncanonical)


def test_verify_rejects_evidence_over_the_byte_limit() -> None:
    with pytest.raises(ExperimentOutputError):
        verify_evidence_bytes(b" " * (2 * 1024 * 1024 + 1))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_unchanged", False),
        ("clone_isolation_verified", False),
        ("deterministic_replay_match", False),
    ),
)
def test_valid_evidence_requires_every_validity_assertion(
    field: str,
    value: bool,
) -> None:
    report = _replace_data(_valid_report(), **{field: value})

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(report)


def test_comparison_complete_must_match_case_completeness() -> None:
    report = _replace_data(_valid_report(), comparison_complete=False)

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(report)


def test_complete_case_requires_both_ordered_manifest_arms() -> None:
    report = _valid_report()
    case = report.data.cases[0].model_copy(
        update={"arms": (report.data.cases[0].arms[0],)}
    )

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(_replace_case(report, case))


def test_complete_arm_requires_observation_utility_and_no_error() -> None:
    report = _valid_report()
    failed_arm = report.data.cases[0].arms[1].model_copy(
        update={"utility_micros": None}
    )
    case = report.data.cases[0].model_copy(
        update={"arms": (report.data.cases[0].arms[0], failed_arm)}
    )

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(_replace_case(report, case))


def test_incomplete_case_requires_null_delta() -> None:
    report = _valid_report()
    failed_arm = report.data.cases[0].arms[1].model_copy(
        update={
            "status": "failed",
            "observation": None,
            "utility_micros": None,
            "error_code": "arm-failed",
            "error_stage": "execute",
        }
    )
    case = report.data.cases[0].model_copy(
        update={
            "status": "incomplete",
            "arms": (report.data.cases[0].arms[0], failed_arm),
            "delta_utility_micros": 750_000,
        }
    )
    tampered = _replace_data(
        _replace_case(report, case),
        comparison_complete=False,
        valid=False,
    )

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(tampered)


def test_complete_case_delta_is_recomputed_from_arm_utilities() -> None:
    report = _valid_report()
    case = report.data.cases[0].model_copy(update={"delta_utility_micros": 1})

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(_replace_case(report, case))


def test_failed_arm_is_valid_only_with_stable_error_fields() -> None:
    report = _valid_report()
    failed_arm = report.data.cases[0].arms[1].model_copy(
        update={
            "status": "failed",
            "observation": None,
            "utility_micros": None,
            "error_code": "arm-failed",
            "error_stage": "execute",
        }
    )
    case = report.data.cases[0].model_copy(
        update={
            "status": "incomplete",
            "arms": (report.data.cases[0].arms[0], failed_arm),
            "delta_utility_micros": None,
        }
    )
    incomplete = _replace_data(
        _replace_case(report, case),
        comparison_complete=False,
        valid=False,
    )

    assert verify_evidence_bytes(canonical_evidence_bytes(incomplete)) == incomplete


def test_incomplete_case_does_not_skip_validation_of_later_cases() -> None:
    report = _valid_report()
    failed_arm = report.data.cases[0].arms[1].model_copy(
        update={
            "status": "failed",
            "observation": None,
            "utility_micros": None,
            "error_code": "arm-failed",
            "error_stage": "execute",
        }
    )
    incomplete = report.data.cases[0].model_copy(
        update={
            "status": "incomplete",
            "arms": (report.data.cases[0].arms[0], failed_arm),
            "delta_utility_micros": None,
        }
    )
    invalid_later_case = report.data.cases[0].model_copy(
        update={
            "case_id": "later-case",
            "arms": (report.data.cases[0].arms[0],),
        }
    )
    tampered = _replace_data(
        report,
        cases=(incomplete, invalid_later_case),
        comparison_complete=False,
        valid=False,
    )

    with pytest.raises(ExperimentOutputError):
        canonical_evidence_bytes(tampered)


def test_profile_keeps_runtime_values_out_of_evidence_and_its_hash() -> None:
    evidence = _valid_report()
    evidence_body = canonical_evidence_bytes(evidence)

    first = canonical_profile_bytes(
        _valid_profile(wall_duration_ns=25, database_bytes=4_096)
    )
    second = canonical_profile_bytes(
        _valid_profile(wall_duration_ns=99, database_bytes=8_192)
    )

    assert first != second
    assert canonical_evidence_bytes(evidence) == evidence_body
    assert b"wall_duration_ns" not in evidence_body
    assert b"database_bytes" not in evidence_body


def test_profile_serializer_failure_is_stable_and_private_detail_free() -> None:
    report = LeakyProfileReport(data=_valid_profile().data)

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_profile_bytes(report)

    assert captured.value.code == "invalid_profile"
    assert (
        captured.value.message
        == "Replay profile does not match the versioned schema"
    )
    assert PRIVATE_SERIALIZER_DETAIL not in str(captured.value)
    assert captured.value.__cause__ is None


def test_profile_preserves_explicit_output_error_from_serializer() -> None:
    expected = ExperimentOutputError("explicit_rejection", "Explicit rejection")

    class RejectingProfileReport(ReplayProfileReportV1):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise expected

    report = RejectingProfileReport(data=_valid_profile().data)

    with pytest.raises(ExperimentOutputError) as captured:
        canonical_profile_bytes(report)

    assert captured.value is expected


def test_profile_does_not_swallow_system_exit_from_serializer() -> None:
    class ExitingProfileReport(ReplayProfileReportV1):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise SystemExit

    report = ExitingProfileReport(data=_valid_profile().data)

    with pytest.raises(SystemExit):
        canonical_profile_bytes(report)


def test_write_artifacts_returns_exact_owned_paths_and_bytes(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "replay",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    evidence = _valid_report()
    profile = _valid_profile()

    artifacts = write_replay_artifacts(
        workspace,
        evidence=evidence,
        profile=profile,
    )

    assert artifacts.evidence_path == workspace.root / "artifacts/evidence.json"
    assert artifacts.profile_path == workspace.root / "artifacts/profile.json"
    assert artifacts.evidence_path.read_bytes() == artifacts.evidence_body
    assert artifacts.profile_path.read_bytes() == artifacts.profile_body


def test_write_artifacts_maps_profile_serializer_failure_without_leaking(
    tmp_path: Path,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "replay",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    profile = LeakyProfileReport(data=_valid_profile().data)

    with pytest.raises(ExperimentOutputError) as captured:
        write_replay_artifacts(
            workspace,
            evidence=_valid_report(),
            profile=profile,
        )

    assert captured.value.code == "invalid_profile"
    assert PRIVATE_SERIALIZER_DETAIL not in str(captured.value)
    assert captured.value.__cause__ is None
    assert not (workspace.root / "artifacts").exists()


def test_profile_write_failure_preserves_previous_valid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "replay",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )
    previous = canonical_evidence_bytes(_valid_report())
    workspace.atomic_write(PurePosixPath("artifacts/evidence.json"), previous)
    replacement_report = _replace_manifest(_valid_report(), seed=8)
    real_atomic_write = OwnedWorkspace.atomic_write

    def fail_profile(
        owned: OwnedWorkspace,
        relative: PurePosixPath,
        body: bytes,
    ) -> Path:
        if relative == PurePosixPath("artifacts/profile.json"):
            raise ExperimentIsolationError("private-code", "/private/detail")
        return real_atomic_write(owned, relative, body)

    monkeypatch.setattr(OwnedWorkspace, "atomic_write", fail_profile)

    with pytest.raises(ExperimentOutputError) as captured:
        write_replay_artifacts(
            workspace,
            evidence=replacement_report,
            profile=_valid_profile(),
        )

    assert (workspace.root / "artifacts/evidence.json").read_bytes() == previous
    assert "/private/detail" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_write_artifacts_accepts_omitted_profile(tmp_path: Path) -> None:
    workspace = prepare_owned_workspace(
        tmp_path / "replay",
        policy=REPLAY_WORKSPACE_POLICY,
        replace_owned=False,
        allow_unmarked_empty=False,
    )

    artifacts = write_replay_artifacts(
        workspace,
        evidence=_valid_report(),
        profile=None,
    )

    assert artifacts.profile_path is None
    assert artifacts.profile_body is None
