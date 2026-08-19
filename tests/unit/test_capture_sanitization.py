from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from experience_hub import sha256_hex
from experience_hub.capture import (
    AdapterDescriptorV1,
    CandidateSignalV1,
    DefaultSecretScanner,
    EvidencePointerV1,
    OutcomeStatus,
    SanitizationDeclarationV1,
    SensitiveField,
    SensitiveMatchV1,
    TrajectoryBundleV1,
    TrajectoryField,
    TrajectoryStepV1,
    hash_trajectory_manifest,
)
from experience_hub.experiences import ExperienceKind

OWNER = UUID("10000000-0000-4000-8000-000000000001")
STARTED_AT = datetime(2026, 7, 22, 1, 0, tzinfo=UTC)


def _synthetic_private_key_marker() -> str:
    return "-----BEGIN " + "PRIVATE KEY-----"


def _synthetic_github_token() -> str:
    return "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"


def _synthetic_openai_key() -> str:
    return "sk-" + "proj-abcdefghijklmnopqrstuvwxyz"


def _step(
    ordinal: int,
    *,
    observation: str = "Observed a benign state.",
    action: str = "Performed a benign action.",
    outcome: str = "Recorded a benign outcome.",
    step_id: str | None = None,
    signal: CandidateSignalV1 | None = None,
) -> TrajectoryStepV1:
    return TrajectoryStepV1(
        step_id=step_id if step_id is not None else f"step-{ordinal}",
        ordinal=ordinal,
        occurred_at=STARTED_AT + timedelta(minutes=ordinal),
        observation=observation,
        action=action,
        outcome=outcome,
        status=OutcomeStatus.SUCCEEDED,
        candidate_signal=signal,
    )


def _bundle(*steps: TrajectoryStepV1) -> TrajectoryBundleV1:
    unchecked = TrajectoryBundleV1.model_construct(
        schema_version=1,
        adapter=AdapterDescriptorV1(kind="generic_jsonl", version=1),
        owner_agent_id=OWNER,
        trajectory_id="trajectory-001",
        source_started_at=STARTED_AT,
        source_completed_at=STARTED_AT + timedelta(hours=1),
        sanitization=SanitizationDeclarationV1(
            profile_id="local-sanitized-v1",
            input_sanitized=True,
        ),
        steps=steps,
        manifest_hash="",
    )
    values = unchecked.model_dump(mode="python")
    values["manifest_hash"] = hash_trajectory_manifest(unchecked)
    return TrajectoryBundleV1.model_validate(values)


def bundle_with_observation(value: str) -> TrajectoryBundleV1:
    return _bundle(_step(1, observation=value))


@pytest.mark.parametrize(
    ("header_field", "reported_field"),
    (
        ("trajectory_id", "trajectory_id"),
        ("sanitization_profile", "sanitization_profile_id"),
    ),
)
def test_scanner_reports_sensitive_persisted_header_without_retaining_it(
    header_field: str,
    reported_field: str,
) -> None:
    probe = _synthetic_openai_key()
    bundle = _bundle(_step(1))
    if header_field == "trajectory_id":
        bundle = bundle.model_copy(update={"trajectory_id": probe})
    else:
        bundle = bundle.model_copy(
            update={
                "sanitization": bundle.sanitization.model_copy(
                    update={"profile_id": probe}
                )
            }
        )

    matches = DefaultSecretScanner().scan(bundle)

    assert tuple(match.model_dump(mode="json") for match in matches) == (
        {
            "field": reported_field,
            "rule_id": "openai_key",
            "step_id": "header",
        },
    )
    assert probe not in repr(matches)


@pytest.mark.parametrize(
    ("value", "rule_id"),
    (
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer_token"),
        (_synthetic_private_key_marker(), "private_key"),
        ("AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
        (_synthetic_github_token(), "github_token"),
        (_synthetic_openai_key(), "openai_key"),
    ),
)
def test_scanner_reports_only_location_and_rule(value: str, rule_id: str) -> None:
    bundle = bundle_with_observation(value)
    matches = DefaultSecretScanner().scan(bundle)

    assert matches == (
        SensitiveMatchV1(
            rule_id=rule_id,
            step_id="step-1",
            field=SensitiveField.OBSERVATION,
        ),
    )
    assert value not in repr(matches)


def test_scanner_uses_exact_raw_field_locations() -> None:
    bundle = _bundle(
        _step(
            1,
            observation="AKIAIOSFODNN7EXAMPLE",
            action=_synthetic_github_token(),
            outcome=_synthetic_openai_key(),
        )
    )

    assert DefaultSecretScanner().scan(bundle) == (
        SensitiveMatchV1(
            rule_id="aws_access_key",
            step_id="step-1",
            field=SensitiveField.OBSERVATION,
        ),
        SensitiveMatchV1(
            rule_id="github_token",
            step_id="step-1",
            field=SensitiveField.ACTION,
        ),
        SensitiveMatchV1(
            rule_id="openai_key",
            step_id="step-1",
            field=SensitiveField.OUTCOME,
        ),
    )


def test_scanner_hashes_an_unsafe_step_id_for_every_match_from_the_step() -> None:
    unsafe_step_id = _synthetic_openai_key()
    safe_marker = f"sha256:{sha256_hex(unsafe_step_id.encode('utf-8'))}"
    signal = CandidateSignalV1(
        kind=ExperienceKind.PROCEDURAL,
        body="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        summary="Benign summary.",
        mechanism="Benign mechanism.",
        tags=("benign",),
        applicability=("local",),
        evidence=(),
        falsifiers=("none",),
    )
    bundle = _bundle(
        _step(
            1,
            step_id=unsafe_step_id,
            observation="AKIAIOSFODNN7EXAMPLE",
            signal=signal,
        )
    )

    matches = DefaultSecretScanner().scan(bundle)

    assert matches == (
        SensitiveMatchV1(
            rule_id="openai_key",
            step_id=safe_marker,
            field=SensitiveField.STEP_ID,
        ),
        SensitiveMatchV1(
            rule_id="aws_access_key",
            step_id=safe_marker,
            field=SensitiveField.OBSERVATION,
        ),
        SensitiveMatchV1(
            rule_id="bearer_token",
            step_id=safe_marker,
            field=SensitiveField.CANDIDATE_SIGNAL,
        ),
    )
    assert unsafe_step_id not in repr(matches)
    assert unsafe_step_id not in repr(
        tuple(match.model_dump(mode="json") for match in matches)
    )


def test_scanner_keeps_order_while_hashing_multiple_unsafe_step_ids() -> None:
    first_id = _synthetic_github_token()
    second_id = _synthetic_openai_key()
    first_marker = f"sha256:{sha256_hex(first_id.encode('utf-8'))}"
    second_marker = f"sha256:{sha256_hex(second_id.encode('utf-8'))}"
    bundle = _bundle(
        _step(
            1,
            step_id=first_id,
            action="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        ),
        _step(
            2,
            step_id=second_id,
            outcome="AKIAIOSFODNN7EXAMPLE",
        ),
    )

    matches = DefaultSecretScanner().scan(bundle)

    assert matches == (
        SensitiveMatchV1(
            rule_id="github_token",
            step_id=first_marker,
            field=SensitiveField.STEP_ID,
        ),
        SensitiveMatchV1(
            rule_id="bearer_token",
            step_id=first_marker,
            field=SensitiveField.ACTION,
        ),
        SensitiveMatchV1(
            rule_id="openai_key",
            step_id=second_marker,
            field=SensitiveField.STEP_ID,
        ),
        SensitiveMatchV1(
            rule_id="aws_access_key",
            step_id=second_marker,
            field=SensitiveField.OUTCOME,
        ),
    )
    retained = repr(tuple(match.model_dump(mode="json") for match in matches))
    assert first_id not in retained
    assert second_id not in retained


def test_scanner_scans_evidence_pointer_step_ids_without_retaining_them() -> None:
    unsafe_step_id = _synthetic_github_token()
    safe_marker = f"sha256:{sha256_hex(unsafe_step_id.encode('utf-8'))}"
    signal = CandidateSignalV1(
        kind=ExperienceKind.PROCEDURAL,
        body="Benign body.",
        summary="Benign summary.",
        mechanism="Benign mechanism.",
        tags=("benign",),
        applicability=("local",),
        evidence=(
            EvidencePointerV1(
                step_id=unsafe_step_id,
                field=TrajectoryField.OBSERVATION,
            ),
        ),
        falsifiers=("none",),
    )
    bundle = _bundle(
        _step(1, step_id=unsafe_step_id),
        _step(2, signal=signal),
    )

    matches = DefaultSecretScanner().scan(bundle)

    assert matches == (
        SensitiveMatchV1(
            rule_id="github_token",
            step_id=safe_marker,
            field=SensitiveField.STEP_ID,
        ),
        SensitiveMatchV1(
            rule_id="github_token",
            step_id="step-2",
            field=SensitiveField.CANDIDATE_SIGNAL,
        ),
    )
    assert unsafe_step_id not in repr(matches)
    assert unsafe_step_id not in repr(
        tuple(match.model_dump(mode="json") for match in matches)
    )


@pytest.mark.parametrize(
    "signal_field",
    ("body", "summary", "mechanism", "tags", "applicability", "falsifiers"),
)
def test_scanner_checks_each_signal_string_location(signal_field: str) -> None:
    value = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    signal = CandidateSignalV1(
        kind=ExperienceKind.PROCEDURAL,
        body=value if signal_field == "body" else "Benign body.",
        summary=value if signal_field == "summary" else "Benign summary.",
        mechanism=value if signal_field == "mechanism" else "Benign mechanism.",
        tags=(value,) if signal_field == "tags" else ("benign",),
        applicability=(value,) if signal_field == "applicability" else ("local",),
        evidence=(),
        falsifiers=(value,) if signal_field == "falsifiers" else ("none",),
    )
    bundle = _bundle(_step(1, signal=signal))

    assert DefaultSecretScanner().scan(bundle) == (
        SensitiveMatchV1(
            rule_id="bearer_token",
            step_id="step-1",
            field=SensitiveField.CANDIDATE_SIGNAL,
        ),
    )


def test_scanner_collapses_duplicate_signal_matches_and_keeps_rule_order() -> None:
    signal = CandidateSignalV1(
        kind=ExperienceKind.PROCEDURAL,
        body="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        summary=_synthetic_private_key_marker(),
        mechanism="AKIAIOSFODNN7EXAMPLE",
        tags=(
            _synthetic_github_token(),
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        ),
        applicability=(_synthetic_openai_key(),),
        evidence=(),
        falsifiers=(
            _synthetic_github_token(),
            _synthetic_private_key_marker(),
        ),
    )
    bundle = _bundle(_step(1, signal=signal))

    assert DefaultSecretScanner().scan(bundle) == tuple(
        SensitiveMatchV1(
            rule_id=rule_id,
            step_id="step-1",
            field=SensitiveField.CANDIDATE_SIGNAL,
        )
        for rule_id in (
            "bearer_token",
            "private_key",
            "aws_access_key",
            "github_token",
            "openai_key",
        )
    )


def test_scanner_does_not_flag_benign_uuid_or_sha256_values() -> None:
    uuid_value = "10000000-0000-4000-8000-000000000001"
    sha256_value = "0123456789abcdef" * 4
    signal = CandidateSignalV1(
        kind=ExperienceKind.SEMANTIC,
        body=f"Observed identifiers {uuid_value} and {sha256_value}.",
        summary=f"Identifier {uuid_value}",
        mechanism=f"Digest {sha256_value}",
        tags=(uuid_value,),
        applicability=(sha256_value,),
        evidence=(),
        falsifiers=(uuid_value, sha256_value),
    )
    bundle = _bundle(
        _step(
            1,
            observation=uuid_value,
            action=sha256_value,
            outcome=f"{uuid_value}:{sha256_value}",
            signal=signal,
        )
    )

    assert DefaultSecretScanner().scan(bundle) == ()
