from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments import (
    CaseEvidenceV1,
    ExperimentInputError,
    ReplayCaseV1,
    ReplayManifestV1,
    ResolvedReplayManifestV1,
    load_replay_cases,
    load_replay_manifest,
)


def _case_document(case_id: str = "queue-case") -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "owner_agent_id": "12345678-1234-5678-1234-567812345678",
        "query": "inspect the deployment queue",
        "mode": "focused",
        "tags": ["queue"],
        "mechanism_cues": ["backpressure"],
        "limit": 5,
        "content_budget_bytes": 512,
        "expand_cold": False,
        "expected": [
            {
                "label": "queue-guide",
                "experience_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            }
        ],
        "forbidden": [
            {
                "label": "foreign-guide",
                "experience_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            }
        ],
    }


def _manifest_document(
    *,
    cases_file: str = "cases.jsonl",
    cases_sha256: str = "0" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "contract-smoke",
        "dataset": {
            "schema_version": 1,
            "dataset_id": "contract-cases",
            "cases_file": cases_file,
            "cases_sha256": cases_sha256,
        },
        "snapshot_binding": "validated_source",
        "frozen_at": "2026-07-26T00:00:00Z",
        "seed": 7,
        "arms": [
            {
                "schema_version": 1,
                "arm_id": "no_memory",
                "kind": "no_memory",
                "required": True,
            },
            {
                "schema_version": 1,
                "arm_id": "experience_hub",
                "kind": "experience_hub",
                "required": True,
            },
        ],
        "oracle": {
            "schema_version": 1,
            "kind": "retrieval_labels",
            "version": 1,
        },
        "evidence_schema_version": 1,
        "profile_schema_version": 1,
        "deterministic_replay_runs": 2,
    }


def _write_fixture_pair(
    tmp_path: Path,
    *,
    cases: bytes | None = None,
    manifest: dict[str, object] | None = None,
) -> Path:
    actual_cases = cases or canonical_json_bytes(_case_document()) + b"\n"
    (tmp_path / "cases.jsonl").write_bytes(actual_cases)
    actual_manifest = manifest or _manifest_document(
        cases_sha256=sha256_hex(actual_cases)
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(actual_manifest))
    return manifest_path


def _resolved_manifest_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "contract-smoke",
        "manifest_sha256": "a" * 64,
        "dataset_id": "contract-cases",
        "cases_sha256": "b" * 64,
        "snapshot_sha256": "c" * 64,
        "source_schema_revision": 1,
        "frozen_at": "2026-07-26T00:00:00Z",
        "seed": 7,
        "policy_arms": _manifest_document()["arms"],
        "oracle": _manifest_document()["oracle"],
        "evidence_schema_version": 1,
        "profile_schema_version": 1,
    }


def test_manifest_and_cases_load_with_exact_hash_closure(tmp_path: Path) -> None:
    cases = canonical_json_bytes(_case_document("queue-case")) + b"\n"
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_bytes(cases)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            _manifest_document(
                cases_file="cases.jsonl",
                cases_sha256=sha256_hex(cases),
            )
        )
    )

    loaded = load_replay_manifest(manifest_path)
    dataset = load_replay_cases(loaded)

    assert loaded.manifest.experiment_id == "contract-smoke"
    assert loaded.body == manifest_path.read_bytes()
    assert dataset.body == cases
    assert tuple(case.case_id for case in dataset.cases) == ("queue-case",)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {**_manifest_document(), "unexpected": "value"},
            "Extra inputs are not permitted",
        ),
        (
            {**_manifest_document(), "schema_version": 2},
            "Input should be 1",
        ),
        (
            {**_manifest_document(), "frozen_at": "2026-07-26T00:00:00+08:00"},
            "UTC",
        ),
        (
            {
                **_manifest_document(),
                "arms": [
                    {
                        "schema_version": 1,
                        "arm_id": "no_memory",
                        "kind": "no_memory",
                        "required": True,
                    },
                    {
                        "schema_version": 1,
                        "arm_id": "no_memory",
                        "kind": "no_memory",
                        "required": True,
                    },
                ],
            },
            "ordered",
        ),
        (
            {
                **_manifest_document(),
                "arms": [
                    {
                        "schema_version": 1,
                        "arm_id": "experience_hub",
                        "kind": "experience_hub",
                        "required": True,
                    },
                    {
                        "schema_version": 1,
                        "arm_id": "no_memory",
                        "kind": "no_memory",
                        "required": True,
                    },
                ],
            },
            "ordered",
        ),
    ],
)
def test_manifest_rejects_invalid_versioned_contract(
    document: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ReplayManifestV1.model_validate_json(canonical_json_bytes(document))


@pytest.mark.parametrize("arm_id", ["no_memory", "experience_hub"])
def test_manifest_rejects_each_missing_required_policy_arm(arm_id: str) -> None:
    document = _manifest_document()
    document["arms"] = [
        {
            "schema_version": 1,
            "arm_id": arm_id,
            "kind": arm_id,
            "required": True,
        }
    ]

    with pytest.raises(ValidationError, match="exactly two"):
        ReplayManifestV1.model_validate_json(canonical_json_bytes(document))


def test_manifest_rejects_duplicate_policy_arm_id() -> None:
    document = _manifest_document()
    document["arms"] = [
        {
            "schema_version": 1,
            "arm_id": "no_memory",
            "kind": "no_memory",
            "required": True,
        },
        {
            "schema_version": 1,
            "arm_id": "no_memory",
            "kind": "no_memory",
            "required": True,
        },
    ]

    with pytest.raises(ValidationError, match="ordered"):
        ReplayManifestV1.model_validate_json(canonical_json_bytes(document))


def test_manifest_rejects_unsupported_policy_kind() -> None:
    document = _manifest_document()
    document["arms"] = [
        {
            "schema_version": 1,
            "arm_id": "no_memory",
            "kind": "recent_notes",
            "required": True,
        },
        {
            "schema_version": 1,
            "arm_id": "experience_hub",
            "kind": "experience_hub",
            "required": True,
        },
    ]

    with pytest.raises(ValidationError, match="no_memory"):
        ReplayManifestV1.model_validate_json(canonical_json_bytes(document))


@pytest.mark.parametrize(
    ("cases_file", "message"),
    [("/tmp/cases.jsonl", "cases_file"), ("../cases.jsonl", "cases_file")],
)
def test_manifest_rejects_nonlocal_cases_filename(
    cases_file: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ReplayManifestV1.model_validate_json(
            canonical_json_bytes(_manifest_document(cases_file=cases_file))
        )


def test_case_rejects_duplicate_and_overlapping_labels() -> None:
    document = _case_document()
    document["expected"] = [
        {
            "label": "queue-guide",
            "experience_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        },
        {
            "label": "queue-guide",
            "experience_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        },
    ]
    with pytest.raises(ValidationError, match="unique"):
        ReplayCaseV1.model_validate_json(canonical_json_bytes(document))

    document = _case_document()
    document["forbidden"] = [
        {
            "label": "queue-guide",
            "experience_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        }
    ]
    with pytest.raises(ValidationError, match="overlap"):
        ReplayCaseV1.model_validate_json(canonical_json_bytes(document))

    document = _case_document()
    document["forbidden"] = [
        {
            "label": "queue-guide",
            "experience_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        }
    ]
    with pytest.raises(
        ValidationError,
        match="expected and forbidden labels must not overlap",
    ):
        ReplayCaseV1.model_validate_json(canonical_json_bytes(document))


def test_case_rejects_boolean_for_integer() -> None:
    document = _case_document()
    document["limit"] = True

    with pytest.raises(ValidationError):
        ReplayCaseV1.model_validate_json(canonical_json_bytes(document))


def test_case_accepts_only_utc_datetime_values_after_decoding() -> None:
    manifest = ReplayManifestV1.model_validate_json(
        canonical_json_bytes(_manifest_document())
    )

    assert manifest.frozen_at == datetime(2026, 7, 26, tzinfo=UTC)
    assert ReplayCaseV1.model_validate_json(
        canonical_json_bytes(_case_document())
    ).owner_agent_id == UUID(
        "12345678-1234-5678-1234-567812345678"
    )


def test_case_evidence_accepts_the_required_no_memory_arm_identifier() -> None:
    evidence = CaseEvidenceV1.model_validate_json(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "case_id": "queue-case",
                "status": "complete",
                "arms": [
                    {
                        "schema_version": 1,
                        "arm_id": "no_memory",
                        "status": "complete",
                        "observation": {
                            "schema_version": 1,
                            "returned_labels": [],
                            "unmapped_count": 0,
                        },
                        "utility_micros": 500_000,
                        "error_code": None,
                        "error_stage": None,
                    }
                ],
                "delta_utility_micros": 0,
            }
        )
    )

    assert evidence.arms[0].arm_id == "no_memory"


def test_resolved_manifest_rejects_unordered_or_duplicate_policy_arms() -> None:
    document = _resolved_manifest_document()
    document["policy_arms"] = [
        {
            "schema_version": 1,
            "arm_id": "no_memory",
            "kind": "no_memory",
            "required": True,
        },
        {
            "schema_version": 1,
            "arm_id": "no_memory",
            "kind": "no_memory",
            "required": True,
        },
    ]

    with pytest.raises(ValidationError, match="ordered"):
        ResolvedReplayManifestV1.model_validate_json(canonical_json_bytes(document))


def test_loader_rejects_noncanonical_manifest_without_exposing_path(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_pair(tmp_path)
    manifest_path.write_bytes(b'{"schema_version": 1}')

    with pytest.raises(ExperimentInputError) as raised:
        load_replay_manifest(manifest_path)

    assert str(tmp_path) not in str(raised.value)


def test_loader_maps_nonfinite_manifest_json_to_a_stable_input_error(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_pair(tmp_path)
    manifest_path.write_bytes(b'{"seed":NaN}')

    with pytest.raises(ExperimentInputError):
        load_replay_manifest(manifest_path)


def test_loader_maps_oversized_manifest_integer_to_invalid_json(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_pair(tmp_path)
    manifest_path.write_bytes(b'{"seed":' + b"1" * 5_000 + b"}")

    with pytest.raises(ExperimentInputError) as raised:
        load_replay_manifest(manifest_path)

    assert raised.value.code == "invalid_json"


def test_loader_maps_deeply_nested_case_to_invalid_json(tmp_path: Path) -> None:
    nested_case = b'{"value":' * 1_000 + b"0" + b"}" * 1_000 + b"\n"
    manifest_path = _write_fixture_pair(
        tmp_path,
        cases=nested_case,
        manifest=_manifest_document(cases_sha256=sha256_hex(nested_case)),
    )

    with pytest.raises(ExperimentInputError) as raised:
        load_replay_cases(load_replay_manifest(manifest_path))

    assert raised.value.code == "invalid_json"


def test_loader_rejects_noncanonical_case_line_and_trailing_blank_line(
    tmp_path: Path,
) -> None:
    noncanonical = b'{"schema_version": 1}\n'
    manifest_path = _write_fixture_pair(
        tmp_path,
        cases=noncanonical,
        manifest=_manifest_document(cases_sha256=sha256_hex(noncanonical)),
    )

    with pytest.raises(ExperimentInputError):
        load_replay_cases(load_replay_manifest(manifest_path))

    canonical = canonical_json_bytes(_case_document()) + b"\n\n"
    manifest_path = _write_fixture_pair(
        tmp_path,
        cases=canonical,
        manifest=_manifest_document(cases_sha256=sha256_hex(canonical)),
    )
    with pytest.raises(ExperimentInputError):
        load_replay_cases(load_replay_manifest(manifest_path))


def test_loader_rejects_hash_mismatch_and_input_bounds(tmp_path: Path) -> None:
    cases = canonical_json_bytes(_case_document()) + b"\n"
    manifest_path = _write_fixture_pair(
        tmp_path,
        cases=cases,
        manifest=_manifest_document(cases_sha256="0" * 64),
    )
    with pytest.raises(ExperimentInputError):
        load_replay_cases(load_replay_manifest(manifest_path))

    too_many = b"".join(
        canonical_json_bytes(_case_document(f"case-{index}")) + b"\n"
        for index in range(257)
    )
    manifest_path = _write_fixture_pair(
        tmp_path,
        cases=too_many,
        manifest=_manifest_document(cases_sha256=sha256_hex(too_many)),
    )
    with pytest.raises(ExperimentInputError):
        load_replay_cases(load_replay_manifest(manifest_path))

    oversized = b" " * (2 * 1024 * 1024 + 1)
    manifest_path = _write_fixture_pair(tmp_path, cases=oversized)
    with pytest.raises(ExperimentInputError):
        load_replay_cases(load_replay_manifest(manifest_path))
