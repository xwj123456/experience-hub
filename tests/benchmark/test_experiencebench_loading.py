from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.benchmark.experiencebench_factories import (
    valid_case_document,
    valid_manifest_document,
    valid_source_agent_document,
    valid_source_experience_document,
)

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    MAX_BENCHMARK_INPUT_BYTES,
    MAX_BENCHMARK_TOTAL_INPUT_BYTES,
)
from experience_hub.experiments.benchmarks.loading import load_benchmark_pack
from experience_hub.experiments.errors import ExperimentInputError


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _case(index: int) -> dict[str, object]:
    document = valid_case_document()
    case_id = f"case-{index}"
    labels = [
        f"{case_id}-required",
        f"{case_id}-optional",
        f"{case_id}-forbidden",
        f"{case_id}-stale",
        f"{case_id}-misleading",
    ]
    document.update(
        {
            "case_id": case_id,
            "source_class": "public_authored" if index < 20 else "reviewed_abstraction",
            "review_status": "authored" if index < 20 else "maintainer_reviewed",
            "stratum": (
                "recurring_workflow",
                "environment_gotcha",
                "state_change",
                "failure_recovery",
                "irrelevant_distractor",
            )[index // 6],
            "language": ("zh", "en", "mixed")[index % 3],
            "owner_label": f"owner-{index}",
            "source_labels": labels,
            "required": [{"label": labels[0], "weight_micros": 450000}],
            "optional": [{"label": labels[1], "weight_micros": 0}],
            "forbidden": [{"label": labels[2], "weight_micros": 100000}],
            "stale": [{"label": labels[3], "weight_micros": 100000}],
            "misleading": [{"label": labels[4], "weight_micros": 100000}],
            "checkpoints": [
                {
                    "predicate": "required_set",
                    "labels": [labels[0]],
                    "weight_micros": 150000,
                }
            ],
        }
    )
    return document


def _source_for(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    source: list[dict[str, object]] = []
    for case in cases:
        agent = valid_source_agent_document()
        agent["label"] = case["owner_label"]
        source.append(agent)
    created_at = datetime(2026, 8, 19, tzinfo=UTC)
    for index, case in enumerate(cases):
        for offset, label in enumerate(case["source_labels"]):
            experience = valid_source_experience_document()
            experience.update(
                {
                    "label": label,
                    "owner_label": case["owner_label"],
                    "created_at": (created_at + timedelta(seconds=index * 5 + offset))
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
            source.append(experience)
    return source


def _write_pack(
    tmp_path: Path,
    *,
    cases: list[dict[str, object]] | None = None,
    source: list[dict[str, object]] | None = None,
    manifest: dict[str, object] | None = None,
) -> Path:
    pack_cases = [_case(index) for index in range(30)] if cases is None else cases
    pack_source = _source_for(pack_cases) if source is None else source
    cases_body = _jsonl(pack_cases)
    source_body = _jsonl(pack_source)
    manifest_document = (
        valid_manifest_document(
            cases_sha256=sha256_hex(cases_body), source_sha256=sha256_hex(source_body)
        )
        if manifest is None
        else manifest
    )
    (tmp_path / "pilot-cases.jsonl").write_bytes(cases_body)
    (tmp_path / "pilot-source.jsonl").write_bytes(source_body)
    manifest_path = tmp_path / "pilot-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest_document))
    return manifest_path


def _rewrite_manifest(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _manifest_for(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _assert_rejected(path: Path, code: str) -> None:
    with pytest.raises(ExperimentInputError) as raised:
        load_benchmark_pack(path)
    assert raised.value.code == code
    assert str(path) not in str(raised.value)


def test_loads_hash_closed_canonical_pilot_pack(tmp_path: Path) -> None:
    manifest_path = _write_pack(tmp_path)

    loaded = load_benchmark_pack(manifest_path)

    assert len(loaded.cases) == 30
    assert tuple(arm.kind for arm in loaded.manifest.arms) == BENCHMARK_ARM_ORDER
    assert loaded.source_labels == frozenset(record.label for record in loaded.source)
    assert loaded.total_input_bytes <= MAX_BENCHMARK_TOTAL_INPUT_BYTES


def test_rejects_noncanonical_manifest_json(tmp_path: Path) -> None:
    manifest_path = _write_pack(tmp_path)
    manifest_path.write_text(
        json.dumps(_manifest_for(manifest_path), indent=2), "utf-8"
    )

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_symbolic_linked_data_file(tmp_path: Path) -> None:
    manifest_path = _write_pack(tmp_path)
    source_path = tmp_path / "pilot-source.jsonl"
    target_path = tmp_path / "outside-source.jsonl"
    source_path.rename(target_path)
    os.symlink(target_path, source_path)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


@pytest.mark.parametrize("filename", ["/tmp/pack.jsonl", "../pack.jsonl"])
def test_rejects_non_plain_manifest_data_filename(
    tmp_path: Path, filename: str
) -> None:
    manifest_path = _write_pack(tmp_path)
    manifest = _manifest_for(manifest_path)
    manifest["cases"]["file"] = filename
    _rewrite_manifest(manifest_path, manifest)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_file_larger_than_the_input_cap(tmp_path: Path) -> None:
    manifest_path = _write_pack(tmp_path)
    (tmp_path / "pilot-cases.jsonl").write_bytes(b" " * (MAX_BENCHMARK_INPUT_BYTES + 1))

    _assert_rejected(manifest_path, "benchmark_resource_limit")


def test_rejects_total_input_larger_than_the_pack_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_pack(tmp_path)
    import experience_hub.experiments.benchmarks.loading as loading

    total_bytes = sum(
        item.stat().st_size
        for item in (
            manifest_path,
            tmp_path / "pilot-cases.jsonl",
            tmp_path / "pilot-source.jsonl",
        )
    )
    monkeypatch.setattr(loading, "MAX_BENCHMARK_TOTAL_INPUT_BYTES", total_bytes - 1)

    _assert_rejected(manifest_path, "benchmark_resource_limit")


def test_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_pack(tmp_path)
    manifest = _manifest_for(manifest_path)
    manifest["cases"]["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    _assert_rejected(manifest_path, "benchmark_pack_hash_mismatch")


def test_rejects_duplicate_case_identifier(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    cases[-1]["case_id"] = cases[0]["case_id"]
    manifest_path = _write_pack(tmp_path, cases=cases)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_unresolved_case_source_label(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    cases[0]["source_labels"] = [
        "missing-label",
        *cases[0]["source_labels"][1:],
    ]
    cases[0]["required"] = [{"label": "missing-label", "weight_micros": 450000}]
    cases[0]["checkpoints"] = [
        {
            "predicate": "required_set",
            "labels": ["missing-label"],
            "weight_micros": 150000,
        }
    ]
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_composition_that_does_not_match_the_cases(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    cases[0]["language"] = "en"
    manifest_path = _write_pack(tmp_path, cases=cases)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_wrong_review_declaration(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    cases[20]["review_status"] = "authored"
    manifest_path = _write_pack(tmp_path, cases=cases)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_source_content_out_of_timestamp_order(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    source[-1]["created_at"] = "2026-08-18T00:00:00Z"
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_private_absolute_path_in_public_source_text(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    source[30]["body"] = "Read /Users/private/project/secrets before retrying."
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_source_owner_that_is_not_defined_before_use(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    source[0], source[30] = source[30], source[0]
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


def test_rejects_candidate_label_referenced_by_an_ordinary_case(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    candidate = deepcopy(source[30])
    candidate.pop("temperature")
    candidate.pop("evidence")
    candidate.pop("importance_micros")
    candidate.pop("confidence_micros")
    candidate.update({"record_type": "candidate", "label": "case-0-required"})
    source[30] = candidate
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")
