from __future__ import annotations

import json
import os
from collections import Counter
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

import experience_hub.experiments.benchmarks.loading as benchmark_loading
from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.experiments.benchmarks.contracts import (
    BENCHMARK_ARM_ORDER,
    MAX_BENCHMARK_INPUT_BYTES,
    MAX_BENCHMARK_TOTAL_INPUT_BYTES,
)
from experience_hub.experiments.benchmarks.loading import load_benchmark_pack
from experience_hub.experiments.errors import ExperimentInputError

REPOSITORY_ROOT = Path(__file__).parents[2]
PILOT_MANIFEST = (
    REPOSITORY_ROOT / "examples" / "experience-bench-s" / "pilot-manifest.json"
)
EXPECTED_CASE_IDS = (
    "recurring-zh-cache-check",
    "recurring-zh-migration-lock",
    "recurring-en-test-scope",
    "recurring-en-release-evidence",
    "recurring-mix-queue-retry",
    "recurring-mix-owner-query",
    "environment-zh-python-version",
    "environment-zh-sqlite-wal",
    "environment-en-macos-path",
    "environment-en-fts5-capability",
    "environment-mix-uv-lock",
    "environment-mix-timezone",
    "state-change-zh-endpoint-version",
    "state-change-zh-schema-revision",
    "state-change-en-branch-head",
    "state-change-en-provider-capability",
    "state-change-mix-token-format",
    "state-change-mix-feature-flag",
    "recovery-zh-stale-cache",
    "recovery-zh-failed-migration",
    "recovery-en-lock-timeout",
    "recovery-en-partial-artifact",
    "recovery-mix-csrf-refresh",
    "recovery-mix-worker-replay",
    "distractor-zh-similar-project",
    "distractor-zh-foreign-owner",
    "distractor-en-old-command",
    "distractor-en-pending-candidate",
    "distractor-mix-keyword-collision",
    "distractor-mix-archived-note",
)
EXPECTED_DIFFICULTIES = (
    "I",
    "A",
    "B",
    "A",
    "I",
    "I",
    "B",
    "A",
    "I",
    "I",
    "B",
    "B",
    "B",
    "A",
    "I",
    "I",
    "B",
    "I",
    "B",
    "A",
    "I",
    "I",
    "I",
    "A",
    "B",
    "B",
    "B",
    "I",
    "A",
    "I",
)
REVIEWED_ABSTRACTIONS = frozenset(
    {
        "recurring-zh-cache-check",
        "recurring-en-release-evidence",
        "environment-zh-sqlite-wal",
        "environment-mix-uv-lock",
        "state-change-en-branch-head",
        "state-change-mix-feature-flag",
        "recovery-zh-stale-cache",
        "recovery-en-partial-artifact",
        "distractor-en-old-command",
        "distractor-mix-keyword-collision",
    }
)


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


def _assert_rejected(
    path: Path,
    code: str,
    *,
    rejected_value: str | None = None,
) -> None:
    with pytest.raises(ExperimentInputError) as raised:
        load_benchmark_pack(path)
    assert raised.value.code == code
    assert str(path) not in str(raised.value)
    if rejected_value is not None:
        assert rejected_value not in str(raised.value)


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


def test_rejects_self_consistent_manifest_with_wrong_pilot_split(
    tmp_path: Path,
) -> None:
    cases = [_case(index) for index in range(30)]
    cases[20]["source_class"] = "public_authored"
    cases[20]["review_status"] = "authored"
    cases[0]["language"] = "en"
    manifest_path = _write_pack(tmp_path, cases=cases)
    manifest = _manifest_for(manifest_path)
    manifest["composition"] = {
        "case_count": 30,
        "public_authored": 21,
        "reviewed_abstractions": 9,
        "cases_per_stratum": 6,
        "chinese": 9,
        "english": 11,
        "mixed": 10,
    }
    _rewrite_manifest(manifest_path, manifest)

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


def test_rejects_source_content_with_equal_timestamps(tmp_path: Path) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    source[-1]["created_at"] = source[-2]["created_at"]
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")


@pytest.mark.parametrize(
    "sentinel",
    (
        "/Users/private/project/secrets",
        r"C:\Users\private\project",
        "private.operator@example.test",
        "api_key=not-a-public-value",
        "-----BEGIN PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz",
    ),
)
def test_rejects_private_or_sensitive_public_source_text(
    tmp_path: Path, sentinel: str
) -> None:
    cases = [_case(index) for index in range(30)]
    source = _source_for(cases)
    source[30]["body"] = f"Do not retain {sentinel}."
    manifest_path = _write_pack(tmp_path, cases=cases, source=source)

    _assert_rejected(
        manifest_path,
        "benchmark_invalid_pack",
        rejected_value=sentinel,
    )


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


def test_rejects_data_file_swapped_for_symlink_at_open_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_pack(tmp_path)
    source_path = tmp_path / "pilot-source.jsonl"
    target_path = tmp_path / "outside-source.jsonl"
    target_body = b"this target must not be read\n"
    target_path.write_bytes(target_body)
    original_open = benchmark_loading.os.open
    original_hash = benchmark_loading.sha256_hex
    hashed_bodies: list[bytes] = []

    def swap_before_open(filename: str, flags: int, mode: int = 0o777) -> int:
        if filename == os.fspath(source_path):
            source_path.unlink()
            source_path.symlink_to(target_path)
        return original_open(filename, flags, mode)

    def record_hash(value: bytes) -> str:
        hashed_bodies.append(value)
        return original_hash(value)

    monkeypatch.setattr(benchmark_loading.os, "open", swap_before_open)
    monkeypatch.setattr(benchmark_loading, "sha256_hex", record_hash)

    _assert_rejected(manifest_path, "benchmark_invalid_pack")

    assert target_body not in hashed_bodies


def test_committed_pilot_pack_has_exact_reviewed_composition_and_privacy() -> None:
    loaded = load_benchmark_pack(PILOT_MANIFEST)
    cases = loaded.cases
    manifest = loaded.manifest

    assert tuple(case.case_id for case in cases) == EXPECTED_CASE_IDS
    assert tuple(case.difficulty for case in cases) == EXPECTED_DIFFICULTIES
    assert tuple(case.stratum.value for case in cases) == (
        ("recurring_workflow",) * 6
        + ("environment_gotcha",) * 6
        + ("state_change",) * 6
        + ("failure_recovery",) * 6
        + ("irrelevant_distractor",) * 6
    )
    assert tuple(case.language.value for case in cases) == (
        "zh",
        "zh",
        "en",
        "en",
        "mixed",
        "mixed",
    ) * 5
    assert {
        case.case_id
        for case in cases
        if case.source_class.value == "reviewed_abstraction"
    } == REVIEWED_ABSTRACTIONS
    assert all(
        case.review_status
        == (
            "maintainer_reviewed"
            if case.case_id in REVIEWED_ABSTRACTIONS
            else "authored"
        )
        for case in cases
    )
    assert len({case.owner_label for case in cases}) == 30
    assert Counter(case.language.value for case in cases) == {
        "zh": 10,
        "en": 10,
        "mixed": 10,
    }
    assert Counter(case.stratum.value for case in cases) == {
        "recurring_workflow": 6,
        "environment_gotcha": 6,
        "state_change": 6,
        "failure_recovery": 6,
        "irrelevant_distractor": 6,
    }

    arm_registry = tuple(
        (arm.arm_id, arm.kind.value, arm.required) for arm in manifest.arms
    )
    assert arm_registry == (
        ("no_memory", "no_memory", True),
        ("recent_notes", "recent_notes", True),
        ("sqlite_bm25", "sqlite_bm25", True),
        ("experience_hub", "experience_hub", True),
    )
    assert manifest.deterministic_replay_runs == 2
    assert (
        manifest.schema_version,
        manifest.oracle_version,
        manifest.metric_version,
        manifest.gate_version,
        manifest.evidence_schema_version,
        manifest.summary_schema_version,
        manifest.profile_schema_version,
    ) == (1, 1, 1, 1, 1, 1, 1)
    assert sha256_hex(loaded.cases_body) == manifest.cases.sha256
    assert sha256_hex(loaded.source_body) == manifest.source.sha256
    manifest_document = json.loads(loaded.manifest_body)
    assert canonical_json_bytes(manifest_document) == loaded.manifest_body
    assert all(
        canonical_json_bytes(json.loads(line)) == line
        for body in (loaded.cases_body, loaded.source_body)
        for line in body.rstrip(b"\n").splitlines()
    )

    source_by_label = {record.label: record for record in loaded.source}
    case_by_id = {case.case_id: case for case in cases}
    seen_case_labels: set[str] = set()
    for case in cases:
        groups = (
            case.required,
            case.optional,
            case.forbidden,
            case.stale,
            case.misleading,
        )
        semantic_labels = {item.label for group in groups for item in group}
        assert set(case.source_labels) == semantic_labels
        assert not (seen_case_labels & semantic_labels)
        assert all(label.startswith(f"{case.case_id}-") for label in semantic_labels)
        assert all(label in source_by_label for label in semantic_labels)
        assert all(
            source_by_label[label].owner_label == case.owner_label
            for label in semantic_labels
            if label != "distractor-zh-foreign-owner-misleading-2"
        )
        seen_case_labels.update(semantic_labels)
        assert sum(item.weight_micros for item in case.required) == 450_000
        assert sum(
            item.weight_micros
            for group in (case.forbidden, case.stale, case.misleading)
            for item in group
        ) == 300_000
        assert sum(item.weight_micros for item in case.checkpoints) == 150_000
        predicates = {item.predicate.value for item in case.checkpoints}
        assert predicates == (
            {"ordered_subsequence"}
            if case.stratum.value == "failure_recovery"
            else {"required_set"}
        )

    foreign_case = case_by_id["distractor-zh-foreign-owner"]
    foreign = source_by_label["distractor-zh-foreign-owner-misleading-2"]
    assert foreign.owner_label != foreign_case.owner_label
    assert foreign.owner_label == "distractor-zh-foreign-owner-foreign-owner"

    pending_case = case_by_id["distractor-en-pending-candidate"]
    pending = source_by_label["distractor-en-pending-candidate-misleading-2"]
    assert pending.record_type == "candidate"
    assert pending.owner_label == pending_case.owner_label
    assert pending.label not in pending_case.source_labels

    archived_case = case_by_id["distractor-mix-archived-note"]
    archived = source_by_label["distractor-mix-archived-note-stale-2"]
    assert archived.owner_label == archived_case.owner_label
    assert archived.temperature.value == "archived"

    public_bodies = (
        loaded.manifest_body,
        loaded.cases_body,
        loaded.source_body,
        (PILOT_MANIFEST.parent / "README.md").read_bytes(),
    )
    for body in public_bodies:
        assert b"/Users/" not in body
        assert b"C:\\Users\\" not in body
        assert b"@" not in body
        assert b"-----BEGIN " not in body
