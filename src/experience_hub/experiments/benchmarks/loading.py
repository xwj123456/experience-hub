"""Read-only validation for hash-closed ExperienceBench-S pilot packs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.errors import CanonicalizationError
from experience_hub.experiments.benchmarks.contracts import (
    MAX_BENCHMARK_INPUT_BYTES,
    MAX_BENCHMARK_SOURCE_RECORDS,
    MAX_BENCHMARK_TOTAL_INPUT_BYTES,
    BenchmarkCaseV1,
    BenchmarkPackManifestV1,
    BenchmarkSourceAgentV1,
    BenchmarkSourceExperienceV1,
    BenchmarkSourceRecordV1,
)
from experience_hub.experiments.errors import ExperimentInputError

_PRIVATE_PATH = re.compile(
    r"(?:^|[\s\"'])/(?:Users|home|private|tmp|var|opt|Volumes)(?:/|$)|"
    r"(?:^|[\s\"'])[A-Za-z]:\\\\"
)
_SOURCE_RECORD: TypeAdapter[BenchmarkSourceRecordV1] = TypeAdapter(
    BenchmarkSourceRecordV1
)


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkPack:
    """Canonical pilot inputs that have passed all cross-file validation."""

    manifest: BenchmarkPackManifestV1
    cases: tuple[BenchmarkCaseV1, ...]
    source: tuple[BenchmarkSourceRecordV1, ...]
    manifest_body: bytes
    cases_body: bytes
    source_body: bytes
    source_labels: frozenset[str]
    total_input_bytes: int
    _parent: Path = field(repr=False, compare=False)


def _reject(code: str, message: str) -> ExperimentInputError:
    return ExperimentInputError(code, message)


def _read_bounded(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise _reject("benchmark_invalid_pack", "benchmark input path is invalid")
    try:
        if path.is_symlink() or not path.is_file():
            raise _reject("benchmark_invalid_pack", "benchmark input is not a file")
        with path.open("rb") as handle:
            body = handle.read(MAX_BENCHMARK_INPUT_BYTES + 1)
    except ExperimentInputError:
        raise
    except OSError:
        raise _reject(
            "benchmark_invalid_pack", "benchmark input is unreadable"
        ) from None
    if len(body) > MAX_BENCHMARK_INPUT_BYTES:
        raise _reject("benchmark_resource_limit", "benchmark input exceeds byte limit")
    return body


def _canonical_object(body: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _reject(
            "benchmark_invalid_pack", "benchmark input is not canonical JSON"
        ) from None
    if not isinstance(decoded, dict):
        raise _reject("benchmark_invalid_pack", "benchmark JSON must be an object")
    try:
        canonical = canonical_json_bytes(decoded)
    except (CanonicalizationError, RecursionError):
        raise _reject(
            "benchmark_invalid_pack", "benchmark JSON is not canonical"
        ) from None
    if canonical != body:
        raise _reject("benchmark_invalid_pack", "benchmark JSON is not canonical")
    return decoded


def _parse_manifest(body: bytes) -> BenchmarkPackManifestV1:
    _canonical_object(body)
    try:
        return BenchmarkPackManifestV1.model_validate_json(body, strict=True)
    except ValidationError:
        raise _reject(
            "benchmark_invalid_pack", "benchmark manifest is invalid"
        ) from None


def _parse_cases(body: bytes) -> tuple[BenchmarkCaseV1, ...]:
    lines = _jsonl_lines(body)
    if len(lines) != 30:
        raise _reject("benchmark_invalid_pack", "benchmark requires thirty cases")
    cases: list[BenchmarkCaseV1] = []
    for line in lines:
        _canonical_object(line)
        try:
            cases.append(BenchmarkCaseV1.model_validate_json(line, strict=True))
        except ValidationError:
            raise _reject(
                "benchmark_invalid_pack", "benchmark case is invalid"
            ) from None
    return tuple(cases)


def _parse_source(body: bytes) -> tuple[BenchmarkSourceRecordV1, ...]:
    lines = _jsonl_lines(body)
    if len(lines) > MAX_BENCHMARK_SOURCE_RECORDS:
        raise _reject(
            "benchmark_resource_limit", "benchmark source exceeds record limit"
        )
    source: list[BenchmarkSourceRecordV1] = []
    for line in lines:
        _canonical_object(line)
        try:
            source.append(_SOURCE_RECORD.validate_json(line, strict=True))
        except ValidationError:
            raise _reject(
                "benchmark_invalid_pack", "benchmark source is invalid"
            ) from None
    return tuple(source)


def _jsonl_lines(body: bytes) -> tuple[bytes, ...]:
    if not body or not body.endswith(b"\n"):
        raise _reject("benchmark_invalid_pack", "benchmark JSONL is invalid")
    lines = tuple(body[:-1].split(b"\n"))
    if not lines or any(not line for line in lines):
        raise _reject("benchmark_invalid_pack", "benchmark JSONL is invalid")
    return lines


def _validate_source_order(
    source: tuple[BenchmarkSourceRecordV1, ...],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    labels: set[str] = set()
    agents: set[str] = set()
    experiences: set[str] = set()
    previous_created_at = None
    for record in source:
        if record.label in labels:
            raise _reject(
                "benchmark_invalid_pack", "benchmark source labels are ambiguous"
            )
        labels.add(record.label)
        if isinstance(record, BenchmarkSourceAgentV1):
            agents.add(record.label)
            continue
        if record.owner_label not in agents:
            raise _reject(
                "benchmark_invalid_pack", "benchmark source owner is unresolved"
            )
        if previous_created_at is not None and record.created_at < previous_created_at:
            raise _reject("benchmark_invalid_pack", "benchmark source order is invalid")
        previous_created_at = record.created_at
        if isinstance(record, BenchmarkSourceExperienceV1):
            experiences.add(record.label)
    return frozenset(labels), frozenset(agents), frozenset(experiences)


def _validate_composition(
    cases: tuple[BenchmarkCaseV1, ...],
    manifest: BenchmarkPackManifestV1,
) -> None:
    composition = manifest.composition
    if len(cases) != composition.case_count:
        raise _reject(
            "benchmark_invalid_pack", "benchmark composition does not match cases"
        )
    source_classes = Counter(case.source_class.value for case in cases)
    languages = Counter(case.language.value for case in cases)
    strata = Counter(case.stratum.value for case in cases)
    if source_classes != {
        "public_authored": composition.public_authored,
        "reviewed_abstraction": composition.reviewed_abstractions,
    }:
        raise _reject(
            "benchmark_invalid_pack", "benchmark composition does not match cases"
        )
    if languages != {
        "zh": composition.chinese,
        "en": composition.english,
        "mixed": composition.mixed,
    }:
        raise _reject(
            "benchmark_invalid_pack", "benchmark composition does not match cases"
        )
    if any(count != composition.cases_per_stratum for count in strata.values()) or len(
        strata
    ) != 5:
        raise _reject(
            "benchmark_invalid_pack", "benchmark composition does not match cases"
        )


def _validate_case_closure(
    cases: tuple[BenchmarkCaseV1, ...],
    *,
    owners: frozenset[str],
    ordinary_labels: frozenset[str],
) -> None:
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise _reject("benchmark_invalid_pack", "benchmark case IDs are ambiguous")
    for case in cases:
        if case.owner_label not in owners:
            raise _reject(
                "benchmark_invalid_pack", "benchmark case owner is unresolved"
            )
        semantic_labels = {
            item.label
            for group in (
                case.required,
                case.optional,
                case.forbidden,
                case.stale,
                case.misleading,
            )
            for item in group
        }
        checkpoint_labels = {
            label for checkpoint in case.checkpoints for label in checkpoint.labels
        }
        referenced_labels = (
            set(case.source_labels) | semantic_labels | checkpoint_labels
        )
        if not referenced_labels.issubset(ordinary_labels):
            raise _reject(
                "benchmark_invalid_pack", "benchmark case labels are unresolved"
            )


def _contains_private_path(value: object) -> bool:
    if isinstance(value, str):
        return _PRIVATE_PATH.search(value) is not None
    if isinstance(value, dict):
        return any(_contains_private_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_private_path(item) for item in value)
    return False


def _validate_public_text(
    cases: tuple[BenchmarkCaseV1, ...], source: tuple[BenchmarkSourceRecordV1, ...]
) -> None:
    if any(
        _contains_private_path(item.model_dump(mode="json"))
        for item in (*cases, *source)
    ):
        raise _reject("benchmark_invalid_pack", "benchmark public text is invalid")


def load_benchmark_pack(path: Path) -> LoadedBenchmarkPack:
    """Load one hash-closed canonical pilot without creating output."""
    manifest_body = _read_bounded(path)
    manifest = _parse_manifest(manifest_body)
    if len({manifest.cases.file, manifest.source.file}) != 2:
        raise _reject("benchmark_invalid_pack", "benchmark data files are invalid")

    cases_body = _read_bounded(path.parent / manifest.cases.file)
    source_body = _read_bounded(path.parent / manifest.source.file)
    total_input_bytes = len(manifest_body) + len(cases_body) + len(source_body)
    if total_input_bytes > MAX_BENCHMARK_TOTAL_INPUT_BYTES:
        raise _reject("benchmark_resource_limit", "benchmark pack exceeds byte limit")
    if sha256_hex(cases_body) != manifest.cases.sha256 or sha256_hex(
        source_body
    ) != manifest.source.sha256:
        raise _reject("benchmark_pack_hash_mismatch", "benchmark digest does not match")

    cases = _parse_cases(cases_body)
    source = _parse_source(source_body)
    source_labels, owners, ordinary_labels = _validate_source_order(source)
    _validate_composition(cases, manifest)
    _validate_case_closure(cases, owners=owners, ordinary_labels=ordinary_labels)
    _validate_public_text(cases, source)
    return LoadedBenchmarkPack(
        manifest=manifest,
        cases=cases,
        source=source,
        manifest_body=manifest_body,
        cases_body=cases_body,
        source_body=source_body,
        source_labels=source_labels,
        total_input_bytes=total_input_bytes,
        _parent=path.parent,
    )
