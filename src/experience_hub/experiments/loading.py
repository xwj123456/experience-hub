"""Bounded canonical loaders for replay fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.errors import CanonicalizationError
from experience_hub.experiments.contracts import (
    MAX_REPLAY_CASES,
    MAX_REPLAY_INPUT_BYTES,
    ReplayCaseV1,
    ReplayManifestV1,
)
from experience_hub.experiments.errors import ExperimentInputError


@dataclass(frozen=True, slots=True)
class LoadedReplayManifest:
    """Validated manifest bytes plus the private directory for its dataset."""

    manifest: ReplayManifestV1
    body: bytes
    _parent: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class LoadedReplayDataset:
    """Validated replay cases and their hash-closed JSONL bytes."""

    cases: tuple[ReplayCaseV1, ...]
    body: bytes


def _reject(code: str, message: str) -> ExperimentInputError:
    return ExperimentInputError(code, message)


def _read_bounded(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise _reject("invalid_path", "input must be a pathlib.Path")
    try:
        if path.is_symlink() or not path.is_file():
            raise _reject("invalid_path", "input must be a regular non-symlink file")
        with path.open("rb") as handle:
            body = handle.read(MAX_REPLAY_INPUT_BYTES + 1)
    except ExperimentInputError:
        raise
    except OSError:
        raise _reject("input_unreadable", "input cannot be read") from None
    if len(body) > MAX_REPLAY_INPUT_BYTES:
        raise _reject("input_too_large", "input exceeds the replay byte limit")
    return body


def _canonical_object(body: bytes, *, subject: str) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _reject("invalid_json", f"{subject} must be UTF-8 JSON") from None
    if not isinstance(decoded, dict):
        raise _reject("invalid_json", f"{subject} must be a JSON object")
    try:
        canonical = canonical_json_bytes(decoded)
    except (CanonicalizationError, RecursionError):
        raise _reject(
            "invalid_json", f"{subject} must contain finite JSON values"
        ) from None
    if canonical != body:
        raise _reject("noncanonical_json", f"{subject} must use canonical JSON")
    return decoded


def load_replay_manifest(path: Path) -> LoadedReplayManifest:
    """Load one canonical, strict replay manifest without exposing its path."""
    body = _read_bounded(path)
    _canonical_object(body, subject="manifest")
    try:
        manifest = ReplayManifestV1.model_validate_json(body, strict=True)
    except ValidationError:
        raise _reject(
            "invalid_manifest", "manifest does not match replay schema"
        ) from None
    return LoadedReplayManifest(manifest=manifest, body=body, _parent=path.parent)


def _load_case(line: bytes) -> ReplayCaseV1:
    _canonical_object(line, subject="case")
    try:
        return ReplayCaseV1.model_validate_json(line, strict=True)
    except ValidationError:
        raise _reject("invalid_case", "case does not match replay schema") from None


def load_replay_cases(loaded: LoadedReplayManifest) -> LoadedReplayDataset:
    """Load hash-closed canonical replay JSONL relative to one manifest."""
    if not isinstance(loaded, LoadedReplayManifest):
        raise _reject("invalid_manifest", "loaded manifest is required")
    path = loaded._parent / loaded.manifest.dataset.cases_file
    body = _read_bounded(path)
    if sha256_hex(body) != loaded.manifest.dataset.cases_sha256:
        raise _reject("cases_hash_mismatch", "cases digest does not match manifest")
    if not body or not body.endswith(b"\n"):
        raise _reject("invalid_jsonl", "cases must end with exactly one newline")
    lines = body[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise _reject(
            "invalid_jsonl", "cases must contain one object per nonempty line"
        )
    if len(lines) > MAX_REPLAY_CASES:
        raise _reject("too_many_cases", "cases exceed the replay case limit")

    cases = tuple(_load_case(line) for line in lines)
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise _reject("duplicate_case", "cases must contain unique case IDs")
    return LoadedReplayDataset(cases=cases, body=body)
