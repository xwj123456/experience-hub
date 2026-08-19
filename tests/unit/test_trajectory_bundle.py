from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture import (
    AdapterDescriptorV1,
    EvidencePointerV1,
    GenericJsonlAdapter,
    OutcomeStatus,
    TrajectoryBundleV1,
    TrajectoryField,
    hash_trajectory_manifest,
    trajectory_manifest_document,
)

OWNER = UUID("10000000-0000-4000-8000-000000000001")
STARTED_AT = datetime(2026, 7, 22, 1, 0, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 7, 22, 2, 0, tzinfo=UTC)

JsonObject = dict[str, object]
Mutator = Callable[[bytes], bytes]


def _header() -> JsonObject:
    return {
        "record_type": "header",
        "schema_version": 1,
        "adapter": {"kind": "generic_jsonl", "version": 1},
        "owner_agent_id": OWNER,
        "trajectory_id": "trajectory-001",
        "source_started_at": STARTED_AT,
        "source_completed_at": COMPLETED_AT,
        "sanitization": {
            "profile_id": "local-sanitized-v1",
            "input_sanitized": True,
        },
    }


def _candidate_signal() -> JsonObject:
    return {
        "kind": "procedural",
        "body": "Retry only after checking whether the operation committed.",
        "summary": "Check commit state before retrying.",
        "mechanism": "A state check prevents duplicate external effects.",
        "tags": ["retry", "audit", "retry"],
        "applicability": ["ambiguous failure", "timeouts"],
        "evidence": [
            {"step_id": "step-2", "field": "outcome"},
            {"step_id": "step-1", "field": "observation"},
        ],
        "falsifiers": ["provider guarantees idempotency", "no state is retained"],
    }


def _step(ordinal: int, *, include_signal: bool = False) -> JsonObject:
    record: JsonObject = {
        "record_type": "step",
        "step_id": f"step-{ordinal}",
        "ordinal": ordinal,
        "occurred_at": datetime(2026, 7, 22, 1, ordinal * 10, tzinfo=UTC),
        "observation": f"Observed state {ordinal}.",
        "action": f"Performed action {ordinal}.",
        "outcome": f"Recorded outcome {ordinal}.",
        "status": "succeeded" if ordinal == 2 else "unknown",
        "candidate_signal": _candidate_signal() if include_signal else None,
    }
    return record


def _encode_records(records: list[JsonObject]) -> bytes:
    return b"\n".join(canonical_json_bytes(record) for record in records)


def _decode_records(data: bytes) -> list[JsonObject]:
    return [cast(JsonObject, json.loads(line)) for line in data.splitlines()]


def valid_jsonl() -> bytes:
    return _encode_records([_header(), _step(1), _step(2, include_signal=True)])


def _rewrite_record(
    data: bytes,
    index: int,
    rewrite: Callable[[JsonObject], None],
) -> bytes:
    records = _decode_records(data)
    rewrite(records[index])
    return _encode_records(records)


def duplicate_header(data: bytes) -> bytes:
    records = _decode_records(data)
    records.insert(1, records[0].copy())
    return _encode_records(records)


def reverse_step_order(data: bytes) -> bytes:
    records = _decode_records(data)
    return _encode_records([records[0], records[2], records[1]])


def duplicate_step_id(data: bytes) -> bytes:
    return _rewrite_record(data, 2, lambda record: record.update(step_id="step-1"))


def omit_sanitization_assertion(data: bytes) -> bytes:
    def omit(record: JsonObject) -> None:
        sanitization = cast(JsonObject, record["sanitization"])
        del sanitization["input_sanitized"]

    return _rewrite_record(data, 0, omit)


def add_unknown_field(data: bytes) -> bytes:
    return _rewrite_record(
        data,
        1,
        lambda record: record.update(unknown_field="retained"),
    )


def append_blank_line(data: bytes) -> bytes:
    return data + b"\n\n"


def append_invalid_utf8(data: bytes) -> bytes:
    return data + b"\n\xff"


def exceed_one_mebibyte(_: bytes) -> bytes:
    return b" " * (1_048_576 + 1)


def test_generic_jsonl_normalizes_one_canonical_bundle() -> None:
    bundle = GenericJsonlAdapter().parse(valid_jsonl())

    assert bundle.schema_version == 1
    assert bundle.adapter == AdapterDescriptorV1(
        kind="generic_jsonl",
        version=1,
    )
    assert bundle.owner_agent_id == OWNER
    assert tuple(step.ordinal for step in bundle.steps) == (1, 2)
    assert bundle.steps[0].status is OutcomeStatus.UNKNOWN
    assert bundle.manifest_hash == hash_trajectory_manifest(bundle)
    manifest = canonical_json_bytes(trajectory_manifest_document(bundle))
    assert bundle.manifest_hash == sha256_hex(manifest)


@pytest.mark.parametrize("schema_version", (True, 1.0, "1"))
def test_generic_jsonl_requires_an_exact_integer_schema_version(
    schema_version: object,
) -> None:
    mutated = _rewrite_record(
        valid_jsonl(),
        0,
        lambda record: record.update(schema_version=schema_version),
    )

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize("adapter_version", (True, 1.0, "1"))
def test_generic_jsonl_requires_an_exact_integer_adapter_version(
    adapter_version: object,
) -> None:
    def replace_version(record: JsonObject) -> None:
        adapter = cast(JsonObject, record["adapter"])
        adapter["version"] = adapter_version

    mutated = _rewrite_record(valid_jsonl(), 0, replace_version)

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize("wire_kind", ("integer", "float", "boolean"))
@pytest.mark.parametrize(
    ("record_index", "field", "epoch_seconds"),
    (
        (0, "source_started_at", int(STARTED_AT.timestamp())),
        (0, "source_completed_at", int(COMPLETED_AT.timestamp())),
        (
            1,
            "occurred_at",
            int(datetime(2026, 7, 22, 1, 10, tzinfo=UTC).timestamp()),
        ),
    ),
)
def test_generic_jsonl_requires_rfc3339_timestamp_strings(
    wire_kind: str,
    record_index: int,
    field: str,
    epoch_seconds: int,
) -> None:
    wire_timestamp: object
    if wire_kind == "integer":
        wire_timestamp = epoch_seconds
    elif wire_kind == "float":
        wire_timestamp = float(epoch_seconds)
    else:
        wire_timestamp = True
    mutated = _rewrite_record(
        valid_jsonl(),
        record_index,
        lambda record: record.update({field: wire_timestamp}),
    )

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize(
    "mutator",
    (
        duplicate_header,
        reverse_step_order,
        duplicate_step_id,
        omit_sanitization_assertion,
        add_unknown_field,
        append_blank_line,
        append_invalid_utf8,
        exceed_one_mebibyte,
    ),
)
def test_generic_jsonl_rejects_noncanonical_or_ambiguous_input(
    mutator: Mutator,
) -> None:
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutator(valid_jsonl()))


def test_generic_jsonl_canonicalizes_set_like_signal_arrays_only() -> None:
    bundle = GenericJsonlAdapter().parse(valid_jsonl())
    signal = bundle.steps[1].candidate_signal

    assert signal is not None
    assert signal.tags == ("audit", "retry")
    assert signal.applicability == ("ambiguous failure", "timeouts")
    assert signal.falsifiers == (
        "no state is retained",
        "provider guarantees idempotency",
    )
    assert signal.evidence == (
        EvidencePointerV1(step_id="step-2", field=TrajectoryField.OUTCOME),
        EvidencePointerV1(step_id="step-1", field=TrajectoryField.OBSERVATION),
    )


def test_generic_jsonl_normalizes_aware_timestamps_to_utc() -> None:
    records = _decode_records(valid_jsonl())
    records[0]["source_started_at"] = "2026-07-22T09:00:00+08:00"
    records[0]["source_completed_at"] = "2026-07-22T10:00:00+08:00"
    records[1]["occurred_at"] = "2026-07-22T09:10:00+08:00"
    records[2]["occurred_at"] = "2026-07-22T09:20:00+08:00"

    bundle = GenericJsonlAdapter().parse(_encode_records(records))

    assert bundle.source_started_at == STARTED_AT
    assert bundle.source_completed_at == COMPLETED_AT
    assert all(
        step.occurred_at.utcoffset().total_seconds() == 0 for step in bundle.steps
    )


@pytest.mark.parametrize(
    ("record_index", "field"),
    (
        (0, "source_started_at"),
        (0, "source_completed_at"),
        (1, "occurred_at"),
    ),
)
def test_generic_jsonl_rejects_naive_timestamps(
    record_index: int,
    field: str,
) -> None:
    mutated = _rewrite_record(
        valid_jsonl(),
        record_index,
        lambda record: record.update({field: "2026-07-22T01:10:00"}),
    )

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize(
    "records",
    (
        [_step(1), _step(2)],
        [_step(1), _header(), _step(2)],
        [_header(), _step(1), _header(), _step(2)],
        [_header()],
    ),
)
def test_generic_jsonl_requires_exactly_one_first_line_header(
    records: list[JsonObject],
) -> None:
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


@pytest.mark.parametrize("ordinals", ((0, 1), (1, 3), (2, 1), (1, 1)))
def test_generic_jsonl_requires_contiguous_declared_step_order(
    ordinals: tuple[int, int],
) -> None:
    records = [_header(), _step(1), _step(2)]
    records[1]["ordinal"], records[2]["ordinal"] = ordinals

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


@pytest.mark.parametrize("ordinal", ("1", True))
def test_generic_jsonl_rejects_coerced_step_ordinals(ordinal: object) -> None:
    mutated = _rewrite_record(
        valid_jsonl(),
        1,
        lambda record: record.update(ordinal=ordinal),
    )

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize("step_id", ("", " ", "\t"))
def test_generic_jsonl_rejects_blank_step_ids(step_id: str) -> None:
    mutated = _rewrite_record(
        valid_jsonl(),
        1,
        lambda record: record.update(step_id=step_id),
    )

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(mutated)


@pytest.mark.parametrize(
    "rewrite",
    (
        lambda records: records[0].update(
            source_started_at="2026-07-22T03:00:00Z"
        ),
        lambda records: records[1].update(occurred_at="2026-07-22T00:59:59Z"),
        lambda records: records[2].update(occurred_at="2026-07-22T01:05:00Z"),
    ),
)
def test_generic_jsonl_rejects_invalid_source_or_step_time_order(
    rewrite: Callable[[list[JsonObject]], None],
) -> None:
    records = _decode_records(valid_jsonl())
    rewrite(records)

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


def test_generic_jsonl_accepts_two_thousand_steps_and_rejects_one_more() -> None:
    records = [_header()]
    for ordinal in range(1, 2_002):
        step = _step(2)
        step["step_id"] = f"step-{ordinal}"
        step["ordinal"] = ordinal
        records.append(step)

    accepted = GenericJsonlAdapter().parse(_encode_records(records[:2_001]))

    assert len(accepted.steps) == 2_000
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


@pytest.mark.parametrize("field", ("observation", "action", "outcome"))
def test_generic_jsonl_bounds_step_text_by_utf8_bytes(field: str) -> None:
    records = _decode_records(valid_jsonl())
    records[1][field] = "界" * 1_365 + "a"
    accepted = GenericJsonlAdapter().parse(_encode_records(records))

    assert len(getattr(accepted.steps[0], field).encode("utf-8")) == 4_096

    records[1][field] = cast(str, records[1][field]) + "b"
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


@pytest.mark.parametrize("field", ("tags", "applicability", "falsifiers"))
def test_generic_jsonl_bounds_signal_string_arrays_before_deduplication(
    field: str,
) -> None:
    records = _decode_records(valid_jsonl())
    signal = cast(JsonObject, records[2]["candidate_signal"])
    signal[field] = [f"item-{index:02d}" for index in range(32)]
    accepted = GenericJsonlAdapter().parse(_encode_records(records))

    accepted_signal = accepted.steps[1].candidate_signal
    assert accepted_signal is not None
    assert len(getattr(accepted_signal, field)) == 32

    signal[field] = ["duplicate"] * 33
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


def test_generic_jsonl_bounds_signal_evidence_and_preserves_order() -> None:
    records = _decode_records(valid_jsonl())
    signal = cast(JsonObject, records[2]["candidate_signal"])
    evidence = [
        {
            "step_id": "step-1" if index % 2 == 0 else "step-2",
            "field": "observation" if index % 2 == 0 else "outcome",
        }
        for index in range(32)
    ]
    signal["evidence"] = evidence
    accepted = GenericJsonlAdapter().parse(_encode_records(records))

    accepted_signal = accepted.steps[1].candidate_signal
    assert accepted_signal is not None
    assert tuple(pointer.step_id for pointer in accepted_signal.evidence) == tuple(
        item["step_id"] for item in evidence
    )

    signal["evidence"] = evidence + [evidence[0]]
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))


def test_generic_jsonl_rejects_noncanonical_json_and_carriage_returns() -> None:
    canonical = valid_jsonl()
    lines = canonical.splitlines()
    noncanonical = b"{ " + lines[0][1:] + b"\n" + b"\n".join(lines[1:])

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(noncanonical)
    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(canonical.replace(b"\n", b"\r\n"))


def test_manifest_contains_hashes_instead_of_raw_step_text() -> None:
    bundle = GenericJsonlAdapter().parse(valid_jsonl())

    manifest = trajectory_manifest_document(bundle)
    first_step = cast(tuple[JsonObject, ...], manifest["steps"])[0]

    assert "observation" not in first_step
    assert "action" not in first_step
    assert "outcome" not in first_step
    assert first_step["observation_hash"] == sha256_hex(
        bundle.steps[0].observation.encode("utf-8")
    )


def test_bundle_validation_rejects_a_copy_with_a_stale_manifest_hash() -> None:
    bundle = GenericJsonlAdapter().parse(valid_jsonl())
    copied = bundle.model_dump(mode="python")
    copied["trajectory_id"] = "tampered-trajectory"

    with pytest.raises(ValidationError, match="manifest hash"):
        TrajectoryBundleV1.model_validate(copied)


def test_bundle_validation_rejects_evidence_for_an_unknown_step() -> None:
    records = _decode_records(valid_jsonl())
    signal = cast(JsonObject, records[2]["candidate_signal"])
    signal["evidence"] = [{"step_id": "missing-step", "field": "outcome"}]

    with pytest.raises(ValueError):
        GenericJsonlAdapter().parse(_encode_records(records))
