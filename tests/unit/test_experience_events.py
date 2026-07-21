from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import StructuredReason, TypedEvidence
from experience_hub.domain.events import EventPayload, EventRegistry
from experience_hub.experiences.contracts import (
    VersionLinkInput,
    canonicalize_version_links,
)
from experience_hub.experiences.events import (
    LIFECYCLE_EXPERIENCE_EVENT_TYPES,
    RETRIEVAL_EXPERIENCE_EVENT_TYPES,
    STATE_EXPERIENCE_EVENT_TYPES,
    TASK2_EXPERIENCE_EVENT_TYPES,
    ExperienceAccessedV1,
    ExperienceArchivedV1,
    ExperienceConfirmedV1,
    ExperienceCorroboratedV1,
    ExperienceCreatedV1,
    ExperienceLifecycleEvaluatedV1,
    ExperiencePinnedV1,
    ExperienceReactivatedV1,
    ExperienceRefutedV1,
    ExperienceRestoredV1,
    ExperienceStateSnapshotV1,
    ExperienceTemperatureChangedV1,
    ExperienceUnpinnedV1,
    ExperienceVersionCreatedV1,
    VersionLinkRefV1,
    is_valid_version_event_sequence,
    register_experience_events,
)
from experience_hub.experiences.models import LinkRelation, Temperature

EXPERIENCE_ID = UUID("00000000-0000-0000-0000-000000000101")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000201")
NEXT_VERSION_ID = UUID("00000000-0000-0000-0000-000000000202")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000301")
TARGET_A = UUID("00000000-0000-0000-0000-000000000011")
TARGET_B = UUID("00000000-0000-0000-0000-000000000012")
CYCLE_ID = UUID("00000000-0000-0000-0000-000000000401")
ADOPTION_ID = UUID("00000000-0000-0000-0000-000000000501")
CAPSULE_ID = UUID("00000000-0000-0000-0000-000000000601")
NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
CONTENT_HASH = "a" * 64
NEXT_CONTENT_HASH = "b" * 64
QUERY_HASH = "c" * 64
ROOT_FINGERPRINT = "d" * 64

SNAPSHOT_FIELDS = {
    "experience_id",
    "owner_agent_id",
    "current_version_id",
    "current_content_hash",
    "temperature",
    "importance",
    "confidence",
    "activation_score",
    "source_trust",
    "access_count",
    "access_strength",
    "strength_updated_at",
    "last_accessed_at",
    "last_transition_at",
    "last_lifecycle_evaluated_at",
    "consecutive_below_threshold",
    "pinned",
}


def snapshot(**overrides: object) -> ExperienceStateSnapshotV1:
    values: dict[str, Any] = {
        "experience_id": EXPERIENCE_ID,
        "owner_agent_id": OWNER_ID,
        "current_version_id": VERSION_ID,
        "current_content_hash": CONTENT_HASH,
        "temperature": Temperature.WARM,
        "importance": 0.35,
        "confidence": 0.5,
        "activation_score": 0.48,
        "source_trust": 1.0,
        "access_count": 0,
        "access_strength": 0.0,
        "strength_updated_at": NOW,
        "last_accessed_at": None,
        "last_transition_at": NOW,
        "last_lifecycle_evaluated_at": None,
        "consecutive_below_threshold": 0,
        "pinned": False,
    }
    values.update(overrides)
    return ExperienceStateSnapshotV1(**values)


def test_snapshot_has_exactly_seventeen_semantic_fields() -> None:
    assert set(ExperienceStateSnapshotV1.model_fields) == SNAPSHOT_FIELDS
    assert len(ExperienceStateSnapshotV1.model_fields) == 17
    assert "projection_event_id" not in ExperienceStateSnapshotV1.model_fields


def test_snapshot_normalizes_every_aware_datetime_to_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))
    local_time = datetime(2026, 7, 18, 16, 30, tzinfo=plus_eight)

    value = snapshot(
        strength_updated_at=local_time,
        last_accessed_at=local_time,
        last_transition_at=local_time,
        last_lifecycle_evaluated_at=local_time,
    )

    assert value.strength_updated_at == NOW
    assert value.last_accessed_at == NOW
    assert value.last_transition_at == NOW
    assert value.last_lifecycle_evaluated_at == NOW
    for field_name in (
        "strength_updated_at",
        "last_accessed_at",
        "last_transition_at",
        "last_lifecycle_evaluated_at",
    ):
        assert getattr(value, field_name).tzinfo is UTC


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_content_hash", "A" * 64),
        ("current_content_hash", "a" * 63),
        ("importance", -0.01),
        ("confidence", 1.01),
        ("activation_score", math.nan),
        ("source_trust", math.inf),
        ("access_strength", 20.01),
        ("access_count", -1),
        ("access_count", True),
        ("consecutive_below_threshold", -1),
        ("consecutive_below_threshold", False),
        ("pinned", 1),
        ("strength_updated_at", NOW.replace(tzinfo=None)),
    ],
)
def test_snapshot_rejects_invalid_strict_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        snapshot(**{field: value})


def test_snapshot_is_frozen_and_rejects_extra_fields() -> None:
    value = snapshot()

    with pytest.raises(ValidationError):
        value.confidence = 0.8
    with pytest.raises(ValidationError):
        ExperienceStateSnapshotV1.model_validate(
            {
                **value.model_dump(),
                "projection_event_id": 1,
            }
        )


def test_registry_contains_cumulative_thirteen_state_event_types() -> None:
    registry = EventRegistry()

    register_experience_events(registry)

    assert frozenset(
        {"experience.created", "experience.version_created"}
    ) == TASK2_EXPERIENCE_EVENT_TYPES
    assert frozenset(
        {
            "experience.accessed",
            "experience.reactivated",
            "experience.temperature_changed",
        }
    ) == RETRIEVAL_EXPERIENCE_EVENT_TYPES
    assert frozenset(
        {
            "experience.lifecycle_evaluated",
            "experience.confirmed",
            "experience.refuted",
            "experience.pinned",
            "experience.unpinned",
            "experience.archived",
            "experience.restored",
        }
    ) == LIFECYCLE_EXPERIENCE_EVENT_TYPES
    assert registry.event_types == STATE_EXPERIENCE_EVENT_TYPES == frozenset(
        {
            "experience.created",
            "experience.version_created",
            "experience.accessed",
            "experience.reactivated",
            "experience.temperature_changed",
            "experience.lifecycle_evaluated",
            "experience.confirmed",
            "experience.refuted",
            "experience.pinned",
            "experience.unpinned",
            "experience.archived",
            "experience.restored",
            "experience.corroborated",
        }
    )


@pytest.mark.parametrize(
    ("version_number", "aggregate_sequence", "expected"),
    [
        (1, 2, True),
        (1, 3, False),
        (2, 2, False),
        (2, 3, True),
        (3, 4, True),
        (3, 9, True),
    ],
)
def test_version_event_sequence_is_independent_after_initial_creation(
    version_number: int,
    aggregate_sequence: int,
    expected: bool,
) -> None:
    assert (
        is_valid_version_event_sequence(
            version_number=version_number,
            aggregate_sequence=aggregate_sequence,
        )
        is expected
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"experience_id":"00000000-0000-0000-0000-000000000101"}',
        b'{"schema_version":2}',
        b'{"extra":true,"schema_version":1}',
    ],
)
def test_registry_rejects_incomplete_wrong_version_or_extra_payloads(
    payload: bytes,
) -> None:
    registry = EventRegistry()
    register_experience_events(registry)

    with pytest.raises(ValidationError):
        registry.decode(
            event_type=ExperienceCreatedV1.event_type,
            payload=payload,
        )


def test_created_event_requires_payload_ids_to_match_after_state() -> None:
    after = snapshot()
    ExperienceCreatedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        version_id=VERSION_ID,
        after=after,
    )

    assert set(ExperienceCreatedV1.model_fields) == {
        "schema_version",
        "experience_id",
        "version_id",
        "after",
    }
    with pytest.raises(ValidationError):
        ExperienceCreatedV1(
            schema_version=1,
            experience_id=TARGET_A,
            version_id=VERSION_ID,
            after=after,
        )
    with pytest.raises(ValidationError):
        ExperienceCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=NEXT_VERSION_ID,
            after=after,
        )


def test_initial_version_event_is_a_semantic_noop() -> None:
    state = snapshot()

    event = ExperienceVersionCreatedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        version_id=VERSION_ID,
        version_number=1,
        supersedes_version_id=None,
        links=(),
        before=state,
        after=state,
    )

    assert event.before == event.after
    with pytest.raises(ValidationError):
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            version_number=1,
            supersedes_version_id=None,
            links=(),
            before=state,
            after=state.model_copy(update={"activation_score": 0.47}),
        )


def test_correction_event_allows_only_the_five_locked_state_changes() -> None:
    before = snapshot(access_strength=4.0)
    after = snapshot(
        current_version_id=NEXT_VERSION_ID,
        current_content_hash=NEXT_CONTENT_HASH,
        access_strength=3.5,
        strength_updated_at=NOW + timedelta(hours=1),
        activation_score=0.52,
    )

    event = ExperienceVersionCreatedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        version_id=NEXT_VERSION_ID,
        version_number=2,
        supersedes_version_id=VERSION_ID,
        links=(),
        before=before,
        after=after,
    )

    assert event.after == after
    with pytest.raises(ValidationError):
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=NEXT_VERSION_ID,
            version_number=2,
            supersedes_version_id=VERSION_ID,
            links=(),
            before=before,
            after=after.model_copy(update={"confidence": 0.75}),
        )


@pytest.mark.parametrize(
    ("version_number", "supersedes_version_id"),
    [(0, None), (1, VERSION_ID), (2, None)],
)
def test_version_event_enforces_number_and_supersession_shape(
    version_number: int,
    supersedes_version_id: UUID | None,
) -> None:
    before = snapshot()
    after = snapshot(
        current_version_id=NEXT_VERSION_ID,
        current_content_hash=NEXT_CONTENT_HASH,
    )

    with pytest.raises(ValidationError):
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=NEXT_VERSION_ID,
            version_number=version_number,
            supersedes_version_id=supersedes_version_id,
            links=(),
            before=before,
            after=after,
        )


def test_accessed_event_has_exact_fields_and_one_access_semantics() -> None:
    accessed_at = NOW + timedelta(hours=1)
    before = snapshot(
        temperature=Temperature.COLD,
        access_count=3,
        access_strength=2.0,
    )
    after = snapshot(
        temperature=Temperature.COLD,
        access_count=4,
        access_strength=2.9,
        strength_updated_at=accessed_at,
        last_accessed_at=accessed_at,
        activation_score=0.52,
    )

    event = ExperienceAccessedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        version_id=VERSION_ID,
        before=before,
        after=after,
    )

    assert set(ExperienceAccessedV1.model_fields) == {
        "schema_version",
        "experience_id",
        "version_id",
        "before",
        "after",
    }
    assert event.after.access_count == event.before.access_count + 1
    assert event.after.strength_updated_at == event.after.last_accessed_at


@pytest.mark.parametrize(
    "after",
    [
        snapshot(
            access_count=2,
            access_strength=1.0,
            strength_updated_at=NOW + timedelta(hours=1),
            last_accessed_at=NOW + timedelta(hours=1),
        ),
        snapshot(
            access_count=1,
            access_strength=1.0,
            strength_updated_at=NOW + timedelta(hours=1),
            last_accessed_at=NOW + timedelta(hours=2),
        ),
        snapshot(
            access_count=1,
            access_strength=1.0,
            strength_updated_at=NOW + timedelta(hours=1),
            last_accessed_at=NOW + timedelta(hours=1),
            confidence=0.75,
        ),
        snapshot(
            current_version_id=NEXT_VERSION_ID,
            access_count=1,
            access_strength=1.0,
            strength_updated_at=NOW + timedelta(hours=1),
            last_accessed_at=NOW + timedelta(hours=1),
        ),
    ],
)
def test_accessed_event_rejects_invalid_or_unauthorized_changes(
    after: ExperienceStateSnapshotV1,
) -> None:
    with pytest.raises(ValidationError):
        ExperienceAccessedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            before=snapshot(),
            after=after,
        )


def test_accessed_event_rejects_archived_experience() -> None:
    accessed_at = NOW + timedelta(hours=1)
    before = snapshot(temperature=Temperature.ARCHIVED)
    after = snapshot(
        temperature=Temperature.ARCHIVED,
        access_count=1,
        access_strength=1.0,
        strength_updated_at=accessed_at,
        last_accessed_at=accessed_at,
        activation_score=0.52,
    )

    with pytest.raises(ValidationError, match="Archived"):
        ExperienceAccessedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            before=before,
            after=after,
        )


def test_reactivated_event_is_a_strict_query_free_semantic_noop() -> None:
    state = snapshot(temperature=Temperature.COLD)

    event = ExperienceReactivatedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        query_hash=QUERY_HASH,
        mode="focused",
        signal=0.72,
        before=state,
        after=state,
    )

    assert set(ExperienceReactivatedV1.model_fields) == {
        "schema_version",
        "experience_id",
        "query_hash",
        "mode",
        "signal",
        "before",
        "after",
    }
    assert event.before == event.after
    assert "query" not in event.model_dump()
    with pytest.raises(ValidationError):
        ExperienceReactivatedV1.model_validate(
            {
                **event.model_dump(mode="python"),
                "query": "raw private query",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_hash", "C" * 64),
        ("query_hash", "c" * 63),
        ("mode", "broad"),
        ("signal", -0.01),
        ("signal", 1.01),
        ("signal", math.nan),
        ("signal", True),
    ],
)
def test_reactivated_event_rejects_invalid_strict_values(
    field: str,
    value: object,
) -> None:
    state = snapshot(temperature=Temperature.COLD)
    values: dict[str, Any] = {
        "schema_version": 1,
        "experience_id": EXPERIENCE_ID,
        "query_hash": QUERY_HASH,
        "mode": "associative",
        "signal": 0.65,
        "before": state,
        "after": state,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        ExperienceReactivatedV1.model_validate(values)


def test_reactivated_event_rejects_state_changes_or_wrong_anchor() -> None:
    state = snapshot(temperature=Temperature.COLD)

    with pytest.raises(ValidationError):
        ExperienceReactivatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=state,
            after=state.model_copy(update={"activation_score": 0.5}),
        )
    with pytest.raises(ValidationError):
        ExperienceReactivatedV1(
            schema_version=1,
            experience_id=TARGET_A,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=state,
            after=state,
        )
    warm = snapshot(temperature=Temperature.WARM)
    with pytest.raises(ValidationError):
        ExperienceReactivatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=warm,
            after=warm,
        )


@pytest.mark.parametrize(
    ("cause", "cycle_id", "before_temperature", "after_temperature"),
    [
        ("cold_reactivation", None, Temperature.COLD, Temperature.WARM),
        ("confirmation", None, Temperature.WARM, Temperature.HOT),
        ("confirmation", None, Temperature.COLD, Temperature.HOT),
        ("pin", None, Temperature.WARM, Temperature.HOT),
        ("pin", None, Temperature.COLD, Temperature.HOT),
        (
            "lifecycle_activation",
            CYCLE_ID,
            Temperature.WARM,
            Temperature.HOT,
        ),
        (
            "lifecycle_demotion",
            CYCLE_ID,
            Temperature.HOT,
            Temperature.WARM,
        ),
        (
            "lifecycle_demotion",
            CYCLE_ID,
            Temperature.WARM,
            Temperature.COLD,
        ),
        (
            "policy_archive",
            CYCLE_ID,
            Temperature.COLD,
            Temperature.ARCHIVED,
        ),
        ("restore", None, Temperature.ARCHIVED, Temperature.WARM),
        (
            "capsule_corroboration",
            None,
            Temperature.COLD,
            Temperature.HOT,
        ),
    ],
)
def test_temperature_changed_event_locks_cause_cycle_and_transition_matrix(
    cause: str,
    cycle_id: UUID | None,
    before_temperature: Temperature,
    after_temperature: Temperature,
) -> None:
    transitioned_at = NOW + timedelta(hours=1)
    before = snapshot(
        temperature=before_temperature,
        consecutive_below_threshold=2,
    )
    after = snapshot(
        temperature=after_temperature,
        last_transition_at=transitioned_at,
        consecutive_below_threshold=0,
    )

    event = ExperienceTemperatureChangedV1.model_validate(
        {
            "schema_version": 1,
            "experience_id": EXPERIENCE_ID,
            "cause": cause,
            "cycle_id": cycle_id,
            "before": before,
            "after": after,
        }
    )

    assert set(ExperienceTemperatureChangedV1.model_fields) == {
        "schema_version",
        "experience_id",
        "cause",
        "cycle_id",
        "before",
        "after",
    }
    assert event.after.temperature is after_temperature
    assert event.after.consecutive_below_threshold == 0


@pytest.mark.parametrize(
    ("cause", "cycle_id", "before_temperature", "after_temperature"),
    [
        (
            "cold_reactivation",
            CYCLE_ID,
            Temperature.COLD,
            Temperature.WARM,
        ),
        (
            "lifecycle_activation",
            None,
            Temperature.WARM,
            Temperature.HOT,
        ),
        ("restore", None, Temperature.COLD, Temperature.WARM),
        (
            "lifecycle_demotion",
            CYCLE_ID,
            Temperature.COLD,
            Temperature.WARM,
        ),
        ("unknown", None, Temperature.COLD, Temperature.WARM),
    ],
)
def test_temperature_changed_event_rejects_wrong_matrix_rows(
    cause: str,
    cycle_id: UUID | None,
    before_temperature: Temperature,
    after_temperature: Temperature,
) -> None:
    with pytest.raises(ValidationError):
        ExperienceTemperatureChangedV1.model_validate(
            {
                "schema_version": 1,
                "experience_id": EXPERIENCE_ID,
                "cause": cause,
                "cycle_id": cycle_id,
                "before": snapshot(temperature=before_temperature),
                "after": snapshot(
                    temperature=after_temperature,
                    last_transition_at=NOW + timedelta(hours=1),
                ),
            }
        )


def test_temperature_changed_event_rejects_noop_counter_and_other_changes(
) -> None:
    transitioned_at = NOW + timedelta(hours=1)
    before = snapshot(
        temperature=Temperature.COLD,
        consecutive_below_threshold=2,
    )
    valid = {
        "schema_version": 1,
        "experience_id": EXPERIENCE_ID,
        "cause": "cold_reactivation",
        "cycle_id": None,
        "before": before,
    }

    invalid_after = (
        before.model_copy(update={"last_transition_at": transitioned_at}),
        snapshot(
            temperature=Temperature.WARM,
            last_transition_at=transitioned_at,
            consecutive_below_threshold=1,
        ),
        snapshot(
            temperature=Temperature.WARM,
            last_transition_at=transitioned_at,
            consecutive_below_threshold=0,
            confidence=0.75,
        ),
    )

    for after in invalid_after:
        with pytest.raises(ValidationError):
            ExperienceTemperatureChangedV1.model_validate(
                {**valid, "after": after}
            )


def test_link_inputs_are_canonical_and_reject_duplicates_or_self_links() -> None:
    links = canonicalize_version_links(
        source_experience_id=EXPERIENCE_ID,
        links=(
            VersionLinkInput(
                target_experience_id=TARGET_B,
                relation=LinkRelation.TESTS,
            ),
            VersionLinkInput(
                target_experience_id=TARGET_A,
                relation=LinkRelation.SUPPORTS,
            ),
            VersionLinkInput(
                target_experience_id=TARGET_A,
                relation=LinkRelation.DERIVED_FROM,
            ),
        ),
    )

    assert links == (
        VersionLinkRefV1(
            target_experience_id=TARGET_A,
            relation=LinkRelation.DERIVED_FROM,
        ),
        VersionLinkRefV1(
            target_experience_id=TARGET_A,
            relation=LinkRelation.SUPPORTS,
        ),
        VersionLinkRefV1(
            target_experience_id=TARGET_B,
            relation=LinkRelation.TESTS,
        ),
    )
    duplicate = VersionLinkInput(
        target_experience_id=TARGET_A,
        relation=LinkRelation.SUPPORTS,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        canonicalize_version_links(
            source_experience_id=EXPERIENCE_ID,
            links=(duplicate, duplicate),
        )
    with pytest.raises(ValueError, match="itself"):
        canonicalize_version_links(
            source_experience_id=EXPERIENCE_ID,
            links=(
                VersionLinkInput(
                    target_experience_id=EXPERIENCE_ID,
                    relation=LinkRelation.SUPPORTS,
                ),
            ),
        )


def test_version_event_rejects_noncanonical_or_duplicate_links() -> None:
    state = snapshot()
    first = VersionLinkRefV1(
        target_experience_id=TARGET_A,
        relation=LinkRelation.SUPPORTS,
    )
    second = VersionLinkRefV1(
        target_experience_id=TARGET_B,
        relation=LinkRelation.TESTS,
    )

    with pytest.raises(ValidationError):
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            version_number=1,
            supersedes_version_id=None,
            links=(second, first),
            before=state,
            after=state,
        )
    with pytest.raises(ValidationError):
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            version_number=1,
            supersedes_version_id=None,
            links=(first, first),
            before=state,
            after=state,
        )


def test_event_payloads_never_contain_body_payload_or_query_fields() -> None:
    state = snapshot()
    accessed_at = NOW + timedelta(hours=1)
    accessed = snapshot(
        access_count=1,
        access_strength=1.0,
        strength_updated_at=accessed_at,
        last_accessed_at=accessed_at,
    )
    cold = snapshot(temperature=Temperature.COLD)
    warm = snapshot(
        temperature=Temperature.WARM,
        last_transition_at=accessed_at,
    )
    payloads = (
        ExperienceCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            after=state,
        ),
        ExperienceVersionCreatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            version_number=1,
            supersedes_version_id=None,
            links=(),
            before=state,
            after=state,
        ),
        ExperienceAccessedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            version_id=VERSION_ID,
            before=state,
            after=accessed,
        ),
        ExperienceReactivatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            query_hash=QUERY_HASH,
            mode="focused",
            signal=0.72,
            before=cold,
            after=cold,
        ),
        ExperienceTemperatureChangedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            cause="cold_reactivation",
            cycle_id=None,
            before=cold,
            after=warm,
        ),
        *lifecycle_payloads(),
    )

    for payload in payloads:
        field_names = set(payload.model_dump())
        assert field_names.isdisjoint({"body", "payload", "query", "query_text"})
        assert b'"body"' not in payload.model_dump_json().encode()


def lifecycle_payloads() -> tuple[
    ExperienceLifecycleEvaluatedV1,
    ExperienceConfirmedV1,
    ExperienceRefutedV1,
    ExperiencePinnedV1,
    ExperienceUnpinnedV1,
    ExperienceArchivedV1,
    ExperienceRestoredV1,
]:
    evaluated_at = NOW + timedelta(hours=1)
    materialized = {
        "access_strength": 0.8,
        "strength_updated_at": evaluated_at,
        "activation_score": 0.42,
    }
    base = snapshot(access_strength=1.0)
    pinned = snapshot(pinned=True, access_strength=1.0)
    cold = snapshot(temperature=Temperature.COLD)
    archived = snapshot(temperature=Temperature.ARCHIVED)
    evidence = (
        TypedEvidence(type="log", id="case-2"),
        TypedEvidence(type="metric", id="case-1"),
    )
    reason = StructuredReason.from_user_text("operator supplied")
    return (
        ExperienceLifecycleEvaluatedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            cycle_id=CYCLE_ID,
            evaluated_at=evaluated_at,
            threshold_target="demote_cold",
            before=base,
            after=base.model_copy(
                update={
                    **materialized,
                    "last_lifecycle_evaluated_at": evaluated_at,
                    "consecutive_below_threshold": 1,
                }
            ),
        ),
        ExperienceConfirmedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            reason=reason,
            evidence=evidence,
            before=base,
            after=base.model_copy(
                update={
                    **materialized,
                    "confidence": 0.6,
                }
            ),
        ),
        ExperienceRefutedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            reason=None,
            evidence=(),
            before=base,
            after=base.model_copy(
                update={
                    **materialized,
                    "confidence": 0.325,
                }
            ),
        ),
        ExperiencePinnedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            reason=None,
            before=base,
            after=base.model_copy(
                update={
                    **materialized,
                    "pinned": True,
                }
            ),
        ),
        ExperienceUnpinnedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            reason=reason,
            before=pinned,
            after=pinned.model_copy(
                update={
                    **materialized,
                    "pinned": False,
                }
            ),
        ),
        ExperienceArchivedV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            cycle_id=CYCLE_ID,
            reason=StructuredReason.policy_due(),
            before=cold,
            after=cold,
        ),
        ExperienceRestoredV1(
            schema_version=1,
            experience_id=EXPERIENCE_ID,
            reason=reason,
            before=archived,
            after=archived.model_copy(update=materialized),
        ),
    )


@pytest.mark.parametrize(
    ("payload_type", "expected_fields"),
    [
        (
            ExperienceLifecycleEvaluatedV1,
            {
                "schema_version",
                "experience_id",
                "cycle_id",
                "evaluated_at",
                "threshold_target",
                "before",
                "after",
            },
        ),
        (
            ExperienceConfirmedV1,
            {
                "schema_version",
                "experience_id",
                "reason",
                "evidence",
                "before",
                "after",
            },
        ),
        (
            ExperienceRefutedV1,
            {
                "schema_version",
                "experience_id",
                "reason",
                "evidence",
                "before",
                "after",
            },
        ),
        (
            ExperiencePinnedV1,
            {
                "schema_version",
                "experience_id",
                "reason",
                "before",
                "after",
            },
        ),
        (
            ExperienceUnpinnedV1,
            {
                "schema_version",
                "experience_id",
                "reason",
                "before",
                "after",
            },
        ),
        (
            ExperienceArchivedV1,
            {
                "schema_version",
                "experience_id",
                "cycle_id",
                "reason",
                "before",
                "after",
            },
        ),
        (
            ExperienceRestoredV1,
            {
                "schema_version",
                "experience_id",
                "reason",
                "before",
                "after",
            },
        ),
    ],
)
def test_lifecycle_events_have_exact_v1_fields(
    payload_type: type[EventPayload],
    expected_fields: set[str],
) -> None:
    assert set(payload_type.model_fields) == expected_fields


def test_lifecycle_events_accept_only_their_permitted_state_deltas() -> None:
    for payload in lifecycle_payloads():
        payload_type = type(payload)
        values = payload.model_dump(mode="python")
        with pytest.raises(ValidationError):
            payload_type.model_validate(
                {
                    **values,
                    "after": payload.after.model_copy(
                        update={"importance": payload.after.importance + 0.01}
                    ),
                }
            )
        other_temperature = (
            Temperature.COLD
            if payload.after.temperature is not Temperature.COLD
            else Temperature.WARM
        )
        with pytest.raises(ValidationError):
            payload_type.model_validate(
                {
                    **values,
                    "after": payload.after.model_copy(
                        update={"temperature": other_temperature}
                    ),
                }
            )


def test_confirmation_and_refutation_lock_design_confidence_formulas() -> None:
    confirmed = lifecycle_payloads()[1]
    refuted = lifecycle_payloads()[2]

    assert confirmed.after.confidence == pytest.approx(
        confirmed.before.confidence
        + (1.0 - confirmed.before.confidence) * 0.20,
        abs=1e-12,
    )
    assert refuted.after.confidence == pytest.approx(
        refuted.before.confidence * 0.65,
        abs=1e-12,
    )
    with pytest.raises(ValidationError, match="confidence"):
        ExperienceConfirmedV1.model_validate(
            {
                **confirmed.model_dump(mode="python"),
                "after": confirmed.after.model_copy(
                    update={"confidence": confirmed.after.confidence + 0.01}
                ),
            }
        )
    with pytest.raises(ValidationError, match="confidence"):
        ExperienceRefutedV1.model_validate(
            {
                **refuted.model_dump(mode="python"),
                "after": refuted.after.model_copy(
                    update={"confidence": refuted.after.confidence - 0.01}
                ),
            }
        )


def corroborated_payload(
    *,
    before_confidence: float = 0.5,
    captured_trust: float = 0.75,
) -> ExperienceCorroboratedV1:
    materialized_at = NOW + timedelta(hours=1)
    before = snapshot(
        confidence=before_confidence,
        access_strength=1.0,
    )
    after = before.model_copy(
        update={
            "confidence": before_confidence
            + (1.0 - before_confidence) * 0.20 * captured_trust,
            "access_strength": 0.8,
            "strength_updated_at": materialized_at,
            "activation_score": 0.42,
        }
    )
    return ExperienceCorroboratedV1(
        schema_version=1,
        experience_id=EXPERIENCE_ID,
        adoption_id=ADOPTION_ID,
        capsule_id=CAPSULE_ID,
        root_fingerprint=ROOT_FINGERPRINT,
        captured_trust=captured_trust,
        before=before,
        after=after,
    )


def test_corroborated_event_has_exact_registered_v1_protocol() -> None:
    payload = corroborated_payload()
    registry = EventRegistry()
    register_experience_events(registry)

    decoded = registry.decode(
        event_type=ExperienceCorroboratedV1.event_type,
        payload=payload.model_dump_json().encode(),
    )

    assert set(ExperienceCorroboratedV1.model_fields) == {
        "schema_version",
        "experience_id",
        "adoption_id",
        "capsule_id",
        "root_fingerprint",
        "captured_trust",
        "before",
        "after",
    }
    assert decoded == payload
    assert ExperienceCorroboratedV1.event_type == "experience.corroborated"
    assert set(payload.model_dump()).isdisjoint(
        {"body", "payload", "query", "provenance_chain"}
    )


def test_corroborated_event_locks_trust_weighted_confidence_formula_without_rounding(
) -> None:
    before_confidence = 1.0 / 3.0
    captured_trust = 7.0 / 13.0

    payload = corroborated_payload(
        before_confidence=before_confidence,
        captured_trust=captured_trust,
    )

    expected = before_confidence + (
        1.0 - before_confidence
    ) * 0.20 * captured_trust
    assert payload.after.confidence == expected
    assert payload.after.confidence != round(expected, 6)

    with pytest.raises(ValidationError, match="confidence"):
        ExperienceCorroboratedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "after": payload.after.model_copy(
                    update={"confidence": expected + 0.000001}
                ),
            }
        )


@pytest.mark.parametrize(
    "captured_trust",
    [-0.000001, 1.000001, math.nan, math.inf, -math.inf, True, False, "0.5"],
)
def test_corroborated_event_rejects_invalid_captured_trust(
    captured_trust: object,
) -> None:
    payload = corroborated_payload()

    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "captured_trust": captured_trust,
            }
        )


@pytest.mark.parametrize(
    "root_fingerprint",
    [
        "D" * 64,
        "d" * 63,
        "d" * 65,
        ("d" * 63) + "g",
        "",
    ],
)
def test_corroborated_event_rejects_noncanonical_root_fingerprint(
    root_fingerprint: str,
) -> None:
    payload = corroborated_payload()

    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "root_fingerprint": root_fingerprint,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("importance", 0.36),
        ("source_trust", 0.99),
        ("access_count", 1),
        ("temperature", Temperature.HOT),
        ("current_version_id", NEXT_VERSION_ID),
        ("current_content_hash", NEXT_CONTENT_HASH),
        ("pinned", True),
    ],
)
def test_corroborated_event_allows_only_confidence_materialization_fields(
    field: str,
    value: object,
) -> None:
    payload = corroborated_payload()

    with pytest.raises(ValidationError, match="unauthorized"):
        ExperienceCorroboratedV1.model_validate(
            {
                **payload.model_dump(mode="python"),
                "after": payload.after.model_copy(update={field: value}),
            }
        )


def test_corroborated_event_rejects_wrong_snapshot_anchors_and_extra_fields(
) -> None:
    payload = corroborated_payload()
    values = payload.model_dump(mode="python")

    for snapshot_name in ("before", "after"):
        with pytest.raises(ValidationError):
            ExperienceCorroboratedV1.model_validate(
                {
                    **values,
                    snapshot_name: getattr(payload, snapshot_name).model_copy(
                        update={"experience_id": TARGET_A}
                    ),
                }
            )
    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {**values, "provenance_chain": []}
        )
    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {
                key: value
                for key, value in values.items()
                if key != "adoption_id"
            }
        )
    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {**values, "capsule_id": str(CAPSULE_ID)}
        )
    with pytest.raises(ValidationError):
        ExperienceCorroboratedV1.model_validate(
            {**values, "schema_version": 2}
        )


@pytest.mark.parametrize(
    ("before_confidence", "captured_trust", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.2),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_corroborated_event_accepts_closed_unit_interval_boundaries(
    before_confidence: float,
    captured_trust: float,
    expected: float,
) -> None:
    payload = corroborated_payload(
        before_confidence=before_confidence,
        captured_trust=captured_trust,
    )

    assert payload.after.confidence == pytest.approx(expected, abs=1e-12)


def test_confirmation_and_refutation_canonicalize_required_evidence() -> None:
    confirmed = lifecycle_payloads()[1]
    first = TypedEvidence(type="trace", id="z")
    second = TypedEvidence(type="metric", id="a")
    values = confirmed.model_dump(mode="python")

    canonical = ExperienceConfirmedV1.model_validate(
        {
            **values,
            "reason": None,
            "evidence": (first, second, first),
        }
    )

    assert canonical.evidence == tuple(
        sorted(
            {canonical_json_bytes(value): value for value in (first, second)}.values(),
            key=canonical_json_bytes,
        )
    )
    for payload_type in (ExperienceConfirmedV1, ExperienceRefutedV1):
        with pytest.raises(ValidationError):
            payload_type.model_validate(
                {
                    key: value
                    for key, value in values.items()
                    if key != "evidence"
                }
            )
        with pytest.raises(ValidationError):
            payload_type.model_validate(
                {
                    key: value
                    for key, value in values.items()
                    if key != "reason"
                }
            )


def test_lifecycle_evaluation_locks_utc_time_and_materialization_timestamp() -> None:
    evaluated = lifecycle_payloads()[0]
    plus_eight = timezone(timedelta(hours=8))
    local_time = evaluated.evaluated_at.astimezone(plus_eight)

    normalized = ExperienceLifecycleEvaluatedV1.model_validate(
        {
            **evaluated.model_dump(mode="python"),
            "evaluated_at": local_time,
        }
    )

    assert normalized.evaluated_at == evaluated.evaluated_at
    assert normalized.evaluated_at.tzinfo is UTC
    with pytest.raises(ValidationError):
        ExperienceLifecycleEvaluatedV1.model_validate(
            {
                **evaluated.model_dump(mode="python"),
                "evaluated_at": evaluated.evaluated_at + timedelta(seconds=1),
            }
        )


def test_pin_events_require_exact_false_true_and_true_false_flips() -> None:
    pinned = lifecycle_payloads()[3]
    unpinned = lifecycle_payloads()[4]

    with pytest.raises(ValidationError, match="false to true"):
        ExperiencePinnedV1.model_validate(
            {
                **pinned.model_dump(mode="python"),
                "after": pinned.after.model_copy(update={"pinned": False}),
            }
        )
    with pytest.raises(ValidationError, match="true to false"):
        ExperienceUnpinnedV1.model_validate(
            {
                **unpinned.model_dump(mode="python"),
                "after": unpinned.after.model_copy(update={"pinned": True}),
            }
        )


def test_archive_and_restore_lock_explanation_states_and_policy_reason() -> None:
    archived = lifecycle_payloads()[5]
    restored = lifecycle_payloads()[6]

    assert archived.before == archived.after
    assert archived.reason == StructuredReason.policy_due()
    with pytest.raises(ValidationError):
        ExperienceArchivedV1.model_validate(
            {
                **archived.model_dump(mode="python"),
                "reason": StructuredReason.from_user_text("manual archive"),
            }
        )
    with pytest.raises(ValidationError):
        ExperienceArchivedV1.model_validate(
            {
                **archived.model_dump(mode="python"),
                "after": archived.after.model_copy(
                    update={"activation_score": 0.1}
                ),
            }
        )
    with pytest.raises(ValidationError):
        ExperienceRestoredV1.model_validate(
            {
                **restored.model_dump(mode="python"),
                "before": restored.before.model_copy(
                    update={"temperature": Temperature.COLD}
                ),
            }
        )


def test_lifecycle_events_require_matching_experience_anchors() -> None:
    for payload in lifecycle_payloads():
        with pytest.raises(ValidationError):
            type(payload).model_validate(
                {
                    **payload.model_dump(mode="python"),
                    "experience_id": TARGET_A,
                }
            )
