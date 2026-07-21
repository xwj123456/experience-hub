from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.experiences.events import ExperienceStateSnapshotV1
from experience_hub.experiences.models import Temperature
from experience_hub.lifecycle.contracts import (
    IdeaArchivePlanner,
    LifecycleResult,
    NullIdeaArchivePlanner,
    decode_lifecycle_result,
    encode_lifecycle_result,
)
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.lifecycle.service import (
    LifecycleThresholdTarget,
    evaluate_transition,
    lifecycle_config_hash,
    lifecycle_cycle_id,
)

CREATED_AT = datetime(2026, 1, 1, 12, tzinfo=UTC)
EXPERIENCE_ID = UUID("30000000-0000-0000-0000-000000000001")
OWNER_ID = UUID("30000000-0000-0000-0000-000000000002")
VERSION_ID = UUID("30000000-0000-0000-0000-000000000003")
CYCLE_ID = UUID("30000000-0000-0000-0000-000000000004")


def _state(**overrides: object) -> ExperienceStateSnapshotV1:
    values: dict[str, Any] = {
        "experience_id": EXPERIENCE_ID,
        "owner_agent_id": OWNER_ID,
        "current_version_id": VERSION_ID,
        "current_content_hash": "a" * 64,
        "temperature": Temperature.WARM,
        "importance": 0.0,
        "confidence": 0.0,
        "activation_score": 0.30,
        "source_trust": 1.0,
        "access_count": 0,
        "access_strength": 0.0,
        "strength_updated_at": CREATED_AT,
        "last_accessed_at": None,
        "last_transition_at": CREATED_AT,
        "last_lifecycle_evaluated_at": None,
        "consecutive_below_threshold": 0,
        "pinned": False,
    }
    values.update(overrides)
    return ExperienceStateSnapshotV1.model_validate(values)


def _next_evaluation_state(
    state: ExperienceStateSnapshotV1,
    *,
    at: datetime,
    materialized_strength: float,
    activation: float,
    counter_after: int,
) -> ExperienceStateSnapshotV1:
    return state.model_copy(
        update={
            "access_strength": materialized_strength,
            "strength_updated_at": at,
            "activation_score": activation,
            "last_lifecycle_evaluated_at": at,
            "consecutive_below_threshold": counter_after,
        }
    )


def test_lifecycle_config_locks_default_policy_and_demo_interval() -> None:
    config = LifecycleConfig()

    assert config.warm_to_hot_threshold == 0.75
    assert config.hot_to_warm_threshold == 0.62
    assert config.warm_to_cold_threshold == 0.30
    assert config.demotion_cycles == 2
    assert config.archive_after_days == 90.0
    assert config.archive_importance_threshold == 0.75
    assert config.archive_confidence_threshold == 0.25
    assert config.archive_strength_threshold == 0.10
    assert config.minimum_cycle_interval == timedelta(minutes=15)
    assert config.worker_interval == timedelta(minutes=15)
    assert config.lease_duration > timedelta(0)
    assert LifecycleConfig.demo().minimum_cycle_interval == timedelta(seconds=60)
    assert LifecycleConfig.demo().worker_interval == timedelta(seconds=60)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warm_to_hot_threshold", math.nan),
        ("hot_to_warm_threshold", -0.01),
        ("warm_to_cold_threshold", 1.01),
        ("demotion_cycles", 0),
        ("demotion_cycles", True),
        ("archive_after_days", 0),
        ("archive_importance_threshold", math.inf),
        ("archive_confidence_threshold", -0.1),
        ("archive_strength_threshold", 20.01),
        ("minimum_cycle_interval_seconds", 0),
        ("worker_interval_seconds", -1),
        ("lease_duration_seconds", math.nan),
    ],
)
def test_lifecycle_config_rejects_invalid_policy_values(
    field: str,
    value: float | int | bool,
) -> None:
    with pytest.raises(ValueError, match=field):
        LifecycleConfig(**cast(Any, {field: value}))


def test_lifecycle_config_requires_ordered_activation_thresholds() -> None:
    with pytest.raises(ValueError, match="threshold"):
        LifecycleConfig(
            warm_to_cold_threshold=0.62,
            hot_to_warm_threshold=0.62,
        )
    with pytest.raises(ValueError, match="threshold"):
        LifecycleConfig(
            hot_to_warm_threshold=0.75,
            warm_to_hot_threshold=0.75,
        )


def test_config_hash_and_cycle_id_are_canonical_and_deterministic() -> None:
    default = LifecycleConfig()
    equivalent = LifecycleConfig(
        recency_half_life_hours=168,
        frequency_half_life_hours=336,
    )
    plus_eight = timezone(timedelta(hours=8))
    local_time = datetime(2026, 7, 18, 20, tzinfo=plus_eight)
    utc_time = datetime(2026, 7, 18, 12, tzinfo=UTC)

    assert lifecycle_config_hash(default) == lifecycle_config_hash(equivalent)
    first = lifecycle_cycle_id(evaluated_at=local_time, config=default)
    second = lifecycle_cycle_id(evaluated_at=utc_time, config=equivalent)
    assert first == second
    assert first.version == 5
    assert lifecycle_cycle_id(
        evaluated_at=utc_time + timedelta(microseconds=1),
        config=default,
    ) != first
    assert lifecycle_cycle_id(
        evaluated_at=utc_time,
        config=LifecycleConfig(warm_to_hot_threshold=0.76),
    ) != first


@pytest.mark.parametrize(
    ("importance", "confidence", "strength", "target", "transition"),
    [
        (
            1.0,
            0.99,
            0.0,
            LifecycleThresholdTarget.NONE,
            None,
        ),
        (
            1.0,
            1.0,
            0.0,
            LifecycleThresholdTarget.PROMOTE_HOT,
            Temperature.HOT,
        ),
        (
            1.0,
            1.0,
            1.0,
            LifecycleThresholdTarget.PROMOTE_HOT,
            Temperature.HOT,
        ),
    ],
)
def test_warm_promotion_respects_below_equal_and_above_threshold(
    importance: float,
    confidence: float,
    strength: float,
    target: LifecycleThresholdTarget,
    transition: Temperature | None,
) -> None:
    result = evaluate_transition(
        state=_state(
            importance=importance,
            confidence=confidence,
            access_strength=strength,
        ),
        created_at=CREATED_AT,
        at=CREATED_AT,
        config=LifecycleConfig(),
        has_active_dependents=False,
    )

    assert result.eligible is True
    assert result.threshold_target is target
    assert result.transition is transition
    assert result.counter_before == 0
    assert result.counter_after == 0
    if confidence == 1.0 and strength == 0.0:
        assert result.activation == pytest.approx(0.75, abs=1e-12)


@pytest.mark.parametrize(
    ("temperature", "target", "transition"),
    [
        (
            Temperature.HOT,
            LifecycleThresholdTarget.DEMOTE_WARM,
            Temperature.WARM,
        ),
        (
            Temperature.WARM,
            LifecycleThresholdTarget.DEMOTE_COLD,
            Temperature.COLD,
        ),
    ],
)
def test_hot_and_warm_demotion_require_two_eligible_cycles(
    temperature: Temperature,
    target: LifecycleThresholdTarget,
    transition: Temperature,
) -> None:
    config = LifecycleConfig()
    first_at = CREATED_AT + config.minimum_cycle_interval
    state = _state(temperature=temperature)

    first = evaluate_transition(
        state=state,
        created_at=CREATED_AT,
        at=first_at,
        config=config,
        has_active_dependents=False,
    )

    assert first.threshold_target is target
    assert first.counter_before == 0
    assert first.counter_after == 1
    assert first.transition is None
    first_state = _next_evaluation_state(
        state,
        at=first_at,
        materialized_strength=first.materialized_strength,
        activation=first.activation,
        counter_after=first.counter_after,
    )
    second_at = first_at + config.minimum_cycle_interval

    second = evaluate_transition(
        state=first_state,
        created_at=CREATED_AT,
        at=second_at,
        config=config,
        has_active_dependents=False,
    )

    assert second.threshold_target is target
    assert second.counter_before == 1
    assert second.counter_after == 2
    assert second.transition is transition


@pytest.mark.parametrize(
    ("temperature", "config", "importance", "confidence"),
    [
        (
            Temperature.HOT,
            LifecycleConfig(
                warm_to_cold_threshold=0.20,
                hot_to_warm_threshold=0.30,
            ),
            0.0,
            0.0,
        ),
        (
            Temperature.WARM,
            LifecycleConfig(),
            0.0,
            0.0,
        ),
    ],
)
def test_demotion_threshold_equality_recovers_and_resets_counter(
    temperature: Temperature,
    config: LifecycleConfig,
    importance: float,
    confidence: float,
) -> None:
    result = evaluate_transition(
        state=_state(
            temperature=temperature,
            importance=importance,
            confidence=confidence,
            consecutive_below_threshold=1,
        ),
        created_at=CREATED_AT,
        at=CREATED_AT,
        config=config,
        has_active_dependents=False,
    )

    assert result.activation == pytest.approx(
        (
            config.hot_to_warm_threshold
            if temperature is Temperature.HOT
            else config.warm_to_cold_threshold
        ),
        abs=1e-12,
    )
    assert result.threshold_target is LifecycleThresholdTarget.NONE
    assert result.counter_before == 1
    assert result.counter_after == 0
    assert result.transition is None


def test_target_change_to_promotion_resets_prior_demotion_counter() -> None:
    result = evaluate_transition(
        state=_state(
            importance=1.0,
            confidence=1.0,
            consecutive_below_threshold=1,
        ),
        created_at=CREATED_AT,
        at=CREATED_AT,
        config=LifecycleConfig(),
        has_active_dependents=False,
    )

    assert result.threshold_target is LifecycleThresholdTarget.PROMOTE_HOT
    assert result.counter_before == 1
    assert result.counter_after == 0
    assert result.transition is Temperature.HOT


def test_minimum_cycle_interval_is_a_complete_noop() -> None:
    config = LifecycleConfig()
    state = _state(
        temperature=Temperature.HOT,
        access_strength=2.0,
        activation_score=0.51,
        last_lifecycle_evaluated_at=CREATED_AT,
        consecutive_below_threshold=1,
    )

    result = evaluate_transition(
        state=state,
        created_at=CREATED_AT,
        at=CREATED_AT + config.minimum_cycle_interval - timedelta(microseconds=1),
        config=config,
        has_active_dependents=False,
    )

    assert result.eligible is False
    assert result.materialized_strength == state.access_strength
    assert result.activation == state.activation_score
    assert result.threshold_target is LifecycleThresholdTarget.NONE
    assert result.counter_before == result.counter_after == 1
    assert result.transition is None


def test_pinned_hot_memory_never_demotes_and_resets_stale_counter() -> None:
    config = LifecycleConfig()

    result = evaluate_transition(
        state=_state(
            temperature=Temperature.HOT,
            pinned=True,
            consecutive_below_threshold=1,
        ),
        created_at=CREATED_AT,
        at=CREATED_AT + config.minimum_cycle_interval,
        config=config,
        has_active_dependents=False,
    )

    assert result.eligible is True
    assert result.threshold_target is LifecycleThresholdTarget.NONE
    assert result.counter_after == 0
    assert result.transition is None


def test_cold_archive_is_immediate_at_all_strict_policy_gates() -> None:
    config = LifecycleConfig()
    at = CREATED_AT + timedelta(days=config.archive_after_days)

    result = evaluate_transition(
        state=_state(
            temperature=Temperature.COLD,
            importance=0.74,
            confidence=0.24,
            access_strength=0.09,
        ),
        created_at=CREATED_AT,
        at=at,
        config=config,
        has_active_dependents=False,
    )

    assert result.eligible is True
    assert result.materialized_strength < config.archive_strength_threshold
    assert result.threshold_target is LifecycleThresholdTarget.ARCHIVE
    assert result.counter_before == result.counter_after == 0
    assert result.transition is Temperature.ARCHIVED


@pytest.mark.parametrize(
    ("state_overrides", "at_delta", "has_active_dependents"),
    [
        ({}, timedelta(days=90) - timedelta(microseconds=1), False),
        ({"importance": 0.75}, timedelta(days=90), False),
        ({"confidence": 0.25}, timedelta(days=90), False),
        ({"access_strength": 20.0}, timedelta(days=90), False),
        ({"pinned": True}, timedelta(days=90), False),
        ({}, timedelta(days=90), True),
    ],
)
def test_cold_archive_requires_every_gate(
    state_overrides: dict[str, object],
    at_delta: timedelta,
    has_active_dependents: bool,
) -> None:
    values: dict[str, object] = {
        "temperature": Temperature.COLD,
        "importance": 0.74,
        "confidence": 0.24,
        "access_strength": 0.09,
    }
    values.update(state_overrides)

    result = evaluate_transition(
        state=_state(**values),
        created_at=CREATED_AT,
        at=CREATED_AT + at_delta,
        config=LifecycleConfig(),
        has_active_dependents=has_active_dependents,
    )

    assert result.threshold_target is LifecycleThresholdTarget.NONE
    assert result.counter_after == 0
    assert result.transition is None


def test_archived_state_is_not_eligible_for_automatic_evaluation() -> None:
    state = _state(
        temperature=Temperature.ARCHIVED,
        access_strength=1.0,
        activation_score=0.4,
        consecutive_below_threshold=1,
    )

    result = evaluate_transition(
        state=state,
        created_at=CREATED_AT,
        at=CREATED_AT + timedelta(days=365),
        config=LifecycleConfig(),
        has_active_dependents=False,
    )

    assert result.eligible is False
    assert result.materialized_strength == state.access_strength
    assert result.activation == state.activation_score
    assert result.counter_before == result.counter_after == 1
    assert result.transition is None


def test_transition_evaluation_rejects_regressed_clock_and_wrong_types() -> None:
    state = _state(strength_updated_at=CREATED_AT + timedelta(minutes=1))
    with pytest.raises(ValueError, match="regress"):
        evaluate_transition(
            state=state,
            created_at=CREATED_AT,
            at=CREATED_AT,
            config=LifecycleConfig(),
            has_active_dependents=False,
        )
    with pytest.raises(ValueError, match="state"):
        evaluate_transition(
            state=cast(Any, object()),
            created_at=CREATED_AT,
            at=CREATED_AT,
            config=LifecycleConfig(),
            has_active_dependents=False,
        )
    with pytest.raises(ValueError, match="has_active_dependents"):
        evaluate_transition(
            state=_state(),
            created_at=CREATED_AT,
            at=CREATED_AT,
            config=LifecycleConfig(),
            has_active_dependents=cast(Any, 1),
        )


def test_lifecycle_result_has_one_strict_canonical_success_codec() -> None:
    result = LifecycleResult(
        cycle_id=CYCLE_ID,
        evaluated_at=CREATED_AT,
        evaluated_count=7,
        transition_count=3,
        archive_count=1,
        idea_archive_count=2,
    )

    encoded = encode_lifecycle_result(result)

    assert encoded == (
        b'{"data":{"archive_count":1,'
        b'"cycle_id":"30000000-0000-0000-0000-000000000004",'
        b'"evaluated_at":"2026-01-01T12:00:00.000000Z",'
        b'"evaluated_count":7,"idea_archive_count":2,'
        b'"transition_count":3}}'
    )
    assert decode_lifecycle_result(encoded) == result


@pytest.mark.parametrize(
    "body",
    [
        b'{"data": {"archive_count":0}}',
        b'{"data":NaN}',
        b'{"data":{},"extra":null}',
        b'{"data":{"archive_count":-1,"cycle_id":'
        b'"30000000-0000-0000-0000-000000000004",'
        b'"evaluated_at":"2026-01-01T12:00:00.000000Z",'
        b'"evaluated_count":0,"idea_archive_count":0,'
        b'"transition_count":0}}',
    ],
)
def test_lifecycle_result_decoder_rejects_noncanonical_or_invalid_body(
    body: bytes,
) -> None:
    with pytest.raises(ValueError):
        decode_lifecycle_result(body)


def test_lifecycle_result_rejects_non_utc_or_invalid_counters() -> None:
    with pytest.raises(ValidationError):
        LifecycleResult(
            cycle_id=CYCLE_ID,
            evaluated_at=datetime(2026, 1, 1, 12),
            evaluated_count=0,
            transition_count=0,
            archive_count=0,
            idea_archive_count=0,
        )
    with pytest.raises(ValidationError):
        LifecycleResult(
            cycle_id=CYCLE_ID,
            evaluated_at=CREATED_AT,
            evaluated_count=True,
            transition_count=0,
            archive_count=0,
            idea_archive_count=0,
        )


@pytest.mark.asyncio
async def test_null_idea_archive_planner_is_strict_and_returns_no_events() -> None:
    planner: IdeaArchivePlanner = NullIdeaArchivePlanner()

    events = await planner.due_archive_events(
        session=cast(AsyncSession, object()),
        evaluated_at=CREATED_AT,
        cycle_id=CYCLE_ID,
    )

    assert events == ()
    with pytest.raises(ValueError, match="cycle_id"):
        await planner.due_archive_events(
            session=cast(AsyncSession, object()),
            evaluated_at=CREATED_AT,
            cycle_id=cast(Any, str(CYCLE_ID)),
        )
    with pytest.raises(ValueError, match="evaluated_at"):
        await planner.due_archive_events(
            session=cast(AsyncSession, object()),
            evaluated_at=cast(Any, "2026-01-01T12:00:00Z"),
            cycle_id=CYCLE_ID,
        )
