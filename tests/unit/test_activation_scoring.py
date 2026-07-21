from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from experience_hub.lifecycle.scoring import (
    MAX_ACCESS_STRENGTH,
    ActivationInputs,
    LifecycleConfig,
    activation_at,
    decay_strength,
    record_access,
)

CREATED_AT = datetime(2026, 7, 1, 12, tzinfo=UTC)


def _state(**overrides: object) -> ActivationInputs:
    values: dict[str, Any] = {
        "importance": 0.35,
        "confidence": 0.50,
        "access_count": 0,
        "access_strength": 0.0,
        "strength_updated_at": CREATED_AT,
        "last_accessed_at": None,
        "created_at": CREATED_AT,
    }
    values.update(overrides)
    return ActivationInputs(**values)


def test_never_accessed_creation_state_materializes_creation_activation() -> None:
    result = activation_at(_state(), CREATED_AT, LifecycleConfig())

    assert result.recency == pytest.approx(1.0, abs=1e-12)
    assert result.decayed_strength == pytest.approx(0.0, abs=1e-12)
    assert result.frequency == pytest.approx(0.0, abs=1e-12)
    assert result.score == pytest.approx(0.48, abs=1e-12)


def test_never_accessed_recency_is_one_half_after_one_half_life() -> None:
    config = LifecycleConfig()

    result = activation_at(
        _state(),
        CREATED_AT + timedelta(hours=config.recency_half_life_hours),
        config,
    )

    assert result.recency == pytest.approx(0.5, abs=1e-12)
    assert result.score == pytest.approx(0.33, abs=1e-12)


def test_decay_strength_uses_the_rehearsal_half_life() -> None:
    config = LifecycleConfig()

    decayed = decay_strength(
        8.0,
        updated_at=CREATED_AT,
        at=CREATED_AT + timedelta(hours=config.frequency_half_life_hours),
        config=config,
    )

    assert decayed == pytest.approx(4.0, abs=1e-12)


def test_activation_and_strength_are_quarter_after_two_half_lives() -> None:
    config = LifecycleConfig(
        recency_half_life_hours=12.0,
        frequency_half_life_hours=12.0,
    )
    state = _state(
        importance=0.0,
        confidence=0.0,
        access_strength=8.0,
        last_accessed_at=CREATED_AT,
    )

    result = activation_at(
        state,
        CREATED_AT + timedelta(hours=24),
        config,
    )

    expected_frequency = math.log1p(2.0) / math.log1p(MAX_ACCESS_STRENGTH)
    assert result.recency == pytest.approx(0.25, abs=1e-12)
    assert result.decayed_strength == pytest.approx(2.0, abs=1e-12)
    assert result.frequency == pytest.approx(expected_frequency, abs=1e-12)
    assert result.score == pytest.approx(
        0.25 * expected_frequency + 0.30 * 0.25,
        abs=1e-12,
    )


def test_rehearsal_strength_can_decay_below_cold_archive_threshold() -> None:
    config = LifecycleConfig()

    decayed = decay_strength(
        1.0,
        updated_at=CREATED_AT,
        at=CREATED_AT
        + timedelta(hours=4 * config.frequency_half_life_hours),
        config=config,
    )

    assert decayed == pytest.approx(0.0625, abs=1e-12)
    assert decayed < 0.10


def test_decay_strength_rejects_values_above_persisted_access_cap() -> None:
    with pytest.raises(ValueError, match="between 0 and 20"):
        decay_strength(
            MAX_ACCESS_STRENGTH + 0.001,
            updated_at=CREATED_AT,
            at=CREATED_AT,
            config=LifecycleConfig(),
        )


def test_record_access_decays_first_then_increments() -> None:
    config = LifecycleConfig()
    at = CREATED_AT + timedelta(hours=config.frequency_half_life_hours)
    state = _state(access_count=2, access_strength=8.0)

    update = record_access(state, at, config)

    assert update.access_count == 3
    assert update.access_strength == pytest.approx(5.0, abs=1e-12)
    assert update.strength_updated_at == at
    assert update.last_accessed_at == at


def test_record_access_rejects_time_before_creation_even_when_other_anchors_are_earlier(
) -> None:
    created_at = CREATED_AT + timedelta(hours=2)
    state = _state(
        created_at=created_at,
        strength_updated_at=CREATED_AT,
        last_accessed_at=None,
    )

    with pytest.raises(
        ValueError,
        match="access time would regress materialized state",
    ):
        record_access(
            state,
            CREATED_AT + timedelta(hours=1),
            LifecycleConfig(),
        )


def test_backward_read_only_scoring_clamps_elapsed_time_to_zero() -> None:
    future = CREATED_AT + timedelta(hours=12)
    state = _state(
        access_strength=3.0,
        strength_updated_at=future,
        last_accessed_at=future,
    )

    result = activation_at(state, CREATED_AT, LifecycleConfig())

    assert result.recency == pytest.approx(1.0, abs=1e-12)
    assert result.decayed_strength == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recency_half_life_hours", 0.0),
        ("recency_half_life_hours", math.inf),
        ("frequency_half_life_hours", -1.0),
        ("frequency_half_life_hours", math.nan),
    ],
)
def test_lifecycle_config_rejects_nonfinite_or_nonpositive_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        if field == "recency_half_life_hours":
            LifecycleConfig(recency_half_life_hours=value)
        else:
            LifecycleConfig(frequency_half_life_hours=value)


def test_access_strength_cap_is_locked_not_runtime_configuration() -> None:
    assert MAX_ACCESS_STRENGTH == 20.0
    with pytest.raises(TypeError, match="access_strength_cap"):
        LifecycleConfig(access_strength_cap=10.0)  # type: ignore[call-arg]

    state = _state(access_count=20, access_strength=MAX_ACCESS_STRENGTH)
    configs = (
        LifecycleConfig(
            recency_half_life_hours=1.0,
            frequency_half_life_hours=1.0,
        ),
        LifecycleConfig(
            recency_half_life_hours=10_000.0,
            frequency_half_life_hours=10_000.0,
        ),
    )

    for config in configs:
        update = record_access(state, CREATED_AT, config)
        result = activation_at(state, CREATED_AT, config)
        assert update.access_strength == MAX_ACCESS_STRENGTH
        assert result.frequency == pytest.approx(1.0, abs=1e-12)


def test_inputs_and_access_outputs_normalize_aware_offsets_to_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))
    local_created_at = datetime(2026, 7, 1, 20, tzinfo=plus_eight)
    state = _state(
        created_at=local_created_at,
        strength_updated_at=local_created_at,
        last_accessed_at=local_created_at,
    )

    assert state.created_at == CREATED_AT
    assert state.created_at.tzinfo is UTC
    assert state.strength_updated_at == CREATED_AT
    assert state.strength_updated_at.tzinfo is UTC
    assert state.last_accessed_at == CREATED_AT
    assert state.last_accessed_at.tzinfo is UTC

    local_access_at = datetime(2026, 7, 1, 21, tzinfo=plus_eight)
    update = record_access(state, local_access_at, LifecycleConfig())
    expected_access_at = datetime(2026, 7, 1, 13, tzinfo=UTC)
    assert update.strength_updated_at == expected_access_at
    assert update.strength_updated_at.tzinfo is UTC
    assert update.last_accessed_at == expected_access_at
    assert update.last_accessed_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("importance", -0.1),
        ("confidence", 1.1),
        ("access_strength", math.nan),
        ("access_strength", MAX_ACCESS_STRENGTH + 0.001),
        ("access_count", -1),
        ("access_count", True),
    ],
)
def test_activation_inputs_reject_invalid_persisted_ranges(
    field: str,
    value: float | int,
) -> None:
    with pytest.raises(ValueError, match=field):
        _state(**{field: value})
