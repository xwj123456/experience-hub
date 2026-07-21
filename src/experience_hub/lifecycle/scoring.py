"""Pure memory activation and rehearsal-strength calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from experience_hub.clock import require_utc

MAX_ACCESS_STRENGTH = 20.0


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite float")
    return converted


def _closed_unit_interval(name: str, value: float) -> float:
    converted = _finite_float(name, value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    recency_half_life_hours: float = 168.0
    frequency_half_life_hours: float = 336.0
    warm_to_hot_threshold: float = 0.75
    hot_to_warm_threshold: float = 0.62
    warm_to_cold_threshold: float = 0.30
    demotion_cycles: int = 2
    archive_after_days: float = 90.0
    archive_importance_threshold: float = 0.75
    archive_confidence_threshold: float = 0.25
    archive_strength_threshold: float = 0.10
    minimum_cycle_interval_seconds: float = 15.0 * 60.0
    worker_interval_seconds: float = 15.0 * 60.0
    lease_duration_seconds: float = 5.0 * 60.0

    def __post_init__(self) -> None:
        for field_name in (
            "recency_half_life_hours",
            "frequency_half_life_hours",
            "archive_after_days",
            "minimum_cycle_interval_seconds",
            "worker_interval_seconds",
            "lease_duration_seconds",
        ):
            value = _finite_float(field_name, getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"{field_name} must be greater than zero")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "warm_to_hot_threshold",
            "hot_to_warm_threshold",
            "warm_to_cold_threshold",
            "archive_importance_threshold",
            "archive_confidence_threshold",
        ):
            object.__setattr__(
                self,
                field_name,
                _closed_unit_interval(field_name, getattr(self, field_name)),
            )
        strength_threshold = _finite_float(
            "archive_strength_threshold",
            self.archive_strength_threshold,
        )
        if not 0.0 < strength_threshold <= MAX_ACCESS_STRENGTH:
            raise ValueError(
                "archive_strength_threshold must be greater than zero "
                "and at most 20"
            )
        object.__setattr__(
            self,
            "archive_strength_threshold",
            strength_threshold,
        )
        if (
            isinstance(self.demotion_cycles, bool)
            or not isinstance(self.demotion_cycles, int)
            or self.demotion_cycles < 1
        ):
            raise ValueError("demotion_cycles must be a positive integer")
        if not (
            self.warm_to_cold_threshold
            < self.hot_to_warm_threshold
            < self.warm_to_hot_threshold
        ):
            raise ValueError(
                "Lifecycle activation thresholds must be strictly ordered"
            )

    @property
    def minimum_cycle_interval(self) -> timedelta:
        return timedelta(seconds=self.minimum_cycle_interval_seconds)

    @property
    def worker_interval(self) -> timedelta:
        return timedelta(seconds=self.worker_interval_seconds)

    @property
    def lease_duration(self) -> timedelta:
        return timedelta(seconds=self.lease_duration_seconds)

    @classmethod
    def demo(cls) -> LifecycleConfig:
        return cls(
            minimum_cycle_interval_seconds=60.0,
            worker_interval_seconds=60.0,
        )


@dataclass(frozen=True, slots=True)
class ActivationInputs:
    importance: float
    confidence: float
    access_count: int
    access_strength: float
    strength_updated_at: datetime
    last_accessed_at: datetime | None
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "importance",
            _closed_unit_interval("importance", self.importance),
        )
        object.__setattr__(
            self,
            "confidence",
            _closed_unit_interval("confidence", self.confidence),
        )
        if (
            isinstance(self.access_count, bool)
            or not isinstance(self.access_count, int)
            or self.access_count < 0
        ):
            raise ValueError("access_count must be a non-negative integer")
        strength = _finite_float("access_strength", self.access_strength)
        if not 0.0 <= strength <= MAX_ACCESS_STRENGTH:
            raise ValueError("access_strength must be between 0 and 20")
        object.__setattr__(self, "access_strength", strength)
        object.__setattr__(
            self,
            "created_at",
            require_utc(self.created_at),
        )
        object.__setattr__(
            self,
            "strength_updated_at",
            require_utc(self.strength_updated_at),
        )
        if self.last_accessed_at is not None:
            object.__setattr__(
                self,
                "last_accessed_at",
                require_utc(self.last_accessed_at),
            )


@dataclass(frozen=True, slots=True)
class ActivationResult:
    score: float
    decayed_strength: float
    recency: float
    frequency: float


@dataclass(frozen=True, slots=True)
class AccessUpdate:
    access_count: int
    access_strength: float
    strength_updated_at: datetime
    last_accessed_at: datetime


def _nonnegative_hours(start: datetime, end: datetime) -> float:
    start = require_utc(start)
    end = require_utc(end)
    return max(0.0, (end - start).total_seconds() / 3_600.0)


def decay_strength(
    strength: float,
    *,
    updated_at: datetime,
    at: datetime,
    config: LifecycleConfig,
) -> float:
    """Decay rehearsal strength at the configured half-life."""
    current = _finite_float("strength", strength)
    if not 0.0 <= current <= MAX_ACCESS_STRENGTH:
        raise ValueError("strength must be between 0 and 20")
    age_hours = _nonnegative_hours(updated_at, at)
    return current * math.exp(
        -math.log(2.0) * age_hours / config.frequency_half_life_hours
    )


def record_access(
    state: ActivationInputs,
    at: datetime,
    config: LifecycleConfig,
) -> AccessUpdate:
    """Materialize decay, then add one rehearsal access up to the cap."""
    at = require_utc(at)
    causal_anchors = [state.created_at, state.strength_updated_at]
    if state.last_accessed_at is not None:
        causal_anchors.append(state.last_accessed_at)
    if at < max(causal_anchors):
        raise ValueError("access time would regress materialized state")
    strength = decay_strength(
        state.access_strength,
        updated_at=state.strength_updated_at,
        at=at,
        config=config,
    )
    return AccessUpdate(
        access_count=state.access_count + 1,
        access_strength=min(MAX_ACCESS_STRENGTH, strength + 1.0),
        strength_updated_at=at,
        last_accessed_at=at,
    )


def activation_at(
    state: ActivationInputs,
    at: datetime,
    config: LifecycleConfig,
) -> ActivationResult:
    """Compute activation at a supplied read clock without mutating state."""
    anchor = state.last_accessed_at or state.created_at
    age_hours = _nonnegative_hours(anchor, at)
    recency = math.exp(
        -math.log(2.0) * age_hours / config.recency_half_life_hours
    )
    strength = decay_strength(
        state.access_strength,
        updated_at=state.strength_updated_at,
        at=at,
        config=config,
    )
    frequency = min(
        1.0,
        math.log1p(strength) / math.log1p(MAX_ACCESS_STRENGTH),
    )
    score = (
        0.30 * state.importance
        + 0.15 * state.confidence
        + 0.25 * frequency
        + 0.30 * recency
    )
    return ActivationResult(
        score=min(1.0, max(0.0, score)),
        decayed_strength=strength,
        recency=recency,
        frequency=frequency,
    )
