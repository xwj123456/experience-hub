"""Pure confidence rules for deliberate capsule adoption."""

from __future__ import annotations

import math

_PRIOR_ALPHA = 2
_PRIOR_BETA = 2
_CORROBORATION_WEIGHT = 0.20


def _unit_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between zero and one")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite number between zero and one")
    return converted


def observer_trust(
    *,
    alpha: int | None = None,
    beta: int | None = None,
) -> float:
    """Return observer-relative Bayesian trust with the locked 2/2 prior."""
    if alpha is None and beta is None:
        alpha = _PRIOR_ALPHA
        beta = _PRIOR_BETA
    elif alpha is None or beta is None:
        raise ValueError("alpha and beta must both be supplied or both omitted")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, int)
        or alpha < _PRIOR_ALPHA
        or isinstance(beta, bool)
        or not isinstance(beta, int)
        or beta < _PRIOR_BETA
    ):
        raise ValueError("alpha and beta must be integers at least two")
    return alpha / (alpha + beta)


def initial_adoption_confidence(
    publisher_confidence: float,
    trust: float,
) -> float:
    """Weight transported confidence by the adopter's captured trust."""
    return _unit_float(
        "publisher_confidence",
        publisher_confidence,
    ) * _unit_float("trust", trust)


def corroboration_delta(
    current_confidence: float,
    trust: float,
) -> float:
    """Return the one-time contribution of one independent provenance root."""
    current = _unit_float("current_confidence", current_confidence)
    captured = _unit_float("trust", trust)
    return (1.0 - current) * _CORROBORATION_WEIGHT * captured


__all__ = [
    "corroboration_delta",
    "initial_adoption_confidence",
    "observer_trust",
]
