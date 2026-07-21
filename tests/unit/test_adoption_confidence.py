from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from experience_hub.sharing.confidence import (
    corroboration_delta,
    initial_adoption_confidence,
    observer_trust,
)


def test_observer_trust_defaults_to_locked_two_two_prior() -> None:
    assert observer_trust() == pytest.approx(0.5, abs=1e-12)
    assert observer_trust(alpha=2, beta=2) == pytest.approx(0.5, abs=1e-12)


def test_observer_trust_uses_effective_bayesian_counts() -> None:
    assert observer_trust(alpha=3, beta=2) == pytest.approx(0.6, abs=1e-12)
    assert observer_trust(alpha=2, beta=3) == pytest.approx(0.4, abs=1e-12)
    assert observer_trust(alpha=7, beta=5) == pytest.approx(
        7.0 / 12.0,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (None, 2),
        (2, None),
        (1, 2),
        (2, 1),
        (True, 2),
        (2, False),
        (2.0, 2),
        (2, 2.0),
        ("2", 2),
        (2, "2"),
    ],
)
def test_observer_trust_rejects_partial_or_invalid_effective_counts(
    alpha: object,
    beta: object,
) -> None:
    with pytest.raises(ValueError):
        observer_trust(alpha=alpha, beta=beta)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("publisher_confidence", "trust", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.8, 0.5, 0.4),
        (0.73, 0.6, 0.438),
    ],
)
def test_initial_adoption_confidence_multiplies_publisher_and_observer_scores(
    publisher_confidence: float,
    trust: float,
    expected: float,
) -> None:
    assert initial_adoption_confidence(
        publisher_confidence,
        trust,
    ) == pytest.approx(expected, abs=1e-12)


def test_initial_adoption_confidence_does_not_apply_business_rounding() -> None:
    publisher_confidence = 0.7333333333333333
    trust = 7.0 / 13.0

    result = initial_adoption_confidence(publisher_confidence, trust)

    assert result == publisher_confidence * trust
    assert result != round(result, 6)


@pytest.mark.parametrize(
    ("current_confidence", "trust", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.2),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.5, 0.5, 0.05),
        (0.8, 0.75, 0.03),
    ],
)
def test_corroboration_delta_uses_remaining_confidence_and_captured_trust(
    current_confidence: float,
    trust: float,
    expected: float,
) -> None:
    delta = corroboration_delta(current_confidence, trust)

    assert delta == pytest.approx(expected, abs=1e-12)
    assert current_confidence + delta <= 1.0


def test_corroboration_delta_does_not_apply_business_rounding() -> None:
    current_confidence = 1.0 / 3.0
    trust = 7.0 / 13.0

    result = corroboration_delta(current_confidence, trust)

    assert result == (1.0 - current_confidence) * 0.20 * trust
    assert result != round(result, 6)


_SCORE_FUNCTIONS: tuple[Callable[[float, float], float], ...] = (
    initial_adoption_confidence,
    corroboration_delta,
)


@pytest.mark.parametrize(
    "invalid",
    [
        -0.000001,
        1.000001,
        math.nan,
        math.inf,
        -math.inf,
        True,
        False,
        "0.5",
        None,
    ],
)
@pytest.mark.parametrize("function", _SCORE_FUNCTIONS)
@pytest.mark.parametrize("invalid_position", [0, 1])
def test_adoption_confidence_functions_reject_non_finite_non_numeric_or_out_of_range(
    function: Callable[[float, float], float],
    invalid_position: int,
    invalid: Any,
) -> None:
    values: list[Any] = [0.5, 0.5]
    values[invalid_position] = invalid

    with pytest.raises(ValueError):
        function(*values)
