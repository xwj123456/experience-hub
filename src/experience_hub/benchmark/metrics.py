"""Exact metrics for the deterministic effectiveness benchmark."""

from decimal import Decimal


class MetricDomainError(ValueError):
    """Raised when a benchmark metric is undefined for its inputs."""


def recall_at_five(
    expected: frozenset[str],
    returned: tuple[str, ...],
) -> Decimal:
    """Return exact unique-label recall over the first five returned positions."""
    if not expected:
        raise MetricDomainError("retrieval recall requires a nonempty expected set")
    relevant_returned = expected.intersection(returned[:5])
    return Decimal(len(relevant_returned)) / Decimal(len(expected))


def macro_recall_at_five(
    cases: tuple[tuple[frozenset[str], tuple[str, ...]], ...],
) -> Decimal:
    """Return the exact unweighted recall mean over relevant retrieval cases."""
    if not cases:
        raise MetricDomainError("macro recall requires at least one relevant case")
    total = sum(
        (recall_at_five(expected, returned) for expected, returned in cases),
        Decimal(0),
    )
    return total / Decimal(len(cases))


def unique_mechanism_ratio(
    valid_ideas: int,
    distinct_clusters: int,
) -> Decimal:
    """Return exact distinct-mechanism coverage over valid generated ideas."""
    _require_non_negative_count(valid_ideas, name="valid_ideas")
    _require_non_negative_count(distinct_clusters, name="distinct_clusters")
    if distinct_clusters > valid_ideas:
        raise MetricDomainError("distinct_clusters cannot exceed valid_ideas")
    if valid_ideas == 0:
        return Decimal(0)
    return Decimal(distinct_clusters) / Decimal(valid_ideas)


def _require_non_negative_count(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricDomainError(f"{name} must be a non-negative integer")
