"""Exact benchmark metric contracts."""

from decimal import Decimal

import pytest

from experience_hub.benchmark.metrics import (
    MetricDomainError,
    macro_recall_at_five,
    recall_at_five,
    unique_mechanism_ratio,
)


def test_recall_at_five_counts_unique_relevant_labels_in_the_first_five() -> None:
    result = recall_at_five(
        frozenset({"alpha", "beta", "gamma"}),
        ("alpha", "alpha", "irrelevant", "beta", "other", "gamma"),
    )

    assert result == Decimal(2) / Decimal(3)
    assert isinstance(result, Decimal)


def test_recall_at_five_does_not_count_relevant_labels_after_rank_five() -> None:
    result = recall_at_five(
        frozenset({"rank-one", "rank-six"}),
        ("rank-one", "two", "three", "four", "five", "rank-six"),
    )

    assert result == Decimal("0.5")


def test_recall_at_five_rejects_an_empty_expected_set() -> None:
    with pytest.raises(
        MetricDomainError,
        match="retrieval recall requires a nonempty expected set",
    ):
        recall_at_five(frozenset(), ("anything",))


def test_macro_recall_at_five_uses_an_exact_unweighted_case_mean() -> None:
    result = macro_recall_at_five(
        (
            (frozenset({"a", "b"}), ("a",)),
            (frozenset({"c"}), ("c",)),
        )
    )

    assert result == Decimal("0.75")
    assert isinstance(result, Decimal)


def test_macro_recall_at_five_rejects_an_empty_case_domain() -> None:
    with pytest.raises(
        MetricDomainError,
        match="macro recall requires at least one relevant case",
    ):
        macro_recall_at_five(())


def test_macro_recall_at_five_rejects_a_distractor_case() -> None:
    with pytest.raises(
        MetricDomainError,
        match="retrieval recall requires a nonempty expected set",
    ):
        macro_recall_at_five(((frozenset(), ()),))


def test_unique_mechanism_ratio_is_exact_and_zero_for_an_empty_valid_set() -> None:
    assert unique_mechanism_ratio(valid_ideas=4, distinct_clusters=3) == Decimal("0.75")
    assert unique_mechanism_ratio(valid_ideas=4, distinct_clusters=0) == Decimal(0)
    assert unique_mechanism_ratio(valid_ideas=0, distinct_clusters=0) == Decimal(0)


@pytest.mark.parametrize(
    ("valid_ideas", "distinct_clusters", "message"),
    (
        (-1, 0, "valid_ideas must be a non-negative integer"),
        (True, 0, "valid_ideas must be a non-negative integer"),
        (1, -1, "distinct_clusters must be a non-negative integer"),
        (1, False, "distinct_clusters must be a non-negative integer"),
        (0, 1, "distinct_clusters cannot exceed valid_ideas"),
        (2, 3, "distinct_clusters cannot exceed valid_ideas"),
    ),
)
def test_unique_mechanism_ratio_rejects_impossible_count_domains(
    valid_ideas: int,
    distinct_clusters: int,
    message: str,
) -> None:
    with pytest.raises(MetricDomainError, match=message):
        unique_mechanism_ratio(
            valid_ideas=valid_ideas,
            distinct_clusters=distinct_clusters,
        )
