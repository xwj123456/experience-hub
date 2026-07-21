from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.hashing import (
    hash_idea_content,
    hash_mechanism,
    hash_snapshot,
    mechanism_similarity,
    normalize_mechanism,
    snapshot_canonical_bytes,
    stable_evidence_key,
    truncate_utf8,
)
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    EvaluationVerdict,
    EvidenceSourceState,
    EvidenceSourceType,
    ExperienceVersionEvidenceReference,
    GeneratorKind,
    IdeaDraft,
    IdeaEvaluation,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.retrieval import RetrievalMode

NOW = datetime(2026, 7, 18, 8, tzinfo=UTC)
OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")
RUN_ID = UUID("00000000-0000-0000-0000-000000000102")
ITEM_ID = UUID("00000000-0000-0000-0000-000000000103")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000104")
VERSION_ID = UUID("00000000-0000-0000-0000-000000000105")
CONTENT_HASH = "a" * 64


def make_snapshot_item(
    *,
    run_id: UUID = RUN_ID,
    snapshot_item_id: UUID = ITEM_ID,
    captured_at: datetime = NOW,
    rank: int = 1,
    summary: str = "缓存失效应在提交后发生。",
) -> SnapshotItem:
    key = stable_evidence_key(
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=SOURCE_ID,
        source_version_id=VERSION_ID,
        content_hash=CONTENT_HASH,
    )
    return SnapshotItem(
        snapshot_item_id=snapshot_item_id,
        stable_evidence_key=key,
        run_id=run_id,
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=SOURCE_ID,
        source_version_id=VERSION_ID,
        source_state=EvidenceSourceState.WARM,
        source_trust=1.0,
        rank=rank,
        summary=summary,
        mechanism="Commit then invalidate cache.",
        applicability=("transaction committed", "cache entry exists"),
        tags=("cache", "consistency"),
        falsifiers=("Invalidation before commit never exposes stale data.",),
        excerpt="Persist first; invalidate only after the commit succeeds.",
        content_hash=CONTENT_HASH,
        captured_at=captured_at,
    )


def make_idea(
    *,
    predictions: tuple[str, ...] = ("B", "A", "A"),
    falsifiers: tuple[str, ...] = ("F2", "F1"),
    assumptions: tuple[str, ...] = ("S2", "S1"),
    evidence: tuple[SnapshotEvidenceReference, ...] | None = None,
) -> IdeaDraft:
    reference = SnapshotEvidenceReference(
        id=ITEM_ID,
        stable_evidence_key=make_snapshot_item().stable_evidence_key,
    )
    return IdeaDraft(
        title="Commit-bound invalidation",
        hypothesis="Invalidation before commit creates an observable stale window.",
        mechanism="Commit then invalidate cache.",
        predictions=predictions,
        falsifiers=falsifiers,
        assumptions=assumptions,
        proposed_test="Inject a rollback between write and invalidation.",
        evidence=evidence or (reference,),
    )


@pytest.mark.parametrize(
    ("limit", "expected"),
    (
        (0, ""),
        (1, "A"),
        (2, "A"),
        (3, "A"),
        (4, "A知"),
        (5, "A知B"),
        (100, "A知B"),
    ),
)
def test_truncate_utf8_never_splits_a_code_point(
    limit: int,
    expected: str,
) -> None:
    result = truncate_utf8("A知B", limit)
    assert result == expected
    assert len(result.encode("utf-8")) <= limit


@pytest.mark.parametrize("limit", (-1, True, 1.5))
def test_truncate_utf8_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        truncate_utf8("text", limit)  # type: ignore[arg-type]


def test_stable_evidence_key_uses_only_canonical_source_identity_and_hash() -> None:
    first = stable_evidence_key(
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=SOURCE_ID,
        source_version_id=VERSION_ID,
        content_hash=CONTENT_HASH,
    )
    second = stable_evidence_key(
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=SOURCE_ID,
        source_version_id=VERSION_ID,
        content_hash=CONTENT_HASH,
    )
    changed = stable_evidence_key(
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=SOURCE_ID,
        source_version_id=UUID(int=VERSION_ID.int + 1),
        content_hash=CONTENT_HASH,
    )
    assert first == second
    assert first == "e283c6dfefc29d138e4b1e73cf60baeb310d3692cbb1a890033d5108a4968631"
    assert first != changed
    assert len(first) == 64


def test_equivalent_snapshot_items_hash_identically_across_runs() -> None:
    first = make_snapshot_item()
    second = make_snapshot_item(
        run_id=UUID(int=RUN_ID.int + 1),
        snapshot_item_id=UUID(int=ITEM_ID.int + 1),
        captured_at=NOW + timedelta(days=1),
    )
    assert first.stable_evidence_key == second.stable_evidence_key
    assert hash_snapshot((first,)) == hash_snapshot((second,))
    assert (
        hash_snapshot((first,))
        == "fc8070be10c3dea515129ea5441566512359d22363c6408d563b7c73ff15a978"
    )
    assert len(snapshot_canonical_bytes((first,))) == 679


def test_snapshot_hash_retains_rank_order_and_canonical_evidence_fields() -> None:
    first = make_snapshot_item()
    second = make_snapshot_item(
        snapshot_item_id=UUID(int=ITEM_ID.int + 1),
        rank=2,
        summary="A second independent observation.",
    )
    assert hash_snapshot((first, second)) != hash_snapshot((second, first))
    assert hash_snapshot((first,)) != hash_snapshot(
        (first.model_copy(update={"excerpt": "Different frozen excerpt."}),)
    )
    assert hash_snapshot((first,)) != hash_snapshot(
        (
            first.model_copy(
                update={"falsifiers": ("No stale read is ever observable.",)}
            ),
        )
    )


def test_idea_content_hash_sorts_and_deduplicates_set_like_fields() -> None:
    first = make_idea()
    second = make_idea(
        predictions=("A", "B"),
        falsifiers=("F1", "F2", "F1"),
        assumptions=("S1", "S2"),
        evidence=(
            SnapshotEvidenceReference(
                id=UUID(int=ITEM_ID.int + 99),
                stable_evidence_key=make_snapshot_item().stable_evidence_key,
            ),
        ),
    )
    assert hash_idea_content(first) == hash_idea_content(second)


def test_idea_content_hash_changes_when_semantic_content_changes() -> None:
    idea = make_idea()
    changed = idea.model_copy(update={"hypothesis": "A different hypothesis."})
    assert hash_idea_content(idea) != hash_idea_content(changed)


def test_mechanism_normalization_is_nfkc_casefolded_and_boundary_aware() -> None:
    assert (
        normalize_mechanism("  ＣＡＣＨＥ—Invalidation！\nAfter COMMIT  ")
        == "cache invalidation after commit"
    )
    assert normalize_mechanism("Straße") == "strasse"
    assert hash_mechanism("CACHE-invalidation") == hash_mechanism(
        "cache invalidation"
    )
    with pytest.raises(ValueError):
        hash_mechanism("———")


def test_mechanism_similarity_uses_padded_character_trigram_jaccard() -> None:
    assert mechanism_similarity("ＣＡＣＨＥ-invalidation", "cache invalidation") == 1
    assert mechanism_similarity("ab", "abc") == pytest.approx(2 / 7)
    assert mechanism_similarity("ab", "xy") == 0


def valid_start(**changes: object) -> StartInspirationRun:
    values: dict[str, object] = {
        "owner_agent_id": OWNER_ID,
        "goal": "Find a falsifiable explanation for stale reads.",
        "context": "service=ledger",
        "mode": RetrievalMode.ASSOCIATIVE,
        "generator": GeneratorKind.DETERMINISTIC,
        "operators": INSPIRATION_OPERATOR_ORDER,
        "include_inbox": False,
    }
    values.update(changes)
    return StartInspirationRun(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "operators",
    (
        (InspirationOperator.CAUSAL_GAP,),
        (InspirationOperator.COUNTERFACTUAL,),
        (InspirationOperator.DISTANT_ANALOGY,),
        (
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        ),
        (
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.DISTANT_ANALOGY,
        ),
        (
            InspirationOperator.COUNTERFACTUAL,
            InspirationOperator.DISTANT_ANALOGY,
        ),
        INSPIRATION_OPERATOR_ORDER,
    ),
)
def test_start_run_accepts_nonempty_fixed_order_operator_subsets(
    operators: tuple[InspirationOperator, ...],
) -> None:
    assert valid_start(operators=operators).operators == operators


@pytest.mark.parametrize(
    "operators",
    (
        (),
        (
            InspirationOperator.COUNTERFACTUAL,
            InspirationOperator.CAUSAL_GAP,
        ),
        (
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.CAUSAL_GAP,
        ),
        [InspirationOperator.CAUSAL_GAP],
        ("causal_gap",),
    ),
)
def test_start_run_rejects_empty_noncanonical_or_non_strict_operators(
    operators: object,
) -> None:
    with pytest.raises(ValueError):
        valid_start(operators=operators)


def test_start_run_defaults_are_the_bounded_protocol_defaults() -> None:
    run = valid_start()
    assert (
        run.branches_per_operator,
        run.output_tokens_per_operator,
        run.total_output_tokens,
        run.operator_timeout_seconds,
        run.global_timeout_seconds,
    ) == (3, 1_200, 3_600, 30, 90)


@pytest.mark.parametrize(
    "changes",
    (
        {"branches_per_operator": 1},
        {"branches_per_operator": 3},
        {"output_tokens_per_operator": 1},
        {"output_tokens_per_operator": 1_200},
        {"total_output_tokens": 1},
        {"total_output_tokens": 3_600},
        {"operator_timeout_seconds": 1},
        {"operator_timeout_seconds": 30},
        {
            "operator_timeout_seconds": 30,
            "global_timeout_seconds": 30,
        },
        {
            "operator_timeout_seconds": 1,
            "global_timeout_seconds": 1,
        },
        {
            "output_tokens_per_operator": 1_200,
            "total_output_tokens": 1,
        },
        {"global_timeout_seconds": 90},
    ),
)
def test_start_run_accepts_every_budget_boundary(
    changes: dict[str, object],
) -> None:
    valid_start(**changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"branches_per_operator": 0},
        {"branches_per_operator": 4},
        {"branches_per_operator": True},
        {"branches_per_operator": 1.0},
        {"branches_per_operator": "1"},
        {"output_tokens_per_operator": 0},
        {"output_tokens_per_operator": 1_201},
        {"total_output_tokens": 0},
        {"total_output_tokens": 3_601},
        {"operator_timeout_seconds": 0},
        {"operator_timeout_seconds": 31},
        {"global_timeout_seconds": 0},
        {"global_timeout_seconds": 91},
        {
            "operator_timeout_seconds": 30,
            "global_timeout_seconds": 29,
        },
    ),
)
def test_start_run_rejects_out_of_range_or_incoherent_budgets(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        valid_start(**changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"owner_agent_id": str(OWNER_ID)},
        {"mode": "associative"},
        {"generator": "deterministic"},
        {"include_inbox": 1},
        {"goal": "   "},
        {"goal": "x" * 2_001},
        {"goal": " canonical but padded "},
        {"context": "x" * 4_001},
        {"context": " canonical but padded "},
        {"goal": "\ud800"},
        {"context": "\ud800"},
    ),
)
def test_start_run_rejects_non_strict_or_noncanonical_inputs(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        valid_start(**changes)


def test_start_run_is_frozen_and_slotted() -> None:
    run = valid_start()
    assert not hasattr(run, "__dict__")
    with pytest.raises(FrozenInstanceError):
        run.goal = "changed"  # type: ignore[misc]


def test_inspiration_models_are_strict_frozen_and_forbid_unknown_fields() -> None:
    reference = SnapshotEvidenceReference(
        id=ITEM_ID,
        stable_evidence_key=make_snapshot_item().stable_evidence_key,
    )
    assert (
        SnapshotEvidenceReference.model_json_schema()["additionalProperties"]
        is False
    )
    with pytest.raises(ValidationError):
        SnapshotEvidenceReference.model_validate(
            {
                **reference.model_dump(mode="python"),
                "id": str(ITEM_ID),
            }
        )
    with pytest.raises(ValidationError):
        IdeaDraft.model_validate(
            {
                **make_idea().model_dump(mode="python"),
                "unexpected": "field",
            }
        )


@pytest.mark.parametrize(
    "changes",
    (
        {
            "source_type": EvidenceSourceType.CAPSULE,
            "source_state": EvidenceSourceState.WARM,
            "source_trust": 0.25,
        },
        {
            "source_type": EvidenceSourceType.CAPSULE,
            "source_state": EvidenceSourceState.QUARANTINED,
            "source_trust": 0.5,
        },
        {
            "source_type": EvidenceSourceType.EXPERIENCE,
            "source_state": EvidenceSourceState.QUARANTINED,
            "source_trust": 1.0,
        },
    ),
)
def test_snapshot_items_reject_incoherent_source_state_or_trust(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SnapshotItem.model_validate(
            make_snapshot_item().model_copy(update=changes).model_dump(
                mode="python"
            )
        )


def test_evaluation_evidence_union_is_strict_nonempty_and_reason_optional() -> None:
    evaluation = IdeaEvaluation(
        evaluator_agent_id=OWNER_ID,
        idea_id=UUID(int=900),
        verdict=EvaluationVerdict.SUPPORTED,
        reason=None,
        evidence=(
            ExperienceVersionEvidenceReference(id=VERSION_ID),
        ),
        evaluated_at=NOW,
    )
    assert evaluation.evidence[0].type == "experience_version"
    with pytest.raises(ValidationError):
        IdeaEvaluation.model_validate(
            {
                **evaluation.model_dump(mode="python"),
                "evidence": (),
            }
        )


def test_idea_draft_rejects_a_mechanism_that_normalizes_to_empty() -> None:
    values = make_idea().model_dump(mode="python")
    values["mechanism"] = "———"
    with pytest.raises(ValidationError):
        IdeaDraft.model_validate(values)
