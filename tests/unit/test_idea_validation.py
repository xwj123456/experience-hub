from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from experience_hub.inspiration.generators.base import OperatorFailureCode
from experience_hub.inspiration.hashing import stable_evidence_key
from experience_hub.inspiration.models import (
    EvidenceSourceState,
    EvidenceSourceType,
    IdeaDraft,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.validation import (
    ValidatedOperatorBatch,
    validate_operator_batch,
)

NOW = datetime(2026, 7, 18, 13, tzinfo=UTC)
RUN_ID = UUID("00000000-0000-0000-0000-000000001001")
OTHER_RUN_ID = UUID("00000000-0000-0000-0000-000000001002")


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def _hash(value: int) -> str:
    return f"{value:064x}"


def _item(rank: int, *, run_id: UUID = RUN_ID) -> SnapshotItem:
    source_id = _uuid(2_000 + rank)
    version_id = _uuid(3_000 + rank)
    content_hash = _hash(4_000 + rank)
    return SnapshotItem(
        snapshot_item_id=_uuid(5_000 + rank),
        stable_evidence_key=stable_evidence_key(
            source_type=EvidenceSourceType.EXPERIENCE,
            source_id=source_id,
            source_version_id=version_id,
            content_hash=content_hash,
        ),
        run_id=run_id,
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=source_id,
        source_version_id=version_id,
        source_state=EvidenceSourceState.WARM,
        source_trust=1.0,
        rank=rank,
        summary=f"Observation {rank}",
        mechanism=f"Mechanism {rank}",
        applicability=(f"Condition {rank}",),
        tags=("validation",),
        falsifiers=(f"Falsifier {rank}",),
        excerpt=f"Excerpt {rank}",
        content_hash=content_hash,
        captured_at=NOW,
    )


def _reference(item: SnapshotItem) -> SnapshotEvidenceReference:
    return SnapshotEvidenceReference(
        id=item.snapshot_item_id,
        stable_evidence_key=item.stable_evidence_key,
    )


def _draft(
    *items: SnapshotItem,
    mechanism: str = "A validated mechanism",
) -> IdeaDraft:
    return IdeaDraft(
        title="A validated branch",
        hypothesis="The branch predicts a bounded outcome.",
        mechanism=mechanism,
        predictions=("Prediction B", "Prediction A"),
        falsifiers=("Falsifier B", "Falsifier A"),
        assumptions=("Assumption B", "Assumption A"),
        proposed_test="Run a controlled comparison.",
        evidence=tuple(_reference(item) for item in items),
    )


def _validate(
    branches: tuple[object, ...],
    *,
    snapshot_items: tuple[SnapshotItem, ...],
) -> ValidatedOperatorBatch:
    return validate_operator_batch(
        run_id=RUN_ID,
        operator=InspirationOperator.CAUSAL_GAP,
        branches=branches,
        snapshot_items=snapshot_items,
        output_tokens_consumed=17,
    )


def test_valid_batch_is_strictly_canonicalized() -> None:
    first = _item(1)
    second = _item(2)
    draft = _draft(second, first)

    result = _validate((draft,), snapshot_items=(first, second))

    assert result.succeeded is True
    assert result.error_code is None
    assert result.output_tokens_consumed == 17
    assert result.ideas[0].predictions == ("Prediction A", "Prediction B")
    assert result.ideas[0].falsifiers == ("Falsifier A", "Falsifier B")
    assert result.ideas[0].assumptions == ("Assumption A", "Assumption B")
    assert result.ideas[0].evidence == (_reference(first), _reference(second))


@pytest.mark.parametrize("mutation", ("missing", "empty", "unknown"))
def test_structurally_invalid_branch_rejects_the_entire_batch(
    mutation: str,
) -> None:
    item = _item(1)
    raw = _draft(item).model_dump(mode="python")
    if mutation == "missing":
        del raw["hypothesis"]
    elif mutation == "empty":
        raw["predictions"] = ()
    else:
        raw["unapproved_field"] = "must be rejected"

    result = _validate((_draft(item), raw), snapshot_items=(item,))

    assert result.succeeded is False
    assert result.ideas == ()
    assert result.error_code is OperatorFailureCode.INVALID_PROVIDER_RESPONSE
    assert "hypothesis" not in repr(result)
    assert "unapproved_field" not in repr(result)


def test_empty_batch_is_a_fixed_no_valid_branches_failure() -> None:
    result = _validate((), snapshot_items=(_item(1),))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.NO_VALID_BRANCHES
    assert result.ideas == ()


def test_unknown_snapshot_item_rejects_the_entire_batch() -> None:
    item = _item(1)
    unknown = _draft(item).model_copy(
        update={
            "evidence": (
                SnapshotEvidenceReference(
                    id=_uuid(99_999),
                    stable_evidence_key=item.stable_evidence_key,
                ),
            )
        }
    )

    result = _validate((_draft(item), unknown), snapshot_items=(item,))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE
    assert result.ideas == ()


def test_foreign_run_snapshot_item_is_never_valid_evidence() -> None:
    foreign = _item(1, run_id=OTHER_RUN_ID)

    result = _validate((_draft(foreign),), snapshot_items=(foreign,))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


def test_unused_foreign_run_snapshot_item_rejects_the_snapshot_boundary() -> None:
    owned = _item(1)
    foreign = _item(2, run_id=OTHER_RUN_ID)

    result = _validate((_draft(owned),), snapshot_items=(owned, foreign))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


def test_snapshot_stable_key_is_recomputed_before_accepting_a_reference() -> None:
    item = _item(1)
    forged_key = _hash(99_998)
    forged = SnapshotItem.model_validate(
        {
            **item.model_dump(mode="python"),
            "stable_evidence_key": forged_key,
        },
        strict=True,
    )
    matching_forged_reference = _draft(forged)

    result = _validate(
        (matching_forged_reference,),
        snapshot_items=(forged,),
    )

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


@pytest.mark.parametrize(
    "snapshot_items",
    (
        (_item(2), _item(1)),
        (_item(1), _item(1).model_copy(update={"snapshot_item_id": _uuid(8_001)})),
    ),
)
def test_snapshot_boundary_requires_canonical_unique_ranks(
    snapshot_items: tuple[SnapshotItem, ...],
) -> None:
    result = _validate((_draft(snapshot_items[0]),), snapshot_items=snapshot_items)

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


def test_stable_key_mismatch_rejects_the_entire_batch() -> None:
    item = _item(1)
    mismatched = _draft(item).model_copy(
        update={
            "evidence": (
                SnapshotEvidenceReference(
                    id=item.snapshot_item_id,
                    stable_evidence_key=_hash(99_999),
                ),
            )
        }
    )

    result = _validate((mismatched,), snapshot_items=(item,))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


def test_repeated_evidence_rejects_the_entire_batch() -> None:
    item = _item(1)
    repeated = _draft(item).model_copy(
        update={"evidence": (_reference(item), _reference(item))}
    )

    result = _validate((repeated,), snapshot_items=(item,))

    assert result.succeeded is False
    assert result.error_code is OperatorFailureCode.INVALID_EVIDENCE_REFERENCE


def test_canonicalization_deduplicates_set_like_idea_lists() -> None:
    item = _item(1)
    repeated = _draft(item).model_copy(
        update={
            "predictions": ("Prediction B", "Prediction A", "Prediction B"),
            "falsifiers": ("Falsifier A", "Falsifier A"),
            "assumptions": ("Assumption B", "Assumption B", "Assumption A"),
        }
    )

    result = _validate((repeated,), snapshot_items=(item,))

    assert result.ideas[0].predictions == ("Prediction A", "Prediction B")
    assert result.ideas[0].falsifiers == ("Falsifier A",)
    assert result.ideas[0].assumptions == ("Assumption A", "Assumption B")


@pytest.mark.parametrize(
    "invalid",
    (
        ["not", "a", "tuple"],
        "not a tuple",
        None,
    ),
)
def test_batch_container_is_strict(invalid: object) -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        validate_operator_batch(
            run_id=RUN_ID,
            operator=InspirationOperator.CAUSAL_GAP,
            branches=invalid,
            snapshot_items=(_item(1),),
        )


def test_validated_batch_revalidates_model_copy_bypasses() -> None:
    invalid = _draft(_item(1)).model_copy(update={"mechanism": 42})

    with pytest.raises(ValueError, match="structurally valid"):
        ValidatedOperatorBatch(
            operator=InspirationOperator.CAUSAL_GAP,
            ideas=(invalid,),
        )
