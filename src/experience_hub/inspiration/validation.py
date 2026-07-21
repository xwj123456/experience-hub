"""Pure, batch-atomic validation for generated inspiration branches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from experience_hub import canonical_json_bytes
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.hashing import stable_evidence_key
from experience_hub.inspiration.models import (
    IdeaDraft,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)


def _strictly_revalidate_idea(idea: IdeaDraft) -> IdeaDraft:
    return IdeaDraft.model_validate(
        idea.model_dump(mode="python", warnings=False),
        strict=True,
    )


@dataclass(frozen=True, slots=True)
class ValidatedOperatorBatch:
    """A complete valid batch or one sanitized, batch-wide failure."""

    operator: InspirationOperator
    ideas: tuple[IdeaDraft, ...]
    error_code: OperatorFailureCode | None = None
    output_tokens_consumed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.operator, InspirationOperator):
            raise TypeError("operator must be an InspirationOperator")
        if not isinstance(self.ideas, tuple) or any(
            not isinstance(idea, IdeaDraft) for idea in self.ideas
        ):
            raise TypeError("ideas must be an immutable tuple of IdeaDraft values")
        try:
            for idea in self.ideas:
                _strictly_revalidate_idea(idea)
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(
                "ideas must contain structurally valid IdeaDraft values"
            ) from error
        if self.error_code is not None and not isinstance(
            self.error_code,
            OperatorFailureCode,
        ):
            raise TypeError("error_code must be an OperatorFailureCode or None")
        if self.error_code is None and not self.ideas:
            raise ValueError("a successful validated batch must contain ideas")
        if self.error_code is not None and self.ideas:
            raise ValueError("a failed validated batch cannot contain ideas")
        if (
            isinstance(self.output_tokens_consumed, bool)
            or not isinstance(self.output_tokens_consumed, int)
            or not 0 <= self.output_tokens_consumed <= 1_200
        ):
            raise ValueError(
                "output_tokens_consumed must be a strict integer from 0 to 1200"
            )

    @property
    def succeeded(self) -> bool:
        """Whether the complete batch passed structural and evidence checks."""
        return self.error_code is None


def _failure(
    *,
    operator: InspirationOperator,
    code: OperatorFailureCode,
    output_tokens_consumed: int,
) -> ValidatedOperatorBatch:
    return ValidatedOperatorBatch(
        operator=operator,
        ideas=(),
        error_code=code,
        output_tokens_consumed=output_tokens_consumed,
    )


def _canonical_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    by_encoding = {canonical_json_bytes(value): value for value in values}
    return tuple(by_encoding[key] for key in sorted(by_encoding))


def _parse_branch(branch: object) -> IdeaDraft:
    if isinstance(branch, IdeaDraft):
        value: Any = branch.model_dump(mode="python", warnings=False)
    elif isinstance(branch, Mapping):
        value = dict(branch)
    else:
        raise TypeError("branch must be an IdeaDraft or mapping")
    return IdeaDraft.model_validate(value, strict=True)


def _validated_snapshot_boundary(
    *,
    run_id: UUID,
    items: tuple[SnapshotItem, ...],
) -> tuple[SnapshotItem, ...] | None:
    try:
        validated = tuple(
            SnapshotItem.model_validate(
                item.model_dump(mode="python", warnings=False),
                strict=True,
            )
            for item in items
        )
    except (TypeError, ValueError, ValidationError):
        return None
    if any(item.run_id != run_id for item in validated):
        return None
    if tuple(item.rank for item in validated) != tuple(
        range(1, len(validated) + 1)
    ):
        return None
    if len({item.snapshot_item_id for item in validated}) != len(validated):
        return None
    if len({item.stable_evidence_key for item in validated}) != len(validated):
        return None
    source_identities = {
        (item.source_type, item.source_id, item.source_version_id)
        for item in validated
    }
    if len(source_identities) != len(validated):
        return None
    if any(
        item.stable_evidence_key
        != stable_evidence_key(
            source_type=item.source_type,
            source_id=item.source_id,
            source_version_id=item.source_version_id,
            content_hash=item.content_hash,
        )
        for item in validated
    ):
        return None
    return validated


def _canonicalize_branch(
    branch: IdeaDraft,
    *,
    snapshot_by_id: dict[UUID, SnapshotItem],
) -> IdeaDraft:
    evidence = tuple(
        sorted(
            branch.evidence,
            key=lambda reference: (
                snapshot_by_id[reference.id].rank,
                reference.stable_evidence_key,
                reference.id.bytes,
            ),
        )
    )
    return IdeaDraft(
        title=branch.title,
        hypothesis=branch.hypothesis,
        mechanism=branch.mechanism,
        predictions=_canonical_strings(branch.predictions),
        falsifiers=_canonical_strings(branch.falsifiers),
        assumptions=_canonical_strings(branch.assumptions),
        proposed_test=branch.proposed_test,
        evidence=evidence,
    )


def _has_valid_evidence(
    branch: IdeaDraft,
    *,
    run_id: UUID,
    snapshot_by_id: dict[UUID, SnapshotItem],
) -> bool:
    ids: set[UUID] = set()
    stable_keys: set[str] = set()
    for reference in branch.evidence:
        if not isinstance(reference, SnapshotEvidenceReference):
            return False
        if reference.id in ids or reference.stable_evidence_key in stable_keys:
            return False
        ids.add(reference.id)
        stable_keys.add(reference.stable_evidence_key)
        item = snapshot_by_id.get(reference.id)
        if (
            item is None
            or item.run_id != run_id
            or item.stable_evidence_key != reference.stable_evidence_key
        ):
            return False
    return True


def _validate_boundary_inputs(
    *,
    run_id: object,
    operator: object,
    branches: object,
    snapshot_items: object,
    output_tokens_consumed: object,
) -> tuple[
    UUID,
    InspirationOperator,
    tuple[object, ...],
    tuple[SnapshotItem, ...],
    int,
]:
    if not isinstance(run_id, UUID):
        raise TypeError("run_id must be a UUID")
    if not isinstance(operator, InspirationOperator):
        raise TypeError("operator must be an InspirationOperator")
    if not isinstance(branches, tuple):
        raise TypeError("branches must be an immutable tuple")
    if not isinstance(snapshot_items, tuple) or any(
        not isinstance(item, SnapshotItem) for item in snapshot_items
    ):
        raise TypeError(
            "snapshot_items must be an immutable tuple of SnapshotItem values"
        )
    if (
        isinstance(output_tokens_consumed, bool)
        or not isinstance(output_tokens_consumed, int)
        or not 0 <= output_tokens_consumed <= 1_200
    ):
        raise ValueError(
            "output_tokens_consumed must be a strict integer from 0 to 1200"
        )
    return (
        run_id,
        operator,
        branches,
        snapshot_items,
        output_tokens_consumed,
    )


def validate_operator_batch(
    *,
    run_id: object,
    operator: object,
    branches: object,
    snapshot_items: object,
    output_tokens_consumed: object = 0,
) -> ValidatedOperatorBatch:
    """Validate all branches atomically against one frozen run snapshot."""
    (
        retained_run_id,
        retained_operator,
        retained_branches,
        retained_items,
        retained_tokens,
    ) = _validate_boundary_inputs(
        run_id=run_id,
        operator=operator,
        branches=branches,
        snapshot_items=snapshot_items,
        output_tokens_consumed=output_tokens_consumed,
    )
    if not retained_branches:
        return _failure(
            operator=retained_operator,
            code=OperatorFailureCode.NO_VALID_BRANCHES,
            output_tokens_consumed=retained_tokens,
        )

    try:
        parsed = tuple(_parse_branch(branch) for branch in retained_branches)
    except (TypeError, ValueError, ValidationError):
        return _failure(
            operator=retained_operator,
            code=OperatorFailureCode.INVALID_PROVIDER_RESPONSE,
            output_tokens_consumed=retained_tokens,
        )

    validated_items = _validated_snapshot_boundary(
        run_id=retained_run_id,
        items=retained_items,
    )
    if validated_items is None:
        return _failure(
            operator=retained_operator,
            code=OperatorFailureCode.INVALID_EVIDENCE_REFERENCE,
            output_tokens_consumed=retained_tokens,
        )
    snapshot_by_id = {
        item.snapshot_item_id: item for item in validated_items
    }
    if any(
        not _has_valid_evidence(
            branch,
            run_id=retained_run_id,
            snapshot_by_id=snapshot_by_id,
        )
        for branch in parsed
    ):
        return _failure(
            operator=retained_operator,
            code=OperatorFailureCode.INVALID_EVIDENCE_REFERENCE,
            output_tokens_consumed=retained_tokens,
        )

    return ValidatedOperatorBatch(
        operator=retained_operator,
        ideas=tuple(
            _canonicalize_branch(branch, snapshot_by_id=snapshot_by_id)
            for branch in parsed
        ),
        output_tokens_consumed=retained_tokens,
    )


__all__ = ["ValidatedOperatorBatch", "validate_operator_batch"]
