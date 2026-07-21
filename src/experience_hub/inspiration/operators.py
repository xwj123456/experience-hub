"""Pure candidate selection for deterministic inspiration operators."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import jieba  # type: ignore[import-untyped]
import jieba.posseg as jieba_posseg  # type: ignore[import-untyped]

from experience_hub.canonical import canonical_json_bytes
from experience_hub.inspiration.models import SnapshotItem
from experience_hub.retrieval.tokenizer import normalize_text

_ENGLISH_NEGATORS = frozenset({"not", "no", "never", "without", "cannot"})
_CJK_NEGATORS = frozenset({"非", "不", "无", "未"})
_CJK_INSISTENCE_COMPLEMENTS = frozenset({"要", "得"})
_CJK_PRODUCTIVE_FUSED_FLAGS = frozenset({"b", "n"})
_CONFLICT_THRESHOLD_NUMERATOR = 13
_CONFLICT_THRESHOLD_DENOMINATOR = 20
_CJK_TOKENIZER = jieba.Tokenizer()
_CJK_POS_TOKENIZER = jieba_posseg.POSTokenizer(_CJK_TOKENIZER)


class ConflictBasis(StrEnum):
    """Closed explanations for why two frozen claims conflict."""

    EXPLICIT_NEGATOR = "explicit_negator_conflict"
    FALSIFIER = "falsifier_conflict"


@dataclass(frozen=True, slots=True)
class CausalPair:
    """One ordered frozen-evidence pair with a possible causal gap."""

    left: SnapshotItem
    right: SnapshotItem
    conflict_basis: ConflictBasis | None


@dataclass(frozen=True, slots=True)
class CounterfactualCandidate:
    """One explicit frozen applicability assumption to invert."""

    item: SnapshotItem
    applicability: str


@dataclass(frozen=True, slots=True)
class AnalogyPair:
    """One shared mechanism ordered from lower to higher evidence rank."""

    left: SnapshotItem
    right: SnapshotItem
    shared_terms: tuple[str, ...]
    lexical_jaccard: Fraction


def _lo_segments(value: str) -> tuple[str, ...]:
    segments: list[str] = []
    current: list[str] = []
    for character in value:
        if unicodedata.category(character) == "Lo":
            current.append(character)
            continue
        if current:
            segments.append("".join(current))
            current.clear()
    if current:
        segments.append("".join(current))
    return tuple(segments)


def operator_terms(value: str) -> frozenset[str]:
    """Return stable word and CJK bigram terms for operator comparisons."""
    normalized = normalize_text(value)
    terms: set[str] = set()
    for token in normalized.split():
        terms.add(token)
        for segment in _lo_segments(token):
            if len(segment) == 1:
                terms.add(segment)
                continue
            terms.update(
                segment[index : index + 2] for index in range(len(segment) - 1)
            )
    return frozenset(terms)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> Fraction:
    union = left | right
    if not union:
        return Fraction(0, 1)
    return Fraction(len(left & right), len(union))


def _meets_conflict_threshold(
    left: frozenset[str],
    right: frozenset[str],
) -> bool:
    union_size = len(left | right)
    if union_size == 0:
        return False
    intersection_size = len(left & right)
    return (
        intersection_size * _CONFLICT_THRESHOLD_DENOMINATOR
        >= union_size * _CONFLICT_THRESHOLD_NUMERATOR
    )


@lru_cache(maxsize=4_096)
def _lexical_units_with_flags(value: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (pair.word, pair.flag)
        for pair in _CJK_POS_TOKENIZER.cut(normalize_text(value), HMM=False)
        if pair.word.strip()
    )


def _lexical_units(value: str) -> tuple[str, ...]:
    return tuple(unit for unit, _flag in _lexical_units_with_flags(value))


@lru_cache(maxsize=4_096)
def _lexical_spans_with_flags(
    value: str,
) -> tuple[tuple[str, str, int, int], ...]:
    normalized = normalize_text(value)
    spans = tuple(
        (unit, start, end)
        for unit, start, end in _CJK_TOKENIZER.tokenize(
            normalized,
            mode="default",
            HMM=False,
        )
        if unit.strip()
    )
    tagged = _lexical_units_with_flags(normalized)
    if tuple(unit for unit, _start, _end in spans) != tuple(
        unit for unit, _flag in tagged
    ):
        raise RuntimeError("fixed Jieba token and POS boundaries disagree")
    return tuple(
        (unit, flag, start, end)
        for (unit, start, end), (_tagged_unit, flag) in zip(
            spans,
            tagged,
            strict=True,
        )
    )


def _is_insistence_negator(value: str, character_index: int) -> bool:
    spans = _lexical_spans_with_flags(value)
    for index, (unit, _flag, start, end) in enumerate(spans):
        if (
            unit == "非"
            and start == character_index
            and end == character_index + 1
            and index + 1 < len(spans)
            and spans[index + 1][0] in _CJK_INSISTENCE_COMPLEMENTS
            and len(spans[index + 1][0]) == 1
            and end == spans[index + 1][2]
        ):
            return True
    return False


def _is_productive_cjk_removal(
    source: str,
    character_index: int,
) -> bool:
    if _is_insistence_negator(source, character_index):
        return False
    spans = _lexical_spans_with_flags(source)
    for span_index, (unit, flag, start, end) in enumerate(spans):
        if not start <= character_index < end:
            continue
        negator = source[character_index]
        offset = character_index - start
        if unit == negator:
            return True
        retained_unit = unit[:offset] + unit[offset + 1 :]
        if (
            offset == 0
            and unit == f"{negator}{retained_unit}"
            and flag in _CJK_PRODUCTIVE_FUSED_FLAGS
            and len(retained_unit) >= 2
            and _lexical_units(retained_unit) == (retained_unit,)
        ):
            return True
        local_end = end
        if span_index + 1 < len(spans) and end == spans[span_index + 1][2]:
            local_end = spans[span_index + 1][3]
        local_source = source[start:local_end]
        local_retained = (
            local_source[:offset] + local_source[offset + 1 :]
        )
        return len(_lexical_units(local_source)) == (
            len(_lexical_units(local_retained)) + 1
        )
    return False


@lru_cache(maxsize=4_096)
def _has_explicit_negator(value: str) -> bool:
    normalized = normalize_text(value)
    if any(token in _ENGLISH_NEGATORS for token in normalized.split()):
        return True
    return any(
        character in _CJK_NEGATORS
        and _is_productive_cjk_removal(normalized, character_index)
        for character_index, character in enumerate(normalized)
    )


def _single_character_removal_index(
    source: str,
    target: str,
) -> int | None:
    if len(source) != len(target) + 1:
        return None
    index = 0
    while index < len(target) and source[index] == target[index]:
        index += 1
    if source[index + 1 :] != target[index:]:
        return None
    return index


def _single_token_removal_index(
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> int | None:
    if len(source) != len(target) + 1:
        return None
    index = 0
    while index < len(target) and source[index] == target[index]:
        index += 1
    if source[index + 1 :] != target[index:]:
        return None
    return index


@lru_cache(maxsize=16_384)
def _matches_single_negator_removal(source: str, target: str) -> bool:
    normalized_source = normalize_text(source)
    normalized_target = normalize_text(target)
    target_terms = operator_terms(normalized_target)
    tokens = tuple(normalized_source.split())
    for negator in _ENGLISH_NEGATORS:
        if negator in tokens:
            token_index = tokens.index(negator)
            retained = operator_terms(
                " ".join((*tokens[:token_index], *tokens[token_index + 1 :]))
            )
            if retained == target_terms:
                return True
    standalone_token_index = _single_token_removal_index(
        tokens,
        tuple(normalized_target.split()),
    )
    if (
        standalone_token_index is not None
        and tokens[standalone_token_index] in _CJK_NEGATORS
    ):
        return True
    character_index = _single_character_removal_index(
        normalized_source,
        normalized_target,
    )
    return (
        character_index is not None
        and normalized_source[character_index] in _CJK_NEGATORS
        and _is_productive_cjk_removal(
            normalized_source,
            character_index,
        )
    )


def _explicit_negator_conflict(left: str, right: str) -> bool:
    left_terms = operator_terms(left)
    right_terms = operator_terms(right)
    if not left_terms or not right_terms:
        return False
    left_explicit = _has_explicit_negator(left)
    right_explicit = _has_explicit_negator(right)
    if left_explicit and right_explicit:
        return False
    left_matches = _matches_single_negator_removal(left, right)
    right_matches = _matches_single_negator_removal(right, left)
    left_negated = left_explicit or left_matches
    right_negated = right_explicit or right_matches
    if left_negated is right_negated:
        return False
    return left_matches if left_negated else right_matches


def _falsifier_conflicts(
    falsifiers: tuple[str, ...],
    *,
    mechanism: str,
    summary: str,
) -> bool:
    targets = (operator_terms(mechanism), operator_terms(summary))
    return any(
        _meets_conflict_threshold(operator_terms(falsifier), target)
        for falsifier in falsifiers
        for target in targets
    )


def evidence_conflict_basis(
    left: SnapshotItem,
    right: SnapshotItem,
) -> ConflictBasis | None:
    """Explain a conflict under closed, replayable lexical rules."""
    if not isinstance(left, SnapshotItem) or not isinstance(right, SnapshotItem):
        raise TypeError("evidence_conflicts requires SnapshotItem values")
    if _explicit_negator_conflict(left.mechanism, right.mechanism):
        return ConflictBasis.EXPLICIT_NEGATOR
    if _falsifier_conflicts(
        left.falsifiers,
        mechanism=right.mechanism,
        summary=right.summary,
    ) or _falsifier_conflicts(
        right.falsifiers,
        mechanism=left.mechanism,
        summary=left.summary,
    ):
        return ConflictBasis.FALSIFIER
    return None


def evidence_conflicts(left: SnapshotItem, right: SnapshotItem) -> bool:
    """Return whether two frozen claims conflict under closed lexical rules."""
    return evidence_conflict_basis(left, right) is not None


def _ordered_items(items: tuple[SnapshotItem, ...]) -> tuple[SnapshotItem, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.rank,
                item.stable_evidence_key,
                item.snapshot_item_id.bytes,
            ),
        )
    )


def causal_gap_pairs(
    items: tuple[SnapshotItem, ...],
) -> tuple[CausalPair, ...]:
    """Select adjacent distinct mechanisms plus every conflicting pair."""
    ordered = _ordered_items(items)
    selected: dict[tuple[bytes, bytes], CausalPair] = {}

    def retain(
        left: SnapshotItem,
        right: SnapshotItem,
        *,
        conflict_basis: ConflictBasis | None,
    ) -> None:
        selected[(left.snapshot_item_id.bytes, right.snapshot_item_id.bytes)] = (
            CausalPair(
                left=left,
                right=right,
                conflict_basis=conflict_basis,
            )
        )

    for left, right in zip(ordered, ordered[1:], strict=False):
        if normalize_text(left.mechanism) != normalize_text(right.mechanism):
            retain(left, right, conflict_basis=None)
    for left, right in combinations(ordered, 2):
        conflict_basis = evidence_conflict_basis(left, right)
        if conflict_basis is not None:
            retain(left, right, conflict_basis=conflict_basis)

    return tuple(
        sorted(
            selected.values(),
            key=lambda pair: (
                pair.left.rank,
                pair.right.rank,
                pair.left.stable_evidence_key,
                pair.right.stable_evidence_key,
            ),
        )
    )


def _canonical_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    unique = {canonical_json_bytes(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def counterfactual_candidates(
    items: tuple[SnapshotItem, ...],
) -> tuple[CounterfactualCandidate, ...]:
    """Expand explicit assumptions in rank then canonical byte order."""
    candidates = [
        CounterfactualCandidate(item=item, applicability=applicability)
        for item in _ordered_items(items)
        for applicability in _canonical_strings(item.applicability)
        if normalize_text(applicability)
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.item.rank,
                canonical_json_bytes(candidate.applicability),
                candidate.item.stable_evidence_key,
            ),
        )
    )


def _lexical_terms(item: SnapshotItem) -> frozenset[str]:
    return operator_terms(" ".join((item.summary, *item.applicability, *item.tags)))


def distant_analogy_pairs(
    items: tuple[SnapshotItem, ...],
) -> tuple[AnalogyPair, ...]:
    """Order shared mechanisms by lexical distance, then stable evidence rank."""
    candidates: list[AnalogyPair] = []
    for left, right in combinations(_ordered_items(items), 2):
        shared = operator_terms(left.mechanism) & operator_terms(right.mechanism)
        if not shared:
            continue
        left_lexical = _lexical_terms(left)
        right_lexical = _lexical_terms(right)
        if not left_lexical and not right_lexical:
            continue
        candidates.append(
            AnalogyPair(
                left=left,
                right=right,
                shared_terms=tuple(sorted(shared, key=canonical_json_bytes)),
                lexical_jaccard=_jaccard(left_lexical, right_lexical),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.lexical_jaccard,
                candidate.left.rank,
                candidate.right.rank,
                candidate.left.stable_evidence_key,
                candidate.right.stable_evidence_key,
            ),
        )
    )


__all__ = [
    "AnalogyPair",
    "CausalPair",
    "ConflictBasis",
    "CounterfactualCandidate",
    "causal_gap_pairs",
    "counterfactual_candidates",
    "distant_analogy_pairs",
    "evidence_conflict_basis",
    "evidence_conflicts",
    "operator_terms",
]
