"""Pure, deterministic multilingual term generation."""

from __future__ import annotations

import logging
import math
import unicodedata
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

with warnings.catch_warnings():
    # Python 3.12 reports known escape literals in the pinned jieba sources on
    # a cold import. Limit suppression to those sources and that warning; other
    # diagnostics remain visible, and CLI responses do not reveal install paths.
    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence.*",
        category=SyntaxWarning,
        module=r".*[\\/]jieba[\\/].*|jieba([.].*)?",
    )
    import jieba  # type: ignore[import-untyped]

TermKind = Literal["word", "char_trigram", "tag", "mechanism"]
_TERM_KINDS = frozenset({"word", "char_trigram", "tag", "mechanism"})

TAG_WEIGHT: Final = 1.50
MECHANISM_WEIGHT: Final = 1.25
WORD_WEIGHT: Final = 1.00
TRIGRAM_WEIGHT: Final = 0.35

_APPLICABILITY_WORD_WEIGHT: Final = 0.45
_APPLICABILITY_TRIGRAM_WEIGHT: Final = 0.16
_RECALL_WORD_WEIGHT: Final = 0.35
_RECALL_TRIGRAM_WEIGHT: Final = 0.12

jieba.setLogLevel(logging.WARNING)
_CJK_TOKENIZER = jieba.Tokenizer()


class VersionTermSource(Protocol):
    """Text fields required to index one canonical experience version."""

    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]


class RankingTermSource(Protocol):
    """Bounded metadata available before any experience body is expanded."""

    @property
    def summary(self) -> str: ...

    @property
    def mechanism(self) -> str: ...

    @property
    def tags(self) -> tuple[str, ...]: ...

    @property
    def applicability(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class TermCue:
    """One weighted normalized term in a search or version cue map."""

    term: str
    term_kind: TermKind
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.term, str) or not self.term:
            raise ValueError("Term must be a non-empty string")
        if not isinstance(self.term_kind, str) or self.term_kind not in _TERM_KINDS:
            raise ValueError("Term kind is not supported")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise ValueError("Term weight must be a finite positive number")
        weight = float(self.weight)
        if not math.isfinite(weight) or not 0.0 < weight <= TAG_WEIGHT:
            raise ValueError("Term weight must be greater than zero and at most 1.5")
        object.__setattr__(self, "weight", weight)


def normalize_text(value: str) -> str:
    """Apply closed NFKC case folding, boundaries, and space collapse."""
    normalized = unicodedata.normalize("NFKC", value)
    folded = unicodedata.normalize("NFKC", normalized.casefold())
    boundary_aware = "".join(
        " "
        if character.isspace()
        or (category := unicodedata.category(character)).startswith("P")
        or category == "Cc"
        else character
        for character in folded
    )
    return " ".join(boundary_aware.split())


def _is_latin_letter(character: str) -> bool:
    return unicodedata.category(character).startswith(
        "L"
    ) and "LATIN" in unicodedata.name(character, "")


def latin_words(value: str) -> tuple[str, ...]:
    """Return normalized contiguous Unicode Latin-script words."""
    words: list[str] = []
    current: list[str] = []
    for character in normalize_text(value):
        if _is_latin_letter(character):
            current.append(character)
            continue
        if unicodedata.category(character).startswith("M") and current:
            current.append(character)
            continue
        if current:
            words.append("".join(current))
            current.clear()
    if current:
        words.append("".join(current))
    return tuple(words)


def _contains_other_letter(value: str) -> bool:
    return any(unicodedata.category(character) == "Lo" for character in value)


def _cjk_words(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for item in _CJK_TOKENIZER.cut(normalize_text(value), HMM=False)
        if (token := item.strip()) and len(token) >= 2 and _contains_other_letter(token)
    )


def padded_char_trigrams(value: str) -> tuple[str, ...]:
    """Return Unicode character trigrams with two spaces at each boundary."""
    normalized = normalize_text(value)
    if not normalized:
        return ()
    padded = f"  {normalized}  "
    return tuple(padded[index : index + 3] for index in range(len(padded) - 2))


def _add_cue(
    terms: dict[tuple[str, TermKind], float],
    *,
    term: str,
    term_kind: TermKind,
    weight: float,
) -> None:
    if not term:
        return
    key = (term, term_kind)
    terms[key] = max(weight, terms.get(key, 0.0))


def _add_trigrams(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        for trigram in padded_char_trigrams(value):
            _add_cue(
                terms,
                term=trigram,
                term_kind="char_trigram",
                weight=TRIGRAM_WEIGHT,
            )


def _add_words(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        for word in latin_words(value):
            _add_cue(
                terms,
                term=word,
                term_kind="word",
                weight=WORD_WEIGHT,
            )


def _add_cjk_words(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        for token in _cjk_words(value):
            _add_cue(
                terms,
                term=token,
                term_kind="word",
                weight=WORD_WEIGHT,
            )


def _add_weighted_ranking_words(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
    *,
    weight: float,
) -> None:
    for value in values:
        for word in latin_words(value):
            _add_cue(
                terms,
                term=word,
                term_kind="word",
                weight=weight,
            )
        for token in _cjk_words(value):
            _add_cue(
                terms,
                term=token,
                term_kind="word",
                weight=weight,
            )


def _add_weighted_trigrams(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
    *,
    weight: float,
) -> None:
    for value in values:
        for trigram in padded_char_trigrams(value):
            _add_cue(
                terms,
                term=trigram,
                term_kind="char_trigram",
                weight=weight,
            )


def _add_ranking_mechanisms(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        for token in (*latin_words(value), *_cjk_words(value)):
            _add_cue(
                terms,
                term=token,
                term_kind="mechanism",
                weight=MECHANISM_WEIGHT,
            )


def _add_tags(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        normalized = normalize_text(value)
        _add_cue(
            terms,
            term=normalized,
            term_kind="tag",
            weight=TAG_WEIGHT,
        )


def _add_mechanisms(
    terms: dict[tuple[str, TermKind], float],
    values: Iterable[str],
) -> None:
    for value in values:
        for token in normalize_text(value).split():
            _add_cue(
                terms,
                term=token,
                term_kind="mechanism",
                weight=MECHANISM_WEIGHT,
            )


def _sorted_cues(
    terms: dict[tuple[str, TermKind], float],
) -> tuple[TermCue, ...]:
    return tuple(
        TermCue(term=term, term_kind=term_kind, weight=weight)
        for (term, term_kind), weight in sorted(terms.items())
    )


def index_version_terms(content: VersionTermSource) -> tuple[TermCue, ...]:
    """Build the complete deterministic term projection for one version."""
    terms: dict[tuple[str, TermKind], float] = {}
    general_values = (content.body, content.summary, *content.applicability)
    _add_words(terms, general_values)
    _add_tags(terms, content.tags)
    _add_mechanisms(terms, (content.mechanism,))
    _add_trigrams(
        terms,
        (*general_values, *content.tags, content.mechanism),
    )
    return _sorted_cues(terms)


def ranking_version_terms(
    content: RankingTermSource,
    *,
    recall_terms: Sequence[TermCue] = (),
) -> tuple[TermCue, ...]:
    """Build field-aware terms from metadata available before body expansion."""
    terms: dict[tuple[str, TermKind], float] = {}
    _add_weighted_ranking_words(
        terms,
        (content.summary,),
        weight=WORD_WEIGHT,
    )
    _add_weighted_ranking_words(
        terms,
        content.applicability,
        weight=_APPLICABILITY_WORD_WEIGHT,
    )
    _add_tags(terms, content.tags)
    _add_mechanisms(terms, (content.mechanism,))
    _add_ranking_mechanisms(terms, (content.mechanism,))
    _add_weighted_trigrams(
        terms,
        (content.summary, *content.tags, content.mechanism),
        weight=TRIGRAM_WEIGHT,
    )
    _add_weighted_trigrams(
        terms,
        content.applicability,
        weight=_APPLICABILITY_TRIGRAM_WEIGHT,
    )
    fallback_weights: dict[TermKind, float] = {
        "word": _RECALL_WORD_WEIGHT,
        "char_trigram": _RECALL_TRIGRAM_WEIGHT,
        "tag": TAG_WEIGHT,
        "mechanism": MECHANISM_WEIGHT,
    }
    for cue in recall_terms:
        if not isinstance(cue, TermCue):
            raise ValueError("recall_terms must contain only TermCue values")
        _add_cue(
            terms,
            term=cue.term,
            term_kind=cue.term_kind,
            weight=min(cue.weight, fallback_weights[cue.term_kind]),
        )
    return _sorted_cues(terms)


def query_cues(
    text: str,
    *,
    tags: Sequence[str] = (),
    mechanisms: Sequence[str] = (),
) -> tuple[TermCue, ...]:
    """Build normalized weighted cues for one retrieval request."""
    terms: dict[tuple[str, TermKind], float] = {}
    _add_words(terms, (text,))
    _add_tags(terms, tags)
    _add_mechanisms(terms, mechanisms)
    _add_trigrams(terms, (text, *tags, *mechanisms))
    return _sorted_cues(terms)


def ranking_query_cues(
    text: str,
    *,
    tags: Sequence[str] = (),
    mechanisms: Sequence[str] = (),
) -> tuple[TermCue, ...]:
    """Build richer query cues for field-aware active-memory ranking."""
    terms: dict[tuple[str, TermKind], float] = {}
    _add_words(terms, (text,))
    _add_cjk_words(terms, (text,))
    _add_tags(terms, tags)
    _add_mechanisms(terms, mechanisms)
    _add_ranking_mechanisms(terms, mechanisms)
    _add_trigrams(terms, (text, *tags, *mechanisms))
    return _sorted_cues(terms)


__all__ = [
    "MECHANISM_WEIGHT",
    "TAG_WEIGHT",
    "TRIGRAM_WEIGHT",
    "WORD_WEIGHT",
    "TermCue",
    "TermKind",
    "VersionTermSource",
    "index_version_terms",
    "latin_words",
    "normalize_text",
    "padded_char_trigrams",
    "query_cues",
    "ranking_query_cues",
    "ranking_version_terms",
]
