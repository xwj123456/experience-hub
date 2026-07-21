"""Pure, deterministic multilingual term generation."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

TermKind = Literal["word", "char_trigram", "tag", "mechanism"]
_TERM_KINDS = frozenset({"word", "char_trigram", "tag", "mechanism"})

TAG_WEIGHT: Final = 1.50
MECHANISM_WEIGHT: Final = 1.25
WORD_WEIGHT: Final = 1.00
TRIGRAM_WEIGHT: Final = 0.35


class VersionTermSource(Protocol):
    """Text fields required to index one canonical experience version."""

    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TermCue:
    """One weighted normalized term in a search or version cue map."""

    term: str
    term_kind: TermKind
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.term, str) or not self.term:
            raise ValueError("Term must be a non-empty string")
        if (
            not isinstance(self.term_kind, str)
            or self.term_kind not in _TERM_KINDS
        ):
            raise ValueError("Term kind is not supported")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
        ):
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
    return (
        unicodedata.category(character).startswith("L")
        and "LATIN" in unicodedata.name(character, "")
    )


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
]
