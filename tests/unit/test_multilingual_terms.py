from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest

from experience_hub.domain import TypedEvidence
from experience_hub.experiences import VersionContent
from experience_hub.retrieval.tokenizer import (
    MECHANISM_WEIGHT,
    TAG_WEIGHT,
    TRIGRAM_WEIGHT,
    WORD_WEIGHT,
    TermCue,
    index_version_terms,
    latin_words,
    normalize_text,
    padded_char_trigrams,
    query_cues,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  ＣＡＣＨＥ—Straße，记忆\t管理  ", "cache strasse 记忆 管理"),
        ("Cafe\u0301 / déjà-vu", "café déjà vu"),
        ("alpha_beta...gamma", "alpha beta gamma"),
        (" \t\n，。—_ ", ""),
        ("", ""),
    ],
)
def test_normalize_text_applies_nfkc_casefold_and_punctuation_boundaries(
    source: str,
    expected: str,
) -> None:
    assert normalize_text(source) == expected


def test_normalize_text_treats_control_characters_as_boundaries() -> None:
    assert normalize_text("alpha\x00beta\x07gamma") == "alpha beta gamma"


def test_normalize_text_preserves_non_punctuation_symbols() -> None:
    assert normalize_text("Alpha+Beta © 记忆") == "alpha+beta © 记忆"


def test_normalize_text_restores_nfkc_after_casefold_decomposition() -> None:
    composed = normalize_text("ǰ")
    decomposed = normalize_text("j\u030c")

    assert composed == decomposed == "ǰ"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Straße缓存 CAFÉ déjà-vu 123", ("strasse", "café", "déjà", "vu")),
        ("alphaΑθήναbeta", ("alpha", "beta")),
        ("记忆管理", ()),
        ("", ()),
    ],
)
def test_latin_words_keep_only_contiguous_latin_words(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert latin_words(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ()),
        ("a", ("  a", " a ", "a  ")),
        ("ab", ("  a", " ab", "ab ", "b  ")),
    ],
)
def test_padded_trigrams_cover_zero_one_and_two_character_boundaries(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert padded_char_trigrams(source) == expected


def test_padded_chinese_character_trigrams_have_two_boundary_spaces() -> None:
    assert padded_char_trigrams("记忆") == (
        "  记",
        " 记忆",
        "记忆 ",
        "忆  ",
    )


def test_padded_trigrams_normalize_mixed_scripts_and_punctuation() -> None:
    assert padded_char_trigrams("Ａ，记") == (
        "  a",
        " a ",
        "a 记",
        " 记 ",
        "记  ",
    )
    assert padded_char_trigrams("，。") == ()
    assert padded_char_trigrams("") == ()


def _cue_map(cues: Iterable[TermCue]) -> dict[tuple[str, str], float]:
    return {(cue.term, cue.term_kind): cue.weight for cue in cues}


@pytest.mark.parametrize(
    ("term", "term_kind", "weight"),
    [
        ("", "word", 1.0),
        ("memory", "unknown", 1.0),
        ("memory", "word", 0.0),
        ("memory", "word", -1.0),
        ("memory", "word", 1.5000001),
        ("memory", "word", float("nan")),
        ("memory", "word", float("inf")),
        ("memory", "word", True),
        (cast(Any, 7), "word", 1.0),
    ],
)
def test_term_cue_rejects_invalid_runtime_values(
    term: Any,
    term_kind: Any,
    weight: Any,
) -> None:
    with pytest.raises(ValueError):
        TermCue(
            term=term,
            term_kind=term_kind,
            weight=weight,
        )


def _content() -> VersionContent:
    return VersionContent(
        body="Cache cache 记忆",
        summary="CACHE!",
        mechanism="Lease-Handoff cache",
        tags=("Memory Ops", "CACHE"),
        applicability=("cache policies",),
        evidence=(),
        falsifiers=(),
    )


def test_index_version_terms_assigns_exact_field_weights() -> None:
    cues = index_version_terms(_content())
    terms = _cue_map(cues)

    assert terms[("cache", "tag")] == TAG_WEIGHT == 1.50
    assert terms[("cache", "mechanism")] == MECHANISM_WEIGHT == 1.25
    assert terms[("lease", "mechanism")] == MECHANISM_WEIGHT
    assert terms[("handoff", "mechanism")] == MECHANISM_WEIGHT
    assert terms[("cache", "word")] == WORD_WEIGHT == 1.00
    assert terms[("policies", "word")] == WORD_WEIGHT
    assert terms[("e 记", "char_trigram")] == TRIGRAM_WEIGHT == 0.35
    assert terms[(" 记忆", "char_trigram")] == TRIGRAM_WEIGHT
    assert terms[("memory ops", "tag")] == TAG_WEIGHT


def test_index_version_terms_deduplicates_by_kind_with_max_not_sum() -> None:
    cues = index_version_terms(_content())

    assert len(cues) == len({(cue.term, cue.term_kind) for cue in cues})
    assert _cue_map(cues)[("cache", "word")] == WORD_WEIGHT
    assert _cue_map(cues)[("cac", "char_trigram")] == TRIGRAM_WEIGHT
    assert cues == tuple(
        sorted(cues, key=lambda cue: (cue.term, cue.term_kind))
    )


def test_index_version_terms_keeps_unspaced_chinese_mechanism_as_one_term() -> None:
    content = VersionContent(
        body="。",
        summary="！",
        mechanism="租约交接",
        tags=(),
        applicability=(),
        evidence=(),
        falsifiers=(),
    )

    terms = _cue_map(index_version_terms(content))

    assert terms[("租约交接", "mechanism")] == MECHANISM_WEIGHT


def test_index_version_terms_ignores_evidence_and_falsifiers() -> None:
    content = VersionContent(
        body="body",
        summary="summary",
        mechanism="mechanism",
        tags=(),
        applicability=(),
        evidence=(TypedEvidence(type="ZephyrQx", id="VortxId"),),
        falsifiers=("PlughXyz",),
    )
    cues = index_version_terms(content)
    terms = _cue_map(cues)
    trigram_terms = {
        cue.term for cue in cues if cue.term_kind == "char_trigram"
    }

    assert ("zephyrqx", "word") not in terms
    assert ("vortxid", "word") not in terms
    assert ("plughxyz", "word") not in terms
    assert "zep" not in trigram_terms
    assert "vor" not in trigram_terms
    assert "plu" not in trigram_terms


def test_query_cues_index_optional_exact_tags_and_mechanism_tokens() -> None:
    cues = query_cues(
        "Cache cache 记忆",
        tags=("Memory-Ops", "CACHE"),
        mechanisms=("Lease-Handoff", "cache"),
    )
    terms = _cue_map(cues)

    assert terms[("cache", "word")] == WORD_WEIGHT
    assert terms[("cache", "tag")] == TAG_WEIGHT
    assert terms[("memory ops", "tag")] == TAG_WEIGHT
    assert terms[("cache", "mechanism")] == MECHANISM_WEIGHT
    assert terms[("lease", "mechanism")] == MECHANISM_WEIGHT
    assert terms[("handoff", "mechanism")] == MECHANISM_WEIGHT
    assert terms[("e 记", "char_trigram")] == TRIGRAM_WEIGHT
    assert len(cues) == len({(cue.term, cue.term_kind) for cue in cues})


def test_query_tag_and_mechanism_trigrams_do_not_cross_field_boundaries() -> None:
    cues = query_cues(
        "",
        tags=("AB",),
        mechanisms=("CD",),
    )
    terms = _cue_map(cues)
    trigram_terms = {
        cue.term for cue in cues if cue.term_kind == "char_trigram"
    }

    assert terms[("ab", "tag")] == TAG_WEIGHT
    assert terms[("cd", "mechanism")] == MECHANISM_WEIGHT
    assert trigram_terms == set(padded_char_trigrams("AB")) | set(
        padded_char_trigrams("CD")
    )
    assert "b c" not in trigram_terms


def test_empty_and_punctuation_only_inputs_produce_no_cues() -> None:
    assert query_cues("") == ()
    assert query_cues("，。—_", tags=("...",), mechanisms=("——",)) == ()
