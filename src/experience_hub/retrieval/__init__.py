"""Public multilingual retrieval queries, results, and services."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from experience_hub.retrieval.tokenizer import (
    MECHANISM_WEIGHT,
    TAG_WEIGHT,
    TRIGRAM_WEIGHT,
    WORD_WEIGHT,
    TermCue,
    TermKind,
    index_version_terms,
    latin_words,
    normalize_text,
    padded_char_trigrams,
    query_cues,
)

if TYPE_CHECKING:
    from experience_hub.retrieval.contracts import (
        CandidateSelection,
        ExperienceView,
        PeekExperiences,
        RetrievalCandidate,
        RetrievalRecord,
        SearchExperiences,
        SearchHit,
        SearchResult,
    )
    from experience_hub.retrieval.ranking import RetrievalMode
    from experience_hub.retrieval.service import (
        ASSOCIATIVE_COLD_EXPANSION_THRESHOLD,
        FOCUSED_COLD_EXPANSION_THRESHOLD,
        ExperienceEvidenceReader,
        RetrievalService,
        retrieval_query_hash,
    )

_LAZY_CONTRACT_EXPORTS = frozenset(
    {
        "CandidateSelection",
        "ExperienceView",
        "PeekExperiences",
        "RetrievalCandidate",
        "RetrievalRecord",
        "SearchExperiences",
        "SearchHit",
        "SearchResult",
    }
)
_LAZY_RANKING_EXPORTS = frozenset({"RetrievalMode"})
_LAZY_SERVICE_EXPORTS = frozenset(
    {
        "ASSOCIATIVE_COLD_EXPANSION_THRESHOLD",
        "FOCUSED_COLD_EXPANSION_THRESHOLD",
        "ExperienceEvidenceReader",
        "RetrievalService",
        "retrieval_query_hash",
    }
)
_LAZY_EXPORT_MODULES = {
    **{
        name: "experience_hub.retrieval.contracts"
        for name in _LAZY_CONTRACT_EXPORTS
    },
    **{
        name: "experience_hub.retrieval.ranking"
        for name in _LAZY_RANKING_EXPORTS
    },
    **{
        name: "experience_hub.retrieval.service"
        for name in _LAZY_SERVICE_EXPORTS
    },
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ASSOCIATIVE_COLD_EXPANSION_THRESHOLD",
    "FOCUSED_COLD_EXPANSION_THRESHOLD",
    "MECHANISM_WEIGHT",
    "TAG_WEIGHT",
    "TRIGRAM_WEIGHT",
    "WORD_WEIGHT",
    "CandidateSelection",
    "ExperienceEvidenceReader",
    "ExperienceView",
    "PeekExperiences",
    "RetrievalCandidate",
    "RetrievalMode",
    "RetrievalRecord",
    "RetrievalService",
    "SearchExperiences",
    "SearchHit",
    "SearchResult",
    "TermCue",
    "TermKind",
    "index_version_terms",
    "latin_words",
    "normalize_text",
    "padded_char_trigrams",
    "query_cues",
    "retrieval_query_hash",
]
