"""Strict public request schemas for experience memory operations."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import StrictModel, StructuredReason, TypedEvidence
from experience_hub.experiences.contracts import (
    CreateExperience,
    CreateExperienceVersion,
    VersionLinkInput,
)
from experience_hub.experiences.models import (
    MAX_BODY_UTF8_BYTES,
    MAX_MECHANISM_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    MAX_VERSION_LIST_ITEMS,
    ExperienceKind,
    LinkRelation,
    VersionContent,
)
from experience_hub.retrieval.contracts import (
    DEFAULT_CONTENT_BUDGET_BYTES,
    DEFAULT_SEARCH_LIMIT,
    MAX_CONTENT_BUDGET_BYTES,
    MAX_QUERY_CHARACTERS,
    SearchExperiences,
)
from experience_hub.retrieval.ranking import MAX_RETRIEVAL_LIMIT, RetrievalMode

UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
SearchLimit = Annotated[StrictInt, Field(ge=1, le=MAX_RETRIEVAL_LIMIT)]
ContentBudget = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_CONTENT_BUDGET_BYTES),
]


class _RequestModel(StrictModel):
    """Transport base that rejects unknown and non-finite input values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _enforce_list_limit(values: Any) -> Any:
    """Bound raw input before any de-duplication can reduce its size."""
    if isinstance(values, (list, tuple)) and len(values) > MAX_VERSION_LIST_ITEMS:
        raise ValueError(
            f"Lists may contain at most {MAX_VERSION_LIST_ITEMS} input items"
        )
    return values


def _canonical_tuple[T](values: tuple[T, ...]) -> tuple[T, ...]:
    by_bytes = {canonical_json_bytes(value): value for value in values}
    return tuple(by_bytes[key] for key in sorted(by_bytes))


def _validate_unicode(value: str, *, field_name: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must contain valid Unicode") from error
    return value


def _canonical_content_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        _validate_unicode(value, field_name="List values")
        if not value.strip():
            raise ValueError("List values must not be blank")
    return _canonical_tuple(values)


def _canonical_query_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    retained: list[str] = []
    for value in values:
        normalized = value.strip()
        _validate_unicode(normalized, field_name="Query list values")
        if not normalized:
            raise ValueError("Query list values must not be blank")
        retained.append(normalized)
    return tuple(sorted(set(retained), key=lambda item: item.encode("utf-8")))


def _canonical_evidence(
    values: tuple[TypedEvidence, ...],
) -> tuple[TypedEvidence, ...]:
    for evidence in values:
        _validate_unicode(evidence.type, field_name="Evidence type")
        _validate_unicode(evidence.id, field_name="Evidence id")
    return _canonical_tuple(values)


class VersionLinkRequest(_RequestModel):
    """One complete replacement-set link for an experience version."""

    target_experience_id: UUID
    relation: LinkRelation

    def to_domain(self) -> VersionLinkInput:
        return VersionLinkInput(
            target_experience_id=self.target_experience_id,
            relation=self.relation,
        )


class _ExperienceContentRequest(_RequestModel):
    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    evidence: tuple[TypedEvidence, ...] = ()
    falsifiers: tuple[str, ...] = ()

    @field_validator(
        "tags",
        "applicability",
        "evidence",
        "falsifiers",
        mode="before",
    )
    @classmethod
    def enforce_raw_list_limits(cls, values: Any) -> Any:
        return _enforce_list_limit(values)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        _validate_unicode(value, field_name="Body")
        if not value.strip():
            raise ValueError("Body must not be blank")
        if len(value.encode("utf-8")) > MAX_BODY_UTF8_BYTES:
            raise ValueError("Body must be at most 65,536 UTF-8 bytes")
        return value

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        _validate_unicode(value, field_name="Summary")
        if not value.strip():
            raise ValueError("Summary must not be blank")
        if len(value) > MAX_SUMMARY_CHARACTERS:
            raise ValueError("Summary must contain at most 1,000 characters")
        return value

    @field_validator("mechanism")
    @classmethod
    def validate_mechanism(cls, value: str) -> str:
        _validate_unicode(value, field_name="Mechanism")
        if not value.strip():
            raise ValueError("Mechanism must not be blank")
        if len(value) > MAX_MECHANISM_CHARACTERS:
            raise ValueError("Mechanism must contain at most 2,000 characters")
        return value

    @field_validator("tags", "applicability", "falsifiers")
    @classmethod
    def canonicalize_content_strings(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _canonical_content_strings(values)

    @field_validator("evidence")
    @classmethod
    def canonicalize_content_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        return _canonical_evidence(values)

    def to_content(self) -> VersionContent:
        return VersionContent(
            body=self.body,
            summary=self.summary,
            mechanism=self.mechanism,
            tags=self.tags,
            applicability=self.applicability,
            evidence=self.evidence,
            falsifiers=self.falsifiers,
        )


class _LinkedExperienceContentRequest(_ExperienceContentRequest):
    links: tuple[VersionLinkRequest, ...] = ()

    @field_validator("links", mode="before")
    @classmethod
    def enforce_raw_link_limit(cls, values: Any) -> Any:
        return _enforce_list_limit(values)

    @field_validator("links")
    @classmethod
    def canonicalize_links(
        cls,
        values: tuple[VersionLinkRequest, ...],
    ) -> tuple[VersionLinkRequest, ...]:
        by_bytes = {canonical_json_bytes(value): value for value in values}
        if len(by_bytes) != len(values):
            raise ValueError("Duplicate experience version link")
        return tuple(by_bytes[key] for key in sorted(by_bytes))

    def to_links(self) -> tuple[VersionLinkInput, ...]:
        return tuple(link.to_domain() for link in self.links)


class CreateExperienceRequest(_LinkedExperienceContentRequest):
    kind: ExperienceKind
    importance: UnitFloat = 0.35
    confidence: UnitFloat = 0.50

    def to_command(self, *, owner_agent_id: UUID) -> CreateExperience:
        return CreateExperience(
            owner_agent_id=owner_agent_id,
            kind=self.kind,
            content=self.to_content(),
            importance=self.importance,
            confidence=self.confidence,
            links=self.to_links(),
        )


class CreateExperienceVersionRequest(_LinkedExperienceContentRequest):
    def to_command(
        self,
        *,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> CreateExperienceVersion:
        return CreateExperienceVersion(
            owner_agent_id=owner_agent_id,
            experience_id=experience_id,
            content=self.to_content(),
            links=self.to_links(),
        )


class SearchExperiencesRequest(_RequestModel):
    query: str
    mode: RetrievalMode
    tags: tuple[str, ...] = ()
    mechanism_cues: tuple[str, ...] = ()
    limit: SearchLimit = DEFAULT_SEARCH_LIMIT
    content_budget_bytes: ContentBudget = DEFAULT_CONTENT_BUDGET_BYTES
    expand_cold: StrictBool = True

    @field_validator("tags", "mechanism_cues", mode="before")
    @classmethod
    def enforce_raw_query_list_limits(cls, values: Any) -> Any:
        return _enforce_list_limit(values)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        retained = value.strip()
        _validate_unicode(retained, field_name="Query")
        if not retained:
            raise ValueError("Query must not be blank")
        if len(retained) > MAX_QUERY_CHARACTERS:
            raise ValueError("Query must contain at most 2,000 characters")
        return retained

    @field_validator("tags", "mechanism_cues")
    @classmethod
    def canonicalize_query_lists(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _canonical_query_strings(values)

    def to_command(self, *, owner_agent_id: UUID) -> SearchExperiences:
        return SearchExperiences(
            owner_agent_id=owner_agent_id,
            query=self.query,
            mode=self.mode,
            tags=self.tags,
            mechanism_cues=self.mechanism_cues,
            limit=self.limit,
            content_budget_bytes=self.content_budget_bytes,
            expand_cold=self.expand_cold,
        )


class ExperienceReasonRequest(_RequestModel):
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        retained = value.strip()
        _validate_unicode(retained, field_name="Reason")
        if not 1 <= len(retained) <= 2_000:
            raise ValueError("Reason must contain 1-2,000 characters after trimming")
        return retained

    def to_reason(self) -> StructuredReason | None:
        if self.reason is None:
            return None
        return StructuredReason.from_user_text(self.reason)


class ExperienceEvidenceReasonRequest(ExperienceReasonRequest):
    evidence: tuple[TypedEvidence, ...] = ()

    @field_validator("evidence", mode="before")
    @classmethod
    def enforce_raw_evidence_limit(cls, values: Any) -> Any:
        return _enforce_list_limit(values)

    @field_validator("evidence")
    @classmethod
    def canonicalize_mutation_evidence(
        cls,
        values: tuple[TypedEvidence, ...],
    ) -> tuple[TypedEvidence, ...]:
        return _canonical_evidence(values)

    def to_evidence(self) -> tuple[TypedEvidence, ...]:
        return self.evidence


__all__ = [
    "CreateExperienceRequest",
    "CreateExperienceVersionRequest",
    "ExperienceEvidenceReasonRequest",
    "ExperienceReasonRequest",
    "SearchExperiencesRequest",
    "VersionLinkRequest",
]
