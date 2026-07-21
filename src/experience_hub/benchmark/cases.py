"""Strict, deterministic benchmark seed and case fixture contracts."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from experience_hub.domain import TypedEvidence
from experience_hub.experiences.models import (
    ExperienceKind,
    Temperature,
    VersionContent,
)
from experience_hub.inspiration import (
    INSPIRATION_OPERATOR_ORDER,
    MAX_CONTEXT_CHARACTERS,
    MAX_GOAL_CHARACTERS,
    GeneratorKind,
    InspirationOperator,
)
from experience_hub.lifecycle.scoring import LifecycleConfig
from experience_hub.retrieval.contracts import (
    MAX_CONTENT_BUDGET_BYTES,
    MAX_QUERY_CHARACTERS,
)
from experience_hub.retrieval.ranking import MAX_RETRIEVAL_LIMIT, RetrievalMode

_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,95}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
    revalidate_instances="always",
)

PositiveInt = Annotated[StrictInt, Field(ge=1)]
UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class BenchmarkFixtureError(ValueError):
    """A stable boundary error for invalid checked-in benchmark fixtures."""


class _FixtureModel(BaseModel):
    model_config = _STRICT_CONFIG


def _label(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a lowercase benchmark label "
            "(letters, digits, underscore, or hyphen)"
        )
    return value


def _text(
    value: Any,
    *,
    field_name: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must contain valid Unicode") from error
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum:,} characters")
    return value


def _label_array(
    value: Any,
    *,
    field_name: str,
    require_nonempty: bool,
    require_empty: bool = False,
) -> Any:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    if require_nonempty and not value:
        raise ValueError(f"{field_name} must be a nonempty expected-label set")
    if require_empty and value:
        raise ValueError(f"{field_name} must be an empty expected-label set")
    for item in value:
        _label(item, field_name=f"{field_name} values")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return frozenset(value)


class BenchmarkClock(_FixtureModel):
    """The two explicit causal clock pins used for every benchmark clone."""

    started_at: datetime
    cold_at: datetime

    @field_validator("started_at", "cold_at", mode="before")
    @classmethod
    def parse_canonical_utc_timestamp(cls, value: Any) -> datetime:
        if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
            raise ValueError(
                "Benchmark clocks must be canonical RFC3339 UTC timestamps"
            )
        return datetime.fromisoformat(value.removesuffix("Z")).replace(tzinfo=UTC)

    @model_validator(mode="after")
    def require_forward_cold_clock(self) -> Self:
        if self.cold_at <= self.started_at:
            raise ValueError("cold_at must be after started_at")
        return self


class BenchmarkLifecycleConfig(_FixtureModel):
    """A completely pinned lifecycle configuration with no hidden defaults."""

    recency_half_life_hours: StrictFloat
    frequency_half_life_hours: StrictFloat
    warm_to_hot_threshold: StrictFloat
    hot_to_warm_threshold: StrictFloat
    warm_to_cold_threshold: StrictFloat
    demotion_cycles: PositiveInt
    archive_after_days: StrictFloat
    archive_importance_threshold: StrictFloat
    archive_confidence_threshold: StrictFloat
    archive_strength_threshold: StrictFloat
    minimum_cycle_interval_seconds: StrictFloat
    worker_interval_seconds: StrictFloat
    lease_duration_seconds: StrictFloat

    @model_validator(mode="after")
    def validate_domain_configuration(self) -> Self:
        LifecycleConfig(**self.model_dump())
        return self

    def to_domain(self) -> LifecycleConfig:
        """Materialize the runtime configuration without supplying defaults."""
        return LifecycleConfig(**self.model_dump())


class BenchmarkConfig(_FixtureModel):
    """Every non-content setting that could change a benchmark result."""

    retrieval_limit: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_RETRIEVAL_LIMIT),
    ]
    content_budget_bytes: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_CONTENT_BUDGET_BYTES),
    ]
    expand_cold: StrictBool
    generator: GeneratorKind
    operators: tuple[InspirationOperator, ...]
    include_inbox: StrictBool
    branches_per_operator: Annotated[StrictInt, Field(ge=1, le=3)]
    output_tokens_per_operator: Annotated[StrictInt, Field(ge=1, le=1_200)]
    total_output_tokens: Annotated[StrictInt, Field(ge=1, le=3_600)]
    operator_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=30)]
    global_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=90)]
    lifecycle: BenchmarkLifecycleConfig

    @field_validator("operators", mode="before")
    @classmethod
    def require_operator_array(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("operators must be an array")
        return tuple(value)

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if not self.expand_cold:
            raise ValueError("expand_cold must be true for cold-cue benchmarks")
        if self.generator is not GeneratorKind.DETERMINISTIC:
            raise ValueError("generator must be deterministic")
        if self.operators != INSPIRATION_OPERATOR_ORDER:
            raise ValueError("operators must use the complete canonical order")
        if self.include_inbox:
            raise ValueError("include_inbox must be false for owned evidence cases")
        if self.total_output_tokens < self.output_tokens_per_operator:
            raise ValueError(
                "total_output_tokens must cover output_tokens_per_operator"
            )
        if self.global_timeout_seconds < self.operator_timeout_seconds:
            raise ValueError(
                "global_timeout_seconds must cover operator_timeout_seconds"
            )
        return self


class SeedAgent(_FixtureModel):
    label: str

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: Any) -> str:
        return _label(value, field_name="Agent label")


class SeedExperience(_FixtureModel):
    """One case-local experience identified only by its stable fixture label."""

    label: str
    owner_label: str
    kind: ExperienceKind
    body: str
    summary: str
    mechanism: str
    tags: tuple[str, ...]
    applicability: tuple[str, ...]
    evidence: tuple[TypedEvidence, ...]
    falsifiers: tuple[str, ...]
    importance: UnitFloat
    confidence: UnitFloat
    target_temperature: Temperature

    @field_validator("label", "owner_label", mode="before")
    @classmethod
    def validate_labels(cls, value: Any, info: Any) -> str:
        return _label(value, field_name=info.field_name)

    @field_validator(
        "tags",
        "applicability",
        "evidence",
        "falsifiers",
        mode="before",
    )
    @classmethod
    def require_arrays(cls, value: Any, info: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError(f"{info.field_name} must be an array")
        return tuple(value)

    @field_validator("body", mode="before")
    @classmethod
    def validate_body(cls, value: Any) -> str:
        return _text(value, field_name="body", maximum=65_536)

    @field_validator("summary", mode="before")
    @classmethod
    def validate_summary(cls, value: Any) -> str:
        return _text(value, field_name="summary", maximum=1_000)

    @field_validator("mechanism", mode="before")
    @classmethod
    def validate_mechanism(cls, value: Any) -> str:
        return _text(value, field_name="mechanism", maximum=2_000)

    @field_validator("tags", "applicability", "falsifiers")
    @classmethod
    def validate_string_arrays(
        cls,
        value: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        for item in value:
            _text(
                item,
                field_name=f"{info.field_name} values",
                maximum=1_000,
            )
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return value

    @field_validator("target_temperature")
    @classmethod
    def reject_archived_seed(cls, value: Temperature) -> Temperature:
        if value is Temperature.ARCHIVED:
            raise ValueError("target_temperature cannot be archived")
        return value

    @model_validator(mode="after")
    def validate_public_experience_content(self) -> Self:
        VersionContent(
            body=self.body,
            summary=self.summary,
            mechanism=self.mechanism,
            tags=self.tags,
            applicability=self.applicability,
            evidence=self.evidence,
            falsifiers=self.falsifiers,
        )
        return self


class BenchmarkSeed(_FixtureModel):
    """The complete, location-independent deterministic seed document."""

    schema_version: Literal[1]
    clock: BenchmarkClock
    config: BenchmarkConfig
    agents: tuple[SeedAgent, ...]
    experiences: tuple[SeedExperience, ...]

    @field_validator("agents", "experiences", mode="before")
    @classmethod
    def require_nonempty_arrays(cls, value: Any, info: Any) -> Any:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{info.field_name} must be a nonempty array")
        return tuple(value)

    @model_validator(mode="after")
    def validate_label_registries(self) -> Self:
        agent_labels = tuple(agent.label for agent in self.agents)
        if len(agent_labels) != len(set(agent_labels)):
            raise ValueError("Duplicate agent label")
        experience_labels = tuple(experience.label for experience in self.experiences)
        if len(experience_labels) != len(set(experience_labels)):
            raise ValueError("Duplicate experience label")
        known_agents = frozenset(agent_labels)
        for experience in self.experiences:
            if experience.owner_label not in known_agents:
                raise ValueError(f"Unknown owner label: {experience.owner_label}")
        return self

    @property
    def agent_labels(self) -> frozenset[str]:
        return frozenset(agent.label for agent in self.agents)

    @property
    def experience_labels(self) -> frozenset[str]:
        return frozenset(experience.label for experience in self.experiences)

    @property
    def experience_owners(self) -> dict[str, str]:
        return {
            experience.label: experience.owner_label for experience in self.experiences
        }

    @property
    def experience_temperatures(self) -> dict[str, Temperature]:
        return {
            experience.label: experience.target_temperature
            for experience in self.experiences
        }


class _CaseBase(_FixtureModel):
    id: str
    owner_label: str

    @field_validator("id", "owner_label", mode="before")
    @classmethod
    def validate_labels(cls, value: Any, info: Any) -> str:
        return _label(value, field_name=info.field_name)


class RetrievalCase(_CaseBase):
    type: Literal["retrieval"]
    query: str
    mode: RetrievalMode
    relevant_labels: frozenset[str]

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        return _text(value, field_name="query", maximum=MAX_QUERY_CHARACTERS)

    @field_validator("relevant_labels", mode="before")
    @classmethod
    def validate_relevant_labels(cls, value: Any) -> Any:
        return _label_array(
            value,
            field_name="relevant_labels",
            require_nonempty=True,
        )


class ColdCueCase(_CaseBase):
    type: Literal["cold_cue"]
    query: str
    mode: RetrievalMode
    relevant_labels: frozenset[str]
    expected_reactivated_labels: frozenset[str]

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        return _text(value, field_name="query", maximum=MAX_QUERY_CHARACTERS)

    @field_validator("relevant_labels", "expected_reactivated_labels", mode="before")
    @classmethod
    def validate_nonempty_expected_labels(cls, value: Any, info: Any) -> Any:
        return _label_array(
            value,
            field_name=info.field_name,
            require_nonempty=True,
        )

    @model_validator(mode="after")
    def validate_reactivation_subset(self) -> Self:
        if not self.expected_reactivated_labels.issubset(self.relevant_labels):
            raise ValueError(
                "expected_reactivated_labels must be a subset of relevant_labels"
            )
        return self


class IrrelevantDistractorCase(_CaseBase):
    type: Literal["irrelevant_distractor"]
    query: str
    mode: RetrievalMode
    relevant_labels: frozenset[str]
    expected_false_reactivations: Literal[0]

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        return _text(value, field_name="query", maximum=MAX_QUERY_CHARACTERS)

    @field_validator("relevant_labels", mode="before")
    @classmethod
    def validate_empty_expected_labels(cls, value: Any) -> Any:
        return _label_array(
            value,
            field_name="relevant_labels",
            require_nonempty=False,
            require_empty=True,
        )


class PropagationCase(_CaseBase):
    type: Literal["propagation"]
    sender_label: str
    recipient_label: str
    source_labels: frozenset[str]
    query: str
    pending_relevant_labels: frozenset[str]
    adopted_relevant_labels: frozenset[str]

    @field_validator("sender_label", "recipient_label", mode="before")
    @classmethod
    def validate_agent_labels(cls, value: Any, info: Any) -> str:
        return _label(value, field_name=info.field_name)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        return _text(value, field_name="query", maximum=MAX_QUERY_CHARACTERS)

    @field_validator("source_labels", "adopted_relevant_labels", mode="before")
    @classmethod
    def validate_nonempty_labels(cls, value: Any, info: Any) -> Any:
        return _label_array(
            value,
            field_name=info.field_name,
            require_nonempty=True,
        )

    @field_validator("pending_relevant_labels", mode="before")
    @classmethod
    def validate_pending_labels(cls, value: Any) -> Any:
        return _label_array(
            value,
            field_name="pending_relevant_labels",
            require_nonempty=False,
            require_empty=True,
        )

    @model_validator(mode="after")
    def validate_recipient_owner(self) -> Self:
        if self.owner_label != self.recipient_label:
            raise ValueError("owner_label must equal recipient_label")
        if self.sender_label == self.recipient_label:
            raise ValueError("sender_label must differ from recipient_label")
        if self.adopted_relevant_labels != self.source_labels:
            raise ValueError("adopted_relevant_labels must equal source_labels")
        return self


class InspirationCase(_CaseBase):
    type: Literal["inspiration"]
    goal: str
    context: str
    mode: RetrievalMode
    evidence_labels: frozenset[str]
    expected_min_valid_ideas: PositiveInt
    expected_min_distinct_mechanisms: PositiveInt

    @field_validator("goal", mode="before")
    @classmethod
    def validate_goal(cls, value: Any) -> str:
        return _text(value, field_name="goal", maximum=MAX_GOAL_CHARACTERS)

    @field_validator("context", mode="before")
    @classmethod
    def validate_context(cls, value: Any) -> str:
        return _text(
            value,
            field_name="context",
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )

    @field_validator("evidence_labels", mode="before")
    @classmethod
    def validate_evidence_labels(cls, value: Any) -> Any:
        return _label_array(
            value,
            field_name="evidence_labels",
            require_nonempty=True,
        )

    @model_validator(mode="after")
    def validate_mechanism_expectation(self) -> Self:
        if self.expected_min_distinct_mechanisms > self.expected_min_valid_ideas:
            raise ValueError(
                "expected_min_distinct_mechanisms cannot exceed "
                "expected_min_valid_ideas"
            )
        return self


BenchmarkCase = Annotated[
    RetrievalCase
    | ColdCueCase
    | IrrelevantDistractorCase
    | PropagationCase
    | InspirationCase,
    Field(discriminator="type"),
]
_CASE_ADAPTER: TypeAdapter[BenchmarkCase] = TypeAdapter(BenchmarkCase)


def _read_bytes(path: Path, *, fixture_name: str) -> bytes:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        body = path.read_bytes()
    except OSError as error:
        raise BenchmarkFixtureError(
            f"Unable to read benchmark {fixture_name}: {path}"
        ) from error
    return body


def load_seed(path: Path) -> BenchmarkSeed:
    """Load one strict seed document without filling any implicit defaults."""
    body = _read_bytes(path, fixture_name="seed")
    try:
        decoded = json.loads(body)
        if not isinstance(decoded, dict):
            raise ValueError("seed root must be an object")
        return BenchmarkSeed.model_validate_json(body, strict=True)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as error:
        raise BenchmarkFixtureError(
            f"Invalid benchmark seed {path}: {error}"
        ) from error


def _case_seed_labels(case: BenchmarkCase) -> frozenset[str]:
    if isinstance(case, ColdCueCase):
        return case.relevant_labels | case.expected_reactivated_labels
    if isinstance(case, (RetrievalCase, IrrelevantDistractorCase)):
        return case.relevant_labels
    if isinstance(case, PropagationCase):
        return (
            case.source_labels
            | case.pending_relevant_labels
            | case.adopted_relevant_labels
        )
    return case.evidence_labels


def _case_agent_labels(case: BenchmarkCase) -> frozenset[str]:
    if isinstance(case, PropagationCase):
        return frozenset({case.owner_label, case.sender_label, case.recipient_label})
    return frozenset({case.owner_label})


def _require_case_label_ownership(
    case: BenchmarkCase,
    *,
    seed: BenchmarkSeed,
) -> None:
    if isinstance(case, (RetrievalCase, ColdCueCase)):
        expected_owner = case.owner_label
        owner_field = "owner_label"
        labels = case.relevant_labels
    elif isinstance(case, InspirationCase):
        expected_owner = case.owner_label
        owner_field = "owner_label"
        labels = case.evidence_labels
    elif isinstance(case, PropagationCase):
        expected_owner = case.sender_label
        owner_field = "sender_label"
        labels = case.source_labels
    else:
        return
    for label in sorted(labels):
        actual_owner = seed.experience_owners[label]
        if actual_owner != expected_owner:
            raise BenchmarkFixtureError(
                f"Case {case.id} label {label} is owned by {actual_owner}, "
                f"not {owner_field} {expected_owner}"
            )


def _require_cold_case_targets(
    case: BenchmarkCase,
    *,
    seed: BenchmarkSeed,
) -> None:
    if not isinstance(case, ColdCueCase):
        return
    for label in sorted(case.relevant_labels):
        if seed.experience_temperatures[label] is not Temperature.COLD:
            raise BenchmarkFixtureError(
                f"Cold-cue case {case.id} label {label} must target cold"
            )


def _require_reachable_inspiration_expectations(
    case: BenchmarkCase,
    *,
    seed: BenchmarkSeed,
) -> None:
    if not isinstance(case, InspirationCase):
        return
    maximum = len(seed.config.operators) * seed.config.branches_per_operator
    if case.expected_min_valid_ideas > maximum:
        raise BenchmarkFixtureError(
            f"Inspiration case {case.id} expects more ideas than the "
            f"pinned branch budget permits ({maximum})"
        )


def load_cases(
    path: Path,
    *,
    seed: BenchmarkSeed,
) -> tuple[BenchmarkCase, ...]:
    """Load strict JSONL cases and resolve every label against the seed."""
    if not isinstance(seed, BenchmarkSeed):
        raise TypeError("seed must be a BenchmarkSeed")
    body = _read_bytes(path, fixture_name="cases")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkFixtureError(
            f"Invalid benchmark cases {path}: content must be UTF-8"
        ) from error
    lines = text.splitlines()
    if not lines:
        raise BenchmarkFixtureError(
            f"Invalid benchmark cases {path}: file must contain cases"
        )

    retained: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise BenchmarkFixtureError(
                f"Invalid benchmark cases {path} line {line_number}: "
                "blank lines are not allowed"
            )
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("case line must be an object")
            case = _CASE_ADAPTER.validate_json(line, strict=True)
        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            raise BenchmarkFixtureError(
                f"Invalid benchmark cases {path} line {line_number}: {error}"
            ) from error
        if case.id in seen_ids:
            raise BenchmarkFixtureError(
                f"Duplicate case ID at line {line_number}: {case.id}"
            )
        seen_ids.add(case.id)

        unknown_agents = _case_agent_labels(case) - seed.agent_labels
        if unknown_agents:
            raise BenchmarkFixtureError(
                f"Unknown agent label in case {case.id}: {sorted(unknown_agents)[0]}"
            )
        unknown_seed_labels = _case_seed_labels(case) - seed.experience_labels
        if unknown_seed_labels:
            raise BenchmarkFixtureError(
                f"Unknown seed label in case {case.id}: "
                f"{sorted(unknown_seed_labels)[0]}"
            )
        _require_case_label_ownership(case, seed=seed)
        _require_cold_case_targets(case, seed=seed)
        _require_reachable_inspiration_expectations(case, seed=seed)
        retained.append(case)
    return tuple(retained)


__all__ = [
    "BenchmarkCase",
    "BenchmarkClock",
    "BenchmarkConfig",
    "BenchmarkFixtureError",
    "BenchmarkLifecycleConfig",
    "BenchmarkSeed",
    "ColdCueCase",
    "InspirationCase",
    "IrrelevantDistractorCase",
    "PropagationCase",
    "RetrievalCase",
    "SeedAgent",
    "SeedExperience",
    "load_cases",
    "load_seed",
]
