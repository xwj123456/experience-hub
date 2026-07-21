"""One-shot OpenAI-compatible inspiration generation without tools or retries."""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import Field, ValidationError

from experience_hub import canonical_json_bytes
from experience_hub.config import Settings
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.inspiration.commands import (
    MAX_CONTEXT_CHARACTERS,
    MAX_GOAL_CHARACTERS,
)
from experience_hub.inspiration.deadlines import (
    MAX_OUTPUT_TOKENS_PER_OPERATOR,
    AsyncioDeadlineRunner,
    BoundedGenerationRunner,
    DeadlineRunner,
    MonotonicClock,
    OperatorGeneration,
    OperatorGenerationRun,
    SystemMonotonicClock,
)
from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.generators.base import (
    GeneratorResult,
    ManagedIdeaGenerator,
)
from experience_hub.inspiration.generators.deterministic import (
    DeterministicIdeaGenerator,
)
from experience_hub.inspiration.hashing import (
    stable_evidence_key as calculate_stable_evidence_key,
)
from experience_hub.inspiration.models import (
    GeneratorKind,
    IdeaDraft,
    InspirationModel,
    InspirationOperator,
    SnapshotItem,
)
from experience_hub.inspiration.validation import validate_operator_batch

_MISSING = object()


class _ProviderSnapshotReference(InspirationModel):
    type: Literal["snapshot_item"]
    id: UUID
    stable_evidence_key: str


class _ProviderIdeaDraft(InspirationModel):
    title: str
    hypothesis: str
    mechanism: str
    predictions: tuple[str, ...]
    falsifiers: tuple[str, ...]
    assumptions: tuple[str, ...]
    proposed_test: str
    evidence: tuple[_ProviderSnapshotReference, ...]


class _ProviderIdeaBatch(InspirationModel):
    ideas: Annotated[
        tuple[_ProviderIdeaDraft, ...],
        Field(min_length=1, max_length=3),
    ]


class ProviderClientFactory(Protocol):
    """Construct a provider client only after selected settings are valid."""

    def __call__(
        self,
        *,
        base_url: str,
        api_key: str,
    ) -> httpx.AsyncClient: ...


class GeneratorNotConfiguredError(ReplayableCommandError):
    """The explicitly selected optional generator is not safely configured."""

    def __init__(self) -> None:
        super().__init__(
            code="generator_not_configured",
            message="The selected inspiration generator is not configured.",
            status_code=422,
        )


def _default_client_factory(*, base_url: str, api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        follow_redirects=False,
        timeout=None,
    )


def _is_safe_base_url(value: str) -> bool:
    try:
        value.encode("utf-8")
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            return False
        compatible_url = httpx.URL(value)
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
        return (
            compatible_url.is_absolute_url
            and parsed.scheme.casefold() in {"http", "https"}
            and bool(parsed.netloc)
            and bool(hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except (ValueError, httpx.InvalidURL):
        return False


def _is_safe_model(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return (
        bool(value)
        and value == value.strip()
        and len(value) <= 256
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _is_safe_api_key(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value.isascii()
        and all(33 <= ord(character) <= 126 for character in value)
    )


def _validated_base_url(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not _is_safe_base_url(value)
    ):
        raise ValueError("base_url must be a safe absolute HTTP(S) URL")
    return value


def _validated_text(
    name: str,
    value: object,
    *,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if value != value.strip():
        raise ValueError(f"{name} must already be trimmed")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum:,} characters")
    return value


def validate_persisted_generator_configuration(
    *,
    kind: GeneratorKind,
    configuration: object,
) -> dict[str, str]:
    """Return the exact credential-free configuration a generator may persist."""
    if not isinstance(kind, GeneratorKind):
        raise TypeError("kind must be a GeneratorKind")
    if kind is GeneratorKind.DETERMINISTIC:
        if configuration != {}:
            raise ValueError("deterministic generator configuration must be empty")
        return {}
    if not isinstance(configuration, dict) or set(configuration) != {
        "base_url",
        "model",
    }:
        raise ValueError("OpenAI-compatible configuration has invalid fields")
    base_url = _validated_base_url(configuration["base_url"])
    model = _validated_text(
        "model",
        configuration["model"],
        maximum=256,
        allow_empty=False,
    )
    if not _is_safe_model(model):
        raise ValueError("model must be a safe canonical identifier")
    return {"base_url": base_url, "model": model}


def _validated_integer(
    name: str,
    value: object,
    *,
    lower: int,
    upper: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be an integer between {lower:,} and {upper:,}")
    return value


def _validated_frozen_items(value: object) -> tuple[SnapshotItem, ...]:
    if not isinstance(value, tuple):
        raise ValueError("frozen_items must be an immutable tuple")
    try:
        items = tuple(
            SnapshotItem.model_validate(
                item.model_dump(mode="python", warnings=False),
                strict=True,
            )
            for item in value
            if isinstance(item, SnapshotItem)
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ValueError("frozen_items contain invalid snapshot values") from error
    if len(items) != len(value):
        raise ValueError("frozen_items must contain only SnapshotItem values")
    if tuple(item.rank for item in items) != tuple(range(1, len(items) + 1)):
        raise ValueError("frozen_items must retain contiguous canonical rank order")
    if len({item.snapshot_item_id for item in items}) != len(items):
        raise ValueError("frozen_items must not repeat snapshot identities")
    if len({item.stable_evidence_key for item in items}) != len(items):
        raise ValueError("frozen_items must not repeat stable evidence keys")
    source_identities = {
        (item.source_type, item.source_id, item.source_version_id) for item in items
    }
    if len(source_identities) != len(items):
        raise ValueError("frozen_items must not repeat source identities")
    if items and any(item.run_id != items[0].run_id for item in items):
        raise ValueError("frozen_items must belong to one inspiration run")
    if any(
        item.stable_evidence_key
        != calculate_stable_evidence_key(
            source_type=item.source_type,
            source_id=item.source_id,
            source_version_id=item.source_version_id,
            content_hash=item.content_hash,
        )
        for item in items
    ):
        raise ValueError("frozen_items contain an invalid stable evidence key")
    return items


def _failure(
    code: OperatorFailureCode,
    *,
    consumed: int,
) -> GeneratorResult:
    return GeneratorResult(
        ideas=(),
        error_code=code,
        output_tokens_consumed=consumed,
    )


def _strict_response_schema(*, branch_limit: int) -> dict[str, Any]:
    schema = _ProviderIdeaBatch.model_json_schema()
    ideas = schema["properties"]["ideas"]
    if not isinstance(ideas, dict):
        raise RuntimeError("provider idea schema has an invalid ideas property")
    ideas["maxItems"] = branch_limit
    return schema


def _request_payload(
    *,
    model: str,
    goal: str,
    context: str,
    frozen_items: tuple[SnapshotItem, ...],
    operator: InspirationOperator,
    branch_limit: int,
    output_token_limit: int,
) -> dict[str, Any]:
    user_content = canonical_json_bytes(
        {
            "branch_limit": branch_limit,
            "context": context,
            "frozen_excerpts": [
                {
                    "excerpt": item.excerpt,
                    "id": str(item.snapshot_item_id),
                    "rank": item.rank,
                    "stable_evidence_key": item.stable_evidence_key,
                }
                for item in frozen_items
            ],
            "goal": goal,
            "operator": operator.value,
        }
    ).decode("utf-8")
    return {
        "max_completion_tokens": output_token_limit,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate evidence-grounded inspiration branches. "
                    "Return only the requested strict JSON schema."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "model": model,
        "n": 1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "inspiration_idea_batch",
                "schema": _strict_response_schema(branch_limit=branch_limit),
                "strict": True,
            },
        },
    }


def _reported_usage(
    payload: dict[str, Any],
    *,
    reservation: int,
) -> tuple[int, OperatorFailureCode | None]:
    usage = payload.get("usage", _MISSING)
    if usage is _MISSING:
        return reservation, None
    if not isinstance(usage, dict) or "completion_tokens" not in usage:
        return reservation, OperatorFailureCode.INVALID_PROVIDER_RESPONSE
    completion_tokens = usage["completion_tokens"]
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        return reservation, OperatorFailureCode.INVALID_PROVIDER_RESPONSE
    if completion_tokens > reservation:
        return reservation, OperatorFailureCode.PROVIDER_BUDGET_VIOLATION
    return completion_tokens, None


def _completion_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    choice_index = choice.get("index") if isinstance(choice, dict) else None
    if (
        not isinstance(choice, dict)
        or isinstance(choice_index, bool)
        or not isinstance(choice_index, int)
        or choice_index != 0
        or choice.get("finish_reason") != "stop"
    ):
        return None
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or message.get("refusal") is not None
        or message.get("tool_calls") is not None
        or message.get("function_call") is not None
    ):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None
    return content


def _parse_ideas(
    content: str,
    *,
    branch_limit: int,
) -> tuple[IdeaDraft, ...] | None:
    try:
        batch = _ProviderIdeaBatch.model_validate_json(content, strict=True)
        if len(batch.ideas) > branch_limit:
            return None
        return tuple(
            IdeaDraft.model_validate(
                idea.model_dump(mode="python", warnings=False),
                strict=True,
            )
            for idea in batch.ideas
        )
    except (TypeError, ValueError, ValidationError):
        return None


class OpenAICompatibleIdeaGenerator:
    """Generate one structured provider batch per enabled operator."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model: str,
        base_url: str | None = None,
        monotonic_clock: MonotonicClock | None = None,
        deadline_runner: DeadlineRunner | None = None,
    ) -> None:
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        self._client = client
        self._monotonic_clock = monotonic_clock or SystemMonotonicClock()
        self._deadline_runner = deadline_runner or AsyncioDeadlineRunner()
        persisted_configuration = validate_persisted_generator_configuration(
            kind=GeneratorKind.OPENAI_COMPATIBLE,
            configuration={
                "base_url": (
                    base_url if base_url is not None else str(client.base_url)
                ),
                "model": model,
            },
        )
        self._model = persisted_configuration["model"]
        self._base_url = persisted_configuration["base_url"]
        self._persisted_configuration = MappingProxyType(persisted_configuration)

    @property
    def model(self) -> str:
        return self._model

    @property
    def persisted_configuration(self) -> dict[str, str]:
        """Return a detached credential-free run configuration."""
        return dict(self._persisted_configuration)

    @property
    def reserves_output_tokens(self) -> bool:
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _generate_one(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int,
    ) -> GeneratorResult:
        request = _request_payload(
            model=self._model,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operator=operator,
            branch_limit=branch_limit,
            output_token_limit=output_token_limit,
        )
        try:
            response = await self._client.post(
                f"{self._base_url.rstrip('/')}/chat/completions",
                json=request,
            )
            response.raise_for_status()
        except (httpx.TimeoutException, TimeoutError):
            return _failure(
                OperatorFailureCode.PROVIDER_TIMEOUT,
                consumed=output_token_limit,
            )
        except (httpx.HTTPStatusError, httpx.RequestError):
            return _failure(
                OperatorFailureCode.PROVIDER_HTTP_ERROR,
                consumed=output_token_limit,
            )
        except Exception:
            return _failure(
                OperatorFailureCode.GENERATOR_ERROR,
                consumed=output_token_limit,
            )

        try:
            raw_payload = response.json()
        except (TypeError, ValueError):
            return _failure(
                OperatorFailureCode.INVALID_PROVIDER_RESPONSE,
                consumed=output_token_limit,
            )
        if not isinstance(raw_payload, dict):
            return _failure(
                OperatorFailureCode.INVALID_PROVIDER_RESPONSE,
                consumed=output_token_limit,
            )
        payload = cast(dict[str, Any], raw_payload)
        consumed, usage_error = _reported_usage(
            payload,
            reservation=output_token_limit,
        )
        if usage_error is not None:
            return _failure(usage_error, consumed=consumed)

        content = _completion_content(payload)
        if content is None:
            return _failure(
                OperatorFailureCode.INVALID_PROVIDER_RESPONSE,
                consumed=consumed,
            )
        ideas = _parse_ideas(content, branch_limit=branch_limit)
        if ideas is None:
            return _failure(
                OperatorFailureCode.INVALID_PROVIDER_RESPONSE,
                consumed=consumed,
            )
        if not frozen_items:
            return _failure(
                OperatorFailureCode.INVALID_EVIDENCE_REFERENCE,
                consumed=consumed,
            )
        validated = validate_operator_batch(
            run_id=frozen_items[0].run_id,
            operator=operator,
            branches=ideas,
            snapshot_items=frozen_items,
            output_tokens_consumed=consumed,
        )
        if validated.error_code is not None:
            return _failure(validated.error_code, consumed=consumed)
        return GeneratorResult(
            ideas=validated.ideas,
            output_tokens_consumed=consumed,
        )

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = MAX_OUTPUT_TOKENS_PER_OPERATOR,
    ) -> GeneratorResult:
        retained_goal = _validated_text(
            "goal",
            goal,
            maximum=MAX_GOAL_CHARACTERS,
            allow_empty=False,
        )
        retained_context = _validated_text(
            "context",
            context,
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )
        items = _validated_frozen_items(frozen_items)
        if not isinstance(operator, InspirationOperator):
            raise ValueError("operator must be an InspirationOperator")
        limit = _validated_integer("branch_limit", branch_limit, lower=1, upper=3)
        token_limit = _validated_integer(
            "output_token_limit",
            output_token_limit,
            lower=1,
            upper=MAX_OUTPUT_TOKENS_PER_OPERATOR,
        )
        if not items:
            return _failure(
                OperatorFailureCode.INSUFFICIENT_EVIDENCE,
                consumed=0,
            )
        return await self._generate_one(
            goal=retained_goal,
            context=retained_context,
            frozen_items=items,
            operator=operator,
            branch_limit=limit,
            output_token_limit=token_limit,
        )

    async def generate_operators(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operators: tuple[InspirationOperator, ...],
        branch_limit: int,
        output_tokens_per_operator: int,
        total_output_tokens: int,
        operator_timeout_seconds: int,
        global_timeout_seconds: int,
    ) -> OperatorGenerationRun:
        _validated_text(
            "goal",
            goal,
            maximum=MAX_GOAL_CHARACTERS,
            allow_empty=False,
        )
        _validated_text(
            "context",
            context,
            maximum=MAX_CONTEXT_CHARACTERS,
            allow_empty=True,
        )
        _validated_frozen_items(frozen_items)
        _validated_integer("branch_limit", branch_limit, lower=1, upper=3)
        return await BoundedGenerationRunner(
            monotonic_clock=self._monotonic_clock,
            deadline_runner=self._deadline_runner,
        ).run(
            generator=self,
            goal=goal,
            context=context,
            frozen_items=frozen_items,
            operators=operators,
            branch_limit=branch_limit,
            output_tokens_per_operator=output_tokens_per_operator,
            total_output_tokens=total_output_tokens,
            operator_timeout_seconds=operator_timeout_seconds,
            global_timeout_seconds=global_timeout_seconds,
        )


def _selected_configuration(settings: Settings) -> tuple[str, str, str]:
    base_url = settings.openai_compatible_base_url
    model = settings.openai_compatible_model
    api_key = settings.openai_compatible_api_key
    if not all(isinstance(value, str) for value in (base_url, model, api_key)):
        raise GeneratorNotConfiguredError
    assert isinstance(base_url, str)
    assert isinstance(model, str)
    assert isinstance(api_key, str)
    if (
        not _is_safe_base_url(base_url)
        or not _is_safe_model(model)
        or not _is_safe_api_key(api_key)
    ):
        raise GeneratorNotConfiguredError
    return base_url, model, api_key


def build_idea_generator(
    *,
    kind: GeneratorKind,
    settings: Settings,
    client_factory: ProviderClientFactory = _default_client_factory,
) -> ManagedIdeaGenerator:
    """Build only the selected adapter, validating before client construction."""
    if not isinstance(kind, GeneratorKind):
        raise ValueError("kind must be a GeneratorKind")
    if not isinstance(settings, Settings):
        raise ValueError("settings must be Settings")
    if kind is GeneratorKind.DETERMINISTIC:
        return DeterministicIdeaGenerator()
    base_url, model, api_key = _selected_configuration(settings)
    client = client_factory(base_url=base_url, api_key=api_key)
    if not isinstance(client, httpx.AsyncClient):
        raise TypeError("client_factory must return an httpx.AsyncClient")
    return OpenAICompatibleIdeaGenerator(
        client=client,
        model=model,
        base_url=base_url,
    )


__all__ = [
    "GeneratorNotConfiguredError",
    "OpenAICompatibleIdeaGenerator",
    "OperatorGeneration",
    "OperatorGenerationRun",
    "ProviderClientFactory",
    "build_idea_generator",
    "validate_persisted_generator_configuration",
]
