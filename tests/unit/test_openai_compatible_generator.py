from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from importlib import import_module
from typing import Any
from uuid import UUID

import pytest

from experience_hub.config import Settings
from experience_hub.errors import DomainError
from experience_hub.inspiration.deadlines import (
    AsyncioDeadlineRunner,
    BoundedGenerationRunner,
    DeadlineExpired,
    DeadlineLimit,
    OperatorGeneration,
    OperatorGenerationRun,
)
from experience_hub.inspiration.generators.base import GeneratorResult
from experience_hub.inspiration.generators.deterministic import (
    DeterministicIdeaGenerator,
)
from experience_hub.inspiration.generators.openai_compatible import (
    OpenAICompatibleIdeaGenerator,
    build_idea_generator,
)
from experience_hub.inspiration.hashing import stable_evidence_key
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    EvidenceSourceState,
    EvidenceSourceType,
    GeneratorKind,
    IdeaDraft,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)

# Imported dynamically so the first TDD red state is specifically the missing
# adapter module, rather than its not-yet-added production dependency.
httpx = import_module("httpx")

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)
RUN_ID = UUID("00000000-0000-0000-0000-000000000100")
GOAL = "Prevent externally visible partial state"
CONTEXT = '{"consistency":"strict","service":"ledger"}'
BASE_URL = "https://provider.example/v1/"
MODEL = "bounded-inspiration-model"
API_KEY = "test-only-secret"


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def _hash(value: int) -> str:
    return f"{value:064x}"


def _snapshot_item(rank: int = 1) -> SnapshotItem:
    source_id = _uuid(1_000 + rank)
    source_version_id = _uuid(2_000 + rank)
    content_hash = _hash(3_000 + rank)
    return SnapshotItem(
        snapshot_item_id=_uuid(4_000 + rank),
        stable_evidence_key=stable_evidence_key(
            source_type=EvidenceSourceType.EXPERIENCE,
            source_id=source_id,
            source_version_id=source_version_id,
            content_hash=content_hash,
        ),
        run_id=RUN_ID,
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=source_id,
        source_version_id=source_version_id,
        source_state=EvidenceSourceState.WARM,
        source_trust=1.0,
        rank=rank,
        summary="Commit precedes cache invalidation.",
        mechanism="commit barrier controls cache invalidation",
        applicability=("database commit succeeds",),
        tags=("cache", "database"),
        falsifiers=("Readers observe pre-commit data.",),
        excerpt="The cache is invalidated only after the durable commit.",
        content_hash=content_hash,
        captured_at=NOW,
    )


def _idea(
    item: SnapshotItem | None = None,
    *,
    evidence_id: UUID | None = None,
    stable_key: str | None = None,
) -> IdeaDraft:
    retained = item or _snapshot_item()
    return IdeaDraft(
        title="Gate visibility on the durable transition",
        hypothesis="A commit barrier prevents externally visible partial state.",
        mechanism="commit barrier gates cache invalidation",
        predictions=(
            "Cache invalidation always follows the durable commit.",
            "Removing the gate permits a stale-read window.",
        ),
        falsifiers=("Invalidation safely precedes every commit.",),
        assumptions=("The cache is the only external read path.",),
        proposed_test="Delay commit and measure the first externally visible read.",
        evidence=(
            SnapshotEvidenceReference(
                id=evidence_id or retained.snapshot_item_id,
                stable_evidence_key=stable_key or retained.stable_evidence_key,
            ),
        ),
    )


def _chat_completion(
    request: Any,
    *,
    ideas: tuple[IdeaDraft, ...] | None = None,
    completion_tokens: int | None = 81,
    content: str | None = None,
    status_code: int = 200,
) -> Any:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_784_377_600,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        content
                        if content is not None
                        else json.dumps(
                            {
                                "ideas": [
                                    idea.model_dump(mode="json")
                                    for idea in (ideas or (_idea(),))
                                ]
                            },
                            separators=(",", ":"),
                        )
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }
    if completion_tokens is not None:
        body["usage"] = {
            "prompt_tokens": 127,
            "completion_tokens": completion_tokens,
            "total_tokens": 127 + completion_tokens,
        }
    return httpx.Response(status_code, json=body, request=request)


class FakeTransport(httpx.AsyncBaseTransport):
    """Record real AsyncClient requests while replacing only the network."""

    def __init__(self, handler: Callable[[Any], Awaitable[Any]]) -> None:
        self._handler = handler
        self.requests: list[Any] = []

    async def handle_async_request(self, request: Any) -> Any:
        self.requests.append(request)
        return await self._handler(request)


class MutableMonotonicClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self.calls = 0

    def now(self) -> float:
        self.calls += 1
        return self.value


class RecordingDeadlineRunner:
    def __init__(
        self,
        *,
        clock: MutableMonotonicClock | None = None,
        advances: tuple[float, ...] = (),
    ) -> None:
        self.clock = clock
        self.advances = list(advances)
        self.timeouts: list[float] = []

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        timeout_seconds: float,
    ) -> Any:
        self.timeouts.append(timeout_seconds)
        try:
            return await operation()
        finally:
            if self.clock is not None and self.advances:
                self.clock.value += self.advances.pop(0)


class CancellingTimeoutRunner:
    """Deterministically enter and cancel a blocked provider await."""

    def __init__(self, provider_started: asyncio.Event) -> None:
        self.provider_started = provider_started
        self.timeouts: list[float] = []

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        timeout_seconds: float,
    ) -> Any:
        self.timeouts.append(timeout_seconds)
        task = asyncio.ensure_future(operation())
        for _ in range(100):
            if self.provider_started.is_set():
                break
            await asyncio.sleep(0)
        assert self.provider_started.is_set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        raise DeadlineExpired


async def _generate(
    transport: FakeTransport,
    *,
    frozen_items: tuple[SnapshotItem, ...] | None = None,
    operators: tuple[InspirationOperator, ...] = INSPIRATION_OPERATOR_ORDER,
    branch_limit: int = 3,
    output_tokens_per_operator: int = 1_200,
    total_output_tokens: int = 3_600,
    operator_timeout_seconds: int = 30,
    global_timeout_seconds: int = 90,
    monotonic_clock: MutableMonotonicClock | None = None,
    deadline_runner: Any | None = None,
) -> Any:
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        transport=transport,
    ) as client:
        constructor_arguments: dict[str, Any] = {
            "client": client,
            "model": MODEL,
        }
        if monotonic_clock is not None:
            constructor_arguments["monotonic_clock"] = monotonic_clock
        if deadline_runner is not None:
            constructor_arguments["deadline_runner"] = deadline_runner
        generator = OpenAICompatibleIdeaGenerator(**constructor_arguments)
        return await generator.generate_operators(
            goal=GOAL,
            context=CONTEXT,
            frozen_items=(
                (_snapshot_item(),)
                if frozen_items is None
                else frozen_items
            ),
            operators=operators,
            branch_limit=branch_limit,
            output_tokens_per_operator=output_tokens_per_operator,
            total_output_tokens=total_output_tokens,
            operator_timeout_seconds=operator_timeout_seconds,
            global_timeout_seconds=global_timeout_seconds,
        )


def _result_codes(run: Any) -> tuple[Any, ...]:
    return tuple(item.result.error_code for item in run.results)


def _json_pointer(root: dict[str, Any], reference: str) -> dict[str, Any]:
    assert reference.startswith("#/")
    current: Any = root
    for component in reference[2:].split("/"):
        current = current[component.replace("~1", "/").replace("~0", "~")]
    assert isinstance(current, dict)
    return current


@pytest.mark.asyncio
async def test_asyncio_deadline_runner_cancels_and_raises_only_its_own_marker(
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def block() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(DeadlineExpired):
        await AsyncioDeadlineRunner().run(block, timeout_seconds=0)

    assert entered.is_set()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_asyncio_deadline_runner_preserves_provider_timeout_error() -> None:
    provider_error = TimeoutError("provider raised its own timeout")

    async def provider_times_out() -> None:
        raise provider_error

    with pytest.raises(TimeoutError) as raised:
        await AsyncioDeadlineRunner().run(
            provider_times_out,
            timeout_seconds=30,
        )

    assert raised.value is provider_error


@pytest.mark.asyncio
async def test_asyncio_deadline_runner_propagates_external_cancellation() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def block() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(
        AsyncioDeadlineRunner().run(block, timeout_seconds=30)
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


def test_generation_audit_models_reject_cross_field_forgery() -> None:
    succeeded = GeneratorResult(
        ideas=(_idea(),),
        output_tokens_consumed=0,
    )
    with pytest.raises(ValueError, match="fixed skip failure"):
        OperatorGeneration(
            operator=InspirationOperator.CAUSAL_GAP,
            result=succeeded,
            output_tokens_reserved=0,
            elapsed_milliseconds_before=0,
            elapsed_milliseconds_after=0,
            applied_timeout_milliseconds=0,
            deadline_limit=None,
            attempted=False,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        OperatorGeneration(
            operator=InspirationOperator.CAUSAL_GAP,
            result=succeeded,
            output_tokens_reserved=0,
            elapsed_milliseconds_before=0,
            elapsed_milliseconds_after=0,
            applied_timeout_milliseconds=30_001,
            deadline_limit=DeadlineLimit.OPERATOR,
            attempted=True,
        )

    first = OperatorGeneration(
        operator=InspirationOperator.CAUSAL_GAP,
        result=succeeded,
        output_tokens_reserved=0,
        elapsed_milliseconds_before=0,
        elapsed_milliseconds_after=10_000,
        applied_timeout_milliseconds=30_000,
        deadline_limit=DeadlineLimit.OPERATOR,
        attempted=True,
    )
    second = OperatorGeneration(
        operator=InspirationOperator.COUNTERFACTUAL,
        result=succeeded,
        output_tokens_reserved=0,
        elapsed_milliseconds_before=0,
        elapsed_milliseconds_after=0,
        applied_timeout_milliseconds=30_000,
        deadline_limit=DeadlineLimit.OPERATOR,
        attempted=True,
    )
    with pytest.raises(ValueError, match="must not move backward"):
        OperatorGenerationRun(
            results=(first, second),
            output_tokens_reserved=0,
            output_tokens_consumed=0,
            elapsed_milliseconds=0,
            timed_out=False,
        )

    consumed = GeneratorResult(
        ideas=(_idea(),),
        output_tokens_consumed=1,
    )
    one_token = OperatorGeneration(
        operator=InspirationOperator.CAUSAL_GAP,
        result=consumed,
        output_tokens_reserved=1,
        elapsed_milliseconds_before=0,
        elapsed_milliseconds_after=0,
        applied_timeout_milliseconds=30_000,
        deadline_limit=DeadlineLimit.OPERATOR,
        attempted=True,
    )
    with pytest.raises(TypeError, match="strict integer"):
        OperatorGenerationRun(
            results=(one_token,),
            output_tokens_reserved=True,  # type: ignore[arg-type]
            output_tokens_consumed=True,  # type: ignore[arg-type]
            elapsed_milliseconds=0,
            timed_out=False,
        )


@pytest.mark.asyncio
async def test_each_operator_makes_one_typed_tool_free_chat_completion_call() -> None:
    item = _snapshot_item()

    async def succeed(request: Any) -> Any:
        return _chat_completion(request, ideas=(_idea(item),))

    transport = FakeTransport(succeed)

    run = await _generate(transport, frozen_items=(item,))

    assert tuple(result.operator for result in run.results) == (
        INSPIRATION_OPERATOR_ORDER
    )
    assert all(result.result.error_code is None for result in run.results)
    assert len(transport.requests) == len(INSPIRATION_OPERATOR_ORDER)
    assert run.output_tokens_reserved == 3_600
    assert run.output_tokens_consumed == 243
    assert run.timed_out is False

    for operator, request in zip(
        INSPIRATION_OPERATOR_ORDER,
        transport.requests,
        strict=True,
    ):
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["n"] == 1
        assert payload["max_completion_tokens"] == 1_200
        assert "max_tokens" not in payload
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert API_KEY not in request.content.decode("utf-8")

        user_content = json.loads(payload["messages"][-1]["content"])
        assert set(user_content) == {
            "branch_limit",
            "context",
            "frozen_excerpts",
            "goal",
            "operator",
        }
        assert user_content["goal"] == GOAL
        assert user_content["context"] == CONTEXT
        excerpt = user_content["frozen_excerpts"][0]
        assert set(excerpt) == {
            "excerpt",
            "id",
            "rank",
            "stable_evidence_key",
        }
        assert excerpt["excerpt"] == item.excerpt
        assert user_content["operator"] == operator.value

        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["ideas"]
        ideas_schema = schema["properties"]["ideas"]
        assert ideas_schema["type"] == "array"
        assert ideas_schema["minItems"] == 1
        assert ideas_schema["maxItems"] == 3
        draft_schema = ideas_schema["items"]
        if "$ref" in draft_schema:
            draft_schema = _json_pointer(schema, draft_schema["$ref"])
        assert draft_schema["additionalProperties"] is False
        assert set(draft_schema["required"]) == set(IdeaDraft.model_fields)
        evidence_schema = draft_schema["properties"]["evidence"]["items"]
        if "$ref" in evidence_schema:
            evidence_schema = _json_pointer(schema, evidence_schema["$ref"])
        assert evidence_schema["additionalProperties"] is False
        assert set(evidence_schema["required"]) == {
            "id",
            "stable_evidence_key",
            "type",
        }
        assert evidence_schema["properties"]["type"]["const"] == "snapshot_item"


@pytest.mark.asyncio
async def test_http_failure_is_sanitized_and_is_not_retried() -> None:
    async def fail(request: Any) -> Any:
        return httpx.Response(
            503,
            json={"error": {"message": "raw provider secret"}},
            request=request,
        )

    transport = FakeTransport(fail)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("provider_http_error",)
    assert run.results[0].result.ideas == ()
    assert len(transport.requests) == 1
    assert "raw provider secret" not in repr(run)


@pytest.mark.asyncio
async def test_provider_timeout_error_is_sanitized_and_not_retried() -> None:
    async def fail(_request: Any) -> Any:
        raise TimeoutError("provider-local raw timeout")

    transport = FakeTransport(fail)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("provider_timeout",)
    assert run.results[0].result.output_tokens_consumed == 1_200
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_body",
    (
        {"id": "missing-choices", "choices": []},
        {
            "id": "missing-message-content",
            "choices": [{"index": 0, "message": {"role": "assistant"}}],
        },
    ),
)
async def test_invalid_completion_envelope_maps_to_fixed_code(
    response_body: dict[str, Any],
) -> None:
    async def invalid(request: Any) -> Any:
        return httpx.Response(200, json=response_body, request=request)

    transport = FakeTransport(invalid)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("invalid_provider_response",)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        "not-json",
        json.dumps({"ideas": [{"title": "missing every required field"}]}),
        json.dumps({"ideas": []}),
    ),
)
async def test_invalid_typed_content_maps_to_fixed_code(content: str) -> None:
    async def invalid(request: Any) -> Any:
        return _chat_completion(request, content=content)

    run = await _generate(
        FakeTransport(invalid),
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("invalid_provider_response",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    (
        lambda body: body.update(
            choices=[*body["choices"], body["choices"][0]]
        ),
        lambda body: body["choices"][0]["message"].update(
            refusal="cannot comply"
        ),
        lambda body: body["choices"][0].update(finish_reason="length"),
        lambda body: body["choices"][0].update(index=99),
        lambda body: body["choices"][0].update(index=False),
        lambda body: body["choices"][0].update(index=0.0),
        lambda body: body["choices"][0]["message"].update(
            tool_calls=[{"type": "function"}]
        ),
    ),
)
async def test_ambiguous_or_incomplete_completion_is_invalid(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    async def invalid(request: Any) -> Any:
        response = _chat_completion(request)
        body = response.json()
        mutate(body)
        return httpx.Response(200, json=body, request=request)

    run = await _generate(
        FakeTransport(invalid),
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("invalid_provider_response",)


@pytest.mark.asyncio
async def test_foreign_snapshot_reference_rejects_the_entire_operator_batch() -> None:
    item = _snapshot_item()
    foreign = _idea(item, evidence_id=_uuid(99_999))

    async def respond(request: Any) -> Any:
        return _chat_completion(request, ideas=(_idea(item), foreign))

    run = await _generate(
        FakeTransport(respond),
        frozen_items=(item,),
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("invalid_evidence_reference",)
    assert run.results[0].result.ideas == ()


@pytest.mark.asyncio
async def test_reported_usage_above_reservation_is_a_budget_violation() -> None:
    async def overrun(request: Any) -> Any:
        return _chat_completion(request, completion_tokens=1_201)

    transport = FakeTransport(overrun)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("provider_budget_violation",)
    assert run.results[0].result.ideas == ()
    assert run.output_tokens_reserved == 1_200
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    (
        {"completion_tokens": True},
        {"completion_tokens": -1},
        {"completion_tokens": 1.5},
        {"prompt_tokens": 10},
        None,
    ),
)
async def test_malformed_present_usage_is_invalid_and_charged_conservatively(
    usage: object,
) -> None:
    async def malformed(request: Any) -> Any:
        response = _chat_completion(request)
        body = response.json()
        body["usage"] = usage
        return httpx.Response(200, json=body, request=request)

    run = await _generate(
        FakeTransport(malformed),
        operators=(InspirationOperator.CAUSAL_GAP,),
    )

    assert _result_codes(run) == ("invalid_provider_response",)
    assert run.output_tokens_consumed == 1_200


@pytest.mark.asyncio
async def test_unused_tokens_are_released_and_missing_usage_uses_full_reservation(
) -> None:
    call_count = 0

    async def respond(request: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _chat_completion(request, completion_tokens=400)
        return _chat_completion(request, completion_tokens=None)

    transport = FakeTransport(respond)

    run = await _generate(transport, total_output_tokens=2_400)

    assert _result_codes(run) == (
        None,
        None,
        "insufficient_token_reservation",
    )
    assert len(transport.requests) == 2
    assert run.output_tokens_reserved == 2_400
    assert run.output_tokens_consumed == 1_600
    assert run.results[0].result.output_tokens_consumed == 400
    assert run.results[1].result.output_tokens_consumed == 1_200
    assert run.results[2].result.output_tokens_consumed == 0


@pytest.mark.asyncio
async def test_insufficient_remaining_reservation_skips_provider_call() -> None:
    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    transport = FakeTransport(must_not_run)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
        output_tokens_per_operator=1_200,
        total_output_tokens=1_199,
    )

    assert _result_codes(run) == ("insufficient_token_reservation",)
    assert run.output_tokens_reserved == 0
    assert run.output_tokens_consumed == 0
    assert transport.requests == []


@pytest.mark.asyncio
async def test_empty_snapshot_skips_every_provider_call_without_reservation() -> None:
    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    transport = FakeTransport(must_not_run)

    run = await _generate(transport, frozen_items=())

    assert _result_codes(run) == (
        "insufficient_evidence",
        "insufficient_evidence",
        "insufficient_evidence",
    )
    assert run.output_tokens_reserved == 0
    assert run.output_tokens_consumed == 0
    assert all(not item.attempted for item in run.results)
    assert transport.requests == []


@pytest.mark.asyncio
async def test_corrupted_snapshot_is_rejected_before_provider_call() -> None:
    corrupted = _snapshot_item().model_copy(
        update={"stable_evidence_key": "f" * 64}
    )

    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    transport = FakeTransport(must_not_run)
    with pytest.raises(ValueError, match="stable evidence key"):
        await _generate(transport, frozen_items=(corrupted,))

    assert transport.requests == []


@pytest.mark.asyncio
async def test_success_survives_global_exhaustion_before_later_operators() -> None:
    async def succeed(request: Any) -> Any:
        return _chat_completion(request)

    clock = MutableMonotonicClock()
    runner = RecordingDeadlineRunner(clock=clock, advances=(90.0,))
    transport = FakeTransport(succeed)

    run = await _generate(
        transport,
        monotonic_clock=clock,
        deadline_runner=runner,
    )

    assert _result_codes(run) == (
        None,
        "global_deadline_exhausted",
        "global_deadline_exhausted",
    )
    assert run.results[0].result.ideas
    assert run.timed_out is True
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_global_exhaustion_precedes_insufficient_token_reservation() -> None:
    class ExhaustedClock:
        def __init__(self) -> None:
            self.calls = 0

        def now(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 90.0

    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    transport = FakeTransport(must_not_run)
    run = await _generate(
        transport,
        output_tokens_per_operator=1_200,
        total_output_tokens=1_199,
        monotonic_clock=ExhaustedClock(),  # type: ignore[arg-type]
        deadline_runner=RecordingDeadlineRunner(),
    )

    assert _result_codes(run) == (
        "global_deadline_exhausted",
        "global_deadline_exhausted",
        "global_deadline_exhausted",
    )
    assert transport.requests == []


@pytest.mark.asyncio
async def test_operator_timeout_cancels_provider_without_wall_clock_sleep() -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def block(request: Any) -> Any:
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        pytest.fail(f"provider unexpectedly resumed: {request.url}")

    runner = CancellingTimeoutRunner(provider_started)
    transport = FakeTransport(block)

    run = await _generate(
        transport,
        operators=(InspirationOperator.CAUSAL_GAP,),
        operator_timeout_seconds=30,
        global_timeout_seconds=90,
        monotonic_clock=MutableMonotonicClock(),
        deadline_runner=runner,
    )

    assert _result_codes(run) == ("provider_timeout",)
    assert runner.timeouts == [30]
    assert provider_cancelled.is_set()
    assert run.timed_out is False
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_call_deadline_is_minimum_of_operator_and_remaining_global() -> None:
    async def succeed(request: Any) -> Any:
        return _chat_completion(request)

    clock = MutableMonotonicClock()
    runner = RecordingDeadlineRunner(clock=clock, advances=(85.0, 0.0))

    run = await _generate(
        FakeTransport(succeed),
        operators=(
            InspirationOperator.CAUSAL_GAP,
            InspirationOperator.COUNTERFACTUAL,
        ),
        operator_timeout_seconds=30,
        global_timeout_seconds=90,
        monotonic_clock=clock,
        deadline_runner=runner,
    )

    assert _result_codes(run) == (None, None)
    assert runner.timeouts == [30, 5]
    assert clock.calls == 5
    assert run.elapsed_milliseconds == 85_000
    assert tuple(
        (
            item.elapsed_milliseconds_before,
            item.elapsed_milliseconds_after,
            item.applied_timeout_milliseconds,
            item.deadline_limit,
            item.attempted,
        )
        for item in run.results
    ) == (
        (0, 85_000, 30_000, DeadlineLimit.OPERATOR, True),
        (85_000, 85_000, 5_000, DeadlineLimit.GLOBAL, True),
    )


@pytest.mark.asyncio
async def test_provider_independent_runner_executes_deterministic_generator(
) -> None:
    run = await BoundedGenerationRunner(
        monotonic_clock=MutableMonotonicClock(),
        deadline_runner=RecordingDeadlineRunner(),
    ).run(
        generator=DeterministicIdeaGenerator(),
        goal=GOAL,
        context=CONTEXT,
        frozen_items=(_snapshot_item(),),
        operators=(InspirationOperator.COUNTERFACTUAL,),
        branch_limit=1,
        output_tokens_per_operator=1_200,
        total_output_tokens=1,
        operator_timeout_seconds=30,
        global_timeout_seconds=90,
    )

    assert run.results[0].result.error_code is None
    assert run.results[0].result.ideas
    assert run.results[0].output_tokens_reserved == 0
    assert run.results[0].result.output_tokens_consumed == 0
    assert run.results[0].attempted is True
    assert run.results[0].deadline_limit is DeadlineLimit.OPERATOR


@pytest.mark.asyncio
async def test_zero_global_remainder_makes_no_call_and_fails_unrun_operators_in_order(
) -> None:
    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    clock = MutableMonotonicClock()

    class ExhaustedBeforeFirstCall:
        def __init__(self) -> None:
            self.calls = 0

        def now(self) -> float:
            self.calls += 1
            return 100.0 if self.calls == 1 else 190.0

    transport = FakeTransport(must_not_run)
    exhausted_clock = ExhaustedBeforeFirstCall()

    run = await _generate(
        transport,
        monotonic_clock=exhausted_clock,  # type: ignore[arg-type]
        deadline_runner=RecordingDeadlineRunner(clock=clock),
    )

    assert tuple(result.operator for result in run.results) == (
        INSPIRATION_OPERATOR_ORDER
    )
    assert _result_codes(run) == (
        "global_deadline_exhausted",
        "global_deadline_exhausted",
        "global_deadline_exhausted",
    )
    assert run.timed_out is True
    assert transport.requests == []


@pytest.mark.asyncio
async def test_global_timeout_cancels_call_and_marks_every_unrun_operator(
) -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def block(request: Any) -> Any:
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        pytest.fail(f"provider unexpectedly resumed: {request.url}")

    runner = CancellingTimeoutRunner(provider_started)
    transport = FakeTransport(block)

    run = await _generate(
        transport,
        operator_timeout_seconds=30,
        global_timeout_seconds=30,
        monotonic_clock=MutableMonotonicClock(),
        deadline_runner=runner,
    )

    assert tuple(result.operator for result in run.results) == (
        INSPIRATION_OPERATOR_ORDER
    )
    assert _result_codes(run) == (
        "global_deadline_exhausted",
        "global_deadline_exhausted",
        "global_deadline_exhausted",
    )
    assert runner.timeouts == [30]
    assert provider_cancelled.is_set()
    assert run.timed_out is True
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_external_cancellation_propagates_instead_of_becoming_operator_failure(
) -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def block(request: Any) -> Any:
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        pytest.fail(f"provider unexpectedly resumed: {request.url}")

    transport = FakeTransport(block)
    async with httpx.AsyncClient(base_url=BASE_URL, transport=transport) as client:
        generator = OpenAICompatibleIdeaGenerator(
            client=client,
            model=MODEL,
            monotonic_clock=MutableMonotonicClock(),
            deadline_runner=RecordingDeadlineRunner(),
        )
        task = asyncio.create_task(
            generator.generate_operators(
                goal=GOAL,
                context=CONTEXT,
                frozen_items=(_snapshot_item(),),
                operators=(InspirationOperator.CAUSAL_GAP,),
                branch_limit=3,
                output_tokens_per_operator=1_200,
                total_output_tokens=3_600,
                operator_timeout_seconds=30,
                global_timeout_seconds=90,
            )
        )
        await provider_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert provider_cancelled.is_set()
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operator_timeout_seconds", "global_timeout_seconds"),
    (
        (31, 90),
        (30, 91),
        (30, 29),
    ),
)
async def test_deadline_limits_are_rejected_before_provider_construction(
    operator_timeout_seconds: int,
    global_timeout_seconds: int,
) -> None:
    async def must_not_run(request: Any) -> Any:
        pytest.fail(f"unexpected provider request: {request.url}")

    transport = FakeTransport(must_not_run)

    with pytest.raises(ValueError):
        await _generate(
            transport,
            operators=(InspirationOperator.CAUSAL_GAP,),
            operator_timeout_seconds=operator_timeout_seconds,
            global_timeout_seconds=global_timeout_seconds,
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_deterministic_factory_needs_no_provider_configuration_or_client(
) -> None:
    client_calls: list[dict[str, Any]] = []

    def client_factory(**arguments: Any) -> Any:
        client_calls.append(arguments)
        raise AssertionError("deterministic generator must not build an HTTP client")

    generator = build_idea_generator(
        kind=GeneratorKind.DETERMINISTIC,
        settings=Settings(),
        client_factory=client_factory,
    )

    assert isinstance(generator, DeterministicIdeaGenerator)
    assert generator.persisted_configuration == {}
    assert client_calls == []
    await generator.aclose()


@pytest.mark.parametrize(
    "missing_field",
    (
        "openai_compatible_base_url",
        "openai_compatible_model",
        "openai_compatible_api_key",
    ),
)
def test_incomplete_selected_provider_configuration_fails_before_client_creation(
    missing_field: str,
) -> None:
    configuration: dict[str, str | None] = {
        "openai_compatible_base_url": BASE_URL,
        "openai_compatible_model": MODEL,
        "openai_compatible_api_key": API_KEY,
    }
    configuration[missing_field] = None
    client_calls: list[dict[str, Any]] = []

    def client_factory(**arguments: Any) -> Any:
        client_calls.append(arguments)
        raise AssertionError("invalid configuration must not build an HTTP client")

    with pytest.raises(DomainError) as raised:
        build_idea_generator(
            kind=GeneratorKind.OPENAI_COMPATIBLE,
            settings=Settings(**configuration),  # type: ignore[arg-type]
            client_factory=client_factory,
        )

    assert raised.value.code == "generator_not_configured"
    assert raised.value.status_code == 422
    assert raised.value.details == {}
    assert client_calls == []


@pytest.mark.asyncio
async def test_complete_selected_configuration_constructs_exactly_one_adapter(
) -> None:
    client = httpx.AsyncClient(base_url=BASE_URL)
    client_calls: list[dict[str, Any]] = []

    def client_factory(**arguments: Any) -> Any:
        client_calls.append(arguments)
        return client

    generator = build_idea_generator(
        kind=GeneratorKind.OPENAI_COMPATIBLE,
        settings=Settings(
            openai_compatible_base_url=BASE_URL,
            openai_compatible_model=MODEL,
            openai_compatible_api_key=API_KEY,
        ),
        client_factory=client_factory,
    )

    assert isinstance(generator, OpenAICompatibleIdeaGenerator)
    assert len(client_calls) == 1
    assert client_calls[0]["base_url"] == BASE_URL
    assert client_calls[0]["api_key"] == API_KEY
    assert generator.model == MODEL
    assert generator.persisted_configuration == {
        "base_url": BASE_URL,
        "model": MODEL,
    }
    assert API_KEY not in json.dumps(generator.persisted_configuration)
    assert API_KEY not in repr(
        Settings(
            openai_compatible_base_url=BASE_URL,
            openai_compatible_model=MODEL,
            openai_compatible_api_key=API_KEY,
        )
    )
    assert client.is_closed is False
    await generator.aclose()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_selected_base_url_is_the_actual_audited_request_endpoint() -> None:
    async def succeed(request: Any) -> Any:
        return _chat_completion(request)

    transport = FakeTransport(succeed)

    def client_factory(**_arguments: Any) -> Any:
        return httpx.AsyncClient(
            base_url="https://different.example/ignored/",
            transport=transport,
        )

    generator = build_idea_generator(
        kind=GeneratorKind.OPENAI_COMPATIBLE,
        settings=Settings(
            openai_compatible_base_url=BASE_URL,
            openai_compatible_model=MODEL,
            openai_compatible_api_key=API_KEY,
        ),
        client_factory=client_factory,
    )
    assert isinstance(generator, OpenAICompatibleIdeaGenerator)

    run = await generator.generate_operators(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=(_snapshot_item(),),
        operators=(InspirationOperator.CAUSAL_GAP,),
        branch_limit=1,
        output_tokens_per_operator=1_200,
        total_output_tokens=1_200,
        operator_timeout_seconds=30,
        global_timeout_seconds=90,
    )

    assert run.results[0].result.error_code is None
    assert str(transport.requests[0].url) == (
        "https://provider.example/v1/chat/completions"
    )
    await generator.aclose()


@pytest.mark.parametrize(
    "base_url",
    (
        "provider.example/v1/",
        "ftp://provider.example/v1/",
        "https://user:secret@provider.example/v1/",
        "https://provider.example/v1/?token=secret",
        "https://provider.example/v1/#secret",
        "https://provider.example:bad/v1/",
        "https://:443/v1/",
        "https://provider.example/\nsecret",
        "https://provider.example/\x00secret",
        "https:// provider.example/v1/",
        "https://provider.ex ample/v1/",
        "https://[v1.foo]/",
    ),
)
def test_selected_provider_rejects_unsafe_persistable_base_url(
    base_url: str,
) -> None:
    client_calls: list[dict[str, Any]] = []

    def client_factory(**arguments: Any) -> object:
        client_calls.append(arguments)
        return object()

    with pytest.raises(DomainError) as raised:
        build_idea_generator(
            kind=GeneratorKind.OPENAI_COMPATIBLE,
            settings=Settings(
                openai_compatible_base_url=base_url,
                openai_compatible_model=MODEL,
                openai_compatible_api_key=API_KEY,
            ),
            client_factory=client_factory,
        )

    assert raised.value.code == "generator_not_configured"
    assert client_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("openai_compatible_model", "\ud800"),
        ("openai_compatible_model", "model\x00name"),
        ("openai_compatible_api_key", "\ud800"),
        ("openai_compatible_api_key", "secret\nheader"),
        ("openai_compatible_api_key", "secret key"),
    ),
)
def test_selected_provider_rejects_unsafe_model_or_api_key_before_client(
    field: str,
    value: str,
) -> None:
    configuration = {
        "openai_compatible_base_url": BASE_URL,
        "openai_compatible_model": MODEL,
        "openai_compatible_api_key": API_KEY,
    }
    configuration[field] = value
    client_calls: list[dict[str, Any]] = []

    def client_factory(**arguments: Any) -> Any:
        client_calls.append(arguments)
        return httpx.AsyncClient(base_url=BASE_URL)

    with pytest.raises(DomainError) as raised:
        build_idea_generator(
            kind=GeneratorKind.OPENAI_COMPATIBLE,
            settings=Settings(**configuration),
            client_factory=client_factory,
        )

    assert raised.value.code == "generator_not_configured"
    assert client_calls == []
