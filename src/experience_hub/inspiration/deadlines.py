"""Monotonic, cancellation-safe deadline boundaries for provider work."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Protocol, TypeVar

from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.generators.base import (
    GeneratorResult,
    IdeaGenerator,
)
from experience_hub.inspiration.models import (
    INSPIRATION_OPERATOR_ORDER,
    InspirationOperator,
    SnapshotItem,
)

_T = TypeVar("_T")

MAX_OUTPUT_TOKENS_PER_OPERATOR = 1_200
MAX_OUTPUT_TOKENS_PER_RUN = 3_600
MAX_OPERATOR_TIMEOUT_SECONDS = 30
MAX_GLOBAL_TIMEOUT_SECONDS = 90


class MonotonicClock(Protocol):
    """Read elapsed process time without using persisted domain time."""

    def now(self) -> float: ...


class DeadlineExpired(Exception):
    """The configured runner deadline actively cancelled the operation."""


class DeadlineLimit(StrEnum):
    """The budget that bounded one attempted operator call."""

    OPERATOR = "operator"
    GLOBAL = "global"


class DeadlineRunner(Protocol):
    """Run a lazily-created operation under one hard elapsed-time bound."""

    async def run(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        timeout_seconds: float,
    ) -> _T: ...


class SystemMonotonicClock:
    """Production monotonic clock backed only by ``time.monotonic``."""

    def now(self) -> float:
        return time.monotonic()


class AsyncioDeadlineRunner:
    """Production runner that distinguishes its deadline from provider errors."""

    async def run(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        timeout_seconds: float,
    ) -> _T:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) < 0.0
        ):
            raise ValueError("timeout_seconds must be a finite nonnegative number")
        timeout_context = asyncio.timeout(float(timeout_seconds))
        try:
            async with timeout_context:
                result = await operation()
        except TimeoutError as error:
            if timeout_context.expired():
                raise DeadlineExpired from error
            raise
        if timeout_context.expired():
            raise DeadlineExpired
        return result


@dataclass(frozen=True, slots=True)
class OperatorGeneration:
    """One operator's terminal result plus exact elapsed-budget accounting."""

    operator: InspirationOperator
    result: GeneratorResult
    output_tokens_reserved: int
    elapsed_milliseconds_before: int
    elapsed_milliseconds_after: int
    applied_timeout_milliseconds: int
    deadline_limit: DeadlineLimit | None
    attempted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.operator, InspirationOperator):
            raise TypeError("operator must be an InspirationOperator")
        if not isinstance(self.result, GeneratorResult):
            raise TypeError("result must be a GeneratorResult")
        try:
            validated_result = GeneratorResult.model_validate(
                self.result.model_dump(mode="python", warnings=False),
                strict=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("result must be a valid GeneratorResult") from error
        object.__setattr__(self, "result", validated_result)
        if (
            isinstance(self.output_tokens_reserved, bool)
            or not isinstance(self.output_tokens_reserved, int)
            or not 0
            <= self.output_tokens_reserved
            <= MAX_OUTPUT_TOKENS_PER_OPERATOR
        ):
            raise ValueError(
                "output_tokens_reserved must be a strict integer from 0 to 1200"
            )
        if self.result.output_tokens_consumed > self.output_tokens_reserved:
            raise ValueError("consumed output tokens cannot exceed the reservation")
        for name, value in (
            ("elapsed_milliseconds_before", self.elapsed_milliseconds_before),
            ("elapsed_milliseconds_after", self.elapsed_milliseconds_after),
            ("applied_timeout_milliseconds", self.applied_timeout_milliseconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative strict integer")
        if self.elapsed_milliseconds_after < self.elapsed_milliseconds_before:
            raise ValueError("operator elapsed time cannot move backward")
        if not isinstance(self.attempted, bool):
            raise TypeError("attempted must be a bool")
        if self.deadline_limit is not None and not isinstance(
            self.deadline_limit,
            DeadlineLimit,
        ):
            raise TypeError("deadline_limit must be a DeadlineLimit or None")
        if self.attempted and self.deadline_limit is None:
            raise ValueError("an attempted operator must carry a deadline limit")
        if self.attempted and self.applied_timeout_milliseconds <= 0:
            raise ValueError("an attempted operator must carry a positive timeout")
        if not self.attempted and self.applied_timeout_milliseconds != 0:
            raise ValueError("a skipped operator cannot carry an applied timeout")
        if (
            not self.attempted
            and self.deadline_limit is DeadlineLimit.OPERATOR
        ):
            raise ValueError("a skipped operator cannot be operator-limited")
        if self.applied_timeout_milliseconds > (
            MAX_OPERATOR_TIMEOUT_SECONDS * 1_000
        ):
            raise ValueError("applied timeout cannot exceed 30000 milliseconds")
        if not self.attempted:
            if self.output_tokens_reserved != 0:
                raise ValueError("a skipped operator cannot reserve output tokens")
            allowed_codes = (
                {OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED}
                if self.deadline_limit is DeadlineLimit.GLOBAL
                else {
                    OperatorFailureCode.INSUFFICIENT_EVIDENCE,
                    OperatorFailureCode.INSUFFICIENT_TOKEN_RESERVATION,
                }
            )
            if self.result.error_code not in allowed_codes:
                raise ValueError(
                    "a skipped operator must carry its fixed skip failure"
                )


@dataclass(frozen=True, slots=True)
class OperatorGenerationRun:
    """Fixed-order provider-independent results for the final transaction."""

    results: tuple[OperatorGeneration, ...]
    output_tokens_reserved: int
    output_tokens_consumed: int
    elapsed_milliseconds: int
    timed_out: bool

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not self.results:
            raise TypeError("results must be a nonempty immutable tuple")
        if any(not isinstance(item, OperatorGeneration) for item in self.results):
            raise TypeError("results must contain only OperatorGeneration values")
        operators = tuple(item.operator for item in self.results)
        if operators != tuple(
            operator
            for operator in INSPIRATION_OPERATOR_ORDER
            if operator in operators
        ):
            raise ValueError("results must retain fixed canonical operator order")
        if len(set(operators)) != len(operators):
            raise ValueError("results must not repeat operators")
        reserved = sum(item.output_tokens_reserved for item in self.results)
        consumed = sum(
            item.result.output_tokens_consumed for item in self.results
        )
        for name, value in (
            ("output_tokens_reserved", self.output_tokens_reserved),
            ("output_tokens_consumed", self.output_tokens_consumed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a strict integer")
        if self.output_tokens_reserved != reserved:
            raise ValueError("output_tokens_reserved must equal result reservations")
        if self.output_tokens_consumed != consumed:
            raise ValueError("output_tokens_consumed must equal result usage")
        if not 0 <= reserved <= MAX_OUTPUT_TOKENS_PER_RUN:
            raise ValueError("run reservations must be from 0 to 3600")
        if not 0 <= consumed <= reserved:
            raise ValueError("run usage cannot exceed reservations")
        if (
            isinstance(self.elapsed_milliseconds, bool)
            or not isinstance(self.elapsed_milliseconds, int)
            or self.elapsed_milliseconds < 0
        ):
            raise ValueError(
                "elapsed_milliseconds must be a nonnegative strict integer"
            )
        if self.elapsed_milliseconds != self.results[-1].elapsed_milliseconds_after:
            raise ValueError("run elapsed time must equal the final operator value")
        if any(
            current.elapsed_milliseconds_before
            < previous.elapsed_milliseconds_after
            for previous, current in zip(
                self.results[:-1],
                self.results[1:],
                strict=True,
            )
        ):
            raise ValueError("operator elapsed ranges must not move backward")
        if not isinstance(self.timed_out, bool):
            raise TypeError("timed_out must be a bool")
        has_global_failure = any(
            item.result.error_code
            is OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
            for item in self.results
        )
        if self.timed_out != has_global_failure:
            raise ValueError(
                "timed_out must exactly match global deadline exhaustion"
            )

    @property
    def global_deadline_exhausted(self) -> bool:
        return self.timed_out


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
        raise ValueError(
            f"{name} must be an integer between {lower:,} and {upper:,}"
        )
    return value


def _validated_operators(value: object) -> tuple[InspirationOperator, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("operators must be a nonempty immutable tuple")
    if any(not isinstance(operator, InspirationOperator) for operator in value):
        raise ValueError("operators must contain only InspirationOperator values")
    operators: tuple[InspirationOperator, ...] = value
    if len(set(operators)) != len(operators):
        raise ValueError("operators must not contain duplicates")
    canonical = tuple(
        operator
        for operator in INSPIRATION_OPERATOR_ORDER
        if operator in operators
    )
    if operators != canonical:
        raise ValueError("operators must follow the fixed canonical order")
    return operators


def _read_monotonic(clock: MonotonicClock) -> float:
    value = clock.now()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError("monotonic clock must return a finite number")
    return float(value)


def _milliseconds(value: float) -> int:
    return int(max(0.0, value) * 1_000.0)


def _timeout_milliseconds(value: float) -> int:
    return max(1, math.ceil(value * 1_000.0))


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


def _validated_result(
    value: object,
    *,
    reservation: int,
) -> GeneratorResult:
    if not isinstance(value, GeneratorResult):
        return _failure(
            OperatorFailureCode.GENERATOR_ERROR,
            consumed=reservation,
        )
    try:
        result = GeneratorResult.model_validate(
            value.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except (TypeError, ValueError):
        return _failure(
            OperatorFailureCode.GENERATOR_ERROR,
            consumed=reservation,
        )
    if result.output_tokens_consumed > reservation:
        return _failure(
            OperatorFailureCode.PROVIDER_BUDGET_VIOLATION,
            consumed=reservation,
        )
    return result


class BoundedGenerationRunner:
    """Apply one provider-independent token pool and monotonic deadline."""

    def __init__(
        self,
        *,
        monotonic_clock: MonotonicClock | None = None,
        deadline_runner: DeadlineRunner | None = None,
    ) -> None:
        self._clock = monotonic_clock or SystemMonotonicClock()
        self._deadline_runner = deadline_runner or AsyncioDeadlineRunner()

    async def run(
        self,
        *,
        generator: IdeaGenerator,
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
        retained_operators = _validated_operators(operators)
        requested_reservation = _validated_integer(
            "output_tokens_per_operator",
            output_tokens_per_operator,
            lower=1,
            upper=MAX_OUTPUT_TOKENS_PER_OPERATOR,
        )
        reserves_output_tokens = generator.reserves_output_tokens
        if not isinstance(reserves_output_tokens, bool):
            raise TypeError("generator.reserves_output_tokens must be a bool")
        reservation = (
            requested_reservation if reserves_output_tokens else 0
        )
        total_budget = _validated_integer(
            "total_output_tokens",
            total_output_tokens,
            lower=1,
            upper=MAX_OUTPUT_TOKENS_PER_RUN,
        )
        operator_timeout = _validated_integer(
            "operator_timeout_seconds",
            operator_timeout_seconds,
            lower=1,
            upper=MAX_OPERATOR_TIMEOUT_SECONDS,
        )
        global_timeout = _validated_integer(
            "global_timeout_seconds",
            global_timeout_seconds,
            lower=1,
            upper=MAX_GLOBAL_TIMEOUT_SECONDS,
        )
        if global_timeout < operator_timeout:
            raise ValueError(
                "global_timeout_seconds must not be less than "
                "operator_timeout_seconds"
            )
        if not frozen_items:
            empty_results = tuple(
                OperatorGeneration(
                    operator=operator,
                    result=_failure(
                        OperatorFailureCode.INSUFFICIENT_EVIDENCE,
                        consumed=0,
                    ),
                    output_tokens_reserved=0,
                    elapsed_milliseconds_before=0,
                    elapsed_milliseconds_after=0,
                    applied_timeout_milliseconds=0,
                    deadline_limit=None,
                    attempted=False,
                )
                for operator in retained_operators
            )
            return OperatorGenerationRun(
                results=empty_results,
                output_tokens_reserved=0,
                output_tokens_consumed=0,
                elapsed_milliseconds=0,
                timed_out=False,
            )

        started_at = _read_monotonic(self._clock)
        results: list[OperatorGeneration] = []
        reserved_total = 0
        consumed_total = 0
        elapsed_floor = 0.0
        timed_out = False

        for index, operator in enumerate(retained_operators):
            observed_elapsed = _read_monotonic(self._clock) - started_at
            if observed_elapsed < 0.0:
                raise RuntimeError("monotonic clock moved backward")
            elapsed_before = max(elapsed_floor, observed_elapsed)
            elapsed_floor = elapsed_before
            before_ms = _milliseconds(elapsed_before)
            remaining_global = float(global_timeout) - elapsed_before
            if remaining_global <= 0.0:
                results.extend(
                    OperatorGeneration(
                        operator=unrun,
                        result=_failure(
                            OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED,
                            consumed=0,
                        ),
                        output_tokens_reserved=0,
                        elapsed_milliseconds_before=before_ms,
                        elapsed_milliseconds_after=before_ms,
                        applied_timeout_milliseconds=0,
                        deadline_limit=DeadlineLimit.GLOBAL,
                        attempted=False,
                    )
                    for unrun in retained_operators[index:]
                )
                timed_out = True
                break
            if consumed_total + reservation > total_budget:
                results.append(
                    OperatorGeneration(
                        operator=operator,
                        result=_failure(
                            OperatorFailureCode.INSUFFICIENT_TOKEN_RESERVATION,
                            consumed=0,
                        ),
                        output_tokens_reserved=0,
                        elapsed_milliseconds_before=before_ms,
                        elapsed_milliseconds_after=before_ms,
                        applied_timeout_milliseconds=0,
                        deadline_limit=None,
                        attempted=False,
                    )
                )
                continue

            timeout_seconds = min(float(operator_timeout), remaining_global)
            deadline_limit = (
                DeadlineLimit.GLOBAL
                if remaining_global <= float(operator_timeout)
                else DeadlineLimit.OPERATOR
            )
            reserved_total += reservation
            expired = False
            try:
                raw_result = await self._deadline_runner.run(
                    partial(
                        generator.generate,
                        goal=goal,
                        context=context,
                        frozen_items=frozen_items,
                        operator=operator,
                        branch_limit=branch_limit,
                        output_token_limit=reservation,
                    ),
                    timeout_seconds=timeout_seconds,
                )
                result = _validated_result(
                    raw_result,
                    reservation=reservation,
                )
            except DeadlineExpired:
                expired = True
                result = _failure(
                    (
                        OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED
                        if deadline_limit is DeadlineLimit.GLOBAL
                        else OperatorFailureCode.PROVIDER_TIMEOUT
                    ),
                    consumed=reservation,
                )
            except Exception:
                result = _failure(
                    OperatorFailureCode.GENERATOR_ERROR,
                    consumed=reservation,
                )

            observed_after = _read_monotonic(self._clock) - started_at
            if observed_after < 0.0:
                raise RuntimeError("monotonic clock moved backward")
            elapsed_after = max(elapsed_before, observed_after)
            if expired:
                elapsed_after = max(
                    elapsed_after,
                    elapsed_before + timeout_seconds,
                )
            elapsed_floor = elapsed_after
            after_ms = _milliseconds(elapsed_after)
            results.append(
                OperatorGeneration(
                    operator=operator,
                    result=result,
                    output_tokens_reserved=reservation,
                    elapsed_milliseconds_before=before_ms,
                    elapsed_milliseconds_after=after_ms,
                    applied_timeout_milliseconds=_timeout_milliseconds(
                        timeout_seconds
                    ),
                    deadline_limit=deadline_limit,
                    attempted=True,
                )
            )
            consumed_total += result.output_tokens_consumed

            if expired and deadline_limit is DeadlineLimit.GLOBAL:
                results.extend(
                    OperatorGeneration(
                        operator=unrun,
                        result=_failure(
                            OperatorFailureCode.GLOBAL_DEADLINE_EXHAUSTED,
                            consumed=0,
                        ),
                        output_tokens_reserved=0,
                        elapsed_milliseconds_before=after_ms,
                        elapsed_milliseconds_after=after_ms,
                        applied_timeout_milliseconds=0,
                        deadline_limit=DeadlineLimit.GLOBAL,
                        attempted=False,
                    )
                    for unrun in retained_operators[index + 1 :]
                )
                timed_out = True
                break

        return OperatorGenerationRun(
            results=tuple(results),
            output_tokens_reserved=reserved_total,
            output_tokens_consumed=consumed_total,
            elapsed_milliseconds=_milliseconds(elapsed_floor),
            timed_out=timed_out,
        )


__all__ = [
    "AsyncioDeadlineRunner",
    "BoundedGenerationRunner",
    "DeadlineExpired",
    "DeadlineLimit",
    "DeadlineRunner",
    "MAX_GLOBAL_TIMEOUT_SECONDS",
    "MAX_OPERATOR_TIMEOUT_SECONDS",
    "MAX_OUTPUT_TOKENS_PER_OPERATOR",
    "MAX_OUTPUT_TOKENS_PER_RUN",
    "MonotonicClock",
    "OperatorGeneration",
    "OperatorGenerationRun",
    "SystemMonotonicClock",
]
