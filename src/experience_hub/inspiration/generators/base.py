"""Narrow, provider-independent contracts for inspiration generation."""

from __future__ import annotations

from typing import Annotated, Protocol, Self

from pydantic import Field, field_validator, model_validator

from experience_hub.inspiration.failures import OperatorFailureCode
from experience_hub.inspiration.models import (
    IdeaDraft,
    InspirationModel,
    InspirationOperator,
    SnapshotItem,
)


class GeneratorResult(InspirationModel):
    """One bounded successful batch or one sanitized operator failure."""

    ideas: tuple[IdeaDraft, ...]
    error_code: OperatorFailureCode | None = None
    output_tokens_consumed: Annotated[int, Field(strict=True, ge=0, le=1_200)] = 0

    @field_validator("ideas")
    @classmethod
    def validate_ideas(cls, values: tuple[IdeaDraft, ...]) -> tuple[IdeaDraft, ...]:
        if len(values) > 3:
            raise ValueError("ideas may contain at most three branches")
        return values

    @model_validator(mode="after")
    def validate_result_state(self) -> Self:
        if self.error_code is None:
            if not self.ideas:
                raise ValueError(
                    "a successful generator result must contain at least one idea"
                )
        elif self.ideas:
            raise ValueError("a failed generator result cannot contain ideas")
        return self


class IdeaGenerator(Protocol):
    """Generate ideas without access to mutable run or source state."""

    @property
    def reserves_output_tokens(self) -> bool: ...

    async def generate(
        self,
        *,
        goal: str,
        context: str,
        frozen_items: tuple[SnapshotItem, ...],
        operator: InspirationOperator,
        branch_limit: int,
        output_token_limit: int = 1_200,
    ) -> GeneratorResult: ...


class ManagedIdeaGenerator(IdeaGenerator, Protocol):
    """A selected generator with a uniform credential-free lifecycle."""

    @property
    def persisted_configuration(self) -> dict[str, str]: ...

    async def aclose(self) -> None: ...


__all__ = [
    "GeneratorResult",
    "IdeaGenerator",
    "ManagedIdeaGenerator",
    "OperatorFailureCode",
]
