"""Provider-independent inspiration generator contracts."""

from importlib import import_module
from typing import Any

from experience_hub.inspiration.generators.base import (
    GeneratorResult,
    IdeaGenerator,
    ManagedIdeaGenerator,
    OperatorFailureCode,
)
from experience_hub.inspiration.generators.deterministic import (
    DeterministicIdeaGenerator,
)

_OPTIONAL_EXPORTS = frozenset(
    {
        "GeneratorNotConfiguredError",
        "OpenAICompatibleIdeaGenerator",
        "OperatorGeneration",
        "OperatorGenerationRun",
        "build_idea_generator",
    }
)


def __getattr__(name: str) -> Any:
    """Load the optional HTTP adapter without creating an import cycle."""
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(name)
    module = import_module(
        "experience_hub.inspiration.generators.openai_compatible"
    )
    return getattr(module, name)


__all__ = [
    "DeterministicIdeaGenerator",
    "GeneratorNotConfiguredError",
    "GeneratorResult",
    "IdeaGenerator",
    "ManagedIdeaGenerator",
    "OpenAICompatibleIdeaGenerator",
    "OperatorFailureCode",
    "OperatorGeneration",
    "OperatorGenerationRun",
    "build_idea_generator",
]
