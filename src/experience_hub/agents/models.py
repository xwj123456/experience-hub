"""Agent command models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateAgent:
    name: str
