"""Injectable UUID generators for deterministic domain behavior."""

from collections import deque
from typing import Protocol
from uuid import UUID, uuid4


class IdGenerator(Protocol):
    def new(self) -> UUID: ...


class Uuid4Generator:
    def new(self) -> UUID:
        return uuid4()


class SequenceIdGenerator:
    def __init__(self, values: list[UUID] | tuple[UUID, ...]) -> None:
        self._values = deque(values)

    def new(self) -> UUID:
        if not self._values:
            raise RuntimeError("SequenceIdGenerator is exhausted")
        return self._values.popleft()
