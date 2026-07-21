"""Controlled transaction fault checkpoints for deterministic safety tests."""

from collections.abc import Callable
from enum import StrEnum


class FaultCheckpoint(StrEnum):
    AFTER_SOURCE_INSERT = "after_source_insert"
    AFTER_EVENT_APPEND = "after_event_append"
    AFTER_PROJECTION_APPLY = "after_projection_apply"
    AFTER_RECEIPT_COMPLETION = "after_receipt_completion"


type FaultInjector = Callable[[FaultCheckpoint], None]


def ignore_faults(checkpoint: FaultCheckpoint) -> None:
    _ = checkpoint
