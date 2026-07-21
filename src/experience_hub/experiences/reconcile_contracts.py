"""Dependency-neutral reports for physical payload reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class PayloadReconcileIssue:
    """Stable, non-sensitive description of one failed version."""

    experience_id: UUID
    version_number: int
    version_id: UUID
    code: str


@dataclass(frozen=True, slots=True)
class PayloadReconcileReport:
    """Exact accounting for one deterministic reconciliation pass."""

    changed_count: int
    skipped_count: int
    error_count: int
    errors: tuple[PayloadReconcileIssue, ...] = ()

    def __post_init__(self) -> None:
        for name in ("changed_count", "skipped_count", "error_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.error_count != len(self.errors):
            raise ValueError("error_count must equal the number of errors")
        if self.errors != tuple(
            sorted(
                self.errors,
                key=lambda issue: (
                    issue.experience_id.int,
                    issue.version_number,
                    issue.version_id.int,
                ),
            )
        ):
            raise ValueError("errors must use deterministic version order")


__all__ = ["PayloadReconcileIssue", "PayloadReconcileReport"]
