"""Closed first-party policy arms for isolated replay execution."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.experiments.contracts import (
    ArmObservationV1,
    PolicyArmDescriptorV1,
    PolicyArmKind,
    ReplayCaseV1,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.ids import SequenceIdGenerator
from experience_hub.retrieval.contracts import PeekExperiences
from experience_hub.runtime import (
    ApplicationRuntime,
    SchemaRevisionError,
    require_current_schema,
)
from experience_hub.storage.database import DatabaseBusy
from experience_hub.storage.projections import ReducerVersionMismatch
from experience_hub.storage.validation import SourceIntegrityError

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True, slots=True)
class PolicyExecutionContext:
    case: ReplayCaseV1
    clone_path: Path
    frozen_at: datetime
    seed: int


@dataclass(frozen=True, slots=True)
class _CloneIdentity:
    path: Path
    device: int
    inode: int


class PolicyArm(Protocol):
    @property
    def descriptor(self) -> PolicyArmDescriptorV1: ...

    async def execute(
        self,
        context: PolicyExecutionContext,
    ) -> ArmObservationV1: ...


def _policy_ids(context: PolicyExecutionContext, arm_id: str) -> SequenceIdGenerator:
    if isinstance(context.seed, bool) or not isinstance(context.seed, int):
        raise TypeError("seed must be an integer")
    scope = f"{context.case.case_id}:{arm_id}"
    values = tuple(
        UUID(
            bytes=hashlib.sha256(
                (
                    "experience-hub:replay-policy:"
                    f"{scope}:{context.seed}:{ordinal}"
                ).encode()
            ).digest()[:16],
            version=4,
        )
        for ordinal in range(1, 4_097)
    )
    return SequenceIdGenerator(values)


def _clone_error() -> ExperimentIsolationError:
    return ExperimentIsolationError(
        "replay_policy_clone_invalid",
        "Replay policy clone is not a valid current-schema database",
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_clone_parent(path: Path) -> int:
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:-1]:
            retained = os.stat(
                part,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(retained.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not _same_identity(retained, opened)
            ):
                os.close(child)
                raise _clone_error()
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_safe_clone(path: Path) -> _CloneIdentity:
    if not isinstance(path, Path):
        raise _clone_error()
    absolute = path.absolute()
    if not absolute.name:
        raise _clone_error()
    parent_descriptor = -1
    clone_descriptor = -1
    try:
        parent_descriptor = _open_clone_parent(absolute)
        parent_retained = os.stat(
            absolute.parent,
            follow_symlinks=False,
        )
        parent_opened = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_retained.st_mode)
            or not _same_identity(parent_retained, parent_opened)
        ):
            raise _clone_error()

        retained = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(retained.st_mode) or retained.st_nlink != 1:
            raise _clone_error()
        clone_descriptor = os.open(
            absolute.name,
            _FILE_FLAGS,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(clone_descriptor)
        declared = os.stat(absolute, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_identity(retained, opened)
            or not _same_identity(opened, declared)
        ):
            raise _clone_error()
        return _CloneIdentity(
            path=absolute,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
    except ExperimentIsolationError:
        raise
    except OSError:
        raise _clone_error() from None
    finally:
        if clone_descriptor >= 0:
            os.close(clone_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _require_same_clone(identity: _CloneIdentity) -> None:
    retained = _require_safe_clone(identity.path)
    if (
        retained.device != identity.device
        or retained.inode != identity.inode
    ):
        raise _clone_error()


def _require_same_clone_without_context(identity: _CloneIdentity) -> None:
    try:
        _require_same_clone(identity)
    except ExperimentIsolationError:
        raise _clone_error() from None


def _clone_settings(path: Path) -> Settings:
    database_url = URL.create(
        "sqlite+aiosqlite",
        database=str(path),
    )
    return Settings(database_url=database_url)


@dataclass(frozen=True, slots=True)
class NoMemoryPolicyArm:
    descriptor: PolicyArmDescriptorV1

    async def execute(
        self,
        context: PolicyExecutionContext,
    ) -> ArmObservationV1:
        del context
        return ArmObservationV1(
            schema_version=1,
            returned_labels=(),
            unmapped_count=0,
        )


@dataclass(frozen=True, slots=True)
class ExperienceHubPolicyArm:
    descriptor: PolicyArmDescriptorV1

    async def execute(
        self,
        context: PolicyExecutionContext,
    ) -> ArmObservationV1:
        try:
            identity = _require_safe_clone(context.clone_path)
            labels_by_id = {
                item.experience_id: item.label
                for item in (*context.case.expected, *context.case.forbidden)
            }
            runtime = ApplicationRuntime(
                _clone_settings(identity.path),
                clock=FrozenClock(context.frozen_at),
                ids=_policy_ids(context, self.descriptor.arm_id),
                migrator=require_current_schema,
            )
            query = PeekExperiences(
                owner_agent_id=context.case.owner_agent_id,
                query=context.case.query,
                mode=context.case.mode,
                tags=context.case.tags,
                mechanism_cues=context.case.mechanism_cues,
                limit=context.case.limit,
                content_budget_bytes=context.case.content_budget_bytes,
                expand_cold=context.case.expand_cold,
            )
            try:
                async with (
                    runtime.initialize(
                        start_lifecycle_worker=False,
                        recover_interrupted=False,
                    ) as container,
                    container.database.read_session() as session,
                ):
                    result = await container.experience_evidence_reader.peek(
                        session=session,
                        query=query,
                    )
            finally:
                _require_same_clone_without_context(identity)
        except ExperimentIsolationError:
            raise
        except (
            DatabaseBusy,
            OSError,
            ReducerVersionMismatch,
            SchemaRevisionError,
            SourceIntegrityError,
            SQLAlchemyError,
            sqlite3.DatabaseError,
        ):
            raise _clone_error() from None

        returned_labels: list[str] = []
        unmapped_count = 0
        for hit in result.hits:
            label = labels_by_id.get(hit.experience.experience_id)
            if label is None:
                unmapped_count += 1
            else:
                returned_labels.append(label)
        return ArmObservationV1(
            schema_version=1,
            returned_labels=tuple(returned_labels),
            unmapped_count=unmapped_count,
        )


def build_policy_arm(descriptor: PolicyArmDescriptorV1) -> PolicyArm:
    """Build one of the two fixed first-party arms without dynamic loading."""
    if not isinstance(descriptor, PolicyArmDescriptorV1):
        raise TypeError("descriptor must be PolicyArmDescriptorV1")
    match descriptor.kind:
        case PolicyArmKind.NO_MEMORY:
            return NoMemoryPolicyArm(descriptor)
        case PolicyArmKind.EXPERIENCE_HUB:
            return ExperienceHubPolicyArm(descriptor)
        case _:
            raise ExperimentInputError(
                "replay_policy_unsupported",
                "Replay policy kind is not supported",
            )


__all__ = [
    "ExperienceHubPolicyArm",
    "NoMemoryPolicyArm",
    "PolicyArm",
    "PolicyExecutionContext",
    "build_policy_arm",
]
