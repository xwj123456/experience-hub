"""Closed first-party policy arms for isolated replay execution."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from experience_hub.clock import FrozenClock
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
from experience_hub.experiments.policy_clones import (
    PolicyCloneIdentity,
    policy_clone_settings,
    require_safe_policy_clone,
    require_same_policy_clone,
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


@dataclass(frozen=True, slots=True)
class PolicyExecutionContext:
    case: ReplayCaseV1
    clone_path: Path
    frozen_at: datetime
    seed: int


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


def _require_same_clone_without_context(identity: PolicyCloneIdentity) -> None:
    try:
        require_same_policy_clone(identity)
    except ExperimentIsolationError:
        raise _clone_error() from None


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
            identity = require_safe_policy_clone(context.clone_path)
            labels_by_id = {
                item.experience_id: item.label
                for item in (*context.case.expected, *context.case.forbidden)
            }
            runtime = ApplicationRuntime(
                policy_clone_settings(identity.path),
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
