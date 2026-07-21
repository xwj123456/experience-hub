"""Experience lifecycle scoring primitives."""

from typing import TYPE_CHECKING, Any

from experience_hub.lifecycle.contracts import (
    IdeaArchivePlanner,
    LifecycleResult,
    NullIdeaArchivePlanner,
    decode_lifecycle_result,
    encode_lifecycle_result,
)
from experience_hub.lifecycle.scoring import (
    MAX_ACCESS_STRENGTH,
    AccessUpdate,
    ActivationInputs,
    ActivationResult,
    LifecycleConfig,
    activation_at,
    decay_strength,
    record_access,
)

if TYPE_CHECKING:
    from experience_hub.lifecycle.repository import (
        LifecycleRecord,
        LifecycleRepository,
    )
    from experience_hub.lifecycle.service import (
        LifecycleEvaluation,
        LifecycleRunMode,
        LifecycleService,
        LifecycleThresholdTarget,
        evaluate_transition,
        lifecycle_config_hash,
        lifecycle_cycle_id,
    )
    from experience_hub.lifecycle.worker import (
        LifecycleTicker,
        LifecycleWorker,
        LifecycleWorkerFailure,
        ManualLifecycleTicker,
        ProductionLifecycleTicker,
    )

_LAZY_SERVICE_EXPORTS = frozenset(
    {
        "LifecycleEvaluation",
        "LifecycleRunMode",
        "LifecycleService",
        "LifecycleThresholdTarget",
        "evaluate_transition",
        "lifecycle_config_hash",
        "lifecycle_cycle_id",
    }
)
_LAZY_REPOSITORY_EXPORTS = frozenset(
    {"LifecycleRecord", "LifecycleRepository"}
)
_LAZY_WORKER_EXPORTS = frozenset(
    {
        "LifecycleTicker",
        "LifecycleWorker",
        "LifecycleWorkerFailure",
        "ManualLifecycleTicker",
        "ProductionLifecycleTicker",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SERVICE_EXPORTS:
        from experience_hub.lifecycle import service

        return getattr(service, name)
    if name in _LAZY_REPOSITORY_EXPORTS:
        from experience_hub.lifecycle import repository

        return getattr(repository, name)
    if name in _LAZY_WORKER_EXPORTS:
        from experience_hub.lifecycle import worker

        return getattr(worker, name)
    raise AttributeError(name)

__all__ = [
    "MAX_ACCESS_STRENGTH",
    "AccessUpdate",
    "ActivationInputs",
    "ActivationResult",
    "IdeaArchivePlanner",
    "LifecycleConfig",
    "LifecycleEvaluation",
    "LifecycleRecord",
    "LifecycleRepository",
    "LifecycleRunMode",
    "LifecycleResult",
    "LifecycleService",
    "LifecycleThresholdTarget",
    "LifecycleTicker",
    "LifecycleWorker",
    "LifecycleWorkerFailure",
    "ManualLifecycleTicker",
    "NullIdeaArchivePlanner",
    "ProductionLifecycleTicker",
    "activation_at",
    "decay_strength",
    "decode_lifecycle_result",
    "encode_lifecycle_result",
    "evaluate_transition",
    "lifecycle_config_hash",
    "lifecycle_cycle_id",
    "record_access",
]
