"""Public replay experiment contracts and fixture loading boundary."""

from experience_hub.experiments.contracts import (
    ArmEvidenceV1,
    ArmObservationV1,
    CaseEvidenceV1,
    OracleDescriptorV1,
    OracleEvidenceV1,
    PolicyArmDescriptorV1,
    PolicyArmKind,
    ReplayCaseV1,
    ReplayDatasetDescriptorV1,
    ReplayEvidenceReportV1,
    ReplayManifestV1,
    ReplayProfileReportV1,
    ResolvedReplayManifestV1,
)
from experience_hub.experiments.errors import (
    ExperimentInputError,
    ExperimentIsolationError,
)
from experience_hub.experiments.loading import (
    LoadedReplayDataset,
    LoadedReplayManifest,
    load_replay_cases,
    load_replay_manifest,
)
from experience_hub.experiments.oracles import score_retrieval_observation
from experience_hub.experiments.policies import (
    ExperienceHubPolicyArm,
    NoMemoryPolicyArm,
    PolicyArm,
    PolicyExecutionContext,
    build_policy_arm,
)
from experience_hub.experiments.reports import (
    ExperimentOutputError,
    ReplayArtifactSet,
    canonical_evidence_bytes,
    canonical_profile_bytes,
    verify_evidence_bytes,
    write_replay_artifacts,
)
from experience_hub.experiments.runner import (
    ReplayExecution,
    ReplayInspection,
    inspect_replay,
    run_replay,
    verify_replay_report,
)
from experience_hub.experiments.snapshots import (
    FrozenSqliteSnapshot,
    checkpoint_owned_sqlite,
    clone_frozen_sqlite,
    freeze_closed_sqlite,
    validate_frozen_snapshot,
    verify_source_unchanged,
)
from experience_hub.experiments.workspace import (
    REPLAY_WORKSPACE_POLICY,
    OwnedWorkspace,
    WorkspacePolicy,
    prepare_owned_workspace,
)

__all__ = [
    "ArmEvidenceV1",
    "ArmObservationV1",
    "CaseEvidenceV1",
    "ExperimentInputError",
    "ExperimentIsolationError",
    "ExperimentOutputError",
    "ExperienceHubPolicyArm",
    "FrozenSqliteSnapshot",
    "LoadedReplayDataset",
    "LoadedReplayManifest",
    "NoMemoryPolicyArm",
    "OracleDescriptorV1",
    "OracleEvidenceV1",
    "OwnedWorkspace",
    "PolicyArm",
    "PolicyArmDescriptorV1",
    "PolicyArmKind",
    "PolicyExecutionContext",
    "ReplayCaseV1",
    "ReplayArtifactSet",
    "ReplayDatasetDescriptorV1",
    "ReplayEvidenceReportV1",
    "ReplayExecution",
    "ReplayInspection",
    "ReplayManifestV1",
    "ReplayProfileReportV1",
    "REPLAY_WORKSPACE_POLICY",
    "ResolvedReplayManifestV1",
    "WorkspacePolicy",
    "build_policy_arm",
    "checkpoint_owned_sqlite",
    "canonical_evidence_bytes",
    "canonical_profile_bytes",
    "clone_frozen_sqlite",
    "freeze_closed_sqlite",
    "inspect_replay",
    "load_replay_cases",
    "load_replay_manifest",
    "prepare_owned_workspace",
    "run_replay",
    "score_retrieval_observation",
    "validate_frozen_snapshot",
    "verify_evidence_bytes",
    "verify_replay_report",
    "verify_source_unchanged",
    "write_replay_artifacts",
]
