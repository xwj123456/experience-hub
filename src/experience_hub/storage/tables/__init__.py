"""SQLAlchemy metadata for durable core tables."""

from experience_hub.storage.tables.base import Base
from experience_hub.storage.tables.core import (
    AgentRow,
    DomainEventRow,
    IdempotencyRecordRow,
    LifecycleLeaseRow,
    ProjectionVersionRow,
)
from experience_hub.storage.tables.experiences import (
    ExperienceLinkRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceTermRow,
    ExperienceVersionRow,
)
from experience_hub.storage.tables.inspiration import (
    IdeaAdoptionRecordRow,
    IdeaOccurrenceRow,
    IdeaStateRow,
    InspirationIdeaRow,
    InspirationRunRow,
    InspirationRunStateRow,
    InspirationSnapshotItemRow,
    MechanismIncubationRow,
)
from experience_hub.storage.tables.sharing import (
    AdoptionRecordRow,
    AgentReputationRow,
    CapsuleFeedbackRow,
    CapsuleStateRow,
    ExperienceCapsuleRow,
    InboxItemRow,
    SubscriptionRow,
    TopicRow,
)

__all__ = [
    "AdoptionRecordRow",
    "AgentRow",
    "AgentReputationRow",
    "Base",
    "CapsuleFeedbackRow",
    "CapsuleStateRow",
    "DomainEventRow",
    "ExperienceCapsuleRow",
    "ExperienceLinkRow",
    "ExperiencePayloadRow",
    "ExperienceRow",
    "ExperienceStateRow",
    "ExperienceTermRow",
    "ExperienceVersionRow",
    "IdeaAdoptionRecordRow",
    "IdeaOccurrenceRow",
    "IdeaStateRow",
    "IdempotencyRecordRow",
    "InboxItemRow",
    "InspirationIdeaRow",
    "InspirationRunRow",
    "InspirationRunStateRow",
    "InspirationSnapshotItemRow",
    "LifecycleLeaseRow",
    "MechanismIncubationRow",
    "ProjectionVersionRow",
    "SubscriptionRow",
    "TopicRow",
]
