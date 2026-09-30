"""Small canonical mutation results; full quarantine content uses owner queries."""

from uuid import UUID

from experience_hub import canonical_json_bytes
from experience_hub.experiences.contracts import ExperienceRecord
from experience_hub.passports.contracts import PassportState
from experience_hub.storage.idempotency import StoredResponse


def passport_import_response(
    *,
    import_id: UUID,
    owner_agent_id: UUID,
    passport_hash: str,
    state: PassportState,
    status_code: int = 200,
) -> StoredResponse:
    return StoredResponse(
        status_code=status_code,
        body=canonical_json_bytes(
            {
                "data": {
                    "import_id": import_id,
                    "owner_agent_id": owner_agent_id,
                    "passport_hash": passport_hash,
                    "state": state,
                }
            }
        ),
    )


def passport_adoption_response(
    *,
    adoption_id: UUID,
    experience: ExperienceRecord,
    created: bool,
) -> StoredResponse:
    return StoredResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "data": {
                    "adoption_id": adoption_id,
                    "created": created,
                    "experience": {
                        "experience_id": experience.experience_id,
                        "owner_agent_id": experience.owner_agent_id,
                        "current_version_id": experience.current_version_id,
                        "current_content_hash": experience.current_content_hash,
                        "temperature": experience.temperature,
                    },
                }
            }
        ),
    )
