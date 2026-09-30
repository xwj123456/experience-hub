"""Deterministic real capture/adoption seed for the public Passport example."""

import json
from pathlib import Path
from uuid import UUID

from sqlalchemy.engine import URL

from experience_hub.clock import FrozenClock
from experience_hub.config import Settings
from experience_hub.ids import SequenceIdGenerator
from experience_hub.passports import PassportDeclarationV1, VerifiedPassportV1
from experience_hub.passports.export import PassportExportService
from experience_hub.runtime import ApplicationRuntime
from experience_hub.storage.readonly import readonly_sqlite_session
from tests.integration.test_candidate_decisions import (
    IDS,
    NOW,
    OWNER_ID,
    DecisionStack,
    FailAt,
    _adopt,
    _capture_pending,
    _create_agent,
)


async def capture_example(path: Path) -> VerifiedPassportV1:
    runtime = ApplicationRuntime(
        Settings(database_url=URL.create("sqlite+aiosqlite", database=str(path))),
        clock=FrozenClock(NOW),
        ids=SequenceIdGenerator(IDS),
    )
    async with runtime.initialize(
        start_lifecycle_worker=False, recover_interrupted=False
    ) as container:
        await _create_agent(container, key="owner", name="Synthetic capture owner")
        await _create_agent(container, key="other", name="Synthetic other owner")
        stack = DecisionStack(container, FailAt())
        candidate = await _capture_pending(stack)
        adopted = await _adopt(stack, candidate_id=candidate, key="example-adopt")
        assert adopted.status_code == 200
        data = json.loads(adopted.body)["data"]
        async with container.database.read_session() as session:
            await container.source_validator.validate(session)
    async with readonly_sqlite_session(path) as session:
        return await PassportExportService().export(
            session=session,
            owner_agent_id=OWNER_ID,
            experience_id=UUID(data["resulting_experience_id"]),
            version_id=UUID(data["resulting_version_id"]),
            declaration=PassportDeclarationV1(
                input_sanitized=True,
                profile_id="synthetic-v1",
                sharing_authorized=True,
            ),
        )
