"""Real lifecycle and owned-source boundaries of native Passport export."""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select, update
from tests.passport_export_fixtures import ExportSeed, create_export_experience
from tests.passport_service_fixtures import (
    MISSING,
    OTHER,
    OWNER,
    PassportStack,
)
from tests.passport_service_fixtures import (
    passport_stack as passport_stack,
)

from experience_hub.domain import CommandContext, CommandRequest
from experience_hub.experiences.contracts import CreateExperience
from experience_hub.experiences.models import (
    ExperienceKind,
    ExperienceOrigin,
    Temperature,
    VersionContent,
)
from experience_hub.passports import PassportDeclarationV1
from experience_hub.passports.errors import PassportError
from experience_hub.storage import StoredResponse, UnitOfWork
from experience_hub.storage.database import payload_rewrite_guard
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    IdempotencyRecordRow,
)
from experience_hub.storage.validation import SourceIntegrityError

DECLARATION = PassportDeclarationV1(
    input_sanitized=True,
    profile_id="synthetic-edge-v1",
    sharing_authorized=True,
)


async def _create_local(stack: PassportStack, *, body: str, key: str) -> ExportSeed:
    content = VersionContent(
        body=body,
        summary="Synthetic native export boundary",
        mechanism="Local source validation precedes portable export.",
        tags=(),
        applicability=(),
        evidence=(),
        falsifiers=(),
    )
    value = CreateExperience(
        owner_agent_id=OWNER,
        kind=ExperienceKind.PROCEDURAL,
        content=content,
        importance=0.05,
        confidence=0.1,
    )
    request = CommandRequest(
        caller_scope=f"agent:{OWNER}",
        operation_scope="experience.create",
        idempotency_key=key,
        method="POST",
        route_template="/v1/agents/{agent_id}/experiences",
        path_parameters={"agent_id": OWNER},
        body={
            "kind": value.kind,
            "content": content,
            "importance": value.importance,
            "confidence": value.confidence,
            "links": (),
        },
    )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await stack.container.experience_service.create(
            uow=uow, command=value, command_context=command
        )

    result = await stack.container.command_executor.execute(request, handler)
    assert result.status_code == 201
    data = json.loads(result.body)["data"]
    return ExportSeed(
        OWNER,
        UUID(data["experience_id"]),
        UUID(data["version_id"]),
        data["content_hash"],
    )


async def _evaluate_lifecycle(stack: PassportStack, *, key: str) -> None:
    evaluated_at = stack.clock.now()
    request = CommandRequest(
        caller_scope="system:local",
        operation_scope="lifecycle.run",
        idempotency_key=key,
        method="POST",
        route_template="/v1/lifecycle:run",
        body={"evaluated_at": evaluated_at, "mode": "manual"},
    )

    async def handler(uow: UnitOfWork, command: CommandContext) -> StoredResponse:
        return await stack.container.lifecycle_service.run(
            uow=uow,
            evaluated_at=evaluated_at,
            command=command,
            mode="manual",
        )

    assert (
        await stack.container.command_executor.execute(request, handler)
    ).status_code == 200


async def test_legally_archived_local_experience_requires_restore_before_export(
    passport_stack: PassportStack,
) -> None:
    # Removing the export archive guard must fail this real, replayable lifecycle.
    seed = await _create_local(
        passport_stack, body="Synthetic aged procedural source.", key="aged-local"
    )
    for ordinal, (days, expected) in enumerate(
        ((7, Temperature.WARM), (1, Temperature.COLD), (91, Temperature.ARCHIVED))
    ):
        passport_stack.clock.advance(timedelta(days=days))
        await _evaluate_lifecycle(passport_stack, key=f"aged-cycle-{ordinal}")
        async with passport_stack.container.database.read_session() as session:
            state = await session.get(ExperienceStateRow, seed.experience_id)
            assert state is not None and state.temperature is expected

    async with passport_stack.container.database.read_session() as session:
        identity = await session.get(ExperienceRow, seed.experience_id)
        assert identity is not None and identity.origin is ExperienceOrigin.LOCAL
        event_types = tuple(
            await session.scalars(
                select(DomainEventRow.event_type)
                .where(DomainEventRow.aggregate_id == seed.experience_id)
                .order_by(DomainEventRow.sequence)
            )
        )
        assert event_types.count("experience.lifecycle_evaluated") == 3
        assert event_types.count("experience.archived") == 1
        await passport_stack.container.source_validator.validate(session)
        with pytest.raises(PassportError) as caught:
            await passport_stack.container.passport_export_service.export(
                session=session,
                owner_agent_id=OWNER,
                experience_id=seed.experience_id,
                version_id=seed.version_id,
                declaration=DECLARATION,
            )
        assert caught.value.code == "passport_restore_required"
        assert caught.value.status_code == 409
    report = await passport_stack.container.projection_manager.verify(
        passport_stack.container.database
    )
    assert report.matches


async def test_corrupt_owned_native_payload_has_fixed_safe_export_error(
    passport_stack: PassportStack,
) -> None:
    # Removing owned payload authentication or its safe wrapper must fail.
    seed = await create_export_experience(passport_stack.container, OWNER)
    async with passport_stack.container.database.read_session() as session:
        await passport_stack.container.source_validator.validate(session)
    async with passport_stack.container.database.transaction(immediate=True) as uow:
        connection = await uow.session.connection()
        with payload_rewrite_guard(connection):
            await uow.session.execute(
                update(ExperiencePayloadRow)
                .where(ExperiencePayloadRow.version_id == seed.version_id)
                .values(payload=b"synthetic corrupted payload")
            )
    async with passport_stack.container.database.read_session() as session:
        with pytest.raises(SourceIntegrityError) as caught:
            await passport_stack.container.passport_export_service.export(
                session=session,
                owner_agent_id=OWNER,
                experience_id=seed.experience_id,
                version_id=seed.version_id,
                declaration=DECLARATION,
            )
    assert caught.value.mismatch_key == "passport_export"
    assert caught.value.code == "source_integrity_error"
    message = str(caught.value)
    assert str(OWNER) not in message and str(seed.experience_id) not in message
    assert str(seed.version_id) not in message
    assert "synthetic corrupted payload" not in message
    assert "sqlite" not in message.lower() and "/" not in message


async def test_secret_looking_native_body_is_scanned_without_echo(
    passport_stack: PassportStack,
) -> None:
    # Skipping retained-content scanning or echoing its input must fail.
    probe = "ghp_" + "a" * 36
    seed = await _create_local(passport_stack, body=probe, key="synthetic-sensitive")
    async with passport_stack.container.database.read_session() as session:
        await passport_stack.container.source_validator.validate(session)
        with pytest.raises(PassportError) as caught:
            await passport_stack.container.passport_export_service.export(
                session=session,
                owner_agent_id=OWNER,
                experience_id=seed.experience_id,
                version_id=None,
                declaration=DECLARATION,
            )
    assert caught.value.code == "passport_sensitive_content"
    assert caught.value.details == {
        "matches": [{"rule_id": "github_token", "position": "subject.content.body"}]
    }
    assert probe not in str(caught.value)
    assert probe not in repr(caught.value.details)


async def test_foreign_selected_version_and_missing_version_are_indistinguishable(
    passport_stack: PassportStack,
) -> None:
    # Removing the selected-version owner/experience binding must fail.
    own = await create_export_experience(passport_stack.container, OWNER)
    foreign = await create_export_experience(passport_stack.container, OTHER)
    errors = []
    async with passport_stack.container.database.read_session() as session:
        await passport_stack.container.source_validator.validate(session)
        for version_id in (foreign.version_id, MISSING):
            with pytest.raises(PassportError) as caught:
                await passport_stack.container.passport_export_service.export(
                    session=session,
                    owner_agent_id=OWNER,
                    experience_id=own.experience_id,
                    version_id=version_id,
                    declaration=DECLARATION,
                )
            errors.append(
                (
                    caught.value.code,
                    caught.value.status_code,
                    str(caught.value),
                    caught.value.details,
                )
            )
    assert errors[0] == errors[1]
    assert errors[0][:2] == ("passport_not_found", 404)
    assert str(foreign.version_id) not in repr(errors)
    assert str(OTHER) not in repr(errors)


@pytest.mark.parametrize("field", ("input_sanitized", "sharing_authorized"))
async def test_export_revalidates_forged_false_declaration(
    passport_stack: PassportStack,
    field: str,
) -> None:
    # Trusting frozen declarations without boundary revalidation must fail.
    seed = await create_export_experience(passport_stack.container, OWNER)
    forged = DECLARATION.model_copy(update={field: False})
    async with passport_stack.container.database.read_session() as session:
        before = await session.scalar(select(func.count()).select_from(DomainEventRow))
        receipts_before = await session.scalar(
            select(func.count()).select_from(IdempotencyRecordRow)
        )
        with pytest.raises(PassportError) as caught:
            await passport_stack.container.passport_export_service.export(
                session=session,
                owner_agent_id=OWNER,
                experience_id=seed.experience_id,
                version_id=None,
                declaration=forged,
            )
        assert (
            await session.scalar(select(func.count()).select_from(DomainEventRow))
            == before
        )
        assert (
            await session.scalar(select(func.count()).select_from(IdempotencyRecordRow))
            == receipts_before
        )
    assert caught.value.code == "passport_invalid"
    assert caught.value.details == {}
