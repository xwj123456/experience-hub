import json
from uuid import UUID

import pytest
from tests.passport_export_fixtures import create_export_experience
from tests.passport_service_fixtures import (
    OTHER,
    OWNER,
    PassportStack,
    decide_passport,
    import_passport,
    prepared_passport,
    result_id,
)
from tests.passport_service_fixtures import passport_stack as passport_stack

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.experiences.models import ExperienceOrigin, VersionContent
from experience_hub.passports.codec import (
    VerifiedPassportV1,
    build_passport_document,
    encode_passport_document,
    verify_passport_bytes,
)
from experience_hub.passports.contracts import (
    PassportDeclarationV1,
    PassportHopV1,
    PassportProvenanceV1,
)
from experience_hub.passports.errors import PassportError
from experience_hub.passports.export import PassportExportService

DECLARATION = PassportDeclarationV1(
    input_sanitized=True, profile_id="synthetic-v1", sharing_authorized=True
)


async def _adopt(stack: PassportStack, prepared: VerifiedPassportV1 | None = None):
    imported = await import_passport(stack, prepared=prepared)
    result = await decide_passport(stack, result_id(imported))
    assert result.status_code == 200
    data = json.loads(result.body)["data"]
    return UUID(data["adoption_id"]), UUID(data["experience"]["experience_id"])


async def _export(stack: PassportStack, experience: UUID, adoption: UUID | None):
    async with stack.container.database.read_session() as session:
        return await PassportExportService().export(
            session=session,
            owner_agent_id=OWNER,
            experience_id=experience,
            version_id=None,
            declaration=DECLARATION,
            parent_adoption_id=adoption,
        )


@pytest.mark.parametrize("reuse_local", [False, True])
async def test_reexport_inherits_closed_evidence_and_appends_explicit_parent(
    passport_stack: PassportStack, reuse_local: bool
) -> None:
    if reuse_local:
        await create_export_experience(passport_stack.container, OWNER)
    parent = prepared_passport()
    adoption, experience = await _adopt(passport_stack, parent)
    output = await _export(passport_stack, experience, adoption)
    assert output.document.evidence_snapshots == parent.document.evidence_snapshots
    assert (
        output.document.provenance.origin_fingerprint
        == parent.document.provenance.origin_fingerprint
    )
    assert output.document.provenance.hops[:-1] == parent.document.provenance.hops
    final = output.document.provenance.hops[-1]
    assert final.source_agent_id == OWNER
    assert final.source_experience_id == experience
    assert final.parent_passport_hash == parent.document.passport_hash
    assert final.source_origin is (
        ExperienceOrigin.LOCAL if reuse_local else ExperienceOrigin.ADOPTED_PASSPORT
    )
    assert output.report.semantic_assessment == "not_assessed"


async def test_adopted_passport_cannot_start_a_new_root(
    passport_stack: PassportStack,
) -> None:
    _, experience = await _adopt(passport_stack)
    with pytest.raises(PassportError) as caught:
        await _export(passport_stack, experience, None)
    assert caught.value.code == "passport_derivation_unsupported"


@pytest.mark.parametrize("mismatch", ["foreign", "missing", "experience"])
async def test_parent_must_be_owned_and_bound_to_selected_experience(
    passport_stack: PassportStack, mismatch: str
) -> None:
    adoption, experience = await _adopt(passport_stack)
    if mismatch == "foreign":
        imported = await import_passport(passport_stack, owner=OTHER, key="foreign")
        result = await decide_passport(
            passport_stack, result_id(imported), owner=OTHER, key="foreign-decision"
        )
        adoption = result_id(result, "adoption_id")
    elif mismatch == "missing":
        adoption = UUID(int=99999)
    else:
        seed = await create_export_experience(
            passport_stack.container,
            OWNER,
            key="other-experience",
            content=prepared_passport("other-content").document.subject.content,
        )
        experience = seed.experience_id
    with pytest.raises(PassportError) as caught:
        await _export(passport_stack, experience, adoption)
    assert caught.value.code == (
        "passport_derivation_unsupported"
        if mismatch == "experience"
        else "passport_not_found"
    )
    assert caught.value.details == {}


async def test_reexport_checks_selected_version_not_current_content(
    passport_stack: PassportStack,
) -> None:
    adoption_id, experience = await _adopt(passport_stack)
    first = await _export(passport_stack, experience, adoption_id)
    await create_export_experience(
        passport_stack.container,
        OWNER,
        experience_id=experience,
        key="changed-version",
        content=VersionContent(
            body="Changed local body",
            summary="Changed summary",
            mechanism="Different mechanism",
            tags=(),
            applicability=(),
            evidence=(),
            falsifiers=(),
        ),
    )
    with pytest.raises(PassportError) as caught:
        await _export(passport_stack, experience, adoption_id)
    assert caught.value.code == "passport_derivation_unsupported"
    async with passport_stack.container.database.read_session() as session:
        historical = await PassportExportService().export(
            session=session,
            owner_agent_id=OWNER,
            experience_id=experience,
            version_id=first.document.subject.source_version_id,
            declaration=DECLARATION,
            parent_adoption_id=adoption_id,
        )
    assert historical.canonical_bytes == first.canonical_bytes


def _with_hops(parent: VerifiedPassportV1, hops: tuple[PassportHopV1, ...]):
    root = hops[0]
    document = build_passport_document(
        subject=parent.document.subject,
        evidence_snapshots=parent.document.evidence_snapshots,
        provenance=PassportProvenanceV1(
            scope="passport_transfers_only",
            hops=hops,
            origin_fingerprint=sha256_hex(
                canonical_json_bytes(
                    {
                        "source_agent_id": root.source_agent_id,
                        "content_hash": root.content_hash,
                    }
                )
            ),
        ),
        declaration=DECLARATION,
    )
    return verify_passport_bytes(encode_passport_document(document))


async def test_fifth_hop_is_refused_without_mutating_parent(
    passport_stack: PassportStack,
) -> None:
    parent = prepared_passport()
    final = parent.document.provenance.hops[0]
    hops = tuple(
        PassportHopV1(
            source_agent_id=UUID(int=100 + index),
            source_experience_id=UUID(int=200 + index),
            source_version_id=UUID(int=300 + index),
            source_origin=ExperienceOrigin.LOCAL,
            content_hash=final.content_hash,
            parent_passport_hash=None if index == 0 else "a" * 64,
        )
        for index in range(3)
    ) + (final.model_copy(update={"parent_passport_hash": "b" * 64}),)
    prepared = _with_hops(parent, hops)
    adoption, experience = await _adopt(passport_stack, prepared)
    with pytest.raises(PassportError) as caught:
        await _export(passport_stack, experience, adoption)
    assert caught.value.code == "passport_provenance_limit"
    assert prepared.document.provenance.hops == hops


async def test_reexport_cycle_cannot_be_hidden_by_reusing_local_content(
    passport_stack: PassportStack,
) -> None:
    local = await create_export_experience(passport_stack.container, OWNER)
    parent = prepared_passport()
    final = parent.document.provenance.hops[0]
    local_hop = PassportHopV1(
        source_agent_id=OWNER,
        source_experience_id=local.experience_id,
        source_version_id=local.version_id,
        source_origin=ExperienceOrigin.LOCAL,
        content_hash=local.content_hash,
        parent_passport_hash=None,
    )
    prepared = _with_hops(
        parent,
        (
            local_hop,
            final.model_copy(update={"parent_passport_hash": "a" * 64}),
        ),
    )
    adoption, experience = await _adopt(passport_stack, prepared)
    assert experience == local.experience_id
    with pytest.raises(PassportError) as caught:
        await _export(passport_stack, experience, adoption)
    assert caught.value.code == "passport_provenance_limit"
