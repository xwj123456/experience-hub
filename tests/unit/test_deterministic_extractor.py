from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes, sha256_hex
from experience_hub.capture import (
    AdapterDescriptorV1,
    CandidateDraftV1,
    CandidateSignalV1,
    CapturedEvidenceV1,
    DeterministicSignalExtractor,
    EvidencePointerV1,
    OutcomeStatus,
    PreparedCaptureV1,
    SanitizationDeclarationV1,
    TrajectoryBundleV1,
    TrajectoryField,
    TrajectoryStepV1,
    extractor_configuration_document,
    extractor_configuration_hash,
    hash_trajectory_manifest,
    trajectory_manifest_document,
)
from experience_hub.domain import TypedEvidence
from experience_hub.experiences import (
    ExperienceKind,
    VersionContent,
    encode_version_content,
)

OWNER = UUID("10000000-0000-4000-8000-000000000001")
STARTED_AT = datetime(2026, 7, 22, 1, 0, tzinfo=UTC)


def _signal(
    *,
    body: str = "Retry only after checking whether the operation committed.",
    evidence: tuple[EvidencePointerV1, ...] = (),
) -> CandidateSignalV1:
    return CandidateSignalV1(
        kind=ExperienceKind.PROCEDURAL,
        body=body,
        summary="Check commit state before retrying.",
        mechanism="A state check prevents duplicate external effects.",
        tags=("retry", "audit", "retry"),
        applicability=("timeouts", "ambiguous failure"),
        evidence=evidence,
        falsifiers=("provider guarantees idempotency", "no state is retained"),
    )


def _step(
    ordinal: int,
    *,
    observation: str | None = None,
    action: str | None = None,
    outcome: str | None = None,
    signal: CandidateSignalV1 | None = None,
) -> TrajectoryStepV1:
    return TrajectoryStepV1(
        step_id=f"step-{ordinal}",
        ordinal=ordinal,
        occurred_at=STARTED_AT + timedelta(minutes=ordinal),
        observation=observation or f"Observed state {ordinal}.",
        action=action or f"Performed action {ordinal}.",
        outcome=outcome or f"Recorded outcome {ordinal}.",
        status=OutcomeStatus.SUCCEEDED,
        candidate_signal=signal,
    )


def _unchecked_bundle(*steps: TrajectoryStepV1) -> TrajectoryBundleV1:
    unchecked = TrajectoryBundleV1.model_construct(
        schema_version=1,
        adapter=AdapterDescriptorV1(kind="generic_jsonl", version=1),
        owner_agent_id=OWNER,
        trajectory_id="trajectory-001",
        source_started_at=STARTED_AT,
        source_completed_at=STARTED_AT + timedelta(hours=1),
        sanitization=SanitizationDeclarationV1(
            profile_id="local-sanitized-v1",
            input_sanitized=True,
        ),
        steps=steps,
        manifest_hash="",
    )
    return unchecked.model_copy(
        update={"manifest_hash": hash_trajectory_manifest(unchecked)}
    )


def _bundle(*steps: TrajectoryStepV1) -> TrajectoryBundleV1:
    unchecked = _unchecked_bundle(*steps)
    return TrajectoryBundleV1.model_validate(unchecked.model_dump(mode="python"))


def signalled_bundle() -> TrajectoryBundleV1:
    return _bundle(
        _step(1),
        _step(
            2,
            signal=_signal(
                evidence=(
                    EvidencePointerV1(
                        step_id="step-2",
                        field=TrajectoryField.OUTCOME,
                    ),
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.OBSERVATION,
                    ),
                )
            ),
        ),
    )


def expected_version_content(bundle: TrajectoryBundleV1) -> VersionContent:
    return VersionContent(
        body="Retry only after checking whether the operation committed.",
        summary="Check commit state before retrying.",
        mechanism="A state check prevents duplicate external effects.",
        tags=("audit", "retry"),
        applicability=("ambiguous failure", "timeouts"),
        evidence=(
            TypedEvidence(
                type="trajectory_field",
                id=(
                    f"{bundle.manifest_hash}:step-1:"
                    f"{TrajectoryField.OBSERVATION.value}"
                ),
            ),
            TypedEvidence(
                type="trajectory_field",
                id=(
                    f"{bundle.manifest_hash}:step-2:"
                    f"{TrajectoryField.OUTCOME.value}"
                ),
            ),
        ),
        falsifiers=("no state is retained", "provider guarantees idempotency"),
    )


def _prepared_capture(
    bundle: TrajectoryBundleV1 | None = None,
) -> PreparedCaptureV1:
    retained_bundle = signalled_bundle() if bundle is None else bundle
    return PreparedCaptureV1(
        bundle=retained_bundle,
        manifest_json=canonical_json_bytes(
            trajectory_manifest_document(retained_bundle)
        ),
        candidates=DeterministicSignalExtractor().extract(retained_bundle),
    )


def _content_for_evidence(
    draft: CandidateDraftV1,
    evidence: tuple[CapturedEvidenceV1, ...],
) -> VersionContent:
    return VersionContent(
        body=draft.content.body,
        summary=draft.content.summary,
        mechanism=draft.content.mechanism,
        tags=draft.content.tags,
        applicability=draft.content.applicability,
        evidence=tuple(
            TypedEvidence(
                type="trajectory_field",
                id=(
                    f"{draft.source_manifest_hash}:{item.step_id}:"
                    f"{item.field.value}"
                ),
            )
            for item in evidence
        ),
        falsifiers=draft.content.falsifiers,
    )


def _candidate_for_evidence(
    draft: CandidateDraftV1,
    evidence: tuple[CapturedEvidenceV1, ...],
) -> CandidateDraftV1:
    content = _content_for_evidence(draft, evidence)
    return CandidateDraftV1(
        source_manifest_hash=draft.source_manifest_hash,
        kind=draft.kind,
        content=content,
        content_hash=encode_version_content(
            kind=draft.kind,
            content=content,
        ).content_hash,
        evidence=evidence,
        extractor_kind=draft.extractor_kind,
        extractor_configuration_hash=draft.extractor_configuration_hash,
    )


def test_signal_extractor_maps_content_without_rewriting_it() -> None:
    bundle = signalled_bundle()
    drafts = DeterministicSignalExtractor().extract(bundle)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.kind is ExperienceKind.PROCEDURAL
    assert draft.content == expected_version_content(bundle)
    assert draft.source_manifest_hash == bundle.manifest_hash
    assert draft.evidence == (
        CapturedEvidenceV1(
            step_id="step-1",
            field=TrajectoryField.OBSERVATION,
            excerpt="Observed state 1.",
            source_hash=sha256_hex(b"Observed state 1."),
            excerpt_hash=sha256_hex(b"Observed state 1."),
        ),
        CapturedEvidenceV1(
            step_id="step-2",
            field=TrajectoryField.OUTCOME,
            excerpt="Recorded outcome 2.",
            source_hash=sha256_hex(b"Recorded outcome 2."),
            excerpt_hash=sha256_hex(b"Recorded outcome 2."),
        ),
    )
    assert len(draft.evidence) == 2
    assert all(len(item.excerpt.encode("utf-8")) <= 512 for item in draft.evidence)
    assert draft.extractor_kind == "deterministic_signal_v1"
    assert draft.extractor_configuration_hash == extractor_configuration_hash()


def test_signal_extractor_uses_the_existing_canonical_content_hash() -> None:
    draft = DeterministicSignalExtractor().extract(signalled_bundle())[0]

    assert draft.content_hash == encode_version_content(
        kind=draft.kind,
        content=draft.content,
    ).content_hash


def test_extractor_configuration_document_is_exact_and_canonically_hashed() -> None:
    expected = {
        "kind": "deterministic_signal_v1",
        "max_evidence_items": 8,
        "max_excerpt_utf8_bytes": 512,
        "version": 1,
    }

    assert extractor_configuration_document() == expected
    assert extractor_configuration_hash() == sha256_hex(canonical_json_bytes(expected))


def test_signal_extractor_rejects_an_unknown_evidence_step_defensively() -> None:
    bundle = _unchecked_bundle(
        _step(
            1,
            signal=_signal(
                evidence=(
                    EvidencePointerV1(
                        step_id="unknown-step",
                        field=TrajectoryField.OUTCOME,
                    ),
                )
            ),
        )
    )

    with pytest.raises(ValueError, match="unknown step"):
        DeterministicSignalExtractor().extract(bundle)


def test_signal_extractor_deduplicates_and_canonically_orders_pointers() -> None:
    bundle = _bundle(
        _step(1),
        _step(
            2,
            signal=_signal(
                evidence=(
                    EvidencePointerV1(
                        step_id="step-2",
                        field=TrajectoryField.OUTCOME,
                    ),
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.OUTCOME,
                    ),
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.ACTION,
                    ),
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.OBSERVATION,
                    ),
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.ACTION,
                    ),
                )
            ),
        ),
    )

    evidence = DeterministicSignalExtractor().extract(bundle)[0].evidence

    assert tuple((item.step_id, item.field) for item in evidence) == (
        ("step-1", TrajectoryField.OBSERVATION),
        ("step-1", TrajectoryField.ACTION),
        ("step-1", TrajectoryField.OUTCOME),
        ("step-2", TrajectoryField.OUTCOME),
    )


def test_signal_extractor_keeps_only_first_eight_canonical_pointers() -> None:
    reversed_pointers = tuple(
        EvidencePointerV1(step_id=f"step-{ordinal}", field=field)
        for ordinal in range(3, 0, -1)
        for field in reversed(tuple(TrajectoryField))
    )
    bundle = _bundle(
        _step(1),
        _step(2),
        _step(3, signal=_signal(evidence=reversed_pointers)),
    )

    evidence = DeterministicSignalExtractor().extract(bundle)[0].evidence

    assert tuple((item.step_id, item.field) for item in evidence) == tuple(
        (f"step-{ordinal}", field)
        for ordinal in range(1, 4)
        for field in TrajectoryField
    )[:8]


def test_signal_extractor_truncates_multibyte_excerpts_on_utf8_boundaries() -> None:
    full_observation = "界" * 171 + "tail"
    bundle = _bundle(
        _step(
            1,
            observation=full_observation,
            signal=_signal(
                evidence=(
                    EvidencePointerV1(
                        step_id="step-1",
                        field=TrajectoryField.OBSERVATION,
                    ),
                )
            ),
        )
    )

    item = DeterministicSignalExtractor().extract(bundle)[0].evidence[0]

    assert item.excerpt == "界" * 170
    assert len(item.excerpt.encode("utf-8")) == 510
    assert item.source_hash == sha256_hex(full_observation.encode("utf-8"))
    assert item.excerpt_hash == sha256_hex(item.excerpt.encode("utf-8"))


def test_signal_extractor_deduplicates_and_retains_first_signal_order() -> None:
    repeated_signal = _signal(
        body="Retain the first occurrence of this candidate.",
        evidence=(
            EvidencePointerV1(
                step_id="step-1",
                field=TrajectoryField.OBSERVATION,
            ),
        )
    )
    bundle = _bundle(
        _step(1),
        _step(2, signal=repeated_signal),
        _step(3, signal=_signal(body="Retain this distinct second candidate.")),
        _step(4, signal=repeated_signal),
    )

    drafts = DeterministicSignalExtractor().extract(bundle)

    assert tuple(draft.content.body for draft in drafts) == (
        "Retain the first occurrence of this candidate.",
        "Retain this distinct second candidate.",
    )


def test_signal_extractor_returns_empty_tuple_without_signals() -> None:
    bundle = _bundle(_step(1), _step(2))

    assert DeterministicSignalExtractor().extract(bundle) == ()


@pytest.mark.parametrize(
    "field",
    ("content_hash", "extractor_configuration_hash"),
)
def test_candidate_draft_rejects_a_stale_reconstructed_hash(field: str) -> None:
    draft = DeterministicSignalExtractor().extract(signalled_bundle())[0]
    values = draft.model_dump(mode="python")
    values[field] = "0" * 64

    with pytest.raises(ValidationError, match="hash"):
        CandidateDraftV1.model_validate(values)


def test_prepared_capture_binds_manifest_and_candidates_to_bundle() -> None:
    prepared = _prepared_capture()

    assert prepared.manifest_json == canonical_json_bytes(
        trajectory_manifest_document(prepared.bundle)
    )
    with pytest.raises(ValidationError, match="manifest"):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=b"{}",
            candidates=prepared.candidates,
        )


def test_prepared_capture_rejects_a_forged_excerpt() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    first = draft.evidence[0]
    forged_first = CapturedEvidenceV1(
        step_id=first.step_id,
        field=first.field,
        excerpt=f"{first.excerpt} forged",
        source_hash=first.source_hash,
        excerpt_hash=first.excerpt_hash,
    )
    forged = _candidate_for_evidence(
        draft,
        (forged_first, *draft.evidence[1:]),
    )

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_a_forged_excerpt_hash() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    first = draft.evidence[0]
    forged_first = CapturedEvidenceV1(
        step_id=first.step_id,
        field=first.field,
        excerpt=first.excerpt,
        source_hash=first.source_hash,
        excerpt_hash="0" * 64,
    )
    forged = _candidate_for_evidence(
        draft,
        (forged_first, *draft.evidence[1:]),
    )

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_forged_evidence_order() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    forged = _candidate_for_evidence(draft, tuple(reversed(draft.evidence)))

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_duplicate_evidence() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    forged = _candidate_for_evidence(
        draft,
        (*draft.evidence, draft.evidence[0]),
    )

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_evidence_from_a_missing_source() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    first = draft.evidence[0]
    missing = CapturedEvidenceV1(
        step_id="missing-step",
        field=first.field,
        excerpt=first.excerpt,
        source_hash=first.source_hash,
        excerpt_hash=first.excerpt_hash,
    )
    forged = _candidate_for_evidence(draft, (missing, *draft.evidence[1:]))

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_forged_candidate_content_evidence() -> None:
    prepared = _prepared_capture()
    draft = prepared.candidates[0]
    source_step = prepared.bundle.steps[0]
    replacement = CapturedEvidenceV1(
        step_id=source_step.step_id,
        field=TrajectoryField.ACTION,
        excerpt=source_step.action,
        source_hash=sha256_hex(source_step.action.encode("utf-8")),
        excerpt_hash=sha256_hex(source_step.action.encode("utf-8")),
    )
    forged = _candidate_for_evidence(draft, (replacement,))

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )


def test_prepared_capture_rejects_forged_candidate_order() -> None:
    bundle = _bundle(
        _step(1),
        _step(2, signal=_signal(body="First candidate.")),
        _step(3, signal=_signal(body="Second candidate.")),
    )
    prepared = _prepared_capture(bundle)
    assert len(prepared.candidates) == 2

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=tuple(reversed(prepared.candidates)),
        )


def test_prepared_capture_rejects_a_forged_candidate_hash() -> None:
    prepared = _prepared_capture()
    forged = prepared.candidates[0].model_copy(
        update={"content_hash": "0" * 64}
    )

    with pytest.raises(ValidationError):
        PreparedCaptureV1(
            bundle=prepared.bundle,
            manifest_json=prepared.manifest_json,
            candidates=(forged,),
        )
