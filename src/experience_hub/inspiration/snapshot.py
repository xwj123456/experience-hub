"""Freeze ranked evidence into one bounded, replayable inspiration snapshot."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from experience_hub.clock import require_utc
from experience_hub.experiences.models import Temperature
from experience_hub.ids import IdGenerator
from experience_hub.inspiration.commands import StartInspirationRun
from experience_hub.inspiration.contracts import (
    ExperienceEvidenceReader,
    InboxEvidenceReader,
)
from experience_hub.inspiration.hashing import (
    hash_snapshot,
    snapshot_canonical_bytes,
    stable_evidence_key,
    truncate_utf8,
)
from experience_hub.inspiration.models import (
    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
    MAX_SNAPSHOT_ITEMS,
    MAX_SNAPSHOT_UTF8_BYTES,
    EvidenceCandidate,
    EvidenceSourceState,
    EvidenceSourceType,
    FrozenSnapshot,
    SnapshotItem,
)
from experience_hub.retrieval.contracts import PeekExperiences, SearchResult
from experience_hub.retrieval.tokenizer import query_cues
from experience_hub.sharing.queries import QuarantinedCapsuleEvidence
from experience_hub.storage.unit_of_work import UnitOfWork

_OWNED_SOURCE_ORDER = 0
_CAPSULE_SOURCE_ORDER = 1
_PLACEHOLDER_ITEM_ID = UUID(int=0)


def _query_text(request: StartInspirationRun) -> str:
    return (
        request.goal
        if not request.context
        else f"{request.goal}\n{request.context}"
    )


def _owned_source_state(temperature: Temperature) -> EvidenceSourceState:
    states = {
        Temperature.HOT: EvidenceSourceState.HOT,
        Temperature.WARM: EvidenceSourceState.WARM,
        Temperature.COLD: EvidenceSourceState.COLD,
    }
    try:
        return states[temperature]
    except KeyError as error:
        raise RuntimeError(
            "Peek returned a non-retrievable experience state"
        ) from error


def _owned_candidates(result: SearchResult) -> tuple[EvidenceCandidate, ...]:
    candidates: list[EvidenceCandidate] = []
    for hit in result.hits:
        experience = hit.experience
        excerpt = (
            experience.body
            if hit.expanded and experience.body is not None
            else ""
        )
        candidates.append(
            EvidenceCandidate(
                source_type=EvidenceSourceType.EXPERIENCE,
                source_id=experience.experience_id,
                source_version_id=experience.version_id,
                source_state=_owned_source_state(experience.temperature),
                source_trust=experience.source_trust,
                relevance=hit.ranking_relevance,
                summary=experience.summary,
                mechanism=experience.mechanism,
                applicability=experience.applicability,
                tags=experience.tags,
                falsifiers=experience.falsifiers,
                excerpt=truncate_utf8(
                    excerpt,
                    MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
                ),
                content_hash=experience.content_hash,
            )
        )
    return tuple(candidates)


def _capsule_candidate(
    evidence: QuarantinedCapsuleEvidence,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        source_type=EvidenceSourceType.CAPSULE,
        source_id=evidence.source_id,
        source_version_id=evidence.source_version_id,
        source_state=EvidenceSourceState.QUARANTINED,
        source_trust=evidence.source_trust,
        relevance=evidence.ranking_relevance,
        summary=evidence.summary,
        mechanism=evidence.mechanism,
        applicability=evidence.applicability,
        tags=evidence.tags,
        falsifiers=evidence.falsifiers,
        excerpt=truncate_utf8(
            evidence.excerpt,
            MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
        ),
        content_hash=evidence.content_hash,
    )


def _candidate_sort_key(
    candidate: EvidenceCandidate,
) -> tuple[float, int, bytes, bytes]:
    source_order = (
        _OWNED_SOURCE_ORDER
        if candidate.source_type is EvidenceSourceType.EXPERIENCE
        else _CAPSULE_SOURCE_ORDER
    )
    return (
        -candidate.relevance,
        source_order,
        candidate.source_id.bytes,
        candidate.source_version_id.bytes,
    )


def _snapshot_item(
    candidate: EvidenceCandidate,
    *,
    snapshot_item_id: UUID,
    run_id: UUID,
    rank: int,
    captured_at: datetime,
    excerpt: str | None = None,
) -> SnapshotItem:
    return SnapshotItem(
        snapshot_item_id=snapshot_item_id,
        stable_evidence_key=stable_evidence_key(
            source_type=candidate.source_type,
            source_id=candidate.source_id,
            source_version_id=candidate.source_version_id,
            content_hash=candidate.content_hash,
        ),
        run_id=run_id,
        source_type=candidate.source_type,
        source_id=candidate.source_id,
        source_version_id=candidate.source_version_id,
        source_state=candidate.source_state,
        source_trust=candidate.source_trust,
        rank=rank,
        summary=candidate.summary,
        mechanism=candidate.mechanism,
        applicability=candidate.applicability,
        tags=candidate.tags,
        falsifiers=candidate.falsifiers,
        excerpt=candidate.excerpt if excerpt is None else excerpt,
        content_hash=candidate.content_hash,
        captured_at=captured_at,
    )


def _fits(items: tuple[SnapshotItem, ...]) -> bool:
    return len(snapshot_canonical_bytes(items)) <= MAX_SNAPSHOT_UTF8_BYTES


def _largest_fitting_excerpt(
    *,
    existing: tuple[SnapshotItem, ...],
    item: SnapshotItem,
) -> str:
    low = 0
    high = len(item.excerpt)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = item.model_copy(
            update={"excerpt": item.excerpt[:middle]}
        )
        if _fits((*existing, candidate)):
            low = middle
        else:
            high = middle - 1
    return item.excerpt[:low]


class SnapshotBuilder:
    """Compose owned and opted-in evidence without opening a transaction."""

    def __init__(
        self,
        *,
        experience_reader: ExperienceEvidenceReader,
        inbox_reader: InboxEvidenceReader,
        id_generator: IdGenerator,
    ) -> None:
        self._experience_reader = experience_reader
        self._inbox_reader = inbox_reader
        self._id_generator = id_generator

    async def freeze(
        self,
        *,
        uow: UnitOfWork,
        request: StartInspirationRun,
        run_id: UUID,
        at: datetime,
    ) -> FrozenSnapshot:
        if not isinstance(request, StartInspirationRun):
            raise ValueError("request must be StartInspirationRun")
        if not isinstance(run_id, UUID):
            raise ValueError("run_id must be a UUID")
        observed_at = require_utc(at)
        text = _query_text(request)
        result = await self._experience_reader.peek(
            session=uow.session,
            query=PeekExperiences(
                owner_agent_id=request.owner_agent_id,
                query=text,
                mode=request.mode,
                limit=MAX_SNAPSHOT_ITEMS,
                content_budget_bytes=MAX_SNAPSHOT_UTF8_BYTES,
                expand_cold=True,
                per_hit_excerpt_bytes=MAX_SNAPSHOT_EXCERPT_UTF8_BYTES,
            ),
        )
        candidates = list(_owned_candidates(result))
        if request.include_inbox:
            capsule_evidence = (
                await self._inbox_reader.list_available_pending(
                    session=uow.session,
                    recipient_agent_id=request.owner_agent_id,
                    as_of=observed_at,
                    query_cues=query_cues(text),
                    mode=request.mode,
                    limit=MAX_SNAPSHOT_ITEMS,
                )
            )
            candidates.extend(
                _capsule_candidate(evidence) for evidence in capsule_evidence
            )
        candidates.sort(key=_candidate_sort_key)

        retained: list[SnapshotItem] = []
        for candidate in candidates[:MAX_SNAPSHOT_ITEMS]:
            rank = len(retained) + 1
            provisional = _snapshot_item(
                candidate,
                snapshot_item_id=_PLACEHOLDER_ITEM_ID,
                run_id=run_id,
                rank=rank,
                captured_at=observed_at,
            )
            existing = tuple(retained)
            if _fits((*existing, provisional)):
                retained.append(
                    _snapshot_item(
                        candidate,
                        snapshot_item_id=self._id_generator.new(),
                        run_id=run_id,
                        rank=rank,
                        captured_at=observed_at,
                    )
                )
                continue

            metadata_only = provisional.model_copy(update={"excerpt": ""})
            if not _fits((*existing, metadata_only)):
                break
            fitted_excerpt = _largest_fitting_excerpt(
                existing=existing,
                item=provisional,
            )
            retained.append(
                _snapshot_item(
                    candidate,
                    snapshot_item_id=self._id_generator.new(),
                    run_id=run_id,
                    rank=rank,
                    captured_at=observed_at,
                    excerpt=fitted_excerpt,
                )
            )
            break

        items = tuple(retained)
        return FrozenSnapshot(
            run_id=run_id,
            items=items,
            snapshot_hash=hash_snapshot(items),
            frozen_at=observed_at,
        )


__all__ = ["SnapshotBuilder"]
