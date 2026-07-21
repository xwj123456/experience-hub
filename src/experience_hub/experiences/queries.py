"""Owner-scoped experience read models safe for cross-module sharing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from experience_hub.canonical import canonical_json_bytes
from experience_hub.domain import EventRegistry, TypedEvidence
from experience_hub.domain.commands import ReplayableCommandError
from experience_hub.experiences.content import decode_payload
from experience_hub.experiences.contracts import ShareableExperienceVersion
from experience_hub.experiences.events import (
    STATE_EXPERIENCE_EVENT_TYPES,
    register_experience_events,
)
from experience_hub.experiences.models import Temperature, VersionContent
from experience_hub.experiences.repository import (
    decode_and_verify_version,
    require_current_aggregate_head,
    snapshot_from_state_row,
)
from experience_hub.retrieval.contracts import (
    CandidateSelection,
    RetrievalCandidate,
    RetrievalRecord,
)
from experience_hub.retrieval.ranking import (
    CandidateMatch,
    raw_overlap,
    select_temperature_pools,
    temperature_pool_quota,
)
from experience_hub.retrieval.tokenizer import TermCue
from experience_hub.storage.tables import (
    DomainEventRow,
    ExperiencePayloadRow,
    ExperienceRow,
    ExperienceStateRow,
    ExperienceTermRow,
    ExperienceVersionRow,
)
from experience_hub.storage.validation import SourceIntegrityError


class ExperienceNotFoundError(ReplayableCommandError):
    def __init__(self) -> None:
        super().__init__(
            code="experience_not_found",
            message="Experience was not found",
            status_code=404,
        )


class ExperienceQuery:
    """Read selected immutable content with current owner-scoped lifecycle data."""

    def __init__(
        self,
        *,
        event_registry: EventRegistry | None = None,
        handled_event_types: frozenset[str] = STATE_EXPERIENCE_EVENT_TYPES,
    ) -> None:
        if event_registry is None:
            event_registry = EventRegistry()
            register_experience_events(event_registry)
        self._event_registry = event_registry
        self._handled_event_types = handled_event_types

    @staticmethod
    def _metadata_content(version: ExperienceVersionRow) -> VersionContent:
        try:
            tags = json.loads(version.tags)
            applicability = json.loads(version.applicability)
            evidence_values = json.loads(version.evidence)
            falsifiers = json.loads(version.falsifiers)
            if not all(
                isinstance(value, list)
                for value in (
                    tags,
                    applicability,
                    evidence_values,
                    falsifiers,
                )
            ):
                raise ValueError("Stored version metadata must use arrays")
            content = VersionContent(
                body="_",
                summary=version.summary,
                mechanism=version.mechanism,
                tags=tuple(tags),
                applicability=tuple(applicability),
                evidence=tuple(
                    TypedEvidence.model_validate(value)
                    for value in evidence_values
                ),
                falsifiers=tuple(falsifiers),
            )
            if (
                canonical_json_bytes(content.tags) != version.tags
                or canonical_json_bytes(content.applicability)
                != version.applicability
                or canonical_json_bytes(content.evidence) != version.evidence
                or canonical_json_bytes(content.falsifiers)
                != version.falsifiers
            ):
                raise ValueError("Stored version metadata is not canonical")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise SourceIntegrityError(
                f"Experience version {version.version_id} has invalid metadata"
            ) from error
        return content

    async def _retrieval_record_from_rows(
        self,
        *,
        session: AsyncSession,
        identity: ExperienceRow,
        version: ExperienceVersionRow,
        state: ExperienceStateRow,
        projection_event: DomainEventRow,
    ) -> RetrievalRecord:
        experience_id = identity.experience_id
        if (
            version.experience_id != experience_id
            or state.experience_id != experience_id
            or state.owner_agent_id != identity.owner_agent_id
            or state.current_version_id != version.version_id
            or state.current_content_hash != version.content_hash
            or projection_event.aggregate_type != "experience"
            or projection_event.aggregate_id != experience_id
            or projection_event.event_id != state.projection_event_id
        ):
            raise SourceIntegrityError(
                f"Owned experience {experience_id} has inconsistent current state"
            )
        await require_current_aggregate_head(
            session=session,
            experience_id=experience_id,
            projection_event=projection_event,
            event_registry=self._event_registry,
            handled_event_types=self._handled_event_types,
        )
        metadata = self._metadata_content(version)
        causal_times = [
            identity.created_at,
            version.created_at,
            projection_event.occurred_at,
            state.strength_updated_at,
            state.last_transition_at,
        ]
        causal_times.extend(
            value
            for value in (
                state.last_accessed_at,
                state.last_lifecycle_evaluated_at,
            )
            if value is not None
        )
        try:
            return RetrievalRecord(
                experience_id=experience_id,
                owner_agent_id=identity.owner_agent_id,
                kind=identity.kind,
                origin=identity.origin,
                created_at=identity.created_at,
                current_version_id=version.version_id,
                current_version_number=version.version_number,
                current_version_created_at=version.created_at,
                current_content_hash=version.content_hash,
                summary=metadata.summary,
                mechanism=metadata.mechanism,
                tags=metadata.tags,
                applicability=metadata.applicability,
                evidence=metadata.evidence,
                falsifiers=metadata.falsifiers,
                state=snapshot_from_state_row(state),
                projection_event_id=projection_event.event_id,
                latest_causal_at=max(causal_times),
            )
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Experience {experience_id} has invalid retrieval values"
            ) from error

    async def get_owned_retrieval_record(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
    ) -> RetrievalRecord | None:
        """Return one current aggregate without revealing foreign existence."""
        try:
            identity = await session.scalar(
                select(ExperienceRow).where(
                    ExperienceRow.owner_agent_id == owner_agent_id,
                    ExperienceRow.experience_id == experience_id,
                )
            )
            if identity is None:
                return None
            aggregate = (
                await session.execute(
                    select(
                        ExperienceVersionRow,
                        ExperienceStateRow,
                        DomainEventRow,
                    )
                    .select_from(ExperienceStateRow)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperienceStateRow.current_version_id,
                    )
                    .join(
                        DomainEventRow,
                        DomainEventRow.event_id
                        == ExperienceStateRow.projection_event_id,
                    )
                    .where(
                        ExperienceStateRow.experience_id == experience_id,
                        ExperienceStateRow.owner_agent_id == owner_agent_id,
                    )
                )
            ).one_or_none()
            if aggregate is None:
                raise SourceIntegrityError(
                    f"Owned experience {experience_id} has no current state"
                )
            version, state, projection_event = aggregate
            return await self._retrieval_record_from_rows(
                session=session,
                identity=identity,
                version=version,
                state=state,
                projection_event=projection_event,
            )
        except SourceIntegrityError:
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Experience {experience_id} has invalid retrieval sources"
            ) from error

    async def select_retrieval_candidates(
        self,
        *,
        session: AsyncSession,
        selection: CandidateSelection,
    ) -> tuple[RetrievalCandidate, ...]:
        """Select positive-overlap owner candidates by independent pools."""
        if not isinstance(selection, CandidateSelection):
            raise ValueError("selection must be a CandidateSelection")
        try:
            cue_document = canonical_json_bytes(
                [
                    {
                        "term": cue.term,
                        "term_kind": cue.term_kind,
                        "weight": cue.weight,
                    }
                    for cue in selection.query_cues
                ]
            ).decode("utf-8")
            quotas = {
                temperature: temperature_pool_quota(
                    selection.mode,
                    temperature,
                    selection.requested_limit,
                )
                for temperature in (
                    Temperature.HOT,
                    Temperature.WARM,
                    Temperature.COLD,
                )
            }
            bounded_rows = (
                await session.execute(
                    text(
                        """
                        WITH query_cues AS (
                            SELECT
                                json_extract(value, '$.term') AS term,
                                json_extract(value, '$.term_kind') AS term_kind,
                                CAST(
                                    json_extract(value, '$.weight') AS REAL
                                ) AS weight
                            FROM json_each(:query_cues)
                        ),
                        cue_overlaps AS (
                            SELECT
                                identity.experience_id AS experience_id,
                                state.owner_agent_id AS state_owner_agent_id,
                                state.temperature AS temperature,
                                query.term AS query_term,
                                query.term_kind AS query_term_kind,
                                MAX(
                                    MIN(query.weight, term.weight)
                                ) AS cue_overlap
                            FROM query_cues AS query
                            JOIN experience_terms AS term
                              ON term.term = query.term
                             AND (
                                (
                                    query.term_kind = 'word'
                                    AND term.term_kind IN (
                                        'word',
                                        'tag',
                                        'mechanism'
                                    )
                                )
                                OR (
                                    query.term_kind = 'tag'
                                    AND term.term_kind = 'tag'
                                )
                                OR (
                                    query.term_kind = 'mechanism'
                                    AND term.term_kind = 'mechanism'
                                )
                                OR (
                                    query.term_kind = 'char_trigram'
                                    AND term.term_kind = 'char_trigram'
                                )
                             )
                            JOIN experiences AS identity
                              ON identity.experience_id = term.experience_id
                            JOIN experience_state AS state
                              ON state.experience_id = identity.experience_id
                            WHERE identity.owner_agent_id = :owner_agent_id
                              AND state.temperature != 'archived'
                            GROUP BY
                                identity.experience_id,
                                state.owner_agent_id,
                                state.temperature,
                                query.term,
                                query.term_kind
                        ),
                        candidate_overlaps AS (
                            SELECT
                                experience_id,
                                state_owner_agent_id,
                                temperature,
                                SUM(cue_overlap) AS raw_overlap
                            FROM cue_overlaps
                            GROUP BY
                                experience_id,
                                state_owner_agent_id,
                                temperature
                            HAVING SUM(cue_overlap) > 0.0
                        ),
                        ranked_pools AS (
                            SELECT
                                experience_id,
                                state_owner_agent_id,
                                temperature,
                                raw_overlap,
                                ROW_NUMBER() OVER (
                                    PARTITION BY temperature
                                    ORDER BY
                                        raw_overlap DESC,
                                        experience_id ASC
                                ) AS pool_rank
                            FROM candidate_overlaps
                        )
                        SELECT /* bounded_candidate_stage */
                            experience_id,
                            state_owner_agent_id,
                            temperature,
                            raw_overlap
                        FROM ranked_pools
                        WHERE (
                            temperature = 'hot'
                            AND pool_rank <= :hot_quota
                        ) OR (
                            temperature = 'warm'
                            AND pool_rank <= :warm_quota
                        ) OR (
                            temperature = 'cold'
                            AND pool_rank <= :cold_quota
                        )
                        ORDER BY raw_overlap DESC, experience_id ASC
                        """
                    ),
                    {
                        "query_cues": cue_document,
                        "owner_agent_id": str(selection.owner_agent_id),
                        "hot_quota": quotas[Temperature.HOT],
                        "warm_quota": quotas[Temperature.WARM],
                        "cold_quota": quotas[Temperature.COLD],
                    },
                )
            ).all()
            matches: list[CandidateMatch] = []
            for (
                experience_id_value,
                state_owner_value,
                temperature_value,
                overlap_value,
            ) in bounded_rows:
                experience_id = UUID(str(experience_id_value))
                state_owner = UUID(str(state_owner_value))
                temperature = Temperature(str(temperature_value))
                if state_owner != selection.owner_agent_id:
                    raise SourceIntegrityError(
                        "Owned candidate projection has an inconsistent owner"
                    )
                matches.append(
                    CandidateMatch(
                        experience_id=experience_id,
                        temperature=temperature,
                        raw_overlap=float(overlap_value),
                    )
                )
            selected = select_temperature_pools(
                matches,
                mode=selection.mode,
                requested_limit=selection.requested_limit,
            )
            if not selected:
                return ()

            selected_ids = tuple(match.experience_id for match in selected)
            full_rows = (
                await session.execute(
                    select(
                        ExperienceRow,
                        ExperienceVersionRow,
                        ExperienceStateRow,
                        DomainEventRow,
                        ExperienceTermRow,
                    )
                    .select_from(ExperienceRow)
                    .join(
                        ExperienceStateRow,
                        ExperienceStateRow.experience_id
                        == ExperienceRow.experience_id,
                    )
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.version_id
                        == ExperienceStateRow.current_version_id,
                    )
                    .join(
                        DomainEventRow,
                        DomainEventRow.event_id
                        == ExperienceStateRow.projection_event_id,
                    )
                    .join(
                        ExperienceTermRow,
                        ExperienceTermRow.experience_id
                        == ExperienceRow.experience_id,
                    )
                    .where(
                        ExperienceRow.owner_agent_id
                        == selection.owner_agent_id,
                        ExperienceRow.experience_id.in_(selected_ids),
                    )
                    .order_by(
                        ExperienceRow.experience_id,
                        ExperienceTermRow.term,
                        ExperienceTermRow.term_kind,
                    )
                )
            ).all()
            grouped: dict[
                UUID,
                tuple[
                    ExperienceRow,
                    ExperienceVersionRow,
                    ExperienceStateRow,
                    DomainEventRow,
                    list[TermCue],
                ],
            ] = {}
            for identity, version, state, projection_event, term in full_rows:
                current = grouped.get(identity.experience_id)
                if current is None:
                    current = (
                        identity,
                        version,
                        state,
                        projection_event,
                        [],
                    )
                    grouped[identity.experience_id] = current
                elif (
                    current[1].version_id != version.version_id
                    or current[2].projection_event_id
                    != state.projection_event_id
                    or current[3].event_id != projection_event.event_id
                ):
                    raise SourceIntegrityError(
                        "Candidate query returned inconsistent aggregate rows"
                    )
                current[4].append(
                    TermCue(
                        term=term.term,
                        term_kind=term.term_kind,
                        weight=term.weight,
                    )
                )
            if set(grouped) != set(selected_ids):
                raise SourceIntegrityError(
                    "Selected candidate sources are incomplete"
                )

            verified = select_temperature_pools(
                (
                    CandidateMatch(
                        experience_id=experience_id,
                        temperature=grouped[experience_id][2].temperature,
                        raw_overlap=raw_overlap(
                            selection.query_cues,
                            grouped[experience_id][4],
                        ),
                    )
                    for experience_id in selected_ids
                ),
                mode=selection.mode,
                requested_limit=selection.requested_limit,
            )
            if {match.experience_id for match in verified} != set(selected_ids):
                raise SourceIntegrityError(
                    "Selected candidate overlap sources are inconsistent"
                )
            selected_ids = tuple(match.experience_id for match in verified)
            overlaps = {
                match.experience_id: match.raw_overlap for match in verified
            }
            values: dict[UUID, RetrievalCandidate] = {}
            for experience_id in selected_ids:
                identity, version, state, projection_event, terms = grouped[
                    experience_id
                ]
                record = await self._retrieval_record_from_rows(
                    session=session,
                    identity=identity,
                    version=version,
                    state=state,
                    projection_event=projection_event,
                )
                values[experience_id] = RetrievalCandidate(
                    record=record,
                    terms=tuple(terms),
                    raw_overlap=overlaps[experience_id],
                )
            return tuple(values[experience_id] for experience_id in selected_ids)
        except SourceIntegrityError:
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                "Experience candidate sources are invalid"
            ) from error

    async def load_decoded_payloads(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        version_ids: Sequence[UUID],
    ) -> dict[UUID, bytes]:
        """Decode an all-or-nothing owner-rechecked version batch."""
        if not isinstance(owner_agent_id, UUID):
            raise ValueError("owner_agent_id must be a UUID")
        if isinstance(version_ids, (str, bytes)) or not isinstance(
            version_ids,
            Sequence,
        ):
            raise ValueError("version_ids must be a sequence of UUID values")
        values = tuple(version_ids)
        if any(not isinstance(value, UUID) for value in values):
            raise ValueError("version_ids must contain only UUID values")
        ordered_ids = tuple(dict.fromkeys(values))
        if not ordered_ids:
            return {}
        try:
            owned_rows = (
                await session.execute(
                    select(ExperienceRow, ExperienceVersionRow)
                    .select_from(ExperienceRow)
                    .join(
                        ExperienceVersionRow,
                        ExperienceVersionRow.experience_id
                        == ExperienceRow.experience_id,
                    )
                    .where(
                        ExperienceRow.owner_agent_id == owner_agent_id,
                        ExperienceVersionRow.version_id.in_(ordered_ids),
                    )
                )
            ).all()
            owned = {
                version.version_id: (identity, version)
                for identity, version in owned_rows
            }
            if set(owned) != set(ordered_ids):
                raise ExperienceNotFoundError

            result: dict[UUID, bytes] = {}
            for version_id in ordered_ids:
                identity, version = owned[version_id]
                payload = await session.get(ExperiencePayloadRow, version_id)
                if payload is None:
                    raise SourceIntegrityError(
                        f"Experience version {version_id} has no payload source"
                    )
                decode_and_verify_version(
                    identity=identity,
                    version=version,
                    payload=payload,
                )
                result[version_id] = decode_payload(
                    payload.codec,
                    payload.payload,
                )
            return result
        except (ExperienceNotFoundError, SourceIntegrityError):
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                "Experience payload batch has invalid sources"
            ) from error

    async def get_owned_shareable_version(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
        version_id: UUID | None,
    ) -> ShareableExperienceVersion:
        try:
            return await self._get_owned_shareable_version(
                session=session,
                owner_agent_id=owner_agent_id,
                experience_id=experience_id,
                version_id=version_id,
            )
        except (ExperienceNotFoundError, SourceIntegrityError):
            raise
        except (LookupError, StatementError, TypeError, ValueError) as error:
            raise SourceIntegrityError(
                f"Experience {experience_id} has invalid source values"
            ) from error

    async def _get_owned_shareable_version(
        self,
        *,
        session: AsyncSession,
        owner_agent_id: UUID,
        experience_id: UUID,
        version_id: UUID | None,
    ) -> ShareableExperienceVersion:
        identity = await session.scalar(
            select(ExperienceRow).where(
                ExperienceRow.owner_agent_id == owner_agent_id,
                ExperienceRow.experience_id == experience_id,
            )
        )
        if identity is None:
            raise ExperienceNotFoundError
        current_version = aliased(ExperienceVersionRow)
        aggregate = (
            await session.execute(
                select(
                    ExperienceStateRow,
                    current_version,
                    DomainEventRow,
                )
                .select_from(ExperienceStateRow)
                .join(
                    current_version,
                    current_version.version_id
                    == ExperienceStateRow.current_version_id,
                )
                .join(
                    DomainEventRow,
                    DomainEventRow.event_id
                    == ExperienceStateRow.projection_event_id,
                )
                .where(
                    ExperienceStateRow.experience_id == experience_id,
                )
            )
        ).one_or_none()
        if aggregate is None:
            raise SourceIntegrityError(
                f"Owned experience {experience_id} has no complete current state"
            )
        state, current, projection_event = aggregate
        if (
            state.owner_agent_id != owner_agent_id
            or current.experience_id != experience_id
            or state.current_content_hash != current.content_hash
            or projection_event.aggregate_type != "experience"
            or projection_event.aggregate_id != experience_id
        ):
            raise SourceIntegrityError(
                "Experience projection event does not belong to its aggregate"
            )
        await require_current_aggregate_head(
            session=session,
            experience_id=experience_id,
            projection_event=projection_event,
            event_registry=self._event_registry,
            handled_event_types=self._handled_event_types,
        )

        selected_id = current.version_id if version_id is None else version_id
        version = await session.scalar(
            select(ExperienceVersionRow).where(
                ExperienceVersionRow.experience_id == experience_id,
                ExperienceVersionRow.version_id == selected_id,
            )
        )
        if version is None:
            raise ExperienceNotFoundError
        payload = await session.get(ExperiencePayloadRow, selected_id)
        if payload is None:
            raise SourceIntegrityError(
                f"Experience version {selected_id} has no payload source"
            )
        content = decode_and_verify_version(
            identity=identity,
            version=version,
            payload=payload,
        )
        causal_times = [
            identity.created_at,
            version.created_at,
            current.created_at,
            projection_event.occurred_at,
            state.strength_updated_at,
            state.last_transition_at,
        ]
        causal_times.extend(
            value
            for value in (
                state.last_accessed_at,
                state.last_lifecycle_evaluated_at,
            )
            if value is not None
        )
        return ShareableExperienceVersion(
            experience_id=identity.experience_id,
            owner_agent_id=identity.owner_agent_id,
            origin=identity.origin,
            kind=identity.kind,
            version_id=version.version_id,
            content=content,
            content_hash=version.content_hash,
            confidence=state.confidence,
            temperature=state.temperature,
            latest_causal_at=max(causal_times),
        )


__all__ = [
    "CandidateSelection",
    "ExperienceNotFoundError",
    "ExperienceQuery",
    "RetrievalCandidate",
    "RetrievalRecord",
    "ShareableExperienceVersion",
]
