"""Guarded physical codec rewrites for semantically immutable payloads."""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub import sha256_hex
from experience_hub.experiences.content import decode_payload, reencode_payload
from experience_hub.experiences.models import PayloadCodec
from experience_hub.storage.database import payload_rewrite_guard
from experience_hub.storage.tables import ExperiencePayloadRow


class PayloadRewriteConflictError(RuntimeError):
    """The guarded payload row changed after it was read."""


async def rewrite_payload_codec(
    *,
    session: AsyncSession,
    version_id: UUID,
    codec: PayloadCodec,
) -> None:
    """Rewrite only codec bytes after proving the decoded hash is unchanged."""
    if not isinstance(codec, PayloadCodec):
        raise TypeError("codec must be a PayloadCodec")

    with session.no_autoflush:
        stored = (
            await session.execute(
                select(
                    ExperiencePayloadRow.codec,
                    ExperiencePayloadRow.payload,
                    ExperiencePayloadRow.payload_hash,
                ).where(ExperiencePayloadRow.version_id == version_id)
            )
        ).one_or_none()
    if stored is None:
        raise LookupError(f"Experience payload does not exist: {version_id}")

    decoded = decode_payload(stored.codec, stored.payload)
    if sha256_hex(decoded) != stored.payload_hash:
        raise ValueError("Stored payload bytes do not match the semantic hash")

    replacement = reencode_payload(decoded, codec)
    replacement_decoded = decode_payload(codec, replacement)
    if replacement_decoded != decoded:
        raise ValueError("Replacement codec changed decoded payload bytes")
    if sha256_hex(replacement_decoded) != stored.payload_hash:
        raise ValueError("Replacement payload does not match the semantic hash")
    if stored.codec is codec and stored.payload == replacement:
        return

    statement = (
        update(ExperiencePayloadRow)
        .where(
            ExperiencePayloadRow.version_id == version_id,
            ExperiencePayloadRow.codec == stored.codec,
            ExperiencePayloadRow.payload == stored.payload,
            ExperiencePayloadRow.payload_hash == stored.payload_hash,
        )
        .values(codec=codec, payload=replacement)
    )
    connection = await session.connection()
    with (
        session.no_autoflush,
        payload_rewrite_guard(connection),
    ):
        result = cast(
            CursorResult[Any],
            await session.execute(
                statement,
                execution_options={"synchronize_session": "fetch"},
            ),
        )
    if result.rowcount != 1:
        raise PayloadRewriteConflictError(
            "Payload changed concurrently during codec rewrite"
        )
