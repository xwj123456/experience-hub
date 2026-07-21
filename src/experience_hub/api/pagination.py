"""Route-bound, canonical opaque cursor encoding."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping
from typing import Any

from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import CanonicalizationError, DomainError

_TOKEN = re.compile(r"[A-Za-z0-9_-]+\Z")
_MAX_CURSOR_CHARACTERS = 8_192
_CURSOR_VERSION = 1


def _invalid_cursor() -> DomainError:
    return DomainError(
        code="invalid_cursor",
        message="The cursor is invalid.",
        status_code=400,
    )


class CursorCodec:
    """Bind a stable sort tuple to one route and optional filter context."""

    def __init__(
        self,
        *,
        route: str,
        version: int = _CURSOR_VERSION,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(route, str) or not route or route != route.strip():
            raise ValueError("route must be a non-empty trimmed string")
        if version != _CURSOR_VERSION:
            raise ValueError("only cursor version 1 is supported")
        self._route = route
        self._version = version
        try:
            normalized_context = (
                None if context is None else json.loads(canonical_json_bytes(context))
            )
        except (
            CanonicalizationError,
            RecursionError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("context must contain canonical JSON values") from error
        if normalized_context is not None and not isinstance(normalized_context, dict):
            raise ValueError("context must be a JSON object")
        self._context = normalized_context
        self._context_bytes = (
            None
            if normalized_context is None
            else canonical_json_bytes(normalized_context)
        )

    def encode(self, sort: tuple[Any, ...]) -> str:
        if not isinstance(sort, tuple) or not sort:
            raise ValueError("sort must be a non-empty tuple")
        try:
            normalized_sort = json.loads(canonical_json_bytes(sort))
        except (
            CanonicalizationError,
            RecursionError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("sort must contain canonical JSON values") from error
        if not isinstance(normalized_sort, list):
            raise ValueError("sort must encode as a JSON array")
        document: dict[str, Any] = {
            "route": self._route,
            "sort": normalized_sort,
            "version": self._version,
        }
        if self._context is not None:
            document["context"] = self._context
        raw = canonical_json_bytes(document)
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def decode(self, cursor: str) -> tuple[Any, ...]:
        try:
            if (
                not isinstance(cursor, str)
                or not cursor
                or cursor != cursor.strip()
                or len(cursor) > _MAX_CURSOR_CHARACTERS
                or "=" in cursor
                or _TOKEN.fullmatch(cursor) is None
            ):
                raise ValueError("invalid token shape")
            encoded = cursor.encode("ascii")
            padding = b"=" * (-len(encoded) % 4)
            raw = base64.b64decode(
                encoded + padding,
                altchars=b"-_",
                validate=True,
            )
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("cursor document must be an object")
            expected_keys = {"route", "sort", "version"}
            if self._context is not None:
                expected_keys.add("context")
            identity_mismatch = (
                set(document) != expected_keys
                or document.get("route") != self._route
                or type(document.get("version")) is not int
                or document.get("version") != self._version
            )
            context_mismatch = (
                canonical_json_bytes(document.get("context")) != self._context_bytes
                if self._context_bytes is not None
                else "context" in document
            )
            if identity_mismatch or context_mismatch:
                raise ValueError("cursor identity mismatch")
            sort = document.get("sort")
            if not isinstance(sort, list) or not sort:
                raise ValueError("cursor sort is invalid")
            if canonical_json_bytes(document) != raw:
                raise ValueError("cursor JSON is not canonical")
            canonical_token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
            if canonical_token != cursor:
                raise ValueError("cursor base64 is not canonical")
            return tuple(sort)
        except (
            binascii.Error,
            CanonicalizationError,
            RecursionError,
            UnicodeDecodeError,
            UnicodeEncodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise _invalid_cursor() from error


__all__ = ["CursorCodec"]
