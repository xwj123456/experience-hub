"""Shared command request and execution contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.errors import CallerScope, DomainError


def _required_trimmed(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def _immutable_canonical_snapshot(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _immutable_canonical_snapshot(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _immutable_canonical_snapshot(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_canonical_snapshot(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """All semantics that identify an idempotent command invocation."""

    caller_scope: str
    operation_scope: str
    idempotency_key: str
    method: str
    route_template: str
    path_parameters: Mapping[str, Any] = field(default_factory=dict)
    query_parameters: Sequence[tuple[str, str]] = ()
    body: Any = None
    semantic_headers: Mapping[str, str] = field(default_factory=dict)
    _request_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.idempotency_key.strip()
        if not 1 <= len(key) <= 128:
            raise ValueError("Idempotency key must contain 1 to 128 characters")
        method = _required_trimmed(self.method, "HTTP method").upper()
        route = _required_trimmed(self.route_template, "Route template")
        operation = _required_trimmed(self.operation_scope, "Operation scope")
        caller = str(CallerScope(self.caller_scope))

        path = MappingProxyType(
            {
                str(name): _immutable_canonical_snapshot(value)
                for name, value in self.path_parameters.items()
            }
        )
        query = tuple(
            sorted(
                (str(name), str(value))
                for name, value in self.query_parameters
            )
        )
        headers = MappingProxyType(
            {
                str(name).strip().lower(): str(value).strip(" \t")
                for name, value in self.semantic_headers.items()
            }
        )
        if any(not name for name in headers):
            raise ValueError("Semantic header names must not be blank")
        body = _immutable_canonical_snapshot(self.body)

        request_semantics = {
            "method": method,
            "route_template": route,
            "path_parameters": path,
            "query_parameters": query,
            "body": body,
            "semantic_headers": headers,
        }
        object.__setattr__(self, "caller_scope", caller)
        object.__setattr__(self, "operation_scope", operation)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "route_template", route)
        object.__setattr__(self, "path_parameters", path)
        object.__setattr__(self, "query_parameters", query)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "semantic_headers", headers)
        object.__setattr__(
            self,
            "_request_hash",
            sha256_hex(canonical_json_bytes(request_semantics)),
        )

    @property
    def request_hash(self) -> str:
        return self._request_hash


@dataclass(frozen=True, slots=True)
class CommandContext:
    receipt_id: UUID
    caller_scope: str
    operation_scope: str
    idempotency_key: str
    request_hash: str


class ReplayableCommandError(DomainError):
    """A stable command failure whose canonical response may be replayed."""
