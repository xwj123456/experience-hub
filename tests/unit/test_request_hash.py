from collections.abc import Mapping, MutableMapping
from operator import setitem
from types import MappingProxyType
from typing import cast
from uuid import UUID

import pytest
from pydantic import BaseModel, computed_field

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.domain.commands import CommandRequest


def make_request(**overrides: object) -> CommandRequest:
    values: dict[str, object] = {
        "caller_scope": "agent:00000000-0000-0000-0000-000000000001",
        "operation_scope": "agent.create",
        "idempotency_key": " create-agent-1 ",
        "method": "post",
        "route_template": "/v1/agents/{agent_id}",
        "path_parameters": {
            "agent_id": UUID("00000000-0000-0000-0000-000000000001")
        },
        "query_parameters": (("tag", "two"), ("tag", "one"), ("page", "1")),
        "body": {"nested": {"b": 2, "a": 1}, "name": "Alice"},
        "semantic_headers": {"If-Match": '"revision-1"'},
    }
    values.update(overrides)
    return CommandRequest(**values)  # type: ignore[arg-type]


def test_idempotency_key_is_trimmed_and_limited_to_128_characters() -> None:
    assert make_request().idempotency_key == "create-agent-1"
    assert make_request(idempotency_key="x" * 128).idempotency_key == "x" * 128

    for invalid in ("", "   ", "x" * 129):
        with pytest.raises(ValueError, match="Idempotency key"):
            make_request(idempotency_key=invalid)


def test_request_hash_canonicalizes_method_maps_query_pairs_and_body() -> None:
    left = make_request()
    right = make_request(
        method="POST",
        path_parameters={
            "agent_id": "00000000-0000-0000-0000-000000000001",
        },
        query_parameters=(("page", "1"), ("tag", "one"), ("tag", "two")),
        body={"name": "Alice", "nested": {"a": 1, "b": 2}},
        semantic_headers={"if-match": '"revision-1"'},
    )

    assert left.request_hash == right.request_hash
    assert len(left.request_hash) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "PUT"),
        ("route_template", "/v1/agents"),
        (
            "path_parameters",
            {"agent_id": "00000000-0000-0000-0000-000000000002"},
        ),
        ("query_parameters", (("page", "1"), ("tag", "one"))),
        ("body", {"name": "Bob", "nested": {"a": 1, "b": 2}}),
        ("semantic_headers", {"if-match": '"revision-2"'}),
    ],
)
def test_request_hash_covers_every_semantic_request_component(
    field: str,
    value: object,
) -> None:
    assert make_request(**{field: value}).request_hash != make_request().request_hash


def test_request_hash_preserves_repeated_query_items() -> None:
    repeated = make_request(query_parameters=(("tag", "one"), ("tag", "one")))
    single = make_request(query_parameters=(("tag", "one"),))

    assert repeated.request_hash != single.request_hash


def test_query_pair_order_is_normalized_without_collapsing_duplicates() -> None:
    left = make_request(
        query_parameters=(("tag", "two"), ("tag", "one"), ("tag", "one"))
    )
    right = make_request(
        query_parameters=(("tag", "one"), ("tag", "one"), ("tag", "two"))
    )

    assert left.request_hash == right.request_hash


def test_request_hash_is_fixed_at_construction_and_normalizes_header_ows() -> None:
    body = {"nested": {"b": 2, "a": 1}, "name": "Alice"}
    path = {"agent_id": "00000000-0000-0000-0000-000000000001"}
    request = make_request(
        body=body,
        path_parameters=path,
        semantic_headers={" If-Match ": "\t\"revision-1\" "},
    )
    original_hash = request.request_hash

    body["name"] = "mutated"
    path["agent_id"] = "00000000-0000-0000-0000-000000000002"

    assert request.request_hash == original_hash
    assert request.request_hash == make_request().request_hash


def test_request_semantics_are_recursively_immutable_after_construction() -> None:
    request = make_request(
        path_parameters={"scope": {"agent_id": "one"}},
        body={"nested": {"value": 1}, "items": [1, 2]},
    )
    original_hash = request.request_hash
    body = cast(Mapping[str, object], request.body)
    nested_body = cast(MutableMapping[str, object], body["nested"])
    nested_path = cast(
        MutableMapping[str, object],
        request.path_parameters["scope"],
    )

    with pytest.raises(TypeError):
        setitem(nested_body, "value", 2)
    with pytest.raises(TypeError):
        setitem(nested_path, "agent_id", "two")

    assert body["items"] == (1, 2)
    assert request.request_hash == original_hash
    assert canonical_json_bytes(request.body) == canonical_json_bytes(
        {"items": [1, 2], "nested": {"value": 1}}
    )


def test_request_accepts_an_already_immutable_canonical_snapshot() -> None:
    request = make_request(
        path_parameters=MappingProxyType(
            {"scope": MappingProxyType({"agent_id": "one"})}
        ),
        body=MappingProxyType(
            {"nested": MappingProxyType({"value": 1})}
        ),
    )

    assert canonical_json_bytes(request.body) == canonical_json_bytes(
        {"nested": {"value": 1}}
    )
    assert canonical_json_bytes(request.path_parameters) == canonical_json_bytes(
        {"scope": {"agent_id": "one"}}
    )


def test_request_hash_uses_the_same_single_materialized_body_snapshot() -> None:
    values = iter((1, 2))

    class StatefulBody(BaseModel):
        @computed_field  # type: ignore[prop-decorator]
        @property
        def value(self) -> int:
            return next(values)

    request = make_request(body=StatefulBody())
    recomputed = sha256_hex(
        canonical_json_bytes(
            {
                "method": request.method,
                "route_template": request.route_template,
                "path_parameters": request.path_parameters,
                "query_parameters": request.query_parameters,
                "body": request.body,
                "semantic_headers": request.semantic_headers,
            }
        )
    )

    assert request.body == {"value": 1}
    assert request.request_hash == recomputed
