import base64

import pytest

from experience_hub.api.pagination import CursorCodec
from experience_hub.canonical import canonical_json_bytes
from experience_hub.errors import DomainError

ROUTE = "agents.list"
OTHER_ROUTE = "inbox.list"
SORT_TUPLE = (
    "2026-07-19T00:00:00.000000Z",
    "00000000-0000-0000-0000-000000000001",
)


def _decode_urlsafe(token: str) -> bytes:
    padding = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(token + padding)


def test_cursor_round_trip_retains_the_stable_sort_tuple() -> None:
    codec = CursorCodec(route=ROUTE)

    token = codec.encode(SORT_TUPLE)

    assert codec.decode(token) == SORT_TUPLE
    assert "=" not in token
    assert _decode_urlsafe(token) == canonical_json_bytes(
        {
            "route": ROUTE,
            "sort": list(SORT_TUPLE),
            "version": 1,
        }
    )


@pytest.mark.parametrize(
    "token",
    [
        "",
        "***not-base64***",
        base64.urlsafe_b64encode(b"not json").decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(b"{}").decode("ascii").rstrip("="),
    ],
)
def test_malformed_cursor_is_rejected_with_invalid_cursor(token: str) -> None:
    codec = CursorCodec(route=ROUTE)

    with pytest.raises(DomainError) as raised:
        codec.decode(token)

    assert raised.value.code == "invalid_cursor"
    assert raised.value.status_code == 400
    assert raised.value.details == {}


def test_wrong_cursor_version_is_rejected_with_invalid_cursor() -> None:
    token = (
        base64.urlsafe_b64encode(
            canonical_json_bytes(
                {
                    "route": ROUTE,
                    "sort": list(SORT_TUPLE),
                    "version": 2,
                }
            )
        )
        .decode("ascii")
        .rstrip("=")
    )

    with pytest.raises(DomainError) as raised:
        CursorCodec(route=ROUTE).decode(token)

    assert raised.value.code == "invalid_cursor"


def test_route_mismatched_cursor_is_rejected_with_invalid_cursor() -> None:
    token = CursorCodec(route=ROUTE).encode(SORT_TUPLE)

    with pytest.raises(DomainError) as raised:
        CursorCodec(route=OTHER_ROUTE).decode(token)

    assert raised.value.code == "invalid_cursor"


def test_cursor_context_prevents_cross_owner_or_filter_reuse() -> None:
    context = {"owner_id": "agent-a", "state": "pending"}
    token = CursorCodec(route=ROUTE, context=context).encode(SORT_TUPLE)

    assert CursorCodec(route=ROUTE, context=context).decode(token) == SORT_TUPLE
    with pytest.raises(DomainError) as raised:
        CursorCodec(
            route=ROUTE,
            context={"owner_id": "agent-b", "state": "pending"},
        ).decode(token)

    assert raised.value.code == "invalid_cursor"


def test_cursor_context_uses_strict_json_type_identity() -> None:
    token = CursorCodec(route=ROUTE, context={"active": 1}).encode(SORT_TUPLE)

    with pytest.raises(DomainError) as raised:
        CursorCodec(route=ROUTE, context={"active": True}).decode(token)

    assert raised.value.code == "invalid_cursor"


@pytest.mark.parametrize(
    ("document", "context"),
    [
        ([], {"owner_id": "agent-a"}),
        (
            {"route": ROUTE, "sort": list(SORT_TUPLE), "version": True},
            None,
        ),
    ],
)
def test_cursor_document_requires_strict_object_and_integer_version(
    document: object,
    context: dict[str, str] | None,
) -> None:
    token = (
        base64.urlsafe_b64encode(canonical_json_bytes(document))
        .decode("ascii")
        .rstrip("=")
    )

    with pytest.raises(DomainError):
        CursorCodec(route=ROUTE, context=context).decode(token)


def test_deeply_nested_cursor_is_a_stable_invalid_cursor() -> None:
    depth = 1_100
    raw = (
        b'{"route":"agents.list","sort":'
        + (b"[" * depth)
        + b"0"
        + (b"]" * depth)
        + b',"version":1}'
    )
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    assert len(token) < 8_192

    with pytest.raises(DomainError) as raised:
        CursorCodec(route=ROUTE).decode(token)

    assert raised.value.code == "invalid_cursor"
