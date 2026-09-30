from uuid import UUID

import pytest
from pydantic import ValidationError

from experience_hub import canonical_json_bytes
from experience_hub.domain import EventRegistry, StructuredReason
from experience_hub.passports.contracts import PassportState


def _payloads() -> tuple[object, ...]:
    try:
        from experience_hub.passports.events import (
            PassportAdoptedV1,
            PassportImportedV1,
            PassportRejectedV1,
        )
    except ImportError:
        pytest.fail("Passport v1 event vocabulary is not implemented")
    common = {
        "schema_version": 1,
        "import_id": UUID(int=101),
        "owner_agent_id": UUID(int=102),
    }
    return (
        PassportImportedV1(
            **common, passport_hash="a" * 64, state_after=PassportState.PENDING
        ),
        PassportAdoptedV1(
            **common,
            state_before=PassportState.PENDING,
            state_after=PassportState.ADOPTED,
            adoption_id=UUID(int=103),
            resulting_experience_id=UUID(int=104),
            resulting_version_id=UUID(int=105),
            resulting_content_hash="b" * 64,
            created=True,
            importance=0.7,
            confidence=0.6,
        ),
        PassportRejectedV1(
            **common,
            state_before=PassportState.PENDING,
            state_after=PassportState.REJECTED,
            reason=StructuredReason.from_user_text("Not applicable."),
        ),
    )


def test_registry_roundtrips_strict_v1_passport_payloads() -> None:
    payloads = _payloads()
    from experience_hub.passports.events import register_passport_events

    registry = EventRegistry()
    register_passport_events(registry)
    assert registry.event_types == {
        "passport.imported",
        "passport.adopted",
        "passport.rejected",
    }
    for payload in payloads:
        assert (
            registry.decode(
                event_type=payload.event_type, payload=canonical_json_bytes(payload)
            )
            == payload
        )


@pytest.mark.parametrize("value", (True, 1.0, 2, "1"))
def test_events_reject_non_integer_v1_schema(value: object) -> None:
    for payload in _payloads():
        with pytest.raises(ValidationError):
            type(payload).model_validate(
                {**payload.model_dump(), "schema_version": value}
            )


@pytest.mark.parametrize(
    "index,updates",
    (
        (0, {"passport_hash": "A" * 64}),
        (0, {"state_after": PassportState.ADOPTED}),
        (1, {"resulting_content_hash": "b" * 63}),
        (1, {"state_before": PassportState.REJECTED}),
        (1, {"state_after": PassportState.REJECTED}),
        (1, {"created": 1}),
        (1, {"importance": True}),
        (1, {"importance": -0.1}),
        (1, {"importance": float("inf")}),
        (1, {"confidence": float("nan")}),
        (1, {"confidence": 1.1}),
        (2, {"state_before": PassportState.ADOPTED}),
        (2, {"state_after": PassportState.ADOPTED}),
        (2, {"extra": "not allowed"}),
    ),
)
def test_events_reject_invalid_hash_state_and_decision_values(
    index: int,
    updates: dict[str, object],
) -> None:
    payload = _payloads()[index]
    with pytest.raises(ValidationError):
        type(payload).model_validate({**payload.model_dump(), **updates})
