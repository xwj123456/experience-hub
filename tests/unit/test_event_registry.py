from typing import ClassVar

import pytest
from pydantic import ValidationError

from experience_hub.domain.events import EventPayload, EventRegistry


class ExampleRecorded(EventPayload):
    event_type: ClassVar[str] = "example.recorded"
    value: str


class DuplicateExampleRecorded(EventPayload):
    event_type: ClassVar[str] = "example.recorded"
    count: int


def test_registry_rejects_duplicate_event_name() -> None:
    registry = EventRegistry()
    registry.register(ExampleRecorded)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DuplicateExampleRecorded)


def test_registry_rejects_unknown_event_name() -> None:
    registry = EventRegistry()

    with pytest.raises(ValueError, match="Unknown event type"):
        registry.decode(
            event_type="example.unknown",
            payload=b'{"schema_version":1,"value":"kept"}',
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value":"missing"}',
        b'{"schema_version":null,"value":"null"}',
        b'{"schema_version":2,"value":"unsupported"}',
        b'{"extra":true,"schema_version":1,"value":"unknown-field"}',
    ],
    ids=[
        "missing-schema-version",
        "null-schema-version",
        "unsupported-schema-version",
        "unknown-field",
    ],
)
def test_registry_strictly_decodes_versioned_payload(payload: bytes) -> None:
    registry = EventRegistry()
    registry.register(ExampleRecorded)

    with pytest.raises(ValidationError):
        registry.decode(event_type=ExampleRecorded.event_type, payload=payload)


def test_registry_returns_the_registered_typed_payload() -> None:
    registry = EventRegistry()
    registry.register(ExampleRecorded)

    decoded = registry.decode(
        event_type=ExampleRecorded.event_type,
        payload=b'{"schema_version":1,"value":"typed"}',
    )

    assert decoded == ExampleRecorded(schema_version=1, value="typed")
    assert type(decoded) is ExampleRecorded
