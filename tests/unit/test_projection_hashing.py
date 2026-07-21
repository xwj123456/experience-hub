import json
from datetime import UTC, datetime
from hashlib import sha256

from experience_hub.storage.projections import canonical_projection_hash


def test_projection_hash_is_canonical_and_keeps_semantic_metadata() -> None:
    first = canonical_projection_hash(
        [
            {
                "projection_id": "b",
                "body": '{"z":1,"a":2}',
                "score": 1.23456789012349,
                "created_at": datetime(2026, 7, 17, 8, 30, tzinfo=UTC),
                "rowid": 9,
                "codec": "zlib",
                "payload": b"physical",
                "payload_hash": "semantic-hash",
            },
            {"projection_id": "a", "body": '{"a":2,"z":1}', "score": 0.0},
        ],
        projection="example",
        primary_key=("projection_id",),
        reducer_version=3,
        checkpoint=41,
    )
    second = canonical_projection_hash(
        [
            {"score": -0.0, "body": '{"z":1,"a":2}', "projection_id": "a"},
            {
                "payload": b"different-physical-bytes",
                "codec": "plain",
                "rowid": 200,
                "created_at": "2026-07-17T08:30:00.000000Z",
                "score": 1.23456789012341,
                "body": '{"a":2,"z":1}',
                "payload_hash": "semantic-hash",
                "projection_id": "b",
            },
        ],
        projection="example",
        primary_key=("projection_id",),
        reducer_version=3,
        checkpoint=41,
    )

    assert first == second
    assert len(first) == 64
    assert first != canonical_projection_hash(
        [{"projection_id": "a"}, {"projection_id": "b"}],
        projection="example",
        primary_key=("projection_id",),
        reducer_version=4,
        checkpoint=41,
    )
    assert first != canonical_projection_hash(
        [{"projection_id": "a"}, {"projection_id": "b"}],
        projection="other",
        primary_key=("projection_id",),
        reducer_version=3,
        checkpoint=41,
    )


def test_projection_hash_orders_numeric_primary_keys_like_sqlite() -> None:
    rows = [
        {"projection_id": "10", "value": "ten"},
        {"projection_id": "2", "value": "two"},
    ]
    expected = sha256(
        json.dumps(
            {
                "checkpoint": 2,
                "projection": "numeric_projection",
                "reducer_version": 1,
                "rows": [rows[1], rows[0]],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    assert (
        canonical_projection_hash(
            rows,
            projection="numeric_projection",
            primary_key=("projection_id",),
            primary_key_types=("INTEGER",),
            reducer_version=1,
            checkpoint=2,
        )
        == expected
    )


def test_projection_hash_uses_stable_sqlite_composite_key_order() -> None:
    rows = [
        {"tenant_id": 2, "item_key": "b", "value": "2-b"},
        {"tenant_id": 10, "item_key": "a", "value": "10-a"},
        {"tenant_id": 2, "item_key": "a", "value": "2-a"},
    ]
    expected = sha256(
        json.dumps(
            {
                "checkpoint": 3,
                "projection": "composite_projection",
                "reducer_version": 1,
                "rows": [rows[2], rows[0], rows[1]],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    assert (
        canonical_projection_hash(
            rows,
            projection="composite_projection",
            primary_key=("tenant_id", "item_key"),
            primary_key_types=("INTEGER", "TEXT"),
            reducer_version=1,
            checkpoint=3,
        )
        == expected
    )
