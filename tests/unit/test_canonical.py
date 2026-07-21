from datetime import UTC, datetime

import pytest

from experience_hub.canonical import canonical_json_bytes, sha256_hex
from experience_hub.errors import CanonicalizationError


def test_canonical_json_is_stable_and_unicode_preserving() -> None:
    left = {
        "中文": [2, 1],
        "at": datetime(2026, 7, 17, tzinfo=UTC),
        "zero": -0.0,
    }
    right = {"zero": 0.0, "at": "2026-07-17T00:00:00.000000Z", "中文": [2, 1]}

    encoded = canonical_json_bytes(left)

    assert encoded == canonical_json_bytes(right)
    assert "中文" in encoded.decode("utf-8")
    assert b"-0.0" not in encoded
    assert sha256_hex(encoded) == sha256_hex(canonical_json_bytes(right))


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(CanonicalizationError, match="finite"):
        canonical_json_bytes({"score": float("nan")})
