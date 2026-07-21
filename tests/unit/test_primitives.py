from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from experience_hub.config import Settings
from experience_hub.domain.values import StructuredReason, TypedEvidence


def test_settings_uses_repository_relative_default_and_explicit_override(
    repository_root: Path,
) -> None:
    default = Settings()
    explicit = Settings(database_url="sqlite+aiosqlite:///tmp/override.db")

    assert default.database_url == (
        f"sqlite+aiosqlite:///{repository_root / '.data' / 'experience_hub.db'}"
    )
    assert explicit.database_url == "sqlite+aiosqlite:///tmp/override.db"


def test_typed_evidence_rejects_blank_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TypedEvidence(type=" ", id="source-1")
    with pytest.raises(ValidationError):
        TypedEvidence(  # type: ignore[call-arg]
            type="document",
            id="source-1",
            unexpected=True,
        )


@pytest.mark.parametrize(
    ("code", "text", "text_hash"),
    [
        ("user_provided", "   ", sha256(b"").hexdigest()),
        ("user_provided", "x" * 2001, sha256(("x" * 2001).encode()).hexdigest()),
        ("user_provided", "retained text", "0" * 64),
    ],
)
def test_structured_reason_rejects_invalid_retained_text_or_hash(
    code: str, text: str, text_hash: str
) -> None:
    with pytest.raises(ValidationError):
        StructuredReason(code=code, text=text, text_hash=text_hash)


@pytest.mark.parametrize("code", ["reason_", "reason__code"])
def test_structured_reason_rejects_malformed_snake_case_codes(code: str) -> None:
    text = "retained text"

    with pytest.raises(ValidationError):
        StructuredReason(
            code=code,
            text=text,
            text_hash=sha256(text.encode("utf-8")).hexdigest(),
        )


def test_structured_reason_hashes_trimmed_retained_text() -> None:
    retained = "use supporting document"
    reason = StructuredReason(
        code="user_provided",
        text=f"  {retained}  ",
        text_hash=sha256(retained.encode("utf-8")).hexdigest(),
    )

    assert reason.text == retained
