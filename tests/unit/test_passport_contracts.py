import pytest
from pydantic import ValidationError

from experience_hub.capture.models import TrajectoryField
from experience_hub.experiences.models import ExperienceOrigin
from experience_hub.passports import (
    EmbeddedExcerptSnapshotV1,
    PassportDeclarationV1,
)


def test_imported_origin_is_distinct_from_local() -> None:
    assert ExperienceOrigin.ADOPTED_PASSPORT.value == "adopted_passport"


@pytest.mark.parametrize("flag", [False, 1, "true", None])
def test_declarations_require_explicit_boolean_true(flag: object) -> None:
    with pytest.raises(ValidationError):
        PassportDeclarationV1(
            profile_id="local-v1", input_sanitized=flag, sharing_authorized=True
        )
    with pytest.raises(ValidationError):
        PassportDeclarationV1(
            profile_id="local-v1", input_sanitized=True, sharing_authorized=flag
        )


def test_profile_utf8_limit() -> None:
    assert PassportDeclarationV1(
        profile_id="界" * 33 + "a", input_sanitized=True, sharing_authorized=True
    )
    with pytest.raises(ValidationError):
        PassportDeclarationV1(
            profile_id="界" * 34, input_sanitized=True, sharing_authorized=True
        )


def test_excerpt_utf8_limit() -> None:
    fields = dict(
        mode="embedded_excerpt",
        reference={"type": "test", "id": "test"},
        source_hash="a" * 64,
        source_manifest_hash="b" * 64,
        step_id="step-1",
        field=TrajectoryField.OUTCOME,
        excerpt_hash="c" * 64,
    )
    assert EmbeddedExcerptSnapshotV1(**fields, excerpt="界" * 170 + "aa")
    with pytest.raises(ValidationError):
        EmbeddedExcerptSnapshotV1(**fields, excerpt="界" * 171)
