from uuid import UUID

import pytest

from experience_hub.passports.errors import PassportError
from experience_hub.passports.requests import passport_adopt_request


def test_adoption_scores_have_one_canonical_request_hash() -> None:
    arguments = {
        "owner_agent_id": UUID(int=1),
        "import_id": UUID(int=2),
        "idempotency_key": "same-score",
    }
    integer = passport_adopt_request(**arguments, importance=1, confidence=0)
    floating = passport_adopt_request(**arguments, importance=1.0, confidence=0.0)
    assert integer.request_hash == floating.request_hash
    assert integer.body == floating.body
    assert type(integer.body["importance"]) is float
    assert type(integer.body["confidence"]) is float


@pytest.mark.parametrize("score", [True, False, float("nan"), float("inf"), -1, 2])
@pytest.mark.parametrize("field", ["importance", "confidence"])
def test_invalid_adoption_score_is_refused_at_request_boundary(
    score: float, field: str
) -> None:
    scores = {"importance": 0.5, "confidence": 0.5, field: score}
    with pytest.raises(PassportError) as caught:
        passport_adopt_request(
            owner_agent_id=UUID(int=1),
            import_id=UUID(int=2),
            idempotency_key="invalid-score",
            **scores,
        )
    assert caught.value.code == "passport_invalid"
    assert caught.value.details == {}
