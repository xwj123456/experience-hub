"""Stable Passport failures with no retained input or filesystem paths."""

from typing import Literal

from experience_hub.capture.sanitization import TextSensitiveMatchV1
from experience_hub.errors import DomainError

type PassportErrorCode = Literal[
    "invalid",
    "size_limit",
    "sensitive_content",
    "not_found",
    "decision_conflict",
    "equivalent_ambiguous",
    "restore_required",
    "derivation_unsupported",
    "provenance_limit",
    "file_invalid",
    "output_conflict",
    "publication_unsupported",
]

_ERRORS: dict[PassportErrorCode, tuple[int, str]] = {
    "invalid": (400, "Invalid evidence passport"),
    "size_limit": (413, "Evidence passport exceeds its size limit"),
    "sensitive_content": (400, "Evidence passport contains sensitive content"),
    "not_found": (404, "Evidence passport not found"),
    "decision_conflict": (409, "Evidence passport already has a decision"),
    "equivalent_ambiguous": (409, "Equivalent local experience is ambiguous"),
    "restore_required": (409, "Equivalent local experience requires restore"),
    "derivation_unsupported": (400, "Passport content derivation is unsupported"),
    "provenance_limit": (400, "Passport provenance limit exceeded"),
    "file_invalid": (400, "Invalid passport file boundary"),
    "output_conflict": (409, "Passport output already contains different bytes"),
    "publication_unsupported": (400, "Passport capsule publication is unsupported"),
}


class PassportError(DomainError):
    def __init__(
        self,
        code: PassportErrorCode,
        *,
        matches: tuple[TextSensitiveMatchV1, ...] = (),
    ) -> None:
        status, message = _ERRORS[code]
        details = (
            {"matches": [match.model_dump() for match in matches]}
            if code == "sensitive_content" and matches
            else None
        )
        super().__init__(
            f"passport_{code}", message, details=details, status_code=status
        )
