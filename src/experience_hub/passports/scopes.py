"""Stable command operations; these do not expose HTTP routes."""

from typing import Final

PASSPORT_IMPORT_SCOPE: Final[str] = "passport.import"
PASSPORT_ADOPT_SCOPE: Final[str] = "passport.adopt"
PASSPORT_REJECT_SCOPE: Final[str] = "passport.reject"
