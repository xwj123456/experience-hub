"""Stable errors for release evidence tooling."""


class ReleaseEvidenceError(ValueError):
    """A stable release-evidence failure without private implementation details."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
