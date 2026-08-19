"""Stable errors for untrusted replay fixture boundaries."""


class ExperimentInputError(ValueError):
    """A path-free, stable rejection for replay manifest or dataset input."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ExperimentIsolationError(RuntimeError):
    """Stable errors for unsafe replay workspace ownership boundaries."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
