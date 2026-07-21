"""Configuration that remains portable across repository locations."""

from dataclasses import dataclass, field
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor containing this project's ``pyproject.toml``."""
    location = (start or Path(__file__)).resolve()
    candidates = (
        (location, *location.parents) if location.is_dir() else location.parents
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Unable to find the experience-hub repository root")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with a repository-relative SQLite default."""

    database_url: str | None = None
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None
    openai_compatible_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.database_url is None:
            path = repository_root() / ".data" / "experience_hub.db"
            object.__setattr__(self, "database_url", f"sqlite+aiosqlite:///{path}")
