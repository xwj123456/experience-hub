from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from experience_hub.config import Settings


def test_default_database_uses_cwd_without_creating_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.database_url is not None
    assert make_url(settings.database_url).database == str(
        tmp_path / ".data" / "experience_hub.db"
    )
    assert not (tmp_path / ".data").exists()


def test_explicit_database_is_preserved_outside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    url = "sqlite+aiosqlite:///explicit.sqlite3"

    assert Settings(database_url=url).database_url == url
    assert not (tmp_path / "explicit.sqlite3").exists()
