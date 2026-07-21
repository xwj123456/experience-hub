"""Shared migration, validation, recovery, and shutdown lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url

from experience_hub.bootstrap import ApplicationContainer
from experience_hub.clock import Clock
from experience_hub.config import Settings, repository_root
from experience_hub.ids import IdGenerator
from experience_hub.storage.database import (
    DatabaseBusy,
    is_sqlite_lock_error,
)


class SchemaRevisionError(RuntimeError):
    """The database revision is not an ancestor supported by this release."""

    code = "schema_version_unsupported"

    def __init__(
        self,
        *,
        current_revision: str | None,
        expected_revision: str,
    ) -> None:
        self.current_revision = current_revision
        self.expected_revision = expected_revision
        super().__init__(
            "Database schema revision is not supported by this application"
        )


SchemaVersionError = SchemaRevisionError


def _synchronous_sqlite_url(settings: Settings) -> URL:
    raw_url = settings.database_url
    if raw_url is None:
        raise ValueError("database_url must be configured")
    url = make_url(raw_url)
    if not url.drivername.startswith("sqlite"):
        raise ValueError("Experience Hub runtime requires SQLite")
    database = url.database
    if database in {None, "", ":memory:"}:
        raise ValueError(
            "Shared runtime initialization requires a file-backed SQLite database"
        )
    assert database is not None
    # Preserve SQLAlchemy's relative-path semantics so the synchronous
    # migration connection and the async application engine select one file.
    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return url.set(drivername="sqlite", database=str(database_path))


def _alembic_config(url: URL) -> Config:
    root = repository_root()
    config = Config(str(root / "alembic.ini"))
    # Embedded migrations run inside a host process whose logging handlers are
    # owned by that host (Uvicorn, a test harness, or another application).
    # Alembic's standalone CLI may still configure logging from alembic.ini.
    config.attributes["configure_logger"] = False
    config.set_main_option(
        "script_location",
        str(root / "src" / "experience_hub" / "storage" / "migrations"),
    )
    rendered_url = url.render_as_string(hide_password=False).replace("%", "%%")
    config.set_main_option("sqlalchemy.url", rendered_url)
    return config


def _current_revision(url: URL) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def _known_ancestors(script: ScriptDirectory, head: str) -> frozenset[str]:
    return frozenset(
        revision.revision for revision in script.iterate_revisions(head, "base")
    )


def _migrate_to_head_sync(settings: Settings) -> str:
    url = _synchronous_sqlite_url(settings)
    config = _alembic_config(url)
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()
    if head is None:
        raise RuntimeError("Alembic migration history has no single head")

    current = _current_revision(url)
    if current is not None:
        try:
            script.get_revision(current)
        except CommandError as error:
            raise SchemaRevisionError(
                current_revision=current,
                expected_revision=head,
            ) from error
        if current not in _known_ancestors(script, head):
            raise SchemaRevisionError(
                current_revision=current,
                expected_revision=head,
            )

    command.upgrade(config, "head")
    migrated = _current_revision(url)
    if migrated != head:
        raise RuntimeError("Database migration did not reach the expected head")
    return head


async def migrate_to_head(settings: Settings) -> str:
    """Migrate a supported file database without blocking the event loop."""
    try:
        return await asyncio.to_thread(_migrate_to_head_sync, settings)
    except BaseException as error:
        if is_sqlite_lock_error(error):
            raise DatabaseBusy from error
        raise


type ContainerFactory = Callable[..., ApplicationContainer]
type SchemaMigrator = Callable[[Settings], Awaitable[str]]


class ApplicationRuntime:
    """Initialize the same dependency graph for ASGI and one-shot commands."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
        *,
        container_factory: ContainerFactory = ApplicationContainer.build,
        migrator: SchemaMigrator = migrate_to_head,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        self.settings = settings
        self.clock = clock
        self.ids = ids
        self._container_factory = container_factory
        self._migrator = migrator
        self._active = False

    @asynccontextmanager
    async def initialize(
        self,
        *,
        start_lifecycle_worker: bool,
        recover_interrupted: bool,
    ) -> AsyncIterator[ApplicationContainer]:
        """Yield only after schema and retained-state validation succeeds."""
        if not isinstance(start_lifecycle_worker, bool):
            raise TypeError("start_lifecycle_worker must be a bool")
        if not isinstance(recover_interrupted, bool):
            raise TypeError("recover_interrupted must be a bool")
        if self._active:
            raise RuntimeError("ApplicationRuntime is already active")
        self._active = True
        container: Any = None
        try:
            container = self._container_factory(
                settings=self.settings,
                clock=self.clock,
                ids=self.ids,
            )
            container.schema_revision = await self._migrator(self.settings)
            await container.projection_manager.validate_startup(container.database)
            if recover_interrupted:
                await container.inspiration_recovery.recover()
            if start_lifecycle_worker:
                container.lifecycle_worker.start()
            yield container
        finally:
            try:
                if container is not None:
                    await container.close()
            finally:
                self._active = False


__all__ = [
    "ApplicationRuntime",
    "SchemaRevisionError",
    "SchemaVersionError",
    "migrate_to_head",
]
