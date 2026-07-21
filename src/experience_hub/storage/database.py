"""Async SQLite connection and transaction boundaries."""

import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from experience_hub.domain.events import EventRegistry
from experience_hub.errors import DomainError
from experience_hub.storage.event_store import EventStore
from experience_hub.storage.faults import FaultInjector, ignore_faults
from experience_hub.storage.projection_contracts import (
    ProjectionApplier,
)
from experience_hub.storage.projections import ProjectionManager
from experience_hub.storage.unit_of_work import UnitOfWork

_PAYLOAD_REWRITE_GUARD_KEY = "experience_hub_payload_rewrite_allowed"


class DatabaseBusy(DomainError):
    retry_after = 5

    def __init__(self) -> None:
        super().__init__(
            "database_busy",
            "The database is busy; retry the request",
            status_code=503,
        )


def is_sqlite_lock_error(error: BaseException) -> bool:
    """Return whether SQLAlchemy wrapped SQLite BUSY or LOCKED."""
    if not isinstance(error, OperationalError):
        return False
    code = getattr(error.orig, "sqlite_errorcode", None)
    return isinstance(code, int) and code & 0xFF in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }


async def _rollback_after_failure(session: AsyncSession) -> None:
    with suppress(Exception):
        await session.rollback()


@contextmanager
def payload_rewrite_guard(connection: AsyncConnection) -> Iterator[None]:
    """Open the payload trigger gate only on one checked-out DBAPI connection."""
    sync_connection = connection.sync_connection
    if sync_connection is None:
        raise RuntimeError("Payload rewrite guard has no synchronous connection")
    info = sync_connection.info
    if _PAYLOAD_REWRITE_GUARD_KEY not in info:
        raise RuntimeError("Payload rewrite guard is unavailable on this connection")
    if info[_PAYLOAD_REWRITE_GUARD_KEY]:
        raise RuntimeError("Nested payload rewrite guards are not allowed")
    info[_PAYLOAD_REWRITE_GUARD_KEY] = True
    try:
        yield
    finally:
        info[_PAYLOAD_REWRITE_GUARD_KEY] = False


class Database:
    """Own the SQLAlchemy engine and expose explicit unit-of-work boundaries."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        event_registry: EventRegistry,
        projection_applier: ProjectionApplier,
        fault_injector: FaultInjector,
    ) -> None:
        self._engine = engine
        self._event_store = EventStore(event_registry)
        self._projection_applier = projection_applier
        self._fault_injector = fault_injector
        self._sessions = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @classmethod
    def create(
        cls,
        url: str,
        *,
        event_registry: EventRegistry | None = None,
        projection_applier: ProjectionApplier | None = None,
        fault_injector: FaultInjector | None = None,
        busy_timeout_ms: int = 5000,
    ) -> "Database":
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or busy_timeout_ms < 0
        ):
            raise ValueError("busy_timeout_ms must be a non-negative integer")
        engine = create_async_engine(url)

        @event.listens_for(engine.sync_engine, "connect")
        def configure_sqlite(connection: Any, connection_record: Any) -> None:
            connection_record.info[_PAYLOAD_REWRITE_GUARD_KEY] = False
            connection.create_function(
                "experience_hub_payload_rewrite_allowed",
                0,
                lambda: int(
                    bool(
                        connection_record.info.get(
                            _PAYLOAD_REWRITE_GUARD_KEY,
                            False,
                        )
                    )
                ),
            )
            cursor = connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            finally:
                cursor.close()

        @event.listens_for(engine.sync_engine, "checkout")
        def reset_payload_guard_on_checkout(
            _: Any,
            connection_record: Any,
            __: Any,
        ) -> None:
            connection_record.info[_PAYLOAD_REWRITE_GUARD_KEY] = False

        @event.listens_for(engine.sync_engine, "checkin")
        def reset_payload_guard_on_checkin(_: Any, connection_record: Any) -> None:
            connection_record.info[_PAYLOAD_REWRITE_GUARD_KEY] = False

        return cls(
            engine,
            event_registry=event_registry or EventRegistry(),
            projection_applier=projection_applier or ProjectionManager(),
            fault_injector=(
                fault_injector if fault_injector is not None else ignore_faults
            ),
        )

    def read_session(self) -> AbstractAsyncContextManager[AsyncSession]:
        return self._read_session()

    @asynccontextmanager
    async def _read_session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessions() as session:
            yield session

    def transaction(
        self,
        *,
        immediate: bool = False,
        exclusive: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        if immediate and exclusive:
            raise ValueError(
                "immediate and exclusive transaction modes are mutually exclusive"
            )
        return self._transaction(immediate=immediate, exclusive=exclusive)

    @asynccontextmanager
    async def _transaction(
        self,
        *,
        immediate: bool,
        exclusive: bool,
    ) -> AsyncIterator[UnitOfWork]:
        async with self._sessions() as session:
            try:
                if immediate:
                    await session.execute(text("BEGIN IMMEDIATE"))
                elif exclusive:
                    await session.execute(text("BEGIN EXCLUSIVE"))
                else:
                    await session.begin()

                yield UnitOfWork(
                    session=session,
                    event_store=self._event_store,
                    projection_applier=self._projection_applier,
                    immediate=immediate,
                    fault_injector=self._fault_injector,
                )
            except BaseException as error:
                await _rollback_after_failure(session)
                if is_sqlite_lock_error(error):
                    raise DatabaseBusy from error
                raise
            else:
                try:
                    await session.commit()
                except BaseException as error:
                    await _rollback_after_failure(session)
                    if is_sqlite_lock_error(error):
                        raise DatabaseBusy from error
                    raise

    async def dispose(self) -> None:
        await self._engine.dispose()
