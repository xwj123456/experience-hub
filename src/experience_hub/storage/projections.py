"""Replayable projection reducers, verification, and atomic repair."""

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from experience_hub.domain.events import StoredEvent
from experience_hub.storage.projection_contracts import ProjectionReducer
from experience_hub.storage.tables import (
    DomainEventRow,
    IdempotencyRecordRow,
    ProjectionVersionRow,
)

if False:  # pragma: no cover - imports used only by static type checkers
    from experience_hub.storage.database import Database

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FLOAT_QUANTUM = Decimal("0.000000000001")
_PHYSICAL_COLUMNS = frozenset({"rowid", "codec", "payload"})


class SourceValidation(Protocol):
    async def validate(self, session: AsyncSession) -> None: ...


@dataclass(frozen=True, slots=True)
class ProjectionDiff:
    projection: str
    online_hash: str
    rebuilt_hash: str
    differing_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationReport:
    event_head: int
    differences: tuple[ProjectionDiff, ...]

    @property
    def matches(self) -> bool:
        return not self.differences


class ProjectionMismatch(RuntimeError):
    """The online projection is not equal to a replay at the same event head."""

    code = "projection_mismatch"

    def __init__(self, report: VerificationReport) -> None:
        super().__init__("Projection verification failed")
        self.report = report


class ReducerVersionMismatch(RuntimeError):
    """The stored reducer version cannot be handled by the running code."""

    code = "reducer_version_mismatch"


class EventHeadChanged(RuntimeError):
    """The event ledger changed during a supposedly exclusive repair."""

    code = "event_head_changed"


class MaintenanceBlockedByInflight(RuntimeError):
    """Repair cannot run while a durable command receipt is in progress."""

    code = "maintenance_blocked_by_inflight"


class SourceValidatorRequired(RuntimeError):
    """Replayable reducers cannot operate without source validation."""

    code = "source_validator_required"


class ProjectionRegistry:
    def __init__(self, reducers: Iterable[ProjectionReducer] = ()) -> None:
        self._reducers: dict[str, ProjectionReducer] = {}
        for reducer in reducers:
            self.register(reducer)

    @property
    def reducers(self) -> tuple[ProjectionReducer, ...]:
        return tuple(self._reducers.values())

    def register(self, reducer: ProjectionReducer) -> None:
        name = reducer.name
        if not name or name != name.strip():
            raise ValueError("Projection name must be a non-empty trimmed string")
        if reducer.version <= 0:
            raise ValueError("Projection reducer version must be positive")
        if name in self._reducers:
            raise ValueError(f"Projection {name!r} is already registered")
        _quoted_identifier(name)
        self._reducers[name] = reducer


def _quoted_identifier(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


def _table_ref(schema: str, table_name: str) -> str:
    if schema not in {"main", "temp"}:
        raise ValueError(f"Unsupported SQLite schema: {schema!r}")
    return f'{_quoted_identifier(schema)}.{_quoted_identifier(table_name)}'


def _normalize_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return value
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Projection datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Projection floats must be finite")
        quantized = Decimal(str(0.0 if value == 0 else value)).quantize(
            _FLOAT_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        return format(quantized, ".12f")
    if isinstance(value, Decimal):
        return format(value.quantize(_FLOAT_QUANTUM, rounding=ROUND_HALF_EVEN), ".12f")
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, str):
        candidate = value.lstrip()
        if candidate.startswith(("{", "[")):
            try:
                return _canonical_value(json.loads(value))
            except (json.JSONDecodeError, ValueError):
                pass
        return _normalize_timestamp(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)


def _canonical_row(
    row: Mapping[str, Any],
    *,
    excluded_columns: frozenset[str] = _PHYSICAL_COLUMNS,
) -> dict[str, Any]:
    return {
        key: _canonical_value(value)
        for key, value in sorted(row.items())
        if key not in excluded_columns
    }


def _sqlite_affinity(declared_type: str) -> str:
    normalized = declared_type.upper()
    if "INT" in normalized:
        return "numeric"
    if any(token in normalized for token in ("CHAR", "CLOB", "TEXT")):
        return "text"
    if any(token in normalized for token in ("REAL", "FLOA", "DOUB")):
        return "numeric"
    if not normalized or "BLOB" in normalized:
        return "blob"
    return "numeric"


def _decimal_key(value: int | float | Decimal | str) -> Decimal | None:
    try:
        decimal = Decimal(str(value))
    except ArithmeticError:
        return None
    if not decimal.is_finite():
        raise ValueError("Projection primary-key numbers must be finite")
    return decimal


def _sqlite_key_value(value: Any, declared_type: str) -> tuple[int, Any]:
    """Mirror SQLite's NULL, numeric, text, blob storage-class ordering."""
    if value is None:
        return (0, 0)
    affinity = _sqlite_affinity(declared_type)
    if affinity == "text" and isinstance(value, (bool, int, float, Decimal)):
        return (2, _normalize_timestamp(str(value)))
    if isinstance(value, bool):
        return (1, Decimal(int(value)))
    if isinstance(value, (int, float, Decimal)):
        numeric = _decimal_key(value)
        assert numeric is not None
        return (1, numeric)
    if isinstance(value, str):
        if affinity == "numeric":
            numeric = _decimal_key(value)
            if numeric is not None:
                return (1, numeric)
        return (2, _normalize_timestamp(value))
    if isinstance(value, bytes):
        return (3, value)
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return (4, encoded)


def _sort_key(
    row: Mapping[str, Any],
    primary_key: tuple[str, ...],
    primary_key_types: tuple[str, ...],
) -> tuple[tuple[int, Any], ...]:
    return tuple(
        _sqlite_key_value(row[column], declared_type)
        for column, declared_type in zip(
            primary_key,
            primary_key_types,
            strict=True,
        )
    )


def canonical_projection_hash(
    rows: Iterable[Mapping[str, Any]],
    *,
    projection: str,
    primary_key: tuple[str, ...],
    primary_key_types: tuple[str, ...] | None = None,
    reducer_version: int,
    checkpoint: int,
    excluded_columns: frozenset[str] = _PHYSICAL_COLUMNS,
) -> str:
    """Hash semantic rows plus the reducer version and relevant-event checkpoint."""
    declared_types = primary_key_types or ("",) * len(primary_key)
    if len(declared_types) != len(primary_key):
        raise ValueError(
            "Primary-key columns and declared types must have equal length"
        )
    ordered = sorted(
        rows,
        key=lambda row: _sort_key(row, primary_key, declared_types),
    )
    document = {
        "checkpoint": checkpoint,
        "projection": projection,
        "reducer_version": reducer_version,
        "rows": [
            _canonical_row(row, excluded_columns=excluded_columns) for row in ordered
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


@dataclass(slots=True)
class _Rebuild:
    prefix: str
    tables: dict[str, str]
    checkpoints: dict[str, int]


class _RebuildFailed(Exception):
    def __init__(self, error: BaseException, rebuild: _Rebuild) -> None:
        super().__init__(str(error))
        self.error = error
        self.rebuild = rebuild


class ProjectionManager:
    """Apply reducers online and replay them for verification or repair."""

    def __init__(
        self,
        registry: ProjectionRegistry | None = None,
        *,
        source_validator: SourceValidation | None = None,
    ) -> None:
        self.registry = registry or ProjectionRegistry()
        self._source_validator = source_validator
        self._require_source_validator()

    def _require_source_validator(self) -> None:
        if self.registry.reducers and self._source_validator is None:
            raise SourceValidatorRequired(
                "A non-empty projection registry requires a source validator"
            )

    async def apply(
        self,
        *,
        session: AsyncSession,
        events: Sequence[StoredEvent],
    ) -> None:
        for event in sorted(events, key=lambda item: item.event_id):
            await self.apply_event(session=session, event=event)

    async def apply_event(
        self,
        *,
        session: AsyncSession,
        event: StoredEvent,
    ) -> None:
        for reducer in self.registry.reducers:
            if event.event_type not in reducer.event_types:
                continue
            version = await session.get(ProjectionVersionRow, reducer.name)
            if version is not None and version.reducer_version != reducer.version:
                raise ReducerVersionMismatch(
                    f"Projection {reducer.name!r} stores reducer version "
                    f"{version.reducer_version}, running version is {reducer.version}"
                )
            if version is not None and event.event_id <= version.last_applied_event_id:
                continue
            await reducer.apply(session, event)
            if version is None:
                version = ProjectionVersionRow(
                    name=reducer.name,
                    reducer_version=reducer.version,
                    last_applied_event_id=event.event_id,
                )
                session.add(version)
            else:
                version.last_applied_event_id = event.event_id
                version.last_verified_hash = None
                version.last_verified_at = None
            await session.flush()

    async def verify(self, database: "Database") -> VerificationReport:
        pending_error: BaseException | None = None
        report: VerificationReport
        async with database.read_session() as session, session.begin():
            await self._validate_sources(session)
            await self._check_versions(session)
            event_head = await self._event_head(session)
            rebuild: _Rebuild | None = None
            try:
                rebuild = await self._rebuild(session, event_head)
                report = await self._compare(session, rebuild, event_head)
                if await self._event_head(session) != event_head:
                    pending_error = EventHeadChanged(
                        "Event head changed during projection verification"
                    )
                elif not report.matches:
                    pending_error = ProjectionMismatch(report)
            except _RebuildFailed as error:
                rebuild = error.rebuild
                pending_error = error.error
            except BaseException as error:
                pending_error = error
            finally:
                if rebuild is not None:
                    await self._drop_temp_tables(session, rebuild.tables.values())
        if pending_error is not None:
            raise pending_error
        return report

    async def validate_startup(self, database: "Database") -> None:
        """Fail closed on source corruption or unsupported reducer state.

        Startup validation deliberately avoids rebuilding or changing projections.
        Full projection equality remains the responsibility of :meth:`verify`.
        """
        async with database.read_session() as session, session.begin():
            await self._validate_sources(session)
            await self._check_versions(session)

    async def repair(self, database: "Database") -> VerificationReport:
        report: VerificationReport
        async with database.transaction(exclusive=True) as uow:
            session = uow.session
            in_progress = await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecordRow)
                .where(IdempotencyRecordRow.state == "in_progress")
            )
            if in_progress:
                raise MaintenanceBlockedByInflight(
                    "Projection repair is blocked by an in-progress receipt"
                )
            await self._validate_sources(session)
            event_head = await self._event_head(session)
            rebuild: _Rebuild | None = None
            try:
                try:
                    rebuild = await self._rebuild(session, event_head)
                except _RebuildFailed as error:
                    rebuild = error.rebuild
                    raise error.error from error
                if await self._event_head(session) != event_head:
                    raise EventHeadChanged(
                        "Event head changed during projection repair"
                    )
                rebuilt_hashes = await self._hashes(session, rebuild, rebuilt=True)
                await self._swap(session, rebuild)
                post_hashes = await self._hashes(session, rebuild, rebuilt=False)
                differences = self._hash_differences(
                    rebuilt_hashes,
                    post_hashes,
                    key_details={name: () for name in rebuilt_hashes},
                    online_first=False,
                )
                report = VerificationReport(event_head, differences)
                if differences:
                    raise ProjectionMismatch(report)
                verified_at = datetime.now(UTC)
                for reducer in self.registry.reducers:
                    version = await session.get(ProjectionVersionRow, reducer.name)
                    assert version is not None
                    version.last_verified_hash = post_hashes[reducer.name]
                    version.last_verified_at = verified_at
                await session.flush()
            except BaseException:
                await session.rollback()
                if rebuild is not None:
                    await self._drop_temp_tables(session, rebuild.tables.values())
                    await session.commit()
                raise
            else:
                if rebuild is not None:
                    await self._drop_temp_tables(session, rebuild.tables.values())
        return report

    async def _validate_sources(self, session: AsyncSession) -> None:
        self._require_source_validator()
        if self._source_validator is not None:
            await self._source_validator.validate(session)

    async def _check_versions(self, session: AsyncSession) -> None:
        for reducer in self.registry.reducers:
            version = await session.get(ProjectionVersionRow, reducer.name)
            if version is not None and version.reducer_version != reducer.version:
                raise ReducerVersionMismatch(
                    f"Projection {reducer.name!r} stores reducer version "
                    f"{version.reducer_version}, running version is {reducer.version}"
                )

    async def _event_head(self, session: AsyncSession) -> int:
        return int(
            await session.scalar(select(func.max(DomainEventRow.event_id))) or 0
        )

    async def _rebuild(self, session: AsyncSession, event_head: int) -> _Rebuild:
        _ = event_head
        prefix = f"_rebuild_{uuid4().hex}_"
        tables = {
            reducer.name: f"{prefix}{reducer.name}"
            for reducer in self.registry.reducers
        }
        checkpoints = {reducer.name: 0 for reducer in self.registry.reducers}
        rebuild = _Rebuild(
            prefix=prefix,
            tables=tables,
            checkpoints=checkpoints,
        )
        try:
            for reducer in self.registry.reducers:
                await reducer.rebuild(session, prefix)
                exists = await session.scalar(
                    text(
                        "SELECT count(*) FROM temp.sqlite_master "
                        "WHERE type = 'table' AND name = :name"
                    ),
                    {"name": tables[reducer.name]},
                )
                if exists != 1:
                    raise RuntimeError(
                        f"Reducer {reducer.name!r} did not create its TEMP table"
                    )
                event_types = tuple(sorted(reducer.event_types))
                if event_types:
                    parameters = {
                        f"event_type_{index}": event_type
                        for index, event_type in enumerate(event_types)
                    }
                    placeholders = ", ".join(f":{name}" for name in parameters)
                    checkpoints[reducer.name] = int(
                        await session.scalar(
                            text(
                                "SELECT max(event_id) FROM main.domain_events "
                                f"WHERE event_type IN ({placeholders}) "
                                "AND event_id <= :event_head"
                            ),
                            {**parameters, "event_head": event_head},
                        )
                        or 0
                    )
            return rebuild
        except BaseException as error:
            raise _RebuildFailed(error, rebuild) from error

    async def _rows(
        self,
        session: AsyncSession,
        schema: str,
        table_name: str,
        primary_key: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        ordering = ", ".join(_quoted_identifier(column) for column in primary_key)
        result = await session.execute(
            text(
                f"SELECT * FROM {_table_ref(schema, table_name)} ORDER BY {ordering}"
            )
        )
        return [dict(row) for row in result.mappings()]

    async def _primary_key(
        self,
        session: AsyncSession,
        table_name: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        result = await session.execute(
            text(f"PRAGMA main.table_info({_quoted_identifier(table_name)})")
        )
        keyed = sorted(
            (int(row["pk"]), str(row["name"]), str(row["type"]))
            for row in result.mappings()
            if int(row["pk"]) > 0
        )
        if not keyed:
            raise RuntimeError(
                f"Projection table {table_name!r} must declare a primary key"
            )
        return (
            tuple(name for _, name, _ in keyed),
            tuple(declared_type for _, _, declared_type in keyed),
        )

    async def _hashes(
        self,
        session: AsyncSession,
        rebuild: _Rebuild,
        *,
        rebuilt: bool,
    ) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for reducer in self.registry.reducers:
            primary_key, primary_key_types = await self._primary_key(
                session,
                reducer.name,
            )
            if rebuilt:
                table_name = rebuild.tables[reducer.name]
                checkpoint = rebuild.checkpoints[reducer.name]
            else:
                table_name = reducer.name
                version = await session.get(ProjectionVersionRow, reducer.name)
                checkpoint = 0 if version is None else version.last_applied_event_id
            rows = await self._rows(
                session,
                "temp" if rebuilt else "main",
                table_name,
                primary_key,
            )
            hashes[reducer.name] = canonical_projection_hash(
                rows,
                projection=reducer.name,
                primary_key=primary_key,
                primary_key_types=primary_key_types,
                reducer_version=reducer.version,
                checkpoint=checkpoint,
            )
        return hashes

    async def _compare(
        self,
        session: AsyncSession,
        rebuild: _Rebuild,
        event_head: int,
    ) -> VerificationReport:
        online_hashes = await self._hashes(session, rebuild, rebuilt=False)
        rebuilt_hashes = await self._hashes(session, rebuild, rebuilt=True)
        details: dict[str, tuple[str, ...]] = {}
        for reducer in self.registry.reducers:
            if online_hashes[reducer.name] == rebuilt_hashes[reducer.name]:
                continue
            primary_key, _ = await self._primary_key(session, reducer.name)
            online = await self._rows(
                session, "main", reducer.name, primary_key
            )
            rebuilt = await self._rows(
                session,
                "temp",
                rebuild.tables[reducer.name],
                primary_key,
            )
            details[reducer.name] = _differing_keys(
                online, rebuilt, primary_key
            )
        differences = self._hash_differences(
            online_hashes,
            rebuilt_hashes,
            key_details=details,
            online_first=True,
        )
        return VerificationReport(event_head, differences)

    @staticmethod
    def _hash_differences(
        online_hashes: Mapping[str, str],
        rebuilt_hashes: Mapping[str, str],
        *,
        key_details: Mapping[str, tuple[str, ...]],
        online_first: bool,
    ) -> tuple[ProjectionDiff, ...]:
        differences = []
        for name in online_hashes:
            online_hash = (
                online_hashes[name] if online_first else rebuilt_hashes[name]
            )
            rebuilt_hash = (
                rebuilt_hashes[name] if online_first else online_hashes[name]
            )
            if online_hash == rebuilt_hash:
                continue
            differences.append(
                ProjectionDiff(
                    projection=name,
                    online_hash=online_hash,
                    rebuilt_hash=rebuilt_hash,
                    differing_keys=key_details.get(name, ()),
                )
            )
        return tuple(differences)

    async def _swap(self, session: AsyncSession, rebuild: _Rebuild) -> None:
        reducers = self.registry.reducers
        for reducer in reversed(reducers):
            online = _table_ref("main", reducer.name)
            await session.execute(text(f"DELETE FROM {online}"))

        for reducer in reducers:
            online = _table_ref("main", reducer.name)
            temporary = _table_ref("temp", rebuild.tables[reducer.name])
            columns_result = await session.execute(
                text(
                    f"PRAGMA main.table_info({_quoted_identifier(reducer.name)})"
                )
            )
            columns = [
                str(row["name"]) for row in columns_result.mappings()
            ]
            column_sql = ", ".join(_quoted_identifier(column) for column in columns)
            await session.execute(
                text(
                    f"INSERT INTO {online} ({column_sql}) "
                    f"SELECT {column_sql} FROM {temporary}"
                )
            )
            version = await session.get(ProjectionVersionRow, reducer.name)
            if version is None:
                version = ProjectionVersionRow(
                    name=reducer.name,
                    reducer_version=reducer.version,
                    last_applied_event_id=rebuild.checkpoints[reducer.name],
                )
                session.add(version)
            else:
                version.reducer_version = reducer.version
                version.last_applied_event_id = rebuild.checkpoints[reducer.name]
                version.last_verified_hash = None
                version.last_verified_at = None
        await session.flush()

    async def _drop_temp_tables(
        self, session: AsyncSession, table_names: Iterable[str]
    ) -> None:
        for table_name in table_names:
            table = _table_ref("temp", table_name)
            await session.execute(text(f"DROP TABLE IF EXISTS {table}"))


def _key_string(row: Mapping[str, Any], primary_key: tuple[str, ...]) -> str:
    values = [_canonical_value(row[column]) for column in primary_key]
    if len(values) == 1:
        return str(values[0])
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _differing_keys(
    online: Sequence[Mapping[str, Any]],
    rebuilt: Sequence[Mapping[str, Any]],
    primary_key: tuple[str, ...],
) -> tuple[str, ...]:
    online_rows = {
        _key_string(row, primary_key): _canonical_row(row) for row in online
    }
    rebuilt_rows = {
        _key_string(row, primary_key): _canonical_row(row) for row in rebuilt
    }
    return tuple(
        key
        for key in sorted(set(online_rows) | set(rebuilt_rows))
        if online_rows.get(key) != rebuilt_rows.get(key)
    )[:50]
