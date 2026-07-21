# Foundation Contracts

This document fixes the boundaries that later Experience Hub plans may depend
on. Public imports are checked by
`tests/contract/test_foundation_exports.py`. A symbol not exported by one of
the packages below is an implementation detail, even when its leaf module is
importable.

## Public package boundaries

| Package | Stable responsibility |
|---|---|
| `experience_hub` | Canonical JSON and hashing, clocks, ID generators, and base errors |
| `experience_hub.domain` | Strict values, command values, and registered event values |
| `experience_hub.storage` | Transactions, event storage, idempotency, projections, and source validation |
| `experience_hub.agents` | The foundation Agent vertical slice |

Later plans may add names to these packages. They must not remove or replace
the foundation exports, reconstruct type aliases, or make callers import ORM
rows. `ReceiptDecision` and `CommandHandler`, in particular, are re-exported
as the original Python type aliases rather than equivalent new aliases.

ORM table classes, physical payload codecs, controlled fault-injection hooks,
and no-op adapters are not application contracts. Feature services receive
public values and a supplied `UnitOfWork`; they do not import another
feature's storage rows.

## Ordinary command ownership

`CommandExecutor` owns an ordinary command from reservation through commit:

1. Open one `BEGIN IMMEDIATE` transaction.
2. Reserve the caller/operation/idempotency-key receipt.
3. Return a completed receipt byte-for-byte, or return the canonical
   in-progress representation, without invoking the handler.
4. For a new receipt, invoke the handler with the existing `UnitOfWork` and
   `CommandContext`.
5. Insert authoritative rows, append ordered events, and apply affected
   projections in that same transaction.
6. Store the allowed response bytes and complete the receipt.
7. Commit once.

A handler never opens, commits, or owns a second transaction. An unexpected
exception rolls back the receipt reservation together with every source,
event, and projection write. A `ReplayableCommandError` is encoded into a
stable response and completed so the same request can replay it exactly.

The receipt scope and canonical request hash jointly define idempotency. The
request hash covers normalized method and route, path parameters, sorted query
pairs, body, and normalized semantic headers. Reusing a scoped key with
different semantics raises `IdempotencyKeyConflict`. `CommandRequest` keeps
the path and body as recursively immutable canonical snapshots, so the public
request semantics cannot drift away from the hash after construction.

The installed `experience-hub` console entry point must always be loadable
and provide help. The foundation app intentionally exposes no operational
commands; the API/CLI plan extends that same Typer application.

## Inspiration's declared split-transaction exception

The future `InspirationRunExecutor` is the only workflow allowed to own the
lower-level three-transaction receipt protocol. Callers must not wrap it in
`CommandExecutor`.

1. Transaction 1 reserves the receipt, inserts the immutable run, attaches
   its run resource, emits `inspiration.started`, projects `running`, and
   commits the retained `in_progress` receipt.
2. Transaction 2 uses a consistent read to retrieve evidence, inserts the
   frozen snapshot, emits `inspiration.snapshot_frozen`, and commits before a
   generator is invoked.
3. Generator work runs outside a write transaction. Transaction 3 inserts
   validated ideas and occurrences, appends operator/idea/terminal events,
   updates projections, and calls `ReceiptStore.complete_existing` for the
   original receipt in the same commit.

All normal run events retain the original receipt as causation. Cancellation
or process interruption before transaction 1 commits rolls that transaction
back, leaving neither a receipt nor a run; the same request may start
normally. Once transaction 1 has committed, interruption leaves the durable
in-progress receipt and run trace for explicit recovery, and the operation is
not silently regenerated or retried. Mutable provider configuration is not
consulted before resolving completed, in-progress, or conflicting receipts.
No other feature may use this exception merely to shorten an ordinary command
transaction.

## Event ledger ordering

`EventPayload` models are strict, frozen, versioned, and registered under one
explicit event name. Unknown event names, extra payload fields, mismatched
payload classes, and unregistered payload classes are rejected.

Events may be appended only through an immediate `UnitOfWork` and must name
an existing non-null causation receipt. The caller's `PendingEvent` order
determines ledger `event_id` order. Aggregate sequence numbers start at one
and remain contiguous under the serialized writer transaction.

`UnitOfWork.append_events` performs these operations as one atomic unit:

1. allocate aggregate sequences and flush all event rows;
2. pass the resulting `StoredEvent` values to the projection applier;
3. let the enclosing command transaction commit source rows, events,
   projections, and receipt together.

Projection batches are processed by ascending `event_id`, even if a caller
supplies an out-of-order collection. A reducer ignores irrelevant events,
does not advance its checkpoint for them, and treats an event at or below its
checkpoint as already applied.

## Canonicalization

`canonical_json_bytes` is the only foundation JSON representation used for
semantic hashes, event payloads, request identity, and stored response
envelopes. It guarantees:

- UTF-8 encoding with Unicode characters preserved;
- compact JSON with lexicographically sorted object keys;
- aware datetimes normalized to UTC with microsecond precision;
- stable UUID, enum, Pydantic model, list, and tuple representations;
- normalization of negative floating-point zero to zero; and
- rejection of non-finite numbers, naive datetimes, non-string object keys,
  and unsupported values.

`sha256_hex` hashes bytes and returns lowercase hexadecimal. Callers must
hash the canonical bytes specified by their feature contract, not a Python
object's display representation.

`StoredResponse` retains response bytes, content type, and normalized headers.
`CommandExecutor` persists only its allowlisted semantic headers. A completed
receipt replays those stored values byte-for-byte and never re-enters the
handler.

## Stable errors and database contention

External adapters may rely on `DomainError.code`, `message`, `details`, and
`status_code`. Codes are stable machine contracts; raw exceptions, SQL, file
paths, provider output, and stack details must not enter a serialized error.

`DatabaseBusy` is the retryable SQLite contention error:

```text
code = database_busy
status_code = 503
retry_after = 5
```

It is raised only for SQLite `SQLITE_BUSY` or `SQLITE_LOCKED` primary or
extended result codes encountered while beginning, using, or committing a
transaction. Other `OperationalError` values remain diagnostic failures and
must not be mislabeled as contention.

Projection and maintenance errors also keep stable codes:

- `projection_mismatch`
- `reducer_version_mismatch`
- `event_head_changed`
- `maintenance_blocked_by_inflight`
- `source_validator_required`
- `source_integrity_error`

## Projection registration, verification, and repair

Every rebuildable projection has one `ProjectionReducer` with a unique,
non-empty name, a positive integer version, a declared event-name set, and
these fixed methods:

```python
async def apply(session, event) -> None: ...
async def rebuild(session, target_prefix) -> None: ...
```

Reducers are registered in `ProjectionRegistry`. A non-empty registry is
invalid without a `SourceValidator`. Feature plans add their source rules as
uniquely named `SourceValidationHook` implementations before registering the
manager in application composition.

Online application checks the stored reducer version and advances a
projection's relevant-event checkpoint in the writer transaction.

Verification:

- validates authoritative sources before rebuilding;
- uses one consistent read transaction and connection-local temporary tables;
- rebuilds and compares at one event head;
- canonicalizes rows in declared primary-key order;
- leaves online rows and projection-version metadata unchanged; and
- drops temporary tables on both success and failure.

Repair:

- acquires `BEGIN EXCLUSIVE`;
- refuses any durable `in_progress` receipt;
- validates sources before creating rebuild tables;
- rebuilds, rechecks the event head, swaps projection contents, and updates
  reducer/checkpoint metadata while holding the lock;
- compares post-swap hashes to the rebuilt hashes; and
- rolls back all online and version changes on any mismatch.

Repair may reconcile a declared reducer-version upgrade. Ordinary online
application may not silently cross a version mismatch.

Projection hashing includes semantic IDs, semantic hashes, reducer versions,
and checkpoints. It orders by declared key types, sorts nested JSON keys,
normalizes aware UTC timestamps to microseconds, and quantizes floats to
twelve decimal places. It excludes SQLite `rowid` and explicitly physical
codec/payload columns.

## Durable data classes

| Class | Rule |
|---|---|
| Authoritative source | Validate constraints, hashes, and explaining events; repair never replaces it |
| Rebuildable projection | Recreate from validated sources plus ordered events |
| Operational | Validate by its own contract; do not replay as domain state |

Idempotency receipts, projection versions, lifecycle leases, and migration
metadata are operational. Physical encoding may change only through an
explicit guarded feature contract; semantic content hashes remain invariant.

## Determinism and testing

Domain time and persisted IDs come from injected `Clock` and `IdGenerator`
implementations. Persisted datetimes are aware UTC values. Unit and
integration tests use frozen clocks, deterministic IDs, temporary SQLite
files, controlled fault hooks, and explicit lock ownership. Foundation tests
must not depend on wall-clock sleeps, network access, provider availability,
or unspecified row ordering.
