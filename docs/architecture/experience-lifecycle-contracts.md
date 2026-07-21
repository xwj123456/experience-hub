# Experience Lifecycle and Retrieval Contracts

This document fixes the package, transaction, event, recall, and maintenance
contracts delivered by the experience lifecycle and retrieval slice. The
public Python surface is enforced by
`tests/contract/test_experience_exports.py`. Symbols that are not re-exported
from their owning package are implementation details, even when their leaf
modules remain importable.

## Public package boundaries

| Package | Stable responsibility |
|---|---|
| `experience_hub.experiences` | Immutable experience values, command values, state snapshots, owner-scoped queries, and transaction-bound writers and services |
| `experience_hub.retrieval` | Multilingual query/result values, ranking mode, mutating retrieval, and the session-bound read-only evidence reader |
| `experience_hub.lifecycle` | Activation and rehearsal scoring, lifecycle policy evaluation, transactional cycle execution, and the idea-archive extension point |

The package exports are the original objects from their leaf modules. They
are not wrapper classes or reconstructed type aliases. Heavy query and
service exports are resolved lazily so importing the three packages, their
leaf modules, or `import *` in any order does not create a circular import.

Cross-feature consumers use these package boundaries. In particular:

- sharing reads an owned current or historical version through
  `ExperienceQuery.get_owned_shareable_version`;
- retrieval mutates through a caller-supplied `UnitOfWork`;
- inspiration reads bounded evidence through
  `ExperienceEvidenceReader.peek(session=..., query=...)`;
- lifecycle extends idea expiry through `IdeaArchivePlanner`; and
- these cross-feature read and service contracts do not return or accept ORM
  rows. The exported `ExperienceRepository` is intentionally a lower-level
  persistence boundary and may expose mapped rows to repository-layer callers.

Both shared read contracts are asynchronous and session-bound.
`get_owned_shareable_version` returns `ShareableExperienceVersion`, while
`peek` returns `SearchResult`. Neither method opens or commits a transaction.

## Authoritative sources and rebuildable projections

The experience data model deliberately separates three kinds of durable
state.

| Class | Tables | Rule |
|---|---|---|
| Authoritative source | `experiences`, `experience_versions`, `experience_payloads`, `experience_links` | Validate and retain; projection repair never replaces these rows |
| Ordered audit source | `domain_events` | Replay in `event_id` order after validating aggregate sequence, schema, causation, and source correspondence |
| Rebuildable projection | `experience_state`, `experience_terms` | Derive completely from validated source anchors and ordered events |

An experience identity has immutable owner, kind, origin, and creation time.
Its versions are contiguous from one, form an adjacent supersession chain,
and each have exactly one payload and one explaining
`experience.version_created` event. The decoded canonical payload hash and
the semantic content hash are recomputed using the identity's kind. Link rows
must exactly equal the canonical links in their explaining version event,
remain within one owner, and may target only an experience whose creation
event precedes the source version event in ledger order. The latest source
version determines current-content uniqueness per owner.

`experience_state` contains the current lifecycle snapshot and projection
checkpoint. `experience_terms` contains the deterministic terms for the
current version. Reducer version 1 creates temporary rebuild tables from
declared schema rather than copying either online projection, then reads only
authoritative anchors and ordered events. Incremental and rebuilt rows must
serialize identically.

Payload codec is a physical representation exception, not a projection.
Decoded payload bytes, `payload_hash`, version metadata, and `content_hash`
remain semantically immutable. Hot and warm experiences prefer `plain`;
cold and archived experiences prefer `zlib`. Projection rebuild never changes
either codec or payload bytes.

## Event ordering and state transitions

Every event sequence below is appended and projected atomically under one
causation receipt. Aggregate sequence numbers are contiguous from one.

| Operation | Exact emitted order |
|---|---|
| Create | `experience.created`, `experience.version_created` |
| Correct content | `experience.version_created` |
| Access hot or warm content | `experience.accessed` |
| Expand and reactivate cold content | `experience.accessed`, `experience.reactivated`, `experience.temperature_changed` with cause `cold_reactivation` |
| Eligible lifecycle evaluation without transition | `experience.lifecycle_evaluated` |
| Lifecycle promotion or demotion | `experience.lifecycle_evaluated`, `experience.temperature_changed` |
| Lifecycle archive | `experience.lifecycle_evaluated`, `experience.archived`, `experience.temperature_changed` with cause `policy_archive` |
| Restore archived memory | `experience.restored`, `experience.temperature_changed` with cause `restore` |

Confirm, refute, pin, and unpin first emit their named event when they change
state. A required promotion follows as
`experience.temperature_changed`; the shared transition event performs the
temperature change and resets hysteresis. Commands that are already in the
requested pin state complete without inventing an event.

Every event carries complete strict before/after snapshots where its schema
requires them. Transaction-bound writers validate actor policy and exact
multi-event sequences. Projectors verify the prior checkpoint, source owner,
current version and content hash, causal time, allowed field delta, and
paired-event causation before applying the next snapshot. Event payloads
never contain body bytes or raw retrieval text.

## Access, cold recall, and expansion thresholds

Direct GET and cue-based search have deliberately different authority.

- Direct GET of hot or warm memory returns the full body and records access.
- Direct GET of cold or archived memory returns metadata with
  `blurred=true`, no body, and no access or temperature mutation.
- Archived memory is absent from search until explicitly restored.
- Search expands hot or warm bodies only when each body fits the remaining
  global UTF-8 content budget.
- Cold search expansion additionally requires `expand_cold=true` and the
  mode-specific signal at or above its inclusive threshold:
  focused lexical-or-trigram relevance `>= 0.72`, or associative mechanism
  relevance `>= 0.65`.
- A candidate must first pass the lower ranking admission threshold:
  focused lexical-or-trigram relevance `>= 0.05`, or associative lexical
  relevance `>= 0.02` or mechanism relevance `>= 0.02`.
- An irrelevant cue, a signal below the cold threshold, or insufficient body
  budget never expands or reactivates cold content.

A successful mutating search fixes the response and all access intents before
writing. A cold body is reactivated only when it is actually included in that
fixed response. The reactivation event stores a canonical query hash, mode,
and numeric matching signal, never the query text; the preceding access event
contains only access-state evidence.

`PeekExperiences` is an internal cross-feature query, not a public HTTP escape
hatch. `ExperienceEvidenceReader.peek` uses only its supplied consistent
session, returns UTF-8-safe bounded excerpts, and never emits access,
reactivation, or temperature events.

## Deterministic multilingual retrieval

Index and query text share one closed tokenizer:

1. normalize with Unicode NFKC;
2. apply Unicode case folding and NFKC again;
3. turn punctuation, control characters, and whitespace into boundaries;
4. collapse spaces;
5. extract contiguous Unicode Latin-script words; and
6. generate padded Unicode character trigrams for every normalized script.

This makes English word matching and Chinese or mixed-language trigram
matching part of one deterministic index. Tags, mechanisms, words, and
trigrams retain distinct kinds and locked weights. Duplicate cues keep the
maximum weight, and final cues are sorted canonically. Candidate selection,
temperature pools, ranking, and UUID tie-breaking are deterministic; no
language detector, locale state, network service, or wall-clock behavior is
consulted.

## Lifecycle, rehearsal, and forgetting

Access increments `access_count`, adds bounded rehearsal strength, records
`last_accessed_at`, and materializes activation. Strength and recency decay
from UTC timestamps under `LifecycleConfig`; all numeric inputs are finite and
bounded.

The default lifecycle policy is:

| Transition | Inclusive/exclusive rule |
|---|---|
| Warm to hot | activation `>= 0.75`, or pinned; immediate on one eligible cycle |
| Hot to warm | activation `< 0.62`, not pinned; two consecutive eligible cycles |
| Warm to cold | activation `< 0.30`, not pinned; two consecutive eligible cycles |
| Cold to archived | cold for at least 90 days, importance `< 0.75`, confidence `< 0.25`, decayed strength `< 0.10`, unpinned, and no active dependent |

An eligible cycle is separated from the prior lifecycle evaluation by at
least 15 minutes by default. Recovery above a demotion threshold resets the
hysteresis counter. Current nonarchived `derived_from`, `supports`, or `tests`
dependents block archival; `contradicts` links and superseded source versions
do not.

Each cycle ID is UUIDv5 over the canonical evaluation time and lifecycle
configuration hash. Manual and background execution share the same
`LifecycleService`, result codec, singleton lease, and completed-resource
replay. A repeated semantic cycle returns its canonical prior result without
emitting events again. The optional `IdeaArchivePlanner` runs after experience
events and must return deterministic idea events carrying that same cycle ID.

## Transaction and clock ownership

Ordinary mutations use one caller-owned `BEGIN IMMEDIATE` unit of work.
Services and writers receive `UnitOfWork` and `CommandContext`; they do not
reserve a second receipt, open another transaction, commit, or roll back.
Source inserts, ordered events, projections, codec transitions, and receipt
completion therefore succeed or fail together.

Read-only evidence uses one caller-supplied `AsyncSession`. Projection verify
uses one consistent read transaction. Repair holds `BEGIN EXCLUSIVE` through
validation, rebuild, event-head recheck, replacement, and post-swap hash
verification. Payload reconciliation requires an active caller-owned
immediate unit of work and encloses all codec rewrites in one savepoint.

Persisted domain times are aware UTC values from an injected `Clock`.
State-changing commands reject a time behind their aggregate's latest causal
anchor with `clock_regression`. Lifecycle evaluation also rejects future
times and times behind state anchors. Cross-aggregate causality uses ledger
`event_id`, not an assumption that independent aggregate timestamps are
globally monotonic.

## Privacy and owner isolation

All retrieval, direct GET, version sharing, and payload expansion start from
the requested owner. A missing value and a foreign-owned value have the same
not-found behavior. Batched payload loading rechecks every version against
that owner, so a mixed-owner batch fails closed.

Bodies remain in immutable payload rows and do not appear in domain events,
projection mismatch keys, or mutation responses. Cold and archived GET
responses expose metadata only. Search events retain only a SHA-256 query
identity and bounded numeric evidence. Canonical reports use stable IDs and
codes rather than decoded bodies, raw queries, exception text, credentials,
or direct personal data.

## Verify, rebuild, repair, and payload reconciliation

Maintenance is fail-closed:

1. validate causation receipts, foreign keys, contiguous aggregate
   sequences, registered event schemas, semantic hashes, source/event
   correspondence, version history, links, and current-content uniqueness;
2. abort before creating or replacing projection data on any source failure;
3. rebuild state and terms from source anchors plus ordered events in
   connection-local temporary tables;
4. compare canonical rows and report stable bounded mismatch keys; and
5. always drop temporary tables.

`projections rebuild --verify` never changes online state or reducer
metadata. `--repair` refuses retained in-progress work, acquires the exclusive
writer lock, rechecks the event head before replacement, swaps only declared
projection rows and version metadata, and rolls back unless post-swap hashes
exactly equal the rebuilt hashes. Neither mode touches authoritative rows or
payload codecs.

`payloads reconcile` is a separate physical maintenance operation. It
prevalidates every version, reports deterministic changed/skipped/error
counts, and rewrites all historical versions according to the current
temperature preference. Each guarded compare-and-set decodes before and
after, preserves `payload_hash`, `content_hash`, identity kind, and complete
semantic content, then post-validates the refreshed rows. Any rewrite or
validation failure rolls the entire reconciliation pass back; unexpected
programming and database exceptions propagate rather than being converted
into a misleading successful report.
