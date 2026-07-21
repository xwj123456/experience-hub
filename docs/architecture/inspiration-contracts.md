# Inspiration and Incubation Contracts

This document fixes the public package, transaction, generation, recovery,
incubation, adoption, and memory-lifecycle boundaries of the inspiration
slice. The public Python surface is enforced by
`tests/contract/test_inspiration_exports.py`. A symbol that is not re-exported
from `experience_hub.inspiration` is an implementation detail even when its
leaf module remains importable.

## Public package boundary

| Group | Stable contracts |
|---|---|
| Run command and views | `StartInspirationRun`, `InspirationRun`, `OperatorOutcome` |
| Idea views and decisions | `Idea`, `IdeaEvaluation`, `AdoptIdea`, `RejectIdea`, `ArchiveIdea`, `IdeaLifecycleService` |
| Frozen evidence | `SnapshotBuilder`, `FrozenSnapshot`, `SnapshotItem`, `ExperienceEvidenceReader`, `InboxEvidenceReader` |
| Generation | `IdeaGenerator`, `ManagedIdeaGenerator`, `GeneratorResult`, `DeterministicIdeaGenerator`, `OpenAICompatibleIdeaGenerator`, `build_idea_generator` |
| Time and budget control | `MonotonicClock`, `DeadlineRunner`, `AsyncioDeadlineRunner`, `BoundedGenerationRunner`, `OperatorGeneration`, `OperatorGenerationRun` |
| Durable run execution | `InspirationRunExecutor`, `GeneratorFactory`, `GenerationRunner` |
| Stored responses | `InspirationResponseCodec`, `InspirationRunResponseV1`, `InspirationErrorResponseV1` |
| Lifecycle integration | `InspirationIdeaArchivePlanner` |

Exports are the original objects from their owning modules, not wrappers or
reconstructed aliases. Importing the package, its generator adapters, or its
service modules in different orders must not create circular-import-dependent
behavior.

`InspirationRunExecutor.execute(request=..., run=...)` is a special
split-transaction workflow. It returns a `StoredResponse` whose canonical JSON
bytes are already final. An API or CLI adapter must pass those bytes, status,
content type, and approved headers through unchanged; it must not wrap this
executor in the ordinary single-transaction `CommandExecutor` or decode and
re-encode its response.

## Evidence is frozen before generation

`SnapshotBuilder.freeze` receives one caller-owned `UnitOfWork`. Its two
evidence ports are asynchronous and session-bound:

- `ExperienceEvidenceReader.peek(session=..., query=...)` reads the owner's
  retrievable experiences without creating access, reactivation, or
  temperature events.
- `InboxEvidenceReader.list_available_pending(session=..., ...)` reads only
  effectively available pending capsules when `include_inbox=true`. Those
  capsule items remain quarantined evidence with fixed source trust `0.25`;
  reading them neither adopts them nor exposes them to ordinary retrieval.

Both ports use the exact `AsyncSession` supplied by the snapshot transaction.
They do not open a connection, start or commit a transaction, or return ORM
rows.

The merged snapshot contains at most 12 items. Each excerpt is truncated at a
valid UTF-8 boundary to at most 2,048 bytes, and the complete canonical
metadata-plus-excerpt document is bounded to 24,576 bytes. Ordering is
deterministic: relevance descending, owned experience before capsule on a tie,
then source and version UUID. Every item carries a stable evidence key derived
from source type, source ID/version, and source content hash. Run-local item
IDs and capture timestamps do not participate in the stable snapshot identity.

After this boundary, generation receives only the goal, canonical context,
operator, explicit budgets, and immutable `SnapshotItem` values. It cannot
re-query live evidence. Source memory may subsequently cool, be archived, be
restored, or receive a new version without changing what the completed run
actually observed.

## The three durable transaction boundaries

Normal synchronous execution has three `BEGIN IMMEDIATE` transactions with
provider work deliberately outside them.

| Phase | Atomic contents | Durable result |
|---|---|---|
| 1. Start | Reserve the original idempotency receipt; validate the selected generator; insert the immutable run configuration; attach the run ID to the receipt; append and project `inspiration.started` | A visible running trace with an attached in-progress receipt |
| 2. Freeze | Read through both session-bound evidence ports; insert immutable snapshot items; append and project `inspiration.snapshot_frozen` | A committed, bounded evidence boundary that generation cannot alter |
| Outside transactions | Run enabled operators sequentially under monotonic token and deadline budgets; validate references and deduplicate returned branches | In-memory sanitized operator outcomes only; no database lock is held |
| 3. Finalize | Re-read current mechanism clusters under the writer lock; insert immutable ideas and occurrences; append idea events and fixed-order operator events; append exactly one terminal run event; project all events; encode the terminal response; complete the original receipt | Ideas, incubation changes, terminal run state, and exact replay bytes commit together |

A normal run captures one aware-UTC logical timestamp in phase 1. Its start,
snapshot, generation-time idea, operator, and terminal events retain that same
time across all three transactions. Elapsed provider deadlines use
`MonotonicClock`; they never consult or persist the domain `Clock` as a
duration source.

If the selected generator is not configured, phase 1 completes the new
receipt with the canonical 422 response and creates no run. If snapshot
preparation fails, phase 2 rolls back and phase 3 records one bounded
`preparation_failed` terminal result. A successful phase 2 therefore always
names the exact immutable snapshot consumed by generation.

## Generation is proposal-only

The deterministic generator is the credential-free baseline. The optional
OpenAI-compatible adapter sends one strict `/chat/completions` request for
each attempted operator. Its request contains a JSON-schema response format
and has no `tools` field. A provider response containing `tool_calls` or
`function_call` is invalid.

Neither generator receives a `UnitOfWork`, repository, tool registry, action
executor, experience writer, or sharing writer. Generation therefore cannot:

- call tools or execute actions;
- mutate an experience, idea decision, capsule, or lifecycle projection;
- adopt evidence or an idea;
- retrieve fresh evidence after the snapshot is frozen; or
- decide that its own proposal is true.

There are no automatic provider retries. Each operator gets at most one
attempt, bounded by the smaller of its per-operator allowance and the
remaining global monotonic budget. Provider timeouts, HTTP failures, invalid
responses, invalid evidence references, token overruns, and exhausted budgets
become fixed sanitized operator codes. One operator's bounded failure does not
discard valid ideas retained from another operator.

The deterministic generator reserves no output tokens. A reserving generator
may start an operator only when `consumed_before + per_operator_reservation`
does not exceed the configured run total; unused reservation can therefore be
reused by a later operator. An empty frozen snapshot calls no generator and
records only zero-accounting `insufficient_evidence` outcomes. Once the global
deadline is exhausted, every remaining operator is a zero-accounting skipped
outcome and the run terminates as `timed_out`.

External task cancellation is not an operator failure. It propagates out,
rolls back the transaction currently in progress, and may leave the already
committed phase-1 or phase-2 trace for startup recovery.

## Idempotency and recovery

The original receipt is reserved before mutable generator configuration is
consulted. The same key and request therefore has three stable outcomes:

- a completed receipt returns its stored response byte-for-byte;
- an attached in-progress receipt returns canonical 409
  `operation_in_progress` with the same run ID; or
- a new receipt starts a new run.

A same-key, different-request hash remains an idempotency conflict. Replaying
a completed run never invokes the generator again.

Startup recovery is ledger-driven and does not resume provider work. A legal
running trace contains `inspiration.started`, optionally followed by
`inspiration.snapshot_frozen`, and no terminal event. It is recoverable only
when its original start receipt still:

- is `in_progress`;
- has the same caller scope, operation scope, and request hash;
- is attached to that exact inspiration run; and
- has no stored response.

Recovery acquires one immediate transaction, reserves a separate
`system:local` recovery receipt, appends exactly one
`inspiration.failed(process_interrupted)` event, and completes both the
original and recovery receipts with the same canonical terminal response. It
does not regenerate ideas, call a provider, refreeze evidence, or modify an
already frozen snapshot. Recovery is the sole timestamp exception: the
failure event uses `max(startup_clock, last_run_event_time)` so a clock that
moved backward cannot violate causal ordering.

A running trace without its exact attached in-progress receipt, a terminal
trace still projected as running, or a completed recovery receipt paired with
a running trace is source corruption. Recovery fails closed instead of
inventing missing provenance.

Source validation also closes causation in both directions. Each run has
exactly one attached start receipt; its causation contains only that run's
start, optional snapshot, retained idea, operator, and normal terminal events.
A recovery receipt causes exactly the one recovery terminal. An unattached
start receipt is valid only for the canonical replayable
`generator_not_configured` response. Provider configuration retained on a run
contains only a safe absolute HTTP(S) base URL and a trimmed model identifier,
never credentials.

## Immutable ideas and mechanism incubation

An accepted generator branch becomes an immutable idea plus exactly one
occurrence anchored to its run and snapshot. Content hashes canonicalize
set-like fields, and mechanism hashes normalize Unicode, case, and
punctuation. Exact and near-duplicate handling is deterministic. Recurring
mechanisms update the separate `mechanism_incubation` projection; they do not
rewrite an earlier idea body.

Mechanism maturity is aggregate evidence, not a truth score:

- recurrence across distinct snapshots can move a mechanism from
  `speculative` to `incubating`;
- effective supported or refuted evaluations update support/refutation
  counts; and
- independent adoption by distinct owners can contribute to candidate
  maturity without exposing another owner's private idea or evidence IDs.

An evaluation records an explicit verdict and evidence. It never runs the
idea's proposed test. Only the latest evaluation revision for one evaluator
and idea is effective.

Per-owner idea state remains independent of aggregate mechanism maturity.
An active idea may be adopted, rejected, or archived. Rejected and adopted
ideas are terminal. A later near-equivalent occurrence can strengthen the
mechanism cluster, but it never silently reopens an archived or rejected
owner decision.

## Explicit adoption is the only bridge into memory

Generating, evaluating, recurring, or promoting a mechanism does not create
an experience. `AdoptIdea` is the only bridge from a proposal to retrievable
memory, and only the idea owner may invoke it.

Adoption maps the immutable idea to one explicit hypothesis experience:

| Hypothesis field | Idea source |
|---|---|
| New identity kind and origin | `hypothesis`, `adopted_idea`; reuse preserves the existing identity origin |
| Summary | idea title |
| Mechanism | copied exactly |
| Tags | canonical `inspiration` and `operator:<name>` |
| Applicability | assumptions |
| Evidence | stable keys from the frozen evidence references |
| Falsifiers | copied exactly |
| Body | canonical JSON of hypothesis, predictions, assumptions, and proposed test |

The command first computes the mapped content hash within the owner scope. A
matching nonarchived current experience is reused without a confidence change;
otherwise exactly one experience and version are created. A matching archived
experience returns `restore_required` so restoration remains an explicit
memory decision. An archived idea itself may still be explicitly adopted, a
rejected idea may not, and an already adopted idea returns its original
adoption result without duplication.

The same immediate transaction writes any new experience/version events
before `inspiration.idea_adopted`, inserts one immutable adoption record, links
owned experience evidence with version-scoped `derived_from` relations, updates
the idea and incubation projections, and completes the command receipt.
Quarantined capsule evidence remains provenance only; idea adoption never
silently adopts a capsule.

New writes use `inspiration.idea_adopted_v2` with schema version 2. It retains
the exact requested importance and confidence, allowing source validation and
later retries to prove the original request body. The original
`inspiration.idea_adopted` schema version 1 remains readable and replayable
without rewriting historical bytes. For a V1 event that created its result,
the original parameters are recoverable from the exact initial experience
state and remain hash-verifiable. A V1 event that reused an existing
experience did not retain those otherwise ignored parameters; validators can
still prove its receipt identity, result, response, state, and timing, but can
only require a structurally valid SHA-256 request hash. Such a legacy reused
adoption therefore returns its original result for any later parameter values.
This is an explicit compatibility proof boundary, not the contract for V2
writes.

One adoption causation is closed: a created result has exactly the ordered
`experience.created` and `experience.version_created` effects before the
adoption event, while a reused result has no experience event under that
causation. Extra access, evidence, lifecycle, or version effects invalidate
the source trace.

## Relationship to hot, warm, cold, and archived memory

Inspiration observes the memory lifecycle without bypassing it:

- hot and warm owned experiences may contribute bounded content through the
  read-only peek path;
- cold experiences participate in ranking, but only a cue strong enough for
  cold expansion contributes body bytes; peeking still does not reactivate
  them;
- an unexpanded cold item contributes metadata and an empty excerpt, so
  blurred content cannot leak into a snapshot;
- archived experiences are absent from evidence retrieval until an explicit
  restore command returns them to the active memory lifecycle; and
- opted-in pending capsules remain quarantined rather than entering the
  owner's normal experience index.

A newly created adopted hypothesis starts warm by default, or hot when
importance is at least `0.85`. It then follows the ordinary experience
lifecycle: access and evidence may keep it hot, decay may move it through warm
and cold, qualifying cue-based search can recall and reactivate cold content,
and an archived experience must be restored before it is mutable or
retrievable again. Reusing an equivalent experience preserves that
experience's existing temperature and confidence.

Idea archival and experience archival are deliberately different:

- a noncandidate, unadopted active idea is policy-archived after 180 days
  without a signal;
- an idea in a candidate mechanism cluster is policy-archived after 365 days
  from the later of its last signal and `candidate_since`;
- adopted and rejected ideas are not policy-archived;
- archiving an idea does not archive an experience; and
- archiving or cooling an experience does not erase the immutable idea,
  snapshot, occurrence, or adoption provenance.

This separation supplies the intended “fuzzy recall” behavior without losing
auditability. A cold memory needs a matching cue before its content can
re-enter working context; an archived experience needs explicit restoration;
an old archived idea can be rediscovered through mechanism recurrence or
owner review but requires explicit adoption to become memory. The immutable
snapshot and stable evidence keys explain which historical evidence produced
that branch even when the live memory temperatures have since changed.

## Verification and privacy

Run, snapshot, idea, occurrence, adoption, receipt, and event sources are
authoritative. Run, idea-state, and mechanism-incubation tables are rebuildable
projections. Source validation checks source/event correspondence, one logical
normal-run timestamp, per-operator idea/outcome ordering, continuous token and
elapsed counters, configured branch limits, snapshot ordering and hashes,
evidence resolution, idea/mechanism hashes, cluster transitions, terminal
trace legality, receipt linkage, and adoption provenance before projection
repair is allowed.

Owner isolation is checked before returning idea or run state. Missing and
foreign-owned resources share not-found behavior. Events and responses contain
stable IDs, hashes, bounded counters, maturity, and sanitized failure codes;
they do not contain provider credentials, raw exceptions, prompts, provider
responses, raw experience bodies, raw capsule bodies, or another owner's
private idea content.
