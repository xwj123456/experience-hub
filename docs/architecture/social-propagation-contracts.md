# Social Experience Propagation Contracts

This document fixes the public, integrity, quarantine, provenance, adoption,
and replay contracts delivered by the social experience propagation slice.
The public Python surface is enforced by
`tests/contract/test_sharing_exports.py`. Symbols that are not re-exported
from `experience_hub.sharing` are implementation details, even when their
leaf modules remain importable.

## Public package boundary

`experience_hub.sharing` owns the stable cross-feature values below.

| Group | Public contracts |
|---|---|
| Commands | `CreateTopic`, `CreateSubscription`, `PublishCapsule`, `AdoptCapsule`, `RetractCapsule`, `RejectInboxItem`, `RecordCapsuleFeedback` |
| Domain values | `Topic`, `Subscription`, `Capsule`, `ProvenanceHop`, `InboxItem`, `AdoptionResult`, `FeedbackRevision`, `Reputation`, and their enums |
| Owner-scoped queries | `SharingQuery`, `InboxPage` |
| Read-only pending evidence | `InboxEvidenceReader`, `QuarantinedCapsuleEvidence` |
| Limits | `MAX_PROVENANCE_HOPS`, `MAX_TOPIC_NAME_CHARACTERS`, `MAX_TOPIC_DESCRIPTION_CHARACTERS` |

These exports are the original objects from their leaf modules, not wrappers
or reconstructed aliases. Query exports are resolved lazily so package import
order does not create a circular dependency.

There are exactly two recipient-facing read paths for quarantined content.
`SharingQuery.list_inbox` returns the capsule through its recipient-owned
inbox route. `InboxEvidenceReader.list_available_pending` is an asynchronous,
session-bound, side-effect-free interface for an explicit inspiration opt-in;
its `excerpt` currently contains the complete capsule body, not a truncated
summary. The publisher's successful publication response also returns its own
capsule, but that command response is not a recipient quarantine read path.
The evidence reader does not open or commit a transaction, emit an event,
create a search row, adopt a capsule, or make pending content ordinarily
retrievable.

Sharing services, repositories, projectors, event payload implementations,
hashing helpers, source validators, physical JSON codecs, and ORM rows are
implementation details. Cross-feature consumers exchange public values and
use caller-supplied `UnitOfWork` or `AsyncSession` objects; they do not pass
sharing ORM rows across the package boundary.

## Authoritative graph and rebuildable projections

The propagation graph deliberately separates immutable evidence from replayed
state.

| Class | Tables | Rule |
|---|---|---|
| Authoritative source | `topics`, `subscriptions`, `experience_capsules`, `adoption_records`, `capsule_feedback` | Validate and retain; projection repair never replaces, edits, or deletes these rows |
| Ordered audit source | `domain_events` | Validate schemas, causation, aggregate sequence, and exact source-row correspondence, then replay in ascending `event_id` order |
| Rebuildable projection | `capsule_state`, `inbox_items`, `agent_reputation` | Recreate completely from validated sources and ordered events |

Topics and subscriptions are immutable identities. Capsule source rows retain
the exact canonical transport document. Adoption rows retain captured trust
and the complete path that reached the adopter. Feedback rows are immutable
revisions; a changed judgment is a new row, never an update.

`capsule_state` projects `active` or `retracted`. `inbox_items` projects
`pending`, `adopted`, or `rejected` while preserving the event-allocated
route identity. `agent_reputation` projects observer-relative effective
feedback counts. Expiry is not a state transition or a projection row:
effective expiry is the pure predicate `observed_at >= expires_at` over an
immutable timestamp and the injected clock.

## Canonical content and capsule transport

Publication first reconstructs and validates the selected immutable
experience version. The semantic body hash and content hash are:

```text
payload_hash = SHA-256(canonical_json({"body": body}))

source_content_hash = SHA-256(canonical_json({
    "kind": kind,
    "summary": summary,
    "mechanism": mechanism,
    "tags": canonical_tags,
    "applicability": canonical_applicability,
    "evidence": canonical_typed_evidence,
    "falsifiers": canonical_falsifiers,
    "payload_hash": payload_hash
}))
```

Canonical arrays use the strict experience value rules: values are validated,
deduplicated by canonical JSON, and sorted by their canonical bytes. The
decoded body and every transferable metadata field must reproduce the source
version's stored content hash exactly. Experience owner, origin, importance,
temperature, access history, rehearsal state, source trust, and lifecycle
timestamps are not semantic content and are not transported as though they
belonged to the adopter.

The transport capsule hash is separate from the semantic content hash:

```text
capsule_hash = SHA-256(canonical_json({
    "transport_schema_version": 1,
    "capsule_id": capsule_id,
    "topic_id": topic_id,
    "source_experience_id": source_experience_id,
    "source_version_id": source_version_id,
    "publisher_agent_id": publisher_agent_id,
    "kind": kind,
    "body": body,
    "summary": summary,
    "mechanism": mechanism,
    "tags": canonical_tags,
    "applicability": canonical_applicability,
    "evidence": canonical_typed_evidence,
    "falsifiers": canonical_falsifiers,
    "publisher_confidence": publisher_confidence,
    "provenance_chain": root_first_prior_hops,
    "root_fingerprint": root_fingerprint,
    "source_content_hash": source_content_hash,
    "created_at": created_at,
    "expires_at": expires_at,
    "hop_count": hop_count
}))
```

All UUIDs, enums, datetimes, numbers, objects, and arrays use the foundation
canonical JSON representation. The hash therefore covers identity, routing,
publisher, copied semantics, publisher confidence, path, root, and transport
lifetime. It does not cover its own `capsule_hash`, projected status, or a
projected transition checkpoint. Retraction consequently changes
availability without changing the immutable transport document.

## Root-first provenance and hop limits

An independent root is identified by both its original publisher and semantic
content:

```text
root_fingerprint = SHA-256(canonical_json({
    "root_publisher_id": root_publisher_id,
    "source_content_hash": source_content_hash
}))
```

Canonical UUID serialization makes the publisher identity lowercase and
stable. Distinct capsules or historical versions from that same root
publisher with identical semantic content therefore share one root
fingerprint.

A capsule's `provenance_chain` contains only prior hops, in root-first order:

```json
[
  {
    "capsule_id": "the-earliest-capsule-id",
    "publisher_agent_id": "the-root-publisher-id"
  },
  {
    "capsule_id": "the-next-capsule-id",
    "publisher_agent_id": "the-next-publisher-id"
  }
]
```

The following invariants are fixed:

- an original publication has an empty chain and `hop_count == 0`;
- `hop_count` always equals the length of the capsule's prior-hop chain;
- a capsule ID cannot repeat in its chain or name itself;
- the maximum capsule hop count is four;
- every referenced capsule exists, names the recorded publisher, preserves
  the same root fingerprint, and has the exact preceding chain prefix;
- every non-root publication is explained by an owned adoption of the final
  prior capsule into the selected source experience;
- republishing appends no implicit hop: it copies the named adoption record's
  full chain, which already includes the publisher's adopted capsule;
- the parent adoption must belong to the publisher and its
  `resulting_experience_id` must equal the selected source experience;
- publication cannot precede the parent adoption; and
- content originating from an adopted capsule requires an explicit parent
  adoption, while local or idea-adopted content may deliberately start a new
  root.

An adoption record stores `capsule.provenance_chain` followed by the current
`(capsule_id, publisher_agent_id)` hop. The capsule root fingerprint is copied
without alteration. Provenance is not translated into `experience_links`;
those links model semantic dependencies, not social transmission.

## Subscription timing, delivery, and quarantine

A subscription is non-backfilling. Its immutable `creation_event_id` is the
ledger boundary: a subscriber is eligible for a capsule only when the
`capsule.published` event ID is strictly greater than that subscription event
ID. The publisher is excluded from its own deliveries.

Publication emits `capsule.published`, then exactly one `capsule.received` for
each eligible recipient, ordered by recipient UUID. All events share one
causation receipt and the capsule creation time. Each receive event allocates
the stable inbox `item_id`; replay must recreate that exact identity and the
unique `(recipient_agent_id, capsule_id)` route. A missing eligible delivery,
an extra ineligible delivery, a duplicate delivery, or a delivery to the
publisher is source corruption rather than a condition for best-effort
repair.

Every received capsule enters `pending` quarantine. Pending content:

- does not appear in ordinary experience retrieval or direct experience GET;
- is visible as a full `Capsule` through the recipient's owner-scoped inbox;
- may expose its body as quarantined evidence only through explicit
  pending-evidence inspiration opt-in;
- must still be pending, active, and unexpired at the supplied read time;
- is labeled `quarantined` and receives fixed source trust `0.25` in that
  read-only evidence interface; and
- cannot mutate retrieval, experience, inbox, or capsule state merely by
  being inspected.

Retraction changes capsule state from `active` to `retracted`. Expiry is
effective at the inclusive boundary `observed_at >= expires_at`. Either
condition prevents later adoption and makes a pending route unavailable, but
neither deletes the capsule, inbox route, adoption history, provenance, or an
already adopted local copy.

## Explicit adoption and local ownership

Adoption is always a deliberate recipient-owned command. Before validating
the current pending state, the service returns an existing
`(adopter_agent_id, capsule_id)` adoption result, so a completed adoption
remains stable even under a new receipt. A new adoption requires:

- an inbox route owned by the caller and still projected `pending`;
- an active, unexpired capsule;
- a command time not behind capsule, inbox, adoption, or affected experience
  causal anchors;
- a valid semantic content hash and complete provenance path; and
- the observer-relative publisher trust captured at that instant.

With no equivalent current content under the adopter, adoption creates one
hot local experience and version with no dependency links. Its copied kind,
body, and transferable metadata reproduce `source_content_hash`; its initial
confidence is:

```text
publisher_confidence * captured_observer_trust
```

The immutable adoption row records the local experience, complete chain,
root, captured trust, time, and `corroboration_applied=false`. Later feedback
may change future trust, but it never rewrites that captured trust or the
adopted experience's fixed initial source trust.

Equivalent-content detection is scoped to the adopter. If a unique,
nonarchived current experience already has the semantic content hash, adoption
adds provenance without creating a duplicate or a new version. A previously
unrepresented independent root may corroborate it once:

```text
confidence_delta =
    (1 - current_confidence) * 0.20 * captured_observer_trust
```

The partial unique key on
`(resulting_experience_id, root_fingerprint)` for rows with
`corroboration_applied=true`, together with the immediate writer transaction,
ensures that concurrent attempts cannot apply the same root twice. Archived
equivalent content requires a separate explicit restore.

Adoption changes only its owned inbox route from `pending` to `adopted`.
Rejection changes it from `pending` to `rejected` and records a structured
reason in the event because no separate rejection source row exists. Both are
terminal for that route. Creation/adoption events, corroboration events,
optional cold-to-hot promotion, adoption sources, inbox projection, and
receipt completion commit atomically.

## Independent-root echo resistance

Confidence follows evidence diversity, not message frequency. At most one
confidence contribution is allowed for each
`(resulting_experience_id, root_fingerprint)`.

Republishing, forwarding through another agent, publishing another capsule
from the same root publisher and semantic content, or adopting the same root
through a different path still creates an immutable adoption record so the
observed route is auditable. It does not apply another confidence increase.
Only a semantically equivalent capsule with a different independently
computed root fingerprint may contribute again.

For a three-agent echo `A -> B -> C`, a capsule republished by B preserves A's
root and carries A's prior hop. If C already represented A's root, adopting
B's copy records the longer path but emits no corroboration. A genuinely
independent publication of the same semantic content from another root may
produce exactly one further confidence rise.

## Feedback revisions and observer-relative reputation

Feedback is authorized only after the observer has adopted or rejected their
own inbox route for the capsule. Pending recipients, unrelated agents, and
foreign-object direct lookups fail closed with the same not-found behavior.
Feedback evaluates the capsule's immediate publisher, not its original root
publisher.

Each `(observer_agent_id, capsule_id)` stream starts at revision one and is
contiguous. A revision source row retains the canonical structured reason and
typed evidence. Its `capsule.feedback_recorded` event names the immutable
revision and carries the prior/current verdict plus effective count
before/after values; raw reasons and evidence are not duplicated into that
event.

Reputation is relative to one subject and one observer. It starts with the
Bayesian prior `alpha=2`, `beta=2`:

```text
useful  -> one effective alpha contribution
refuted -> one effective beta contribution
harmful -> one effective beta contribution
trust   = alpha / (alpha + beta)
```

Only the latest revision for an observer/capsule contributes. Revising a
verdict first removes the prior effective contribution, then applies the new
one. Replay in event order must therefore produce the same useful, refuted,
harmful, alpha, beta, and trust values as incremental projection. A later
reputation change affects only future trust lookups; it never retroactively
changes an adoption record, local confidence, or source trust.

## Event ownership and atomic ordering

Aggregate ownership is fixed:

| Events | Aggregate |
|---|---|
| `topic.created` | topic ID |
| `subscription.created` | subscription ID |
| `capsule.published`, `capsule.retracted`, `capsule.feedback_recorded` | capsule ID |
| `capsule.received`, `capsule.adopted`, `capsule.rejected` | inbox item ID |

The event-allocated inbox item sequence starts with `capsule.received` and may
have exactly one terminal adoption or rejection transition. Publication and
delivery use one transaction and one receipt. Adoption may also emit
experience creation or corroboration events, but every source insert, ordered
event, affected projection, and receipt completion remains inside the
caller-owned `BEGIN IMMEDIATE` transaction.

Event payloads are strict registered V1 values. Unknown event names, schema
versions, fields, state transitions, aggregate owners, actors, or sequence
shapes fail closed. Capsule bodies are retained only in immutable capsule
sources and never enter domain events, projection mismatch keys, or command
responses that are specified as metadata-only.

## Complete source validation

Sharing source validation proves the graph before any projection rebuild or
repair. In addition to foundation ledger, receipt, schema, foreign-key, and
aggregate-sequence checks, it validates:

1. a closed sharing aggregate namespace: every event on a `topic`,
   `subscription`, `capsule`, or `inbox_item` aggregate is a sharing event,
   and every sharing event uses its one declared aggregate type;
2. a bijection between topic/subscription rows and their exact creation
   events, including the subscription's stored creation event ID;
3. a bijection between capsule rows and `capsule.published` events;
4. source experience/version ownership, decoded semantic equality,
   `source_content_hash`, and the complete recomputed capsule transport hash;
5. hop count, root-first prefix continuity, publisher identity, parent
   adoption, time ordering, and root preservation at every provenance hop;
6. exact delivery correspondence with subscription eligibility at the
   publication ledger boundary, including publisher exclusion and stable
   inbox identities;
7. adoption row/event correspondence, owned result experience, copied
   semantic hash, complete current-hop chain, captured trust bounds, and
   first-independent-root contribution semantics;
8. feedback row/event correspondence, contiguous observer/capsule revisions,
   terminal-route authorization, prior verdict continuity, and exact
   effective reputation count transitions;
9. nondecreasing clocks in each sharing aggregate, across inbox terminal and
   prior capsule transitions, and across each publisher/observer reputation
   stream;
10. exact binding of each command event to a completed receipt with the
   expected operation scope, caller, result resource type/ID, causal event
   group, and enclosing command clock; and
11. reverse correspondence, so an unexplained source row and an event naming
   no source row are equally invalid.

Mismatch keys are stable, bounded identifiers such as a topic, subscription,
capsule, delivery, adoption, feedback revision, or projection row ID. They
never include transported bodies, reasons, typed evidence payloads, SQL,
stack traces, or other unbounded diagnostic data.

## Verify, replay, and atomic repair

The three sharing reducers rebuild capsule state, inbox state, and reputation
from validated source anchors plus ordered events. Incremental and rebuilt
rows must serialize identically, including semantic IDs, enum values,
effective counts, reducer versions, and projection event checkpoints.

Maintenance is fail-closed:

1. source validation runs before temporary rebuild tables are created or
   online rows are considered replaceable;
2. `--verify` uses one consistent read and event head, rebuilds into
   connection-local temporary tables, reports stable row mismatches, drops
   temporary tables, and never changes online projections or reducer
   metadata;
3. `--repair` acquires the exclusive writer lock, refuses retained
   in-progress receipts, validates the complete source graph, captures and
   rechecks the event head, rebuilds every registered reducer, and swaps only
   declared projections and their version/checkpoint metadata;
4. post-swap canonical hashes must equal the rebuilt hashes; and
5. any validation, replay, head, swap, or post-swap failure rolls the whole
   operation back, so reducers are never partially repaired.

A corrupt capsule semantic hash, transport hash, provenance chain, delivery,
adoption, feedback revision, or row/event correspondence raises
`source_integrity_error`. Repair is not permitted to infer or heal the
authoritative truth, and all online projections remain byte-for-byte
untouched. A valid source graph with only projection divergence produces
`projection_mismatch` during verification and is eligible for exact repair.

## Stable failure semantics

The following distinctions are part of the contract:

- foreign-owned and missing sharing resources are indistinguishable to the
  caller;
- a pending route is not authority to send feedback;
- retracted or expired capsules cannot be newly adopted;
- the expiry boundary is inclusive and creates no event;
- a nonpending route cannot perform another terminal transition;
- an adopted-capsule experience cannot be republished without its exact
  owned parent adoption;
- a publication cannot exceed four prior hops;
- a command clock behind a relevant causal anchor fails with
  `clock_regression`;
- an overlapping root records provenance but does not inflate confidence;
- source corruption is never downgraded into a repairable projection
  mismatch; and
- idempotent replay returns the stored prior result without repeating
  delivery, adoption, corroboration, or feedback effects.

Stable domain errors expose only their code, bounded message, details, and
status. Unexpected exceptions, decoded capsule bodies, reasons, evidence,
database paths, and internal implementation data never enter serialized
errors.
