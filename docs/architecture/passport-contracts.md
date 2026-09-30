# Evidence Passport v1 contracts

An Evidence Passport transfers one immutable experience version between local
databases. It is an unsigned UTF-8 JSON file, not a database backup, an executable
attachment, or a bundle of experience dependencies. The transport schema version
is `1`; it is independent of the package release version.

Passport v1 exposes local CLI commands and Python services. It has no HTTP
Passport API, automatic adoption, network propagation, model assessment, or
cryptographic publisher authentication. No model key or external tool is needed.
See the [transfer tutorial](../tutorials/passport-transfer.md) for a runnable
two-database example.

## File and integrity boundary

The document contains `schema_version`, `format`, `subject`,
`evidence_snapshots`, `provenance`, `declaration`, and `passport_hash`.
`format` is exactly `experience_passport`. The subject retains source agent,
experience, and version UUIDs, source origin, kind, semantic content, and its
existing content hash. These source IDs never become the receiving owner's
local experience IDs.

| Boundary | v1 rule |
|---|---|
| Encoding | Exact canonical UTF-8 JSON, without a BOM or trailing newline |
| Schema | Integer `1`, strict types, no unknown fields or duplicate JSON keys |
| File size | At most 512 KiB; over-limit input is rejected, never truncated |
| JSON container depth | At most 16 |
| Evidence | At most 32 references, with exactly one snapshot per reference |
| Embedded excerpt | At most 512 UTF-8 bytes per item |
| Sanitization profile | Nonblank identifier, at most 100 UTF-8 bytes |
| Provenance | One to four source hops, root first, without repeated identities |

Invalid Unicode, non-finite numbers, and booleans in place of the schema integer
are rejected. Existing experience-content limits also apply. Evidence snapshots
use canonical reference order; provenance retains transfer order.

The semantic content hash uses the existing experience encoder. The separate
Passport SHA-256 covers the canonical document except its own `passport_hash`
field, including evidence, declarations, and provenance. Export adds no clock,
random export ID, database path, or host path. Selecting the same immutable
version, evidence, declaration, and lineage produces the same bytes.

Hashes detect inconsistent retained data. Anyone can edit an unsigned document
and recompute its hashes; they do not authenticate a publisher.

## Evidence coverage and honest inspection

| Snapshot | What the file retains | What inspection can establish |
|---|---|---|
| `reference_only` | The original typed evidence reference | Reference coverage, not the referenced source's contents |
| `embedded_excerpt` | Reference, bounded excerpt, excerpt hash, declared source and manifest hashes, step, and trajectory field | Excerpt integrity and exact reference/step/field/hash binding |

Automatic excerpt export is limited to retained capture evidence accessible to
the selected owner and version. An unresolved or unavailable reference remains
`reference_only`; no URL, arbitrary file, or external service is opened to
resolve it. A located owned source with corrupt or conflicting evidence causes
export to fail, rather than silently downgrading it to a reference.

Neither snapshot includes the full original trajectory field or manifest
preimage. Declared source and manifest hashes are not proof that the recipient
has verified those original sources. Inspection reports:

- `embedded_excerpt_count` and `reference_only_count` for coverage;
- `unavailable_preimage_count` for evidence whose original source preimages are
  absent, including embedded excerpts;
- `publisher_identity: "unverified"`;
- `semantic_assessment: "not_assessed"`; and
- `persisted: false`.

`inspect` validates only the bounded file. It does not initialize the runtime,
open SQLite, persist data, or print the body or excerpt. It does not determine
whether the experience is true or whether its evidence supports the conclusion.

## Explicit declarations and file access

Export requires `--input-sanitized`, `--sanitization-profile PROFILE`, and
`--sharing-authorized`. These record the caller's explicit declarations, not a
privacy audit or rights certification. Retained strings are scanned with the
shared sensitive-text rules; a match fails closed without echoing the secret.
A clean scan does not establish that all sensitive information was removed.

Experience contents do not automatically inherit the source code's Apache 2.0
license. The sender must establish permission to share the contents and evidence;
the recipient must establish permitted use.

File inputs and outputs reject symlinks, special files, and unsafe directory
chains. Output uses a private temporary file and no-clobber atomic publication:
an existing byte-identical target succeeds; different existing bytes produce
`passport_output_conflict` and are retained. These protections do not promise
to defeat an arbitrary malicious process running as the same OS user.

## Read-only export

`passport export` requires an existing file-backed SQLite database at the
installed schema head. It does not create or migrate the database, checkpoint
it, alter journal mode, run lifecycle work or interrupted-run recovery, record
an export event, or update access history.

The source must be quiescent and fully checkpointed. Nonempty `-wal`, `-shm`,
or `-journal` sidecars are refused; changing files or unsafe paths are refused
with `readonly_database_invalid`. Stop writers and checkpoint through the
database's normal owning process before retrying. Do not delete sidecars to
force an export. Upgrade older databases through normal runtime initialization
before a later read-only export.

`--version VERSION` selects an owned immutable historical version; omitting it
selects current. Archived experiences require an explicit restore before export,
including when a historical version is selected.

## Quarantine, decisions, and retries

Import validates the file before entering a short write transaction. The target
owner must already exist and is supplied explicitly; the source identity in the
file is only unauthenticated provenance, not a local caller or owner.

Deduplication is scoped to `(owner_agent_id, passport_hash)`. Reimporting the same
file for that owner returns the existing item and preserves its current state;
it reveals nothing about another owner's imports.

The decision state machine is `pending -> adopted` or `pending -> rejected`.
Pending Passports are accessible through owner-scoped `list` and `show` only.
They do not enter ordinary experience retrieval, search terms, sharing, or
inspiration snapshots. The pending-capsule inspiration opt-in does not include
Passports. Rejection records a structured reason and creates no experience.

Adoption requires explicit finite importance and confidence scores in `[0, 1]`.
For new content, adoption creates new local IDs, origin `adopted_passport`,
temperature `warm`, and fixed source trust `0.25`, while preserving the semantic
content hash. Importance and confidence are the receiver's local inputs, not
copied publisher scores or a truth assessment.

If exactly one owned, non-archived current experience has equivalent content,
adoption reuses it and adds immutable Passport lineage. It does not change its
version, confidence, trust, temperature, or links. Ambiguous equivalent content
fails closed; archived equivalent content requires restore. Adoption copies no
semantic dependency links and performs no independent-source corroboration.

Import, adopt, and reject require an idempotency key. The request hash binds the
owner, operation, and exact parameters, not filesystem paths. The same key and
request replay the stored response byte-for-byte. Reusing that key for another
request conflicts. A new decision key against an adopted or rejected item yields
`passport_decision_conflict`; it cannot reverse or repeat the decision.

Imports, immutable adoption lineage, ordered events, decision projections, and
receipts commit atomically. `projections rebuild --verify` and `--repair` cover
Passport state; repair does not rewrite imported bytes or adoption lineage.

## Provenance and existing sharing

`provenance.scope` is `passport_transfers_only`: the chain is not a complete
capsule, idea, or candidate history. Each hop retains source agent/experience/
version identity, source origin, content hash, and a parent Passport hash. The
first parent is null; subsequent parents are SHA-256 values. All hops preserve
the subject content hash, and the final hop matches the subject exactly.

The origin fingerprint binds the first source agent and semantic content hash.
It is not authenticated identity or evidence of independent corroboration.
Historical parents are not embedded recursively, so their absent preimages
cannot be independently authenticated from this file.

Re-exporting an `adopted_passport` experience requires an explicit
`--parent-adoption ADOPTION_ID` owned by the exporter. The selected content must
still match that adoption; modified-content derivation is unsupported in v1.
Re-export retains the imported snapshots and appends the local source hop.
The fifth hop or a repeated source identity is refused.

Capsule publication from origin `adopted_passport` is blocked with
`passport_publication_unsupported`. A genuinely local experience reused during
equivalent adoption keeps its local origin and existing publication capability.

## Public interfaces and CLI

The `experience_hub.passports` package exports strict transport values,
`VerifiedPassportV1`, pure build/encode/hash/verify functions, `PassportService`,
`PassportQuery`, and `PassportExportService`. Services use a caller-supplied
`UnitOfWork` or `AsyncSession`; filesystem adapters and ORM rows are not
cross-feature contracts.

The local commands are:

```text
experience-hub passport export OWNER EXPERIENCE --database DB --output FILE
  --input-sanitized --sanitization-profile PROFILE --sharing-authorized
  [--version VERSION] [--parent-adoption ADOPTION_ID]
experience-hub passport inspect FILE
experience-hub passport import OWNER FILE --database DB --idempotency-key KEY
experience-hub passport list OWNER --database DB
  [--state pending|adopted|rejected] [--limit N] [--cursor CURSOR]
experience-hub passport show OWNER IMPORT_ID --database DB
experience-hub passport adopt OWNER IMPORT_ID --database DB
  --importance SCORE --confidence SCORE --idempotency-key KEY
experience-hub passport reject OWNER IMPORT_ID --database DB
  --reason TEXT --idempotency-key KEY
```

All database commands require `--database`; `inspect` needs none. The list page
limit is 1–100, with an opaque owner/state-bound cursor. Missing and foreign
import IDs return the same `passport_not_found` response. Commands emit one line
of canonical JSON; errors do not echo input text, SQL, or host paths.
