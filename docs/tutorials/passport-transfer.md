# Transfer one experience between local databases

This tutorial uses synthetic data, two SQLite files, and local Passport commands.
It calls no HTTP API, external service, model, or tool and reads no model key.
The small Python setup uses the same application services as the API because
there is no agent-creation CLI command. Passport v1 itself has no HTTP endpoints.

Run from the repository root after `uv sync --all-groups --frozen`, using Bash
or Zsh. Use a fresh tutorial directory; the commands do not remove existing
data. All IDs below come from actual command responses, not fixed UUIDs.
Installed users can use their environment's `python` and `experience-hub`
instead of `uv run`; neither the runtime nor its migrations needs a checkout.

## 1. Create source and receiving owners

The setup initializes the databases and creates one local source experience.
Its synthetic note is intentionally a reference without retained source text.
The runtime closes before export, so no source writer remains active.

```bash
umask 077
mkdir -p .data/passport-transfer outputs/passport-transfer
uv run python - <<'PY' > outputs/passport-transfer/seed.json
import asyncio
import json
from uuid import UUID

from sqlalchemy.engine import URL

from experience_hub import canonical_json_bytes
from experience_hub.agents import CreateAgent
from experience_hub.config import Settings
from experience_hub.domain import CommandRequest, TypedEvidence
from experience_hub.experiences import (
    CreateExperience, ExperienceKind, VersionContent,
)
from experience_hub.runtime import ApplicationRuntime


async def seed(database, name, *, source=False):
    settings = Settings(database_url=URL.create(
        "sqlite+aiosqlite", database=database,
    ))
    async with ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False, recover_interrupted=False,
    ) as container:
        async def create_owner(uow, context):
            return await container.agent_service.create(
                uow=uow, command=CreateAgent(name=name),
                command_context=context,
            )

        result = await container.command_executor.execute(CommandRequest(
            caller_scope="system:local", operation_scope="agent.create",
            idempotency_key="tutorial-owner", method="POST",
            route_template="/v1/agents", body={"name": name},
        ), create_owner)
        assert result.status_code == 201, result.body.decode()
        owner = json.loads(result.body)["data"]
        if not source:
            return owner

        owner_id = UUID(owner["agent_id"])
        content = VersionContent(
            body="Resume a failed local batch from its committed checkpoint.",
            summary="Checkpoint recovery",
            mechanism="A checkpoint avoids repeating committed batch work.",
            tags=("checkpoint", "recovery"),
            applicability=("local batches",),
            evidence=(TypedEvidence(type="note", id="synthetic-note"),),
            falsifiers=("The checkpoint was not committed.",),
        )

        async def create_experience(uow, context):
            return await container.experience_service.create(
                uow=uow, command=CreateExperience(
                    owner_agent_id=owner_id, kind=ExperienceKind.PROCEDURAL,
                    content=content, importance=0.7, confidence=0.6,
                ), command_context=context,
            )

        result = await container.command_executor.execute(CommandRequest(
            caller_scope=f"agent:{owner_id}",
            operation_scope="experience.create",
            idempotency_key="tutorial-experience", method="POST",
            route_template="/v1/agents/{agent_id}/experiences",
            path_parameters={"agent_id": owner_id},
            body={"kind": ExperienceKind.PROCEDURAL,
                  **content.model_dump(mode="python"),
                  "importance": 0.7, "confidence": 0.6, "links": ()},
        ), create_experience)
        assert result.status_code == 201, result.body.decode()
        return {**owner, **json.loads(result.body)["data"]}


async def main():
    source = await seed(
        ".data/passport-transfer/source.db", "Source owner", source=True,
    )
    target = await seed(".data/passport-transfer/target.db", "Receiving owner")
    print(canonical_json_bytes({"source": source, "target": target}).decode())


asyncio.run(main())
PY
```

Use this helper to read a field from a saved response:

```bash
json_field() {
  uv run python -c '
import json, sys
from functools import reduce
value = reduce(lambda obj, key: obj[key], sys.argv[1].split("."),
               json.load(sys.stdin))
print(value)
' "$2" < "$1"
}
SOURCE_OWNER=$(json_field outputs/passport-transfer/seed.json source.agent_id)
SOURCE_EXPERIENCE=$(json_field outputs/passport-transfer/seed.json source.experience_id)
SOURCE_VERSION=$(json_field outputs/passport-transfer/seed.json source.version_id)
TARGET_OWNER=$(json_field outputs/passport-transfer/seed.json target.agent_id)
```

## 2. Export and inspect

For real content, review every retained field and establish permission to share
the body and evidence before making the declarations below. A sanitization
profile is your own bounded identifier, not a built-in certification. Sensitive
pattern scanning is an additional check, not a complete privacy audit. Content
does not automatically inherit this project's source-code license.

```bash
uv run experience-hub passport export "$SOURCE_OWNER" "$SOURCE_EXPERIENCE" \
  --version "$SOURCE_VERSION" \
  --database .data/passport-transfer/source.db \
  --output outputs/passport-transfer/checkpoint.passport.json \
  --input-sanitized --sanitization-profile synthetic-v1 --sharing-authorized
uv run experience-hub passport inspect \
  outputs/passport-transfer/checkpoint.passport.json
uv run experience-hub passport inspect \
  examples/passports/capture-recovery.passport.json
```

The tutorial export reports one `reference_only` item: the file carries the note
reference, not its text. The committed capture example demonstrates an
`embedded_excerpt`: a bounded retained snippet with exact evidence bindings.
Neither form includes the complete original source preimage. Read all three
coverage counts, including `unavailable_preimage_count`.

Successful inspection reports `publisher_identity: "unverified"`,
`semantic_assessment: "not_assessed"`, and `persisted: false`. It checks file
integrity, not publisher identity or the truth of the conclusion. It opens no
database and does not echo the body or snippet.

Export is read-only: it requires an existing database at the supported schema
head and refuses nonempty WAL, shared-memory, or journal sidecars. If
`readonly_database_invalid` occurs, stop writers and checkpoint through the
normal owning process; do not delete sidecars. Export does not migrate or
checkpoint for you. A byte-identical existing output is accepted; a different
file at that output path is never overwritten.

## 3. Import into quarantine

```bash
uv run experience-hub passport import "$TARGET_OWNER" \
  outputs/passport-transfer/checkpoint.passport.json \
  --database .data/passport-transfer/target.db \
  --idempotency-key tutorial-import \
  > outputs/passport-transfer/import.json
IMPORT_ID=$(json_field outputs/passport-transfer/import.json data.import_id)
uv run experience-hub passport list "$TARGET_OWNER" \
  --database .data/passport-transfer/target.db --state pending --limit 10
uv run experience-hub passport show "$TARGET_OWNER" "$IMPORT_ID" \
  --database .data/passport-transfer/target.db
```

The state is `pending`. `show` intentionally exposes the full quarantined
document to its owner. It is not yet ordinary experience memory: retrieval,
search terms, sharing, and inspiration cannot consume it. Source IDs identify
unauthenticated provenance only; they do not select the receiving owner.

## 4. Choose adoption or rejection

To adopt, explicitly choose local importance and confidence in `[0, 1]`:

```bash
uv run experience-hub passport adopt "$TARGET_OWNER" "$IMPORT_ID" \
  --database .data/passport-transfer/target.db \
  --importance 0.7 --confidence 0.6 --idempotency-key tutorial-adopt \
  > outputs/passport-transfer/adopt.json
TARGET_EXPERIENCE=$(json_field outputs/passport-transfer/adopt.json \
  data.experience.experience_id)
ADOPTION_ID=$(json_field outputs/passport-transfer/adopt.json data.adoption_id)
uv run experience-hub passport show "$TARGET_OWNER" "$IMPORT_ID" \
  --database .data/passport-transfer/target.db
```

For this fresh target, `created` is `true` and the response contains new local
experience/version IDs, the unchanged content hash, and temperature `warm`.
New Passport adoption fixes source trust at `0.25`. Your confidence is a local
decision, not copied publisher confidence or an automatic truth assessment.

If a unique owned, unarchived current experience already has the same content,
`created` is `false`: adoption adds lineage without changing its version,
confidence, trust, temperature, or links. It never corroborates a source merely
because the same content was transferred again. Origin `adopted_passport` cannot
be published as a capsule.

Alternatively, instead of adopting, reject the still-pending item:

```bash
uv run experience-hub passport reject "$TARGET_OWNER" "$IMPORT_ID" \
  --database .data/passport-transfer/target.db \
  --reason "Insufficient source evidence for this use" \
  --idempotency-key tutorial-reject
```

Rejection records the reason and creates no experience. Skip the remaining
adoption-only steps if you chose rejection.

## 5. Retrieve adopted content and optionally forward it

Ordinary retrieval goes through the public application adapter; this is a
durable access operation, unlike Passport inspection or export:

```bash
uv run python - "$TARGET_OWNER" "$TARGET_EXPERIENCE" <<'PY'
import asyncio
import sys
from uuid import UUID

from sqlalchemy.engine import URL

from experience_hub.config import Settings
from experience_hub.runtime import ApplicationRuntime


async def main():
    settings = Settings(database_url=URL.create(
        "sqlite+aiosqlite", database=".data/passport-transfer/target.db",
    ))
    async with ApplicationRuntime(settings).initialize(
        start_lifecycle_worker=False, recover_interrupted=False,
    ) as container:
        result = await container.retrieval_adapter.get(
            owner_agent_id=UUID(sys.argv[1]), experience_id=UUID(sys.argv[2]),
            idempotency_key="tutorial-retrieve",
        )
        assert result.status_code == 200, result.body.decode()
        print(result.body.decode())


asyncio.run(main())
PY
uv run experience-hub passport export "$TARGET_OWNER" "$TARGET_EXPERIENCE" \
  --database .data/passport-transfer/target.db \
  --output outputs/passport-transfer/forwarded.passport.json \
  --parent-adoption "$ADOPTION_ID" \
  --input-sanitized --sanitization-profile synthetic-v1 --sharing-authorized
```

The forwarding command preserves the retained evidence and appends a source
hop. `--parent-adoption` must name this owner's adoption of the selected content;
it is required for origin `adopted_passport`. Modified-content derivation is not
supported. A chain may contain at most four source hops, not four extra forwards.
It describes Passport transfers only, not a complete history of other sharing
or idea/candidate origins.

## Retries and input limits

Retry a write with the same key and exact request to receive the stored response
byte-for-byte. Importing the same Passport again with a new key returns the
existing owner-scoped item, even after adoption or rejection. It does not reopen
quarantine. Reusing a key for changed parameters conflicts; using a new key to
adopt or reject an already decided item yields `passport_decision_conflict`.

The file must be exact canonical JSON and no larger than 512 KiB. Unknown
fields, duplicate keys, noncanonical formatting, invalid hashes, and incomplete
or extra evidence snapshots are rejected. Do not hand-edit or pretty-print a
Passport and expect it to remain valid. See the
[public contract](../architecture/passport-contracts.md) for the strict schema,
excerpt limits, owner isolation, and file-access boundary.
