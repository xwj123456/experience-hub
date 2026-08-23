# ExperienceBench-S pilot

ExperienceBench-S is a small, reproducible comparison of four retrieval policies
under the same frozen source data, owner boundary, query, result limit, and content
budget. It measures whether useful evidence is selected while stale, misleading,
forbidden, foreign-owned, pending, and archived material stays out of ordinary
results.

## Reproduce the pilot

Inspecting validates canonical input, hashes, composition, and the required SQLite
FTS5 capability without creating a workspace:

```bash
uv run experience-hub replay benchmark inspect \
  --pack examples/experience-bench-s/pilot-manifest.json
```

Run both deterministic passes in a new owned workspace:

```bash
uv run experience-hub replay benchmark run \
  --pack examples/experience-bench-s/pilot-manifest.json \
  --workspace .data/experiencebench-s-pilot
```

Verify the published canonical evidence without rerunning a policy:

```bash
uv run experience-hub replay benchmark verify \
  --report .data/experiencebench-s-pilot/artifacts/benchmark-evidence.json
```

The run compares `no_memory`, `recent_notes`, `sqlite_bm25`, and
`experience_hub`. The pack contains 30 cases: six in each of five strata, ten in
each language group (Chinese, English, and mixed), and ten reviewed workflow
abstractions alongside twenty public-authored scenarios.

## Scoring and gate

Each case has a 1,000,000-micro utility rubric: 450,000 for required evidence,
300,000 for avoiding forbidden, stale, or misleading evidence, 150,000 for the
declared checkpoint, and 100,000 for evidence efficiency. Recovery checkpoints
use an ordered subsequence; all others use a required set. Integer arithmetic is
used throughout.

The fixed pilot gate requires complete, safe, byte-identical replay; a mean paired
gain of at least 50,000 micros over the strongest baseline; and a mean paired gain
of at least -20,000 micros in every stratum. A complete and valid run can still
fail this effectiveness gate. In that case the command exits nonzero while
retaining the negative evidence and summary for inspection.

## Boundary and limitations

The benchmark is local and offline. It does not read model credentials, invoke a
network provider, mutate an online database, or treat a missing FTS5 capability as
a favorable partial result. Logical labels are converted to deterministic local
identities only inside the generated fixture database.

This 30-case pilot is an expansion decision aid, not a general effectiveness
claim, a security proof, or a model of human memory or consciousness. It has no
confidence interval and does not represent production workloads. The frozen input
must not be adjusted after a measured run to improve its outcome.
