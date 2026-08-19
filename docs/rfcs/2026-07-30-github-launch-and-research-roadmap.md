# RFC: Evidence-First GitHub Launch and Research Roadmap

- Status: accepted for implementation planning
- Date: 2026-07-30
- Audience: agent-tool developers, open-source maintainers, and contributors
- Scope: public presentation, reproducible launch evidence, community entry
  points, and staged research validation

## Summary

Experience Hub will launch as an **experience engineering** developer tool:

> Record what an agent learned. Decide whether to trust it. Replay its effect
> before allowing it to shape future work.

The public story is “replay before adoption”: capture a candidate experience,
keep it quarantined, replay its effect in isolation, inspect the evidence, and
make an explicit adoption decision.

The selected launch strategy combines:

- executable proof as the primary repository experience;
- a concise experience-engineering category narrative;
- a later community challenge that is not required for launch.

This RFC extends the accepted
[`Experience Replay Lab`](./2026-07-22-experience-replay-lab.md) RFC. It does not
change its domain, storage, owner-isolation, quarantine, replay, or adoption
contracts.

## Goals

- Explain the product category and trust boundary within 30 seconds.
- Provide a five-minute, offline, deterministic demonstration.
- Generate public proof values and graphics from reviewable evidence.
- Maintain an English primary README and a complete Chinese translation.
- Give first-time contributors small, contract-shaped entry points.
- Support a credible launch without requiring external participants.
- Add stronger human and multi-project evidence only after launch.
- Establish testable research directions without presenting them as shipped.

## Non-goals

- A hosted service, web dashboard, or telemetry pipeline.
- Automatic publication, GitHub account changes, or license selection.
- A promise of stars, security, intelligence, truth, or universal improvement.
- Treating several accounts operated by one person as several participants.
- Presenting synthetic tasks as real-user outcomes.
- Replacing Replay Lab contracts or duplicating domain rules in documentation.

## Selected public experience

### Progressive README

The English README becomes a progressive proof path:

1. **First 30 seconds:** category, value statement, constraints, evidence, demo
   link, and Chinese-language link.
2. **30 to 120 seconds:** the operational problem and the
   `capture -> quarantine -> replay -> adopt` workflow.
3. **Two to five minutes:** a clean local run using a committed fixture.
4. **After proof:** links to integration, evaluation, research, limitations, and
   contribution material.

Large reference indexes remain available but move out of the primary reading
path.

The first screen uses the selected **Evidence Console** direction. It emphasizes
local-first operation, deterministic replay, SQLite authority, offline defaults,
and current verified evidence. Planned capabilities use a separate visual label.

### Five-minute reproduction

The quick start uses existing local commands:

```bash
uv sync
uv run experience-hub demo --reset
uv run experience-hub benchmark
```

Documented results must come from the same evidence source as the public proof
card. README output must not drift independently.

### Launch assets

The public package contains:

- an Evidence Console README hero;
- a short scripted replay demonstration with static and text fallbacks;
- an adoption and simulation trust-boundary diagram;
- a release evidence card;
- a repository social-preview image;
- a proof-first release-note template.

Assets remain repository-local and require no remote image service. Uploading the
social preview remains an explicit maintainer action.

## Evidence contract

Launch evidence follows one flow:

```text
required verification
    -> bounded canonical evidence
    -> deterministic renderers
    -> README and launch assets
    -> staleness validation
```

The canonical evidence document includes only public, stable fields:

- schema and renderer versions;
- tested commit identity and bounded platform information;
- required command identities and results;
- test and benchmark counts;
- replay and non-persistence invariant results;
- explicit release verification metadata.

It excludes machine paths, usernames, credentials, raw environment dumps,
wall-clock profiles, and unstable temporary identifiers.

Given identical evidence and renderer versions, supported text and SVG outputs
must be byte-identical. Validation fails closed when:

- the tested commit does not match the intended release;
- a required command result or invariant is missing;
- generated output differs from its checked-in form;
- the README references absent evidence;
- public files contain a local path or sensitive value.

Stale evidence blocks the release candidate instead of retaining an older claim.

## Public claim rules

- Current, next, and research capabilities use visibly different labels.
- Benchmarks identify their fixture, scope, commands, and material exclusions.
- Known limits appear beside the strongest relevant claims.
- Maintainer-only activity is not presented as user evidence.
- No remote badge, tracking pixel, or product telemetry is added.
- Metaphors remain presentation devices, not descriptions of mechanisms.
- Stars and repository traffic are reach signals, not technical validation.

## Contributor entry points

Contribution material defines at least four bounded paths:

- documentation correction;
- replay invariant or regression case;
- versioned Replay Pack;
- adapter proposal backed by a real integration.

Issue templates request sanitized reproduction material and prohibit credentials,
private prompts, proprietary traces, and personal data.

### Replay Pack

A Replay Pack is a small, versioned failure scenario containing canonical inputs,
provenance, expected invariants, and replay assertions. Initial examples use
synthetic, sanitized fixtures and must pass schema, owner-isolation,
determinism, and sensitive-content checks.

### Evidence Passport

An Evidence Passport is a planned canonical record of provenance, replay cases,
observed effects, limitations, tested versions, adoption state, and decision
lineage. It reuses Replay Lab records and must not create a second adoption state
machine.

### Open Replay Challenge

The later challenge invites versioned counterexamples that expose unsafe, stale,
conflicting, or non-generalizable experience. It opens only after Replay Pack
validation and moderation rules are stable. It is not a launch requirement.

## Growth and evidence maturity

The project uses native repository insights and voluntary feedback instead of
product telemetry. A small weekly snapshot may record:

- unique visitors, referring sources, and new stars;
- unique clones and voluntary demo-success reports;
- first-time contributors and time to first response;
- reproducibility failures and stale-evidence detections.

Small counts remain raw counts. Technical release decisions continue to use
validation gates.

Human evidence increases claim strength but does not determine whether the
project can exist:

| Level | Evidence | Participants | Permitted interpretation |
| --- | --- | ---: | --- |
| 0 | Tests, fixed fixtures, replay, failure injection | 0 | Engineering release |
| 1 | Maintainer pilot | 1 operator | Instruction and UI debugging |
| 2 | Independent first-run pilot | 3–5 | Directional usability findings |
| 3 | Counterbalanced comparison | 8–12 | Exploratory comparison |
| 4 | Reviewed study or varied corpus | Study-defined | Broader claim |

Several accounts operated by one person remain one operator.

## Research directions

All three directions are planned and hypothesis-led.

### Counterfactual Replay Lab

Run the same historical context with a candidate experience included and
excluded without modifying the live database.

Hypothesis: explicit counterfactual replay reduces harmful adoption decisions
compared with immediate trust.

Candidate measures include harmful-adoption rate, false rejection, task utility
delta, decision time, and replay mismatch rate.

### Evidence Passport

Present provenance, replay evidence, limitations, version, and adoption lineage
as one inspectable decision record.

Hypothesis: structured evidence improves adoption-decision accuracy without an
unacceptable increase in review time.

Candidate measures include oracle-aligned decisions, review time,
missing-limit detection, and confidence calibration.

### Experience Firewall

Express owner isolation, quarantine, conflict, staleness, and regression
requirements as versioned gates applied before adoption.

Hypothesis: versioned gates catch more unsafe changes than ad hoc review while
preserving deterministic reproduction.

Candidate measures include leak count, unsafe-change detection, false blocks,
incomplete-run handling, and gate reproducibility.

## Delivery and validation

### Phase 1: launch foundation, zero to four weeks

- restructure the English README and add the Chinese version;
- create the Evidence Console assets and five-minute demo;
- add canonical evidence generation and staleness checks;
- add contribution, security, issue, PR, and release material;
- specify Replay Packs and add two or three repository-owned examples.

Release gate: all project checks pass, a clean environment reproduces the demo,
all proof values trace to current evidence, generated assets match their sources,
and public artifacts pass sensitive-content and local-path audits.

Participant requirement: zero.

### Phase 2: pilot and Evidence Passport, month two

- prototype the Evidence Passport on existing replay records;
- observe three to five independent developers using the first-run path;
- record completion, time, errors, blockers, and trust-boundary comprehension;
- convert confirmed blockers into regression tests or documentation fixes;
- open bounded Replay Pack contribution issues.

Results remain directional and use only consented, anonymized aggregates.

### Phase 3: counterfactual study, months three to six

- implement include/exclude counterfactual replay;
- use counterbalanced tasks to reduce small-sample ordering bias;
- add stale, conflicting, adversarial, and negative-transfer Replay Packs;
- draft an interoperability RFC for Evidence Passports.

Public findings preserve incomplete and negative results and report uncertainty,
exclusions, and scope. Target evidence is eight to twelve independent developers
or a suitably varied historical corpus.

### Phase 4: benchmark and protocol, months six to twelve

- publish a versioned Experience Engineering Benchmark;
- stabilize Replay Pack and Evidence Passport contracts;
- add adapters only after two real integrations need the same boundary;
- run the Open Replay Challenge with transparent moderation;
- evaluate cross-project reproducibility and maintenance cost.

Broader claims require independent reproduction, varied project evidence, or an
appropriately reviewed human study.

## Release boundaries

This RFC does not authorize merging, pushing, creating a release, changing
repository visibility, or uploading the social preview.

A public open-source launch also requires an explicit maintainer license
decision. Implementation must not add or alter a license.

## Required verification

Before the launch package is described as complete:

```bash
uv lock --check
uv run ruff check .
uv run mypy src
uv run pytest --no-cov
uv run experience-hub demo --reset
uv run experience-hub benchmark
uv build
```

Focused tests must cover evidence validation, canonical serialization,
deterministic assets, stale evidence, fallbacks, public-content scanning, README
evidence references, and invalid Replay Packs.

## Acceptance criteria

- The English first screen defines the category, current value, constraints, and
  reproducible proof.
- The Chinese README covers the same current behavior and limitations.
- A clean developer can complete the primary offline path in five minutes.
- Public evidence has one canonical, reviewable source of truth.
- Identical evidence produces identical supported text and SVG assets.
- Stale or mismatched evidence blocks the release candidate.
- Planned research cannot be mistaken for shipped behavior.
- Contributor paths are bounded, safe, and test-shaped.
- All required project checks pass on the final implementation state.
- No license, push, release, or account-setting change occurs without explicit
  maintainer approval.
