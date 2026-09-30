# Changelog

All notable changes are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning while public compatibility stabilizes.

## [Unreleased]

### Added

- Evidence Passport v1 for unsigned, offline transfer of one immutable
  experience version, with canonical JSON, bounded evidence snapshots, and
  explicit verification limitations.
- Local Passport export, inspect, import, list, show, adopt, and reject commands;
  owner-scoped quarantine, idempotent decisions, immutable adoption lineage,
  and rebuildable decision projections.
- Read-only export from existing, checkpointed SQLite files; strict file
  validation and no-clobber output publication.
- A synthetic capture Passport example, transfer tutorial, and public contracts.

### Changed

- Runtime database defaults are relative to the current working directory;
  installed migrations work without a source checkout. Passport database
  commands require explicit database paths.

## [0.1.0] - 2026-07-21

### Added

- Auditable experience identities, immutable versions, event causation, idempotent commands, and rebuildable projections.
- Hot, warm, cold, and archived lifecycle with cue-driven cold-memory reactivation.
- Focused and associative multilingual retrieval with bounded content expansion.
- Quarantined cross-agent sharing, provenance chains, trust feedback, and explicit adoption.
- Frozen-evidence inspiration runs with causal-gap, counterfactual, and distant-analogy operators.
- Idea deduplication, incubation, evaluation, archival, and explicit hypothesis adoption.
- FastAPI, Typer CLI, deterministic demo, offline benchmark, migration, recovery, and maintenance tools.

[0.1.0]: https://github.com/xwj123456/experience-hub/releases/tag/v0.1.0
