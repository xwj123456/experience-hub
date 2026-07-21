# Contributing to Experience Hub

Thank you for helping improve reliable, auditable agent memory.

## Before opening a change

- Search existing issues and discussions first.
- For behavioral or architectural changes, open an issue describing the use case, invariants, and failure modes before investing in a large patch.
- Keep pull requests focused. Unrelated refactoring should be proposed separately.
- Never include credentials, private data, model transcripts, or machine-specific paths.

## Development setup

```bash
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```

New behavior should include deterministic tests. Prefer injected clocks, deterministic IDs, temporary SQLite databases, fake providers, and explicit failure assertions. Network-dependent tests are not accepted in the default suite.

## Commit sign-off

This project uses the [Developer Certificate of Origin](DCO). Sign every commit with:

```bash
git commit -s -m "Brief description"
```

The sign-off certifies the contribution under the terms in [DCO](DCO); it is not a copyright assignment.

## Pull requests

A good pull request explains:

- the problem and why it matters;
- the behavioral and compatibility impact;
- the authoritative data or event-contract impact;
- tests and commands used for verification;
- documentation or migration changes.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
