# Security policy

## Supported versions

Security fixes currently target the latest release and the `main` branch. Version `0.x` is a local MVP and may introduce compatibility changes with release notes.

## Reporting a vulnerability

Please do not publish exploitable details in a public issue. Use GitHub's **Report a vulnerability** feature in the repository Security tab. Include the affected version, reproduction steps, impact, and any suggested mitigation.

The maintainer will aim to acknowledge a complete report within seven days. Timelines depend on severity and reproducibility; no response-time guarantee is implied.

## Current boundary

Experience Hub does not yet provide production authentication, authorization, tenancy, or public-network hardening. Bind the API to localhost or place it behind a separately reviewed security boundary. Do not store secrets in experience bodies, benchmark fixtures, logs, or issue reports.
