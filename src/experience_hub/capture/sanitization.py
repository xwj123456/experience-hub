"""Pure defense-in-depth scanning without retained secret material."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from experience_hub import sha256_hex
from experience_hub.capture.models import (
    SensitiveField,
    SensitiveMatchV1,
    TrajectoryBundleV1,
)

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bearer_token",
        re.compile(
            r"(?:Authorization[ \t]*:[ \t]*)?Bearer[ \t]+"
            r"[A-Za-z0-9._~+/=-]{16,}",
            re.ASCII | re.IGNORECASE,
        ),
    ),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----",
            re.ASCII,
        ),
    ),
    (
        "aws_access_key",
        re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII),
    ),
    (
        "github_token",
        re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}", re.ASCII),
    ),
    (
        "openai_key",
        re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}", re.ASCII),
    ),
)
_HEADER_STEP_ID = "header"


class SecretScanner(Protocol):
    def scan(
        self,
        bundle: TrajectoryBundleV1,
    ) -> tuple[SensitiveMatchV1, ...]: ...


def _matching_rule_ids(values: Iterable[str]) -> tuple[str, ...]:
    retained = tuple(values)
    return tuple(
        rule_id
        for rule_id, pattern in _SECRET_RULES
        if any(pattern.search(value) is not None for value in retained)
    )


class DefaultSecretScanner:
    def scan(
        self,
        bundle: TrajectoryBundleV1,
    ) -> tuple[SensitiveMatchV1, ...]:
        matches: list[SensitiveMatchV1] = []
        header_fields = (
            (SensitiveField.TRAJECTORY_ID, bundle.trajectory_id),
            (
                SensitiveField.SANITIZATION_PROFILE_ID,
                bundle.sanitization.profile_id,
            ),
        )
        for field, value in header_fields:
            matches.extend(
                SensitiveMatchV1(
                    rule_id=rule_id,
                    step_id=_HEADER_STEP_ID,
                    field=field,
                )
                for rule_id in _matching_rule_ids((value,))
            )
        for step in bundle.steps:
            step_id_rule_ids = _matching_rule_ids((step.step_id,))
            retained_step_id = (
                f"sha256:{sha256_hex(step.step_id.encode('utf-8'))}"
                if step_id_rule_ids
                else step.step_id
            )
            matches.extend(
                SensitiveMatchV1(
                    rule_id=rule_id,
                    step_id=retained_step_id,
                    field=SensitiveField.STEP_ID,
                )
                for rule_id in step_id_rule_ids
            )
            raw_fields = (
                (SensitiveField.OBSERVATION, step.observation),
                (SensitiveField.ACTION, step.action),
                (SensitiveField.OUTCOME, step.outcome),
            )
            for field, value in raw_fields:
                matches.extend(
                    SensitiveMatchV1(
                        rule_id=rule_id,
                        step_id=retained_step_id,
                        field=field,
                    )
                    for rule_id in _matching_rule_ids((value,))
                )

            signal = step.candidate_signal
            if signal is None:
                continue
            signal_values = (
                signal.body,
                signal.summary,
                signal.mechanism,
                *signal.tags,
                *signal.applicability,
                *(pointer.step_id for pointer in signal.evidence),
                *signal.falsifiers,
            )
            matches.extend(
                SensitiveMatchV1(
                    rule_id=rule_id,
                    step_id=retained_step_id,
                    field=SensitiveField.CANDIDATE_SIGNAL,
                )
                for rule_id in _matching_rule_ids(signal_values)
            )
        return tuple(matches)
