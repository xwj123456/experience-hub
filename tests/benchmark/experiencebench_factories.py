from __future__ import annotations


def valid_manifest_document(
    *,
    cases_sha256: str,
    source_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_id": "experiencebench-s-pilot",
        "maturity": "pilot-30",
        "cases": {"file": "pilot-cases.jsonl", "sha256": cases_sha256},
        "source": {"file": "pilot-source.jsonl", "sha256": source_sha256},
        "frozen_at": "2026-08-20T00:00:00Z",
        "seed": 20260820,
        "arms": [
            {"arm_id": value, "kind": value, "required": True, "schema_version": 1}
            for value in (
                "no_memory",
                "recent_notes",
                "sqlite_bm25",
                "experience_hub",
            )
        ],
        "oracle_version": 1,
        "metric_version": 1,
        "gate_version": 1,
        "evidence_schema_version": 1,
        "summary_schema_version": 1,
        "profile_schema_version": 1,
        "deterministic_replay_runs": 2,
        "composition": {
            "case_count": 30,
            "public_authored": 20,
            "reviewed_abstractions": 10,
            "cases_per_stratum": 6,
            "chinese": 10,
            "english": 10,
            "mixed": 10,
        },
    }


def valid_case_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "queue-recovery",
        "source_class": "public_authored",
        "review_status": "authored",
        "stratum": "failure_recovery",
        "language": "en",
        "difficulty": "B",
        "owner_label": "queue-owner",
        "query": "recover the bounded queue",
        "mode": "focused",
        "tags": ["queue"],
        "mechanism_cues": ["recovery"],
        "limit": 5,
        "content_budget_bytes": 4096,
        "source_labels": [
            "queue-required",
            "queue-optional",
            "queue-forbidden",
            "queue-stale",
            "queue-misleading",
        ],
        "required": [{"label": "queue-required", "weight_micros": 450000}],
        "optional": [{"label": "queue-optional", "weight_micros": 0}],
        "forbidden": [{"label": "queue-forbidden", "weight_micros": 100000}],
        "stale": [{"label": "queue-stale", "weight_micros": 100000}],
        "misleading": [
            {"label": "queue-misleading", "weight_micros": 100000}
        ],
        "checkpoints": [
            {
                "predicate": "ordered_subsequence",
                "labels": ["queue-required"],
                "weight_micros": 150000,
            }
        ],
        "oracle_version": 1,
    }


def valid_source_agent_document() -> dict[str, object]:
    return {"schema_version": 1, "record_type": "agent", "label": "queue-owner"}


def valid_source_experience_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "experience",
        "label": "queue-required",
        "owner_label": "queue-owner",
        "created_at": "2026-08-19T00:00:00Z",
        "temperature": "warm",
        "kind": "procedural",
        "body": "Restore the bounded queue before accepting another retry.",
        "summary": "Restore the queue before retrying.",
        "mechanism": "Bounded recovery prevents duplicate work.",
        "tags": ["queue"],
        "applicability": ["bounded worker queue"],
        "evidence": [{"type": "fixture", "id": "queue-evidence"}],
        "falsifiers": ["The queue was never interrupted."],
        "importance_micros": 800000,
        "confidence_micros": 900000,
    }


def valid_source_candidate_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "candidate",
        "label": "queue-candidate",
        "owner_label": "queue-owner",
        "created_at": "2026-08-19T00:00:01Z",
        "kind": "procedural",
        "body": "Keep this candidate in quarantine.",
        "summary": "Quarantined queue candidate.",
        "mechanism": "Pending evidence must not enter ordinary retrieval.",
        "tags": ["queue"],
        "applicability": ["candidate review"],
        "falsifiers": ["The candidate was explicitly adopted."],
    }
