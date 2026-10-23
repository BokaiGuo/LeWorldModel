"""Canonicalize GDD candidate identity to the declared action sequence.

Candidate-set contexts intentionally prefix candidate IDs with a context name.
That is useful provenance, but it must not hide an action shared across
contexts from the D1 repeated-preference audit.  This adapter changes only the
*audit identity* to a SHA-256 of ``action_sequence`` and preserves the source
candidate ID and all other fields.  It never scores, filters, or reorders rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _action_hash(sequence: Any) -> str:
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("action_sequence must be a non-empty list")
    encoded = json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata, rows = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("source must contain metadata and non-empty records")
    rewritten: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("all source records must be objects")
        action_hash = _action_hash(row.get("action_sequence"))
        copy = dict(row)
        copy["source_candidate_id"] = copy.get("source_candidate_id", copy.get("candidate_id"))
        copy["candidate_id"] = f"action_sha256::{action_hash}"
        copy["action_sequence_sha256"] = action_hash
        cell = (str(copy.get("instance_id")), str(copy.get("teacher_set_id")), action_hash)
        if cell in seen:
            raise ValueError("a teacher set contains duplicate action sequences after canonicalization")
        seen.add(cell)
        rewritten.append(copy)
    new_metadata = dict(metadata)
    new_metadata["candidate_identity_protocol"] = {
        "identity": "sha256(action_sequence)",
        "source_dataset_sha256": _file_hash(source),
        "context_prefix_not_used_as_action_identity": True,
        "outcome_used_for_canonicalization": False,
    }
    result = {"metadata": new_metadata, "records": rewritten}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "gdd_wm_canonical_candidate_action_identity.v0.1",
        "status": "built",
        "record_count": len(rewritten),
        "source_dataset_sha256": _file_hash(source),
        "output_dataset_sha256": _file_hash(output),
        "claim_boundary": "audit-identity adapter only; not a teacher, feature transform, or planning result",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(canonicalize(args.source, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
