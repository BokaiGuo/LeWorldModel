"""Materialize and audit a held-root PointMaze effect-adapter replication.

Older intervention-supervised artifacts reported a positive PointMaze ranking
control but did not retain their exact dataset paths.  This script is a
fail-closed replacement: it accepts one frozen, shared-world candidate bank,
proves its action identity and root partition, then materializes *only* the
train/calibration and outcome partitions for
``evaluate_candidate_effect_adapter_external``.  The generated outcome JSON
renames its frozen ``outcome`` split to ``external_outcome`` solely to satisfy
that evaluator's cross-bank contract; the source hash and original split are
persisted so this is never described as a newly collected external panel.

It neither selects a checkpoint nor reads outcome labels to choose the bank,
feature family, training duration, or thresholds.  It is a reproducible
intervention-supervised positive-control audit, not frozen world-model or
closed-loop planning evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "pointmaze_effect_adapter_reproducibility_audit.v0.1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected candidate dataset with records")
    if not payload["records"]:
        raise ValueError(f"{path}: candidate dataset is empty")
    return payload


def _root_splits(records: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in records:
        root, split = str(row.get("instance_id", "")), str(row.get("split", ""))
        if not root or split not in {"train", "calibration", "outcome"}:
            raise ValueError("records require a root and train/calibration/outcome split")
        previous = result.setdefault(root, split)
        if previous != split:
            raise ValueError(f"root {root} occurs in multiple splits: {previous}, {split}")
    return result


def _shared_action_identity(records: list[dict[str, Any]]) -> dict[str, str]:
    by_candidate: dict[str, set[str]] = defaultdict(set)
    for row in records:
        candidate = str(row.get("candidate_id", ""))
        sequence = row.get("action_sequence")
        if not candidate or not isinstance(sequence, list) or not sequence:
            raise ValueError("records require candidate_id and non-empty action_sequence")
        by_candidate[candidate].add(_canonical_hash(sequence))
    nonshared = sorted(candidate for candidate, values in by_candidate.items() if len(values) != 1)
    if nonshared:
        raise ValueError(
            "candidate action identity is root-conditioned, not one shared world-action bank: "
            f"{nonshared[:3]}"
        )
    return {candidate: next(iter(values)) for candidate, values in sorted(by_candidate.items())}


def materialize(
    source_dataset: Path,
    source_arrays: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate one frozen bank and write evaluator-compatible derived JSONs."""

    payload = _load(source_dataset)
    records = [dict(row) for row in payload["records"]]
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("shared_world_action_bank") is not True:
        raise ValueError("source metadata must declare shared_world_action_bank=true")
    if metadata.get("candidate_bank_coordinate_frame") != "world_fixed":
        raise ValueError("source metadata must declare world_fixed action coordinates")
    if not source_arrays.is_file():
        raise FileNotFoundError(f"missing candidate arrays: {source_arrays}")

    root_splits = _root_splits(records)
    actions = _shared_action_identity(records)
    by_split = {
        split: sorted(root for root, value in root_splits.items() if value == split)
        for split in ("train", "calibration", "outcome")
    }
    if not by_split["train"] or not by_split["calibration"] or not by_split["outcome"]:
        raise ValueError("source needs non-empty train, calibration, and outcome root partitions")
    candidate_sets: dict[str, set[str]] = defaultdict(set)
    for row in records:
        candidate_sets[str(row["instance_id"])].add(str(row["candidate_id"]))
    expected = set(actions)
    mismatched = sorted(root for root, values in candidate_sets.items() if values != expected)
    if mismatched:
        raise ValueError(f"roots do not carry the same complete candidate IDs: {mismatched[:3]}")

    source_hash = _sha256_file(source_dataset)
    arrays_hash = _sha256_file(source_arrays)
    provenance = {
        "schema": SCHEMA,
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": source_hash,
        "source_arrays": str(source_arrays.resolve()),
        "source_arrays_sha256": arrays_hash,
        "source_split_semantics": "frozen root-disjoint train/calibration/outcome",
        "replication_semantics": "held-root replication; not newly collected external data",
        "root_ids_sha256": {
            split: _canonical_hash(values) for split, values in by_split.items()
        },
        "root_counts": {split: len(values) for split, values in by_split.items()},
        "candidate_action_sha256": actions,
        "candidate_action_bank_sha256": _canonical_hash(actions),
        "outcome_used_to_select_source_or_protocol": False,
    }

    development_records = [row for row in records if str(row["split"]) in {"train", "calibration"}]
    outcome_records = []
    for row in records:
        if str(row["split"]) != "outcome":
            continue
        derived = dict(row)
        derived["split"] = "external_outcome"
        derived["source_split"] = "outcome"
        outcome_records.append(derived)
    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "development_candidate_dataset.json"
    outcome_path = output / "held_root_candidate_dataset.json"
    development_path.write_text(
        json.dumps({"metadata": {**metadata, **provenance, "derived_role": "development"}, "records": development_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outcome_path.write_text(
        json.dumps({"metadata": {**metadata, **provenance, "derived_role": "held_root_outcome"}, "records": outcome_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        **provenance,
        "status": "materialized_valid_held_root_replication_inputs",
        "development_dataset": str(development_path.resolve()),
        "held_root_dataset": str(outcome_path.resolve()),
        "claim_boundary": (
            "Input-provenance audit only. The derived held partition is root-disjoint but is not "
            "an independently collected external dataset."
        ),
    }
    (output / "input_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = materialize(args.source_dataset, args.source_arrays, args.output)
    print(json.dumps({key: report[key] for key in ("status", "root_counts", "candidate_action_bank_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
