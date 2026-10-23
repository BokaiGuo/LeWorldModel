"""Contracts for the PointMaze unseen-action-bank transfer test.

The original effect-adapter positive control evaluates held *roots* under a
fixed action bank.  That leaves open a simpler explanation: the adapter could
learn an action-template lookup table.  This module keeps the source-context
protocol but makes the action sequences themselves a held-out object.

It is deliberately independent of candidate IDs.  A valid unseen bank must
have all of the following properties:

* root partitions are disjoint and immutable;
* test candidate IDs do not occur in the training bank;
* canonical full-sequence hashes do not occur in the training bank; and
* coarse temporal-template fingerprints are disjoint as a guard against an
  accidental re-labeling of the same pulse templates.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np


SCHEMA = "pointmaze_unseen_action_bank_contract.v0.1"


def _canonical_array(sequence: Any) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 4:
        raise ValueError("action_sequence must have shape [steps>=4, 2]")
    if not np.isfinite(value).all():
        raise ValueError("action_sequence must be finite")
    value = np.round(value, 10)
    value[value == 0.0] = 0.0
    return value


def sequence_hash(sequence: Any) -> str:
    """Hash the numeric action sequence, not its candidate label."""

    return hashlib.sha256(_canonical_array(sequence).tobytes()).hexdigest()


def template_fingerprint(sequence: Any) -> str:
    """Hash four ordered temporal means plus signed transition structure.

    Exact sequence hashes catch literal reuse.  This lower-resolution hash is
    an additional template-leakage guard: renaming the old four-pulse actions
    or adding tiny per-step noise cannot make them look like a new family.
    """

    value = _canonical_array(sequence)
    blocks = np.stack([part.mean(axis=0) for part in np.array_split(value, 4)])
    transitions = np.diff(blocks, axis=0)
    compact = np.concatenate((blocks.reshape(-1), np.sign(transitions).reshape(-1)))
    return hashlib.sha256(np.round(compact, 8).tobytes()).hexdigest()


def _root_partition(records: list[dict[str, Any]], *, allowed: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in records:
        root = str(row.get("instance_id", ""))
        split = str(row.get("split", ""))
        if not root or split not in allowed:
            raise ValueError(f"record requires instance_id and split in {sorted(allowed)}")
        previous = result.setdefault(root, split)
        if previous != split:
            raise ValueError(f"root {root} occurs in multiple splits: {previous}, {split}")
    if not result:
        raise ValueError("candidate bank has no roots")
    return result


def _candidate_map(records: list[dict[str, Any]]) -> dict[int, str]:
    by_candidate: dict[int, set[str]] = defaultdict(set)
    for row in records:
        if "candidate_id" not in row:
            raise ValueError("record lacks candidate_id")
        by_candidate[int(row["candidate_id"])].add(sequence_hash(row.get("action_sequence")))
    inconsistent = sorted(candidate for candidate, values in by_candidate.items() if len(values) != 1)
    if inconsistent:
        raise ValueError(
            "candidate ID is root-conditioned or maps to several sequences: "
            f"{inconsistent[:3]}"
        )
    return {candidate: next(iter(values)) for candidate, values in sorted(by_candidate.items())}


def _root_candidate_sets(records: list[dict[str, Any]]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for row in records:
        result[str(row["instance_id"])].add(int(row["candidate_id"]))
    return result


def audit_unseen_action_bank_contract(
    train_records: list[dict[str, Any]],
    unseen_records: list[dict[str, Any]],
    *,
    minimum_template_l2_distance: float = 1.0e-3,
) -> dict[str, Any]:
    """Fail closed unless the unseen bank is action- and root-disjoint.

    ``train_records`` may include only train and calibration roots; labels in
    ``unseen_records`` are not inspected here.  The caller owns the separate
    rule that no unseen outcomes are read until after fitting.
    """

    train_partition = _root_partition(train_records, allowed={"train", "calibration"})
    unseen_partition = _root_partition(unseen_records, allowed={"train", "calibration", "outcome"})
    root_overlap = sorted(set(train_partition) & {
        root for root, split in unseen_partition.items() if split == "outcome"
    })
    if root_overlap:
        raise ValueError(f"outcome roots overlap development roots: {root_overlap[:3]}")

    train_candidates = _candidate_map(train_records)
    unseen_candidates = _candidate_map(unseen_records)
    reused_ids = sorted(set(train_candidates) & set(unseen_candidates))
    if reused_ids:
        raise ValueError(f"unseen bank reuses training candidate IDs: {reused_ids[:3]}")
    reused_hashes = sorted(set(train_candidates.values()) & set(unseen_candidates.values()))
    if reused_hashes:
        raise ValueError("unseen bank reuses training action-sequence hashes")

    train_templates = {
        template_fingerprint(row["action_sequence"])
        for row in train_records
    }
    unseen_templates = {
        template_fingerprint(row["action_sequence"])
        for row in unseen_records
    }
    template_overlap = sorted(train_templates & unseen_templates)
    if template_overlap:
        raise ValueError("unseen bank reuses training temporal-action templates")

    train_values = [
        _canonical_array(row["action_sequence"]).reshape(-1)
        for row in train_records
    ]
    unseen_values = [
        _canonical_array(row["action_sequence"]).reshape(-1)
        for row in unseen_records
    ]
    distances = np.asarray([
        np.linalg.norm(candidate - reference) / np.sqrt(candidate.size)
        for candidate in unseen_values
        for reference in train_values
    ], dtype=np.float64)
    minimum_distance = float(np.min(distances))
    if minimum_distance < float(minimum_template_l2_distance):
        raise ValueError(
            "unseen actions are too close to a training sequence under the declared "
            f"template-distance guard: {minimum_distance:.8f}"
        )

    expected = set(unseen_candidates)
    bad_roots = sorted(
        root for root, ids in _root_candidate_sets(unseen_records).items() if ids != expected
    )
    if bad_roots:
        raise ValueError(f"unseen roots do not all carry the full candidate bank: {bad_roots[:3]}")
    if "train" not in unseen_partition.values() or "calibration" not in unseen_partition.values() or "outcome" not in unseen_partition.values():
        raise ValueError("unseen bank must retain non-empty train, calibration, and outcome root splits")

    return {
        "schema": SCHEMA,
        "status": "passed",
        "train_root_counts": {
            split: sum(value == split for value in train_partition.values())
            for split in ("train", "calibration")
        },
        "unseen_root_counts": {
            split: sum(value == split for value in unseen_partition.values())
            for split in ("train", "calibration", "outcome")
        },
        "training_candidate_ids": sorted(train_candidates),
        "unseen_candidate_ids": sorted(unseen_candidates),
        "training_action_hashes": sorted(train_candidates.values()),
        "unseen_action_hashes": sorted(unseen_candidates.values()),
        "candidate_id_overlap": reused_ids,
        "action_hash_overlap": reused_hashes,
        "template_fingerprint_overlap": template_overlap,
        "minimum_cross_bank_action_l2": minimum_distance,
        "minimum_template_l2_distance": float(minimum_template_l2_distance),
        "root_overlap_outcome_vs_development": root_overlap,
        "candidate_identity_rule": "numeric candidate ID and full canonical action hash are both disjoint",
        "template_leakage_rule": "ordered four-block means and transition signs are disjoint",
    }


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
