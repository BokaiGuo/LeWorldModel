"""Audit whether the same candidate pair keeps its task ordering across roots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("candidate dataset must be an object with records")
    return dict(payload.get("metadata", {})), list(payload["records"])


def _sign(value: float, margin: float) -> int:
    if abs(value) <= margin:
        return 0
    return 1 if value > 0.0 else -1


def _action_signature(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    actions = row["action_sequence"]
    if metadata.get("candidate_bank_coordinate_frame") == "goal_relative":
        source = np.asarray(row["source_state_raw"], dtype=float)[:2]
        goal = np.asarray(row["goal_raw"], dtype=float)[:2]
        direct = goal - source
        norm = float(np.linalg.norm(direct))
        if norm <= 1.0e-8:
            raise ValueError("goal-relative action signature requires nonzero source-goal displacement")
        direct = direct / norm
        perpendicular = np.asarray([-direct[1], direct[0]])
        raw = np.asarray(actions, dtype=float)
        canonical = np.stack([raw @ direct, raw @ perpendicular], axis=1).round(8)
        canonical[np.abs(canonical) < 5.0e-8] = 0.0
        actions = canonical.tolist()
    return json.dumps(actions, sort_keys=True)


def _split_report(rows: list[dict[str, Any]], metadata: dict[str, Any], margin: float) -> dict[str, Any]:
    by_root: dict[str, dict[str, float]] = defaultdict(dict)
    action_signatures: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        root = str(row["instance_id"])
        candidate = str(row["candidate_id"])
        by_root[root][candidate] = float(row["real_task_cost"])
        action_signatures[candidate].add(_action_signature(row, metadata))

    candidates = sorted({candidate for values in by_root.values() for candidate in values})
    pair_rows = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            signs = []
            for values in by_root.values():
                if left in values and right in values:
                    signs.append(_sign(values[left] - values[right], margin))
            comparable = [sign for sign in signs if sign]
            if not comparable:
                continue
            positive = sum(sign > 0 for sign in comparable)
            negative = sum(sign < 0 for sign in comparable)
            pair_rows.append(
                {
                    "candidate_pair": [left, right],
                    "root_count": len(comparable),
                    "positive_fraction": positive / len(comparable),
                    "flip_across_roots": positive > 0 and negative > 0,
                    "majority_consistency": max(positive, negative) / len(comparable),
                }
            )
    flip_count = sum(bool(item["flip_across_roots"]) for item in pair_rows)
    return {
        "root_count": len(by_root),
        "candidate_count": len(candidates),
        "candidate_action_signature_count": {
            candidate: len(signatures) for candidate, signatures in sorted(action_signatures.items())
        },
        "shared_action_bank": bool(action_signatures) and all(
            len(signatures) == 1 for signatures in action_signatures.values()
        ),
        "pair_count": len(pair_rows),
        "pair_flip_fraction": None if not pair_rows else flip_count / len(pair_rows),
        "mean_majority_consistency": None
        if not pair_rows
        else sum(item["majority_consistency"] for item in pair_rows) / len(pair_rows),
        "pair_rows": pair_rows,
    }


def audit(candidate_dataset: Path, *, margin: float = 1.0e-3) -> dict[str, Any]:
    metadata, records = _load(candidate_dataset)
    splits = {
        split: _split_report(
            [row for row in records if str(row.get("split")) == split],
            metadata,
            margin,
        )
        for split in ("train", "calibration", "outcome", "environment_test")
        if any(str(row.get("split")) == split for row in records)
    }
    return {
        "schema": "gdd_wm_cross_root_action_effect_consistency.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "real-cost ordering diagnostic only; no teacher validation, causal claim, "
            "distillation, or planning evidence"
        ),
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "pair_margin": float(margin),
            "outcome_used_for_fit_or_variant_selection": False,
        },
        "metadata": metadata,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="name|candidate_dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    args = parser.parse_args()
    cases = {}
    for specification in args.case:
        parts = specification.split("|")
        if len(parts) != 2 or not all(parts):
            raise ValueError("each --case must be name|candidate_dataset")
        name, dataset = parts
        if name in cases:
            raise ValueError(f"duplicate case {name!r}")
        print(f"[cross-root-effect] auditing {name}")
        cases[name] = audit(Path(dataset), margin=args.margin)
    report = {
        "schema": "gdd_wm_cross_root_action_effect_consistency_panel.v0.1",
        "status": "completed_diagnostic",
        "case_order": list(cases),
        "cases": cases,
        "decision": {
            "interpretation": (
                "high same-pair rank flips with a shared action bank indicate "
                "state-conditioned action-effect transfer instability"
            ),
            "teacher_validated": False,
            "planning_validated": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
