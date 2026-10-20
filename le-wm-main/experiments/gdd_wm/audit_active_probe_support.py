"""Audit whether the predeclared active probes lie in policy action support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_benchmark.fit_pointmaze_online_effect_policy import (
    OnlineEffectPolicy,
    action_sequence_basis,
    support_distance,
)
from experiments.run_live_pointmaze_planning import GOAL_STATE, active_probe_actions


def audit(policy_path: Path, dataset_path: Path, output_path: Path) -> dict:
    policy = OnlineEffectPolicy.load(policy_path, torch.device("cpu"))
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    goal = np.asarray(metadata.get("goal_coordinate", GOAL_STATE), dtype=np.float64)
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate dataset must contain records")
    roots: dict[str, dict] = {}
    for row in rows:
        roots.setdefault(str(row["instance_id"]), row)
    distances: list[float] = []
    per_root: dict[str, dict] = {}
    for root, row in sorted(roots.items()):
        probes = active_probe_actions(np.asarray(row["source_state_raw"]), goal)
        bases = np.stack([
            action_sequence_basis(np.repeat(action[None], policy.action_steps, axis=0))
            for action in probes
        ])
        values = np.asarray([
            support_distance(policy.action_support, basis) for basis in bases
        ], dtype=np.float64)
        distances.extend(values.tolist())
        per_root[root] = {
            "support_distances": values.tolist(),
            "pass_count": int(np.sum(values <= float(policy.action_support["threshold"]))),
        }
    values = np.asarray(distances, dtype=np.float64)
    threshold = float(policy.action_support["threshold"])
    report = {
        "schema": "gdd_wm_active_probe_support_audit.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "policy": str(policy_path),
            "dataset": str(dataset_path),
            "probe_definition": "goal-relative direct/opposite/perpendicular/opposite at magnitude 0.5",
            "outcome_used_for_probe_selection": False,
            "support_threshold": threshold,
        },
        "summary": {
            "root_count": len(roots),
            "probe_count": int(len(values)),
            "mean_support_distance": float(np.mean(values)),
            "p90_support_distance": float(np.quantile(values, 0.90)),
            "max_support_distance": float(np.max(values)),
            "support_pass_fraction": float(np.mean(values <= threshold)),
        },
        "roots": per_root,
        "claim_boundary": "action-support diagnostic only; no closed-loop or planning claim",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.policy, args.dataset, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
