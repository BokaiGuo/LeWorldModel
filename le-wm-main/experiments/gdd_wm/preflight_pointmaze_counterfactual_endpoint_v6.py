"""Static preflight for the sealed PointMaze counterfactual-endpoint V6 panel.

This tool performs no encoding, model fitting, CUDA allocation, or outcome
execution.  It verifies that the frozen root range and candidate construction
can satisfy the contracts required before the CUDA-only builder is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control import (
    MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY,
    MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT,
    _calibration_role,
)
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _canonical_action_summary


SCHEMA = "pointmaze_counterfactual_endpoint_v6_static_preflight.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collision_pairs(bank: np.ndarray) -> int:
    value = np.asarray(bank, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (panel.TOTAL_STEPS, 2):
        raise ValueError("candidate bank must be [candidate, 20, 2]")
    count = 0
    for left in range(len(value)):
        left_summary = _canonical_action_summary({"candidate_id": str(left), "action_sequence": value[left].tolist()})
        for right in range(left + 1, len(value)):
            right_summary = _canonical_action_summary({"candidate_id": str(right), "action_sequence": value[right].tolist()})
            if not np.array_equal(value[left], value[right]) and np.array_equal(left_summary, right_summary):
                count += 1
    return count


def preflight(
    *, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int, train_action_seed: int,
    train_root_count: int, calibration_root_count: int, outcome_root_count: int,
    root_id_min: int, root_id_max_exclusive: int, goal_qpos: tuple[float, float] = (0.8, 0.8),
) -> dict[str, Any]:
    """Validate roots, action support, collision pairs, and design diversity."""

    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("dataset requires states.pth and obses before V6 can be built")
    loaded_contexts = [Path(path).resolve() for path in contexts]
    old_roots, old_hashes, _old_templates, context_sha256 = panel._all_prior_identities(loaded_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (100, 4):
        raise ValueError("PointMaze state pool must be [roots,100,4]")
    counts = panel._validated_split_counts({
        "train": int(train_root_count), "calibration": int(calibration_root_count), "outcome": int(outcome_root_count),
    })
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=old_roots, root_count=len(states), split_counts=counts,
        root_id_min=int(root_id_min), root_id_max_exclusive=int(root_id_max_exclusive),
    )
    visual_roots = panel._renderer_chart_roots_v6(root, set(split))
    design_roots = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=old_roots | set(visual_roots), evaluation_roots=set(split), root_count=len(states),
        root_id_min=int(root_id_min), root_id_max_exclusive=int(root_id_max_exclusive),
    )
    goal = panel.validate_goal_qpos(goal_qpos)
    held, train, interaction = panel.goal_balanced_interaction_candidate_banks(
        states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    collision_pairs = _collision_pairs(held)
    role_counts = {name: 0 for name in ("selector", "arm_certificate", "risk_calibration")}
    for root_id, role in split.items():
        if role == "calibration":
            role_counts[("selector", "arm_certificate", "risk_calibration")[_calibration_role(str(root_id))]] += 1
    checks = {
        "contexts_present": bool(loaded_contexts),
        "split_disjoint_from_prior_roots": not (set(split) & old_roots),
        "design_disjoint_from_evaluation": not (set(design_roots) & set(split)),
        "design_disjoint_from_prior_roots": not (set(design_roots) & old_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not ((held_hashes | train_hashes) & old_hashes),
        "goal_balanced_design_entropy": float(interaction["design_winner_entropy_normalized"]) >= MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY,
        "minimum_summary_collision_pairs": collision_pairs >= MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT,
        "all_calibration_roles_populated": all(role_counts.values()),
    }
    return {
        "schema": SCHEMA,
        "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks,
        "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path),
        "prior_context_count": len(loaded_contexts), "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in counts},
        "calibration_role_counts": role_counts,
        "visual_chart_root_count": len(visual_roots), "interaction_design_root_count": len(design_roots),
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "canonical_summary_collision_pair_count": collision_pairs,
            "minimum_collision_pairs_per_root": MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT,
            "design_winner_entropy_normalized": interaction["design_winner_entropy_normalized"],
            "design_dominant_winner_fraction": interaction["design_dominant_winner_fraction"],
            "design_distinct_winner_count": interaction["design_distinct_winner_count"],
        },
        "claim_boundary": "Static root/action-bank preflight only; it is not a CUDA build, learned-model fit, or outcome result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--held-action-seed", type=int, required=True)
    parser.add_argument("--train-action-seed", type=int, required=True)
    parser.add_argument("--train-root-count", type=int, required=True)
    parser.add_argument("--calibration-root-count", type=int, required=True)
    parser.add_argument("--outcome-root-count", type=int, required=True)
    parser.add_argument("--root-id-min", type=int, required=True)
    parser.add_argument("--root-id-max-exclusive", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    if output is not None and output.exists():
        raise FileExistsError(f"refusing to overwrite static preflight output: {output}")
    result = preflight(**{key: value for key, value in vars(args).items() if key != "output"})
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
