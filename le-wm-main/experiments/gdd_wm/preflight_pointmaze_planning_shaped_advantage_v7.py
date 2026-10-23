"""Fail-closed static preflight for the fresh V7 advantage-router panel.

No encoder, learned scorer, CUDA allocation, or outcome execution occurs here.
The preflight instead freezes the exact fresh root range, all prior-context
exclusions, supported interaction candidate bank, and V5-compatible
calibration partition before a GPU job may be launched.
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
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import _root_fold
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _canonical_action_summary


SCHEMA = "pointmaze_planning_shaped_advantage_v7_static_preflight.v1"
V7_EXTENSION_SEED = 202610370
V7_NEW_ROOT_RANGE = (10296, 10808)
V7_SPLIT_COUNTS = {"train": 128, "calibration": 192, "outcome": 64}
V7_MIN_COLLISION_PAIRS = 8
# Task-unit requirements, evaluated only on disjoint design roots before any
# train/calibration/outcome materialization.  A high-entropy bank with nearly
# tied winners cannot reliably expose action-ranking improvements.
V7_MIN_TOP2_GAP_Q10 = 0.001
V7_MIN_TOP2_GAP_GE_0P003_FRACTION = 0.50
V7_MIN_COST_SPREAD_Q10 = 0.05
# Full ordered root features have 91 task/history/action coordinates plus
# four frozen effect-disagreement coordinates.  Require a declared root
# surplus so p >= n cannot masquerade as a repairability failure.
V7_ROUTER_FEATURE_DIMENSION = 95
V7_MIN_ROUTER_ROOT_SURPLUS = 16
V7_ROUTER_KIND = "advantage_ridge"
V7_ROUTER_ACTION_REPRESENTATION = "full_ordered_20x2"
V7_ROUTER_REGULARIZATION_SELECTION = "train_root_only_four_fold_blocked_advantage_mse"
# C1 is a post-execution mechanism audit, but its scale and contrast threshold
# must still be declared before any V7 outcome labels exist.  Its full pair
# coverage is required because the subsequent disagreement-overlap audit must
# never reason from a truncated subset of outcome roots.
V7_POST_EXECUTION_ALIASING_CONTRACT = {
    "representation_key": "source_history_model",
    "radius_quantile": 0.25,
    "contrast_gap_threshold": 0.05,
    "max_reported_pairs": V7_SPLIT_COUNTS["outcome"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _extension_contract(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "extension_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V7 dataset requires extension_manifest.json")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V7 extension manifest must be an object")
    checks = {
        "extension_schema": value.get("schema") == "pointmaze_independent_state_action_extension.v1",
        "extension_seed": value.get("seed") == V7_EXTENSION_SEED,
        "extension_episode_count": value.get("new_episode_count") == V7_NEW_ROOT_RANGE[1] - V7_NEW_ROOT_RANGE[0],
        "extension_root_range": value.get("new_root_id_range") == [V7_NEW_ROOT_RANGE[0], V7_NEW_ROOT_RANGE[1] - 1],
        "extension_base_root_count": value.get("base_root_count") == V7_NEW_ROOT_RANGE[0],
        "extension_total_root_count": value.get("extended_root_count") == V7_NEW_ROOT_RANGE[1],
    }
    if not all(checks.values()):
        raise ValueError(f"V7 extension manifest violates frozen contract: {checks}")
    return {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path), "checks": checks}


def preflight(*, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
              train_action_seed: int) -> dict[str, Any]:
    """Freeze V7 identities and candidate support before GPU work begins."""

    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V7 dataset requires states.pth and obses")
    extension = _extension_contract(root)
    resolved_contexts = [Path(path).resolve() for path in contexts]
    old_roots, old_hashes, _old_templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.shape != (V7_NEW_ROOT_RANGE[1], 100, 4) or not np.isfinite(states).all():
        raise ValueError("V7 state pool must be finite [10808,100,4]")
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=old_roots, root_count=len(states), split_counts=V7_SPLIT_COUNTS,
        root_id_min=V7_NEW_ROOT_RANGE[0], root_id_max_exclusive=V7_NEW_ROOT_RANGE[1],
    )
    visual_roots = panel._renderer_chart_roots_v6(root, set(split))
    design_roots = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=old_roots | set(visual_roots), evaluation_roots=set(split), root_count=len(states),
        root_id_min=V7_NEW_ROOT_RANGE[0], root_id_max_exclusive=V7_NEW_ROOT_RANGE[1],
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    held, train, interaction = panel.goal_balanced_interaction_candidate_banks(
        states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    collision_pairs = _collision_pairs(held)
    headroom = interaction.get("design_oracle_headroom")
    if not isinstance(headroom, dict):
        raise RuntimeError("goal-balanced interaction report lacks design-root oracle headroom")
    top2_gap = headroom.get("best_second_cost_gap")
    cost_spread = headroom.get("candidate_cost_spread")
    if not isinstance(top2_gap, dict) or not isinstance(cost_spread, dict):
        raise RuntimeError("goal-balanced interaction report has malformed oracle headroom")
    calibration_roles = {"selector": 0, "certification": 0}
    for root_id, split_name in split.items():
        if split_name == "calibration":
            calibration_roles["selector" if _root_fold(str(root_id)) % 2 == 0 else "certification"] += 1
    checks = {
        "contexts_present": bool(resolved_contexts),
        "split_disjoint_from_prior_roots": not (set(split) & old_roots),
        "design_disjoint_from_evaluation": not (set(design_roots) & set(split)),
        "design_disjoint_from_prior_roots": not (set(design_roots) & old_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not ((held_hashes | train_hashes) & old_hashes),
        "goal_balanced_design_entropy": float(interaction["design_winner_entropy_normalized"]) >= 0.50,
        "goal_balanced_top2_gap_lower_tail": float(top2_gap["q10"]) >= V7_MIN_TOP2_GAP_Q10,
        "goal_balanced_decision_sensitive_root_fraction": float(top2_gap["ge_0p003_fraction"]) >= V7_MIN_TOP2_GAP_GE_0P003_FRACTION,
        "goal_balanced_candidate_cost_spread": float(cost_spread["q10"]) >= V7_MIN_COST_SPREAD_Q10,
        "router_train_root_surplus": V7_SPLIT_COUNTS["train"] >= V7_ROUTER_FEATURE_DIMENSION + V7_MIN_ROUTER_ROOT_SURPLUS,
        "minimum_summary_collision_pairs": collision_pairs >= V7_MIN_COLLISION_PAIRS,
        "router_calibration_roles_populated": all(calibration_roles.values()),
    }
    return {
        "schema": SCHEMA,
        "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks,
        "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path), "extension": extension,
        "construction": {
            "root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed),
            "root_id_min": V7_NEW_ROOT_RANGE[0], "root_id_max_exclusive": V7_NEW_ROOT_RANGE[1],
        },
        "prior_context_count": len(resolved_contexts), "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in V7_SPLIT_COUNTS},
        "split_roots": {name: sorted(root_id for root_id, value in split.items() if value == name) for name in V7_SPLIT_COUNTS},
        "visual_chart_root_count": len(visual_roots), "interaction_design_root_count": len(design_roots),
        "interaction_design_roots": design_roots, "router_calibration_role_counts": calibration_roles,
        "post_execution_aliasing_contract": dict(V7_POST_EXECUTION_ALIASING_CONTRACT),
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": collision_pairs,
            "minimum_collision_pairs_per_root": V7_MIN_COLLISION_PAIRS,
            "design_winner_entropy_normalized": interaction["design_winner_entropy_normalized"],
            "design_dominant_winner_fraction": interaction["design_dominant_winner_fraction"],
            "design_distinct_winner_count": interaction["design_distinct_winner_count"],
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": float(top2_gap["q10"]),
            "design_best_second_cost_gap_ge_0p003_fraction": float(top2_gap["ge_0p003_fraction"]),
            "design_candidate_cost_spread_q10": float(cost_spread["q10"]),
            "minimum_design_best_second_cost_gap_q10": V7_MIN_TOP2_GAP_Q10,
            "minimum_design_best_second_cost_gap_ge_0p003_fraction": V7_MIN_TOP2_GAP_GE_0P003_FRACTION,
            "minimum_design_candidate_cost_spread_q10": V7_MIN_COST_SPREAD_Q10,
            "router_feature_contract": "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1",
            "router_feature_dimension": V7_ROUTER_FEATURE_DIMENSION,
            "router_train_root_count": V7_SPLIT_COUNTS["train"],
            "minimum_router_train_root_surplus": V7_MIN_ROUTER_ROOT_SURPLUS,
            "router_kind": V7_ROUTER_KIND,
            "router_action_representation": V7_ROUTER_ACTION_REPRESENTATION,
            "router_regularization_selection": V7_ROUTER_REGULARIZATION_SELECTION,
        },
        "claim_boundary": "Static V7 root/action-bank preflight only; it is not a CUDA build, learned-model fit, outcome execution, or prospective repairability result.",
    }


def _identical_or_write(result: dict[str, Any], output: Path, *, reuse_identical_output: bool) -> None:
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if not reuse_identical_output:
            raise FileExistsError(f"refusing to overwrite static preflight output: {output}")
        if output.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing V7 preflight differs from current frozen inputs")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--held-action-seed", type=int, required=True)
    parser.add_argument("--train-action-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-identical-output", action="store_true")
    args = vars(parser.parse_args())
    output, reuse = args.pop("output"), args.pop("reuse_identical_output")
    result = preflight(**args)
    _identical_or_write(result, output, reuse_identical_output=reuse)
    print(json.dumps({"status": result["status"], "root_counts": result["root_counts"]}, indent=2))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
