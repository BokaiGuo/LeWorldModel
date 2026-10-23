"""Freeze V10's actual endpoint-repair routing intervention before CUDA fitting.

V10 is deliberately narrower than a generic repairability claim.  It tests
whether a pre-outcome router can decide when to replace a frozen direct-cost
scorer with an *executed* endpoint-coupled counterfactual repair.  The static
ledger binds a fresh root range, the shared in-support action bank, compact
router capacity, and the cost-only fail-closed fallback.  It cannot observe
scorer disagreement, train a model, or establish efficacy.
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
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _collision_pairs, _sha256
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import (
    V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
    V9_MIN_ROUTER_ROOT_SURPLUS,
    V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
    V9_ROUTER_FEATURE_CONTRACT,
    V9_ROUTER_FEATURE_DIMENSION,
    V7_POST_EXECUTION_ALIASING_CONTRACT,
)


SCHEMA = "pointmaze_endpoint_repairability_router_v10_static_preflight.v1"
EXTENSION_SEED = 202610970
ROOT_RANGE = (16312, 16824)
# The panel keeps the prior 384-root budget.  At V5's observed ~22% scorer
# disagreement rate, 192 training roots provide ~42 opportunities in
# expectation for the fixed 31-root capacity gate; 128 calibration roots give
# about 14 expected opportunities in each of its two independent roles.
SPLIT_COUNTS = {"train": 192, "calibration": 128, "outcome": 64}
MIN_TOP2_GAP_Q10 = 0.001
MIN_TOP2_GAP_FRACTION = 0.50
MIN_COST_SPREAD_Q10 = 0.05


def _design_quality(interaction: dict[str, Any]) -> dict[str, float]:
    """Extract the shared quality contract from robust-design output."""

    headroom = interaction.get("design_all_root_headroom")
    if not isinstance(headroom, dict):
        raise ValueError("V10 robust candidate design lacks aggregate headroom")
    required = {
        "winner_entropy_normalized",
        "best_second_gap_q10",
        "best_second_gap_ge_0p003_fraction",
        "candidate_cost_spread_q10",
    }
    if not required <= set(headroom):
        raise ValueError("V10 robust candidate design has incomplete aggregate headroom")
    return {key: float(headroom[key]) for key in required}


def _extension_contract(root: Path) -> dict[str, Any]:
    path = root / "extension_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("V10 dataset requires extension_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": value.get("schema") == "pointmaze_independent_state_action_extension.v1",
        "seed": value.get("seed") == EXTENSION_SEED,
        "new_episode_count": value.get("new_episode_count") == ROOT_RANGE[1] - ROOT_RANGE[0],
        "range": value.get("new_root_id_range") == [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
        "base_root_count": value.get("base_root_count") == ROOT_RANGE[0],
        "extended_root_count": value.get("extended_root_count") == ROOT_RANGE[1],
    }
    if not all(checks.values()):
        raise ValueError(f"V10 extension manifest violates frozen contract: {checks}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "checks": checks}


def preflight(*, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
              train_action_seed: int) -> dict[str, Any]:
    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V10 dataset requires states.pth and obses")
    extension = _extension_contract(root)
    resolved_contexts = [Path(item).resolve() for item in contexts]
    old_roots, old_hashes, _templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.shape != (ROOT_RANGE[1], 100, 4) or not np.isfinite(states).all():
        raise ValueError("V10 state pool must be finite [16824,100,4]")
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=old_roots, root_count=len(states), split_counts=SPLIT_COUNTS,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
    )
    visual = panel._renderer_chart_roots_v6(root, set(split))
    design = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=old_roots | set(visual), evaluation_roots=set(split), root_count=len(states),
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    held, train, interaction = panel.expanded_tail_robust_goal_balanced_interaction_candidate_banks(
        states[np.asarray(design, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(item) for item in held}
    train_hashes = {panel.sequence_hash(item) for item in train}
    quality = _design_quality(interaction)
    calibration_roles = {"selector": 0, "certification": 0}
    for root_id, role in split.items():
        if role == "calibration":
            # Match the runtime partition exactly; otherwise this would only
            # prove two arbitrary non-empty buckets, not V10's certificate
            # support roles.
            fold = int(hashlib.sha256(f"v10/calibration/{root_id}".encode("utf-8")).hexdigest()[:8], 16) % 2
            calibration_roles["selector" if fold == 0 else "certification"] += 1
    checks = {
        "contexts_present": bool(resolved_contexts),
        "split_disjoint_from_prior_roots": not (set(split) & old_roots),
        "design_disjoint_from_evaluation": not (set(design) & set(split)),
        "design_disjoint_from_prior_roots": not (set(design) & old_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not ((held_hashes | train_hashes) & old_hashes),
        "goal_balanced_design_entropy": quality["winner_entropy_normalized"] >= 0.50,
        "top2_gap_lower_tail": quality["best_second_gap_q10"] >= MIN_TOP2_GAP_Q10,
        "decision_sensitive_fraction": quality["best_second_gap_ge_0p003_fraction"] >= MIN_TOP2_GAP_FRACTION,
        "candidate_cost_spread": quality["candidate_cost_spread_q10"] >= MIN_COST_SPREAD_Q10,
        "minimum_summary_collision_pairs": _collision_pairs(held) >= 8,
        "router_train_root_surplus": SPLIT_COUNTS["train"] >= V9_ROUTER_FEATURE_DIMENSION + V9_MIN_ROUTER_ROOT_SURPLUS,
        "calibration_roles_populated": all(calibration_roles.values()),
    }
    return {
        "schema": SCHEMA, "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path), "extension": extension,
        "construction": {"root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed), "root_id_min": ROOT_RANGE[0], "root_id_max_exclusive": ROOT_RANGE[1]},
        "prior_context_count": len(resolved_contexts), "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
        "split_roots": {name: sorted(root_id for root_id, value in split.items() if value == name) for name in SPLIT_COUNTS},
        "visual_chart_root_count": len(visual), "interaction_design_root_count": len(design), "interaction_design_roots": design,
        "post_execution_aliasing_contract": dict(V7_POST_EXECUTION_ALIASING_CONTRACT),
        "candidate_bank": {
            "candidate_design": "expanded_tail_robust_goal_balanced_interaction_pairs", "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": _collision_pairs(held),
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": quality["best_second_gap_q10"],
            "design_best_second_cost_gap_ge_0p003_fraction": quality["best_second_gap_ge_0p003_fraction"],
            "design_candidate_cost_spread_q10": quality["candidate_cost_spread_q10"],
            "router_feature_contract": V9_ROUTER_FEATURE_CONTRACT,
            "router_feature_dimension": V9_ROUTER_FEATURE_DIMENSION, "router_train_root_count": SPLIT_COUNTS["train"],
            "minimum_router_train_root_surplus": V9_MIN_ROUTER_ROOT_SURPLUS, "router_kind": "advantage_ridge",
            "router_action_representation": "full_ordered_20x2",
            "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
            "router_training_scope": "scorer_disagreement_only",
            "opportunity_gate": "cost_endpoint_candidate_disagreement",
            "minimum_router_opportunity_root_count": V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
            "minimum_calibration_opportunity_roots_per_role": V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
            "repair_arm": "endpoint_coupled_counterfactual",
        },
        "claim_boundary": "Static V10 identity/support/capacity ledger only; no scorer fit, actual disagreement, repair efficacy, certificate, or planning result is implied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--contexts", required=True, nargs="+", type=Path)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--held-action-seed", required=True, type=int)
    parser.add_argument("--train-action-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-identical-output", action="store_true")
    args = vars(parser.parse_args())
    output, reuse = args.pop("output"), args.pop("reuse_identical_output")
    result = preflight(**args)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if not reuse or output.read_text(encoding="utf-8") != serialized:
            raise ValueError("refusing to overwrite or alter an immutable V10 preflight")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": result["status"], "root_counts": result["root_counts"]}, indent=2))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
