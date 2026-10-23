"""Freeze a state-conditional action-pair contrast panel before CUDA fitting.

V11 deliberately does not reuse the full-bank top-two margin gate.  That gate
is necessary for certifying a 16-way planner choice, but it rejects a valid
counterfactual-identifiability experiment whenever an unrelated pair happens
to be nearly tied.  This preflight instead demands a margin for the exact
same-summary action pairs that C1 will evaluate, while retaining root/action
support isolation and a held-exact scorer bank.
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
from experiments.gdd_wm.audit_pointmaze_candidate_design_frontier import pairwise_contrast_metrics
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _collision_pairs, _sha256


SCHEMA = panel.PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA
MIN_PAIRWISE_Q10 = 0.003
MIN_PAIRWISE_FRACTION = 0.80


def preflight(
    *, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
    train_action_seed: int, root_id_min: int, root_id_max_exclusive: int,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V11 dataset requires states.pth and obses")
    resolved_contexts = [Path(path).resolve() for path in contexts]
    prior_roots, prior_hashes, _templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    counts = panel._validated_split_counts(split_counts)
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=prior_roots, root_count=len(states), split_counts=counts,
        root_id_min=int(root_id_min), root_id_max_exclusive=int(root_id_max_exclusive),
    )
    chart_roots = panel._renderer_chart_roots_v6(root, set(split))
    design_roots = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=prior_roots | set(chart_roots), evaluation_roots=set(split),
        root_count=len(states), root_id_min=int(root_id_min), root_id_max_exclusive=int(root_id_max_exclusive),
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    held, train, interaction = panel.pairwise_contrast_robust_interaction_candidate_banks(
        states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    pairs = pairwise_contrast_metrics(
        states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], held, goal,
    )
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    checks = {
        "contexts_present": bool(resolved_contexts),
        "split_disjoint_from_prior_roots": not bool(set(split) & prior_roots),
        "design_disjoint_from_evaluation": not bool(set(design_roots) & set(split)),
        "design_disjoint_from_prior_roots": not bool(set(design_roots) & prior_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not bool((held_hashes | train_hashes) & prior_hashes),
        "minimum_summary_collision_pairs": _collision_pairs(held) >= len(held) // 2,
        "pairwise_absolute_cost_gap_q10": float(pairs["pooled_absolute_gap_q10"]) >= MIN_PAIRWISE_Q10,
        "pairwise_decision_sensitive_fraction": float(pairs["pooled_gap_ge_0p003_fraction"]) >= MIN_PAIRWISE_FRACTION,
        "pairwise_state_conditional_signs": int(pairs["sign_mixed_pair_count"]) == len(held) // 2,
    }
    return {
        "schema": SCHEMA,
        "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks,
        "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path),
        "construction": {
            "root_seed": int(root_seed), "held_action_seed": int(held_action_seed),
            "train_action_seed": int(train_action_seed), "root_id_min": int(root_id_min),
            "root_id_max_exclusive": int(root_id_max_exclusive),
        },
        "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in counts},
        "interaction_design_roots": design_roots,
        "candidate_bank": {
            "candidate_design": "pairwise_contrast_robust_interaction_pairs",
            "goal_qpos": goal.tolist(), "held_shape": list(held.shape), "train_shape": list(train.shape),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": _collision_pairs(held),
            "pairwise_contrast_contract": "state_conditional_summary_collision_pairs_v1",
            "pairwise_absolute_cost_gap_q10": float(pairs["pooled_absolute_gap_q10"]),
            "pairwise_gap_ge_0p003_fraction": float(pairs["pooled_gap_ge_0p003_fraction"]),
            "pairwise_sign_mixed_pair_count": int(pairs["sign_mixed_pair_count"]),
            "pairwise_per_pair": pairs["per_pair"],
            "selection_order": interaction.get("selection_order"),
        },
        "claim_boundary": "Static root/action/pairwise-contrast preflight only; no encoder fit, model fit, calibration label, outcome label, or planning claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--held-action-seed", type=int, required=True)
    parser.add_argument("--train-action-seed", type=int, required=True)
    parser.add_argument("--root-id-min", type=int, required=True)
    parser.add_argument("--root-id-max-exclusive", type=int, required=True)
    parser.add_argument("--train-root-count", type=int, required=True)
    parser.add_argument("--calibration-root-count", type=int, required=True)
    parser.add_argument("--outcome-root-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V11 static preflight {args.output}")
    result = preflight(
        dataset_root=args.dataset_root, contexts=args.contexts, root_seed=args.root_seed,
        held_action_seed=args.held_action_seed, train_action_seed=args.train_action_seed,
        root_id_min=args.root_id_min, root_id_max_exclusive=args.root_id_max_exclusive,
        split_counts={"train": args.train_root_count, "calibration": args.calibration_root_count, "outcome": args.outcome_root_count},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
