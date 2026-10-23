"""Freeze an independent shared-future PointMaze panel before CUDA work.

V8 is deliberately disjoint from the V7 advantage-router panel.  It exists
solely to evaluate the action-independent shared-future control on fresh roots
and an exact supported candidate bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _collision_pairs, _sha256


SCHEMA = "pointmaze_shared_future_v8_static_preflight.v1"
# The original 512-root V8 interval was consumed by later immutable context
# ledgers before this control was launched.  It could supply the 384
# evaluation roots but not the additional 64 disjoint interaction-design
# roots.  V8-r2 therefore starts from the newest unconsumed state/action pool
# and reserves a 640-root interval before any panel/model work.
EXTENSION_SEED = 202611500
ROOT_RANGE = (17720, 18360)
SPLIT_COUNTS = {"train": 128, "calibration": 192, "outcome": 64}
MIN_TOP2_GAP_Q10 = 0.001
MIN_TOP2_GAP_FRACTION = 0.50
MIN_COST_SPREAD_Q10 = 0.05


def _extension_contract(root: Path) -> dict[str, Any]:
    path = root / "extension_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("V8 dataset requires extension_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": value.get("schema") == "pointmaze_independent_state_action_extension.v1",
        "seed": value.get("seed") == EXTENSION_SEED,
        "base_root_count": value.get("base_root_count") == ROOT_RANGE[0],
        "extended_root_count": value.get("extended_root_count") == ROOT_RANGE[1],
        "range": value.get("new_root_id_range") == [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    if not all(checks.values()):
        raise ValueError(f"V8 extension manifest violates frozen contract: {checks}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "checks": checks}


def preflight(*, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
              train_action_seed: int) -> dict[str, Any]:
    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V8 dataset requires states.pth and obses")
    extension = _extension_contract(root)
    contexts = [Path(path).resolve() for path in contexts]
    old_roots, old_hashes, _templates, context_sha256 = panel._all_prior_identities(contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.shape != (ROOT_RANGE[1], 100, 4) or not np.isfinite(states).all():
        raise ValueError(f"V8-r2 state pool must be finite [{ROOT_RANGE[1]},100,4]")
    available = [
        root_id for root_id in range(*ROOT_RANGE)
        if root_id not in old_roots
    ]
    required = sum(SPLIT_COUNTS.values()) + panel.INTERACTION_DESIGN_ROOTS
    if len(available) < required:
        raise ValueError(
            "V8-r2 root interval lacks capacity for disjoint evaluation and "
            f"interaction design roots: need {required}, have {len(available)}"
        )
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
    held, train, interaction = panel.goal_balanced_interaction_candidate_banks(
        states[np.asarray(design, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(item) for item in held}
    train_hashes = {panel.sequence_hash(item) for item in train}
    headroom = interaction.get("design_oracle_headroom", {})
    gap, spread = headroom.get("best_second_cost_gap", {}), headroom.get("candidate_cost_spread", {})
    checks = {
        "contexts_present": bool(contexts),
        "split_disjoint_from_prior_roots": not (set(split) & old_roots),
        "design_disjoint_from_evaluation": not (set(design) & set(split)),
        "design_disjoint_from_prior_roots": not (set(design) & old_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not ((held_hashes | train_hashes) & old_hashes),
        "goal_balanced_design_entropy": float(interaction["design_winner_entropy_normalized"]) >= 0.50,
        "top2_gap_lower_tail": float(gap.get("q10", -1.0)) >= MIN_TOP2_GAP_Q10,
        "decision_sensitive_fraction": float(gap.get("ge_0p003_fraction", -1.0)) >= MIN_TOP2_GAP_FRACTION,
        "candidate_cost_spread": float(spread.get("q10", -1.0)) >= MIN_COST_SPREAD_Q10,
    }
    return {
        "schema": SCHEMA, "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path), "extension": extension,
        "construction": {"root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed), "root_id_min": ROOT_RANGE[0], "root_id_max_exclusive": ROOT_RANGE[1]},
        "prior_context_count": len(contexts), "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
        "split_roots": {name: sorted(key for key, value in split.items() if value == name) for name in SPLIT_COUNTS},
        "fresh_root_capacity": {
            "declared_interval_count": ROOT_RANGE[1] - ROOT_RANGE[0],
            "available_after_prior_context_exclusion": len(available),
            "required_evaluation_plus_interaction_design": required,
            "surplus_after_reservation": len(available) - required,
        },
        "visual_chart_root_count": len(visual), "interaction_design_root_count": len(design), "interaction_design_roots": design,
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": _collision_pairs(held),
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": float(gap["q10"]),
            "design_best_second_cost_gap_ge_0p003_fraction": float(gap["ge_0p003_fraction"]),
            "design_candidate_cost_spread_q10": float(spread["q10"]),
            "shared_future_action_representation": "full_ordered_20x2",
        },
        "claim_boundary": "Static V8 root/action-bank preflight only; it is not a CUDA fit, shared-future efficacy, or planning result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path, nargs="+")
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
            raise ValueError("refusing to overwrite or alter an immutable V8 preflight")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": result["status"], "root_counts": result["root_counts"]}, indent=2))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
